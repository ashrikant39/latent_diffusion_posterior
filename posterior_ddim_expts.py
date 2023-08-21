import torch, random
from scripts.sample_diffusion import *
import argparse, os, sys, glob, datetime, yaml
from omegaconf import OmegaConf
import torchvision
import PIL
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from ldm_testing import *
from jscc_baseline_testing import deep_jscc_testing
from torch.utils.data import DataLoader, random_split, Subset
from ldm_testing import compute_metrics
from einops import rearrange
from tqdm import tqdm
from ldm.models.diffusion.ddim import DDIMSamplerJSCC
from scripts.sample_diffusion import custom_to_pil


if __name__=="__main__":


    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-l",
        "--log_dir",
        type = str,
        help = "Directory with model logs",
        default = "none"
    )

    parser.add_argument(
        "-s",
        "--test_snr_db",
        type = int,
        help = "SNR for noise in dB",
        default = 20
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        nargs="?",
        help="batch size",
        default=2
    )

    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        nargs="?",
        help="num_workers for data loading",
        default=2
    )
    parser.add_argument(
        "-de",
        "--ddim_eta",
        type=float,
        nargs="?",
        help="ddim eta for ddim sampling (0.0 yields deterministic sampling)",
        default=0.0
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for scaling the gradient of the likelihood",
        default=1.0
    )

    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "-mi",
        "--max_images",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=64
    )
    parser.add_argument(
        "-sa",
        "--sampling_algorithm",
        type=str,
        help="sampling algorithm, 'ddpm' or 'ddim'",
        default='ddim'
    )
    parser.add_argument(
        "-re",
        "--re_encode",
        action='store_true',
        help="Re-encoding"
    )

    args, unknown = parser.parse_known_args()
    LOGDIR = args.log_dir
    TEST_SNR_DB = args.test_snr_db
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    DEVICE = "cuda"

    test_snr = 10**(TEST_SNR_DB/10)
    config = OmegaConf.load("./configs/latent-diffusion/ldm_all_models_iterative.yaml")
    config.model.params.channel_snr_dB = TEST_SNR_DB

    from scripts.sample_diffusion import load_model
    ldm_posterior_model, _ = load_model(config, "new_loss_log/2023-08-21T00-10-29_ldm_vq_all_models_iterative/checkpoints/last.ckpt", True, True)
    #ldm_posterior_model, _ = load_model(config, "models/ldm/lsun_beds256/model.ckpt", True, True)

    ddim_model = DDIMSamplerJSCC(ldm_posterior_model)
    ddim_model.make_schedule(ddim_num_steps=args.custom_steps, ddim_eta=args.ddim_eta, verbose=False)

    ldm_posterior_model = ldm_posterior_model.to("cuda")

    root_dir = "/tmpssd/ashri/LSUN/"
    val_txt_path = os.path.join(root_dir, "bedrooms_val.txt")
    val_data_dir = os.path.join(root_dir, "bedrooms")
    config.data.params.validation.params.txt_file = val_txt_path
    config.data.params.validation.params.data_root = val_data_dir

    main_dataset = instantiate_from_config(config.data.params.validation)
    dataloader = DataLoader(main_dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS)

    metric_dict = {
            "PSNR": 0.0,
            "MS_SSIM": 0.0,
            "FID": 0.0,
            "LPIPS_VGG": 0.0,
        }

    total_examples = main_dataset.__len__()
    var_inv = 10 ** (TEST_SNR_DB / 10)

    num_samples_processed = 0

    torch.manual_seed(0)
    iterator = tqdm(dataloader)
    for i, image_dict in enumerate(iterator):
        if i == args.max_images // args.batch_size:
            break
        

        with torch.no_grad():
            images = rearrange(image_dict["image"].cuda(), 'b h w c -> b c h w')
            codewords = ldm_posterior_model.first_stage_model.encode(images)
            signal_power = torch.mean(codewords**2)
            noise = torch.randn_like(codewords)*torch.sqrt(signal_power/test_snr)
            noisy_codeword = codewords + noise

            #with torch.no_grad():
            #    reconstructed = torch.clamp(ldm_posterior_model.first_stage_model.decode(noisy_codeword), -1.0, 1.0)
            #    scores = compute_metrics(images, reconstructed)

        if args.sampling_algorithm == 'ddim':
            sampled_codeword, _, total_steps = ddim_model.ddim_sampling(None, noisy_codeword.size(), test_snr / signal_power, x_T=noisy_codeword, scale_grad=args.eta, re_encode=args.re_encode)
        elif args.sampling_algorithm == 'ddpm':
            sampled_codeword, total_steps = ldm_posterior_model.posterior_sampling(test_snr / signal_power, noisy_codeword, scale_grad=args.eta)
        with torch.no_grad():
            reconstructed = torch.clamp(ldm_posterior_model.first_stage_model.decode(sampled_codeword), -1.0, 1.0)
            scores = compute_metrics(images, reconstructed)
            #print("Eta: {:5.2f}".format(args.eta), "PSNR:", scores[0]/args.batch_size, "LPIPS:", scores[-1]/args.batch_size)
        for idx, key in enumerate(metric_dict.keys()):
            metric_dict[key] += scores[idx]
        num_samples_processed += images.size(0)
        iterator.set_description("Avg PSNR: {:.2f}, Avg LPIPS: {:.4f}, steps: {}".format(metric_dict["PSNR"]/num_samples_processed, metric_dict["LPIPS_VGG"]/num_samples_processed, total_steps))


    reconstructed = custom_to_pil(reconstructed[-1,:,:,:])
    reconstructed.save("out.png")

    images = custom_to_pil(images[-1,:,:,:])
    images.save("target.png")
    exit()
    for key in metric_dict.keys():
        metric_dict[key] = metric_dict[key]/total_examples

    
    filename = os.path.join(LOGDIR, f"posterior_test_{TEST_SNR_DB}_metrics.pkl")
    with open(filename, "wb") as fp:
        pickle.dump(metric_dict, fp)
