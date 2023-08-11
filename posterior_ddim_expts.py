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
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
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
    args, unknown = parser.parse_known_args()
    LOGDIR = args.log_dir
    TEST_SNR_DB = args.test_snr_db
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    DEVICE = "cuda"

    test_snr = 10**(TEST_SNR_DB/10)
    config = OmegaConf.load("./configs/latent-diffusion/ldm_all_models_iterative.yaml")
    config.model.params.channel_snr_dB = TEST_SNR_DB

    ldm_posterior_model = instantiate_from_config(config["model"])

    ddim_model = DDIMSamplerJSCC(ldm_posterior_model)
    ddim_model.make_schedule(ddim_num_steps=args.custom_steps, ddim_eta=args.eta, verbose=False)

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

    torch.manual_seed(0)
    for image_dict in tqdm(dataloader):
        

        with torch.no_grad():
            images = rearrange(image_dict["image"].cuda(), 'b h w c -> b c h w')
            codewords = ldm_posterior_model.first_stage_model.encode(images)
            signal_power = torch.mean(codewords**2)
            noise = torch.randn_like(codewords)*torch.sqrt(signal_power/test_snr)
            noisy_codeword = codewords + noise

            print("Signal Power:", torch.mean(codewords**2).item() , "Noise Power:", torch.mean(noise ** 2).item())

            with torch.no_grad():
                reconstructed = torch.clamp(ldm_posterior_model.first_stage_model.decode(noisy_codeword), -1.0, 1.0)
                scores = compute_metrics(images, reconstructed)
                print("Before Denoising", "PSNR:", scores[0]/args.batch_size, "LPIPS:", scores[-1]/args.batch_size)

        etas = [0.0, 1.0, 10.0]
        for eta in etas:
            #sampled_codeword = ldm_posterior_model.posterior_sampling(test_snr / signal_power, noisy_codeword, eta, re_encode=False)
            sampled_codeword, _ = ddim_model.ddim_sampling(None, noisy_codeword.size(), test_snr / signal_power, x_T=noisy_codeword, scale_grad=eta)
            with torch.no_grad():
                reconstructed = torch.clamp(ldm_posterior_model.first_stage_model.decode(sampled_codeword), -1.0, 1.0)
                scores = compute_metrics(images, reconstructed)
                print("Eta: {:5.2f}".format(eta), "PSNR:", scores[0]/args.batch_size, "LPIPS:", scores[-1]/args.batch_size)
        for idx, key in enumerate(metric_dict.keys()):
                metric_dict[key] += scores[idx]

    for key in metric_dict.keys():
        metric_dict[key] = metric_dict[key]/total_examples

    filename = os.path.join(LOGDIR, f"posterior_test_{TEST_SNR_DB}_metrics.pkl")
    with open(filename, "wb") as fp:
        pickle.dump(metric_dict, fp)
