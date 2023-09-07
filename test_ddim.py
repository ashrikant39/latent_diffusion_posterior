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
from ldm.models.diffusion.ddim import DDIMSamplerJSCC
from ldm_testing import compute_metrics
from einops import rearrange
from tqdm import tqdm


if __name__=="__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_dir",
        type = str,
        help = "Directory with model logs",
        default = "none"
    )
    
    parser.add_argument(
        "--num_images",
        type = int,
        help = "Number of Images to use",
        default = 200
    )

    parser.add_argument(
        "-de",
        "--ddim_eta",
        type=float,
        nargs="?",
        help="ddim eta for ddim sampling (0.0 yields deterministic sampling)",
        default=0.0
    )

    # parser.add_argument(
    #     "--l_simple_weight",
    #     type=float,
    #     nargs="?",
    #     help="Weight for Denoising Loss wrt. AutoEncoder Loss",
    #     default=1.0
    # )

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
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )

    parser.add_argument(
        "--scale_grad",
        type = float,
        help = "Scaling for Grad of NLL",
        default = 1.0
    )


    args, unknown = parser.parse_known_args()
    CHECKPOINT_DIR = args.ckpt_dir
    TEST_SNR_DB = args.test_snr_db
    NUM_IMAGES = args.num_images
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    CUSTOM_STEPS = args.custom_steps
    DDIM_ETA = args.ddim_eta
    SCALE_GRAD = args.scale_grad
    # L_SIMPLE_WEIGHT = args.l_simple_weight
    DEVICE = "cuda"

         
    test_snr = 10**(TEST_SNR_DB/10)
    config = OmegaConf.load("/home/ashri/latent-diffusion/configs/latent-diffusion/ldm_all_models_iterative.yaml")
    config.model.params.channel_snr_dB = TEST_SNR_DB 
    # config.model.params.l_simple_weight = L_SIMPLE_WEIGHT

    # load_dir = glob.glob(os.path.join(CHECKPOINT_DIR, f"snr_{TEST_SNR_DB}/*/"))[0] ####################################################
    load_dir = CHECKPOINT_DIR
    # import pdb; pdb.set_trace()
    # checkpoint_path = os.path.join(load_dir, "checkpoints/last.ckpt")
    # config = OmegaConf.load(glob.glob(os.path.join(load_dir, "configs/*project.yaml")))
    

    # ldm_posterior_model, _ = load_model(config, checkpoint_path, gpu=True, eval_mode=True)
    ldm_posterior_model = instantiate_from_config(config["model"]).to(DEVICE)
    
    ddim_model = DDIMSamplerJSCC(ldm_posterior_model)
    ddim_model.make_schedule(ddim_num_steps=CUSTOM_STEPS, ddim_eta=DDIM_ETA, verbose=False)

    root_dir = "/tmpssd/ashri/LSUN/"
    val_txt_path = os.path.join(root_dir, "bedrooms_val.txt")
    val_data_dir = os.path.join(root_dir, "bedrooms")
    config.data.params.validation.params.txt_file = val_txt_path
    config.data.params.validation.params.data_root = val_data_dir

    main_dataset = instantiate_from_config(config.data.params.validation)
    dataset = Subset(main_dataset, indices=range(NUM_IMAGES))
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS)

    metric_dict = {
            "PSNR": 0.0,
            "MS_SSIM": 0.0,
            "FID": 0.0,
            "LPIPS_VGG": 0.0,
        }

    total_examples = dataset.__len__()

    print(f"CHANNEL SNR: {TEST_SNR_DB}")

    progress_bar = tqdm(dataloader)
    for image_dict in progress_bar:
        images = rearrange(image_dict["image"].cuda(), 'b h w c -> b c h w')
        codewords, _, (_, _, _) = ldm_posterior_model.first_stage_model.encode(images)
        signal_power = torch.mean(codewords**2)
        noisy_codeword = codewords + torch.randn_like(codewords)*torch.sqrt(signal_power/test_snr)
        sampled_codeword, _, total_steps = ddim_model.ddim_sampling(None, noisy_codeword.size(), test_snr / signal_power, x_T=noisy_codeword, scale_grad=SCALE_GRAD, re_encode=False)
        reconstructed = torch.clamp(ldm_posterior_model.first_stage_model.decode(sampled_codeword), -1.0, 1.0)
        scores = compute_metrics(images, reconstructed)

        progress_bar.set_postfix(psnr=scores[0]/BATCH_SIZE)

        for idx, key in enumerate(metric_dict.keys()):
                metric_dict[key] += scores[idx]

    for key in metric_dict.keys():
        metric_dict[key] = metric_dict[key]/total_examples
    
    savedir = os.path.join(load_dir, f"ddim_pretrained_posterior_{NUM_IMAGES}_images") ############################################################
    
    if os.path.isdir(savedir) is False:
         os.mkdir(savedir)

    filename = os.path.join(savedir, f"posterior_test_{TEST_SNR_DB}_metrics.pkl")
    with open(filename, "wb") as fp:
        pickle.dump(metric_dict, fp)
