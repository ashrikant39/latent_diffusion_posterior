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


if __name__=="__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_dir",
        type = str,
        help = "Directory with model logs",
        default = "none"
    )
    
    parser.add_argument(
        "--scale_grad",
        type = float,
        help = "Scaling for Grad of NLL",
        default = 1.0
    )
    
    parser.add_argument(
        "--save_dir",
        type = str,
        help = "Directory name to save metrics",
        default = "none"
    )

    parser.add_argument(
        "--num_images",
        type = int,
        help = "Number of Images to use",
        default = 200
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

    args, unknown = parser.parse_known_args()
    CHECKPOINT_DIR = args.ckpt_dir
    TEST_SNR_DB = args.test_snr_db
    NUM_IMAGES = args.num_images
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    SCALE_GRAD = args.scale_grad
    SAVE_DIR = args.save_dir
    DEVICE = "cuda"

         
    test_snr = 10**(TEST_SNR_DB/10)
    config = OmegaConf.load("/home/ashri/latent-diffusion/configs/latent-diffusion/ldm_all_models_iterative.yaml")
    config.model.params.channel_snr_dB = TEST_SNR_DB
    
    load_dir = glob.glob(os.path.join(CHECKPOINT_DIR, f"snr_{TEST_SNR_DB}/*/"))[0]
    checkpoint_path = os.path.join(load_dir, "checkpoints/last.ckpt")
    config_path = glob.glob(os.path.join(load_dir, "configs/*project.yaml"))
    

    ldm_posterior_model, _ = load_model(config, checkpoint_path, gpu=True, eval_mode=True)

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
        sampled_codeword = ldm_posterior_model.posterior_sampling(TEST_SNR_DB, noisy_codeword, signal_power, re_encode = False, scale_grad = SCALE_GRAD)
        reconstructed = torch.clamp(ldm_posterior_model.first_stage_model.decode(sampled_codeword), -1.0, 1.0)
        scores = compute_metrics(images, reconstructed)

        progress_bar.set_postfix(psnr=scores[0]/BATCH_SIZE)

        for idx, key in enumerate(metric_dict.keys()):
                metric_dict[key] += scores[idx]

    for key in metric_dict.keys():
        metric_dict[key] = metric_dict[key]/total_examples
    
    savedir = os.path.join(load_dir, SAVE_DIR)

    if os.path.isdir(savedir) is False:
         os.mkdir(savedir)

    filename = os.path.join(savedir, f"test_{TEST_SNR_DB}_metrics.pkl")
    with open(filename, "wb") as fp:
        pickle.dump(metric_dict, fp)
