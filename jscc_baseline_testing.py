import torch 
import glob
from scripts.sample_diffusion import *
import argparse, pickle
from omegaconf import OmegaConf
import torchvision
from tqdm import tqdm
from ldm.models.autoencoder import VQModel, VQModelInterface, AutoencoderKL, AutoEncoder_JSCC
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import multiscale_structural_similarity_index_measure, psnr
from torchvision.datasets import LSUN
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor, Compose, Resize

import warnings
warnings.filterwarnings("ignore")

def fload_to_int(x:torch.FloatTensor):
    x = torch.clamp(x, -1, 1)
    x= (x+1)/2.
    x = (255.*x).to(torch.uint8)
    return x

def deep_jscc_testing(model:AutoEncoder_JSCC, images, snr_db):
    noisy_latent = model.encode_through_channel(images, snr_db)
    reconstruction = torch.clamp(model.decode_noisy_latent(noisy_latent), -1.0, 1.0)
    return reconstruction

def encode_through_channel(latent_diffusion_model, images, snr_dB_value, denoise_latent = False, verbose = False):

    autoencoder = latent_diffusion_model.first_stage_model
    encoded_obj = autoencoder.encode(images)

    if isinstance(autoencoder, AutoencoderKL):
        latent_signal  = encoded_obj.mean

    elif isinstance(autoencoder, VQModelInterface):
        latent_signal = encoded_obj
    
    elif isinstance(autoencoder, VQModel):
        latent_signal, _, _  = encoded_obj
    
    signal_power = torch.mean(latent_signal**2)
    noise_power = signal_power/(10**(snr_dB_value/10))

    alpha_cumulative = latent_diffusion_model.alphas_cumprod
    alpha_noisy = signal_power/(signal_power+noise_power)
    enter_timestep = torch.argmin(abs(alpha_cumulative - alpha_noisy)).item()
    noisy_latent = torch.normal(mean = torch.sqrt(alpha_cumulative[enter_timestep])*latent_signal, std = torch.sqrt(1-alpha_cumulative[enter_timestep]))

    if denoise_latent is True:
        denoised_latent, _ = latent_diffusion_model.progressive_denoising(cond=None, shape=noisy_latent.shape, timesteps = enter_timestep, x_T = noisy_latent, verbose = verbose)
        reconstructed = torch.clamp(autoencoder.decode(denoised_latent), -1.0, 1.0)
    
    else:
        reconstructed = torch.clamp(autoencoder.decode(noisy_latent), -1.0, 1.0)

    return reconstructed

    
    

def compute_metrics(true_images, reconstructed_images, device = "cuda"):

    psnr_val = psnr(reconstructed_images, true_images, reduction='sum', dim = [1, 2, 3], data_range = 2.0).item()
    total_multiscale_ssim = (multiscale_structural_similarity_index_measure(reconstructed_images, true_images, data_range=2.0)*true_images.size(0)).item()

    fid_obj = FrechetInceptionDistance().to(device)

    true_images_uint8 = fload_to_int(true_images)
    reconstructed_uint8 = fload_to_int(reconstructed_images)
    fid_obj.update(true_images_uint8, real=True)
    fid_obj.update(reconstructed_uint8, real=False)
    total_fid_score = (fid_obj.compute()*true_images.size(0)).item()
    vgg_lpips_obj = LearnedPerceptualImagePatchSimilarity(net_type = "vgg", reduction= 'sum').to(device)
    vgg_lpips_score = vgg_lpips_obj(reconstructed_images, true_images).item()

    torch.cuda.empty_cache()

    return psnr_val, total_multiscale_ssim, total_fid_score, vgg_lpips_score


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
        "--num_images",
        type = int,
        help = "Directory with model logs",
        default = 100
    )

    parser.add_argument(
        "-s",
        "--test_snr",
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
    LOGDIR = args.log_dir
    NUM_IMAGES = args.num_images
    TEST_SNR = args.test_snr
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    DEVICE = "cuda"

    # import pdb; pdb.set_trace()
    LOGDIR = glob.glob(os.path.join(LOGDIR, f"snr_{TEST_SNR}", "*"))[0]
    config = OmegaConf.load(glob.glob(os.path.join(LOGDIR, "configs", "*project.yaml"))[0])
    checkpoint_path = os.path.join(LOGDIR, "checkpoints", "last.ckpt")

    # latent_diff_model, _ = load_model(config, checkpoint_path, gpu = True, eval_mode = True)
    jscc_model, _ = load_model(config, checkpoint_path, gpu = True, eval_mode = True)
    scale_function = lambda img: (2*img - 1.0)
    ScaleShiftTransform = torchvision.transforms.Lambda(scale_function)
    data_transform = Compose(
        [
            Resize(size = (256, 256)),
            ToTensor(),
            ScaleShiftTransform
        ]
    )

    # import pdb; pdb.set_trace()
    dataset = LSUN(root = "val_data_lsun/lsun_beds",
                    classes = ["bedroom_val"],
                    transform = data_transform)
    
    subset = Subset(dataset, range(NUM_IMAGES))
    dataloader = DataLoader(dataset, batch_size= BATCH_SIZE, num_workers = NUM_WORKERS)

    metric_dict = {
        "PSNR": 0.0,
        "MS_SSIM": 0.0,
        "FID": 0.0,
        "LPIPS_VGG": 0.0,
    }

    total_examples = subset.__len__()

    for images, _ in tqdm(dataloader):

        images = images.to(DEVICE)
        jscc_model = jscc_model.to(DEVICE)

        reconstructed = deep_jscc_testing(jscc_model, images, TEST_SNR)
        scores = compute_metrics(images, reconstructed)

        for idx, key in enumerate(metric_dict.keys()):
            metric_dict[key] += scores[idx]
    
    for key in metric_dict.keys():
        metric_dict[key] = metric_dict[key]/total_examples
    
    filename = f"post_exp_metrics/baseline_expts/jscc_baseline_scores_{int(TEST_SNR)}_dict.pkl"

    with open(filename, "wb") as fp:
        pickle.dump(metric_dict, fp)

    