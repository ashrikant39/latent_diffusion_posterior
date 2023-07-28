import numpy as np
import torch
from PIL import Image
from scripts.sample_diffusion import custom_to_pil, load_model

from ldm.models.autoencoder import AutoEncoder_JSCC, VQModel

from omegaconf import OmegaConf


if __name__ == "__main__":

    conf = OmegaConf.load("/nfs/turbo/coe-hunseok/ashri/lsun_data/loggings/jscc_models_finetune/snr_10/2023-07-05T18-07-12_jscc_64x64x3/configs/2023-07-05T18-07-12-project.yaml")
    model, _ = load_model(conf, ckpt="/nfs/turbo/coe-hunseok/ashri/lsun_data/loggings/jscc_models_updated/snr_10/2023-07-07T01-43-10_jscc_64x64x3/checkpoints/last.ckpt", gpu = True, eval_mode=True)
    
    # import pdb; pdb.set_trace()
    
    im = Image.open("images/test/test_in.webp")
    img = np.array(im).astype(np.uint8)
    img = (img / 127.5 - 1.0).astype(np.float32)
    img = img[:,256:512,:]
    img = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).cuda()
    out = model(img)
    out = custom_to_pil(out[0,:,:,:])
    out.save("images/test/test_jscc_out.png")

    im = custom_to_pil(img[0,:,:,:])
    im.save("images/test/test_input.png")
