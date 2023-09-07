import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
from packaging import version
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
import sys, os
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from torch.optim.lr_scheduler import LambdaLR
from ldm.util import instantiate_from_config
import wandb

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import multiscale_structural_similarity_index_measure, psnr
from scripts.sample_diffusion import load_model
from omegaconf import OmegaConf

def normalize_tensor(x:torch.FloatTensor):
    normalized = (x - x.min())/(x.max() - x.min())
    return 2*normalized - 1.0

def schedule_function(step):
    if step%40 == 39:
        return 0.8
    else:
        return 1.0

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        # import pdb; pdb.set_trace()
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        # if len(missing) > 0:
        #     print(f"Missing Keys: {missing}")
        #     print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            # scheduler = instantiate_from_config(self.scheduler_config)
            scheduler = torch.nn.optim.StepLR()
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class AutoEncoder_JSCC(VQModel):
    def __init__(self, channel_SNR_dB, checkpoint_path, freeze_modules, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channel_snr = channel_SNR_dB
        print(f"Loading model from {checkpoint_path}...")        
        pretrined_autoencoder_state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(pretrined_autoencoder_state_dict)

        self.quantize = torch.nn.Identity()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Params: {total_params}")
        print(f"Total Trainable Params: {trainable_params}")


    def encode_through_channel(self, x, test_snr = None):

        latent_signal = self.quant_conv(self.encoder(x))
        signal_power = torch.mean(latent_signal**2)
        snr = test_snr if test_snr is not None else self.channel_snr
        noise_power = signal_power/(10**(snr/10))
        noisy_latent = latent_signal + torch.randn_like(latent_signal)*torch.sqrt(noise_power)
        return noisy_latent

    def decode_noisy_latent(self, h):
        return self.decoder(self.post_quant_conv(h))

    def forward(self, input):
        noisy_latent = self.encode_through_channel(input)
        decoded = self.decode_noisy_latent(noisy_latent)
        return decoded
    
    def configure_optimizers(self):
            lr_d = self.learning_rate
            lr_g = self.lr_g_factor*self.learning_rate
            print("lr_d", lr_d)
            print("lr_g", lr_g)

            parameters = list(self.encoder.parameters()) + \
                         list(self.decoder.parameters()) + \
                         list(self.quant_conv.parameters()) + \
                         list(self.post_quant_conv.parameters())
            opt_ae = torch.optim.Adam(parameters, lr=lr_g, betas=(0.5, 0.9))
            # opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
            #                             lr=lr_d, betas=(0.5, 0.9))

            if self.scheduler_config is None:
                # scheduler = instantiate_from_config(self.scheduler_config)

                print("Setting up LambdaLR scheduler...")
                scheduler = [
                    {
                        'scheduler': LambdaLR(opt_ae, lr_lambda=schedule_function),
                        'interval': 'step',
                        'frequency': 1
                    }
                ]
                return [opt_ae], scheduler
            return [opt_ae], []

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec = self(x)

        aeloss = self.loss(xrec, x)
        log_dict_ae = dict()
        lpip_obj = LearnedPerceptualImagePatchSimilarity(net_type = "vgg").to(device = x.device)
        log_dict_ae["train/rec_loss"] = aeloss.detach()
        log_dict_ae["train/rec_psnr"] = psnr(normalize_tensor(xrec.detach()), x)
        log_dict_ae["train/rec_lpips"] = lpip_obj(normalize_tensor(xrec.detach()), x)

        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch)
        # with self.ema_scope():
        #     log_dict_ema = self._validation_step(batch)
        return log_dict

    def _validation_step(self, batch):

        x = self.get_input(batch, self.image_key)
        xrec = self(x)
        aeloss = self.loss(xrec, x)
        
        log_dict_ae = dict()
        lpip_obj = LearnedPerceptualImagePatchSimilarity(net_type = "vgg").to(device = x.device)

        log_dict_ae["val/rec_loss"] = aeloss.detach()
        log_dict_ae["val/rec_psnr"] = psnr(normalize_tensor(xrec.detach()), x)
        log_dict_ae["val/rec_lpips"] = lpip_obj(normalize_tensor(xrec.detach()), x)

        self.log(f"val/rec_loss", aeloss,
                prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"val/psnr", log_dict_ae["val/rec_psnr"],
                prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val/rec_loss"]
        self.log_dict(log_dict_ae)
        return self.log_dict
    
    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log


class AutoEncoder(VQModel):

    def __init__(self, checkpoint_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print(f"Loading model from {checkpoint_path}...")        
        pretrined_autoencoder_state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(pretrined_autoencoder_state_dict)
        self.quantize = torch.nn.Identity()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Params: {total_params}")
        print(f"Total Trainable Params: {trainable_params}")


    def encode(self, x):
        return self.quant_conv(self.encoder(x))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, input):
        latent = self.encode(input)
        decoded = self.decode(latent)
        aeloss = self.loss(input, decoded)
        return latent, decoded, aeloss
    
    def configure_optimizers(self):
            lr_d = self.learning_rate
            lr_g = self.lr_g_factor*self.learning_rate
            print("lr_d", lr_d)
            print("lr_g", lr_g)

            parameters = list(self.encoder.parameters()) + \
                         list(self.decoder.parameters()) + \
                         list(self.quant_conv.parameters()) + \
                         list(self.post_quant_conv.parameters())
            opt_ae = torch.optim.Adam(parameters, lr=lr_g, betas=(0.5, 0.9))

            if self.scheduler_config is None:
                print("Setting up LambdaLR scheduler...")
                scheduler = [
                    {
                        'scheduler': LambdaLR(opt_ae, lr_lambda=schedule_function),
                        'interval': 'step',
                        'frequency': 1
                    }
                ]
                return [opt_ae], scheduler
            return [opt_ae], []

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        _, xrec, aeloss = self(x)

        log_dict_ae = dict()
        lpip_obj = LearnedPerceptualImagePatchSimilarity(net_type = "vgg").to(device = x.device)
        log_dict_ae["train/rec_loss"] = aeloss.detach()
        log_dict_ae["train/rec_psnr"] = psnr(normalize_tensor(xrec.detach()), x)
        log_dict_ae["train/rec_lpips"] = lpip_obj(normalize_tensor(xrec.detach()), x)

        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch)
        # with self.ema_scope():
        #     log_dict_ema = self._validation_step(batch)
        return log_dict

    def _validation_step(self, batch):

        x = self.get_input(batch, self.image_key)
        _, xrec, aeloss = self(x)
        
        log_dict_ae = dict()
        lpip_obj = LearnedPerceptualImagePatchSimilarity(net_type = "vgg").to(device = x.device)

        log_dict_ae["val/rec_loss"] = aeloss.detach()
        log_dict_ae["val/rec_psnr"] = psnr(normalize_tensor(xrec.detach()), x)
        log_dict_ae["val/rec_lpips"] = lpip_obj(normalize_tensor(xrec.detach()), x)

        self.log(f"val/rec_loss", aeloss,
                prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"val/psnr", log_dict_ae["val/rec_psnr"],
                prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val/rec_loss"]
        self.log_dict(log_dict_ae)
        return self.log_dict
    
    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log


class VQModel_JSCC(VQModel):

    def __init__(self, checkpoint_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print(f"Loading model from {checkpoint_path}...")        
        pretrined_autoencoder_state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(pretrined_autoencoder_state_dict)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Params: {total_params}")
        print(f"Total Trainable Params: {trainable_params}")

    
    def configure_optimizers(self):
            lr_d = self.learning_rate
            lr_g = self.lr_g_factor*self.learning_rate
            print("lr_d", lr_d)
            print("lr_g", lr_g)

            parameters = list(self.encoder.parameters()) + \
                         list(self.decoder.parameters()) + \
                         list(self.quant_conv.parameters()) + \
                         list(self.post_quant_conv.parameters()) +\
                         list(self.quantize.parameters())
            
            opt_ae = torch.optim.Adam(parameters, lr=lr_g, betas=(0.5, 0.9))

            if self.scheduler_config is None:
                print("Setting up LambdaLR scheduler...")
                scheduler = [
                    {
                        'scheduler': LambdaLR(opt_ae, lr_lambda=schedule_function),
                        'interval': 'step',
                        'frequency': 1
                    }
                ]
                return [opt_ae], scheduler
            return [opt_ae], []

    def forward(self, input, return_pred_indices=False):

        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff


    def training_step(self, batch, batch_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss = self.loss(x, xrec)
        log_dict_ae = dict()
        lpip_obj = LearnedPerceptualImagePatchSimilarity(net_type = "vgg").to(device = x.device)
        log_dict_ae["train/rec_loss"] = aeloss.detach()
        log_dict_ae["train/embed_loss"] = qloss.detach()
        log_dict_ae["train/rec_psnr"] = psnr(normalize_tensor(xrec.detach()), x)
        log_dict_ae["train/rec_lpips"] = lpip_obj(normalize_tensor(xrec.detach()), x)

        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch)
        # with self.ema_scope():
        #     log_dict_ema = self._validation_step(batch)
        return log_dict

    def _validation_step(self, batch):

        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss = self.loss(x , xrec)
        log_dict_ae = dict()
        lpip_obj = LearnedPerceptualImagePatchSimilarity(net_type = "vgg").to(device = x.device)

        log_dict_ae["val/rec_loss"] = aeloss.detach()
        log_dict_ae["val/embed_loss"] = qloss.detach()
        log_dict_ae["val/rec_psnr"] = psnr(normalize_tensor(xrec.detach()), x)
        log_dict_ae["val/rec_lpips"] = lpip_obj(normalize_tensor(xrec.detach()), x)

        self.log(f"val/rec_loss", aeloss,
                prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"val/psnr", log_dict_ae["val/rec_psnr"],
                prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val/rec_loss"]
        self.log_dict(log_dict_ae)
        return self.log_dict
    
    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log    

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
    




