from stable_diffusion.sd_15.clip import CLIP
from stable_diffusion.sd_15.encoder import VAE_Encoder
from stable_diffusion.sd_15.decoder import VAE_Decoder
from stable_diffusion.sd_15.diffuser.unet import Diffusion

import stable_diffusion.sd_15.converter as converter

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }