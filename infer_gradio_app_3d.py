import cv2
import torch
import numpy as np
from PIL import Image
from torch import nn
from torchvision import transforms
import math
from scipy.io import loadmat
import os
import argparse
import glob
from pathlib import Path
import time
import json
import itertools
import trimesh
from diffusers.utils import load_image
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from tqdm.auto import tqdm
from diffusers.utils.torch_utils import randn_tensor
from uvpipeline.unet_1d_condition import UNet1DConditionModel
from RGB_Fitting.utils.data_utils import read_img, img3channel, img2mask, np2pillow, pillow2np, np2tensor, tensor2np, img3channel, draw_mask, draw_landmarks, save_img
from RGB_Fitting.model import ours_fit_model
from safetensors.torch import load_file
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from models import FitDataset
import gradio as gr
from gradio_litmodel3d import LitModel3D

class IPAdapterUnet_geo(torch.nn.Module):
    """IPAdapterUnet_geo"""
    def __init__(self, unet, ckpt_path=None, mode='train'):
        super().__init__()
        self.unet = unet
        self.mode = mode
        self.load_ip_adapter_iduvidm(ckpt_path)

        if mode == 'train':
            self.unet.requires_grad_(True)
            self.unet.train()
        else:
            self.unet.requires_grad_(False)
            self.unet.eval()

    def load_ip_adapter_iduvidm(self, model_ckpt, image_emb_dim=512, num_tokens=16, scale=0.5):     
        if model_ckpt is not None:
            state_dict = torch.load(model_ckpt, map_location="cpu")
            if 'ip_adapter_unet' in state_dict:
                state_dict = state_dict['ip_adapter_unet']
                self.unet.load_state_dict(state_dict)
                print(f"Successfully loaded ip_adapter_unet from checkpoint {model_ckpt}")
            else:
                raise ValueError("ip_adapter_unet state_dict not found in checkpoint")

    def forward(self, noisy_latents, timesteps, encoder_hidden_states):
        batch = noisy_latents.shape[0]
        noise_pred = self.unet(noisy_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states).sample
        return noise_pred
    
    def save_checkpoint(self, save_path: str):
        torch.save({
            "ip_adapter_unet": self.unet.state_dict(),
        }, save_path)
        print(f"Successfully saved checkpoint to {save_path}")


class IPAdapterUnet_all(torch.nn.Module):
    """IPAdapterUnet_all"""
    def __init__(self, unet, ckpt_path=None, mode='train', ip_adapter_scale=None):
        super().__init__()
        self.unet = unet
        self.mode = mode
        self.load_ip_adapter_iduvidm(ckpt_path)

        # 0. set ip_adapter_scale
        if ip_adapter_scale is not None:
            self.set_ip_adapter_scale(ip_adapter_scale)

        if mode == 'train':
            self.image_proj_model.requires_grad_(True)
            self.unet.requires_grad_(True)
            self.image_proj_model.train()
            self.unet.train()
        else:
            self.image_proj_model.requires_grad_(False)
            self.unet.requires_grad_(False)
            self.image_proj_model.eval()
            self.unet.eval()

    def load_ip_adapter_iduvidm(self, model_ckpt, image_emb_dim=512, num_tokens=16, scale=1.0):     
        self.set_image_proj_model(model_ckpt, image_emb_dim=image_emb_dim, num_tokens=num_tokens)
        self.set_ip_adapter_unet(model_ckpt, num_tokens=num_tokens, scale=scale)
        if model_ckpt is not None:
            state_dict = torch.load(model_ckpt, map_location="cpu")
            if 'image_proj' in state_dict:
                state_dict = state_dict["image_proj"]
                self.image_proj_model.load_state_dict(state_dict)
                print(f"Successfully loaded image_proj_model from checkpoint {model_ckpt}")
            else:
                raise ValueError("image_proj_model state_dict not found in checkpoint")
            if 'ip_adapter_unet' in state_dict:
                state_dict = state_dict['ip_adapter_unet']
                self.unet.load_state_dict(state_dict)
                print(f"Successfully loaded ip_adapter_unet from checkpoint {model_ckpt}")
            else:
                raise ValueError("ip_adapter_unet state_dict not found in checkpoint")
            
    def set_image_proj_model(self, model_ckpt=None, image_emb_dim=512, num_tokens=16):
        self.image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=self.unet.config.cross_attention_dim,
            ff_mult=4,
        )

        self.image_proj_model_in_features = image_emb_dim


    def set_ip_adapter_unet(self, model_ckpt=None, num_tokens=16, scale=1.0):
        print("Initializing the Ip_adapter_unet UNet from the pretrained UNet.")
        in_channels = 8
        out_channels = self.unet.conv_in.out_channels
        self.unet.register_to_config(in_channels=in_channels)
        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels, out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride, self.unet.conv_in.padding
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            self.unet.conv_in = new_conv_in

        attn_procs = {}
        unet_sd = self.unet.state_dict()
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, 
                                                   cross_attention_dim=cross_attention_dim,
                                                   scale=scale,
                                                   num_tokens=num_tokens)
        self.unet.set_attn_processor(attn_procs)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        batch = noisy_latents.shape[0]
        image_embeds = image_embeds.reshape([batch, 1, self.image_proj_model_in_features])
        ip_tokens = self.image_proj_model(image_embeds) # [b, 1, 512] -> [b, 16, 1024]
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred
    
    def set_ip_adapter_scale(self, scale):
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        for attn_processor in unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale
    
class IPAdapterUnet_only_incomplete(torch.nn.Module):
    """IPAdapterUnet_only_incomplete"""
    def __init__(self, unet, ckpt_path=None, mode='train'):
        super().__init__()
        self.unet = unet
        self.mode = mode
        self.load_ip_adapter_iduvidm(ckpt_path)

        if mode == 'train':
            self.unet.requires_grad_(True)
            self.unet.train()
        else:
            self.unet.requires_grad_(False)
            self.unet.eval()

    def load_ip_adapter_iduvidm(self, model_ckpt):     
        self.set_ip_adapter_unet()
        if model_ckpt is not None:
            state_dict = torch.load(model_ckpt, map_location="cpu")
            if 'ip_adapter_unet' in state_dict:
                state_dict = state_dict['ip_adapter_unet']
                self.unet.load_state_dict(state_dict)
                print(f"Successfully loaded ip_adapter_unet from checkpoint {model_ckpt}")
            else:
                raise ValueError("ip_adapter_unet state_dict not found in checkpoint")

    def set_ip_adapter_unet(self):
        print("Initializing the Ip_adapter_unet UNet from the pretrained UNet.")
        in_channels = 8
        out_channels = self.unet.conv_in.out_channels
        self.unet.register_to_config(in_channels=in_channels)
        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels, out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride, self.unet.conv_in.padding
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            self.unet.conv_in = new_conv_in


    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds=None):
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred
    
    def save_checkpoint(self, save_path: str):
        torch.save({
            "ip_adapter_unet": self.unet.state_dict(),
        }, save_path)
        print(f"Successfully saved checkpoint to {save_path}")


class IPAdapterUnet_only_emb(torch.nn.Module):
    """IPAdapterUnet_only_emb"""
    def __init__(self, unet, ckpt_path=None, mode='train'):
        super().__init__()
        self.unet = unet
        self.mode = mode
        self.load_ip_adapter_iduvidm(ckpt_path)

        if mode == 'train':
            self.image_proj_model.requires_grad_(True)
            self.unet.requires_grad_(True)
            self.image_proj_model.train()
            self.unet.train()
        else:
            self.image_proj_model.requires_grad_(False)
            self.unet.requires_grad_(False)
            self.image_proj_model.eval()
            self.unet.eval()

    def load_ip_adapter_iduvidm(self, model_ckpt, image_emb_dim=512, num_tokens=16, scale=1.0):     
        self.set_image_proj_model(model_ckpt, image_emb_dim=image_emb_dim, num_tokens=num_tokens)
        self.set_ip_adapter_unet(model_ckpt, num_tokens=num_tokens, scale=scale)
        if model_ckpt is not None:
            state_dict = torch.load(model_ckpt, map_location="cpu")
            if 'image_proj' in state_dict:
                state_dict = state_dict["image_proj"]
                self.image_proj_model.load_state_dict(state_dict)
                print(f"Successfully loaded image_proj_model from checkpoint {model_ckpt}")
            else:
                raise ValueError("image_proj_model state_dict not found in checkpoint")
            if 'ip_adapter_unet' in state_dict:
                state_dict = state_dict['ip_adapter_unet']
                self.unet.load_state_dict(state_dict)
                print(f"Successfully loaded ip_adapter_unet from checkpoint {model_ckpt}")
            else:
                raise ValueError("ip_adapter_unet state_dict not found in checkpoint")

    def set_image_proj_model(self, model_ckpt=None, image_emb_dim=512, num_tokens=16):
        self.image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=self.unet.config.cross_attention_dim,
            ff_mult=4,
        )

        self.image_proj_model_in_features = image_emb_dim


    def set_ip_adapter_unet(self, model_ckpt=None, num_tokens=16, scale=1.0):
        attn_procs = {}
        unet_sd = self.unet.state_dict()
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, 
                                                   cross_attention_dim=cross_attention_dim,
                                                   scale=scale,
                                                   num_tokens=num_tokens)
                attn_procs[name].load_state_dict(weights)
        self.unet.set_attn_processor(attn_procs)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        batch = noisy_latents.shape[0]
        image_embeds = image_embeds.reshape([batch, 1, self.image_proj_model_in_features])
        ip_tokens = self.image_proj_model(image_embeds) # [b, 1, 512] -> [b, 16, 1024]
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred
    
    def save_checkpoint(self, save_path: str):
        torch.save({
            "image_proj": self.image_proj_model.state_dict(),
            "ip_adapter_unet": self.unet.state_dict(),
        }, save_path)
        print(f"Successfully saved checkpoint to {save_path}")
    
    def set_ip_adapter_scale(self, scale):
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        for attn_processor in unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

def inference_geo_null(text_input_ids, text_input_ids_null, generator, noise_scheduler, text_encoder, ipadapterunet, weight_dtype, guild_weight=7.5):
    timesteps = noise_scheduler.timesteps
    timesteps = timesteps.long()
    bs = text_input_ids.shape[0]
    shape = (bs, 1, 532)
    with torch.no_grad():
        encoder_hidden_states = text_encoder(text_input_ids)[0]
        encoder_hidden_states_null = text_encoder(text_input_ids_null)[0]

    with tqdm(total=len(timesteps)) as progress_bar:
        noisy_latents = randn_tensor(shape, generator=generator, device=text_input_ids.device, dtype=weight_dtype)
        for i, t in enumerate(timesteps):
            noise_pred = ipadapterunet(noisy_latents, t, encoder_hidden_states)
            noise_pred_null = ipadapterunet(noisy_latents, t, encoder_hidden_states_null)
            noise_pred = guild_weight * noise_pred + (1 - guild_weight) * noise_pred_null
            noisy_latents = noise_scheduler.step(noise_pred, t, noisy_latents, return_dict=False)[0]
            progress_bar.update(1)
    return noisy_latents

def inference_wo_null(cond_embed, text_input_id, incomplete, text_input_id_null,
              drop_image_embed, drop_incomplete,
              generator, noise_scheduler, vae, text_encoder, 
              ipadapterunet, weight_dtype, infer_mode, device, guild_weight):
    
    timesteps = noise_scheduler.timesteps
    timesteps = timesteps.long()

    shape = (1, 4, 64, 64)
    with torch.no_grad():
        encoder_hidden_state = text_encoder(text_input_id)[0]
        encoder_hidden_state_null = text_encoder(text_input_id_null)[0]

        cond_embed_null = torch.zeros((1, 512)).to(device, dtype=weight_dtype)
        if drop_image_embed == 1:
            cond_embed = torch.zeros((1, 512)).to(device, dtype=weight_dtype)

        if drop_incomplete == 1:
            ip2p_latent = torch.zeros(shape).to(device, dtype=weight_dtype)
            ip2p_latent_null = torch.zeros(shape).to(device, dtype=weight_dtype)
        else:
            if infer_mode == 'all' or infer_mode == 'incomplete':
                ip2p_latent = vae.encode(incomplete).latent_dist.sample()
                ip2p_latent = ip2p_latent * vae.config.scaling_factor
                ip2p_latent_null = torch.zeros(shape).to(device, dtype=weight_dtype)
                ip2p_latent_null = ip2p_latent_null * vae.config.scaling_factor
           
    with tqdm(total=len(timesteps)) as progress_bar:
        noisy_latent = randn_tensor(shape, generator=generator, device=cond_embed.device, 
                                     dtype=weight_dtype)
        # noisy_latent = noise_scheduler.add_noise(ip2p_latent, noisy_latent, torch.IntTensor([999]))
        for i, t in enumerate(timesteps):
            if infer_mode == 'all' or infer_mode == 'incomplete':
                concatenated_noisy_latent = torch.cat([noisy_latent, ip2p_latent], dim=1)
                concatenated_noisy_latent_null = torch.cat([noisy_latent, ip2p_latent_null], dim=1)
            else:
                concatenated_noisy_latent = noisy_latent
                concatenated_noisy_latent_null = noisy_latent
            noise_pred = ipadapterunet(concatenated_noisy_latent, t, encoder_hidden_state, cond_embed)
            noise_pred_null = ipadapterunet(concatenated_noisy_latent_null, t, encoder_hidden_state_null, cond_embed_null)
            noise_pred = guild_weight * noise_pred + (1 - guild_weight) * noise_pred_null
            noisy_latent = noise_scheduler.step(noise_pred, t, noisy_latent, return_dict=False)[0]
            progress_bar.update(1)

        image = vae.decode(noisy_latent/vae.config.scaling_factor, return_dict=False)[0]

    return image


def load_model(args):
    # Load scheduler, tokenizer and models.
    print(f"Loading noise_scheduler from {args.base_model_path}")
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    noise_scheduler.rescale_betas_zero_snr = True

    print(f"Loading tokenizer from {args.base_model_path}")
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model_path, subfolder="tokenizer")
    print(f"Loading text_encoder from {args.base_model_path}")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model_path, subfolder="text_encoder")
    print(f"Loading vae from {args.fintune_vae_ckpt_path}")
    vae = AutoencoderKL.from_pretrained(args.fintune_vae_ckpt_path, subfolder="vae")
    print(f"Loading unet from {args.base_model_path}")
    unet = UNet2DConditionModel.from_pretrained(args.base_model_path, subfolder="unet")

    print(f"Loading geo_unet from {args.base_model_path}")
    # self customized unet model, no need for resample
    geo_unet = UNet1DConditionModel(
        sample_size=532,
        in_channels=33,
        out_channels=1,
    )

    vae.requires_grad_(False).to(args.device)
    text_encoder.requires_grad_(False).to(args.device)
    
    if args.infer_mode == 'all':
        ipadapterunet = IPAdapterUnet_all(unet, ckpt_path=None, mode='eval', ip_adapter_scale=1.0)
        ipadapterunet.load_state_dict(load_file(args.ipadapterunet_all_ckpt_path))
        print(f"Successfully loaded ip_adapter_unet from checkpoint {args.ipadapterunet_all_ckpt_path}")

    elif args.infer_mode == 'incomplete':
        ipadapterunet = IPAdapterUnet_only_incomplete(unet, ckpt_path=None, mode='eval')
        ipadapterunet.load_state_dict(load_file(args.ipadapterunet_tex_ckpt_path))
        print(f"Successfully loaded ip_adapter_unet from checkpoint {args.ipadapterunet_tex_ckpt_path}")
    elif args.infer_mode == 'emb':
        ipadapterunet = IPAdapterUnet_only_emb(unet, ckpt_path=None, mode='eval')
        ipadapterunet.load_state_dict(load_file(args.ipadapterunet_emb_ckpt_path))
        print(f"Successfully loaded ip_adapter_unet from checkpoint {args.ipadapterunet_emb_ckpt_path}")

    ipadapterunet_geo = IPAdapterUnet_geo(geo_unet, ckpt_path=None, mode='eval')
    ipadapterunet_geo.load_state_dict(load_file(args.ipadapterunet_geo_ckpt_path))
    print(f"Successfully loaded ip_adapter_geo_unet from checkpoint {args.ipadapterunet_geo_ckpt_path}")

    ipadapterunet.to(args.device)
    ipadapterunet_geo.to(args.device)

    return noise_scheduler, tokenizer, text_encoder, vae, ipadapterunet, ipadapterunet_geo

def load_caption(text_file):
    return json.load(open(text_file, 'r'))['caption']

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--base_model_path',
                        type=str,
                        default='checkpoints/sd21',
                        help='path to the base model')
    parser.add_argument('--fintune_vae_ckpt_path',
                        type=str,
                        default='checkpoints/vae_fintuned',
                        help='path to the finetune_vae checkpoint')
    parser.add_argument('--ipadapterunet_tex_ckpt_path',
                        type=str,
                        default='checkpoints/TDM_model/model.safetensors',
                        help='path to the ipadapterunet_incomplete checkpoint')
    parser.add_argument('--ipadapterunet_geo_ckpt_path',
                        type=str,
                        default='checkpoints/GDM_model/model.safetensors')
    parser.add_argument('--RealESRGAN_ckpt_path',
                        type=str,
                        default='checkpoints/Real-ESRGAN/weights/RealESRGAN_x4plus.pth',
                        help='path to the RealESRGAN_model')
    parser.add_argument('--checkpoints_dir',
                        type=str,
                        default="checkpoints",
                        help='path to the checkpoints_dir')
    parser.add_argument('--topo_dir',
                        type=str,
                        default="assets_topo",
                        help='path to the topo_dir')
    parser.add_argument('--weight_dtype',
                        default=torch.float32,
                        help='weight dtype')
    parser.add_argument('--num_inference_steps',
                        type=int,
                        default=50,
                        help='number of inference steps')
    parser.add_argument('--infer_mode',
                        type=str,
                        default='incomplete',
                        help='infer_mode')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device cuda/cpu')
    parser.add_argument('--guild_weight_geo', type=int, default=1.5, help='guild_weight')
    parser.add_argument('--guild_weight_tex', type=int, default=6, help='guild_weight')
    args = parser.parse_args()

    return args

def initial():
    args = get_args()

    dataset_op = FitDataset(parsing_model_pth=os.path.join(args.checkpoints_dir, 'parsing_model/79999_iter.pth'),
                        parsing_resnet18_path=os.path.join(args.checkpoints_dir,
                                                            'resnet_model/resnet18-5c106cde.pth'),
                        lm68_3d_path=os.path.join(args.topo_dir, 'similarity_Lm3D_all.mat'),
                        unwrap_info_path = os.path.join(args.topo_dir, 'unwrap_1024_info.mat'),
                        unwrap_info_mask_path = os.path.join(args.topo_dir, 'unwrap_1024_info_mask.png'),
                        pfm_model_path = os.path.join(args.topo_dir, 'hifi3dpp_model_info.mat'),
                        recon_model_path = os.path.join(args.checkpoints_dir, 'deep3d_model/epoch_latest.pth'),
                        face_analysis_model_path = os.path.join(args.checkpoints_dir, 'face_analysis_model_path'),
                        focal=1015.0,
                        camera_distance=10.0,
                        batch_size=1,
                        device=args.device
                        )
    
    rec_model = ours_fit_model.RecModel(args.checkpoints_dir, args.topo_dir, device=args.device)  

    # restorer
    print("Loading RealESRGANer ...")
    RealESRGANer_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    RealESRGANer_upsampler = RealESRGANer(
        scale=4,
        model_path=args.RealESRGAN_ckpt_path,
        dni_weight=None,
        model=RealESRGANer_model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=None)  

    # ----------------- 1. Load the model -----------------
    # Load scheduler, tokenizer and models.
    noise_scheduler, tokenizer, text_encoder, vae, ipadapterunet, ip_adapter_geo_unet = load_model(args)

    return args, dataset_op, rec_model, RealESRGANer_upsampler, noise_scheduler, tokenizer, text_encoder, vae, ipadapterunet, ip_adapter_geo_unet


def run_text(args, prompt_base, prompt_tex, prompt_geo, rec_model, RealESRGANer_upsampler, noise_scheduler, tokenizer, text_encoder, vae, ipadapterunet, ip_adapter_geo_unet, seed, save_dir):

    tic = time.time()
    save_dir_sub = os.path.join(save_dir, f"text2avatar_{args.num_inference_steps}")
    os.makedirs(save_dir_sub, exist_ok=True)
    saved_dirs = os.listdir(save_dir_sub)
    ind = len(saved_dirs)
    rec_dir = os.path.join(save_dir_sub,f"{ind}")
    os.makedirs(rec_dir, exist_ok=True)

    # Load image
    face_emb = None
    incomplete = None
    input_data = None
    drop_image_embed = 1
    drop_incomplete = 1

    # save prompt to json
    with open(os.path.join(rec_dir, f"prompt.json"), 'w') as f:
        json.dump({"caption": 
                        {'base': prompt_base, 
                        'tex': prompt_tex, 
                        'geo': prompt_geo}}, f)

    prompt_tex = prompt_base + '' + prompt_tex
    prompt_geo = prompt_base + '' + prompt_geo
    print(f"prompt_tex: {prompt_tex}")
    print(f"prompt_geo: {prompt_geo}")
    
    text_input_ids = tokenizer(prompt_tex, max_length=77, padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids
    text_input_ids = text_input_ids.unsqueeze(0).to(args.device)

    text_input_ids_geo = tokenizer(prompt_geo, max_length=77, padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids
    text_input_ids_geo = text_input_ids_geo.unsqueeze(0).to(args.device)

    prompt_null = ""
    text_input_ids_null = tokenizer(prompt_null, max_length=77, padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids
    text_input_ids_null = text_input_ids_null.unsqueeze(0).to(args.device)
    
    # Set the timesteps
    noise_scheduler.set_timesteps(num_inference_steps=args.num_inference_steps, device=args.device)

    # Set the random seed
    generator = torch.Generator(device=args.device).manual_seed(seed)

    # Generate image
    with torch.autocast("cuda"):
        with torch.no_grad():
            id_coeffs_weight = inference_geo_null(text_input_ids_geo, text_input_ids_null, generator=generator,noise_scheduler=noise_scheduler, text_encoder=text_encoder,ipadapterunet=ip_adapter_geo_unet, weight_dtype=args.weight_dtype, guild_weight=args.guild_weight_geo)

            image_weight = inference_wo_null(face_emb, text_input_ids,incomplete,  text_input_ids_null, drop_image_embed=1, drop_incomplete=1,
                                generator=generator, noise_scheduler=noise_scheduler,
                                vae=vae, text_encoder=text_encoder, ipadapterunet=ipadapterunet, 
                                weight_dtype=args.weight_dtype,
                                infer_mode=args.infer_mode,
                                device=args.device,
                                guild_weight=args.guild_weight_tex)

            pred_uv_map_weight = image_weight*0.5+0.5

            # RealESRGANer
            pred_uv_map_numpy_weight = tensor2np(pred_uv_map_weight)

            pred_uv_map_numpy_2K_weight, _ = RealESRGANer_upsampler.enhance(pred_uv_map_numpy_weight[:,:,::-1], outscale=4)

            pred_uv_map_tensor_2K_weight = np2tensor(pred_uv_map_numpy_2K_weight[:,:,::-1], device=args.device)

            rec_model.rec_fitting(input_data=input_data, rec_dir=rec_dir, uv_map=pred_uv_map_tensor_2K_weight, id_coeffs=id_coeffs_weight.squeeze_(1), stage=4)

            id_rendering = rec_model.rec_id_rendering(rec_dir=rec_dir, uv_map=pred_uv_map_tensor_2K_weight, id_coeffs=id_coeffs_weight.squeeze_(1), stage=4)

    toc = time.time()
    print(f"Processing {ind} took {toc-tic:.2f} seconds")

    return np2pillow(pred_uv_map_numpy_2K_weight[:,:,::-1]), np2pillow(id_rendering), rec_dir, os.path.join(rec_dir, "stage4_mesh_id.obj")


def run_image(args, img, prompt_base, prompt_tex, dataset_op, rec_model, RealESRGANer_upsampler, noise_scheduler, tokenizer, text_encoder, vae, ipadapterunet, seed, save_dir):
    
    tic = time.time()
    save_dir_sub = os.path.join(save_dir, f"image2avatar_{args.num_inference_steps}")
    os.makedirs(save_dir_sub, exist_ok=True)
    saved_dirs = os.listdir(save_dir_sub)
    ind = len(saved_dirs)
    rec_dir = os.path.join(save_dir_sub,f"{ind}")
    os.makedirs(rec_dir, exist_ok=True)

    # Load image
    drop_image_embed = 0
    drop_incomplete = 0

    save_input_data_path = os.path.join(rec_dir, "input_data.pt")
    if os.path.exists(save_input_data_path):
        input_data = torch.load(save_input_data_path)
    else:
        input_data = dataset_op.get_input_data(img)
        if input_data is not None:
            torch.save(input_data, save_input_data_path)
    if input_data is None:
        log_text = f"No face detected"
        print(f"No face detected")
        return None, None, log_text

    # save prompt to json
    with open(os.path.join(rec_dir, f"prompt.json"), 'w') as f:
        json.dump({"caption": 
                        {'base': prompt_base, 
                        'tex': prompt_tex}}, f)

    prompt_tex = prompt_base + '' + prompt_tex
    # prompt_tex = ""
    print(f"prompt_tex: {prompt_tex}")

    text_input_ids = tokenizer(
        prompt_tex,
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids
    text_input_ids = text_input_ids.unsqueeze(0).to(args.device)
    prompt_null = ''
    text_input_ids_null = tokenizer(
        prompt_null,
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids
    text_input_ids_null = text_input_ids_null.unsqueeze(0).to(args.device)

    face_emb = input_data['face_emb']
    face_emb = torch.tensor(face_emb).unsqueeze(0).to(args.device)

    incomplete_pillow = input_data['incomplete']
    save_incomplete_path = os.path.join(rec_dir, "incomplete.png")
    incomplete_pillow.save(save_incomplete_path)

    data_transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    incomplete = data_transform(incomplete_pillow).unsqueeze(0).to(args.device)
    
    # Set the timesteps
    noise_scheduler.set_timesteps(num_inference_steps=args.num_inference_steps, device=face_emb.device)

    # Set the random seed
    generator = torch.Generator(device=face_emb.device).manual_seed(seed)

    # Generate image
    with torch.autocast("cuda"):
        with torch.no_grad():
            image_weight = inference_wo_null(face_emb, text_input_ids,incomplete,  text_input_ids_null, drop_image_embed=0, drop_incomplete=0,
                                    generator=generator, noise_scheduler=noise_scheduler,
                                    vae=vae, text_encoder=text_encoder, ipadapterunet=ipadapterunet, 
                                    weight_dtype=args.weight_dtype,
                                    infer_mode=args.infer_mode,
                                    device=args.device,
                                    guild_weight=args.guild_weight_tex)
        
        pred_uv_map_weight = image_weight*0.5+0.5
        pred_uv_map_numpy_weight = tensor2np(pred_uv_map_weight)
        pred_uv_map_numpy_2K_weight, _ = RealESRGANer_upsampler.enhance(pred_uv_map_numpy_weight[:,:,::-1], outscale=4)
        pred_uv_map_tensor_2K_weight = np2tensor(pred_uv_map_numpy_2K_weight[:,:,::-1], device=args.device)
        rec_model.rec_opitming_gr(input_data=input_data, rec_dir=rec_dir, uv_map_hr=pred_uv_map_tensor_2K_weight, stage=1, is_opt=False)

        id_rendering = rec_model.rec_id_rendering(rec_dir=rec_dir, uv_map=pred_uv_map_tensor_2K_weight, id_coeffs=None, stage=5)


    toc = time.time()
    print(f'Fit image: done, took {toc - tic:.4f} seconds.')

    return np2pillow(pred_uv_map_numpy_2K_weight[:,:,::-1]), np2pillow(id_rendering), rec_dir

def read_obj_file(file_name):
    # 读取上传的.obj文件内容
    with open(file_name, "r") as f:
        obj_content = f.read()
    return obj_content


def get_image_example():
        case = [
            [
                './examples/1004.jpg',
                "",
                "",
                50,
                6.5,
                42,
                "output_examples",
            ],
            [
                './examples/9341.jpg',
                "",
                "",
                50,
                6.5,
                42,
                "output_examples",
            ],
            [
                './examples/9435.jpg',
                "",
                "",
                50,
                6.5,
                42,
                "output_examples",
            ],
            [
                './examples/17486.jpg',
                "",
                "",
                50,
                6.5,
                42,
                "output_examples",
            ],
            [
                './examples/Einstein.png',
                "",
                "",
                50,
                6.5,
                42,
                "output_examples",
            ],
                [
                './examples/smith.jpeg',
                "",
                "",
                50,
                6.5,
                42,
                "output_examples",
            ],
        ]
        return case
    
def get_text_example():
        case = [
            [
                "A man in his 40s with a Caucasian appearance",
                "has olive skin with a smooth and clear complexion, facial hair around the mouth, chin, and jawline",
                "an oval face shape, straight eyebrows, a prominent nose with a rounded tip, and a round chin.",
                50,6.5,1.5,42,"output_examples",
            ],
            [
                "A young Caucasian female.",
                "She has light skin, thin eyebrows, subtle laugh lines near the mouth, faint dark circles under the eyes, rosy and slightly chapped lips.",
                "a square face and a square chin.",
                50,6.5,1.5,42,"output_examples",
            ],
            [
                "An adult male of Asian descent in his 20s.",
                "has a yellow skin tone, smooth skin",
                "a square-shaped face with overall balanced proportions, and arched eyebows",
                50,6.5,1.5,42,"output_examples",
            ],            
        ]
        return case

### Description
title = r"""
<h1 align="center">🧙‍♂️ PromptAvatar: Text-Image Prompted Generation of 3D Animatable Avatars 🧙‍♀️</h1>
"""

description_image = r"""
<style>
    .demo-description {
        font-size: 1.2em;
        color: #e0e6ed;
        margin-bottom: 20px;
        font-weight: bold;
        text-align: center;
    }
    .instructions {
        background-color: #1e2a38;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3b4a5a;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    .instruction-step {
        margin-bottom: 15px;
        font-size: 1.05em;
        color: #cfd8e3;
    }
    .instruction-step b {
        color: #60a3ff;
    }
    .note {
        font-style: italic;
        color: #a8b2bf;
        margin-top: 25px;
        background-color: #2a3647;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3b4a5a;
        box-shadow: 0 1px 5px rgba(0, 0, 0, 0.2);
    }
    .note p {
        margin: 0;
    }
</style>

<div class="demo-description">
    🤗 Gradio Demo for PromptAvatar: Create amazing 3D Animatable Avatars with text and image prompts! 🎭✨
</div>

<div class="instructions">
    <h3>🚀 How to use:</h3>
    <p class="instruction-step">1. 📸 Upload an image with a face. For group photos, we'll focus on the largest face. Make sure it's clear and visible!</p>
    <p class="instruction-step">2. 🖱️ Click the <b>Generate</b> button to start crafting your 3D Avatar. 😊</p>
    <p class="instruction-step">3. 💾 Find your .obj and .glb files in the save directory. Magic!</p>
</div>

<div class="note">
    <p>⚠️ Please note: The 3D model might have some quirks at the back of the head and mouth. Don't worry, it's just the texture coordinates being playful!</p>
    <p>🔍 For the best view, check out the perfect OBJ file in MeshLab or Blender. It's like HD for 3D! 🖥️✨</p>
</div>
"""

description_text = r"""
<style>
    .demo-description {
        font-size: 1.2em;
        color: #e0e6ed;
        margin-bottom: 20px;
        font-weight: bold;
        text-align: center;
    }
    .instructions {
        background-color: #1e2a38;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3b4a5a;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    .instruction-step {
        margin-bottom: 15px;
        font-size: 1.05em;
        color: #cfd8e3;
    }
    .instruction-step b {
        color: #60a3ff;
    }
    .note {
        font-style: italic;
        color: #a8b2bf;
        margin-top: 25px;
        background-color: #2a3647;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3b4a5a;
        box-shadow: 0 1px 5px rgba(0, 0, 0, 0.2);
    }
    .note p {
        margin: 0;
    }
</style>
<div class="demo-description">
    <b>🤗 Gradio Demo</b> for PromptAvatar: Text-Image Prompted Generation of 3D Animatable Avatars.
</div>

<div class="instructions">
    <h3>🚀 How to use:</h3>
    <p class="instruction-step">1. Please specify the base, texture, and geometry prompts separately. You can refer to the examples provided for specific formats.</p>
    <p class="instruction-step">2. Click the <b>"Generate"</b> button to start customizing your 3D avatar. 😊</p>
    <p class="instruction-step">3. The .obj and .glb files will be saved to the <b>save_dir</b> directory.</p>
</div>

<div class="note">
    <p>⚠️ Please note: The 3D model may show imperfections at the back of the head and mouth. This is due to the texture coordinate indexing order and does not affect the actual model quality.</p>
    <p>🔍 For the best representation, we provide a perfect OBJ file that can be viewed in MeshLab or Blender.</p>
</div>

"""

def run_gradio():
    args, dataset_op, rec_model, RealESRGANer_upsampler, noise_scheduler, tokenizer, text_encoder, vae, ipadapterunet, ip_adapter_geo_unet = initial()

    # 初始化点击次数计数器
    def gradio_run_text_inference(prompt_base, prompt_tex, prompt_geo, num_inference_steps, guild_weight_tex, guild_weight_geo, seed, save_dir):
        args.num_inference_steps = num_inference_steps
        args.guild_weight_tex = guild_weight_tex
        args.guild_weight_geo = guild_weight_geo
        uvmap, id_rendering, save_path, id_obj_path = run_text(args, prompt_base, prompt_tex, prompt_geo, rec_model, RealESRGANer_upsampler, noise_scheduler, tokenizer, text_encoder, vae, ipadapterunet, ip_adapter_geo_unet, seed, save_dir)
        # return uvmap, id_rendering, save_path, id_obj_path
        mesh_name = os.path.join(save_path, "stage4_mesh_id.glb")
        return uvmap, gr.update(value=mesh_name, visible=True), save_path

    
    def gradio_run_image_inference(img, prompt_base, prompt_tex, num_inference_steps, guild_weight_tex, seed, save_dir):
        args.num_inference_steps = num_inference_steps
        args.guild_weight_tex = guild_weight_tex
        uvmap, id_rendering, save_path = run_image(args, img, prompt_base, prompt_tex, dataset_op, rec_model, RealESRGANer_upsampler, noise_scheduler, tokenizer, text_encoder, vae, ipadapterunet, seed, save_dir)
        # return uvmap, id_rendering, save_path

        mesh_name = os.path.join(save_path, "stage2_mesh_id.glb")
        return uvmap, gr.update(value=mesh_name, visible=True), save_path
    
    css = '''
    .gradio-container {width: 85% !important}
    '''
    with gr.Blocks(css=css) as demo:
        # gr.Markdown("# PromptAvatar: Text-Image Prompted Generation of 3D Animatable Avatars")
        gr.Markdown(title)
        # gr.desc
        with gr.Tab(label="Image-to-Avatar"):
            gr.Markdown(description_image)
            with gr.Row():
                with gr.Column(scale=1):
                    image_upload = gr.Image(label="facial_image_upload", type="pil", image_mode="RGB")
                with gr.Column(scale=2):
                    save_dir = gr.Textbox(label = "save_dir", value="output_examples/gradio")
            btn = gr.Button("Generate")

            with gr.Accordion(label="Advanced Options", open=False):
                with gr.Column():
                    prompt_base = gr.Textbox(label = "base_prompt", value="", visible=True)
                    prompt_tex = gr.Textbox(label = "prompt_tex", value="", visible=True)
                    num_inference_steps = gr.Slider(value=50, label="num_inference_steps", maximum=1000, minimum=5, step=1)
                    guild_weight_tex = gr.Slider(value=6., label="guild_weight_tex", maximum=20., minimum=0., step=0.1)
                    seed = gr.Slider(value=args.seed, label="seed", step=1)
                    # guild_weight_geo = gr.Slider(value=1.5, label="guild_weight_geo", maximum=20., minimum=0., step=0.1)   
            with gr.Column():
                with gr.Row():
                    uvmap = gr.Image(value=None, label="pred_uv_map_weight", type="pil")
                    # id_rendering = gr.Image(value=None, label="nvdiffrast_rendering", type="pil")
                    output_3d = LitModel3D(
                        label="3D Model",
                        visible=True,
                        clear_color=[0.0, 0.0, 0.0, 0.0],
                        tonemapping="aces",
                        contrast=1.0,
                        scale=1.0,
                    )
                with gr.Row():
                    save_path = gr.Textbox(label="Avatar generated successfully, saved to ", value="", interactive=False)
                    with gr.Column(visible=True, scale=1.0) as hdr_row:
                        gr.Markdown("""## HDR Environment Map

                        Select an HDR environment map to light the 3D model. You can also upload your own HDR environment maps.
                        """)

                        with gr.Row():
                            hdr_illumination_file = gr.File(
                                label="HDR Env Map", file_types=[".hdr"], file_count="single"
                            )
                            example_hdris = [
                                os.path.join("examples/hdri", f)
                                for f in os.listdir("examples/hdri")
                            ]
                            hdr_illumination_example = gr.Examples(
                                examples=example_hdris,
                                inputs=hdr_illumination_file,
                            )

                            hdr_illumination_file.change(
                                lambda x: gr.update(env_map=x.name if x is not None else None),
                                inputs=hdr_illumination_file,
                                outputs=[output_3d],
                            )
            btn.click(fn=gradio_run_image_inference, inputs=[image_upload, prompt_base, prompt_tex, num_inference_steps, guild_weight_tex, seed, save_dir], outputs=[uvmap, output_3d, save_path])

            gr.Examples(
                examples=get_image_example(),
                inputs=[image_upload, prompt_base, prompt_tex, num_inference_steps, guild_weight_tex, seed, save_dir],
                run_on_click=True,
                fn=gradio_run_image_inference,
                outputs=[uvmap, output_3d, save_path],
                cache_examples=True,
            )
            
        with gr.Tab(label="Text-to-Avatar"):
            gr.Markdown(description_text)
            with gr.Row():
                with gr.Column(scale=4):
                    prompt_base = gr.Textbox(label = "base_prompt", value="")

                    prompt_tex = gr.Textbox(label = "prompt_tex", value="")

                    prompt_geo = gr.Textbox(label = "prompt_geo", value="")

                with gr.Column(scale=1, min_width="50"):
                    save_dir = gr.Textbox(label = "save_dir", value="output_examples/gradio")
                    btn = gr.Button("Generate")
            
            with gr.Accordion(label="Advanced Options", open=False):
                with gr.Column():
                    num_inference_steps = gr.Slider(value=50, label="num_inference_steps", maximum=1000, minimum=5, step=1)
                    guild_weight_tex = gr.Slider(value=6., label="guild_weight_tex", maximum=20., minimum=0., step=0.1)
                    guild_weight_geo = gr.Slider(value=1.5, label="guild_weight_geo", maximum=20., minimum=0., step=0.1)
                    seed = gr.Slider(value=args.seed, label="seed", step=1)
            
            with gr.Column():
                with gr.Row():
                    uvmap = gr.Image(value=None, label="pred_uv_map_weight", type="pil")
                    # id_rendering = gr.Image(value=None, label="nvdiffrast_rendering", type="pil")
                    output_3d = LitModel3D(
                        label="3D Model",
                        visible=True,
                        clear_color=[0.0, 0.0, 0.0, 0.0],
                        tonemapping="aces",
                        contrast=1.0,
                        scale=1.0,
                    )
                with gr.Row():
                    save_path = gr.Textbox(label="Avatar generated successfully, saved to ", value="", interactive=False)

                    with gr.Column(visible=True, scale=1.0) as hdr_row:
                        gr.Markdown("""## HDR Environment Map

                        Select an HDR environment map to light the 3D model. You can also upload your own HDR environment maps.
                        """)

                        with gr.Row():
                            hdr_illumination_file = gr.File(
                                label="HDR Env Map", file_types=[".hdr"], file_count="single"
                            )
                            example_hdris = [
                                os.path.join("examples/hdri", f)
                                for f in os.listdir("examples/hdri")
                            ]
                            hdr_illumination_example = gr.Examples(
                                examples=example_hdris,
                                inputs=hdr_illumination_file,
                            )

                            hdr_illumination_file.change(
                                lambda x: gr.update(env_map=x.name if x is not None else None),
                                inputs=hdr_illumination_file,
                                outputs=[output_3d],
                            )

            btn.click(fn=gradio_run_text_inference, inputs=[prompt_base, prompt_tex, prompt_geo, num_inference_steps, guild_weight_tex, guild_weight_geo, seed, save_dir], outputs=[uvmap, output_3d, save_path])

            gr.Examples(
                examples=get_text_example(),
                inputs=[prompt_base, prompt_tex, prompt_geo, num_inference_steps, guild_weight_tex, guild_weight_geo, seed, save_dir],
                run_on_click=True,
                fn=gradio_run_text_inference,
                outputs=[uvmap, output_3d, save_path],
                cache_examples=True,
            )
        

    demo.launch(share=False, server_port=8011)

if __name__ == '__main__':
    os.environ['GRADIO_TEMP_DIR'] = 'tmp'
    run_gradio()
    