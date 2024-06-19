from dataclasses import dataclass

from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionInstructPix2PixPipeline
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from tqdm import tqdm
import math
import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *

from threestudio.models.ip_adapter import IPAdapterXL
from threestudio.models.ip_adapter.utils import (
    register_cross_attention_hook,
    get_net_attn_map,
    attnmaps2images,
    get_generator
)
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel

from threestudio.utils.dge_utils import register_pivotal, register_batch_idx, register_cams, register_epipolar_constrains, register_extended_attention, register_normal_attention, register_extended_attention, make_dge_block, isinstance_str, compute_epipolar_constrains

@threestudio.register("dge-guidance")
class DGEGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        ddim_scheduler_name_or_path: str = "CompVis/stable-diffusion-v1-4"
        ip2p_name_or_path: str = "timbrooks/instruct-pix2pix"

        # zest params
        base_model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
        image_encoder_path: str = "/data2/manan/zest_code/models/image_encoder"
        ip_ckpt: str = "/data2/manan/zest_code/sdxl_models/ip-adapter_sdxl_vit-h.bin"
        controlnet_path: str = "diffusers/controlnet-depth-sdxl-1.0"
        controlnet_conditioning_scale: float = 0.9
        num_inference_steps: int = 30

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float = 1.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True
        fixed_size: int = -1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        diffusion_steps: int = 20
        use_sds: bool = False
        camera_batch_size: int = 5

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading InstructPix2Pix ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.cfg.ip2p_name_or_path, **pipe_kwargs
        ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        # dtype is torch.float16 !! 
        # For encoding, both vae and input should be of max precision! (Refer : https://github.com/huggingface/diffusers/issues/7188)
        self.vae = self.pipe.vae.eval()         
        self.vae = self.vae.to(torch.float32)       # Convert to torch.float32 for encoding 
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded InstructPix2Pix!")
        for _, module in self.unet.named_modules():
            if isinstance_str(module, "BasicTransformerBlock"):
                make_block_fn = make_dge_block 
                module.__class__ = make_block_fn(module.__class__)
                # Something needed for older versions of diffusers
                if not hasattr(module, "use_ada_layer_norm_zero"):
                    module.use_ada_layer_norm = False
                    module.use_ada_layer_norm_zero = False
            register_extended_attention(self)

    
    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self, latents: Float[Tensor, "B 4 DH DW"]
    ) -> Float[Tensor, "B 3 H W"]:
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def use_normal_unet(self):
        print("use normal unet")
        register_normal_attention(self)

    def edit_latents(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
        cams= None,
    ) -> Float[Tensor, "B 4 DH DW"]:
        
        self.scheduler.config.num_train_timesteps = t.item() if len(t.shape) < 1 else t[0].item()
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        current_H = image_cond_latents.shape[2]
        current_W = image_cond_latents.shape[3]

        camera_batch_size = self.cfg.camera_batch_size
        print("Start editing images...")

        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t) 

            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
            split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)
            
            for t in self.scheduler.timesteps:
                # predict the noise residual with unet, NO grad!
                with torch.no_grad():
                    # pred noise
                    noise_pred_text = []
                    noise_pred_image = []
                    noise_pred_uncond = []
                    # say, camera_batch_size = 5, len(latents) = 20, then pivotal_idx = [1 random value between each set of b:b+batch_size] (len 4)
                    pivotal_idx = torch.randint(camera_batch_size, (len(latents)//camera_batch_size,)) + torch.arange(0, len(latents), camera_batch_size) 
                    register_pivotal(self.unet, True)
                    
                    key_cams = [cams[cam_pivotal_idx] for cam_pivotal_idx in pivotal_idx.tolist()]  # len = 4
                    latent_model_input = torch.cat([latents[pivotal_idx]] * 3)      # Only the pivotal latents are selected for passing thru unet, but why concatenate 3 times ?
                    pivot_text_embeddings = torch.cat([positive_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx]], dim=0)
                    pivot_image_cond_latetns = torch.cat([split_image_cond_latents[pivotal_idx], split_image_cond_latents[pivotal_idx], zero_image_cond_latents[pivotal_idx]], dim=0)
                    latent_model_input = torch.cat([latent_model_input, pivot_image_cond_latetns], dim=1)

                    # Why are these pivotal latents passed to the unet first?
                    # Since unet is for denoising, pivotal latents are possibly being denoised.
                    # But their predicted noise is not used, so what happens here?
                    self.forward_unet(latent_model_input, t, encoder_hidden_states=pivot_text_embeddings)
                    register_pivotal(self.unet, False)

                    for i, b in enumerate(range(0, len(latents), camera_batch_size)):
                        register_batch_idx(self.unet, i)
                        register_cams(self.unet, cams[b:b + camera_batch_size], pivotal_idx[i] % camera_batch_size, key_cams) 
                        
                        # establish epipolar constraints between each of the latents and the pivotal latents (20*4*4096*4096)
                        # register these constraints in the unet (prolly?) so that it does constrained denoising of latents
                        epipolar_constrains = {}
                        for down_sample_factor in [1, 2, 4, 8]:
                            H = current_H // down_sample_factor
                            W = current_W // down_sample_factor
                            epipolar_constrains[H * W] = []
                            for cam in cams[b:b + camera_batch_size]:   # len = 5
                                cam_epipolar_constrains = []
                                for key_cam in key_cams:    # len = 4
                                    cam_epipolar_constrains.append(compute_epipolar_constrains(key_cam, cam, current_H=H, current_W=W))  # 10 values
                                epipolar_constrains[H * W].append(torch.stack(cam_epipolar_constrains, dim=0))
                            epipolar_constrains[H * W] = torch.stack(epipolar_constrains[H * W], dim=0)     # 5 x 4 x (64**2) x (64**2) (pixel-wise)
                        register_epipolar_constrains(self.unet, epipolar_constrains)

                        batch_model_input = torch.cat([latents[b:b + camera_batch_size]] * 3)   # Again, latents of a given camera batch size are concatenated 3 times. Is it because the output is of 3 channels?
                        batch_text_embeddings = torch.cat([positive_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size]], dim=0)
                        batch_image_cond_latents = torch.cat([split_image_cond_latents[b:b + camera_batch_size], split_image_cond_latents[b:b + camera_batch_size], zero_image_cond_latents[b:b + camera_batch_size]], dim=0)
                        batch_model_input = torch.cat([batch_model_input, batch_image_cond_latents], dim=1)

                        # After the pivotal latents are passed to the model and the epipolar constraints registered, 
                        # then latents for each camera in a batch are passed to unet for denoising
                        # Possibly, now that the pivotal latent features are passed to the 
                        batch_noise_pred = self.forward_unet(batch_model_input, t, encoder_hidden_states=batch_text_embeddings)
                        batch_noise_pred_text, batch_noise_pred_image, batch_noise_pred_uncond = batch_noise_pred.chunk(3)
                        noise_pred_text.append(batch_noise_pred_text)
                        noise_pred_image.append(batch_noise_pred_image)
                        noise_pred_uncond.append(batch_noise_pred_uncond)

                    noise_pred_text = torch.cat(noise_pred_text, dim=0)
                    noise_pred_image = torch.cat(noise_pred_image, dim=0)
                    noise_pred_uncond = torch.cat(noise_pred_uncond, dim=0)

                    # perform classifier-free guidance
                    noise_pred = (
                        noise_pred_uncond
                        + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                        + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
                    )

                    # get previous sample, continue loop
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                    
        print("Editing finished.")
        return latents

    def compute_grad_sds(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
        cams= None,
    ):
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, t) 
        positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
        split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)
        current_H = image_cond_latents.shape[2]
        current_W = image_cond_latents.shape[3]
        camera_batch_size = self.cfg.camera_batch_size
        
        with torch.no_grad():
            noise_pred_text = []
            noise_pred_image = []
            noise_pred_uncond = []
            pivotal_idx = torch.randint(camera_batch_size, (len(latents)//camera_batch_size,)) + torch.arange(0,len(latents),camera_batch_size) 
            print(pivotal_idx)
            register_pivotal(self.unet, True)

            latent_model_input = torch.cat([latents[pivotal_idx]] * 3)
            pivot_text_embeddings = torch.cat([positive_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx]], dim=0)
            pivot_image_cond_latetns = torch.cat([split_image_cond_latents[pivotal_idx], split_image_cond_latents[pivotal_idx], zero_image_cond_latents[pivotal_idx]], dim=0)
            latent_model_input = torch.cat([latent_model_input, pivot_image_cond_latetns], dim=1)
            
            key_cams = cams[pivotal_idx]
            self.forward_unet(latent_model_input, t, encoder_hidden_states=pivot_text_embeddings)
            register_pivotal(self.unet, False)


            for i, b in enumerate(range(0, len(latents), camera_batch_size)):
                register_batch_idx(self.unet, i)
                register_cams(self.unet, cams[b:b + camera_batch_size], pivotal_idx[i] % camera_batch_size, key_cams) 
                
                epipolar_constrains = {}
                for down_sample_factor in [1, 2, 4, 8]:
                    H = current_H // down_sample_factor
                    W = current_W // down_sample_factor
                    epipolar_constrains[H * W] = []
                    for cam in cams[b:b + camera_batch_size]:
                        cam_epipolar_constrains = []
                        for key_cam in key_cams:
                            cam_epipolar_constrains.append(compute_epipolar_constrains(key_cam, cam, current_H=H, current_W=W))
                        epipolar_constrains[H * W].append(torch.stack(cam_epipolar_constrains, dim=0))
                    epipolar_constrains[H * W] = torch.stack(epipolar_constrains[H * W], dim=0)
                register_epipolar_constrains(self.unet, epipolar_constrains)

                batch_model_input = torch.cat([latents[b:b + camera_batch_size]] * 3)
                batch_text_embeddings = torch.cat([positive_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size]], dim=0)
                batch_image_cond_latents = torch.cat([split_image_cond_latents[b:b + camera_batch_size], split_image_cond_latents[b:b + camera_batch_size], zero_image_cond_latents[b:b + camera_batch_size]], dim=0)
                batch_model_input = torch.cat([batch_model_input, batch_image_cond_latents], dim=1)
                batch_noise_pred = self.forward_unet(batch_model_input, t, encoder_hidden_states=batch_text_embeddings)
                batch_noise_pred_text, batch_noise_pred_image, batch_noise_pred_uncond = batch_noise_pred.chunk(3)
                noise_pred_text.append(batch_noise_pred_text)
                noise_pred_image.append(batch_noise_pred_image)
                noise_pred_uncond.append(batch_noise_pred_uncond)

            noise_pred_text = torch.cat(noise_pred_text, dim=0)
            noise_pred_image = torch.cat(noise_pred_image, dim=0)
            noise_pred_uncond = torch.cat(noise_pred_uncond, dim=0)

            # perform classifier-free guidance
            noise_pred = (
                noise_pred_uncond
                + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
            )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        return grad
    



    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        cond_rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        gaussians = None,
        cams= None,
        render=None,
        pipe=None,
        background=None,
        **kwargs,
    ):
        assert cams is not None, "cams is required for dge guidance"
        batch_size, H, W, _ = rgb.shape
        factor = 512 / max(W, H)
        factor = math.ceil(min(W, H) * factor / 64) * 64 / min(W, H)

        width = int((W * factor) // 64) * 64
        height = int((H * factor) // 64) * 64
        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        RH, RW = height, width

        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )
        latents = self.encode_images(rgb_BCHW_HW8)  # 20 x 4 x 64 x 64
        
        cond_rgb_BCHW = cond_rgb.permute(0, 3, 1, 2)
        cond_rgb_BCHW_HW8 = F.interpolate(      # 20 x 3 x 512 x 512
            cond_rgb_BCHW,
            (RH, RW),
            mode="bilinear",
            align_corners=False,
        )
        cond_latents = self.encode_cond_images(cond_rgb_BCHW_HW8)       # 60 x 4 x 64 x 64

        temp = torch.zeros(batch_size).to(rgb.device)
        text_embeddings = prompt_utils.get_text_embeddings(temp, temp, temp, False)
        positive_text_embeddings, negative_text_embeddings = text_embeddings.chunk(2)
        text_embeddings = torch.cat(
            [positive_text_embeddings, negative_text_embeddings, negative_text_embeddings], dim=0)  # [positive, negative, negative]

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.max_step - 1,
            self.max_step,
            [1],
            dtype=torch.long,
            device=self.device,
        ).repeat(batch_size)

        if self.cfg.use_sds:
            grad = self.compute_grad_sds(text_embeddings, latents, cond_latents, t)
            grad = torch.nan_to_num(grad)
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            target = (latents - grad).detach()
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
            return {
                "loss_sds": loss_sds,
                "grad_norm": grad.norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
            }
        else:
            edit_latents = self.edit_latents(text_embeddings, latents, cond_latents, t, cams)   # 20 x 4 x 64 x 64
            edit_images = self.decode_latents(edit_latents)
            edit_images = F.interpolate(edit_images, (H, W), mode="bilinear")   # 20 x 3 x 512 x 512

            return {"edit_images": edit_images.permute(0, 2, 3, 1)}

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


@threestudio.register("dge-zest-guidance")
class DGEZestGuidance(DGEGuidance):
    cfg = DGEGuidance.Config
    def configure(self) -> None:
        threestudio.info(f"Loading ZeST checkpoints ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        torch.cuda.empty_cache()

        # load SDXL pipeline
        self.controlnet = ControlNetModel.from_pretrained(self.cfg.controlnet_path, variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device="cuda")
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            self.cfg.base_model_path,
            controlnet=self.controlnet,
            use_safetensors=True,
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(device="cuda")
        self.pipe.unet = register_cross_attention_hook(self.pipe.unet)

        self.ip_model = IPAdapterXL(self.pipe, self.cfg.image_encoder_path, self.cfg.ip_ckpt, "cuda")

        # self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        #     self.cfg.ip2p_name_or_path, **pipe_kwargs
        # ).to(self.device)

        # self.scheduler = self.pipe.scheduler
        # self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # vae dtype is torch.float16 !! 
        # For encoding, both vae and input should be of max precision! (Refer : https://github.com/huggingface/diffusers/issues/7188)
        self.vae = self.pipe.vae.eval()         
        self.vae = self.vae.to(torch.float32)       # Convert to torch.float32 for encoding 
        self.unet = self.pipe.unet.eval()
        self.unet = self.unet.to(torch.float32)

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.pipe.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.pipe.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded ZeST checkpoints!")
        for name, module in self.unet.named_modules():
            if isinstance_str(module, "BasicTransformerBlock"):
                print(name)
                make_block_fn = make_dge_block 
                module.__class__ = make_block_fn(module.__class__)
                # Something needed for older versions of diffusers
                if not hasattr(module, "use_ada_layer_norm_zero"):
                    module.use_ada_layer_norm = False
                    module.use_ada_layer_norm_zero = False
            register_extended_attention(self)


    def edit_latents(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
        cams= None,
    ) -> Float[Tensor, "B 4 DH DW"]:
        
        self.scheduler.config.num_train_timesteps = t.item() if len(t.shape) < 1 else t[0].item()
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        current_H = image_cond_latents.shape[2]
        current_W = image_cond_latents.shape[3]

        camera_batch_size = self.cfg.camera_batch_size
        print("Start editing images...")

        with torch.no_grad():
            # add noise
            ### TODO: Replace this with SDXL.prepare_latents (pass return_noise=True) -> Gives outputs noise, latents
            ######################
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t) 
            ######################

            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
            split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)
            
            for t in self.scheduler.timesteps:
                # predict the noise residual with unet, NO grad!
                with torch.no_grad():
                    # pred noise
                    noise_pred_text = []
                    noise_pred_image = []
                    noise_pred_uncond = []
                    # say, camera_batch_size = 5, len(latents) = 20, then pivotal_idx = [1 random value between each set of b:b+batch_size] (len 4)
                    # register pivot idxs
                    pivotal_idx = torch.randint(camera_batch_size, (len(latents)//camera_batch_size,)) + torch.arange(0, len(latents), camera_batch_size) 
                    register_pivotal(self.unet, True)
                    
                    key_cams = [cams[cam_pivotal_idx] for cam_pivotal_idx in pivotal_idx.tolist()]  # len = 4
                    latent_model_input = torch.cat([latents[pivotal_idx]] * 3)      # Only the pivotal latents are selected for passing thru unet, but why concatenate 3 times ?
                    pivot_text_embeddings = torch.cat([positive_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx]], dim=0)
                    pivot_image_cond_latetns = torch.cat([split_image_cond_latents[pivotal_idx], split_image_cond_latents[pivotal_idx], zero_image_cond_latents[pivotal_idx]], dim=0)
                    latent_model_input = torch.cat([latent_model_input, pivot_image_cond_latetns], dim=1)

                    # Why are these pivotal latents passed to the unet first?
                    # Since unet is for denoising, pivotal latents are possibly being denoised.
                    # But their predicted noise is not used, so what happens here?

                    ### TODO: Replace this with self.pipe.forward_unet()
                    ######################
                    self.forward_unet(latent_model_input, t, encoder_hidden_states=pivot_text_embeddings)
                    register_pivotal(self.unet, False)
                    ######################

                    for i, b in enumerate(range(0, len(latents), camera_batch_size)):
                        register_batch_idx(self.unet, i)
                        register_cams(self.unet, cams[b:b + camera_batch_size], pivotal_idx[i] % camera_batch_size, key_cams) 
                        
                        # establish epipolar constraints between each of the latents and the pivotal latents (20*4*4096*4096)
                        # register these constraints in the unet (prolly?) so that it does constrained denoising of latents
                        epipolar_constrains = {}
                        for down_sample_factor in [1, 2, 4, 8]:
                            H = current_H // down_sample_factor
                            W = current_W // down_sample_factor
                            epipolar_constrains[H * W] = []
                            for cam in cams[b:b + camera_batch_size]:   # len = 5
                                cam_epipolar_constrains = []
                                for key_cam in key_cams:    # len = 4
                                    cam_epipolar_constrains.append(compute_epipolar_constrains(key_cam, cam, current_H=H, current_W=W))  # 10 values
                                epipolar_constrains[H * W].append(torch.stack(cam_epipolar_constrains, dim=0))
                            epipolar_constrains[H * W] = torch.stack(epipolar_constrains[H * W], dim=0)     # 5 x 4 x (64**2) x (64**2) (pixel-wise)
                        register_epipolar_constrains(self.unet, epipolar_constrains)

                        batch_model_input = torch.cat([latents[b:b + camera_batch_size]] * 3)   # Again, latents of a given camera batch size are concatenated 3 times. Is it because the output is of 3 channels?
                        batch_text_embeddings = torch.cat([positive_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size]], dim=0)
                        batch_image_cond_latents = torch.cat([split_image_cond_latents[b:b + camera_batch_size], split_image_cond_latents[b:b + camera_batch_size], zero_image_cond_latents[b:b + camera_batch_size]], dim=0)
                        batch_model_input = torch.cat([batch_model_input, batch_image_cond_latents], dim=1)

                        # After the pivotal latents are passed to the model and the epipolar constraints registered, 
                        # then latents for each camera in a batch are passed to unet for denoising
                        # Possibly, now that the pivotal latent features are passed to the 
                        batch_noise_pred = self.forward_unet(batch_model_input, t, encoder_hidden_states=batch_text_embeddings)
                        batch_noise_pred_text, batch_noise_pred_image, batch_noise_pred_uncond = batch_noise_pred.chunk(3)
                        noise_pred_text.append(batch_noise_pred_text)
                        noise_pred_image.append(batch_noise_pred_image)
                        noise_pred_uncond.append(batch_noise_pred_uncond)

                    noise_pred_text = torch.cat(noise_pred_text, dim=0)
                    noise_pred_image = torch.cat(noise_pred_image, dim=0)
                    noise_pred_uncond = torch.cat(noise_pred_uncond, dim=0)

                    # perform classifier-free guidance
                    noise_pred = (
                        noise_pred_uncond
                        + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                        + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
                    )

                    # get previous sample, continue loop
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                    
        print("Editing finished.")
        return latents
    

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        images: List[torch.Tensor]=None,         # FG-Grayscaled images
        timestep=None,          # latent timestep
        is_strength_max=True,
        add_noise=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (batch_size, num_channels_latents, height // self.pipe.vae_scale_factor, width // self.pipe.vae_scale_factor)   # 1 x 4 x 64 x 64
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            images: List[torch.Tensor] = [image.to(device=device, dtype=dtype) for image in images]     # list of tensors (1,3,512,512), dtype=torch.float32

            if images[0].shape[1] == 4:
                images_latents = images.copy()
            else:
                ########
                ## TODO: This line causes the _encode_vae_image to give nan outputs even though the images are normalized to -1, 1
                # This is because 
                images_latents = [self.pipe._encode_vae_image(image=image, generator=generator) for image in images]
            images_latents: List[torch.Tensor] = [image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1) for image_latents in images_latents]

        noise_samples: List[torch.Tensor] = [randn_tensor(shape, generator=generator, device=device, dtype=dtype) for _ in images_latents]
        # if strength is 1. then initialise the latents to noise, else initial to image + noise
        latent_samples: List[torch.Tensor] = [noise if is_strength_max else self.pipe.scheduler.add_noise(image_latents, noise, timestep) for noise, image_latents in zip(noise_samples, images_latents)]
        # if pure noise then scale the initial latents by the  Scheduler's init sigma
        latent_samples = [latents * self.pipe.scheduler.init_noise_sigma if is_strength_max else latents for latents in latent_samples]

        outputs: Tuple[List[torch.Tensor]] = (latent_samples,)
        outputs += (noise_samples,)

        if return_image_latents:
            outputs += (images_latents,)

        return outputs
    

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask

        masked_image_latents = None
        if masked_image is not None:
            masked_image = masked_image.to(device=device, dtype=dtype)
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)
            if masked_image_latents.shape[0] < batch_size:
                if not batch_size % masked_image_latents.shape[0] == 0:
                    raise ValueError(
                        "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                        f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                        " Make sure the number of images that you pass is divisible by the total requested batch size."
                    )
                masked_image_latents = masked_image_latents.repeat(
                    batch_size // masked_image_latents.shape[0], 1, 1, 1
                )

            masked_image_latents = (
                torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
            )

            # aligning device to prevent device errors when concating it with the latent model input
            masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        return mask, masked_image_latents
    

    
    def edit_zest_latents(
        self,
        prompt_embeds,    # positive prompt + material exemplar
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        images,
        texture_exemplar: Image.Image,
        depths: List[Image.Image],             # control image (original control_img input was just one PIL Image)
        masks: List[Image.Image],
        num_images_per_prompt: int = 1,
        height: int = None,
        width: int = None,
        strength: float = 0.9999,
        guidance_scale: float = 5.0,
        guidance_rescale: float = 0.0,
        num_inference_steps: int = 30,
        controlnet_conditioning_scale: float = 0.9,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        cams = None,
    ) -> Float[Tensor, "B 4 DH DW"]:
        
        ####
        controlnet = self.pipe.controlnet
        mult = 1
        control_guidance_start, control_guidance_end = (
            mult * [control_guidance_start],
            mult * [control_guidance_end],
            )

        self.pipe._guidance_scale = guidance_scale      # 5.0
        batch_size = prompt_embeds.shape[0]

        device = self.device
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.pipe.get_timesteps(           # [925, 892, ...], len = 29
            num_inference_steps,
            strength,
            device,
            denoising_start=None
        )

        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)      # [925., ]
        self.pipe._num_timesteps = len(timesteps)       # 29
        crops_coords = None
        resize_mode = "default"

        #######
        x = images[0]      # For debugging   (PIL IMAGE : (512, 512))   (FG-Grayscaled image IS CORRECT! - CHECKED using visualizer)
        #######

        # prepare FG-Grayscaled images
        original_images: List[Image.Image] = images     # List of FG-Grayscaled images
        init_images = [                      # list of image tensors (len = 20) (each of shape 1,3,512,512)
                        self.pipe.image_processor.preprocess(
                            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
                        ).to(dtype=torch.float32)
                        for
                        image in original_images
                    ]

        # prepare depth maps (control images)
        depth_maps = [                      # list of control image tensors (len = 20) (each of shape 2,3,512,512)
                        self.pipe.prepare_control_image(
                            image=control_image,
                            width=width,
                            height=height,
                            batch_size=batch_size * num_images_per_prompt,
                            num_images_per_prompt=num_images_per_prompt,
                            device=device,
                            dtype=controlnet.dtype,
                            crops_coords=crops_coords,
                            resize_mode=resize_mode,
                            do_classifier_free_guidance=self.pipe.do_classifier_free_guidance,
                            guess_mode=False,
                        )
                        for control_image in depths
                    ]
        
        # Prepare masks
        masks = [
                self.pipe.mask_processor.preprocess(            # list of mask tensors (len = 20) (each of shape 1,1,512,512)
                    mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
                )
            for mask_image in masks
        ]

        # Mask the original FG-Grayscaled images with mask
        masked_images = [
            init_image * (mask < 0.5)
            for init_image, mask in zip(init_images, masks)
        ]

        assert len(init_images) > 0, "No images to edit"
        _, _, height, width = init_images[0].shape

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels      # 4
        num_channels_unet = self.pipe.unet.config.in_channels       # 4
        return_image_latents = num_channels_unet == 4               # True

        add_noise = True
        is_strength_max = strength == 1.0

        # current_H = image_cond_latents.shape[2]
        # current_W = image_cond_latents.shape[3]

        camera_batch_size = self.cfg.camera_batch_size      # 5
        print("Start editing images...")

        with torch.no_grad():
            latent_samples, noise_samples, images_latents = [], [], []
            for image in init_images:
                # create the noised image latents
                latent_outputs: Tuple[List[torch.Tensor]] = self.pipe.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    torch.float32,      # prompt_embeds.dtype
                    device,
                    generator,
                    None,
                    image=image,
                    timestep=latent_timestep,
                    is_strength_max=is_strength_max,
                    add_noise=add_noise,
                    return_noise=True,
                    return_image_latents=return_image_latents,
                )

                latent_outputs: Tuple[List[torch.Tensor]] = self.pipe.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    torch.float32,      # prompt_embeds.dtype
                    device,
                    generator,
                    None,
                    image=image,
                    timestep=latent_timestep,
                    is_strength_max=is_strength_max,
                    add_noise=add_noise,
                    return_noise=True,
                    return_image_latents=return_image_latents,
                )

                if return_image_latents:
                    latent_sample, noise_sample, image_latents = latent_outputs
                else:
                    latent_sample, noise_sample = latent_outputs

                print("Latent vals : ", latent_sample.shape, latent_sample.max(), latent_sample.min())
                latent_samples.append(latent_sample)
                noise_samples.append(noise_sample)
                images_latents.append(image_latents)

            print("Latents vals : ", latent_samples[0].shape, latent_samples[0].max(), latent_samples[0].min())

            #### TODO: (17th June 12 PM) 
            # - Fix the mask latents and other control input formats from SDXl pipeline
            # - Then, follow the for t in timestep loop in DGE to select pivot indexes

            masks, masked_images_latents = [
                self.prepare_mask_latents(
                mask,
                masked_image,
                batch_size * num_images_per_prompt,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                self.pipe.do_classifier_free_guidance,
            )
                for mask, masked_image in zip(masks, masked_images)
            ]

            if num_channels_unet == 9:
                # default case for runwayml/stable-diffusion-inpainting
                num_channels_mask = masks[0].shape[1]
                num_channels_masked_image = masked_images_latents[0].shape[1]
                if num_channels_latents + num_channels_mask + num_channels_masked_image != self.pipe.unet.config.in_channels:
                    raise ValueError(
                        f"Incorrect configuration settings! The config of `pipeline.unet`: {self.pipe.unet.config} expects"
                        f" {self.pipe.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                        f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                        f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                        " `pipeline.unet` or your `mask_image` or `image` input."
                    )
            elif num_channels_unet != 4:
                raise ValueError(
                    f"The unet {self.pipe.unet.__class__} should have either 4 or 9 input channels, not {self.pipe.unet.config.in_channels}."
                )
            

            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)

            # 8.2 Create tensor stating which controlnets to keep
            controlnet_keep: List[float] = []    # len(timesteps) with each value = 1.0
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0])

            # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            height, width = latent_samples[0].shape[-2:]
            height = height * self.pipe.vae_scale_factor
            width = width * self.pipe.vae_scale_factor

            original_size = (height, width)
            target_size = (height, width)

            # 10. Prepare added time ids & embeddings
            add_text_embeds = pooled_prompt_embeds
            if self.pipe.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = self.pipe.text_encoder_2.config.projection_dim

            add_time_ids, add_neg_time_ids = self.pipe._get_add_time_ids(
                original_size,
                (0,0),
                target_size,
                aesthetic_score,
                negative_aesthetic_score,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

            if self.pipe.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
                add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device)

            # 11. Denoising loop
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.pipe.scheduler.order, 0)

            #############################################
            ### TODO: Start the timesteps bit.
            # Take stuff up from both below (DGEGuidance) and SDXL-PIPELINE (Line 1669 onwards)
            #############################################

            with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    with torch.no_grad():
                        # expand the latents if we are doing classifier free guidance
                        latent_model_inputs = [
                            torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                            for latents in latent_samples
                        ]

                        # concat latents, mask, masked_image_latents in the channel dimension
                        latent_model_inputs = [
                            self.pipe.scheduler.scale_model_input(latent_model_input, t)
                            for latent_model_input in latent_model_inputs
                        ]

                        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                        # controlnet(s) inference
                        control_model_inputs: List[torch.Tensor] = latent_model_inputs
                        controlnet_prompt_embeds = prompt_embeds
                        controlnet_added_cond_kwargs = added_cond_kwargs

                        controlnet_cond_scale = controlnet_conditioning_scale
                        cond_scale: List[float] = controlnet_cond_scale * controlnet_keep[i]

                        # # Resize control_image to match the size of the input to the controlnet
                        # if control_image.shape[-2:] != control_model_input.shape[-2:]:
                        #     control_image = F.interpolate(control_image, size=control_model_input.shape[-2:], mode="bilinear", align_corners=False)

                        down_block_res_samples, mid_block_res_samples = [], []
                        for control_model_input, depth_map in zip(control_model_inputs, depth_maps):
                            down_block_res_sample, mid_block_res_sample = self.controlnet(
                                control_model_input,
                                t,
                                encoder_hidden_states=controlnet_prompt_embeds,
                                controlnet_cond=depth_map,
                                conditioning_scale=cond_scale,
                                guess_mode=False,
                                added_cond_kwargs=controlnet_added_cond_kwargs,
                                return_dict=False,
                            )
                            down_block_res_samples.append(down_block_res_sample)
                            mid_block_res_samples.append(mid_block_res_sample)

                        if num_channels_unet == 9:
                            latent_model_inputs = [
                                torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
                                for latent_model_input, mask, masked_image_latents in zip(
                                    latent_model_inputs, masks, masked_images_latents
                                )
                            ]

                        # find pivotal indices
                        pivotal_idx = torch.randint(camera_batch_size, (len(latent_samples)//camera_batch_size,)) + torch.arange(0, len(latent_samples), camera_batch_size) 
                        register_pivotal(self.pipe.unet, True)

                        key_cams = [cams[cam_pivotal_idx] for cam_pivotal_idx in pivotal_idx.tolist()]  # len = 4
                        # Now, only select latent model inputs for pivotal indices
                        latent_model_inputs = [latent_model_inputs[cam_pivotal_idx] for cam_pivotal_idx in pivotal_idx.tolist()]
                        latent_model_inputs: torch.Tensor = torch.cat(latent_model_inputs, dim=0) # convert to tensor
                        
                        # Now, do a single forward pass through the unet and register the pivotal indices
                        self.pipe.unet(
                            latent_model_inputs,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )
                        register_pivotal(self.pipe.unet, False)

                        noise_preds_text = []
                        noise_preds_uncond = []

                        for i, b in enumerate(range(0, len(latent_samples), camera_batch_size)):
                            register_batch_idx(self.pipe.unet, i)
                            register_cams(self.pipe.unet, cams[b:b + camera_batch_size], pivotal_idx[i] % camera_batch_size, key_cams) 
                            
                            # establish epipolar constraints between each of the latents and the pivotal latents (20*4*4096*4096)
                            # register these constraints in the unet (prolly?) so that it does constrained denoising of latents
                            epipolar_constraints = {}
                            for down_sample_factor in [1, 2, 4, 8]:
                                H = height // down_sample_factor
                                W = width // down_sample_factor
                                epipolar_constraints[H * W] = []
                                for cam in cams[b:b + camera_batch_size]:   # len = 5
                                    cam_epipolar_constraints = []
                                    for key_cam in key_cams:    # len = 4
                                        cam_epipolar_constraints.append(compute_epipolar_constrains(key_cam, cam, current_H=H, current_W=W))  # 10 values
                                    epipolar_constraints[H * W].append(torch.stack(cam_epipolar_constraints, dim=0))
                                epipolar_constraints[H * W] = torch.stack(epipolar_constraints[H * W], dim=0)     # 5 x 4 x (64**2) x (64**2) (pixel-wise)
                            register_epipolar_constrains(self.pipe.unet, epipolar_constraints)

                            # This time, just pick latents in a given camera batch size
                            batch_model_inputs: torch.Tensor = latent_model_inputs[b:b+camera_batch_size]   # batch_size * latent_size
                            # predict the noise residual
                            batch_noise_preds = self.pipe.unet(
                                batch_model_inputs,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                down_block_additional_residuals=down_block_res_samples[b:b+camera_batch_size],
                                mid_block_additional_residual=mid_block_res_samples[b:b+camera_batch_size],
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )

                            # perform guidance
                            if self.do_classifier_free_guidance:
                                batch_noise_preds_uncond, batch_noise_preds_text = batch_noise_preds.chunk(2)
                                noise_preds_uncond.append(batch_noise_preds_uncond)
                                noise_preds_text.append(batch_noise_preds_text)

                        if self.do_classifier_free_guidance:
                            noise_preds = noise_preds_uncond + guidance_scale * (noise_preds_text - noise_preds_uncond)

                        if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            noise_preds = rescale_noise_cfg(noise_preds, noise_preds_text, guidance_rescale=guidance_rescale)

                        # compute the previous noisy sample x_t -> x_t-1
                        latent_samples = self.pipe.scheduler.step(noise_preds, t, latent_samples, **extra_step_kwargs, return_dict=False)

                        if num_channels_unet == 4:
                            init_latents_proper = images_latents
                            if self.do_classifier_free_guidance:
                                init_masks, _ = masks.chunk(2)
                            else:
                                init_masks = masks

                            if i < len(timesteps) - 1:
                                noise_timestep = timesteps[i + 1]
                                init_latents_proper = self.pipe.scheduler.add_noise(
                                    init_latents_proper, noise_samples, torch.tensor([noise_timestep])
                                )

                            latent_samples = (1 - init_masks) * init_latents_proper + init_masks * latent_samples

        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.pipe.upcast_vae()
            latent_samples = latent_samples.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.pipe.final_offload_hook is not None:
            self.pipe.unet.to("cpu")
            self.pipe.controlnet.to("cpu")
            torch.cuda.empty_cache()

        images = self.vae.decode(latent_samples / self.vae.config.scaling_factor, return_dict=False)

        images = self.image_processor.postprocess(images, output_type="pil")

        # Offload all models
        self.pipe.maybe_free_model_hooks()

        print("Editing finished.")

        return StableDiffusionXLPipelineOutput(images=images)


        # with torch.no_grad():
        #     # add noise
        #     ### TODO: Replace this with SDXL.prepare_latents (pass return_noise=True) -> Gives outputs noise, latents
        #     ######################
        #     # noise = torch.randn_like(latents)
        #     # latents = self.scheduler.add_noise(latents, noise, t)       # noised latents
        #     ######################

        #     # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        #     positive_text_embedding, negative_text_embedding, _ = text_embeddings.chunk(3)
        #     split_image_cond_latents, _, zero_image_cond_latents = image_cond_latents.chunk(3)
            
        #     for t in self.scheduler.timesteps:
        #         # predict the noise residual with unet, NO grad!
        #         with torch.no_grad():
        #             # pred noise
        #             noise_pred_text = []
        #             noise_pred_image = []
        #             noise_pred_uncond = []
        #             # say, camera_batch_size = 5, len(latents) = 20, then pivotal_idx = [1 random value between each set of b:b+batch_size] (len 4)
        #             # register pivot idxs
        #             pivotal_idx = torch.randint(camera_batch_size, (len(latents)//camera_batch_size,)) + torch.arange(0, len(latents), camera_batch_size) 
        #             register_pivotal(self.unet, True)
                    
        #             key_cams = [cams[cam_pivotal_idx] for cam_pivotal_idx in pivotal_idx.tolist()]  # len = 4
        #             latent_model_input = torch.cat([latents[pivotal_idx]] * 3)      # if each latent 1x4x64x64, this gives 3x4x64x64
        #             pivot_text_embeddings = torch.cat([positive_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx], negative_text_embedding[pivotal_idx]], dim=0)
        #             pivot_image_cond_latetns = torch.cat([split_image_cond_latents[pivotal_idx], split_image_cond_latents[pivotal_idx], zero_image_cond_latents[pivotal_idx]], dim=0)
        #             latent_model_input = torch.cat([latent_model_input, pivot_image_cond_latetns], dim=1)

        #             # Why are these pivotal latents passed to the unet first?
        #             # Since unet is for denoising, pivotal latents are possibly being denoised.
        #             # But their predicted noise is not used, so what happens here?

        #             ### TODO: Replace this with self.pipe.forward_unet()
        #             ######################
        #             self.forward_unet(latent_model_input, t, encoder_hidden_states=pivot_text_embeddings)
        #             register_pivotal(self.unet, False)
        #             ######################

        #             for i, b in enumerate(range(0, len(latents), camera_batch_size)):
        #                 register_batch_idx(self.unet, i)
        #                 register_cams(self.unet, cams[b:b + camera_batch_size], pivotal_idx[i] % camera_batch_size, key_cams) 
                        
        #                 # establish epipolar constraints between each of the latents and the pivotal latents (20*4*4096*4096)
        #                 # register these constraints in the unet (prolly?) so that it does constrained denoising of latents
        #                 epipolar_constrains = {}
        #                 for down_sample_factor in [1, 2, 4, 8]:
        #                     H = current_H // down_sample_factor
        #                     W = current_W // down_sample_factor
        #                     epipolar_constrains[H * W] = []
        #                     for cam in cams[b:b + camera_batch_size]:   # len = 5
        #                         cam_epipolar_constrains = []
        #                         for key_cam in key_cams:    # len = 4
        #                             cam_epipolar_constrains.append(compute_epipolar_constrains(key_cam, cam, current_H=H, current_W=W))  # 10 values
        #                         epipolar_constrains[H * W].append(torch.stack(cam_epipolar_constrains, dim=0))
        #                     epipolar_constrains[H * W] = torch.stack(epipolar_constrains[H * W], dim=0)     # 5 x 4 x (64**2) x (64**2) (pixel-wise)
        #                 register_epipolar_constrains(self.unet, epipolar_constrains)

        #                 batch_model_input = torch.cat([latents[b:b + camera_batch_size]] * 3)   # Again, latents of a given camera batch size are concatenated 3 times. Is it because the output is of 3 channels?
        #                 batch_text_embeddings = torch.cat([positive_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size], negative_text_embedding[b:b + camera_batch_size]], dim=0)
        #                 batch_image_cond_latents = torch.cat([split_image_cond_latents[b:b + camera_batch_size], split_image_cond_latents[b:b + camera_batch_size], zero_image_cond_latents[b:b + camera_batch_size]], dim=0)
        #                 batch_model_input = torch.cat([batch_model_input, batch_image_cond_latents], dim=1)

        #                 # After the pivotal latents are passed to the model and the epipolar constraints registered, 
        #                 # then latents for each camera in a batch are passed to unet for denoising
        #                 # Possibly, now that the pivotal latent features are passed to the 
        #                 batch_noise_pred = self.forward_unet(batch_model_input, t, encoder_hidden_states=batch_text_embeddings)
        #                 batch_noise_pred_text, batch_noise_pred_image, batch_noise_pred_uncond = batch_noise_pred.chunk(3)
        #                 noise_pred_text.append(batch_noise_pred_text)
        #                 noise_pred_image.append(batch_noise_pred_image)
        #                 noise_pred_uncond.append(batch_noise_pred_uncond)

        #             noise_pred_text = torch.cat(noise_pred_text, dim=0)
        #             noise_pred_image = torch.cat(noise_pred_image, dim=0)
        #             noise_pred_uncond = torch.cat(noise_pred_uncond, dim=0)

        #             # perform classifier-free guidance
        #             noise_pred = (
        #                 noise_pred_uncond
        #                 + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
        #                 + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
        #             )

        #             # get previous sample, continue loop
        #             latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                    
        pass

        
    def __call__(
        self,
        images: List[Image.Image],
        # cond_rgb: Float[Tensor, "B H W C"],
        # prompt_utils: PromptProcessorOutput,
        texture_exemplar: Image.Image,
        gaussians = None,
        cams= None,
        depths=None,
        masks=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        seed=None,
        num_samples=1,
        render=None,
        pipe=None,
        background=None,
        **kwargs,
    ):
        '''
        Orig Args:
            rgb: Rendered image
            cond_rgb: Original image at that viewpoint
            cams: Camera parameters
            
        New Args:
            images: List[Image.Image] (FG-Grayscaled images)
            texture_exemplar: (Texture exempla6r)
            depth_map: Depth map of the scene
            mask_image: Mask of the scene
        '''
        # assert cams is not None, "cams is required for dge guidance"
        # batch_size, H, W, _ = rgb.shape
        # factor = 512 / max(W, H)
        # factor = math.ceil(min(W, H) * factor / 64) * 64 / min(W, H)

        # width = int((W * factor) // 64) * 64
        # height = int((H * factor) // 64) * 64
        # rgb_BCHW = rgb.permute(0, 3, 1, 2)      # 20 x 3 x 512 x 512

        # RH, RW = height, width

        # rgb_BCHW_HW8 = F.interpolate(
        #     rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        # )   # 20 x 3 x 512 x 512
        # latents = self.encode_images(rgb_BCHW_HW8)  # 20 x 4 x 64 x 64
        
        # cond_rgb_BCHW = cond_rgb.permute(0, 3, 1, 2)
        
        # cond_rgb_BCHW_HW8 = F.interpolate(      # 20 x 3 x 512 x 512
        #     cond_rgb_BCHW,
        #     (RH, RW),
        #     mode="bilinear",
        #     align_corners=False,
        # )
        # cond_latents = self.encode_cond_images(cond_rgb_BCHW_HW8)       # 60 x 4 x 64 x 64

        # temp = torch.zeros(batch_size).to(rgb.device)
        # text_embeddings = prompt_utils.get_text_embeddings(temp, temp, temp, False)
        # positive_text_embeddings, negative_text_embeddings = text_embeddings.chunk(2)
        # text_embeddings = torch.cat(
        #     [positive_text_embeddings, negative_text_embeddings, negative_text_embeddings], dim=0)  # [positive, negative, negative]

        ############### IP-ADAPTER CODE ################

        self.ip_model.set_scale(scale)

        # assert len(texture_exemplar) == 1, "Only one texture exemplar is supported"
        num_prompts = 1

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        # assert isinstance(images, List[Image.Image]), "images must be a list of PIL images"
        image_prompt_embeds, uncond_image_prompt_embeds = self.ip_model.get_image_embeds(texture_exemplar)  # 1x4x2048 (both)
        bs_embed, seq_len, _ = image_prompt_embeds.shape    # bs, 4
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1) # bs x seq_len x 2048
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)     # 1x4x2048
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)       # 1x4x2048

        with torch.inference_mode():
            (
                prompt_embeds,      # 1x77x2048
                negative_prompt_embeds,     # 1x77x2048
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            # force model to generate more like (positive prompt + material exemplar) and less like (negative prompt + original image)
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)
        

        ''' TODO: 
            Replace this pipe call with low-level noising and denoising code from SDXL+Inpaint
            Instead, replace the self.edit_latents() call with the code from SDXL+Inpaint, using below inputs
        '''

        # Extra params (not for IP-Adapter but for SDXL+Inpaint)
        # image, control_image, mask_image, controlnet_conditioning_scale
        imgs = self.edit_zest_latents(
            prompt_embeds=prompt_embeds,    # positive prompt + material exemplar
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            controlnet_conditioning_scale=self.cfg.controlnet_conditioning_scale,
            num_inference_steps=self.cfg.num_inference_steps,
            generator=self.generator,
            images=images,
            texture_exemplar=texture_exemplar,
            cams=cams,
            depths=depths,
            masks=masks

        ).images

        print(images)
        exit(0)


        #######################################################

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.max_step - 1,
            self.max_step,
            [1],
            dtype=torch.long,
            device=self.device,
        ).repeat(batch_size)

        if self.cfg.use_sds:
            # grad = self.compute_grad_sds(text_embeddings, latents, cond_latents, t)
            # grad = torch.nan_to_num(grad)
            # if self.grad_clip_val is not None:
            #     grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            # target = (latents - grad).detach()
            # loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
            # return {
            #     "loss_sds": loss_sds,
            #     "grad_norm": grad.norm(),
            #     "min_step": self.min_step,
            #     "max_step": self.max_step,
            # }
            pass
        else:
            edit_latents = self.edit_latents(text_embeddings, latents, cond_latents, t, cams)   # 20 x 4 x 64 x 64
            edit_images = self.decode_latents(edit_latents)
            edit_images = F.interpolate(edit_images, (H, W), mode="bilinear")   # 20 x 3 x 512 x 512

            return {"edit_images": edit_images.permute(0, 2, 3, 1)}