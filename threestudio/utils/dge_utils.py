from typing import Type
import torch
import os

from pathlib import Path
from PIL import Image
import torch
import yaml
import math

from gaussiansplatting.utils.graphics_utils import get_fundamental_matrix_with_H
import torchvision.transforms as T
from torchvision.io import read_video,write_video
import os
import random
import numpy as np
from torchvision.io import write_video
from kornia.geometry.transform import remap

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity


def resize_bool_tensor(bool_tensor, size):
    """
    Resizes a boolean tensor to a new size using nearest neighbor interpolation.
    """
    # Convert boolean tensor to float
    H_new, W_new = size
    tensor_float = bool_tensor.float()

    # Resize using nearest interpolation
    resized_float = torch.nn.functional.interpolate(tensor_float, size=(H_new, W_new), mode='nearest')

    # Convert back to boolean
    resized_bool = resized_float > 0.5
    return resized_bool

def point_to_line_dist(points, lines):
    """
    Calculate the distance from points to lines in 2D.
    points: Nx3
    lines: Mx3

    return distance: NxM
    """
    numerator = torch.abs(lines @ points.T)
    denominator = torch.linalg.norm(lines[:,:2], dim=1, keepdim=True)
    return numerator / denominator

def save_video_frames(video_path, img_size=(512,512)):
    video, _, _ = read_video(video_path, output_format="TCHW")
    # rotate video -90 degree if video is .mov format. this is a weird bug in torchvision
    if video_path.endswith('.mov'):
        video = T.functional.rotate(video, -90)
    video_name = Path(video_path).stem
    os.makedirs(f'data/{video_name}', exist_ok=True)
    for i in range(len(video)):
        ind = str(i).zfill(5)
        image = T.ToPILImage()(video[i])
        image_resized = image.resize((img_size),  resample=Image.Resampling.LANCZOS)
        image_resized.save(f'data/{video_name}/{ind}.png')

def add_dict_to_yaml_file(file_path, key, value):
    data = {}

    # If the file already exists, load its contents into the data dictionary
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

    # Add or update the key-value pair
    data[key] = value

    # Save the data back to the YAML file
    with open(file_path, 'w') as file:
        yaml.dump(data, file)
        
def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity


def load_imgs(data_path, n_frames, device='cuda', pil=False):
    imgs = []
    pils = []
    for i in range(n_frames):
        img_path = os.path.join(data_path, "%05d.jpg" % i)
        if not os.path.exists(img_path):
            img_path = os.path.join(data_path, "%05d.png" % i)
        img_pil = Image.open(img_path)
        pils.append(img_pil)
        img = T.ToTensor()(img_pil).unsqueeze(0)
        imgs.append(img)
    if pil:
        return torch.cat(imgs).to(device), pils
    return torch.cat(imgs).to(device)


def save_video(raw_frames, save_path, fps=10):
    video_codec = "libx264"
    video_options = {
        "crf": "18",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    }

    frames = (raw_frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(save_path, frames, fps=fps, video_codec=video_codec, options=video_options)


def compute_epipolar_constrains(cam1, cam2, current_H=64, current_W=64):
    n_frames = 1
    sequence_length = current_W * current_H
    fundamental_matrix_1 = []
    
    fundamental_matrix_1.append(get_fundamental_matrix_with_H(cam1, cam2, current_H, current_W))        # fundamental matrix has very large values (1e+6 range!)
    fundamental_matrix_1 = torch.stack(fundamental_matrix_1, dim=0)     # 1,3,3

    x = torch.arange(current_W)     # 0,1,...,64
    y = torch.arange(current_H)
    x, y = torch.meshgrid(x, y, indexing='xy')      # 64x64
    x = x.reshape(-1)       # 4096
    y = y.reshape(-1)       # 4096

    # 0,0,1
    # 1,0,1         heto_cam : Column-wise ordering of latent pixels (homogeneous coords)
    # ....
    # 63,63,1
    heto_cam2 = torch.stack([x, y, torch.ones(size=(len(x),))], dim=1).view(-1, 3).cuda()                   # 4096,3
    heto_cam1 = torch.stack([x, y, torch.ones(size=(len(x),))], dim=1).view(-1, 3).cuda()
    # epipolar_line: n_frames X seq_len,  3
    line1 = (heto_cam2.unsqueeze(0).repeat(n_frames, 1, 1) @ fundamental_matrix_1.cuda()).view(-1, 3)       # 4096,3
    # line1 has inf, -inf, nan values (in normal DGE, DGE+ZeST)!! 
    distance1 = point_to_line_dist(heto_cam1, line1)        # 4096,4096 (nan values in DGE, DGE+ZeST! (Is this expected?))

    
    idx1_epipolar = distance1 > 1 # sequence_length x sequence_lengths

    return idx1_epipolar

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def register_epipolar_constrains(diffusion_model, epipolar_constrains):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "epipolar_constrains", epipolar_constrains)

def register_cams(diffusion_model, cams, pivot_this_batch, key_cams):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "cams", cams)
            setattr(module, "pivot_this_batch", pivot_this_batch)
            setattr(module, "key_cams", key_cams)

def register_pivotal(diffusion_model, is_pivotal):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "pivotal_pass", is_pivotal)
            
def register_batch_idx(diffusion_model, batch_idx):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "batch_idx", batch_idx)


def register_t(diffusion_model, t):

    for _, module in diffusion_model.named_modules():
    # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "t", t)

### NEW FUNCTIONS FOR ZEST ###
def register_conditioning_factor(diffusion, conditioning_factor):

    for _, module in diffusion.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "conditioning_factor", conditioning_factor)

            '''TODO: Maybe also add this factor as a attention layer-level attribute too, since also needed in extended_attention()
            '''

            if hasattr(module, 'attn1'):
                setattr(module.attn1, "conditioning_factor", conditioning_factor)

def register_normal_attention(model):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        def forward(x, encoder_hidden_states=None, attention_mask=None):
            # assert encoder_hidden_states is None 
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.head_to_batch_dim(q)
            key = self.head_to_batch_dim(k)
            value = self.head_to_batch_dim(v)

            attention_probs = self.get_attention_scores(query, key)
            hidden_states = torch.bmm(attention_probs, value)
            out = self.batch_to_head_dim(hidden_states)

            return to_out(out)

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.normal_attn = sa_forward(module.attn1)
            module.use_normal_attn = True


def register_extended_attention(model):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        def forward(x, encoder_hidden_states=None, attention_mask=None):
            '''
            NOTE:
              x = pixel-wise embeddings of latents
              encoder_hidden_states = prompt embeddings (or, latent pixels itself)

              In cross attention, prompt embeddings form the key and value, and pixel-wise embeddings form the query.
              This gives a pixel-wise correspondence/grounding with each word in the prompt. For ZeST, since self-attention between pixel embeddings is used instead, the prompt embeddings are not used.
            '''
            assert encoder_hidden_states is None        # For ZeST, self-attention between pixel embeddings is used 
            batch_size, sequence_length, dim = x.shape      # 8,4096,320
            h = self.heads      # 8

            # For ZeST (i.e., SDXl), there is only text (condition) and uncond input. So, divide the batch size by 2
            factor = self.conditioning_factor
            assert factor in [2,3]          # Support both SD-1.5 and IP2P
            n_frames = batch_size // factor          # 3
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x        # 8,4096,320
            q = self.to_q(x)                            # 8,4096,320 ... 8,1024, 640 -> 8,4096,320 each
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
            
            '''
            NOTE: This replicating key and value tensors n_frames is done to allow each pixel to simultaneously attend to pixels in ALL KEYFRAMES AT ONCE, INSTEAD OF n_frame latent-pairwise computations.
            TODO: Make this code usable for both normal DGE (IP2P) and ZeST (SD-1.5)
            '''

            k_text = k[:n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)         # 4,4096x4,320
            if factor == 3:     # IP2P
                k_image = k[n_frames: 2*n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
                k_uncond = k[2*n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            else:               # SD-1.5
                k_uncond = k[n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            v_text = v[:n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            if factor == 3:     # IP2P
                v_image = v[n_frames:2*n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
                v_uncond = v[2*n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            else:               # SD-1.5   
                v_uncond = v[n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            q_text = self.head_to_batch_dim(q[:n_frames])                           # 4,4096,320 -> 32,4096,320//8(40)
            if factor == 3:     # IP2P
                q_image = self.head_to_batch_dim(q[n_frames: 2*n_frames])
                q_uncond = self.head_to_batch_dim(q[2 * n_frames:])
            else:               # SD-1.5
                q_uncond = self.head_to_batch_dim(q[n_frames:])

            k_text = self.head_to_batch_dim(k_text)                                 # 32,16384,40
            if factor == 3:
                k_image = self.head_to_batch_dim(k_image)
            k_uncond = self.head_to_batch_dim(k_uncond)
            
            v_text = self.head_to_batch_dim(v_text)                                 # 32,16384,40
            if factor == 3:
                v_image = self.head_to_batch_dim(v_image)
            v_uncond = self.head_to_batch_dim(v_uncond)

            out_text = []
            if factor == 3:
                out_image = []
            out_uncond = []

            q_text = q_text.view(n_frames, h, sequence_length, dim // h)                # 4,8,4096,40
            k_text = k_text.view(n_frames, h, sequence_length * n_frames, dim // h)     # 4,8,16384,40
            v_text = v_text.view(n_frames, h, sequence_length * n_frames, dim // h)     # 4,8,16384,40

            if factor == 3:
                q_image = q_image.view(n_frames, h, sequence_length, dim // h)
                k_image = k_image.view(n_frames, h, sequence_length * n_frames, dim // h)
                v_image = v_image.view(n_frames, h, sequence_length * n_frames, dim // h)

            q_uncond = q_uncond.view(n_frames, h, sequence_length, dim // h)
            k_uncond = k_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_uncond = v_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)

            for j in range(h):      # torch.bmm : batch matrix multiplication 
                sim_text = torch.bmm(q_text[:, j], k_text[:, j].transpose(-1, -2)) * self.scale         # 4,4096,16384  (pixel-pairwise attention scores)
                if factor == 3:
                    sim_image = torch.bmm(q_image[:, j], k_image[:, j].transpose(-1, -2)) * self.scale
                sim_uncond = torch.bmm(q_uncond[:, j], k_uncond[:, j].transpose(-1, -2)) * self.scale
                
                out_text.append(torch.bmm(sim_text.softmax(dim=-1), v_text[:, j]))              # output embeddings : 
                if factor == 3:
                    out_image.append(torch.bmm(sim_image.softmax(dim=-1), v_image[:, j]))
                out_uncond.append(torch.bmm(sim_uncond.softmax(dim=-1), v_uncond[:, j]))

            out_text = torch.cat(out_text, dim=0).view(h, n_frames, sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
            if factor == 3:
                out_image = torch.cat(out_image, dim=0).view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
            out_uncond = torch.cat(out_uncond, dim=0).view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)

            if factor == 3:
                out = torch.cat([out_text, out_image, out_uncond], dim=0)
            else:
                out = torch.cat([out_text, out_uncond], dim=0)
            out = self.batch_to_head_dim(out)       # 8,4096,320

            return to_out(out)

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.forward = sa_forward(module.attn1)


def compute_camera_distance(cams, key_cams):
    cam_centers = [cam.camera_center for cam in cams]
    key_cam_centers = [cam.camera_center for cam in key_cams] 
    cam_centers = torch.stack(cam_centers).cuda()
    key_cam_centers = torch.stack(key_cam_centers).cuda()
    cam_distance = torch.cdist(cam_centers, key_cam_centers)

    return cam_distance   

def make_dge_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class DGEBlock(block_class):
        def forward(
            self,
            hidden_states,                      # 2,B,4096,320  (Latent Pixel-wise embeddings)    
            attention_mask=None,
            encoder_hidden_states=None,         # 2B,81,768    (Prompt embeds)
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            
            # hidden states : pixel-wise embeddings (dim=320) of each pixel in 64x64 latent
            # Reshape 8 -> 2,4 (two rows of replicas of the same 4 frames)
            batch_size, sequence_length, dim = hidden_states.shape      # 8,4096,320 ... 8,1024,640         # Normal DGE : 3,4096,320
            factor = self.conditioning_factor      # 2 for ZeST (SD-1.5), 3 for normal DGE (IP2P)
            assert factor in [2,3]                 # Support both SD-1.5 and IP2P
            n_frames = batch_size // factor      # 3
            hidden_states = hidden_states.view(factor, n_frames, sequence_length, dim)   # 2,4,4096,320  (Actual : 3)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)
        
            norm_hidden_states = norm_hidden_states.view(factor, n_frames, sequence_length, dim)  # 3
            if self.pivotal_pass:
                self.pivot_hidden_states = norm_hidden_states       
            if not hasattr(self, 'use_normal_attn'):
                if self.pivotal_pass:
                    self.pivot_hidden_states = norm_hidden_states       # 2,4,4096,320
                else:
                    '''
                    NOTE:
                        When self.pivotal_pass is True, self.pivot_hidden_states is initialized to norm_hidden_states of the pivotal frames.
                        Hence, before batched forward pass, a single forward pass step (just before Line with register_pivotal=False in edit_zest_latents())
                        is done to initialize self.pivot_hidden_states. 

                        Next, when pivotal_pass is False,
                        Now, batched inputs are passed here, and self.pivot_hidden_states is used to get the closest cam pivot hidden states in this block.
                    '''
                    batch_idxs = [self.batch_idx]
                    if self.batch_idx > 0:
                        batch_idxs.append(self.batch_idx - 1)
                    idx1 = []
                    idx2 = []
                    cam_distance = compute_camera_distance(self.cams, self.key_cams)        # shape : 5,4
                    cam_distance_min = cam_distance.sort(dim=-1)
                    closest_cam = cam_distance_min[1][:,:len(batch_idxs)]                   # 5,1
                    closest_cam_pivot_hidden_states = self.pivot_hidden_states[1][closest_cam]      # 5,1,4096,320
                    sim = torch.einsum('bld,bcsd->bcls', norm_hidden_states[1] / norm_hidden_states[1].norm(dim=-1, keepdim=True), closest_cam_pivot_hidden_states / closest_cam_pivot_hidden_states.norm(dim=-1, keepdim=True)).squeeze()      
                    # 5,4096,4096 (cosine similarity between each pixel in the current frame and the closest pivot frame)    
                    if len(batch_idxs) == 2:
                        sim1, sim2 = sim.chunk(2, dim=1)
                        sim1 = sim1.view(-1, sequence_length)
                        sim2 = sim2.view(-1, sequence_length)
                        sim1_max = sim1.max(dim=-1)
                        sim2_max = sim2.max(dim=-1)
                        idx1.append(sim1_max[1])
                        idx2.append(sim2_max[1])

                    else:
                        sim = sim.view(-1, sequence_length)         # 5*4096,4096
                        sim_max = sim.max(dim=-1)
                        idx1.append(sim_max[1])

                    if len(batch_idxs) == 2:
                        idx1 = []
                        idx2 = []
                        pivot_this_batch = self.pivot_this_batch
                        
                        idx1_epipolar, idx2_epipolar = self.epipolar_constrains[sequence_length].gather(dim=1, index=closest_cam[:, :, None, None].expand(-1, -1, self.epipolar_constrains[sequence_length].shape[2], self.epipolar_constrains[sequence_length].shape[3])).cuda().chunk(2, dim=1)
                        idx1_epipolar = idx1_epipolar.reshape(n_frames, sequence_length, sequence_length)
    
                        idx1_epipolar[pivot_this_batch, ...] = False
                        idx2_epipolar = idx2_epipolar.reshape(n_frames, sequence_length, sequence_length)

                        idx1_epipolar = idx1_epipolar.reshape(n_frames * sequence_length, sequence_length)
                        idx2_epipolar = idx2_epipolar.reshape(n_frames * sequence_length, sequence_length)
                        idx2_sum = idx2_epipolar.sum(dim=-1)
                        idx1_sum = idx1_epipolar.sum(dim=-1)

                        idx1_epipolar[idx1_sum == sequence_length, :] = False
                        idx2_epipolar[idx2_sum == sequence_length, :] = False
                        sim1[idx1_epipolar] = 0
                        sim2[idx2_epipolar] = 0

                        sim1_max = sim1.max(dim=-1)
                        sim2_max = sim2.max(dim=-1)
                        idx1.append(sim1_max[1])
                        idx2.append(sim2_max[1])

                        
                    else:
                        idx1 = []
                        pivot_this_batch = self.pivot_this_batch    # Index of pivot of this batch

                        idx1_epipolar = self.epipolar_constrains[sequence_length].gather(dim=1, index=closest_cam[:, :, None, None].expand(-1, -1, self.epipolar_constrains[sequence_length].shape[2], self.epipolar_constrains[sequence_length].shape[3])).cuda()

                        idx1_epipolar = idx1_epipolar.view(n_frames, -1, sequence_length)       # 5,4096,4096
                        idx1_epipolar[pivot_this_batch, ...] = False            # This is set to False to remove similar (pivot,pivot) pair contribution

                        idx1_epipolar = idx1_epipolar.view(n_frames * sequence_length, sequence_length)
                        idx1_sum = idx1_epipolar.sum(dim=-1)
                        idx1_epipolar[idx1_sum == sequence_length, :] = False
                        sim[idx1_epipolar] = 0
                        sim_max = sim.max(dim=-1)
                        idx1.append(sim_max[1])
                            
                    #####
                    # TODO: Check this part, 
                    idx1 = torch.stack(idx1 * factor, dim=0) # 2, 4096 * 5
                    idx1 = idx1.squeeze(1)


                    if len(batch_idxs) == 2:
                        idx2 = torch.stack(idx2 * factor, dim=0) # 3, n_frames * seq_len
                        idx2 = idx2.squeeze(1)

                    #####
            
            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if hasattr(self, 'use_normal_attn'):
                # print("use normal attn")
                self.attn_output = self.attn1.normal_attn(
                        norm_hidden_states.view(batch_size, sequence_length, dim),
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        **cross_attention_kwargs,
                    )         
            else:
                '''
                NOTE: Control entered here during the first pivotal pass.
                '''
                if self.pivotal_pass:
                    # norm_hidden_states.shape = 3, n_frames * seq_len, dim
                    self.attn_output = self.attn1(
                            norm_hidden_states.view(batch_size, sequence_length, dim),
                            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,     # None
                            **cross_attention_kwargs,
                        )
                    # 3, n_frames * seq_len, dim - > 3 * n_frames, seq_len, dim     (8,4096,320)
                    self.kf_attn_output = self.attn_output      # Store the key frame attention outputs in the first pivotal pass

                else:
                    batch_kf_size, _, _ = self.kf_attn_output.shape
                    self.attn_output = self.kf_attn_output.view(factor, batch_kf_size // factor, sequence_length, dim)[:,     # 3
                                    closest_cam]

            if self.use_ada_layer_norm_zero:
                self.n = gate_msa.unsqueeze(1) * self.attn_output

            # gather values from attn_output, using idx as indices, and get a tensor of shape 3, n_frames, seq_len, dim
            if not hasattr(self, 'use_normal_attn'):
                if not self.pivotal_pass:
                    if len(batch_idxs) == 2:            # Normal DGE : batch_idxs = [0] (len = 1)
                        attn_1, attn_2 = self.attn_output[:, :, 0], self.attn_output[:, :, 1]
                        idx1 = idx1.view(factor, n_frames, sequence_length)          # 2, n_frames, 4096
                        idx2 = idx2.view(factor, n_frames, sequence_length)
                        attn_output1 = attn_1.gather(dim=2, index=idx1.unsqueeze(-1).repeat(1, 1, 1, dim))
                        attn_output2 = attn_2.gather(dim=2, index=idx2.unsqueeze(-1).repeat(1, 1, 1, dim))
                        d1 = cam_distance_min[0][:,0]           # 20,1  (closest key cam distance)
                        d2 = cam_distance_min[0][:,1]           # 20,1  (2nd closest key cam distance)
                        w1 = d2 / (d1 + d2)                     # average the two closest key cam distances
                        w1 = torch.sigmoid(w1)
                        w1 = w1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(factor, 1, sequence_length, dim)     # // 3
                        attn_output1 = attn_output1.view(factor, n_frames, sequence_length, dim)     # // 3
                        attn_output2 = attn_output2.view(factor, n_frames, sequence_length, dim)     # // 3
                        attn_output = w1 * attn_output1 + (1 - w1) * attn_output2
                        attn_output = attn_output.reshape(
                            batch_size, sequence_length, dim).half()
                    else:
                        idx1 = idx1.view(factor, n_frames, sequence_length)      # 3    2, n_frames, 4096
                        attn_output = self.attn_output[:,:,0].gather(dim=2, index=idx1.unsqueeze(-1).repeat(1, 1, 1, dim))
                        attn_output = attn_output.reshape(batch_size, sequence_length, dim).half()                       
                else:   #pivotal pass
                    attn_output = self.attn_output      # 8,4096,320
            else:
                attn_output = self.attn_output
            
            
            hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)  # 8,4096,320
            hidden_states = attn_output + hidden_states             # x + attn_output(x)
            hidden_states = hidden_states.to(self.norm2.weight.dtype)
            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)          # 8,4096,320

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]


            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

    return DGEBlock

