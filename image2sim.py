from typing import List, NamedTuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from enum import IntEnum
from scipy.ndimage import distance_transform_edt
from torchvision import transforms
import math
from torch.func import functional_call, jvp
import os
import torch.hub
import torchvision.models._api as torch_api
from torch.utils.checkpoint import checkpoint
import torchvision.transforms.functional as TF
from torch_kdtree import build_kd_tree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import distance_transform_edt

# ==========================================================
# Please ensure these two files genuinely exist on your server
# ==========================================================
LOCAL_VGG_PATH = os.path.abspath("pretrained_models/vgg16-397923af.pth")
LOCAL_DINO_PATH = os.path.abspath("pretrained_models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")

def universal_offline_loader(url, *args, **kwargs):
    """
    For the convenience of offline servers, local files will be loaded
    whenever the keywords match, regardless of whether LPIPS or DINOv3 is called.
    """
    target_path = None
    
    if "vgg16" in url:
        target_path = LOCAL_VGG_PATH
    elif "dinov3" in url or "dino" in url:
        target_path = LOCAL_DINO_PATH

    if target_path:
        print(f"\n[Offline-Hack] URL: {url}")
        print(f"[Offline-Hack]: {target_path}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Error：Local file missing -> {target_path}")
        
        return torch.load(target_path, map_location='cpu', weights_only=False)

    raise RuntimeError(f"Attempt to request an unknown URL in an offline environment: {url}")

# ==========================================================
# [Multiple Hijacks] Covering all possible entry points
# ==========================================================

# 1. Hijack torch.hub (commonly used by DINOv3)
torch.hub.load_state_dict_from_url = universal_offline_loader

# 2. Hijack torchvision (commonly used by LPIPS)
if hasattr(torch_api, 'load_state_dict_from_url'):
    torch_api.load_state_dict_from_url = universal_offline_loader

# 3. Specific Target: Intercept DINOv3 if it directly uses torch.load with a URL path
_orig_torch_load = torch.load
def hacked_torch_load(f, *args, **kwargs):
    # If f is a URL string containing 'dino'
    if isinstance(f, str) and ("http" in f) and ("dino" in f):
        return universal_offline_loader(f, *args, **kwargs)
    # Otherwise, load normally, but enforce weights_only=False to prevent subsequent errors
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(f, *args, **kwargs)

torch.load = hacked_torch_load

print(">>> [Done] Offline load interceptor fully injected. Ready to launch DINOv3 & LPIPS <<<\n")

try:
    from lpips import LPIPS
except ImportError:
    print("[Warning] lpips not found. Perceptual loss will be disabled.")
    LPIPS = None
    exit()

try:
    from diff_gaussian_pano_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except ImportError:
    print("[Warning] diff_gaussian_pano_rasterization not found. PanoAnisotropicSplatting will fail.")
    exit()

class PanoData(NamedTuple):
  """Data corresponding to a Matterport3D panorama."""
  position: Tensor
  rgb: Tensor
  semantic: Tensor
  depth: Tensor

class OutputData(NamedTuple):
  """Output tuple for Video2Sim model outputs."""
  proj_features: Tensor       # Projected 3D features
  aligned_features: Tensor    # Output from Semantic Aligner
  pred_rgb: Tensor            # Final Refined RGB
  proj_rgb: Tensor            # Projected RGB
  pred_depth: Tensor          # Final Refined Depth
  proj_depth: Tensor          # Projected Depth
  proj_alpha: Tensor          # Projected Alpha

class MemoryState(NamedTuple):
  """Tuple for memory state."""
  coords: Tensor        # (B, 3, N)
  feats: Tensor         # (B, 16, N) - Pixel-level features
  rgb: Tensor           # (B, 3, N) - Basic RGB color
  gs_attrs: Tensor      # (B, 8, N) - Gaussian Attributes (3 Scale + 4 Quat + 1 Opacity)

class PanoAnisotropicSplatting(nn.Module):
    """
    [Fixed] Supports Network Predicted Scale and Rotation.
    """
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.register_buffer('identity_matrix', torch.eye(4).unsqueeze(0))
        self.register_buffer('pi_inv', torch.tensor(1.0 / math.pi))
        self.register_buffer('two_pi_inv', torch.tensor(1.0 / (2 * math.pi)))

    def forward(self, position, heading, points, rgb_colors, features, opacity, scales, rotations, max_depth=15.0):
        B, C, N = features.shape
        H, W = self.height, self.width
        device = points.device
        
        rendered_rgb_list = []
        rendered_feats_list = []
        rendered_depth_list = []
        rendered_alpha_list = []

        # 1. Calculate the panorama scale factor (must be consistent with the logic in forward.cu)
        factor_u = W / (2 * math.pi)
        factor_v = H / math.pi
        
        # 2. Construct a pseudo-projection matrix
        # preprocessCUDA in backward.cu will read proj[0] and proj[5] as scale factors
        proj_matrix = torch.zeros((4, 4), device=device, dtype=torch.float32)
        proj_matrix[0, 0] = factor_u
        proj_matrix[1, 1] = factor_v
        proj_matrix[2, 2] = 1.0
        proj_matrix[3, 3] = 1.0
        
        # Construct the World Matrix (object-to-world, or camera position in world coordinates)
        heading = torch.pi / 2 - heading
        cos_t = torch.cos(heading)
        sin_t = torch.sin(heading)

        # Construct the rotation component (assuming Z-up)
        R = torch.zeros((B, 3, 3), device=device)
        R[:, 0, 0] = cos_t
        R[:, 0, 1] = -sin_t
        R[:, 1, 0] = sin_t
        R[:, 1, 1] = cos_t
        R[:, 2, 2] = 1

        # Construct the View Matrix (R_inv, -R_inv * t)
        # For a pure rotation matrix, R_inv = R.T (transpose)
        R_view = R.transpose(1, 2) 
        t_view = -torch.bmm(R_view, position.unsqueeze(2)).squeeze(2)

        view_matrix = torch.zeros((B, 4, 4), device=device, dtype=torch.float32)
        view_matrix[:, :3, :3] = R_view
        view_matrix[:, :3, 3] = t_view
        view_matrix[:, 3, 3] = 1.0


        # 1. Define the standard coordinate transformation matrix (Math Matrix)
        # Convert Body Frame (X-Fwd, Y-Left, Z-Up) to Camera/CUDA Frame (X-Right, Y-Down, Z-Fwd)
        # Row 0 (New X/Right) = -Old Y
        # Row 1 (New Y/Down)  = -Old Z
        # Row 2 (New Z/Fwd)   = Old X
        conversion = torch.tensor([
            [ 0, -1,  0,  0],
            [ 0,  0, -1,  0],
            [ 1,  0,  0,  0],
            [ 0,  0,  0,  1]
        ], device=device, dtype=view_matrix.dtype)

        # 2. Apply coordinate transformation (C * V)
        # At this point, view_matrix is the standard World-to-Camera matrix
        view_matrix = torch.matmul(conversion, view_matrix).to(torch.float32)

        # 3. [Critical Step] Transpose the matrix to adapt to the CUDA memory layout
        # The 3DGS CUDA kernels expect the matrix to be transposed (i.e., with basis vectors as columns) when reading it
        view_matrix = view_matrix.transpose(1, 2).contiguous()

        for b in range(B):
            pts = points[b]       # (3, N)
            colors = rgb_colors[b]
            feats = features[b]   # (C, N)
            opac = opacity[b]     # (1, N)
            sc = scales[b]        # (3, N)
            rot = rotations[b]    # (4, N)


            rel_pos = pts - position[b].unsqueeze(1) # (3, N)
            dist = torch.norm(rel_pos, dim=0)        # (N,)
            
            # Filter out points that are too close to the camera (e.g., less than 0.05m) to prevent division by zero
            is_finite = torch.isfinite(dist)
            valid_mask = ((dist > 0.01) & (dist < max_depth) & is_finite).view(-1)
            
            if not valid_mask.any():
                rendered_rgb_list.append(torch.zeros((3, H, W), device=device))
                rendered_feats_list.append(torch.zeros((C, H, W), device=device))
                rendered_depth_list.append(torch.zeros((1, H, W), device=device))
                rendered_alpha_list.append(torch.zeros((1, H, W), device=device))
                continue
            
            x_v = pts[0, valid_mask]
            y_v = pts[1, valid_mask]
            z_v = pts[2, valid_mask]
            
            # For attributes with shape (C, N), use slicing [:, mask]
            feats_v = feats[:, valid_mask].transpose(0, 1).contiguous()   # (N_v, C)
            colors_v = colors[:, valid_mask].transpose(0, 1).contiguous()
            opac_v = opac[:, valid_mask].squeeze(0).unsqueeze(1).contiguous()          # (N_v,)
            
            # Process Scale and Rotation
            sc_v = sc[:, valid_mask].transpose(0, 1).contiguous()
            rot_v = rot[:, valid_mask].transpose(0, 1)
            rot_v = F.normalize(rot_v, p=2, dim=1, eps=1e-4)
            
            means3D_input = torch.stack([x_v, y_v, z_v], dim=-1).squeeze() # Shape: (N_v, 3)

            means2D_input = torch.zeros((means3D_input.shape[0], 2), dtype=means3D_input.dtype, device=device, requires_grad=True)

            raster_settings = GaussianRasterizationSettings(
                image_height=int(H),
                image_width=int(W),
                tanfovx=1.0, tanfovy=1.0,
                bg=torch.zeros(C + 1, device=device),
                scale_modifier=1.0,
                viewmatrix=view_matrix[b].contiguous(),
                projmatrix=proj_matrix.contiguous(),
                sh_degree=0, 
                campos=position[b],
                prefiltered=False,
                debug=False
            )
            
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            rendered_rgb, rendered_feat, _, rendered_depth, rendered_alpha = rasterizer(
                means3D=means3D_input,
                means2D=means2D_input,
                opacities=opac_v,
                shs=None,
                colors_precomp=colors_v,       
                semantic_feature=feats_v,      
                scales=sc_v,
                rotations=rot_v,
                cov3D_precomp=None
            )

            rendered_rgb_list.append(rendered_rgb)
            rendered_feats_list.append(rendered_feat)
            rendered_depth_list.append(rendered_depth)
            rendered_alpha_list.append(rendered_alpha)
            
        final_rgb = torch.stack(rendered_rgb_list, dim=0) 
        final_feat = torch.stack(rendered_feats_list, dim=0)   
        final_depth = torch.stack(rendered_depth_list, dim=0)  
        final_alpha = torch.stack(rendered_alpha_list, dim=0)
        
        return final_rgb, final_feat, final_depth, final_alpha

# --- Architecture Modules ---

class DINOv3Wrapper(nn.Module):
    """Wrapper for DINOv3 ViT-L/16 (Frozen)."""
    def __init__(self):
        super().__init__()
        # Using hub load as proxy. Ensure path is correct in your env.
        # Try local/custom path first as per prompt
        self.backbone = torch.hub.load("dinov3", 'dinov3_vitl16', source='local', weights="pretrained_models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.embed_dim = 1024
        self.patch_size = 16 # DINOv3 is 16

    def forward(self, x):
        # x: (B, 3, H, W)
        B, C, H, W = x.shape
        h_p, w_p = H // self.patch_size, W // self.patch_size
        
        with torch.no_grad():
            if isinstance(self.backbone, nn.Identity):
                return torch.zeros((B, 1024, h_p, w_p), device=x.device)

            # DINOv2/v3 forward_features usually returns a dict
            res = self.backbone.forward_features(x, masks=None)
            patch_tokens = res['x_norm_patchtokens']
            
            # Reshape
            patch_tokens = patch_tokens.reshape(B, h_p, w_p, self.embed_dim).permute(0, 3, 1, 2)
            
                
        return patch_tokens # (B, 1024, H/16, W/16)


    
class PanoConv2d(nn.Module):
    """
    Conv2D for Panorama。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=False):
        super().__init__()
        
        # Use user-specified padding if provided (e.g., 0);
        # Otherwise, fall back to 'same' padding logic (k // 2)
        if padding is not None:
            self.pad_amt = padding
        else:
            self.pad_amt = kernel_size // 2 
            
        # Set the padding of the actual convolutional layer to 0, as we pad manually in the forward pass
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        # x shape: (B, C, H, W)
        H = x.shape[-2]
        W = x.shape[-1]
        
        if self.pad_amt > 0:
            if 2*H == W: # Panorama mode
                x_h = F.pad(x, (self.pad_amt, self.pad_amt, 0, 0), mode='circular')
                x_final = F.pad(x_h, (0, 0, self.pad_amt, self.pad_amt), mode='replicate')
            else: # Pinhole mode
                x_final = F.pad(x, (self.pad_amt, self.pad_amt, self.pad_amt, self.pad_amt), mode='replicate')
        else:
            x_final = x
        
        feature = self.conv(x_final)
        return feature


class LayerNorm2d(nn.Module):
    """(N, C, H, W)  LayerNorm"""
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class GRN(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x):
        Gx = torch.sqrt((x ** 2).sum(dim=(1, 2), keepdim=True) + self.eps)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x
    

class PanoBlock(nn.Module):
    """
    A unified block architecture incorporating PanoConv, ConvNeXt V2 with GRN, and GLU.
    """
    def __init__(self, dim, kernel_size=7, drop_path=0.):
        super().__init__()
        
        # 1. PanoConv for spatial fusion
        self.dwconv = PanoConv2d(dim, dim, kernel_size=kernel_size)
        self.norm = LayerNorm2d(dim)
        
        # 2. MLP with GLU & GRN for chanel fusion
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.grn = GRN(2 * dim)
        self.pwconv2 = nn.Linear(2 * dim, dim)
        
        self.act = nn.GELU()
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        
        x = x.permute(0, 2, 3, 1) 
        
        # GLU
        x_split = self.pwconv1(x)
        x1, x2 = torch.chunk(x_split, 2, dim=-1)
        x = x1 * self.act(x2) # Gating
        
        x = self.grn(x)
        x = self.pwconv2(x)
        
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x
    
    
class SurfaceNormalEstimator(nn.Module):
    """
    Robust 5x5 Sobel Normal Estimator.
    Larger receptive field naturally suppresses high-frequency noise from raw depth sensors
    """
    def __init__(self):
        super().__init__()
        # 5x5 Sobel Kernels
        self.register_buffer('sobel_x', torch.tensor([
            [-1, -2, 0, 2, 1],
            [-4, -8, 0, 8, 4],
            [-6,-12, 0,12, 6],
            [-4, -8, 0, 8, 4],
            [-1, -2, 0, 2, 1]
        ], dtype=torch.float32).view(1, 1, 5, 5))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -4, -6, -4, -1],
            [-2, -8,-12, -8, -2],
            [ 0,  0,  0,  0,  0],
            [ 1,  4,  6,  4,  1],
            [ 2,  8, 12,  8,  2]
        ], dtype=torch.float32).view(1, 1, 5, 5))
        
        # Sum of absolute values in one half of X kernel is 48. 
        self.scale = 1.0 / 48.0 

    def forward(self, depth):
        # depth: (B, 1, H, W)
        
        # 1. Bilateral-like simple approximation
        depth = F.avg_pool2d(depth, kernel_size=3, stride=1, padding=1)

        # 2. Padding (Size 2 for 5x5 kernel)
        depth_pad = F.pad(depth, (2, 2, 2, 2), mode='replicate')
        
        # 3. Compute Gradients
        grad_x = F.conv2d(depth_pad, self.sobel_x) * self.scale
        grad_y = F.conv2d(depth_pad, self.sobel_y) * self.scale
        
        # 4. Construct Normals
        # The choice of z_value determines the "steepness" of the normals.
        # When sensor noise is significant, increasing z_value makes the normals lean 
        # more towards the camera (smoother), thereby suppressing lateral random noise.
        z_value = 0.5 # If noise persists, try increasing to 1.0
        z = torch.ones_like(grad_x) * z_value
        
        # (-dx, -dy, z)
        normal = torch.cat([-grad_x, -grad_y, z], dim=1)
        normal = F.normalize(normal, p=2, dim=1, eps=1e-4) 
        
        return normal


class DetailEncoder(nn.Module):
    """
    A shallow CNN specifically designed to encode high-resolution RGB and Depth/Normal data.
    Generates a feature pyramid: [H, H/2, H/4, H/8]
    """
    def __init__(self, in_ch=3+1+3): # in_ch = RGB(3) + Depth(1) + CoarseNormal(3) + Ray(3) = 10
        super().__init__()
        
        self.stem = nn.Sequential(
            PanoConv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Level 1: H (Full Res) -> 32ch
        self.layer1 = PanoBlock(32)
        
        # Level 2: H/2 -> 64ch
        self.down2 = nn.Sequential(
            PanoConv2d(32, 64, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(64), nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Level 3: H/4 -> 128ch
        self.down3 = nn.Sequential(
            PanoConv2d(64, 128, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(128), nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Level 4: H/8 -> 256ch
        self.down4 = nn.Sequential(
            PanoConv2d(128, 256, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(256), nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # x: (B, 7, H, W)
        x = self.stem(x)
        f1 = self.layer1(x)  # H
        f2 = self.down2(f1)  # H/2
        f3 = self.down3(f2)  # H/4
        f4 = self.down4(f3)  # H/8
        return [f1, f2, f3, f4]


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Upsample/Expand channels first
        self.conv = PanoConv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        # Add a smoothing layer to mitigate checkerboard artifacts
        self.smooth = PanoConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True) # Switch to LeakyReLU to sustain/prevent vanishing gradient flow

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.smooth(x)
        return self.act(x)
       

class FeatureUpsampler(nn.Module):
    """
    [Refactored] Geometry-Aware Upsampler with NextGen Blocks.
    
    Inputs:
        - DINO Tokens (B, 1024, H/16, W/16)
        - RGB Image   (B, 3, H, W)
        - Depth Image (B, 1, H, W)
        - Ray Map     (B, 3, H, W)
    """
    def __init__(self, in_dim=1024, out_feat_dim=16):
        super().__init__()
        
        # 1. Auxiliary geometric computing for coarse hints
        self.coarse_normal_estimator = SurfaceNormalEstimator() 

        # 2. Detail encoder (retaining original pipeline; highly lightweight yet effective)
        # in_ch = RGB(3) + Depth(1) + CoarseNormal(3) + Ray(3) = 10
        self.detail_encoder = DetailEncoder(in_ch=10) 
        
        # 3. DINO feature projection
        self.proj_dino = nn.Sequential(
            PanoConv2d(in_dim, 512, kernel_size=1),
            LayerNorm2d(512),
            nn.GELU()
        )
        
        # --- Upsampling Stages (Refactored) ---
        
        # Stage 4: H/16 -> H/8
        self.up4 = PixelShuffleUpsample(512, 256) 
        # Fusion: Cat(256, 256) -> 512 -> Proj(256) -> EncoderBlock
        self.proj4 = PanoConv2d(256 + 256, 256, 1)
        self.fusion4 = PanoBlock(256)
        
        # Stage 3: H/8 -> H/4
        self.up3 = PixelShuffleUpsample(256, 128) 
        # Fusion: Cat(128, 128) -> 256 -> Proj(128) -> EncoderBlock
        self.proj3 = PanoConv2d(128 + 128, 128, 1)
        self.fusion3 = PanoBlock(128)
        
        # Stage 2: H/4 -> H/2
        self.up2 = PixelShuffleUpsample(128, 64) 
        # Fusion: Cat(64, 64) -> 128 -> Proj(64) -> EncoderBlock
        self.proj2 = PanoConv2d(64 + 64, 64, 1)
        self.fusion2 = PanoBlock(64)
        
        # Stage 1: H/2 -> H/1
        self.up1 = PixelShuffleUpsample(64, 64) 
        # Fusion: Cat(64, 32) -> 96 -> Proj(64) -> EncoderBlock
        self.proj1 = PanoConv2d(64 + 32, 64, 1)
        self.fusion1 = PanoBlock(64)
        
        # 4. Heads
        
        # Head A: semantic (16 dims)
        self.feat_head = nn.Sequential(
            PanoConv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            PanoConv2d(64, out_feat_dim, kernel_size=1)
        )
        
        # Head B: 3DGS  (Scale 3 + Rot 4 + Opacity 1 = 8)
        self.gs_head = nn.Sequential(
            PanoConv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            PanoConv2d(64, 8, kernel_size=1)
        )

        # Head C: Learnable Normal (3 dims)
        self.normal_head = nn.Sequential(
            PanoConv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            PanoConv2d(64, 3, kernel_size=1)
        )

    def _shortest_arc_quat(self, n):
        """Calculate the rotation quaternion from (0,0,1) to the normal vector n"""
        nx, ny, nz = n[:, 0:1], n[:, 1:2], n[:, 2:3]
        eps = 1e-6
        w = 1.0 + nz + eps
        x = -ny
        y = nx
        z = torch.zeros_like(nx)
        q = torch.cat([w, x, y, z], dim=1)
        return F.normalize(q, p=2, dim=1, eps=1e-4)

    def _quat_multiply(self, q1, q2):
        """Quaternion multiplication"""
        w1, x1, y1, z1 = torch.split(q1, 1, dim=1)
        w2, x2, y2, z2 = torch.split(q2, 1, dim=1)
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.cat((w, x, y, z), dim=1)

    def forward(self, dino_tokens, rgb_img, depth_img, ray_map):
        """
        Input:
            dino_tokens: DINOv3 Features
            rgb_img: (B, 3, H, W)
            depth_img: (B, 1, H, W)
            ray_map: (B, 3, H, W)
        """
        # 1. Input
        with torch.no_grad():
            coarse_normals = self.coarse_normal_estimator(depth_img) 
        
        detail_input = torch.cat([rgb_img, depth_img, coarse_normals, ray_map], dim=1)
        
        # 2. Detail Encoder Pyramid
        # details[0]: H, details[1]: H/2, details[2]: H/4, details[3]: H/8
        details = self.detail_encoder(detail_input)
        
        # 3. DINO Main Path
        x = self.proj_dino[0](dino_tokens)
        x = self.proj_dino[1](x) # LayerNorm
        x = self.proj_dino[2](x) # GELU
        
        # Stage 4 (H/16 -> H/8)
        x = self.up4(x) 
        x = torch.cat([x, details[3]], dim=1) 
        x = self.proj4(x)   # Reduce dim
        x = self.fusion4(x) # Encoder Block
        
        # Stage 3 (H/8 -> H/4)
        x = self.up3(x) 
        x = torch.cat([x, details[2]], dim=1) 
        x = self.proj3(x)
        x = self.fusion3(x)

        # Stage 2 (H/4 -> H/2)
        x = self.up2(x) 
        x = torch.cat([x, details[1]], dim=1) 
        x = self.proj2(x)
        x = self.fusion2(x)

        # Stage 1 (H/2 -> H/1)
        x = self.up1(x) 
        x = torch.cat([x, details[0]], dim=1) 
        x = self.proj1(x)
        x = self.fusion1(x)
        
        # 4. Heads Output
        feats = torch.tanh(self.feat_head(x))
        
        # --- 3DGS Attributes Prediction ---
        # A. Learnable Normal (Refine coarse normal)
        pred_normal_raw = self.normal_head(x)
        pred_normal = F.normalize(pred_normal_raw, p=2, dim=1, eps=1e-4)
        
        # B. Raw Attributes
        raw_gs = self.gs_head(x)
        
        # Scale: sigmoid ensure positive
        scales = torch.sigmoid(raw_gs[:, :3]) * 0.04 + 1e-4
        
        # Opacity: Sigmoid [0, 1]
        opacity = torch.sigmoid(raw_gs[:, 7:8])
        
        # Rotation: Normal-guided + Residual
        q_base = self._shortest_arc_quat(pred_normal) # Base rotation aligning Z to Normal
        
        raw_residual = raw_gs[:, 3:7].clone()
        raw_residual[:, 0] += 1.0 # Identity bias
        q_res = F.normalize(raw_residual, p=2, dim=1, eps=1e-4)
        
        rotations = self._quat_multiply(q_base, q_res) # Apply residual
        
        gs_attrs = torch.cat([scales, rotations, opacity], dim=1)
        
        return feats, gs_attrs



class TransformerBottleneck(nn.Module):
    """
    Standard Transformer layer with built-in 2D learnable position embeddings.
    Supports input shapes of (B, C, H, W) and outputs shapes of (B, C, H, W).
    """
    def __init__(self, dim, num_heads=8, h_base=32, w_base=64):
        super().__init__()
        # 1. Position Embeddings (H/16 resolution: 512x1024 input -> 32x64 feature)
        # Initialized with common dimensions; if resolution changes during inference, 
        # it will automatically interpolate in the forward pass.
        self.pos_embed = nn.Parameter(torch.randn(1, dim, h_base, w_base) * .02)
        
        # 2. Standard Transformer
        # batch_first=True, norm_first=True (Pre-Norm for more stable training)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Add PE before flattening
        x = x + self.pos_embed 
        
        # --- 2. (B, C, H, W) -> (B, H*W, C) ---
        x_flat = x.flatten(2).transpose(1, 2)
        
        # --- 3. Attention ---
        x_out = self.transformer(x_flat)
        
        # --- 4. (B, H*W, C) -> (B, C, H, W) ---
        x_out = x_out.transpose(1, 2).view(B, C, H, W)
        
        return x_out


class SemanticAligner(nn.Module):
    def __init__(self, in_feat_dim=24, embed_dim=1024, num_heads=8):
        super().__init__()

        # padding=0，1024 -> 64
        self.down_proj = nn.Sequential(
            # H -> H/4
            PanoConv2d(in_feat_dim, embed_dim // 2, kernel_size=4, stride=4, padding=0),
            LayerNorm2d(embed_dim // 2),
            nn.GELU(),
            # H/4 -> H/16
            PanoConv2d(embed_dim // 2, embed_dim, kernel_size=4, stride=4, padding=0)
        )

        # Transformer
        self.transformer = nn.Sequential(TransformerBottleneck(embed_dim,num_heads=num_heads),
                                         TransformerBottleneck(embed_dim,num_heads=num_heads))

    def forward(self, x):
        """
        x: (B, 16, H, W)
        """
        # Patchify Features
        x = self.down_proj(x) # (B, 1024, H_p, W_p)

        # Transformer
        x_aligned = self.transformer(x)
        
        return x_aligned



class SemanticInjector(nn.Module):
    """
    Modulate spatial features using DINO semantic features (a simplified SPADE variant).
    Significantly outperforms simple concatenation (Concat), preventing noise in geometric 
    features from contaminating semantic generation.
    """
    def __init__(self, spatial_dim, semantic_dim):
        super().__init__()
        self.norm = LayerNorm2d(spatial_dim)
        # gamma and beta
        self.modulator = nn.Sequential(
            nn.SiLU(),
            PanoConv2d(semantic_dim, spatial_dim * 2, kernel_size=1)
        )

    def forward(self, spatial_feat, semantic_feat): 

        style = self.modulator(semantic_feat)
        gamma, beta = torch.chunk(style, 2, dim=1)
        
        # x_out = gamma * norm(x) + beta
        return self.norm(spatial_feat) * (1 + gamma) + beta
    

class AlphaPyramid(nn.Module):
    """
    Rapidly generate multi-scale Alpha masks to guide feature fusion across different hierarchies.
    """
    def __init__(self):
        super().__init__()

    def forward(self, alpha_full):
        # alpha_full: (B, 1, H, W)
        pyramid = []
        curr = alpha_full
        # H/2, H/4, H/8, H/16
        for _ in range(4): 
            curr = F.avg_pool2d(curr, kernel_size=2, stride=2)
            pyramid.append(curr)
        return pyramid # [H/2, H/4, H/8, H/16]
    


class AlphaGatedSkip(nn.Module):
    """
    Skip Connection Mechanism
    1. Adapter: Project Encoder features into the Decoder space first (resolves semantic misalignment).
    2. Alpha-Gating: Leverage the Alpha channel to explicitly suppress Encoder noise in occluded regions.
    """
    def __init__(self, dim):
        super().__init__()
        # Adapter: Transform Encoder features into the Decoder's feature space
        self.adapter = nn.Sequential(
            PanoConv2d(dim, dim, kernel_size=1),
            LayerNorm2d(dim),
            nn.GELU()
        )
        # Mixer Gate: Predict fusion weights based on the concatenated features
        self.mixer = nn.Sequential(
            PanoConv2d(dim * 2 + 1, 1, kernel_size=1, bias=True), 
            nn.Sigmoid()
        )
        # Zero-convolution warm-start: Default bias initialized to 2.0 (Sigmoid(2.0) ≈ 0.88).
        # Forces the network to heavily trust 3DGS priors in step 1 for ultra-fast convergence,
        # allowing it to adaptively fine-tune the routing weights over subsequent iterations.
        nn.init.zeros_(self.mixer[0].conv.weight)
        nn.init.constant_(self.mixer[0].conv.bias, 2.0)

    def forward(self, dec_feat, enc_feat, alpha):
        enc_adapted = self.adapter(enc_feat)
        combined = torch.cat([dec_feat, enc_adapted, alpha], dim=1)
        # The shape of 'gate' is [B, 1, H, W]
        gate = self.mixer(combined)
        return gate * enc_adapted + (1 - gate) * dec_feat
    
    
# --- 双输入时间编码器 (参考 meanflow) ---

class TimestepEmbedder(nn.Module):
    """
    Sinusoidal embedding for MeanFlow frequency handling.
    Reference: meanflow/models/unet.py
    """
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        t = t.to(torch.float32)
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DualTimeEmbedding(nn.Module):
    """
    Encodes both absolute time 't' and step size 'h = t-r'.
    Uses concatenation to preserve independence, fused by MLP.
    """
    def __init__(self, time_embed_dim=256):
        super().__init__()
        self.freq_dim = 256
        # Input dim = freq_dim * 2 (concatenated t and h embeddings)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.freq_dim * 2, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, t, h):
        # t, h: (B,)
        t_freq = TimestepEmbedder.timestep_embedding(t, self.freq_dim)
        h_freq = TimestepEmbedder.timestep_embedding(h, self.freq_dim)
        combined = torch.cat([t_freq, h_freq], dim=1)
        return self.fusion_mlp(combined)
    
    
# --- Dual-Input Temporal Encoder (Reference: MeanFlow) ---
class RenderingDecoder(nn.Module):
    """
    MeanFlow-adapted Decoder.
    Highlights:
    1. High Capacity: Base Dim 96
    2. Deep Bottleneck: 3x Transformer for robust global context.
    3. Learnable Skip Connections: Better feature alignment.
    4. Clean Semantic Injection: Only at bottleneck (Simple & Effective).
    """
    def __init__(self, in_channels=16+3+1+1+3, semantic_dim=1024):
        super().__init__()
        
        base_dim = 64
        
        # --- Time Encoding ---
        self.time_encoder = DualTimeEmbedding(time_embed_dim=base_dim*4)
        
        # Time Injection Params
        # Predict scale(gamma) 和 shift(beta)
        self.time_injector = nn.Sequential(
            nn.SiLU(),
            nn.Linear(base_dim*4, base_dim * 8 * 2) 
        )

        self.alpha_pyramid = AlphaPyramid()

        # ==========================================================
        # 1. Encoder Path (H -> H/16)
        # ==========================================================
        
        # Step 1: H -> H/2 (In -> 64)
        self.stem_h2 = nn.Sequential(
            PanoConv2d(in_channels + 4, base_dim, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(base_dim),
            nn.GELU()
        )
        
        # Step 2: H/2 -> H/4 (64 -> 128)
        self.stem_h4 = nn.Sequential(
            PanoConv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(base_dim * 2),
            nn.GELU()
        )
        
        # H/4 -> H/8 (128 -> 256)
        self.stage1_process = nn.Sequential(
            PanoBlock(base_dim * 2),
            PanoBlock(base_dim * 2)
        )
        self.down1 = PanoConv2d(base_dim * 2, base_dim * 4, kernel_size=2, stride=2, padding=0) 
        
        # H/8 -> H/16 (256 -> 512)

        self.stage2_process = nn.Sequential(
            PanoBlock(base_dim * 4), 
            PanoBlock(base_dim * 4)
        )
        self.down2 = PanoConv2d(base_dim * 4, base_dim * 8, kernel_size=2, stride=2, padding=0)
        
        # ==========================================================
        # 2. Deep Bottleneck (H/16, 512 dim)
        # ==========================================================
        bn_dim = base_dim * 8 # 512
        
        # Deep Bottleneck with 8 Blocks
        # 8 layers for global layout understand
        self.bottleneck_blocks = nn.ModuleList(
            [TransformerBottleneck(bn_dim) for _ in range(8)]
        )
        
        # Semantic Injection
        self.sem_proj_bn = PanoConv2d(semantic_dim, bn_dim, 1)
        self.sem_injector_bn = SemanticInjector(spatial_dim=bn_dim, semantic_dim=bn_dim)

        # ==========================================================
        # 3. Decoder Path (H/16 -> H/1)
        # ==========================================================
        
        # --- Stage 1: H/16 -> H/8 (512 -> 256) ---
        self.up1 = PixelShuffleUpsample(bn_dim, base_dim * 4) 
        # Learnable Adapter
        self.skip1 = AlphaGatedSkip(base_dim * 4)
        self.refine1 = nn.Sequential(
            PanoBlock(base_dim * 4), PanoBlock(base_dim * 4)
        )
        
        # --- Stage 2: H/8 -> H/4 (256 -> 128) ---
        self.up2 = PixelShuffleUpsample(base_dim * 4, base_dim * 2)
        self.skip2 = AlphaGatedSkip(base_dim * 2)
        self.refine2 = nn.Sequential(
            PanoBlock(base_dim * 2), PanoBlock(base_dim * 2)
        )
        
        # --- Stage 3: H/4 -> H/2 (128 -> 64) ---
        self.up3 = PixelShuffleUpsample(base_dim * 2, base_dim) 
        self.skip3 = AlphaGatedSkip(base_dim)
        self.refine3 = nn.Sequential(
            PanoBlock(base_dim), PanoBlock(base_dim)
        )
        
        # --- Stage 4: H/2 -> H/1 (64 -> In_Channels) ---
        self.up4 = PixelShuffleUpsample(base_dim, in_channels) 
        # Final AlphaGatedSkip for details
        self.skip4 = AlphaGatedSkip(in_channels) 

        # ==========================================================
        # 4. Prediction Heads
        # ==========================================================
        
        self.final_refine = nn.Sequential(
            PanoBlock(in_channels, kernel_size=5), 
            PanoBlock(in_channels, kernel_size=3)
        )
        self.rgb_head = PanoConv2d(in_channels, 3, kernel_size=1, bias=True)  
        self.depth_head = PanoConv2d(in_channels, 1, kernel_size=1, bias=True)  


    def forward(self, detail_input, semantic_input, z_state, times, return_features=False):
        """
        detail_input: (B, C, H, W) - 3DGS Projection Features
        semantic_input: (B, 1024, H, W) - DINO Features
        z_state: (B, 4, H, W) - Noisy RGBD State
        times: (B, 2) - [t, t-r]
        """
        
        # 1. Time Encoding
        t = times[:, 0]
        h = times[:, 1]
        time_emb = self.time_encoder(t, h) 
        
        # Time Injection Params (Applied at Bottleneck)
        style = self.time_injector(time_emb)
        gamma, beta = style.chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)

        # Alpha Pyramid
        raw_alpha = detail_input[:, 20:21, :, :]  # Check carefully !!!!!!!!!!!
        alphas = self.alpha_pyramid(raw_alpha) # [H/2, H/4, H/8]
        
        # =========================
        # Encoder Path
        # =========================
        x_in = torch.cat([detail_input, z_state], dim=1)

        # H -> H/2 (64 ch)
        x_h2 = self.stem_h2(x_in)
        
        # H/2 -> H/4 (128 ch)
        x0 = self.stem_h4(x_h2)
        
        # H/4 -> H/8 (256 ch)
        x1_feat = self.stage1_process(x0)  
        x1 = self.down1(x1_feat)           
        
        # H/8 -> H/16 (512 ch)
        x2_feat = self.stage2_process(x1)  
        x2 = self.down2(x2_feat)           
        
        # =========================
        # Deep Bottleneck
        # =========================
        
        # Semantic Injection
        sem_feat_bn = self.sem_proj_bn(semantic_input)
        x2 = self.sem_injector_bn(x2, sem_feat_bn)
        
        # Time Injection (Scale & Shift)
        x2 = x2 * (1 + gamma) + beta
        
        # Global Layout
        for block in self.bottleneck_blocks:
            x2 = block(x2)
        
        # =========================
        # Decoder Path
        # =========================
        
        # --- Stage 1 (H/16 -> H/8) ---
        d1 = self.up1(x2) # 512 -> 256
        # Learnable Adapter fuse the features from Encoder
        d1 = self.skip1(dec_feat=d1, enc_feat=x2_feat, alpha=alphas[2])
        d1 = self.refine1(d1)
        
        # --- Stage 2 (H/8 -> H/4) ---
        d2 = self.up2(d1) # 256 -> 128
        d2 = self.skip2(dec_feat=d2, enc_feat=x1_feat, alpha=alphas[1])
        d2 = self.refine2(d2)
        
        # --- Stage 3 (H/4 -> H/2) ---
        d3 = self.up3(d2) # 128 -> 64
        d3 = self.skip3(dec_feat=d3, enc_feat=x_h2, alpha=alphas[0])
        d3 = self.refine3(d3)
        
        # --- Stage 4 (H/2 -> H/1) ---
        d4 = self.up4(d3) # 64 -> In_Channels
        # Final AlphaGatedSkip
        d4 = self.skip4(dec_feat=d4, enc_feat=detail_input, alpha=raw_alpha)
        
        # Prediction Heads
        final_feat = self.final_refine(d4)
        rgb = torch.sigmoid(self.rgb_head(final_feat))
        depth = torch.relu(self.depth_head(final_feat))

        output = torch.cat([rgb,depth], dim=1)

        if return_features:
            features_dict = {
                "enc_h16": x2,         # Bottleneck
                "dec_h8":  d1,         # decoder stage 1
                "dec_h4":  d2,         # decoder stage 2
                "dec_h2":  d3,         # decoder stage 3
                "dec_h1":  d4,         # decoder stage 4
                "final_feat": final_feat # decoder final features
            }
            return output, features_dict
            
        return output



class GaussianModel(nn.Module):
  """
    Image2Sim Model.
    Supports:
      - 3DGS Projection (Condition Generation)
      - MeanFlow Training (forward_meanflow_loss)
      - 1-Step Sampling (sample_1step)
    """

  def __init__(self, config = type('Config', (), {'image_height': 512, 'batch_size': 1, 'max_depth': 10.0})):
    super().__init__()
    self.config = config
    self.max_depth = config.max_depth
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Large Encoder (DINO + Upsampler)
    self.encoder_backbone = DINOv3Wrapper()
    self.feature_upsampler = FeatureUpsampler(in_dim=1024, out_feat_dim=16)
    
    # 2. Initialize Small Decoder Parts
    self.semantic_aligner = SemanticAligner(in_feat_dim=24, embed_dim=1024)
    # in_channels=24 because: 16 (features) + 3 (RGB) + 1 (depth) + 1 (alpha) + 3 (ray direction, xyz)  from projection
    self.rendering_decoder = RenderingDecoder(in_channels=16+3+1+1+3, semantic_dim=1024)
      
    self.to(self.device)
    self.encoder_backbone.eval() 

    self.batch_size = config.batch_size
    self.height = config.image_height
    self.width = config.image_height * 2
    self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    self.rasterizer = PanoAnisotropicSplatting(self.height, self.width)

    self.reset_memory()

    if LPIPS is not None:
        self.lpips = LPIPS(net='vgg', model_path="pretrained_models/vgg.pth").eval()
        for param in self.lpips.parameters():
            param.requires_grad = False
    else:
        self.lpips = None

  def torch_compile(self):
    self.encoder_backbone = torch.compile(self.encoder_backbone)
    self.feature_upsampler = torch.compile(self.feature_upsampler)
    self.semantic_aligner = torch.compile(self.semantic_aligner)
    self.rendering_decoder = torch.compile(self.rendering_decoder)


  def _transform_position(self, xyz: Tensor) -> Tensor:
    transformed = torch.stack(
        [xyz[:, 0], xyz[:, 1], xyz[:, 2],
         torch.zeros(self.batch_size, device=xyz.device, dtype=xyz.dtype)], dim=1)
    return transformed

  def reset_memory(self):
    self._memory = MemoryState(
        coords=torch.zeros((self.batch_size, 3, 0), device=self.device),
        feats=torch.zeros((self.batch_size, 16, 0), dtype=torch.float32, device=self.device),
        rgb=torch.zeros((self.batch_size, 3, 0), dtype=torch.float32, device=self.device),
        gs_attrs=torch.zeros((self.batch_size, 8, 0), dtype=torch.float32, device=self.device)
    )

  def _append_points_raw(self, new_xyz: Tensor, new_feats: Tensor, new_rgb: Tensor, new_gs_attrs: Tensor):
      new_memory_state = MemoryState(
          coords=torch.cat([self._memory.coords, new_xyz], dim=2),
          feats=torch.cat([self._memory.feats, new_feats], dim=2),
          rgb=torch.cat([self._memory.rgb, new_rgb], dim=2),
          gs_attrs=torch.cat([self._memory.gs_attrs, new_gs_attrs], dim=2)
      )
      self._memory = new_memory_state 
  

  def get_ray_map(self, B, H, W, intrinsics=None):
    """
    Generate a ray direction map of shape (B, 3, H, W) based on the camera model.
    - intrinsics is None   -> Treated as Panorama (Equirectangular projection)
    - intrinsics is Tensor -> Treated as Pinhole camera model
    """
    device = self.device
    # Generate pixel grid coordinates
    grid_v, grid_u = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    
    if intrinsics is None:
        # --- Panorama Mode ---
        # Pixel -> Longitude/Latitude -> Unit Vector
        phi = (grid_u / W) * 2 * np.pi - np.pi
        theta = -((grid_v / H) * np.pi - (np.pi / 2))
        
        x = torch.cos(theta) * torch.sin(phi)
        y = torch.cos(theta) * torch.cos(phi)
        z = torch.sin(theta)
        
        ray_map = torch.stack([x, y, z], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return ray_map

    else:
        # --- Pinhole Mode ---
        # Pixel -> Camera Coordinates -> Normalization
        if intrinsics.dim() == 2: intrinsics = intrinsics.unsqueeze(0).expand(B, -1, -1)
        
        fx = intrinsics[:, 0, 0].view(B, 1, 1)
        fy = intrinsics[:, 1, 1].view(B, 1, 1)
        cx = intrinsics[:, 0, 2].view(B, 1, 1)
        cy = intrinsics[:, 1, 2].view(B, 1, 1)
        
        u = grid_u.unsqueeze(0).expand(B, -1, -1)
        v = grid_v.unsqueeze(0).expand(B, -1, -1)
        
        # Back-projection under Z-forward coordinate system
        x = (u - cx) / fx
        y = (v - cy) / fy
        z = torch.ones_like(x)
        
        raw_rays = torch.stack([x, y, z], dim=1) # (B, 3, H, W)
        ray_map = F.normalize(raw_rays, p=2, dim=1, eps=1e-4) # 归一化为方向向量
        
        return ray_map
        

  def extract_features_from_image(self, rgb_tensor: Tensor, depth_tensor: Tensor = None, intrinsics: Tensor = None):
    """
    rgb_tensor: (B, H, W, 3) range [0, 255]
    depth_tensor: (B, H, W) metric depth
    intrinsics: (B, 3, 3) metric depth
    """
    B, H, W, _ = rgb_tensor.shape

    with torch.no_grad():
        # 1. DINO input
        rgb_input = rgb_tensor.permute(0, 3, 1, 2) / 255.0 # (B, 3, H, W), range [0, 1]
        rgb_norm = self.normalize(rgb_input) # DINO 需要标准化
        tokens = self.encoder_backbone(rgb_norm)

        # 2. Detail input
        # RGB, Depth normalize [0, 1]
        depth_input = torch.clamp(depth_tensor.unsqueeze(1), 0, self.max_depth) / self.max_depth

    # Ray direction map
    ray_map = self.get_ray_map(B, H, W, intrinsics).to(tokens.dtype)

    # 3. Call Upsampler (Gradients are generated here)
# Note: rgb_input and depth_input need to be involved in gradient calculation 
# (if they are part of the learnable parameters; however, they are usually GT data. 
# The weights of the Upsampler are learnable).
    feats, gs_attrs = self.feature_upsampler(tokens, rgb_input, depth_input, ray_map=ray_map)
    return feats, gs_attrs, tokens
  


  def import_scene_gaussian(self, xyz: Tensor, rgb: Tensor, feats: Tensor, gs_attrs: Tensor):
      """
      Imports pre-processed tensors into memory.
      xyz: (B, 3, N)
      rgb: (B, 3, N)
      feats: (B, 16, N)
      gs_attrs: (B, 8, N)
      """
      self.reset_memory()
      # Ensure correct dimensions
      if xyz.dim() == 2: xyz = xyz.unsqueeze(0).repeat(self.batch_size, 1, 1)
      if rgb.dim() == 2: rgb = rgb.unsqueeze(0).repeat(self.batch_size, 1, 1)
      if feats.dim() == 2: feats = feats.unsqueeze(0).repeat(self.batch_size, 1, 1)
      if gs_attrs.dim() == 2: gs_attrs = gs_attrs.unsqueeze(0).repeat(self.batch_size, 1, 1)
      
      # Transpose if needed (expecting channels in dim 1)
      # Usually passed as (N, C), needs (B, C, N)
      if xyz.shape[2] != 3 and xyz.shape[1] == 3: pass 
      elif xyz.shape[2] == 3: xyz = xyz.transpose(1, 2)
      
      if rgb.shape[2] != 3 and rgb.shape[1] == 3: pass
      elif rgb.shape[2] == 3: rgb = rgb.transpose(1, 2)
      
      if feats.shape[2] != 16 and feats.shape[1] == 16: pass
      elif feats.shape[2] == 16: feats = feats.transpose(1, 2)
      
      if gs_attrs.shape[2] != 8 and gs_attrs.shape[1] == 8: pass
      elif gs_attrs.shape[2] == 8: gs_attrs = gs_attrs.transpose(1, 2)

      self._append_points_raw(xyz, feats, rgb, gs_attrs)



  def gaussian_splatting(self, position, heading):
    position = position.to(self.device)
    heading = heading.to(self.device)
    
    # Original data
    points = self._memory.coords      # (B, 3, N)
    feats = self._memory.feats        # (B, 16, N)
    
    # Extract GS Attributes
    gs_scales = self._memory.gs_attrs[:, 0:3, :]  # (B, 3, N)
    gs_quats  = self._memory.gs_attrs[:, 3:7, :]  # (B, 4, N)
    gs_opac   = self._memory.gs_attrs[:, 7:8, :]  # (B, 1, N)

    # =========================================================================
    # [Safety Guard Start] Avoid NaN/Inf and CUDA Illegal Memory Access
    # =========================================================================

    with torch.no_grad():
        cam_pos = position.view(1, 3) if position.dim() == 1 else position
        
        # 1. Check values
        mask_pos = torch.isfinite(points).all(dim=1)
        mask_scale = torch.isfinite(gs_scales).all(dim=1) & (gs_scales < 10.0).all(dim=1)
        mask_quat = torch.isfinite(gs_quats).all(dim=1) & (gs_quats.norm(dim=1) > 1e-5)
        mask_opac = torch.isfinite(gs_opac).all(dim=1)
        mask_feats = torch.isfinite(feats).all(dim=1)
        
        # 2. Distance Singularity Protection (Near Clipping)
        # Cull points within 5cm of the camera center to prevent division-by-zero (1/0) 
        # crashes in the panoramic Jacobian computation.
        dist_to_cam = torch.norm(points - cam_pos.unsqueeze(-1), dim=1)
        mask_dist = dist_to_cam > 0.05 
        
        valid_indices = mask_pos & mask_scale & mask_quat & mask_opac & mask_feats & mask_dist

    if not valid_indices.all():
        points = points.clone()
        gs_scales = gs_scales.clone()
        gs_opac = gs_opac.clone()
        gs_quats = gs_quats.clone()
        feats = feats.clone()
        
        inv_mask = (~valid_indices).unsqueeze(1) 
        
        # 3. Large values for point positions
        points.masked_fill_(inv_mask.expand_as(points), 1000.)
        gs_scales.masked_fill_(inv_mask.expand_as(gs_scales), 0.001) 
        gs_opac.masked_fill_(inv_mask.expand_as(gs_opac), 0.0)       
        feats.masked_fill_(inv_mask.expand_as(feats), 0.0)
        
        default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=points.device, dtype=points.dtype).view(1, 4, 1)
        gs_quats = torch.where(inv_mask.expand_as(gs_quats), default_quat, gs_quats)
    # =========================================================================
    # [Safety Guard End]
    # =========================================================================
    
    # 1. Call Rasterizer
    proj_rgb, proj_features, proj_depth, proj_alpha = self.rasterizer(
        position=position,
        heading=heading,
        points=points,       # Cleaned points
        rgb_colors = self._memory.rgb,
        features=feats, 
        opacity=gs_opac,     # Cleaned opacity
        scales=gs_scales,    # Cleaned scales
        rotations=gs_quats
    )

    B, _, H, W = proj_features.shape
    ray_map = self.get_ray_map(B, H, W, intrinsics=None).to(proj_features.device)

    return proj_features, proj_rgb, proj_depth/self.max_depth, proj_alpha, ray_map


  def prepare_training_conditions(self, proj_output):
    B, C, H, W = proj_output.shape
    device = proj_output.device
    
    # ==============================================================================
    # 1. Channel Separation (Isolating Content Features from Absolute Coordinates)
    # Exclude the last 3 channels (RayMap) since ray directions act as absolute 
    # physical constraints and should not be masked.
    # ==============================================================================
    feat_content = proj_output[:, :-3, :, :] 
    feat_rays = proj_output[:, -3:, :, :]

    # ==============================================================================
    # 2. Spatial Block Masking
    # ==============================================================================
    spatial_mask = torch.ones((B, 1, H, W), device=device)
    
    for b in range(B):
        if torch.rand(1) < 0.3:  # 30% large missing region
            w_len = torch.randint(W // 8, W // 3, (1,)).item()
            h_len = torch.randint(H // 8, H // 3, (1,)).item()
            x_start = torch.randint(0, W - w_len, (1,)).item()
            y_start = torch.randint(0, H - h_len, (1,)).item()
            
            # mask to 0
            spatial_mask[b, :, y_start:y_start+h_len, x_start:x_start+w_len] = 0.0

    feat_content = feat_content * spatial_mask

    
    # ==============================================================================
    # 3. Concatenate
    # ==============================================================================
    proj_output_final = torch.cat([feat_content, feat_rays], dim=1)
    
    return proj_output_final, spatial_mask


  def compute_saturation_loss(self, pred_rgb, gt_rgb, mask=None, eps=1e-3):
    """
    Leverages GT channel indices to guide the optimization path of Pred RGB.
    Fixes gather runtime errors, gradient explosion risks, and the "vanishing gradient 
    under incorrect hue" issue.
    """
    # 1. Extract guidance from GT (Detached to prevent backward gradients to GT)
    # keepdim=True ensures shape (B, 1, H, W) for seamless torch.gather operations
    gt_max_val, gt_max_idx = gt_rgb.max(dim=1, keepdim=True)
    gt_min_val, gt_min_idx = gt_rgb.min(dim=1, keepdim=True)
    
    # Force Long tensor type to prevent gather indexing errors
    gt_max_idx = gt_max_idx.detach().long()
    gt_min_idx = gt_min_idx.detach().long()
    
    # 2. Extract corresponding channel values from Pred based on GT indices
    # Core mechanism: Forces Pred to follow GT's alignment, regardless of Pred's own internal channel magnitudes
    pred_val_at_gt_max = torch.gather(pred_rgb, 1, gt_max_idx)
    pred_val_at_gt_min = torch.gather(pred_rgb, 1, gt_min_idx)
    
    # 3. Compute "Guided Saturation"
    # Numerator: Forces Pred to widen the gap between the GT-designated Max/Min channels.
    # Yields a positive value if hue is correct; yields a negative value if hue is incorrect (penalizing with massive loss)
    numerator = pred_val_at_gt_max - pred_val_at_gt_min
    
    # Critical numerical stability fix: Normalize using Pred's actual maximum value (detached).
    # Reason: We exclusively want to optimize saturation by altering the numerator (widening channel contrast).
    # If gradients flow through the denominator, a model with incorrect hue might trick the loss by 
    # scaling up global brightness (luminance) instead of correcting the actual color/chrominance.
    # Detaching also guarantees that if Pred_Red=0, Pred_Green still receives backpropagated gradients.
    pred_actual_max = pred_rgb.max(dim=1, keepdim=True)[0].detach()
    
    # Append eps to prevent division-by-zero (specifically for pure black pixels)
    pred_guided_sat = numerator / (pred_actual_max + eps)
    
    # Calculate Ground-Truth saturation
    gt_sat = (gt_max_val - gt_min_val) / (gt_max_val + eps)
    
    # 4. Compute Loss
    diff = torch.abs(pred_guided_sat - gt_sat)
    
    # 5. Low-Luminance Suppression (Luminance Weighting)
    # Weighted by GT luminance to ignore saturation computation in dark regions (which are highly noisy)
    weight = gt_max_val.detach()
    if mask is not None:
        weight = weight * mask

    # Normalize Loss
    loss = (diff * weight).float().sum() / (weight.float().sum() + eps)
    return loss


  def forward_meanflow_loss(self, position, heading, gt_rgb, gt_depth, gt_patch_tokens, rgb_mask=None, depth_mask=None, training_stage=0, teacher_decoder=None):

    # --- 1. Prepare Target Data ---
    B = gt_rgb.shape[0]
    gt_depth = gt_depth.unsqueeze(1) if gt_depth.dim() == 3 else gt_depth
    gt_rgb = gt_rgb.permute(0, 3, 1, 2) if gt_rgb.shape[-1] == 3 else gt_rgb

    if rgb_mask is None: rgb_mask = torch.ones((B, 1, self.height, self.width), device=self.device)
    if depth_mask is None: depth_mask = torch.ones((B, 1, self.height, self.width), device=self.device)
    
    rgb_mask = rgb_mask.unsqueeze(1) if rgb_mask.dim() == 3 else rgb_mask
    depth_mask = depth_mask.unsqueeze(1) if depth_mask.dim() == 3 else depth_mask
    depth_mask = depth_mask * (gt_depth > 1e-3).float()

    gt_rgb = gt_rgb.to(torch.float32) / 255. * rgb_mask
    gt_depth = gt_depth / self.max_depth * depth_mask

    x_target = torch.cat([gt_rgb, gt_depth], dim=1) 

    # --- 2. Get Conditions (3DGS) ---
    if training_stage == 0:
        proj_features, proj_rgb, proj_depth, proj_alpha, ray_map = checkpoint(self.gaussian_splatting, position, heading, use_reentrant=False)
    else:
        with torch.no_grad():
            proj_features, proj_rgb, proj_depth, proj_alpha, ray_map = self.gaussian_splatting(position, heading)

    proj_mask = (gt_depth > 1e-3).float()
    diff_rgb = torch.abs(proj_rgb - gt_rgb) * rgb_mask * proj_mask
    loss_proj_rgb = diff_rgb.float().sum() / torch.clamp((rgb_mask * proj_mask).float().sum(), min=1.0)
    
    diff_depth = torch.abs(proj_depth - gt_depth) * depth_mask * proj_mask
    loss_proj_depth = diff_depth.float().sum() / torch.clamp((depth_mask * proj_mask).float().sum(), min=1.0)
    proj_loss = 0.1 * loss_proj_rgb + 0.1 * loss_proj_depth

    gs_scales = self._memory.gs_attrs[:, 0:3, :]
    excess_scale = torch.relu(gs_scales - 0.01)
    scale_reg_loss = excess_scale.float().mean()
    proj_loss = proj_loss + scale_reg_loss
    
    # =====================================================================
    # Masked Semantic Autoencoder
    # =====================================================================
    proj_output = torch.cat([proj_features, proj_rgb, proj_depth, proj_alpha, ray_map], dim=1)
    
    # 1. Generate conditions featuring natural blind spots alongside 30% artificial large-block occlusions/masks
    proj_output_student, spatial_mask = self.prepare_training_conditions(proj_output)
    
    # 2. Force the Transformer to leverage global attention to perform semantic inpainting across the blocked holes
    aligned_features_student = checkpoint(self.semantic_aligner, proj_output_student, use_reentrant=False)

    # 3. Compute semantic supervision loss
    cos_sim = F.cosine_similarity(aligned_features_student.float(), gt_patch_tokens.float(), dim=1, eps=1e-4)
    semantic_mask = F.interpolate(rgb_mask.float(), size=cos_sim.shape[-2:], mode='nearest').squeeze(1)
    valid_semantic_pixels = torch.clamp(semantic_mask.sum(), min=1.0)
    
    # NOTE: Here we utilize the original GT mask (semantic_mask) for supervision!
# This means the network is required to output flawless GT tokens even in regions hollowed out 
# by the spatial_mask, fully unlocking its potential for semantic inpainting!
    align_loss = 0.1 * ((1.0 - cos_sim) * semantic_mask).sum() / valid_semantic_pixels

    # Detach the 3DGS base structure from the computational graph to prevent 
    # subsequent Flow Matching from disrupting 3D geometric priors.
    proj_output_student = proj_output_student.detach()
    # DO NOT detach 'aligned_features_student', ensure gradients from the subsequent MeanFlow can still backpropagate to the Transformer.

    # =====================================================================
    # # [Perfect God's-Eye View: Teacher Extraction]
    # =====================================================================
    if teacher_decoder is not None:
        with torch.no_grad():
            proj_rgb_det = proj_rgb.detach()
            proj_depth_det = proj_depth.detach()
            proj_features_det = proj_features.detach()
            
            # A. Image-Level Stitching
            teacher_rgb = gt_rgb * rgb_mask + proj_rgb_det * (1.0 - rgb_mask)
            teacher_depth = gt_depth * depth_mask + proj_depth_det * (1.0 - depth_mask)

            teacher_rgb_ext = (teacher_rgb.permute(0, 2, 3, 1) * 255.0).clamp(0, 255)
            teacher_depth_ext = teacher_depth.squeeze(1) * self.max_depth

            # B. Extract Seamless God's-Eye View Features
            teacher_gt_feats, _, teacher_patch_tokens = self.extract_features_from_image(teacher_rgb_ext, teacher_depth_ext)
            teacher_gt_feats = teacher_gt_feats * rgb_mask + proj_features_det * (1.0 - rgb_mask)

            teacher_proj_alpha = proj_alpha
            teacher_proj_output = torch.cat([
                teacher_gt_feats,        
                teacher_rgb,             
                teacher_depth,           
                teacher_proj_alpha,      
                ray_map
            ], dim=1).detach()
            
            # The Teacher enjoys perfect semantics without needing any inpainting/reasoning
            teacher_aligned_features = teacher_patch_tokens.detach()

    # --- 3. Time Sampling ---
    t = torch.rand(B, device=self.device)
    is_flow_matching = torch.rand(B, device=self.device) < 0.5
    r_meanflow = torch.rand_like(t) * t 
    r = torch.where(is_flow_matching, t, r_meanflow)
    
    t_b = t.view(B, 1, 1, 1)
    r_b = r.view(B, 1, 1, 1)

    # --- 4. Construct Flow State ---
    x_prior = torch.cat([proj_rgb, proj_depth], dim=1).detach() * spatial_mask
    alpha = proj_alpha.detach() * spatial_mask 
    
    e = torch.randn_like(x_prior)
    noise_scale_rgb = alpha * 0.1 + (1 - alpha) * 1.0
    noise_scale_depth = alpha * 0.01 + (1 - alpha) * 1.0
    noise_scales = torch.cat([noise_scale_rgb.repeat(1, 3, 1, 1), noise_scale_depth], dim=1)
    
    x_source = (x_prior * alpha) + (noise_scales * e)
    z = (1 - t_b) * x_target + t_b * x_source
    v_target = x_source - x_target

    # =====================================================================
    # 5. Functional JVP (has_aux=True)
    # =====================================================================
    params = dict(self.rendering_decoder.named_parameters())
    buffers = dict(self.rendering_decoder.named_buffers())
    
    def u_from_x_func(z_in, t_in, r_in):
        if z_in.dim() == 3: z_in = z_in.unsqueeze(0)
        if t_in.dim() == 0: t_in = t_in.view(1)
        if r_in.dim() == 0: r_in = r_in.view(1)
        
        times_in = torch.stack([t_in, t_in - r_in], dim=1)
        x_pred, student_feats_dict = functional_call(
            self.rendering_decoder, (params, buffers), 
            (proj_output_student, aligned_features_student, z_in, times_in), 
            kwargs={'return_features': True}
        )
        
        t_safe = t_in.view(-1, 1, 1, 1).clamp(min=1e-3)
        u_val = (z_in - x_pred) / t_safe
        return u_val, student_feats_dict

    with torch.no_grad():
        v_pred, _ = u_from_x_func(z, t, t)
        v_pred = v_pred.detach() 

    primals = (z, t, r)
    tangents = (v_pred, torch.ones_like(t), torch.zeros_like(r))
    
    with torch.amp.autocast(device_type='cuda', enabled=False, dtype=torch.float32):
        u_derived, dudt, student_feats = jvp(u_from_x_func, primals, tangents, has_aux=True)

    # --- 6. Predict x_0 ---
    t_safe_b = t_b.clamp(min=1e-3)
    x_recovered = z - u_derived * t_safe_b

    # --- 7. Compute Flow Loss ---
    V_theta = u_derived + (t_b - r_b) * dudt.detach()
    
    if training_stage == 0:
        focus_weight = 1.
        lpips_scale = 1.

    elif training_stage == 1:
        focus_weight = 1.0 + 4.0 * (1.0 - alpha)
        lpips_scale = 2.

    elif training_stage == 2:
        focus_weight = 1.0 + 4.0 * (1.0 - alpha)
        lpips_scale = 4.
    else:
        print("Error: No suitable training_stage!!!")
        exit()
  
    
    def compute_adaptive_loss(pred, target, mask_rgb, mask_depth, weight_map=1):
        sq_diff = (pred - target) ** 2
        masked_sq_rgb = sq_diff[:, :3] * mask_rgb * weight_map
        masked_sq_depth = sq_diff[:, 3:] * mask_depth * weight_map
        
        valid_rgb = (mask_rgb * weight_map).sum(dim=(1, 2, 3)) + 1e-3
        valid_depth = (mask_depth * weight_map).sum(dim=(1, 2, 3)) + 1e-3
        
        loss_val = masked_sq_rgb.sum(dim=(1, 2, 3)) / (valid_rgb * 3) + \
                   masked_sq_depth.sum(dim=(1, 2, 3)) / valid_depth
                   
        adp_wt = loss_val.detach() + 1e-3 
        return (loss_val / adp_wt).mean(), loss_val.mean()

    loss_flow, _ = compute_adaptive_loss(V_theta, v_target, rgb_mask, depth_mask, focus_weight)
    
    # --- 8. LPIPS & Saturation Loss ---
    perc_loss = torch.tensor(0.0, device=self.device)
    if self.lpips is not None:
        pred_rgb_masked = x_recovered[:, :3] * rgb_mask
        gt_rgb_masked = x_target[:, :3] * rgb_mask

        pred_rgb_lpips = pred_rgb_masked * 2.0 - 1.0
        gt_rgb_lpips = gt_rgb_masked * 2.0 - 1.0
        perc_loss = lpips_scale * self.lpips(pred_rgb_lpips, gt_rgb_lpips).mean()

    loss_sat = self.compute_saturation_loss(x_recovered[:, :3], x_target[:, :3], rgb_mask)

    # --- 9. Distill ---
    distill_loss = torch.tensor(0.0, device=self.device)
    if teacher_decoder is not None and training_stage > 0:
        zero_times = torch.zeros_like(t).unsqueeze(1).repeat(1, 2) + 1e-3
        with torch.no_grad():
            _, teacher_feats = teacher_decoder(
                teacher_proj_output, teacher_aligned_features, torch.cat([teacher_rgb, teacher_depth], dim=1).detach(), zero_times, return_features=True
            )
            
        if training_stage == 1:
            layer_weights = {"enc_h16": 1., "dec_h8": 0.8, "dec_h4": 0.6, "dec_h2": 0.4, "dec_h1": 0.2, "final_feat": 0.1}
        if training_stage == 2:
            layer_weights = {"dec_h8": 0.8, "dec_h4": 0.6, "dec_h2": 0.4, "dec_h1": 0.2, "final_feat": 0.1}

        total_w = sum(layer_weights.values())
        
        for k, w in layer_weights.items():
            t_f, s_f = teacher_feats[k].detach(), student_feats[k]
            cos_loss = 1.0 - F.cosine_similarity(s_f.float(), t_f.float(), dim=1).mean()
            l2_loss = F.mse_loss(s_f.float(), t_f.float())
            distill_loss += (w / total_w) * (cos_loss + 0.1 * l2_loss)

    distill_loss = 2. * distill_loss

    meanflow_loss = loss_flow + 0.2 * loss_sat

    with torch.no_grad():
        rgb_loss = torch.abs(x_recovered[:, :3] - x_target[:, :3]).mean()
        depth_loss = torch.abs(x_recovered[:, 3:] - x_target[:, 3:]).mean()

    if torch.isnan(proj_loss) or torch.isinf(proj_loss):
        print("Warning: NaN encountered in proj_loss, using DDP-safe dummy loss.")
        dummy_loss = 0.0
        for p in self.parameters():
            if p.requires_grad:
                dummy_loss = dummy_loss + p.sum() * 0.0
        proj_loss = dummy_loss

    return proj_loss, align_loss, meanflow_loss, perc_loss, distill_loss, rgb_loss, depth_loss


  @torch.no_grad()
  def render(self, position: Tensor, heading: Optional[Tensor] = None, z_1: Optional[Tensor] = None, num_steps: int = 1) -> OutputData:
        """
        [Image2Sim Inference] Combines explicit 3DGS projection with Alpha-Gated Flow Matching 
        to perform multi-step Euler integration inference.
        """
        
        # 1. Acquire conditional inputs (3DGS Projection)
        proj_features, proj_rgb, proj_depth, proj_alpha, ray_map = self.gaussian_splatting(position, heading)
        proj_output = torch.cat([proj_features, proj_rgb, proj_depth, proj_alpha, ray_map], dim=1)
        aligned_features = self.semantic_aligner(proj_output)
    
        B, _, H, W = proj_output.shape
        
        # Extract 3DGS priors and Alpha masks
        x_prior = torch.cat([proj_rgb, proj_depth], dim=1)
        alpha = proj_alpha
    
        # 2. Sample pure Gaussian noise (Preserve location-based deterministic random seed logic)
        if z_1 is None:
            gen = torch.Generator(device=self.device)
            if position is not None:
                pos_flat = position.view(-1)
                primes = torch.tensor([73856093, 19349663, 83492791], device=self.device, dtype=torch.float32)
                p_len = pos_flat.shape[0]
                primes = primes.repeat((p_len // 3) + 1)[:p_len]
                pos_val = (pos_flat * primes).sum().item()
                head_val = 0
                if heading is not None:
                    head_flat = heading.view(-1)
                    primes_h = torch.tensor([99991, 99989, 99971], device=self.device, dtype=torch.float32)
                    h_len = head_flat.shape[0]
                    primes_h = primes_h.repeat((h_len // 3) + 1)[:h_len]
                    head_val = (head_flat * primes_h).sum().item()
                seed_val = int(abs(pos_val + head_val)) % (2**32)
                gen.manual_seed(seed_val)
            z_1_raw = torch.randn((B, 4, H, W), device=self.device, generator=gen, dtype=proj_output.dtype)
        else:
            z_1_raw = z_1
            
        noise_scale_rgb = alpha * 0.03 + (1.0 - alpha)
        noise_scale_depth = alpha * 0.01 + (1.0 - alpha)
        
        noise_scales = torch.cat([noise_scale_rgb.repeat(1, 3, 1, 1), noise_scale_depth], dim=1)
        z_t = (x_prior * alpha) + (noise_scales * z_1_raw)
        
        # ----------------------------------------------------------------------
        # 4. Coupled Alpha & Time Multi-Step Integration Scheduler
        # ----------------------------------------------------------------------
        # Compute velocity field and perform integration
        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)

        for i in range(num_steps):
            t_current = t_steps[i]
            t_next = t_steps[i+1]
            
            times = torch.tensor([[t_current.item(), t_current.item()]], device=self.device).repeat(B, 1)
            x_pred = self.rendering_decoder(proj_output, aligned_features, z_t, times)
            
            # Compute velocity field and perform integration
            t_safe = torch.clamp(t_current, min=1e-3)
            v_t = (z_t - x_pred) / t_safe
            dt = t_next - t_current
            z_t = z_t + v_t * dt

        x_pred = z_t
        # ----------------------------------------------------------------------
        # 5. Output Post-Processing
        pred_rgb = x_pred[:, :3]
        pred_depth = x_pred[:, 3:4]

        pred_rgb = torch.clamp(pred_rgb, 0.0, 1.0)
        pred_depth = torch.clamp(pred_depth, 0.0, 1.0) * self.max_depth

        return OutputData(
            proj_features=proj_features,
            aligned_features=aligned_features,
            pred_rgb=pred_rgb, 
            proj_rgb=proj_rgb,
            pred_depth=pred_depth,
            proj_depth=proj_depth,
            proj_alpha=proj_alpha
        )

  def forward(self, position=None, heading=None, gt_rgb=None, gt_depth=None, gt_patch_tokens=None, rgb_mask=None, depth_mask=None, z_1=None, num_steps: int = 1, training_stage=0, teacher_decoder=None):
        """
        Automatically triggered by DDP upon invoking model().
        Toggles between training (loss computation) and inference (image generation) modes via parameters.
        """
        if gt_rgb is None:
            return self.render(position=position, heading=heading, z_1=z_1, num_steps=num_steps)
        else:
            return self.forward_meanflow_loss(position=position, heading=heading, gt_rgb=gt_rgb, gt_depth=gt_depth, gt_patch_tokens=gt_patch_tokens, rgb_mask=rgb_mask, depth_mask=depth_mask, training_stage=training_stage, teacher_decoder=teacher_decoder)
        

  @torch.no_grad()
  def inpaint_depth_2d(self, rgb_tensor: Tensor, depth_tensor: Tensor, valid_mask: Tensor = None, intrinsics: Tensor = None) -> Tensor:
      """
        [Zero-3D Rendering] A pure 2D depth completion approach.
        Leverages 2D features and a Rendering Decoder to complete depth maps, perfectly 
        circumventing artifacts caused by 3D ray penetration through specular or transparent surfaces (e.g., mirrors/glass).
        
        Args:
            rgb_tensor: (B, H, W, 3) or (H, W, 3), range [0, 255]
            depth_tensor: (B, H, W) or (H, W), metric depth
            valid_mask: (B, H, W) or (H, W), boolean mask. If None, automatically generated based on max_depth.
            intrinsics: (B, 3, 3), optional, for pinhole cameras
            
        Returns:
            final_depth: (B, H, W) or (H, W), completed hybrid depth
        """
        # 1. Dimension Alignment (Unifying by prepending a batch dimension)
      if rgb_tensor.dim() == 3:
          rgb_tensor = rgb_tensor.unsqueeze(0)
      if depth_tensor.dim() == 2:
          depth_tensor = depth_tensor.unsqueeze(0)
          
      B, H, W, _ = rgb_tensor.shape
      
      if valid_mask is None:
          valid_mask = (depth_tensor > 0.01) & (depth_tensor < self.max_depth)
      elif valid_mask.dim() == 2:
          valid_mask = valid_mask.unsqueeze(0)

      # 2. Directly extract 2D spatial features
      feats, _, _ = self.extract_features_from_image(rgb_tensor, depth_tensor, intrinsics=intrinsics)
      
      # 3. Construct a Pseudo-Projection condition in the 2D spatial dimensions
      proj_rgb = rgb_tensor.permute(0, 3, 1, 2) / 255.0
      proj_depth = depth_tensor.unsqueeze(1) / self.max_depth
      proj_alpha = valid_mask.unsqueeze(1).float()
      ray_map = self.get_ray_map(B, H, W, intrinsics).to(self.device).to(feats.dtype)
      
      proj_output = torch.cat([feats, proj_rgb, proj_depth, proj_alpha, ray_map], dim=1)
      
      # 4. Align semantic features
      aligned_features = self.semantic_aligner(proj_output)
      
      # 5. Construct an Alpha-gated adaptive initial state (Z_start)
      x_prior = torch.cat([proj_rgb, proj_depth], dim=1)
      
      gen = torch.Generator(device=self.device)
      gen.manual_seed(42) # Fix the seed to ensure deterministic inference and completion
      z_1_raw = torch.randn((B, 4, H, W), device=self.device, generator=gen, dtype=proj_output.dtype)
      
      noise_scale_rgb = proj_alpha * 0.03 + (1 - proj_alpha) * 1.0
      noise_scale_depth = proj_alpha * 0.01 + (1 - proj_alpha) * 1.0
      noise_scales = torch.cat([noise_scale_rgb.repeat(1, 3, 1, 1), noise_scale_depth], dim=1)
      
      z_start = (x_prior * proj_alpha) + (noise_scales * z_1_raw)
      times = torch.tensor([[1.0, 1.0]], device=self.device).repeat(B, 1)
      
      # 6. Direct decoding for prediction
      x_pred = self.rendering_decoder(proj_output, aligned_features, z_start, times)
      pred_depth = (torch.clamp(x_pred[:, 3:4], 0.0, 1.0) * self.max_depth).squeeze(1) # (B, H, W)
      
      # 7. Precisely backfill holes in the GT depth map
      final_depth = torch.where(valid_mask, depth_tensor, pred_depth)
      
      return final_depth.squeeze(0) if final_depth.shape[0] == 1 else final_depth


class Action(IntEnum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    LOOK_UP = 4
    LOOK_DOWN = 5


class NeuralSimulator(GaussianModel):
    """
    Simulates the world state, agent physics, AND full path planning.
    Includes robust collision detection.
    """
    @staticmethod
    def cfg_get(cfg, key, default):
        return getattr(cfg, key, default)

    def __init__(
        self,
        config=type(
            "Config",
            (),
            {
                # GaussianModel
                "image_height": 512,
                "batch_size": 1,
                "max_depth": 10.0,

                # Physics
                "step_size": 0.25,
                "max_step_height": 0.15,
                "turn_angle": 15.0,
                "agent_radius": 0.15,
                "eye_height": 1.25,

                # Camera
                "hfov_deg": 90.0,
                "vfov_deg": 90.0,
                "output_resolution": (336, 336), # (H, W)

                # Planning
                "planning_voxel_size": 0.05,
                "planning_safety_weight": 5.0,
                "goal_threshold": 0.5,
            },
        ),
    ):
        super().__init__(config)

        self.config = config

        # --------------------------------------------------
        # Physics & Navigation Parameters
        # --------------------------------------------------
        self.step_size = self.cfg_get(config, "step_size", 0.25)

        self.max_step_height = self.cfg_get(
            config,
            "max_step_height",
            0.15,
        )

        self.turn_angle_rad = np.deg2rad(
            self.cfg_get(config, "turn_angle", 15.0)
        )

        self.agent_radius = self.cfg_get(
            config,
            "agent_radius",
            0.15,
        )

        self.eye_height = self.cfg_get(
            config,
            "eye_height",
            1.25,
        )

        # --------------------------------------------------
        # Camera Parameters
        # --------------------------------------------------
        self.hfov_deg = self.cfg_get(
            config,
            "hfov_deg",
            90.0,
        )

        self.vfov_deg = self.cfg_get(
            config,
            "vfov_deg",
            90.0,
        )

        self.output_resolution = self.cfg_get(
            config,
            "output_resolution",
            (336, 336), # (H, W)
        )

        # --------------------------------------------------
        # Planning Parameters
        # --------------------------------------------------
        self.planning_voxel_size = self.cfg_get(
            config,
            "planning_voxel_size",
            0.05,
        )

        self.planning_safety_weight = self.cfg_get(
            config,
            "planning_safety_weight",
            5.0,
        )

        self.goal_threshold = self.cfg_get(
            config,
            "goal_threshold",
            0.5,
        )

        self.waypoint_threshold = self.step_size

        self.loose_tolerance = (
            self.turn_angle_rad * 0.8
        )

        # --------------------------------------------------
        # Agent State
        # --------------------------------------------------
        self.agent_pos = torch.zeros(
            (self.batch_size, 3),
            device=self.device,
            dtype=torch.float32,
        )

        self.agent_pos[:, 2] = self.eye_height

        self.agent_heading = torch.zeros(
            self.batch_size,
            device=self.device,
            dtype=torch.float32,
        )

        self.agent_pitch = torch.zeros(
            self.batch_size,
            device=self.device,
            dtype=torch.float32,
        )

        # --------------------------------------------------
        # Navigation Structures
        # --------------------------------------------------
        self.navigable_kdtree = None
        self.navigable_points = None
        self.planner_graph = None

        print(
            f"Neural Simulator Initialized "
            f"(Batch: {self.batch_size})."
        )


    def load_navigable_pcd(self, nav_pcd, scene_pcd=None, distance_threshold=0.05):
        """Loads scene for rendering AND builds robust planning graph."""

        if self._memory.coords.shape[2] == 0:
            print("Error: Simulator memory is empty. Please run image extraction and import_scene_gaussian first.")
            exit()
        elif scene_pcd is None:
            print("Warning: Scene PCD is empty! Skipping GS filtering.")
        elif len(scene_pcd.points) != 0:
            # Convert point cloud to CUDA Tensor
            scene_pts = torch.tensor(np.asarray(scene_pcd.points), device=self.device, dtype=torch.float32)
            
            # Fetch current GS coordinates (N, 3), which are already a CUDA Tensor
            gs_coords = self._memory.coords[0].transpose(0, 1).contiguous()
            
            # 1. Build GPU-accelerated KD-Tree
            kdtree = build_kd_tree(scene_pts)
            
            # 2. Query nearest neighbors (Note: Returns squared distances)
            dists_sq, _ = kdtree.query(gs_coords, nr_nns_searches=1)
            dists = torch.sqrt(dists_sq.squeeze(-1))
            
            # 3. Establish retention mask
            keep_tensor = dists <= distance_threshold
            
            original_n = len(keep_tensor)
            kept_n = keep_tensor.sum().item()
            print(f"    GS Points: {original_n} -> {kept_n} (Removed {original_n - kept_n} hallucinated noise points)")
            
            # Slice-update memory (PyTorch advanced indexing automatically filters along the last dimension)
            self._memory = MemoryState(
                coords=self._memory.coords[:, :, keep_tensor],
                feats=self._memory.feats[:, :, keep_tensor],
                rgb=self._memory.rgb[:, :, keep_tensor],
                gs_attrs=self._memory.gs_attrs[:, :, keep_tensor]
            )
            
        # =========================================================
        # Import Nav PCD and reconstruct the underlying path planning structures
        # =========================================================
        if nav_pcd is not None:
            self.navigable_points = torch.tensor(np.asarray(nav_pcd.points), device=self.device, dtype=torch.float32)
            self.navigable_kdtree = build_kd_tree(self.navigable_points)

            print(f">>> Building robust navigation graph from {len(self.navigable_points)} navigable points...")
            # Invoke the underlying routing pipeline to voxelize, map, and formulate the planning graph 
            # leveraging the freshly loaded ground-truth navigable geometry
            self._prepare_planning_data(self.navigable_points)
            print(">>> Navigation graph built successfully.")
        elif scene_pcd is None:
            print("Warning: Navigable PCD is empty!")


    def reset_agents(self, position=None, heading=None):
        if self.navigable_points is not None:
            indices = torch.tensor(np.random.randint(0, len(self.navigable_points), size=self.batch_size), device=self.device)
            base_pos = self.navigable_points[indices]
            self.agent_pos = base_pos + torch.tensor([[0, 0, self.eye_height]], dtype=torch.float32, device=self.device)
            self.agent_heading = torch.tensor(np.random.rand(self.batch_size), device=self.device) * 2 * math.pi
            if position is not None:
                position = position.to(self.device)
                self.agent_pos = position + torch.tensor([[0, 0, self.eye_height]], dtype=torch.float32, device=self.device)
            if heading is not None:
                heading = heading.to(self.device)
                self.agent_heading = heading
        else:
            self.agent_pos = torch.zeros((self.batch_size, 3), device=self.device, dtype=torch.float32)
            self.agent_pos[:, 2] = self.eye_height
            self.agent_heading = torch.zeros(self.batch_size, device=self.device, dtype=torch.float32)

        self.agent_pitch = np.zeros(self.batch_size, dtype=np.float32)
        return self._render_current_view()
    

    @torch.no_grad()
    def step(self, actions, render_observation=True):
        actions = torch.tensor(actions, device=self.device)
        move_mask = (actions == Action.MOVE_FORWARD)
        left_mask = (actions == Action.TURN_LEFT)
        right_mask = (actions == Action.TURN_RIGHT)
        
        # =======================================================================
        # 1. Heading Update (Unified Compass Coordinate System: 
        #    Clockwise increases with right turns, counter-clockwise decreases with left turns)
        # =======================================================================
        d_h = torch.where(right_mask, self.turn_angle_rad, 0.0) - torch.where(left_mask, self.turn_angle_rad, 0.0)
        self.agent_heading = (self.agent_heading + d_h) % (2 * np.pi)
        
        collided = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        if not move_mask.any() or self.navigable_kdtree is None:
            return (self._render_current_view() if render_observation else None), \
                   {"position": self.agent_pos.clone(), "heading": self.agent_heading.clone(), "collided": collided}

        # =======================================================================
        # 2. Precompute Physical Parameters (Microscopic Collision Avoidance + Macroscopic Pathfinding)
        # =======================================================================
        macro_radius = self.agent_radius * 2.0
        
        expected_pts = int(math.pi * (self.agent_radius**2) / (self.planning_voxel_size**2))
        expected_pts_macro = int(math.pi * (macro_radius**2) / (self.planning_voxel_size**2))
        
        # Microscopic physical collision threshold
        thr_xy = int(expected_pts * 0.6)     
        thr_slide = int(expected_pts * 0.3)  
        
        k_n = expected_pts_macro
        
        # Dynamic Protection: If k_n is truncated, scale down the physical threshold synchronously 
        # to prevent normal walking from being falsely classified as a wall collision.
        thr_xy = min(thr_xy, k_n - 5)
        thr_slide = min(thr_slide, k_n // 3)

        num_steps = max(int(self.step_size // self.planning_voxel_size), 1)
        step_dist = self.step_size / num_steps
        
        # -- Pure Sideslip Angle (Sideslip Only) ---
        slide_angle = math.pi / 2.0
        
        # =======================================================================
        # 3. Core Fix: Displacement Decomposition in the Compass Frame (X maps to sin, Y maps to cos)
        # =======================================================================
        dx_fwd = torch.sin(self.agent_heading) * step_dist
        dy_fwd = torch.cos(self.agent_heading) * step_dist
        
        # Left Strafe: In the compass system, strafing left corresponds to heading minus 90 degrees
        dx_L = torch.sin(self.agent_heading - slide_angle) * step_dist
        dy_L = torch.cos(self.agent_heading - slide_angle) * step_dist
        
        # Right Strafe: In the compass system, strafing right corresponds to heading plus 90 degrees
        dx_R = torch.sin(self.agent_heading + slide_angle) * step_dist
        dy_R = torch.cos(self.agent_heading + slide_angle) * step_dist
        
        z_off = torch.tensor([0, 0, self.eye_height], device=self.device)
        curr_p = self.agent_pos.clone()
        active = move_mask.clone()
        xy_tol = self.planning_voxel_size * 2.0

        p_fwd = torch.empty_like(curr_p)
        p_L = torch.empty_like(curr_p)
        p_R = torch.empty_like(curr_p)
        for _ in range(num_steps):
            if not active.any(): break
            
            # Three-way Candidates: Frontal, Left-slip, Right-slip
            p_fwd.copy_(curr_p); p_fwd[:, 0] += dx_fwd; p_fwd[:, 1] += dy_fwd
            p_L.copy_(curr_p); p_L[:, 0]   += dx_L;   p_L[:, 1]   += dy_L
            p_R.copy_(curr_p); p_R[:, 0]   += dx_R;   p_R[:, 1]   += dy_R
            
            # Batched Query / Vectorized Query
            f_batch = torch.cat([(p_fwd[active]-z_off), (p_L[active]-z_off), (p_R[active]-z_off)], dim=0)
            search_f = f_batch.clone(); search_f[:, 2] += (self.max_step_height * 0.5)
            
            _, idxs = self.navigable_kdtree.query(search_f.contiguous(), nr_nns_searches=k_n)
            pts = self.navigable_points[idxs.clamp(max=len(self.navigable_points)-1)]
            
            d_xy = torch.norm(f_batch.unsqueeze(1)[..., :2] - pts[..., :2], dim=-1)
            d_z = (pts[..., 2] - f_batch.unsqueeze(1)[..., 2]).abs()
            
            # Distinction: Snap Mask / Microscopic Collision Mask / Macroscopic Pathfinding Mask
            m_cnt = (d_xy <= xy_tol) & (d_z <= self.max_step_height)
            m_rad = (d_xy <= self.agent_radius) & (d_z <= self.max_step_height)
            m_macro = (d_xy <= macro_radius) & (d_z <= self.max_step_height)
            
            hc = m_cnt.any(dim=1)
            sc = m_rad.sum(dim=1)         
            
            N = (active.sum()).item()
            hc_fwd, hc_L, hc_R = hc[:N], hc[N:2*N], hc[2*N:]
            
            # Extract microscopic counts to strictly regulate physical collisions
            sc_fwd, sc_L, sc_R = sc[:N], sc[N:2*N], sc[2*N:]
            
            is_safe_fwd = hc_fwd & (sc_fwd >= thr_xy)
            is_safe_L   = hc_L   & (sc_L >= thr_slide)
            is_safe_R   = hc_R   & (sc_R >= thr_slide)
            
            # =======================================================================
            # 💡 Perfect Fusion: Macroscopic Field of View (FoV) + Geometric Centroid Cross Product
            # =======================================================================
            pts_fwd = pts[:N]
            m_macro_fwd = m_macro[:N] 
            
            sc_macro_fwd = m_macro_fwd.sum(dim=1)
            safe_sc_macro_fwd = sc_macro_fwd.unsqueeze(-1).clamp(min=1) 
            
            com_fwd = (pts_fwd * m_macro_fwd.unsqueeze(-1)).sum(dim=1) / safe_sc_macro_fwd
            
            n_nav_x = com_fwd[:, 0] - p_fwd[active, 0]
            n_nav_y = com_fwd[:, 1] - p_fwd[active, 1]
            
            # Cross-Product Decision: This algorithm preserves directional consistency even under orthogonal coordinate transformations!
            cross_prod = dx_fwd[active] * n_nav_y - dy_fwd[active] * n_nav_x
            prefer_L = cross_prod >= 0
            
            use_fwd = is_safe_fwd
            use_L = (~is_safe_fwd) & is_safe_L & (prefer_L | (~is_safe_R))
            use_R = (~is_safe_fwd) & is_safe_R & (~use_L)
            # ==========================================================
            
            # --- Height Snapping / Altitude Snapping ---
            d_xy.masked_fill_(~m_cnt, float('inf'))
            best_z = pts[torch.arange(len(f_batch)), d_xy.argmin(dim=1), 2]
            bz_fwd, bz_L, bz_R = best_z[:N], best_z[N:2*N], best_z[2*N:]
            
            # Update Position
            new_p = curr_p[active].clone()
            
            new_p = torch.where(use_fwd.unsqueeze(-1), p_fwd[active], new_p)
            new_p[:, 2] = torch.where(use_fwd, bz_fwd + self.eye_height, new_p[:, 2])
            
            new_p = torch.where(use_L.unsqueeze(-1), p_L[active], new_p)
            new_p[:, 2] = torch.where(use_L, bz_L + self.eye_height, new_p[:, 2])
            
            new_p = torch.where(use_R.unsqueeze(-1), p_R[active], new_p)
            new_p[:, 2] = torch.where(use_R, bz_R + self.eye_height, new_p[:, 2])
            
            curr_p[active] = new_p
            
            # Check deadlocks / Settle stuck states
            stuck = (~is_safe_fwd) & (~is_safe_L) & (~is_safe_R)
            collided[active] |= stuck
            active[active.clone()] &= (~stuck)

        self.agent_pos = curr_p
        obs = self._render_current_view() if render_observation else None
        return obs, {"position": self.agent_pos.clone(), "heading": self.agent_heading.clone(), "collided": collided}


    def _render_current_view(self):
        pos_tensor = self.agent_pos.clone()
        heading_tensor = self.agent_heading.clone()
        rgb, depth = self.get_agent_observation(
            position=pos_tensor,
            heading=heading_tensor
        )
        return {"rgb": rgb, "depth": depth}
    

    def _get_perspective_grid(self, batch_size: int) -> Tensor:
        out_h, out_w = self.output_resolution
        hfov_deg = self.hfov_deg
        vfov_deg = self.vfov_deg
        hfov_rad = hfov_deg * np.pi / 180.0
        vfov_rad = vfov_deg * np.pi / 180.0
        
        x = torch.linspace(-1, 1, out_w, device=self.device)
        y = torch.linspace(-1, 1, out_h, device=self.device)
        mesh_y, mesh_x = torch.meshgrid(y, x, indexing='ij')
        
        tan_h = torch.tan(torch.tensor(hfov_rad / 2.0, device=self.device))
        tan_v = torch.tan(torch.tensor(vfov_rad / 2.0, device=self.device))
        
        x_cart = mesh_x * tan_h
        y_cart = mesh_y * tan_v  
        z_cart = torch.ones_like(x_cart)
        
        r = torch.sqrt(x_cart**2 + y_cart**2 + z_cart**2)
        lon = torch.atan2(x_cart, z_cart)
        lat = torch.asin(y_cart / r)
        
        grid_u = lon / np.pi
        grid_v = lat / (np.pi / 2.0)
        
        grid = torch.stack([grid_u, grid_v], dim=-1)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return grid
    

    @torch.no_grad()
    def get_agent_observation(self, position: Tensor, heading: Tensor):

        pano_out = self.render(
            position=position, 
            heading=heading
        )
        
        rgb_in = pano_out.pred_rgb 
        
        depth_tensor = pano_out.pred_depth
        if depth_tensor.dim() == 3: 
            depth_in = depth_tensor.unsqueeze(1) 
        elif depth_tensor.dim() == 4: 
            if depth_tensor.shape[1] == 1:
                depth_in = depth_tensor
            elif depth_tensor.shape[-1] == 1:
                depth_in = depth_tensor.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unexpected depth shape: {depth_tensor.shape}")

        grid = self._get_perspective_grid(batch_size=position.shape[0])
        
        rgb_persp = F.grid_sample(rgb_in, grid, mode='bilinear', align_corners=False)
        depth_persp = F.grid_sample(depth_in, grid, mode='nearest', align_corners=False)
        
        rgb_persp = rgb_persp.permute(0, 2, 3, 1)
        depth_persp = depth_persp.squeeze(1)
        
        rgb_persp = (torch.clamp(rgb_persp, 0, 1) * 255).to(torch.uint8)
        
        return rgb_persp, depth_persp
    

    @torch.no_grad()
    def get_agent_12_views(self, position: Tensor, heading: Tensor):

        pano_out = self.render(
            position=position, 
            heading=heading
        )
        
        rgb_in = pano_out.pred_rgb

        depth_in = pano_out.pred_depth
        if depth_in.dim() == 3: depth_in = depth_in.unsqueeze(1)
        
        num_views = 12
        batch_size = position.shape[0]
        
        rgb_batch = rgb_in.repeat_interleave(num_views, dim=0)
        depth_batch = depth_in.repeat_interleave(num_views, dim=0)
        
        base_grid = self._get_perspective_grid(batch_size) 
        grids = base_grid.repeat_interleave(num_views, dim=0)
        
        angles = torch.arange(0, 360, step=30, device=self.device, dtype=torch.float32)
        offsets = (angles / 180.0) 
        offsets = offsets.view(1, num_views, 1, 1).repeat(batch_size, 1, 1, 1).view(-1, 1, 1)
        
        u_coords = grids[..., 0] + offsets
        grids[..., 0] = ((u_coords + 1) % 2) - 1
        
        rgb_persp_batch = F.grid_sample(rgb_batch, grids, mode='bilinear', align_corners=False)
        depth_persp_batch = F.grid_sample(depth_batch, grids, mode='nearest', align_corners=False)
        
        rgb_persp_batch = rgb_persp_batch.permute(0, 2, 3, 1)
        depth_persp_batch = depth_persp_batch.squeeze(1)
        
        rgb_persp_batch = (torch.clamp(rgb_persp_batch, 0, 1) * 255).to(torch.uint8)
        return rgb_persp_batch, depth_persp_batch


    @torch.no_grad()
    def get_panorama_observation(self, position: Tensor, heading: Tensor):
        pano_out = self.render(
            position=position, 
            heading=heading
        )
        pred_rgb = pano_out.pred_rgb * 255.
        rgb = pred_rgb.permute(0, 2, 3, 1).to(torch.uint8)

        pred_depth = pano_out.pred_depth
        depth = pred_depth.permute(0, 2, 3, 1)

        return rgb, depth



    def _prepare_planning_data(self, raw_positions_arr):
        """Builds 3D Voxel Grid, 2D Clearance Map, and Neighbor Graph via PyTorch Vectorization."""
        print(">>> Building Navigation Graph...")
        voxel_size = self.planning_voxel_size
        
        # 1. CUDA Tensor
        if isinstance(raw_positions_arr, np.ndarray):
            raw_tensor = torch.tensor(raw_positions_arr, device=self.device, dtype=torch.float32)
        else:
            raw_tensor = raw_positions_arr.to(self.device, dtype=torch.float32)
            
        # =====================================================================
        # Phase 1: voxel downsample
        # =====================================================================
        ds_points_tensor = raw_tensor.view(-1,3)

        # =====================================================================
        # Phase 2: 3D Grid
        # =====================================================================
        min_coords, _ = torch.min(raw_tensor, dim=0)
        max_coords, _ = torch.max(raw_tensor, dim=0)
        padding = 4
        grid_origin = min_coords - voxel_size * padding
        grid_shape = torch.ceil((max_coords - grid_origin) / voxel_size).to(torch.int32) + (padding * 2)
        
        raw_grid_indices = torch.floor((raw_tensor - grid_origin) / voxel_size).to(torch.int32)
        valid_mask = (raw_grid_indices >= 0).all(dim=1) & (raw_grid_indices < grid_shape.unsqueeze(0)).all(dim=1)
        raw_grid_indices = raw_grid_indices[valid_mask]
        
        # occupancy_grid_3d is True means Navigable
        occupancy_grid_3d = torch.zeros(tuple(grid_shape.tolist()), dtype=torch.bool, device=self.device)
        occupancy_grid_3d[raw_grid_indices[:, 0], raw_grid_indices[:, 1], raw_grid_indices[:, 2]] = True

        # =====================================================================
        # Phase 3: 2D Clearance (CPU EDT)
        # =====================================================================
        grid_shape_2d = grid_shape[:2].tolist()
        occupancy_grid_2d = torch.zeros(grid_shape_2d, dtype=torch.bool, device=self.device)
        occupancy_grid_2d[raw_grid_indices[:, 0], raw_grid_indices[:, 1]] = True
        
        occ_2d_np = occupancy_grid_2d.cpu().numpy()
        clearance_grid_2d_np = distance_transform_edt(occ_2d_np) * voxel_size
        clearance_grid_2d = torch.tensor(clearance_grid_2d_np, device=self.device, dtype=torch.float32)
        
        ds_indices = torch.floor((ds_points_tensor - grid_origin) / voxel_size).to(torch.long)
        ds_indices = torch.clamp(ds_indices, torch.zeros_like(grid_shape), grid_shape - 1)
        clearance_values = clearance_grid_2d[ds_indices[:, 0], ds_indices[:, 1]]

        # =====================================================================
        # Phase 4: KD-Tree scan
        # =====================================================================
        kdtree = build_kd_tree(ds_points_tensor)
        search_radius = voxel_size * 2.5
        search_radius_sq = search_radius ** 2
        
        K = min(64, len(ds_points_tensor)) 
        dists_sq, idxs = kdtree.query(ds_points_tensor, nr_nns_searches=K)
        
        if dists_sq.dim() == 3:
            dists_sq = dists_sq.squeeze(-1)
            idxs = idxs.squeeze(-1)

        # Pure PyTorch Fully Parallelized Ray Casting (with Fault-Tolerant Anti-Penetration Mechanism)
        N = ds_points_tensor.size(0)
        
        # 1. Filter candidate edges
        i_indices = torch.arange(N, device=self.device).view(N, 1).expand(N, K)
        candidate_mask = (dists_sq <= search_radius_sq) & (i_indices < idxs)
        
        i_cand = i_indices[candidate_mask]
        j_cand = idxs[candidate_mask]
        dist_cand = torch.sqrt(dists_sq[candidate_mask])
        
        # 2. Prepare parallel sampling parameters
        p1 = ds_points_tensor[i_cand]
        p2 = ds_points_tensor[j_cand]
        diff = p2 - p1
        
        step_len = voxel_size * 0.5
        num_steps = int(search_radius / step_len) + 2
        t = torch.linspace(0.0, 1.0, num_steps, device=self.device).view(1, num_steps, 1)
        
        # 3. Broadcast and compute coordinates for all 3D sample points
        ray_points = p1.unsqueeze(1) + t * diff.unsqueeze(1)
        
        # 4. Transform to grid indices and clip boundaries
        grid_coords = torch.floor((ray_points - grid_origin.view(1, 1, 3)) / voxel_size).to(torch.long)
        gx = torch.clamp(grid_coords[..., 0], 0, grid_shape[0] - 1)
        gy = torch.clamp(grid_coords[..., 1], 0, grid_shape[1] - 1)
        gz = torch.clamp(grid_coords[..., 2], 0, grid_shape[2] - 1)
        
        # 5. Extract voxel occupancy profiles along rays (M, 1, num_steps)
        ray_walkable = occupancy_grid_3d[gx, gy, gz].float().unsqueeze(1)
        
        # --- 1D Closing Operation along Rays (Anti-Penetration Voxel Gap-Filling) ---
        # Define the maximum physical point-cloud gap you wish to tolerate (e.g., 0.06m = 6cm)
        # As long as the actual wall thickness exceeds 6cm, rays are strictly prevented from penetrating!
        tolerance_m = 1.5 * voxel_size 
        
        K_size = int(tolerance_m / step_len)
        if K_size % 2 == 0: K_size += 1
        K_size = max(3, K_size)
        pad = K_size // 2
        
        # a. 1D Dilation: Instantly close voxel gaps under 6cm encountered along the ray
        ray_dilated = F.max_pool1d(ray_walkable, kernel_size=K_size, stride=1, padding=pad)
        
        # b. 1D Erosion: Leverage negative-inversion properties to accurately push the wall boundaries back to their true positions
        ray_closed = -F.max_pool1d(-ray_dilated, kernel_size=K_size, stride=1, padding=pad)
        
        # If the ray remains fully unblocked (all values are 1.0) after the closing operation, it indicates a clear, collision-free path.
        is_clear = (ray_closed.squeeze(1) > 0.5).all(dim=1)
        
        # 6. Extract strictly valid edges
        i_final = i_cand[is_clear]
        j_final = j_cand[is_clear]
        dist_final = dist_cand[is_clear]

        # =======================================================================
        # Phase 6: Generate Physical Costs and Export to C++ Dijkstra Sparse Graph
        # =======================================================================
        c_i = torch.clamp(clearance_values[i_final], min=0.01)
        c_j = torch.clamp(clearance_values[j_final], min=0.01)
        
        # Establish a broad safety corridor buffer zone
        safe_margin = 5 * self.agent_radius
        
        # Compute normalized intrusion level [0, 1]: 0 means completely outside the safety zone, 1 means flushing against the obstacle
        invasion_j = torch.clamp(safe_margin - c_j, min=0.0) / safe_margin
        invasion_i = torch.clamp(safe_margin - c_i, min=0.0) / safe_margin
        
        # --- Decouple Base Safety Penalty ---
        cost_i_to_j = dist_final + self.planning_safety_weight * (invasion_j ** 2)
        cost_j_to_i = dist_final + self.planning_safety_weight * (invasion_i ** 2)
        
        # --- Soft Linear Collision Penalty (Eliminate Stepwise Mutations / Discontinuities) ---
        # Cap the maximum detour cost to prevent the agent from circumnavigating the entire room just to bypass a single 3DGS outlier point
        max_detour_cost = self.planning_safety_weight
        
        # Compute actual penetration depth [0, agent_radius]
        penetration_j = torch.clamp(self.agent_radius - c_j, min=0.0)
        penetration_i = torch.clamp(self.agent_radius - c_i, min=0.0)
        
        # Deeper penetration yields higher additive costs, transitioning smoothly up to max_detour_cost
        collision_cost_j = max_detour_cost * (penetration_j / self.agent_radius)
        collision_cost_i = max_detour_cost * (penetration_i / self.agent_radius)
        
        cost_i_to_j += collision_cost_j
        cost_j_to_i += collision_cost_i
        
        rows = torch.cat([i_final, j_final]).cpu().numpy()
        cols = torch.cat([j_final, i_final]).cpu().numpy()
        
        data_cost = torch.cat([cost_i_to_j, cost_j_to_i]).cpu().numpy()
        data_dist = torch.cat([dist_final, dist_final]).cpu().numpy() # Pure physical distance
        
        ds_points_np = ds_points_tensor.cpu().numpy()
        N = len(ds_points_np)
        
        sparse_graph = csr_matrix((data_cost, (rows, cols)), shape=(N, N))
        sparse_dist_graph = csr_matrix((data_dist, (rows, cols)), shape=(N, N))

        print(f"    Graph Built: {N} nodes, {len(rows)} edges via PyTorch Ray-Marching.")

        point_to_idx = {tuple(map(float, p)): i for i, p in enumerate(ds_points_np)}
        clearance_map = {tuple(map(float, p)): float(c) for p, c in zip(ds_points_np, clearance_values.cpu().numpy())}

        self.planner_graph = {
            'points': ds_points_np,
            'kdtree': kdtree,
            'clearance_map': clearance_map,
            'occupancy_grid': occupancy_grid_3d.cpu().numpy(),
            'grid_origin': grid_origin.cpu().numpy(),
            'grid_size': voxel_size,
            
            'sparse_graph': sparse_graph,            
            'sparse_dist_graph': sparse_dist_graph,
            'point_to_idx': point_to_idx,
            'idx_to_point': {i: p for p, i in point_to_idx.items()}
        }


    def get_shortest_paths_to(self, target_positions, start_positions=None):
        """
        Calculates shortest paths using robust C++ Dijkstra and Hybrid Smoothing.
        """
        if self.planner_graph is None:
            return [[] for _ in range(self.batch_size)], np.full(self.batch_size, np.inf)

        targets = np.array(target_positions)
        if targets.ndim == 1: targets = np.tile(targets, (self.batch_size, 1))
        
        if start_positions is None:
            starts = self.agent_pos.cpu().numpy() - np.array([[0, 0, self.eye_height]]) # 兼容如果agent_pos改成了Tensor
        else:
            starts = np.array(start_positions)
            if starts.ndim == 1: starts = np.tile(starts, (self.batch_size, 1))
            
        final_paths = []
        final_lengths = []
        
        graph = self.planner_graph
        kdtree = graph['kdtree']
        
        for b in range(self.batch_size):
            start_raw = starts[b]
            end_raw = targets[b]
            
            s_tensor = torch.tensor(start_raw, device=self.device, dtype=torch.float32).unsqueeze(0)
            e_tensor = torch.tensor(end_raw, device=self.device, dtype=torch.float32).unsqueeze(0)
            
            _, s_idx_t = kdtree.query(s_tensor, nr_nns_searches=1)
            _, e_idx_t = kdtree.query(e_tensor, nr_nns_searches=1)
            
            s_idx = s_idx_t.item()
            e_idx = e_idx_t.item()
            
            # C++ Dijkstra
            dist_matrix, predecessors = dijkstra(
                csgraph=graph['sparse_graph'], 
                directed=False, 
                indices=s_idx, 
                return_predecessors=True
            )
            
            # If the path is unreachable (-9999 is the sentinel value / flag established by SciPy)
            if predecessors[e_idx] == -9999 and s_idx != e_idx:
                final_paths.append([])
                final_lengths.append(np.inf)
                continue
                
            # Backtrack the path based on the predecessor array
            raw_path = []
            curr_idx = e_idx
            while curr_idx != s_idx:
                raw_path.append(graph['idx_to_point'][curr_idx])
                curr_idx = predecessors[curr_idx]
            raw_path.append(graph['idx_to_point'][s_idx])
            raw_path.reverse() # From source to destination / From start to goal
            
            path_arr = np.array(raw_path)
            # --- Deep Gaussian-like Smoothing ---
            # Discard sparsification and retain the 0.05m high-precision density; 
            # switch instead to multiple passes of a large-window mean filter.
            if len(path_arr) > 5:
                smoothed = np.copy(path_arr)
                for _ in range(5):
                    # 5-point moving average with an expanded receptive field, completely wiping out the staircase artifacts of the A* voxel grid.
                    smoothed[2:-2, :2] = (
                        smoothed[:-4, :2] +
                        smoothed[1:-3, :2] +
                        smoothed[2:-2, :2] +
                        smoothed[3:-1, :2] +
                        smoothed[4:, :2]
                    ) / 5.0
                path_nodes = smoothed.tolist()
            else:
                path_nodes = raw_path
                
            # Length calculation
            path_arr = np.array(path_nodes)
            if len(path_arr) > 1:
                diffs = path_arr[1:] - path_arr[:-1]
                total_length = np.sum(np.linalg.norm(diffs, axis=1))
                total_length += np.linalg.norm(start_raw - path_arr[0])
                total_length += np.linalg.norm(end_raw - path_arr[-1])
            else:
                total_length = np.linalg.norm(start_raw - end_raw)

            final_paths.append(path_nodes)
            final_lengths.append(total_length)
            
        return final_paths, np.array(final_lengths)


    def generate_action_sequence(self, target_positions=None, start_positions=None, reference_paths=None, max_actions=500):
        """
        [Physics-Informed Lookahead Closed-Loop Mode]
        - Global Mode: Accepts target_positions and utilizes voxel-based global pathfinding.
        - Follow Mode: Accepts reference_paths, directly trusts the native trajectories, 
                       and leverages low-level Lookahead to achieve perfect inner-cutting smooth cornering
        """
        # =======================================================================
        # 1. If reference_paths are provided, directly bypass all pathfinding and concatenation
        # =======================================================================
        if reference_paths is not None:
            paths = reference_paths
            # Force-start from the very first point of the reference path
            start_positions = [rp[0] for rp in reference_paths]
            # Placeholder length, solely configured to pass the subsequent validity check
            lengths = [1.0] * self.batch_size 
        else:
            paths, lengths = self.get_shortest_paths_to(target_positions, start_positions)

        # 1. State Backup
        original_pos = self.agent_pos.clone()
        original_heading = self.agent_heading.clone()

        if start_positions is not None:
            self.agent_pos = torch.tensor(start_positions, device=self.device, dtype=torch.float32) + torch.tensor([[0, 0, self.eye_height]], device=self.device, dtype=torch.float32)

        active_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        batch_sequences = [[] for _ in range(self.batch_size)]
        current_wp_indices = [1] * self.batch_size
        waypoints_list = [torch.tensor(p, device=self.device, dtype=torch.float32) if p else None for p in paths]

        # Error check
        for b in range(self.batch_size):
            if not paths[b] or len(paths[b]) < 2 or lengths[b] == np.inf:
                active_mask[b] = False
                batch_sequences[b].append("STOP")

        # 2. Physical Rollout
        for _ in range(max_actions):
            if not active_mask.any():
                break

            actions_this_step = []
            actions_str_this_step = []

            for b in range(self.batch_size):
                if not active_mask[b]:
                    actions_this_step.append(Action.STOP)
                    actions_str_this_step.append("STOP")
                    continue

                # feet position of agent
                feet_pos = self.agent_pos[b].clone()
                feet_pos[2] -= self.eye_height
                current_heading_val = self.agent_heading[b].item()
                
                waypoints_t = waypoints_list[b]
                final_goal = waypoints_t[-1]

                # Check destination
                if torch.norm(feet_pos - final_goal).cpu().item() < self.goal_threshold:
                    actions_this_step.append(Action.STOP)
                    actions_str_this_step.append("STOP")
                    batch_sequences[b].append("STOP")
                    active_mask[b] = False
                    continue

                # Lookahead next waypoint
                while current_wp_indices[b] < len(waypoints_t) - 1:
                    if torch.norm(feet_pos - waypoints_t[current_wp_indices[b]]) < self.waypoint_threshold:
                        current_wp_indices[b] += 1
                    else:
                        break

                target_wp = waypoints_t[current_wp_indices[b]]
                dx = target_wp[0] - feet_pos[0]
                dy = target_wp[1] - feet_pos[1]

                # Close-loop action
                if torch.norm(torch.tensor([dx, dy], device=self.device)) < 0.05:
                    act_str, act_enum = "MOVE_FORWARD", Action.MOVE_FORWARD
                else:
                    target_heading_val = (math.pi / 2) - torch.atan2(dy, dx).item()
                    target_heading_val = (target_heading_val + math.pi) % (2 * math.pi) - math.pi
                    
                    diff = (target_heading_val - current_heading_val + np.pi) % (2 * np.pi) - np.pi
                    if abs(diff) > getattr(self, 'loose_tolerance', 0.17):
                        if diff > 0:
                            act_str, act_enum = "TURN_RIGHT", Action.TURN_RIGHT
                        else:
                            act_str, act_enum = "TURN_LEFT", Action.TURN_LEFT
                    else:
                        act_str, act_enum = "MOVE_FORWARD", Action.MOVE_FORWARD

                actions_this_step.append(act_enum)
                actions_str_this_step.append(act_str)

            # Action sequence
            for b in range(self.batch_size):
                if active_mask[b]:
                    batch_sequences[b].append(actions_str_this_step[b])

            # Action step
            if active_mask.any():
                self.step(actions_this_step, render_observation=False)

        # 3. State Rewind
        self.agent_pos = original_pos
        self.agent_heading = original_heading
        return batch_sequences[0] if self.batch_size == 1 else batch_sequences