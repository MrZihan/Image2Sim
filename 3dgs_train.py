import os
import copy
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torch.nn.functional as F
from PIL import Image
import random
import torch.optim as optim
import image2sim
import data_tools
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from tqdm import tqdm
import glob
from torch.distributed.optim import ZeroRedundancyOptimizer
import gc
from torch.optim.lr_scheduler import CosineAnnealingLR

class EMADecoder:
    def __init__(self, student_decoder, decay=0.999):
        """
        EMA Teacher
        decay=0.999
        """
        self.decay = decay
        self.model = copy.deepcopy(student_decoder)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, student_decoder):
        for ema_param, student_param in zip(self.model.parameters(), student_decoder.parameters()):
            if ema_param.dtype.is_floating_point:
                # EMA = decay * EMA + (1 - decay) * Student
                ema_param.data.mul_(self.decay).add_(student_param.data, alpha=1.0 - self.decay)

                
class Trainer(torch.nn.Module):
    def __init__(self, model, config, max_batch_size=1):
        super().__init__()
        self.model = model
        self.config = config
        self.max_batch_size = max_batch_size

    def forward(self, scene_path, dataset_type, training_stage=0, teacher_decoder=None, num_grad_src = 1, num_nograd_src = 2, num_targets = 2):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        device = torch.device("cuda")
        
        H = self.config.image_height
        W = self.config.image_height * 2

        is_valid = False
        loaded_data = None

        if True:
            if training_stage == 0:
                ret = data_tools.build_scene_nvs_pointcloud_data(
                    scene_path, 
                    dataset_type=dataset_type, 
                    device=device, 
                    voxel_size=0.01, 
                    model=self.model, 
                    num_grad_src=num_grad_src,
                    num_nograd_src=num_nograd_src,
                    num_targets=num_targets
                )
            else:
                with torch.no_grad():
                    ret = data_tools.build_scene_nvs_pointcloud_data(
                        scene_path, 
                        dataset_type=dataset_type, 
                        device=device, 
                        voxel_size=0.01, 
                        model=self.model, 
                        num_grad_src=num_grad_src,
                        num_nograd_src=num_nograd_src,
                        num_targets=num_targets
                    )
                    
            if ret[0] is not None and len(ret[-1]) > 0:
                is_valid = True
                loaded_data = ret
        #except Exception as e:
        #    print(f"[Rank {dist.get_rank()}] Data Load Error: {e}")
        #    del e 
        #    torch.cuda.empty_cache() 
        #    is_valid = False

        if is_valid:
            (scene_xyz, scene_rgb, scene_feats, scene_gs, target_frames) = loaded_data
            loss_scale = 1.0
        else:
            loss_scale = 0.0
            scene_xyz = torch.tensor([[[0.0, 0.0], [0.0, 0.1], [1.0, 1.0]]], device=device, dtype=torch.float32)
            scene_rgb = torch.ones((1, 3, 2), device=device, dtype=torch.float32)
            scene_feats = torch.zeros((1, 16, 2), device=device, dtype=torch.float32)
            one_gs = torch.tensor([[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [1.0]], device=device, dtype=torch.float32)
            scene_gs = one_gs.repeat(1, 1, 2)
            target_frames = [{'type': 'dummy', 'rgb_path': None, 'pose': np.eye(4)}]

        self.model.batch_size = len(target_frames)
        self.model.import_scene_gaussian(xyz=scene_xyz, rgb=scene_rgb, feats=scene_feats, gs_attrs=scene_gs)

        del loaded_data
        
        is_realsee_real = (dataset_type == 'realsee') and ('real_world_data' in scene_path)
        is_matterport = (dataset_type == 'matterport')
        
        strip_mask = torch.ones((1, H, W), device=device, dtype=torch.float32)
        if is_realsee_real:
            strip_mask[:, :int((115 / 800) * H), :] = 0
            strip_mask[:, (H - int((125 / 800) * H)):, :] = 0
        elif is_matterport:
            strip_mask[:, :int((145 / 1024) * H), :] = 0
            strip_mask[:, (H - int((149 / 1024) * H)):, :] = 0

        batch_gt_rgb_list, batch_gt_depth_list = [], []
        batch_pos_list, batch_heading_list, batch_strip_mask_list = [], [], []

        for frame in target_frames:
            if is_valid:
                pos_tensor, heading_tensor = data_tools.get_position_heading_from_pose(frame, device)
                try:
                    gt_img = Image.open(frame['rgb_path']).convert('RGB')
                    gt_rgb_tensor = torch.tensor(np.asarray(gt_img.resize((W, H), Image.BILINEAR)), device=device, dtype=torch.float32).unsqueeze(0)
                except Exception:
                    gt_rgb_tensor = torch.zeros((1, H, W, 3), device=device)
                try:
                    gt_depth_tensor = torch.tensor(data_tools.load_gt_depth_normalized(frame, H, W), device=device, dtype=torch.float32).unsqueeze(0)
                except Exception:
                     gt_depth_tensor = torch.zeros((1, 1, H, W), device=device)
            else:
                pos_tensor, heading_tensor = torch.zeros((1, 3), device=device), torch.zeros((1, 1), device=device)
                gt_rgb_tensor, gt_depth_tensor = torch.zeros((1, H, W, 3), device=device), torch.zeros((1, 1, H, W), device=device)

            batch_gt_rgb_list.append(gt_rgb_tensor)
            batch_gt_depth_list.append(gt_depth_tensor)
            batch_pos_list.append(pos_tensor)
            batch_heading_list.append(heading_tensor)
            batch_strip_mask_list.append(strip_mask)

        batch_gt_rgb = torch.cat(batch_gt_rgb_list)
        batch_gt_depth = torch.cat(batch_gt_depth_list)
        batch_pos_tensor = torch.cat(batch_pos_list)
        batch_heading = torch.cat(batch_heading_list)
        batch_strip_mask = torch.cat(batch_strip_mask_list)

        with torch.no_grad():
            if is_valid:
                rgb_input = batch_gt_rgb.permute(0, 3, 1, 2) / 255.0
                gt_patch_tokens = self.model.encoder_backbone(self.model.normalize(rgb_input))
            else:
                gt_patch_tokens = torch.zeros((len(target_frames), 1024, H // 16, W // 16), device=device)
        
        del batch_gt_rgb_list, batch_gt_depth_list, batch_pos_list, batch_heading_list

        proj_loss, align_loss, meanflow_loss, perc_loss, distill_loss, rgb_loss, depth_loss = self.model(
            batch_pos_tensor, 
            batch_heading, 
            batch_gt_rgb, 
            batch_gt_depth, 
            gt_patch_tokens, 
            batch_strip_mask, 
            batch_strip_mask,
            training_stage=training_stage,
            teacher_decoder=teacher_decoder
        )
        
        raw_loss = proj_loss + align_loss + meanflow_loss + perc_loss + distill_loss
        final_loss = raw_loss * loss_scale

        if dist.get_rank() == 0 and is_valid:
            print(f"proj:{proj_loss.item():.4f}, align:{align_loss.item():.4f}, flow:{meanflow_loss.item():.4f}, lpips:{perc_loss.item():.4f}, distill:{distill_loss.item():.4f}, rgb:{rgb_loss.item():.4f}, depth:{depth_loss.item():.4f}")

        del batch_gt_rgb, batch_gt_depth, batch_pos_tensor, gt_patch_tokens, scene_xyz, scene_feats
        
        return final_loss


def save_checkpoint(model, optimizer, scaler, iteration, save_dir, ema_teacher=None):
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"image2sim_iter_{iteration:06d}.pth")
        raw_model = model.module if hasattr(model, "module") else model
        checkpoint = {
            'iteration': iteration,
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict()
        }
        if ema_teacher is not None:
            checkpoint['ema_teacher'] = ema_teacher.model.state_dict()
        torch.save(checkpoint, save_path)
        print(f"\n[Rank 0] Successfully saved checkpoint to: {save_path}")



def load_checkpoint(model, optimizer, scaler, scheduler, load_path_or_dir, device, ema_teacher=None):
    """
    
    Args:
        model: DDP wrapped model or standard model
        optimizer: Optimizer instance
        scaler: GradScaler instance
        load_path_or_dir: Checkpoint path or file
    
    Returns:
        start_iteration
    """
    load_path = load_path_or_dir
    
    if os.path.isdir(load_path_or_dir):
        pattern = os.path.join(load_path_or_dir, "image2sim_iter_*.pth")
        files = glob.glob(pattern)
        if not files:
            print(f"[Warning] No checkpoints found in {load_path_or_dir}. Starting from scratch.")
            return 0
        load_path = max(files, key=lambda x: int(x.split("_iter_")[-1].split(".pth")[0]))
    
    if not os.path.isfile(load_path):
        print(f"[Warning] Checkpoint file {load_path} not found. Starting from scratch.")
        return 0

    print(f"[{dist.get_rank()}] Loading checkpoint from {load_path}...")
    
    checkpoint = torch.load(load_path, map_location=device)
    
    ckpt_dict = checkpoint['model']
    
    is_model_ddp = hasattr(model, 'module')
    is_ckpt_ddp = list(ckpt_dict.keys())[0].startswith('module.')
    
    if is_model_ddp and not is_ckpt_ddp:
        ckpt_dict = {f'module.{k}': v for k, v in ckpt_dict.items()}
    elif not is_model_ddp and is_ckpt_ddp:
        ckpt_dict = {k.replace('module.', ''): v for k, v in ckpt_dict.items()}
        
    try:
        model.load_state_dict(ckpt_dict, strict=True)
    except RuntimeError as e:
        print(f"[Error] Key mismatch during loading. Trying with strict=False. Details: {e}")
        model.load_state_dict(ckpt_dict, strict=False)

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
        
    start_iteration = checkpoint.get('iteration', 0)
    print(f"[{dist.get_rank()}] Successfully resumed from iteration {start_iteration}.")
    
    if ema_teacher is not None:
        if 'ema_teacher' in checkpoint:
            ema_teacher.model.load_state_dict(checkpoint['ema_teacher'])
            print(f"[{dist.get_rank()}] EMA Teacher state restored from checkpoint.")
        else:
            raw_model = model.module if hasattr(model, "module") else model
            ema_teacher.model.load_state_dict(raw_model.rendering_decoder.state_dict())
            print(f"[{dist.get_rank()}] Legacy checkpoint detected. Initialized EMA Teacher with Student.")

    scheduler.last_epoch = start_iteration - 1
    scheduler.step()

    return start_iteration


def seed_everything(global_rank, base_seed=0):
    """Set the random seed"""
    seed = base_seed + global_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    config = type('Config', (), {'image_height': 512, 'batch_size': 2, 'max_depth': 10.})
    checkpoint_path = "pretrained_models"
    max_batch_size = config.batch_size # Training batch_size
    num_grad_src = max_batch_size
    num_nograd_src = max_batch_size*64
    num_targets = max_batch_size
    training_stage = 0
    teacher_decoder = None
    total_iterations = 400009
    save_interval = 5000

    dataset_sources = {
        'realsee_synthetic': ("data/scene_datasets/RealSee3D/synthetic_data", "realsee"),
        'realsee_real':      ("data/scene_datasets/RealSee3D/real_world_data", "realsee"),
        'structured3d':      ("data/scene_datasets/Structured3D/Structured3D", "structured3d"),
        'matterport':        ("data/scene_datasets/Matterport3D_360/data", "matterport"),
    }

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    seed_everything(global_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
 
    print("Indexing datasets...")
    scene_index = {}
    
    for key, (root_dir, dtype) in dataset_sources.items():
        if not os.path.exists(root_dir):
            print(f"Warning: {root_dir} not found.")
            scene_index[key] = []
            continue
            
        if dtype == "matterport":
            scenes = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        elif dtype == "structured3d":
            scenes = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if d.startswith("scene_")]
        else: # realsee
            scenes = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            
        scene_index[key] = sorted(scenes)
        print(f"  [{key}] Found {len(scenes)} scenes.")

    peek_iteration = 0
    actual_load_path = checkpoint_path
    
    if os.path.isdir(actual_load_path):
        pattern = os.path.join(actual_load_path, "image2sim_iter_*.pth")
        files = glob.glob(pattern)
        if files:
            actual_load_path = max(files, key=lambda x: int(x.split("_iter_")[-1].split(".pth")[0]))

    if os.path.isfile(actual_load_path):
        try:
            peek_iteration = int(actual_load_path.split("_iter_")[-1].split(".pth")[0])
            print(f"=>  {peek_iteration} steps")
        except ValueError:
            loc = f'cuda:{local_rank}'
            checkpoint_dict = torch.load(actual_load_path, map_location=loc, weights_only=False)
            peek_iteration = checkpoint_dict.get("iteration", 0) 
            del checkpoint_dict
            print(f"=>  {peek_iteration} steps")

    print(f"\nInitializing Model...")
    model = image2sim.GaussianModel(config).to(device)

    if peek_iteration >= total_iterations * 3 // 4:
        print(f"\n=> [Stage 2] >= 3/4，refinement mode")
        model.encoder_backbone.requires_grad_(False)
        model.feature_upsampler.requires_grad_(False)
        model.semantic_aligner.requires_grad_(False)
        model.rendering_decoder.time_encoder.requires_grad_(False)
        model.rendering_decoder.time_injector.requires_grad_(False)
        model.rendering_decoder.stem_h2.requires_grad_(False)
        model.rendering_decoder.stem_h4.requires_grad_(False)
        model.rendering_decoder.stage1_process.requires_grad_(False)
        model.rendering_decoder.down1.requires_grad_(False)
        model.rendering_decoder.stage2_process.requires_grad_(False)
        model.rendering_decoder.down2.requires_grad_(False)
        model.rendering_decoder.bottleneck_blocks.requires_grad_(False)
        model.rendering_decoder.sem_proj_bn.requires_grad_(False)
        model.rendering_decoder.sem_injector_bn.requires_grad_(False)

    ema_teacher = EMADecoder(model.rendering_decoder, decay=0.999)

    run_training_loop = Trainer(model, config, max_batch_size=max_batch_size)

    run_training_loop = DDP(run_training_loop, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, run_training_loop.parameters()), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=total_iterations, eta_min=1e-6)
    
    start_iteration = load_checkpoint(model, optimizer, scaler, scheduler, checkpoint_path, device, ema_teacher=ema_teacher)

    source_keys = list(dataset_sources.keys())
    
    print(f"\nStarting Training Loop ({total_iterations} iterations)...")
    
    for i in tqdm(range(start_iteration, total_iterations)):
        optimizer.zero_grad()

        key_id = i % len(source_keys)
        available_scenes = scene_index[source_keys[key_id]]
        
        if not available_scenes:
            print("The dataset cannot be found:", source_keys[key_id])
            continue
            
        chosen_scene = random.choice(available_scenes)
        dtype = dataset_sources[source_keys[key_id]][1]
        scene_name = chosen_scene.split("/")[-1]
        print(f"\n--- Iter {i+1}/{total_iterations} | Source: {source_keys[key_id]}, {scene_name}---")
        
        if i == total_iterations * 3 // 4 - 1:
            print(f"Pre-training done, please restart the code to fine-tune the super resolution CNN")
            save_checkpoint(model, optimizer, scaler, i + 1, save_dir=checkpoint_path, ema_teacher=ema_teacher)
            exit()

        if i >= total_iterations // 4:
            num_grad_src = 0
            training_stage = 1
            teacher_decoder = ema_teacher.model

        if i >= total_iterations * 3 // 4:
            training_stage = 2


        with torch.amp.autocast(device_type='cuda'):
            loss = run_training_loop(chosen_scene, dtype, training_stage=training_stage, teacher_decoder=teacher_decoder, num_grad_src = num_grad_src, num_nograd_src = num_nograd_src, num_targets = num_targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(run_training_loop.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        raw_model = model.module if hasattr(model, 'module') else model
        ema_teacher.update(raw_model.rendering_decoder)
        
        del loss
        model.reset_memory()
        # gc.collect()
        # torch.cuda.empty_cache()
        
        # Save Checkpoint
        if (i + 1) % save_interval == 0:
            if dist.get_rank() == 0:
                print(f"Iteration {i} completed, consolidating and saving...")
                save_checkpoint(model, optimizer, scaler, i + 1, save_dir=checkpoint_path, ema_teacher=ema_teacher)