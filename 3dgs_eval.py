import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import image2sim
import data_tools
import random
import gc


from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

@torch.no_grad()
def evaluate():
    print(f"\n{'='*60}")
    print("[INIT] Starting NVS Metrics Evaluation")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = type('Config', (), {'image_height': 512, 'batch_size': 1, 'max_depth': 10.})
    checkpoint_dir = "pretrained_models"
    
    dataset_sources = {
        #'realsee_synthetic': ("realsee", "data/datasets/RealSee3D/synthetic_data"),
        #'realsee_real':      ("realsee", "data/datasets/RealSee3D/real_world_data"),
        'matterport':        ("matterport", "data/datasets/Matterport3D_360/data"),
        #'hm3d':              ("hm3d", "data/datasets/hm3d_360"),
        #'gibson':              ("gibson", "data/datasets/gibson_360")
        #'structured3d':      ("structured3d", "data/datasets/Structured3D/Structured3D"),
    }

    print(">>> Indexing Evaluation Datasets...")
    scene_index = {}
    total_scenes = 0
    for key, (dtype, root_dir) in dataset_sources.items():
        if not os.path.exists(root_dir):
            print(f"  [Warning] {root_dir} not found. Skipping.")
            continue
            
        if dtype == "structured3d":
            scenes = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if d.startswith("scene_")]
        else: 
            scenes = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            
        scene_index[key] = sorted(scenes)
        total_scenes += len(scenes)
        print(f"  [{key}] Found {len(scenes)} scenes.")

    if total_scenes == 0:
        print("No scenes found for evaluation. Exiting.")
        return

    print("\n>>> Initializing Image2Sim Model...")
    model = image2sim.GaussianModel(config).to(device)
    
    model = data_tools.load_checkpoint(model, checkpoint_dir)
    model.torch_compile()
    model.eval()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

    H = config.image_height
    W = config.image_height * 2

    with torch.amp.autocast(device_type='cuda'):
        with torch.no_grad():
            for source_key, scenes in scene_index.items():
                dataset_type = dataset_sources[source_key][0]
                print(f"\n{'#'*50}")
                print(f"Evaluating Dataset: {source_key.upper()}")
                print(f"{'#'*50}")

                for scene_dir in tqdm(scenes, desc=f"Eval {source_key}"):
                    deterministic_seed = hash(scene_dir.split("/")[-1]) % (2**32)
                    random.seed(deterministic_seed)
                    np.random.seed(deterministic_seed)
                    torch.manual_seed(deterministic_seed)
                    
                    ret = data_tools.build_scene_pointcloud_data(
                        scene_dir, 
                        dataset_type=dataset_type, 
                        device=device, 
                        voxel_size=0.005, 
                        model=model,
                        inpaint_depth=True,
                        max_batch_size=1000,
                        num_targets=1
                    )
                    
                    if ret[0] is None or len(ret[-1]) == 0:
                        continue
                        
                    scene_xyz, scene_rgb, scene_feats, scene_gs, target_frames = ret
                    
                    model.batch_size = len(target_frames)
                    model.import_scene_gaussian(xyz=scene_xyz, rgb=scene_rgb, feats=scene_feats, gs_attrs=scene_gs)
                    
                    for frame in target_frames:
                        pos_tensor, heading_tensor = data_tools.get_position_heading_from_pose(frame, device)

                        try:
                            gt_img = Image.open(frame['rgb_path']).convert('RGB')
                            gt_rgb_np = np.asarray(gt_img.resize((W, H), Image.BILINEAR))
                            gt_rgb_tensor = torch.tensor(gt_rgb_np, device=device, dtype=torch.float32).unsqueeze(0) / 255.0
                            gt_rgb_tensor = gt_rgb_tensor.permute(0, 3, 1, 2) # (1, 3, H, W)
                        except Exception as e:
                            continue

                        render_output = model.render(position=pos_tensor, heading=heading_tensor, num_steps=2)
                        
                        pred_rgb_tensor = render_output.pred_rgb

                        is_realsee_real = (dataset_type == 'realsee') and ('real_world_data' in scene_dir)
                        is_matterport = (dataset_type == 'matterport')

                        top_h = 0
                        bottom_h = H

                        if is_realsee_real:
                            top_h = int((115 / 800) * H)
                            bottom_h = H - int((125 / 800) * H)
                        elif is_matterport:
                            top_h = int((145 / 1024) * H)
                            bottom_h = H - int((149 / 1024) * H)

                        gt_rgb_eval = gt_rgb_tensor[:, :, top_h:bottom_h, :]
                        pred_rgb_eval = pred_rgb_tensor[:, :, top_h:bottom_h, :]

                        psnr_metric.update(pred_rgb_eval, gt_rgb_eval)
                        ssim_metric.update(pred_rgb_eval, gt_rgb_eval)
                        lpips_metric.update(pred_rgb_eval, gt_rgb_eval)
                        

                        model.reset_memory()
                        gc.collect()
                        torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("Evaluation Metrics")
    print("="*50)
    
    try:
        print(f"PSNR  ↑: {psnr_metric.compute().item():.4f}")
    except Exception as e:
        print(f"PSNR Error: {e}")

    try:
        print(f"SSIM  ↑: {ssim_metric.compute().item():.4f}")
    except Exception as e:
        print(f"SSIM Error: {e}")

    try:
        print(f"LPIPS ↓: {lpips_metric.compute().item():.4f}")
    except Exception as e:
        print(f"LPIPS Error: {e}")

        
    print("="*50)

if __name__ == "__main__":
    evaluate()