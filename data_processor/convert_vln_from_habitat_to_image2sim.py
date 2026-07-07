import json
import torch
import numpy as np
import cv2
import math
import os
import glob
import open3d as o3d
from collections import defaultdict
import image2sim
import data_tools 


def save_video(frames, filename="debug.mp4", fps=5):
    if not frames: return
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))
    [out.write(cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2BGR)) for f in frames]
    out.release()
    print(f"  -> Video saved to {filename}")

if __name__ == "__main__":

    device = "cuda"
    SAVE_DEBUG_VIDEO = False
    
    config = type('Config', (), {'image_height': 512, 'batch_size': 1, 'max_depth': 10., "hfov_deg": 90.0, "vfov_deg": 90., "output_resolution": (336, 336)})
    neural_simulator = image2sim.NeuralSimulator(config).to(device)
    neural_simulator = data_tools.load_checkpoint(neural_simulator, "checkpoints")
    neural_simulator.torch_compile()
    neural_simulator.eval()

    # Input the VLN data path (e.g., R2R-CE, RxR-CE, REVERIE-CE, SRDF...)
    EXTERNAL_DATA_DIR = "data/datasets/R2R_VLNCE_v1-3/train"
    #EXTERNAL_DATA_DIR = "data/datasets/RxR_VLNCE_v0/train/train_guide.json"
    #EXTERNAL_DATA_DIR = "data/datasets/REVERIE/reverie_train.json" # Refer to Dynam3D  https://github.com/MrZihan/Dynam3D

    vln_episodes = []

    if os.path.isdir(EXTERNAL_DATA_DIR):
        json_files = glob.glob(os.path.join(EXTERNAL_DATA_DIR, "*.json"))
    else:
        json_files = [EXTERNAL_DATA_DIR]
    print(f">>> Found {len(json_files)} JSON files，loading...")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    vln_episodes.extend(data)
                else:
                    vln_episodes.extend(data["episodes"])
        except Exception as e:
            print(f"Read file error: {file_path}, {e}")

    print(f">>> Read json files done {len(vln_episodes)} trajectories�?)

    episodes_by_scene = defaultdict(list)
    for ep in vln_episodes:
        if ("language" in ep["instruction"] and "en" in ep["instruction"]["language"]) or "language" not in ep["instruction"]: # for rxr
            scene_name = ep["scene_id"].split("/")[-1].split(".")[0]
            episodes_by_scene[scene_name].append(ep)
        
    print(f">>> Found trajectories for {len(episodes_by_scene)} unique scenes in JSON.")

    # dataset_sources
    dataset_sources = {
        #'realsee_synthetic': ("realsee", "data/datasets/RealSee3D/synthetic_data", "data/nav_map/RealSee3D"),
        #'realsee_real':      ("realsee", "data/datasets/RealSee3D/real_world_data", "data/nav_map/RealSee3D"),
        #'structured3d':      ("structured3d", "data/datasets/Structured3D/Structured3D", "data/nav_map/Structured3D"),
        'matterport':        ("matterport", "data/datasets/Matterport3D_360/data", "data/nav_map/mp3d"),
        #'hm3d':              ("hm3d", "data/datasets/hm3d_360", "data/nav_map/hm3d"),
        #'gibson':              ("gibson", "data/datasets/gibson_360", "data/nav_map/gibson")
    }

    all_results = []

    for source_key, dataset_info in dataset_sources.items():
        dataset_type = dataset_info[0]
        IMAGES_DIR = dataset_info[1]
        SCENES_DIR = dataset_info[2]
        
        print(f"\n{'#'*60}")
        print(f"🚀 [START] Processing Dataset: {source_key.upper()}")
        print(f"{'#'*60}")

        if not os.path.exists(SCENES_DIR):
            print(f"Error: Scenes directory '{SCENES_DIR}' not found.")
            os.makedirs(SCENES_DIR, exist_ok=True)
            print("Created empty scenes directory. Skipping for now...")
            continue

        all_files = glob.glob(os.path.join(SCENES_DIR, "*_navigable.pcd"))
        scene_names = [os.path.basename(f).replace("_navigable.pcd", "") for f in all_files]
        print(f"Found {len(scene_names)} scenes in {source_key}.")
        
        with torch.amp.autocast(device_type='cuda'):
            with torch.no_grad():
                for scene_name in scene_names:

                    scene_id = scene_name.split("-")[-1]
                    if scene_id not in episodes_by_scene:
                        continue
                    if os.path.exists(f"{dataset_type}_{scene_name}.json"):
                        continue

                    print(f"\n{'='*50}")
                    print(f"Processing Scene: {scene_name} ({source_key}) | Contains {len(episodes_by_scene[scene_id])} Trajectories")
                    print(f"{'='*50}")
            
                    full_pcd_path = os.path.join(SCENES_DIR, f"{scene_name}.pcd")
                    nav_pcd_path = os.path.join(SCENES_DIR, f"{scene_name}_navigable.pcd")
                    if not os.path.exists(nav_pcd_path):
                        continue
            
                    
                    scene_path = os.path.join(IMAGES_DIR, scene_name)
                    nav_pcd = o3d.io.read_point_cloud(nav_pcd_path)
    
                    (scene_xyz, scene_rgb, scene_feats, scene_gs, all_frames_data) = data_tools.build_scene_pointcloud_data(
                        scene_path, 
                        dataset_type=dataset_type, 
                        device=device,
                        voxel_size=0.01,
                        model=neural_simulator, 
                        max_batch_size=1000,
                        inpaint_depth=True
                    )
                    
                    if scene_xyz is None:
                        print(f"Skipping scene {scene_name}: Failed to build point cloud data.")
                        continue
                        
                    neural_simulator.import_scene_gaussian(xyz=scene_xyz, rgb=scene_rgb, feats=scene_feats, gs_attrs=scene_gs)
                    neural_simulator.load_navigable_pcd(nav_pcd, None)
                        
                    
                    torch.cuda.empty_cache()

                    target_episodes = episodes_by_scene[scene_id]
                    scene_results = []
                    for ep in target_episodes:
                        ep["start_position"][0], ep["start_position"][1], ep["start_position"][2]  = ep["start_position"][0], - ep["start_position"][2], ep["start_position"][1]
                        start_pos = ep["start_position"]
                        for i in range(len(ep["reference_path"])):
                            ep["reference_path"][i][0], ep["reference_path"][i][1], ep["reference_path"][i][2] = ep["reference_path"][i][0], - ep["reference_path"][i][2], ep["reference_path"][i][1]

                        end_pos = ep["reference_path"][-1].copy()
                        
                        camera_start_pos = np.array([start_pos])
                        camera_start_pos[:,2] += neural_simulator.eye_height 
    
                        neural_simulator.agent_pos = torch.tensor(camera_start_pos, device=neural_simulator.device, dtype=torch.float32)
                        vec = np.array(ep["reference_path"][1]) - np.array(start_pos) if len(ep["reference_path"]) > 1 else [1, 0]
                        init_heading = (math.pi / 2) - math.atan2(vec[1], vec[0])
                        neural_simulator.agent_heading = torch.tensor([init_heading], device=neural_simulator.device, dtype=torch.float32)

                        max_actions = 500
                        actions = neural_simulator.generate_action_sequence(
                            reference_paths = [ep["reference_path"]],
                            max_actions = max_actions
                        )

                        if len(actions) == 1 or len(actions) >= max_actions:
                            print(f"  -> Episode {ep['episode_id']} Fail: Recorded {len(actions)} steps.")
                            continue
                            
                        trajectory_log = []
                        frames = []

                        for step_idx, act in enumerate(actions):

                            feet_pos = neural_simulator.agent_pos[0].cpu().numpy().copy()
                            feet_pos[2] -= neural_simulator.eye_height
                            current_heading = neural_simulator.agent_heading[0].item()
                            
                            trajectory_log.append({
                                "step": step_idx,
                                "pos": feet_pos.tolist(),
                                "heading": current_heading,
                                "action": act
                            })
                            
                            if SAVE_DEBUG_VIDEO:
                                rgb, _ = neural_simulator.get_panorama_observation(neural_simulator.agent_pos, neural_simulator.agent_heading)
                                frames.append(rgb[0, 88:424, 256:768].cpu().numpy()) 
                            
                            if act != "STOP":
                                neural_simulator.step([getattr(image2sim.Action, act)], render_observation=False)

                        inst_text = ep.get("instruction", {}).get("instruction_text", "Follow the reference path to the goal.")
                        
                        traj_data = {
                            "episode_id": f"{dataset_type}_{scene_name}_{ep['episode_id']}",
                            "instruction_data": {
                                "style": "rxr_ground_truth",
                                "instruction": inst_text
                            },
                            "trajectory": trajectory_log
                        }
                        
                        scene_results.append(traj_data)
                        print(f"  -> Episode {ep['episode_id']} Success: Recorded {len(actions)} steps.")
                        
                        if SAVE_DEBUG_VIDEO and frames:
                            video_name = f"{dataset_type}_{scene_name}_{ep['episode_id']}.mp4"
                            save_video(frames, video_name, fps=10)

                    if scene_results:
                        scene_json_name = f"{dataset_type}_{scene_name}.json"
                        with open(scene_json_name, 'w', encoding='utf-8') as f:
                            json.dump(scene_results, f, indent=4, ensure_ascii=False)
                        print(f"\n>>> [Scene Completed] Saved {len(scene_results)} episodes to {scene_json_name}")
                    else:
                        print(f"\n>>> [Scene Skipped] No valid episodes generated for {scene_name}.")

                    neural_simulator.reset_memory()
                        
        
    print("\n🎉 All matched datasets and scenes processed successfully!")