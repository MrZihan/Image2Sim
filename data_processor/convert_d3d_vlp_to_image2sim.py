import json
import os
import cv2
import torch
import numpy as np
import open3d as o3d
import math
from tqdm import tqdm
from PIL import Image
from scipy.sparse.csgraph import dijkstra
import random
from copy import deepcopy
import image2sim
import data_tools

def save_video(frames, filename="debug_traj.mp4", fps=5):
    if not frames: 
        return
    frames_pil = [Image.fromarray(img) for img in frames]
    w, h = frames_pil[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    for img in frames_pil:
        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out.write(open_cv_image)
    out.release()

def sample_start_position(sim, target_pos, min_dist=1.0, max_dist=5.0):
    graph_data = sim.planner_graph
    kdtree = graph_data['kdtree']
    dist_graph = graph_data['sparse_dist_graph']
    
    target_tensor = torch.tensor(target_pos, device=sim.device, dtype=torch.float32).unsqueeze(0)
    _, target_idx_t = kdtree.query(target_tensor, nr_nns_searches=1)
    target_idx = target_idx_t.item()
    
    distances = dijkstra(csgraph=dist_graph, directed=False, indices=target_idx, limit=max_dist + 1.0)
    valid_indices = np.where((distances >= min_dist) & (distances <= max_dist))[0]
    
    if len(valid_indices) == 0:
        return None
    
    start_idx = np.random.choice(valid_indices)
    return np.array(graph_data['idx_to_point'][start_idx])

def generate_annotated_episode(sim, start_pos, target_pos, instruction_text, goal_position=None, goal_id=None, render_video=False):
    end_pos_safe = np.array(target_pos).copy()
    end_pos_safe[2] += 0.1
    target_positions = np.array([end_pos_safe])

    start_pos_safe = np.array(start_pos).copy()
    start_pos_safe[2] += 0.1
    
    paths, lengths = sim.get_shortest_paths_to(target_positions, np.array([start_pos_safe]))
    if not paths or not paths[0] or len(paths[0]) < 2: 
        return None
        
    waypoints = np.array(paths[0])
    
    current_feet_pos = np.array(start_pos_safe, dtype=np.float64)
    if len(waypoints) > 1:
        vec = waypoints[1] - waypoints[0]
        current_heading = (np.pi / 2) - np.arctan2(vec[1], vec[0])
        current_heading = (current_heading + np.pi) % (2 * np.pi) - np.pi
    else:
        current_heading = 0.0
        
    camera_start_pos = current_feet_pos.copy()
    camera_start_pos[2] += sim.eye_height 
    sim.agent_pos = torch.tensor([camera_start_pos], device=sim.device, dtype=torch.float32)
    sim.agent_heading = torch.tensor([current_heading], device=sim.device, dtype=torch.float32)
    
    action_sequence = sim.generate_action_sequence(target_positions=target_positions)
    if isinstance(action_sequence[0], list):
        action_sequence = action_sequence[0]
        
    episode_data = {"trajectory": [], "frames": []}
    state_window = []
    
    for step_count, action_type in enumerate(action_sequence):
        curr_pos_t = sim.agent_pos.clone()
        curr_heading_t = sim.agent_heading.clone()  
        
        if render_video:
            pano_rgb, _ = sim.get_panorama_observation(position=curr_pos_t, heading=curr_heading_t)
            # pano_rgb  (1, H, W, 3)
            img_np = pano_rgb[0].cpu().numpy()
            episode_data['frames'].append(img_np)

        state_window.append({'pos': curr_pos_t[0, :2].clone(), 'heading': curr_heading_t[0].item()})
        if len(state_window) > 4: 
            state_window.pop(0)
            
        if len(state_window) == 4:
            disp = torch.norm(state_window[-1]['pos'] - state_window[0]['pos']).item()
            heading_diff = abs((state_window[-1]['heading'] - state_window[0]['heading'] + math.pi) % (2*math.pi) - math.pi)
            
            if disp <= 2 * sim.planning_voxel_size and heading_diff <= sim.turn_angle_rad:
                if len(episode_data['trajectory']) > 0:
                    episode_data['trajectory'][-1]['action'] = "STOP"
                return None
                
        log_feet = curr_pos_t[0].cpu().numpy()
        log_feet[2] -= sim.eye_height
        episode_data['trajectory'].append({
            "step": step_count,
            "pos": log_feet.tolist(),
            "heading": float(curr_heading_t[0].item()),
            "action": action_type
        })
        
        if action_type == "STOP":
            break
            
        sim.step([image2sim.Action[action_type]], render_observation=False)
        
    if len(episode_data['trajectory']) < 5: 
        return None
        
    return {
        "instruction_data": {
            "style": "grounding",
            "instruction": instruction_text,
        },
        "goal_position": goal_position,
        "goal_id": goal_id,
        "trajectory": episode_data['trajectory'],
        "frames": episode_data['frames']
    }


short_description_prompts = [
            # --- Direct Commands ---
            "Navigate to the {target}.",
            "Go to the {target}.",
            "Find the {target}.",
            "Show me the {target}.",
            "Take me to the {target}.",
            "Lead me to the {target}.",
            "Head to the {target}.",
            "Move towards the {target}.",
            "Proceed to the {target}.",
            "Locate the {target}.",
            "Direct me to the {target}.",
            "Go find the {target}.",
            "Show me where the {target} is.",
            "Get me to the {target}.",
            "Find {target}.",
            "Go to {target}.",

            # --- Polite Requests ---
            "Please navigate to the {target}.",
            "Can you help me find the {target}?",
            "Could you find the {target} for me?",
            "Can you take me to the {target}?",
            "Please go to the {target}.",
            "Would you mind showing me where the {target} is?",
            "Please show me the way to the {target}.",
            "Could you please guide me to the {target}?",
            "Help me find the {target}, please.",
            "I'd like you to find the {target}.",
            "Could you locate the {target}?",
            "If you can, please find the {target}.",
            "May I see the {target}?",
            "Would you be able to navigate to the {target}?",

            # --- Interrogative / Question-based ---
            "Where is the {target}?",
            "Can you find the {target}?",
            "Do you know where the {target} is?",
            "What's the way to the {target}?",
            "How do I get to the {target}?",
            "Is the {target} nearby?",
            "Can you see the {target}?",
            "Where can I find the {target}?",
            "Could you tell me the location of the {target}?",

            # --- Goal-Oriented Statements ---
            "I need to find the {target}.",
            "I'm looking for the {target}.",
            "My destination is the {target}.",
            "I want to go to the {target}.",
            "The goal is to reach the {target}.",
            "Let's go to the {target}.",
            "I need to get to the {target}.",
            "I have to go to the {target}.",
            "My objective is the {target}.",
            "We need to end up at the {target}.",
            "I'm trying to locate the {target}.",
            "The next stop is the {target}.",
            "I'm searching for the {target}.",
            "I wish to be taken to the {target}.",
            
            # --- More Conversational & Natural Language ---
            "Alright, let's find the {target} now.",
            "Okay, time to head over to the {target}.",
            "I wonder where the {target} is, let's look.",
            "Let's see if we can find the {target}.",
            "Now, for the {target}.",
            "Okay, on to the {target}.",
            "Let's try to locate the {target}.",
            "Time to find the {target}.",
            "I am in need of the {target}.",
            "I think the {target} is what I need next.",
            "Let's go on a search for the {target}.",
            "The {target} is what we're looking for.",
            "I need to see the {target}.",
            "Okay robot, find me the {target}.",
            "Now, I'd like to see the {target}.",
            
            # --- Task-Based Instructions (Navigation is a prerequisite) ---
            "Go get the {target}.",
            "I need the {target}, please go to it.",
            "To proceed, we must go to the {target}.",
            "First, go to the {target}, then wait for instructions.",
            "Before we do anything else, find the {target}.",
            "The plan is to navigate to the {target} first.",
            "Go to the {target} and report back.",
            "I need to interact with the {target}, take me there.",
            "I'll need you to go to the {target} to complete the task.",

            # --- Variations with different phrasing ---
            "Guide me to the {target}.",
            "Show me the path to the {target}.",
            "I require assistance in finding the {target}.",
            "Could you chart a course to the {target}?",
            "I'm on a quest for the {target}.",
            "Make your way to the {target}.",
            "I command you to find the {target}.",
            "Your task is to find the {target}.",
            "The mission is to locate the {target}.",
            "Let's venture to the {target}.",
            "Pathfind to the {target}.",
            "Seek out the {target}.",
            "Travel to the {target}.",
            "Advance to the {target}'s location.",
            "I am trying to find the {target}.",
            "Can you figure out where the {target} is?",
            "Point me to the {target}.",
            "Find me a {target}.",
            "Navigate to a {target}.",
            "I want to see the {target}.",
            "Let's make our way to the {target}.",
            "Please, the {target}.",
            "The {target}, please.",
        ]


if __name__ == "__main__":
    # ==========================================
    # Configure
    # ==========================================
    RENDER_VIDEO = True
    ANNOTATIONS_FILE = "D3D-VLP/data/datasets/annotation.json"
    dataset_sources = {
        'matterport':        ("matterport", "/data/scene_datasets/Matterport3D_360/data", "/data/nav_map/mp3d"),
        'hm3d':              ("hm3d", "/data/scene_datasets/hm3d_360", "/data/nav_map/hm3d"),
        'scannet':           ("scannet", "/data/scene_datasets/ScanNet/scannet_train_images/frames_square", "/data/nav_map/ScanNet"),
        #'3rscan':            ("3rscan", "/data/scene_datasets/3RScan/scenes", "/data/nav_map/3RScan"),
        'arkitscenes':       ("arkitscenes", "/data/scene_datasets/ARKitScenes/3dod/Training", "/data/nav_map/ARKitScenes")
    }

    with open(ANNOTATIONS_FILE, 'r') as f:
        annotations = json.load(f)

    print(">>> Initializing Neural Simulator...")
    config = type('Config', (), {'image_height': 512, 'batch_size': 1, 'max_depth': 10.})
    sim = image2sim.NeuralSimulator(config).to("cuda")
    sim = data_tools.load_checkpoint(sim, "checkpoints") 
    sim.torch_compile()
    sim.eval()

    SCENES_DIR = IMAGES_DIR = None
    with torch.amp.autocast(device_type='cuda'):
        with torch.no_grad():
            for scene_key, annotation in annotations.items():
                for dataset_type in dataset_sources:
                    if dataset_type.replace("matterport","mp3d") in scene_key.lower():
                        SCENES_DIR = dataset_sources[dataset_type][2]
                        IMAGES_DIR = dataset_sources[dataset_type][1]
                        DATASET_TYPE = dataset_type

                if SCENES_DIR is None:
                    continue


                scene_name = scene_key.split('/')[-1]
                print(f"\n{'='*60}")
                print(f"🚀 Processing Scene: {scene_name} ({len(annotation)} annotations)")
                print(f"{'='*60}")
                
                if os.path.exists(f"output_{DATASET_TYPE}_{scene_name}.json"):
                    continue

                full_pcd_path = os.path.join(SCENES_DIR, f"{scene_name}.pcd")
                nav_pcd_path = os.path.join(SCENES_DIR, f"{scene_name}_navigable.pcd")
                scene_path = os.path.join(IMAGES_DIR, scene_name)
                if not os.path.exists(nav_pcd_path):
                    continue

                try:
                    scene_pcd = o3d.io.read_point_cloud(full_pcd_path) if os.path.exists(full_pcd_path) else None
                    nav_pcd = o3d.io.read_point_cloud(nav_pcd_path)

                    (scene_xyz, scene_rgb, scene_feats, scene_gs, _) = data_tools.build_scene_pointcloud_data(
                        scene_path, dataset_type=DATASET_TYPE, device="cuda",
                        voxel_size=0.01, model=sim, max_batch_size=1000, inpaint_depth=True
                    )
                    
                    if scene_xyz is None: continue
                        
                    sim.import_scene_gaussian(xyz=scene_xyz, rgb=scene_rgb, feats=scene_feats, gs_attrs=scene_gs)
                    sim.load_navigable_pcd(nav_pcd, scene_pcd)
                    
                except Exception as e:
                    print(f"❌ Failed to load scene {scene_name}: {e}")
                    continue

                scene_episodes = []
                random.shuffle(annotation)
                for i, ann in enumerate(tqdm(annotation[:1000], desc="Generating Trajectories")):
                    try:
                        target_pos = ann['instance_position'][0]
                        goal_position = deepcopy(ann['instance_position'][0])
                        instance_type = ann['instance_type'][0]
                        goal_id = ann['instance_id'][0]

                        if instance_type is None : continue

                        if ann['context'] != "":
                            instruction = ann["context"]
                        else:
                            instruction = ann['response']

                        if isinstance(ann["response"], list):
                            instruction = ann["response"][0]


                        if isinstance(ann['instance_type'][0], list):
                            ann['instance_type'][0] = ann['instance_type'][0][0]

                        if isinstance(instruction, list):
                            instruction = instruction[0]

                        if len(instruction) > 100:
                            instruction = random.choice(short_description_prompts).replace("{target}", ann['instance_type'][0])
                        instruction = instruction.replace("object", ann['instance_type'][0])

                        instruction = instruction.replace("1. ","")

                        if "object" in instruction or instruction == "":
                            continue

                        print(instruction)

                        nav_points = sim.planner_graph['points'] 
                        
                        dist_xy = np.linalg.norm(nav_points[:, :2] - target_pos[:2], axis=1)
                        nearby_mask = dist_xy < 1.
                        
                        if np.any(nearby_mask):
                            nearby_points = nav_points[nearby_mask]
                            nearby_dists = dist_xy[nearby_mask]
                            
                            z_max = target_pos[2] + 0.2  
                            z_min = target_pos[2] - 1.5
                            
                            vertical_mask = (nearby_points[:, 2] <= z_max) & (nearby_points[:, 2] >= z_min)
                            
                            if np.any(vertical_mask):
                                valid_points = nearby_points[vertical_mask]
                                valid_dists = nearby_dists[vertical_mask]
                                
                                true_floor_z = np.min(valid_points[:, 2])
                                
                                on_floor_mask = valid_points[:, 2] <= (true_floor_z + 0.2)
                                floor_points = valid_points[on_floor_mask]
                                floor_dists = valid_dists[on_floor_mask]
                                
                                best_local_idx = np.argmin(floor_dists)
                                target_pos = floor_points[best_local_idx].copy()
                            else:
                                nearest_idx = np.argmin(dist_xy)
                                target_pos = nav_points[nearest_idx].copy()
                        else:
                            nearest_idx = np.argmin(dist_xy)
                            target_pos = nav_points[nearest_idx].copy()

                        start_pos = sample_start_position(sim, target_pos, min_dist=2.0, max_dist=5.0)
                        if start_pos is None: continue
                            
                        episode = generate_annotated_episode(
                            sim, start_pos, target_pos, instruction, goal_position=goal_position, goal_id=goal_id, render_video=RENDER_VIDEO
                        )
                        
                        if episode:

                            clean_type = instance_type.replace(' ', '_')
                            ep_id = f"{scene_name}_human_{i}_{clean_type}"
                            
                            if RENDER_VIDEO and episode.get("frames"):
                                video_name = f"debug_vid_{ep_id}.mp4"
                                save_video(episode["frames"], filename=video_name, fps=10)
                                print(f"\n  -> [Debug] Saved video: {video_name}")
                            
                            if "frames" in episode:
                                del episode["frames"]
                            
                            episode["episode_id"] = ep_id
                            scene_episodes.append(episode)

                    except Exception as e:
                        print(f"❌ Failed to process scene {scene_name}: {e}")

                if scene_episodes:
                    out_filename = f"output_{DATASET_TYPE}_{scene_name}.json"
                    with open(out_filename, 'w', encoding='utf-8') as f:
                        json.dump(scene_episodes, f, indent=4, ensure_ascii=False)
                    print(f"\n✅ [Success] Saved {len(scene_episodes)} clean trajectories to {out_filename}")