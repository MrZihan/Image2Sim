import torch
import argparse
import numpy as np
import torchvision.transforms.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast

import open3d as o3d
import matplotlib.pyplot as plt
import math
import random
from PIL import Image
import cv2
from scipy.sparse.csgraph import dijkstra

import numpy as np
from collections import defaultdict
import os
import glob
import networkx as nx
import json
from ultralytics import YOLOWorld
import traceback
import image2sim
import data_tools
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def get_dynamic_prompt(style_choice):
    """
    R2R-, REVERIE-, Demand- style prompt matrix。
    """
    # ==========================================
    # 1. R2R (path-following)
    # ==========================================
    r2r_syntax_pool = [
        "Structure: Ensure every action verb is paired with at least one specific landmark.",
        "Structure: Describe the path as a sequence of object-to-object transitions.",
        "Structure: Sequence the instruction by listing objects in the order they appear in the field of view, connected by short action verbs.",
        "Structure: Replace directional fluff with object-relative directions.",
    ]
    r2r_tone_pool = [
        "Tone: Landmark-centric. Treat objects as the primary anchors for every directional change.",
        "Tone: Observant and precise, identifying unique visual markers at every junction.",
        "Tone: Instructive, like a guide directing a follower through a complex indoor space."
    ]

    # ==========================================
    # 2. REVERIE (high-level)
    # ==========================================
    reverie_syntax_pool = [
        "Structure: Use passive voice to describe the target's location.",
        "Structure: Specify the destination room/area first, followed by the specific receptacle holding the target.",
        "Structure: Formulate as a direct search command.",
        "Structure: Use spatial prepositions to emphasize the target's relationship within environment."
    ]
    reverie_tone_pool = [
        "Tone: Objective and analytical, focusing solely on the existence and position of the target.",
        "Tone: Goal-driven, completely ignoring the intermediate walking path.",
        "Tone: Definitive, as if confirming the final location of a misplaced item."
    ]

    # ==========================================
    # 3. Demand
    # ==========================================
    demand_syntax_pool = [
        "Structure: Frame it as a request for assistance to fetch or check a STRICTLY VISIBLE object in the final frame.",
        "Structure: State the user's need first, followed by the location. The demanded object MUST be physically present in the image.",
        "Structure: Phrase it as a casual reminder or thought about an item that is unambiguously captured by the camera.",
        "Structure: Formulate an action-oriented demand targeting an explicitly visible object, without inventing any items."
    ]
    demand_tone_pool = [
        "Tone: Casual and conversational, but absolutely fact-based regarding the visible environment.",
        "Tone: Urgent and direct, demanding interaction with an item that is clearly right there in the scene.",
        "Tone: Polite but firm, requesting navigation to the exact visible location of a real object."
    ]

    # ==========================================
    # 4. Route sample
    # ==========================================
    if style_choice == 'r2r':
        core_task = "Generate a path-following navigation instruction describing the path with important landmarks (objects)."
        selected_syntax = random.choice(r2r_syntax_pool)
        selected_tone = random.choice(r2r_tone_pool)
    elif style_choice == 'reverie':
        core_task = "Generate a high-level, goal-oriented instruction. IGNORE the walking path entirely. Focus ONLY on identifying the final target object and its immediate surrounding/receptacle."
        selected_syntax = random.choice(reverie_syntax_pool)
        selected_tone = random.choice(reverie_tone_pool)
    else:  # demand
        core_task = "Generate a natural human-centric demand. Simulate an everyday spoken request to interact with, fetch, or find a specific object at the destination."
        selected_syntax = random.choice(demand_syntax_pool)
        selected_tone = random.choice(demand_tone_pool)



    # Final Prompt
    prompt = (
        f"{core_task}\n\n"
        f"- {selected_syntax}\n"
        f"- {selected_tone}\n\n"
    )
    
    return prompt


class InstructionAnnotator:
    def __init__(self, model_id="Qwen/Qwen3-VL-32B-Instruct", device="cuda"):
        print(f">>> Loading VLM: {model_id} ...")
        self.device = device
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def generate_instructions(self, frames_list, trajectory_log, style_choice, style_prompt):
        """
        End-to-End Trajectory Annotator: Accepts wide Field-of-View (FoV) image sequences 
        containing median strips along with low-level action logs; executes semantic segmentation, 
        quality self-inspection, and generates high-quality natural language instructions.
        """
        if not trajectory_log or not frames_list:
            return None

        # =======================================================================
        # 1. Odometry-based Semantic Chunking
        # =======================================================================
        macro_steps = []
        start_idx = 0
        
        # Golden Thresholds: 1.0-meter displacement OR 45-degree cumulative yaw/rotation
        DIST_THRESHOLD = 1.0      
        TURN_THRESHOLD = 45
        
        for i in range(1, len(trajectory_log)):
            start_pos = np.array(trajectory_log[start_idx]['pos'])
            start_heading = trajectory_log[start_idx]['heading']
            curr_pos = np.array(trajectory_log[i]['pos'])
            curr_heading = trajectory_log[i]['heading']
            action = trajectory_log[i]['action']
            
            dist = np.linalg.norm(curr_pos[:2] - start_pos[:2])
            diff_h = (curr_heading - start_heading + np.pi) % (2 * np.pi) - np.pi
            
            if action == "STOP" or dist >= DIST_THRESHOLD or abs(diff_h) >= np.deg2rad(TURN_THRESHOLD):
                macro_steps.append({
                    'start_idx': start_idx,
                    'end_idx': i,
                    'dist': dist,
                    'diff_h': diff_h,
                    'final_action': action
                })
                start_idx = i

        if start_idx < len(trajectory_log) - 1:
            end_idx = len(trajectory_log) - 1
            dist = np.linalg.norm(np.array(trajectory_log[end_idx]['pos'][:2]) - np.array(trajectory_log[start_idx]['pos'][:2]))
            diff_h = (trajectory_log[end_idx]['heading'] - trajectory_log[start_idx]['heading'] + np.pi) % (2 * np.pi) - np.pi
            macro_steps.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'dist': dist,
                'diff_h': diff_h,
                'final_action': trajectory_log[end_idx]['action']
            })

        # ==========================================================
        # 2. 	VLM content
        # ==========================================================
        content = [
            {
                "type": "text", 
                "text": "You are an expert navigation instructor for an embodied AI agent. "
                        "The following alternating sequence of images and text represents a navigation trajectory. "
                        "Each image shows a 180-degree ultra-wide panoramic view from the agent's front-facing camera, "
                        "followed by a description of the agent's movement from that viewpoint."
            }
        ]
        
        for idx, m_step in enumerate(macro_steps):
            img_np = frames_list[m_step['start_idx']]
            img_pil = Image.fromarray(img_np)
            
            dist = m_step['dist']
            deg_diff = np.rad2deg(m_step['diff_h'])
            action = m_step['final_action']
            
            # Action -> Text description
            if action == "STOP":
                act_desc = "Stop. This is the final destination."
            elif dist <= DIST_THRESHOLD/2: 
                if deg_diff >= TURN_THRESHOLD:
                    act_desc = "Turn right."
                elif deg_diff <= -TURN_THRESHOLD:
                    act_desc = "Turn left."
            else:
                if deg_diff >= TURN_THRESHOLD:
                    act_desc = "Walk forward while turning right."
                elif deg_diff <= -TURN_THRESHOLD:
                    act_desc = "Walk forward while turning left."
                else:
                    act_desc = "Walk straight forward."

            content.append({"type": "text", "text": f"\n--- Observation {idx+1} ---"})
            content.append({"type": "image", "image": img_pil})
            content.append({"type": "text", "text": f"Movement: {act_desc}"})

        content.append({
            "type": "text", 
            "text": f"\nTASK 1: STRICT QUALITY ASSURANCE (CRITICAL)\n"
                    f"You must act as a harsh quality inspector. Carefully compare the consecutive images. Look for these fatal errors:\n"
                    f"1. Freeze/Stuck: The visual viewpoint does NOT change across multiple frames, indicating the agent is stuck.\n"
                    f"2. Clipping: The agent teleports through solid walls, closed doors, or furniture unnaturally.\n"
                    f"3. Artifacts: The images contain severe rendering bugs, black holes, or chaotic meshes.\n\n"
                    f"TASK 2: INSTRUCTION GENERATION\n"
                    f"Only if the trajectory is flawlessly realistic, fulfill the following request:\n"
                    f"{style_prompt}\n\n"
                    f"CRITICAL ANTI-HALLUCINATION RULE: You MUST NOT invent objects. You MUST NOT copy examples from the prompt. Every object or landmark you mention MUST be clearly visible in the provided images. If you hallucinate objects, the entire dataset will be ruined.\n\n" # <--- 新增防幻觉护栏
                    f"OUTPUT FORMAT REQUIREMENT:\n"
                    f"You must structure your response EXACTLY as follows:\n"
                    f"REASONING: [Write 1-2 sentences explicitly evaluating if the movement is visible and checking for clipping/freezing]\n"
                    f"RESULT: [Output exactly 'VALID' or 'INVALID_TRAJECTORY']\n"
                    f"INSTRUCTION: [If VALID, write the natural language instruction here. If INVALID, leave blank]"
        })

        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.device)

        # =======================================================================
        # 4. Inference & Parsing
        # =======================================================================
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=512,  
            use_cache=True,      # KV Cache must be enabled.
            do_sample=False      # Use Greedy Search; disabling random sampling significantly accelerates generation speed and enhances logical stability.
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        output_text = output_text.strip()
        
        if "INVALID_TRAJECTORY" in output_text or "INVALID" in output_text.split("INSTRUCTION:")[0].upper():
            try:
                reason = output_text.split("RESULT:")[0].replace("REASONING:", "").strip()
            except:
                reason = "Unknown parsing error"
            print(f"  [VLM QA REJECTED] Reason: {reason}")
            return None
            
        try:
            clean_instruction = output_text.split("INSTRUCTION:")[1].replace("```", "").strip().strip('"').strip("'")
        except Exception as e:
            clean_instruction = output_text.replace("```", "").strip().strip('"').strip("'")
            
        return {"style": style_choice, "instruction": clean_instruction}
    

# =======================================================================
# 1. Base Configuration & Constants
# =======================================================================

LVIS_INDOOR_VOCABULARY = [
    # === (Furniture - Large & Storage) ===
    "bed", "bunk bed", "crib", "cabinet", "filing cabinet", "bookcase", 
    "wardrobe", "closet", "dresser", "chest of drawers", "shelf", "rack", 
    "table", "dining table", "coffee table", "desk", "nightstand", 
    "kitchen island", "counter", "pantry", "tv stand",

    # === (Furniture - Seating) ===
    "chair", "armchair", "sofa", "couch", "futon", "stool", "bar stool", 
    "bench", "ottoman", "recliner", "rocking chair", "highchair", 
    "beanbag", "cushion", "pillow",

    # === (Appliances - Major) ===
    "refrigerator", "freezer", "oven", "microwave", "dishwasher", 
    "washing machine", "dryer", "stove", "water heater", "air conditioner", 
    "radiator", "heater", "vacuum cleaner",

    # === (Appliances & Electronics - Minor) ===
    "television", "monitor", "computer", "laptop", "tablet", "keyboard", 
    "mouse", "router", "speaker", "stereo", "telephone", "camera", 
    "printer", "scanner", "fan", "ceiling fan", "toaster", "blender", 
    "coffee maker", "kettle", "iron", "hair dryer", "scale", "clock",

    # === (Kitchen & Dining - Manipulation Targets) ===
    "plate", "bowl", "cup", "mug", "glass", "wine glass", "bottle", "jug", 
    "pitcher", "pot", "pan", "skillet", "tray", "cutting board", "knife", 
    "fork", "spoon", "spatula", "whisk", "can opener", "saltshaker", 
    "pepper shaker", "napkin", "paper towel", "trash can", "bin",

    # === (Bathroom & Hygiene) ===
    "toilet", "urinal", "sink", "bathtub", "shower", "shower head", 
    "towel", "bath towel", "hand towel", "washcloth", "soap", "soap dispenser", 
    "toothbrush", "toothpaste", "shampoo", "toilet paper", "tissue", 
    "mirror", "sponge", "plunger", "laundry basket",

    # === (Decor, Lighting & Textiles) ===
    "painting", "picture", "poster", "photograph", "frame", "sculpture", 
    "vase", "potted plant", "houseplant", "flower", "lamp", "desk lamp", 
    "floor lamp", "chandelier", "sconce", "lampshade", "candle", "candlestick", 
    "curtain", "blind", "drape", "rug", "carpet", "mat", "blanket", "quilt",

    # === (Daily Necessities & Office) ===
    "book", "magazine", "newspaper", "notebook", "folder", "binder", 
    "pen", "pencil", "marker", "eraser", "scissors", "stapler", "tape", 
    "box", "carton", "crate", "basket", "bag", "backpack", "handbag", 
    "suitcase", "briefcase", "purse", "wallet", "keys", "umbrella", 
    "coat rack", "hanger",

    # === (Apparel & Personal) ===
    "clothing", "shirt", "t-shirt", "sweater", "jacket", "coat", "pants", 
    "jeans", "shorts", "skirt", "dress", "suit", "shoe", "boot", "sneaker", 
    "slipper", "sock", "hat", "cap", "glove", "scarf", "tie", "belt", "glasses",

    # === (Structural Landmarks & Hardware) ===
    "door", "window", "stairs", "step", 
    "bannister", "handrail", "column", "pillar", "fireplace", "mantel", 
    "doorknob", "door handle", "drawer pull", "light switch", "power outlet", 
    "faucet", "pipe", "vent", "thermostat", "smoke detector", "sign", "whiteboard"
]

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def save_video(frames, filename="pano.mp4", fps=5):
    if not frames: return
    frames = [Image.fromarray(img) for img in frames]
    w, h = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    for img in frames:
        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out.write(open_cv_image)
    out.release()
    print(f"Video saved to {os.path.abspath(filename)}")

# =============================================================================
# 2. Segmenter, Target Miner
# =============================================================================


class GlobalObjectMiner:
    def __init__(self):
        self.model = YOLOWorld('pretrained_models/yolov8l-worldv2.pt') 
        self.target_names = LVIS_INDOOR_VOCABULARY
        self.model.set_classes(self.target_names)
        
        self.detected_objects = defaultdict(list)
        
        self.min_area_threshold = 300
        
    def process_frame(self, rgb_numpy, depth_numpy, pose_matrix, intrinsic_matrix):
        h, w = rgb_numpy.shape[:2]
        results = self.model.predict(rgb_numpy, verbose=False)
        boxes = results[0].boxes
        
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = self.target_names[class_id]
            conf = float(box.conf[0])
            
            # --- low confidence for diverse objects ---
            if conf < 0.08:  
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            
            if area < self.min_area_threshold: 
                continue
            
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            d_patch = depth_numpy[max(0, cy-2):min(h, cy+3), max(0, cx-2):min(w, cx+3)]
            valid_d = d_patch[d_patch > 0.1]
            if len(valid_d) == 0: continue
            z_depth = np.median(valid_d)
            
            if z_depth > 12.0: continue 
            
            fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
            cx_cam, cy_cam = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
            
            x_local = (cx - cx_cam) * z_depth / fx
            y_local = (cy - cy_cam) * z_depth / fy
            
            point_local = np.array([x_local, y_local, z_depth, 1.0])
            point_world = (pose_matrix @ point_local)[:3]
            
            is_duplicate = False
            for existing_obj in self.detected_objects[class_name]:
                # Allow high-density clusters of homogeneous objects (e.g., a row of chairs)
                if np.linalg.norm(existing_obj - point_world) < 0.5: 
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                self.detected_objects[class_name].append(point_world)

    def get_summary(self):
        print(f"--- Object Mining Summary ---")
        for cls, objs in self.detected_objects.items():
            print(f"Found {len(objs)} {cls}(s)")
        return self.detected_objects
        
        
    def clear(self):
        self.detected_objects.clear()

# =============================================================================
# 3. Long Range Generator / Smart Generator)
# =============================================================================

class LongRangeTaskGenerator:
    """
    Unified Data Generator for Embodied Navigation and Mobile Manipulation.

    Introduced Features: Area-Adaptive Capacity, Trajectory Length Distribution Buckets, 
    Long-Tail Category Frequency Penalty, and Manipulation-Level Docking Thresholds.
    """
    def __init__(self, simulator):
        self.sim = simulator
        self.nav_graph = None 
        self.history_starts = []
        self.history_ends = []
        
        # Global category sampling frequency statistics (utilized for long-tail mining)
        self.category_sample_counts = defaultdict(int)
        
        # Trajectory length distribution bucket configuration: (min_dist, max_dist, probability)
        # Ensures an equitable distribution across short-, medium-, and long-range tasks, 
        # preventing the model from overfitting to specific path scales.
        self.length_buckets = [
            (3, 6, 0.20),  # Short-range (20%): Tailored for mobile manipulation and localized instructions
            (6, 10, 0.30),  # Medium-range (30%): Cross-room navigation
            (10, 15, 0.50)  # Long-range (50%): Long-sequence instructions and environmental exploration
        ]
        
        # Define Mobile Manipulation targets requiring precision docking
        self.manipulation_targets = {
            "doorknob", "door handle", "drawer pull", "light switch", "power outlet",
            "faucet", "thermostat",

            "soap", "soap dispenser", "toothbrush", "toothpaste", "shampoo",
            "toilet paper", "tissue", "sponge", "plunger",

            "plate", "bowl", "cup", "mug", "glass", "wine glass", "bottle",
            "jug", "pitcher", "pot", "pan", "skillet", "tray", "cutting board",
            "knife", "fork", "spoon", "spatula", "whisk", "can opener",
            "saltshaker", "pepper shaker", "napkin",

            "laptop", "tablet", "keyboard", "mouse", "telephone", "camera",
            "router", "speaker", "clock",

            "book", "notebook", "folder", "binder", "pen", "pencil", "marker",
            "eraser", "scissors", "stapler", "tape", "wallet", "keys",
            "glasses",

            "shoe", "boot", "sneaker", "slipper", "sock", "hat", "cap",
            "glove", "scarf", "tie", "belt",

            "box", "carton", "basket", "bag", "handbag", "purse"
        }
        self.instruction_annotator = InstructionAnnotator(model_id="pretrained_models/models--Qwen--Qwen3-VL-32B-Instruct", device=self.sim.device)

    def reset_history(self):
        self.history_starts = []
        self.history_ends = []

    def build_networkx_graph(self):
        if self.sim.planner_graph is None: 
            print("Warning: Planner graph not found in simulator.")
            return
        
        print("Building NetworkX graph for geodesic distance calculation...")
        sparse_dist_graph = self.sim.planner_graph['sparse_dist_graph']
        idx_to_point = self.sim.planner_graph['idx_to_point']
        
        try:
            self.nav_graph = nx.from_scipy_sparse_array(sparse_dist_graph)
        except AttributeError:
            self.nav_graph = nx.from_scipy_sparse_matrix(sparse_dist_graph)
        
        nx.relabel_nodes(self.nav_graph, idx_to_point, copy=False)
        self.node_list = list(self.nav_graph.nodes())
        print(f"Graph built with {self.nav_graph.number_of_nodes()} nodes.")


    def calculate_scene_capacity(self, density_factor=5):
        """
        Dynamically computes the sampling upper bound based on the scene's actual navigable area 
        to prevent topological redundancy in small rooms.
        """
        if self.sim.planner_graph is None: return 0
        
        num_nodes = self.sim.planner_graph['sparse_dist_graph'].shape[0]
        voxel_size = self.sim.planner_graph['grid_size']
        
        # Estimate the navigable area (in square meters)
        navigable_area = num_nodes * (voxel_size ** 2)
        
        # Assume each square meter supports a maximum of 'density_factor' trajectories 
        # (e.g., density_factor trajectories per square meter)
        capacity = int(navigable_area * density_factor)
        
        # Enforce reasonable lower and upper bounds (minimum 50, maximum 1000 trajectories)
        return max(50, min(capacity, 1000))

    def _get_furthest_candidates(self, candidates_pts, history_pts, top_k=3):
        if not history_pts:
            indices = np.arange(len(candidates_pts))
            np.random.shuffle(indices)
            return indices[:top_k]
            
        cand_arr = np.array(candidates_pts)
        hist_arr = np.array(history_pts)
        dists = np.linalg.norm(cand_arr[:, None, :] - hist_arr[None, :, :], axis=2)
        min_dists_to_hist = np.min(dists, axis=1)
        furthest_indices = np.argsort(min_dists_to_hist)[-top_k:]
        return furthest_indices

    def sample_bucketed_task(self, scale=1., max_trials=20):
        """Targeted sampling based on the trajectory length bucket distribution."""
        if self.sim.planner_graph is None: 
            return None, None, 0.0

        graph_data = self.sim.planner_graph
        dist_graph = graph_data['sparse_dist_graph']
        num_nodes = dist_graph.shape[0]
        all_indices = np.arange(num_nodes)

        # 1. Roulette wheel selection to determine the current length bucket
        r = random.random()
        cumulative = 0.0
        target_min, target_max = 1.5, 3.0
        for b_min, b_max, prob in self.length_buckets:
            cumulative += prob
            if r <= cumulative:
                target_min, target_max = b_min, b_max
                break

        target_min *= scale
        target_max *= scale

        # 2. Attempt to sample start and goal points that satisfy the selected length bucket
        for _ in range(max_trials):
            cand_start_indices = np.random.choice(all_indices, size=min(100, num_nodes), replace=False)
            cand_start_pts = [graph_data['idx_to_point'][i] for i in cand_start_indices]
            
            best_start_local_idxs = self._get_furthest_candidates(cand_start_pts, self.history_starts, top_k=5)
            start_idx = cand_start_indices[random.choice(best_start_local_idxs)]
            start_node = graph_data['idx_to_point'][start_idx]
            
            # Use 'target_max' as the Dijkstra distance limit to dramatically accelerate computation
            distances = dijkstra(csgraph=dist_graph, directed=False, indices=start_idx, limit=target_max + 1.0)
            
            # Filter for goal points that fall strictly within the [target_min, target_max] range
            valid_indices = np.where((distances >= target_min) & (distances <= target_max))[0]
            
            if len(valid_indices) == 0:
                continue 
            
            valid_end_pts = [graph_data['idx_to_point'][i] for i in valid_indices]
            best_end_local_idxs = self._get_furthest_candidates(valid_end_pts, self.history_ends, top_k=3)
            
            e_idx = valid_indices[random.choice(best_end_local_idxs)]
            end_node = graph_data['idx_to_point'][e_idx]
            dist = distances[e_idx]
            
            self.history_starts.append(np.array(start_node))
            self.history_ends.append(np.array(end_node))
                
            return np.array(start_node), np.array(end_node), dist

        # Fallback: Revert to uniform valid sampling if the selected bucket fails to sample
        return self._fallback_sample(all_indices, dist_graph, graph_data)

    def _fallback_sample(self, all_indices, dist_graph, graph_data):
        start_idx = np.random.choice(all_indices)
        distances = dijkstra(csgraph=dist_graph, directed=False, indices=start_idx, limit=10.0)
        valid_indices = np.where((distances >= 1.5) & (distances != np.inf))[0]
        if len(valid_indices) > 0:
            e_idx = np.random.choice(valid_indices)
            return np.array(graph_data['idx_to_point'][start_idx]), np.array(graph_data['idx_to_point'][e_idx]), distances[e_idx]
        return None, None, 0.0

    def _scan_objects_at_location(self, miner, location):
        """
        Adaptive Docking Thresholds + Geodesic Verification + IDF Category Balancing Weights
        """
        object_candidates = []
        graph_data = self.sim.planner_graph
        kdtree = graph_data['kdtree']
        dist_graph = graph_data['sparse_dist_graph']
        
        _, agent_node_idx_t = kdtree.query(torch.tensor(location, device=self.sim.device, dtype=torch.float32).unsqueeze(0), nr_nns_searches=1)
        
        geo_distances = dijkstra(csgraph=dist_graph, directed=False, indices=agent_node_idx_t.item(), limit=4.0)
        
        for cls_name, coords_list in miner.detected_objects.items():
            max_dist = 1.5 if cls_name in self.manipulation_targets else 3.0
            min_dist = 0.5 if cls_name in self.manipulation_targets else 0.3
            
            for obj_pos in coords_list:
                euclidean_dist = np.linalg.norm(obj_pos - location)
                if euclidean_dist > max_dist or euclidean_dist < min_dist:
                    continue 
                
                _, obj_node_idx_t = kdtree.query(torch.tensor(obj_pos, device=self.sim.device, dtype=torch.float32).unsqueeze(0), nr_nns_searches=1)
                geo_dist = geo_distances[obj_node_idx_t.item()]
                
                if geo_dist == np.inf or geo_dist > (euclidean_dist * 1.5):
                    continue
                    
                object_candidates.append((cls_name, euclidean_dist))
                
        if not object_candidates:
            return "a specific place in the room"
            
        unique_classes_dict = {}
        for cls_name, dist in object_candidates:
            if cls_name in ["wall", "floor", "ceiling"]: continue
            if cls_name not in unique_classes_dict or dist < unique_classes_dict[cls_name]:
                unique_classes_dict[cls_name] = dist
                
        if not unique_classes_dict:
            return "a specific place in the room"

        # --- IDF-Weighted Randomized Sampling ---
        candidates = list(unique_classes_dict.keys())
        weights = []
        for cls in candidates:
            # Higher-frequency categories receive lower sampling weights (offset by +1 to prevent division-by-zero)
            weight = 1.0 / (self.category_sample_counts[cls] + 1)
            weights.append(weight)
            
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        selected_target = np.random.choice(candidates, p=probs)
        
        self.category_sample_counts[selected_target] += 1
        
        return selected_target

    def generate_episode(self, object_miner):
        style_choice = random.choice(['r2r', 'reverie', 'demand'])
        if style_choice == 'r2r':
            length_scale = 1.0
        elif style_choice == 'reverie': # shorter path
            length_scale = 0.5
        elif style_choice == 'demand': # shorter path
            length_scale = 0.5
        else:
            length_scale = 1.0

        style_prompt = get_dynamic_prompt(style_choice)
        start_pos, end_pos, geo_dist = self.sample_bucketed_task(scale=length_scale)
        if start_pos is None: return None

        start_pos_safe, end_pos_safe = start_pos.copy(), end_pos.copy()
        start_pos_safe[2] += 0.1 
        end_pos_safe[2] += 0.1

        try:
            paths, _ = self.sim.get_shortest_paths_to(
                target_positions=np.array([end_pos_safe]),
                start_positions=np.array([start_pos_safe])
            )
        except Exception as e:
            print("\n[Path Planning Error]")
            traceback.print_exc()
            return None

        if not paths or not paths[0] or len(paths[0]) < 2: return None
        waypoints = np.array(paths[0])

        target_name = self._scan_objects_at_location(object_miner, end_pos)
        
        episode_data = self.generate_data(start_pos, waypoints, style_choice, style_prompt, target_name) # 注意：需要在 sim 中暴露 _simulate_agent
        return episode_data
    

    def generate_data(self, start_pos, waypoints, style_choice, style_prompt, target_name):

        end_pos_safe = np.array(waypoints[-1]).copy()
        end_pos_safe[2] += 0.1
        target_positions = np.array([end_pos_safe])
        
        current_feet_pos = np.array(start_pos, dtype=np.float64)
        if len(waypoints) > 1:
            vec = waypoints[1] - waypoints[0]
            # Utilize compass heading !!!!
            current_heading = (np.pi / 2) - np.arctan2(vec[1], vec[0])
            current_heading = (current_heading + np.pi) % (2 * np.pi) - np.pi
        else:
            current_heading = 0.0
            
        camera_start_pos = current_feet_pos.copy()
        camera_start_pos[2] += self.sim.eye_height 
        
        self.sim.agent_pos = torch.tensor([camera_start_pos], device=self.sim.device, dtype=torch.float32)
        self.sim.agent_heading = torch.tensor([current_heading], device=self.sim.device, dtype=torch.float32)
        
        episode_data = {"frames": [], "trajectory": []}
        wide_frames_for_vlm = [] 
        
        action_sequence = self.sim.generate_action_sequence(target_positions=target_positions)
        if isinstance(action_sequence[0], list):
            action_sequence = action_sequence[0]

        state_window = []

        for step_count, action_type in enumerate(action_sequence):
            
            curr_pos_t = self.sim.agent_pos.clone()
            curr_heading_t = self.sim.agent_heading.clone()  

            state_window.append({
                'pos': curr_pos_t[0, :2].clone(),
                'heading': curr_heading_t[0].item()
            })
            
            if len(state_window) > 4:
                state_window.pop(0)
                
            if len(state_window) == 4:
                start_state = state_window[0]
                end_state = state_window[-1]
                
                disp = torch.norm(end_state['pos'] - start_state['pos']).item()
                
                heading_diff = abs((end_state['heading'] - start_state['heading'] + math.pi) % (2*math.pi) - math.pi)
                
                if disp <= 2 * self.sim.planning_voxel_size and heading_diff <= self.sim.turn_angle_rad:
                    print(f"  [Simulator] Agent stuck (moved <10cm in past 4 steps) at step {step_count}. Truncating.")
                    
                    if len(episode_data['trajectory']) > 0:
                        episode_data['trajectory'][-1]['action'] = "STOP"
                        
                    break
            # ==========================================================
            
            pano_rgb, pano_depth = self.sim.get_panorama_observation(
                position=curr_pos_t, heading=curr_heading_t
            )

            wide_img = pano_rgb[0, 88:424, 256:768].cpu().numpy()
            wide_frames_for_vlm.append(wide_img)
            
            log_feet = curr_pos_t[0].cpu().numpy()
            log_feet[2] -= self.sim.eye_height
            episode_data['trajectory'].append({
                "step": step_count,
                "pos": log_feet.tolist(),
                "heading": float(curr_heading_t[0].item()),
                "action": action_type
            })
            
            episode_data['frames'].append(wide_img)
            
            if action_type == "STOP":
                break
                
            self.sim.step([image2sim.Action[action_type]], render_observation=False)

        # ==========================================================
        # Discard short path
        # ==========================================================
        if len(episode_data['frames']) < 5:
            print("  [Simulator] ⚠️ Trajectory too short after truncation. Discarding.")
            return None

        if getattr(self, 'instruction_annotator', None) is not None:
            vlm_result = self.instruction_annotator.generate_instructions(wide_frames_for_vlm, episode_data['trajectory'], style_choice, style_prompt)
            
            if vlm_result is None:
                print("  [Data Gen] Episode discarded by VLM QA.")
                return None 
            else:
                episode_data['instruction_data'] = vlm_result
                print(f"  [VLM Output | Style: {vlm_result['style']}]: {vlm_result['instruction']}")

        return episode_data

# =============================================================================
# 4. Main Execution
# =============================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Image2Sim VLN Data Generation")
    parser.add_argument("--group_num", type=int, default=1, help="Total shards")
    parser.add_argument("--group_id", type=int, default=0, help="Current shard ID, 0 to group_num-1")
    parser.add_argument("--save_video", type=bool, default=False, help="save the video")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"[INIT] Running Worker Node: Group {args.group_id} / {args.group_num}")
    print(f"{'='*60}\n")

    # 1. Configure
    config = type('Config', (), {'image_height': 512, 'batch_size': 1, 'max_depth': 10.})
    checkpoint_path = "checkpoints"
    max_batch_size = config.batch_size 
    device = "cuda"
    
    # 2. dataset path
    # (dataset_type, image data path, pcd path)
    dataset_sources = {
        'matterport':        ("matterport", "data/scene_datasets/Matterport3D_360/data", "data/nav_map/mp3d"),
        'realsee_synthetic': ("realsee", "data/scene_datasets/RealSee3D/synthetic_data", "data/nav_map/RealSee3D_synthetic"),
        'realsee_real':      ("realsee", "data/scene_datasets/RealSee3D/real_world_data", "data/nav_map/RealSee3D_real"),
        'structured3d':      ("structured3d", "data/scene_datasets/Structured3D", "data/nav_map/Structured3D"),
        'hm3d':              ("hm3d", "data/scene_datasets/hm3d_360", "data/nav_map/hm3d"),
        'gibson':              ("gibson", "data/scene_datasets/gibson_360", "data/nav_map/gibson")
    }

    print(">>> Initializing Neural Simulator...")
    neural_simulator = image2sim.NeuralSimulator(config).to("cuda")
    neural_simulator = data_tools.load_checkpoint(neural_simulator, "pretrained_models")
    neural_simulator.torch_compile()
    neural_simulator.eval()

    print(">>> Initializing Components...")
    object_miner = GlobalObjectMiner() 
    long_range_gen = LongRangeTaskGenerator(neural_simulator)

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
        scene_names.sort()
        print(f"Found {len(scene_names)} scenes in {source_key}.")
        
        with torch.amp.autocast(device_type='cuda'):
            with torch.no_grad():
                for scene_id, scene_name in enumerate(scene_names):
                    if scene_id % args.group_num != args.group_id:
                        continue

                    scene_json_name = f"output_{dataset_type}_{scene_name}.json"
                    if os.path.exists(scene_json_name):
                        print(f"\n>>> [Skip] Scene {scene_name} already processed (found {scene_json_name}). Skipping...")
                        continue
                    
                    print(f"\n{'='*50}")
                    print(f"Processing Scene: {scene_name} ({source_key})")
                    print(f"{'='*50}")
            
                    full_pcd_path = os.path.join(SCENES_DIR, f"{scene_name}.pcd")
                    nav_pcd_path = os.path.join(SCENES_DIR, f"{scene_name}_navigable.pcd")
                    if not os.path.exists(nav_pcd_path):
                        continue
            
                    try:
                        scene_path = os.path.join(IMAGES_DIR, scene_name)
                        if os.path.exists(full_pcd_path) and 'matterport' not in dataset_type:
                            scene_pcd = o3d.io.read_point_cloud(full_pcd_path)
                        else:
                            scene_pcd = None
                        nav_pcd = o3d.io.read_point_cloud(nav_pcd_path)
        
                        (scene_xyz, scene_rgb, scene_feats, scene_gs, all_frames_data) = data_tools.build_scene_pointcloud_data(
                            scene_path, 
                            dataset_type=dataset_type, 
                            device=device,
                            voxel_size=0.005,
                            model=neural_simulator, 
                            max_batch_size=1000,
                            inpaint_depth=True
                        )
                        
                        if scene_xyz is None:
                            print(f"Skipping scene {scene_name}: Failed to build point cloud data.")
                            continue
                            
                        neural_simulator.import_scene_gaussian(xyz=scene_xyz, rgb=scene_rgb, feats=scene_feats, gs_attrs=scene_gs)
                        neural_simulator.load_navigable_pcd(nav_pcd, scene_pcd)
                        
                        long_range_gen.build_networkx_graph()
                        num_episodes_per_scene = long_range_gen.calculate_scene_capacity()
                        
                    except Exception as e:
                        print(f"Error loading scene {scene_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
                    torch.cuda.empty_cache()
                    print(">>> Phase 1: Scanning for objects...")
                    object_miner.clear()
                    
                    nav_points_arr = np.asarray(nav_pcd.points)
                    scan_indices = np.random.choice(len(nav_points_arr), size=min(num_episodes_per_scene, len(nav_points_arr)), replace=False)
                    
                    for idx in tqdm(scan_indices, desc="Mining"):
                        pos = nav_points_arr[idx]
                        for heading in [0, np.pi/2, np.pi, 3*np.pi/2]:
                            pos_t = torch.tensor(pos, device=neural_simulator.device).float().unsqueeze(0)
                            pos_t[:, 2] += neural_simulator.eye_height
                            head_t = torch.tensor([heading], device=neural_simulator.device).float()
                            
                            rgb_t, depth_t = neural_simulator.get_agent_observation(pos_t, head_t)
                            
                            rgb_np = rgb_t.cpu().numpy()[0]
                            depth_np = depth_t.cpu().numpy()[0]
                            
                            cos_h, sin_h = np.cos(heading), np.sin(heading)
                            pose = np.eye(4)
                            pose[:3, :3] = np.array([
                                [cos_h,   0,  sin_h],
                                [-sin_h,  0,  cos_h],
                                [0,      -1,      0]
                            ])
                            pose[:3, 3] = pos + np.array([0, 0, neural_simulator.eye_height])
                            
                            h_sim, w_sim = rgb_np.shape[:2]
                            approx_intrinsic = np.array([
                                [w_sim/2, 0, w_sim/2],
                                [0, h_sim/2, h_sim/2],
                                [0, 0, 1]
                            ])
                            object_miner.process_frame(rgb_np, depth_np, pose, approx_intrinsic)
            
                    object_miner.get_summary()
                    torch.cuda.empty_cache()

                    print(f"\n>>> Phase 2: Generating Long-Range Episodes for {scene_name}")
                    long_range_gen.reset_history()
                    
                    scene_episodes = []
                    
                    for i in range(num_episodes_per_scene):
                        print(f"Generating episode {i+1}/{num_episodes_per_scene}...")
                        
                        episode = long_range_gen.generate_episode(object_miner)
                        
                        if episode and 'instruction_data' in episode:
                            if args.save_video:
                                video_name = f"output_{dataset_type}_{scene_name}_long_{i}.mp4"
                                save_video(episode['frames'], filename=video_name, fps=10)
                            
                            traj_data = {
                                "episode_id": f"{scene_name}_long_{i}",
                                "instruction_data": episode['instruction_data'],
                                "trajectory": episode['trajectory']
                            }
                            
                            scene_episodes.append(traj_data)
                    
                    if scene_episodes:
                        scene_json_name = f"output_{dataset_type}_{scene_name}.json"
                        with open(scene_json_name, 'w', encoding='utf-8') as f:
                            json.dump(scene_episodes, f, indent=4, ensure_ascii=False)
                        print(f"\n>>> [Scene Completed] Saved {len(scene_episodes)} episodes to {scene_json_name}")
                    else:
                        print(f"\n>>> [Scene Skipped] No valid episodes generated for {scene_name}.")

    print("\n All Datasets Processed Successfully.")