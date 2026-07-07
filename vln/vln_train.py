import os
import datetime

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "7200"
os.environ["NCCL_TIMEOUT"] = "7200"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

import torch.distributed as dist
_orig_pt_init = dist.init_process_group
def _patched_pt_init(*args, **kwargs):
    kwargs['timeout'] = datetime.timedelta(seconds=7200)
    print("\n🚀 [Monkey Patch] PyTorch process group timeout forced to 7200s!")
    return _orig_pt_init(*args, **kwargs)
dist.init_process_group = _patched_pt_init

try:
    import deepspeed.comm as dist_comm
    import deepspeed
    _orig_ds_init = dist_comm.init_distributed
    
    def _patched_ds_init(*args, **kwargs):
        kwargs['timeout'] = datetime.timedelta(seconds=7200)
        print("\n🚀 [Monkey Patch] DeepSpeed process group timeout forced to 7200s!")
        return _orig_ds_init(*args, **kwargs)
        
    dist_comm.init_distributed = _patched_ds_init
    deepspeed.init_distributed = _patched_ds_init
except ImportError:
    pass

import os
import json
import math
import glob
import copy
import random
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import torch.distributed as dist
import numpy as np
import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset, Sampler
from transformers import Trainer, AutoProcessor
import re
from vln.model.vln_model import VLNForCausalLM

# Simulator related
import image2sim
import data_tools
import open3d as o3d

IGNORE_INDEX = -100


ACTION_IDX_TO_TOKEN = {
    0: "STOP",
    1: "↑",
    2: "←",
    3: "→",
}

ROUND_PREFIXES = [
    "At step {obs_idx}, you can see",
    "Observation at step {obs_idx}: in front of you is",
    "By step {obs_idx}, there is",
    "Current view at step {obs_idx}: you can spot",
    "Reaching step {obs_idx}, ahead of you is",
    "At step {obs_idx}, in your sight is",
]

def get_round_prefix(round_idx: int, obs_idx: int) -> str:
    template = ROUND_PREFIXES[round_idx % len(ROUND_PREFIXES)]
    return template.format(obs_idx=obs_idx)

def actions_to_text(actions: List[int]) -> str:
    return "".join(ACTION_IDX_TO_TOKEN[a] for a in actions)


# ==============================================================================
# 1. Arguments
# ==============================================================================

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-VL-4B-Instruct")
    tune_mm_llm: bool = field(default=True)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    simulator_ckpt: str = field(default="pretrained_models")
    attn_implementation: str = field(default="flash_attention_2")


@dataclass
class DataArguments:
    dataset_config: str = field(
        default="dataset_config.json",
        metadata={"help": "Path to multi-source dataset JSON config."},
    )
    num_train_observation_frames: int = field(
        default=32,
        metadata={"help": "Total number of uniformly sampled observation rounds used in each training sample."},
    )
    num_future_steps: int = field(
        default=2,
        metadata={"help": "Number of actions predicted for each observation round."},
    )
    remove_init_turns: bool = field(
        default=False,
        metadata={"help": "Whether to skip initial pure-rotation actions when building training anchors."},
    )

    batch_size: int = field(default=1)
    scene_voxel_size: float = field(default=0.005)

    image_width: int = field(default=336)
    image_height: int = field(default=336)
    image_hfov: float = field(default=90.0)
    image_vfov: float = field(default=90.0)
    max_depth: float = field(default=10.0)

    step_size: float = field(default=0.25)
    eye_height: float = field(default=1.25)
    turn_angle: float = field(default=15.0)
    max_step_height: float = field(default=0.15)
    agent_radius: float = field(default=0.15)
    goal_threshold: float = field(default=0.5)

    # === NEW: Sampling Control Arguments ===
    max_train_scenes: int = field(
        default=200000,
        metadata={"help": "Total number of scenes to build for this dataset."}
    )
    sample_ratios: str = field(
        default='{\"image2sim_batch_1\": 10, \"image2sim_batch_2\": 10, \"image2sim_batch_3\": 10, \"room_grounding\": 1, \"house_grounding\": 1, \"r2r\": 1, \"reverie\": 1, \"rxr\": 1, \"srdf\": 5}', 
        metadata={"help": "JSON string for instruction dir sampling probabilities, e.g. '{\"r2r\": 1, \"rxr\": 1}'. Unspecified types will default to weight 1"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=32 * 1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded."},
    )
    ddp_timeout: int = field(
        default=7200, 
        metadata={"help": "Timeout for NCCL collective operations (in seconds)."}
    )


# ==============================================================================
# 2. Dataset
# ==============================================================================

class Qwen3VLNOnlineDataset(Dataset):
    def __init__(self, data_args: DataArguments, processor: transformers.ProcessorMixin):
        super().__init__()
        self.data_args = data_args
        self.processor = processor
        self.tokenizer = processor.tokenizer

        self.num_sampled_observation_frames = data_args.num_train_observation_frames
        self.num_future_steps = data_args.num_future_steps
        self.remove_init_turns = data_args.remove_init_turns

        self.action_str2idx = {
            "STOP": 0,
            "MOVE_FORWARD": 1,
            "TURN_LEFT": 2,
            "TURN_RIGHT": 3,
        }

        print(f"Loading multi-source config from {data_args.dataset_config}...")
        with open(data_args.dataset_config, "r", encoding="utf-8") as f:
            self.dataset_sources = json.load(f)

        self.file_index: List[Dict[str, Any]] = []
        self.anchor_index: List[Tuple[int, int, int]] = []  # (ep_id, valid_obs_start, anchor_obs_idx)

        self._build_dataset_index()

        print(
            f"[Dataset] Scanned dataset and built {len(self.anchor_index)} training anchors via Lazy Loading."
        )

    def _clean_initial_rotations(self, actions: List[int]) -> int:
        idx = 0
        while idx < len(actions) and actions[idx] in (2, 3):
            idx += 1
        return idx

    def _build_dataset_index(self) -> None:
        import os
        import glob
        import json
        import random
        import re
        import concurrent.futures
        from collections import defaultdict
        import torch.distributed as dist
        
        self.file_index = []
        self.anchor_index = []
        visited_dirs = set()
        
        try:
            target_ratios = json.loads(self.data_args.sample_ratios)
        except Exception:
            target_ratios = {}

        # =====================================================================
        # Pre-cache all scenes
        # =====================================================================
        print("[Dataset] Pre-caching valid scenes from all images_dirs to memory...")
        valid_scenes_cache = {}
        for src_key, src_info in self.dataset_sources.items():
            img_dir = src_info.get("images_dir", "")
            if img_dir and os.path.exists(img_dir):
                valid_scenes_cache[src_key] = {
                    "scenes": set(os.listdir(img_dir)),
                    "info": src_info
                }
        print(f"[Dataset] Cached {len(valid_scenes_cache)} valid source directories.")

        # =====================================================================
        # Scan all json files to sample the navigation training data
        # =====================================================================
        for source_key, source_info in self.dataset_sources.items():
            traj_dir = source_info.get("traj_dir")
            if not traj_dir or not os.path.exists(traj_dir) or traj_dir in visited_dirs:
                continue
            visited_dirs.add(traj_dir)

            instr_types = [d for d in os.listdir(traj_dir) if os.path.isdir(os.path.join(traj_dir, d))]
            instr_type_to_files = {}
            for itype in instr_types:
                files = glob.glob(os.path.join(traj_dir, itype, "*.json"))
                if files:
                    instr_type_to_files[itype] = files

            if not instr_type_to_files:
                continue

            available_types = list(instr_type_to_files.keys())
            weights = [target_ratios.get(t, 1.0) for t in available_types]

            sampled_files = []
            max_scenes = self.data_args.max_train_scenes
            for _ in range(max_scenes):
                chosen_type = random.choices(available_types, weights=weights, k=1)[0]
                chosen_file = random.choice(instr_type_to_files[chosen_type])
                sampled_files.append(chosen_file)


            print(f"Multi-threading fast regex parse for {len(sampled_files)} files ...")
            
            EP_PATTERN = re.compile(r'"episode_id"\s*:\s*"?([^",\s}]+)"?')

            def _process_single_file(jf: str):
                # 1. scene name
                scene_name = jf.split("/")[-1].replace(".json","").replace("output_","")
                scene_name = scene_name.replace(scene_name.split("_")[0]+"_","")

                # 2. O(1) check
                true_source_info = None
                for cache_data in valid_scenes_cache.values():
                    if scene_name in cache_data["scenes"]:
                        true_source_info = cache_data["info"]
                        break
                
                if true_source_info is None:
                    return None

                results = []
                try:
                    # read text file without json.load for high speed
                    with open(jf, "r", encoding="utf-8") as f:
                        text_content = f.read()
                except Exception:
                    return None

                # 3. search all episode_id
                ep_matches = EP_PATTERN.findall(text_content)
                random.shuffle(ep_matches)
                max_sample_per_file = 256
                ep_matches = ep_matches[:max_sample_per_file]

                is_list_json = text_content.lstrip().startswith('[')
                
                if is_list_json:
                    if ep_matches:
                        for idx, ep_id_str in enumerate(ep_matches):
                            results.append({
                                "file_path": jf,
                                "source_info": true_source_info,
                                "scene_name": scene_name,
                                "episode_str": f"{jf.split('/')[-1]}_{ep_id_str}",
                                "list_index": idx
                            })
                    else:
                        list_count = text_content.count('"trajectory"') 
                        for idx in range(max(1, list_count)):
                            results.append({
                                "file_path": jf,
                                "source_info": true_source_info,
                                "scene_name": scene_name,
                                "episode_str": f"{jf.split('/')[-1]}_{idx}",
                                "list_index": idx
                            })
                else:
                    ep_id_str = ep_matches[0] if ep_matches else "0"
                    results.append({
                        "file_path": jf,
                        "source_info": true_source_info,
                        "scene_name": scene_name,
                        "episode_str": f"{jf.split('/')[-1]}_{ep_id_str}",
                        "list_index": 0
                    })
                    
                return results

            valid_results = []
            max_workers = min(64, (os.cpu_count() or 1) * 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = executor.map(_process_single_file, sampled_files)
                for res_list in results:
                    if res_list is not None:
                        valid_results.extend(res_list) # 展平 list

            for res in valid_results:
                ep_id = len(self.file_index)
                self.file_index.append(res)
                self.anchor_index.append(ep_id) 

        # =====================================================================
        # Global Synchronous Blocks
        # =====================================================================
        print("[Dataset] Grouping trajectories into strictly aligned blocks...")
        scene_to_ep_ids = defaultdict(list)
        for ep_id in self.anchor_index:
            meta = self.file_index[ep_id]
            bucket = meta["scene_name"]
            scene_to_ep_ids[bucket].append(ep_id)

        sim_bs = self.data_args.batch_size
        SYNC_BLOCK_SIZE = 32
        
        scene_buckets = list(scene_to_ep_ids.keys())
        random.shuffle(scene_buckets)
        
        blocks_by_scene = []
        for bucket in scene_buckets:
            ep_ids = scene_to_ep_ids[bucket].copy()
            random.shuffle(ep_ids)

            scene_blocks = []
            for i in range(0, len(ep_ids), SYNC_BLOCK_SIZE):
                block_eps = ep_ids[i : i + SYNC_BLOCK_SIZE]
                if len(block_eps) < SYNC_BLOCK_SIZE:
                    pad_size = SYNC_BLOCK_SIZE - len(block_eps)
                    block_eps.extend(random.choices(block_eps, k=pad_size))
                scene_blocks.append(block_eps)
                
            blocks_by_scene.append(scene_blocks)

        # =================================================================
        # Round-Robin
        # =================================================================
        all_blocks = []
        max_blocks_per_scene = max(len(blocks) for blocks in blocks_by_scene) if blocks_by_scene else 0
        
        for block_idx in range(max_blocks_per_scene):
            for scene_blocks in blocks_by_scene:
                if block_idx < len(scene_blocks):
                    all_blocks.append(scene_blocks[block_idx])
                
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        remainder = len(all_blocks) % self.world_size
        if remainder != 0:
            pad_blocks = self.world_size - remainder
            all_blocks.extend(all_blocks[:pad_blocks])

        self.prebuilt_batches = []
        for block in all_blocks:
            for j in range(0, len(block), sim_bs):
                self.prebuilt_batches.append(block[j : j + sim_bs])

        chunk_size_in_blocks = len(all_blocks) // self.world_size
        batches_per_block = SYNC_BLOCK_SIZE // sim_bs
        
        self.current_ptr = self.rank * (chunk_size_in_blocks * batches_per_block)
        
        print(f"[Dataset] Generated {len(all_blocks)} synchronous blocks (size={SYNC_BLOCK_SIZE}).")
        print(f"[Dataset] DDP Status -> Rank {self.rank} starts at pointer {self.current_ptr}.")
        
        self._cached_json_path = None
        self._cached_json_data = None


    def __len__(self) -> int:
        return len(self.prebuilt_batches)

    def __getitem__(self, dummy_index: int) -> List[Dict[str, Any]]:
        import json
        import numpy as np
        import traceback
        import copy
        import math
        max_retries = 20
        
        for attempt in range(max_retries):
            try:
                actual_idx = self.current_ptr % len(self.prebuilt_batches)

                self.current_ptr += 1
                
                batch_ep_ids = self.prebuilt_batches[actual_idx]
                batch_features = []
                
                for ep_id in batch_ep_ids:
                    meta = self.file_index[ep_id]
                    file_path = meta["file_path"]
                    
                    # =====================================================================
                    # IO cache
                    # =====================================================================
                    if getattr(self, "_cached_json_path", None) != file_path:
                        with open(file_path, "r", encoding="utf-8") as f:
                            self._cached_json_data = json.load(f)
                        self._cached_json_path = file_path
                        
                    data = self._cached_json_data

                    list_idx = meta.get("list_index", 0)
                    item = data[list_idx] if isinstance(data, list) else data
                    
                    trajectory = item.get("trajectory", [])
                    instruction = item.get("instruction_data", {}).get("instruction", "")[:2000]
                    source_info = meta["source_info"]
                    
                    # ========== random direction at step 0 ==========
                    if len(trajectory) > 0 and random.random() < 0.8:
                        
                        offset_steps = random.choice(list(range(-11,12)))
                        turn_angle_rad = math.radians(self.data_args.turn_angle)
                        correct_action_str = "TURN_RIGHT" if offset_steps < 0 else "TURN_LEFT"
                        num_turns = abs(offset_steps)
                        
                        prepend_traj = []
                        curr_heading = trajectory[0]["heading"] + offset_steps * turn_angle_rad
                        step_heading = turn_angle_rad if offset_steps < 0 else -turn_angle_rad
                        
                        for _ in range(num_turns):
                            fake_step = copy.deepcopy(trajectory[0])
                            fake_step["heading"] = curr_heading
                            fake_step["action"] = correct_action_str
                            prepend_traj.append(fake_step)
                            curr_heading += step_heading
                            
                        trajectory = prepend_traj + trajectory

                    actions_full = [self.action_str2idx.get(step.get("action", "STOP"), 0) for step in trajectory]
                    valid_obs_start = 0
                    anchor_obs_idx = len(trajectory) - 1
                    
                    # =====================================================================
                    # sample frames with little random
                    # =====================================================================
                    sampled_obs_indices = self._select_uniform_observation_indices(
                        valid_obs_start=valid_obs_start,
                        anchor_obs_idx=anchor_obs_idx,
                    ).astype(np.int32)
                    
                    if len(sampled_obs_indices) == 0:
                        sampled_obs_indices = np.array([anchor_obs_idx], dtype=np.int32)
                        
                    sampled_obs_indices_list = sampled_obs_indices.tolist()

                    positions = [trajectory[idx]["pos"] for idx in sampled_obs_indices_list]
                    headings = [trajectory[idx]["heading"] for idx in sampled_obs_indices_list]

                    # =====================================================================
                    # Prompt (Interleaved Messages)
                    # =====================================================================
                    messages = self._build_interleaved_messages(
                        sampled_obs_indices=sampled_obs_indices_list,
                        actions_full=actions_full,
                        instruction=instruction,
                    )

                    batch_features.append({
                        "scene_name": meta["scene_name"],
                        "dataset_type": source_info["dataset_type"],
                        "images_dir": source_info["images_dir"],
                        "scenes_dir": source_info["scenes_dir"],
                        "positions": positions,
                        "headings": headings,
                        "messages": messages,
                        "num_images": len(sampled_obs_indices_list),
                        "episode_id": meta["episode_str"],
                        "anchor_obs_idx": anchor_obs_idx,
                        "sampled_obs_indices": sampled_obs_indices_list,
                    })
                    
                return batch_features

            except Exception as e:
                print(f"[Dataset Warning] Error loading data at ptr {self.current_ptr-1}: {e}. Retrying...")

        raise RuntimeError(f"Failed to load a valid sample after {max_retries} attempts.")
        

    def _select_uniform_observation_indices(
        self,
        valid_obs_start: int,
        anchor_obs_idx: int,
    ) -> np.ndarray:
        
        F = self.num_future_steps          
        K = self.num_sampled_observation_frames 

        if F <= 0:
            raise ValueError(f"num_future_steps must be positive, got {F}")

        if anchor_obs_idx < valid_obs_start:
            return np.array([], dtype=np.int32)

        obs_pool = np.arange(valid_obs_start, anchor_obs_idx + 1, F)
        
        if len(obs_pool) == 0:
            return np.array([], dtype=np.int32)

        if len(obs_pool) <= K:
            return obs_pool.astype(np.int32)

        
        # retain (Index 0) and (Index -1)
        start_idx = 0
        end_idx = len(obs_pool) - 1
        
        base_middle_indices = np.linspace(1, end_idx - 1, num=K - 2)
        
        step = (end_idx - 1) / (K - 1)
        noise = np.random.uniform(-step * 0.4, step * 0.4, size=K - 2)
        
        jittered_middle = np.round(base_middle_indices + noise).astype(np.int32)
        
        jittered_middle = np.clip(jittered_middle, 1, end_idx - 1)
        jittered_middle = np.unique(jittered_middle)
        
        if len(jittered_middle) < K - 2:
            remaining_pool = np.setdiff1d(np.arange(1, end_idx), jittered_middle)
            needed = (K - 2) - len(jittered_middle)
            if needed > 0 and len(remaining_pool) >= needed:
                fillers = np.random.choice(remaining_pool, size=needed, replace=False)
                jittered_middle = np.sort(np.concatenate([jittered_middle, fillers]))
        
        final_indices = np.concatenate(([start_idx], jittered_middle, [end_idx]))
        
        sampled = obs_pool[final_indices]
        
        return sampled.astype(np.int32)

    def _pad_action_chunk(self, actions_full: List[int], obs_idx: int) -> List[int]:
        step_actions = actions_full[obs_idx: obs_idx + self.num_future_steps]
        if len(step_actions) < self.num_future_steps:
            step_actions = step_actions + [0] * (self.num_future_steps - len(step_actions))
        return step_actions


    def _build_interleaved_messages(
        self,
        sampled_obs_indices: List[int],
        actions_full: List[int],
        instruction: str,
    ) -> List[Dict[str, Any]]:
        
        system_prompt_with_goal = (
            "You are an autonomous navigation assistant. "
            f"Your ultimate goal is: {instruction}. "
            f"Based on the current observation and step history, output exactly {self.num_future_steps} actions to reach the goal. "
            "Actions must be chosen from: TURN LEFT (←), TURN RIGHT (→), MOVE FORWARD (↑), or STOP. "
            "Respond with actions only. Do not output any extra words, spaces, or punctuation."
        )
        
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt_with_goal}
        ]

        for round_idx, obs_idx in enumerate(sampled_obs_indices):
            prefix = get_round_prefix(round_idx, obs_idx)

            if round_idx > 0 and round_idx % 8 == 0:
                user_text = (
                    f"Reminder of your goal: '{instruction}'.\n"
                    f"{prefix}. Output the next {self.num_future_steps} actions."
                )
            else:
                user_text = f"{prefix}. Output the next {self.num_future_steps} actions."

            user_content = [
                {"type": "image", "image": "file://dummy"},
                {"type": "text", "text": user_text},
            ]
            messages.append({"role": "user", "content": user_content})

            action_chunk = self._pad_action_chunk(actions_full, obs_idx)
            assistant_text = actions_to_text(action_chunk)
            
            messages.append({"role": "assistant", "content": assistant_text})

        return messages


# ==============================================================================
# 4. Online collator with simulator rendering
# ==============================================================================

@dataclass
class DataCollatorWithOnlineSim:
    processor: transformers.ProcessorMixin
    simulator: Any
    scene_voxel_size: float = 0.005

    def __post_init__(self):
        self.current_scene_name = None
        self.current_dataset_type = None
        self.current_images_dir = None
        self.current_scenes_dir = None

    def load_scene_if_needed(self, scene_name: str, dataset_type: str, images_dir: str, scenes_dir: str):
        if (
            self.current_scene_name == scene_name
            and self.current_dataset_type == dataset_type
            and self.current_images_dir == images_dir
            and self.current_scenes_dir == scenes_dir
        ):
            return

        print(f"\n[Online Simulator] Swapping to scene: {scene_name} (Dataset: {dataset_type})...")

        full_pcd_path = os.path.join(scenes_dir, f"{scene_name}.pcd")
        nav_pcd_path = os.path.join(scenes_dir, f"{scene_name}_navigable.pcd")
        scene_path = os.path.join(images_dir, scene_name)

        scene_pcd = o3d.io.read_point_cloud(full_pcd_path) if os.path.exists(full_pcd_path) else None
        nav_pcd = o3d.io.read_point_cloud(nav_pcd_path) if os.path.exists(nav_pcd_path) else None

        (
            scene_xyz,
            scene_rgb,
            scene_feats,
            scene_gs,
            _,
        ) = data_tools.build_scene_pointcloud_data(
            scene_path=scene_path,
            dataset_type=dataset_type,
            device=self.simulator.device,
            voxel_size=self.scene_voxel_size,
            model=self.simulator,
            max_batch_size=1000,
            inpaint_depth=True
        )


        self.simulator.import_scene_gaussian(
            xyz=scene_xyz,
            rgb=scene_rgb,
            feats=scene_feats,
            gs_attrs=scene_gs,
        )
        self.simulator.load_navigable_pcd(nav_pcd=None, scene_pcd=scene_pcd) # Only load the scene_pcd to filter out the noise

        self.current_scene_name = scene_name
        self.current_dataset_type = dataset_type
        self.current_images_dir = images_dir
        self.current_scenes_dir = scenes_dir

    def _render_batch_images(
        self,
        positions: List[List[float]],
        headings: List[float],
    ) -> List[Image.Image]:
        if len(positions) == 0:
            return []

        all_rgb = []
        sim_bs = self.simulator.batch_size

        for i in range(0, len(positions), sim_bs):
            pos_chunk = positions[i : i + sim_bs]
            head_chunk = headings[i : i + sim_bs]
            actual_len = len(pos_chunk)

            while len(pos_chunk) < sim_bs:
                pos_chunk.append(pos_chunk[-1])
                head_chunk.append(head_chunk[-1])

            pos_tensor = torch.tensor(pos_chunk, dtype=torch.float32, device=self.simulator.device)
            heading_tensor = torch.tensor(head_chunk, dtype=torch.float32, device=self.simulator.device)

            with torch.no_grad():
                rgb, _ = self.simulator.get_agent_observation(
                    position=pos_tensor + torch.tensor([[0, 0, self.simulator.eye_height]], device=pos_tensor.device, dtype=torch.float32),
                    heading=heading_tensor
                )

            if torch.is_tensor(rgb):
                rgb = rgb.detach().float().cpu().numpy()
                if rgb.ndim == 4 and rgb.shape[1] in [1, 3, 4]:
                    rgb = np.transpose(rgb, (0, 2, 3, 1))
            else:
                rgb = np.asarray(rgb)

            for j in range(actual_len):
                img = np.clip(rgb[j], 0, 255).astype(np.uint8)
                all_rgb.append(Image.fromarray(img).convert("RGB"))

        return all_rgb

    def _build_labels(
        self,
        input_ids: torch.Tensor,
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D [B, T], got shape={tuple(input_ids.shape)}")

        labels = torch.full_like(input_ids, IGNORE_INDEX)

        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        
        assistant_ids = tokenizer.encode("assistant\n", add_special_tokens=False)
        assistant_start_ids = [im_start_id] + assistant_ids
        assistant_end_ids = [im_end_id]

        if len(assistant_start_ids) == 0 or len(assistant_end_ids) == 0:
            raise RuntimeError(
                "[Label Build Error] Failed to encode assistant boundary tokens."
            )

        def find_subseq(src, pat, start=0):
            m = len(pat)
            if m == 0:
                return -1
            for i in range(start, len(src) - m + 1):
                if src[i:i + m] == pat:
                    return i
            return -1

        found_any = False

        for b in range(input_ids.shape[0]):
            input_list = input_ids[b].tolist()
            search_pos = 0
            found_this_sample = False

            while True:
                st = find_subseq(input_list, assistant_start_ids, search_pos)
                if st < 0:
                    break

                content_st = st + len(assistant_start_ids)
                ed = find_subseq(input_list, assistant_end_ids, content_st)
                if ed < 0:
                    break

                if ed > content_st:
                    labels[b, content_st:ed] = input_ids[b, content_st:ed]
                    found_any = True
                    found_this_sample = True

                search_pos = ed + len(assistant_end_ids)

            if not found_this_sample:
                raise RuntimeError(
                    f"[Label Build Error] No assistant supervision span found for batch item {b}."
                )

        if not found_any:
            raise RuntimeError(
                "[Label Build Error] No assistant supervision span found in packed sequence."
            )

        return labels
    

    def __call__(self, features: List[Any]) -> Dict[str, torch.Tensor]:
        try:
            if len(features) == 1 and isinstance(features[0], list):
                features = features[0]
                
            if len(features) == 0:
                raise ValueError("Empty batch received by DataCollatorWithOnlineSim.")

            first = features[0]
            scene_name = first["scene_name"]
            dataset_type = first["dataset_type"]
            images_dir = first["images_dir"]
            scenes_dir = first["scenes_dir"]

            for feat in features[1:]:
                if not (
                    feat["scene_name"] == scene_name
                    and feat["dataset_type"] == dataset_type
                    and feat["images_dir"] == images_dir
                    and feat["scenes_dir"] == scenes_dir
                ):
                    raise RuntimeError(
                        "Guarantees same-scene batches, "
                        "but collator received a mixed-scene batch."
                    )

            # 2. Ensure simulator is loaded with the current scene
            self.load_scene_if_needed(
                scene_name=scene_name,
                dataset_type=dataset_type,
                images_dir=images_dir,
                scenes_dir=scenes_dir,
            )

            # 3. Flatten all positions/headings for simulator rendering
            flat_positions: List[List[float]] = []
            flat_headings: List[float] = []
            num_images_per_sample: List[int] = []

            for feat in features:
                positions = feat["positions"]
                headings = feat["headings"]

                if len(positions) != len(headings):
                    raise ValueError(
                        f"positions/headings length mismatch: {len(positions)} vs {len(headings)}"
                    )

                flat_positions.extend(positions)
                flat_headings.extend(headings)
                num_images_per_sample.append(len(positions))

            rendered_images = self._render_batch_images(flat_positions, flat_headings)

            if len(rendered_images) != sum(num_images_per_sample):
                raise RuntimeError(
                    f"Rendered image count mismatch: got {len(rendered_images)}, "
                    f"expected {sum(num_images_per_sample)}"
                )

            # 4. Split rendered images back to per-sample image lists
            per_sample_images: List[List[Image.Image]] = []
            cursor = 0
            for n in num_images_per_sample:
                per_sample_images.append(rendered_images[cursor: cursor + n])
                cursor += n

            # 5. Build per-sample text and image inputs
            batch_texts: List[str] = []
            batch_images: List[List[Image.Image]] = []

            for feat, images in zip(features, per_sample_images):
                messages = copy.deepcopy(feat["messages"])

                image_idx = 0
                for msg in messages:
                    if msg["role"] != "user":
                        continue
                    for item in msg["content"]:
                        if item["type"] == "image":
                            if image_idx >= len(images):
                                raise RuntimeError(
                                    f"Not enough rendered images for one sample: "
                                    f"need > {image_idx}, got {len(images)}"
                                )
                            item["image"] = images[image_idx]
                            image_idx += 1

                if image_idx != len(images):
                    raise RuntimeError(
                        f"Unused rendered images in one sample: used {image_idx}, total {len(images)}"
                    )

                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                batch_texts.append(text)
                batch_images.append(images)

            # 6. Tokenize packed multimodal conversations
            model_inputs = self.processor(
                text=batch_texts,
                images=batch_images,
                padding=True,
                return_tensors="pt",
            )

            # 7. Build labels from assistant spans only
            labels = self._build_labels(
                input_ids=model_inputs["input_ids"],
                tokenizer=self.processor.tokenizer,
            )
            model_inputs["labels"] = labels

            return model_inputs

        except Exception as e:
            # Dummy Batch for Erroe
            import traceback
            print(f"[Collator Error Catch] Failed to collate batch: {e}. Generating dummy batch to maintain DDP sync.")
            traceback.print_exc()
            
            dummy_text = "<|im_start|>system\nYou are an AI.<|im_end|>\n<|im_start|>user\nskip<|im_end|>\n<|im_start|>assistant\nskip<|im_end|>"
            
            model_inputs = self.processor(
                text=[dummy_text], # batch_size = 1
                padding=True,
                return_tensors="pt",
            )
            
            # set labels to IGNORE_INDEX (-100)，so that loss is 0
            model_inputs["labels"] = torch.full_like(model_inputs["input_ids"], IGNORE_INDEX)
            
            return model_inputs


# ==============================================================================
# 6. Train entry
# ==============================================================================

def train():
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Use training_args.seed to avoid collision
    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoints:
        try:
            latest_ckpt = max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))
            
            completed_steps = int(latest_ckpt.name.split("-")[-1])
            
            training_args.seed += completed_steps
            print(f"⚠️ [Warning] Resuming from step {completed_steps}! Resetting seed to {training_args.seed} to reshuffle data.")
            
        except Exception as e:
            training_args.seed += len(checkpoints) * 100 
            print(f"⚠️ [Warning] Failed to parse steps, fallback seed to {training_args.seed}.")

    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_args.seed)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = VLNForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
    )

    if not model_args.tune_mm_llm:
        model.language_model.requires_grad_(False)

    if hasattr(model, "multi_modal_projector") and not model_args.tune_mm_mlp:
        model.multi_modal_projector.requires_grad_(False)

    if hasattr(model, "visual") and not model_args.tune_mm_vision:
        model.visual.requires_grad_(False)

    if hasattr(model, "get_vision_tower") and not model_args.unfreeze_mm_vision_tower:
        try:
            model.get_vision_tower().requires_grad_(False)
        except Exception:
            pass

    device = training_args.device 

    sim_config = type(
        "Config",
        (),
        {
            "image_height": 512,
            "batch_size": data_args.batch_size,
            "max_depth": data_args.max_depth,
            "output_resolution": (data_args.image_height, data_args.image_width),
            "hfov_deg": data_args.image_hfov,
            "vfov_deg": data_args.image_vfov,
            "step_size": data_args.step_size,
            "eye_height": data_args.eye_height,
            "turn_angle": data_args.turn_angle,
            "max_step_height": data_args.max_step_height,
            "agent_radius": data_args.agent_radius,
            "goal_threshold":  data_args.goal_threshold,
        },
    )

    neural_simulator = image2sim.NeuralSimulator(sim_config).to(device)
    neural_simulator = data_tools.load_checkpoint(neural_simulator, model_args.simulator_ckpt)
    # neural_simulator.torch_compile()
    neural_simulator.eval()

    for p in neural_simulator.parameters():
        p.requires_grad = False

    train_dataset = Qwen3VLNOnlineDataset(data_args, processor)

    data_collator = DataCollatorWithOnlineSim(
        processor=processor,
        simulator=neural_simulator,
        scene_voxel_size=data_args.scene_voxel_size,
    )

    training_args.per_device_train_batch_size = 1
    training_args.per_device_eval_batch_size = 1
    training_args.dataloader_num_workers = 0
    training_args.remove_unused_columns = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )

    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoints:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.state_dict_type = "FULL_STATE_DICT"

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()