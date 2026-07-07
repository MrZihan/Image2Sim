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

import json
import math
import copy
import random
import pathlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, AutoProcessor

from vln.model.vln_model import VLNForCausalLM
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

def get_round_prefix(obs_idx: int) -> str:
    template = ROUND_PREFIXES[obs_idx % len(ROUND_PREFIXES)]
    return template.format(obs_idx=obs_idx)

def build_user_text(instruction: str, prefix: str, num_future_steps: int, obs_idx: int) -> str:
    if obs_idx > 0 and obs_idx % 8 == 0:
        return (
            f"Reminder of your goal: '{instruction}'.\n"
            f"{prefix}. Output the next {num_future_steps} actions."
        )
    else:
        return f"{prefix}. Output the next {num_future_steps} actions."

def actions_to_text(actions: List[int], expected_len: int) -> str:
    padded_actions = actions.copy()
    while len(padded_actions) < expected_len:
        padded_actions.append(0) 
    padded_actions = padded_actions[:expected_len]
    return "".join(ACTION_IDX_TO_TOKEN[a] for a in padded_actions)

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
    attn_implementation: str = field(default="sdpa")

@dataclass
class DataArguments:
    dataset_config: str = field(default="dataset_config.json")
    
    num_sparse_history_frames: int = field(default=48)
    num_recent_observation_frames: int = field(default=16)
    
    num_future_steps: int = field(default=2)
    max_episode_steps: int = field(default=200)
    
    online_training_ratio: float = field(default=0.8)
    dagger_ratio: float = field(default=0.8)
    
    chunk_size_min: int = field(default=4)
    chunk_size_max: int = field(default=32)

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

    max_train_scenes: int = field(default=200000)
    sample_ratios: str = field(default='{\"image2sim_batch_1\": 5, \"image2sim_batch_2\": 5, \"image2sim_batch_3\": 5, \"room_grounding\": 1, \"house_grounding\": 1, \"r2r\": 1, \"reverie\": 1, \"rxr\": 1, \"srdf\": 5}')

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=64 * 1024)
    ddp_timeout: int = field(default=7200)

# ==============================================================================
# 2. KV Cache & Context Management
# ==============================================================================

@dataclass
class RoundRecord:
    image: Image.Image
    executed_action_text: str  
    target_action_text: str
    obs_idx: int

class DAggerContextManager:
    def __init__(self, num_sparse=48, num_recent=16, num_future_steps=2, instruction=""):
        self.num_sparse = num_sparse
        self.num_recent = num_recent
        self.max_frames = num_sparse + num_recent
        self.num_future_steps = num_future_steps
        self.instruction = instruction

        self.all_historical_rounds: List[RoundRecord] = []
        self.active_rounds: List[RoundRecord] = []
        self.pending_current_image: Optional[Image.Image] = None
        self.pending_obs_idx: int = 0

    def reset(self, instruction: str):
        self.instruction = instruction
        self.all_historical_rounds = []
        self.active_rounds = []
        self.pending_current_image = None
        self.pending_obs_idx = 0

    def close(self):
        self.instruction = None
        del self.all_historical_rounds
        del self.active_rounds
        self.all_historical_rounds = []
        self.active_rounds = []
        self.pending_current_image = None
        self.pending_obs_idx = 0

    def observe(self, image: Image.Image, obs_idx: int):
        self.pending_current_image = image
        self.pending_obs_idx = obs_idx

    def _uniform_downsample(self, rounds: List[RoundRecord], max_keep: int) -> List[RoundRecord]:
        if max_keep <= 0 or len(rounds) == 0: return []
        if len(rounds) <= max_keep: return rounds
        idxs = torch.linspace(0, len(rounds) - 1, steps=max_keep).round().long().tolist()
        return [rounds[i] for i in idxs]

    def add_round_result(self, executed_text: str, target_action_text: str):
        record = RoundRecord(
            image=self.pending_current_image,
            executed_action_text=executed_text,
            target_action_text=target_action_text,
            obs_idx=self.pending_obs_idx,
        )
        self.all_historical_rounds.append(record)
        self.active_rounds.append(record)
        self.pending_current_image = None

    def build_messages_for_inference(self, system_prompt: str) -> Tuple[List[Dict[str, Any]], List[Image.Image], bool]:
        need_flush = False
        if len(self.active_rounds) >= self.max_frames:
            self.active_rounds = self._uniform_downsample(self.all_historical_rounds, self.num_sparse)
            need_flush = True

        messages = [{"role": "system", "content": system_prompt}]
        images = []

        for rr in self.active_rounds:
            prefix = get_round_prefix(rr.obs_idx)
            user_text = build_user_text(self.instruction, prefix, self.num_future_steps, rr.obs_idx)
            messages.append({"role": "user", "content": [{"type": "image", "image": rr.image}, {"type": "text", "text": user_text}]})
            messages.append({"role": "assistant", "content": rr.executed_action_text})
            images.append(rr.image)

        curr_prefix = get_round_prefix(self.pending_obs_idx)
        curr_text = build_user_text(self.instruction, curr_prefix, self.num_future_steps, self.pending_obs_idx)
        messages.append({"role": "user", "content": [{"type": "image", "image": self.pending_current_image}, {"type": "text", "text": curr_text}]})
        images.append(self.pending_current_image)

        return messages, images, need_flush

    def build_dense_training_data(self, system_prompt: str) -> Tuple[List[Dict[str, Any]], List[Image.Image], List[str]]:
        final_rounds = self.all_historical_rounds
        if len(final_rounds) > self.max_frames:
            final_rounds = self._uniform_downsample(final_rounds, self.max_frames)

        messages = [{"role": "system", "content": system_prompt}]
        images = []
        target_texts = []

        for rr in final_rounds:
            prefix = get_round_prefix(rr.obs_idx)
            user_text = build_user_text(self.instruction, prefix, self.num_future_steps, rr.obs_idx)
            messages.append({"role": "user", "content": [{"type": "image", "image": rr.image}, {"type": "text", "text": user_text}]})
            messages.append({"role": "assistant", "content": rr.executed_action_text})
            
            images.append(rr.image)
            target_texts.append(rr.target_action_text)

        return messages, images, target_texts

# ==============================================================================
# 3. DAgger Dataset (The Rollout Engine)
# ==============================================================================

class Qwen3VLNDAggerDataset(Dataset):
    def __init__(self, data_args: DataArguments, processor: transformers.ProcessorMixin, model: Any, simulator: Any):
        super().__init__()
        self.data_args = data_args
        self.processor = processor
        self.model = model
        self.simulator = simulator
        self.device = model.device

        self.action_str2idx = {"STOP": [0], "↑": [1], "←": [2], "→": [3]}
        self.gt_action_str2idx = {"STOP": [0], "MOVE_FORWARD": [1], "TURN_LEFT": [2], "TURN_RIGHT": [3]}

        print(f"Loading multi-source config from {data_args.dataset_config}...")
        with open(data_args.dataset_config, "r", encoding="utf-8") as f:
            self.dataset_sources = json.load(f)

        self.file_index: List[Dict[str, Any]] = []
        self.anchor_index: List[Tuple[int, int, int]] = []

        self._build_dataset_index()
        print(f"[Dataset] Scanned dataset and built {len(self.anchor_index)} training anchors via Lazy Loading.")
        
        self.current_scene_name = None
        self.past_key_values = None
        self.past_seq_len = 0
        self._vision_cache = {}

    def _build_dataset_index(self) -> None:
        import glob
        import concurrent.futures
        from collections import defaultdict
        
        self.file_index = []
        self.anchor_index = []
        visited_dirs = set()
        
        try:
            target_ratios = json.loads(self.data_args.sample_ratios)
        except Exception:
            target_ratios = {}

        print("[Dataset] Pre-caching valid scenes from all images_dirs to memory...")
        valid_scenes_cache = {}
        for src_key, src_info in self.dataset_sources.items():
            img_dir = src_info.get("images_dir", "")
            if img_dir and os.path.exists(img_dir):
                valid_scenes_cache[src_key] = {
                    "scenes": set(os.listdir(img_dir)),
                    "info": src_info
                }

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

            if not instr_type_to_files: continue

            available_types = list(instr_type_to_files.keys())
            weights = [target_ratios.get(t, 1.0) for t in available_types]

            sampled_files = []
            max_scenes = self.data_args.max_train_scenes
            for _ in range(max_scenes):
                chosen_type = random.choices(available_types, weights=weights, k=1)[0]
                chosen_file = random.choice(instr_type_to_files[chosen_type])
                sampled_files.append(chosen_file)

            EP_PATTERN = re.compile(r'"episode_id"\s*:\s*"?([^",\s}]+)"?')

            def _process_single_file(jf: str):
                scene_name = jf.split("/")[-1].replace(".json","").replace("output_","")
                scene_name = scene_name.replace(scene_name.split("_")[0]+"_","")

                true_source_info = None
                for cache_data in valid_scenes_cache.values():
                    if scene_name in cache_data["scenes"]:
                        true_source_info = cache_data["info"]
                        break
                
                if true_source_info is None: return None

                results = []
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        text_content = f.read()
                except Exception:
                    return None

                ep_matches = EP_PATTERN.findall(text_content)
                random.shuffle(ep_matches)
                ep_matches = ep_matches[:256]

                is_list_json = text_content.lstrip().startswith('[')
                
                if is_list_json:
                    if ep_matches:
                        for idx, ep_id_str in enumerate(ep_matches):
                            results.append({"file_path": jf, "source_info": true_source_info, "scene_name": scene_name, "episode_str": f"{jf.split('/')[-1]}_{ep_id_str}", "list_index": idx})
                    else:
                        list_count = text_content.count('"trajectory"') 
                        for idx in range(max(1, list_count)):
                            results.append({"file_path": jf, "source_info": true_source_info, "scene_name": scene_name, "episode_str": f"{jf.split('/')[-1]}_{idx}", "list_index": idx})
                else:
                    ep_id_str = ep_matches[0] if ep_matches else "0"
                    results.append({"file_path": jf, "source_info": true_source_info, "scene_name": scene_name, "episode_str": f"{jf.split('/')[-1]}_{ep_id_str}", "list_index": 0})
                return results

            valid_results = []
            max_workers = min(64, (os.cpu_count() or 1) * 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = executor.map(_process_single_file, sampled_files)
                for res_list in results:
                    if res_list is not None: valid_results.extend(res_list)

            for res in valid_results:
                ep_id = len(self.file_index)
                self.file_index.append(res)
                self.anchor_index.append(ep_id) 

        print("[Dataset] Grouping trajectories into strictly aligned blocks...")
        scene_to_ep_ids = defaultdict(list)
        for ep_id in self.anchor_index:
            meta = self.file_index[ep_id]
            scene_to_ep_ids[meta["scene_name"]].append(ep_id)

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
                    block_eps.extend(random.choices(block_eps, k=SYNC_BLOCK_SIZE - len(block_eps)))
                scene_blocks.append(block_eps)
            blocks_by_scene.append(scene_blocks)

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
            all_blocks.extend(all_blocks[:self.world_size - remainder])

        self.prebuilt_batches = []
        for block in all_blocks:
            for j in range(0, len(block), sim_bs):
                self.prebuilt_batches.append(block[j : j + sim_bs])

        chunk_size_in_blocks = len(all_blocks) // self.world_size
        batches_per_block = SYNC_BLOCK_SIZE // sim_bs
        self.current_ptr = self.rank * (chunk_size_in_blocks * batches_per_block)
        
        self._cached_json_path = None
        self._cached_json_data = None

    def load_scene_if_needed(self, scene_name: str, dataset_type: str, images_dir: str, scenes_dir: str):
        if self.current_scene_name == scene_name: return
        full_pcd_path = os.path.join(scenes_dir, f"{scene_name}.pcd")
        nav_pcd_path = os.path.join(scenes_dir, f"{scene_name}_navigable.pcd")
        scene_path = os.path.join(images_dir, scene_name)

        scene_pcd = o3d.io.read_point_cloud(full_pcd_path) if os.path.exists(full_pcd_path) else None
        nav_pcd = o3d.io.read_point_cloud(nav_pcd_path) if os.path.exists(nav_pcd_path) else None

        scene_xyz, scene_rgb, scene_feats, scene_gs, _ = data_tools.build_scene_pointcloud_data(
            scene_path=scene_path, dataset_type=dataset_type, device=self.simulator.device,
            voxel_size=self.data_args.scene_voxel_size, model=self.simulator, max_batch_size=1000, inpaint_depth=True
        )
        self.simulator.import_scene_gaussian(scene_xyz, scene_rgb, scene_feats, scene_gs)
        self.simulator.load_navigable_pcd(nav_pcd=nav_pcd, scene_pcd=scene_pcd)
        self.current_scene_name = scene_name

    def parse_actions(self, output: str) -> List[int]:
        regex = re.compile("|".join(re.escape(action) for action in self.action_str2idx))
        return [self.action_str2idx[match][0] for match in regex.findall(output)]

    def _get_image_cache_key(self, image: Image.Image):
        return id(image)


    def _get_cached_vision_inputs(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Cache Qwen-VL image processor outputs for each PIL image.
        This avoids repeatedly resizing / normalizing / patchifying all history images.
        Cache tensors stay on CPU to avoid GPU memory blow-up.
        """
        key = self._get_image_cache_key(image)

        if key not in self._vision_cache:
            if hasattr(self.processor, "image_processor"):
                vision_inputs = self.processor.image_processor(
                    images=[image],
                    return_tensors="pt",
                )
            else:
                # Fallback: slower, but only called once per image.
                dummy_text = "<|vision_start|><|image_pad|><|vision_end|>"
                vision_inputs = self.processor(
                    text=[dummy_text],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                )

            pixel_values = vision_inputs["pixel_values"].detach().cpu()
            image_grid_thw = vision_inputs["image_grid_thw"].detach().cpu()

            self._vision_cache[key] = {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }

        return self._vision_cache[key]


    def _expand_image_placeholders(self, text: str, image_grid_thw: torch.Tensor) -> str:
        """
        Mimic Qwen-VL processor's image token expansion without re-running image preprocessing.
        Qwen-VL uses image_grid_thw and merge_size to expand <|image_pad|>.
        """
        image_token = getattr(self.processor, "image_token", "<|image_pad|>")
        placeholder = "<|placeholder|>"

        merge_size = getattr(getattr(self.processor, "image_processor", None), "merge_size", 2)
        merge_length = merge_size ** 2

        expanded_text = text

        for grid in image_grid_thw:
            num_image_tokens = int(grid.prod().item() // merge_length)
            expanded_text = expanded_text.replace(
                image_token,
                placeholder * num_image_tokens,
                1,
            )

        expanded_text = expanded_text.replace(placeholder, image_token)
        return expanded_text


    def _prepare_cached_processor_inputs(
        self,
        messages: List[Dict[str, Any]],
        images: List[Image.Image],
        include_all_pixels: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Build model inputs without repeatedly image-processing all history images.
        
        include_all_pixels=True:
            used for full prefill after flush; concatenate all image pixel_values.
        include_all_pixels=False:
            used for KV-cache incremental step; only return latest image pixel_values.
        """
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        vision_items = [self._get_cached_vision_inputs(img) for img in images]
        image_grid_thw = torch.cat([v["image_grid_thw"] for v in vision_items], dim=0)

        expanded_text = self._expand_image_placeholders(text, image_grid_thw)

        tokenized = self.processor.tokenizer(
            [expanded_text],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        inputs = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized.get(
                "attention_mask",
                torch.ones_like(tokenized["input_ids"], dtype=torch.bool),
            ),
            "image_grid_thw": image_grid_thw,
        }

        if include_all_pixels:
            inputs["pixel_values"] = torch.cat(
                [v["pixel_values"] for v in vision_items],
                dim=0,
            )
        else:
            # incremental KV-cache case: only the new/current image is needed
            inputs["pixel_values"] = vision_items[-1]["pixel_values"]

        # Optional but useful for Qwen3-VL variants that accept mm_token_type_ids.
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            getattr(self.processor, "image_token", "<|image_pad|>")
        )
        if image_token_id is not None and image_token_id >= 0:
            mm_token_type_ids = torch.zeros_like(inputs["input_ids"])
            mm_token_type_ids[inputs["input_ids"] == image_token_id] = 1
            inputs["mm_token_type_ids"] = mm_token_type_ids

        inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        return inputs


    @torch.no_grad()
    def _predict_actions_stream(
        self,
        inference_model,
        messages: List[Dict[str, Any]],
        images: List[Image.Image],
        need_flush: bool,
    ) -> List[int]:
        import inspect

        if need_flush or self.past_key_values is None:
            self.past_key_values = None
            self.past_seq_len = 0

        is_prefill = self.past_key_values is None

        inputs = self._prepare_cached_processor_inputs(
            messages=messages,
            images=images,
            include_all_pixels=is_prefill,
        )

        if "attention_mask" not in inputs or inputs["attention_mask"] is None:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.bool)

        full_input_ids = inputs["input_ids"]

        rope_kwargs = {
            "image_grid_thw": inputs.get("image_grid_thw"),
            "video_grid_thw": inputs.get("video_grid_thw"),
            "attention_mask": inputs["attention_mask"],
        }

        if "mm_token_type_ids" in inspect.signature(inference_model.model.get_rope_index).parameters:
            rope_kwargs["mm_token_type_ids"] = inputs.get("mm_token_type_ids", None)

        full_position_ids, rope_deltas = inference_model.model.get_rope_index(
            full_input_ids,
            **rope_kwargs,
        )
        inference_model.model.rope_deltas = rope_deltas

        prefill_seq_len = full_input_ids.shape[1]

        if self.past_key_values is None:
            outputs = inference_model(
                **inputs,
                position_ids=full_position_ids,
                use_cache=True,
                return_dict=True,
            )
        else:
            new_ids = full_input_ids[:, self.past_seq_len:]
            new_pos = full_position_ids[:, :, self.past_seq_len:]

            grid_thw = inputs["image_grid_thw"][-1:]
            new_pixels = inputs["pixel_values"]

            image_token = getattr(self.processor, "image_token", "<|image_pad|>")
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token)
            merge_size = getattr(self.processor.image_processor, "merge_size", 2)
            expected_img_tokens = int(grid_thw[0].prod().item() // (merge_size ** 2))
            actual_img_tokens = int((new_ids == image_token_id).sum().item())
            assert actual_img_tokens == expected_img_tokens, (
                f"KV/image mismatch: got {actual_img_tokens}, expected {expected_img_tokens}"
            )

            tgt_mask = torch.ones(
                (1, self.past_seq_len + new_ids.shape[1]),
                device=self.device,
                dtype=torch.bool,
            )

            forward_kwargs = dict(
                input_ids=new_ids,
                position_ids=new_pos,
                pixel_values=new_pixels,
                image_grid_thw=grid_thw,
                attention_mask=tgt_mask,
                past_key_values=self.past_key_values,
                use_cache=True,
                return_dict=True,
            )

            if "mm_token_type_ids" in inputs:
                forward_kwargs["mm_token_type_ids"] = inputs["mm_token_type_ids"][:, self.past_seq_len:]

            outputs = inference_model(**forward_kwargs)

        self.past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)

        decode_start_len = prefill_seq_len
        generated_ids = [next_token.item()]

        for step_i in range(self.data_args.num_future_steps - 1):
            curr_seq_len = decode_start_len + step_i
            step_mask = torch.ones((1, curr_seq_len + 1), device=self.device, dtype=torch.bool)
            step_pos = full_position_ids[..., -1:] + (step_i + 1)

            outputs = inference_model(
                input_ids=next_token,
                position_ids=step_pos,
                attention_mask=step_mask,
                past_key_values=self.past_key_values,
                use_cache=True,
                return_dict=True,
            )

            self.past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated_ids.append(next_token.item())

        action_seq = self.parse_actions(
            self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        )

        while len(action_seq) < self.data_args.num_future_steps:
            action_seq.append(0)

        if hasattr(self.past_key_values, "crop"):
            self.past_key_values.crop(decode_start_len)
        else:
            self.past_key_values = tuple(
                tuple(t[:, :, :decode_start_len, :] for t in layer)
                for layer in self.past_key_values
            )

        self.past_seq_len = decode_start_len
        return action_seq[: self.data_args.num_future_steps]


    def __len__(self) -> int:
        return len(self.prebuilt_batches)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        max_retries = 20
        for attempt in range(max_retries):
            try:
                actual_idx = self.current_ptr % len(self.prebuilt_batches)
                self.current_ptr += 1
                
                meta = self.file_index[self.prebuilt_batches[actual_idx][0]]
                file_path = meta["file_path"]
                
                if getattr(self, "_cached_json_path", None) != file_path:
                    with open(file_path, "r", encoding="utf-8") as f:
                        self._cached_json_data = json.load(f)
                    self._cached_json_path = file_path
                    
                data = self._cached_json_data
                item = data[meta.get("list_index", 0)] if isinstance(data, list) else data
                trajectory = item.get("trajectory", [])
                
                if not trajectory or len(trajectory) < 3: continue 
                
                instruction = item.get("instruction_data", {}).get("instruction", "")
                source_info = meta["source_info"]
                target_pos = trajectory[-1]["pos"]  
                
                if random.random() < self.data_args.online_training_ratio:
                    offset_steps = random.choice(list(range(-11, 12)))
                    turn_angle_rad = math.radians(self.data_args.turn_angle)
                    correct_action_str = "TURN_RIGHT" if offset_steps < 0 else "TURN_LEFT"
                    
                    prepend_traj = []
                    curr_heading = trajectory[0]["heading"] + offset_steps * turn_angle_rad
                    step_heading = turn_angle_rad if offset_steps < 0 else -turn_angle_rad
                    
                    for _ in range(abs(offset_steps)):
                        fake_step = copy.deepcopy(trajectory[0])
                        fake_step["heading"] = curr_heading
                        fake_step["action"] = correct_action_str
                        prepend_traj.append(fake_step)
                        curr_heading += step_heading
                        
                    trajectory = prepend_traj + trajectory

                self.load_scene_if_needed(meta["scene_name"], source_info["dataset_type"], source_info["images_dir"], source_info["scenes_dir"])
                
                pos = torch.tensor(trajectory[0]["pos"], device=self.device, dtype=torch.float32).unsqueeze(0)
                pos[:, 2] += self.simulator.eye_height
                self.simulator.agent_pos = pos
                self.simulator.agent_heading = torch.tensor([trajectory[0]["heading"]], device=self.device, dtype=torch.float32)
                
                obs = self.simulator._render_current_view()
                    
                system_prompt = (
                    "You are an autonomous navigation assistant. "
                    f"Your ultimate goal is: {instruction}. "
                    f"Based on the current observation and step history, output exactly {self.data_args.num_future_steps} actions to reach the goal. "
                    "Actions must be chosen from: TURN LEFT (←), TURN RIGHT (→), MOVE FORWARD (↑), or STOP. "
                    "Respond with actions only. Do not output any extra words, spaces, or punctuation."
                )

                ctx_mgr = DAggerContextManager(
                    num_sparse=self.data_args.num_sparse_history_frames,
                    num_recent=self.data_args.num_recent_observation_frames,
                    num_future_steps=self.data_args.num_future_steps,
                    instruction=instruction
                )
                
                self.past_key_values = None
                self.past_seq_len = 0
                
                is_dagger_mode = random.random() < self.data_args.online_training_ratio
                run_pure_gt = not is_dagger_mode

                if len(trajectory) > self.data_args.max_episode_steps / 2:
                    run_pure_gt = True
                    is_dagger_mode = False

                inference_model = self.model.module if hasattr(self.model, "module") else self.model
                inference_model.eval() 
                
                if getattr(inference_model.config, "gradient_checkpointing", False) or getattr(inference_model, "gradient_checkpointing", False):
                    inference_model.gradient_checkpointing_disable()

                if is_dagger_mode:
                    try:
                        chunk_timer = 0
                        use_model = False
                        initial_pos = self.simulator.agent_pos[0].cpu().numpy()
                        initial_dist = np.linalg.norm(initial_pos[:2] - np.array(target_pos)[:2])
                        stuck_counter = 0
                        prev_pos = initial_pos.copy()
                        physical_step = 0
                        past_use_model = True
                        MAX_MODEL_CALLS = self.data_args.chunk_size_max * 2
                        model_call_count = 0
                        current_gt_idx = 0

                        for step_idx in range(self.data_args.max_episode_steps):
                            if chunk_timer <= 0:
                                use_model = random.random() < self.data_args.dagger_ratio
                                chunk_timer = random.randint(self.data_args.chunk_size_min, self.data_args.chunk_size_max)

                            if model_call_count >= MAX_MODEL_CALLS:
                                use_model = False
                                chunk_timer = self.data_args.chunk_size_max

                            rgb = obs["rgb"].detach().float().cpu().numpy()
                            if rgb.ndim == 4 and rgb.shape[1] in [1, 3, 4]:
                                rgb = np.transpose(rgb, (0, 2, 3, 1))

                            image = Image.fromarray(np.clip(rgb[0], 0, 255).astype(np.uint8)).convert("RGB")
                            ctx_mgr.observe(image, physical_step)

                            curr_pos_np = self.simulator.agent_pos[0].cpu().numpy()
                            curr_dist_to_goal = np.linalg.norm(curr_pos_np[:2] - np.array(target_pos)[:2])
                            safe_distance_threshold = max(1.0, self.data_args.goal_threshold * 2)
                            
                            # Strategy 1: Target-driven Dagger
                            # gt_action_strs = self.simulator.generate_action_sequence(target_positions=[target_pos], max_actions=self.data_args.num_future_steps)[:self.data_args.num_future_steps]

                            # Strategy 2: Path-following Dagger
                            #################
                            search_window = min(len(trajectory) - current_gt_idx, 15)
                            min_dist = float('inf')
                            best_idx_offset = 0
                            for offset in range(search_window):
                                idx = current_gt_idx + offset
                                dist = np.linalg.norm(curr_pos_np[:2] - np.array(trajectory[idx]["pos"])[:2])
                                if dist < min_dist:
                                    min_dist = dist
                                    best_idx_offset = offset
                            
                            current_gt_idx += best_idx_offset 
                            
                            # Ghost Teacher Waypoint
                            start_idx = min(current_gt_idx + 1, len(trajectory) - 1)
                            min_subgoal_dist = (self.data_args.num_future_steps + 1) * self.data_args.step_size + self.data_args.goal_threshold

                            curr_xy = np.asarray(curr_pos_np[:2], dtype=np.float32)

                            sub_goal_idx = len(trajectory) - 1
                            for idx in range(start_idx, len(trajectory)):
                                cand_pos = np.asarray(trajectory[idx]["pos"], dtype=np.float32)
                                cand_dist = np.linalg.norm(cand_pos[:2] - curr_xy)

                                if cand_dist >= min_subgoal_dist:
                                    sub_goal_idx = idx
                                    break

                            local_sub_goal = trajectory[sub_goal_idx]["pos"]

                            gt_action_strs = self.simulator.generate_action_sequence(
                                target_positions=[local_sub_goal],
                                max_actions=self.data_args.num_future_steps
                            )[:self.data_args.num_future_steps]
                            #################


                            if "STOP" in gt_action_strs and curr_dist_to_goal > safe_distance_threshold:
                                print(f"Oracle Navigation Failed! Agent is {curr_dist_to_goal:.2f}m away but Oracle output STOP.")
                                break

                            gt_action_seq = [self.gt_action_str2idx.get(a, [0])[0] for a in gt_action_strs]
                            if len(gt_action_seq) == 0: gt_action_seq = [0] * self.data_args.num_future_steps

                            if use_model:
                                messages, images, need_flush = ctx_mgr.build_messages_for_inference(system_prompt)
                                if not past_use_model: need_flush = True
                                executed_action_seq = self._predict_actions_stream(inference_model, messages, images, need_flush)
                                model_call_count += self.data_args.num_future_steps
                            else:
                                executed_action_seq = gt_action_seq
                                
                            past_use_model = use_model
                            executed_text = actions_to_text(executed_action_seq, self.data_args.num_future_steps)

                            target_action_text = ACTION_IDX_TO_TOKEN[gt_action_seq[0]]
                            ctx_mgr.add_round_result(executed_text=executed_text, target_action_text=target_action_text)

                            done = False
                            for a in executed_action_seq:
                                obs, info = self.simulator.step([a], render_observation=True)
                                chunk_timer -= 1
                                physical_step += 1
                                if a == self.gt_action_str2idx.get("STOP", [0])[0] and curr_dist_to_goal <= safe_distance_threshold:
                                    done = True
                                    break

                            if done: break

                            curr_pos_loop = self.simulator.agent_pos[0].cpu().numpy()
                            if np.linalg.norm(curr_pos_loop - prev_pos) < self.data_args.step_size / 2: 
                                stuck_counter += 1
                            else:
                                stuck_counter = 0
                            prev_pos = curr_pos_loop.copy()
                            
                            if stuck_counter >= 4:
                                use_model = False

                            if stuck_counter >= 12:
                                print(f"Navigation stucked! Agent is {curr_dist_to_goal:.2f}m away.")
                                break
                            
                        dense_messages, dense_images, target_texts = ctx_mgr.build_dense_training_data(system_prompt)
                        if hasattr(ctx_mgr, "close"): ctx_mgr.close()

                        if len(dense_images) == 0:
                            raise RuntimeError("No valid data for Dagger training.")

                    except Exception as e:
                        print(e)
                        run_pure_gt = True
                        self.past_key_values = None 
                        torch.cuda.empty_cache()

                if run_pure_gt:
                    ctx_mgr.reset(instruction)
                    limit = min(self.data_args.max_episode_steps, len(trajectory))
                    for i in range(limit):
                        if i % self.data_args.num_future_steps != 0: continue
                            
                        step_data = trajectory[i]
                        pos = torch.tensor(step_data["pos"], device=self.device, dtype=torch.float32).unsqueeze(0)
                        pos[:, 2] += self.simulator.eye_height
                        self.simulator.agent_pos = pos
                        self.simulator.agent_heading = torch.tensor([step_data["heading"]], device=self.device, dtype=torch.float32)
                        
                        obs = self.simulator._render_current_view()
                        rgb = obs["rgb"].detach().float().cpu().numpy()
                        if rgb.ndim == 4 and rgb.shape[1] in [1, 3, 4]: rgb = np.transpose(rgb, (0, 2, 3, 1))
                            
                        image = Image.fromarray(np.clip(rgb[0], 0, 255).astype(np.uint8)).convert("RGB")
                        ctx_mgr.observe(image, i)
                        
                        future_actions = []
                        for j in range(self.data_args.num_future_steps):
                            if i + j < len(trajectory):
                                act_val = trajectory[i + j].get("action", "STOP")
                                future_actions.append(act_val if isinstance(act_val, int) else self.gt_action_str2idx.get(act_val, [0])[0])
                            else:
                                future_actions.append(0)
                                
                        full_gt_text = actions_to_text(future_actions, self.data_args.num_future_steps)
                        
                        ctx_mgr.add_round_result(executed_text=full_gt_text, target_action_text=full_gt_text)
                        
                        if 0 in future_actions: break

                    dense_messages, dense_images, target_texts = ctx_mgr.build_dense_training_data(system_prompt)
                    if hasattr(ctx_mgr, "close"): ctx_mgr.close()

                    if len(dense_images) == 0:
                        raise RuntimeError("No valid data collected in this rollout.")

                self._vision_cache.clear()
                inference_model.train() 
                inference_model.gradient_checkpointing_enable()
                return {
                    "messages": dense_messages,
                    "images": dense_images,
                    "gt_texts": target_texts, 
                }
            
            except Exception as e:
                print(e)
                self._vision_cache.clear()
                inference_model = self.model.module if hasattr(self.model, "module") else self.model
                inference_model.train() 
                inference_model.gradient_checkpointing_enable()
            finally:
                torch.cuda.empty_cache()

        raise RuntimeError(f"Failed to load a valid sample after {max_retries} attempts.")

# ==============================================================================
# 4. Dense Collator (The Token Aligner)
# ==============================================================================

@dataclass
class DAggerDataCollator:
    processor: transformers.ProcessorMixin

    def _build_labels(self, input_ids: torch.Tensor, gt_texts_batch: List[List[str]]) -> torch.Tensor:
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_ids = self.processor.tokenizer.encode("assistant\n", add_special_tokens=False)
        assistant_start_ids = [im_start_id] + assistant_ids
        assistant_end_ids = [im_end_id]

        def find_subseq(src, pat, start=0):
            m = len(pat)
            if m == 0: return -1
            for i in range(start, len(src) - m + 1):
                if src[i:i + m] == pat: return i
            return -1

        for b in range(input_ids.shape[0]):
            input_list = input_ids[b].tolist()
            gt_texts = gt_texts_batch[b]
            gt_idx = 0
            search_pos = 0
            
            while True:
                st = find_subseq(input_list, assistant_start_ids, search_pos)
                if st < 0: break
                content_st = st + len(assistant_start_ids)
                ed = find_subseq(input_list, assistant_end_ids, content_st)
                if ed < 0: break
                
                if ed > content_st:
                    span_len = ed - content_st
                    
                    if gt_idx < len(gt_texts):
                        target_text = gt_texts[gt_idx]
                        gt_tokens = self.processor.tokenizer.encode(target_text, add_special_tokens=False)
                        gt_len = len(gt_tokens)
                        
                        copy_len = min(span_len, gt_len)
                        labels[b, content_st : content_st + copy_len] = torch.tensor(
                            gt_tokens[:copy_len], dtype=labels.dtype, device=labels.device
                        )
                        gt_idx += 1
                        
                search_pos = ed + len(assistant_end_ids)
                
        return labels

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        try:
            batch_texts = []
            batch_images = []
            gt_texts_batch = []

            for feat in features:
                text = self.processor.apply_chat_template(
                    feat["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                batch_texts.append(text)
                batch_images.append(feat["images"])
                gt_texts_batch.append(feat["gt_texts"])

            model_inputs = self.processor(
                text=batch_texts,
                images=batch_images,
                padding=True,
                return_tensors="pt",
            )

            labels = self._build_labels(model_inputs["input_ids"], gt_texts_batch)
            model_inputs["labels"] = labels
            return model_inputs

        except Exception as e:
            dummy_text = "<|im_start|>system\nYou are an AI.<|im_end|>\n<|im_start|>user\nskip<|im_end|>\n<|im_start|>assistant\nskip<|im_end|>"
            
            model_inputs = self.processor(
                text=[dummy_text],
                padding=True,
                return_tensors="pt",
            )
            model_inputs["labels"] = torch.full_like(model_inputs["input_ids"], IGNORE_INDEX)
            return model_inputs

# ==============================================================================
# 5. Train entry
# ==============================================================================

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoints:
        try:
            latest_ckpt = max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))
            completed_steps = int(latest_ckpt.name.split("-")[-1])
            training_args.seed += completed_steps
        except Exception:
            training_args.seed += len(checkpoints) * 100 

    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_args.seed)

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    
    model = VLNForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
    ).to(training_args.device)

    if not model_args.tune_mm_llm: model.language_model.requires_grad_(False)
    if not model_args.tune_mm_mlp: model.multi_modal_projector.requires_grad_(False)
    if not model_args.tune_mm_vision: model.visual.requires_grad_(False)

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
    
    neural_simulator = image2sim.NeuralSimulator(sim_config).to(training_args.device)
    neural_simulator = data_tools.load_checkpoint(neural_simulator, model_args.simulator_ckpt)
    neural_simulator.eval()
    for p in neural_simulator.parameters(): p.requires_grad = False

    train_dataset = Qwen3VLNDAggerDataset(data_args, processor, model, neural_simulator)
    data_collator = DAggerDataCollator(processor=processor)

    training_args.per_device_train_batch_size = 1
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