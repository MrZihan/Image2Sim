import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import json
import tqdm
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math
import torch
import torch.distributed as dist
from PIL import Image
import numpy as np
import habitat
from habitat import Env
from habitat_extensions import measures
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

from transformers import AutoProcessor
from vln.model.vln_model import VLNForCausalLM
from utils.dist import *
from omegaconf import OmegaConf

# ==============================================================================
# 0. Shared prompt helpers
# ==============================================================================

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

def build_system_prompt(instruction: str, num_future_steps: int) -> str:
    return (
        "You are an autonomous navigation assistant. "
        f"Your ultimate goal is: {instruction}. "
        f"Based on the current observation and step history, output exactly {num_future_steps} actions to reach the goal. "
        "Actions must be chosen from: TURN LEFT (←), TURN RIGHT (→), MOVE FORWARD (↑), or STOP. "
        "Respond with actions only. Do not output any extra words, spaces, or punctuation."
    )

def get_round_prefix(round_idx: int, obs_idx: int) -> str:
    template = ROUND_PREFIXES[round_idx % len(ROUND_PREFIXES)]
    return template.format(obs_idx=obs_idx)

def build_user_text(instruction: str, prefix: str, num_future_steps: int, round_idx: int) -> str:
    if round_idx > 0 and round_idx % 8 == 0:
        return (
            f"Reminder of your goal: '{instruction}'.\n"
            f"{prefix}. Output the next {num_future_steps} actions."
        )
    else:
        return f"{prefix}. Output the next {num_future_steps} actions."

def actions_to_text(actions: List[int]) -> str:
    return "".join(ACTION_IDX_TO_TOKEN[a] for a in actions)


@dataclass
class RoundRecord:
    image: Image.Image
    assistant_text: str
    obs_idx: int


class InterleavedHistoryContextManager:
    """
    Unified prompt reconstruction for evaluation.

    Design:
      - sparse_history_rounds: uniformly sampled older replay rounds
      - recent_rounds: recent dense replay rounds
      - every historical round is represented as:
            user(image + followup text) -> assistant(action chunk)
      - current observation is appended as the final user turn

    This matches the dense interleaved training topology much better than the
    previous "sparse image block + recent replay" mixed format.

    Important:
      - This is still prompt reconstruction.
      - It is NOT persistent cross-call KV-cache reuse.
    """
    def __init__(
        self,
        num_sparse_history_frames: int,
        num_recent_observation_frames: int,
        num_future_steps: int,
        instruction: str,
    ):
        self.num_sparse_history_frames = num_sparse_history_frames
        self.num_recent_observation_frames = num_recent_observation_frames
        self.num_future_steps = num_future_steps
        self.instruction = instruction

        self.sparse_history_rounds: List[RoundRecord] = []
        self.recent_rounds: List[RoundRecord] = []
        self.pending_current_image: Optional[Image.Image] = None
        self.pending_obs_idx: int = 0 

    def reset(self, instruction: str):
        self.instruction = instruction
        self.sparse_history_rounds = []
        self.recent_rounds = []
        self.pending_current_image = None
        self.pending_obs_idx = 0

    def observe(self, image: Image.Image, obs_idx: int):
        self.pending_current_image = image
        self.pending_obs_idx = obs_idx

    def _uniform_downsample_rounds(
        self,
        rounds: List[RoundRecord],
        max_keep: int,
    ) -> List[RoundRecord]:
        if max_keep <= 0 or len(rounds) == 0:
            return []
        if len(rounds) <= max_keep:
            return rounds

        idxs = torch.linspace(0, len(rounds) - 1, steps=max_keep).round().long().tolist()
        return [rounds[i] for i in idxs]

    def _maybe_fold_recent_into_sparse(self):
        if len(self.recent_rounds) > self.num_recent_observation_frames:
            overflow = len(self.recent_rounds) - self.num_recent_observation_frames
            to_fold = self.recent_rounds[:overflow]
            self.recent_rounds = self.recent_rounds[overflow:]
            self.sparse_history_rounds.extend(to_fold)

    def add_round_result(self, assistant_text: str):
        if self.pending_current_image is None:
            raise RuntimeError("Current observation image is missing before adding round result.")

        self.recent_rounds.append(
            RoundRecord(
                image=self.pending_current_image,
                assistant_text=assistant_text,
                obs_idx=self.pending_obs_idx,
            )
        )
        self.pending_current_image = None
        self._maybe_fold_recent_into_sparse()

    def build_messages_and_images(self, system_prompt: str) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        if self.pending_current_image is None:
            raise RuntimeError("Current observation image is missing.")

        if len(self.sparse_history_rounds) > self.num_sparse_history_frames:
            sampled_sparse = self._uniform_downsample_rounds(
                self.sparse_history_rounds, 
                self.num_sparse_history_frames
            )
        else:
            sampled_sparse = self.sparse_history_rounds

        replay_rounds = sampled_sparse + self.recent_rounds
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        image_inputs: List[Image.Image] = []

        # Historical replay rounds: fully interleaved user -> assistant
        for round_idx, rr in enumerate(replay_rounds):
            prefix = get_round_prefix(round_idx, rr.obs_idx)
            user_text = build_user_text(self.instruction, prefix, self.num_future_steps, round_idx)

            user_content = [
                {"type": "image", "image": rr.image},
                {"type": "text", "text": user_text},
            ]
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": rr.assistant_text})
            image_inputs.append(rr.image)

        # Current planning turn
        current_round_idx = len(replay_rounds)
        current_prefix = get_round_prefix(current_round_idx, self.pending_obs_idx)
        current_text = build_user_text(self.instruction, current_prefix, self.num_future_steps, current_round_idx)

        current_user_content = [
            {"type": "image", "image": self.pending_current_image},
            {"type": "text", "text": current_text},
        ]
        messages.append({"role": "user", "content": current_user_content})
        image_inputs.append(self.pending_current_image)

        return messages, image_inputs


def _read_jsonl_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                records.append(rec)
    return records


def _merge_rank_result_files(output_path: str, world_size: int) -> List[Dict[str, Any]]:
    """
    Merge result_rank{r}.jsonl -> result.jsonl
    Deduplicate by (scene_id, episode_id), keeping the first valid occurrence.
    """
    merged: List[Dict[str, Any]] = []
    seen = set()

    for r in range(world_size):
        rank_path = os.path.join(output_path, f"result_rank{r}.jsonl")
        rank_records = _read_jsonl_records(rank_path)

        for rec in rank_records:
            if "scene_id" not in rec or "episode_id" not in rec:
                continue

            key = (rec["scene_id"], str(rec["episode_id"]))
            if key in seen:
                continue

            seen.add(key)
            rec = dict(rec)
            rec["episode_id"] = str(rec["episode_id"])
            merged.append(rec)

    merged.sort(key=lambda x: (x["scene_id"], x["episode_id"]))

    merged_path = os.path.join(output_path, "result.jsonl")
    with open(merged_path, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return merged

class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        processor: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)

        with habitat.config.read_write(self.config):
            self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = self.local_rank
            self.config.habitat.dataset.split = self.split
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_resolution=1024,
                        draw_shortest_path=True,
                        draw_view_points=True,
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        self.model = model
        self.processor = processor
        self.model.eval()

        self.actions2idx = {
            "STOP": [0],
            "↑": [1],
            "←": [2],
            "→": [3],
        }

        self.context_manager = InterleavedHistoryContextManager(
            num_sparse_history_frames=args.num_sparse_history_frames,
            num_recent_observation_frames=args.num_recent_observation_frames,
            num_future_steps=args.num_future_steps,
            instruction="",
        )

    def config_env(self) -> Env:
        OmegaConf.set_readonly(self.config, False)
        return Env(config=self.config)

    def parse_actions(self, output: str) -> List[int]:
        action_patterns = "|".join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        return [self.actions2idx[match][0] for match in matches]

    def _prepare_inputs(self, messages: List[Dict[str, Any]], images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    @torch.no_grad()
    def _predict_actions(self, messages: List[Dict[str, Any]], images: List[Image.Image]) -> Tuple[str, List[int]]:
        inputs = self._prepare_inputs(messages, images)
        prompt_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.args.num_future_steps,
            do_sample=False,
            use_cache=True,  # cache is only used inside this single generate call
            return_dict_in_generate=True,
            output_scores=True
        )
        logits = torch.stack(outputs.scores, dim=1)

        sequences = outputs.sequences
        generated_ids = sequences[0, prompt_len:]
        llm_outputs = self.processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        action_seq = self.parse_actions(llm_outputs)

        while len(action_seq) < self.args.num_future_steps:
            action_seq.append(0)
        action_seq = action_seq[: self.args.num_future_steps]

        assistant_text = actions_to_text(action_seq)
        return assistant_text, action_seq, logits
    

    def eval_action(self, idx: int, args: argparse.Namespace):
        env = self.config_env()
        scene_episode_dict: Dict[str, List[Any]] = {}

        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        sucs, spls, oss, nes, ndtws = [], [], [], [], []
        done_res = set()

        os.makedirs(self.output_path, exist_ok=True)

        rank = get_rank()
        world_size = get_world_size()
        result_path = os.path.join(self.output_path, f"result_rank{rank}.jsonl")

        # Resume: read all json files
        for r in range(world_size):
            prev_result_path = os.path.join(self.output_path, f"result_rank{r}.jsonl")
            if not os.path.exists(prev_result_path):
                continue

            with open(prev_result_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        res = json.loads(line)
                    except Exception:
                        continue

                    if "scene_id" in res and "episode_id" in res:
                        done_res.add((res["scene_id"], str(res["episode_id"])))

        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split("/")[-2]
            process_bar = tqdm.tqdm(
                range(len(episodes[idx::self.env_num])),
                desc=f"scene {scene_id}",
            )

            for episode in episodes[idx::self.env_num]:
                episode_instruction = (
                    episode.instruction.instruction_text
                    if "objectnav" not in self.config_path
                    else episode.object_category
                )
                episode_id = str(episode.episode_id)

                if (scene_id, episode_id) in done_res:
                    process_bar.update(1)
                    continue

                self.model.reset_for_env(idx)
                env.current_episode = episode
                observations = env.reset()
                self.context_manager.reset(episode_instruction)
                
                system_prompt = build_system_prompt(episode_instruction, self.args.num_future_steps)

                vis_frames = []
                step_id = 0

                
                # ==========================================================
                # At step 0, select the correct direction
                # ==========================================================
                num_views = 12
                steps_per_view = 24 // num_views  # 15°
                
                valid_actions = ["STOP", "↑", "←", "→"]
                action_token_ids = []
                for act in valid_actions:
                    token_id = self.processor.tokenizer(act, add_special_tokens=False).input_ids[0]
                    action_token_ids.append(token_id)
                
                # Index of "↑" action is 1
                forward_idx_in_subset = 1
                stop_idx_in_subset = 0
                
                best_direction_idx = 0
                max_forward_prob = -float('inf')
                
                for view_idx in range(num_views):
                    if env.episode_over:
                        break
                        
                    current_image = Image.fromarray(observations["rgb"]).convert("RGB")
                    
                    sniff_prefix = get_round_prefix(0, 0)
                    sniff_text = build_user_text(episode_instruction, sniff_prefix, self.args.num_future_steps, 0)
                    sniff_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "image", "image": current_image},
                            {"type": "text", "text": sniff_text}
                        ]}
                    ]
                    
                    inputs = self._prepare_inputs(sniff_messages, [current_image])
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        next_token_logits = outputs.logits[0, -1, :]
                        
                        subset_logits = next_token_logits[action_token_ids]
                        
                        subset_probs = torch.nn.functional.softmax(subset_logits, dim=0)
                        
                        forward_prob = subset_probs[forward_idx_in_subset].item()
                    
                    if forward_prob > max_forward_prob:
                        max_forward_prob = forward_prob
                        best_direction_idx = view_idx
                        
                    for _ in range(steps_per_view):
                        if env.episode_over:
                            break
                        observations = env.step(3)  # Index of "→" is 3
                        step_id += 1
                        
                        info = env.get_metrics()
                        if info["top_down_map"] is not None:
                            frame = observations_to_image({"rgb": observations["rgb"]}, info)
                            vis_frames.append(frame)

                # ----------------------------------------------------------
                # Turn to the correct direction
                # ----------------------------------------------------------
                optimal_steps = best_direction_idx * steps_per_view
                for _ in range(optimal_steps):
                    if env.episode_over:
                        break
                    observations = env.step(3)
                    step_id += 1
                    
                    info = env.get_metrics()
                    if info["top_down_map"] is not None:
                        frame = observations_to_image({"rgb": observations["rgb"]}, info)
                        vis_frames.append(frame)
                
                # ==========================================================
                #  Start Navigation
                # ==========================================================
                self.context_manager.reset(episode_instruction)
                logical_step_id = 0
                consecutive_non_forward = 0
 
                while not env.episode_over:
                    # ----------------------------------------------------------
                    # 1) Observe ONLY at chunk boundary
                    # ----------------------------------------------------------
                    image = Image.fromarray(observations["rgb"]).convert("RGB")
                    self.context_manager.observe(image, logical_step_id)

                    info = env.get_metrics()
                    if info["top_down_map"] is not None:
                        frame = observations_to_image({"rgb": observations["rgb"]}, info)
                        vis_frames.append(frame)

                    # ----------------------------------------------------------
                    # 2) Reconstruct the full prompt and predict ONE full chunk
                    # ----------------------------------------------------------
                    messages, images = self.context_manager.build_messages_and_images(system_prompt)
                    assistant_text, action_seq, logits = self._predict_actions(messages, images)

                    # ----------------------------------------------------------
                    # 3) Execute the predicted chunk open-loop
                    # ----------------------------------------------------------

                    #  if stuck
                    if consecutive_non_forward >= 12:
                      if action_seq == [2,2]:
                        action_seq = [2,1]
                      if action_seq == [3,3]:
                        action_seq = [3,1]
                        
                    elif consecutive_non_forward >= 24:
                        action_seq == [0, 0]
     
                    assistant_text = actions_to_text(action_seq)
                    self.context_manager.add_round_result(assistant_text)
                          
                    
                    for action in action_seq:
                        if env.episode_over:
                            break

                        observations = env.step(action)
                        step_id += 1            
                        logical_step_id += 1   
                        
                        if action == 1:  # "↑" is 1
                            consecutive_non_forward = 0
                        else:
                            consecutive_non_forward += 1

                        info = env.get_metrics()
                        if info["top_down_map"] is not None:
                            frame = observations_to_image({"rgb": observations["rgb"]}, info)
                            vis_frames.append(frame)

                process_bar.update(1)
                metrics = env.get_metrics()

                if self.save_video:
                    os.makedirs(os.path.join(self.output_path, f"vis_{self.epoch}"), exist_ok=True)
                    images_to_video(
                        vis_frames,
                        os.path.join(self.output_path, f"vis_{self.epoch}"),
                        f"{scene_id}_{episode_id}",
                        fps=6,
                        quality=9,
                    )

                try:
                  metrics["ndtw"]
                except:
                  metrics["ndtw"] = 0
	
                if not math.isfinite(metrics["spl"]) or not math.isfinite(metrics["distance_to_goal"]):
                  continue
                 
                sucs.append(metrics["success"])
                spls.append(metrics["spl"])
                oss.append(metrics["oracle_success"])
                nes.append(metrics["distance_to_goal"])
                ndtws.append(metrics["ndtw"])

                record = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics["oracle_success"],
                    "ne": metrics["distance_to_goal"],
                    "ndtw": metrics["ndtw"],
                    "steps": step_id,
                }

                with open(result_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                done_res.add((scene_id, episode_id))

        if get_rank() == 0 and len(sucs) > 0:
            print(
                f"[{self.split}] SR={np.mean(sucs):.4f}, "
                f"SPL={np.mean(spls):.4f}, "
                f"OS={np.mean(oss):.4f}, "
                f"NE={np.mean(nes):.4f}"
                f"nDTW={np.mean(ndtws):.4f}"
            )

        env.close()

        return (
            torch.tensor(sucs, dtype=torch.float32, device=self.device),
            torch.tensor(spls, dtype=torch.float32, device=self.device),
            torch.tensor(oss, dtype=torch.float32, device=self.device),
            torch.tensor(nes, dtype=torch.float32, device=self.device),
            torch.tensor(ndtws, dtype=torch.float32, device=self.device),
            torch.tensor([len(sucs)], dtype=torch.long, device=self.device),  # ep_num
        )
    

def evaluate(model, processor, args):
    model.eval()
    world_size = get_world_size()
    model.reset(world_size)

    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        processor=processor,
        epoch=0,
        args=args,
    )

    sucs, spls, oss, nes, ndtws, ep_num = evaluator.eval_action(get_rank(), args)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    sucs_list = sucs.cpu().tolist()
    spls_list = spls.cpu().tolist()
    oss_list = oss.cpu().tolist()
    nes_list = nes.cpu().tolist()
    ndtws_list = ndtws.cpu().tolist()

    sucs_all_list = [None for _ in range(world_size)]
    spls_all_list = [None for _ in range(world_size)]
    oss_all_list = [None for _ in range(world_size)]
    nes_all_list = [None for _ in range(world_size)]
    ndtws_all_list = [None for _ in range(world_size)]

    dist.all_gather_object(sucs_all_list, sucs_list)
    dist.all_gather_object(spls_all_list, spls_list)
    dist.all_gather_object(oss_all_list, oss_list)
    dist.all_gather_object(nes_all_list, nes_list)
    dist.all_gather_object(ndtws_all_list, ndtws_list)

    sucs_all = [item for sublist in sucs_all_list for item in sublist]
    spls_all = [item for sublist in spls_all_list for item in sublist]
    oss_all = [item for sublist in oss_all_list for item in sublist]
    nes_all = [item for sublist in nes_all_list for item in sublist]
    ndtws_all = [item for sublist in ndtws_all_list for item in sublist]

    result_all = {
        "sucs_all": float(sum(sucs_all) / len(sucs_all)) if len(sucs_all) > 0 else 0.0,
        "spls_all": float(sum(spls_all) / len(spls_all)) if len(spls_all) > 0 else 0.0,
        "oss_all": float(sum(oss_all) / len(oss_all)) if len(oss_all) > 0 else 0.0,
        "nes_all": float(sum(nes_all) / len(nes_all)) if len(nes_all) > 0 else 0.0,
        "ndtws_all": float(sum(ndtws_all) / len(ndtws_all)) if len(ndtws_all) > 0 else 0.0,
        "length": len(sucs_all),
    }

    if get_rank() == 0:
        os.makedirs(args.output_path, exist_ok=True)

        merged_records = _merge_rank_result_files(args.output_path, world_size)
        result_all["merged_length"] = len(merged_records)

        print(result_all)

        with open(os.path.join(args.output_path, "result.json"), "a", encoding="utf-8") as f:
            f.write(json.dumps(result_all, ensure_ascii=False) + "\n")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--habitat_config_path", type=str, default="config/vln_r2r.yaml")
    parser.add_argument("--eval_split", type=str, default="val_unseen")
    parser.add_argument("--output_path", type=str, default="./results/val_unseen/vln")
    parser.add_argument("--num_sparse_history_frames", type=int, default=48)
    parser.add_argument("--num_recent_observation_frames", type=int, default=16)
    parser.add_argument("--num_future_steps", type=int, default=2)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--port", default="1111")
    parser.add_argument("--dist_url", default="env://")

    args = parser.parse_args()
    init_distributed_mode(args)

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = VLNForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    evaluate(model, processor, args)


if __name__ == "__main__":
    eval()