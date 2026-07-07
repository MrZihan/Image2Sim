import torch
from typing import List, Optional, Union, Tuple
from transformers import Qwen3VLForConditionalGeneration
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast


class VLNForCausalLM(Qwen3VLForConditionalGeneration):
    """
    Minimal VLN wrapper around Qwen3-VL.

    Design intent:
      - Keep the model interface compatible with VLN training / eval code.
      - Explicitly support message-level prompt reconstruction pipelines.

    Notes:
      - `use_cache=True` still works exactly as in the base HF model, i.e. it only
        enables cache usage inside the current forward / generation call.
      - This wrapper does not store, evict, or reuse `past_key_values` across
        rollout steps on behalf of the caller.
      - The step counters below are bookkeeping helpers for eval loops only.
    """

    # Older dataset / collator code may still pass these deprecated keys.
    FORWARD_LEGACY_KEYS = [
        "images",
        "depths",
        "poses",
        "intrinsics",
        "time_ids",
        "task_type",
        "modalities",
        "image_sizes",
    ]

    GENERATE_LEGACY_KEYS = [
        "images",
        "image_sizes",
        "depths",
        "poses",
        "intrinsics",
        "time_ids",
        "task_type",
    ]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        # Backward-compatible config field retained only as metadata.
        # It is not used for persistent cache control.
        self.num_history = getattr(config, "num_history", 8)

        # Per-environment rollout step counters for evaluation bookkeeping.
        self.rollout_step_counters: List[int] = []

    def reset(self, env_num: int) -> None:
        """Reset rollout bookkeeping for a vectorized evaluator."""
        self.rollout_step_counters = [0] * env_num

    def reset_for_env(self, env_idx: int) -> None:
        """Reset rollout bookkeeping for a single environment."""
        while len(self.rollout_step_counters) <= env_idx:
            self.rollout_step_counters.append(0)
        self.rollout_step_counters[env_idx] = 0

    @staticmethod
    def _pop_legacy_kwargs(kwargs, legacy_keys):
        for key in legacy_keys:
            kwargs.pop(key, None)
        return kwargs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Thin compatibility wrapper.

        Important:
          Passing `past_key_values` here behaves exactly like the base HF model
          for this single call only. This wrapper does not automatically manage
          cross-step cache state for rollout inference.
        """
        kwargs = self._pop_legacy_kwargs(kwargs, self.FORWARD_LEGACY_KEYS)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Thin generation wrapper used by the VLN evaluator.

        `env_id` is accepted only for rollout bookkeeping. It does not imply that
        this class is maintaining persistent per-environment KV caches.
        """
        env_id = kwargs.pop("env_id", None)
        kwargs = self._pop_legacy_kwargs(kwargs, self.GENERATE_LEGACY_KEYS)

        if env_id is not None and len(self.rollout_step_counters) > env_id:
            self.rollout_step_counters[env_id] += 1

        return super().generate(
            inputs=inputs,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )
