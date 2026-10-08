# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Union

from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
)

import trl
from trl.experimental.orpo import ORPOConfig as _ORPOConfig


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)


class H4ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(
        self, yaml_arg: str, other_args: Optional[List[str]] = None
    ) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))
        # print("Parsed YAML:", arg_list, os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {
            arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args
        }
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type is bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(
                            f"Duplicate argument provided: {arg}, may cause unexpected behavior"
                        )
            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self):
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(
                os.path.abspath(sys.argv[1]), sys.argv[2:]
            )
        # parse command line args only
        else:
            raise RuntimeError("Do not support commandline args.")
            # output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]

        return output


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    base_model_revision: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The base model checkpoint for weights initialization with PEFT adapters."
            )
        },
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    model_code_revision: str = field(
        default=None, metadata={"help": "The branch of the IFT model"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The path to the tokenizer. Useful if you want to use a different tokenizer to the one stored in `model_name_or_path`."
            )
        },
    )

    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use; you can use --attn_implementation=flash_attention_2, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    lora_target_modules: Union[List[str], str] = field(
        default="all",
        metadata={"help": ("LoRA target modules.")},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "use 8 bit precision"})
    load_in_4bit: bool = field(default=False, metadata={"help": "use 4 bit precision"})

    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(
        default=False, metadata={"help": "use nested quantization"}
    )
    bnb_4bit_quant_storage: Optional[str] = field(
        default="uint8",
        metadata={"help": "storage type to pack the quanitzed 4-bit prarams."},
    )
    # use_flash_attention_2: bool = field(default=False)

    # max position embedding scaling
    max_position_embeddings: Optional[int] = field(
        default=None,
        metadata={"help": ("The maximum position embeddings to use.")},
    )

    rope_scaling_method: Optional[str] = field(
        default="yarn",
        metadata={
            "help": (
                "The method to use for RoPE scaling. If not set, the default method will be used."
            )
        },
    )

    rope_scaling_factor: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "The scaling factor to use for the RoPE embeddings. If not set, the default scaling factor will be used."
            )
        },
    )

    train_last_step_only: bool = field(
        default=False,
        metadata={
            "help": (
                "For looped/Universal-Transformer models (e.g. Ouro) that return per-step "
                "hidden states: compute logits/loss from only the final step instead of the "
                "model's native multi-step weighted combination. No-op for standard models."
            )
        },
    )

    freeze_param_patterns: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "Substrings; any parameter whose name contains one of these is frozen "
                "(requires_grad_(False)) before training."
            )
        },
    )

    huginn_forward_patch: bool = field(
        default=False,
        metadata={
            "help": (
                "For Huginn-style depth-recurrent CausalLMs (e.g. tomg-group-umd/huginn-0125) "
                "whose forward() does not shift logits/labels internally: patch forward to "
                "drop attention_mask/position_ids (unsupported for batch>1 in the upstream "
                "modeling code), shift logits/labels the standard way at loss time, and skip "
                "returning latent_states. No-op for standard models."
            )
        },
    )

    huginn_mean_recurrence: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Override model.config.mean_recurrence before training (Huginn only). The "
                "randomized depth sampler always stays on during training -- this only "
                "re-centers it, e.g. set to 8 if you plan to serve cheaply at num_steps=8. "
                "Leave unset to keep the pretrained default (32, matching full-depth serving)."
            )
        },
    )

    huginn_mean_backprop_depth: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Override model.config.mean_backprop_depth before training (Huginn only): "
                "the truncated-backprop window within the randomized depth sampler."
            )
        },
    )

    huginn_fixed_num_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Force this exact num_steps on every forward (train and eval), bypassing "
                "the randomized depth sampler entirely -- every iteration runs with "
                "gradient, with zero depth variance. This deviates from how Huginn was "
                "pretrained and from the official finetuning script, which never fixes "
                "num_steps. Mutually exclusive in effect with huginn_mean_recurrence/"
                "huginn_mean_backprop_depth: if set, those two become dead config since "
                "the sampler they'd re-center never runs. Leave unset to keep the "
                "stochastic-depth training behavior."
            )
        },
    )

    def __post_init__(self):
        # although HF support other methods, we use yarn since it is the most efficient method than dynamic or
        if self.rope_scaling_factor:
            if self.rope_scaling_method == "llama3":
                self.rope_scaling = {
                    "type": self.rope_scaling_method,
                    "factor": self.rope_scaling_factor,
                    "original_max_position_embeddings": 8192,
                    "high_freq_factor": 4.0,
                    "low_freq_factor": 1.0,
                }
            else:
                self.rope_scaling = {
                    "type": self.rope_scaling_method,
                    "factor": self.rope_scaling_factor,
                    "original_max_position_embeddings": 8192,
                }
            self.max_position_embeddings = self.rope_scaling_factor * 8192

        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    chat_template: Optional[str] = field(
        default=None, metadata={"help": "The chat template to use."}
    )
    dataset_mixer: Optional[Dict[str, float]] = field(
        default=None,
        metadata={
            "help": ("Datasets and their proportions to be used for training ift/rl.")
        },
    )
    text_column: Optional[str] = field(
        default="text",
        metadata={
            "help": "The column name to use for the text in the dataset (only used for continued pretraining)."
        },
    )
    dataset_splits: Optional[List[str]] = field(
        default_factory=lambda: ["train", "test"],
        metadata={"help": ("List of train test splits to use in the dataset")},
    )
    dataset_configs: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of dataset config names. If given must be the same length as 'dataset_mixer' keys."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )
    auto_insert_empty_system_msg: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to automatically insert an empty system message as the first message if `system` is mentioned in the chat template."
            )
        },
    )
    additional_special_tokens: Optional[Dict[str, str]] = field(
        default=None, metadata={"help": "special tokens add to the tokenizer."}
    )
    special_tokens: Optional[Dict[str, str]] = field(
        default=None,
        metadata={
            "help": (
                "Directly remap tokenizer special-token attributes (e.g. bos_token, "
                "eos_token, unk_token, pad_token) to tokens that already exist in the "
                "vocab. Unlike additional_special_tokens, this does not add new tokens "
                "or resize embeddings."
            )
        },
    )
    additional_non_special_tokens: Optional[List[str]] = field(
        default=None, metadata={"help": "new tokens add to the tokenizer."}
    )
    role_map: Optional[Dict[str, str]] = field(
        default=None,
        metadata={
            "help": (
                "Rename message roles (e.g. {'assistant': 'Huginn'}) before applying the "
                "tokenizer's chat template. Needed for models whose chat template uses a "
                "non-standard assistant role name."
            )
        },
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments
    Also used for the continued pretraining task.
    """

    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={
            "help": ("Whether to log and evaluate the first global_step or not.")
        },
    )
    prompt_loss_weight: float = field(default=0.1)
    use_plw: bool = field(default=False)
    use_plw_sample_template: bool = field(default=False)
    # Override transformers TrainingArguments.use_cache to default False (matching TRL's std
    # SFTTrainer, which forces use_cache=False at forward time). The KV cache is never reused
    # during full-sequence training passes, and recurrent / Universal-Transformer models like
    # Ouro make it a 4x per-UT-step memory multiplier. PLWTrainer honors this at forward time
    # and additionally forces it off under any activation/gradient checkpointing.
    use_cache: bool = field(
        default=False,
        metadata={
            "help": (
                "KV cache during training. Defaults to False (matches TRL's std SFTTrainer, "
                "which forces use_cache=False at forward time) -- the cache is never reused "
                "during full-sequence training passes, and recurrent / Universal-Transformer "
                "models like Ouro make it a 4x per-UT-step memory multiplier. PLWTrainer "
                "honors this field at forward time and additionally forces it off under any "
                "activation/gradient checkpointing."
            )
        },
    )
    # Qwen3 thinking mode. True (default) keeps thinking on (the model's pretrained
    # behavior); set to false in a recipe to suppress reasoning -- the chat template then
    # inserts an empty thinking/answer block at generation prompts so the model answers
    # directly. Only affects generation-prompt rendering (add_generation_prompt=True);
    # loss-based SFT training/eval (add_generation_prompt=False) already uses Qwen3's
    # empty-think-block no-think format regardless, so this flag is a no-op on the SFT
    # data and only matters for generation-based eval / inference.
    enable_thinking: bool = field(
        default=True,
        metadata={
            "help": (
                "Qwen3 thinking mode. True keeps thinking on (pretrained behavior); set false "
                "in a recipe to suppress reasoning (empty think block at generation prompts). "
                "Only affects generation-prompt rendering; loss-based SFT data is unchanged."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum sequence length of the model. Sequences longer than this will be truncated, so using a larger value should be used with caution."
            )
        },
    )
    # PLW prompt/completion boundary. False (default): completion is the full assistant
    # turn (header + think block + answer + eos) -- the model is trained to generate the
    # whole turn. True: the prompt is built with the generation prompt
    # (add_generation_prompt=True), so for Qwen3 no-think the assistant header + empty
    # think block are pre-filled in the prompt and the completion is answer+eos only --
    # exactly matching the inference generation target. The full token sequence is
    # identical either way; only the loss-weight boundary (prompt vs completion) moves.
    # Generalizes to think-on mode (completion = think + answer). No effect when
    # use_plw is false.
    plw_match_inference_boundary: bool = field(
        default=False,
        metadata={
            "help": (
                "PLW prompt/completion boundary. False (default): completion = full "
                "assistant turn (model generates the whole turn). True: prompt pre-fills "
                "the generation prompt (assistant header +, for Qwen3 no-think, the empty "
                "think block) and completion = answer+eos only, matching inference. Only "
                "used when use_plw is true."
            )
        },
    )
    # How PLW handles examples whose prompt+completion length exceeds the length cap
    # (args.max_length, resolved from max_seq_length). "drop" (default): skip over-length
    # examples (clean signal, but skews the mix if a hard task is over-represented -- run the
    # per-task length audit before committing). "truncate_right": keep the first max_len
    # tokens (drop the tail), parity with std SFTTrainer keep_start; preserves the system
    # prompt + task instruction, may lose the completion tail (and drops the example if the
    # completion would be entirely trimmed). "error": raise so the run fails loudly.
    plw_overlength_strategy: str = field(
        default="drop",
        metadata={
            "help": (
                "PLW over-length example handling: 'drop' (skip), 'truncate_right' (keep "
                "first max_len tokens, std keep_start parity), or 'error'. Informed by the "
                "per-task length audit; only used when use_plw is true."
            )
        },
    )
    use_weighted_json_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Weight per-token CE by decision-vs-scaffold class for JSON-formatted "
                "targets (med multitask mix), using the ported schema in "
                "med_target_schema.py. Mutually exclusive with use_plw."
            )
        },
    )
    # Weight applied to scaffold tokens (mandatory JSON punctuation, required-key names).
    # Decision tokens (values, optional-key presence, array continue/stop, the stop token)
    # always get weight 1.0. A control run with this set to 1.0 reproduces plain CE on
    # canonicalized targets, isolating the weighting effect from the canonicalization one.
    weighted_json_scaffold_weight: float = field(
        default=0.1,
        metadata={
            "help": (
                "Loss weight for scaffold-class tokens under use_weighted_json_loss "
                "(decision-class tokens always get weight 1.0). Set to 1.0 for a plain-CE "
                "control run on the same canonicalized-target data."
            )
        },
    )
    weighted_json_overlength_strategy: str = field(
        default="drop",
        metadata={
            "help": (
                "Over-length example handling under use_weighted_json_loss: 'drop' (skip), "
                "'truncate_right' (keep first max_len tokens), or 'error'."
            )
        },
    )
    # Log token accuracy every N train steps (eval always logs) to avoid a cross-rank
    # gather_for_metrics sync every step. 0 or 1 => every step. Guarded against 0.
    plw_accuracy_log_steps: int = field(
        default=50,
        metadata={
            "help": (
                "Log PLW mean_token_accuracy every N train steps (eval always logs). "
                "0 or 1 => every step."
            )
        },
    )
    kl_coef: float = field(default=0.02)
    kl_start_step: int = field(
        default=0,
        metadata={
            "help": (
                "Step at which to start applying the KL penalty. Recommend to start after warmup."
            )
        },
    )
    kl_warmup_steps: int = field(
        default=100,
        metadata={
            "help": (
                "Linearly ramp KL coef from 0 to target over this many steps after kl_start_step"
            )
        },
    )

    asft_sync_original_model: bool = field(
        default=True,
    )
    asft_kl_weight: float = field(
        default=0.1,
        metadata={"help": ("The KL weight for ASFT trainer.")},
    )
    asft_clip_min: float = field(
        default=0.1,
        metadata={"help": ("The minimum clipping value for ASFT trainer.")},
    )
    asft_clip_max: float = field(
        default=2.0,
        metadata={"help": ("The maximum clipping value for ASFT trainer.")},
    )


@dataclass
class ORPOConfig(_ORPOConfig):
    """
    Arguments related to the ORPO training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments
    """

    use_fix_overflow_patch: bool = field(
        default=True,
    )
    plw_weight: Optional[float] = field(
        default=0.0,
    )


@dataclass
class DPOConfig(trl.DPOConfig):
    """
    Arguments related to the DPO training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments
    """

    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={
            "help": ("Whether to log and evaluate the first global_step or not.")
        },
    )
    optim: Optional[str] = field(default="rmsprop")
    remove_unused_columns: bool = field(default=False)
    max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, so using a larger value should be used with caution."
            )
        },
    )
    positive_lambda: float = field(
        default=0.0,
        metadata={
            "help": (
                "The positive lambda value for the DPO positive loss. This is a hyperparameter that controls the strength of the positive feedback in the DPO loss."
            )
        },
    )


@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    Arguments related to the DPO training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments
    """

    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={
            "help": ("Whether to log and evaluate the first global_step or not.")
        },
    )
    remove_unused_columns: bool = field(default=False)
    reward_functions: List[str] = field(default=None)
    max_length: Optional[float] = field(default=None)
    do_sft_to_grpo_conversion: bool = field(default=True)
    temperature_schedule: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": (
                "Multi-temperature rollout (MDP-GRPO): sample the num_generations completions per "
                "prompt at DIFFERENT temperatures -- one per bucket -- instead of the single "
                "self.temperature. Raises within-group reward dispersion so all-correct / all-wrong "
                "groups (zero advantage -> no gradient) occur less often -- the main failure mode "
                "for our deterministic 0/1 medical rewards. Only the vLLM colocate path supports "
                "this (Qwen3 recipes); HF .generate() and non-colocate fall back to the single global "
                "temperature. num_generations must be divisible by len(temperature_schedule); e.g. "
                "[0.2, 0.5, 0.8, 1.1] with num_generations: 8, or [0.3, 0.7, 1.1] with num_generations: 9."
            )
        },
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16
        # print("!!!", self.generation_batch_size, self.steps_per_generation)

        TrainingArguments.__post_init__(self)

        self.scale_rewards = {True: "group", False: "none"}.get(
            self.scale_rewards, self.scale_rewards
        )

        num_processes = self.world_size

        # print("@@@", self.generation_batch_size, self.steps_per_generation)

        self.steps_per_generation = self.generation_batch_size // (
            self.per_device_train_batch_size * num_processes
        )

        if self.do_eval and self.eval_strategy != "no":
            # Just ensure the value is divisible by the global batch size
            if (
                self.per_device_eval_batch_size * num_processes
            ) % self.num_generations != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by num_generations ({self.num_generations})."
                )

        # The generation batch must contain full prompt groups (no partials), so it must be divisible by
        # num_generations.
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )

        if self.num_generations < 2:
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )

        if self.delta is not None and self.use_liger_loss:
            raise ValueError("Liger loss does not support two-sided GRPO loss yet.")
