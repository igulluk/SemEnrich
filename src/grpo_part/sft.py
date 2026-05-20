# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys

import datasets
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, set_seed, AutoProcessor, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import dataclass, field
from typing import Optional
from qwen_vl_utils import process_vision_info

# Import Qwen VL model classes for different versions
from transformers import Qwen2VLForConditionalGeneration

# Try to import Qwen2.5-VL (requires transformers >= 4.45.0)
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    QWEN2_5_VL_AVAILABLE = True
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None
    QWEN2_5_VL_AVAILABLE = False

# Try to import Qwen3-VL (requires newer transformers version when available)
try:
    from transformers import Qwen3VLForConditionalGeneration
    QWEN3_VL_AVAILABLE = True
except ImportError:
    Qwen3VLForConditionalGeneration = None
    QWEN3_VL_AVAILABLE = False


logger = logging.getLogger(__name__)

# Global variables set in main()
processor = None
max_seq_length = None


def is_qwen3_vl_model(model_name_or_path: str) -> bool:
    """Check if the model is a Qwen3-VL model."""
    model_name_lower = model_name_or_path.lower()
    return "qwen3-vl" in model_name_lower or "qwen3_vl" in model_name_lower


def get_qwen_vl_model_class(model_name_or_path: str):
    """
    Determine the appropriate Qwen VL model class based on the model name/path.
    
    Args:
        model_name_or_path: The model identifier or path
        
    Returns:
        The appropriate model class for the Qwen VL version
    """
    model_name_lower = model_name_or_path.lower()
    
    if "qwen3-vl" in model_name_lower or "qwen3_vl" in model_name_lower:
        if not QWEN3_VL_AVAILABLE:
            raise ImportError(
                "Qwen3-VL model requested but Qwen3VLForConditionalGeneration is not available. "
                "Please upgrade transformers to a version that supports Qwen3-VL."
            )
        logger.info(f"Using Qwen3VLForConditionalGeneration for model: {model_name_or_path}")
        return Qwen3VLForConditionalGeneration
    elif "qwen2.5-vl" in model_name_lower or "qwen2_5-vl" in model_name_lower or "qwen2.5_vl" in model_name_lower:
        if not QWEN2_5_VL_AVAILABLE:
            raise ImportError(
                "Qwen2.5-VL model requested but Qwen2_5_VLForConditionalGeneration is not available. "
                "Please upgrade transformers to version >= 4.45.0."
            )
        logger.info(f"Using Qwen2_5_VLForConditionalGeneration for model: {model_name_or_path}")
        return Qwen2_5_VLForConditionalGeneration
    else:
        # Default to Qwen2-VL
        logger.info(f"Using Qwen2VLForConditionalGeneration for model: {model_name_or_path}")
        return Qwen2VLForConditionalGeneration


@dataclass
class CustomScriptArguments(ScriptArguments):
    data_json_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the JSON data file"}
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={"help": "Stop training if eval_loss doesn't improve for this many evaluations. Set to None to disable."}
    )

@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})


def convert_example(example):
    """
    correct example into "messages" 
    eg:
    {
      "system": "You are a helpful assistant.",
      "conversations": [
          {"from": "user", "value": "How many objects are included in this image?",
           "image_path": "/path/to/image.png"},
          {"from": "assistant", "value": "<think>\nI can see 10 objects\n</think>\n<answer>\n10\n</answer>"}
      ]
    }
    """
    messages = []
    if "system" in example:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": example["system"]}],
        })
    else:
    #     SYSTEM_PROMPT = (
    # "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    # "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    # "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    # "<think> reasoning process here </think><answer> answer here </answer>"
    #     )
        SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the findings in the image and then provides the user with the final impression. The findings "
    "and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> findings here </think><answer> impression here </answer>"
        )
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        })

    problem = "Given the provided medical image, analyze it carefully and generate findings and impression."
    findings = example.get("Findings")
    impression = example.get("Impression")
    solution = f"<think> {findings} </think><answer> {impression} </answer>"
    image = example.get("image")
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": problem},
            {"type": "image", "image": image},
            ]
    })
    messages.append({
        "role": "assistant",
        "content": f"\n\n{solution}",
    })
    example["messages"] = messages
    return example


def collate_fn(examples):
    texts = [
        processor.apply_chat_template(convert_example(example)["messages"], tokenize=False, add_generation_prompt=True)
        for example in examples
    ]
    image_inputs = []
    for example in examples:
        imgs, vids = process_vision_info(example["messages"])
        image_inputs.append(imgs)
    
    # Use max_seq_length from training args for consistent memory usage
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding="max_length",      # Pad to max_length for consistent batch sizes
        truncation=True,           # Truncate sequences longer than max_length
        max_length=max_seq_length, # From --max_seq_length command line arg
    )
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################

    # Load your custom JSON dataset
    print(f'script_args.data_json_path : {script_args.data_json_path}')
    raw_dataset = load_dataset('json', data_files=script_args.data_json_path)
    

    # Split into train and test based on the 'split' column
    train_dataset = raw_dataset['train'].filter(lambda x: x['split'] == 'train')
    test_dataset = raw_dataset['train'].filter(lambda x: x['split'] == 'test')

    # Create a DatasetDict with the same structure
    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset  # or you might want to keep only 'train' if your original code only used that
    })

    ################
    # Load tokenizer
    ################
    global processor, max_seq_length
    max_seq_length = training_args.max_seq_length
    logger.info(f"Using max_seq_length: {max_seq_length}")
    
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, 
            trust_remote_code=model_args.trust_remote_code,
            min_pixels=32 * 28 * 28,    # Minimum image size
            max_pixels=128 * 28 * 28,    # Maximum image size (reduces tokens)
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # training_args.model_init_kwargs = model_kwargs
    model_class = get_qwen_vl_model_class(model_args.model_name_or_path)
    
    # Qwen3-VL doesn't support use_cache in __init__, so remove it
    if is_qwen3_vl_model(model_args.model_name_or_path):
        model_kwargs.pop("use_cache", None)
        logger.info("Removed 'use_cache' from model_kwargs for Qwen3-VL compatibility")

    model = model_class.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False
    
    # Setup callbacks
    callbacks = []
    if script_args.early_stopping_patience is not None:
        if not training_args.load_best_model_at_end:
            logger.warning(
                "Early stopping requires --load_best_model_at_end true. Enabling it automatically."
            )
            training_args.load_best_model_at_end = True
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=script_args.early_stopping_patience))
        logger.info(f"Early stopping enabled with patience={script_args.early_stopping_patience}")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        callbacks=callbacks if callbacks else None,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["R1-V"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
        processor.push_to_hub(training_args.hub_model_id)




if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    model_args.lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    model_args.use_peft = True
    main(script_args, training_args, model_args)
