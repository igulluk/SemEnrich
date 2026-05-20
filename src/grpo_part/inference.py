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


import argparse
import json
import logging
import os
import sys
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, set_seed
from peft import PeftModel

from qwen_vl_utils import process_vision_info

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

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on test dataset with fine-tuned VLM")
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model checkpoint (adapter for PEFT or full model)"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Base model name/path (used for loading PEFT adapters)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=".",
        help="Path to the JSON dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./inference_results.json",
        help="Path to save the inference results JSON"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (1 recommended for VLMs)"
    )
    parser.add_argument(
        "--is_peft",
        type=lambda x: x.lower() == 'true',
        default=True,
        help="Whether the checkpoint is a PEFT/LoRA adapter (default: True)"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--trust_remote_code",
        type=lambda x: x.lower() == 'true',
        default=True,
        help="Whether to trust remote code"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (None for all)"
    )
    parser.add_argument(
        "--sft_checkpoint_path",
        type=str,
        default=None,
        help="Path to SFT LoRA checkpoint (required for GRPO inference, as GRPO was trained on top of merged SFT model)"
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=32 * 28 * 28,  # ~32 visual tokens min
        help="Minimum number of pixels for the image"
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=128 * 28 * 28,  # ~128 visual tokens max
        help="Maximum number of pixels for the image"
    )
    
    return parser.parse_args()


def prepare_messages(example, include_ground_truth=False):
    """
    Prepare messages for the model from a dataset example.
    Returns messages for generation (without assistant response).
    """
    messages = []
    SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the findings in the image and then provides the user with the final impression. The findings "
    "and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> findings here </think><answer> impression here </answer>"
        )
    # Add system prompt
    if "system" in example and example["system"]:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": example["system"]}],
        })
    else:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        })
    
    # Add user message with image and text
    problem = "Given the provided medical image, analyze it carefully and generate findings and impression."
    findings = example.get("Findings")
    impression = example.get("Impression")
    solution = f"<think> {findings} </think><answer> {impression} </answer>"
    image = example.get("image")
    
    user_content = [{"type": "text", "text": problem}]
    if image:
        user_content.append({"type": "image", "image": image})
    
    messages.append({
        "role": "user",
        "content": user_content,
    })
    
    return messages


def load_model_and_processor(args):
    """Load the model and processor based on configuration."""
    
    logger.info(f"Loading processor from {args.base_model_name}")
    processor = AutoProcessor.from_pretrained(
        args.base_model_name,
        trust_remote_code=args.trust_remote_code,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    logger.info(f"Processor loaded with min_pixels={args.min_pixels}, max_pixels={args.max_pixels}")
    
    # Set pad token if needed
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor, "tokenizer") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Determine torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)
    
    # Get the appropriate model class based on model name
    model_class = get_qwen_vl_model_class(args.base_model_name)
    
    # Prepare model kwargs
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if args.device == "cuda" else None,
    )
    
    if args.is_peft:
        # Load base model first
        logger.info(f"Loading base model from {args.base_model_name}")
        base_model = model_class.from_pretrained(
            args.base_model_name,
            **model_kwargs,
        )
        
        # If SFT checkpoint is provided, merge it first (required for GRPO checkpoints)
        # This is because GRPO LoRA was trained on top of (base + SFT merged)
        if args.sft_checkpoint_path is not None:
            logger.info(f"Loading SFT adapter from {args.sft_checkpoint_path}")
            base_model = PeftModel.from_pretrained(base_model, args.sft_checkpoint_path)
            base_model = base_model.merge_and_unload()
            logger.info("SFT adapter merged with base model")
        
        logger.info(f"Loading PEFT adapter from {args.checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
        model = model.merge_and_unload()  # Merge for faster inference
        logger.info("PEFT adapter merged with base model")
    else:
        # Load full fine-tuned model directly
        logger.info(f"Loading full model from {args.checkpoint_path}")
        model = model_class.from_pretrained(
            args.checkpoint_path,
            **model_kwargs,
        )
    
    model.eval()
    
    if args.device != "cuda" or not hasattr(model, 'hf_device_map'):
        model = model.to(args.device)
    
    return model, processor


def generate_single(model, processor, example, args):
    """Generate output for a single example."""
    
    # Prepare messages for generation
    messages = prepare_messages(example)
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Prepare inputs
    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
    )
    
    # Move inputs to device
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        # generated_ids = model.generate(
        #     **inputs,
        #     max_new_tokens=args.max_new_tokens,
        #     do_sample=False,  # Greedy decoding for reproducibility
        #     pad_token_id=processor.tokenizer.pad_token_id,
        # )
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,  # Lower = more focused, Higher = more random
            top_p=0.9,  # Nucleus sampling
            top_k=50,   # Top-k sampling
            pad_token_id=processor.tokenizer.pad_token_id,
    )
    # Decode only the generated tokens (exclude input tokens)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_len:]
    generated_text = processor.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]
    
    return generated_text


def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("=" * 50)
    logger.info("Inference Configuration:")
    logger.info(f"  Checkpoint: {args.checkpoint_path}")
    logger.info(f"  Base Model: {args.base_model_name}")
    logger.info(f"  SFT Checkpoint: {args.sft_checkpoint_path}")
    logger.info(f"  Data Path: {args.data_path}")
    logger.info(f"  Output Path: {args.output_path}")
    logger.info(f"  Is PEFT: {args.is_peft}")
    logger.info(f"  Max New Tokens: {args.max_new_tokens}")
    logger.info(f"  Device: {args.device}")
    logger.info("=" * 50)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")
    raw_dataset = load_dataset('json', data_files=args.data_path)
    
    # Filter test split
    test_dataset = raw_dataset['train'].filter(lambda x: x['split'] == 'test')
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Limit samples if specified
    if args.num_samples is not None:
        test_dataset = test_dataset.select(range(min(args.num_samples, len(test_dataset))))
        logger.info(f"Limited to {len(test_dataset)} samples")
    
    # Load model and processor
    model, processor = load_model_and_processor(args)
    logger.info("Model and processor loaded successfully")
    
    # Run inference
    results = []
    
    for idx, example in enumerate(tqdm(test_dataset, desc="Generating")):
        try:
            generated_text = generate_single(model, processor, example, args)
            findings = example.get("Findings")
            impression = example.get("Impression")
            solution = f"<think> {findings} </think><answer> {impression} </answer>"
    
            result = {
                "idx": idx,
                "image": example.get("image", ""),
                "ground_truth": solution,
                "generated": generated_text,
            }
            
            # Include any additional fields from the original example
            for key in example.keys():
                if key not in ["Findings", "Impression", "image", "split", "system"]:
                    result[key] = example[key]
            
            results.append(result)
            
            # # Log progress every 10 samples
            # if (idx + 1) % 10 == 0:
            #     logger.info(f"Processed {idx + 1}/{len(test_dataset)} samples")
                
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            findings = example.get("Findings")
            impression = example.get("Impression")
            solution = f"<think> {findings} </think><answer> {impression} </answer>"
            results.append({
                "idx": idx,
                "Findings": findings,
                "Impression": impression,
                "image": example.get("image", ""),
                "ground_truth": solution,
                "generated": f"ERROR: {str(e)}",
                "error": True,
            })
    
    # Save results
    logger.info(f"Saving results to {args.output_path}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Inference complete! Results saved to {args.output_path}")
    logger.info(f"Total samples processed: {len(results)}")
    
    # Print summary
    error_count = sum(1 for r in results if r.get("error", False))
    if error_count > 0:
        logger.warning(f"Errors encountered: {error_count}/{len(results)}")
    
    # Print a sample result
    if results:
        logger.info("\n" + "=" * 50)
        logger.info("Sample result:")
        sample = results[0]
        logger.info(f"ground_truth: {sample['ground_truth'][:200]}...")
        logger.info(f"Generated: {sample['generated'][:200]}...")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()

