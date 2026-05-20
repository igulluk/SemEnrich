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

import logging
import sys

import os
import re
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import Qwen2VLForConditionalGeneration, EarlyStoppingCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint

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

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


from dataclasses import dataclass, field
from typing import Optional
from qwen_vl_utils import process_vision_info
from peft import PeftModel

# Global variables for cluster dictionaries (loaded once)
FINDINGS_SENTENCE_TO_CLUSTER = None
IMPRESSION_SENTENCE_TO_CLUSTER = None

# Global variables for cluster criticality scores (loaded once)
FINDINGS_CLUSTER_CRITICALITY = None
IMPRESSION_CLUSTER_CRITICALITY = None

logger = logging.getLogger(__name__)


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
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'bleu', 'semantic_cluster', 'exact_sentence_match', 'weighted_semantic_cluster'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "bleu", "semantic_cluster"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'bleu', 'semantic_cluster', 'exact_sentence_match', 'weighted_semantic_cluster'"},
    )
    max_pixels: Optional[int] = field(
        default=128 * 28 * 28,  # ~128 visual tokens max
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=32 * 28 * 28,   # ~32 visual tokens min
        metadata={"help": "Minimum number of pixels for the image"},
    )
    data_json_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the JSON data file"}
    )
    sft_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to an SFT LoRA checkpoint to initialize from (will be merged into base model)"}
    )
    findings_cluster_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the findings sentence-to-cluster JSON file (thought_cluster for findings)"}
    )
    impression_cluster_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the impression sentence-to-cluster JSON file (thought_cluster for impression)"}
    )
    findings_criticality_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the findings cluster criticality CSV file (columns: element, criticality_score)"}
    )
    impression_criticality_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the impression cluster criticality CSV file (columns: element, criticality_score)"}
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={"help": "Stop training if eval_reward doesn't improve for this many evaluations. Set to None to disable."}
    )


def accuracy_reward(completions, Findings, Impression, **kwargs):
    solution = [f"<think> {findings} </think><answer> {impression} </answer>" for findings, impression in zip(Findings, Impression)]
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        # import numpy as np 
        # if np.random.uniform()<5:
        #     print(f'content: \n')
        #     print(content)
        #     print(f'='*200)
        #     print(f'sol: \n')
        #     print(sol)
        #     print(f'='*200)
        #     print(f'*'*300)
        #     print(f'*'*300)

        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # Pattern accounts for optional leading whitespace/newlines (as trained in SFT with "\n\n" prefix)
    pattern = r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*"
    completion_contents = [completion[0]["content"] for completion in completions]
    # print(completion_contents)
    # print(f'='*150)
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def bleu_reward(completions, Findings, Impression, **kwargs):
    """
    Reward function that calculates sentence BLEU scores between:
    - Generated findings (from <think> tags) and ground truth Findings
    - Generated impression (from <answer> tags) and ground truth Impression
    
    Returns the sum of both BLEU scores as the reward (max 2.0).
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    # Use smoothing to handle cases with few matching n-grams
    smoothing = SmoothingFunction().method1
    
    for content, gt_findings, gt_impression in zip(contents, Findings, Impression):
        findings_bleu = 0.0
        impression_bleu = 0.0
        
        try:
            # Extract findings from <think> tags in generated content
            findings_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            generated_findings = findings_match.group(1).strip() if findings_match else ""
            
            # Extract impression from <answer> tags in generated content
            impression_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            generated_impression = impression_match.group(1).strip() if impression_match else ""
            
            # Tokenize by splitting on whitespace (simple tokenization)
            # For BLEU, reference should be a list of reference sentences (each tokenized)
            # and hypothesis should be a tokenized sentence
            
            if generated_findings and gt_findings:
                reference_findings = [gt_findings.strip().split()]
                hypothesis_findings = generated_findings.split()
                findings_bleu = sentence_bleu(
                    reference_findings, 
                    hypothesis_findings, 
                    smoothing_function=smoothing
                )
            
            if generated_impression and gt_impression:
                reference_impression = [gt_impression.strip().split()]
                hypothesis_impression = generated_impression.split()
                impression_bleu = sentence_bleu(
                    reference_impression, 
                    hypothesis_impression, 
                    smoothing_function=smoothing
                )
                
        except Exception as e:
            # Keep scores as 0.0 if extraction or BLEU calculation fails
            pass
        
        # Sum of both BLEU scores (max reward = 2.0)
        reward = findings_bleu + impression_bleu
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} BLEU reward: {reward} -------------\n")
                f.write(f"Findings BLEU: {findings_bleu}, Impression BLEU: {impression_bleu}\n")
                f.write(f"Content: {content}\n")
                f.write(f"GT Findings: {gt_findings}\n")
                f.write(f"GT Impression: {gt_impression}\n")
    
    return rewards


def load_cluster_dictionaries(findings_path: str, impression_path: str):
    """
    Load the sentence-to-cluster dictionaries for findings and impression.
    Should be called once before training starts.
    """
    global FINDINGS_SENTENCE_TO_CLUSTER, IMPRESSION_SENTENCE_TO_CLUSTER
    
    if findings_path is not None:
        with open(findings_path, 'r') as f:
            FINDINGS_SENTENCE_TO_CLUSTER = json.load(f)
        logger.info(f"Loaded findings cluster dictionary with {len(FINDINGS_SENTENCE_TO_CLUSTER)} sentences")
    
    if impression_path is not None:
        with open(impression_path, 'r') as f:
            IMPRESSION_SENTENCE_TO_CLUSTER = json.load(f)
        logger.info(f"Loaded impression cluster dictionary with {len(IMPRESSION_SENTENCE_TO_CLUSTER)} sentences")


def load_criticality_dictionaries(findings_criticality_path: str, impression_criticality_path: str):
    """
    Load the cluster criticality dictionaries from CSV files.
    CSV format: columns 'element' (cluster ID) and 'criticality_score'.
    Should be called once before training starts.
    """
    import pandas as pd
    global FINDINGS_CLUSTER_CRITICALITY, IMPRESSION_CLUSTER_CRITICALITY
    
    if findings_criticality_path is not None:
        df = pd.read_csv(findings_criticality_path)
        FINDINGS_CLUSTER_CRITICALITY = dict(zip(df['element'], df['criticality_score']))
        logger.info(f"Loaded findings criticality dictionary with {len(FINDINGS_CLUSTER_CRITICALITY)} clusters")
    
    if impression_criticality_path is not None:
        df = pd.read_csv(impression_criticality_path)
        IMPRESSION_CLUSTER_CRITICALITY = dict(zip(df['element'], df['criticality_score']))
        logger.info(f"Loaded impression criticality dictionary with {len(IMPRESSION_CLUSTER_CRITICALITY)} clusters")


def split_into_sentences(text: str) -> list:
    """
    Split text into sentences by splitting on '.' and cleaning up.
    Returns list of non-empty, lowercase, stripped sentences.
    """
    if not text:
        return []
    
    # Split on '.'
    sentences = text.split('.')
    sentences = [x.strip().lower() for x in sentences]

    # Clean up: strip whitespace, convert to lowercase, filter empty
    cleaned_sentences = []
    itemize_numericals = set([str(x) for x in range(1,10)]) ## some impressions are itemized like "1. imp-1 2. imp-2"
    for sent in sentences:
        sent = sent.strip().lower()
        if sent and not sent in itemize_numericals:  # Only keep non-empty sentences
            cleaned_sentences.append(sent)

    return cleaned_sentences


def get_cluster_ids(sentences: list, sentence_to_cluster: dict) -> set:
    """
    Map a list of sentences to their cluster IDs.
    Sentences not found in the dictionary are skipped.
    
    Args:
        sentences: List of sentences (should already be lowercase and stripped)
        sentence_to_cluster: Dictionary mapping sentences to cluster IDs
    
    Returns:
        Set of cluster IDs
    """
    cluster_ids = set()
    for sent in sentences:
        if sent in sentence_to_cluster and sentence_to_cluster[sent]!=-1:
            cluster_ids.add(sentence_to_cluster[sent])
    return cluster_ids


def compute_f1_score(generated_clusters: set, ground_truth_clusters: set) -> float:
    """
    Compute F1 score based on cluster overlap.
    
    Args:
        generated_clusters: Set of cluster IDs from generated text
        ground_truth_clusters: Set of cluster IDs from ground truth
    
    Returns:
        F1 score (0.0 to 1.0)
    """
    
    if len(generated_clusters) == 0 or len(ground_truth_clusters) == 0:
        return 0.0  # One empty, one not = no match
    
    intersection = generated_clusters & ground_truth_clusters
    
    precision = len(intersection) / len(generated_clusters)
    recall = len(intersection) / len(ground_truth_clusters)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_weighted_f1_score(generated_clusters: set, ground_truth_clusters: set, criticality_dict: dict) -> float:
    """
    Compute weighted F1 score based on cluster overlap, weighted by criticality.
    
    Args:
        generated_clusters: Set of cluster IDs from generated text
        ground_truth_clusters: Set of cluster IDs from ground truth
        criticality_dict: Dictionary mapping cluster IDs to criticality scores
    
    Returns:
        Weighted F1 score (0.0 to 1.0)
    """
    
    if len(generated_clusters) == 0 or len(ground_truth_clusters) == 0:
        return 0.0  # One empty, one not = no match
    
    intersection = generated_clusters & ground_truth_clusters
    
    # Sum of criticality weights for each set
    intersection_weight = sum(criticality_dict.get(c, 1.0) for c in intersection)
    generated_weight = sum(criticality_dict.get(c, 1.0) for c in generated_clusters)
    ground_truth_weight = sum(criticality_dict.get(c, 1.0) for c in ground_truth_clusters)
    
    # Weighted precision: what fraction of predicted weight is correct
    precision = intersection_weight / generated_weight if generated_weight > 0 else 0.0
    
    # Weighted recall: what fraction of ground truth weight is captured
    recall = intersection_weight / ground_truth_weight if ground_truth_weight > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def semantic_cluster_reward(completions, Findings, Impression, **kwargs):
    """
    Reward function that computes semantic cluster-based F1 scores between:
    - Generated findings (from <think> tags) and ground truth Findings
    - Generated impression (from <answer> tags) and ground truth Impression
    
    The reward is based on the overlap of semantic clusters between generated
    and ground truth sentences. Sentences are mapped to pre-computed clusters
    where sentences with similar meanings belong to the same cluster.
    
    Returns the sum of both F1 scores as the reward (max 2.0).
    """
    global FINDINGS_SENTENCE_TO_CLUSTER, IMPRESSION_SENTENCE_TO_CLUSTER
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, gt_findings, gt_impression in zip(contents, Findings, Impression):
        findings_f1 = 0.0
        impression_f1 = 0.0
        
        try:
            # Extract findings from <think> tags in generated content
            findings_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            generated_findings = findings_match.group(1).strip() if findings_match else ""
            
            # Extract impression from <answer> tags in generated content
            impression_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            generated_impression = impression_match.group(1).strip() if impression_match else ""
            
            # Compute F1 for findings
            if FINDINGS_SENTENCE_TO_CLUSTER is not None and generated_findings and gt_findings:
                # Split into sentences
                gen_findings_sentences = split_into_sentences(generated_findings)
                gt_findings_sentences = split_into_sentences(gt_findings)
                
                # Get cluster IDs
                gen_findings_clusters = get_cluster_ids(gen_findings_sentences, FINDINGS_SENTENCE_TO_CLUSTER)
                gt_findings_clusters = get_cluster_ids(gt_findings_sentences, FINDINGS_SENTENCE_TO_CLUSTER)
                
                # Compute F1
                findings_f1 = compute_f1_score(gen_findings_clusters, gt_findings_clusters)
            
            # Compute F1 for impression
            if IMPRESSION_SENTENCE_TO_CLUSTER is not None and generated_impression and gt_impression:
                # Split into sentences
                gen_impression_sentences = split_into_sentences(generated_impression)
                gt_impression_sentences = split_into_sentences(gt_impression)
                
                # Get cluster IDs
                gen_impression_clusters = get_cluster_ids(gen_impression_sentences, IMPRESSION_SENTENCE_TO_CLUSTER)
                gt_impression_clusters = get_cluster_ids(gt_impression_sentences, IMPRESSION_SENTENCE_TO_CLUSTER)
                
                # Compute F1
                impression_f1 = compute_f1_score(gen_impression_clusters, gt_impression_clusters)
                
        except Exception as e:
            # Keep scores as 0.0 if extraction or computation fails
            pass
        
        # Sum of both F1 scores (max reward = 2.0)
        reward = findings_f1 + impression_f1
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Semantic Cluster reward: {reward} -------------\n")
                f.write(f"Findings F1: {findings_f1}, Impression F1: {impression_f1}\n")
                f.write(f"Content: {content}\n")
                f.write(f"GT Findings: {gt_findings}\n")
                f.write(f"GT Impression: {gt_impression}\n")
    
    return rewards


def exact_sentence_match_reward(completions, Findings, Impression, **kwargs):
    """
    Reward function that computes exact sentence match F1 scores between:
    - Generated findings (from <think> tags) and ground truth Findings
    - Generated impression (from <answer> tags) and ground truth Impression
    
    Unlike semantic_cluster_reward which maps sentences to clusters, this function
    compares sentences directly using exact string matching (after lowercase and strip).
    This allows comparing the effectiveness of clustering vs exact matching.
    
    Returns the sum of both F1 scores as the reward (max 2.0).
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, gt_findings, gt_impression in zip(contents, Findings, Impression):
        findings_f1 = 0.0
        impression_f1 = 0.0
        
        try:
            # Extract findings from <think> tags in generated content
            findings_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            generated_findings = findings_match.group(1).strip() if findings_match else ""
            
            # Extract impression from <answer> tags in generated content
            impression_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            generated_impression = impression_match.group(1).strip() if impression_match else ""
            
            # Compute F1 for findings using exact sentence matching
            if generated_findings and gt_findings:
                # Split into sentences (lowercase, stripped)
                gen_findings_sentences = set(split_into_sentences(generated_findings))
                gt_findings_sentences = set(split_into_sentences(gt_findings))
                
                # Compute F1 using exact sentence sets
                findings_f1 = compute_f1_score(gen_findings_sentences, gt_findings_sentences)
            
            # Compute F1 for impression using exact sentence matching
            if generated_impression and gt_impression:
                # Split into sentences (lowercase, stripped)
                gen_impression_sentences = set(split_into_sentences(generated_impression))
                gt_impression_sentences = set(split_into_sentences(gt_impression))
                
                # Compute F1 using exact sentence sets
                impression_f1 = compute_f1_score(gen_impression_sentences, gt_impression_sentences)
                
        except Exception as e:
            # Keep scores as 0.0 if extraction or computation fails
            pass
        
        # Sum of both F1 scores (max reward = 2.0)
        reward = findings_f1 + impression_f1
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Exact Sentence Match reward: {reward} -------------\n")
                f.write(f"Findings F1: {findings_f1}, Impression F1: {impression_f1}\n")
                f.write(f"Content: {content}\n")
                f.write(f"GT Findings: {gt_findings}\n")
                f.write(f"GT Impression: {gt_impression}\n")
    
    return rewards


def weighted_semantic_cluster_reward(completions, Findings, Impression, **kwargs):
    """
    Reward function that computes weighted semantic cluster-based F1 scores between:
    - Generated findings (from <think> tags) and ground truth Findings
    - Generated impression (from <answer> tags) and ground truth Impression
    
    Similar to semantic_cluster_reward, but weights clusters by their criticality scores.
    Higher criticality clusters contribute more to the F1 score.
    
    Returns the sum of both weighted F1 scores as the reward (max 2.0).
    """
    global FINDINGS_SENTENCE_TO_CLUSTER, IMPRESSION_SENTENCE_TO_CLUSTER
    global FINDINGS_CLUSTER_CRITICALITY, IMPRESSION_CLUSTER_CRITICALITY
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, gt_findings, gt_impression in zip(contents, Findings, Impression):
        findings_f1 = 0.0
        impression_f1 = 0.0
        
        try:
            # Extract findings from <think> tags in generated content
            findings_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            generated_findings = findings_match.group(1).strip() if findings_match else ""
            
            # Extract impression from <answer> tags in generated content
            impression_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            generated_impression = impression_match.group(1).strip() if impression_match else ""
            
            # Compute weighted F1 for findings
            if (FINDINGS_SENTENCE_TO_CLUSTER is not None and 
                FINDINGS_CLUSTER_CRITICALITY is not None and 
                generated_findings and gt_findings):
                # Split into sentences
                gen_findings_sentences = split_into_sentences(generated_findings)
                gt_findings_sentences = split_into_sentences(gt_findings)
                
                # Get cluster IDs
                gen_findings_clusters = get_cluster_ids(gen_findings_sentences, FINDINGS_SENTENCE_TO_CLUSTER)
                gt_findings_clusters = get_cluster_ids(gt_findings_sentences, FINDINGS_SENTENCE_TO_CLUSTER)
                
                # Compute weighted F1
                findings_f1 = compute_weighted_f1_score(
                    gen_findings_clusters, 
                    gt_findings_clusters, 
                    FINDINGS_CLUSTER_CRITICALITY
                )
            
            # Compute weighted F1 for impression
            if (IMPRESSION_SENTENCE_TO_CLUSTER is not None and 
                IMPRESSION_CLUSTER_CRITICALITY is not None and 
                generated_impression and gt_impression):
                # Split into sentences
                gen_impression_sentences = split_into_sentences(generated_impression)
                gt_impression_sentences = split_into_sentences(gt_impression)
                
                # Get cluster IDs
                gen_impression_clusters = get_cluster_ids(gen_impression_sentences, IMPRESSION_SENTENCE_TO_CLUSTER)
                gt_impression_clusters = get_cluster_ids(gt_impression_sentences, IMPRESSION_SENTENCE_TO_CLUSTER)
                
                # Compute weighted F1
                impression_f1 = compute_weighted_f1_score(
                    gen_impression_clusters, 
                    gt_impression_clusters, 
                    IMPRESSION_CLUSTER_CRITICALITY
                )
                
        except Exception as e:
            # Keep scores as 0.0 if extraction or computation fails
            pass
        
        # Sum of both weighted F1 scores (max reward = 2.0)
        reward = findings_f1 + impression_f1
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Weighted Semantic Cluster reward: {reward} -------------\n")
                f.write(f"Findings Weighted F1: {findings_f1}, Impression Weighted F1: {impression_f1}\n")
                f.write(f"Content: {content}\n")
                f.write(f"GT Findings: {gt_findings}\n")
                f.write(f"GT Impression: {gt_impression}\n")
    
    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "bleu": bleu_reward,
    "semantic_cluster": semantic_cluster_reward,
    "exact_sentence_match": exact_sentence_match_reward,
    "weighted_semantic_cluster": weighted_semantic_cluster_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility - ensures consistent dataloader ordering
    set_seed(training_args.seed)
    
    # Check for last checkpoint to enable resumption
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected at {last_checkpoint}. Training will resume from this checkpoint.")
    
    # Load cluster dictionaries if semantic_cluster or weighted_semantic_cluster reward is used
    if "semantic_cluster" in script_args.reward_funcs or "weighted_semantic_cluster" in script_args.reward_funcs:
        if script_args.findings_cluster_path is None or script_args.impression_cluster_path is None:
            raise ValueError(
                "When using 'semantic_cluster' or 'weighted_semantic_cluster' reward, both "
                "--findings_cluster_path and --impression_cluster_path must be provided."
            )
        load_cluster_dictionaries(
            script_args.findings_cluster_path, 
            script_args.impression_cluster_path
        )
    
    # Load criticality dictionaries if weighted_semantic_cluster reward is used
    if "weighted_semantic_cluster" in script_args.reward_funcs:
        if script_args.findings_criticality_path is None or script_args.impression_criticality_path is None:
            raise ValueError(
                "When using 'weighted_semantic_cluster' reward, both --findings_criticality_path and "
                "--impression_criticality_path must be provided."
            )
        load_criticality_dictionaries(
            script_args.findings_criticality_path,
            script_args.impression_criticality_path
        )
    
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load and merge SFT checkpoint if provided
    model_to_use = model_args.model_name_or_path

    if script_args.sft_checkpoint_path is not None:
        # Get local rank for debugging
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        print(f"[Rank {local_rank}/{world_size}] Loading SFT checkpoint from {script_args.sft_checkpoint_path}")

        # Get the appropriate model class based on model name
        model_class = get_qwen_vl_model_class(model_args.model_name_or_path)
        
        # Build model kwargs
        # CRITICAL: Load model to CPU to avoid multi-GPU conflicts
        # DeepSpeed will handle distribution to GPUs later
        model_kwargs = dict(
            torch_dtype="auto",
            attn_implementation=model_args.attn_implementation,
            device_map="cpu",  # Explicitly load to CPU
        )
        
        # Qwen3-VL doesn't support use_cache in __init__, so don't add it
        if not is_qwen3_vl_model(model_args.model_name_or_path):
            model_kwargs["use_cache"] = False
        
        # Load the base model to CPU
        print(f"[Rank {local_rank}/{world_size}] Loading base model to CPU...")
        base_model = model_class.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs
        )
        print(f"[Rank {local_rank}/{world_size}] Base model loaded. Device: {next(base_model.parameters()).device}")
        
        # Load the LoRA adapter - also to CPU
        print(f"[Rank {local_rank}/{world_size}] Loading PEFT adapter to CPU...")
        model_with_adapter = PeftModel.from_pretrained(
            base_model,
            script_args.sft_checkpoint_path,
            torch_device="cpu",  # Explicitly load adapter weights to CPU
        )
        print(f"[Rank {local_rank}/{world_size}] Adapter loaded successfully.")
        
        # Merge the adapter into the base model and unload
        model_to_use = model_with_adapter.merge_and_unload()
        logger.info("\n\nSFT LoRA adapter merged into base model successfully\n\n")

    # # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Load your custom JSON dataset
    raw_dataset = load_dataset('json', data_files=script_args.data_json_path)

    # Split into train and test based on the 'split' column
    train_dataset = raw_dataset['train'].filter(lambda x: x['split'] == 'train')
    test_dataset = raw_dataset['train'].filter(lambda x: x['split'] == 'test')

    # Create a DatasetDict with the same structure
    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset  # or you might want to keep only 'train' if your original code only used that
    })
    print(f'dataset : ')
    print(dataset)
    print(f'script_args')
    print(script_args)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["input"]},
            ],
        }

    # def make_conversation_image(example):
    #     return {
    #         "prompt": [
    #             {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image"},
    #                     {"type": "text", "text": example["problem"]},
    #                 ],
    #             },
    #         ],
    #     }

    # QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the findings in the image and then provides the user with the final impression. The findings "
        "and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> findings here </think><answer> impression here </answer>"
    )

    USER_PROMPT = "Given the provided medical image, analyze it carefully and generate findings and impression."
    def make_conversation_image(example):
        return {
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        # IMPORTANT: Match SFT order - text first, then image
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "image"},
                    ],
                },
            ],
        }


    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    
    # Setup callbacks
    callbacks = []
    if script_args.early_stopping_patience is not None:
        # Early stopping based on eval_reward (higher is better)
        if not training_args.load_best_model_at_end:
            logger.warning(
                "Early stopping requires --load_best_model_at_end true. Enabling it automatically."
            )
            training_args.load_best_model_at_end = True
        if training_args.metric_for_best_model is None:
            training_args.metric_for_best_model = "eval_reward"
            training_args.greater_is_better = True  # Higher reward is better
            logger.info("Setting metric_for_best_model to 'eval_reward' (greater_is_better=True)")
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=script_args.early_stopping_patience))
        logger.info(f"Early stopping enabled with patience={script_args.early_stopping_patience}")
    
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_to_use,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        callbacks=callbacks if callbacks else None,
    )

    # Train and push the model to the Hub
    # Resume from checkpoint if available, otherwise start fresh
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    model_args.lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    model_args.use_peft = True
    main(script_args, training_args, model_args)
