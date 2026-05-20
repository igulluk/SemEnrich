from dataclasses import dataclass, field
import torch
from torch import nn
import transformers
from typing import Optional
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)
from transformers import GPTNeoForCausalLM, AutoModelForImageTextToText

@dataclass
class PEFTArguments:
    is_lora: Optional[bool] = field(default=True)
    peft_mode: str = field(default="lora")
    lora_rank: int = field(default=8)
    num_virtual_tokens: int = field(default=32)  # Used for prompt tuning, prefix tuning, and p-tuning
    mapping_hidden_dim: int = field(default=1024)


class LLModel(nn.Module):
    def __init__(self, config):  
        super(LLModel, self).__init__()  
        peft_args = PEFTArguments
        self.config = config
                
        ''' Large Language Model'''
        self.model = self.get_model(peft_args)
        

    def get_model(self, peft_args):
        print("Setup Model")
        if self.config.LLM.model_name == 'chaoyi-wu/PMC_LLAMA_7B':
            model = transformers.LlamaForCausalLM.from_pretrained(
                self.config.LLM.model_name,
                cache_dir=self.config.LLM.cache_dir,
                force_download=False,
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.config.LLM.model_name,
                cache_dir=self.config.LLM.cache_dir,
                force_download=False,    
            )
        if peft_args.is_lora:
            print("Setup PEFT")
            peft_config = self.get_peft_config(peft_args=peft_args)
            model = get_peft_model(model, peft_config)
        return model
    
    
    def forward(self, input_ids, image_features, labels=None):

        input_embedding = self.model.get_input_embeddings()(input_ids)
        input_embedding = torch.cat([image_features, input_embedding], dim=1)

        output = self.model(inputs_embeds=input_embedding, labels=labels)

        return output


    def get_peft_config(self, peft_args: PEFTArguments):
        if peft_args.peft_mode == "lora":
            if self.config.LLM.model_name in {
                        'google/gemma-2-2b', 
                        'google/gemma-2-2b-it', 
                        'google/gemma-2-9b', 
                        'google/gemma-2-9b-it',
                        'google/gemma-2-27b', 
                        'google/gemma-2-27b-it',
                        'google/gemma-3-4b-pt',
                        'google/gemma-3-12b-pt',
                        'meta-llama/Llama-3.2-11B-Vision',
                        'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                        'Qwen/Qwen2-7B',
                    }:
                peft_config = LoraConfig(
                    target_modules='all-linear',
                    task_type=TaskType.CAUSAL_LM, inference_mode=False,
                    r=peft_args.lora_rank,
                    lora_alpha=32, lora_dropout=0.1
                )
            else:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, 
                    inference_mode=False,
                    r=peft_args.lora_rank,
                    lora_alpha=32, lora_dropout=0.1
                )
        elif peft_args.peft_mode == "prefix":
            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=peft_args.num_virtual_tokens,
                encoder_hidden_size=peft_args.mapping_hidden_dim,
                prefix_projection=True,
            )
        elif peft_args.peft_mode == "ptuning":
            peft_config = PromptEncoderConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=peft_args.num_virtual_tokens,
                encoder_hidden_size=peft_args.mapping_hidden_dim,
            )
        elif peft_args.peft_mode == "prompt":
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=peft_args.num_virtual_tokens,
            )
        else:
            raise KeyError(peft_args.peft_mode)
        return peft_config




