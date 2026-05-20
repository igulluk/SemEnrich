import torch

from torch import nn
from einops import rearrange
from dataclasses import dataclass, field
from typing import Optional
from .transformer import TransformerDecoderLayer, TransformerDecoder
from .vision_model import OutputDim


@dataclass
class LLMArguments:
    vocab_size: int
    hidden_dim: int 

@dataclass
class QueryArguments:
    N: Optional[int] = field(default=12)
    H: Optional[int] = field(default=16)
    

llm_configs = {
    "chaoyi-wu/PMC_LLAMA_7B": LLMArguments(hidden_dim=4096, vocab_size=32000),
    "meta-llama/Meta-Llama-3-8B": LLMArguments(hidden_dim=4096, vocab_size=128000),
    "meta-llama/Llama-3.1-8B": LLMArguments(hidden_dim=4096, vocab_size=128000),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": LLMArguments(hidden_dim=4096, vocab_size=128000),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": LLMArguments(hidden_dim=1536, vocab_size=151646),
    "meta-llama/Llama-3.2-11B-Vision": LLMArguments(hidden_dim=4096, vocab_size=128000),
    "google/gemma-2-2b": LLMArguments(hidden_dim=2304, vocab_size=256128),
    "google/gemma-2-2b-it": LLMArguments(hidden_dim=2304, vocab_size=256128),

    "google/gemma-2-9b": LLMArguments(hidden_dim=3584, vocab_size=256128),
    "google/gemma-2-27b": LLMArguments(hidden_dim=4608, vocab_size=256128),
    "google/gemma-2-9b-it": LLMArguments(hidden_dim=3584, vocab_size=256128),
    "google/gemma-2-27b-it": LLMArguments(hidden_dim=4608, vocab_size=256128),
    "bigscience/bloom-7b1": LLMArguments(hidden_dim=4096, vocab_size=250680),
    "meta-llama/Llama-3.2-1B-Instruct": LLMArguments(hidden_dim=2048, vocab_size=128256),
    "EleutherAI/gpt-neo-2.7B": LLMArguments(hidden_dim=2560, vocab_size=50257),
    "openai-community/gpt2-large": LLMArguments(hidden_dim=1280, vocab_size=50257),
    "google/gemma-2b": LLMArguments(hidden_dim=2048, vocab_size=256128),
    "meta-llama/Llama-2-7b-hf": LLMArguments(hidden_dim=4096, vocab_size=32000),
    "distilbert/distilgpt2": LLMArguments(hidden_dim=768, vocab_size=50257),   
    "Qwen/Qwen3-8B": LLMArguments(hidden_dim=4096, vocab_size=151936),   
    "Qwen/Qwen3-14B": LLMArguments(hidden_dim=4096, vocab_size=151936),   

    "google/gemma-3-4b-pt": LLMArguments(hidden_dim=2560, vocab_size=262208),   
    "google/gemma-3-12b-pt": LLMArguments(hidden_dim=3840, vocab_size=262208),   

    "mistralai/Mistral-7B-v0.1": LLMArguments(hidden_dim=4096, vocab_size=32000),   
}



def get_model_config(model_name: str) -> LLMArguments:
    return llm_configs.get(model_name, None)


class QueryDecoder(nn.Module):
    def __init__(self, config):  
        super(QueryDecoder, self).__init__()  
        self.img_tokens = config.img_token_num
        model_args = QueryArguments
        self.H = model_args.H 
        self.N = model_args.N 
        self.config = config

        output_dim = OutputDim
        num_ftrs = getattr(output_dim, config.vision_module)

        llm_args = get_model_config(config.LLM.model_name)
        self.hidden_dim = llm_args.hidden_dim

        ''' Query Decoder'''
        self.query_embed = nn.Embedding(self.img_tokens, num_ftrs) 
        
        decoder_layer = TransformerDecoderLayer(num_ftrs, self.H, 1024, 0.1, 'relu', normalize_before=True)
        decoder_norm = nn.LayerNorm(num_ftrs)
        self.decoder = TransformerDecoder(decoder_layer, self.N, decoder_norm, return_intermediate=False)

        ''' FC '''        
        self.fc_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.fc_l2 = nn.Linear(num_ftrs, self.hidden_dim)

    def forward(self, x, B):
        features = x.transpose(0, 1)  # patch_num b dim
        ### Q-Former ###
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # query_number, batch, dim
        features, ws = self.decoder(query_embed, features, memory_key_padding_mask=None, pos=None, query_pos=None)
        features = features.transpose(0, 1)
        ### fc  ### 
        features = rearrange(features, 'b n d  -> (b n) d')
        features = self.fc_l1(features)
        features = torch.relu(features)
        features = self.fc_l2(features)
        features = rearrange(features, '(b n) d -> b n d', b=B )  
        return features