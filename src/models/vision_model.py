from dataclasses import dataclass, field
import os 
import torch
from torch import nn
from einops import rearrange
import transformers
from transformers import CLIPVisionConfig
from .blocks import ModifiedResNet, PMC_CLIP_cfg
import timm
from huggingface_hub import hf_hub_download
from typing import Optional
# from open_clip import create_model_from_pretrained 
from transformers import AutoProcessor, AutoModel


@dataclass
class OutputDim:
    PMC_CLIP: int = 1024
    CLIP: int = 768
    Scratch: int = 768
    UNI: int = 1024
    BiomedCLIP: int = 768
    PMC_OA: int = 768
    ConvNext: int = 768
    Vit512: int = 768 


@dataclass
class VisionModelArguments:
    output_dim = OutputDim
    CLIP_name: Optional[str] = field(default='openai/clip-vit-base-patch32')
    UNI_path: Optional[str] = field(default='.')
    BiomedCLIP_name: Optional[str] = field(default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    PMC_OA_name: Optional[str] = field(default='hf-hub:ryanyip7777/pmc_vit_l_14')
    Vit512_name: Optional[str] = field(default='vit_base_patch16_siglip_512')


class VisionModel(nn.Module):
    def __init__(self, config):  
        super(VisionModel, self).__init__()  
        self.config = config
        self.vision_module = config.vision_module
        self.arguments = VisionModelArguments
        
        ''' Vision Model'''
        self.model, num_ftrs = self.get_model()
    
    def forward(self, xis):
        if self.vision_module in {'PMC_CLIP'}:
            batch_size = xis.shape[0]
            res_fea = self.model(xis) #batch_size,feature_size,patch_num,patch_num
            out_emb = rearrange(res_fea, 'b d n1 n2 -> b (n1 n2) d')
        if self.vision_module in {'CLIP', 'Scratch'}:
            out_emb = self.model(pixel_values=xis)['last_hidden_state'][:, 1:, :]  # dismiss the cls token
        if self.vision_module in {'UNI', 'BiomedCLIP'}:
            out_emb = self.model.forward_features(xis)[:, 1:, :]  # dismiss the cls token
        if self.vision_module in {'PMC_OA'}:
            out_emb = self.model(xis)
        if self.vision_module in {'ConvNext'}:
            out_emb = self.model(xis)
        if self.vision_module == 'Vit512':
            out_emb = self.model(pixel_values=xis)['last_hidden_state'][:, 1:, :]  # dismiss the cls token

        return out_emb

    def vision_load_pretrain(self, resnet, model_path):
        checkpoint = torch.load(model_path, map_location='cpu') 
        state_dict = checkpoint['state_dict'] 
        state_dict = {k.replace('module.visual.', ''): v for k, v in state_dict.items() if '.visual' in k}
        resnet.load_state_dict(state_dict)
        return resnet

    def get_model(self):

        """Factory function to get the appropriate vision model."""
        if self.vision_module == 'PMC_CLIP':
            vision_cfg = PMC_CLIP_cfg()
            vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
            model = ModifiedResNet(
                layers=vision_cfg.layers,
                heads=vision_heads,
                output_dim=768,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width
            )
            model = self.vision_load_pretrain(model, './img_checkpoint/CLIP/clip-vit-base-patch32')
            model = nn.Sequential(*list(model.children())[:-2])
        
        elif self.vision_module == "CLIP":
            model = transformers.CLIPVisionModel.from_pretrained(
                self.arguments.CLIP_name, ignore_mismatched_sizes=True, cache_dir=self.config.LLM.cache_dir
            )

        elif self.vision_module == 'Scratch':
            model = transformers.CLIPVisionModel(config=CLIPVisionConfig(image_size=self.config.image_size[0]))

        elif self.vision_module == 'UNI':
            local_dir = self.arguments.UNI_path
            os.makedirs(local_dir, exist_ok=True)
            if not os.path.exists(os.path.join(local_dir, "pytorch_model.bin")):
                hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
            model = timm.create_model(
                "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
            )
            model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)

        elif self.vision_module == 'BiomedCLIP':
            # model, _ = create_model_from_pretrained(self.arguments.BiomedCLIP_name, cache_dir=self.config.LLM.cache_dir)

            # model = model.visual.trunk
            for _ in range(5):
                print(f'CONVERT FROM TIMM TO BIOMEDCLIP IN VISION MODEL')
            import timm
            model = timm.create_model('vit_base_patch16_224', pretrained=False)

        elif self.vision_module == 'PMC_OA':
            model, _ = create_model_from_pretrained(self.arguments.PMC_OA_name, cache_dir=self.config.LLM.cache_dir)
            model = model.visual
        
        elif self.vision_module == 'Vit512':

            model = AutoModel.from_pretrained(
                    "google/siglip-base-patch16-512", 
                    cache_dir=self.config.LLM.cache_dir
                ).vision_model

        else:
            raise ValueError(f"Unknown Vision Module: {self.vision_module}")

        num_ftrs = getattr(self.arguments.output_dim, self.vision_module)
        
        return model, num_ftrs