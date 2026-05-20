import torch
from torch import nn
from .vision_model import VisionModel
from .query_decoder import QueryDecoder
from .llm import LLModel, PEFTArguments
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)
# from transformers import GPTNeoForCausalLM, AutoModelForImageTextToText, Gemma3ForConditionalGeneration
from transformers import AutoModelForImageTextToText


class VLModel(nn.Module):
    def __init__(self, config):  
        super(VLModel, self).__init__()  
    
        self.config = config
    
        ''' Visual Model'''
        self.vision_model = VisionModel(config)
        
        ''' Query Decoder'''
        self.query_decoder = QueryDecoder(config)
        
        ''' Large Language Model'''
        self.llm = LLModel(config)
    

    def forward(self, input_ids, images, labels=None):
        #### When the dataset is only question-answering dataset without images ####
        if images is None:
            output = self.llm(input_ids, None, labels)
        
        else:
            B = images.shape[0]
            image_features = self.vision_model(images)
            image_features = self.query_decoder(image_features, B)
            output = self.llm(input_ids, image_features, labels)

        return output

    def generate(self, input_ids, images):
        with torch.no_grad():
            if images is None:
                generation = self.llm(input_ids, None)['logits']
            else:
                B = images.shape[0]
                image_features = self.vision_model(images)
                image_features = self.query_decoder(image_features, B)
                generation = self.llm(input_ids, image_features)['logits']
            return generation
    
    def generate_long_sentence(self, input_ids, images):
        with torch.no_grad():

            if images is None:
                input_embedding = self.llm.model.get_input_embeddings()(input_ids)

                # Create attention mask for the image features (0 means "ignore")
                image_mask = torch.zeros((input_embedding.shape[0], self.config.img_token_num), device=input_ids.device)
                image_features = torch.zeros((input_embedding.shape[0], self.config.img_token_num, input_embedding.shape[2]), device=image_mask.device)

                # Create the usual attention mask for the text input (1 means "attend")
                text_mask = torch.ones(input_ids.shape, device=input_embedding.device)

                # Concatenate both masks (image + text)
                attention_mask = torch.cat([image_mask, text_mask], dim=1)

                input_embedding = torch.cat([image_features, input_embedding], dim=1)
                generation = self.llm.model.generate(inputs_embeds=input_embedding, max_new_tokens=self.config.LLM.seq_length, attention_mask=attention_mask)


            else:
                B = images.shape[0]
                image_features = self.vision_model(images)
                image_features = self.query_decoder(image_features, B)
                input_embedding = self.llm.model.get_input_embeddings()(input_ids)
                input_embedding = torch.cat([image_features, input_embedding], dim=1)

                generation = self.llm.model.generate(inputs_embeds=input_embedding, max_new_tokens=self.config.LLM.seq_length)
                
            return generation



class VLModelInstruct(nn.Module):
    def __init__(self, config):  
        super(VLModelInstruct, self).__init__()  
        peft_args = PEFTArguments
        self.config = config
                
        ''' Large Language Model'''
        self.model = self.get_model(peft_args)

        

    def get_model(self, peft_args):
        print("Setup Model")
        if "qwen" in self.config.LLM.model_name.lower():

            model = AutoModelForImageTextToText.from_pretrained(
                self.config.LLM.model_name,
                device_map="auto",  # 🚀 Distributes layers across available GPUs
                cache_dir=self.config.LLM.cache_dir
            )

        if "gemma" in self.config.LLM.model_name.lower():

            model = Gemma3ForConditionalGeneration.from_pretrained(
                "google/gemma-3-27b-it",
                # device_map="auto", 
                cache_dir=self.config.LLM.cache_dir,
                torch_dtype=torch.bfloat16, 
                # low_cpu_mem_usage=False  # 🚀 Force full loading of weights
            )

        if peft_args.is_lora:
            print("Setup PEFT")
            peft_config = self.get_peft_config(peft_args=peft_args)
            # model.model = get_peft_model(model.model, peft_config)
            # model.visual = get_peft_model(model.visual, peft_config)
            model = get_peft_model(model, peft_config)
        return model
    
    
    def forward(self, inputs):
        if "qwen" in self.config.LLM.model_name:
            output = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                image_grid_thw=inputs["image_grid_thw"],
                pixel_values=inputs["pixel_values"]  # ✅ Required for vision models
            )
        elif "gemma"  in self.config.LLM.model_name:
            output = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                pixel_values=inputs["pixel_values"]  # ✅ Required for vision models
            )
        else:
            raise ValueError
        return output


    def get_peft_config(self, peft_args: PEFTArguments):
        if peft_args.peft_mode == "lora":
            peft_config = LoraConfig(
                r=peft_args.lora_rank,
                lora_alpha=32,
                lora_dropout=0.1,
                # bias="none",
                # target_modules='all-linear',  
                target_modules=["q_proj", "v_proj", "k_proj", "qkv", "fc1", "fc2", "o_proj", "gate_proj", "up_proj", "down_proj"],
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


    
    def generate_long_sentence(self, inputs):

        with torch.no_grad():
            B = images.shape[0]
            image_features = self.vision_model(images)
            image_features = self.query_decoder(image_features, B)
            input_embedding = self.llm.model.get_input_embeddings()(input_ids)
            input_embedding = torch.cat([image_features, input_embedding], dim=1)
            generation = self.llm.model.generate(inputs_embeds=input_embedding, max_new_tokens=self.config.LLM.seq_length)
            
        return generation
                        


