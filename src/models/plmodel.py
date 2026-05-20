import torch
import lightning.pytorch as pl
import torch.nn.functional as F
import numpy as np 
import copy 
import random 
import json 
import os 
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoProcessor

from torchmetrics import Accuracy
from torch import nn, Tensor
from src.models.vl_model import VLModel, VLModelInstruct
from src.models.vqa_eval import VQAEval
from collections import defaultdict 
from src.data.data_loaders import get_sequential_dataloaders


# import nltk
# nltk.download('wordnet')


class PreTrainingLightningModule(pl.LightningModule):
    def __init__(
        self,
        config,
    ) -> None:

        super().__init__()
        self.config = config 
        self.evaluator = VQAEval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.LLM.model_name,
            cache_dir=config.LLM.cache_dir, 
            force_download=False, 
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.model = VLModel(config)

        self.train_step_outputs = []
        self.valid_step_outputs = []
        self.is_inference = self.config.is_inference 

    def training_step(self, batch, batch_idx):
        step_out = self._step(batch, batch_idx)   
        step_loss = step_out['loss']  
        loss_value = step_loss.detach().item()
        loss_value = round(loss_value, 4)
        step_out['loss'] = loss_value
        self.train_step_outputs.append(step_out)
        return step_loss

    def validation_step(self, batch, batch_idx):
        
        step_out = self._step(batch, batch_idx)
        step_loss = step_out['loss']  
        loss_value = step_loss.detach().item()
        loss_value = round(loss_value, 4)
        step_out['loss'] = loss_value
        self.valid_step_outputs.append(step_out)
        return step_loss

    def on_train_epoch_end(self):

        self._logging(self.train_step_outputs, phase='train')
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        self._logging(self.valid_step_outputs, phase='valid')
        self.valid_step_outputs.clear()

    def _logging(self, step_outputs, phase):
        logs = {}
        loss = np.array([x['loss'] for x in step_outputs]).mean()
        logs.update({phase + "_loss": loss})
        for dataset_name in self.config.dataset_list:
            loss = np.array([x['loss'] for x in step_outputs if x['dset_name']==dataset_name]).mean()
            logs.update({phase + "_" + dataset_name + "_loss": loss})

  
        if self.is_inference:
            inference_json_dir = self.config.inference_out_dir
            os.makedirs(inference_json_dir, exist_ok=True)
            with open(f'{inference_json_dir}{self.config.experiment_name}.json', "w") as file:
                json.dump(step_outputs, file)
            eval_res = self.evaluator.get_results(step_outputs, phase)
            logs.update(eval_res)
         
        self.log_dict(logs, sync_dist=True)

    def _step(self, batch, batch_idx):

        images = batch.get("image", None)
        input_ids = batch.get("input_ids", None)
        labels = batch.get("labels", None)
        given_texts = batch.get("given_text", None)
        given_text_lens = batch.get("given_text_len", None)


        output = self.model(
            images=images,
            input_ids=input_ids,
            labels=labels,
        )
        step_loss = output.loss
        
        if self.is_inference:
            gt_texts, generated_texts = self.get_predictions(
                images, 
                input_ids,
                labels,
                given_texts,
                given_text_lens
        )
        else:
            gt_texts, generated_texts = None, None

        dset_name = batch.get("dset_name", None)[0]
        img_path = batch.get("img_path", None)

        step_out = {
            'loss':step_loss,
            'gt_texts':gt_texts,
            'generated_texts':generated_texts,
            'dset_name':dset_name,
            'img_path': img_path,
        }

        return step_out

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.optimizer.learning_rate,
            eps=self.config.optimizer.adam_eps,
            weight_decay = self.config.optimizer.adam_weight_decay,
            betas=self.config.optimizer.adam_betas,
        )
        return optimizer

    def get_predictions(
            self, 
            images, 
            input_ids,
            labels,
            given_texts,
            given_text_lens
        ):

        bsize = input_ids.shape[0]
        generated_texts = []
        with torch.no_grad():
            for i in range(bsize):
                prompt = given_texts[i][:given_text_lens[i]]
                image_i = images[i].unsqueeze(0) if images is not None else None 
                generation_ids = self.model.generate_long_sentence(prompt.unsqueeze(0), image_i)
                gen_text = self.tokenizer.batch_decode(generation_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                input_text = self.tokenizer.batch_decode(input_ids[i].unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                generated_texts.append(gen_text[1:] if gen_text.startswith(' ') else gen_text)
    
        labels_ = labels.detach().cpu()
        # labels_[labels_==-100] = 0
        labels_[labels_==-100] = self.tokenizer.pad_token_id
        gt_texts = self.tokenizer.batch_decode(labels_, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        gt_texts = [x[1:] if x.startswith(' ') else x for x in gt_texts]

        # print(f'========gt_texts=======')
        # for x in gt_texts:
        #     print(f'\n{x}')
        # print(f'========generated_texts=======')
        # for x in generated_texts:
        #     print(f'\n{x}')

        return gt_texts, generated_texts

    # def get_predictions(
    #         self, 
    #         images, 
    #         input_ids,
    #         labels,
    #         given_texts,
    #         given_text_lens
    #     ):

    #     bsize = input_ids.shape[0]
    #     generated_texts = []
    #     with torch.no_grad():
    #         for i in range(bsize):
    #             prompt = given_texts[i][:given_text_lens[i]]
    #             image_i = images[i].unsqueeze(0) if images is not None else None 
    #             generation_ids = self.model.generate_long_sentence(prompt.unsqueeze(0), image_i)
    #             gen_text = self.tokenizer.batch_decode(generation_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #             input_text = self.tokenizer.batch_decode(input_ids[i].unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #             generated_texts.append(gen_text[1:] if gen_text.startswith(' ') else gen_text)
    
    #     labels_ = labels.detach().cpu()
    #     # labels_[labels_==-100] = 0
    #     labels_[labels_==-100] = self.tokenizer.pad_token_id
    #     gt_texts = self.tokenizer.batch_decode(labels_, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    #     gt_texts = [x[1:] if x.startswith(' ') else x for x in gt_texts]

    #     return gt_texts, generated_texts



    def train_dataloader(self):
        return get_sequential_dataloaders(self.config, phase='train')

    def val_dataloader(self):
        return get_sequential_dataloaders(self.config, phase='valid')
    
