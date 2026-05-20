"""Data loaders for Medical Report Generation training."""
import random
import cv2
import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from transformers import AutoTokenizer, AutoProcessor
import torch.distributed as dist
import lightning.pytorch as pl
from PIL import Image
from collections import defaultdict
import copy 
from src.data.rexgrad_dataset import RexGradDataset, RexGradDatasetFactory





class Transforms:
    """Image transformations for training and validation."""
    
    def __init__(self, config):
        self.train_transforms = A.Compose([
            A.augmentations.geometric.resize.Resize(
                height=config.image_size[0], 
                width=config.image_size[1]
            )
        ])
        self.valid_transforms = A.Compose([
            A.augmentations.geometric.resize.Resize(
                height=config.image_size[0], 
                width=config.image_size[1]
            )
        ])


class ImageTextDataset(Dataset):
    def __init__(self, config, transform, tokenizer, img_padding, dataset_name, phase):
        self.config = config
        self.transform = transform
        self.tokenizer = tokenizer
        self.img_padding = img_padding
        self.seq_length = config.LLM.seq_length
        self.dset_name = dataset_name
        self.phase = phase

        # Load data (to be defined in the subclass)
        # self.data = self.load_data()
        self.base_dataset = self._create_base_dataset(dataset_name, phase)

    def _create_base_dataset(self, dataset_name, phase):
        """Create the appropriate RexGrad dataset."""
        # Parse dataset name to determine configuration
        if dataset_name == 'rexgrad':
            return RexGradDataset(
                data_path=self.config.rexgrad.main_path,
                phase=phase,
                use_findings_only=False,
                transform=self.transform,
            )
        elif dataset_name == 'rexgrad-only-findings':
            return RexGradDataset(
                data_path=self.config.rexgrad.main_path,
                phase=phase,
                use_findings_only=True,
                transform=self.transform,
            )
        else:
            # Default to RexGradDatasetFactory
            return RexGradDatasetFactory.create(
                config=self.config,
                config_name=dataset_name,
                data_path=self.config.rexgrad.main_path,
                phase=phase,
                transform=self.transform,
            )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Define in the subclass how to handle specific indexing

        # item_data = self.get_item_data(idx)
        item_data = self.base_dataset[idx]
        img_path = item_data['img_path']
        text_input = item_data['given_text']
        full_text = item_data['full_text']

        # Read image 
        output = {"image": self.read_img(img_path, self.transform), "dset_name": self.dset_name} if img_path is not None else {"dset_name": self.dset_name}
        
        # Prepare the tokenized text output
        text_out = self.prepare_text(
            text_input, full_text, self.tokenizer, self.seq_length, self.img_padding
        )

        output.update(text_out)
        output.update({'img_path':img_path})

        return output

    def read_img(self, img_path, image_transforms):
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32)
        img = image_transforms(image=img)["image"]
        img = img - img.min()
        if img.max() != 0:
            img = img / img.max()
        if img.ndim != 3:
            img = img[..., np.newaxis]
            img = np.concatenate([img, img, img], axis=-1)
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img)
        img = img[:3]
        return img

    def prepare_text(self, given_text, text_all, tokenizer, seq_length, img_padding):
        text_all = tokenizer(text_all)
        input_ids = text_all["input_ids"]
        input_ids.append(tokenizer.eos_token_id)
        input_ids = np.array(input_ids)

        if len(input_ids) < seq_length:
            input_ids = np.pad(
                input_ids,
                (0, seq_length - len(input_ids)),
                "constant",
                constant_values=0,
            )
        else:
            input_ids = input_ids[:seq_length]

        label = copy.deepcopy(input_ids)
        label[label == 0] = -100

        given_text = tokenizer(given_text)
        if len(given_text["input_ids"]) < len(label):
            label[: len(given_text["input_ids"])] = -100
        label = label.tolist()
        label = np.array(img_padding + label)

        given_text = np.array(given_text.input_ids)
        given_text_len = given_text.shape[0]
        given_text_len = np.array(given_text_len)

        if len(given_text) < seq_length:
            given_text = np.pad(
                given_text,
                (0, seq_length - len(given_text)),
                "constant",
                constant_values=0,
            )
        else:
            given_text = given_text[:seq_length]

        output = {
            "input_ids": input_ids,
            "labels": label,
            "given_text": given_text,
            "given_text_len": given_text_len,
        }
        
        return output

    def load_data(self):
        """Method to load dataset data. Implement in subclass."""
        raise NotImplementedError

    def get_item_data(self, idx):
        """Method to return image path and text data. Implement in subclass."""
        raise NotImplementedError




class MultiDataLoader:
    """Multi-dataset loader that yields from multiple dataloaders sequentially."""
    
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        for loader in self.dataloaders:
            for batch in loader:
                yield batch

    def __len__(self):
        return sum(len(loader) for loader in self.dataloaders)


def collate_fn(batch):
    """Custom collate function to handle batching."""
    # Stack tensors
    result = {}
    
    # Handle image data
    if 'image' in batch[0] and batch[0]['image'] is not None:
        result['image'] = torch.stack([item['image'] for item in batch])
    else:
        result['image'] = None
    
    # Handle text data
    if 'input_ids' in batch[0]:
        result['input_ids'] = torch.stack([item['input_ids'] for item in batch])
    
    if 'labels' in batch[0]:
        result['labels'] = torch.stack([item['labels'] for item in batch])
    
    if 'given_text' in batch[0]:
        # Pad to same length
        max_len = max(item['given_text_len'] for item in batch)
        given_texts = []
        for item in batch:
            gt = item['given_text']
            if len(gt) < max_len:
                pad = torch.zeros(max_len - len(gt), dtype=gt.dtype)
                gt = torch.cat([gt, pad])
            given_texts.append(gt)
        result['given_text'] = torch.stack(given_texts)
        result['given_text_len'] = torch.tensor([item['given_text_len'] for item in batch])
    
    # Handle metadata (combine lists)
    result['dset_name'] = [item for sublist in [item['dset_name'] for item in batch] for item in sublist]
    result['ans_type'] = [item for sublist in [item['ans_type'] for item in batch] for item in sublist]
    result['img_path'] = [item for sublist in [item['img_path'] for item in batch] for item in sublist]
    result['question'] = [item for sublist in [item['question'] for item in batch] for item in sublist]
    result['mc_options'] = [item for sublist in [item['mc_options'] for item in batch] for item in sublist]
    
    # Handle VL instruct specific fields
    if 'attention_mask' in batch[0]:
        result['attention_mask'] = torch.stack([item['attention_mask'] for item in batch])
    if 'pixel_values' in batch[0]:
        result['pixel_values'] = torch.stack([item['pixel_values'] for item in batch])
    if 'image_grid_thw' in batch[0]:
        result['image_grid_thw'] = torch.stack([item['image_grid_thw'] for item in batch])
    
    return result


def get_sequential_dataloaders(cfg, phase):
    """Get data loaders for training or validation.
    
    Args:
        cfg: Configuration object
        phase: 'train' or 'valid'
        
    Returns:
        MultiDataLoader with data from all specified datasets
    """
    dataloaders = []
    name_list = cfg.dataset_list

    # Create DataLoader for each dataset separately
    for name in name_list:
        transforms = Transforms(cfg)
        if phase == 'train':
            transform = transforms.train_transforms
        elif phase == 'valid':
            transform = transforms.valid_transforms
        else:
            raise ValueError(f"Invalid phase: {phase}")


        tokenizer = AutoTokenizer.from_pretrained(
            cfg.LLM.model_name,
            cache_dir=cfg.LLM.cache_dir,
            force_download=False,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # tokenizer.padding_side = 'right'
        img_padding = [-100 for _ in range(cfg.img_token_num)]
        
        dataset = ImageTextDataset(
            config=cfg,
            dataset_name=name,
            phase=phase,
            tokenizer=tokenizer,
            img_padding=img_padding,
            transform=transform,
        )

        print(f'---- DATASET LENGTH ----')
        print(f'Name of the dataset: {name} and phase: {phase}')
        print(len(dataset))

        # Optional subsampling for inference mode
        if cfg.is_inference and len(dataset) > cfg.n_inference_lim:
            num_samples = cfg.n_inference_lim
            seed = 42
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            pl.seed_everything(seed)
            random.seed(seed)

            indices = random.sample(range(len(dataset)), num_samples)
            dataset = Subset(dataset, indices)

        # Use DistributedSampler for multi-GPU
        if dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=(phase == 'train'))
        else:
            sampler = None

        # Create DataLoader for each individual dataset
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            shuffle=(sampler is None and phase == 'train'),
            prefetch_factor=cfg.prefetch_factor,
            # collate_fn=collate_fn,
        )

        dataloaders.append(loader)

    dataloader = MultiDataLoader(dataloaders)
    return dataloader

