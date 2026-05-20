"""Training script for Medical Report Generation model."""
import os
import argparse
import torch
import yaml
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.plmodel import PreTrainingLightningModule
from src.training.train_utils import get_strategy, get_logger, get_callback, get_precision, get_early_stopping

pl.seed_everything(42)
torch.set_float32_matmul_precision("high")


def train(config):
    """Main training function."""
    strategy, num_devices = get_strategy(config)
    wandb_logger = get_logger(config)
    precision = get_precision(config)    


    pre_trained_path = config.ckpt_path if config.ckpt_path != "" else None
    if pre_trained_path:
        print(f'-----Checkpoint weights are loaded-----')
        torch.cuda.empty_cache()  # Clear GPU memory
        model = PreTrainingLightningModule.load_from_checkpoint(
            pre_trained_path, 
            config=config,
            map_location='cpu'
        )
    else:
        print(f'-----Model weights are initialized randomly-----')
        model = PreTrainingLightningModule(config)
    
    
    checkpoint_callback = get_callback(config)
    early_stopping = get_early_stopping(config)  # Add this line

    trainer = pl.Trainer(
        devices=num_devices,
        num_nodes=1,
        accelerator="gpu",
        strategy=strategy,
        max_epochs=config.max_epochs,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, early_stopping],  # Add early_stopping here
        logger=wandb_logger,
        precision=precision,
        accumulate_grad_batches=config.accumulation,
        use_distributed_sampler=False,
    )


    trainer.logger._version = datetime.now().strftime("%m-%d-%H-%M-%S")

    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(dict(config))

    trainer.fit(
        model=model,
        ckpt_path=config.ckpt_path if config.ckpt_path != "" else None,
    )


def update_config(config, args):
    """Update config with command line arguments."""
    print(f'args : {args}')

    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    if args.project:
        config["project"] = args.project
    if args.ckpt_path:
        config["ckpt_path"] = args.ckpt_path
    if args.lora_rank:
        config["lora_rank"] = args.lora_rank
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.img_token_num:
        config["img_token_num"] = args.img_token_num
    if args.vision_module:
        config["vision_module"] = args.vision_module
    if args.model_name:
        config["LLM"]["model_name"] = args.model_name
    if args.dataset_list:
        config["dataset_list"] = args.dataset_list
    if args.max_epochs:
        config["max_epochs"] = args.max_epochs
    if args.accumulation:
        config["accumulation"] = args.accumulation
    if args.seq_length:
        config["LLM"]["seq_length"] = args.seq_length

    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Training script for Medical Report Generation"
    )
    parser.add_argument("--config", type=str, default="experiments/configs/training.yaml",
                       help="Path to config file")
    parser.add_argument("--experiment_name", type=str, help="experiment_name")
    parser.add_argument("--project", type=str, help="project")
    parser.add_argument("--img_token_num", type=int, help="img_token_num")
    parser.add_argument("--lora_rank", type=int, help="lora_rank")
    parser.add_argument("--max_epochs", type=int, help="max_epochs")
    parser.add_argument("--accumulation", type=int, help="accumulation")
    parser.add_argument("--seq_length", type=int, help="seq_length")
    parser.add_argument("--batch_size", type=int, help="batch_size")
    parser.add_argument("--model_name", type=str, help="model_name")
    parser.add_argument("--vision_module", type=str, help="vision_module")
    parser.add_argument("--ckpt_path", type=str, help="ckpt_path")
    parser.add_argument("--dataset_list", help="dataset_list", nargs="+")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Make path relative to project root
        config_path = Path(__file__).parent.parent.parent / args.config
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = OmegaConf.create(config)
    config = update_config(config, args)

    train(config)


if __name__ == "__main__":
    main()

