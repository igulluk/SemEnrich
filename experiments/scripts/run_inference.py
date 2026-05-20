"""Inference script for Medical Report Generation model."""
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
from src.training.train_utils import get_strategy, get_logger, get_callback, get_precision

pl.seed_everything(42)
torch.set_float32_matmul_precision("high")


def run_inference(config):
    """Main inference function."""
    strategy, num_devices = get_strategy(config)
    wandb_logger = get_logger(config)
    precision = get_precision(config)

    assert config.ckpt_path is not None and config.ckpt_path != "", "Checkpoint path must be provided for inference"
    pre_trained_path = config.ckpt_path 

    # Load model from checkpoint
    # model = PreTrainingLightningModule.load_from_checkpoint(pre_trained_path, config=config)

    print(f'-----Checkpoint weights are loaded-----')
    torch.cuda.empty_cache()  # Clear GPU memory
    model = PreTrainingLightningModule.load_from_checkpoint(
        pre_trained_path, 
        config=config,
        map_location='cpu'
    )


    checkpoint_callback = get_callback(config)

    trainer = pl.Trainer(
        devices=num_devices,
        num_nodes=1,
        accelerator="gpu",
        strategy=strategy,
        max_epochs=config.max_epochs,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        precision=precision,
        accumulate_grad_batches=config.accumulation,
        use_distributed_sampler=False,
    )

    trainer.logger._version = datetime.now().strftime("%m-%d-%H-%M-%S")

    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(dict(config))

    # Run validation/inference
    trainer.validate(   
        model=model,
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
    if args.n_inference_lim:
        config["n_inference_lim"] = args.n_inference_lim
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
    if args.seq_length:
        config["LLM"]["seq_length"] = args.seq_length
    if args.max_epochs:
        config["max_epochs"] = args.max_epochs
        
    # Inference specific settings
    assert len(config["dataset_list"]) == 1, "Only one dataset allowed for inference"
    config['is_inference'] = True 
    config["batch_size"] = 4 
    
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inference script for Medical Report Generation"
    )
    parser.add_argument("--config", type=str, default="experiments/configs/training.yaml",
                       help="Path to config file")
    parser.add_argument("--experiment_name", type=str, required=True, help="experiment_name")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--img_token_num", type=int, help="img_token_num")
    parser.add_argument("--lora_rank", type=int, help="lora_rank")
    parser.add_argument("--n_inference_lim", type=int, help="n_inference_lim")
    parser.add_argument("--max_epochs", type=int, help="max_epochs")
    parser.add_argument("--batch_size", type=int, help="batch_size")
    parser.add_argument("--model_name", type=str, help="model_name")
    parser.add_argument("--vision_module", type=str, help="vision_module")
    parser.add_argument("--dataset_list", help="dataset_list", nargs="+")
    parser.add_argument("--seq_length", type=int, help="seq_length")
    parser.add_argument("--project", type=str, help="project")


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

    
    run_inference(config)


if __name__ == "__main__":
    main()

