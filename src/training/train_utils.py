"""Training utilities for PyTorch Lightning."""
import os 
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy




def get_strategy(config):
    """Get the distributed training strategy based on model configuration."""
    if config.LLM.model_name in {"axiong/PMC_LLaMA_13B", "google/gemma-2-27b-it", "google/gemma-2-27b"}:
        strategy = "deepspeed_stage_2"
        num_devices = 'auto'
    else:
        strategy = DDPStrategy(find_unused_parameters=True)
        num_devices = 'auto'
    return strategy, num_devices


def get_logger(config):
    """Get WandB logger for experiment tracking."""
    wandb_logger = WandbLogger(
        log_model="False",
        project=config.project,
        name=config.experiment_name,
        save_dir=config.wandb_folder_path,
    )
    return wandb_logger


def get_callback(config):
    """Get ModelCheckpoint callback for saving best models."""
    os.makedirs(config.checkpoint_dirpath, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dirpath + config.experiment_name + "/",
        save_top_k=1,
        monitor="valid_loss",
        mode="min",
        filename="{epoch}-{valid_loss:.3f}",
    )
    return checkpoint_callback

def get_early_stopping(config):
    """Get EarlyStopping callback to stop training when validation loss stops improving."""
    early_stopping = EarlyStopping(
        monitor="valid_loss",
        patience=3,  # Stop after 3 epochs without improvement
        mode="min",  # We want to minimize the loss
        verbose=True,
        strict=True,  # Crash if monitored metric is not found
    )
    return early_stopping


def get_precision(config):
    """Get precision setting based on model type."""
    if config.LLM.model_name in {
        'google/gemma-2-2b', 
        'google/gemma-2-2b-it', 
        'google/gemma-2-9b', 
        'google/gemma-2-9b-it', 
        'google/gemma-2-27b-it', 
        'google/gemma-2-27b'
    }:
        return "bf16-true"
    return "bf16-mixed"

