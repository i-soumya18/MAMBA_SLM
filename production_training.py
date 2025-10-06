"""
Production Training Infrastructure
Supports multi-GPU distributed training with DeepSpeed, FSDP, and DDP
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Comprehensive training configuration for production runs
    """
    # Model
    model_size: str = "1.3B"  # "1.3B", "2.7B", "6.7B", "13B"
    model_name: str = "MAMBA-SLM-1.3B"
    
    # Dataset
    dataset_name: str = "combined"  # Will combine multiple sources
    dataset_path: Optional[str] = None
    max_samples: int = -1  # -1 for all
    validation_split: float = 0.01  # 1% for validation
    
    # Training
    num_train_epochs: int = 1
    max_steps: int = 100000  # Override epochs if specified
    gradient_accumulation_steps: int = 4
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 2000
    warmup_ratio: Optional[float] = None
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True  # BF16 preferred for large models
    fp16_opt_level: str = "O1"
    
    # Distributed training
    world_size: int = 1  # Number of GPUs
    local_rank: int = -1
    distributed_type: str = "deepspeed"  # "deepspeed", "fsdp", "ddp"
    
    # DeepSpeed
    deepspeed: bool = True
    deepspeed_config_file: Optional[str] = None
    zero_stage: int = 2  # 0, 1, 2, 3 (3 for very large models)
    offload_optimizer: bool = False  # Offload to CPU
    offload_param: bool = False  # Offload parameters to CPU
    
    # FSDP (PyTorch native)
    fsdp: bool = False
    fsdp_sharding_strategy: str = "full_shard"  # "full_shard", "shard_grad_op", "no_shard"
    fsdp_auto_wrap_policy: str = "transformer_based"
    
    # Gradient checkpointing
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Dict = None
    
    # Logging
    logging_dir: str = "./logs"
    logging_steps: int = 10
    log_level: str = "info"
    report_to: List[str] = None  # ["wandb", "tensorboard"]
    
    # Checkpointing
    output_dir: str = "./checkpoints_production"
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 5  # Keep only best 5 checkpoints
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 1000
    eval_accumulation_steps: int = 1
    
    # Data loading
    dataloader_num_workers: int = 8
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
    
    # Reproducibility
    seed: int = 42
    
    # Advanced features
    use_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    # Performance
    torch_compile: bool = False  # PyTorch 2.0 compile
    torch_compile_backend: str = "inductor"
    
    # W&B
    wandb_project: str = "mamba-slm-production"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        if self.report_to is None:
            self.report_to = ["tensorboard"]
        
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        if self.gradient_checkpointing_kwargs is None:
            self.gradient_checkpointing_kwargs = {"use_reentrant": False}
        
        # Compute effective batch size
        self.effective_batch_size = (
            self.per_device_train_batch_size 
            * self.gradient_accumulation_steps 
            * self.world_size
        )
        
        # Set run name if not specified
        if self.wandb_run_name is None:
            self.wandb_run_name = f"{self.model_name}_bs{self.effective_batch_size}_lr{self.learning_rate}"


def create_deepspeed_config(
    training_config: TrainingConfig,
    output_path: Optional[str] = None
) -> Dict:
    """
    Create DeepSpeed configuration for production training
    
    ZeRO Stages:
    - Stage 0: Disabled (DDP)
    - Stage 1: Optimizer state partitioning
    - Stage 2: Optimizer + Gradient partitioning (recommended for most cases)
    - Stage 3: Optimizer + Gradient + Parameter partitioning (for very large models)
    """
    config = {
        "train_batch_size": training_config.effective_batch_size,
        "train_micro_batch_size_per_gpu": training_config.per_device_train_batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "gradient_clipping": training_config.max_grad_norm,
        "steps_per_print": training_config.logging_steps,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": training_config.learning_rate,
                "betas": [training_config.adam_beta1, training_config.adam_beta2],
                "eps": training_config.adam_epsilon,
                "weight_decay": training_config.weight_decay
            }
        },
        
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": training_config.learning_rate,
                "warmup_num_steps": training_config.warmup_steps,
                "total_num_steps": training_config.max_steps
            }
        },
        
        "zero_optimization": {
            "stage": training_config.zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
        },
        
        "fp16": {
            "enabled": training_config.fp16 and not training_config.bf16,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        "bf16": {
            "enabled": training_config.bf16
        },
        
        "activation_checkpointing": {
            "partition_activations": training_config.gradient_checkpointing,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": False,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        },
        
        "wall_clock_breakdown": False,
        "tensorboard": {
            "enabled": "tensorboard" in training_config.report_to,
            "output_path": training_config.logging_dir,
            "job_name": training_config.wandb_run_name
        }
    }
    
    # Add ZeRO-3 specific configs
    if training_config.zero_stage == 3:
        config["zero_optimization"].update({
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        })
    
    # Add CPU offloading if enabled
    if training_config.offload_optimizer:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    if training_config.offload_param:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    # Save config if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"DeepSpeed config saved to {output_path}")
    
    return config


def create_fsdp_config(training_config: TrainingConfig) -> Dict:
    """
    Create FSDP (Fully Sharded Data Parallel) configuration
    PyTorch native alternative to DeepSpeed
    """
    config = {
        "fsdp_config": {
            "fsdp_transformer_layer_cls_to_wrap": ["ProductionMambaBlock", "ProductionTransformerBlock"],
            "fsdp_backward_prefetch": "backward_pre",
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_forward_prefetch": False,
            "fsdp_offload_params": training_config.offload_param,
            "fsdp_sharding_strategy": training_config.fsdp_sharding_strategy,
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "fsdp_sync_module_states": True,
            "fsdp_use_orig_params": True,
        }
    }
    
    return config


def setup_distributed_environment():
    """
    Setup distributed training environment
    Detects available GPUs and configures torch.distributed
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        logger.info(f"Distributed training: Rank {local_rank}/{world_size}")
    else:
        logger.info("Single GPU training")
    
    return local_rank, world_size


def get_training_args(config: TrainingConfig):
    """
    Convert TrainingConfig to HuggingFace TrainingArguments
    """
    from transformers import TrainingArguments
    
    # Create DeepSpeed config if needed
    deepspeed_config = None
    if config.deepspeed:
        ds_config_path = os.path.join(config.output_dir, "ds_config.json")
        deepspeed_config = create_deepspeed_config(config, ds_config_path)
        config.deepspeed_config_file = ds_config_path
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        
        # Training
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # Optimization
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        max_grad_norm=config.max_grad_norm,
        
        # Schedule
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        warmup_ratio=config.warmup_ratio,
        
        # Mixed precision
        fp16=config.fp16,
        bf16=config.bf16,
        fp16_full_eval=config.fp16,
        bf16_full_eval=config.bf16,
        
        # Distributed
        local_rank=config.local_rank,
        deepspeed=config.deepspeed_config_file if config.deepspeed else None,
        fsdp=config.fsdp_sharding_strategy if config.fsdp else "",
        fsdp_config=create_fsdp_config(config) if config.fsdp else None,
        
        # Gradient checkpointing
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs=config.gradient_checkpointing_kwargs,
        
        # Logging
        logging_dir=config.logging_dir,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        report_to=config.report_to,
        
        # Checkpointing
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        
        # Evaluation
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        eval_accumulation_steps=config.eval_accumulation_steps,
        
        # Data loading
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        
        # Reproducibility
        seed=config.seed,
        data_seed=config.seed,
        
        # Performance
        torch_compile=config.torch_compile,
        
        # W&B
        run_name=config.wandb_run_name,
        
        # Other
        remove_unused_columns=False,
        include_tokens_per_second=True,
        ddp_find_unused_parameters=False,
    )
    
    return training_args


def estimate_training_time(
    config: TrainingConfig,
    tokens_per_sample: int = 2048,
    gpu_type: str = "A100"
) -> Dict:
    """
    Estimate training time and resource requirements
    
    Based on empirical measurements:
    - A100 80GB: ~300 TFLOPS BF16
    - A100 40GB: ~300 TFLOPS BF16
    - V100 32GB: ~125 TFLOPS FP16
    - RTX 3090: ~70 TFLOPS FP16
    """
    gpu_tflops = {
        "A100": 300,
        "V100": 125,
        "RTX3090": 70,
        "RTX4090": 83,
        "H100": 1000,
    }
    
    tflops = gpu_tflops.get(gpu_type, 100)
    
    # Estimate FLOPS per token (approximate)
    from production_config import get_config
    model_config = get_config(config.model_size)
    
    # Forward pass: ~6 * params * tokens (MatMul dominates)
    # Backward pass: ~2 * forward
    flops_per_token = 6 * model_config.total_params * 3  # forward + backward
    
    # Total tokens to train
    total_tokens = config.max_steps * config.effective_batch_size * tokens_per_sample
    
    # Total FLOPS
    total_flops = flops_per_token * total_tokens
    
    # Time in seconds (accounting for ~50% efficiency)
    time_seconds = (total_flops / (tflops * 1e12)) / 0.5
    
    # Convert to hours/days
    time_hours = time_seconds / 3600
    time_days = time_hours / 24
    
    # Memory estimate per GPU
    model_memory_gb = (model_config.total_params * 2) / (1024**3)  # BF16
    optimizer_memory_gb = model_memory_gb * 2  # Adam states
    gradient_memory_gb = model_memory_gb
    activation_memory_gb = (
        config.per_device_train_batch_size 
        * tokens_per_sample 
        * model_config.d_model 
        * model_config.n_layers 
        * 4  # rough estimate
    ) / (1024**3)
    
    total_memory_gb = model_memory_gb + optimizer_memory_gb + gradient_memory_gb + activation_memory_gb
    
    # With DeepSpeed ZeRO-2
    if config.zero_stage == 2:
        total_memory_gb = (model_memory_gb + (optimizer_memory_gb + gradient_memory_gb) / config.world_size 
                          + activation_memory_gb)
    
    # With DeepSpeed ZeRO-3
    elif config.zero_stage == 3:
        total_memory_gb = ((model_memory_gb + optimizer_memory_gb + gradient_memory_gb) / config.world_size 
                          + activation_memory_gb)
    
    return {
        "total_tokens": f"{total_tokens / 1e9:.2f}B",
        "total_flops": f"{total_flops / 1e15:.2f}P",
        "estimated_time_hours": f"{time_hours:.1f}",
        "estimated_time_days": f"{time_days:.2f}",
        "memory_per_gpu_gb": f"{total_memory_gb:.1f}",
        "tokens_per_second": f"{total_tokens / time_seconds / 1000:.1f}K",
        "model_params": f"{model_config.total_params / 1e9:.2f}B",
        "effective_batch_size": config.effective_batch_size,
    }


# Predefined production training configurations
PRODUCTION_TRAINING_CONFIGS = {
    "quick_test": TrainingConfig(
        model_size="1.3B",
        max_steps=1000,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        eval_steps=200,
        save_steps=200,
        deepspeed=False,
    ),
    
    "small_scale": TrainingConfig(
        model_size="1.3B",
        max_steps=50000,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        world_size=4,
        zero_stage=2,
    ),
    
    "medium_scale": TrainingConfig(
        model_size="2.7B",
        max_steps=100000,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        world_size=8,
        zero_stage=2,
    ),
    
    "large_scale": TrainingConfig(
        model_size="6.7B",
        max_steps=150000,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        world_size=8,
        zero_stage=3,
        offload_optimizer=True,
    ),
    
    "xlarge_scale": TrainingConfig(
        model_size="13B",
        max_steps=200000,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        world_size=8,
        zero_stage=3,
        offload_optimizer=True,
        offload_param=True,
    ),
}


if __name__ == "__main__":
    print("=" * 70)
    print("Production Training Infrastructure")
    print("=" * 70)
    
    # Show all configurations
    for name, config in PRODUCTION_TRAINING_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Configuration: {name}")
        print(f"{'='*70}")
        print(f"Model: {config.model_name}")
        print(f"GPUs: {config.world_size}")
        print(f"Effective Batch Size: {config.effective_batch_size}")
        print(f"Max Steps: {config.max_steps:,}")
        print(f"DeepSpeed ZeRO Stage: {config.zero_stage}")
        
        # Estimate training time
        estimates = estimate_training_time(config, gpu_type="A100")
        print(f"\n📊 Estimates (on {config.world_size}x A100):")
        for key, value in estimates.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("💡 To use a configuration:")
    print("  from production_training import PRODUCTION_TRAINING_CONFIGS")
    print("  config = PRODUCTION_TRAINING_CONFIGS['medium_scale']")
    print("=" * 70)
