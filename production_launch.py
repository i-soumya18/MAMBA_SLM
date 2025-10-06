"""
Production Training Launcher
Complete end-to-end training script for production models
Integrates all components: model, data, training, evaluation
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk, Dataset, concatenate_datasets

# Import production components
from production_config import get_config, print_config_summary, PRODUCTION_CONFIGS
from production_model import ProductionHybridModel, create_production_model
from production_training import (
    TrainingConfig, get_training_args, create_deepspeed_config,
    setup_distributed_environment, PRODUCTION_TRAINING_CONFIGS
)
from production_dataset import CURATION_CONFIGS, CurationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Production Model Training")
    
    # Model configuration
    parser.add_argument(
        "--model_size",
        type=str,
        default="1.3B",
        choices=["1.3B", "2.7B", "6.7B", "13B"],
        help="Model size to train"
    )
    
    # Training configuration
    parser.add_argument(
        "--training_config",
        type=str,
        default="quick_test",
        choices=list(PRODUCTION_TRAINING_CONFIGS.keys()),
        help="Pre-defined training configuration"
    )
    
    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to tokenized dataset (if already prepared)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name (if preparing fresh)"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration/subset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="Maximum samples for testing (-1 for all)"
    )
    
    # Tokenizer
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Tokenizer to use"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_production",
        help="Output directory for checkpoints"
    )
    
    # Overrides
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max training steps"
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=None,
        help="Override batch size per device"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps"
    )
    
    # Features
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable BF16 (use FP16 instead)"
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient training"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA rank"
    )
    
    # Evaluation
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip final evaluation"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Override evaluation frequency"
    )
    
    # Distributed
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    # Testing
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print configuration and exit (don't train)"
    )
    
    return parser.parse_args()


def load_and_prepare_data(args, tokenizer):
    """
    Load and prepare dataset for training
    """
    logger.info("Loading and preparing dataset...")
    
    # If pre-tokenized data exists, load it
    if args.data_path and Path(args.data_path).exists():
        logger.info(f"Loading pre-tokenized dataset from {args.data_path}")
        dataset = load_from_disk(args.data_path)
        
        # Split into train/val if not already
        if "validation" not in dataset:
            dataset = dataset.train_test_split(test_size=0.01, seed=42)
            dataset["validation"] = dataset["test"]
            del dataset["test"]
        
        return dataset
    
    # Otherwise, load from HuggingFace and tokenize
    logger.info(f"Loading dataset: {args.dataset_name}")
    from datasets import load_dataset
    
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config if args.dataset_config != "None" else None,
        split="train"
    )
    
    # Limit samples for testing
    if args.max_samples > 0 and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
        logger.info(f"Limited to {args.max_samples} samples")
    
    # Tokenize
    logger.info("Tokenizing dataset...")
    
    def tokenize_function(examples):
        # Get text column (try common names)
        text_column = None
        for col in ["text", "content", "article", "document"]:
            if col in examples:
                text_column = col
                break
        
        if text_column is None:
            raise ValueError(f"Could not find text column in {examples.keys()}")
        
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=2048,  # Standard for pre-training
            padding="max_length",
            return_tensors=None,
        )
    
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    # Split train/val
    dataset = dataset.train_test_split(test_size=0.01, seed=42)
    dataset["validation"] = dataset["test"]
    del dataset["test"]
    
    logger.info(f"Dataset prepared: {len(dataset['train'])} train, {len(dataset['validation'])} val")
    
    return dataset


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup distributed environment
    local_rank, world_size = setup_distributed_environment()
    if local_rank != -1:
        args.local_rank = local_rank
    
    # Load configurations
    logger.info(f"Loading configurations for {args.model_size} model...")
    
    # Get model config
    model_config = get_config(args.model_size)
    
    # Get training config
    training_config = PRODUCTION_TRAINING_CONFIGS[args.training_config]
    training_config.model_size = args.model_size
    training_config.output_dir = args.output_dir
    training_config.local_rank = args.local_rank
    training_config.world_size = world_size if world_size > 0 else 1
    
    # Apply overrides
    if args.max_steps is not None:
        training_config.max_steps = args.max_steps
    if args.per_device_batch_size is not None:
        training_config.per_device_train_batch_size = args.per_device_batch_size
    if args.learning_rate is not None:
        training_config.learning_rate = args.learning_rate
    if args.gradient_accumulation_steps is not None:
        training_config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.eval_steps is not None:
        training_config.eval_steps = args.eval_steps
    
    # Features
    if args.no_bf16:
        training_config.bf16 = False
        training_config.fp16 = True
    if args.no_gradient_checkpointing:
        training_config.gradient_checkpointing = False
    if args.use_lora:
        training_config.use_lora = True
        training_config.lora_r = args.lora_r
    
    # Update effective batch size
    training_config.effective_batch_size = (
        training_config.per_device_train_batch_size
        * training_config.gradient_accumulation_steps
        * training_config.world_size
    )
    
    # Print configurations
    print("\n" + "=" * 80)
    print("PRODUCTION TRAINING CONFIGURATION")
    print("=" * 80)
    print_config_summary(model_config)
    print("\n" + "=" * 80)
    print("TRAINING SETUP")
    print("=" * 80)
    print(f"  GPUs: {training_config.world_size}")
    print(f"  Batch Size/GPU: {training_config.per_device_train_batch_size}")
    print(f"  Gradient Accumulation: {training_config.gradient_accumulation_steps}")
    print(f"  Effective Batch Size: {training_config.effective_batch_size}")
    print(f"  Max Steps: {training_config.max_steps:,}")
    print(f"  Learning Rate: {training_config.learning_rate}")
    print(f"  Mixed Precision: {'BF16' if training_config.bf16 else 'FP16'}")
    print(f"  Gradient Checkpointing: {training_config.gradient_checkpointing}")
    print(f"  DeepSpeed: {'✅' if training_config.deepspeed else '❌'}")
    if training_config.deepspeed:
        print(f"  ZeRO Stage: {training_config.zero_stage}")
    print(f"  LoRA: {'✅' if training_config.use_lora else '❌'}")
    print(f"  Output: {training_config.output_dir}")
    print("=" * 80)
    
    if args.dry_run:
        logger.info("Dry run mode - exiting without training")
        return
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Verify vocab size matches
    if len(tokenizer) != model_config.vocab_size:
        logger.warning(f"Tokenizer vocab size ({len(tokenizer)}) != model vocab size ({model_config.vocab_size})")
        logger.warning("Updating model vocab size to match tokenizer")
        model_config.vocab_size = len(tokenizer)
    
    # Load and prepare data
    dataset = load_and_prepare_data(args, tokenizer)
    
    # Create model
    logger.info("Creating model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ProductionHybridModel(model_config)
    
    # Apply LoRA if requested
    if training_config.use_lora:
        logger.info(f"Applying LoRA (r={training_config.lora_r})...")
        from peft import LoraConfig, get_peft_model
        
        peft_config = LoraConfig(
            r=training_config.lora_r,
            lora_alpha=training_config.lora_alpha,
            lora_dropout=training_config.lora_dropout,
            target_modules=training_config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Enable gradient checkpointing
    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Get training arguments
    training_args = get_training_args(training_config)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Create Trainer
    logger.info("Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    train_result = trainer.train()
    
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Final loss: {train_result.training_loss:.4f}")
    
    # Save final model
    logger.info(f"Saving model to {training_config.output_dir}/final_model")
    trainer.save_model(f"{training_config.output_dir}/final_model")
    tokenizer.save_pretrained(f"{training_config.output_dir}/final_model")
    
    # Save training metrics
    metrics_path = f"{training_config.output_dir}/training_metrics.json"
    import json
    with open(metrics_path, 'w') as f:
        json.dump(train_result.metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Evaluation
    if not args.skip_evaluation:
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        
        logger.info("Evaluation results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Save eval metrics
        eval_path = f"{training_config.output_dir}/eval_metrics.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"Evaluation metrics saved to {eval_path}")
    
    logger.info("=" * 80)
    logger.info("✅ ALL DONE!")
    logger.info(f"Model saved to: {training_config.output_dir}/final_model")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
