"""
Comprehensive Training Script for Hybrid Mamba-Transformer
Supports pre-training, fine-tuning, and LoRA/QLoRA
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, TrainingArguments, Trainer
from pathlib import Path
import json
import sys
from typing import Optional

# Import modules
try:
    from dataset_loader import create_dataset, TextDatasetLoader
    from advanced_sampling import AdvancedSampler, GenerationConfig
    from quantization import QuantizationConfig, quantize_model
    from lora_finetuning import LoRAConfig, QLoRAConfig, add_lora_to_model, prepare_model_for_qlora
    from flash_attention import replace_attention_with_flash
    from extended_context import upgrade_model_context_length
except ImportError:
    print("Warning: Some enhancement modules not found. Using basic functionality.")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Hybrid Mamba-Transformer Model")
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--vocab_size', type=int, default=32000)
    model_group.add_argument('--d_model', type=int, default=512)
    model_group.add_argument('--n_layers', type=int, default=8)
    model_group.add_argument('--n_heads', type=int, default=8)
    model_group.add_argument('--d_state', type=int, default=16)
    model_group.add_argument('--expand_factor', type=int, default=2)
    model_group.add_argument('--dropout', type=float, default=0.1)
    model_group.add_argument('--max_seq_length', type=int, default=1024)
    
    # Training configuration
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--output_dir', type=str, default='./outputs')
    train_group.add_argument('--batch_size', type=int, default=2)
    train_group.add_argument('--gradient_accumulation_steps', type=int, default=8)
    train_group.add_argument('--learning_rate', type=float, default=5e-4)
    train_group.add_argument('--weight_decay', type=float, default=0.01)
    train_group.add_argument('--num_train_epochs', type=int, default=3)
    train_group.add_argument('--max_steps', type=int, default=-1)
    train_group.add_argument('--warmup_steps', type=int, default=500)
    train_group.add_argument('--save_steps', type=int, default=500)
    train_group.add_argument('--eval_steps', type=int, default=500)
    train_group.add_argument('--logging_steps', type=int, default=100)
    train_group.add_argument('--save_total_limit', type=int, default=3)
    
    # Dataset configuration
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--dataset', type=str, default='wikitext',
                           help='Dataset name (wikitext, c4, openwebtext) or path to files')
    data_group.add_argument('--dataset_split', type=str, default='train')
    data_group.add_argument('--num_samples', type=int, default=None,
                           help='Limit number of samples (for testing)')
    data_group.add_argument('--cache_dir', type=str, default='./data_cache')
    
    # Optimization features
    opt_group = parser.add_argument_group('Optimization Features')
    opt_group.add_argument('--fp16', action='store_true', default=True)
    opt_group.add_argument('--bf16', action='store_true', default=False)
    opt_group.add_argument('--gradient_checkpointing', action='store_true', default=True)
    opt_group.add_argument('--flash_attention', action='store_true', default=False)
    
    # LoRA/QLoRA configuration
    lora_group = parser.add_argument_group('LoRA Configuration')
    lora_group.add_argument('--use_lora', action='store_true', default=False)
    lora_group.add_argument('--use_qlora', action='store_true', default=False)
    lora_group.add_argument('--lora_r', type=int, default=8)
    lora_group.add_argument('--lora_alpha', type=int, default=16)
    lora_group.add_argument('--lora_dropout', type=float, default=0.05)
    
    # Quantization
    quant_group = parser.add_argument_group('Quantization')
    quant_group.add_argument('--load_in_8bit', action='store_true', default=False)
    quant_group.add_argument('--load_in_4bit', action='store_true', default=False)
    
    # Fine-tuning
    ft_group = parser.add_argument_group('Fine-tuning')
    ft_group.add_argument('--pretrained_model', type=str, default=None,
                         help='Path to pretrained model for fine-tuning')
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    
    return parser.parse_args()


def setup_model(args):
    """Initialize model with specified configuration"""
    # Import here to avoid circular imports
    from MAMBA_SLM import HybridMambaTransformer
    
    print("\n" + "="*60)
    print("Setting up model...")
    print("="*60)
    
    # Create model
    model = HybridMambaTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_state=args.d_state,
        expand_factor=args.expand_factor,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length
    )
    
    # Load pretrained weights if specified
    if args.pretrained_model:
        print(f"Loading pretrained model from {args.pretrained_model}")
        try:
            state_dict = torch.load(
                f"{args.pretrained_model}/pytorch_model.bin",
                map_location='cpu'
            )
            model.load_state_dict(state_dict, strict=False)
            print("✓ Pretrained weights loaded")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    # Apply optimizations
    if args.flash_attention:
        try:
            model = replace_attention_with_flash(model, use_flash=True)
            print("✓ Flash Attention enabled")
        except Exception as e:
            print(f"Warning: Could not enable Flash Attention: {e}")
    
    # Apply quantization
    if args.load_in_8bit or args.load_in_4bit:
        try:
            quant_config = QuantizationConfig(
                load_in_8bit=args.load_in_8bit,
                load_in_4bit=args.load_in_4bit
            )
            model = quantize_model(model, quant_config)
            print(f"✓ Model quantized ({'8-bit' if args.load_in_8bit else '4-bit'})")
        except Exception as e:
            print(f"Warning: Could not quantize model: {e}")
    
    # Apply LoRA/QLoRA
    if args.use_qlora:
        try:
            qlora_config = QLoRAConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout
            )
            model = prepare_model_for_qlora(model, qlora_config)
            print("✓ QLoRA enabled")
        except Exception as e:
            print(f"Warning: Could not enable QLoRA: {e}")
    elif args.use_lora:
        try:
            lora_config = LoRAConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout
            )
            model = add_lora_to_model(model, lora_config)
            print("✓ LoRA enabled")
        except Exception as e:
            print(f"Warning: Could not enable LoRA: {e}")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    return model


def setup_dataset(args, tokenizer):
    """Load and prepare dataset"""
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    
    try:
        dataset = create_dataset(
            args.dataset,
            tokenizer,
            max_length=args.max_seq_length,
            split=args.dataset_split,
            cache_dir=args.cache_dir,
            num_samples=args.num_samples
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using default sample dataset")
        
        # Fallback to sample dataset
        from dataset_loader import SimpleTextDataset
        sample_texts = [
            "The future of artificial intelligence is bright.",
            "Machine learning models continue to improve.",
            "Natural language processing enables human-computer interaction.",
        ] * 100  # Repeat for more samples
        
        return SimpleTextDataset(sample_texts, tokenizer, args.max_seq_length)


def main():
    """Main training function"""
    args = parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    
    # Setup model
    model = setup_model(args)
    model = model.to(device)
    
    # Load dataset
    train_dataset = setup_dataset(args, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        prediction_loss_only=True,
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=None,
        seed=args.seed,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    
    # Save final model
    print("\n" + "="*60)
    print("Saving model...")
    print("="*60)
    
    output_path = Path(args.output_dir) / "final_model"
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    # Save configuration
    config = {
        'vocab_size': args.vocab_size,
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'd_state': args.d_state,
        'expand_factor': args.expand_factor,
        'dropout': args.dropout,
        'max_seq_length': args.max_seq_length,
    }
    
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Model saved to {output_path}")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
