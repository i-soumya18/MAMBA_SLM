"""
Evaluation and Inference Script for Hybrid Mamba-Transformer
Supports various generation modes and benchmarking
"""

import argparse
import torch
from transformers import AutoTokenizer
from pathlib import Path
import json
import time
from typing import List, Dict

# Import modules
try:
    from advanced_sampling import AdvancedSampler, GenerationConfig
    from quantization import QuantizationConfig, load_quantized_model
    from onnx_export import ONNXInferenceSession, export_to_onnx
except ImportError:
    print("Warning: Some enhancement modules not found")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate/Inference Hybrid Mamba-Transformer")
    
    # Model
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--device', type=str, default='cuda')
    
    # Generation mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--interactive', action='store_true',
                           help='Interactive chat mode')
    mode_group.add_argument('--prompt', type=str,
                           help='Single prompt generation')
    mode_group.add_argument('--benchmark', action='store_true',
                           help='Run performance benchmark')
    mode_group.add_argument('--evaluate', type=str,
                           help='Evaluate on dataset')
    
    # Generation configuration
    gen_group = parser.add_argument_group('Generation Configuration')
    gen_group.add_argument('--max_length', type=int, default=200)
    gen_group.add_argument('--temperature', type=float, default=0.8)
    gen_group.add_argument('--top_k', type=int, default=50)
    gen_group.add_argument('--top_p', type=float, default=0.9)
    gen_group.add_argument('--repetition_penalty', type=float, default=1.1)
    gen_group.add_argument('--num_beams', type=int, default=1)
    gen_group.add_argument('--penalty_alpha', type=float, default=0.0,
                          help='Contrastive search penalty')
    gen_group.add_argument('--stream', action='store_true',
                          help='Stream generation token by token')
    
    # Quantization
    quant_group = parser.add_argument_group('Quantization')
    quant_group.add_argument('--load_in_8bit', action='store_true')
    quant_group.add_argument('--load_in_4bit', action='store_true')
    
    # ONNX
    parser.add_argument('--use_onnx', action='store_true',
                       help='Use ONNX runtime for inference')
    parser.add_argument('--export_onnx', type=str,
                       help='Export model to ONNX format')
    
    return parser.parse_args()


def load_model(args):
    """Load model and tokenizer"""
    print(f"Loading model from {args.model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load configuration
    config_path = Path(args.model_path) / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print("Warning: config.json not found, using defaults")
        config = {}
    
    # Load model
    from model import HybridMambaTransformer
    
    if args.load_in_8bit or args.load_in_4bit:
        # Load with quantization
        quant_config = QuantizationConfig(
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit
        )
        model = load_quantized_model(
            HybridMambaTransformer,
            args.model_path,
            quant_config,
            device=args.device,
            **config
        )
    else:
        # Load normally
        model = HybridMambaTransformer(**config)
        
        # Load weights - try safetensors first, then pytorch_model.bin
        model_path = Path(args.model_path)
        safetensors_path = model_path / "model.safetensors"
        pytorch_path = model_path / "pytorch_model.bin"
        
        try:
            if safetensors_path.exists():
                from safetensors.torch import load_file
                state_dict = load_file(str(safetensors_path))
                model.load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded weights from {safetensors_path.name}")
            elif pytorch_path.exists():
                state_dict = torch.load(str(pytorch_path), map_location=args.device)
                model.load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded weights from {pytorch_path.name}")
            else:
                print("Warning: Model weights not found, using random initialization")
        except Exception as e:
            print(f"Warning: Failed to load weights: {e}")
        
        model = model.to(args.device)
        model.eval()
    
    print("✓ Model loaded successfully")
    
    return model, tokenizer


def interactive_chat(model, tokenizer, args):
    """Interactive chat interface"""
    print("\n" + "="*60)
    print("Interactive Chat Mode")
    print("Type 'quit' to exit, 'clear' to reset context")
    print("="*60 + "\n")
    
    sampler = AdvancedSampler(model, tokenizer, device=args.device)
    conversation_history = ""
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                conversation_history = ""
                print("Context cleared!")
                continue
            
            if not user_input:
                continue
            
            # Build prompt
            if conversation_history:
                prompt = f"{conversation_history}\nHuman: {user_input}\nAssistant:"
            else:
                prompt = f"Human: {user_input}\nAssistant:"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors='pt').to(args.device)
            
            # Generate
            gen_config = GenerationConfig(
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                num_beams=args.num_beams,
                penalty_alpha=args.penalty_alpha,
                stream=args.stream
            )
            
            print("Assistant: ", end="", flush=True)
            
            if args.stream:
                # Streaming generation
                full_response = ""
                for token_text in sampler.streaming_generate(inputs['input_ids'], gen_config):
                    print(token_text, end="", flush=True)
                    full_response += token_text
                print()
            else:
                # Standard generation
                outputs = sampler.generate(inputs['input_ids'], gen_config)
                
                # Decode
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = full_text[len(prompt):].strip()
                print(response)
                full_response = response
            
            # Update history
            conversation_history = f"{conversation_history}\nHuman: {user_input}\nAssistant: {full_response}"
            
            # Truncate if too long
            if len(conversation_history) > 2000:
                lines = conversation_history.split('\n')
                conversation_history = '\n'.join(lines[-10:])
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def single_generation(model, tokenizer, args):
    """Generate from a single prompt"""
    print(f"\nPrompt: {args.prompt}\n")
    
    sampler = AdvancedSampler(model, tokenizer, device=args.device)
    
    # Tokenize
    inputs = tokenizer(args.prompt, return_tensors='pt').to(args.device)
    
    # Generate
    gen_config = GenerationConfig(
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        num_beams=args.num_beams,
        penalty_alpha=args.penalty_alpha
    )
    
    start_time = time.time()
    outputs = sampler.generate(inputs['input_ids'], gen_config)
    generation_time = time.time() - start_time
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Generated:\n{generated_text}\n")
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Tokens: {outputs.shape[1]}")
    print(f"Speed: {outputs.shape[1]/generation_time:.1f} tokens/s")


def run_benchmark(model, tokenizer, args):
    """Run performance benchmark"""
    print("\n" + "="*60)
    print("Performance Benchmark")
    print("="*60 + "\n")
    
    sampler = AdvancedSampler(model, tokenizer, device=args.device)
    
    test_prompts = [
        "The future of artificial intelligence",
        "In the field of machine learning",
        "Python programming language is",
        "Deep learning models can",
        "Natural language processing enables"
    ]
    
    gen_config = GenerationConfig(
        max_length=100,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty
    )
    
    total_time = 0
    total_tokens = 0
    
    for i, prompt in enumerate(test_prompts):
        print(f"Test {i+1}/5: {prompt}...")
        
        inputs = tokenizer(prompt, return_tensors='pt').to(args.device)
        
        start_time = time.time()
        outputs = sampler.generate(inputs['input_ids'], gen_config)
        generation_time = time.time() - start_time
        
        num_tokens = outputs.shape[1]
        speed = num_tokens / generation_time
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Time: {generation_time:.2f}s, Tokens: {num_tokens}, Speed: {speed:.1f} tok/s")
        print(f"  Output: {generated_text[:100]}...\n")
        
        total_time += generation_time
        total_tokens += num_tokens
    
    avg_speed = total_tokens / total_time
    
    print("="*60)
    print(f"Average Performance: {avg_speed:.1f} tokens/second")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total tokens: {total_tokens}")
    print("="*60)


def export_onnx_model(model, tokenizer, args):
    """Export model to ONNX"""
    print(f"\nExporting model to ONNX: {args.export_onnx}")
    
    from onnx_export import export_model_with_config
    
    # Get config
    config_path = Path(args.model_path) / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = None
    
    # Export
    export_model_with_config(model, args.export_onnx, config)
    
    print("✓ ONNX export complete")


def main():
    """Main function"""
    args = parse_args()
    
    # Load model
    model, tokenizer = load_model(args)
    
    # Export ONNX if requested
    if args.export_onnx:
        export_onnx_model(model, tokenizer, args)
        return
    
    # Run appropriate mode
    if args.interactive:
        interactive_chat(model, tokenizer, args)
    elif args.prompt:
        single_generation(model, tokenizer, args)
    elif args.benchmark:
        run_benchmark(model, tokenizer, args)
    elif args.evaluate:
        print("Evaluation mode not yet implemented")
    else:
        print("Please specify a mode: --interactive, --prompt, --benchmark, or --evaluate")
        print("Use --help for more information")


if __name__ == "__main__":
    main()
