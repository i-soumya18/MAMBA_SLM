"""
ONNX Export for Hybrid Mamba-Transformer
Enables cross-platform deployment and optimized inference
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from typing import Optional, Dict, List, Tuple
import numpy as np
from pathlib import Path
import json


class ONNXExportConfig:
    """Configuration for ONNX export"""
    
    def __init__(self,
                 opset_version: int = 14,
                 do_constant_folding: bool = True,
                 dynamic_axes: bool = True,
                 optimize: bool = True,
                 export_params: bool = True):
        """
        Args:
            opset_version: ONNX opset version (14+ recommended)
            do_constant_folding: Fold constant operations for optimization
            dynamic_axes: Support dynamic batch size and sequence length
            optimize: Apply ONNX optimizations
            export_params: Include model parameters in export
        """
        self.opset_version = opset_version
        self.do_constant_folding = do_constant_folding
        self.dynamic_axes = dynamic_axes
        self.optimize = optimize
        self.export_params = export_params


def export_to_onnx(model: nn.Module,
                  save_path: str,
                  config: ONNXExportConfig = None,
                  dummy_input_shape: Tuple[int, int] = (1, 128),
                  input_names: List[str] = None,
                  output_names: List[str] = None,
                  verbose: bool = True) -> str:
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model to export
        save_path: Path to save ONNX model
        config: Export configuration
        dummy_input_shape: Shape of dummy input (batch_size, seq_len)
        input_names: Names for input tensors
        output_names: Names for output tensors
        verbose: Print export information
        
    Returns:
        Path to exported ONNX model
    """
    config = config or ONNXExportConfig()
    input_names = input_names or ['input_ids']
    output_names = output_names or ['logits']
    
    # Prepare model
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    batch_size, seq_len = dummy_input_shape
    dummy_input = torch.randint(
        0, model.vocab_size if hasattr(model, 'vocab_size') else 32000,
        (batch_size, seq_len),
        dtype=torch.long,
        device=device
    )
    
    # Define dynamic axes if enabled
    dynamic_axes = None
    if config.dynamic_axes:
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    
    # Export to ONNX
    if verbose:
        print(f"Exporting model to ONNX format...")
        print(f"  Opset version: {config.opset_version}")
        print(f"  Dynamic axes: {config.dynamic_axes}")
        print(f"  Input shape: {dummy_input.shape}")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=config.export_params,
            opset_version=config.opset_version,
            do_constant_folding=config.do_constant_folding,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        if verbose:
            print(f"✓ Model exported to {save_path}")
        
        # Verify the exported model
        if config.optimize:
            save_path = optimize_onnx_model(save_path, verbose=verbose)
        
        verify_onnx_model(save_path, verbose=verbose)
        
        return save_path
        
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        raise


def optimize_onnx_model(onnx_path: str, verbose: bool = True) -> str:
    """
    Optimize ONNX model using onnxoptimizer or onnxruntime
    
    Args:
        onnx_path: Path to ONNX model
        verbose: Print optimization information
        
    Returns:
        Path to optimized model
    """
    try:
        import onnxoptimizer
        
        # Load model
        model = onnx.load(onnx_path)
        
        # Apply optimizations
        passes = onnxoptimizer.get_available_passes()
        optimized_model = onnxoptimizer.optimize(model, passes)
        
        # Save optimized model
        optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        onnx.save(optimized_model, optimized_path)
        
        if verbose:
            original_size = Path(onnx_path).stat().st_size / (1024**2)
            optimized_size = Path(optimized_path).stat().st_size / (1024**2)
            print(f"✓ Model optimized")
            print(f"  Original size: {original_size:.2f} MB")
            print(f"  Optimized size: {optimized_size:.2f} MB")
            print(f"  Reduction: {(1 - optimized_size/original_size)*100:.1f}%")
        
        return optimized_path
        
    except ImportError:
        if verbose:
            print("onnxoptimizer not available, skipping optimization")
            print("Install with: pip install onnxoptimizer")
        return onnx_path


def verify_onnx_model(onnx_path: str, verbose: bool = True) -> bool:
    """
    Verify ONNX model validity
    
    Args:
        onnx_path: Path to ONNX model
        verbose: Print verification information
        
    Returns:
        True if model is valid
    """
    try:
        # Load and check model
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        
        if verbose:
            print(f"✓ ONNX model is valid")
            
            # Print model info
            print(f"\nModel Information:")
            print(f"  Inputs: {[input.name for input in model.graph.input]}")
            print(f"  Outputs: {[output.name for output in model.graph.output]}")
            print(f"  Opset version: {model.opset_import[0].version}")
            
            # File size
            file_size = Path(onnx_path).stat().st_size / (1024**2)
            print(f"  File size: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ ONNX model validation failed: {e}")
        return False


class ONNXInferenceSession:
    """ONNX Runtime inference session wrapper"""
    
    def __init__(self, 
                 onnx_path: str,
                 providers: Optional[List[str]] = None,
                 sess_options: Optional[ort.SessionOptions] = None):
        """
        Initialize ONNX inference session
        
        Args:
            onnx_path: Path to ONNX model
            providers: Execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            sess_options: Session options for optimization
        """
        # Default providers
        if providers is None:
            providers = [
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]
        
        # Default session options
        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
        
        # Create session
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=providers
        )
        
        # Get input/output info
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"ONNX Runtime session created")
        print(f"  Providers: {self.session.get_providers()}")
        print(f"  Inputs: {self.input_names}")
        print(f"  Outputs: {self.output_names}")
    
    def run(self, input_ids: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference
        
        Args:
            input_ids: Input token IDs as numpy array
            
        Returns:
            Dictionary of output tensors
        """
        # Prepare inputs
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy()
        
        inputs = {self.input_names[0]: input_ids.astype(np.int64)}
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        
        # Return as dictionary
        return {name: output for name, output in zip(self.output_names, outputs)}
    
    def benchmark(self, 
                  batch_size: int = 1,
                  seq_len: int = 128,
                  num_iterations: int = 100,
                  warmup_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark inference performance
        
        Args:
            batch_size: Batch size for testing
            seq_len: Sequence length
            num_iterations: Number of iterations
            warmup_iterations: Warmup iterations
            
        Returns:
            Performance statistics
        """
        import time
        
        # Create dummy input
        dummy_input = np.random.randint(
            0, 32000,
            size=(batch_size, seq_len),
            dtype=np.int64
        )
        
        # Warmup
        print(f"Warming up ({warmup_iterations} iterations)...")
        for _ in range(warmup_iterations):
            self.run(dummy_input)
        
        # Benchmark
        print(f"Benchmarking ({num_iterations} iterations)...")
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.run(dummy_input)
            times.append(time.perf_counter() - start)
        
        # Calculate statistics
        times = np.array(times)
        stats = {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'throughput_samples_per_sec': batch_size / np.mean(times),
            'throughput_tokens_per_sec': (batch_size * seq_len) / np.mean(times)
        }
        
        print(f"\nBenchmark Results:")
        print(f"  Mean latency: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
        print(f"  Min/Max: {stats['min_ms']:.2f} / {stats['max_ms']:.2f} ms")
        print(f"  Throughput: {stats['throughput_samples_per_sec']:.1f} samples/s")
        print(f"  Throughput: {stats['throughput_tokens_per_sec']:.1f} tokens/s")
        
        return stats


def compare_pytorch_onnx(model: nn.Module,
                        onnx_path: str,
                        num_samples: int = 10,
                        tolerance: float = 1e-3) -> bool:
    """
    Compare PyTorch and ONNX model outputs
    
    Args:
        model: Original PyTorch model
        onnx_path: Path to ONNX model
        num_samples: Number of samples to test
        tolerance: Acceptable difference threshold
        
    Returns:
        True if outputs match within tolerance
    """
    print(f"\nComparing PyTorch and ONNX outputs...")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Create ONNX session
    onnx_session = ONNXInferenceSession(onnx_path)
    
    all_match = True
    max_diff = 0.0
    
    for i in range(num_samples):
        # Random input
        input_ids = torch.randint(
            0, model.vocab_size if hasattr(model, 'vocab_size') else 32000,
            (1, 128),
            dtype=torch.long,
            device=device
        )
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = model(input_ids)
            if isinstance(pytorch_output, dict):
                pytorch_logits = pytorch_output['logits'].cpu().numpy()
            else:
                pytorch_logits = pytorch_output.cpu().numpy()
        
        # ONNX inference
        onnx_output = onnx_session.run(input_ids.cpu().numpy())
        onnx_logits = onnx_output['logits']
        
        # Compare
        diff = np.abs(pytorch_logits - onnx_logits).max()
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            print(f"  Sample {i+1}: ✗ Difference {diff:.6f} exceeds tolerance {tolerance}")
            all_match = False
        else:
            print(f"  Sample {i+1}: ✓ Difference {diff:.6f}")
    
    print(f"\nMaximum difference: {max_diff:.6f}")
    
    if all_match:
        print("✓ All outputs match within tolerance")
    else:
        print("✗ Some outputs differ beyond tolerance")
    
    return all_match


def export_model_with_config(model: nn.Module,
                            save_dir: str,
                            config: Optional[Dict] = None):
    """
    Export model to ONNX along with configuration
    
    Args:
        model: PyTorch model
        save_dir: Directory to save model and config
        config: Model configuration dictionary
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Export ONNX model
    onnx_path = save_dir / "model.onnx"
    export_to_onnx(model, str(onnx_path))
    
    # Save configuration
    if config:
        config_path = save_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration saved to {config_path}")
    
    # Create README
    readme_path = save_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write("# ONNX Model Export\n\n")
        f.write("## Usage\n\n")
        f.write("```python\n")
        f.write("import onnxruntime as ort\n")
        f.write("import numpy as np\n\n")
        f.write(f"session = ort.InferenceSession('{onnx_path.name}')\n")
        f.write("input_ids = np.random.randint(0, 32000, (1, 128), dtype=np.int64)\n")
        f.write("outputs = session.run(['logits'], {'input_ids': input_ids})\n")
        f.write("```\n")
    
    print(f"✓ Export complete in {save_dir}")


if __name__ == "__main__":
    print("ONNX Export Module")
    print("\nBenefits:")
    print("  - Cross-platform deployment (Windows, Linux, macOS, mobile)")
    print("  - Hardware acceleration (CPU, GPU, NPU)")
    print("  - Optimized inference engines")
    print("  - Reduced dependencies")
    print("\nSupported runtimes:")
    print("  - ONNX Runtime (Python, C++, C#, Java)")
    print("  - TensorRT (NVIDIA GPUs)")
    print("  - DirectML (Windows)")
    print("  - CoreML (Apple devices)")
