"""
Model Quantization Support for Hybrid Mamba-Transformer
Implements 8-bit and 4-bit quantization using bitsandbytes
Significantly reduces memory footprint for inference
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import warnings

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("bitsandbytes not available. Install with: pip install bitsandbytes")


class QuantizationConfig:
    """Configuration for model quantization"""
    
    def __init__(self,
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = False,
                 bnb_4bit_compute_dtype: torch.dtype = torch.float16,
                 bnb_4bit_use_double_quant: bool = True,
                 bnb_4bit_quant_type: str = "nf4",
                 llm_int8_threshold: float = 6.0,
                 llm_int8_skip_modules: Optional[list] = None):
        """
        Args:
            load_in_8bit: Load model in 8-bit precision
            load_in_4bit: Load model in 4-bit precision (higher compression)
            bnb_4bit_compute_dtype: Computation dtype for 4-bit (float16 or bfloat16)
            bnb_4bit_use_double_quant: Use nested quantization for 4-bit
            bnb_4bit_quant_type: Quantization type ('fp4' or 'nf4')
            llm_int8_threshold: Threshold for outlier detection in 8-bit
            llm_int8_skip_modules: Modules to skip quantization
        """
        if load_in_8bit and load_in_4bit:
            raise ValueError("Cannot use both 8-bit and 4-bit quantization")
        
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.llm_int8_threshold = llm_int8_threshold
        self.llm_int8_skip_modules = llm_int8_skip_modules or []


class QuantizedLinear(nn.Module):
    """Wrapper for quantized linear layers"""
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 quantization_config: Optional[QuantizationConfig] = None):
        super().__init__()
        
        if not BITSANDBYTES_AVAILABLE:
            # Fallback to regular linear
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            self.quantized = False
            return
        
        self.quantized = True
        config = quantization_config or QuantizationConfig()
        
        if config.load_in_8bit:
            self.linear = bnb.nn.Linear8bitLt(
                in_features,
                out_features,
                bias=bias,
                has_fp16_weights=False,
                threshold=config.llm_int8_threshold
            )
        elif config.load_in_4bit:
            self.linear = bnb.nn.Linear4bit(
                in_features,
                out_features,
                bias=bias,
                compute_dtype=config.bnb_4bit_compute_dtype,
                compress_statistics=config.bnb_4bit_use_double_quant,
                quant_type=config.bnb_4bit_quant_type
            )
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            self.quantized = False
    
    def forward(self, x):
        return self.linear(x)


def quantize_model(model: nn.Module,
                  quantization_config: QuantizationConfig,
                  skip_modules: Optional[list] = None) -> nn.Module:
    """
    Quantize a model using bitsandbytes
    
    Args:
        model: Model to quantize
        quantization_config: Quantization configuration
        skip_modules: List of module names to skip
        
    Returns:
        Quantized model
    """
    if not BITSANDBYTES_AVAILABLE:
        warnings.warn("bitsandbytes not available, returning unquantized model")
        return model
    
    skip_modules = skip_modules or []
    skip_modules.extend(quantization_config.llm_int8_skip_modules)
    
    # Recursively replace Linear layers
    for name, module in model.named_children():
        if any(skip_name in name for skip_name in skip_modules):
            continue
        
        if isinstance(module, nn.Linear):
            # Replace with quantized version
            quantized_layer = QuantizedLinear(
                module.in_features,
                module.out_features,
                module.bias is not None,
                quantization_config
            )
            
            # Copy weights if not quantized yet
            if hasattr(quantized_layer, 'linear') and not quantized_layer.quantized:
                quantized_layer.linear.weight = module.weight
                if module.bias is not None:
                    quantized_layer.linear.bias = module.bias
            
            setattr(model, name, quantized_layer)
        else:
            # Recursively quantize child modules
            quantize_model(module, quantization_config, skip_modules)
    
    return model


def load_quantized_model(model_class,
                        model_path: str,
                        quantization_config: QuantizationConfig,
                        device: str = 'cuda',
                        **model_kwargs) -> nn.Module:
    """
    Load a pre-trained model with quantization
    
    Args:
        model_class: Model class to instantiate
        model_path: Path to model weights
        quantization_config: Quantization configuration
        device: Device to load model on
        **model_kwargs: Additional arguments for model initialization
        
    Returns:
        Loaded and quantized model
    """
    # Initialize model
    model = model_class(**model_kwargs)
    
    # Load weights
    if model_path:
        try:
            state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {model_path}")
        except FileNotFoundError:
            warnings.warn(f"No weights found at {model_path}, using random initialization")
    
    # Quantize model
    if quantization_config.load_in_8bit or quantization_config.load_in_4bit:
        print("Quantizing model...")
        model = quantize_model(model, quantization_config)
        
        # Print memory savings
        if quantization_config.load_in_8bit:
            print("Model quantized to 8-bit (expected 50% memory reduction)")
        elif quantization_config.load_in_4bit:
            print("Model quantized to 4-bit (expected 75% memory reduction)")
    
    model = model.to(device)
    model.eval()
    
    return model


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Calculate model size in memory
    
    Args:
        model: Model to measure
        
    Returns:
        Dictionary with size information
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        'param_size_mb': param_size / 1024**2,
        'buffer_size_mb': buffer_size / 1024**2,
        'total_size_mb': total_size / 1024**2,
        'total_size_gb': total_size / 1024**3
    }


def compare_quantization_methods(model: nn.Module):
    """
    Compare different quantization methods
    
    Args:
        model: Base model to compare
    """
    print("=" * 60)
    print("Quantization Comparison")
    print("=" * 60)
    
    # Original size
    original_size = get_model_size(model)
    print(f"\nOriginal Model (FP32):")
    print(f"  Size: {original_size['total_size_mb']:.2f} MB")
    
    # FP16
    model_fp16 = model.half()
    fp16_size = get_model_size(model_fp16)
    print(f"\nFP16 Model:")
    print(f"  Size: {fp16_size['total_size_mb']:.2f} MB")
    print(f"  Reduction: {(1 - fp16_size['total_size_mb']/original_size['total_size_mb'])*100:.1f}%")
    
    if BITSANDBYTES_AVAILABLE:
        # 8-bit
        config_8bit = QuantizationConfig(load_in_8bit=True)
        model_8bit = quantize_model(model.__class__(**model.config), config_8bit)
        int8_size = get_model_size(model_8bit)
        print(f"\nINT8 Model:")
        print(f"  Size: {int8_size['total_size_mb']:.2f} MB (estimated)")
        print(f"  Reduction: {(1 - int8_size['total_size_mb']/original_size['total_size_mb'])*100:.1f}%")
        
        # 4-bit
        config_4bit = QuantizationConfig(load_in_4bit=True)
        model_4bit = quantize_model(model.__class__(**model.config), config_4bit)
        int4_size = get_model_size(model_4bit)
        print(f"\nINT4 Model:")
        print(f"  Size: {int4_size['total_size_mb']:.2f} MB (estimated)")
        print(f"  Reduction: {(1 - int4_size['total_size_mb']/original_size['total_size_mb'])*100:.1f}%")
    else:
        print("\nINT8/INT4 quantization not available (install bitsandbytes)")
    
    print("=" * 60)


class DynamicQuantization:
    """Dynamic quantization for inference"""
    
    @staticmethod
    def quantize_dynamic(model: nn.Module,
                        dtype: torch.dtype = torch.qint8,
                        modules_to_quantize: Optional[set] = None) -> nn.Module:
        """
        Apply PyTorch dynamic quantization
        
        Args:
            model: Model to quantize
            dtype: Quantization dtype
            modules_to_quantize: Set of module types to quantize
            
        Returns:
            Dynamically quantized model
        """
        if modules_to_quantize is None:
            modules_to_quantize = {nn.Linear, nn.LSTM, nn.GRU}
        
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            modules_to_quantize,
            dtype=dtype
        )
        
        print(f"Applied dynamic quantization with {dtype}")
        return quantized_model


# Utility functions for quantization-aware training
class QuantizationAwareTraining:
    """Helper for quantization-aware training (QAT)"""
    
    @staticmethod
    def prepare_qat(model: nn.Module,
                   backend: str = 'fbgemm') -> nn.Module:
        """
        Prepare model for quantization-aware training
        
        Args:
            model: Model to prepare
            backend: Quantization backend ('fbgemm' or 'qnnpack')
            
        Returns:
            Prepared model with fake quantization
        """
        model.train()
        
        # Set quantization config
        model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        
        # Prepare for QAT
        torch.quantization.prepare_qat(model, inplace=True)
        
        print(f"Model prepared for quantization-aware training (backend: {backend})")
        return model
    
    @staticmethod
    def convert_qat(model: nn.Module) -> nn.Module:
        """
        Convert QAT model to quantized model
        
        Args:
            model: Trained QAT model
            
        Returns:
            Fully quantized model
        """
        model.eval()
        torch.quantization.convert(model, inplace=True)
        
        print("Converted QAT model to quantized model")
        return model


if __name__ == "__main__":
    print(f"bitsandbytes available: {BITSANDBYTES_AVAILABLE}")
    
    if BITSANDBYTES_AVAILABLE:
        print("\nQuantization options available:")
        print("  - 8-bit quantization (INT8): ~50% memory reduction")
        print("  - 4-bit quantization (NF4/FP4): ~75% memory reduction")
        print("  - Double quantization: Additional compression for 4-bit")
        print("\nUsage:")
        print("  config = QuantizationConfig(load_in_4bit=True)")
        print("  model = quantize_model(model, config)")
    else:
        print("\nInstall bitsandbytes for advanced quantization:")
        print("  pip install bitsandbytes")
        print("\nAlternatively, use PyTorch dynamic quantization:")
        print("  model = DynamicQuantization.quantize_dynamic(model)")
