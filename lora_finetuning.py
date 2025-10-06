"""
LoRA (Low-Rank Adaptation) and QLoRA for Hybrid Mamba-Transformer
Enables parameter-efficient fine-tuning with minimal memory overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import math


class LoRAConfig:
    """Configuration for LoRA"""
    
    def __init__(self,
                 r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 target_modules: Optional[List[str]] = None,
                 bias: str = "none",
                 task_type: str = "CAUSAL_LM",
                 inference_mode: bool = False):
        """
        Args:
            r: LoRA rank (lower = more compression, 4-64 typical)
            lora_alpha: LoRA scaling factor (typically 2*r or r)
            lora_dropout: Dropout probability for LoRA layers
            target_modules: Names of modules to apply LoRA to
            bias: Bias training ("none", "all", or "lora_only")
            task_type: Type of task (for configuration)
            inference_mode: Whether in inference mode (merges weights)
        """
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["qkv", "o_proj", "mlp"]
        self.bias = bias
        self.task_type = task_type
        self.inference_mode = inference_mode
        self.scaling = lora_alpha / r


class LoRALayer(nn.Module):
    """Base LoRA layer"""
    
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else lambda x: x


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation
    
    Original: y = Wx + b
    LoRA: y = Wx + b + (BA)x * scaling
    where A is [r, in_features], B is [out_features, r]
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 bias: bool = True,
                 merge_weights: bool = False):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merge_weights = merge_weights
        self.merged = False
        
        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False
        
        # LoRA low-rank matrices
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else lambda x: x
            
            # Initialize
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 0 and not self.merged:
            # Original output
            result = self.linear(x)
            
            # LoRA adaptation
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            result = result + lora_out * self.scaling
            
            return result
        else:
            return self.linear(x)
    
    def merge_lora_weights(self):
        """Merge LoRA weights into base weights for inference"""
        if self.r > 0 and not self.merged:
            # W' = W + BA * scaling
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True
    
    def unmerge_lora_weights(self):
        """Unmerge LoRA weights from base weights"""
        if self.r > 0 and self.merged:
            self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False


class LoRAAttention(nn.Module):
    """Attention layer with LoRA on Q, K, V projections"""
    
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 lora_config: LoRAConfig):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Use LoRA for QKV projection
        self.qkv = LoRALinear(
            d_model,
            3 * d_model,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            bias=False
        )
        
        # Use LoRA for output projection
        self.o_proj = LoRALinear(
            d_model,
            d_model,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            bias=False
        )
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # QKV projection with LoRA
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project with LoRA
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.o_proj(attn_output)
        
        return output


def add_lora_to_model(model: nn.Module,
                     lora_config: LoRAConfig,
                     verbose: bool = True) -> nn.Module:
    """
    Add LoRA adapters to a model
    
    Args:
        model: Base model
        lora_config: LoRA configuration
        verbose: Print information about applied LoRA
        
    Returns:
        Model with LoRA adapters
    """
    trainable_params = 0
    all_params = 0
    
    # Freeze all base parameters
    for param in model.parameters():
        param.requires_grad = False
        all_params += param.numel()
    
    # Replace target modules with LoRA versions
    for name, module in model.named_modules():
        # Check if this module should have LoRA
        should_apply_lora = any(target in name for target in lora_config.target_modules)
        
        if should_apply_lora and isinstance(module, nn.Linear):
            # Get parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            
            # Create LoRA layer
            lora_layer = LoRALinear(
                module.in_features,
                module.out_features,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                bias=module.bias is not None
            )
            
            # Copy original weights
            lora_layer.linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_layer.linear.bias.data = module.bias.data.clone()
            
            # Replace module
            setattr(parent, attr_name, lora_layer)
            
            # Count trainable parameters
            trainable_params += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
    
    if verbose:
        print(f"LoRA Configuration:")
        print(f"  Rank (r): {lora_config.r}")
        print(f"  Alpha: {lora_config.lora_alpha}")
        print(f"  Scaling: {lora_config.scaling:.2f}")
        print(f"  Target modules: {lora_config.target_modules}")
        print(f"\nTrainable parameters: {trainable_params:,} ({trainable_params/all_params*100:.2f}%)")
        print(f"Total parameters: {all_params:,}")
        print(f"Memory reduction: ~{(1 - trainable_params/all_params)*100:.1f}%")
    
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights into base weights for inference
    
    Args:
        model: Model with LoRA adapters
        
    Returns:
        Model with merged weights
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_lora_weights()
    
    print("LoRA weights merged into base model")
    return model


def save_lora_weights(model: nn.Module, save_path: str):
    """
    Save only LoRA adapter weights
    
    Args:
        model: Model with LoRA adapters
        save_path: Path to save weights
    """
    lora_state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data
    
    torch.save(lora_state_dict, save_path)
    print(f"Saved LoRA weights to {save_path}")
    print(f"LoRA parameters: {sum(p.numel() for p in lora_state_dict.values()):,}")


def load_lora_weights(model: nn.Module, load_path: str) -> nn.Module:
    """
    Load LoRA adapter weights
    
    Args:
        model: Model with LoRA adapters
        load_path: Path to load weights from
        
    Returns:
        Model with loaded LoRA weights
    """
    lora_state_dict = torch.load(load_path)
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if f"{name}.lora_A" in lora_state_dict:
                module.lora_A.data = lora_state_dict[f"{name}.lora_A"]
                module.lora_B.data = lora_state_dict[f"{name}.lora_B"]
    
    print(f"Loaded LoRA weights from {load_path}")
    return model


class QLoRAConfig(LoRAConfig):
    """
    Configuration for QLoRA (Quantized LoRA)
    Combines 4-bit quantization with LoRA for maximum efficiency
    """
    
    def __init__(self,
                 r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 target_modules: Optional[List[str]] = None,
                 load_in_4bit: bool = True,
                 bnb_4bit_compute_dtype: torch.dtype = torch.float16,
                 bnb_4bit_use_double_quant: bool = True,
                 bnb_4bit_quant_type: str = "nf4"):
        super().__init__(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.bnb_4bit_quant_type = bnb_4bit_quant_type


def prepare_model_for_qlora(model: nn.Module,
                            qlora_config: QLoRAConfig) -> nn.Module:
    """
    Prepare model for QLoRA fine-tuning
    
    Args:
        model: Base model
        qlora_config: QLoRA configuration
        
    Returns:
        Model ready for QLoRA training
    """
    # First quantize the model
    from quantization import QuantizationConfig, quantize_model
    
    quant_config = QuantizationConfig(
        load_in_4bit=qlora_config.load_in_4bit,
        bnb_4bit_compute_dtype=qlora_config.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=qlora_config.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=qlora_config.bnb_4bit_quant_type
    )
    
    model = quantize_model(model, quant_config)
    
    # Then add LoRA adapters
    model = add_lora_to_model(model, qlora_config)
    
    print("Model prepared for QLoRA training")
    print("Expected memory usage: ~25% of original model")
    
    return model


# Utility functions
def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get only LoRA parameters for optimization"""
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_params.append(module.lora_A)
            lora_params.append(module.lora_B)
    return lora_params


def print_lora_stats(model: nn.Module):
    """Print statistics about LoRA layers"""
    lora_layers = 0
    total_lora_params = 0
    total_params = sum(p.numel() for p in model.parameters())
    
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_layers += 1
            total_lora_params += module.lora_A.numel() + module.lora_B.numel()
    
    print(f"LoRA Statistics:")
    print(f"  Number of LoRA layers: {lora_layers}")
    print(f"  LoRA parameters: {total_lora_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable ratio: {total_lora_params/total_params*100:.2f}%")
    print(f"  Memory savings: ~{(1-total_lora_params/total_params)*100:.1f}%")


if __name__ == "__main__":
    print("LoRA/QLoRA module for parameter-efficient fine-tuning")
    print("\nBenefits:")
    print("  - Train only 0.1-1% of parameters")
    print("  - 10-100x less memory for gradients")
    print("  - Faster training and iteration")
    print("  - Multiple task-specific adapters from one base model")
    print("\nQLoRA combines:")
    print("  - 4-bit quantization (75% memory reduction)")
    print("  - LoRA adapters (99% fewer trainable params)")
    print("  - Result: Fine-tune large models on consumer GPUs!")
