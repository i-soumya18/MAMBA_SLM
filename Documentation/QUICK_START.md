# Quick Start Guide: MAMBA_SLM Complete Workflow

## 🚀 One-Command Training Pipeline

### Full Workflow (Copy & Paste Ready):

```powershell
# Step 1: Load Dataset
python dataset_loader.py --dataset_name wikitext --dataset_config wikitext-2-raw-v1 --split train --output_dir ./data --max_samples 5000

# Step 2: Train Base Model
$env:WANDB_DISABLED="true"
python train.py --d_model 512 --n_layers 8 --num_train_epochs 3 --batch_size 2 --learning_rate 5e-4 --fp16 --gradient_accumulation_steps 4 --output_dir ./checkpoints/base_model --logging_steps 10 --save_steps 100 --dataset wikitext --num_samples 1000 --max_steps 500

# Step 3: Fine-Tune with LoRA
python train.py --pretrained_model ./checkpoints/base_model/final_model --use_lora --lora_r 8 --lora_alpha 16 --d_model 512 --n_layers 8 --num_train_epochs 2 --batch_size 2 --learning_rate 1e-4 --fp16 --output_dir ./checkpoints/lora_finetuned --max_steps 200

# Step 4: Evaluate Performance
python evaluate.py --model_path ./checkpoints/lora_finetuned/final_model --benchmark

# Step 5: Test Generation
python evaluate.py --model_path ./checkpoints/lora_finetuned/final_model --prompt "Artificial intelligence is" --max_length 100 --temperature 0.8

# Step 6 (Optional): Launch Dashboard
python launch_dashboard.py
```

---

## 📋 Quick Commands Reference

### Dataset Loading:
```powershell
# WikiText-2 (Small, Fast)
python dataset_loader.py --dataset_name wikitext --dataset_config wikitext-2-raw-v1 --max_samples 1000

# OpenWebText (Larger, Better Quality)
python dataset_loader.py --dataset_name openwebtext --max_samples 10000

# Custom Text Files
python dataset_loader.py --files "path/to/data/*.txt" --output_dir ./data
```

### Training Configurations:

#### Fast Demo (5-10 minutes):
```powershell
python train.py --d_model 256 --n_layers 4 --max_steps 50 --batch_size 2 --fp16
```

#### Balanced (30-60 minutes):
```powershell
python train.py --d_model 512 --n_layers 8 --max_steps 500 --batch_size 4 --fp16 --gradient_accumulation_steps 2
```

#### High Quality (2-4 hours):
```powershell
python train.py --d_model 768 --n_layers 12 --num_train_epochs 3 --batch_size 8 --fp16 --gradient_accumulation_steps 4
```

### LoRA Fine-Tuning:

#### Minimal (Best for 4GB VRAM):
```powershell
python train.py --use_lora --lora_r 4 --lora_alpha 8 --batch_size 1 --max_steps 100
```

#### Recommended (8GB+ VRAM):
```powershell
python train.py --use_lora --lora_r 8 --lora_alpha 16 --batch_size 4 --max_steps 300
```

#### High-Rank (16GB+ VRAM):
```powershell
python train.py --use_lora --lora_r 16 --lora_alpha 32 --batch_size 8 --max_steps 500
```

### Evaluation & Inference:

#### Benchmark Speed:
```powershell
python evaluate.py --model_path ./checkpoints/lora_finetuned/final_model --benchmark
```

#### Single Prompt:
```powershell
python evaluate.py --model_path <model_path> --prompt "Your prompt here" --max_length 200
```

#### Interactive Chat:
```powershell
python evaluate.py --model_path <model_path> --interactive
```

---

## 🎛️ Parameter Tuning Guide

### Model Size (d_model):
- **256**: Ultra-fast, minimal quality (~30M params)
- **384**: Fast, decent quality (~45M params)
- **512**: Balanced speed/quality (~68M params) ✅ Recommended
- **768**: High quality, slower (~150M params)

### Layers (n_layers):
- **4**: Very fast training, basic capability
- **6**: Good for prototyping
- **8**: Balanced ✅ Recommended
- **12**: Better quality, longer training

### Batch Size:
- **1**: For 4GB VRAM (slow)
- **2**: For 4-6GB VRAM ✅ Recommended for GTX 1650
- **4**: For 8GB VRAM
- **8+**: For 12GB+ VRAM

### Learning Rate:
- **5e-4**: Base training ✅
- **1e-4**: Fine-tuning ✅
- **5e-5**: Stable convergence
- **1e-5**: Very conservative

### LoRA Rank:
- **r=4**: Minimal adaptation (fastest, least flexible)
- **r=8**: Good balance ✅ Recommended
- **r=16**: Better adaptation (slower, more flexible)
- **r=32**: Maximum adaptation (slowest, most powerful)

---

## 🐛 Troubleshooting

### Out of Memory (CUDA OOM):
```powershell
# Solution 1: Reduce batch size
--batch_size 1 --gradient_accumulation_steps 8

# Solution 2: Smaller model
--d_model 256 --n_layers 4

# Solution 3: Use LoRA
--use_lora --lora_r 8

# Solution 4: Enable quantization
--load_in_8bit  # or --load_in_4bit
```

### Slow Training:
```powershell
# Solution 1: Enable FP16
--fp16

# Solution 2: Increase batch size
--batch_size 4 --gradient_accumulation_steps 2

# Solution 3: Use Flash Attention (if available)
--flash_attention
```

### Poor Generation Quality:
```powershell
# Solution 1: Train longer
--num_train_epochs 5 --max_steps 2000

# Solution 2: More training data
--num_samples 10000  # or more

# Solution 3: Larger model
--d_model 512 --n_layers 8

# Solution 4: Adjust generation params
--temperature 0.7 --top_p 0.9 --repetition_penalty 1.2
```

### WandB Errors:
```powershell
# Disable WandB logging
$env:WANDB_DISABLED="true"
```

---

## 📊 Expected Performance

### GTX 1650 (4GB VRAM):
| Configuration | Speed | Quality | Memory |
|---------------|-------|---------|--------|
| 256d, 4 layers | ~100 tok/s | Basic | 2.5GB |
| 384d, 6 layers | ~85 tok/s | Good | 3.2GB |
| 512d, 8 layers | ~70 tok/s | Better | 3.8GB ✅ |

### RTX 3060 (12GB VRAM):
| Configuration | Speed | Quality | Memory |
|---------------|-------|---------|--------|
| 512d, 8 layers | ~120 tok/s | Good | 4.2GB |
| 768d, 12 layers | ~80 tok/s | High | 7.5GB |
| 1024d, 16 layers | ~50 tok/s | Very High | 11GB |

---

## 🎯 Workflow Cheat Sheet

```
1. LOAD DATASET
   ↓ (WikiText, OpenWebText, or custom files)
   
2. TRAIN BASE MODEL
   ↓ (FP16, gradient accumulation, checkpointing)
   
3. FINE-TUNE WITH LORA (Optional)
   ↓ (99.9% param reduction, 4-bit quantization available)
   
4. EVALUATE PERFORMANCE
   ↓ (Benchmark speed, test prompts)
   
5. DEPLOY / USE
   ↓ (Interactive chat, API serving, or dashboard)
```

---

## 🚀 Advanced Features

### Enable Flash Attention 2:
```powershell
pip install flash-attn --no-build-isolation
python train.py --flash_attention ...
```

### Extended Context (4K tokens):
```python
from extended_context import upgrade_model_context_length
model = upgrade_model_context_length(model, new_max_length=4096)
```

### 4-bit Quantized Fine-Tuning (QLoRA):
```powershell
python train.py --use_qlora --load_in_4bit --lora_r 8 ...
```

### Export to ONNX:
```powershell
python onnx_export.py --model_path <path> --output_path ./model.onnx --optimize
```

---

## 📱 Dashboard Usage

```powershell
# Launch GUI (easiest method)
python launch_dashboard.py

# Or on Windows (double-click)
launch_dashboard.bat

# Or on Linux/Mac
chmod +x launch_dashboard.sh
./launch_dashboard.sh
```

**Dashboard Features:**
- 📂 Dataset Tab: Load WikiText, C4, or custom files
- 🏋️ Training Tab: Configure and monitor training
- 🔧 Fine-Tuning Tab: LoRA/QLoRA with visual controls
- 💬 Inference Tab: Interactive chat with streaming
- 📊 Export & Eval Tab: ONNX export, benchmarking
- 📁 Model Manager Tab: Browse and load checkpoints

---

## 🔗 File References

- **WORKFLOW_SUMMARY.md** - Complete detailed workflow documentation
- **DASHBOARD_README.md** - Full dashboard documentation
- **README.md** - Main project README
- **requirements.txt** - All dependencies

---

*Quick Start Guide - MAMBA_SLM Project*
*For detailed information, see WORKFLOW_SUMMARY.md*
