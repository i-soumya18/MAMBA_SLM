# Quick Start Guide - Hybrid Mamba-Transformer

## 🚀 Get Started in 5 Minutes

### Step 1: Setup Environment
```bash
# Clone and navigate
cd MAMBA_SLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install basic requirements
pip install torch transformers datasets accelerate
pip install bitsandbytes peft  # For quantization and LoRA

# Optional but recommended
pip install flash-attn --no-build-isolation  # 2-3x speedup
```

### Step 2: Quick Training Test (5 minutes)
```bash
python train.py \
  --dataset wikitext \
  --num_samples 1000 \
  --max_steps 100 \
  --batch_size 2 \
  --output_dir ./test_model
```

### Step 3: Interactive Chat
```bash
python evaluate.py \
  --model_path ./test_model/final_model \
  --interactive
```

---

## 💡 Common Workflows

### Train from Scratch with All Features
```bash
python train.py \
  --dataset wikitext \
  --output_dir ./my_model \
  --num_train_epochs 3 \
  --flash_attention \
  --gradient_checkpointing \
  --fp16 \
  --max_seq_length 2048
```

### Fine-tune with LoRA (Fast & Efficient)
```bash
python train.py \
  --pretrained_model ./my_model/final_model \
  --use_lora \
  --lora_r 16 \
  --dataset ./custom_data \
  --output_dir ./finetuned
```

### Inference with Quantization (Low Memory)
```bash
python evaluate.py \
  --model_path ./my_model/final_model \
  --load_in_4bit \
  --interactive
```

### Export to ONNX for Production
```bash
python evaluate.py \
  --model_path ./my_model/final_model \
  --export_onnx ./production_model
```

---

## 🎯 Training Recipes

### Fastest Training
```bash
python train.py --dataset wikitext --max_steps 500 --flash_attention --fp16
```

### Best Quality
```bash
python train.py --d_model 768 --n_layers 12 --num_train_epochs 5
```

### Lowest Memory
```bash
python train.py --use_qlora --batch_size 1 --gradient_accumulation_steps 32
```

---

## 🔍 All Features at a Glance

✅ Real dataset support (WikiText, C4, custom files)
✅ Advanced sampling (beam search, contrastive, streaming)
✅ Flash Attention (2-3x speedup)
✅ 4-bit/8-bit quantization (75% memory reduction)
✅ LoRA/QLoRA (99% fewer trainable params)
✅ Extended context (up to 4K tokens)
✅ ONNX export (cross-platform)
✅ Complete training & evaluation scripts

Happy training! 🚀
