# 🎛️ MAMBA_SLM Unified Dashboard

A comprehensive PyQt6-based graphical interface for managing all aspects of the Hybrid Mamba-Transformer project.

![Dashboard](https://img.shields.io/badge/GUI-PyQt6-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

The unified dashboard provides an intuitive interface for:

### 📚 Dataset Management
- **Multiple Sources**: Load from HuggingFace Hub, local files, or directories
- **Real-time Preview**: View dataset samples before training
- **Configuration**: Set max sequence length, batch size, caching options
- **Supported Formats**: TXT, JSON, JSONL
- **Popular Datasets**: WikiText, C4, OpenWebText, Wikipedia, and more

### 🎓 Training Configuration
- **Model Architecture**: Configure hidden size, layers, attention heads
- **Training Parameters**: Epochs, learning rate, warmup steps, gradient accumulation
- **Advanced Optimizations**:
  - ✅ Flash Attention 2 (2-3x speedup)
  - ✅ Mixed Precision (FP16/BF16)
  - ✅ Gradient Checkpointing
  - ✅ Quantization (8-bit/4-bit)
- **Real-time Monitoring**: Live training metrics, loss curves, progress tracking
- **Control Panel**: Start, stop, resume training with one click

### 🎯 LoRA/QLoRA Fine-tuning
- **LoRA Configuration**: Set rank, alpha, dropout, target modules
- **QLoRA Support**: 4-bit quantization + LoRA for efficient fine-tuning
- **Model Selection**: Load pretrained checkpoints easily
- **Weight Management**: Apply, merge, and save LoRA weights
- **Fine-tuning Settings**: Custom epochs, learning rates for adaptation

### 💬 Interactive Inference
- **Chat Interface**: Real-time conversation with your model
- **Advanced Sampling**:
  - Nucleus Sampling (top-p, top-k)
  - Beam Search
  - Contrastive Search
  - Greedy Decoding
- **Generation Controls**:
  - Temperature adjustment (slider)
  - Top-p/top-k filtering
  - Repetition penalty
  - Max token length
- **Streaming Output**: Token-by-token generation display
- **Model Loading**: Quick model selection and loading

### 📦 Export & Evaluation
- **ONNX Export**:
  - Convert models to ONNX format
  - Dynamic batch/sequence support
  - Optimization for inference
  - Cross-platform deployment
- **Benchmarking**:
  - Measure latency and throughput
  - Memory usage tracking
  - Configurable test parameters
  - Detailed performance metrics

### 🗂️ Model Management
- **Checkpoint Browser**: View all saved checkpoints
- **Model Information**: Detailed stats and configuration
- **Quick Actions**: Load, delete, manage checkpoints
- **Auto-discovery**: Finds checkpoints in common directories

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
# Install all requirements including PyQt6
pip install -r requirements.txt
```

### Step 2: Install Optional Components

```bash
# For Flash Attention (recommended for training)
pip install flash-attn --no-build-isolation

# For GPU ONNX acceleration
pip install onnxruntime-gpu
```

---

## 🎮 Usage

### Quick Start

**Option 1: Using Launcher Script**
```bash
python launch_dashboard.py
```

**Option 2: Direct Launch**
```bash
python mamba_dashboard.py
```

**Option 3: Windows Double-Click**
```powershell
# Create a shortcut to launch_dashboard.py
# Double-click to launch
```

### First-Time Setup

1. **Launch Dashboard**: Run `python launch_dashboard.py`
2. **Load Dataset**: Go to "📚 Dataset" tab
   - Select source (HuggingFace or local files)
   - Configure settings
   - Click "Load Dataset"
3. **Configure Training**: Switch to "🎓 Training" tab
   - Set model architecture
   - Choose optimizations
   - Click "Start Training"

---

## 📖 Tab Guide

### 1. 📚 Dataset Tab

**Purpose**: Load and preview training data

**Workflow**:
```
1. Select Source Type
   ├─ HuggingFace Dataset → Enter dataset name (e.g., "wikitext")
   ├─ Local File(s) → Browse to .txt, .json, .jsonl
   ├─ Local Directory → Select folder with text files
   └─ Custom Mix → Combine multiple sources

2. Configure Settings
   ├─ Max Sequence Length: 128-4096 tokens
   ├─ Batch Size: 1-128 samples
   └─ Enable Caching: Speed up repeated loads

3. Load Dataset → View preview and stats
```

**Tips**:
- Start with smaller datasets for testing
- Enable caching for faster iterations
- Preview samples to verify formatting

---

### 2. 🎓 Training Tab

**Purpose**: Configure and monitor model training

**Workflow**:
```
1. Model Configuration
   ├─ Hidden Size: 128-2048 (default: 512)
   ├─ Layers: 2-32 (default: 8)
   └─ Attention Heads: 2-32 (default: 8)

2. Training Settings
   ├─ Epochs: 1-1000
   ├─ Learning Rate: 0.00001-0.1
   ├─ Warmup Steps: 0-10000
   └─ Gradient Accumulation: 1-64

3. Enable Optimizations
   ├─ ✅ Flash Attention 2 (2-3x speedup)
   ├─ ✅ Mixed Precision (FP16/BF16)
   ├─ ✅ Gradient Checkpointing (save memory)
   └─ ✅ Quantization (8-bit/4-bit)

4. Start Training → Monitor progress
```

**Real-time Monitoring**:
- Progress bar showing epoch completion
- Live loss metrics
- Training speed (tokens/sec)
- Estimated time remaining

**Tips**:
- Use Flash Attention for 2-3x speedup
- Enable gradient accumulation for larger effective batch sizes
- Start with smaller models for testing

---

### 3. 🎯 Fine-tuning Tab

**Purpose**: Efficiently fine-tune with LoRA/QLoRA

**Workflow**:
```
1. Load Pretrained Model
   └─ Browse to checkpoint directory

2. Configure LoRA
   ├─ Rank (r): 1-256 (default: 8)
   ├─ Alpha: 1-256 (default: 16)
   ├─ Dropout: 0.0-0.5 (default: 0.05)
   └─ Target Modules: "q_proj,k_proj,v_proj,o_proj"

3. Choose Method
   ├─ LoRA: Standard parameter-efficient fine-tuning
   └─ ✅ QLoRA: 4-bit quantization + LoRA (recommended)

4. Fine-tuning Settings
   ├─ Epochs: 1-100 (default: 3)
   └─ Learning Rate: 0.00001-0.01

5. Apply LoRA → Start Fine-tuning → Merge Weights
```

**Memory Savings**:
- LoRA: Train 0.1-1% of parameters
- QLoRA: 75% less memory vs full fine-tuning
- Can fine-tune 100M model on 2GB VRAM

**Tips**:
- Use QLoRA for maximum memory efficiency
- Lower rank (r) = fewer parameters but less capacity
- Start with rank 8-16 for most tasks

---

### 4. 💬 Inference Tab

**Purpose**: Interactive chat and text generation

**Workflow**:
```
1. Load Model
   └─ Click "Load Model" button

2. Configure Generation
   ├─ Max Tokens: 10-2048
   ├─ Temperature: 0.1-2.0 (use slider)
   ├─ Top-p: 0.01-1.00
   ├─ Top-k: 1-200
   ├─ Repetition Penalty: 1.0-2.0
   └─ Sampling Method: Nucleus/Beam/Contrastive/Greedy

3. Enable Streaming
   └─ ✅ Real-time token-by-token output

4. Enter Prompt → Click Generate (or press Enter)
```

**Sampling Methods**:
- **Nucleus**: Balanced quality and diversity (recommended)
- **Beam Search**: Higher quality, slower
- **Contrastive**: Reduces repetition
- **Greedy**: Fastest, deterministic

**Parameter Guide**:
- **Temperature**: Higher = more creative, Lower = more focused
- **Top-p**: Probability threshold (0.9 = 90% probability mass)
- **Top-k**: Number of top tokens to consider
- **Repetition Penalty**: Penalize repeated tokens (>1.0)

**Tips**:
- Use streaming for real-time feedback
- Lower temperature (0.7-0.8) for factual text
- Higher temperature (0.9-1.2) for creative writing

---

### 5. 📦 Export & Eval Tab

**Purpose**: Export models and benchmark performance

**ONNX Export**:
```
1. Select Model
   └─ Browse to checkpoint directory

2. Configure Export
   ├─ ONNX Opset: 11-17 (default: 14)
   ├─ ✅ Optimize for Inference
   └─ ✅ Dynamic Batch/Sequence

3. Export to ONNX
   └─ Generates .onnx file for deployment
```

**Benchmarking**:
```
1. Configure Test
   ├─ Batch Size: 1-32
   ├─ Sequence Length: 128-4096
   └─ Iterations: 10-1000

2. Run Benchmark
   └─ View latency, throughput, memory usage
```

**Use Cases**:
- **ONNX Export**: Deploy to edge devices, mobile, web
- **Benchmarking**: Compare configurations, optimize settings

---

### 6. 🗂️ Models Tab

**Purpose**: Manage checkpoints and model files

**Features**:
- **Auto-discovery**: Scans common checkpoint directories
- **Model Info**: View configuration, training stats
- **Quick Actions**: Load, delete, organize checkpoints
- **Checkpoint Browser**: List all saved models

**Workflow**:
```
1. Click "Refresh" to scan for checkpoints
2. Select checkpoint from list
3. View model information
4. Load or delete as needed
```

---

## 🎨 User Interface

### Modern Dark Theme
- Clean, professional dark interface
- Color-coded tabs for easy navigation
- Responsive layout adapts to window size
- High contrast for readability

### Keyboard Shortcuts
- **Enter** in prompt field → Generate
- **Tab** → Navigate between fields
- **Ctrl+C** → Stop generation (inference tab)

### Visual Feedback
- ✓ Success indicators
- ❌ Error messages with details
- 🔄 Progress bars and spinners
- 📊 Real-time metric updates

---

## ⚙️ Configuration

### Default Directories
The dashboard looks for files in these locations:

```
./checkpoints/      # Saved model checkpoints
./outputs/          # Training outputs
./models/           # Downloaded models
./cache/            # Dataset cache
```

### Customization
Edit `mamba_dashboard.py` to customize:
- Default parameter values
- Theme colors and styling
- Tab order and names
- Checkpoint search paths

---

## 🐛 Troubleshooting

### Common Issues

**1. "PyQt6 not found"**
```bash
pip install PyQt6
```

**2. "Dataset loader module not available"**
- Ensure `dataset_loader.py` is in the same directory
- Check Python path includes project directory

**3. "Model failed to load"**
- Verify checkpoint path is correct
- Check model files are not corrupted
- Ensure compatible PyTorch version

**4. Training doesn't start**
- Check dataset is loaded first
- Verify CUDA is available (for GPU training)
- Check sufficient disk space

**5. Inference is slow**
- Enable streaming for responsiveness
- Use quantization (4-bit/8-bit)
- Export to ONNX for faster inference

### Getting Help

1. Check error messages in status labels
2. Review console output for detailed errors
3. Verify all dependencies are installed
4. Ensure project modules are available

---

## 📊 Performance Tips

### Training
- ✅ Enable Flash Attention (2-3x speedup)
- ✅ Use mixed precision (FP16)
- ✅ Increase gradient accumulation for larger effective batches
- ✅ Enable caching for datasets

### Inference
- ✅ Use quantization (4-bit/8-bit)
- ✅ Export to ONNX for production
- ✅ Reduce max tokens for faster responses
- ✅ Lower beam size in beam search

### Memory
- ✅ Use gradient checkpointing
- ✅ Enable 4-bit quantization
- ✅ Use QLoRA instead of full fine-tuning
- ✅ Reduce batch size if OOM

---

## 🔮 Future Enhancements

Planned features:
- [ ] Distributed training support
- [ ] Advanced metrics visualization (graphs, charts)
- [ ] Dataset augmentation tools
- [ ] Model comparison dashboard
- [ ] Hyperparameter tuning automation
- [ ] Export to TensorRT, TFLite
- [ ] Custom theme builder
- [ ] Plugin system for extensions

---

## 🤝 Contributing

Contributions welcome! Areas to improve:
- UI/UX enhancements
- Additional sampling methods
- More export formats
- Performance optimizations
- Documentation improvements

---

## 📝 Technical Details

### Architecture
- **Framework**: PyQt6 (Qt 6.6+)
- **Threading**: QThread for background operations
- **Signals**: Qt signals/slots for async communication
- **Styling**: Custom QSS (Qt Style Sheets)

### Integration
The dashboard integrates with:
- `dataset_loader.py` - Dataset management
- `advanced_sampling.py` - Generation strategies
- `lora_finetuning.py` - Parameter-efficient training
- `quantization.py` - Model compression
- `onnx_export.py` - Model conversion
- `train.py` - Training loop
- `evaluate.py` - Evaluation utilities

### Thread Safety
- Training runs in `TrainingThread`
- Inference runs in `InferenceThread`
- UI updates via Qt signals
- Thread-safe model access

---

## 📄 License

MIT License - see main project README

---

## 🎯 Quick Reference

### Typical Workflow

**1. Pre-training from Scratch**
```
Dataset Tab → Load WikiText
Training Tab → Configure model → Start Training
Models Tab → Monitor checkpoints
```

**2. Fine-tuning Existing Model**
```
Fine-tuning Tab → Load checkpoint → Configure LoRA → Apply LoRA
Training Tab → Start Fine-tuning
Models Tab → Save fine-tuned model
```

**3. Inference and Testing**
```
Inference Tab → Load Model → Configure sampling
Enter prompts → Generate responses
Export & Eval Tab → Run benchmarks
```

**4. Production Deployment**
```
Export & Eval Tab → Select model → Export to ONNX
Benchmark → Optimize settings
Deploy ONNX model to target platform
```

---

## 🌟 Key Benefits

✅ **No Command Line Required**: Everything in GUI
✅ **Real-time Feedback**: See progress instantly
✅ **Beginner Friendly**: Tooltips and clear labels
✅ **Advanced Users**: Full control over all parameters
✅ **Time Saving**: Configure complex setups in seconds
✅ **Visual**: See model behavior immediately
✅ **Integrated**: All features in one place

---

**Made with ❤️ for the MAMBA_SLM project**

*Transform your LLM workflow from complex to convenient!* 🚀
