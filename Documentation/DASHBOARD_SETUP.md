# 🚀 Dashboard Quick Setup Guide

Get the MAMBA_SLM Dashboard running in under 5 minutes!

## 📋 Prerequisites

- ✅ Python 3.8 or higher
- ✅ pip (Python package manager)
- ✅ 4GB RAM minimum (8GB recommended)
- ✅ Windows, Linux, or macOS

## ⚡ Quick Install

### Windows

**Method 1: One-Click Launch (Recommended)**
```powershell
# Simply double-click this file:
launch_dashboard.bat
```
The batch file will:
- Check Python installation
- Install PyQt6 if needed
- Launch the dashboard automatically

**Method 2: Manual Install**
```powershell
# Install dependencies
pip install PyQt6

# Launch dashboard
python launch_dashboard.py
```

### Linux/Mac

**Method 1: Shell Script**
```bash
# Make executable
chmod +x launch_dashboard.sh

# Run launcher
./launch_dashboard.sh
```

**Method 2: Manual Install**
```bash
# Install dependencies
pip3 install PyQt6

# Launch dashboard
python3 launch_dashboard.py
```

## 📦 Full Installation (All Features)

For complete functionality with training, fine-tuning, and export:

```bash
# Install all dependencies
pip install -r requirements.txt

# Optional: Flash Attention (2-3x speedup)
pip install flash-attn --no-build-isolation

# Optional: GPU ONNX support
pip install onnxruntime-gpu
```

## 🎮 First Launch

1. **Run the launcher**:
   - Windows: Double-click `launch_dashboard.bat`
   - Linux/Mac: Run `./launch_dashboard.sh`
   - Or: `python launch_dashboard.py`

2. **You should see**:
   ```
   🐍 MAMBA_SLM Unified Dashboard Launcher
   ==================================================
   Checking dependencies...
   ✓ All dependencies installed
   
   Launching dashboard...
   ```

3. **Dashboard window appears** with 6 tabs:
   - 📚 Dataset
   - 🎓 Training
   - 🎯 Fine-tuning
   - 💬 Inference
   - 📦 Export & Eval
   - 🗂️ Models

## 🏃 Quick Start Tutorial

### Example 1: Load a Dataset

1. Click **📚 Dataset** tab
2. Select "HuggingFace Dataset"
3. Enter "wikitext" in the text field
4. Click "Load Dataset"
5. View preview in the bottom panel

### Example 2: Interactive Chat

1. Click **💬 Inference** tab
2. Click "Load Model" button
3. Type a prompt: "Once upon a time"
4. Press Enter or click "Generate"
5. Watch streaming output appear!

### Example 3: Start Training

1. Load dataset first (see Example 1)
2. Click **🎓 Training** tab
3. Configure settings:
   - Hidden Size: 512
   - Layers: 8
   - Enable Flash Attention ✓
   - Enable Mixed Precision ✓
4. Click "Start Training"
5. Monitor progress in real-time

## 🔧 Troubleshooting

### Issue: "PyQt6 not found"
```bash
pip install PyQt6
```

### Issue: "Module not found" errors
```bash
# Ensure you're in the project directory
cd /path/to/MAMBA_SLM

# Install all requirements
pip install -r requirements.txt
```

### Issue: Dashboard won't launch
```bash
# Check Python version (must be 3.8+)
python --version

# Try launching directly
python mamba_dashboard.py
```

### Issue: "Import error" for project modules
```bash
# Make sure you're in the project directory
# The dashboard looks for modules in the current directory
```

## 💡 Tips for Best Experience

### Performance
- Enable "Flash Attention" for 2-3x speedup
- Use "Mixed Precision (FP16)" to save memory
- Enable "Streaming Generation" for real-time output

### Memory Optimization
- Use 4-bit quantization for large models
- Enable gradient checkpointing during training
- Use QLoRA instead of full fine-tuning

### Workflow
- Always load dataset before training
- Save checkpoints regularly
- Test with small models first
- Use inference tab to validate models

## 📝 Common Workflows

### Workflow 1: Pre-train a Model
```
1. Dataset Tab → Load WikiText
2. Training Tab → Configure model (512 hidden, 8 layers)
3. Training Tab → Enable Flash Attention + Mixed Precision
4. Training Tab → Start Training
5. Models Tab → Monitor saved checkpoints
```

### Workflow 2: Fine-tune with LoRA
```
1. Fine-tuning Tab → Load pretrained checkpoint
2. Fine-tuning Tab → Configure LoRA (rank=8, alpha=16)
3. Fine-tuning Tab → Enable QLoRA for memory efficiency
4. Fine-tuning Tab → Start Fine-tuning
5. Fine-tuning Tab → Merge weights when done
```

### Workflow 3: Chat with Model
```
1. Inference Tab → Load Model
2. Inference Tab → Adjust temperature (0.8 for balanced)
3. Inference Tab → Enable Streaming
4. Inference Tab → Enter prompt and Generate
5. Experiment with different sampling methods
```

### Workflow 4: Export for Production
```
1. Export & Eval Tab → Select trained model
2. Export & Eval Tab → Enable optimization
3. Export & Eval Tab → Export to ONNX
4. Export & Eval Tab → Run benchmark
5. Deploy .onnx file to target platform
```

## 🎯 What to Try First

**Absolute Beginner?**
1. Launch dashboard → Inference tab → Load model → Chat!

**Want to Train?**
1. Load small dataset (wikitext)
2. Use default settings
3. Start training
4. Watch the magic happen!

**Advanced User?**
1. Configure all optimizations
2. Mix multiple datasets
3. Use QLoRA fine-tuning
4. Export to ONNX

## 📚 Next Steps

- Read **DASHBOARD_README.md** for complete documentation
- Explore each tab to discover all features
- Check **README_ENHANCED.md** for technical details
- Review **QUICKSTART.md** for module usage

## ❓ Getting Help

If you encounter issues:
1. Check the status labels in the dashboard
2. Look at console output for errors
3. Verify all dependencies are installed
4. Ensure project files are in same directory

## 🎉 You're Ready!

The dashboard is designed to be intuitive. Don't hesitate to:
- Click around and explore
- Try different settings
- Experiment with models
- Have fun!

**Happy training! 🚀**

---

**Pro Tip**: The dashboard auto-saves your settings between sessions, so your preferred configurations are always ready!
