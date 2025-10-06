#!/bin/bash
# Setup script for Hybrid Mamba-Transformer Training

echo "Setting up Hybrid Mamba-Transformer Training Environment..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv hybrid_mamba_env
source hybrid_mamba_env/bin/activate  # On Windows: hybrid_mamba_env\Scripts\activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Download base tokenizer
echo "Downloading base tokenizer..."
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')"

echo "Setup complete!"
echo ""
echo "To activate the environment: source hybrid_mamba_env/bin/activate"
echo "To start training: python hybrid_mamba_training.py"
echo "To run inference: python inference.py --chat"
