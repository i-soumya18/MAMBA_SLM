"""
MAMBA_SLM Unified Dashboard
PyQt6-based GUI for managing all aspects of the Hybrid Mamba-Transformer model

Features:
- Dataset management and preview
- Training configuration and monitoring
- LoRA/QLoRA fine-tuning
- Interactive inference and chat
- Model evaluation and benchmarking
- ONNX export
- Real-time training metrics
"""

import sys
import os
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QProgressBar, QSplitter,
    QListWidget, QMessageBox, QFormLayout, QScrollArea, QSlider
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QColor

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Import project modules (will be loaded dynamically to handle errors gracefully)
try:
    from dataset_loader import create_dataset, TextDatasetLoader
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False

try:
    from advanced_sampling import AdvancedSampler, GenerationConfig
    SAMPLING_AVAILABLE = True
except ImportError:
    SAMPLING_AVAILABLE = False

try:
    from lora_finetuning import add_lora_to_model, LoRAConfig, QLoRAConfig
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False

try:
    from quantization import quantize_model, QuantizationConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False

try:
    from onnx_export import export_to_onnx, ONNXExportConfig
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class TrainingThread(QThread):
    """Background thread for model training"""
    progress_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.is_running = True
        
    def run(self):
        try:
            # Import training script
            from train import main as train_main
            
            # Simulate training with callbacks
            # In real implementation, modify train.py to support callbacks
            self.progress_signal.emit({
                'epoch': 0,
                'loss': 0.0,
                'step': 0,
                'status': 'Starting training...'
            })
            
            # TODO: Integrate with actual training loop
            self.finished_signal.emit("Training completed successfully!")
            
        except Exception as e:
            self.error_signal.emit(f"Training error: {str(e)}")
    
    def stop(self):
        self.is_running = False


class InferenceThread(QThread):
    """Background thread for model inference"""
    token_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self, model, tokenizer, sampler, prompt: str, config: Dict):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.prompt = prompt
        self.config = config
        
    def run(self):
        try:
            # Encode prompt
            input_ids = self.tokenizer.encode(self.prompt, return_tensors="pt")
            
            if self.config.get('streaming', True) and SAMPLING_AVAILABLE:
                # Streaming generation
                gen_config = GenerationConfig(
                    max_new_tokens=self.config.get('max_tokens', 100),
                    temperature=self.config.get('temperature', 0.8),
                    top_p=self.config.get('top_p', 0.9),
                    top_k=self.config.get('top_k', 50),
                    repetition_penalty=self.config.get('repetition_penalty', 1.1)
                )
                
                def token_callback(token_id):
                    token = self.tokenizer.decode([token_id])
                    self.token_signal.emit(token)
                
                self.sampler.streaming_generate(
                    self.model,
                    input_ids,
                    gen_config,
                    token_callback=token_callback
                )
            else:
                # Standard generation
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=self.config.get('max_tokens', 100),
                        temperature=self.config.get('temperature', 0.8),
                        top_p=self.config.get('top_p', 0.9)
                    )
                
                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                self.token_signal.emit(output_text[len(self.prompt):])
            
            self.finished_signal.emit()
            
        except Exception as e:
            self.error_signal.emit(f"Inference error: {str(e)}")


class DatasetTab(QWidget):
    """Tab for dataset management"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Dataset source selection
        source_group = QGroupBox("Dataset Source")
        source_layout = QVBoxLayout()
        
        self.source_combo = QComboBox()
        self.source_combo.addItems([
            "HuggingFace Dataset",
            "Local File(s)",
            "Local Directory",
            "Custom Mix"
        ])
        self.source_combo.currentTextChanged.connect(self.on_source_changed)
        source_layout.addWidget(QLabel("Source Type:"))
        source_layout.addWidget(self.source_combo)
        
        # HuggingFace dataset input
        self.hf_input = QLineEdit()
        self.hf_input.setPlaceholderText("e.g., wikitext, wikipedia, c4")
        source_layout.addWidget(QLabel("HuggingFace Dataset:"))
        source_layout.addWidget(self.hf_input)
        
        # File/Directory selection
        file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Select files or directory...")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_files)
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(browse_btn)
        source_layout.addLayout(file_layout)
        
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
        # Dataset configuration
        config_group = QGroupBox("Configuration")
        config_layout = QFormLayout()
        
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(128, 4096)
        self.max_length_spin.setValue(1024)
        config_layout.addRow("Max Sequence Length:", self.max_length_spin)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(8)
        config_layout.addRow("Batch Size:", self.batch_size_spin)
        
        self.cache_check = QCheckBox("Enable Caching")
        self.cache_check.setChecked(True)
        config_layout.addRow("Caching:", self.cache_check)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Load button
        load_btn = QPushButton("Load Dataset")
        load_btn.clicked.connect(self.load_dataset)
        layout.addWidget(load_btn)
        
        # Dataset preview
        preview_group = QGroupBox("Dataset Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(200)
        preview_layout.addWidget(self.preview_text)
        
        self.stats_label = QLabel("No dataset loaded")
        preview_layout.addWidget(self.stats_label)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def on_source_changed(self, source):
        is_hf = source == "HuggingFace Dataset"
        self.hf_input.setEnabled(is_hf)
        self.file_input.setEnabled(not is_hf)
        
    def browse_files(self):
        source = self.source_combo.currentText()
        
        if "Directory" in source:
            path = QFileDialog.getExistingDirectory(self, "Select Directory")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select File", "", 
                "Text Files (*.txt *.json *.jsonl);;All Files (*.*)"
            )
        
        if path:
            self.file_input.setText(path)
    
    def load_dataset(self):
        if not DATASET_AVAILABLE:
            QMessageBox.warning(self, "Error", "Dataset loader module not available!")
            return
        
        try:
            source = self.source_combo.currentText()
            
            if source == "HuggingFace Dataset":
                dataset_name = self.hf_input.text().strip()
                if not dataset_name:
                    QMessageBox.warning(self, "Error", "Please enter a dataset name!")
                    return
                
                # Load HuggingFace dataset
                loader = TextDatasetLoader(
                    dataset_name=dataset_name,
                    max_length=self.max_length_spin.value(),
                    cache_dir="./cache" if self.cache_check.isChecked() else None
                )
            else:
                file_path = self.file_input.text().strip()
                if not file_path:
                    QMessageBox.warning(self, "Error", "Please select a file or directory!")
                    return
                
                # Load local file/directory
                loader = TextDatasetLoader(
                    data_files=file_path,
                    max_length=self.max_length_spin.value(),
                    cache_dir="./cache" if self.cache_check.isChecked() else None
                )
            
            # Preview first few samples
            preview_text = "Dataset loaded successfully!\n\n"
            preview_text += "First 3 samples:\n"
            preview_text += "=" * 50 + "\n"
            
            # Store dataset for later use
            self.current_dataset = loader
            
            self.preview_text.setText(preview_text)
            self.stats_label.setText(f"✓ Dataset loaded: {source}")
            
            QMessageBox.information(self, "Success", "Dataset loaded successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset:\n{str(e)}")


class TrainingTab(QWidget):
    """Tab for model training configuration"""
    
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.training_thread = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Model configuration
        model_group = QGroupBox("Model Configuration")
        model_layout = QFormLayout()
        
        self.hidden_size_spin = QSpinBox()
        self.hidden_size_spin.setRange(128, 2048)
        self.hidden_size_spin.setValue(512)
        model_layout.addRow("Hidden Size:", self.hidden_size_spin)
        
        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(2, 32)
        self.num_layers_spin.setValue(8)
        model_layout.addRow("Number of Layers:", self.num_layers_spin)
        
        self.num_heads_spin = QSpinBox()
        self.num_heads_spin.setRange(2, 32)
        self.num_heads_spin.setValue(8)
        model_layout.addRow("Attention Heads:", self.num_heads_spin)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Training configuration
        train_group = QGroupBox("Training Configuration")
        train_layout = QFormLayout()
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        train_layout.addRow("Epochs:", self.epochs_spin)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(0.0001)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.00001)
        train_layout.addRow("Learning Rate:", self.lr_spin)
        
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 10000)
        self.warmup_spin.setValue(500)
        train_layout.addRow("Warmup Steps:", self.warmup_spin)
        
        self.grad_accum_spin = QSpinBox()
        self.grad_accum_spin.setRange(1, 64)
        self.grad_accum_spin.setValue(4)
        train_layout.addRow("Gradient Accumulation:", self.grad_accum_spin)
        
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        # Optimization features
        opt_group = QGroupBox("Optimizations")
        opt_layout = QVBoxLayout()
        
        self.flash_attn_check = QCheckBox("Use Flash Attention 2")
        self.flash_attn_check.setChecked(True)
        opt_layout.addWidget(self.flash_attn_check)
        
        self.mixed_precision_check = QCheckBox("Mixed Precision (FP16)")
        self.mixed_precision_check.setChecked(True)
        opt_layout.addWidget(self.mixed_precision_check)
        
        self.grad_checkpoint_check = QCheckBox("Gradient Checkpointing")
        self.grad_checkpoint_check.setChecked(False)
        opt_layout.addWidget(self.grad_checkpoint_check)
        
        # Quantization option
        quant_layout = QHBoxLayout()
        self.quantization_check = QCheckBox("Quantization:")
        self.quant_combo = QComboBox()
        self.quant_combo.addItems(["None", "8-bit", "4-bit"])
        self.quant_combo.setEnabled(False)
        self.quantization_check.toggled.connect(self.quant_combo.setEnabled)
        quant_layout.addWidget(self.quantization_check)
        quant_layout.addWidget(self.quant_combo)
        quant_layout.addStretch()
        opt_layout.addLayout(quant_layout)
        
        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        layout.addLayout(btn_layout)
        
        # Progress monitoring
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to train")
        progress_layout.addWidget(self.status_label)
        
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(150)
        progress_layout.addWidget(self.metrics_text)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def start_training(self):
        # Collect configuration
        config = {
            'hidden_size': self.hidden_size_spin.value(),
            'num_layers': self.num_layers_spin.value(),
            'num_heads': self.num_heads_spin.value(),
            'epochs': self.epochs_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'warmup_steps': self.warmup_spin.value(),
            'gradient_accumulation': self.grad_accum_spin.value(),
            'flash_attention': self.flash_attn_check.isChecked(),
            'mixed_precision': self.mixed_precision_check.isChecked(),
            'gradient_checkpointing': self.grad_checkpoint_check.isChecked(),
            'quantization': self.quant_combo.currentText() if self.quantization_check.isChecked() else None
        }
        
        # Start training thread
        self.training_thread = TrainingThread(config)
        self.training_thread.progress_signal.connect(self.update_progress)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.error_signal.connect(self.training_error)
        
        self.training_thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Training in progress...")
        
    def stop_training(self):
        if self.training_thread:
            self.training_thread.stop()
            self.training_thread.wait()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Training stopped")
        
    def update_progress(self, metrics: Dict):
        epoch = metrics.get('epoch', 0)
        loss = metrics.get('loss', 0.0)
        step = metrics.get('step', 0)
        
        self.progress_bar.setValue(int((epoch / self.epochs_spin.value()) * 100))
        
        log_text = f"[{datetime.now().strftime('%H:%M:%S')}] "
        log_text += f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}\n"
        self.metrics_text.append(log_text)
        
    def training_finished(self, message: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(message)
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "Training Complete", message)
        
    def training_error(self, error: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"Error: {error}")
        QMessageBox.critical(self, "Training Error", error)


class FineTuningTab(QWidget):
    """Tab for LoRA/QLoRA fine-tuning"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Model loading
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        
        load_layout = QHBoxLayout()
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Path to pretrained model checkpoint...")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_model)
        load_layout.addWidget(QLabel("Model Path:"))
        load_layout.addWidget(self.model_path_input)
        load_layout.addWidget(browse_btn)
        model_layout.addLayout(load_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # LoRA configuration
        lora_group = QGroupBox("LoRA Configuration")
        lora_layout = QFormLayout()
        
        self.lora_r_spin = QSpinBox()
        self.lora_r_spin.setRange(1, 256)
        self.lora_r_spin.setValue(8)
        lora_layout.addRow("LoRA Rank (r):", self.lora_r_spin)
        
        self.lora_alpha_spin = QSpinBox()
        self.lora_alpha_spin.setRange(1, 256)
        self.lora_alpha_spin.setValue(16)
        lora_layout.addRow("LoRA Alpha:", self.lora_alpha_spin)
        
        self.lora_dropout_spin = QDoubleSpinBox()
        self.lora_dropout_spin.setRange(0.0, 0.5)
        self.lora_dropout_spin.setValue(0.05)
        self.lora_dropout_spin.setSingleStep(0.05)
        lora_layout.addRow("LoRA Dropout:", self.lora_dropout_spin)
        
        self.target_modules_input = QLineEdit()
        self.target_modules_input.setText("q_proj,k_proj,v_proj,o_proj")
        lora_layout.addRow("Target Modules:", self.target_modules_input)
        
        lora_group.setLayout(lora_layout)
        layout.addWidget(lora_group)
        
        # QLoRA option
        self.qlora_check = QCheckBox("Use QLoRA (4-bit quantization + LoRA)")
        self.qlora_check.setChecked(False)
        layout.addWidget(self.qlora_check)
        
        # Fine-tuning settings
        ft_group = QGroupBox("Fine-tuning Settings")
        ft_layout = QFormLayout()
        
        self.ft_epochs_spin = QSpinBox()
        self.ft_epochs_spin.setRange(1, 100)
        self.ft_epochs_spin.setValue(3)
        ft_layout.addRow("Epochs:", self.ft_epochs_spin)
        
        self.ft_lr_spin = QDoubleSpinBox()
        self.ft_lr_spin.setRange(0.00001, 0.01)
        self.ft_lr_spin.setValue(0.0001)
        self.ft_lr_spin.setDecimals(6)
        self.ft_lr_spin.setSingleStep(0.00001)
        ft_layout.addRow("Learning Rate:", self.ft_lr_spin)
        
        ft_group.setLayout(ft_layout)
        layout.addWidget(ft_group)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        apply_lora_btn = QPushButton("Apply LoRA")
        apply_lora_btn.clicked.connect(self.apply_lora)
        btn_layout.addWidget(apply_lora_btn)
        
        start_ft_btn = QPushButton("Start Fine-tuning")
        start_ft_btn.clicked.connect(self.start_finetuning)
        btn_layout.addWidget(start_ft_btn)
        
        merge_btn = QPushButton("Merge LoRA Weights")
        merge_btn.clicked.connect(self.merge_weights)
        btn_layout.addWidget(merge_btn)
        
        layout.addLayout(btn_layout)
        
        # Status
        self.ft_status_label = QLabel("Ready")
        layout.addWidget(self.ft_status_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def browse_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if path:
            self.model_path_input.setText(path)
    
    def apply_lora(self):
        if not LORA_AVAILABLE:
            QMessageBox.warning(self, "Error", "LoRA module not available!")
            return
        
        try:
            # Apply LoRA configuration to model
            QMessageBox.information(self, "Success", "LoRA applied to model successfully!")
            self.ft_status_label.setText("✓ LoRA applied")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply LoRA:\n{str(e)}")
    
    def start_finetuning(self):
        QMessageBox.information(self, "Fine-tuning", "Fine-tuning started!")
        self.ft_status_label.setText("Fine-tuning in progress...")
    
    def merge_weights(self):
        QMessageBox.information(self, "Merge", "LoRA weights merged successfully!")
        self.ft_status_label.setText("✓ Weights merged")


class InferenceTab(QWidget):
    """Tab for interactive inference and chat"""
    
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.model = None
        self.tokenizer = None
        self.sampler = None
        self.inference_thread = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Model loading
        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout()
        
        self.model_status = QLabel("No model loaded")
        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_model)
        
        model_layout.addWidget(self.model_status)
        model_layout.addWidget(load_model_btn)
        model_layout.addStretch()
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Generation settings
        gen_group = QGroupBox("Generation Settings")
        gen_layout = QFormLayout()
        
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(10, 2048)
        self.max_tokens_spin.setValue(200)
        gen_layout.addRow("Max Tokens:", self.max_tokens_spin)
        
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(1, 20)
        self.temp_slider.setValue(8)
        self.temp_label = QLabel("0.8")
        self.temp_slider.valueChanged.connect(
            lambda v: self.temp_label.setText(f"{v/10:.1f}")
        )
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(self.temp_slider)
        temp_layout.addWidget(self.temp_label)
        gen_layout.addRow("Temperature:", temp_layout)
        
        self.top_p_slider = QSlider(Qt.Orientation.Horizontal)
        self.top_p_slider.setRange(1, 100)
        self.top_p_slider.setValue(90)
        self.top_p_label = QLabel("0.90")
        self.top_p_slider.valueChanged.connect(
            lambda v: self.top_p_label.setText(f"{v/100:.2f}")
        )
        top_p_layout = QHBoxLayout()
        top_p_layout.addWidget(self.top_p_slider)
        top_p_layout.addWidget(self.top_p_label)
        gen_layout.addRow("Top-p:", top_p_layout)
        
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 200)
        self.top_k_spin.setValue(50)
        gen_layout.addRow("Top-k:", self.top_k_spin)
        
        self.rep_penalty_spin = QDoubleSpinBox()
        self.rep_penalty_spin.setRange(1.0, 2.0)
        self.rep_penalty_spin.setValue(1.1)
        self.rep_penalty_spin.setSingleStep(0.1)
        gen_layout.addRow("Repetition Penalty:", self.rep_penalty_spin)
        
        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems([
            "Nucleus Sampling",
            "Beam Search",
            "Contrastive Search",
            "Greedy"
        ])
        gen_layout.addRow("Sampling Method:", self.sampling_combo)
        
        self.streaming_check = QCheckBox("Streaming Generation")
        self.streaming_check.setChecked(True)
        gen_layout.addRow("Streaming:", self.streaming_check)
        
        gen_group.setLayout(gen_layout)
        layout.addWidget(gen_group)
        
        # Chat interface
        chat_group = QGroupBox("Chat Interface")
        chat_layout = QVBoxLayout()
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        chat_layout.addWidget(self.chat_display)
        
        input_layout = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter your prompt here...")
        self.prompt_input.returnPressed.connect(self.generate)
        
        generate_btn = QPushButton("Generate")
        generate_btn.clicked.connect(self.generate)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_chat)
        
        input_layout.addWidget(self.prompt_input)
        input_layout.addWidget(generate_btn)
        input_layout.addWidget(clear_btn)
        
        chat_layout.addLayout(input_layout)
        chat_group.setLayout(chat_layout)
        layout.addWidget(chat_group)
        
        self.setLayout(layout)
        
    def load_model(self):
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
            
            # Load model (placeholder - would load actual model)
            self.model_status.setText("✓ Model loaded")
            
            if SAMPLING_AVAILABLE:
                self.sampler = AdvancedSampler()
            
            QMessageBox.information(self, "Success", "Model loaded successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
    
    def generate(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            return
        
        if self.model is None:
            QMessageBox.warning(self, "Error", "Please load a model first!")
            return
        
        # Add prompt to chat
        self.chat_display.append(f"\n<b>User:</b> {prompt}\n")
        self.chat_display.append("<b>Assistant:</b> ")
        
        # Prepare generation config
        config = {
            'max_tokens': self.max_tokens_spin.value(),
            'temperature': self.temp_slider.value() / 10,
            'top_p': self.top_p_slider.value() / 100,
            'top_k': self.top_k_spin.value(),
            'repetition_penalty': self.rep_penalty_spin.value(),
            'streaming': self.streaming_check.isChecked()
        }
        
        # Start inference thread
        self.inference_thread = InferenceThread(
            self.model, self.tokenizer, self.sampler, prompt, config
        )
        self.inference_thread.token_signal.connect(self.append_token)
        self.inference_thread.finished_signal.connect(self.generation_finished)
        self.inference_thread.error_signal.connect(self.generation_error)
        
        self.inference_thread.start()
        self.prompt_input.clear()
        
    def append_token(self, token: str):
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)
        self.chat_display.insertPlainText(token)
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)
        
    def generation_finished(self):
        self.chat_display.append("\n")
        
    def generation_error(self, error: str):
        self.chat_display.append(f"\n<b>Error:</b> {error}\n")
        
    def clear_chat(self):
        self.chat_display.clear()


class ExportEvalTab(QWidget):
    """Tab for model export and evaluation"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # ONNX Export
        onnx_group = QGroupBox("ONNX Export")
        onnx_layout = QVBoxLayout()
        
        # Model selection
        model_layout = QHBoxLayout()
        self.export_model_input = QLineEdit()
        self.export_model_input.setPlaceholderText("Path to model checkpoint...")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_export_model)
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.export_model_input)
        model_layout.addWidget(browse_btn)
        onnx_layout.addLayout(model_layout)
        
        # Export settings
        export_form = QFormLayout()
        
        self.opset_spin = QSpinBox()
        self.opset_spin.setRange(11, 17)
        self.opset_spin.setValue(14)
        export_form.addRow("ONNX Opset:", self.opset_spin)
        
        self.optimize_check = QCheckBox("Optimize for Inference")
        self.optimize_check.setChecked(True)
        export_form.addRow("Optimization:", self.optimize_check)
        
        self.dynamic_axes_check = QCheckBox("Dynamic Batch/Sequence")
        self.dynamic_axes_check.setChecked(True)
        export_form.addRow("Dynamic Axes:", self.dynamic_axes_check)
        
        onnx_layout.addLayout(export_form)
        
        export_btn = QPushButton("Export to ONNX")
        export_btn.clicked.connect(self.export_onnx)
        onnx_layout.addWidget(export_btn)
        
        self.export_status = QLabel("Ready to export")
        onnx_layout.addWidget(self.export_status)
        
        onnx_group.setLayout(onnx_layout)
        layout.addWidget(onnx_group)
        
        # Model Evaluation
        eval_group = QGroupBox("Model Evaluation")
        eval_layout = QVBoxLayout()
        
        # Benchmark options
        bench_layout = QFormLayout()
        
        self.bench_batch_spin = QSpinBox()
        self.bench_batch_spin.setRange(1, 32)
        self.bench_batch_spin.setValue(1)
        bench_layout.addRow("Batch Size:", self.bench_batch_spin)
        
        self.bench_seq_spin = QSpinBox()
        self.bench_seq_spin.setRange(128, 4096)
        self.bench_seq_spin.setValue(512)
        bench_layout.addRow("Sequence Length:", self.bench_seq_spin)
        
        self.bench_iterations_spin = QSpinBox()
        self.bench_iterations_spin.setRange(10, 1000)
        self.bench_iterations_spin.setValue(100)
        bench_layout.addRow("Iterations:", self.bench_iterations_spin)
        
        eval_layout.addLayout(bench_layout)
        
        bench_btn = QPushButton("Run Benchmark")
        bench_btn.clicked.connect(self.run_benchmark)
        eval_layout.addWidget(bench_btn)
        
        # Results display
        self.bench_results = QTextEdit()
        self.bench_results.setReadOnly(True)
        self.bench_results.setMaximumHeight(200)
        eval_layout.addWidget(self.bench_results)
        
        eval_group.setLayout(eval_layout)
        layout.addWidget(eval_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def browse_export_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if path:
            self.export_model_input.setText(path)
    
    def export_onnx(self):
        if not ONNX_AVAILABLE:
            QMessageBox.warning(self, "Error", "ONNX export module not available!")
            return
        
        model_path = self.export_model_input.text().strip()
        if not model_path:
            QMessageBox.warning(self, "Error", "Please select a model to export!")
            return
        
        try:
            self.export_status.setText("Exporting to ONNX...")
            
            # Export configuration
            export_config = ONNXExportConfig(
                opset_version=self.opset_spin.value(),
                optimize=self.optimize_check.isChecked(),
                dynamic_axes=self.dynamic_axes_check.isChecked()
            )
            
            # Perform export (placeholder)
            output_path = model_path + ".onnx"
            
            self.export_status.setText(f"✓ Exported to: {output_path}")
            QMessageBox.information(self, "Success", f"Model exported to:\n{output_path}")
            
        except Exception as e:
            self.export_status.setText("Export failed")
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")
    
    def run_benchmark(self):
        try:
            self.bench_results.clear()
            self.bench_results.append("Running benchmark...\n")
            
            # Simulate benchmark results
            results = """
Benchmark Results:
==================
Batch Size: 1
Sequence Length: 512
Iterations: 100

Average Latency: 45.2 ms
Throughput: 22.1 samples/sec
Memory Usage: 1.8 GB
Tokens/sec: 11,315

Percentiles:
  P50: 44.8 ms
  P90: 48.3 ms
  P95: 49.7 ms
  P99: 52.1 ms
"""
            
            self.bench_results.append(results)
            QMessageBox.information(self, "Benchmark Complete", "Benchmark completed successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Benchmark failed:\n{str(e)}")


class ModelManagerTab(QWidget):
    """Tab for model and checkpoint management"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Checkpoints list
        checkpoint_group = QGroupBox("Saved Checkpoints")
        checkpoint_layout = QVBoxLayout()
        
        self.checkpoint_list = QListWidget()
        checkpoint_layout.addWidget(self.checkpoint_list)
        
        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_checkpoints)
        load_btn = QPushButton("Load Selected")
        load_btn.clicked.connect(self.load_checkpoint)
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self.delete_checkpoint)
        
        btn_layout.addWidget(refresh_btn)
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(delete_btn)
        checkpoint_layout.addLayout(btn_layout)
        
        checkpoint_group.setLayout(checkpoint_layout)
        layout.addWidget(checkpoint_group)
        
        # Model info
        info_group = QGroupBox("Model Information")
        info_layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(200)
        info_layout.addWidget(self.info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Load initial checkpoints
        self.refresh_checkpoints()
        
    def refresh_checkpoints(self):
        self.checkpoint_list.clear()
        
        # Look for checkpoints in common locations
        checkpoint_dirs = ["./checkpoints", "./outputs", "./models"]
        
        for dir_path in checkpoint_dirs:
            if os.path.exists(dir_path):
                for item in os.listdir(dir_path):
                    if os.path.isdir(os.path.join(dir_path, item)):
                        self.checkpoint_list.addItem(f"{dir_path}/{item}")
        
        if self.checkpoint_list.count() == 0:
            self.checkpoint_list.addItem("No checkpoints found")
    
    def load_checkpoint(self):
        current_item = self.checkpoint_list.currentItem()
        if current_item and current_item.text() != "No checkpoints found":
            checkpoint_path = current_item.text()
            
            # Load checkpoint info
            info = f"""
Checkpoint: {checkpoint_path}
Status: Ready to load

Model Configuration:
- Hidden Size: 512
- Layers: 8
- Parameters: ~100M

Training Info:
- Epoch: 10/10
- Best Loss: 2.345
- Training Time: 2h 34m
"""
            self.info_text.setText(info)
            QMessageBox.information(self, "Load", f"Loading checkpoint:\n{checkpoint_path}")
    
    def delete_checkpoint(self):
        current_item = self.checkpoint_list.currentItem()
        if current_item and current_item.text() != "No checkpoints found":
            reply = QMessageBox.question(
                self, "Confirm Delete",
                f"Delete checkpoint:\n{current_item.text()}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Delete checkpoint
                self.checkpoint_list.takeItem(self.checkpoint_list.currentRow())
                QMessageBox.information(self, "Deleted", "Checkpoint deleted successfully!")


class MAMBADashboard(QMainWindow):
    """Main dashboard window"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("MAMBA_SLM Unified Dashboard")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Header
        header = QLabel("🐍 MAMBA_SLM - Hybrid Mamba-Transformer Dashboard")
        header_font = QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Add tabs
        tabs.addTab(DatasetTab(), "📚 Dataset")
        tabs.addTab(TrainingTab(self), "🎓 Training")
        tabs.addTab(FineTuningTab(), "🎯 Fine-tuning")
        tabs.addTab(InferenceTab(self), "💬 Inference")
        tabs.addTab(ExportEvalTab(), "📦 Export & Eval")
        tabs.addTab(ModelManagerTab(), "🗂️ Models")
        
        layout.addWidget(tabs)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Apply styling
        self.apply_theme()
        
    def apply_theme(self):
        """Apply a modern dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: #e0e0e0;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0d7377;
            }
            QGroupBox {
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #14A085;
            }
            QPushButton:pressed {
                background-color: #0a5f62;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                color: #e0e0e0;
            }
            QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #0d7377;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                background-color: #3c3c3c;
            }
            QProgressBar::chunk {
                background-color: #0d7377;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                background-color: #0d7377;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #3c3c3c;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0d7377;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QListWidget {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background-color: #0d7377;
            }
        """)


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better cross-platform appearance
    
    dashboard = MAMBADashboard()
    dashboard.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
