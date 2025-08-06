#!/bin/bash

# SL2 Query API Model Training Setup Script
# This script installs dependencies and runs the training pipeline

set -e  # Exit on any error

echo "================================================================"
echo "SL2 QUERY API MODEL TRAINING SETUP"
echo "================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
print_status "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Python $PYTHON_VERSION found"

# Check if pip is installed
print_status "Checking pip installation..."
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip."
    exit 1
fi

print_success "pip3 found"

# Check CUDA/GPU availability
print_status "Checking CUDA/GPU availability..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('No GPU detected. Training will be slow on CPU.')
" 2>/dev/null || print_warning "PyTorch not installed yet - will check GPU after installation"

# Create virtual environment if it doesn't exist
VENV_DIR="venv_training"
if [ ! -d "$VENV_DIR" ]; then
    print_status "Creating virtual environment for training..."
    python3 -m venv $VENV_DIR
    print_success "Virtual environment created at $VENV_DIR"
else
    print_status "Using existing virtual environment at $VENV_DIR"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (if available)
print_status "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    print_status "No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install training dependencies
print_status "Installing training dependencies..."

echo "Installing transformers..."
pip install "transformers>=4.36.0"

echo "Installing datasets..."
pip install "datasets>=2.15.0"

echo "Installing PEFT (Parameter Efficient Fine-Tuning)..."
pip install "peft>=0.7.0"

echo "Installing TRL (Transformer Reinforcement Learning)..."
pip install "trl>=0.7.0"

echo "Installing bitsandbytes for quantization..."
pip install "bitsandbytes>=0.41.0"

echo "Installing accelerate..."
pip install "accelerate>=0.24.0"

echo "Installing huggingface_hub..."
pip install huggingface_hub

echo "Installing additional dependencies..."
pip install pandas numpy

# Optional: Install monitoring tools
echo "Installing optional monitoring tools..."
pip install tensorboard  # For training monitoring
pip install wandb --quiet || print_warning "Weights & Biases installation failed (optional)"

print_success "All training dependencies installed successfully!"

# Check if dataset exists
DATASET_PATH="colab/datasets/sl2-query-api-dataset-alpaca"
if [ ! -d "$DATASET_PATH" ]; then
    print_warning "Training dataset not found at $DATASET_PATH"
    print_status "Checking for dataset creation script..."
    
    if [ ! -f "create_dataset.py" ]; then
        print_error "Dataset creation script not found!"
        print_status "Please ensure you have:"
        echo "  1. create_dataset.py script"
        echo "  2. examples.md file in colab/input/"
        echo "  3. Run dataset creation first"
        
        read -p "Do you want to run dataset creation now? (y/n): " CREATE_DATASET
        if [[ $CREATE_DATASET =~ ^[Yy]$ ]]; then
            if [ -f "create_dataset.py" ]; then
                print_status "Running dataset creation..."
                python3 create_dataset.py
            else
                print_error "create_dataset.py not found in current directory"
                exit 1
            fi
        else
            print_error "Cannot proceed without dataset. Please create dataset first."
            exit 1
        fi
    fi
else
    print_success "Training dataset found at $DATASET_PATH"
fi

# Verify dataset after creation/check
if [ ! -d "$DATASET_PATH" ]; then
    print_error "Training dataset still not available. Please create dataset first."
    exit 1
fi

# Check available disk space
print_status "Checking available disk space..."
AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))

if [ $AVAILABLE_GB -lt 10 ]; then
    print_warning "Low disk space: ${AVAILABLE_GB}GB available"
    print_warning "Training may require 5-20GB of space for model checkpoints"
    read -p "Continue anyway? (y/n): " CONTINUE_TRAINING
    if [[ ! $CONTINUE_TRAINING =~ ^[Yy]$ ]]; then
        print_error "Training cancelled due to insufficient disk space"
        exit 1
    fi
else
    print_success "Sufficient disk space available: ${AVAILABLE_GB}GB"
fi

# GPU check after PyTorch installation
print_status "Verifying GPU setup after PyTorch installation..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    if torch.cuda.get_device_properties(0).total_memory < 8e9:
        print('WARNING: GPU has less than 8GB memory. Consider using smaller batch size.')
else:
    print('No GPU detected. Training will be slow on CPU.')
"

# Create output directories
print_status "Creating output directory structure..."
mkdir -p colab/models
mkdir -p colab/logs
mkdir -p colab/models/backups

# Ask user for training options
echo ""
echo "================================================================"
echo "TRAINING CONFIGURATION"
echo "================================================================"

# Ask for training parameters
read -p "Number of training epochs (default: 3): " EPOCHS
EPOCHS=${EPOCHS:-3}

read -p "Batch size (default: 4, reduce if OOM): " BATCH_SIZE
BATCH_SIZE=${BATCH_SIZE:-4}

read -p "Learning rate (default: 2e-4): " LEARNING_RATE
LEARNING_RATE=${LEARNING_RATE:-2e-4}

echo ""
read -p "Do you want to upload to Hugging Face Hub after training? (y/n): " UPLOAD_CHOICE
echo ""

if [[ $UPLOAD_CHOICE =~ ^[Yy]$ ]]; then
    read -p "Enter your Hugging Face repository name (e.g., username/model-name): " HF_REPO
    echo ""
fi

# Start training
print_status "Starting SL2 Query API model training..."
echo "================================================================"

TRAINING_CMD="python3 train.py --epochs $EPOCHS --batch-size $BATCH_SIZE --learning-rate $LEARNING_RATE"

if [[ $UPLOAD_CHOICE =~ ^[Yy]$ ]] && [[ -n "$HF_REPO" ]]; then
    TRAINING_CMD="$TRAINING_CMD --upload"
fi

print_status "Running training command: $TRAINING_CMD"
echo ""

# Execute training
eval $TRAINING_CMD

# Check if training was successful
if [ $? -eq 0 ]; then
    print_success "Training completed successfully!"
    
    echo ""
    echo "================================================================"
    echo "TRAINING COMPLETED"
    echo "================================================================"
    
    # Show created files
    print_status "Created files and directories:"
    if [ -d "colab/models" ]; then
        find colab/models -name "*.json" -o -name "*.bin" -o -name "*.safetensors" | head -10 | while read file; do
            echo "  📄 $file"
        done
        
        find colab/models -type d -name "*sl2*" | while read dir; do
            echo "  📁 $dir/"
        done
    fi
    
    echo ""
    print_status "Next steps:"
    echo "  1. Review the trained model in colab/models/"
    echo "  2. Test the model with your own prompts"
    echo "  3. Deploy the model for code generation"
    
    if [[ ! $UPLOAD_CHOICE =~ ^[Yy]$ ]]; then
        echo "  4. Optionally upload to Hugging Face Hub later using --upload flag"
    fi
    
    # Show model loading example
    echo ""
    print_status "To load your trained model:"
    echo "```python"
    echo "from transformers import AutoTokenizer, AutoModelForCausalLM"
    echo "from peft import PeftModel"
    echo ""
    echo "# Load base model"
    echo "base_model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-6.7b-instruct')"
    echo ""
    echo "# Load your trained LoRA model"
    echo "model = PeftModel.from_pretrained(base_model, 'colab/models/[your-model-directory]/final_model')"
    echo ""
    echo "# Load tokenizer"
    echo "tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-6.7b-instruct')"
    echo "```"
    
    echo ""
    print_success "Setup and training completed! 🚀"
    
else
    print_error "Training failed. Check the error messages above."
    
    echo ""
    print_status "Troubleshooting tips:"
    echo "  1. Check if you have enough GPU memory (reduce batch size if needed)"
    echo "  2. Ensure dataset is properly formatted"
    echo "  3. Check CUDA installation if using GPU"
    echo "  4. Try running with --no-test flag to skip model testing"
    
    exit 1
fi

# Deactivate virtual environment
deactivate

echo ""
print_status "Virtual environment deactivated"
print_status "To reactivate later, run: source $VENV_DIR/bin/activate"
