#!/bin/bash

# SL2 Query API Dataset Creation Setup Script
# This script installs dependencies and runs the dataset creation pipeline

set -e  # Exit on any error

echo "================================================================"
echo "SL2 QUERY API DATASET CREATION SETUP"
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

# Check Python version compatibility
print_status "Checking Python version compatibility..."
PYTHON_VERSION_MAJOR=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1)
PYTHON_VERSION_MINOR=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f2)

if [ "$PYTHON_VERSION_MAJOR" -eq 3 ] && [ "$PYTHON_VERSION_MINOR" -ge 13 ]; then
    print_warning "Python 3.13+ detected. PyTorch may not have stable wheels yet."
    print_status "Will use CPU-only PyTorch or skip if not available"
    USE_PYTORCH_NIGHTLY=true
elif [ "$PYTHON_VERSION_MAJOR" -eq 3 ] && [ "$PYTHON_VERSION_MINOR" -lt 8 ]; then
    print_error "Python 3.8 or higher is required. Please upgrade Python."
    exit 1
else
    print_success "Python version is compatible with stable PyTorch"
    USE_PYTORCH_NIGHTLY=false
fi

# Create virtual environment if it doesn't exist
VENV_DIR="venv_dataset"
if [ ! -d "$VENV_DIR" ]; then
    print_status "Creating virtual environment..."
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

# Install required packages
print_status "Installing required packages..."
echo "Installing datasets..."
pip install datasets

echo "Installing transformers..."
pip install transformers

echo "Installing huggingface_hub..."
pip install huggingface_hub

echo "Installing pandas..."
pip install pandas

echo "Installing numpy..."
pip install numpy

# Install PyTorch with version compatibility handling
echo "Installing torch (for datasets compatibility)..."
if [ "$USE_PYTORCH_NIGHTLY" = true ]; then
    print_status "Attempting to install PyTorch for Python 3.13..."
    
    # Try CPU-only PyTorch first
    pip install torch --index-url https://download.pytorch.org/whl/cpu --no-deps 2>/dev/null || {
        print_warning "Stable PyTorch not available for Python 3.13"
        
        # Try nightly build
        print_status "Trying PyTorch nightly build..."
        pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu --no-deps 2>/dev/null || {
            print_warning "PyTorch nightly also failed. Continuing without PyTorch..."
            print_status "The datasets library can work without PyTorch for basic operations"
            TORCH_INSTALLED=false
        }
    }
    
    if [ "$TORCH_INSTALLED" != false ]; then
        print_success "PyTorch installed successfully"
        TORCH_INSTALLED=true
    fi
else
    # Standard PyTorch installation for compatible Python versions
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    TORCH_INSTALLED=true
    print_success "PyTorch installed successfully"
fi

# Install additional dependencies for datasets without PyTorch if needed
if [ "$TORCH_INSTALLED" = false ]; then
    print_status "Installing alternative backend for datasets..."
    pip install tensorflow-cpu 2>/dev/null || {
        print_warning "TensorFlow also failed. Using JAX as backend..."
        pip install jax jaxlib 2>/dev/null || {
            print_warning "No ML framework backend available. Some dataset features may be limited."
        }
    }
fi

print_success "All available dependencies installed successfully!"

# Check if examples file exists
EXAMPLES_FILE="colab/input/examples.md"
if [ ! -f "$EXAMPLES_FILE" ]; then
    print_warning "Examples file not found at $EXAMPLES_FILE"
    print_status "Creating input directory structure..."
    mkdir -p colab/input
    
    print_warning "Please place your examples.md file in colab/input/ directory"
    print_status "Expected file structure:"
    echo "  colab/"
    echo "  ├── input/"
    echo "  │   └── examples.md        # <-- Place your examples file here"
    echo "  └── datasets/              # <-- Output will be created here"
    
    read -p "Press Enter when you've placed the examples.md file, or Ctrl+C to exit..."
fi

# Verify examples file exists now
if [ ! -f "$EXAMPLES_FILE" ]; then
    print_error "Examples file still not found. Please ensure examples.md is in colab/input/"
    exit 1
fi

print_success "Examples file found at $EXAMPLES_FILE"

# Create output directory
print_status "Creating output directory structure..."
mkdir -p colab/datasets
mkdir -p colab/logs

# Run the dataset creation script
print_status "Starting dataset creation..."
echo "================================================================"

# Ask user for options
echo ""
read -p "Do you want to upload to Hugging Face Hub? (y/n): " UPLOAD_CHOICE
echo ""

if [[ $UPLOAD_CHOICE =~ ^[Yy]$ ]]; then
    read -p "Enter your Hugging Face repository name (e.g., username/repo-name): " HF_REPO
    echo ""
    
    print_status "Running dataset creation with Hugging Face upload..."
    python3 create_dataset.py \
        --examples "$EXAMPLES_FILE" \
        --output "colab/datasets" \
        --name "sl2-query-api-dataset" \
        --upload \
        --hf-repo "$HF_REPO"
else
    print_status "Running dataset creation without Hugging Face upload..."
    python3 create_dataset.py \
        --examples "$EXAMPLES_FILE" \
        --output "colab/datasets" \
        --name "sl2-query-api-dataset"
fi

# Check if script ran successfully
if [ $? -eq 0 ]; then
    print_success "Dataset creation completed successfully!"
    
    echo ""
    echo "================================================================"
    echo "DATASET CREATION SUMMARY"
    echo "================================================================"
    
    # Show created files
    print_status "Created files and directories:"
    if [ -d "colab/datasets" ]; then
        find colab/datasets -type f -name "*.json" -o -name "README.md" | head -10 | while read file; do
            echo "  📄 $file"
        done
        
        find colab/datasets -type d -name "*dataset*" | while read dir; do
            echo "  📁 $dir/"
        done
    fi
    
    echo ""
    print_status "Next steps:"
    echo "  1. Review the generated datasets in colab/datasets/"
    echo "  2. Check dataset_statistics.json for detailed information"
    echo "  3. Use the datasets for training your model"
    
    if [[ ! $UPLOAD_CHOICE =~ ^[Yy]$ ]]; then
        echo "  4. Optionally upload to Hugging Face Hub later using --upload flag"
    fi
    
    echo ""
    print_success "Setup and dataset creation completed! 🚀"
    
else
    print_error "Dataset creation failed. Check the error messages above."
    exit 1
fi

# Deactivate virtual environment
deactivate

echo ""
print_status "Virtual environment deactivated"
print_status "To reactivate later, run: source $VENV_DIR/bin/activate"
