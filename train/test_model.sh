#!/bin/bash

# SL2 Query API Model Testing Script
# Easy interface for testing trained models

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

echo "================================================================"
echo "SL2 QUERY API MODEL TESTING"
echo "================================================================"

# Check if virtual environment exists
VENV_DIR="venv_training"
if [ -d "$VENV_DIR" ]; then
    print_status "Activating virtual environment..."
    source $VENV_DIR/bin/activate
else
    print_warning "Training virtual environment not found. Using system Python."
fi

# Check if test script exists
if [ ! -f "test_model.py" ]; then
    print_error "test_model.py not found in current directory"
    exit 1
fi

# Function to show help
show_help() {
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -l, --list          List available trained models"
    echo "  -m, --model PATH    Specify model path"
    echo "  -t, --latest        Use latest trained model"
    echo "  -i, --interactive   Interactive testing (default)"
    echo "  -p, --predefined    Run predefined tests"
    echo "  -b, --batch FILE    Run batch testing with file"
    echo ""
    echo "Examples:"
    echo "  $0                          # Interactive testing with latest model"
    echo "  $0 --list                   # List available models"
    echo "  $0 --latest --predefined    # Run predefined tests with latest model"
    echo "  $0 -m /path/to/model -i     # Interactive testing with specific model"
    echo ""
}

# Parse command line arguments
MODE="interactive"
MODEL_PATH=""
USE_LATEST=false
BATCH_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -l|--list)
            print_status "Listing available models..."
            python3 test_model.py --list
            exit 0
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -t|--latest)
            USE_LATEST=true
            shift
            ;;
        -i|--interactive)
            MODE="interactive"
            shift
            ;;
        -p|--predefined)
            MODE="predefined"
            shift
            ;;
        -b|--batch)
            MODE="batch"
            BATCH_FILE="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build command
CMD="python3 test_model.py"

if [ "$USE_LATEST" = true ]; then
    CMD="$CMD --latest"
elif [ -n "$MODEL_PATH" ]; then
    CMD="$CMD --model '$MODEL_PATH'"
fi

CMD="$CMD --mode $MODE"

if [ "$MODE" = "batch" ] && [ -n "$BATCH_FILE" ]; then
    if [ ! -f "$BATCH_FILE" ]; then
        print_error "Batch file not found: $BATCH_FILE"
        exit 1
    fi
    CMD="$CMD --batch-file '$BATCH_FILE'"
fi

# Check GPU availability
print_status "Checking GPU availability..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('No GPU detected. Model will load on CPU (slower).')
" 2>/dev/null || print_warning "PyTorch not available - please install dependencies first"

# Start testing
print_status "Starting model testing..."
print_status "Mode: $MODE"

if [ "$USE_LATEST" = true ]; then
    print_status "Using latest trained model"
elif [ -n "$MODEL_PATH" ]; then
    print_status "Using model: $MODEL_PATH"
fi

echo ""
echo "================================================================"

# Run the command
eval $CMD

# Check if testing was successful
if [ $? -eq 0 ]; then
    echo ""
    print_success "Testing completed successfully!"
else
    print_error "Testing failed. Check the error messages above."
    
    echo ""
    print_status "Troubleshooting tips:"
    echo "  1. Ensure you have trained a model first"
    echo "  2. Check if the model path is correct"
    echo "  3. Verify you have enough GPU/CPU memory"
    echo "  4. Make sure all dependencies are installed"
    
    exit 1
fi

# Deactivate virtual environment if we activated it
if [ -d "$VENV_DIR" ]; then
    deactivate
fi

echo ""
print_status "Model testing session ended"
