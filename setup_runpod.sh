#!/bin/bash

# Display current Python version
echo "Python version:"
python --version

pip uninstall numpy -y

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify numpy is installed
echo "Verifying numpy installation:"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Verify pytorch is installed
echo "Verifying PyTorch installation:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Setup complete. You can now run the main script with: python main.py"
