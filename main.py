import os
import sys
import json

try:
    import numpy as np
    import torch
    from js_code_generator import JSCodeGenerator
except ImportError as e:
    missing_package = str(e).split("'")[-2]
    print(f"Error: Required package '{missing_package}' is not available.")
    print("Please install the required packages using: pip install -r requirements.txt")
    sys.exit(1)

from config import METADATA_PATH, MODEL_PATH, print_config

def main():
    print_config()
    
    print("PyTorch version:", torch.__version__)
    print("NumPy version:", np.__version__)
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("Number of GPUs:", torch.cuda.device_count())
        device = "cuda"
    else:
        print("No GPU found, using CPU")
        device = "cpu"
    
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        sys.exit(1)
    
    print(f"Initializing JS Code Generator with model: {MODEL_PATH}")
    js_generator = JSCodeGenerator(metadata_path=METADATA_PATH, model_path=MODEL_PATH)
    
    try:
        js_generator.initialize()
        print("Initialization complete!")
        
        js_generator.interactive_mode()
    except Exception as e:
        print(f"Error during initialization or execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
