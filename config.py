import os
import torch

# Default model paths
MODEL_OPTIONS = {
    "codellama": "codellama/CodeLlama-13b-Instruct-hf",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "qwen": "Qwen/Qwen-14B-Chat"
}

# Default configuration
DEFAULT_MODEL = "codellama"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENABLE_4BIT = True if DEVICE == "cuda" else False
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Allow environment overrides
MODEL_KEY = os.environ.get('MODEL_KEY', DEFAULT_MODEL)
MODEL_PATH = os.environ.get('MODEL_PATH', MODEL_OPTIONS.get(MODEL_KEY, MODEL_OPTIONS[DEFAULT_MODEL]))
METADATA_PATH = os.environ.get('METADATA_PATH', './meta_with_field(with option)_50.json')

# Print configuration for debugging
def print_config():
    print(f"Using model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"4-bit quantization: {'Enabled' if ENABLE_4BIT else 'Disabled'}")
    print(f"Metadata path: {METADATA_PATH}")

if __name__ == "__main__":
    print_config()
