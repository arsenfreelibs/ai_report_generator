import os
import torch

# Default model paths
MODEL_OPTIONS = {
    "codellama": "codellama/CodeLlama-13b-Instruct-hf",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "qwen": "Qwen/Qwen-14B-Chat",
    "codellama_small": "codellama/CodeLlama-7b-Instruct-hf",
    "deepseek": "deepseek-ai/deepseek-coder-33b-instruct"
}

# OpenAI model options
OPENAI_MODEL_OPTIONS = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4": "gpt-4",
    "gpt-4-turbo": "gpt-4-turbo-preview",
    "gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k",
    "gpt-4.1": "gpt-4.1",
    "gpt-4o": "gpt-4o",
}

# Claude model options
CLAUDE_MODEL_OPTIONS = {
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-4": "claude-sonnet-4-20250514",
    "claude-opus-4": "claude-opus-4-20250514"
}

# LLM Processor Configuration
LLM_PROCESSOR_TYPE = os.environ.get('LLM_PROCESSOR_TYPE', 'openai')  # 'local', 'openai', or 'claude'

# Default configuration
DEFAULT_MODEL = "codellama_small"
DEFAULT_OPENAI_MODEL = "gpt-4.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENABLE_4BIT = True if DEVICE == "cuda" else False
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Use data from config
MODEL_KEY = os.environ.get('MODEL_KEY', DEFAULT_MODEL)
MODEL_PATH = os.environ.get('MODEL_PATH', MODEL_OPTIONS.get(MODEL_KEY, MODEL_OPTIONS[DEFAULT_MODEL]))
METADATA_PATH = os.environ.get('METADATA_PATH', './meta_co2sandbox.json')

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_MODEL_KEY = os.environ.get('OPENAI_MODEL_KEY', DEFAULT_OPENAI_MODEL)
OPENAI_MODEL_NAME = OPENAI_MODEL_OPTIONS.get(OPENAI_MODEL_KEY, DEFAULT_OPENAI_MODEL)

# Claude Configuration
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY', '')
CLAUDE_MODEL_KEY = os.environ.get('CLAUDE_MODEL_KEY', 'claude-3.5-sonnet')
CLAUDE_MODEL_NAME = CLAUDE_MODEL_OPTIONS.get(CLAUDE_MODEL_KEY, 'claude-3-5-sonnet-20241022')

# Google Drive Configuration
GDRIVE_FILE_ID = None  # Set your Google Drive file ID here
GDRIVE_CREDENTIALS_PATH = 'credentials.json'

JS_API_FALLBACK_FILE = 'co2_js_api.json'
REPORT_EXAMPLES_KNOWLEDGE_PATH = 'knowledge/report_examples'

# Print configuration for debugging
def print_config():
    print(f"LLM Processor Type: {LLM_PROCESSOR_TYPE}")
    if LLM_PROCESSOR_TYPE == 'local':
        print(f"Using local model: {MODEL_PATH}")
        print(f"Device: {DEVICE}")
        print(f"4-bit quantization: {'Enabled' if ENABLE_4BIT else 'Disabled'}")
    elif LLM_PROCESSOR_TYPE == 'openai':
        print(f"Using OpenAI model: {OPENAI_MODEL_NAME}")
        print(f"API Key configured: {'Yes' if OPENAI_API_KEY else 'No'}")
    elif LLM_PROCESSOR_TYPE == 'claude':
        print(f"Using Claude model: {CLAUDE_MODEL_NAME}")
        print(f"API Key configured: {'Yes' if CLAUDE_API_KEY else 'No'}")
    print(f"Metadata path: {METADATA_PATH}")

def get_llm_processor_config():
    """Get the appropriate LLM processor configuration"""
    if LLM_PROCESSOR_TYPE == 'openai':
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI processor")
        return {
            'type': 'openai',
            'api_key': OPENAI_API_KEY,
            'model_name': OPENAI_MODEL_NAME
        }
    elif LLM_PROCESSOR_TYPE == 'claude':
        if not CLAUDE_API_KEY:
            raise ValueError("CLAUDE_API_KEY environment variable is required for Claude processor")
        return {
            'type': 'claude',
            'api_key': CLAUDE_API_KEY,
            'model_name': CLAUDE_MODEL_NAME
        }
    else:
        return {
            'type': 'local',
            'model_path': MODEL_PATH,
            'device': DEVICE,
            'enable_4bit': ENABLE_4BIT
        }

if __name__ == "__main__":
    print_config()
