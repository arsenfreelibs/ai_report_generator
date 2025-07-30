import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class LLMProcessor:
    """Handles language model initialization and query processing"""

    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", 
                 enable_4bit: bool = True):
        """Initialize the LLM processor"""
        self.model_path = model_path
        self.device = device
        self.enable_4bit = enable_4bit
        self.tokenizer = None
        self.model = None

    def initialize_model(self):
        """Initialize the language model with optimizations"""
        print(f"Initializing model {self.model_path} on {self.device}...")

        # Configure quantization parameters for efficient inference
        if self.enable_4bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = None

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="offload",
            offload_state_dict=True if self.device == "cuda" else False,
        )

        return self.tokenizer, self.model

    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Generate a response from the language model"""
        if not self.model or not self.tokenizer:
            self.initialize_model()

        # Format prompt according to model's expected format
        if "mixtral" in self.model_path.lower():
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        elif "codellama" in self.model_path.lower():
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        elif "qwen" in self.model_path.lower():
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif "deepseek" in self.model_path.lower():
            formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        else:
            formatted_prompt = prompt

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        # Generate with appropriate parameters
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the part after the prompt
        response = response.split(prompt)[-1].strip()

        # Clean up any model-specific formatting
        response = response.replace("<|im_end|>", "").replace("[/INST]", "").replace("### Response:", "").strip()

        return response
