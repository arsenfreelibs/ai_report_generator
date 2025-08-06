#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SL2 Query API Model Interactive Testing Script
Test your trained model with custom prompts
"""

import os
import json
import torch
import argparse
import sys
from datetime import datetime
from typing import Optional, List, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from peft import PeftModel

class SL2ModelTester:
    def __init__(self, model_path: str, base_model: str = "deepseek-ai/deepseek-coder-6.7b-instruct"):
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.pipe = None
        
    def check_model_path(self) -> bool:
        """Check if model path exists and is valid."""
        if not os.path.exists(self.model_path):
            print(f"❌ Model path not found: {self.model_path}")
            return False
            
        # Check if it's a LoRA model or merged model
        has_adapter_config = os.path.exists(os.path.join(self.model_path, "adapter_config.json"))
        has_config = os.path.exists(os.path.join(self.model_path, "config.json"))
        
        if not (has_adapter_config or has_config):
            print(f"❌ Invalid model directory: {self.model_path}")
            print("Directory should contain either adapter_config.json (LoRA) or config.json (merged model)")
            return False
            
        return True
    
    def load_model(self) -> bool:
        """Load the trained model."""
        print(f"Loading model from: {self.model_path}")
        
        try:
            # Check if it's a LoRA model or merged model
            if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
                print("📦 Loading LoRA model...")
                
                # Load base model first
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
                
                # Apply LoRA
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
                
            else:
                print("📦 Loading merged model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Add pad token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print("✅ Model loaded successfully!")
            print(f"  Model type: {'LoRA' if os.path.exists(os.path.join(self.model_path, 'adapter_config.json')) else 'Merged'}")
            print(f"  Device: {next(self.model.parameters()).device}")
            print(f"  Memory usage: {torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format prompt for the model."""
        if input_text:
            return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:"""
        else:
            return f"""### Instruction:
{instruction}

### Response:"""
    
    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate response from the model."""
        try:
            result = self.pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = result[0]['generated_text']
            # Extract only the response part
            response = generated_text.split("### Response:")[-1].strip()
            return response
            
        except Exception as e:
            return f"❌ Error generating response: {e}"
    
    def run_predefined_tests(self):
        """Run a set of predefined test prompts."""
        print("\n" + "="*60)
        print("RUNNING PREDEFINED TESTS")
        print("="*60)
        
        test_prompts = [
            {
                "name": "Basic Query",
                "instruction": "Write a SL2 Query API function for basic database queries with conditions",
                "input": ""
            },
            {
                "name": "Bulk Updates",
                "instruction": "Create a JavaScript function using SL2 Query API to perform bulk updates on multiple records",
                "input": ""
            },
            {
                "name": "Data Aggregation",
                "instruction": "Show me how to group data and apply aggregation functions using SL2 Query API",
                "input": ""
            },
            {
                "name": "Table Joins",
                "instruction": "Implement a SL2 database query function that joins multiple tables",
                "input": ""
            },
            {
                "name": "Array Operators",
                "instruction": "Write a SL2 Query API function that uses array operators like RTL and Multichoice",
                "input": ""
            }
        ]
        
        for i, test in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}: {test['name']} ---")
            print(f"Instruction: {test['instruction']}")
            if test['input']:
                print(f"Input: {test['input']}")
            
            prompt = self.format_prompt(test['instruction'], test['input'])
            response = self.generate_response(prompt)
            
            print(f"\n🤖 Generated Response:")
            print("-" * 40)
            print(response[:500] + ("..." if len(response) > 500 else ""))
            print("-" * 40)
            
            # Wait for user input to continue
            input("\nPress Enter to continue to next test...")
    
    def interactive_testing(self):
        """Run interactive testing session."""
        print("\n" + "="*60)
        print("INTERACTIVE TESTING MODE")
        print("="*60)
        print("Enter your prompts to test the model.")
        print("Commands:")
        print("  'exit' or 'quit' - Exit testing")
        print("  'help' - Show this help")
        print("  'examples' - Show example prompts")
        print("  'settings' - Adjust generation settings")
        print("="*60)
        
        # Generation settings
        max_tokens = 512
        temperature = 0.1
        
        while True:
            try:
                user_input = input("\n💭 Enter instruction (or command): ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  exit/quit/q - Exit testing")
                    print("  help - Show this help")
                    print("  examples - Show example prompts")
                    print("  settings - Adjust generation settings")
                    print("  clear - Clear screen")
                    continue
                
                if user_input.lower() == 'examples':
                    print("\nExample prompts you can try:")
                    print("• Write a SL2 function to find users by status")
                    print("• Create a pagination function using SL2 Query API")
                    print("• Show how to implement search with filters in SL2")
                    print("• Build a data export function with SL2 Query API")
                    print("• Create a function to handle batch operations")
                    continue
                
                if user_input.lower() == 'settings':
                    print(f"\nCurrent settings:")
                    print(f"  Max tokens: {max_tokens}")
                    print(f"  Temperature: {temperature}")
                    
                    try:
                        new_tokens = input(f"Max tokens ({max_tokens}): ").strip()
                        if new_tokens:
                            max_tokens = int(new_tokens)
                        
                        new_temp = input(f"Temperature ({temperature}): ").strip()
                        if new_temp:
                            temperature = float(new_temp)
                        
                        print(f"✅ Settings updated: max_tokens={max_tokens}, temperature={temperature}")
                    except ValueError:
                        print("❌ Invalid input. Settings unchanged.")
                    continue
                
                if user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                if not user_input:
                    continue
                
                # Check if user wants to provide input as well
                additional_input = input("📝 Additional input (optional, press Enter to skip): ").strip()
                
                # Format and generate
                prompt = self.format_prompt(user_input, additional_input)
                
                print("\n🤖 Generating response...")
                start_time = datetime.now()
                response = self.generate_response(prompt, max_tokens, temperature)
                end_time = datetime.now()
                
                duration = (end_time - start_time).total_seconds()
                
                print(f"\n✨ Generated Response (took {duration:.1f}s):")
                print("=" * 60)
                print(response)
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run_batch_test(self, prompts_file: str):
        """Run batch testing from a file."""
        print(f"\n📁 Running batch test from: {prompts_file}")
        
        if not os.path.exists(prompts_file):
            print(f"❌ Prompts file not found: {prompts_file}")
            return
        
        try:
            with open(prompts_file, 'r') as f:
                if prompts_file.endswith('.json'):
                    prompts_data = json.load(f)
                else:
                    # Assume each line is a prompt
                    prompts_data = [{"instruction": line.strip()} for line in f if line.strip()]
            
            results = []
            for i, prompt_data in enumerate(prompts_data, 1):
                instruction = prompt_data.get('instruction', '')
                input_text = prompt_data.get('input', '')
                
                print(f"\n--- Batch Test {i}/{len(prompts_data)} ---")
                print(f"Instruction: {instruction}")
                
                prompt = self.format_prompt(instruction, input_text)
                response = self.generate_response(prompt)
                
                result = {
                    'prompt': instruction,
                    'input': input_text,
                    'response': response,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
                
                print(f"Response: {response[:100]}...")
            
            # Save results
            results_file = f"batch_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✅ Batch testing completed!")
            print(f"📄 Results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Error in batch testing: {e}")

def find_latest_model(models_dir: str = "colab/models") -> Optional[str]:
    """Find the latest trained model."""
    if not os.path.exists(models_dir):
        return None
    
    model_dirs = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            final_model_path = os.path.join(item_path, "final_model")
            if os.path.exists(final_model_path):
                model_dirs.append((item, item_path))
    
    if not model_dirs:
        return None
    
    # Sort by name (which includes timestamp) and return latest
    model_dirs.sort(reverse=True)
    latest_dir = model_dirs[0][1]
    return os.path.join(latest_dir, "final_model")

def list_available_models(models_dir: str = "colab/models"):
    """List all available trained models."""
    print(f"\n📦 Available models in {models_dir}:")
    
    if not os.path.exists(models_dir):
        print("  No models directory found.")
        return []
    
    models = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            final_model_path = os.path.join(item_path, "final_model")
            if os.path.exists(final_model_path):
                # Get model info if available
                info_file = os.path.join(models_dir, f"{item.split('-')[0]}-info.json")
                info = {}
                if os.path.exists(info_file):
                    try:
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                    except:
                        pass
                
                models.append((item, final_model_path, info))
                print(f"  📁 {item}")
                if 'training_date' in info:
                    print(f"      Trained: {info['training_date']}")
                if 'config' in info:
                    config = info['config']
                    print(f"      Config: {config.get('epochs', 'N/A')} epochs, LR {config.get('learning_rate', 'N/A')}")
    
    return models

def main():
    parser = argparse.ArgumentParser(description="Test trained SL2 Query API model")
    parser.add_argument("--model", "-m",
                       help="Path to trained model directory")
    parser.add_argument("--base-model", "-b",
                       default="deepseek-ai/deepseek-coder-6.7b-instruct",
                       help="Base model name")
    parser.add_argument("--mode", "-t",
                       choices=["interactive", "predefined", "batch"],
                       default="interactive",
                       help="Testing mode")
    parser.add_argument("--batch-file", "-f",
                       help="File with prompts for batch testing")
    parser.add_argument("--list", "-l",
                       action="store_true",
                       help="List available models and exit")
    parser.add_argument("--latest",
                       action="store_true",
                       help="Use latest trained model")
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list:
        list_available_models()
        return 0
    
    # Determine model path
    model_path = args.model
    
    if args.latest:
        model_path = find_latest_model()
        if not model_path:
            print("❌ No trained models found. Run training first.")
            return 1
        print(f"🔍 Using latest model: {model_path}")
    
    if not model_path:
        # Try to find latest model automatically
        model_path = find_latest_model()
        if not model_path:
            print("❌ No model specified and no trained models found.")
            print("Usage:")
            print("  python test_model.py --model /path/to/model")
            print("  python test_model.py --latest")
            print("  python test_model.py --list")
            return 1
        print(f"🔍 Auto-detected model: {model_path}")
    
    # Create tester
    tester = SL2ModelTester(model_path, args.base_model)
    
    # Check model path
    if not tester.check_model_path():
        return 1
    
    # Load model
    if not tester.load_model():
        return 1
    
    # Run testing based on mode
    try:
        if args.mode == "predefined":
            tester.run_predefined_tests()
        elif args.mode == "batch":
            if not args.batch_file:
                print("❌ Batch mode requires --batch-file argument")
                return 1
            tester.run_batch_test(args.batch_file)
        else:  # interactive
            tester.interactive_testing()
    
    except KeyboardInterrupt:
        print("\n\n👋 Testing interrupted. Goodbye!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
