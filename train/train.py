#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SL2 Query API Model Fine-tuning with QLoRA
Local version of the Colab training notebook
"""

import os
import json
import torch
import shutil
import zipfile
from datetime import datetime
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer
from huggingface_hub import login
import pandas as pd
import numpy as np
import argparse
import sys
from typing import Optional, Tuple

# Configuration
DEFAULT_MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
DEFAULT_NEW_MODEL_NAME = "deepseek-coder-6.7b-sl2-query-api"

class SL2ModelTrainer:
    def __init__(self, 
                 model_name: str = DEFAULT_MODEL_NAME,
                 new_model_name: str = DEFAULT_NEW_MODEL_NAME,
                 dataset_path: str = "colab/datasets/sl2-query-api-dataset-alpaca",
                 output_dir: str = "colab/models",
                 logs_dir: str = "colab/logs"):
        
        self.model_name = model_name
        self.new_model_name = new_model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.logs_dir = logs_dir
        
        # Training hyperparameters
        self.batch_size = 4
        self.gradient_accumulation_steps = 4
        self.learning_rate = 2e-4
        self.num_epochs = 3
        self.max_seq_length = 2048
        self.warmup_ratio = 0.03
        self.weight_decay = 0.001
        self.logging_steps = 10
        self.save_steps = 100
        self.eval_steps = 100
        
        # QLoRA configuration
        self.lora_r = 64
        self.lora_alpha = 16
        self.lora_dropout = 0.1
        self.lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        # Initialize directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate unique run name
        self.run_name = f"sl2-query-api-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.run_output_dir = os.path.join(output_dir, self.run_name)
        
        print(f"Training Configuration:")
        print(f"  Model: {self.model_name}")
        print(f"  New model name: {self.new_model_name}")
        print(f"  Dataset path: {self.dataset_path}")
        print(f"  Output directory: {self.run_output_dir}")
        print(f"  Max sequence length: {self.max_seq_length}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Epochs: {self.num_epochs}")

    def check_gpu(self):
        """Check GPU availability and print information."""
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️ No GPU detected. Training will be very slow on CPU.")
            response = input("Continue without GPU? (y/n): ").lower().strip()
            if response != 'y':
                print("Training cancelled.")
                return False
        return True

    def load_dataset(self):
        """Load the training dataset."""
        try:
            if os.path.exists(self.dataset_path):
                print(f"Loading dataset from local path: {self.dataset_path}")
                dataset = load_from_disk(self.dataset_path)
            else:
                print(f"❌ Dataset not found at {self.dataset_path}")
                print("Please run create_dataset.py first or provide correct dataset path")
                return None

            print(f"✅ Dataset loaded successfully!")
            print(f"  Train samples: {len(dataset['train'])}")
            print(f"  Validation samples: {len(dataset['validation'])}")
            print(f"  Test samples: {len(dataset['test'])}")

            # Preview dataset
            print("\n🔍 Dataset sample:")
            sample = dataset['train'][0]
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")

            return dataset

        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None

    def setup_quantization_config(self):
        """Setup BitsAndBytes quantization configuration."""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print("✅ Quantization config created")
        return bnb_config

    def load_model_and_tokenizer(self, bnb_config):
        """Load model and tokenizer with quantization."""
        print(f"Loading model: {self.model_name}")
        print("This may take a few minutes...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )

        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        # Disable cache for training
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        print("✅ Model and tokenizer loaded successfully")
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Model dtype: {next(model.parameters()).dtype}")

        return model, tokenizer

    def setup_lora_config(self):
        """Setup LoRA configuration."""
        peft_config = LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.lora_target_modules,
        )

        print("✅ LoRA config created")
        print(f"  LoRA rank (r): {self.lora_r}")
        print(f"  LoRA alpha: {self.lora_alpha}")
        print(f"  LoRA dropout: {self.lora_dropout}")
        print(f"  Target modules: {self.lora_target_modules}")

        return peft_config

    def format_dataset(self, dataset):
        """Format the dataset for instruction following."""
        def format_instruction(sample):
            instruction = sample['instruction']
            input_text = sample['input']
            output = sample['output']

            if input_text:
                prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
            else:
                prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""

            return {"text": prompt}

        # Format datasets
        train_dataset = dataset['train'].map(format_instruction)
        eval_dataset = dataset['validation'].map(format_instruction)

        print("✅ Training data formatted")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Evaluation samples: {len(eval_dataset)}")

        # Preview formatted sample
        print("\n🔍 Formatted sample:")
        print(train_dataset[0]['text'][:500] + "...")

        return train_dataset, eval_dataset

    def setup_training_arguments(self):
        """Setup training arguments."""
        training_arguments = TrainingArguments(
            output_dir=self.run_output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=self.logging_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=self.warmup_ratio,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="tensorboard",
            eval_strategy="steps",
            eval_steps=self.eval_steps,
            save_strategy="steps",
            save_steps=self.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            push_to_hub=False,
            logging_dir=os.path.join(self.logs_dir, self.run_name),
        )

        print("✅ Training arguments configured")
        print(f"  Output directory: {self.run_output_dir}")
        print(f"  Run name: {self.run_name}")

        return training_arguments

    def create_trainer(self, model, train_dataset, eval_dataset, training_arguments, tokenizer):
        """Create the trainer."""
        # Format dataset for training
        def format_dataset_simple(dataset):
            def format_example(example):
                return {"text": example["text"]}
            return dataset.map(format_example)

        print("Formatting datasets...")
        simple_train_dataset = format_dataset_simple(train_dataset)
        simple_eval_dataset = format_dataset_simple(eval_dataset)

        # Try SFTTrainer first, fallback to standard Trainer
        try:
            print("Creating SFTTrainer...")
            trainer = SFTTrainer(
                model=model,
                train_dataset=simple_train_dataset,
                eval_dataset=simple_eval_dataset,
                args=training_arguments,
            )
            print("✅ SFTTrainer initialized successfully")
            return trainer

        except Exception as e:
            print(f"⚠️ SFTTrainer failed: {e}")
            print("Trying fallback with standard Trainer...")

            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_seq_length,
                )

            # Tokenize datasets
            tokenized_train = simple_train_dataset.map(tokenize_function, batched=True)
            tokenized_eval = simple_eval_dataset.map(tokenize_function, batched=True)

            # Create standard trainer
            trainer = Trainer(
                model=model,
                args=training_arguments,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                tokenizer=tokenizer,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False,
                ),
            )
            print("✅ Fallback Trainer initialized successfully")
            return trainer

    def save_model(self, trainer, tokenizer):
        """Save the trained model."""
        print("\nSaving trained model...")

        # Save paths
        final_model_path = os.path.join(self.run_output_dir, "final_model")
        backup_path = os.path.join(self.output_dir, "backups", f"{self.new_model_name}-{self.run_name}")

        # Save to primary location
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        # Create backup copy
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        try:
            shutil.copytree(final_model_path, backup_path)
            print(f"✅ Backup created: {backup_path}")
        except Exception as e:
            print(f"⚠️ Failed to create backup: {e}")

        # Save model info
        model_info = {
            "model_name": self.new_model_name,
            "base_model": self.model_name,
            "training_date": datetime.now().isoformat(),
            "run_name": self.run_name,
            "paths": {
                "primary": final_model_path,
                "backup": backup_path
            },
            "config": {
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "learning_rate": self.learning_rate,
                "epochs": self.num_epochs,
                "batch_size": self.batch_size
            }
        }

        info_path = os.path.join(self.output_dir, f"{self.new_model_name}-info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        print(f"✅ Model saved:")
        print(f"  Primary: {final_model_path}")
        print(f"  Backup: {backup_path}")
        print(f"  Info: {info_path}")

        return final_model_path, backup_path

    def create_model_archive(self, model_path: str):
        """Create a compressed archive of the model."""
        archive_name = f"{self.new_model_name}-{self.run_name}.zip"
        archive_path = os.path.join(self.output_dir, "backups", archive_name)

        try:
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, model_path)
                        zipf.write(file_path, arcname)

            archive_size = os.path.getsize(archive_path) / (1024 * 1024)
            print(f"✅ Model archive created: {archive_path} ({archive_size:.1f} MB)")
            return archive_path

        except Exception as e:
            print(f"❌ Failed to create archive: {e}")
            return None

    def test_model(self, model_path: str):
        """Test the trained model."""
        print("\n" + "="*50)
        print("TESTING THE TRAINED MODEL")
        print("="*50)

        try:
            # Load base model and apply LoRA
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            test_model = PeftModel.from_pretrained(base_model, model_path)
            test_tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=test_model,
                tokenizer=test_tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
                top_p=0.95,
                pad_token_id=test_tokenizer.eos_token_id
            )

            # Test prompts
            test_prompts = [
                "### Instruction:\nWrite a SL2 Query API function for basic database queries with conditions\n\n### Response:",
                "### Instruction:\nCreate a JavaScript function using SL2 Query API to perform bulk updates on multiple records\n\n### Response:",
                "### Instruction:\nShow me how to group data and apply aggregation functions using SL2 Query API\n\n### Response:",
                "### Instruction:\nImplement a SL2 database query function that joins multiple tables\n\n### Response:"
            ]

            print("\n🧪 Testing trained model:")

            for i, prompt in enumerate(test_prompts, 1):
                print(f"\n--- Test {i} ---")
                prompt_text = prompt.split('### Instruction:')[1].split('### Response:')[0].strip()
                print(f"Prompt: {prompt_text}")

                try:
                    result = pipe(prompt)
                    generated_text = result[0]['generated_text']
                    response = generated_text.split("### Response:")[-1].strip()
                    print(f"Generated Response:\n{response[:300]}...")

                except Exception as e:
                    print(f"❌ Error generating response: {e}")

            print("✅ Model testing completed!")
            return pipe

        except Exception as e:
            print(f"⚠️ Model testing failed: {e}")
            return None

    def train(self, test_after_training: bool = True, create_archive: bool = True):
        """Main training method."""
        print("="*50)
        print("SL2 QUERY API MODEL TRAINING")
        print("="*50)

        # Check GPU
        if not self.check_gpu():
            return False

        # Load dataset
        dataset = self.load_dataset()
        if dataset is None:
            return False

        # Setup quantization
        bnb_config = self.setup_quantization_config()

        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer(bnb_config)

        # Setup LoRA
        peft_config = self.setup_lora_config()
        print("Applying PEFT adapters to model...")
        model = get_peft_model(model, peft_config)
        print("✅ PEFT adapters applied to model")

        # Format dataset
        train_dataset, eval_dataset = self.format_dataset(dataset)

        # Setup training arguments
        training_arguments = self.setup_training_arguments()

        # Create trainer
        trainer = self.create_trainer(model, train_dataset, eval_dataset, training_arguments, tokenizer)

        # Start training
        print("\n" + "="*50)
        print("STARTING TRAINING")
        print("="*50)

        estimated_time = self.num_epochs * len(train_dataset) // (self.batch_size * self.gradient_accumulation_steps) // 60
        print(f"Training will start now. Estimated duration: ~{estimated_time} minutes")

        try:
            trainer.train()
            print("✅ Training completed successfully!")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False

        # Save model
        final_model_path, backup_path = self.save_model(trainer, tokenizer)

        # Create archive if requested
        if create_archive:
            self.create_model_archive(final_model_path)

        # Test model if requested
        if test_after_training:
            self.test_model(final_model_path)

        # Print summary
        self.print_training_summary(dataset, final_model_path)

        return True

    def print_training_summary(self, dataset, model_path):
        """Print training summary."""
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)

        print(f"\n📊 Training Summary:")
        print(f"  Base Model: {self.model_name}")
        print(f"  Training Method: QLoRA")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Training Samples: {len(dataset['train'])}")
        print(f"  Validation Samples: {len(dataset['validation'])}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Batch Size: {self.batch_size}")

        print(f"\n📁 Model Files:")
        print(f"  Model: {model_path}")
        print(f"  Logs: {os.path.join(self.logs_dir, self.run_name)}")

        print(f"\n🔄 Loading Model Later:")
        print(f"```python")
        print(f"from transformers import AutoTokenizer, AutoModelForCausalLM")
        print(f"from peft import PeftModel")
        print(f"")
        print(f"base_model = AutoModelForCausalLM.from_pretrained('{self.model_name}')")
        print(f"model = PeftModel.from_pretrained(base_model, '{model_path}')")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{self.model_name}')")
        print(f"```")

        print(f"\n🎯 Model is ready for SL2 Query API code generation!")

def main():
    parser = argparse.ArgumentParser(description="Train SL2 Query API model with QLoRA")
    parser.add_argument("--model", "-m", 
                       default=DEFAULT_MODEL_NAME,
                       help="Base model name")
    parser.add_argument("--new-model-name", "-n", 
                       default=DEFAULT_NEW_MODEL_NAME,
                       help="New model name")
    parser.add_argument("--dataset", "-d", 
                       default="colab/datasets/sl2-query-api-dataset-alpaca",
                       help="Dataset path")
    parser.add_argument("--output", "-o", 
                       default="colab/models",
                       help="Output directory")
    parser.add_argument("--logs", "-l", 
                       default="colab/logs",
                       help="Logs directory")
    parser.add_argument("--epochs", "-e", 
                       type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", 
                       type=int, default=4,
                       help="Batch size")
    parser.add_argument("--learning-rate", "-lr", 
                       type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--no-test", 
                       action="store_true",
                       help="Skip model testing after training")
    parser.add_argument("--no-archive", 
                       action="store_true",
                       help="Skip creating model archive")
    parser.add_argument("--upload", "-u", 
                       action="store_true",
                       help="Upload to Hugging Face Hub (requires login)")

    args = parser.parse_args()

    # Create trainer
    trainer = SL2ModelTrainer(
        model_name=args.model,
        new_model_name=args.new_model_name,
        dataset_path=args.dataset,
        output_dir=args.output,
        logs_dir=args.logs
    )

    # Update hyperparameters
    trainer.num_epochs = args.epochs
    trainer.batch_size = args.batch_size
    trainer.learning_rate = args.learning_rate

    # Login to HF if uploading
    if args.upload:
        print("Logging in to Hugging Face...")
        try:
            login()
        except Exception as e:
            print(f"❌ Failed to login to Hugging Face: {e}")
            return 1

    # Start training
    success = trainer.train(
        test_after_training=not args.no_test,
        create_archive=not args.no_archive
    )

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
