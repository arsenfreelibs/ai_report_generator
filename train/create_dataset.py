#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SL2 Query API Dataset Creation for Fine-tuning
Local version of the Colab notebook for creating training datasets
"""

import os
import re
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import login, HfApi
import numpy as np
from typing import List, Dict, Tuple
import random
from datetime import datetime
import argparse
import sys

# Configuration
DATASET_VERSION = "1.0"
DEFAULT_HF_REPO_NAME = "arsen-r-a/sl2-query-api-dataset"

class SL2DatasetCreator:
    def __init__(self, examples_file: str, output_dir: str, dataset_name: str = "sl2-query-api-dataset"):
        self.examples_file = examples_file
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.hf_repo_name = DEFAULT_HF_REPO_NAME
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_examples_file(self) -> str:
        """Load the examples file."""
        try:
            with open(self.examples_file, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"✅ Successfully loaded examples file ({len(content)} characters)")
            return content
        except FileNotFoundError:
            print(f"❌ Error: File not found at {self.examples_file}")
            print("Please ensure examples.md exists at the specified path")
            return ""
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return ""

    def extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract JavaScript code blocks and their descriptions."""
        # Pattern to match function definitions and their code
        function_pattern = r'(// [=]{5,}\s*\n// ([A-Z\s\(\),]+)\s*\n// [=]{5,}\s*\n)(.*?)(?=\n// [=]{5,}|\nif \(typeof|$)'

        examples = []
        matches = re.finditer(function_pattern, content, re.DOTALL)

        for match in matches:
            header = match.group(1).strip()
            category = match.group(2).strip()
            code_block = match.group(3).strip()

            # Extract individual functions from the code block
            func_pattern = r'(const\s+\w+\s*=\s*async\s*\(\s*\)\s*=>\s*{.*?};)'
            func_matches = re.finditer(func_pattern, code_block, re.DOTALL)

            for func_match in func_matches:
                func_code = func_match.group(1)
                func_name_match = re.search(r'const\s+(\w+)', func_code)
                if func_name_match:
                    func_name = func_name_match.group(1)

                    examples.append({
                        'category': category,
                        'function_name': func_name,
                        'code': func_code,
                        'description': f"SL2 Query API example for {category.lower()}: {func_name}"
                    })

        print(f"✅ Extracted {len(examples)} code examples")
        return examples

    def create_instruction_data(self, examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Create instruction-following dataset from code examples."""
        instruction_templates = [
            "Write a SL2 Query API function for {task}",
            "Create a JavaScript function using SL2 Query API to {task}",
            "Implement a SL2 database query function that {task}",
            "Show me how to {task} using SL2 Query API",
            "Generate SL2 Query API code for {task}",
            "Write a function that {task} using the SL2 JavaScript API",
            "Create a SL2 query function to {task}",
            "How do I {task} with SL2 Query API?"
        ]

        task_descriptions = {
            'BASIC FIND OPERATIONS': [
                'performs basic database queries with conditions',
                'finds records with simple and complex filters',
                'searches for records using AND/OR conditions'
            ],
            'ARRAY OPERATORS (RTL, MULTICHOICE)': [
                'queries arrays and multichoice fields',
                'searches records with array contains operations',
                'filters data using array operators'
            ],
            'RTL, REF, GLOBALREF QUERIES': [
                'queries related records using references',
                'searches using relationship fields',
                'finds records through model relationships'
            ],
            'ORDERING': [
                'sorts query results by multiple fields',
                'orders records in ascending/descending order',
                'applies custom sorting to database queries'
            ],
            'JOINS': [
                'joins multiple tables in queries',
                'performs complex multi-table operations',
                'combines data from related models'
            ],
            'GROUPING OPERATIONS': [
                'groups data and applies aggregation functions',
                'calculates statistics using GROUP BY',
                'performs data aggregation with MAX, MIN, SUM, COUNT, AVG'
            ],
            'BULK OPERATIONS': [
                'performs bulk updates on multiple records',
                'executes mass delete operations',
                'handles batch data operations efficiently'
            ],
            'PAGINATION AND LIMITS': [
                'implements pagination for large datasets',
                'limits query results with offset',
                'handles data pagination efficiently'
            ]
        }

        dataset_entries = []

        for example in examples:
            category = example['category']
            code = example['code']

            # Get task descriptions for this category
            tasks = task_descriptions.get(category, [f"works with {category.lower()}"])

            for task in tasks:
                # Random instruction template
                instruction_template = random.choice(instruction_templates)
                instruction = instruction_template.format(task=task)

                # Create training entry
                entry = {
                    'instruction': instruction,
                    'input': '',
                    'output': code,
                    'category': category,
                    'function_name': example['function_name']
                }
                dataset_entries.append(entry)

        print(f"✅ Created {len(dataset_entries)} instruction-following examples")
        return dataset_entries

    def create_chat_format(self, entry: Dict[str, str]) -> Dict[str, str]:
        """Convert to chat format for training."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert JavaScript developer specializing in SL2 Query API. You write clean, efficient, and well-documented code following SL2 best practices."
            },
            {
                "role": "user",
                "content": entry['instruction']
            },
            {
                "role": "assistant",
                "content": entry['output']
            }
        ]

        return {
            'messages': messages,
            'category': entry['category'],
            'function_name': entry['function_name']
        }

    def create_alpaca_format(self, entry: Dict[str, str]) -> Dict[str, str]:
        """Convert to Alpaca format for training."""
        return {
            'instruction': entry['instruction'],
            'input': entry['input'],
            'output': entry['output'],
            'category': entry['category'],
            'function_name': entry['function_name']
        }

    def split_dataset(self, data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Split dataset into train/validation/test sets."""
        random.shuffle(data)
        n = len(data)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        return train_data, val_data, test_data

    def save_datasets(self, chat_dataset: DatasetDict, alpaca_dataset: DatasetDict, 
                     train_chat: List[Dict], train_alpaca: List[Dict], 
                     examples: List[Dict], instruction_data: List[Dict]):
        """Save datasets to disk."""
        # Define paths
        chat_path = os.path.join(self.output_dir, f"{self.dataset_name}-chat")
        alpaca_path = os.path.join(self.output_dir, f"{self.dataset_name}-alpaca")
        json_backup_dir = os.path.join(self.output_dir, "json_backups")
        
        print("Saving datasets...")
        print(f"📁 Chat format: {chat_path}")
        print(f"📁 Alpaca format: {alpaca_path}")

        # Save as Hugging Face Dataset format
        chat_dataset.save_to_disk(chat_path)
        alpaca_dataset.save_to_disk(alpaca_path)

        # Create JSON backups
        os.makedirs(json_backup_dir, exist_ok=True)
        
        print("Creating JSON backups...")
        with open(os.path.join(json_backup_dir, "chat_train_sample.json"), 'w') as f:
            json.dump(train_chat[:10], f, indent=2)

        with open(os.path.join(json_backup_dir, "chat_full_train.json"), 'w') as f:
            json.dump(train_chat, f, indent=2)

        with open(os.path.join(json_backup_dir, "alpaca_train_sample.json"), 'w') as f:
            json.dump(train_alpaca[:10], f, indent=2)

        with open(os.path.join(json_backup_dir, "alpaca_full_train.json"), 'w') as f:
            json.dump(train_alpaca, f, indent=2)

        # Save dataset statistics
        stats = {
            "creation_date": datetime.now().isoformat(),
            "dataset_version": DATASET_VERSION,
            "original_examples": len(examples),
            "instruction_entries": len(instruction_data),
            "categories": list(set([e['category'] for e in examples])),
            "splits": {
                "train": len(chat_dataset['train']),
                "validation": len(chat_dataset['validation']),
                "test": len(chat_dataset['test'])
            },
            "category_breakdown": {cat: len([e for e in instruction_data if e['category'] == cat])
                                  for cat in set([e['category'] for e in instruction_data])},
            "paths": {
                "chat_dataset": chat_path,
                "alpaca_dataset": alpaca_path,
                "json_backups": json_backup_dir
            }
        }

        stats_path = os.path.join(self.output_dir, "dataset_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Create README
        self._create_readme(stats, chat_path, alpaca_path, json_backup_dir)
        
        print(f"✅ Datasets saved:")
        print(f"  📊 Chat format: {chat_path}")
        print(f"  📊 Alpaca format: {alpaca_path}")
        print(f"  📄 JSON backups: {json_backup_dir}")
        print(f"  📈 Statistics: {stats_path}")

    def _create_readme(self, stats: Dict, chat_path: str, alpaca_path: str, json_backup_dir: str):
        """Create README file for the datasets."""
        readme_content = f"""# SL2 Query API Training Datasets

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Version: {DATASET_VERSION}

## Dataset Information

- **Original Examples**: {stats['original_examples']}
- **Training Entries**: {stats['instruction_entries']}
- **Categories**: {len(stats['categories'])}

## Dataset Splits

- **Training**: {stats['splits']['train']} samples
- **Validation**: {stats['splits']['validation']} samples
- **Test**: {stats['splits']['test']} samples

## Formats Available

### 1. Chat Format (`{self.dataset_name}-chat/`)
For training with chat-based fine-tuning:
```python
from datasets import load_from_disk
dataset = load_from_disk('{chat_path}')
```

### 2. Alpaca Format (`{self.dataset_name}-alpaca/`)
For training with instruction-following format:
```python
from datasets import load_from_disk
dataset = load_from_disk('{alpaca_path}')
```

### 3. JSON Backups (`json_backups/`)
Raw JSON files for inspection and backup purposes.

## Category Breakdown

{chr(10).join([f"- **{cat}**: {count} examples" for cat, count in sorted(stats['category_breakdown'].items())])}

## Usage in Training

```python
# Load for training
from datasets import load_from_disk

# For QLoRA/LoRA training (recommended)
dataset = load_from_disk('{alpaca_path}')

# For chat-based training
dataset = load_from_disk('{chat_path}')

train_dataset = dataset['train']
eval_dataset = dataset['validation']
test_dataset = dataset['test']
```

Generated by SL2 Query API Dataset Creation Pipeline
"""

        readme_path = os.path.join(self.output_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(readme_content)

    def upload_to_hub(self, chat_dataset: DatasetDict, alpaca_dataset: DatasetDict):
        """Upload datasets to Hugging Face Hub."""
        try:
            print("Uploading chat format dataset to Hugging Face Hub...")
            chat_dataset.push_to_hub(
                f"{self.hf_repo_name}-chat",
                private=False,
                commit_description=f"SL2 Query API training dataset v{DATASET_VERSION} - Chat format"
            )

            print("Uploading Alpaca format dataset to Hugging Face Hub...")
            alpaca_dataset.push_to_hub(
                f"{self.hf_repo_name}-alpaca",
                private=False,
                commit_description=f"SL2 Query API training dataset v{DATASET_VERSION} - Alpaca format"
            )

            print("✅ Successfully uploaded datasets to Hugging Face Hub!")
            print(f"Chat format: https://huggingface.co/datasets/{self.hf_repo_name}-chat")
            print(f"Alpaca format: https://huggingface.co/datasets/{self.hf_repo_name}-alpaca")

        except Exception as e:
            print(f"❌ Error uploading to Hub: {e}")
            print("You can upload manually later or check your credentials")

    def create_dataset(self, upload_to_hf: bool = False):
        """Main method to create the dataset."""
        print("="*50)
        print("SL2 QUERY API DATASET CREATION")
        print("="*50)

        # Load examples
        print("Loading examples file...")
        content = self.load_examples_file()
        if not content:
            print("❌ Cannot proceed without examples file.")
            return False

        # Extract code examples
        print("\nExtracting code examples...")
        examples = self.extract_code_blocks(content)
        if not examples:
            print("❌ No examples found. Please check the file format.")
            return False

        # Create instruction dataset
        print("\nCreating instruction dataset...")
        instruction_data = self.create_instruction_data(examples)

        # Create formats
        print("Creating chat format dataset...")
        chat_data = [self.create_chat_format(entry) for entry in instruction_data]

        print("Creating Alpaca format dataset...")
        alpaca_data = [self.create_alpaca_format(entry) for entry in instruction_data]

        # Split datasets
        train_chat, val_chat, test_chat = self.split_dataset(chat_data)
        train_alpaca, val_alpaca, test_alpaca = self.split_dataset(alpaca_data)

        print(f"\nDataset splits:")
        print(f"Chat format - Train: {len(train_chat)}, Val: {len(val_chat)}, Test: {len(test_chat)}")
        print(f"Alpaca format - Train: {len(train_alpaca)}, Val: {len(val_alpaca)}, Test: {len(test_alpaca)}")

        # Create Hugging Face datasets
        chat_dataset = DatasetDict({
            'train': Dataset.from_list(train_chat),
            'validation': Dataset.from_list(val_chat),
            'test': Dataset.from_list(test_chat)
        })

        alpaca_dataset = DatasetDict({
            'train': Dataset.from_list(train_alpaca),
            'validation': Dataset.from_list(val_alpaca),
            'test': Dataset.from_list(test_alpaca)
        })

        print("✅ Created Hugging Face datasets")

        # Save datasets
        self.save_datasets(chat_dataset, alpaca_dataset, train_chat, train_alpaca, 
                          examples, instruction_data)

        # Upload to HF Hub if requested
        if upload_to_hf:
            self.upload_to_hub(chat_dataset, alpaca_dataset)

        # Print summary
        print("\n" + "="*50)
        print("DATASET CREATION SUMMARY")
        print("="*50)
        print(f"Original examples extracted: {len(examples)}")
        print(f"Instruction entries created: {len(instruction_data)}")
        print(f"Categories covered: {len(set([e['category'] for e in examples]))}")

        # Category breakdown
        category_counts = {}
        for entry in instruction_data:
            cat = entry['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1

        print(f"\nCategory breakdown:")
        for cat, count in sorted(category_counts.items()):
            print(f"  {cat}: {count} examples")

        print(f"\n✅ Dataset creation completed successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Create SL2 Query API training dataset")
    parser.add_argument("--examples", "-e", 
                       default="colab/input/examples.md",
                       help="Path to examples.md file")
    parser.add_argument("--output", "-o", 
                       default="colab/datasets",
                       help="Output directory for datasets")
    parser.add_argument("--name", "-n", 
                       default="sl2-query-api-dataset",
                       help="Dataset name")
    parser.add_argument("--upload", "-u", 
                       action="store_true",
                       help="Upload to Hugging Face Hub")
    parser.add_argument("--hf-repo", 
                       default=DEFAULT_HF_REPO_NAME,
                       help="Hugging Face repository name")

    args = parser.parse_args()

    # Make paths absolute
    examples_file = os.path.abspath(args.examples)
    output_dir = os.path.abspath(args.output)

    print(f"Examples file: {examples_file}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset name: {args.name}")

    # Create dataset creator
    creator = SL2DatasetCreator(examples_file, output_dir, args.name)
    creator.hf_repo_name = args.hf_repo

    # Login to HF if uploading
    if args.upload:
        print("Logging in to Hugging Face...")
        try:
            login()
        except Exception as e:
            print(f"❌ Failed to login to Hugging Face: {e}")
            return 1

    # Create dataset
    success = creator.create_dataset(upload_to_hf=args.upload)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
