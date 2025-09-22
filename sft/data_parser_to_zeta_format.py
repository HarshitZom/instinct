import os
import sys
# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from utils.datautils import get_dataset
from utils.synthdatautils import get_synth_dataset
from datasets import concatenate_datasets
from datetime import datetime
import torch
import math
from accelerate import Accelerator
import gc
import argparse
import json
import re

os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

accelerator = Accelerator()

parser = argparse.ArgumentParser(description="Instinct SFT Training")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--selekt_alpha", type=float, default=0.05, help="SeleKT alpha parameter")
parser.add_argument("--weight_decay", type=float, default=0.01, help="regularization")
parser.add_argument("--use-synth-data", action="store_true", help="Use synthetic data for training")
args = parser.parse_args()

CONFIG = {
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "model_size": "7B",
    "lr":args.lr,
    "selekt_alpha":args.selekt_alpha,
    "epochs":5,
    "batch_size":1,
    "accumulation_steps":4,
    "decay": args.weight_decay,
    "py_samples": 750,
    "c_samples": 200,
    "java_samples": 400,
    "rust_samples": 100,
}

model_name = f"Qwen/Qwen2.5-Coder-{CONFIG['model_size']}"
tokenizer = AutoTokenizer.from_pretrained(model_name)

ts_train_dataset   = get_dataset(train=True)
if args.use_synth_data:
    py_train_dataset   = get_synth_dataset("Python", train=True, num=CONFIG["py_samples"])
    c_train_dataset    = get_synth_dataset("C",      train=True, num=CONFIG["c_samples"])
    java_train_dataset = get_synth_dataset("Java",   train=True, num=CONFIG["java_samples"])
    rust_train_dataset = get_synth_dataset("Rust",   train=True, num=CONFIG["rust_samples"])

    train_dataset = concatenate_datasets([
            ts_train_dataset,
            py_train_dataset, 
            c_train_dataset, 
            java_train_dataset, 
            rust_train_dataset
        ])
    del py_train_dataset, c_train_dataset, java_train_dataset, rust_train_dataset
else:
    train_dataset = ts_train_dataset

del ts_train_dataset

train_dataset = train_dataset.filter(lambda example: example["messages"] is not None)

def format_chat(example):
    formatted = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return {"text": formatted}

train_dataset = train_dataset.map(lambda example: format_chat(example))
train_dataset = train_dataset.shuffle(seed=42)
eval_dataset = get_dataset(train=False)
eval_dataset = eval_dataset.filter(lambda example: example["messages"] is not None)
eval_dataset = eval_dataset.map(lambda example: format_chat(example))


def clean_tokens(text):
    """Remove unwanted tokens from text"""
    if not text:
        return text
    
    # Remove the specific tokens
    text = text.replace("<|im_start|>", "")
    text = text.replace("<|im_end|>", "")
    
    # Clean up any extra whitespace
    text = text.strip()
    
    return text


def parse_conversational_text(text):
    """
    Parse the conversational text to extract events, input, and output fields.
    
    Args:
        text: The formatted text from chat template
        
    Returns:
        dict with keys 'events', 'input', 'output' or None if parsing fails
    """
    if not text:
        return None
    
    try:
        # Extract the user content between <|im_start|>user and <|im_end|>
        user_match = re.search(r'<\|im_start\|>user\s*(.*?)\s*<\|im_end\|>', text, re.DOTALL)
        if not user_match:
            return None
        
        user_content = user_match.group(1).strip()
        
        # Extract the assistant content between <|im_start|>assistant and <|im_end|>
        assistant_match = re.search(r'<\|im_start\|>assistant\s*(.*?)\s*<\|im_end\|>', text, re.DOTALL)
        if not assistant_match:
            return None
        
        assistant_content = assistant_match.group(1).strip()
        
        # Extract events: everything between "### User Edits:" and "### User Excerpt:"
        events_match = re.search(r'### User Edits:\s*(.*?)\s*### User Excerpt:', user_content, re.DOTALL)
        if not events_match:
            return None
        
        events = events_match.group(1).strip()
        
        # Extract input: everything between "### User Excerpt:" and the end of user content
        input_match = re.search(r'### User Excerpt:\s*(.*)', user_content, re.DOTALL)
        if not input_match:
            return None
        
        input_content = input_match.group(1).strip()
        
        # Clean tokens from all fields
        events = clean_tokens(events)
        input_content = clean_tokens(input_content)
        assistant_content = clean_tokens(assistant_content)
        
        return {
            "events": events,
            "input": input_content,
            "output": assistant_content
        }
        
    except Exception as e:
        print(f"Error parsing conversational text: {e}")
        return None


def convert_to_zeta_format(example):
    """
    Convert a dataset example to the required format with events, input, and output fields.
    
    Args:
        example: Dataset example with 'text' field
        
    Returns:
        dict with 'events', 'input', 'output' fields or None if parsing fails
    """
    if "text" not in example:
        return None
    
    parsed = parse_conversational_text(example["text"])
    return parsed


def save_dataset_as_jsonl(dataset, filename):
    """
    Save dataset as JSONL file with the required format.
    
    Args:
        dataset: HuggingFace dataset
        filename: Output filename
    """
    print(f"Converting and saving dataset to {filename}...")
    
    # Convert dataset to the required format
    converted_dataset = dataset.map(convert_to_zeta_format)
    
    # Filter out None values (failed parses)
    converted_dataset = converted_dataset.filter(lambda x: x is not None)
    converted_dataset = converted_dataset.remove_columns(['messages', 'text'])
    # Save as JSONL
    with open(filename, 'w', encoding='utf-8') as f:
        for example in converted_dataset:
            if example and all(key in example for key in ["events", "input", "output"]):
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
    print(f"Saved {len(converted_dataset)} examples to {filename}")


# Generate output filenames with timestamp
train_output_file = f"train_zeta_format.jsonl"
eval_output_file = f"eval_zeta_format.jsonl"

# Save both datasets in the new format
print("Converting datasets to Zeta format...")
save_dataset_as_jsonl(train_dataset, train_output_file)
save_dataset_as_jsonl(eval_dataset, eval_output_file)

print(f"\nConversion complete!")
print(f"Train dataset saved as: {train_output_file}")
print(f"Eval dataset saved as: {eval_output_file}")


