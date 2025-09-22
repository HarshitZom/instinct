import sys
import os
# sys.path.append("../utils/")
# sys.path.append("../eval")
from transformers import AutoTokenizer
from utils.datautils import get_dataset, TRAIN_PROMPT_PREFACE
from utils.synthdatautils import get_synth_dataset
from datasets import concatenate_datasets
from datetime import datetime
import wandb
from trl import SFTTrainer, SFTConfig
from utils.utils import show_memory_stats, show_final_memory_stats, get_model_and_tokenizer, token_count, print_with_separation
from utils.selekt import Callback
import torch
import math
import numpy as np
from accelerate import Accelerator
from eval.eval_callback import EvalCallback
from eval.synthdata_callback import SynthDataEvalCallback
import os
import gc
import argparse
from .custom_collator import MaskingCollator

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

run_name = f"{CONFIG['model_size']}-{CONFIG['lr']:.0e}-{CONFIG['selekt_alpha']}-{CONFIG['decay']}-{CONFIG['timestamp']}"
if args.use_synth_data:
    run_name = "synth-" + run_name

# run = wandb.init(
#     project="instinct-sft",
#     name=run_name,
#     config=CONFIG,
# )
# wandb.define_metric("*", step_metric="train/epoch")

model_name = f"Qwen/Qwen2.5-Coder-{CONFIG['model_size']}"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# base_model, _ = get_model_and_tokenizer(model_name, torch.bfloat16)
# base_model.cpu()
# base_state_dict = base_model.state_dict()

# Move base state dict to CPU and ensure proper cleanup
# for key in base_state_dict:
#     base_state_dict[key] = base_state_dict[key].cpu()

# del base_model
for _ in range(3):
    gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


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
    
train_dataset_size = len(train_dataset)
effective_batch_size = CONFIG["batch_size"] * CONFIG["accumulation_steps"] * 8
steps_per_epoch = math.ceil(train_dataset_size / effective_batch_size)
total_training_steps = steps_per_epoch * CONFIG["epochs"]
print("TOTAL_TRAINING_STEPS:", total_training_steps)

if accelerator.is_main_process:
    print_with_separation(train_dataset[0]["text"])

save_dir = f"../ckpt/instinct-sft/qwen-{CONFIG['model_size']}-{CONFIG['lr']:.0e}-{CONFIG['selekt_alpha']}-{CONFIG['decay']}-{CONFIG['timestamp']}"
if args.use_synth_data:
    save_dir = f"../ckpt/instinct-sft/synth-qwen-{CONFIG['model_size']}-{CONFIG['lr']:.0e}-{CONFIG['selekt_alpha']}-{CONFIG['decay']}-{CONFIG['timestamp']}"

run_name = f"{CONFIG['model_size']}-{CONFIG['timestamp']}"


training_args = SFTConfig(
    output_dir=save_dir,
    run_name=f"{CONFIG['model_size']}-{CONFIG['timestamp']}",
    seed=3407,

    dataset_text_field="text",
    max_length=20000,
    dataset_num_proc=8,
    dataset_kwargs={"load_from_cache_file": False},
    deepspeed="../configs/ds_config.json",
    bf16=True,

    num_train_epochs=CONFIG["epochs"],
    gradient_accumulation_steps=CONFIG["accumulation_steps"],
    per_device_train_batch_size=CONFIG["batch_size"],
    learning_rate=CONFIG["lr"],
    lr_scheduler_type="cosine",
    warmup_steps=15,
    optim="adamw_8bit",
    weight_decay=CONFIG["decay"],

    logging_steps=1,
    report_to="wandb",
    do_eval=True,
    eval_strategy="steps",
    eval_steps=25,
    per_device_eval_batch_size=1,
    save_strategy="no",
    save_steps=None,
    gradient_checkpointing=True,
)

selekt_callback = Callback(
    base_model_state_dict=base_state_dict, 
    flush_steps=1, 
    alpha=CONFIG["selekt_alpha"],
    selekt_steps=25
)
# eval_model, eval_tokenizer = get_model_and_tokenizer("Qwen/Qwen2.5-Coder-32B-Instruct", torch.bfloat16)
# eval_model.eval()
eval_callback  = EvalCallback(         tokenizer, eval_dataset, run, accelerator, model, tokenizer, n_eval_samples=64)
synth_callback = SynthDataEvalCallback(tokenizer,               run, accelerator, model, tokenizer, n_eval_samples=64)

collator = MaskingCollator(tokenizer)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    data_collator=collator,
    callbacks = [selekt_callback, eval_callback, synth_callback],

)
selekt_callback.set_trainer(trainer)


start_gpu_memory, max_memory = show_memory_stats()
trainer_stats = trainer.train()
model.save_pretrained(save_dir)
show_final_memory_stats(start_gpu_memory, max_memory, trainer_stats)

run.finish()