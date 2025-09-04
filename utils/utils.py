import os.path as osp
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
import math

def get_merged_from_adapter(
    adapter_path: str,
    max_seq_length: int = 16384,
    dtype=None,
    load_in_4bit: bool = False,
):
    """returns unsloth model and tokenizer"""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Merge the adapter weights into the model
    model = model.merge_and_unload()
    return model, tokenizer


def get_model_and_tokenizer(model_name: str, dtype: torch.dtype = torch.bfloat16):
    load_kwargs = dict(
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def token_count(text: str, tokenizer) -> int:
    tokenized_example = tokenizer(text, return_tensors="pt")
    return tokenized_example["input_ids"].size(1)


def show_memory_stats():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    return start_gpu_memory, max_memory


def show_final_memory_stats(start_gpu_memory, max_memory, trainer_stats):
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


def get_checkpoint_path(
    relative_to_ckpt_path, slug, rank, alpha, steps, lr, batchsize, beta=None
):
    ckpt_path = "/home/ubuntu/next-edit-model-training/ckpt"
    if beta is not None:
        filename = (
            slug
            + f"-rank{rank:05d}-alpha{alpha:05d}-steps{steps:05d}-lr{lr:05f}-bs{batchsize:03d}-beta{beta:03f}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
    else:
        filename = (
            slug
            + f"-rank{rank:05d}-alpha{alpha:05d}-steps{steps:05d}-lr{lr:05f}-bs{batchsize:03d}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )

    return osp.join(ckpt_path, relative_to_ckpt_path, filename)


def print_with_separation(*args):
    sep = "\r" + "=" * 80
    print(sep)
    for arg in args:
        print(arg, "\n", sep)
    print("\n")