import sys

sys.path.append("../utils/")
sys.path.append("../eval/")
from datautils import get_dataset
from utils import get_model_and_tokenizer, print_with_separation
from synthetic_data_creator import SyntheticDataCreator
from tqdm import tqdm
import os
import argparse
import numpy as np
from dotenv import load_dotenv
load_dotenv()

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

parser = argparse.ArgumentParser(description="GPU-sliced synthetic data generator")
parser.add_argument("--gpu", type=int, help="GPU to use")
args = parser.parse_args()

DATASET_OPTS = {
    "prompt_preface": "Predict a rewritten version of the user's excerpt given previous edits and some context",
    "diff_context_lines": 5,
    "excerpt_lines_around_cursor": 25,
    "cursor_line_jitter_stddev": 0,
    "cursor_char_jitter_stddev": 0,
    "render_cursor_pos": True,
    "editable_range_radius": 5,
    "editable_range_jitter_stddev": 0,
    "coeditor_excerpt_format": False,
}

synthdata_creator = SyntheticDataCreator(
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    languages=["Python", "C", "C++", "Java", "C#", "Ruby", "Rust", "Swift", "Kotlin", "Lua"],
)

for dataset_type in ["train", "test"]:
    dataset = get_dataset(DATASET_OPTS, version="v1.1", split=dataset_type)
    split_points = np.linspace(0, len(dataset), 9).astype(int)
    split_indices = np.arange(split_points[args.gpu], split_points[args.gpu + 1])
    for i in tqdm(split_indices, desc=f"Translating {dataset_type} dataset on GPU {args.gpu}..."):
        i = int(i)
        item = dataset[i]
        prompt, response = None, None
        for message in item["messages"]:
            if message["role"] == "user":
                prompt = message["content"]
            elif message['role'] == "assistant":
                response = message["content"]
        text = f"{prompt}\n### Response:\n{response}"
        try:
            prompts = synthdata_creator.translate_example(
                text, excerpt_languages=["Python", "C", "Rust", "Java"]
            )
        except ValueError as e:
            print(f"Error translating item {i}: {e}")
            continue

        os.makedirs(
            os.path.dirname(f"./v11_{dataset_type}_set/python/{i:04d}.txt"),
            exist_ok=True,
        )
        with open(f"./v11_{dataset_type}_set/python/{i:04d}.txt", "w") as f:
            f.write(prompts[0])

        os.makedirs(
            os.path.dirname(f"./v11_{dataset_type}_set/c/{i:04d}.txt"), exist_ok=True
        )
        with open(f"./v11_{dataset_type}_set/c/{i:04d}.txt", "w") as f:
            f.write(prompts[1])

        os.makedirs(
            os.path.dirname(f"./v11_{dataset_type}_set/rust/{i:04d}.txt"), exist_ok=True
        )
        with open(f"./v11_{dataset_type}_set/rust/{i:04d}.txt", "w") as f:
            f.write(prompts[2])

        os.makedirs(
            os.path.dirname(f"./v11_{dataset_type}_set/java/{i:04d}.txt"),
            exist_ok=True,
        )
        with open(f"./v11_{dataset_type}_set/java/{i:04d}.txt", "w") as f:
            f.write(prompts[3])
