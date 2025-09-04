import anthropic
import openai
from eval_suite.codebleu import calculate_codebleu
from eval_suite.llm_judge import get_llm_judge_scores_batch
from eval_suite.keystroke_distance import get_keystroke_distance
import os
from tqdm import tqdm
import sys
import json
import numpy as np
from datetime import datetime
import re
import signal


sys.path.append("../../utils/")
from utils import print_with_separation, get_model_and_tokenizer
from datautils import get_dataset, TRAIN_PROMPT_PREFACE, get_unidiff
from synthdatautils import get_synth_dataset
from evalutils import generate_model_output
import dotenv
dotenv.load_dotenv()
import requests

sys.path.append("../")
from evalutils import (
    JUDGE_SYSTEM_PROMPT, 
    MERCURY_JUDGE_SYSTEM_PROMPT, 
    ZETA_JUDGE_SYSTEM_PROMPT, 
    JUDGE_USER_PROMPT,
    ZETA_FRONTMATTER,
    ZETA_ENDMATTER,
    get_file_extension, 
    extract_content_between_flags
)

class EvalRunner:
    def __init__(
        self,
        model: str,  # instinct, zeta, or mercury
        dataset_lang: str,
        dataset_version: str,
    ):
        self.DATASET_OPTS = {
            "prompt_preface": TRAIN_PROMPT_PREFACE,
            "diff_context_lines": 5,
            "excerpt_lines_around_cursor": 25,
            "cursor_line_jitter_stddev": 0,
            "cursor_char_jitter_stddev": 0,
            "render_cursor_pos": True,
            "editable_range_radius": 5,
            "editable_range_jitter_stddev": 0,
            "coeditor_excerpt_format": False,
        }

        self.model = model
        if self.model.upper() == "ZETA":
            self.zeta_model, self.zeta_tokenizer = get_model_and_tokenizer("zed-industries/zeta")
            self.zeta_model.to("cuda")

        if dataset_lang == "Typescript":
            self.dataset = get_dataset(self.DATASET_OPTS, dataset_version, split="test")
        else:
            self.dataset = get_synth_dataset(
                dataset_lang, train=False
            )
        
        print("MAPPING DATASET")
        self.dataset = self.dataset.map(lambda example: self._format_dataset(example))
        self.dataset_lang = dataset_lang
        self.generations = []
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        self.eval_results = {"codebleu": [], "judge_comments": [], "judge_scores": [], "devtime": []}
        print(f"Constructed EvalRunner for {self.model}")

    def _format_dataset(self, example):
        if   self.model.upper() == "INSTINCT": # fmt: skip
            result = self._format_dataset_for_instinct(example)
        elif self.model.upper() == "ZETA":
            result = self._format_dataset_for_zeta(example)
        elif self.model.upper() == "MERCURY":
            result =  self._format_dataset_for_mercury(example)
        return result

    def _format_dataset_for_instinct(self, example):
        messages = []
        gt = ""
        for message in example["messages"]:
            if message["role"] == "system":
                messages.append(message)
            if message["role"] == "user":
                messages.append(message)
            if message["role"] == "assistant":
                gt = message["content"]
        return {"prompt": messages, "gt": gt}

    def _format_dataset_for_zeta(self, example):
        user_prompt = ""
        gt = ""
        for msg in example["messages"]:
            if msg["role"] == "user":
                user_prompt = msg["content"]
            if msg["role"] == "assistant":
                gt = msg["content"]
        idx = user_prompt.rfind("### User Edits:\n")
        idx2 = user_prompt.find("### User Excerpt:")
        excerpt = user_prompt[idx2:]
        user_prompt = ZETA_FRONTMATTER + "\n" + user_prompt[idx:idx2]

        filename_match = excerpt.split('\n')[1].strip('"')
        
        start_marker = "<|editable_region_start|>"
        end_marker = "<|editable_region_end|>"
        start_idx = excerpt.index(start_marker)
        end_idx = excerpt.index(end_marker)
        lines = excerpt.split('\n')
        start_line = excerpt[:start_idx].count('\n')
        end_line = excerpt[:end_idx].count('\n')
        before_lines = lines[max(0, start_line-2):start_line]
        after_lines = lines[end_line+1:end_line+3]
        
        editable = excerpt[start_idx + len(start_marker)+1:end_idx-1]
        
        excerpt = f"""### User Excerpt:

```{filename_match}
{'\n'.join(before_lines)}
{start_marker}
{editable}
{end_marker}
{'\n'.join(after_lines)}
```"""
        user_prompt += excerpt + "\n\n" + ZETA_ENDMATTER

        return {"prompt": user_prompt, "gt": gt, "original_messages": example["messages"], "filename": str(filename_match)}

    def _extract_edits(self, diff_history: str) -> tuple[list[str], list[str]]:
        # Pattern to match filename and diff blocks
        pattern = r'User edited file "([^"]+)"\s*```diff\n(.*?)\n```'
        matches = re.findall(pattern, diff_history, re.DOTALL)
        filenames = [match[0] for match in matches]
        diffs = [match[1].strip() for match in matches]
        return filenames, diffs


    def _format_dataset_for_mercury(self, item):
        user_prompt = ""
        gt = ""
        for msg in item["messages"]:
            if msg["role"] == "user":
                user_prompt = msg["content"]
            if msg["role"] == "assistant":
                gt = msg["content"]
        context_idx = user_prompt.find("### Context:\n")
        idx = user_prompt.rfind("### User Edits:\n")
        idx2 = user_prompt.find("### User Excerpt:")
        context = user_prompt[context_idx + 13 : idx]
        context_items = context.split("<|context_file|> ")[1:]
        context_filenames = [
            context_item.split("<|snippet|>")[0][:-1] for context_item in context_items
        ]
        context_items = [
            context_item.split("<|snippet|>")[1][:-1] for context_item in context_items
        ]

        excerpt = user_prompt[idx2:]
        filename_match = excerpt.split('\n')[1].strip('"')
        excerpt = "\n".join(excerpt.splitlines()[1:])
        excerpt = "<|current_file_contents|>\n" + excerpt + "\n<|/current_file_contents|>"
        excerpt = excerpt.replace("<|editable_region_start|>", "<|code_to_edit|>")
        excerpt = excerpt.replace("<|editable_region_end|>", "<|/code_to_edit|>")

        diff_history = user_prompt[idx:idx2]
        diff_filenames, diffs = self._extract_edits(diff_history)

        user_prompt = "<|recently_viewed_code_snippets|>\n"

        for filename, snippet in zip(context_filenames, context_items):
            user_prompt += f"<|recently_viewed_code_snippet|>\ncode_snippet_file_path: {filename}\n{snippet}\n<|/recently_viewed_code_snippet|>\n"

        user_prompt += "<|/recently_viewed_code_snippets|>\n\n"

        user_prompt += excerpt + "\n\n"

        user_prompt += "<|edit_diff_history|>\n"
        for filename, diff in zip(diff_filenames, diffs):
            user_prompt += f"--- {filename}\tbefore\n+++ {filename}\tafter\n{diff}\n\n"
        user_prompt += "<|/edit_diff_history|>"

        return {"prompt": user_prompt, "gt": gt, "filename": str(filename_match)}

    def _generate_outputs(self):
        if self.model.upper() == "INSTINCT":

            client = openai.Client(base_url="http://localhost:30000/v1", api_key="dummy") # fmt: skip
            stop_token = "<|im_end|>"
            for example in tqdm(self.dataset, desc=f"Running inference on {self.model}..."):
                prompt = ""
                for message in example["prompt"]:
                    if message["role"] == "user": prompt = message["content"]
                
                response = client.chat.completions.create(
                    model="/root/.cache/huggingface",
                    messages=example["prompt"],
                    temperature=0.01,
                    max_tokens=128,
                    stop=[stop_token],
                )

                if response.choices[0].message.content:
                    self.generations.append(
                        {
                            "prompt":   prompt, # fmt: skip
                            "gt":       example["gt"], # fmt: skip
                            "response": response.choices[0].message.content,
                            # "filename": example["filename"],
                        }
                    )

        elif self.model.upper() == "ZETA":

            for example in tqdm(self.dataset, desc=f"Running inference on {self.model}..."):
                context_prompt = ""
                for message in example["original_messages"]:
                    if message["role"] == "user": context_prompt = message["content"]
                response = generate_model_output(example["prompt"], self.zeta_model, self.zeta_tokenizer, True)
                self.generations.append(
                        {
                            "prompt":   context_prompt, # fmt: skip
                            "gt":       example["gt"], # fmt: skip
                            "response": response,
                            # "filename": example["filename"],
                        }
                    )

        elif self.model.upper() == "MERCURY":

            client = openai.Client(base_url="https://api.inceptionlabs.ai/v1/edit/completions/", api_key=os.environ["INCEPTION_API_KEY"]) # fmt: skip
            stop_token = "<|/code_to_edit|>"
            for i, example in enumerate(self.dataset):
                # if i > 5:
                #     break
                prompt = example["prompt"]
                # print(example.keys())
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ['INCEPTION_API_KEY']}"
                }
                
                data = {
                    "model": "mercury-coder",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.01,
                    "max_tokens": 128,
                    "stop": ["<|/code_to_edit|>"]
                }
                
                response = requests.post(
                    "https://api.inceptionlabs.ai/v1/edit/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("choices") and result["choices"][0].get("message", {}).get("content"):
                        model_response = result["choices"][0]["message"]["content"]
                        model_response = "\n".join(model_response.splitlines()[1:-1])
                        self.generations.append({
                            "prompt": prompt,
                            "gt": example["gt"],
                            "response": model_response,
                            # "filename": example["filename"],
                        })
                else:
                    print(f"Error: {response.status_code} - {response.text}")
        
    def run_evals(self, evals: list[str], verbose=False) -> dict:
        """Runs selected evals on model"""
        self._generate_outputs()
        # here, iterate through and check gt, response diff format for judge

        for result in tqdm(self.generations, desc="Evaluating non-LLM metrics..."):
            if "codebleu" in evals:
                self._run_codebleu_eval(result)
            if "devtime" in evals:
                self._run_devtime_eval(result)
        
        claude_requests = []
        if "claude" in evals:
            claude_requests = self._prepare_claude_batch_requests()
            self._process_claude_batch_results(claude_requests)

        median_codebleu = np.median(self.eval_results["codebleu"])
        total_judge_score = 0.0
        for i in range(len(self.eval_results["judge_scores"])):
            total_judge_score += sum(self.eval_results["judge_scores"][i]) / len(self.eval_results["judge_scores"][i])
        avg_judge_score = total_judge_score / len(self.eval_results["judge_scores"]) if self.eval_results["judge_scores"] else 0.0
        avg_dev_time = sum(self.eval_results["devtime"]) / len(self.eval_results["devtime"])

        if verbose:
            print(f"Median CodeBLEU score: {median_codebleu}")
            print(f"Average Judge score: {avg_judge_score}")
            print(f"Average Dev time: {avg_dev_time}")
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        with open(f"./results/{self.model}-{self.dataset_lang}-{avg_judge_score:.5f}-{median_codebleu:.5f}-{timestamp}.json", "w") as f:
            json.dump(self.eval_results, f)
        return self.eval_results
    

    def _run_codebleu_eval(self, result: dict) -> float:
        """Runs CodeBLEU eval"""
        extension = get_file_extension(result["prompt"])
        score = calculate_codebleu(result["gt"], result["response"], extension)
        self.eval_results["codebleu"].append(score)


    def _run_devtime_eval(self, result: dict) -> float:
        """Runs keystroke distance eval"""
        prev = extract_content_between_flags(result["prompt"], "<|editable_region_start|>\n", "\n<|editable_region_end|>")
        prev = prev.replace("<|user_cursor_is_here|>", "")
        post = result["gt"]

        def timeout_handler(signum, frame):
            raise TimeoutError("Operation timed out")

        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(20)  # 20 second timeout
            devtime = get_keystroke_distance(prev, post)
            signal.alarm(0)  # Cancel the alarm
            self.eval_results["devtime"].append(devtime)
        except TimeoutError:
            print("TIMED OUT, SKIPPING")
            signal.alarm(0)
            pass

    def _prepare_claude_batch_requests(self) -> list:
        """Prepare all claude requests for batching"""
        batch_requests = []
        for i, result in enumerate(self.generations):
            # Create 3 requests per result
            for j in range(3):
                custom_id = f"judge_{i}_{j}"  # Unique identifier for each request
                batch_requests.append({
                    "custom_id": custom_id,
                    "result_index": i,
                    "iteration": j,
                    "prompt": result["prompt"],
                    "response": result["response"],
                    "gt": result["gt"]
                })
        return batch_requests
    
    def _process_claude_batch_results(self, requests: list):
        """Process batched claude results"""
        if self.model.upper() == "INSTINCT":
            batch_results = get_llm_judge_scores_batch(
                requests, 
                self.anthropic_client,
                JUDGE_SYSTEM_PROMPT,
                JUDGE_USER_PROMPT
            )
        elif self.model.upper() == "MERCURY":
            batch_results = get_llm_judge_scores_batch(
                requests,
                self.anthropic_client,
                MERCURY_JUDGE_SYSTEM_PROMPT,
                JUDGE_USER_PROMPT
            )
        elif self.model.upper() == "ZETA":
            batch_results = get_llm_judge_scores_batch(
                requests,
                self.anthropic_client,
                ZETA_JUDGE_SYSTEM_PROMPT,
                JUDGE_USER_PROMPT
            )

        if not batch_results:
            print("No batch results returned. Using default values.")
        
        # Initialize with defaults for ALL generations first
        num_generations = len(self.generations)
        judge_scores = [[0.0, 0.0, 0.0] for _ in range(num_generations)]
        judge_comments = [["Error", "Error", "Error"] for _ in range(num_generations)]
        
        # Fill in actual results
        for result in batch_results:
            idx = result["result_index"]
            iteration = result["iteration"]
            
            # Add bounds checking
            if 0 <= idx < num_generations and 0 <= iteration < 3:
                judge_scores[idx][iteration] = result["score"]
                judge_comments[idx][iteration] = result["text"]
            else:
                print(f"Warning: Invalid result indices - idx: {idx}, iteration: {iteration}")
        
        # Set the results (don't append!)
        self.eval_results["judge_scores"] = judge_scores
        self.eval_results["judge_comments"] = judge_comments

