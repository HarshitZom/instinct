import torch
from transformers import TrainerCallback
from eval_suite import codebleu, llm_judge
from evalutils import (
    generate_model_output,
    extract_content_between_flags,
    get_excerpt,
    get_file_extension,
    multi_GPU_generate_model_output,
)
import sys

sys.path.append("../utils")
from utils import print_with_separation
from synthdatautils import get_synth_dataset
from datautils import TRAIN_PROMPT_PREFACE
import random
from tqdm import tqdm
import random


class SynthDataEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        run,
        accelerator,
        eval_llm=None,
        eval_tokenizer=None,
        n_eval_samples=32,
    ):
        self.tokenizer = tokenizer
        self.n_eval_samples = n_eval_samples
        self.run = run
        self.accelerator = accelerator
        self.eval_llm = eval_llm
        self.eval_tokenizer = eval_tokenizer
        self.py_dataset   = get_synth_dataset("Python", train=False)
        self.c_dataset    = get_synth_dataset("C",      train=False)
        self.java_dataset = get_synth_dataset("Java",   train=False)
        self.rust_dataset = get_synth_dataset("Rust",   train=False)

    def generate(self, model: str, example: dict, tokenizer=None) -> tuple[str, str, str]:
        """Multi-GPU-compatible inference function"""
        if tokenizer is None:
            tokenizer = self.tokenizer
        prompt, ground_truth = "", ""
        for message in example["messages"]:
            if   message["role"] == "user":      prompt       = message["content"]
            elif message["role"] == "assistant": ground_truth = message["content"]
        response = multi_GPU_generate_model_output(
            model, prompt, tokenizer, self.accelerator
        )
        return prompt, ground_truth, response

    def on_evaluate(self, args, state, control, model, **kwargs):
        """Called at the end of evaluation"""
        torch.cuda.empty_cache()
        model.eval()
        py_items, c_items, rs_items, java_items = self._get_random_synth_samples()
        py_results, c_results, rs_results, java_results = self._get_model_responses(
            model, py_items, c_items, rs_items, java_items
        )
        if py_results:
            self._evaluate_responses(py_results,   state.global_step, lang="python")
        if c_results:
            self._evaluate_responses(c_results,    state.global_step, lang="c")
        if rs_results:
            self._evaluate_responses(rs_results,   state.global_step, lang="rust")
        if java_results:
            self._evaluate_responses(java_results, state.global_step, lang="java")

        model.train()

    def _evaluate_responses(self, results: list[str], step: int, lang: str):
        """Custom evaluation logic on string prompt/response pairs"""
        self._log_text(results, step, lang)
        avg_gen_length = sum(len(r["generated_response"]) for r in results) / len(
            results
        )
        codebleu_scores = [
            self._get_codebleu_score(
                r["prompt"], r["ground_truth"], r["generated_response"]
            )
            for r in results
        ]
        avg_codebleu_score = sum(codebleu_scores) / len(codebleu_scores)

        llm_judge_scores = []
        for r in tqdm(results[:8], desc=f"Getting {lang} judge scores...", leave=False):
            score, _ = self._get_judge_score(
                r["prompt"], r["ground_truth"], r["generated_response"]
            )
            llm_judge_scores.append(score)
        avg_judge_score = sum(llm_judge_scores) / len(llm_judge_scores)


        self.run.log(
            {
                f"eval/{lang}-avg_generation_length": avg_gen_length,
                f"eval/{lang}-avg_codebleu_score": avg_codebleu_score,
                f"eval/{lang}-avg_judge_score": avg_judge_score,
            },
            commit=False,
            # step=self.run.step
        )

    def _get_codebleu_score(
        self, prompt: str, ground_truth: str, response: str
    ) -> float:
        """Calculate CodeBLEU score"""
        extension = get_file_extension(prompt)
        return codebleu.calculate_codebleu(ground_truth, response, extension)

    def _get_judge_score(self, prompt: str, ground_truth: str, response: str) -> float:
        """Calculate LLM judge score"""
        return 0.0, ""
        return llm_judge.get_llm_judge_score(
            prompt,
            response,
            ground_truth,
            self.accelerator,
            method="local",
            eval_model=self.eval_llm,
            eval_tokenizer=self.eval_tokenizer,
        )
    
    def _log_text(self, results: dict, step: int, lang: str):
        """Log text data in table form"""
        sample_table_data = []

        for i, result in enumerate(results[:10]):
            sample_table_data.append(
                [
                    i,
                    result["prompt"],
                    result["generated_response"],
                    result["ground_truth"],
                ]
            )

        if hasattr(self.run, "Table"):
            self.run.log(
                {
                    f"eval/{lang}-eval_samples": self.run.Table(
                        columns=["idx", "prompt_suffix", "generated", "ground_truth"],
                        data=sample_table_data,
                    )
                },
                commit=False,
                # step=self.run.step
            )

    def _get_random_synth_samples(self) -> tuple[list[str], list[str], list[str], list[str]]:
        """Gets random samples from the synthetic datasets"""
        indices = random.sample(range(len(self.py_dataset)), self.n_eval_samples)
        py_items   = self.py_dataset.select(indices)

        indices = random.sample(range(len(self.c_dataset)), self.n_eval_samples)
        c_items    = self.c_dataset.select(indices)

        indices = random.sample(range(len(self.java_dataset)), self.n_eval_samples)
        java_items = self.java_dataset.select(indices)

        indices = random.sample(range(len(self.rust_dataset)), self.n_eval_samples)
        rust_items = self.rust_dataset.select(indices)

        return py_items, c_items, java_items, rust_items

    def _get_model_responses(
        self, model, py_items: list[dict], c_items: list[dict], rs_items: list[dict], java_items: list[dict],
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        """Runs model inference on the synthetically translated examples"""
        py_results, c_results, rs_results, java_results = [], [], [], []

        with torch.no_grad():
            for example in tqdm(py_items, desc="Inferring Python examples...", leave=False):
                prompt, gt, response = self.generate(model, example)
                if prompt is not None:
                    py_results.append(
                        {
                            "prompt": prompt,
                            "generated_response": response,
                            "ground_truth": gt,
                        }
                    )
                    if self.accelerator.is_main_process:
                        print_with_separation(gt, response)
            for example in tqdm(c_items, desc="Inferring C examples...", leave=False):
                prompt, gt, response = self.generate(model, example)
                if prompt is not None:
                    c_results.append(
                        {
                            "prompt": prompt,
                            "generated_response": response,
                            "ground_truth": gt,
                        }
                    )
                    if self.accelerator.is_main_process:
                        print_with_separation(gt, response)
            for example in tqdm(rs_items, desc="Inferring Rust examples...", leave=False):
                prompt, gt, response = self.generate(model, example)
                if prompt is not None:
                    rs_results.append(
                        {
                            "prompt": prompt,
                            "generated_response": response,
                            "ground_truth": gt,
                        }
                    )
                    if self.accelerator.is_main_process:
                        print_with_separation(gt, response)
            for example in tqdm(java_items, desc="Inferring Java examples...", leave=False):
                prompt, gt, response = self.generate(model, example)
                if prompt is not None:
                    java_results.append(
                        {
                            "prompt": prompt,
                            "generated_response": response,
                            "ground_truth": gt,
                        }
                    )
                    if self.accelerator.is_main_process:
                        print_with_separation(gt, response)

        return py_results, c_results, rs_results, java_results

