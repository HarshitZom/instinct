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
import random
from tqdm import tqdm


class EvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        eval_dataset,
        run,
        accelerator,
        eval_llm=None,
        eval_tokenizer=None,
        n_eval_samples=32,
    ):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.n_eval_samples = n_eval_samples
        self.run = run
        self.accelerator = accelerator
        self.eval_llm = eval_llm
        self.eval_tokenizer = eval_tokenizer

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
        indices = random.sample(range(len(self.eval_dataset)), self.n_eval_samples)
        eval_samples = self.eval_dataset.select(indices)
        model.eval()
        results = []
        with torch.no_grad():
            for sample in tqdm(eval_samples, desc="Inferring eval samples...", leave=False):
                prompt, ground_truth, generated_response = self.generate(
                    model, sample
                )
                if prompt is not None:
                    results.append(
                        {
                            "prompt": prompt,
                            "generated_response": generated_response,
                            "ground_truth": ground_truth,
                        }
                    )
                    if self.accelerator.is_main_process:
                        print_with_separation(ground_truth, generated_response)
        if results:
            self._evaluate_responses(results, state.global_step)
        model.train()

    def _evaluate_responses(self, results: list[dict], step: int):
        """Custom evaluation logic on string prompt/response pairs"""
        self._log_text(results, step)
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
        for r in tqdm(results[:8], desc="Getting eval judge scores...", leave=False):
            score, _ = self._get_judge_score(
                r["prompt"], r["ground_truth"], r["generated_response"]
            )
            llm_judge_scores.append(score)
        avg_judge_score = sum(llm_judge_scores) / len(llm_judge_scores)

        self.run.log(
            {
                "eval/avg_generation_length": avg_gen_length,
                "eval/avg_codebleu_score": avg_codebleu_score,
                "eval/avg_judge_score": avg_judge_score,
            },
            commit=False,
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

    def _log_text(self, results: dict, step: int):
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
                    "eval/eval_samples": self.run.Table(
                        columns=["idx", "prompt_suffix", "generated", "ground_truth"],
                        data=sample_table_data,
                    )
                },
                commit=False,
            )
