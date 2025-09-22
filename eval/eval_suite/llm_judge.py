# function that takes prompt, generated response, ground truth, and local/claude, and returns a score
import sys

sys.path.append("../")
sys.path.append("../utils")
from evalutils import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT,
    multi_GPU_generate_model_output,
    extract_content_between_flags,
)
from utils.utils import print_with_separation
from utils.datautils import get_unidiff
import torch
import anthropic
import asyncio
from typing import List, Dict, Any
from tqdm import tqdm


def get_llm_judge_score(
    prompt: str,
    generated_response: str,
    ground_truth: str,
    accelerator = None,
    method: str = "local",
    eval_model=None,
    eval_tokenizer=None,
    anthropic_client=None,
    judge_system_prompt = JUDGE_SYSTEM_PROMPT,
    judge_user_prompt = JUDGE_USER_PROMPT
) -> tuple[float, str]:
    """Gets LLM-as-a-judge score for a predicted edit using local judge model"""
    user_prompt = judge_user_prompt.format(
        ground_truth, generated_response, prompt
    )
    return get_local_score(
        judge_system_prompt, user_prompt, accelerator, eval_model, eval_tokenizer
    )
    pass


def get_local_score(
    system_prompt: str, user_prompt: str, accelerator, eval_model, eval_tokenizer
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = eval_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    output = multi_GPU_generate_model_output(
        eval_model, text, eval_tokenizer, accelerator, max_new_tokens=2048
    )
    try:
        score = float(output[-3:])
    except:
        return 0.0, "Invalid output format from LLM judge"
    text = "\n".join(output.splitlines()[:-1])

    return score, text


def get_llm_judge_scores_batch(
    requests: List[Dict[str, Any]],
    anthropic_client,
    judge_system_prompt=JUDGE_SYSTEM_PROMPT,
    judge_user_prompt=JUDGE_USER_PROMPT,
) -> List[Dict[str, Any]]:
    """Process multiple judge requests using Anthropic's Message Batches API"""
    # Prepare batch requests in Anthropic's format
    batch_requests = []
    for req in requests:
        gt_diff, response_diff = _get_diffs(req)
        user_prompt = judge_user_prompt.format(
            gt_diff, response_diff, "\n".join(req["prompt"].splitlines()[1:])
        )

        batch_request = {
            "custom_id": req["custom_id"],
            "params": {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 1000,
                "temperature": 0,
                "system": judge_system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
        }
        batch_requests.append(batch_request)

    try:
        # Create the batch
        batch = anthropic_client.messages.batches.create(requests=batch_requests)

        # Wait for completion
        print(f"Created batch {batch.id}. Waiting for completion...")
        completed_batch = _wait_for_batch_completion(anthropic_client, batch.id)

        # Process results
        results = []
        for result in anthropic_client.messages.batches.results(completed_batch.id):
            processed_result = _process_single_batch_result(result, requests)
            if processed_result:
                results.append(processed_result)

        return results

    except Exception as e:
        print(f"Error in batch processing: {e}")
        return []


def _get_diffs(req):
    prev = extract_content_between_flags(
        req["prompt"], "<|editable_region_start|>", "<|editable_region_end|>"
    )
    gt_diff = get_unidiff(prev, req["gt"], 10000)
    response_diff = get_unidiff(prev, req["response"], 10000)
    return gt_diff, response_diff


def _wait_for_batch_completion(client, batch_id: str, max_wait_time: int = 7200):
    """Wait for batch to complete with polling"""
    import time

    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        batch = client.messages.batches.retrieve(batch_id)

        if batch.processing_status == "ended":
            return batch
        elif batch.processing_status in ["failed", "canceled"]:
            raise Exception(
                f"Batch {batch_id} failed with status: {batch.processing_status}"
            )

        print(f"Batch status: {batch.processing_status}. Waiting...")
        time.sleep(30)  # Check every 30 seconds

    raise Exception(f"Batch {batch_id} timed out after {max_wait_time} seconds")


def _process_single_batch_result(batch_result, original_requests):
    """Process a single result from the batch"""
    try:
        # Find the original request
        original_req = None
        for req in original_requests:
            if req["custom_id"] == batch_result.custom_id:
                original_req = req
                break

        if not original_req:
            return None

        if batch_result.result.type == "succeeded":
            output = batch_result.result.message.content[0].text.strip()

            try:
                score = float(output[-3:])
            except:
                print(f"ERROR PARSING BATCH RESULT:\n{output}")
                score = 0.0

            text = "\n".join(output.splitlines()[:-1])

            return {
                "custom_id": batch_result.custom_id,
                "result_index": original_req["result_index"],
                "iteration": original_req["iteration"],
                "score": score,
                "text": text,
            }
        else:
            print(
                f"Batch result failed for {batch_result.custom_id}: {batch_result.result.error}"
            )
            return {
                "custom_id": batch_result.custom_id,
                "result_index": original_req["result_index"],
                "iteration": original_req["iteration"],
                "score": 0.0,
                "text": "Batch request failed",
            }

    except Exception as e:
        print(f"Error processing batch result: {e}")
        return None
