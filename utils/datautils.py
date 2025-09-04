import dotenv

dotenv.load_dotenv()
from datasets import load_dataset
import difflib
import os
import re
import numpy as np
import json
import sys

sys.path.append("./eval/")
from evalutils import extract_content_between_flags

SYSTEM_PROMPT = """You are Instinct, an intelligent next-edit predictor developed by Continue.dev. Your role as an AI assistant is to help developers complete their code tasks by predicting the next edit that they will make within the section of code marked by <|editable_region_start|> and <|editable_region_end|> tags.

You have access to the following information to help you make informed suggestions:

- Context: In the section marked \"### Context\", there are context items from potentially relevant files in the developer's codebase. Each context item consists of a <|context_file|> marker, the filepath, a <|snippet|> marker, and then some content from that file, in that order. Keep in mind that not all of the context information may be relevant to the task, so use your judgement to determine which parts to consider.
- User Edits: In the section marked \"### User Edits:\", there is a record of the most recent changes made to the code, helping you understand the evolution of the code and the developer's intentions. These changes are listed from most recent to least recent. It's possible that some of the edit diff history is entirely irrelevant to the developer's change. The changes are provided in a unified line-diff format, i.e. with pluses and minuses for additions and deletions to the code.
- User Excerpt: In the section marked \"### User Excerpt:\", there is a filepath to the developer's current file, and then an excerpt from that file. The <|editable_region_start|> and <|editable_region_end|> markers are within this excerpt. Your job is to rewrite only this editable region, not the whole excerpt. The excerpt provides additional context on the surroundings of the developer's edit.
- Cursor Position: Within the user excerpt's editable region, the <|user_cursor_is_here|> flag indicates where the developer's cursor is currently located, which can be crucial for understanding what part of the code they are focusing on. Do not produce this marker in your output; simply take it into account.

Your task is to predict and complete the changes the developer would have made next in the editable region. The developer may have stopped in the middle of typing. Your goal is to keep the developer on the path that you think they're following. Some examples include further implementing a class, method, or variable, or improving the quality of the code. Make sure the developer doesn't get distracted by ensuring your suggestion is relevant. Consider what changes need to be made next, if any. If you think changes should be made, ask yourself if this is truly what needs to happen. If you are confident about it, then proceed with the changes.

# Steps

1. **Review Context**: Analyze the context from the resources provided, such as recently viewed snippets, edit history, surrounding code, and cursor location.
2. **Evaluate Current Code**: Determine if the current code within the tags requires any corrections or enhancements.
3. **Suggest Edits**: If changes are required, ensure they align with the developer's patterns and improve code quality.
4. **Maintain Consistency**: Ensure indentation and formatting follow the existing code style.

# Output Format

- Provide only the revised code within the tags. Do not include the tags in your output.
- Ensure that you do not output duplicate code that exists outside of these tags.
- Avoid undoing or reverting the developer's last change unless there are obvious typos or errors.

"""


TRAIN_PROMPT_PREFACE = """Reference the user excerpt, user edits, and the snippets to understand the developer's intent. Update the editable region of the user excerpt by predicting and completing the changes they would have made next. This may be a deletion, addition, or modification of code."""


def get_dataset(train=True):
    if train:
        split = "train_typescript"
    else:
        split = "test_typescript"
    full_dataset = load_dataset("continuedev/instinct-data")
    dataset = full_dataset[split]
    return dataset

def transform_item(item, dataset_opts):
    """Transform a single dataset item into the required format"""
    prompt, _, response = get_prompt_excerpt_and_rewrite(dataset_opts, item)
    prompt, response = mask_urls(prompt), mask_urls(response)
    prompt, response = mask_api_keys(prompt), mask_api_keys(response)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def mask_urls(text):
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    return url_pattern.sub("https://", text)


def mask_api_keys(text):
    api_key = os.environ["OLD_ANTHROPIC_KEY"]
    text = text.replace(api_key, "API_KEY")
    return text


def get_prompt_excerpt_and_rewrite(
    dataset_opts: dict,
    data,
) -> tuple[str, str, str]:
    """Top-level function to render the input prompt, input user excerpt, and target output response"""

    # get previous edit data
    filenames, diffs = [], []
    for previous_edit in data["previous_edits"]:
        filenames.append(previous_edit["filename"])
        diffs.append(
            reduce_diff_context_lines(
                previous_edit["diff"], dataset_opts["diff_context_lines"]
            )
        )

    # get context
    context = parse_context(data["context"])

    # trim excerpt and rewrite
    excerpt, rewrite, cursor_line = trim_excerpt_and_rewrite(
        data["large_excerpt"],
        data["large_rewrite"],
        data["cursor_pos_line"],
        dataset_opts["excerpt_lines_around_cursor"],
    )
    # get cursor position
    cursor_line, cursor_char = get_cursor_position(
        excerpt,
        cursor_line,
        data["cursor_pos_char"],
        dataset_opts["cursor_line_jitter_stddev"],
        dataset_opts["cursor_char_jitter_stddev"],
    )
    # get editable range
    editable_startline, editable_endline = get_editable_region(
        excerpt,
        cursor_line,
        dataset_opts["editable_range_radius"],
        dataset_opts["editable_range_jitter_stddev"],
    )

    # format excerpt and response in either coeditor or normal format
    if dataset_opts["coeditor_excerpt_format"]:
        rendered_excerpt, rendered_response = render_coeditor_excerpt_and_response(
            excerpt, rewrite, editable_startline, editable_endline
        )
    else:
        delta_n_lines = len(rewrite.splitlines()) - len(excerpt.splitlines())
        response_lines = rewrite.splitlines()[
            editable_startline : editable_endline + delta_n_lines
        ]
        rendered_response = "\n".join(response_lines)
        rendered_excerpt = render_excerpt(
            excerpt,
            cursor_line,
            cursor_char,
            editable_startline,
            editable_endline,
            dataset_opts["render_cursor_pos"],
        )

    # put it all together
    prompt = dataset_opts["prompt_preface"]
    prompt += f"\n\n### Context:\n\n{context}\n\n"

    prompt += "\n### User Edits:\n\n"
    for i in range(len(filenames)):
        prompt += f'User edited file "{filenames[i]}"\n\n'
        prompt += "```diff\n" + diffs[i] + "\n```\n\n"

    prompt += f"### User Excerpt:\n"
    prompt += '"' + data["filename"] + '"\n\n'
    prompt += rendered_excerpt

    prompt = truncate_prompt_context(prompt)

    # returns prompt, excerpt, and response; excerpt is included for convenience
    return prompt, rendered_excerpt, rendered_response


def render_excerpt(
    excerpt: str,
    cursor_line: int,
    cursor_char: int,
    editable_startline: int,
    editable_endline: int,
    render_cursor: bool,
) -> str:
    """Helper to render the excerpt with cursor and editable region tags"""

    lines = excerpt.splitlines()
    before_lines = (
        lines[:editable_startline]
        + ["<|editable_region_start|>"]
        + lines[editable_startline:cursor_line]
    )

    if render_cursor:
        cursor_tag_line = (
            lines[cursor_line][:cursor_char]
            + "<|user_cursor_is_here|>"
            + lines[cursor_line][cursor_char:]
        )
    else:
        cursor_tag_line = lines[cursor_line]

    after_lines = (
        lines[cursor_line + 1 : editable_endline]
        + ["<|editable_region_end|>"]
        + lines[editable_endline:]
    )
    return "\n".join(before_lines + [cursor_tag_line] + after_lines)


def reduce_diff_context_lines(diff_text: str, context_lines: int = 3) -> str:
    """Reduces the number of context lines in a diff to the desired amount,
    updating the header as necessary"""

    lines = diff_text.splitlines()
    if not lines:
        return diff_text
    header, content = lines[0], lines[1:]
    match = re.match(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@(.*)", header)
    if not match:
        return diff_text
    old_start, new_start = int(match.group(1)), int(match.group(3))
    header_desc = match.group(5)
    change_indices = [
        i for i, line in enumerate(content) if line.startswith(("+", "-"))
    ]
    if not change_indices:
        return diff_text
    min_change_idx, max_change_idx = min(change_indices), max(change_indices)
    start = max(0, min_change_idx - context_lines)
    end = min(len(content), max_change_idx + context_lines + 1)
    result_content = content[start:end]

    old_lines = sum(1 for l in result_content if not l.startswith("+"))
    new_lines = sum(1 for l in result_content if not l.startswith("-"))
    old_offset = sum(1 for l in content[:start] if not l.startswith("+"))
    new_offset = sum(1 for l in content[:start] if not l.startswith("-"))
    new_header = f"@@ -{old_start + old_offset},{old_lines} +{new_start + new_offset},{new_lines} @@{header_desc}"

    content = "\n".join(result_content)
    content = content.replace("<|editable_region_start|>", "")
    content = content.replace("<|editable_region_end|>", "")
    content = content.replace("<|user_cursor_is_here|>", "")

    return "\n".join([new_header] + [content])


def parse_context(context: str) -> str:
    """Parse Codestral-style context (with +++++ markers) into filesep tags"""

    if not context.strip():
        return context

    lines = context.split("\n")
    result = []
    current_content = []

    for line in lines:
        if line.startswith("+++++"):
            # If we have accumulated content, close the previous snippet
            if current_content:
                result.append("<|snippet|>")
                # Remove trailing empty lines from current_content
                while current_content and current_content[-1] == "":
                    current_content.pop()
                result.extend(current_content)
                result.append("\n")
                current_content = []

            # Start new file section
            filename = line[5:].strip()  # Remove '+++++' prefix
            result.append(f"<|context_file|> {filename}")
        else:
            # Accumulate content lines
            current_content.append(line)

    # Handle the last file's content
    if current_content:
        result.append("<|snippet|>")
        # Remove trailing empty lines before adding end tag
        while current_content and current_content[-1] == "":
            current_content.pop()
        result.extend(current_content)

    context = "\n".join(result)

    context = context.replace("<|editable_region_start|>", "")
    context = context.replace("<|editable_region_end|>", "")
    context = context.replace("<|user_cursor_is_here|>", "")
    return context


def truncate_prompt_context(prompt, max_tokens=20000):
    """
    Truncates context so that prompt is under max_tokens
    """
    if len(prompt) // 4 <= max_tokens:
        return prompt

    context_start = prompt.find("### Context:\n")
    context_end = prompt.find("### User Edits:\n")
    if context_start == -1 or context_end == -1:
        return prompt

    before_context = prompt[: context_start + len("### Context:\n")]
    after_context = prompt[context_end:]
    context_section = prompt[context_start + len("### Context:\n") : context_end]
    non_context_tokens = len(before_context) // 4 + len(after_context) // 4
    available_context_tokens = max_tokens - non_context_tokens

    if available_context_tokens <= 0:
        return before_context + after_context

    context_items = re.split(r"<\|context_file\|>", context_section)
    context_items = [item.strip() for item in context_items if item.strip()]
    selected_items = []
    used_tokens = 0

    for item in context_items:
        item_tokens = len(item) // 4
        if used_tokens + item_tokens <= available_context_tokens:
            selected_items.append(item)
            used_tokens += item_tokens

    if selected_items:
        new_context = "<|context_file|>" + "\n\n<|context_file|> ".join(selected_items)
        return before_context + new_context + "\n\n" + after_context
    else:
        return before_context + after_context










def get_unidiff(before: str, after: str, n_context_lines: int = 3) -> str:
    """Helper to run a diff (returns only content, without header)"""

    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(), lineterm="", n=n_context_lines
        )
    )
    diff = "\n".join(diff_lines[3:])
    diff = diff.replace("<|editable_region_start|>", "")
    diff = diff.replace("<|editable_region_end|>", "")
    diff = diff.replace("<|user_cursor_is_here|>", "")
    return diff



