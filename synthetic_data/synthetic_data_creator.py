import sys

sys.path.append("../utils/")
sys.path.append("../eval/")
import datautils
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_model_and_tokenizer, print_with_separation
from evalutils import (
    get_excerpt,
    extract_content_between_flags,
    separate_input_output,
    generate_model_output,
)
from synthdataprompts import (
    DIFF_SYSTEM_PROMPT,
    DIFF_USER_PROMPT,
    EXCERPT_SYSTEM_PROMPT,
    EXCERPT_USER_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    REWRITE_USER_PROMPT,
    CONTEXTITEM_SYSTEM_PROMPT,
    CONTEXTITEM_USER_PROMPT,
)
import random
import os
from tqdm import tqdm


class SyntheticDataCreator:

    def __init__(self, translator_model_name: str, languages: list[str]):
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        self.model, self.tokenizer = get_model_and_tokenizer(translator_model_name)
        self.model = self.model.to("cuda")
        self.model.eval()
        # self.langs is used for diffs and context; language for excerpt and rewrite passed in separately
        self.langs = languages
        self.extension_map = {
            "Python": "py",
            "Java": "java",
            "C++": "cpp",
            "C": "c",
            "C#": "cs",
            "C++": "cpp",
            "Ruby": "rb",
            "Rust": "rs",
            "Go": "go",
            "Swift": "swift",
            "Kotlin": "kt",
            "Lua": "lua",
        }
        self.permitted_exts = [
            "ts",
            "js",
            "tsx",
            "jsx",
            "kt",
            "py",
            "sh",
            "swift",
            "java",
            "lua",
            "ps1",
        ]

    def translate_example(self, prompt: str, excerpt_languages: list[str]) -> str:
        """Deconstructs a prompt, changes up the languages, and constructs it back"""
        # Deconstruct
        prompt_preface = self.extract_prompt_preface(prompt)
        diffs, diff_filenames = self.extract_diffs_and_filenames(prompt)
        context_items, context_filenames = self.extract_context_items_and_filenames(
            prompt
        )
        excerpt, rewrite, filename = self.extract_excerpt_rewrite_and_filename(prompt)

        # Translate
        translated_diffs, diff_filenames = self.translate_diffs(diffs, diff_filenames)
        translated_context_items, context_filenames = self.translate_context_items(
            context_items, context_filenames
        )
        prompts = []
        for language in tqdm(
            excerpt_languages,
            desc="Translating excerpts in four languages",
            leave=False,
        ):
            translated_excerpt, translated_rewrite, excerpt_filename = (
                self.translate_excerpt_and_rewrite(excerpt, rewrite, filename, language)
            )

            # Reconstruct
            prompt = prompt_preface + "\n### User Edits:\n\n"
            for diff, filename in zip(translated_diffs, diff_filenames):
                prompt += f'User edited file "{filename}"\n\n{diff}\n\n'
            prompt += "### Context:\n"
            for context_item, filename in zip(
                translated_context_items, context_filenames
            ):
                prompt += (
                    f"<|context_file|> `{filename}`\n<|snippet|>\n{context_item}\n\n"
                )
            prompt += f'### User Excerpt:\n"{excerpt_filename}"\n\n{translated_excerpt}\n\n### Response:\n{translated_rewrite}'
            prompts.append(prompt)

        return prompts

    def extract_prompt_preface(self, prompt: str) -> str:
        return prompt.split("### User Edits:")[0].strip()

    def extract_diffs_and_filenames(self, prompt: str) -> tuple[list[str], list[str]]:
        diff_section = extract_content_between_flags(
            prompt, "### User Edits:", "### Context:"
        )
        split_diffs = diff_section.split("User edited file ")[1:]
        filenames = [diff.splitlines()[0][1:-1] for diff in split_diffs]
        filenames = self._remove_query_from_filenames(filenames)
        diffs = ["\n".join(diff.splitlines()[1:]).strip() for diff in split_diffs]
        return diffs, filenames

    def extract_context_items_and_filenames(
        self, prompt: str
    ) -> tuple[list[str], list[str]]:
        context_section = extract_content_between_flags(
            prompt, "### Context:", "### User Excerpt:"
        )
        context_chunks = context_section.split("<|context_file|>")[1:]
        filenames = [
            context_chunk.splitlines()[0].strip() for context_chunk in context_chunks
        ]
        filenames = self._remove_query_from_filenames(filenames)
        context_items = [
            "\n".join(context_chunk.splitlines()[2:])
            for context_chunk in context_chunks
        ]
        return context_items, filenames

    def extract_excerpt_rewrite_and_filename(self, prompt: str) -> tuple[str, str, str]:
        excerpt_section = extract_content_between_flags(
            prompt, "### User Excerpt:\n", "### Response:"
        )
        rewrite = prompt[prompt.find("### Response:\n") + len("### Response:\n") :]
        filename = excerpt_section.splitlines()[0][1:-1].strip()

        original_ext = self._get_file_ext(filename)
        if original_ext not in self.permitted_exts:
            raise ValueError(f"File extension {original_ext} not permitted")
        
        filename = self._remove_query_from_filenames([filename])[0]
        excerpt = "\n".join(excerpt_section.splitlines()[2:])
        return excerpt, rewrite, filename

    def translate_diffs(self, diffs: list[str], filenames: list[str]) -> list[str]:
        # translate filenames
        unique_filenames = list(set(filenames))
        new_langs = [random.choice(self.langs) for _ in unique_filenames]
        filename_map = {
            unique_filenames[i]: self._change_file_ext(
                unique_filenames[i], new_langs[i]
            )
            for i in range(len(unique_filenames))
        }
        new_filenames = [filename_map[filename] for filename in filenames]

        new_diffs = []
        for i in tqdm(range(len(diffs)), desc="Translating diffs...", leave=False):
            # translate diffs
            language = new_langs[unique_filenames.index(filenames[i])]
            system_prompt, user_prompt = DIFF_SYSTEM_PROMPT, DIFF_USER_PROMPT.format(
                language, diffs[i]
            )
            prompt = self._format_request(system_prompt, user_prompt)
            response = generate_model_output(prompt, self.model, self.tokenizer)
            # get content between triple backticks (as insurance)
            start = response.find("```")
            end = response.find("```", start + 3)
            if start != -1 and end != -1:
                response = response[start : end + 3]
            new_diffs.append(response.strip())

        return new_diffs, new_filenames

    def translate_context_items(
        self, context_items: list[str], filenames: list[str]
    ) -> list[str]:
        new_filenames = []
        new_context_items = []
        for i in range(len(context_items)):
            lang = random.choice(self.langs)
            new_filenames.append(self._change_file_ext(filenames[i], lang))
            system_prompt, user_prompt = (
                CONTEXTITEM_SYSTEM_PROMPT,
                CONTEXTITEM_USER_PROMPT.format(lang, context_items[i]),
            )
            prompt = self._format_request(system_prompt, user_prompt)
            response = generate_model_output(prompt, self.model, self.tokenizer)
            response = self._extract_lines_between_backticks(response)
            new_context_items.append(response)
        return new_context_items, new_filenames

    def translate_excerpt_and_rewrite(
        self, excerpt: str, rewrite: str, filename: str, language: str
    ) -> tuple[str, str, str]:
        # get translated filename and excerpt
        filename = self._change_file_ext(filename, language)
        system_prompt, user_prompt = EXCERPT_SYSTEM_PROMPT, EXCERPT_USER_PROMPT.format(
            language, excerpt
        )
        prompt = self._format_request(system_prompt, user_prompt)
        new_excerpt = generate_model_output(
            prompt, self.model, self.tokenizer, split_by_start_tag=False
        )
        # extract content between the first and second set of triple backticks (as insurance)
        new_excerpt = self._extract_lines_between_backticks(new_excerpt)

        # get translated rewrite
        system_prompt, user_prompt = REWRITE_SYSTEM_PROMPT, REWRITE_USER_PROMPT.format(
            language, excerpt, new_excerpt, rewrite
        )
        prompt = self._format_request(system_prompt, user_prompt)
        new_rewrite = generate_model_output(
            prompt, self.model, self.tokenizer, split_by_start_tag=False
        )
        new_rewrite = self._extract_lines_between_backticks(new_rewrite)
        return new_excerpt, new_rewrite, filename

    def _change_file_ext(self, filepath: str, language: str) -> str:
        return os.path.splitext(filepath)[0] + "." + self.extension_map[language]

    def _get_file_ext(self, filepath: str) -> str:
        return os.path.splitext(filepath)[1][1:]

    def _remove_query_from_filenames(self, filenames):
        return [
            filename[: filename.find("?")] if "?" in filename else filename
            for filename in filenames
        ]

    def _format_request(self, system_message, user_message):
        messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text

    def _extract_lines_between_backticks(self, text):
        response = text
        lines = response.splitlines()
        start_line = next((i for i, line in enumerate(lines) if "```" in line), -1)
        end_line = next(
            (i for i, line in enumerate(lines[start_line + 1 :]) if "```" in line), -1
        )
        if start_line != -1 and end_line != -1:
            response = "\n".join(
                lines[start_line + 1 : start_line + 1 + end_line]
            ).strip()
        else:
            response = response.strip()
        return response
