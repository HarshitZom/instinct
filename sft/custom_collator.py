from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import torch
import sys
sys.path.append("../utils")
from utils import print_with_separation

class MaskingCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=False, **kwargs)
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"
        self.assistant = "assistant\n"
        self.eos_token = "<|endoftext|>"

    def torch_call(self, features):
        # Use parent class to handle padding and initial label creation
        batch = super().torch_call(features)
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        for i in range(len(features)):
            input_seq = input_ids[i]
            label_seq = labels[i]
            text = self.tokenizer.decode(input_seq, skip_special_tokens=False)
            # Split by <|im_start|> to process each message
            segments = text.split(self.im_start)
            assistant_content = None
            assistant_start_idx = None
            assistant_end_idx = None

            # Find the last assistant message
            for segment in reversed(segments):
                if self.assistant in segment and (self.im_end in segment or self.eos_token in segment):
                    # Extract content between assistant tag and <|im_end|> or <|endoftext|>
                    start = segment.find(self.assistant) + len(self.assistant)
                    if self.im_end in segment:
                        end = segment.find(self.im_end) + len(self.im_end)  # Include <|im_end|>
                    else:
                        end = segment.find(self.eos_token) + len(self.eos_token)  # Include <|endoftext|>
                    if end != -1:
                        assistant_content = segment[start:end].strip()
                        # Calculate token indices
                        full_text_up_to_segment = text[:text.rfind(segment) + start]
                        full_text_up_to_end = text[:text.rfind(segment) + end]
                        assistant_start_idx = len(self.tokenizer.encode(full_text_up_to_segment, add_special_tokens=False))
                        assistant_end_idx = len(self.tokenizer.encode(full_text_up_to_end, add_special_tokens=False))
                        break

            if assistant_content:
                label_seq[:assistant_start_idx] = -100
                label_seq[assistant_end_idx:] = -100
            else:
                label_seq[:] = -100

        batch["labels"] = labels
        return batch