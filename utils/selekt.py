# NOTE: This code is adapted from https://github.com/microsoft/NextCoder/blob/main/src/train/SeleKT/selekt.py

import os
import torch
import torch.distributed as dist
from transformers import TrainerCallback
from deepspeed.accelerator import get_accelerator
import deepspeed
from tqdm import tqdm
import gc


class Callback(TrainerCallback):
    def __init__(self, base_model_state_dict, flush_steps, alpha, selekt_steps=12):
        self.flush_steps = flush_steps
        self.trainer = None
        self.alpha = alpha
        self.selekt_steps = selekt_steps
        self.base_model_state_dict = base_model_state_dict

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, model, **kwargs):
        # memory cleanup
        if state.global_step % self.flush_steps == 0:
            get_accelerator().empty_cache()
            if dist.is_initialized():
                dist.barrier()

        # SeleKT application
        if state.global_step > 0 and state.global_step % self.selekt_steps == 0:
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            self._apply_selekt_in_place(local_rank)

    def _apply_selekt_in_place(self, local_rank):
        """Apply SeleKT directly to trainer model parameters"""

        if dist.is_initialized():
            dist.barrier()

        model = self.trainer.model_wrapped
        
        # Force no_grad context for entire operation
        with torch.no_grad():
            for name, param in tqdm(
                model.named_parameters(), desc="SeleKT", disable=(local_rank != 0), leave=False
            ):
                with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                    if local_rank == 0:
                        # Handle parameter name mapping
                        clean_name = name.replace("module.", "").replace("_orig_mod.", "")

                        if clean_name in self.base_model_state_dict:
                            base_param = self.base_model_state_dict[clean_name].to(
                                device=param.device, dtype=param.dtype, non_blocking=True
                            )

                            delta = param.data - base_param
                            
                            delta_abs = delta.abs()
                            delta_flat = delta_abs.view(-1)
                            _, indices = torch.topk(delta_flat, int(self.alpha * delta.numel()))
                            
                            mask = torch.zeros_like(delta_flat)
                            mask[indices] = 1
                            mask = mask.view_as(delta)

                            delta *= mask
                            param.data.copy_(base_param + delta)

                            del base_param, delta, delta_abs, delta_flat, mask, indices
                            
                if local_rank == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.barrier()

        for _ in range(3):
            gc.collect()
        
        # PyTorch CUDA memory management
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Force memory defragmentation
            torch.cuda.reset_peak_memory_stats()
            
        get_accelerator().empty_cache()
        
        # Additional CUDA cleanup if available
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
            
        # Clear any cached kernels
        if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
            torch.cuda.reset_accumulated_memory_stats()

        if dist.is_initialized():
            dist.barrier()