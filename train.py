#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import random
from typing import Any, Dict, List, Optional

import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,  # not used, but handy for quick tests
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniThinkerForConditionalGeneration
from qwen_omni_utils import process_mm_info
from dataloader import ShadowDataset, PadToMultipleCollator
from trainer import QwenOmniTrainer



def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y", "t"}

# -----------------------------
# Tokenizer & model utilities
# -----------------------------
def load_tokenizer(args):
    processor = Qwen2_5OmniProcessor.from_pretrained(
        args.model_name_or_path,
    )
    return processor


LLM_LORA_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def get_llm_lora_targets(model):
    targets = []
    for name, module in model.named_modules():
        if not name.startswith("model.layers."):
            continue
        if not name.endswith(LLM_LORA_SUFFIXES):
            continue
        if isinstance(module, torch.nn.Linear):
            targets.append(name)
    if not targets:
        raise RuntimeError("No LLM LoRA targets found.")
    return targets


def default_lora_config(
    model,
    r=32,
    alpha=64,
    dropout=0.05,
):
    targets = get_llm_lora_targets(model)
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
        modules_to_save=None,
    )


def build_model(args):
    attn_impl = "flash_attention_2" if args.flash_attn else "sdpa"
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        device_map=None,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if not args.fulltune:
        # Freeze everything first.
        for p in model.parameters():
            p.requires_grad = False

        lora_cfg = default_lora_config(
            model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

        model = get_peft_model(model, lora_cfg)

        # Sanity check: only LoRA parameters should be trainable.
        for name, p in model.named_parameters():
            if p.requires_grad and "lora_" not in name:
                raise RuntimeError(
                    f"Unexpected trainable non-LoRA parameter: {name}"
                )

        model.print_trainable_parameters()

        print("\nActually adapted modules:")
        for name in model.targeted_module_names:
            print("  ", name)
    else:
        # Full-tune text backbone only, freeze multimodal encoders.
        for name, p in model.named_parameters():
            if (
                name.startswith("visual.")
                or name.startswith("audio_tower.")
            ):
                p.requires_grad = False
    return model


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-Omni-Instruct")

    # Data
    p.add_argument("--dataset", type=str, required=True, help="Local .json/.jsonl or HF dataset path")
    p.add_argument("--val_dataset", type=str, required=True, help="Local .json/.jsonl or HF dataset path")
    p.add_argument("--lora_ckpt", type=str, default="no")
    p.add_argument("--max_seq_length", type=int, default=4096)

    # Output / training
    p.add_argument("--output_dir", type=str, default="qwen25_omni_sft_custom")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_train_epochs", type=float, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--save_total_limit", type=int, default=5)
    p.add_argument("--bf16", type=str2bool, default=True)
    p.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    p.add_argument("--flash_attn", type=str2bool, default=False)

    # LoRA & quantization
    p.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--fulltune", type=str2bool, default=False)
    p.add_argument("--merge_lora_on_save", type=str2bool, default=False)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deepspeed", type=str, default=None)
    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args)
    model = build_model(args)

    # Build dataset
    train_ds = ShadowDataset(
        source=args.dataset,
        tokenizer=tokenizer,
    )
    eval_ds = ShadowDataset(
        source=args.val_dataset,
        tokenizer=tokenizer,
        split="valid"
    )

    collator = PadToMultipleCollator(tokenizer, pad_to_multiple_of=8)

    # Let HF Trainer construct DDP-aware DataLoaders; still our dataset + collator are used.
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=not args.bf16,
        dataloader_pin_memory=True,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        max_grad_norm=1.0,
        optim="adamw_torch",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    train_args.gradient_checkpointing_kwargs={"use_reentrant": False}
    train_args.do_validation = True

    trainer = QwenOmniTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    print("Training done. Saved in:", args.output_dir)


if __name__ == "__main__":
    main()
