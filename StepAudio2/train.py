#!/usr/bin/env python3
import argparse
import inspect
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftModel
from dataset import StepAudioDataset, StepAudioCollator


def str2bool(x):
    return str(x).lower() in {'1', 'true', 'yes', 'y', 't'}


class StepAudioTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        # The remote Step-Audio2 forward exposes logits, not a labels/loss interface.
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.float().reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1), ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


def lora_targets(model):
    suffixes = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
    matches = [name for name, layer in model.named_modules()
               if name.startswith('model.layers.') and name.endswith(suffixes)
               and isinstance(layer, torch.nn.Linear)]
    if not matches:
        raise RuntimeError('No LLM Linear layers found for LoRA; inspect model.named_modules()')
    return matches


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name_or_path', default='stepfun-ai/Step-Audio-2-mini')
    p.add_argument('--dataset', required=True)
    p.add_argument('--val_dataset', required=True)
    p.add_argument('--output_dir', default='stepaudio2_sft')
    p.add_argument('--max_seq_length', type=int, default=4096)
    p.add_argument('--per_device_train_batch_size', type=int, default=1)
    p.add_argument('--gradient_accumulation_steps', type=int, default=2)
    p.add_argument('--learning_rate', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--num_train_epochs', type=float, default=10)
    p.add_argument('--lr_scheduler_type', default='cosine')
    p.add_argument('--warmup_ratio', type=float, default=0.03)
    p.add_argument('--logging_steps', type=int, default=2)
    p.add_argument('--save_steps', type=int, default=500)
    p.add_argument('--save_total_limit', type=int, default=5)
    p.add_argument('--bf16', type=str2bool, default=True)
    p.add_argument('--gradient_checkpointing', type=str2bool, default=False)
    p.add_argument('--lora_r', type=int, default=32)
    p.add_argument('--lora_alpha', type=int, default=64)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--fulltune', type=str2bool, default=False)
    p.add_argument('--lora_ckpt', default='no')
    p.add_argument('--deepspeed', default=None)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.per_device_train_batch_size != 1:
        raise ValueError('StepAudio2 custom forward currently requires batch size 1 for multi-chunk audio')

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, padding_side='right'
    )
    tokenizer.eos_token = '<|EOT|>'
    eot = tokenizer.convert_tokens_to_ids('<|EOT|>')
    if eot is None or eot == tokenizer.unk_token_id:
        raise ValueError('Missing <|EOT|> token')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map=None,
    )
    model.config.eos_token_id = eot
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # Preserve runtime generation behavior after saving; do NOT apply generation during SFT.
    model.generation_config = GenerationConfig(
        eos_token_id=eot, pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=32, num_beams=1, do_sample=False,
    )

    if args.gradient_checkpointing:
        # Delegate to native implementation; test one backward pass for this remote-code version.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()

    if args.fulltune:
        # Full tuning of the LLM, with audio encoder and adaptor frozen.
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith('model.') or name.startswith('lm_head.')
    else:
        for param in model.parameters():
            param.requires_grad = False
        targets = lora_targets(model)
        # task_type omitted: generic PeftModel does not inject 'labels' into custom forward.
        if args.lora_ckpt != 'no':
            model = PeftModel.from_pretrained(
                model, args.lora_ckpt, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout, bias='none',
                target_modules=targets,
            ))
        model.print_trainable_parameters()

    train_ds = StepAudioDataset(args.dataset, tokenizer, args.max_seq_length)
    eval_ds = StepAudioDataset(args.val_dataset, tokenizer, args.max_seq_length)
    collator = StepAudioCollator(tokenizer.pad_token_id)

    kw = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
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
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False} if args.gradient_checkpointing else None,
        deepspeed=args.deepspeed,
        max_grad_norm=1.0,
        optim='adamw_torch',
        remove_unused_columns=False,
        dataloader_num_workers=0,
        eval_strategy='steps',
        eval_steps=args.save_steps,
        report_to='none',
        seed=args.seed,
    )
    # compatibility with versions that use evaluation_strategy
    if 'eval_strategy' not in inspect.signature(TrainingArguments).parameters:
        kw['evaluation_strategy'] = kw.pop('eval_strategy')
    training_args = TrainingArguments(**kw)
    trainer = StepAudioTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    model.generation_config.save_pretrained(args.output_dir)
    print('Saved to', args.output_dir)


if __name__ == '__main__':
    main()
