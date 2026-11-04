#!/usr/bin/env python3
"""SALMONN-2 membership-inference evaluation, batch size 1.

Requires the SALMONN-2 `salmonn` Python package and the same `mia_utils.py`
used in the user's Qwen/StepAudio MIA evaluation scripts.
"""
import argparse
import json
from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor
from peft import PeftModel
import numpy as np

from salmonn import AudioProcessor, SalmonnForConditionalGeneration
from salmonn.training import SalmonnCollator
from mia_utils import get_meta_metrics, get_img_metric


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes", "y", "t")


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--datapath", required=True, help="JSON list with audio, question, answer, optional label")
    p.add_argument("--output_dir", default="./exp")
    p.add_argument("--model_path", required=True, help="Original SALMONN-2 base checkpoint")
    p.add_argument("--lora_ckpt", default="no", help="Directory with adapter_config.json; 'no' for base model")
    p.add_argument("--bare_question", type=str2bool, default=True, help="Generate or score reference answer")
    p.add_argument("--from_audio", type=str2bool, default=False, help="Score greedy pseudo tokens on audio-only perturbation")
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--metrics", default="all", help="all,minkpp or all,minkpp")
    p.add_argument("--change", default="none", choices=["none", "noise"])
    p.add_argument("--chunk_size", type=int, default=50000, help="Waveform samples per chunk for reverse perturbation")
    p.add_argument("--othertag", default="")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=1.0)
    p.add_argument("--noise_level", type=float, default=20.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_audio_paths(row):
    paths = row.get("audios", row.get("audio"))
    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, list) or not paths:
        raise ValueError("Each example must contain audio (string) or audios (nonempty list)")
    return paths


def make_sample(row):
    question = row["question"]
    answer = str(row["answer"])
    paths = get_audio_paths(row)
    # Exactly one placeholder per input audio; keep this aligned with SalmonnCollator.
    content = ("<audio>\n" * len(paths)) + question
    return {
        "audios": paths,
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": answer},
        ],
    }


def move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def reverse_audio_chunks(waveform, chunk_size):
    """Reverse the order of waveform chunks; preserve sample order in chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    pieces = list(waveform.split(chunk_size, dim=-1))
    return torch.cat(pieces[::-1], dim=-1) if pieces else waveform


def add_noise(audio, snr_db=20.0):
    signal_power = torch.mean(audio ** 2)
    if signal_power.item() != 0:
        noise_power = signal_power / (10 ** (snr_db / 10))
        # Generate noise with the same shape, device, and dtype
        noise = torch.randn_like(audio) * torch.sqrt(noise_power)
        noisy_audio = audio + noise
        # Clip to [-1, 1]
        noisy_audio = torch.clamp(noisy_audio, -1.0, 1.0)
    else:
        noisy_audio = audio
    return noisy_audio


class WaveformAudioProcessor(AudioProcessor):
    """SALMONN-2 AudioProcessor with optional pre-fbank waveform reversal.

    This overrides audio loading because the stock processor extracts mel
    features immediately after loading audio from disk.
    """
    def __init__(self, reverse=False, chunk_size=50000, change="none", noise_level=20):
        super().__init__()
        self.reverse = reverse
        self.chunk_size = chunk_size
        self.change = change
        self.noise_level = noise_level

    def __call__(self, path):
        waveform, sample_rate = torchaudio.load(str(Path(path).expanduser()))
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
        if "noise" in self.change:
            waveform = add_noise(waveform, snr_db=self.noise_level)
        if self.reverse:
            waveform = reverse_audio_chunks(waveform, self.chunk_size)
        # Use the same SALMONN-2 fbank as its original AudioProcessor.
        features = self.fbank.extract(
            waveform.squeeze(0), sampling_rate=self.sample_rate
        )
        return features.to(torch.float32)


def aligned_logits_labels(model, batch, from_audio=False):
    """Score only the assistant tokens after multimodal expansion."""
    # Produces labels with -100 at inserted acoustic-embedding positions.
    with torch.no_grad():
        prepared = model.prepare_multimodal_inputs(
            batch["input_ids"], batch["attention_mask"],
            batch["audio_features"], batch["audio_lengths"],
            batch["audio_counts"], batch["labels"],
        )
        audio_end = - batch["input_ids"][0][4:].size(-1)
        audio_start = 4
        expanded_labels = prepared[2]
        outputs = model(**batch)
        logits = outputs.logits

        if from_audio:
            selected_logits = logits[0, audio_start:audio_end]
            # pseudo labels
            target_ids = selected_logits.argmax(dim=-1)
        else:
            # A target at position t is predicted by logit t-1.
            targets = expanded_labels[:, 1:]
            previous_logits = logits[:, :-1, :]
            keep = targets.ne(-100)
            target_ids = targets[keep].long()
            selected_logits = previous_logits[keep].float()
    return selected_logits, target_ids


def minkpp(log_probs, token_ids):
    actual = log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    zscore = (actual - log_probs.mean(-1)) / log_probs.std(-1, unbiased=False).clamp_min(1e-6)
    result = {}
    for fraction in (0.05, 0.10, 0.30, 0.70, 0.90):
        k = max(1, min(len(zscore), int(len(zscore) * fraction)))
        result[f"min_k_pp_{fraction:.2f}"] = zscore.topk(k, largest=False).values.mean().item()
    return result


def compute_metrics(logits, token_ids, modes, comparison_logits=None):
    modes = set(modes.split(","))
    out = {}
    # Run these two compatible helpers exactly as in the previous MIA inference code.
    if "all" in modes:
        probabilities = F.softmax(logits, dim=-1)
        log_probabilities = F.log_softmax(logits, dim=-1)
        data = get_meta_metrics(token_ids, probabilities, log_probabilities)
        if comparison_logits is not None:
            cp = F.softmax(comparison_logits, dim=-1)
            clp = F.log_softmax(comparison_logits, dim=-1)
            cmp_data = get_meta_metrics(comparison_logits.argmax(-1), cp, clp)
            comparison_seq = cmp_data["sequence_sm_entropy"]
            comparison_tok = cmp_data["sm_entropy"]
        else:
            comparison_seq = data["sequence_sm_entropy"]
            comparison_tok = data["sm_entropy"]
        out.update(get_img_metric(
            data["ppl"], data["all_prob"], data["loss"], data["entropies"],
            data["modified_entropies"], data["max_prob"], data["probabilities"],
            data["gap_prob"], data["renyi_05"], data["renyi_2"], data["log_probs"],
            data["mod_renyi_05"], data["mod_renyi_2"],
            data["sequence_sm_entropy"], data["sm_entropy"],
            comparison_seq, comparison_tok,
        ))
    if "minkpp" in modes:
        out.update(minkpp(F.log_softmax(logits, dim=-1), token_ids))
    return out


def build_model(args):
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        fix_mistral_regex=False,
    )
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    model = SalmonnForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )
    if args.lora_ckpt != "no":
        model.base_llm = PeftModel.from_pretrained(
            model.base_llm, args.lora_ckpt, is_trainable=False
        )
        model.base_llm = model.base_llm.merge_and_unload()
    # if model.config.inject_temporal_embedding_nl:
    #     model.register_nl_timestamp_tokenizer(tokenizer)
    model = model.cuda().eval()
    processor.prepare_model(model)
    return model, processor


def get_prompt_batch(collator, tokenizer, sample, scored_batch, device):
    text = collator._render(sample["messages"][:-1], generation_prompt=True)
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return move_batch({
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "audio_features": scored_batch["audio_features"],
        "audio_lengths": scored_batch["audio_lengths"],
        "audio_counts": scored_batch["audio_counts"],
    }, device)


def generate_answers(model, processor, batch, n_samples, max_new_tokens, temperature=1.0):
    samples = []
    for i in range(n_samples):
        # With inputs_embeds, SALMONN-2 returns generated text IDs, not the expanded prompt.
        with torch.no_grad():
            output = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=i > 0,
                temperature=temperature if i > 0 else None,
                top_p=0.99 if i > 0 else None,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        text = processor.decode(output.sequences[0]).strip()
        samples.append(text)
        if i == 0:
            logits = torch.cat(output.scores, dim=0)
            generated_ids = output.sequences[0]
    return samples, logits, generated_ids


def main():
    args = args_parser()
    torch.manual_seed(args.seed)
    if args.n_samples < 1:
        raise ValueError("--n_samples must be >= 1")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")
    if not 0 <= args.start <= args.end <= 1:
        raise ValueError("Require 0 <= start <= end <= 1")
    with open(args.datapath, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list")
    data = data[int(len(data) * args.start):int(len(data) * args.end)]
    model, processor = build_model(args)
    tokenizer = processor.tokenizer
    collator = SalmonnCollator(
        tokenizer, WaveformAudioProcessor(
            reverse=False, chunk_size=args.chunk_size, change=args.change, noise_level=args.noise_level,
        )
    )
    if args.from_audio:
        comparison_collator = SalmonnCollator(
            tokenizer, WaveformAudioProcessor(
                reverse=True, chunk_size=args.chunk_size, change=args.change, noise_level=args.noise_level,
            )
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = ""
    if args.from_audio:
        tag += "fromaudio"
    if not args.bare_question:
        tag += "no_generation"
    tag += f"_{args.n_samples}_samples_{args.metrics.replace(',', '_')}"
    tag += "_origmodel" if args.lora_ckpt == "no" else "_lora"
    if "noise" in args.change:
        tag += f"_{args.change}_snr{args.noise_level}"
    if args.othertag:
        tag += f"_{args.othertag}"
    out_file = output_dir / f"mia_salmonn2_{tag}.json"
    results = []
    for row in tqdm(data):
        preds = []
        sample = make_sample(row)
        # Training-style collator constructs the same assistant-only label mask.
        batch = move_batch(collator([sample]), model.device)
        comparison_logits = None
        if args.from_audio:
            batch = move_batch(collator([sample]), model.device)
            logits, answer_ids = aligned_logits_labels(model, batch, from_audio=True)
            comparison_batch = move_batch(comparison_collator([sample]), model.device)
            comparison_logits, comparison_ids = aligned_logits_labels(model, comparison_batch, from_audio=True)
        else:
            if args.bare_question:
                prompt_batch = get_prompt_batch(collator, tokenizer, sample, batch, model.device)
                preds, logits, answer_ids = generate_answers(
                    model, processor, prompt_batch, args.n_samples, args.max_new_tokens, temperature=args.temperature
                )
                print(preds)
            else:
                batch = move_batch(collator([sample]), model.device)
                logits, answer_ids = aligned_logits_labels(model, batch)
        scores = compute_metrics(
            logits, answer_ids, args.metrics, comparison_logits=comparison_logits
        )
        import pdb; pdb.set_trace()
        scores["pred"] = preds
        results.append({
            "audio": get_audio_paths(row), "question": row["question"],
            "answer": row["answer"], "label": row.get("label"), "pred": scores,
        })

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} examples to {out_file}")


if __name__ == "__main__":
    main()
