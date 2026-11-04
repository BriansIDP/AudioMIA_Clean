#!/usr/bin/env python3
"""Minimal audio adaptation of VideoMIA for Qwen2.5-Omni.

Input JSON: [{"audio": "/path/to/file.wav", ...}, ...]. Other fields are kept.
Each trial asks (1) earliest candidate and (2) full temporal ordering.
The model sees chronological context clips and four individually labelled,
shuffled candidate clips. No logits or training QAs are used.

This is an AUDIO ADAPTATION of VideoMIA, not the authors' implementation.
"""
import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from peft import PeftModel

LETTERS = "ABCD"


def sample_trial(audio, sr, rng, segment_sec, n_context, gap_sec, poisson_sec):
    """Sample fixed-length clips with increasing candidate start times."""
    seg = round(segment_sec * sr)
    minimum_gap = round(gap_sec * sr)
    # Poisson increments in integer seconds, at least one segment apart.
    total = len(audio)
    for _ in range(100):
        increments = [seg + round(float(rng.poisson(poisson_sec)) * sr)
                      for _ in range(3)]
        span = n_context * seg + minimum_gap + sum(increments) + seg
        if span > total:
            continue
        start = int(rng.integers(0, total - span + 1))
        context = [audio[start + i*seg: start + (i+1)*seg]
                   for i in range(n_context)]
        first = start + n_context*seg + minimum_gap
        positions = [first]
        for inc in increments:
            positions.append(positions[-1] + inc)
        candidates = [audio[p:p+seg] for p in positions]  # chronological
        order = rng.permutation(4).tolist()  # label -> chronological index
        labeled = {LETTERS[i]: candidates[order[i]] for i in range(4)}
        true_order = [LETTERS[order.index(i)] for i in range(4)]
        return context, labeled, true_order, start/sr
    return None


def make_conversation(paths, task):
    context_paths, candidate_paths = paths
    content = [{"type": "text", "text":
                "The following context audio clips are consecutive and ordered chronologically. "
                "The four candidate clips are shuffled, independently presented excerpts from later in the "
                "SAME original recording. Use their content, not the order in this message."}]
    for i, path in enumerate(context_paths, 1):
        content.extend([{"type": "text", "text": f"Context {i}:"},
                        {"type": "audio", "audio": path}])
    for letter, path in candidate_paths.items():
        content.extend([{"type": "text", "text": f"Candidate {letter}:"},
                        {"type": "audio", "audio": path}])
    if task == "identify":
        instruction = "Which candidate occurs EARLIEST in the original recording after the context? Reply with exactly one letter: A, B, C, or D."
    else:
        instruction = "Put candidates A, B, C, D in their ORIGINAL chronological order, earliest to latest. Reply only like A,C,D,B."
    content.append({"type": "text", "text": instruction})
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {"role": "user", "content": content}
    ]


def query(model, processor, paths, task, max_new_tokens):
    conversation = make_conversation(paths, task)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, return_audio=False,
                       use_audio_in_video=True)
    # Match the user's existing inference pipeline.
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, use_audio_in_video=True, do_sample=False,
                             max_new_tokens=max_new_tokens)
    token_ids = out[0, inputs["input_ids"].shape[1]:]
    return processor.decode(token_ids, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False).strip()


def parse_identify(s):
    s = s.strip().upper()
    m = re.fullmatch(r"(?:CANDIDATE\s*)?([A-D])[.)]?", s)
    if m:
        return m.group(1)
    # Accept slightly verbose but unambiguous outputs.
    found = re.findall(r"\b[A-D]\b", s)
    return found[0] if len(found) == 1 else None


def parse_rank(s):
    found = re.findall(r"\b[A-D]\b", s.upper())
    return found if len(found) == 4 and set(found) == set(LETTERS) else None


def trial_scores(true_order, identity, rank):
    if identity is None:
        identify_score = 0.0
    else:
        j = true_order.index(identity) + 1
        identify_score = 2**(-(j-1)) if j < 4 else 0.0
    if rank is None:
        rank_score = 0.0
    else:
        pos = {c: i for i, c in enumerate(rank)}
        rank_score = sum(pos[true_order[i]] < pos[true_order[j]]
                         for i in range(4) for j in range(i+1, 4)) / 6.0
    return identify_score, rank_score


def load_audio(path, target_sr):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # mono
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return np.clip(wav, -1, 1).astype(np.float32)


def run_file(record, model, processor, args, rng):
    audio = load_audio(record["audio"], args.sample_rate)
    result = {"audio": record["audio"], "trials": [],
              "identify_score": None, "rank_score": None,
              "videomia_audio_score": None}
    for k in range(args.trials):
        trial = sample_trial(audio, args.sample_rate, rng,
                             args.segment_seconds, args.context_count,
                             args.gap_seconds, args.poisson_seconds)
        if trial is None:
            result["error"] = "Audio too short for configured trial sampling"
            break
        context, candidates, true_order, offset = trial
        # Temporary files let qwen_omni_utils process each audio independently.
        with tempfile.TemporaryDirectory(prefix="audio_mia_") as directory:
            cpaths = []
            for i, clip in enumerate(context):
                path = os.path.join(directory, f"context_{i}.wav")
                sf.write(path, clip, args.sample_rate)
                cpaths.append(path)
            paths = {}
            for letter, clip in candidates.items():
                path = os.path.join(directory, f"candidate_{letter}.wav")
                sf.write(path, clip, args.sample_rate)
                paths[letter] = path
            clips = (cpaths, paths)
            identify_text = query(model, processor, clips, "identify", args.max_new_tokens)
            rank_text = query(model, processor, clips, "rank", args.max_new_tokens)
        identity, rank = parse_identify(identify_text), parse_rank(rank_text)
        si, sr = trial_scores(true_order, identity, rank)
        result["trials"].append({"trial": k, "context_start_sec": offset,
                                  "ground_truth_order": true_order,
                                  "identify_raw": identify_text, "identify_parsed": identity,
                                  "rank_raw": rank_text, "rank_parsed": rank,
                                  "identify_score": si, "rank_score": sr})
    if result["trials"]:
        result["identify_score"] = float(np.mean([t["identify_score"] for t in result["trials"]]))
        result["rank_score"] = float(np.mean([t["rank_score"] for t in result["trials"]]))
        result["videomia_audio_score"] = 0.5 * (result["identify_score"] + result["rank_score"])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datapath", required=True)
    p.add_argument("--output_dir", default="./exp")
    p.add_argument("--model_path", default="Qwen/Qwen2.5-Omni-7B")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--lora_ckpt", default="no")
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--segment_seconds", type=float, default=2.0)
    p.add_argument("--context_count", type=int, default=3)
    p.add_argument("--gap_seconds", type=float, default=2.0)
    p.add_argument("--poisson_seconds", type=float, default=2.0)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.segment_seconds <= 0 or args.trials < 1 or args.context_count < 1 or args.gap_seconds < 0 or args.poisson_seconds < 0:
        p.error("Invalid segment/trial/sampling parameters")
    with open(args.datapath) as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        p.error("datapath must contain a JSON list of items with an 'audio' path")
    os.makedirs(args.output_dir, exist_ok=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto", cache_dir=args.cache_dir)
    if args.lora_ckpt != "no":
        model = PeftModel.from_pretrained(model, args.lora_ckpt)
        model = model.to(torch.bfloat16).merge_and_unload()
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path, cache_dir=args.cache_dir)
    rng = np.random.default_rng(args.seed)
    outfile = os.path.join(args.output_dir, "audio_videomia_qwen_{}.json".format(args.trials))
    results = []
    for record in tqdm(data):
        # Preserve existing metadata (including membership label) without using it.
        new_record = dict(record)
        # try:
        new_record["videomia"] = run_file(record, model, processor, args, rng)
        # except Exception as exc:
        #     new_record["videomia"] = {"audio": record.get("audio"), "error": repr(exc)}
        results.append(new_record)
        # Save incrementally so long HPC runs retain partial results.
        with open(outfile, "w") as fp:
            json.dump(results, fp, indent=2)
    print("Saved:", outfile)


if __name__ == "__main__":
    main()

