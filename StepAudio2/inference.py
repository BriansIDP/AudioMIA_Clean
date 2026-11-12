import soundfile as sf
import librosa
import argparse
import sys, os
import json
import random
import time

from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
from transformers import StoppingCriteria, StoppingCriteriaList
import numpy as np
from mia_utils import get_meta_metrics, get_img_metric
from transformers import GenerationConfig
from stepaudio_utils import compute_token_num, load_audio, log_mel_spectrogram, padding_mels
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def compute_minkpp(log_probs, sudo_input_ids):
    mu_all = log_probs.mean(dim=-1)
    sigma_all = log_probs.std(dim=-1)
    log_probs_norm = (log_probs[torch.arange(sudo_input_ids.size(0)), sudo_input_ids] - mu_all) / sigma_all
    return_dict = {}
    for ratio in [0.05, 0.1, 0.3, 0.7, 0.9]:
        number_items = int(ratio * log_probs_norm.size(0))
        number_items = max(1, min(number_items, log_probs_norm.size(0)))
        mink_prob = (-log_probs_norm).topk(number_items)
        return_dict["min_k_pp_{:.2f}".format(ratio)] = - mink_prob.values.mean().item()
    return return_dict

# Custom stopping criteria
class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        last_tokens = input_ids[:, -1]
        return (last_tokens == self.stop_token_id).all()

stop_token_id = 151645
stopping_criteria = StoppingCriteriaList([StopOnToken(stop_token_id)])

random.seed(12345)
torch.manual_seed(12345)

def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y", "t"}


def change_speed(audios, speed_factor=1.0):
    newaudios = []
    speed_factors = []
    for audio in audios:
        # speed_factor = random.uniform(0.75, 1.5)
        stretched = librosa.effects.time_stretch(audio, rate=speed_factor)
        newaudios.append(stretched)
        speed_factors.append(speed_factor)
    return newaudios, speed_factors


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

def get_metrics(logits, text_ids_generated):
    probabilities = torch.nn.functional.softmax(torch.clamp(logits, -1e9, 1e9), dim=-1)
    log_probabilities = torch.nn.functional.log_softmax(torch.clamp(logits, -1e9, 1e9), dim=-1)
    pred = {}
    if "all" in args.metrics:
        metrics_dict = get_meta_metrics(text_ids_generated, probabilities, log_probabilities)
        pred = get_img_metric(
            metrics_dict["ppl"], metrics_dict["all_prob"], metrics_dict["loss"], metrics_dict["entropies"], 
            metrics_dict["modified_entropies"], metrics_dict["max_prob"], metrics_dict["probabilities"], 
            metrics_dict["gap_prob"], metrics_dict["renyi_05"], metrics_dict["renyi_2"], metrics_dict["log_probs"],
            metrics_dict["mod_renyi_05"], metrics_dict["mod_renyi_2"], metrics_dict["sequence_sm_entropy"], metrics_dict["sm_entropy"],
            metrics_dict["sequence_sm_entropy"], metrics_dict["sm_entropy"]
        )
    if "minkpp" in args.metrics:
        minkpp = compute_minkpp(log_probabilities, text_ids_generated)
        pred.update(minkpp)
    return pred


def reverse_audio_chunks(audio, chunk_size=50000):
    chunks = [
        audio[i:i + chunk_size]
        for i in range(0, len(audio), chunk_size)
    ]
    return np.concatenate(chunks[::-1])


def tokenize_prompt(messages, tokenizer):
    prompt_ids = []
    for msg in messages:
        if isinstance(msg, str):
            prompt_ids.append(tokenizer(text=msg, return_tensors="pt", padding=True)["input_ids"])
        elif isinstance(msg, list):
            prompt_ids.append(torch.tensor([msg], dtype=torch.int32))
        else:
            raise ValueError(f"Unsupported content type: {type(msg)}")
    prompt_ids = torch.cat(prompt_ids, dim=-1)
    attention_mask = torch.ones_like(prompt_ids)
    return prompt_ids, attention_mask


def apply_chat_template(messages: list, change="none", snr=20):
    results = []
    mels = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            role = "human"
        if isinstance(content, str):
            text_with_audio = f"<|BOT|>{role}\n{content}"
            text_with_audio += '<|EOT|>' if msg.get('eot', True) else ''
            results.append(text_with_audio)
        elif isinstance(content, list):
            results.append(f"<|BOT|>{role}\n")
            for item in content:
                if item["type"] == "text":
                    results.append(f"{item['text']}")
                elif item["type"] == "audio":
                    audio = load_audio(item['audio'])
                    if change == "reverse":
                        audio = reverse_audio_chunks(audio)
                    if change == "noise":
                        audio = add_noise(audio, snr_db=snr)
                    for i in range(0, audio.shape[0], 16000 * 25):
                        mel = log_mel_spectrogram(audio[i:i+16000*25], n_mels=128, padding=479)
                        mels.append(mel)
                        audio_tokens = "<audio_patch>" * compute_token_num(mel.shape[1])
                        results.append(f"<audio_start>{audio_tokens}<audio_end>")
                elif item["type"] == "token":
                    results.append(item["token"])
            if msg.get('eot', True):
                results.append('<|EOT|>')
        elif content is None:
            results.append(f"<|BOT|>{role}\n")
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
    return results, mels


class MyDataset(Dataset):
    def __init__(self, data, tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        # Returns the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Returns one sample and its label at the given index
        audiopath = self.data[idx]["audio"]
        prompt = self.data[idx]["question"]
        answer = self.data[idx]["answer"]
        conversation = [
            {
                "role": "system",
                "content": "You are an expert in audio analysis, please analyze the audio content and answer the questions accurately.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audiopath},
                    {"type": "text", "text": prompt}
                ],
            },
        ]
        prompt_only_text, mels = apply_chat_template(conversation, change=self.args.change, snr=self.args.snr_db)
        prompt_only_text.append("<|BOT|>assistant\n")
        if not self.args.bare_question:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            )
            fulltext, mels = apply_chat_template(conversation, change=self.args.change, snr=self.args.snr_db)
        if len(mels)==0:
            mels = None
            mel_lengths = None
        else:
            mels, mel_lengths = padding_mels(mels)

        if self.args.bare_question:
            input_ids, attention_mask = tokenize_prompt(prompt_only_text, self.tokenizer)
            generation_starts = -1
        else:
            input_ids, attention_mask = tokenize_prompt(fulltext, self.tokenizer)
            prompt_input_ids, _ = tokenize_prompt(prompt_only_text, self.tokenizer)
            generation_starts = prompt_input_ids.size(1)
        inputs = {
            "input_ids": input_ids,
            "wavs": mels,
            "wav_lens": mel_lengths,
            "attention_mask":attention_mask,
            "prompt": prompt,
            "audio": audiopath,
            "answer": answer,
            "label": self.data[idx]["label"],
        }
        if self.args.from_audio:
            prompt_only_text_reverse, mels_reverse = apply_chat_template(conversation, change="reverse")
            mels_reverse, mel_lengths_reverse = padding_mels(mels_reverse)
            inputs["reverse_mels"] = mels_reverse
        return inputs, generation_starts


def identity_collate(batch):
    return batch


def prediction_step(
    args,
    model,
    inputs,
    processor,
    change="none",
    return_dict_in_generate=False,
    generation_starts=0,
    generation_config=None,
):
    # Inference: Generation of the output text and audio
    logits = None
    if args.from_audio:
        with torch.no_grad():
            outputs = model(**inputs)
            audio_positions = torch.where(inputs["input_ids"] == 151690)[1]
            logits = outputs.logits[0, audio_positions]
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
            sudo_input_ids = logits.argmax(dim=-1)
            # Forward reversed features
            inputs["wavs"] = inputs["reverse_mels"]
            reverse_outputs = model(**inputs)
            reverse_logits = reverse_outputs.logits[0, audio_positions]
            reverse_probabilities = torch.nn.functional.softmax(reverse_logits, dim=-1)
            reverse_log_probabilities = torch.nn.functional.log_softmax(reverse_logits, dim=-1)
            revrese_sudo_input_ids = reverse_logits.argmax(dim=-1)

        pred = {}
        if "all" in args.metrics:
            metrics_dict = get_meta_metrics(sudo_input_ids, probabilities, log_probabilities)
            metrics_reverse = get_meta_metrics(revrese_sudo_input_ids, reverse_probabilities, reverse_log_probabilities)
            pred = get_img_metric(
                metrics_dict["ppl"], metrics_dict["all_prob"], metrics_dict["loss"], metrics_dict["entropies"], 
                metrics_dict["modified_entropies"], metrics_dict["max_prob"], metrics_dict["probabilities"], 
                metrics_dict["gap_prob"], metrics_dict["renyi_05"], metrics_dict["renyi_2"], metrics_dict["log_probs"],
                metrics_dict["mod_renyi_05"], metrics_dict["mod_renyi_2"], metrics_dict["sequence_sm_entropy"], metrics_dict["sm_entropy"],
                metrics_reverse["sequence_sm_entropy"], metrics_reverse["sm_entropy"]
            )
        if "minkpp" in args.metrics:
            minkpp = compute_minkpp(log_probabilities, sudo_input_ids)
            pred.update(minkpp)
    else:
        textsamples = []
        pred = {}
        with torch.no_grad():
            if not args.bare_question:
                outputs = model(**inputs)
                logits = outputs.logits[0, generation_starts-1:-1]
                text_ids_generated = inputs["input_ids"][0, generation_starts:]
                pred = get_metrics(logits, text_ids_generated)
            else:
                for i in range(args.n_samples):
                    outputs = model.generate(
                        **inputs,
                        generation_config=generation_config,
                        tokenizer=processor,
                        return_dict_in_generate=True if "puregen" not in args.othertag else False,
                        output_scores=True if "puregen" not in args.othertag else False,
                        use_cache=True,
                        max_new_tokens=512,
                        stopping_criteria=stopping_criteria,
                        do_sample=True if (args.n_samples > 1 and i > 0) else False,
                        temperature=1.2,
                        top_p=0.9,
                    )

                    if "puregen" not in args.othertag:
                        text_ids = outputs.sequences
                        text_ids_generated = text_ids[0, inputs["input_ids"].shape[1]:]
                        text = processor.decode(text_ids_generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        textsamples.append(text)
                        logits = torch.cat(outputs.scores, dim=0)
                        pred_dict = get_metrics(logits, text_ids_generated)
                        for key, value in pred_dict.items():
                            if key not in pred:
                                pred[key] = []
                            pred[key].append(float(value))
                    else:
                        text = processor.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        textsamples.append(text)
                pred["pred"] = textsamples
    # pred.update(additional_factors)
    return pred

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--datapath", type=str, default="./dataset")
    args.add_argument("--output_dir", type=str, default="./exp")
    args.add_argument("--bare_question", type=str2bool, default=True)
    args.add_argument("--return_logits", type=str2bool, default=False)
    args.add_argument("--lora_r", type=int, default=32)
    args.add_argument("--lora_alpha", type=int, default=64)
    args.add_argument("--lora_dropout", type=float, default=0.05)
    args.add_argument("--lora_ckpt", type=str, default="no")
    args.add_argument("--from_audio", type=str2bool, default=False)
    args.add_argument("--n_samples", type=int, default=1)
    args.add_argument("--metrics", type=str, default="all")
    args.add_argument("--change", type=str, default="none")
    args.add_argument("--speed_factor", type=float, default=1.0)
    args.add_argument("--snr_db", type=float, default=20.0)
    args.add_argument("--start", type=float, default=0.0)
    args.add_argument("--end", type=float, default=1.0)
    args.add_argument("--othertag", type=str, default="")

    args = args.parse_args()
    with open(args.datapath) as fin:
        data = json.load(fin)
    # start = int(args.start * len(data))
    # end = int(args.end * len(data))
    # data = data[start:end]

    letters = ["A", "B", "C", "D", "E"]

    MODEL_PATH = 'Step-Audio-2-mini'

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer.eos_token = "<|EOT|>"
    model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
    eos_token_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
    generation_config = dict(max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|EOT|>"),
    )
    # kwargs = {
    #     "max_new_tokens": 32,
    #     "num_beams": 1
    # }
    # generation_config.update(kwargs)
    generation_config = GenerationConfig(**generation_config)
    if args.lora_ckpt != "no":
        model = PeftModel.from_pretrained(model, args.lora_ckpt)
        model = model.to(torch.bfloat16)
        model = model.merge_and_unload()
        model.cuda().eval()
    else:
        model.eval().cuda()

    letter_ids = [tokenizer(l).input_ids[0] for l in letters]
    dataset = MyDataset(data, tokenizer, args)
    dataloader = DataLoader(
        dataset, 
        batch_size=1,   # Number of samples per pass
        shuffle=False,    # Mix data every epoch to prevent overfitting
        num_workers=8,    # Set >0 for multi-process loading (e.g., 2 or 4)
        collate_fn=identity_collate,
    )
    results = []

    for datapiece in tqdm(dataloader):
        inputs, generation_starts = datapiece[0]
        res_i = {
            "audio": inputs.pop("audio", None),
            "question": inputs.pop("prompt", None),
            "answer": inputs.pop("answer", None),
            "label": inputs.pop("label", None),
        }
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        pred = prediction_step(
            args,
            model,
            inputs,
            tokenizer,
            return_dict_in_generate=args.return_logits,
            change=args.change,
            generation_config=generation_config,
            generation_starts=generation_starts,
        )
        res_i["pred"] = pred
        if "answer" in res_i and "pred" in res_i["pred"]:
            print("REF:", res_i["answer"])
            print("PRED:", res_i["pred"]["pred"][0])
            print("="*89)
        results.append(res_i)
    tag = "no_generation" if args.from_audio else "generation"
    tag += "_{}_samples".format(args.n_samples)
    tag += "_{}".format(args.metrics)
    if "per_question" in args.datapath:
        tag += "_per_question"
    if "per_audio_all_question" in args.datapath:
        tag += "_per_audio_all_question"
    if "_alt" in args.datapath:
        tag += "_alt"
    if args.lora_ckpt == "no":
        tag += "_origmodel"
    if args.change != "none":
        tag += "_{}".format(args.change)
        if "speed" in args.change:
            tag += "_{}".format(args.speed_factor)
        if "noise" in args.change:
            tag += "_{}".format(args.snr_db)
    tag += "_{}".format(args.othertag)
    with open(os.path.join(args.output_dir, "mia_stepaudio_{}.json".format(tag)), 'w') as fp:
        json.dump(results, fp, indent=4)
