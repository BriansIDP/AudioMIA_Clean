import soundfile as sf
import librosa
import argparse
import sys, os
import json
import random
from tqdm import tqdm
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from peft import PeftModel
from transformers import StoppingCriteria, StoppingCriteriaList
import numpy as np
from mia_utils import get_meta_metrics, get_img_metric



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

def reverse_audio_chunks(audio, chunk_size=50000):
    chunks = [
        audio[i:i + chunk_size]
        for i in range(0, len(audio), chunk_size)
    ]
    return np.concatenate(chunks[::-1])

# Custom stopping criteria
class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        last_tokens = input_ids[:, -1]
        return (last_tokens == self.stop_token_id).all()

stop_token_id = 151645
stopping_criteria = StoppingCriteriaList([StopOnToken(stop_token_id)])


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


def add_noise(audios, snr_db=20.0):
    newaudios = []
    noise_snr = []
    for audio in audios:
        # snr_db = random.uniform(10.0, 30.0)
        signal_power = np.mean(audio ** 2)
        if signal_power != 0:
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(audio.dtype)
            noisy_audio = audio + noise
            # Clip to [-1, 1] to avoid distortion
            noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
        else:
            noisy_audio = audio
        newaudios.append(noisy_audio)
        noise_snr.append(snr_db)
    return newaudios, noise_snr


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


def prediction_step(
    args,
    model,
    processor,
    audiopath,
    prompt,
    change="none",
    return_dict_in_generate=False,
    answer="",
    transcription="",
):
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audiopath},
                {"type": "text", "text": prompt}
            ],
        },
    ]
    if not args.bare_question:
        prompt_only_text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        conversation.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        )

    # Set whether to use audio in video
    USE_AUDIO_IN_VIDEO = True

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True if args.bare_question else False, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    if "completion" in args.metrics:
        # cutoff = audios[0].shape[0] // 3
        cutoff = int(16000 * 1)
        # random_start = random.randint(0, cutoff)
        audios[0] = audios[0][0:cutoff]

    # Further change audio
    additional_factors = {}
    if "speed" in change:
        audios, speed_factor = change_speed(audios, args.speed_factor)
        additional_factors["speed_factor"] = speed_factor
    if "noise" in change:
        audios, snr = add_noise(audios, args.snr_db)
        additional_factors["noise_snr"] = snr
    if "permute" in change:
        audios = [audios[0][::-1]]

    inputs = processor(text=text,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                    return_audio=False,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)
    if not args.bare_question:
        prompt_only_text = processor(text=prompt_only_text,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=True,
                    return_audio=False,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO)
        generation_starts = prompt_only_text["input_ids"].size(1)

    # Inference: Generation of the output text and audio
    logits = None
    if args.from_audio:
        with torch.no_grad():
            # Make reverse audio
            reverse_audios = [reverse_audio_chunks(audios[0])]
            reverse_inputs = processor(text=text,
                audio=reverse_audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                return_audio=False,
                use_audio_in_video=USE_AUDIO_IN_VIDEO)
            reverse_inputs = reverse_inputs.to(model.device).to(model.dtype)

            outputs = model(**inputs)
            audio_positions = torch.where(inputs["input_ids"] == 151646)[1]
            logits = outputs.logits[0, audio_positions]
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
            sudo_input_ids = logits.argmax(dim=-1)
            # Forward reversed features
            # feature_end = inputs["feature_attention_mask"].sum()
            # reverse_feature = inputs["input_features"][:, :, :feature_end]
            # reverse_feature = torch.flip(reverse_feature, dims=[2])
            # reverse_feature = torch.cat([reverse_feature, inputs["input_features"][:, :, feature_end:]], dim=-1)
            # inputs["input_features"] = reverse_feature
            reverse_outputs = model(**reverse_inputs)
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
        with torch.no_grad():
            if not args.bare_question:
                outputs = model(**inputs)
                logits = outputs.logits[0, generation_starts-1:-1]
                text_ids_generated = inputs["input_ids"][0, generation_starts:]
                pred = get_metrics(logits, text_ids_generated)
            else:
                nsamples = args.n_samples
                pred = {}
                for i in range(nsamples):
                    outputs = model.generate(
                        **inputs,
                        use_audio_in_video=USE_AUDIO_IN_VIDEO,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=1024,
                        stopping_criteria=stopping_criteria,
                        do_sample=True if (nsamples > 1 and i > 0) else False,
                        temperature=1.0,
                        top_p=0.9,
                    )
                    text_ids = outputs.sequences
                    text_ids_generated = text_ids[0, inputs["input_ids"].shape[1]:]
                    text = processor.decode(text_ids_generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    textsamples.append(text)
                    if i == 0:
                        logits = torch.cat(outputs.scores, dim=0)
                        pred = get_metrics(logits, text_ids_generated)
                pred["pred"] = textsamples
    pred.update(additional_factors)
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
    args.add_argument("--othertag", type=str, default="")

    args = args.parse_args()
    with open(args.datapath) as fin:
        data = json.load(fin)

    letters = ["A", "B", "C", "D", "E"]

    MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.lora_ckpt != "no":
        model = PeftModel.from_pretrained(model, args.lora_ckpt)
        model = model.to(torch.bfloat16)
        model = model.merge_and_unload()
        model.cuda()
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    letter_ids = [processor(l).input_ids[0][0] for l in letters]

    for datapiece in tqdm(data):
        pred = prediction_step(
            args,
            model,
            processor,
            datapiece["audio"],
            datapiece["question"],
            return_dict_in_generate=args.return_logits,
            change=args.change,
            answer=datapiece["answer"],
        )
        datapiece["pred"] = pred
        if "answer" in datapiece and not args.from_audio:
            print("QUESTION:", datapiece["question"])
            print("REF:", datapiece["answer"])
            if args.bare_question:
                print("PRED:", pred["pred"][0])
        print("="*89)
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
    with open(os.path.join(args.output_dir, "mia_qwen25omni_{}.json".format(tag)), 'w') as fp:
        json.dump(data, fp, indent=4)