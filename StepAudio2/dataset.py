import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from stepaudio_utils import load_audio, log_mel_spectrogram, compute_token_num, padding_mels

SYSTEM = 'You are an expert in audio analysis, please analyze the audio content and answer the questions accurately.'


def read_records(source):
    path = Path(source)
    if path.suffix == '.jsonl':
        with path.open(encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open(encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('Expected a JSON array of records')
    return data


class StepAudioDataset(Dataset):
    def __init__(self, source, tokenizer, max_seq_length=4096, system_prompt=SYSTEM):
        self.records = read_records(source)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.system_prompt = system_prompt
        self.audio_start_id = tokenizer.convert_tokens_to_ids('<audio_start>')
        self.audio_patch_id = tokenizer.convert_tokens_to_ids('<audio_patch>')
        self.audio_end_id = tokenizer.convert_tokens_to_ids('<audio_end>')
        self.eot_id = tokenizer.convert_tokens_to_ids('<|EOT|>')
        for name, value in [('audio_start', self.audio_start_id), ('audio_patch', self.audio_patch_id), ('audio_end', self.audio_end_id), ('EOT', self.eot_id)]:
            if value is None or value == tokenizer.unk_token_id:
                raise ValueError(f'Special token {name} is not registered')

    def __len__(self):
        return len(self.records)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __getitem__(self, idx):
        row = self.records[idx]
        audio_path = row['audio']
        question = str(row['question'])
        answer = str(row['answer'])

        # Identical 16 kHz / 25-second chunking and log-mel generation to inference.
        waveform = load_audio(audio_path)
        if waveform.numel() == 0:
            raise ValueError(f'Empty audio: {audio_path}')
        mels = []
        audio_ids = []
        for start in range(0, waveform.shape[0], 16000 * 25):
            mel = log_mel_spectrogram(
                waveform[start:start + 16000 * 25], n_mels=128, padding=479
            )
            mels.append(mel)
            patch_count = compute_token_num(mel.shape[1])
            audio_ids.extend([self.audio_start_id] + [self.audio_patch_id] * patch_count + [self.audio_end_id])

        # Same chat format as Step-Audio2 inference. The prompt ends at assistant\n.
        prompt_ids = (
            self.encode(f'<|BOT|>system\n{self.system_prompt}<|EOT|>')
            + self.encode('<|BOT|>human\n')
            + audio_ids
            + self.encode(question + '<|EOT|>')
            + self.encode('<|BOT|>assistant\n')
        )
        # Explicit EOT supervision so generation learns to stop.
        answer_ids = self.encode(answer) + [self.eot_id]
        if len(prompt_ids) + 1 > self.max_seq_length:
            raise ValueError(
                f'Example {idx} needs {len(prompt_ids)} prompt tokens, exceeding max_seq_length={self.max_seq_length}. '
                'Increase max_seq_length or shorten audio; do not truncate audio_patch tokens.'
            )
        answer_ids = answer_ids[:self.max_seq_length - len(prompt_ids)]
        if answer_ids[-1] != self.eot_id:
            answer_ids[-1] = self.eot_id
        input_ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.long)
        labels = torch.tensor([-100] * len(prompt_ids) + answer_ids, dtype=torch.long)
        wavs, wav_lens = padding_mels(mels)
        assert wavs.shape[0] == len(mels)
        return {'input_ids': input_ids, 'labels': labels, 'wavs': wavs, 'wav_lens': wav_lens}


class StepAudioCollator:
    """Batch size 1: remote Step-Audio2 forward maps each mel chunk to its audio_start token."""
    def __init__(self, pad_token_id, pad_to_multiple_of=8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples):
        if len(examples) != 1:
            raise ValueError('Set per_device_*_batch_size=1 (multi-chunk audio uses multiple wav rows).')
        ex = examples[0]
        n = ex['input_ids'].numel()
        pad = (-n) % self.pad_to_multiple_of
        return {
            'input_ids': torch.nn.functional.pad(ex['input_ids'], (0, pad), value=self.pad_token_id).unsqueeze(0),
            'attention_mask': torch.tensor([[1] * n + [0] * pad], dtype=torch.long),
            'labels': torch.nn.functional.pad(ex['labels'], (0, pad), value=-100).unsqueeze(0),
            'wavs': ex['wavs'],
            'wav_lens': ex['wav_lens'],
        }
