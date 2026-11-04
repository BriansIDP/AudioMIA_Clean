# AudioTrace: File-Level Membership Detection for Audio Large Language Models
Significant privacy and copyright concerns are raised by the use of publicly available audio recordings to train audio large language models (LLMs), yet effective means of auditing such use remain limited. Existing membership inference attacks typically assume knowledge of complete training examples, including their associated textual supervision. In practice, however, audio recordings may be paired with privately generated transcripts, captions, or question-answer pairs that are unavailable to recording owners. This motivates audio-file-level membership inference, in which the inclusion of a recording in model training is inferred without knowledge of its associated annotations. This paper systematically investigates the challenges of audio-file-level membership inference, and proposes AudioTraceBench as a controlled benchmark comprising 1,285 audio recordings with explicit fine-tuning membership labels. Through evaluations across three audio LLMs, existing text- and video-based membership inference methods adapted to audio are shown to perform close to random guessing. To address this limitation, AudioTrace is proposed, where audio-conditioned proxy questions are generated using the target model itself, and membership evidence is extracted from its responses and aggregated at the recording level. Consistent improvements over existing baselines are observed, with AUROC reaching 0.60 and exceeding this level under increased fine-tuning exposure or adaptation capacity. Further analyses reveal that membership inference effectiveness is influenced by the similarity between proxy and training questions, acoustic integrity, and fine-tuning configurations. Through these findings, audio-file-level membership inference is established as a distinct and practically relevant auditing challenge, and an initial approach toward detecting the use of individual audio recordings in model training is provided.

### Preparation
```
conda create -n audiomia python=3.11
conda activate audiomia
pip install -r requirements.txt
```

### Downloading Data
The raw audio files will be released upon acceptance\
Store downloaded data under `data/`

### Finetune Qwen2.5-Omni
```
expdir=exp/qwen25omni_newdata_capasr_QA_all_linear
mkdir -p $expdir
train_data=data/newdata_train_capasr_QA.json
valid_data=data/valid_data.json

torchrun --nproc_per_node=1 --master_port=12345 train.py \
  --model_name_or_path Qwen/Qwen2.5-Omni-7B \
  --dataset $train_data \
  --val_dataset $valid_data \
  --output_dir $expdir \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_seq_length 4096 \
  --save_steps 1000 \
  --logging_steps 1 \
  --gradient_checkpointing true \
  --flash_attn false \
  --lora_r 128 \
  --lora_alpha 256 \
  --bf16 true \
```
Examine `train.sh` for various fine-tuning options.

### Membership Inference for Qwen2.5-Omni
```
# The expdir should be the dir of the finetuned model
expdir=exp/qwen25omni_newdata_capasr_QA_all_linear
ckpt=$expdir/checkpoint-2271
metrics=minkpp_all
nsample=1

# Run AudioTrace Inference
evaldata=data/self_questions_capasr.json
python inference.py \
    --datapath $evaldata \
    --return_logits true \
    --lora_ckpt $ckpt \
    --from_audio false \
    --output_dir $expdir \
    --n_samples $nsample \
    --metrics $metrics \
    --othertag selfquestion_capasr
```
To score the inference results:
```
python score.py \
  exp/qwen25omni_newdata_capasr_QA_all_linear/mia_qwen25omni_generation_1_samples_minkpp_all_selfquestion_capasr.json
```

Related configurable scoring parameters:
```
per_audio = True  # evaluate membership inference at per-sample or per-audio level
score_topk = 2  # The K value in top-k used to average
to_combine  # The list of metric names to combine as the final singal
```
