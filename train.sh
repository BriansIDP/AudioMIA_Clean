
# Activate the conda environment here

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
