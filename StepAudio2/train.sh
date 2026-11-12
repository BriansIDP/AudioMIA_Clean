expdir=exp/step_audio2_sft_out_r128_newdata

mkdir -p $expdir

train_data=/path/to/same/train/data
valid_data=/path/to/same/valid/data

# torchrun --nproc_per_node=1 
python train.py \
    --model_name_or_path stepfun-ai/Step-Audio-2-mini \
    --dataset $train_data \
    --val_dataset $valid_data \
    --output_dir $expdir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --save_steps 1000 \
    --bf16 true \
    --gradient_checkpointing false
