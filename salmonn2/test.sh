. /home/gs534/rds/hpc-work/work/anaconda3/etc/profile.d/conda.sh  && conda deactivate && conda activate audiomia
export CUDA_VISIBLE_DEVICES=0

expdir="exp/salmonn2_capasr_QA_all"
lora_ckpt=$expdir/checkpoint-2500
# lora_ckpt=no


evaldata="data/self_questions_capasr.json"
python inference.py \
    --datapath $evaldata \
    --output_dir $expdir \
    --model_path marcoyang/salmonn-2-8b-test \
    --lora_ckpt $lora_ckpt \
    --bare_question true \
    --from_audio false \
    --n_samples 1 \
    --metrics all \
    --max_new_tokens 512 \
    --othertag selfquestion_capasr \
    # --change noise \
    # --noise_level 0 \

# Run Audio-SME
evaldata="../data/testdata_Qgen.json"
python inference.py \
    --datapath $evaldata \
    --output_dir $expdir \
    --model_path marcoyang/salmonn-2-8b-test \
    --lora_ckpt $lora_ckpt \
    --bare_question true \
    --from_audio true \
    --n_samples 1 \
    --metrics all \
    --max_new_tokens 128 \
    --othertag audioSME \

# Run VideoMIA-Audio
evaldata="../data/testdata_Qgen.json"
python inference_videomia.py \
    --datapath $evaldata \
    --output_dir $expdir \
    --lora_ckpt $lora_ckpt \
    --trials 15 \
    --segment_seconds 2 \
    --context_count 3 \
    --gap_seconds 2 \
    --poisson_seconds 2