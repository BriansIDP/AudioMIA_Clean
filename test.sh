# The expdir should be the dir of the finetuned model
expdir=exp/qwen25omni_newdata_capasr_QA_all_linear

ckpt=$expdir/checkpoint-2271
# ckpt=no

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

# Run Audio-SME
evaldata=data/testdata_Qgen.json
python inference.py \
    --datapath $evaldata \
    --return_logits true \
    --lora_ckpt $ckpt \
    --from_audio false \
    --output_dir $expdir \
    --n_samples $nsample \
    --metrics $metrics \
    --othertag AudioSME

# Run VideoMIA-Audio
evaldata=data/testdata_Qgen.json
python inference_videomia.py \
    --datapath $evaldata \
    --output_dir $expdir \
    --lora_ckpt $ckpt \
    --trials 15 \
    --cache_dir /path/to/your/hf_home \
    --segment_seconds 2 \
    --context_count 3 \
    --gap_seconds 2 \
    --poisson_seconds 2