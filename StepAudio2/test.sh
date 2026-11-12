expdir=exp/step_audio2_sft_out_r128_newdata
ckpt=$expdir/checkpoint-3780
# ckpt=no

metrics=minkpp_all
nsample=1

# evaldata=../data/newdata/testdata_capasr.json
evaldata="data/self_questions_capasr.json"

python inference_stepaudio.py \
        --datapath $evaldata \
        --return_logits true \
        --lora_ckpt $ckpt \
        --from_audio false \
        --output_dir $expdir \
        --n_samples $nsample \
        --bare_question true \
        --metrics $metrics \
        --othertag selfquestions \
