. /home/gs534/rds/hpc-work/work/anaconda3/etc/profile.d/conda.sh  && conda deactivate && conda activate audiomia
export CUDA_VISIBLE_DEVICES=0

traindata=data/newdata_train_capasr_QA.json
expdir=exp/salmonn2_capasr_QA_all

python train.py \
    --config configs/finetune.json \
    --data_path $traindata \
    --output_dir $expdir