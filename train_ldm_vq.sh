#!/bin/bash

#SBATCH --job-name=post_ldm_training
#SBATCH --mail-user=ashri@umich.edu ##MODIFY THIS LINE!
#SBATCH --mail-type=END,FAILED
#SBATCH --nodes=1
#SBATCH --account=hunseok0
#SBATCH --partition=spgpu
#SBATCH --gpus=2
#SBATCH --mem-per-gpu=48GB
#SBATCH --cpus-per-gpu=4
#SBATCH --time=00-16:00:00
#SBATCH --output=/home/%u/latent-diffusion/job_logs/training/%x-%j.log

module purge
module load cuda cudnn
module load python3.9-anaconda
conda init bash
source ~/.bashrc
conda activate ldm
cd /home/ashri/latent-diffusion

source prepare_lsun.sh
training_steps=1000
model_dir="weighted_loss_$training_steps"

if [ ! -d "$log_dir/$model_dir/" ]
then
    mkdir "$log_dir/$model_dir/"
fi

for snr in {20..0..-5}
do
    if [ ! -d "$log_dir/$model_dir/snr_$snr/" ]
        then
            mkdir "$log_dir/$model_dir/snr_$snr/"
    fi

    for l_weight in 0.01 0.1 1.0 10.0
    do
        python main_jscc.py --base configs/latent-diffusion/ldm_all_models_iterative.yaml --max_steps $training_steps \
            --train_txt_path "$data_root_dir/bedrooms_train.txt" --val_txt_path \
        "$data_root_dir/bedrooms_val.txt" --root_dir $data_dir --channel_snr $snr --gpus=2 \
        -t --logdir $log_dir/$model_dir/snr_${snr}_weight_${l_weight}/
    done
done
module unload python3.9-anaconda cuda cudnn
