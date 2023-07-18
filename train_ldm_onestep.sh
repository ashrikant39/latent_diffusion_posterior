#!/bin/bash

#SBATCH --job-name=ldm_onestep
#SBATCH --mail-user=ashri@umich.edu ##MODIFY THIS LINE!
#SBATCH --mail-type=END,FAILED
#SBATCH --nodes=1
#SBATCH --account=hunseok2
#SBATCH --partition=spgpu
#SBATCH --gpus=2
#SBATCH --mem-per-gpu=48GB
#SBATCH --cpus-per-gpu=4
#SBATCH --time=00-08:00:00
#SBATCH --output=/home/%u/latent-diffusion/job_logs/training/%x-%j.log

module load python3.9-anaconda
conda init bash
source ~/.bashrc
conda activate ldm
cd /home/ashri/latent-diffusion

rm -rf /tmpssd/ashri/
source prepare_lsun.sh
training_steps=533066
model_dir="ldm_onestep_training_$training_steps"

if [ ! -d "$log_dir/$model_dir/" ]
then
    mkdir "$log_dir/$model_dir/"
fi

for snr in {0..20..5}
do
    if [ ! -d "$log_dir/$model_dir/snr_$snr/" ]
        then
            mkdir "$log_dir/$model_dir/snr_$snr/"
    fi
    python main.py --base configs/latent-diffusion/lsun_bedrooms_onestep.yaml --max_steps $training_steps \
        --train_txt_path "$data_root_dir/bedrooms_train.txt" --val_txt_path \
    "$data_root_dir/bedrooms_val.txt" --root_dir $data_dir --channel_snr $snr --gpus=2 \
    -t --logdir $log_dir/$model_dir/snr_$snr/
done
module unload python3.9-anaconda cuda cudnn
