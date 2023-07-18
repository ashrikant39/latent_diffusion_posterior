#!/bin/bash


#SBATCH --job-name=ldm_jscc
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

source prepare_lsun.sh
model_dir="jscc_models_updated"

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
    python main.py --base configs/autoencoder/jscc_64x64x3.yaml --max_steps 200 \
        --train_txt_path "$data_root_dir/bedrooms_train.txt" --val_txt_path \
    "$data_root_dir/bedrooms_val.txt" --root_dir $data_dir -c $snr --gpus=2 \
    -t --logdir $log_dir/$model_dir/snr_$snr/
done
module unload python3.9-anaconda cuda cudnn
