#!/bin/bash

#SBATCH --job-name=posterior_test
#SBATCH --mail-user=ashri@umich.edu ##MODIFY THIS LINE!
#SBATCH --mail-type=END,FAILED
#SBATCH --nodes=1
#SBATCH --account=hunseok2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=48GB
#SBATCH --cpus-per-gpu=4
#SBATCH --time=00-04:00:00
#SBATCH --output=/home/%u/latent-diffusion/job_logs/testing/%x-%j.log

module load python3.9-anaconda
conda init bash
source ~/.bashrc
conda activate ldm
cd /home/ashri/latent-diffusion

source prepare_lsun.sh
images=200

for test_snr in 20 15
do
    python trained_posterior_expts.py --ckpt_dir "/nfs/turbo/coe-hunseok/ashri/lsun_data/loggings/beta_changed_1000" \
    --test_snr_db $test_snr --scale_grad 1.0 --num_images $images --batch_size 2 --num_workers 4 --save_dir "posterior_expts_${images}_images"
done
