#!/bin/bash

#SBATCH --job-name=verify_code
#SBATCH --mail-user=ashri@umich.edu ##MODIFY THIS LINE!
#SBATCH --mail-type=END,FAILED
#SBATCH --nodes=1
#SBATCH --account=hunseok2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=48GB
#SBATCH --cpus-per-gpu=4
#SBATCH --time=02-00:00:00
#SBATCH --output=/home/%u/latent-diffusion/job_logs/testing/%x-%j.log

module load python3.9-anaconda
conda init bash
source ~/.bashrc
conda activate ldm
cd /home/ashri/latent-diffusion

source prepare_lsun.sh

for test_snr in {20..0..-5}
do
    python scale_grad_expts.py --log_dir "verification_post_code" --test_snr_db $test_snr --scale_grad 0.0 --batch_size 2 --num_workers 4
done
