#!/bin/bash

#SBATCH --job-name=posterior_testing
#SBATCH --mail-user=ashri@umich.edu ##MODIFY THIS LINE!
#SBATCH --mail-type=END,FAILED
#SBATCH --nodes=1
#SBATCH --account=hunseok1
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=48GB
#SBATCH --cpus-per-gpu=4
#SBATCH --time=00-20:00:00
#SBATCH --output=/home/%u/latent-diffusion/job_logs/testing/%x-%j.log

module load python3.9-anaconda
conda init bash
source ~/.bashrc
conda activate ldm
cd /home/ashri/latent-diffusion

source prepare_lsun.sh

for test_snr in 10 15
do
    python posterior_expts.py --log_dir "post_scale_corrected" --test_snr_db $test_snr --batch_size 2 --num_workers 4
done
