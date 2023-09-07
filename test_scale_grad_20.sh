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
#SBATCH --time=00-02:00:00
#SBATCH --output=/home/%u/latent-diffusion/job_logs/testing/%x-%j.log

module load python3.9-anaconda
conda init bash
source ~/.bashrc
conda activate ldm
cd /home/ashri/latent-diffusion

source prepare_lsun.sh

test_snr=20
dir="eta_sweeps_corrected/test_snr_$test_snr"
if [ ! -d $dir ]
then
    mkdir $dir
fi

for scale_factor in 10.0 1.0 0.1 0.01 0.001
do
    python scale_grad_expts.py --log_dir $dir --test_snr_db $test_snr --scale_grad $scale_factor --batch_size 2 \
    --num_workers 4
done
