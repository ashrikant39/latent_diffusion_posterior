#!/bin/bash


#SBATCH --job-name=jscc_testing
#SBATCH --mail-user=ashri@umich.edu ##MODIFY THIS LINE!
#SBATCH --mail-type=END,FAILED
#SBATCH --nodes=1
#SBATCH --account=hunseok2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=48GB
#SBATCH --cpus-per-gpu=4
#SBATCH --time=00-08:00:00
#SBATCH --output=/home/%u/latent-diffusion/job_logs/testing/%x-%j.log

module load python3.9-anaconda
conda init bash
source ~/.bashrc
conda activate ldm
cd /home/ashri/latent-diffusion


for test_snr in {5..20..5}
do
    python ldm_testing.py --log_dir "models/ldm/lsun_beds256" --test_snr $test_snr -d --batch_size 2 --num_workers 4
done

module unload python3.9-anaconda cuda cudnn
