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


train_snr=20
for test_snr in 15 20
do
    python jscc_baseline_testing.py --log_dir "/nfs/turbo/coe-hunseok/ashri/lsun_data/loggings/jscc_models_updated/"\
    --test_snr $test_snr --train_snr $train_snr --num_workers 4
done

test_snr=0
python ldm_testing.py --log_dir "models/ldm/lsun_beds256" --test_snr $test_snr -d --batch_size 2 --num_workers 4
module unload python3.9-anaconda cuda cudnn
