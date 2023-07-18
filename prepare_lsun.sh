#!/bin/bash

conda init bash
source ~/.bashrc
conda activate ldm

source_path="/home/$USER/latent-diffusion/val_data_lsun/data.py"
log_dir="/nfs/turbo/coe-hunseok/$USER/lsun_data/loggings"
train_data_mbd_dir="/nfs/turbo/coe-hunseok/$USER/lsun_data/bedroom_train_lmdb/"
val_data_mbd_dir="/nfs/turbo/coe-hunseok/$USER/lsun_data/bedroom_val_lmdb/"


super_root="/tmpssd/$USER"


if [ ! -d $super_root ]
then
	mkdir $super_root
    mkdir "$super_root/LSUN" "$super_root/LSUN/bedrooms"
	data_root_dir="$super_root/LSUN"
	data_dir="$super_root/LSUN/bedrooms/"

	echo "Preparing LSUN Dataset"
	echo "Copying Text files to $data_root_dir"
	rsync -ah /home/ashri/latent-diffusion/data/lsun/bedrooms_train.txt "$data_root_dir/"
	rsync -ah /home/ashri/latent-diffusion/data/lsun/bedrooms_val.txt "$data_root_dir/"
	
	echo "Extracting training images to $data_dir"
	python $source_path export $train_data_mbd_dir --out_dir $data_dir --flat
    echo "Extracting validation images to at $data_dir"
    python $source_path export $val_data_mbd_dir --out_dir $data_dir --flat
	echo "Unzip finish"
	export data_root_dir data_dir
else
	export log_dir
	export data_root_dir="$super_root/LSUN"
	export data_dir="$super_root/LSUN/bedrooms/"
fi