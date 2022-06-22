#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J predict_cpu
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 1GB of memory per core/slot -- 
#BSUB -R "rusage[mem=6GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 7GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo out
#BSUB -eo err_predict

module load python3/3.8.11
# module load ffmpeg/5.0.1
###python3 -m venv venv_1
source ../venv_1/bin/activate

###pip install --upgrade pip
###python3 -m pip --disable-pip-version-check install numpy
###python3 -m pip --disable-pip-version-check install tifffile
###python3 -m pip --disable-pip-version-check install pickle
###python3 -m pip --disable-pip-version-check install matplotlib
###python3 -m pip --disable-pip-version-check install opencv-python
###python3 -m pip --disable-pip-version-check install torch
###python3 -m pip --disable-pip-version-check install torch==1.8.2+cu111 torchvision==0.9.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
###python3 -m pip --disable-pip-version-check install torchvision
###python3 -m pip --disable-pip-version-check install torch-summary
###python3 -m pip --disable-pip-version-check install pycocotools

# Run file
python3 -m cProfile -s tottime ../src/models/predict_model.py > pyout_predict
