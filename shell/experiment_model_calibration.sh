#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J model_calibration
### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 23:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=6GB]"
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
#BSUB -eo err_calib

module load python3/3.8.11
module load cuda/11.1
module load opencv/3.4.16-python-3.8.11-cuda-11.1
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
python3 ../src/experiments/calibration/model_calibration.py > pyout_calib
