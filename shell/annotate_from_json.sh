#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### For Gpu: -q gpuv100 or gpua100
### -- set the job Name --
#BSUB -J annotate_from_json
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
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
#BSUB -eo err

module load python3/3.8.11
module load cuda/11.1
module load opencv/3.4.16-python-3.8.11-cuda-11.1
###python3 -m venv venv_1
source ../venv_1/bin/activate

###python3 -m pip --disable-pip-version-check install --upgrade pip
###python3 -m pip --disable-pip-version-check install numpy
###python3 -m pip --disable-pip-version-check install pickle
###python3 -m pip --disable-pip-version-check install matplotlib
###python3 -m pip --disable-pip-version-check install opencv-python==3.4.16.57
###python3 -m pip --disable-pip-version-check install -e ../.

# Run file
python3 ../src/data/annotate_from_json.py > pyout
