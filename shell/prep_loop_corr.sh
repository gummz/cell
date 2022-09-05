#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J prep_loop_corr
### -- ask for number of cores (default: 1) -- 
#BSUB -n 8 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 1GB of memory per core/slot -- 
#BSUB -R "rusage[mem=8GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 30GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 

### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo out
#BSUB -eo err_corr

module load python3/3.8.11
###python3 -m venv ../venv_1
source ../venv_1/bin/activate

###pip --disable-pip-version-check install --upgrade pip
###python3 -m pip --disable-pip-version-check install numpy
###python3 -m pip --disable-pip-version-check install tifffile
###python3 -m pip --disable-pip-version-check install pickle
###python3 -m pip --disable-pip-version-check install matplotlib
###python3 -m pip --disable-pip-version-check install opencv-python
###python3 -m pip --disable-pip-version-check install pillow
###python3 -m pip install -e ../.
###python3 -m pip --disable-pip-version-check install aicsimageio


# Run file
python3 ../src/tracking/prep_loop_corr.py > pyout_corr
