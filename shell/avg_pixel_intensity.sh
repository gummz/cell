#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J avg_pixel_intensity
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 1GB of memory per core/slot -- 
#BSUB -R "rusage[mem=1GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 3GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 

### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo out
#BSUB -eo err

module load python3/3.8.2
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

# -m cProfile -s tottime
# Run file
python3 ../src/data/avg_pixel_intensity.py > pyout
