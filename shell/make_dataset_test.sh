#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J make_dataset_test
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 9:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=6GB]"

### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo out
#BSUB -eo err

module load python3/3.8.11
module load cuda/11.1
module load opencv/3.4.16-python-3.8.11-cuda-11.1
###python3 -m venv ../venv_1
source ../venv_1/bin/activate

# Run file
python3 -m cProfile -s tottime ../src/data/make_dataset_test.py > pyout
