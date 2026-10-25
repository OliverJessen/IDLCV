#!/bin/sh
#BSUB -J perframe
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 02:00
#BSUB -o logs/perframe_%J.out
#BSUB -e logs/perframe_%J.err

module load python3
source ~/venvs/dlcv/bin/activate

python3 simple_train-2.py
