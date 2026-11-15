#!/bin/bash
#BSUB -J train_weak
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:30
#BSUB -o hpc/logs/train_weak_%J.out
#BSUB -e hpc/logs/train_weak_%J.err

cd /dtu/blackhole/04/231940/IDLCV/P03
uv run train.py