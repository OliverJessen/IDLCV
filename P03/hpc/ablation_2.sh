#!/bin/bash
#BSUB -J ablation
#BSUB -q gpua10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:30
#BSUB -o hpc/logs/ablation_%J.out
#BSUB -e hpc/logs/ablation_%J.err

cd /dtu/blackhole/04/231940/IDLCV/P03

for strategy in center random; do
    for clicks in 5 10 20 40 80; do
        echo "===== Strategy: $strategy | Clicks: $clicks ====="

        uv run train_weak_auto.py --strategy $strategy --clicks $clicks

        uv run evaluate_weak_auto.py \
            checkpoints/PH2_unet_improved_clicks_${strategy}_${clicks}_best.pth
    done
done