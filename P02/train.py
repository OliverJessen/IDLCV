"""Train ONE model with the shared recipe and evaluate it once on the test set.

    python train.py --data leak   --model early    --backbone resnet18 --seed 0
    python train.py --data noleak --model temporal --backbone scratch  --seed 1

Shared recipe (identical for every model):
  input 112x112, ImageNet normalisation, 30 epochs, AdamW (wd 1e-4), cosine
  schedule, batch 16 videos (160 frames for per-frame), cross-entropy,
  random horizontal flip applied identically to all frames of a video
  (for flow: dx changes sign), dropout 0.5 before the classifier.
  lr = 1e-3 for scratch networks, 1e-4 for pre-trained ResNet-18.
  The checkpoint with the highest VALIDATION accuracy is kept; the TEST set is
  evaluated once with it.

Writes runs/<data>/<model>_<backbone>[_submean]_s<seed>.json with the learning
curves, best epoch, val/test accuracy and the per-video softmax on val and test
(used later for two-stream fusion and per-class analysis).
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

from datasets import FrameImageDataset, FrameVideoDataset, FlowVideoDataset
from models import build_model

ROOTS = {
    'leak': '/dtu/datasets1/02516/ufc10',
    'noleak': '/dtu/datasets1/02516/ucf101_noleakage',
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LR = {'scratch': 1e-3, 'resnet18': 1e-4}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, choices=list(ROOTS))
    p.add_argument('--model', required=True,
                   choices=['perframe', 'late_avg', 'late_concat', 'early', '3d', 'temporal'])
    p.add_argument('--backbone', required=True, choices=['scratch', 'resnet18'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--subtract-mean', action='store_true', help='temporal only: remove mean flow')
    p.add_argument('--root', default=None, help='override dataset path')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--img-size', type=int, default=112)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', default='runs')
    return p.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def video_names(ds):
    if isinstance(ds, FlowVideoDataset):
        return ds.video_names
    return [os.path.splitext(os.path.basename(p))[0] for p in ds.video_paths]


def make_data(args, root):
    tf = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    if args.model == 'temporal':
        mk = lambda s: FlowVideoDataset(root, s, args.img_size, subtract_mean=args.subtract_mean)
        train = mk('train')
    else:
        mk = lambda s: FrameVideoDataset(root, s, tf, stack_frames=True)
        train = FrameImageDataset(root, 'train', tf) if args.model == 'perframe' else mk('train')
    kw = dict(num_workers=args.workers, pin_memory=torch.cuda.is_available())
    g = torch.Generator().manual_seed(args.seed)
    train_bs = args.bs * 10 if args.model == 'perframe' else args.bs
    val, test = mk('val'), mk('test')
    return (DataLoader(train, train_bs, shuffle=True, drop_last=True, generator=g, **kw),
            DataLoader(val, args.bs, shuffle=False, **kw),
            DataLoader(test, args.bs, shuffle=False, **kw),
            video_names(val), video_names(test), len(train))


def augment(x, is_flow):
    """Random horizontal flip per sample, the same for all frames of a video.
    Works for [B,C,H,W], [B,C,T,H,W] and flow [B,18,H,W] (width = last dim)."""
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    if not flip.any():
        return x
    x = x.clone()
    xf = x[flip].flip(-1)
    if is_flow:
        xf[:, 0::2] = -xf[:, 0::2]          # mirrored motion: dx -> -dx
    x[flip] = xf
    return x


def run_epoch(model, loader, device, optimizer=None, is_flow=False, keep_probs=False):
    train = optimizer is not None
    model.train(train)
    tot, correct, n, probs, labels = 0.0, 0, 0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train:
                x = augment(x, is_flow)
            out = model(x)
            loss = F.cross_entropy(out, y)   # for PerFrame on videos: NLL of averaged softmax
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            tot += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
            if keep_probs:
                probs.append(out.softmax(1).detach().cpu())
                labels.append(y.cpu())
    res = (tot / n, correct / n)
    return res + (torch.cat(probs), torch.cat(labels).tolist()) if keep_probs else res


def main():
    args = get_args()
    set_seed(args.seed)
    device = get_device()
    root = args.root or ROOTS[args.data]
    name = f'{args.model}_{args.backbone}' + ('_submean' if args.subtract_mean else '') + f'_s{args.seed}'
    out_dir = os.path.join(args.out, args.data)
    os.makedirs(out_dir, exist_ok=True)

    train_ld, val_ld, test_ld, val_names, test_names, n_train = make_data(args, root)
    model = build_model(args.model, args.backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    lr = LR[args.backbone]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    is_flow = args.model == 'temporal'
    print(f'{name} | data={root} | {n_params / 1e6:.2f}M params | lr={lr} | '
          f'{n_train} train samples | {device}', flush=True)

    hist, best_val, best_state, best_epoch = [], -1.0, None, 0
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_l, tr_a = run_epoch(model, train_ld, device, opt, is_flow)
        va_l, va_a = run_epoch(model, val_ld, device, is_flow=is_flow)
        sched.step()
        hist.append(dict(epoch=ep, train_loss=tr_l, train_acc=tr_a, val_loss=va_l, val_acc=va_a))
        flag = ''
        if va_a > best_val:                           # model selection on VALIDATION only
            best_val, best_epoch = va_a, ep
            best_state = copy.deepcopy(model.state_dict())
            flag = ' *'
        print(f'ep {ep:3d} | train {tr_l:.3f} / {tr_a:.3f} | val {va_l:.3f} / {va_a:.3f} '
              f'| {time.time() - t0:.0f}s{flag}', flush=True)

    model.load_state_dict(best_state)
    _, va_a, va_p, va_y = run_epoch(model, val_ld, device, is_flow=is_flow, keep_probs=True)
    _, te_a, te_p, te_y = run_epoch(model, test_ld, device, is_flow=is_flow, keep_probs=True)   # ONCE
    print(f'RESULT {args.data} {name} | best epoch {best_epoch} | val {va_a:.3f} | TEST {te_a:.3f}',
          flush=True)

    with open(os.path.join(out_dir, f'{name}.json'), 'w') as f:
        json.dump(dict(
            args=vars(args), root=root, name=name, params=n_params, lr=lr,
            best_epoch=best_epoch, val_acc=va_a, test_acc=te_a,
            train_acc_at_best=hist[best_epoch - 1]['train_acc'],
            minutes=(time.time() - t_start) / 60, history=hist,
            val_probs={n: p.tolist() for n, p in zip(val_names, va_p)},
            test_probs={n: p.tolist() for n, p in zip(test_names, te_p)},
            val_labels=dict(zip(val_names, va_y)), test_labels=dict(zip(test_names, te_y)),
        ), f)


if __name__ == '__main__':
    main()
