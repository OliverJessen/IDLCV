#!/usr/bin/env python
"""
Project 3 - Part 1 (fully supervised segmentation), all experiments in one script.

Uses the group's own models (models.py) and fixed splits (datasets/*_splits.json),
and the settings described in the report's Method section:
    PH2 256x256, DRIVE 512x512, Adam, lr 1e-4, 50 epochs, batch 4 (PH2) / 2 (DRIVE),
    best checkpoint chosen on validation Dice, seed 42, no augmentation.

Experiments
    models   : 4 models (baseline/improved encoder-decoder and U-Net) x 2 datasets, BCE loss
               -> Table 1 (best validation Dice) and Table 2 (test metrics)
    ablation : U-Net improved with weighted BCE and focal loss x 2 datasets (BCE is reused)
               -> Table 3 (loss ablation)

All five metrics (Dice, IoU, Accuracy, Sensitivity, Specificity) are computed per image and
averaged, on train, val and test. For DRIVE they are computed inside the field-of-view (FOV)
mask only, the standard protocol for DRIVE; otherwise the black border inflates accuracy and
specificity.

Put this file in the P03 folder (next to models.py and datasets/) and run
    python phase1.py --shapes            # feature-map sizes for the U-Net sketch (fast, no GPU)
    python phase1.py --epochs 2          # quick smoke test
    python phase1.py                     # everything (submit with: bsub < phase1.sh)
    python phase1.py --report-only       # only rebuild tables/figures from finished runs
Finished runs are skipped, so a job that hits its time limit can simply be resubmitted.
Outputs go to results_phase1/.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:  # the group's train.py uses models/models.py, the repo has models.py - support both
    from models.models import EncoderDecoderCNN, EncoderDecoderImproved, UNet, UNetImproved
except ImportError:
    from models import EncoderDecoderCNN, EncoderDecoderImproved, UNet, UNetImproved

# ----------------------------------------------------------------------------- configuration
DATA = {
    "PH2": dict(root=os.environ.get("PH2_ROOT", "/dtu/datasets1/02516/PH2_Dataset_images"),
                split=HERE / "datasets" / "ph2_splits.json", size=(256, 256), batch=4),
    "DRIVE": dict(root=os.environ.get("DRIVE_ROOT", "/dtu/datasets1/02516/DRIVE"),
                  split=HERE / "datasets" / "drive_splits.json", size=(512, 512), batch=2),
}
MODELS = {
    "encoder_decoder": EncoderDecoderCNN,
    "encoder_decoder_improved": EncoderDecoderImproved,
    "unet": UNet,
    "unet_improved": UNetImproved,
}
LABEL = {
    "encoder_decoder": "Encoder-Decoder baseline",
    "encoder_decoder_improved": "Encoder-Decoder improved",
    "unet": "U-Net baseline",
    "unet_improved": "U-Net improved",
    "bce": "BCE", "wbce": "Weighted BCE", "focal": "Focal",
}
METRICS = ("dice", "iou", "accuracy", "sensitivity", "specificity")
METRIC_HDR = ("Dice", "IoU", "Acc.", "Sens.", "Spec.")
LR, SEED = 1e-4, 42


def experiment_list(exp, datasets):
    runs = []
    if exp in ("models", "all"):
        runs += [(d, m, "bce") for d in datasets for m in MODELS]
    if exp in ("ablation", "all"):
        runs += [(d, "unet_improved", l) for d in datasets for l in ("wbce", "focal")]
    return runs


# ----------------------------------------------------------------------------- data
class SegData(Dataset):
    """Same preprocessing as datasets/datasets.py (bilinear image, nearest mask, mask > 0),
    but everything is cached in memory and DRIVE also returns its FOV mask."""

    def __init__(self, dataset, ids):
        cfg = DATA[dataset]
        h, w = cfg["size"]
        root = Path(cfg["root"])
        self.ids, self.images, self.masks, self.fovs = [], [], [], []
        for i in ids:
            if dataset == "PH2":
                img_p = root / i / f"{i}_Dermoscopic_Image" / f"{i}.bmp"
                msk_p = root / i / f"{i}_lesion" / f"{i}_lesion.bmp"
                fov_p = None
            else:
                img_p = next((root / "training" / "images").glob(f"{i}_*.tif"))
                msk_p = root / "training" / "1st_manual" / f"{i}_manual1.gif"
                fov_p = next((root / "training" / "mask").glob(f"{i}_*mask.gif"))
            img = Image.open(img_p).convert("RGB").resize((w, h), Image.BILINEAR)
            msk = Image.open(msk_p).convert("L").resize((w, h), Image.NEAREST)
            self.ids.append(i)
            self.images.append(torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1))
            self.masks.append(torch.from_numpy((np.asarray(msk) > 0).astype(np.float32))[None])
            if fov_p is None:
                self.fovs.append(torch.ones(1, h, w))
            else:
                fov = Image.open(fov_p).convert("L").resize((w, h), Image.NEAREST)
                self.fovs.append(torch.from_numpy((np.asarray(fov) > 0).astype(np.float32))[None])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        return self.images[k], self.masks[k], self.fovs[k]


def load_splits(dataset):
    with open(DATA[dataset]["split"]) as f:
        s = json.load(f)
    return {k: SegData(dataset, s[k]) for k in ("train", "val", "test")}


# ----------------------------------------------------------------------------- losses
def make_loss(name, pos_weight=None, gamma=2.0):
    """Loss over all pixels (as in the group's train.py).
    bce   : binary cross-entropy
    wbce  : BCE with foreground weighted by pos_weight = #background / #foreground (train set)
    focal : (1 - p_t)^gamma * BCE, down-weights easy pixels (no extra class weighting, so it
            isolates the 'focusing' effect from the re-weighting tested by wbce)"""
    if name == "bce":
        return lambda logits, y: F.binary_cross_entropy_with_logits(logits, y)
    if name == "wbce":
        pw = torch.tensor([pos_weight])
        return lambda logits, y: F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw.to(logits.device))
    if name == "focal":
        def focal(logits, y):
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            p = torch.sigmoid(logits)
            p_t = p * y + (1 - p) * (1 - y)
            return ((1 - p_t) ** gamma * bce).mean()
        return focal
    raise ValueError(name)


# ----------------------------------------------------------------------------- metrics
def per_image_metrics(logits, y, valid):
    """Returns a dict metric -> tensor (B,), counted over valid pixels only."""
    pred = (torch.sigmoid(logits) > 0.5).float()
    d = (1, 2, 3)
    tp = (pred * y * valid).sum(d)
    fp = (pred * (1 - y) * valid).sum(d)
    fn = ((1 - pred) * y * valid).sum(d)
    tn = ((1 - pred) * (1 - y) * valid).sum(d)
    one, eps = torch.ones_like(tp), 1e-7
    empty = (tp + fp + fn) == 0  # nothing predicted and nothing there -> perfect
    return {
        "dice": torch.where(empty, one, 2 * tp / (2 * tp + fp + fn + eps)),
        "iou": torch.where(empty, one, tp / (tp + fp + fn + eps)),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "sensitivity": torch.where(tp + fn == 0, one, tp / (tp + fn + eps)),
        "specificity": torch.where(tn + fp == 0, one, tn / (tn + fp + eps)),
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    vals, loss_sum, n = {m: [] for m in METRICS}, 0.0, 0
    for x, y, v in loader:
        x, y, v = x.to(device), y.to(device), v.to(device)
        logits = model(x)
        loss_sum += loss_fn(logits, y).item() * x.size(0)
        n += x.size(0)
        for k, t in per_image_metrics(logits, y, v).items():
            vals[k] += t.cpu().tolist()
    return loss_sum / n, {k: float(np.mean(v)) for k, v in vals.items()}


# ----------------------------------------------------------------------------- training
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_name(dataset, model, loss):
    return f"{dataset}_{model}_{loss}"


def done_runs(csv_path):
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {r["run"] for r in csv.DictReader(f)}


def train_one(dataset, model_name, loss_name, data, args, device):
    name = run_name(dataset, model_name, loss_name)
    run_dir = args.out / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    bs = DATA[dataset]["batch"]
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(data["train"], batch_size=bs, shuffle=True, generator=g)
    loaders = {k: DataLoader(v, batch_size=bs) for k, v in data.items()}

    pos_weight = None
    if loss_name == "wbce":
        fg = torch.stack(data["train"].masks).mean().item()
        pos_weight = (1 - fg) / fg
    loss_fn = make_loss(loss_name, pos_weight)

    model = MODELS[model_name]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_ep, hist = -1.0, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss, n = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
            n += x.size(0)
        va_loss, va = evaluate(model, loaders["val"], loss_fn, device)
        hist.append({"epoch": ep, "train_loss": tr_loss / n, "val_loss": va_loss, "val_dice": va["dice"]})
        if va["dice"] > best:
            best, best_ep = va["dice"], ep
            torch.save(model.state_dict(), run_dir / "best.pth")
        print(f"[{name}] ep {ep:3d}/{args.epochs}  train loss {tr_loss / n:.4f} | "
              f"val loss {va_loss:.4f} dice {va['dice']:.4f}  (best {best:.4f} @ {best_ep})", flush=True)
    json.dump(hist, open(run_dir / "history.json", "w"), indent=1)

    model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device))
    row = {"run": name, "dataset": dataset, "model": model_name, "loss": loss_name,
           "epochs": args.epochs, "best_epoch": best_ep, "best_val_dice": round(best, 4),
           "pos_weight": round(pos_weight, 3) if pos_weight else "",
           "minutes": round((time.time() - t0) / 60, 2)}
    for split, loader in loaders.items():
        _, m = evaluate(model, loader, loss_fn, device)
        for k in METRICS:
            row[f"{split}_{k}"] = round(m[k], 4)
    print(f"[{name}] test " + "  ".join(f"{k} {row['test_' + k]:.4f}" for k in METRICS), flush=True)

    csv_path = args.out / "results.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ----------------------------------------------------------------------------- report
def latex_table(header, rows, caption, label, bold_cols=None, groups=None):
    """rows: list of lists (strings or floats). bold_cols: indices of numeric columns where the
    max per group is bolded. groups: list of group sizes for \\midrule separation."""
    rows = [list(r) for r in rows]
    if bold_cols:
        start = 0
        for size in (groups or [len(rows)]):
            block = rows[start:start + size]
            for c in bold_cols:
                best = max(r[c] for r in block)
                for r in block:
                    if r[c] == best:
                        r[c] = f"\\textbf{{{r[c]:.3f}}}"
            start += size
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    ncol = len(header)
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             f"\\begin{{tabular}}{{{'l' * (ncol - len(bold_cols or []))}{'c' * len(bold_cols or [])}}}",
             "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    start = 0
    for gi, size in enumerate(groups or [len(rows)]):
        for r in rows[start:start + size]:
            lines.append(" & ".join(fmt(v) for v in r) + " \\\\")
        start += size
        if gi < len(groups or [1]) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def make_report(args, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = args.out / "results.csv"
    if not csv_path.exists():
        print("no results yet")
        return
    rows = {}
    for r in csv.DictReader(open(csv_path)):
        rows[r["run"]] = r  # later rows overwrite earlier ones
    get = lambda d, m, l: rows.get(run_name(d, m, l))
    datasets = [d for d in DATA if any(r["dataset"] == d for r in rows.values())]
    tex = []

    # Table 1: best validation Dice
    t1 = []
    for m in MODELS:
        vals = [float(get(d, m, "bce")["best_val_dice"]) if get(d, m, "bce") else "-" for d in datasets]
        t1.append([LABEL[m]] + vals)
    if all(isinstance(v, float) for r in t1 for v in r[1:]):
        tex.append(latex_table(["Model"] + datasets, t1,
                               "Best validation Dice during fully supervised training (BCE loss).",
                               "tab:val_dice", bold_cols=list(range(1, len(datasets) + 1))))

    # Table 2: test metrics, BCE, all models, both datasets
    t2, groups = [], []
    for d in datasets:
        block = [[d if i == 0 else "", LABEL[m]] + [float(get(d, m, "bce")[f"test_{k}"]) for k in METRICS]
                 for i, m in enumerate(MODELS) if get(d, m, "bce")]
        t2 += block
        groups.append(len(block))
    if t2:
        tex.append(latex_table(["Data", "Model", *METRIC_HDR], t2,
                               "Test-set metrics of the fully supervised models (BCE loss). "
                               "DRIVE metrics are computed inside the field-of-view mask. Best per dataset in bold.",
                               "tab:test_metrics", bold_cols=list(range(2, 7)), groups=groups))

    # Table 3: loss ablation with U-Net improved
    t3, groups = [], []
    for d in datasets:
        block = [[d if i == 0 else "", LABEL[l]] + [float(get(d, "unet_improved", l)[f"test_{k}"]) for k in METRICS]
                 for i, l in enumerate(("bce", "wbce", "focal")) if get(d, "unet_improved", l)]
        if len(block) > 1:
            t3 += block
            groups.append(len(block))
    if t3:
        tex.append(latex_table(["Data", "Loss", *METRIC_HDR], t3,
                               "Loss ablation with the improved U-Net, test-set metrics. Weighted BCE uses "
                               "pos\\_weight = \\#background / \\#foreground pixels of the training set.",
                               "tab:loss_ablation", bold_cols=list(range(2, 7)), groups=groups))
    (args.out / "tables.tex").write_text("\n".join(tex))

    # Appendix: every run, every split, markdown
    md = ["| run | split | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for r in rows.values():
        for s in ("train", "val", "test"):
            md.append(f"| {r['run']} | {s} | " + " | ".join(f"{float(r[f'{s}_{k}']):.3f}" for k in METRICS) + " |")
    (args.out / "all_metrics.md").write_text("\n".join(md) + "\n")

    # Curves: validation Dice per epoch (models) and per loss (ablation)
    for d in datasets:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for m in MODELS:
            p = args.out / "runs" / run_name(d, m, "bce") / "history.json"
            if p.exists():
                h = json.load(open(p))
                axes[0].plot([e["epoch"] for e in h], [e["val_dice"] for e in h], label=LABEL[m])
        for l in ("bce", "wbce", "focal"):
            p = args.out / "runs" / run_name(d, "unet_improved", l) / "history.json"
            if p.exists():
                h = json.load(open(p))
                axes[1].plot([e["epoch"] for e in h], [e["val_dice"] for e in h], label=LABEL[l])
        axes[0].set(title=f"{d}: models (BCE)", xlabel="epoch", ylabel="validation Dice", ylim=(0, 1))
        axes[1].set(title=f"{d}: U-Net improved, loss ablation", xlabel="epoch", ylim=(0, 1))
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(args.out / f"curves_{d}.pdf")
        fig.savefig(args.out / f"curves_{d}.png", dpi=150)
        plt.close(fig)

    # Qualitative figure: test images, ground truth and predictions of each model
    for d in datasets:
        avail = [m for m in MODELS if (args.out / "runs" / run_name(d, m, "bce") / "best.pth").exists()]
        if not avail:
            continue
        with open(DATA[d]["split"]) as f:
            test_ids = json.load(f)["test"][:3]
        data = SegData(d, test_ids)
        preds = {}
        for m in avail:
            net = MODELS[m]().to(device)
            net.load_state_dict(torch.load(args.out / "runs" / run_name(d, m, "bce") / "best.pth", map_location=device))
            net.eval()
            with torch.no_grad():
                preds[m] = [((torch.sigmoid(net(x[None].to(device)))[0, 0] > 0.5).float() * v[0].to(device)).cpu()
                            for x, _, v in data]
        cols = 2 + len(avail)
        fig, axes = plt.subplots(len(data), cols, figsize=(2.2 * cols, 2.2 * len(data)), squeeze=False)
        for i, (x, y, _) in enumerate(data):
            panels = [(x.permute(1, 2, 0), "Image"), (y[0], "Ground truth")]
            panels += [(preds[m][i], LABEL[m].replace("Encoder-Decoder", "Enc-Dec")) for m in avail]
            for j, (im, title) in enumerate(panels):
                axes[i, j].imshow(im, cmap=None if j == 0 else "gray", vmin=0, vmax=1)
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(title, fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out / f"examples_{d}.pdf")
        fig.savefig(args.out / f"examples_{d}.png", dpi=150)
        plt.close(fig)

    print(f"\nReport files written to {args.out}/: tables.tex, all_metrics.md, curves_*.pdf/png, examples_*.pdf/png")
    print("\nTest metrics:")
    for r in rows.values():
        print(f"  {r['run']:<40} " + "  ".join(f"{k[:4]} {float(r['test_' + k]):.3f}" for k in METRICS))


# ----------------------------------------------------------------------------- shapes
def print_shapes(size=(256, 256)):
    """Feature-map size after every block - use it to check the U-Net sketch (task 3)."""
    for name in ("unet", "unet_improved"):
        model = MODELS[name]().eval()
        print(f"\n{LABEL[name]}  input (3, {size[0]}, {size[1]})  "
              f"params {sum(p.numel() for p in model.parameters()):,}")
        hooks = [mod.register_forward_hook(
            lambda _m, _i, o, n=n: print(f"  {n:<11} -> {tuple(o.shape[1:])}"))
            for n, mod in model.named_children()]
        with torch.no_grad():
            model(torch.zeros(1, 3, *size))
        for h in hooks:
            h.remove()
    print("\nNote: each dec block receives up-sampled + skip channels concatenated (e.g. 128 + 128 = 256).")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="all", choices=["models", "ablation", "all"])
    ap.add_argument("--datasets", nargs="+", default=["PH2", "DRIVE"], choices=list(DATA))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", type=Path, default=HERE / "results_phase1")
    ap.add_argument("--rerun", action="store_true", help="retrain runs that are already finished")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--shapes", action="store_true")
    args = ap.parse_args()

    if args.shapes:
        print_shapes()
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        finished = set() if args.rerun else done_runs(args.out / "results.csv")
        todo = [r for r in experiment_list(args.exp, args.datasets) if run_name(*r) not in finished]
        print(f"{len(todo)} runs to do:", [run_name(*r) for r in todo], flush=True)
        cache = {}
        for d, m, l in todo:
            if d not in cache:
                cache[d] = load_splits(d)
                print(d, {k: len(v) for k, v in cache[d].items()}, flush=True)
            train_one(d, m, l, cache[d], args, device)
    make_report(args, device)


if __name__ == "__main__":
    main()
