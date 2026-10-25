"""Collect all runs/ results: tables (mean +- std over seeds), two-stream fusion,
per-class accuracy, confusion matrix and learning curves. CPU only, no GPU needed:

    python analyze.py            (run in the folder that contains runs/)

Prints everything for the report and writes summary.json, fig_curves.pdf and
fig_confusion.pdf.
"""
import glob
import json
import os

import numpy as np

SEEDS = [0, 1, 2]
BACKBONES = ['scratch', 'resnet18']
MODELS = [('perframe', 'Per-frame + avg.'), ('late_avg', 'Late fusion, avg.'),
          ('late_concat', 'Late fusion, concat'), ('early', 'Early fusion'), ('3d', '3D CNN')]
WEIGHTS = [0.0, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0]
summary = {}


def load(data, name):
    p = f'runs/{data}/{name}.json'
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def ms(vals):
    v = 100 * np.array(vals, dtype=float)
    if len(v) == 0:
        return '      --      '
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f'{v.mean():5.1f} +- {sd:4.1f} (n={len(v)})'


def accs(data, model, bb, key='test_acc', suffix=''):
    out = []
    for s in SEEDS:
        r = load(data, f'{model}_{bb}{suffix}_s{s}')
        if r is not None:
            out.append(r[key])
    return out


# ------------------------------------------------------------------ 4.1 / 4.2.1
for data, title in [('leak', 'WITH leakage (ufc10)'), ('noleak', 'NO leakage (ucf101_noleakage)')]:
    print(f'\n=== {title}: TEST accuracy %, mean +- std over seeds ===')
    print(f'{"model":22s} {"scratch":>22s} {"resnet18 (ImageNet)":>22s}')
    for m, label in MODELS:
        row = {}
        for bb in BACKBONES:
            row[bb] = accs(data, m, bb)
            summary[f'{data}/{m}_{bb}/test'] = row[bb]
            summary[f'{data}/{m}_{bb}/val'] = accs(data, m, bb, 'val_acc')
        print(f'{label:22s} {ms(row["scratch"]):>22s} {ms(row["resnet18"]):>22s}')

print('\n=== Drop when leakage is removed (pp, mean over seeds) ===')
for m, label in MODELS:
    parts = []
    for bb in BACKBONES:
        a, b = accs('leak', m, bb), accs('noleak', m, bb)
        parts.append(f'{bb}: {100 * (np.mean(a) - np.mean(b)):5.1f}' if a and b else f'{bb}:   -- ')
    print(f'{label:22s} ' + '   '.join(parts))

print('\n=== Best epoch / train acc at best epoch (seed 0, no leakage) ===')
for m, label in MODELS + [('temporal', 'Temporal (flow)')]:
    for bb in BACKBONES:
        r = load('noleak', f'{m}_{bb}_s0')
        if r:
            print(f'{label:22s} {bb:9s} best ep {r["best_epoch"]:2d}/{len(r["history"])}  '
                  f'train {100 * r["train_acc_at_best"]:5.1f}  val {100 * r["val_acc"]:5.1f}  '
                  f'test {100 * r["test_acc"]:5.1f}  ({r["minutes"]:.1f} min)')


# ------------------------------------------------------------------ 4.2.2 two-stream
def fuse(rgb, flow, split, w):
    names = sorted(rgb[f'{split}_probs'])
    assert set(names) == set(flow[f'{split}_probs']), 'RGB and flow videos differ'
    P = np.array([(1 - w) * np.array(rgb[f'{split}_probs'][n]) + w * np.array(flow[f'{split}_probs'][n])
                  for n in names])
    y = np.array([rgb[f'{split}_labels'][n] for n in names])
    return float((P.argmax(1) == y).mean()), P, y, names


print('\n=== Two-stream (no leakage): TEST accuracy %, mean +- std over seeds ===')
for bb in BACKBONES:
    rows = {'RGB stream (per-frame)': [], 'Flow stream (temporal)': [],
            'Two-stream, w=0.5': [], 'Two-stream, w from val': []}
    gain05, chosen = [], []
    for s in SEEDS:
        rgb, flow = load('noleak', f'perframe_{bb}_s{s}'), load('noleak', f'temporal_{bb}_s{s}')
        if rgb is None or flow is None:
            continue
        rows['RGB stream (per-frame)'].append(rgb['test_acc'])
        rows['Flow stream (temporal)'].append(flow['test_acc'])
        t05 = fuse(rgb, flow, 'test', 0.5)[0]
        rows['Two-stream, w=0.5'].append(t05)
        val_scores = [fuse(rgb, flow, 'val', w)[0] for w in WEIGHTS]
        w_best = WEIGHTS[int(np.argmax(val_scores))]                 # chosen on VAL
        chosen.append(w_best)
        rows['Two-stream, w from val'].append(fuse(rgb, flow, 'test', w_best)[0])
        gain05.append(t05 - flow['test_acc'])
    print(f'-- backbone: {bb}')
    for k, v in rows.items():
        summary[f'twostream/{bb}/{k}'] = v
        print(f'   {k:26s} {ms(v)}   ' + ' '.join(f'{100 * x:.1f}' for x in v))
    if gain05:
        print(f'   gain of w=0.5 over flow per seed: ' + ' '.join(f'{100 * g:+.1f}' for g in gain05)
              + f'   (mean {100 * np.mean(gain05):+.1f})')
        print(f'   w chosen on val per seed: {chosen}')

rgb0, sub0, flow0 = (load('noleak', 'perframe_resnet18_s0'),
                     load('noleak', 'temporal_resnet18_submean_s0'), load('noleak', 'temporal_resnet18_s0'))
if rgb0 and sub0 and flow0:
    print('\n=== Ablation: mean flow subtracted (resnet18, seed 0) ===')
    print(f'   flow stream          {100 * flow0["test_acc"]:.1f}  ->  with mean subtracted {100 * sub0["test_acc"]:.1f}')
    print(f'   two-stream w=0.5     {100 * fuse(rgb0, flow0, "test", .5)[0]:.1f}  ->  '
          f'with mean subtracted {100 * fuse(rgb0, sub0, "test", .5)[0]:.1f}')

# ------------------------------------------------------------------ per class + confusion
if rgb0 and flow0:
    cls = {}
    for d in glob.glob('/dtu/datasets1/02516/ucf101_noleakage/flows/test/*/*'):
        cls[os.path.basename(d)] = os.path.basename(os.path.dirname(d))
    _, P2, y, names = fuse(rgb0, flow0, 'test', 0.5)
    Pr = np.array([rgb0['test_probs'][n] for n in names])
    Pf = np.array([flow0['test_probs'][n] for n in names])
    C = 10
    cname = {}
    for n, lab in zip(names, y):
        cname[int(lab)] = cls.get(n, n.split('_')[1] if n.startswith('v_') else f'class{lab}')
    class_names = [cname.get(c, f'class{c}') for c in range(C)]
    print('\n=== Per-class TEST accuracy %, resnet18, seed 0 ===')
    print(f'{"class":20s} {"n":>3s} {"RGB":>7s} {"Flow":>7s} {"Two-str":>8s}')
    pc = {}
    for c in range(C):
        m = y == c
        if not m.any():
            continue
        a = [100 * float((P[m].argmax(1) == c).mean()) for P in (Pr, Pf, P2)]
        pc[class_names[c]] = a
        print(f'{class_names[c]:20s} {m.sum():3d} {a[0]:7.1f} {a[1]:7.1f} {a[2]:8.1f}')
    summary['per_class_resnet18_s0'] = pc
    conf = np.zeros((C, C), int)
    for t, p in zip(y, P2.argmax(1)):
        conf[t, p] += 1
    summary['confusion_twostream_resnet18_s0'] = conf.tolist()
    summary['class_names'] = class_names
    print('\nMost frequent confusions (two-stream, resnet18, seed 0):')
    off = sorted(((conf[i, j], class_names[i], class_names[j]) for i in range(C) for j in range(C)
                  if i != j and conf[i, j]), reverse=True)[:6]
    for k, a, b in off:
        print(f'  {k} x  {a} -> predicted {b}')

with open('summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# ------------------------------------------------------------------ figures
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False})
    panels = [('RGB, scratch', 'perframe_scratch_s0'), ('RGB, ResNet-18', 'perframe_resnet18_s0'),
              ('Flow, ResNet-18', 'temporal_resnet18_s0')]
    panels = [(t, load('noleak', n)) for t, n in panels]
    panels = [(t, r) for t, r in panels if r]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(3.3, 1.45), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (t, r) in zip(axes, panels):
            ep = [h['epoch'] for h in r['history']]
            ax.plot(ep, [h['train_acc'] for h in r['history']], '--', color='#2a6fdb', lw=1.3, label='train')
            ax.plot(ep, [h['val_acc'] for h in r['history']], '-', color='#d9622b', lw=1.3, label='val')
            ax.set_title(t, fontsize=7)
            ax.set_xlabel('epoch')
            ax.set_ylim(0, 1.02)
            ax.grid(axis='y', color='#dddddd', lw=0.5)
        axes[0].set_ylabel('accuracy')
        axes[-1].legend(frameon=False, fontsize=6, loc='lower right')
        fig.tight_layout(pad=0.3)
        fig.savefig('fig_curves.pdf')
    if rgb0 and flow0:
        fig, ax = plt.subplots(figsize=(3.3, 3.0))
        ax.imshow(conf, cmap='Blues', vmin=0)
        short = [n.replace('Handstand', 'Hs.').replace('BodyWeight', 'BW') for n in class_names]
        ax.set_xticks(range(C))
        ax.set_xticklabels(short, rotation=60, ha='right')
        ax.set_yticks(range(C))
        ax.set_yticklabels(short)
        ax.set_xlabel('predicted'); ax.set_ylabel('true')
        for i in range(C):
            for j in range(C):
                if conf[i, j]:
                    ax.text(j, i, conf[i, j], ha='center', va='center', fontsize=6.5,
                            color='white' if conf[i, j] > conf.max() / 2 else '#333333')
        fig.tight_layout(pad=0.3)
        fig.savefig('fig_confusion.pdf')
    print('\nsaved summary.json, fig_curves.pdf, fig_confusion.pdf')
except ImportError:
    print('\nmatplotlib missing (pip install matplotlib) - tables and summary.json are done')
