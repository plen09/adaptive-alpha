#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe-gradient meta-controller for the bidirectional loss weight alpha.

Plain-language summary of the idea
----------------------------------
The old controller looked only at clean training losses and gave more weight
to whichever loss was "falling slower". The s42 experiments showed that this
points the wrong way: the T2I objective is the better teacher for the shared
visual tower even though its own loss falls faster.

This controller instead asks a direct question every `--probe-every` steps:

    "If I take one gradient step on L_i2t, or one on L_t2i, which one would
     reduce the loss on a small held-out PROBE set more?"

Using a first-order approximation, the answer is decided by comparing how
well each direction's gradient aligns (cosine similarity) with the gradient
of the probe objective. alpha (the weight on I2T) moves toward whichever
direction wins. Because the probe set is evaluated CLEAN *and* CORRUPTED
(same deterministic noise as the robustness evaluation), the controller can
"see" the robustness trade-off that clean training losses are blind to.

Data budget (fairness)
----------------------
For every run launched through THIS script, the 45,000-image training pool is
re-split into 44,000 train + 1,000 probe (per-class: 4400 + 100, using the
same seeded stratified permutation as the original experiment, so the 5,000
image validation split is IDENTICAL to before). Fixed-alpha baselines trained
with this script use the same 44k subset and never touch the probe, so the
comparison is apples-to-apples. The final validation split stays untouched;
the official test split is never used here.

Usage
-----
    # meta-controller
    python probe_meta_clip_cifar10.py --controller probe --seed 42 \
        --out-dir runs_meta_s42

    # matched fixed-alpha baselines on the same 44k training subset
    python probe_meta_clip_cifar10.py --controller fixed --alpha 0.3 \
        --seed 42 --out-dir runs_meta_s42
    python probe_meta_clip_cifar10.py --controller fixed --alpha 0.5 \
        --seed 42 --out-dir runs_meta_s42

    # pipeline check (results NOT usable for comparison)
    python probe_meta_clip_cifar10.py --controller probe --epochs 1 \
        --max-train-batches 12 --warmup-steps 4 --probe-every 4 \
        --probe-batch 64 --self-test

Outputs per run are the same files as the original script (steps.csv,
epochs.csv, alpha_curve.png, validation_curve.png, best.pt, summary.json)
plus alpha_alignment.png and extra columns with the alignment statistics.
best.pt / summary.json keep the original schema, so
evaluate_clip_robustness.py can consume these runs unchanged.
"""

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import open_clip

from adaptive_clip_cifar10 import (
    set_seed, pick_device, class_text_features, image_text_logits,
    paired_direction_losses, evaluate, select_trainable,
    trainable_snapshot, write_csv, plot_logs,
)
from evaluate_clip_robustness import corrupted_pil, NOISE_SIGMA, BLUR_RADIUS


def parse_args():
    p = argparse.ArgumentParser(description="Probe-gradient meta-controller for alpha.")
    p.add_argument("--controller", choices=["probe", "fixed"], default="probe",
                   help="'probe' = meta-controller; 'fixed' = matched baseline.")
    p.add_argument("--model", default="ViT-B-32-quickgelu")
    p.add_argument("--pretrained", default="./weights/open_clip_model.safetensors")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./runs_meta")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--train-blocks", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Fixed I2T weight, only used with --controller fixed.")
    # Probe / controller hyperparameters.
    p.add_argument("--probe-per-class", type=int, default=100,
                   help="Images per class removed from training to form the probe.")
    p.add_argument("--probe-batch", type=int, default=128,
                   help="Probe images scored per controller visit.")
    p.add_argument("--probe-every", type=int, default=50,
                   help="Controller updates every this-many training steps.")
    p.add_argument("--probe-condition", default="noise1",
                   choices=tuple(NOISE_SIGMA) + tuple(BLUR_RADIUS),
                   help="Corruption applied to half of each probe batch.")
    p.add_argument("--probe-noise-weight", type=float, default=0.5,
                   help="Weight of the corrupted part of the probe objective.")
    p.add_argument("--eta-alpha", type=float, default=0.1,
                   help="Step size for moving alpha per controller visit.")
    p.add_argument("--delta-ema", type=float, default=0.9,
                   help="Momentum for the alignment-difference signal.")
    p.add_argument("--warmup-steps", type=int, default=100,
                   help="Steps before the controller starts moving alpha.")
    p.add_argument("--min-alpha", type=float, default=0.2)
    p.add_argument("--max-alpha", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--max-train-batches", type=int, default=None,
                   help="Debug only; results with this flag are not comparable.")
    p.add_argument("--self-test", action="store_true",
                   help="Check controller math on synthetic gradients, then exit.")
    args = p.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        p.error("--alpha must be in [0,1].")
    if not 0 <= args.min_alpha < args.max_alpha <= 1:
        p.error("Invalid alpha clipping bounds.")
    if args.probe_every < 1 or args.warmup_steps < 0:
        p.error("--probe-every >= 1 and --warmup-steps >= 0 are required.")
    if not 0.0 <= args.probe_noise_weight <= 1.0:
        p.error("--probe-noise-weight must be in [0,1].")
    if args.probe_per_class < 1 or args.probe_per_class > 500:
        p.error("--probe-per-class must be in 1..500 (training pool has 4500/class).")
    return args


class ProbeMetaAlpha:
    """Moves alpha toward the direction whose gradient better serves the probe.

    update rule (plain words):
        delta   = cos(g_probe, g_i2t) - cos(g_probe, g_t2i)
        signal  = EMA(delta)
        alpha  <- clamp(alpha + eta * signal, min_alpha, max_alpha)

    delta > 0 means a step on the I2T loss would reduce the probe objective
    more than a step on the T2I loss, so alpha (the I2T weight) increases.
    alpha never receives gradients; this is a pure outer-loop heuristic.
    """

    def __init__(self, eta_alpha, delta_ema, min_alpha, max_alpha, start=0.5):
        self.alpha = float(start)
        self.eta_alpha = float(eta_alpha)
        self.delta_ema = float(delta_ema)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.ema_delta = 0.0
        self.visits = 0

    @staticmethod
    def _flat(grads):
        return torch.cat([g.detach().reshape(-1).float() for g in grads])

    @torch.no_grad()
    def update(self, g_i, g_t, g_p):
        vi, vt, vp = self._flat(g_i), self._flat(g_t), self._flat(g_p)
        c_i = F.cosine_similarity(vp, vi, dim=0).item()
        c_t = F.cosine_similarity(vp, vt, dim=0).item()
        c_it = F.cosine_similarity(vi, vt, dim=0).item()
        delta = c_i - c_t
        self.ema_delta = self.delta_ema * self.ema_delta + (1 - self.delta_ema) * delta
        self.alpha = min(self.max_alpha,
                         max(self.min_alpha, self.alpha + self.eta_alpha * self.ema_delta))
        self.visits += 1
        return {"cos_probe_i2t": c_i, "cos_probe_t2i": c_t,
                "cos_i2t_t2i": c_it, "delta": delta, "ema_delta": self.ema_delta}


def self_test():
    p = [torch.zeros(4, requires_grad=True)]
    g_i = [torch.tensor([1.0, 0.0, 0.0, 0.0])]
    g_t = [torch.tensor([0.0, 1.0, 0.0, 0.0])]
    # Probe gradient closer to g_i => alpha must rise.
    c = ProbeMetaAlpha(eta_alpha=0.5, delta_ema=0.0, min_alpha=0.2, max_alpha=0.8)
    st = c.update(g_i, g_t, [torch.tensor([1.0, 0.2, 0.0, 0.0])])
    assert st["delta"] > 0 and c.alpha > 0.5, st
    # Probe gradient closer to g_t => alpha must fall.
    c2 = ProbeMetaAlpha(eta_alpha=0.5, delta_ema=0.0, min_alpha=0.2, max_alpha=0.8)
    st2 = c2.update(g_i, g_t, [torch.tensor([0.2, 1.0, 0.0, 0.0])])
    assert st2["delta"] < 0 and c2.alpha < 0.5, st2
    # Huge step size must be clipped to the bounds.
    c3 = ProbeMetaAlpha(eta_alpha=100.0, delta_ema=0.0, min_alpha=0.2, max_alpha=0.8)
    c3.update(g_i, g_t, [torch.tensor([1.0, 0.2, 0.0, 0.0])])
    assert c3.alpha == 0.8, c3.alpha
    # EMA smoothing: with delta_ema=1.0 the signal never changes alpha.
    c4 = ProbeMetaAlpha(eta_alpha=0.5, delta_ema=1.0, min_alpha=0.2, max_alpha=0.8)
    c4.update(g_i, g_t, [torch.tensor([1.0, 0.2, 0.0, 0.0])])
    assert c4.alpha == 0.5, c4.alpha
    print("SELF-TEST PASSED: alignment sign, clipping and EMA behaviour are correct.")


def make_loaders(args, preprocess_val, device):
    """Same seeded stratified split as the original experiment, plus a probe.

    Per class: val = first 500 of the permutation (identical to the original),
    probe = next `probe_per_class`, train = the remaining 4400 (or 4500 when
    probe_per_class is 0). The rng is consumed in exactly the original order,
    so the validation split matches runs/ byte-for-byte for the same seed.
    """
    train_preprocess = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        preprocess_val,
    ])
    train_set = datasets.CIFAR10(args.data_dir, train=True,
                                 download=True, transform=train_preprocess)
    val_set = datasets.CIFAR10(args.data_dir, train=True,
                               download=False, transform=preprocess_val)
    raw = datasets.CIFAR10(args.data_dir, train=True,
                           download=False, transform=None)

    rng = np.random.default_rng(args.seed)
    all_labels = np.asarray(raw.targets)
    train_idx, val_idx, probe_idx = [], [], []
    for label in range(len(raw.classes)):
        indices = rng.permutation(np.where(all_labels == label)[0])
        val_idx.extend(indices[:500].tolist())
        probe_idx.extend(indices[500:500 + args.probe_per_class].tolist())
        train_idx.extend(indices[500 + args.probe_per_class:].tolist())

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    train_loader = DataLoader(
        Subset(train_set, train_idx), shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), **loader_kw)
    val_loader = DataLoader(Subset(val_set, val_idx), shuffle=False, **loader_kw)
    return train_loader, val_loader, raw, probe_idx


def build_probe_cache(raw, probe_indices, preprocess, condition, seed):
    """Precompute (clean, corrupted) probe tensors once.

    The corruption is the exact deterministic 32px corruption from
    evaluate_clip_robustness.corrupted_pil, keyed by the original image
    index, so the controller optimizes the same perturbation the final
    robustness report measures.
    """
    cleans, noisys, labels = [], [], []
    t0 = time.perf_counter()
    for i in probe_indices:
        image, label = raw[i]
        cleans.append(preprocess(image))
        noisys.append(preprocess(corrupted_pil(image, condition, i, seed)))
        labels.append(int(label))
    cache = (torch.stack(cleans), torch.stack(noisys), torch.tensor(labels))
    print(f"Probe cache built: {len(labels)} images x2 views "
          f"(condition={condition}) in {time.perf_counter()-t0:.1f}s")
    return cache


def probe_objective(model, text_feats, clean, noisy, labels,
                    device, amp, noise_weight):
    """Half clean + half corrupted, each averaging both direction losses."""
    ctx = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
    x = torch.cat([clean, noisy]).to(device, non_blocking=True)
    with ctx:
        logits = image_text_logits(model, x, text_feats)
    logits = logits.float()
    lc, ln = logits.split(len(clean))
    li_c, lt_c = paired_direction_losses(lc, labels)
    li_n, lt_n = paired_direction_losses(ln, labels)
    clean_part = 0.5 * (li_c + lt_c)
    noise_part = 0.5 * (li_n + lt_n)
    return (1.0 - noise_weight) * clean_part + noise_weight * noise_part


def plot_alignment(out_path, step_rows):
    visits = [r for r in step_rows if r.get("probe_visit") == 1]
    if not visits:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot([r["step"] for r in step_rows], [r["alpha"] for r in step_rows],
             color="tab:blue", linewidth=1.4, label="alpha (I2T weight)")
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("alpha", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot([r["step"] for r in visits], [r["ema_delta"] for r in visits],
             color="tab:red", linewidth=1.0, marker=".", markersize=3,
             label="EMA[cos(probe,i2t) - cos(probe,t2i)]")
    ax2.axhline(0.0, color="tab:red", linestyle=":", linewidth=0.6)
    ax2.set_ylabel("alignment difference", color="tab:red")
    ax1.set_title("Meta-controller: alpha and probe-alignment signal")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path / "alpha_alignment.png", dpi=160)
    plt.close(fig)


def run(args):
    set_seed(args.seed)
    device = pick_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    print(f"Device={device}; AMP={amp}; controller={args.controller}; "
          f"model={args.model}; pretrained={args.pretrained}")

    model, _, preprocess_val = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained)
    model = model.to(device)
    model.eval()
    train_loader, val_loader, raw, probe_idx = make_loaders(args, preprocess_val, device)
    text_feats, prompts = class_text_features(model, args.model, raw.classes, device)

    probe_cache = None
    probe_rng = np.random.default_rng(args.seed + 1234)
    if args.controller == "probe":
        probe_cache = build_probe_cache(raw, probe_idx, preprocess_val,
                                        args.probe_condition, args.seed)

    params = select_trainable(model, args.train_blocks)
    print("Trainable parameters:", f"{sum(p.numel() for p in params):,}")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    controller = None
    if args.controller == "probe":
        controller = ProbeMetaAlpha(args.eta_alpha, args.delta_ema,
                                    args.min_alpha, args.max_alpha)

    tag = ("probe_meta" if args.controller == "probe"
           else f"fixed_a{args.alpha:.2f}")
    output = Path(args.out_dir) / f"{tag}_s{args.seed}"
    output.mkdir(parents=True, exist_ok=True)

    initial_val = evaluate(model, val_loader, text_feats, device, amp)
    best_val, best_epoch = initial_val, 0
    best_state = trainable_snapshot(model)
    epoch_rows = [{
        "epoch": 0, "train_i2t": "", "train_t2i": "", "train_total": "",
        "alpha_mean": "", "alpha_min": "", "alpha_max": "",
        "probe_visits": "", "cos_i2t_t2i_mean": "", "delta_mean": "",
        "val_acc_i2t": initial_val["acc_i2t"],
        "val_map_t2i": initial_val["map_t2i"],
        "val_score": initial_val["score"], "seconds": 0.0,
    }]
    step_rows = []
    global_step = 0
    print(f"Epoch 0 (pretrained): {initial_val}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_i, total_t, total_loss, alphas = [], [], [], []
        visit_cos, visit_delta = [], []
        start = time.perf_counter()

        for batch_index, (images, labels) in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()

            with ctx:
                logits = image_text_logits(model, images, text_feats)
                li, lt = paired_direction_losses(logits, labels)

            stats = None
            due = (controller is not None
                   and global_step + 1 > args.warmup_steps
                   and (global_step + 1 - args.warmup_steps) % args.probe_every == 0)
            if due:
                g_i = torch.autograd.grad(li, params, retain_graph=True)
                g_t = torch.autograd.grad(lt, params, retain_graph=True)
                sel = probe_rng.choice(len(probe_idx),
                                       size=min(args.probe_batch, len(probe_idx)),
                                       replace=False)
                p_clean = probe_cache[0][sel]
                p_noisy = probe_cache[1][sel]
                p_labels = probe_cache[2][sel].to(device)
                l_probe = probe_objective(model, text_feats, p_clean, p_noisy,
                                          p_labels, device, amp,
                                          args.probe_noise_weight)
                g_p = torch.autograd.grad(l_probe, params)
                stats = controller.update(g_i, g_t, g_p)
                visit_cos.append(stats["cos_i2t_t2i"])
                visit_delta.append(stats["delta"])
                del g_i, g_t, g_p, l_probe, p_clean, p_noisy

            a = controller.alpha if controller is not None else args.alpha
            loss = a * li + (1 - a) * lt
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            f_li, f_lt = li.detach().item(), lt.detach().item()
            total_i.append(f_li)
            total_t.append(f_lt)
            total_loss.append(loss.detach().item())
            alphas.append(a)
            step_rows.append({
                "step": global_step, "epoch": epoch,
                "loss_i2t": round(f_li, 7),
                "loss_t2i": round(f_lt, 7),
                "loss_total": round(loss.detach().item(), 7),
                "alpha": round(a, 7),
                "probe_visit": 1 if stats else 0,
                "cos_probe_i2t": round(stats["cos_probe_i2t"], 6) if stats else "",
                "cos_probe_t2i": round(stats["cos_probe_t2i"], 6) if stats else "",
                "cos_i2t_t2i": round(stats["cos_i2t_t2i"], 6) if stats else "",
                "ema_delta": round(stats["ema_delta"], 6) if stats else "",
            })

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        val = evaluate(model, val_loader, text_feats, device, amp)
        epoch_rows.append({
            "epoch": epoch,
            "train_i2t": round(float(np.mean(total_i)), 6),
            "train_t2i": round(float(np.mean(total_t)), 6),
            "train_total": round(float(np.mean(total_loss)), 6),
            "alpha_mean": round(float(np.mean(alphas)), 6),
            "alpha_min": round(float(np.min(alphas)), 6),
            "alpha_max": round(float(np.max(alphas)), 6),
            "probe_visits": len(visit_delta) if controller else 0,
            "cos_i2t_t2i_mean": round(float(np.mean(visit_cos)), 6) if visit_cos else "",
            "delta_mean": round(float(np.mean(visit_delta)), 6) if visit_delta else "",
            "val_acc_i2t": val["acc_i2t"],
            "val_map_t2i": val["map_t2i"],
            "val_score": val["score"],
            "seconds": round(elapsed, 2),
        })
        extra = ""
        if controller:
            extra = (f" cos(i,t)={np.mean(visit_cos):.3f}" if visit_cos else "")
        print(f"Epoch {epoch}: alpha={np.mean(alphas):.3f} "
              f"I2T={np.mean(total_i):.4f} T2I={np.mean(total_t):.4f} "
              f"val_acc={val['acc_i2t']:.4f} val_mAP={val['map_t2i']:.4f} "
              f"time={elapsed:.1f}s{extra}")

        if val["score"] > best_val["score"]:
            best_val, best_epoch = val, epoch
            best_state = trainable_snapshot(model)

    checkpoint_path = output / "best.pt"
    torch.save({
        "model_name": args.model,
        "pretrained": args.pretrained,
        "trainable_state": best_state,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "prompts": prompts,
    }, checkpoint_path)
    write_csv(output / "steps.csv", step_rows)
    write_csv(output / "epochs.csv", epoch_rows)
    plot_logs(output, step_rows, epoch_rows)
    plot_alignment(output, step_rows)

    summary = {
        "method": "probe_meta" if args.controller == "probe" else "fixed",
        "seed": args.seed,
        "alpha_fixed": args.alpha if args.controller == "fixed" else None,
        "tau": None,
        "best_epoch": best_epoch,
        "pretrained_validation": initial_val,
        "best_validation": best_val,
        "checkpoint": str(checkpoint_path),
        "settings": vars(args),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Saved:", output)
    print("Best validation epoch:", best_epoch, best_val)
    if controller:
        print(f"Final alpha={controller.alpha:.4f} after {controller.visits} visits")
    if args.max_train_batches:
        print("WARNING: --max-train-batches was set; this run is a pipeline "
              "check only and must NOT be used for comparisons.")


if __name__ == "__main__":
    a = parse_args()
    if a.self_test:
        self_test()
    else:
        run(a)