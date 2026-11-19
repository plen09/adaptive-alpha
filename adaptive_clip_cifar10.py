#!/usr/bin/env python3
"""Adaptive-alpha CLIP ViT-B/32 experiment on CIFAR-10.

CIFAR-10 has category labels, not unique image captions. We therefore use ten
unique class text prototypes and a multi-positive text-to-image objective.
This is supervised CLIP adaptation, NOT an exact reproduction of CLIP pretraining.
"""

import argparse
import csv
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import open_clip


CLASS_ALIASES = {"automobile": "car"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["zero_shot", "fixed", "adaptive", "evaluate"],
                   default="adaptive")
    p.add_argument("--model", default="ViT-B-32-quickgelu")
    p.add_argument("--pretrained", default="./weights/open_clip_model.safetensors",
                   help="OpenCLIP pretrained tag or path to a compatible checkpoint.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./runs")
    p.add_argument("--checkpoint", default=None,
                   help="Best checkpoint path, required for --mode evaluate.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--train-blocks", type=int, default=2,
                   help="Unfreeze this many final visual transformer blocks.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Fixed image-to-text weight for --mode fixed.")
    p.add_argument("--tau", type=float, default=1.0,
                   help="Adaptive weight sensitivity; smaller => more aggressive.")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--loss-ema", type=float, default=0.98,
                   help="Momentum of EMA for per-direction losses.")
    p.add_argument("--alpha-update", type=float, default=0.05,
                   help="EMA update rate for alpha itself.")
    p.add_argument("--min-alpha", type=float, default=0.2)
    p.add_argument("--max-alpha", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--eval-test", action="store_true",
                   help="Only enable AFTER hyperparameters/method are chosen on val.")
    p.add_argument("--max-train-batches", type=int, default=None,
                   help="For quick debugging only; do not use for formal comparisons.")
    args = p.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        p.error("--alpha must be in [0,1].")
    if args.tau <= 0 or not 0 < args.loss_ema < 1:
        p.error("--tau must be positive and --loss-ema must be in (0,1).")
    if args.warmup_steps < 1 or not 0 < args.alpha_update <= 1:
        p.error("--warmup-steps >= 1 and --alpha-update in (0,1] are required.")
    if not 0 <= args.min_alpha < args.max_alpha <= 1:
        p.error("Invalid alpha clipping bounds.")
    if args.mode == "evaluate" and args.checkpoint is None:
        p.error("--checkpoint is required for --mode evaluate.")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(args, preprocess_val, device):
    # Slight augmentation at ORIGINAL 32px, before resizing for CLIP.
    train_preprocess = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        preprocess_val,
    ])
    train_set = datasets.CIFAR10(args.data_dir, train=True,
                                 download=True, transform=train_preprocess)
    val_set = datasets.CIFAR10(args.data_dir, train=True,
                               download=False, transform=preprocess_val)
    test_set = datasets.CIFAR10(args.data_dir, train=False,
                                download=True, transform=preprocess_val)

    rng = np.random.default_rng(args.seed)
    all_labels = np.asarray(train_set.targets)
    train_indices, val_indices = [], []
    for label in range(len(train_set.classes)):
        indices = rng.permutation(np.where(all_labels == label)[0])
        val_indices.extend(indices[:500].tolist())
        train_indices.extend(indices[500:].tolist())

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    train_loader = DataLoader(
        Subset(train_set, train_indices),
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_kw,
    )
    val_loader = DataLoader(Subset(val_set, val_indices), shuffle=False, **loader_kw)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kw)
    return train_loader, val_loader, test_loader, train_set.classes


def class_text_features(model, model_name, class_names, device):
    tokenizer = open_clip.get_tokenizer(model_name)
    prompts = [f"a photo of a {CLASS_ALIASES.get(name, name)}."
               for name in class_names]
    tokens = tokenizer(prompts).to(device)
    model.eval()
    with torch.no_grad():
        feats = model.encode_text(tokens).float()
        feats = F.normalize(feats, dim=-1)
    return feats, prompts


def image_text_logits(model, images, text_feats):
    im_feats = F.normalize(model.encode_image(images).float(), dim=-1)
    # Use fp32 for logits, logsumexp, and KL, even when visual forward uses AMP.
    scale = model.logit_scale.exp().detach().float()
    return scale * (im_feats @ text_feats.T.float())


def paired_direction_losses(logits, labels):
    """I2T: 10-way cross entropy.

    T2I: per-class KL(Uniform(positive images) || Softmax(all batch images)).
    Subtract log(number of positive images) so the attainable minimum is 0.
    """
    n_classes = logits.shape[1]
    loss_i = F.cross_entropy(logits, labels)

    count = torch.bincount(labels, minlength=n_classes)
    present = count > 0
    pos_scores = logits.gather(1, labels[:, None]).squeeze(1)
    sums = torch.zeros(n_classes, device=logits.device, dtype=logits.dtype)
    sums = sums.scatter_add(0, labels, pos_scores)

    log_partition = torch.logsumexp(logits, dim=0)
    count_pos = count[present].to(logits.dtype)
    loss_t_per_class = (
        log_partition[present]
        - sums[present] / count_pos
        - count_pos.log()
    )
    loss_t = loss_t_per_class.mean()
    return loss_i, loss_t


class AdaptiveAlpha:
    """Detached, GPU-side EMA controller for relative loss progress."""

    def __init__(self, device, tau, warmup, loss_ema,
                 alpha_update, min_alpha, max_alpha):
        self.alpha = torch.tensor(0.5, device=device)
        self.ema_i = None
        self.ema_t = None
        self.ref_sum_i = torch.zeros((), device=device)
        self.ref_sum_t = torch.zeros((), device=device)
        self.ref_i = None
        self.ref_t = None
        self.steps = 0
        self.tau = tau
        self.warmup = warmup
        self.loss_ema = loss_ema
        self.alpha_update = alpha_update
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    @torch.no_grad()
    def update(self, li, lt):
        li, lt = li.detach(), lt.detach()
        if self.ema_i is None:
            self.ema_i, self.ema_t = li.clone(), lt.clone()
        else:
            self.ema_i.mul_(self.loss_ema).add_(li, alpha=1-self.loss_ema)
            self.ema_t.mul_(self.loss_ema).add_(lt, alpha=1-self.loss_ema)

        self.steps += 1
        if self.steps <= self.warmup:
            self.ref_sum_i.add_(li)
            self.ref_sum_t.add_(lt)
            if self.steps == self.warmup:
                self.ref_i = (self.ref_sum_i / self.warmup).clamp_min(1e-8)
                self.ref_t = (self.ref_sum_t / self.warmup).clamp_min(1e-8)
            return self.alpha.detach().clone()

        rel_i = (self.ema_i / self.ref_i).clamp_min(1e-8)
        rel_t = (self.ema_t / self.ref_t).clamp_min(1e-8)
        target = torch.sigmoid((rel_i.log() - rel_t.log()) / self.tau)
        self.alpha.lerp_(target, self.alpha_update)
        self.alpha.clamp_(self.min_alpha, self.max_alpha)
        return self.alpha.detach().clone()


@torch.inference_mode()
def evaluate(model, loader, text_feats, device, amp):
    """I2T top-1 accuracy; T2I mean average precision over ten class queries."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        ctx = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
        with ctx:
            logits = image_text_logits(model, images, text_feats)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    scores = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    top1 = (scores.argmax(dim=1) == labels).float().mean().item()
    aps = []
    for c in range(scores.shape[1]):
        order = scores[:, c].argsort(descending=True)
        relevant = (labels[order] == c).float()
        ranks = torch.arange(1, len(relevant)+1, dtype=torch.float32)
        precision = relevant.cumsum(dim=0) / ranks
        ap = (precision * relevant).sum() / relevant.sum().clamp_min(1)
        aps.append(ap.item())
    mean_ap = float(np.mean(aps))
    return {
        "acc_i2t": round(top1, 6),
        "map_t2i": round(mean_ap, 6),
        "score": round((top1 + mean_ap)/2, 6),
    }


def select_trainable(model, train_blocks):
    for p in model.parameters():
        p.requires_grad_(False)
    visual = model.visual
    blocks = visual.transformer.resblocks
    if train_blocks < 0 or train_blocks > len(blocks):
        raise ValueError(f"--train-blocks must be 0..{len(blocks)}")
    if train_blocks > 0:
        for p in blocks[-train_blocks:].parameters():
            p.requires_grad_(True)
    for p in visual.ln_post.parameters():
        p.requires_grad_(True)
    if visual.proj is not None:
        if isinstance(visual.proj, torch.nn.Parameter):
            visual.proj.requires_grad_(True)
        else:
            for p in visual.proj.parameters():
                p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def trainable_snapshot(model):
    return {name: p.detach().cpu().clone()
            for name, p in model.named_parameters() if p.requires_grad}


def restore_snapshot(model, snapshot):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in snapshot.items():
            if name not in params:
                raise KeyError(f"Checkpoint parameter not in model: {name}")
            params[name].copy_(tensor.to(params[name].device,
                                          dtype=params[name].dtype))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_logs(out_path, step_rows, epoch_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    if step_rows:
        plt.figure(figsize=(8, 4))
        plt.plot([r["step"] for r in step_rows],
                 [r["alpha"] for r in step_rows], linewidth=1.4)
        plt.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        plt.xlabel("Training step")
        plt.ylabel("Image-to-text weight (alpha)")
        plt.title("Adaptive alpha trajectory")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(out_path / "alpha_curve.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot([r["epoch"] for r in epoch_rows],
             [r["val_acc_i2t"] for r in epoch_rows], marker="o", label="I2T accuracy")
    plt.plot([r["epoch"] for r in epoch_rows],
             [r["val_map_t2i"] for r in epoch_rows], marker="o", label="T2I mAP")
    plt.xlabel("Epoch")
    plt.ylabel("Validation metric")
    plt.title("Validation performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "validation_curve.png", dpi=160)
    plt.close()


def run(args):
    set_seed(args.seed)
    device = pick_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    print(f"Device={device}; AMP={amp}; model={args.model}; pretrained={args.pretrained}")

    # Official OpenCLIP notes: OpenAI ViT-B/32 uses QuickGELU.
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model = model.to(device)
    model.eval()
    train_loader, val_loader, test_loader, classes = make_loaders(
        args, preprocess_val, device
    )
    text_feats, prompts = class_text_features(model, args.model, classes, device)
    print("Class prompts:", prompts)

    if args.mode == "evaluate":
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if ckpt["model_name"] != args.model:
            raise ValueError("Checkpoint --model differs from current --model.")
        if ckpt["pretrained"] != args.pretrained:
            raise ValueError("Checkpoint --pretrained differs from current value.")
        restore_snapshot(model, ckpt["trainable_state"])
        val_metrics = evaluate(model, val_loader, text_feats, device, amp)
        test_metrics = evaluate(model, test_loader, text_feats, device, amp)
        print(json.dumps({"validation": val_metrics, "test": test_metrics},
                         indent=2, ensure_ascii=False))
        return

    if args.mode == "zero_shot":
        results = {"validation": evaluate(model, val_loader, text_feats, device, amp)}
        if args.eval_test:
            results["test"] = evaluate(model, test_loader, text_feats, device, amp)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    params = select_trainable(model, args.train_blocks)
    print("Trainable parameters:", f"{sum(p.numel() for p in params):,}")
    optimizer = torch.optim.AdamW(params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    controller = None
    if args.mode == "adaptive":
        controller = AdaptiveAlpha(
            device, args.tau, args.warmup_steps, args.loss_ema,
            args.alpha_update, args.min_alpha, args.max_alpha
        )

    label = (f"fixed_a{args.alpha:.2f}_s{args.seed}"
             if args.mode == "fixed"
             else f"adaptive_tau{args.tau:.2f}_s{args.seed}")
    output = Path(args.out_dir) / label
    output.mkdir(parents=True, exist_ok=True)

    initial_val = evaluate(model, val_loader, text_feats, device, amp)
    best_val = initial_val
    best_epoch = 0
    best_state = trainable_snapshot(model)
    epoch_rows = [{
        "epoch": 0, "train_i2t": "", "train_t2i": "", "train_total": "",
        "alpha_mean": "", "alpha_min": "", "alpha_max": "",
        "val_acc_i2t": initial_val["acc_i2t"],
        "val_map_t2i": initial_val["map_t2i"],
        "val_score": initial_val["score"], "seconds": 0.0,
    }]
    step_rows = []
    global_step = 0
    print(f"Epoch 0 (pretrained): {initial_val}")

    for epoch in range(1, args.epochs+1):
        model.train()
        total_i, total_t, total_loss = [], [], []
        alphas = []
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
                if controller is not None:
                    a = controller.update(li, lt)
                else:
                    a = args.alpha
                loss = a * li + (1-a) * lt

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            f_li, f_lt, f_loss = li.detach().item(), lt.detach().item(), loss.detach().item()
            f_alpha = a.item() if torch.is_tensor(a) else float(a)
            total_i.append(f_li)
            total_t.append(f_lt)
            total_loss.append(f_loss)
            alphas.append(f_alpha)
            step_rows.append({
                "step": global_step, "epoch": epoch,
                "loss_i2t": round(f_li, 7),
                "loss_t2i": round(f_lt, 7),
                "loss_total": round(f_loss, 7),
                "alpha": round(f_alpha, 7),
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
            "val_acc_i2t": val["acc_i2t"],
            "val_map_t2i": val["map_t2i"],
            "val_score": val["score"],
            "seconds": round(elapsed, 2),
        })
        print(f"Epoch {epoch}: alpha={np.mean(alphas):.3f} "
              f"I2T={np.mean(total_i):.4f} T2I={np.mean(total_t):.4f} "
              f"val_acc={val['acc_i2t']:.4f} val_mAP={val['map_t2i']:.4f} "
              f"time={elapsed:.1f}s")

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

    summary = {
        "method": args.mode,
        "seed": args.seed,
        "alpha_fixed": args.alpha if args.mode == "fixed" else None,
        "tau": args.tau if args.mode == "adaptive" else None,
        "best_epoch": best_epoch,
        "pretrained_validation": initial_val,
        "best_validation": best_val,
        "checkpoint": str(checkpoint_path),
        "settings": vars(args),
    }
    if args.eval_test:
        print("WARNING: test results must not be used for model/hyperparameter selection.")
        restore_snapshot(model, best_state)
        summary["test"] = evaluate(model, test_loader, text_feats, device, amp)
    with (output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Saved:", output)
    print("Best validation epoch:", best_epoch, best_val)


if __name__ == "__main__":
    run(parse_args())
