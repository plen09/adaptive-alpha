#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate saved Adaptive-alpha CLIP ViT-B/32 checkpoints on corrupted CIFAR-10.

Matches the previous adaptive_clip_cifar10.py:
- seed=42 stratified validation indices (500/class from CIFAR-10 TRAIN set)
- same 10 class prompts ("automobile" -> "car")
- saved `best.pt` contains only fine-tuned parameters, so load the ORIGINAL base
  pretrained checkpoint first, then overlay `trainable_state`.
- I2T top-1, T2I class-query mAP.

Corruptions are deterministic synthetic Gaussian noise and Gaussian blur at 32px.
They are NOT the official CIFAR-10-C benchmark.

Example:
    python evaluate_clip_robustness.py --check-only
    python evaluate_clip_robustness.py --runs-dir runs --per-class 50
    python evaluate_clip_robustness.py --runs-dir runs --per-class 500
"""

import argparse
import csv
import json
import math
import os
import warnings
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset


SUPPORTED = ("clean", "noise1", "noise2", "blur1", "blur2")
PROMPT_ALIASES = {"automobile": "car"}
NOISE_SIGMA = {"noise1": 12.0, "noise2": 26.0}  # pixel intensity / 255 at native 32px
BLUR_RADIUS = {"blur1": 0.6, "blur2": 1.2}      # PIL GaussianBlur radius at native 32px

SUMMARY_COLUMNS = [
    "run", "method", "group", "seed", "split", "condition",
    "n_images", "acc_i2t", "error_count_i2t", "map_t2i", "combined_score",
    "acc_drop_vs_clean_pp", "map_drop_vs_clean_pp",
    "original_clean_acc", "original_clean_map",
]
PAIRED_COLUMNS = [
    "condition", "baseline", "candidate", "n_images",
    "baseline_acc", "candidate_acc", "delta_acc_pp",
    "baseline_map", "candidate_map", "delta_map_pp",
    "baseline_wrong_candidate_right", "baseline_right_candidate_wrong",
    "mcnemar_exact_p_two_sided",
]
PER_IMAGE_COLUMNS = [
    "run", "condition", "split", "sample_index", "true_label",
    "predicted_label", "correct",
]


def args_parser():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"),
                   help="Directory containing experiment subdirs with best.pt and summary.json.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", type=Path, default=Path("robustness_results"))
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--per-class", type=int, default=50,
                   help="Use 50 for a quick trial; 500 for complete validation.")
    p.add_argument("--conditions", nargs="+", choices=SUPPORTED,
                   default=list(SUPPORTED))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42,
                   help="Must match original training split seed (42 for your runs).")
    p.add_argument("--pretrained", default=None,
                   help="Override original base weights; e.g. ./weights/open_clip_model.safetensors")
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--check-only", action="store_true",
                   help="List checkpoints without downloading a dataset or loading a model.")
    p.add_argument("--self-test", action="store_true",
                   help="Test evaluation math/corruption determinism with synthetic data.")
    return p.parse_args()


def discover(runs_dir):
    runs = []
    for meta_path in sorted(runs_dir.rglob("summary.json")):
        d = meta_path.parent
        try:
            summary = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print("WARN: cannot read", meta_path, exc)
            continue
        if not isinstance(summary, dict) or "best_validation" not in summary:
            continue
        runs.append({
            "dir": d,
            "run": str(d.relative_to(runs_dir)).replace("\\", "/"),
            "checkpoint": d / "best.pt",
            "summary": summary,
        })
    return runs


def maybe_check(runs):
    print(f"Found {len(runs)} experiment summaries.")
    for r in runs:
        present = "OK" if r["checkpoint"].is_file() else "MISSING"
        print(f"  [{present:7}] {r['run']}/best.pt")
    available = [r for r in runs if r["checkpoint"].is_file()]
    print(f"Ready to evaluate: {len(available)}")
    if not available:
        print("No best.pt files found. The previously uploaded run_s42.zip contains logs "
              "and PNGs only. Use the ORIGINAL local runs/ directory where training "
              "wrote best.pt, or copy the checkpoints into the matching subfolders.")
    return available


def choose_device(arg):
    if arg != "auto":
        device = torch.device(arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def chosen_indices(all_labels, split, per_class, seed, nclasses=10):
    """For val: EXACT same 500/class split as the original training program."""
    if per_class <= 0:
        raise ValueError("--per-class must be positive")
    labels = np.asarray(all_labels)
    if split == "val":
        if per_class > 500:
            raise ValueError("Validation split has 500 images per class; use --per-class <= 500.")
        rng = np.random.default_rng(seed)
        result = []
        for c in range(nclasses):
            indices = rng.permutation(np.where(labels == c)[0])
            # Important: always first select the exact original 500 val samples.
            result.extend(indices[:500][:per_class].tolist())
        return result

    if per_class > 1000:
        raise ValueError("CIFAR-10 test split has 1000 images per class.")
    return [int(i) for c in range(nclasses)
            for i in np.where(labels == c)[0][:per_class]]


def corrupted_pil(image, condition, original_index, seed):
    """Deterministic corruption applied BEFORE OpenCLIP resize/normalization."""
    if condition == "clean":
        return image
    if condition in BLUR_RADIUS:
        return image.filter(ImageFilter.GaussianBlur(BLUR_RADIUS[condition]))
    if condition in NOISE_SIGMA:
        # local RNG (NOT global random state); repeated across all model checkpoints
        key = (int(seed)*1000003 + int(original_index)*9176 +
               (1 if condition == "noise1" else 2)*97409) % (2**63-1)
        rng = np.random.default_rng(key)
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        noise = rng.normal(loc=0.0, scale=NOISE_SIGMA[condition],
                           size=rgb.shape).astype(np.float32)
        rgb = np.uint8(np.clip(np.rint(rgb + noise), 0, 255))
        return Image.fromarray(rgb, mode="RGB")
    raise ValueError(f"Unknown condition: {condition}")


class CorruptedSubset(Dataset):
    def __init__(self, raw_cifar, indices, preprocess, condition, seed):
        self.raw_cifar = raw_cifar
        self.indices = [int(i) for i in indices]
        self.preprocess = preprocess
        self.condition = condition
        self.seed = seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos):
        original_idx = self.indices[pos]
        image, label = self.raw_cifar[original_idx]
        image = corrupted_pil(image, self.condition, original_idx, self.seed)
        return self.preprocess(image), int(label), original_idx


def restore_checkpoint(model, record, expected_keys=None):
    """Only overlay fine-tuned weights. Text tower and other frozen parts stay at base."""
    checkpoint = torch.load(record["checkpoint"], map_location="cpu", weights_only=True)
    meta = record["summary"].get("settings", {})
    checkpoint_model = checkpoint.get("model_name")
    if checkpoint_model != meta.get("model"):
        raise ValueError(f"Model mismatch in {record['checkpoint']}: "
                         f"best.pt={checkpoint_model!r}, summary={meta.get('model')!r}")
    trained = checkpoint["trainable_state"]
    keys = set(trained)
    if expected_keys is not None and keys != expected_keys:
        raise RuntimeError("Runs have different fine-tuned parameter sets; "
                           "cannot safely reuse the same base model.")
    named = dict(model.named_parameters())
    extra = keys - set(named)
    if extra:
        raise KeyError(f"Unexpected checkpoint parameter names: {list(extra)[:5]}")
    with torch.no_grad():
        for name, value in trained.items():
            dest = named[name]
            if tuple(value.shape) != tuple(dest.shape):
                raise ValueError(f"Shape mismatch for {name}: "
                                 f"{tuple(value.shape)} vs {tuple(dest.shape)}")
            dest.copy_(value.to(device=dest.device, dtype=dest.dtype))
    model.eval()
    return keys


def prompts_for_classes(names):
    return [f"a photo of a {PROMPT_ALIASES.get(name, name)}." for name in names]


@torch.inference_mode()
def compute_text_embeddings(model, model_name, classes, device, open_clip):
    tokenizer = open_clip.get_tokenizer(model_name)
    text = tokenizer(prompts_for_classes(classes)).to(device)
    embeddings = model.encode_text(text).float()
    return F.normalize(embeddings, dim=-1)


@torch.inference_mode()
def score_loader(model, loader, text_features, device, use_amp):
    model.eval()
    logits_all, labels_all, idx_all = [], [], []
    scale = model.logit_scale.exp().detach().float()
    for image, label, idx in loader:
        image = image.to(device, non_blocking=True)
        ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
        with ctx:
            emb = F.normalize(model.encode_image(image).float(), dim=-1)
            scores = scale * (emb @ text_features.T)
        logits_all.append(scores.float().cpu())
        labels_all.append(label.cpu())
        idx_all.append(idx.cpu())
    if not logits_all:
        raise RuntimeError("No samples in dataloader.")
    return (torch.cat(logits_all).numpy(), torch.cat(labels_all).numpy(),
            torch.cat(idx_all).numpy())


def metrics(scores, labels):
    """Mirrors original script: I2T acc and T2I class-query mAP."""
    scores = torch.from_numpy(np.asarray(scores)).float()
    labels = torch.from_numpy(np.asarray(labels)).long()
    pred = scores.argmax(dim=1)
    top1 = (pred == labels).float().mean().item()
    aps = []
    for c in range(scores.shape[1]):
        order = scores[:, c].argsort(descending=True)
        relevant = (labels[order] == c).float()
        precision = relevant.cumsum(0) / torch.arange(
            1, len(relevant)+1, dtype=torch.float32)
        aps.append(((precision * relevant).sum() /
                   relevant.sum().clamp_min(1)).item())
    map_value = float(np.mean(aps))
    return {"acc_i2t": top1, "error_count_i2t": int((pred != labels).sum().item()),
            "map_t2i": map_value, "combined_score": (top1 + map_value)/2,
            "pred": pred.numpy()}


def mcnemar_exact_p(wins, losses):
    """Two-sided exact McNemar (conditional Binomial(n, 0.5)), no scipy needed."""
    n = int(wins + losses)
    if n == 0:
        return 1.0
    k = min(int(wins), int(losses))
    log_terms = [
        math.lgamma(n+1) - math.lgamma(j+1) - math.lgamma(n-j+1)
        - n*math.log(2)
        for j in range(k+1)
    ]
    pivot = max(log_terms)
    tail = math.exp(pivot) * sum(math.exp(t - pivot) for t in log_terms)
    return min(1.0, 2.0*tail)


def write_table(path, rows, cols):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def draw_curves(summary_rows, outdir, metric_name, corruption, filename):
    # In each plot, include the clean point as zero severity, then both levels.
    conditions = ("clean", f"{corruption}1", f"{corruption}2")
    groups = defaultdict(dict)
    for row in summary_rows:
        groups[row["run"]][row["condition"]] = row
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    drawn = False
    for run, by_condition in sorted(groups.items()):
        if all(k in by_condition for k in conditions):
            values = [float(by_condition[k][metric_name]) for k in conditions]
            ax.plot(range(len(conditions)), values, marker="o", linewidth=1.5,
                    label=run.split("/")[-1])
            drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.set_xticks(range(3), ["Clean", f"{corruption.capitalize()} mild",
                              f"{corruption.capitalize()} strong"])
    ax.set_ylabel("I2T accuracy" if metric_name == "acc_i2t" else "T2I mAP")
    ax.set_title(f"{'I2T accuracy' if metric_name == 'acc_i2t' else 'T2I mAP'} "
                 f"under {corruption}")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01,1), fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_delta_heatmap(paired_rows, outdir, key, filename):
    if not paired_rows:
        return
    conditions = [k for k in SUPPORTED if any(r["condition"] == k for r in paired_rows)]
    runs = sorted(set(r["candidate"] for r in paired_rows))
    values = np.full((len(runs), len(conditions)), np.nan, dtype=float)
    for r in paired_rows:
        if r["candidate"] in runs and r["condition"] in conditions:
            values[runs.index(r["candidate"]), conditions.index(r["condition"])] = r[key]
    if not np.any(np.isfinite(values)):
        return
    fig, ax = plt.subplots(figsize=(max(8, len(conditions)*1.6),
                                    max(3.5, len(runs)*0.75+1.8)))
    im = ax.imshow(np.ma.masked_invalid(values), aspect="auto")
    ax.set_xticks(range(len(conditions)), conditions)
    ax.set_yticks(range(len(runs)), [r.split("/")[-1] for r in runs])
    ax.set_title(("I2T accuracy" if key == "delta_acc_pp" else "T2I mAP") +
                 " delta vs fixed alpha=0.5 (pp)")
    for i in range(len(runs)):
        for j in range(len(conditions)):
            if math.isfinite(values[i,j]):
                ax.text(j, i, f"{values[i,j]:+.2f}", ha="center",
                        va="center", fontsize=9, color="white")
    fig.colorbar(im, ax=ax, label="Candidate minus baseline (percentage points)")
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_corruption_examples(raw, indices, outdir, seed):
    if not indices:
        return
    index = indices[0]
    image, label = raw[index]
    fig, axs = plt.subplots(1, len(SUPPORTED), figsize=(10,2.3))
    for ax, cond in zip(axs, SUPPORTED):
        ax.imshow(corrupted_pil(image, cond, index, seed))
        ax.set_title(cond, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Corruptions on the same 32x32 source image (class {label})",fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "00_example_corruptions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def self_test():
    im = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
    a = np.asarray(corrupted_pil(im, "noise1", 123, 42))
    b = np.asarray(corrupted_pil(im, "noise1", 123, 42))
    c = np.asarray(corrupted_pil(im, "noise2", 123, 42))
    assert np.array_equal(a, b)
    assert np.mean(np.abs(a.astype(float)-128)) < np.mean(np.abs(c.astype(float)-128))
    assert np.array_equal(np.asarray(corrupted_pil(im,"clean",123,42)),np.asarray(im))

    # For class c, each correct sample has higher score in its own text column.
    labels = np.repeat(np.arange(10), 4)
    scores = np.full((40,10), -3.0, dtype=np.float32)
    scores[np.arange(40),labels] = 3.0
    mm = metrics(scores, labels)
    assert abs(mm["acc_i2t"]-1.0) < 1e-6 and mm["map_t2i"] > 0.99
    assert abs(mcnemar_exact_p(5, 0)-0.0625) < 1e-10
    # Validate reproducible nested validation split.
    faux_labels = np.repeat(np.arange(10), 5000)
    idx_50 = chosen_indices(faux_labels, "val", 50, 42)
    idx_500 = chosen_indices(faux_labels, "val", 500, 42)
    assert len(idx_50)==500 and len(idx_500)==5000
    for c in range(10):
        assert idx_50[c*50:(c+1)*50] == idx_500[c*500:c*500+50]
    print("SELF-TEST PASSED: corruption, metrics, exact paired test, split reproducibility.")


def run():
    args = args_parser()
    if args.self_test:
        self_test()
        return
    runs = discover(args.runs_dir)
    available = maybe_check(runs)
    if args.check_only:
        return
    if not available:
        raise SystemExit("\nNo saved model weights: the analysis cannot reconstruct "
                         "best.pt from the previous CSV/PNG logs.")
    if len(available) < len(runs):
        print(f"WARN: {len(runs)-len(available)} runs missing checkpoints and will be skipped.")
    available.sort(key=lambda r:r["run"])
    # Keep exact same model, seed, text prompts, and base weights across all six runs.
    ref = available[0]["summary"].get("settings") or {}
    model_name = ref.get("model", "ViT-B-32-quickgelu")
    original_base = ref.get("pretrained", "openai")
    for r in available:
        st = r["summary"].get("settings") or {}
        for field in ("model", "pretrained", "seed"):
            if st.get(field) != ref.get(field):
                raise ValueError(f"Runs have different {field}: "
                                 f"{available[0]['run']} vs {r['run']}; "
                                 "evaluate them separately.")
        if st.get("seed") != args.seed:
            raise ValueError(f"--seed {args.seed} differs from training "
                             f"seed {st.get('seed')}.")
    base = args.pretrained if args.pretrained is not None else original_base
    if (base.endswith(".safetensors") or base.endswith(".pt") or base.endswith(".pth")) \
            and not Path(base).exists():
        raise FileNotFoundError(
            f"Base CLIP weights missing: {base}\n"
            "Your summaries reference './weights/open_clip_model.safetensors'. "
            "Use --pretrained /full/path/to/the/same/original/base_weights.safetensors"
        )

    try:
        import open_clip
        from torchvision import datasets
    except ImportError as exc:
        raise SystemExit("Please install dependencies: "
                         "pip install open_clip_torch torchvision numpy matplotlib") from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    use_amp = (device.type=="cuda" and not args.no_amp)
    print(f"Device: {device} | AMP: {use_amp} | Dataset split: {args.split} | "
          f"per class: {args.per_class}")
    print("Pretrained base:", base)
    print("Model:", model_name)
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        model_name, pretrained=base
    )
    model = model.to(device).eval()
    raw = datasets.CIFAR10(args.data_dir, train=(args.split=="val"),
                           download=True, transform=None)
    indices = chosen_indices(raw.targets, args.split, args.per_class, args.seed,
                             nclasses=len(raw.classes))
    labels_expected = np.asarray(raw.targets)[indices]
    print(f"Selected {len(indices)} images, labels/class:", args.per_class)
    if "clean" not in args.conditions:
        args.conditions.insert(0, "clean")
        print("Included clean automatically for robustness delta computation.")
    draw_corruption_examples(raw, indices, args.out_dir, args.seed)

    texts = compute_text_embeddings(model, model_name, raw.classes, device, open_clip)
    all_rows = []
    per_image_rows = []
    results = {}
    use_keyset = None

    for r in available:
        use_keyset = restore_checkpoint(model, r, expected_keys=use_keyset)
        print(f"\nExperiment: {r['run']}")
        for cond in args.conditions:
            ds = CorruptedSubset(raw, indices, preprocess_val, cond, args.seed)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers,
                                pin_memory=(device.type == "cuda"),
                                persistent_workers=(args.workers>0))
            scores, labels, sample_indices = score_loader(model, loader, texts,
                                                           device, use_amp)
            if not np.array_equal(sample_indices, np.asarray(indices)):
                raise RuntimeError("Unexpected validation index ordering.")
            if not np.array_equal(labels, labels_expected):
                raise RuntimeError("Unexpected CIFAR-10 labels.")
            m = metrics(scores, labels)
            results[(r["run"],cond)] = {"metrics": m, "labels":labels,
                                       "ids":sample_indices, "pred":m["pred"]}
            summ = r["summary"]
            orig = summ.get("best_validation") or {}
            row = {
                "run":r["run"],"method":summ.get("method",""),
                "group":r["run"].split("/")[-1],
                "seed":args.seed,"split":args.split,
                "condition":cond, "n_images":len(indices),
                "acc_i2t":m["acc_i2t"],
                "error_count_i2t":m["error_count_i2t"],
                "map_t2i":m["map_t2i"],
                "combined_score":m["combined_score"],
                "original_clean_acc":orig.get("acc_i2t",""),
                "original_clean_map":orig.get("map_t2i",""),
            }
            all_rows.append(row)
            for original_id, true_label, pred in zip(sample_indices,labels,m["pred"]):
                per_image_rows.append({
                    "run":r["run"],"condition":cond,"split":args.split,
                    "sample_index":int(original_id),
                    "true_label":int(true_label),
                    "predicted_label":int(pred),
                    "correct":int(pred==true_label),
                })
            print(f"  {cond:7} | I2T {m['acc_i2t']*100:7.3f}% "
                  f"({m['error_count_i2t']} wrong) | "
                  f"T2I mAP {m['map_t2i']*100:7.3f}%")
            del loader, ds, scores

    clean = {r["run"]: results.get((r["run"],"clean")) for r in available}
    clean_metrics = {run:rec["metrics"] for run,rec in clean.items() if rec is not None}
    for row in all_rows:
        c = clean_metrics.get(row["run"])
        row["acc_drop_vs_clean_pp"] = (100*(c["acc_i2t"]-row["acc_i2t"])
                                        if c else math.nan)
        row["map_drop_vs_clean_pp"] = (100*(c["map_t2i"]-row["map_t2i"])
                                        if c else math.nan)

    baseline = [r["run"] for r in available
                if r["summary"].get("method")=="fixed"
                and abs(float(r["summary"].get("alpha_fixed",99))-.5)<1e-6]
    baseline = baseline[0] if len(baseline)==1 else None
    paired_rows = []
    if baseline is None:
        print("WARN: Cannot find a unique fixed alpha=0.5 baseline; "
              "paired comparison skipped.")
    else:
        for cond in args.conditions:
            b = results[(baseline,cond)]
            truth = b["labels"]
            bcorrect = b["pred"] == truth
            bm = b["metrics"]
            for r in available:
                if r["run"] == baseline:
                    continue
                c = results[(r["run"],cond)]
                if not np.array_equal(c["ids"],b["ids"]):
                    raise RuntimeError("Cannot do paired tests on different image IDs.")
                ccorrect = c["pred"] == truth
                wins = int(np.sum(~bcorrect & ccorrect))
                losses = int(np.sum(bcorrect & ~ccorrect))
                paired_rows.append({
                    "condition":cond,
                    "baseline":baseline,
                    "candidate":r["run"],
                    "n_images":len(truth),
                    "baseline_acc":bm["acc_i2t"],
                    "candidate_acc":c["metrics"]["acc_i2t"],
                    "delta_acc_pp":100*(c["metrics"]["acc_i2t"]-bm["acc_i2t"]),
                    "baseline_map":bm["map_t2i"],
                    "candidate_map":c["metrics"]["map_t2i"],
                    "delta_map_pp":100*(c["metrics"]["map_t2i"]-bm["map_t2i"]),
                    "baseline_wrong_candidate_right":wins,
                    "baseline_right_candidate_wrong":losses,
                    "mcnemar_exact_p_two_sided":mcnemar_exact_p(wins,losses),
                })

    write_table(args.out_dir/"robustness_summary.csv", all_rows, SUMMARY_COLUMNS)
    write_table(args.out_dir/"per_image_predictions.csv", per_image_rows,
                PER_IMAGE_COLUMNS)
    write_table(args.out_dir/"paired_vs_fixed_05.csv", paired_rows,
                PAIRED_COLUMNS)

    draw_curves(all_rows,args.out_dir,"acc_i2t","noise","01_noise_accuracy.png")
    draw_curves(all_rows,args.out_dir,"map_t2i","noise","02_noise_map.png")
    draw_curves(all_rows,args.out_dir,"acc_i2t","blur","03_blur_accuracy.png")
    draw_curves(all_rows,args.out_dir,"map_t2i","blur","04_blur_map.png")
    draw_delta_heatmap(paired_rows,args.out_dir,
                       "delta_acc_pp","05_delta_accuracy_heatmap.png")
    draw_delta_heatmap(paired_rows,args.out_dir,
                       "delta_map_pp","06_delta_map_heatmap.png")

    if args.split=="val" and args.per_class==500:
        for row in all_rows:
            if row["condition"]!="clean":continue
            ref_acc = row["original_clean_acc"]
            ref_map = row["original_clean_map"]
            if isinstance(ref_acc,(float,int)) and abs(ref_acc-row["acc_i2t"]) > 0.0005:
                warnings.warn(
                    f"{row['run']}: clean acc ({row['acc_i2t']:.6f}) is not close to "
                    f"original ({ref_acc:.6f}); check checkpoint or preprocessing."
                )
            if isinstance(ref_map,(float,int)) and abs(ref_map-row["map_t2i"]) > 0.0005:
                warnings.warn(
                    f"{row['run']}: clean mAP ({row['map_t2i']:.6f}) is not close to "
                    f"original ({ref_map:.6f}); check checkpoint or preprocessing."
                )

    note = (
        "# CLIP robustness experiment\n\n"
        f"- Dataset: CIFAR-10 {args.split} split; seed={args.seed}; "
        f"{args.per_class} images per class; {len(indices)} total images.\n"
        f"- Checkpoints: {len(available)}; source: {args.runs_dir}.\n"
        f"- Conditions: {', '.join(args.conditions)}.\n"
        "- Perturbations are deterministic synthetic noise/blur on original 32px RGB "
        "BEFORE CLIP preprocessing; they are NOT official CIFAR-10-C conditions.\n"
        "- Original test split should only be used after method selection.\n"
        "- Accuracy and text-query mAP are measured on the same samples for each run.\n"
        "- Positive delta means the candidate is better than the fixed alpha=0.5 baseline.\n"
        "- McNemar p-values assess paired image classification disagreements, "
        "NOT retrieval mAP, and are unadjusted for multiple tests.\n"
        "- If per-class <500 on the validation split, clean scores need not match the "
        "original 5000-image summary; subset estimates can be noisy.\n"
        "- Model checkpoint base is loaded once; fine-tuned visual parameters are overlaid "
        "separately for every run; text tower is the same frozen base.\n"
        "\n## Output files\n\n"
        "- `robustness_summary.csv`: main aggregate table.\n"
        "- `paired_vs_fixed_05.csv`: same-image baseline comparison, wins/losses and McNemar.\n"
        "- `per_image_predictions.csv`: image-wise classifications for further analysis.\n"
        "- `00_example_corruptions.png`: inspect perturbation strengths.\n"
        "- `01...06...png`: noise/blur performance and baseline delta charts.\n"
    )
    (args.out_dir/"README_results.md").write_text(note,encoding="utf-8")
    print(f"\nFinished: {len(all_rows)} run-condition evaluations "
          f"({len(indices)} images each)")
    print(f"All tables and charts saved to: {args.out_dir.resolve()}")


if __name__=="__main__":
    run()
