# CLIP robustness experiment

- Dataset: CIFAR-10 val split; seed=43; 500 images per class; 5000 total images.
- Checkpoints: 6; source: runs_s43.
- Conditions: clean, noise1, noise2, blur1, blur2.
- Perturbations are deterministic synthetic noise/blur on original 32px RGB BEFORE CLIP preprocessing; they are NOT official CIFAR-10-C conditions.
- Original test split should only be used after method selection.
- Accuracy and text-query mAP are measured on the same samples for each run.
- Positive delta means the candidate is better than the fixed alpha=0.5 baseline.
- McNemar p-values assess paired image classification disagreements, NOT retrieval mAP, and are unadjusted for multiple tests.
- If per-class <500 on the validation split, clean scores need not match the original 5000-image summary; subset estimates can be noisy.
- Model checkpoint base is loaded once; fine-tuned visual parameters are overlaid separately for every run; text tower is the same frozen base.

## Output files

- `robustness_summary.csv`: main aggregate table.
- `paired_vs_fixed_05.csv`: same-image baseline comparison, wins/losses and McNemar.
- `per_image_predictions.csv`: image-wise classifications for further analysis.
- `00_example_corruptions.png`: inspect perturbation strengths.
- `01...06...png`: noise/blur performance and baseline delta charts.
