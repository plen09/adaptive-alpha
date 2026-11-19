# CLIP ViT-B/32 · CIFAR-10 无需重训的鲁棒性评估

这是针对之前 `adaptive_clip_cifar10.py` 的评估脚本，已对照原训练实现：
- seed=42、从官方 CIFAR-10 **训练集**分层选出的 5000 张 validation；
- 使用 **ViT-B-32-quickgelu** 和原始 `open_clip_model.safetensors` 权重；
- 使用原来的 10 条类别文本原型；
- 每组 `best.pt` 只覆盖微调的视觉塔参数；
- 计算 I2T Accuracy、10 类文本原型作为查询的 T2I mAP。

## 1. 安装

和训练时一样，保持现有 PyTorch / torchvision 环境即可：

```bash
pip install open_clip_torch numpy matplotlib pillow
```

## 2. 把脚本放在项目根目录

```text
project/
  evaluate_clip_robustness.py
  weights/
    open_clip_model.safetensors
  data/
  runs/
    fixed_a0.30_s42/
      summary.json
      best.pt
    fixed_a0.50_s42/
      summary.json
      best.pt
    fixed_a0.70_s42/
      summary.json
      best.pt
    adaptive_tau0.50_s42/
      summary.json
      best.pt
    adaptive_tau1.00_s42/
      summary.json
      best.pt
    adaptive_tau2.00_s42/
      summary.json
      best.pt
```

## 3. 三条命令

### 检查权重是否齐全（不加载模型、不下载数据集）

```bash
python evaluate_clip_robustness.py --check-only
```

### 快速试跑（每类 50 张、共 500 张）

```bash
python evaluate_clip_robustness.py --runs-dir runs --data-dir data --per-class 50 --batch-size 32
```

### 正式验证（每类 500 张、共 5000 张，严格复用原来的 val）

```bash
python evaluate_clip_robustness.py --runs-dir runs --data-dir data --per-class 500 --batch-size 64
```

如果你的权重文件不在当时的 `./weights/open_clip_model.safetensors`，可显式覆盖：

```bash
python evaluate_clip_robustness.py --runs-dir runs --per-class 500 --pretrained "/path/to/open_clip_model.safetensors"
```

该 `--pretrained` 必须是你当初六组训练共同使用的 **同一个底座权重文件**，
不能随意换成其他版本或随机初始化。

对特别吃力的设备：

```bash
python evaluate_clip_robustness.py --per-class 20 --batch-size 16 --workers 0 --conditions clean noise1 blur1
```

提示：较小的 subset 是快速观察，并非最终严谨结论。
本脚本默认 `--split val`。如最终确定了方法再做一次官方 test，可显式用：
`--split test --per-class 1000`。不要反复基于 test 结果调超参。

## 4. 图像扰动设置

原始图像大小是 32×32 像素，**先施加扰动，再做原始 CLIP 的 224px 预处理**：

| condition | 含义 |
|---|---|
| `clean` | 无扰动 |
| `noise1` | Gaussian noise, σ=12/255 |
| `noise2` | Gaussian noise, σ=26/255 |
| `blur1` | Gaussian blur, PIL radius=0.6 |
| `blur2` | Gaussian blur, PIL radius=1.2 |

噪声采样由“原始图像索引 + condition + seed”决定，
同一张图片在 6 个模型下的腐蚀结果完全一致。
这只是可控的**合成扰动实验**，不是官方 CIFAR-10-C benchmark。

## 5. 输出（默认 `robustness_results/`）

CSV 文件：

- `robustness_summary.csv`：所有模型×条件的 I2T Accuracy、错分数、T2I mAP、
  综合 Score，以及相对 clean 的性能下降量。
- `paired_vs_fixed_05.csv`：每种条件下每个模型与同图的固定 α=0.5 模型进行对比；
  输出谁改正了对方的错误、谁出现额外错误、准确率差值、McNemar 双侧精确 p 值。
- `per_image_predictions.csv`：每个图像的索引、真实标签、预测标签、对错情况。

PNG 图片：

- `00_example_corruptions.png`：真实 CIFAR-10 图像的扰动样例；
- `01_noise_accuracy.png`、`02_noise_map.png`；
- `03_blur_accuracy.png`、`04_blur_map.png`；
- `05_delta_accuracy_heatmap.png`、`06_delta_map_heatmap.png`。

额外说明：`README_results.md`。

## 6. 怎么判断是否有效？

最关键的不是一个模型的绝对性能，而是：

1. 固定 α=0.5 相比于自适应 α，在**相同被扰动图像**上的差距；
2. 相较原图，模型在更强噪声/模糊下的下降幅度；
3. 哪些样本被候选模型改对了，哪些样本反而被改错了；
4. T2I 类别文本检索 mAP 有没有同步变好。

`paired_vs_fixed_05.csv` 中：
- `delta_acc_pp > 0`：候选模型在该条件下 Accuracy 优于固定 0.5；
- `baseline_wrong_candidate_right`：候选模型救回的错分图像数；
- `baseline_right_candidate_wrong`：候选模型新错的图像数。

McNemar p 值针对相同图像上的**分类正确与否**，不代表 mAP 的显著性；
而且有多组方法、多种扰动，会存在多重检验问题，不能只看 `p < 0.05`。
如果你已经使用验证集筛选了最佳模型，此阶段只做探索；
最后的方法应该使用没有参与调参的官方 test 集进行最终验证。

## 7. 功能自检

```bash
python evaluate_clip_robustness.py --self-test
```

自检不需要 CLIP 预训练权重或 CIFAR-10，检验了：
噪声是否确定性、噪声强度、指标计算、配对检验、验证集分层划分。
（注意这不等于已在六个真实 checkpoint 上完成评估。）