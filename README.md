# CIFAR-10 / OpenAI CLIP ViT-B/32 Adaptive-α Experiment

这是一个可复现的**类别级 CLIP 微调**最小实验，不是原始 image-caption CLIP 预训练复现。
CIFAR-10 没有逐图像 caption。我们固定 10 条类别文本提示，冻结文本塔，
微调 CLIP ViT-B/32 视觉塔最后两个 Transformer block、`ln_post` 和 `proj`。
这样两个方向的 Loss 都能更新视觉塔，而权重 α 是否变化是主要实验变量。

## 安装
建议 Python 3.10+、GPU。先根据你的 CUDA 版本安装匹配的 PyTorch 和 torchvision，
再安装：

    pip install open_clip_torch numpy matplotlib

OpenAI 权重使用 `ViT-B-32-quickgelu` 搭配 `pretrained=openai`
（OpenCLIP 官方 README 说明此组合匹配 QuickGELU）。
第一次运行需要网络下载预训练权重和 CIFAR-10。

## 训练目标

- Image→Text：图像嵌入在 10 个**唯一**文本类别原型中做交叉熵分类。
- Text→Image：对每个出现在 batch 的类别 c，把同类别的所有图像作为正样本；
  以 `KL[ Uniform(positive images) || Softmax(similarity to all batch images) ]` 为 Loss。
  展开为 `logsumexp(scores[:,c]) - mean(scores[positive,c]) - log(n_positive)`。
- Total：`alpha * loss_i2t + (1-alpha) * loss_t2i`。
- Adaptive alpha：两个 Loss 的 EMA，除以各自在 warmup 阶段的均值作为相对剩余难度。
  对两者 log ratio 用 sigmoid/temperature 得到 α 目标，再做 EMA 平滑并截断到 [0.2,0.8]。
  α 不参与反向传播。

注意：两项 Loss 虽都非负并可在理想情况下趋近 0，但数值尺度仍可能不同；
采用相对初期损失的比值，减少它们的尺度差异对 α 的直接影响。

## 数据与指标

- CIFAR-10 官方训练集 50,000 张，以固定 seed 分层随机划分：
  45,000 train / 5,000 validation（每类验证 500 张）。
- 官方 test 10,000 张只在验证集选定方法/超参数之后使用。
- 图像 32px，先进行轻度训练增强，再用 CLIP 验证预处理缩放到 224px。
- 评估：Image→Text top-1 分类准确率；
  Text→Image mAP（10 个类别文本作为查询，各类别对应的所有图片都是 relevant）。
- 按验证集 `(accuracy + mAP)/2` 选择最佳 epoch；同时保留 epoch 0 不微调的模型。

## 命令

先检查原始预训练模型（验证集）：

    python adaptive_clip_cifar10.py --mode zero_shot

固定权重基线（推荐每个设置跑 seed 42 / 43 / 44）：

    python adaptive_clip_cifar10.py --mode fixed --alpha 0.5 --seed 42 --epochs 5
    python adaptive_clip_cifar10.py --mode fixed --alpha 0.3 --seed 42 --epochs 5
    python adaptive_clip_cifar10.py --mode fixed --alpha 0.7 --seed 42 --epochs 5

自适应权重：

    python adaptive_clip_cifar10.py --mode adaptive --tau 1.0 --seed 42 --epochs 5
    python adaptive_clip_cifar10.py --mode adaptive --tau 0.5 --seed 42 --epochs 5
    python adaptive_clip_cifar10.py --mode adaptive --tau 2.0 --seed 42 --epochs 5

显存有限，可以将 `--batch-size 128` 改为 `--batch-size 64`；
各比较组必须使用**一致**的 batch size、seed、数据划分、训练超参数和训练预算。
第一次建议仅跑 1 epoch 或使用 `--max-train-batches 5` 检查流程是否畅通
（带有该调试参数的结果不可用于正式比较）。

比较全部验证集结果：

    python compare_runs.py --root runs

确定最佳方法/超参后，选中的最佳验证集 checkpoint 单独运行 test：

    python adaptive_clip_cifar10.py --mode evaluate --checkpoint runs/adaptive_tau1.00_s42/best.pt

上面的命令需要与训练时保持相同的 `--seed`（因为验证集分层划分由 seed 决定）。
如果使用非默认模型或权重，也要在 evaluate 命令中保持一致。

## 输出

每个训练设置输出到 `runs/<method+parameters+seed>/`：

- `steps.csv`：每步的 alpha、I2T/T2I/total Loss
- `epochs.csv`：每 epoch 的验证指标、平均 alpha 与训练时间
- `alpha_curve.png`：alpha 轨迹
- `validation_curve.png`：验证集准确率与 mAP
- `best.pt`：只保存已微调参数和元数据；恢复时重新加载 OpenAI CLIP 初始权重
- `summary.json`：最佳 epoch 与验证指标等

**解读边界**：
这验证的是“固定类别文本原型条件下，自适应双向 Loss 权重是否优于固定权重”。
不能直接推广为“原始图文配对 CLIP 预训练中必然有效”。
如要研究原始 CLIP，可第二阶段改用有真实 captions 的 Flickr30k 或 COCO，
并明确处理一对多正例和重复 caption。

资料：
- https://github.com/mlfoundations/open_clip
- https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
- https://proceedings.mlr.press/v80/chen18a.html