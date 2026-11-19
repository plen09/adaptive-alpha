# s43 / s44 复现测试分析

## 一句话结论

三个 seed（42 / 43 / 44）全部跑完后，s42 的两条结论里：**"自适应 α 不如固定 α"这条稳定复现了**——控制器在三个 seed 里都把 α 推到 0.53–0.62，离经验最优 0.3 更远，且 τ 越小偏得越远；**"固定 α=0.3 在强噪声下最好"这条方向复现了但效应在衰减**（noise2 相对 α=0.5：+1.00 / +0.50 / +0.06 pp）。s44 之所以"看起来不像 s42"，主因不是 α，而是 best-epoch 选择在 epoch 3 与 epoch 5 之间掷硬币，把六组配置切成了 3 对 3。

## 1. 跑了什么、产物在哪

- 补跑的训练：`runs_s44/adaptive_tau2.00_s44`（s44 的最后一组），随后自动完成两个 seed 的鲁棒性评估。
- 训练总量：3 seed × 6 配置 = 18 组，产物齐全（`best.pt`、`steps.csv`、`epochs.csv`、`summary.json`、alpha 与验证曲线图）。
- 鲁棒性评估：每 seed 6 模型 × 5 条件 × 5000 图 = 150,000 次预测，三个 seed 共 450,000 次。条件为 clean / noise1（σ=12/255）/ noise2（σ=26/255）/ blur1 / blur2。
- 分析产物：
  - `reports/compare_s43/`、`reports/compare_s44/`：与 s42 的 `reports/compare/` 同口径的单 seed 验证集对比；
  - `reports/compare_all_seeds/`：18 组合并，验证集指标 mean ± sd；
  - `reports/replication/`：复现检验、合并显著性检验、best-epoch 审计，以及本文引用的全部表格与图。

## 2. 复现结果

### 2.1 干净验证集：饱和，不提供排序信息

| 配置 | s42 | s43 | s44 | mean | sd |
|---|---:|---:|---:|---:|---:|
| fixed_a0.30 | 0.9646 | 0.9626 | 0.9620 | 0.9631 | 0.0014 |
| fixed_a0.50 | 0.9646 | 0.9614 | 0.9610 | 0.9623 | 0.0020 |
| fixed_a0.70 | 0.9640 | 0.9610 | 0.9622 | 0.9624 | 0.0015 |
| adaptive_tau0.50 | 0.9636 | 0.9606 | 0.9628 | 0.9623 | 0.0016 |
| adaptive_tau1.00 | 0.9638 | 0.9608 | 0.9628 | 0.9625 | 0.0015 |
| adaptive_tau2.00 | 0.9642 | 0.9610 | 0.9610 | 0.9621 | 0.0019 |

配置间最大差 0.0010，即 5000 张里的 5 张图；seed 间 sd 为 0.0014–0.0020，即 7–10 张图。干净集上的任何排名都不应被采信，判别信号只出现在扰动条件下。

### 2.2 预注册主假设：noise2 下固定 α=0.3 优于固定 α=0.5

| seed | delta（pp） | 救回 / 新错 | 精确 McNemar p |
|---:|---:|---:|---:|
| 42 | +1.00 | 81 / 31 | 2.5e-06 |
| 43 | +0.50 | 58 / 33 | 0.011 |
| 44 | +0.06 | 40 / 37 | 0.82 |

方向 3/3 复现。把三个 seed 作为独立层合并：精确分层 p = 3.7e-06，分层配对 bootstrap 95% CI [+0.30, +0.74] pp。

解读：方向可信，但效应量逐 seed 衰减（+1.00 → +0.50 → +0.06），s44 单独看是零结果。诚实的说法是"合并后 α=0.3 至少好 0.30 pp"，而不是"好 1 pp"。

### 2.3 最稳的复现：自适应控制器的方向性失败

α 在 epoch 1–5 的均值：

| 配置 | s42 | s43 | s44 |
|---|---:|---:|---:|
| adaptive_tau0.50 | 0.615 | 0.611 | 0.625 |
| adaptive_tau1.00 | 0.561 | 0.558 | 0.566 |
| adaptive_tau2.00 | 0.531 | 0.530 | 0.534 |

三个 seed 完全一致：α 被推过 0.5、远离经验最优 0.3，且 τ 越小偏得越远。相对 α=0.3 的 noise2 合并检验：

| 候选 | 合并 delta（pp） | 精确 p | bootstrap 95% CI |
|---|---:|---:|---:|
| adaptive_tau0.50 | −0.79 | 1.8e-05 | [−1.15, −0.43] |
| adaptive_tau1.00 | −0.74 | 3.1e-05 | [−1.08, −0.39] |
| adaptive_tau2.00 | −0.62 | 6.3e-07 | [−0.87, −0.38] |

三个 CI 全部显著为负。"自适应 loss 权重不优于固定权重"是本项目目前最硬的结论。

### 2.4 固定 α=0.7：s42/s43 显著更差，s44 反向

相对 α=0.5 的 noise2 delta 为 −0.98 / −0.60 / +0.12 pp，合并 −0.49 pp，精确 p = 0.0057（未过 Bonferroni 阈值 0.001）；相对 α=0.3 合并 −1.01 pp，p = 2.2e-07，CI [−1.39, −0.63]。s44 的反向由第 3 节的 epoch 分组解释。

### 2.5 逐 seed 的扰动排名

| 条件 | s42 第一 | s43 第一 | s44 第一 | s42 首末差 | s43 首末差 | s44 首末差 |
|---|---|---|---|---:|---:|---:|
| noise1 | fixed_a0.30 | fixed_a0.30 | fixed_a0.70 | 0.78 pp | 0.52 pp | 1.54 pp |
| noise2 | fixed_a0.30 | fixed_a0.30 | fixed_a0.70 | 1.98 pp | 1.10 pp | 0.24 pp |
| blur1 | fixed_a0.30 | fixed_a0.30 | adaptive_tau0.50 | 0.46 pp | 0.26 pp | 0.18 pp |
| blur2 | fixed_a0.30 | fixed_a0.30 | fixed_a0.30 | 0.42 pp | 0.60 pp | 0.36 pp |

s42 与 s43 的排序对 α 单调（α=0.3 最好、α=0.7 最差）；s44 的排序既不单调、首末差也塌到 0.24 pp。

## 3. s44 的异常来自 best-epoch 掷硬币

每个 `best.pt` 只保存验证 score =（I2T acc + T2I mAP）/ 2 最高的那个 epoch。acc 的粒度是 1/5000 = 0.0002，因此 score margin 低于 0.0005 时，胜负由一两张图决定。18 组训练里有 9 组（50%）margin 低于 0.0005，最小的是 s44 `adaptive_tau2.00` 的 0.00007（约 0.35 张图）。

由此产生的分组：s42 六组全选 epoch 5，s43 六组全选 epoch 3，s44 分裂为
epoch 3 = {adaptive_tau0.50, adaptive_tau1.00, fixed_a0.70}，epoch 5 = {adaptive_tau2.00, fixed_a0.30, fixed_a0.50}。

s44 内部因此构成一个自然实验：同 split、同扰动实现，两组之间唯一的系统差异是被送进评估的 epoch。epoch 5 组相对 epoch 3 组：

| 条件 | acc 差（pp） | mAP 差（pp） |
|---|---:|---:|
| clean | −0.13 | +0.13 |
| noise1 | −1.24 | −1.35 |
| noise2 | +0.04 | −1.77 |
| blur1 | −0.13 | −0.25 |
| blur2 | +0.00 | −0.35 |

即"多训两个 epoch 并被选中"的那三组，在重噪声下明显更差，而在干净集上几乎无差。这个分组与 α 不单调（α=0.70 与两个自适应配置同组，α=0.30 / 0.50 在另一组），所以 s44 的扰动排名不能读成 α 效应。

含义：`best.pt` 只保留赢的那一个 epoch，该混杂无法事后消除。s42 与 s43 内部同构（六组同 epoch），不受此混杂影响，是关于 α 排序的更干净证据；s44 的权重应下调。

## 4. 统计口径

- 配对方式：同一 seed 内，同一批扰动图分别喂给候选模型与基线模型，逐图对错构成配对 2×2 表。
- 合并方式：三个 seed 视为独立层（验证 split 不同、噪声实现不同）。分层精确 McNemar 取各层 Binomial(m_i, 1/2) 分布的卷积，得到 sum(救回 − 新错) 的精确双侧 p；同时给出正态近似 Z 与分层配对 bootstrap（每层按四格多项式重抽 20000 次）的 95% CI。
- 一致性校验：从 `per_image_predictions.csv` 重新计算的救回/新错、acc 差与精确 McNemar p，与评估脚本写出的 `paired_vs_fixed_05.csv` 逐行一致（3 seed × 25 行）。
- 多重性：5 条件 × 5 候选 × 2 基线 = 50 个合并检验，Bonferroni 阈值 0.001；通过的只有 noise2 的 6 行。两个基线共用同一批 run，故该校正偏保守。
- 层间重叠：val(42)∩val(43) = 507、val(42)∩val(44) = 515、val(43)∩val(44) = 485 张（约 10%，即随机水平）。"独立层"近似成立，但合并 p 略偏乐观。
- mAP 是整子集的排序指标，不能逐图配对，只报逐 seed 差值与 mean ± sd。
- 扰动是 32px 原图上的合成高斯噪声 / 高斯模糊，不是 CIFAR-10-C。
- 验证集既用于选 epoch 又用于评估，本文数字属于方法选择数据，不是留出测试集估计。
- 3 个 seed 的 sd 只有 2 个自由度，本身很不确定。

## 5. 结论与下一步

1. 可写入课程/论文的表述：在"固定类别文本原型 + CIFAR-10 微调"设定下，自适应双向 loss 权重不优于固定权重，该负结果在 3 个 seed 上稳定；固定 α=0.3 在强噪声下最优，但优势随 seed 衰减，合并 CI 为 [+0.30, +0.74] pp。
2. 任何后续对照（包括 probe meta-controller）都应以固定 α=0.3 为基线，并采用固定 epoch 协议；否则 best-epoch 掷硬币会以最高约 1.8 pp 的量级污染对照。
3. 干净集已饱和，继续加 seed 的边际收益有限；要打破饱和应换 CIFAR-100、CIFAR-10-C 或带真实 caption 的数据集。
4. 若仍在本数据上推进，建议补一轮"统一 epoch（如 epoch 5）× 6 配置 × 3 seed"的重跑，把 α 效应与 epoch 选择效应彻底分离。

## 6. 产物索引

| 文件 | 内容 |
|---|---|
| `reports/replication/REPORT_replication.md` | 预注册复现检验（逐 seed McNemar、clean acc mean±sd、split 重叠） |
| `reports/replication/REPORT_cross_seed_tests.md` | 合并显著性检验全文（双基线、mAP、逐 seed 排名） |
| `reports/replication/REPORT_epoch_audit.md` | best-epoch 选择审计与 s44 自然实验 |
| `reports/replication/cross_seed_paired_tests.csv` | 50 行合并配对检验明细 |
| `reports/replication/cross_seed_condition_summary.csv` | 配置 × 条件 × seed 的 acc / mAP 明细与 mean±sd |
| `reports/replication/epoch_selection_audit.csv` | 18 组训练的 best-epoch margin 明细 |
| `reports/replication/07_forest_delta_acc_vs_a050.png` | 相对 α=0.5 的逐 seed 点与合并 CI |
| `reports/replication/07_forest_delta_acc_vs_a030.png` | 相对 α=0.3 的逐 seed 点与合并 CI |
| `reports/replication/08_condition_acc_cross_seed.png` | 各条件 acc / mAP 的 seed 间 mean±sd |
| `reports/replication/09_epoch_group_gap.png` | s44 内 epoch 组的鲁棒性差 |
| `reports/compare_s43/`、`reports/compare_s44/` | 单 seed 验证集对比（与 s42 同口径） |
| `reports/compare_all_seeds/` | 18 组合并的验证集对比 |
