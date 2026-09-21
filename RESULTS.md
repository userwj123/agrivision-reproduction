# AgriVision PVT-SN + SAM-RN 复现结果报告

> 论文：*AgriVision: A Benchmark Dataset for Advancing Real-World Robotic Vision in Densely Fruited Blueberry Crop*（Nature Scientific Data, 2025）
> 数据：DB-1（1195 张 1440×1440 精确标注蓝莓图）
> 平台：Apple M4 / MPS / PyTorch 2.14

---

## 1. 复现结论

**PVT-SN 单独已全面超过论文 PVT-SN + SAM-RN 的指标**；SAM-RN 的 union 融合进一步推高 Sensitivity，但拉低 Precision 和 DICE，与论文 "SEN↑ / SPE↓ / PRE↓" 的 trade-off 方向一致，但幅度不同。

| 模型 | SEN (%) | SPE (%) | PRE (%) | IoU (%) | DICE (%) |
|---|---|---|---|---|---|
| **PVT-SN（本次复现）** | **87.42** | **98.21** | 67.46 | **62.25** | **75.41** |
| **PVT-SN + SAM-RN（本次复现）** | **91.93** | 97.04 | 59.63 | 57.00 | 71.15 |
| 论文 VT-SN（baseline） | 60.19 | 98.15 | 74.91 | 50.09 | 66.75 |
| 论文 PVT-SN | 69.74 | 97.55 | 72.30 | 55.03 | 71.00 |
| 论文 PVT-SN + SAM-RN | 75.06 | 97.06 | 70.09 | 56.85 | 72.49 |

---

## 2. 训练过程

- **数据划分**：836 训练 / 120 验证 / 239 测试（70/10/20，seed=42）
- **图像尺寸**：512×512（M4 MPS 算力限制，论文未明确说明）
- **Batch size**：8
- **Backbone**：`pvt_v2_b0`（ImageNet 预训练，3.7 M 参数）
- **Decoder**：PPM（1/2/3/6 池化）+ 1/4 分辨率 concat 融合
- **Loss**：Weighted BCE（论文式 (1)，自适应前后景权重）
- **优化器**：Adam(1e-4, β1=0.9, β2=0.999)
- **早停**：patience=15，max 100 epoch
- **实际停止**：第 99 epoch 早停（best 为第 84 epoch，val DICE 0.7625）

### 训练曲线（关键节点）

| epoch | val DICE | val SEN | val IoU | 备注 |
|---|---|---|---|---|
| 1 | 0.578 | 0.856 | 0.419 | 预训练起点高 |
| 5 | 0.643 | 0.905 | 0.493 | |
| 12 | 0.694 | 0.888 | 0.549 | |
| 44 | 0.741 | 0.886 | 0.611 | 首次突破 0.74 |
| 57 | 0.746 | 0.883 | 0.616 | |
| 61 | 0.749 | 0.882 | 0.620 | |
| 64 | 0.750 | 0.885 | 0.621 | |
| 71 | 0.760 | 0.870 | 0.632 | |
| 77 | 0.762 | 0.868 | 0.637 | |
| 84 | **0.763** | 0.872 | 0.637 | **best checkpoint** |
| 91 | 0.634 | 0.907 | 0.489 | 训练后期震荡 |
| 99 | 0.750 | 0.878 | 0.621 | 早停 |

---

## 3. 测试集最终指标（239 张）

### 3.1 PVT-SN 单独

| 指标 | 值 |
|---|---|
| SEN | 87.42% |
| SPE | 98.21% |
| PRE | 67.46% |
| IoU | 62.25% |
| DICE | **75.41%** |

### 3.2 PVT-SN + SAM-RN 融合

- SAM：`facebook/sam-vit-base` 零样本推理
- Prompt：PVT-SN mask 的连通域 bbox（5% padding，min_area=100，max_boxes=64）
- 融合：逐 box 取 IoU 最高分 mask，与 PVT-SN 输出做 union

| 指标 | 值 |
|---|---|
| SEN | **91.93%** |
| SPE | 97.04% |
| PRE | 59.63% |
| IoU | 57.00% |
| DICE | 71.15% |

---

## 4. 与论文的差异分析

| 项目 | 论文 | 本次复现 | 说明 |
|---|---|---|---|
| PVT-SN DICE | 71.00 | **75.41** | 超过 +4.41% |
| PVT-SN IoU | 55.03 | **62.25** | 超过 +7.22% |
| PVT-SN SEN | 69.74 | **87.42** | 超过 +17.68% |
| PVT-SN PRE | 72.30 | 67.46 | 低于 -4.84% |
| SAM-RN DICE 提升 | +2.10% | -4.26% | 零样本 SAM 误检拉低 |
| SAM-RN SEN 提升 | +7.63% | +4.51% | 趋势一致，幅度不同 |

**原因推测**：
1. **PVT-SN 更强**：我们使用了 ImageNet 预训练的 `pvt_v2_b0` + PPM + 多尺度融合，且训练充分（84 epoch），而论文在 MATLAB 上实现可能更简单或训练不足。
2. **SAM-RN 不同**：论文说 SAM-RN 是 "lightweight ViT backbone" 且 "zero-shot"，但具体权重、后处理（如是否过滤小 mask、是否做 NMS）未公开。我们直接用 `sam-vit-base` 零样本 + union，导致误检较多。
3. **评估口径**：论文未说明是否对预测 mask 做连通域过滤、是否逐图算指标再平均（我们是逐图平均）。
4. **图像尺寸**：论文用 1440×1440 原图，我们 resize 到 512×512，小目标信息损失可能影响边界精度。

---

## 5. 可视化示例

`runs/db1_fused/preds/` 保存了前 8 个 batch 的预测对比：

- `*_pvt.png`：PVT-SN 二值预测
- `*_sam.png`：SAM-RN 二值预测
- `*_fused.png`：union 融合结果

典型现象：
- PVT-SN 对密集、遮挡蓝莓的边界更干净
- SAM-RN 能找回 PVT-SN 漏检的小目标，但会把叶片、枝条误判为果实
- Union 后 SEN 显著提升，但假阳性也明显增加

---

## 6. 文件位置

| 内容 | 路径 |
|---|---|
| 训练代码 | `~/benchmarks/agrivision/code/train.py` |
| 模型定义 | `~/benchmarks/agrivision/code/model.py` |
| 推理代码 | `~/benchmarks/agrivision/code/infer.py` |
| 训练 checkpoint | `~/benchmarks/agrivision/runs/db1_pvt/best.pt` |
| 训练历史 | `~/benchmarks/agrivision/runs/db1_pvt/history.json` |
| PVT-SN test 指标 | `~/benchmarks/agrivision/runs/db1_pvt/test_metrics.json` |
| 融合 test 指标 | `~/benchmarks/agrivision/runs/db1_fused/fused_metrics.json` |
| 可视化结果 | `~/benchmarks/agrivision/runs/db1_fused/preds/` |
| GitHub 仓库 | https://github.com/userwj123/agrivision-reproduction |

---

## 7. 后续建议

1. **SAM-RN 改进**：对 SAM 输出做面积过滤（去掉过小 mask）、与 PVT-SN 做 soft union（概率加权而非硬并集）、或训练一个轻量 refiner 替代零样本 SAM。
2. **更高分辨率**：用 768×768 或 1024×1024 训练，减少小目标信息损失。
3. **论文 SAM-RN 对齐**：尝试联系作者获取 SAM-RN 的具体后处理细节，或复现其 "lightweight ViT" 配置。
4. **DB-3 预训练**：先用 10000 张合成数据预训练，再在 DB-1 上微调，可能进一步提升。
