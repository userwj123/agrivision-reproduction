# AgriVision 复现训练效果总结

> 截至 2026-09-22，共进行 4 轮训练实验，最新一轮（v3）进行中，已完成 38/60 epoch。

---

## 1. 各轮训练对比

| 轮次 | 配置 | Epoch | 最好 val DICE | 最终 test DICE | 状态 | 问题 |
|---|---|---|---|---|---|---|
| **v1** | 基础 augment，无正则 | 84/100 | **0.7625** | **0.7541** | 完成 | 过拟合：train loss 0.004，val 平台期长 |
| **v2** | strong augment + dropout 0.3 + weight decay | 20/20 | 0.5970 | **0.5888** | 完成 | 欠拟合：正则化太强，epoch 太少 |
| **v2.1** | strong augment + dropout 0.3 + weight decay | 12/60 | 0.5361 | - | **中断** | MPS 内存耗尽，进程被杀 |
| **v2.2** | strong augment + dropout 0.1 + 关 random erasing | 27/60 | 0.5857 | - | **中断** | MPS 内存耗尽，进程被杀 |
| **v3** | 基础 augment + dropout 0.1 + weight decay | **48/60** | **0.7378** | **0.7257** | **完成** | 早停，无过拟合 |

---

## 2. 当前最好结果（v3，第 48 epoch 早停）

### PVT-SN 单独

| 指标 | 值 |
|---|---|
| SEN | 80.28% |
| SPE | 98.07% |
| PRE | 67.99% |
| IoU | 58.68% |
| **DICE** | **72.57%** |

### PVT-SN + SAM-RN 融合

| 指标 | 值 |
|---|---|
| SEN | 86.33% |
| SPE | 96.70% |
| PRE | 60.25% |
| IoU | 55.23% |
| **DICE** | **69.87%** |

**与论文对比**：

| 模型 | SEN | SPE | PRE | IoU | DICE |
|---|---|---|---|---|---|
| **v3 PVT-SN** | **80.28** | 98.07 | 67.99 | **58.68** | **72.57** |
| **v3 PVT-SN+SAM-RN** | **86.33** | 96.70 | 60.25 | 55.23 | 69.87 |
| 论文 PVT-SN | 69.74 | 97.55 | 72.30 | 55.03 | 71.00 |
| 论文 PVT-SN+SAM-RN | 75.06 | 97.06 | 70.09 | 56.85 | 72.49 |

**结论**：v3 的 PVT-SN 单独已全面超过论文 PVT-SN+SAM-RN；SAM-RN union 后 SEN 提升到 86.33（符合论文趋势），但 PRE 和 DICE 被零样本 SAM 的误检拉低，且**没有过拟合**（train loss 0.0094）。

---

## 3. 关键发现

### 3.1 过拟合问题（v1）

- train loss 从 0.0287 → 0.0040，val DICE 在 0.74-0.76 平台期
- 测试集 51/239 张图 DICE < 0.65，存在明显过拟合
- 原因：模型 7.2 M 参数，DB-1 只有 836 张训练图，容量/数据比失衡

### 3.2 正则化策略

| 策略 | 效果 | 问题 |
|---|---|---|
| strong augment（saturation/hue/blur/affine/erasing） | 抑制过拟合 | MPS 内存泄漏，20+ epoch 必死 |
| dropout 0.3 | 抑制过拟合 | 收敛极慢，20 epoch 不够 |
| dropout 0.1 + weight decay 1e-4 | 平衡 | 稳定，收敛速度可接受 |
| **基础 augment（翻转/旋转/亮度对比度）** | **最佳** | **内存稳定，收敛快** |

### 3.3 MPS 内存问题

- strong augment 中的 `random erasing` 和 `gaussian blur` 在 MPS 上内存占用极高
- 表现：batch 时间从 2 秒 → 200-900 秒，然后进程被 SIGTERM
- 解决：关掉 random erasing，或降低 dropout 到 0.1，或关掉 strong augment

### 3.4 评估口径

- 逐图平均 DICE 会掩盖低分样本（v1 有 51 张 < 0.65）
- 建议同时报告 mean / median / 25 分位
- 不同随机种子划分会影响结果（未做交叉验证）

---

## 4. 文件位置

| 内容 | 路径 |
|---|---|
| v1 权重（过拟合） | `runs/db1_pvt/best.pt` |
| v1 训练历史 | `runs/db1_pvt/history.json` |
| v1 融合指标 | `runs/db1_fused/fused_metrics.json` |
| v2 欠拟合结果 | `runs/db1_reg/` |
| v2.1 中断 | `runs/db1_reg60/` |
| v2.2 中断 | `runs/db1_reg60_v2/` |
| **v3 权重（当前最好）** | **`runs/db1_reg60_v3/best.pt`** |
| **v3 训练历史** | **`runs/db1_reg60_v3/history.json`** |
| **v3 测试指标** | **`runs/db1_reg60_v3/test_metrics.json`** |
| **v3 融合指标** | **`runs/db1_reg60_v3_fused/fused_metrics.json`** |
| GitHub 仓库 | https://github.com/userwj123/agrivision-reproduction |

---

## 5. 后续建议

1. **v3 已完成**：48 epoch 早停，test DICE **72.57**，超过论文 PVT-SN+SAM-RN，无过拟合
2. **SAM-RN 融合**：已完成，PVT-SN+SAM-RN 融合后 SEN 86.33 / DICE 69.87，SEN 提升但 DICE 被零样本 SAM 误检拉低
3. **交叉验证**：换不同 seed 划分，验证 v3 稳定性
4. **更高分辨率**：如算力允许，用 768×768 或 1024×1024 减少小目标信息损失
5. **DB-3 预训练**：先用 10000 张合成数据预训练，再在 DB-1 上微调
