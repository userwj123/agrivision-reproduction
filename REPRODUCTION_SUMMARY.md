# AgriVision 复现项目总结

> 目标：基于论文 *AgriVision: A Benchmark Dataset for Advancing Real-World Robotic Vision in Densely Fruited Blueberry Crop*（Nature Scientific Data, 2025）复现其提出的 **PVT-SN + SAM-RN** 密集蓝莓分割框架，并还原其 **DB-3 合成数据生成算法**。
>
> 工作区：`~/benchmarks/agrivision/`

---

## 1. GitHub 项目分析

**仓库**: https://github.com/Owais-CodeHub/PVT-SN-SAM-RN

| 项目 | 结论 |
|---|---|
| 仓库性质 | 空壳项目，仅用于挂论文链接和两张示意图 |
| 代码文件 | `Code/main.m` 为 **1 字节空文件**；Git 全历史（22 个提交）中从未出现过任何真实代码 |
| 其他内容 | `README.md`（结果表）、`Images/Model.png`（架构示意图）、`Images/Dataset.png`（数据集示意图） |
| 论文出处 | Owais, M. et al. *AgriVision: A Benchmark Dataset for Advancing Real-World Robotic Vision in Densely Fruited Blueberry Crop*. Sci Data 12, 2014 (2025). DOI: 10.1038/s41597-025-06263-3 |
| 数据托管 | Figshare 记录 29222462（CC BY 4.0），含 DB-1 / DB-2 / DB-3 / 官方 DB-3 生成代码 |

**结论**：GitHub 仓库不可用于复现，必须依赖论文 PDF + Figshare 资源。

---

## 2. 论文方法要点

- **任务**：密集蓝莓果实二值分割（berry vs. 背景）
- **框架**：PVT-SN（Pyramid Vision Transformer Segmentation Network）+ SAM-RN（Segment Anything Model Refinement Network）
- **数据流**：
  1. PVT-SN 用层级 PVT 主干 + PPM 提取全局/局部特征，输出初始 mask
  2. 从初始 mask 提取连通域 bbox 作为 SAM 的 prompt
  3. SAM-RN 零样本精修，提升边界、找回漏检
  4. Post-fusion：PVT-SN 与 SAM-RN 输出做 **union**，保证高 SEN
- **训练**（论文）：Adam(1e-4, 0.9/0.999)，batch 12，WCE loss，早停 15，max 100 epoch，70/10/20 split
- **结果**（论文，DB-1）：SEN 75.06 / SPE 97.06 / PRE 70.09 / IoU 56.85 / DICE 72.49

---

## 3. 数据集下载与还原

从 Figshare（ID 29222462）下载并 MD5 校验：

| 文件 | 大小 | 内容 | 状态 |
|---|---|---|---|
| `DB-1.zip` | 6.02 GB | 1195 对 1440×1440 图 + 精确 mask | 已下载、已解压 |
| `DB-3.zip` | 10.03 GB | 10000 对合成图 + mask（uncluttered） | 已下载、已解压 |
| `Code_DB-3 Data Generation.zip` | 0.50 GB | 官方 `mainCode.m` + 样例素材 | 已下载、已解压 |
| `DB-2.zip` | 78.04 GB | 141453 帧半自动标注视频 | **未下载** |

**网络问题记录**：figshare 的 `ndownloader` 端点最初因我额外附加 UA/Cookie 触发 AWS WAF 返回 403；裸 `curl -L` 即可拿到 302 跳转和 S3 预签名 URL。切换代理节点（台湾）后速度从 ~200 KB/s 提升到可接受水平。

---

## 4. 我产生的代码与结果

### 4.1 `code/db3_generator.py`
**功能**：将官方 `mainCode.m`（184 行 MATLAB）完整移植为 Python/OpenCV 向量化实现。

- 忠实还原：HSV 绿色区域提取、10-50 个果实 patch、0.6-0.9 缩放、0-360° 旋转、30% 遮挡、0.9 透明度混合、Canny 边缘 + 膨胀 + 高斯平滑
- 输出 uncluttered / cluttered 两对图+mask，命名与官方一致
- 生成 50 张耗时 4 秒（官方 MATLAB 双重循环版本极慢）
- **验证**：与官方 40 组样例对比，前景像素分布基本一致（mean 15.5 万 vs 13.8 万）

### 4.2 `code/dataset.py`
- `AgriVisionSeg`：支持 DB-1（`Images/`+`Masks/`）和 DB-3（`Generated Images/`+`Generated Masks/`）
- 图像/掩码 resize 到 512×512，训练侧做翻转/旋转/亮度对比度增强
- `make_splits`：70/10/20 固定随机种子划分

### 4.3 `code/model.py`
| 模块 | 实现 | 说明 |
|---|---|---|
| `PPM` | 1/2/3/6 池化 + 1×1 降维 + concat + 3×3 融合 | MPS 友好（用 interpolate 替代 `adaptive_avg_pool2d` 绕开 PyTorch #96056） |
| `PVTSN` | PVTv2 主干 + PPM + 1/4 分辨率 concat decoder | 输出单通道 logit；融合放在 1/4 分辨率避免 MPS INT_MAX 张量尺寸溢出 |
| `SAMRN` | `facebook/sam-vit-base` 零样本推理 | bbox prompt，逐 box 选 IoU 最高分 mask，最后 union |
| `mask_to_boxes` | 连通域 + 5% padding | 从 PVT-SN mask 生成 SAM prompt |
| `WeightedBCE` | 论文式 (1) 自适应加权 BCE | 处理前景/背景不平衡 |
| `metrics_from_mask` | SEN/SPE/PRE/IoU/DICE | 与论文评估口径一致 |

### 4.4 `code/train.py`
- Adam(1e-4, 0.9/0.999)，batch 8，早停 15，max 100 epoch
- 训练时把 mask 下采样到 1/4 分辨率算 loss，验证/测试时上采样回 512
- 保存 `best.pt`、`history.json`、`test_metrics.json`

### 4.5 `code/infer.py`
- 加载训练好的 PVT-SN checkpoint，跑完整 pipeline：PVT-SN → bbox → SAM-RN → union
- 同时输出 PVT-SN 与 PVT-SN+SAM-RN 两套指标

---

## 5. 实验结果

### 5.1 DB-3 官方样例（40 张）
- 配置：pvt_v2_b0，512×512，batch 8，40 epoch
- **Test**: SEN 0.981 / SPE 0.984 / PRE 0.781 / IoU 0.769 / **DICE 0.869**
- 说明端到端管线（数据加载 → 训练 → 评估）已完整跑通

### 5.2 DB-1 正式复现（1195 张）
- 配置：pvt_v2_b0，512×512，batch 8，ImageNet 预训练
- 训练被提前终止（M4 MPS 约 5 min/epoch，全量 100 epoch 需 ~8 小时）
- 已完成 5 epoch，最好 **val DICE 0.668**（checkpoint：`runs/db1_pvt/best.pt`）

### 5.3 SAM-RN 零样本验证（DB-1 单图）
- 用真实 mask 加 8% 噪声模拟 PVT-SN 输出，提取 34 个 bbox
- SAM 单独：**DICE 0.791**（SEN 0.929 / PRE 0.688）
- Union 后：SEN 升到 0.994，SPE/PRE 下降，与论文 "SEN↑ / SPE↓" 的 trade-off 完全一致

---

## 6. 与论文的差异说明

| 项目 | 论文 | 本次复现 | 原因 |
|---|---|---|---|
| 框架 | MATLAB R2024a + Deep Learning Toolbox | PyTorch 2.14 + timm + transformers | 平台差异 |
| PVT-SN 主干 | 未公开具体实现，仅给出 ViT-B/16 级超参（d=768, 12 层, 12 头） | `pvt_v2_b0`（3.7 M 参数） | 论文代码为空；M4 MPS 算力限制 |
| SAM-RN | 未公开，描述为轻量 ViT（d=256, 12 头） | `facebook/sam-vit-base` 零样本 | 官方无权重/代码 |
| 图像尺寸 | 1440×1440（推测） | 512×512 | 显存/算力限制 |
| Batch size | 12 | 8 | 同上 |

---

## 7. 文件清单

```
~/benchmarks/agrivision/
├── paper/
│   ├── nature.pdf                  # 论文 PDF
│   ├── epmc.xml                    # Europe PMC 全文 XML
│   └── fig_p*.png / page*.png      # 从 PDF 提取的图
├── data/
│   ├── DB-1.zip / DB-1/            # 1195 对图+mask
│   ├── DB-3.zip / DB-3/            # 10000 对合成图+mask
│   └── db1_dl.log / db3_dl.log
├── code/
│   ├── code_db3.zip                # 官方代码包
│   ├── extracted/Code_DB-3 Data Generation/
│   │   ├── mainCode.m              # 官方 MATLAB 生成代码
│   │   └── DB-3 Sample Folder/     # 样例素材与输出
│   ├── db3_generator.py            # Python 版 DB-3 生成器
│   ├── dataset.py                  # 数据集加载
│   ├── model.py                    # PVT-SN / SAM-RN / WCE / 指标
│   ├── train.py                    # 训练脚本
│   └── infer.py                    # 推理与融合脚本
├── runs/
│   ├── db3_sample/                 # DB-3 样例训练结果
│   │   ├── best.pt
│   │   ├── history.json
│   │   └── test_metrics.json
│   └── db1_pvt/                    # DB-1 训练结果（中断）
│       ├── best.pt
│       ├── history.json
│       └── test_metrics.json
└── work/                           # 中间脚本与网络探测
```

---

## 8. 已知问题与后续建议

1. **GitHub 仓库无代码**：无法直接复现作者原始实现，只能依据论文文字 + Figure 2 示意图 + 官方 DB-3 生成代码重建。
2. **算力瓶颈**：M4 MPS 上 512×512 batch 8 约 5 min/epoch，完整 DB-1 训练需 8 小时以上；如需复现论文指标，建议换 CUDA GPU 或减小输入尺寸/缩短 epoch。
3. **MPS 兼容性**：已通过两处修改绕开 PyTorch MPS 限制（PPM 池化方式、decoder 融合分辨率）。
4. **DB-2 未下载**：78 GB 视频帧数据量太大，且当前复现以 DB-1 监督分割为主，暂未需要。
5. **后续可选项**：
   - 续训 DB-1（`runs/db1_pvt/best.pt`）
   - 用现有 checkpoint 跑 `code/infer.py` 出 PVT-SN+SAM-RN 融合指标
   - 下载 DB-2 做半监督/弱监督实验
