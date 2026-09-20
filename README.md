# AgriVision PVT-SN + SAM-RN 复现

基于论文 *AgriVision: A Benchmark Dataset for Advancing Real-World Robotic Vision in Densely Fruited Blueberry Crop*（Nature Scientific Data, 2025）的复现代码。

原 GitHub 仓库（Owais-CodeHub/PVT-SN-SAM-RN）为空壳，仅包含空 `main.m` 与示意图。本仓库根据论文 PDF 与 Figshare 官方资源重建了：

- **DB-3 合成数据生成器**：官方 `mainCode.m` 的 Python 向量化移植
- **PVT-SN 分割网络**：PVTv2 主干 + Pyramid Pooling Module + concat decoder
- **SAM-RN 精修**：零样本 SAM，使用 PVT-SN 输出的 bbox 作为 prompt
- **Union 后融合**：保证高 Sensitivity

详细过程与结果见 [REPRODUCTION_SUMMARY.md](REPRODUCTION_SUMMARY.md)。

## 代码结构

```
code/
├── db3_generator.py   # DB-3 合成数据生成
├── dataset.py         # DB-1 / DB-3 数据集加载与划分
├── model.py           # PVT-SN / SAM-RN / WCE loss / 指标
├── train.py           # 训练脚本
└── infer.py           # 推理与融合脚本
```

## 快速开始

```bash
# 环境
python -m venv venv
source venv/bin/activate
pip install torch torchvision timm transformers opencv-python-headless pillow numpy scikit-image tqdm matplotlib pymupdf

# 训练 PVT-SN（以 DB-1 为例）
python code/train.py --data data/DB-1/DB-1 --backbone pvt_v2_b0 --img-size 512 --batch-size 8

# 推理 + SAM-RN 融合
python code/infer.py --data data/DB-1/DB-1 --ckpt runs/db1_pvt/best.pt

# 生成 DB-3 风格合成数据
python code/db3_generator.py --backgrounds <bg_dir> --fruits <fruit_dir> --out <out_dir>
```

## 数据

数据来自 Figshare 记录 29222462（CC BY 4.0）：

- DB-1: 1195 对 1440×1440 图 + 精确 mask
- DB-2: 141453 帧半自动标注视频
- DB-3: 10000 对合成图 + mask
- Code_DB-3 Data Generation: 官方 MATLAB 生成代码与样例

## 引用

```bibtex
@article{owais2025agrivision,
  title={AgriVision: A Benchmark Dataset for Advancing Real-World Robotic Vision in Densely Fruited Blueberry Crop},
  author={Owais, Muhammad and Shafay, Muhammad and Zubair, Muhammad and Mohammed, Shamal and Seneviratne, Lakmal and Hussain, Irfan},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={2014},
  year={2025},
  publisher={Nature Publishing Group}
}
```
