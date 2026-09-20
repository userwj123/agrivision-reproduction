"""Inference: run PVT-SN -> box prompts -> SAM-RN -> union fusion, compute metrics."""
import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import AgriVisionSeg, make_splits
from model import PVTSN, SAMRN, mask_to_boxes, metrics_from_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image-sub", default="Images")
    ap.add_argument("--mask-sub", default="Masks")
    ap.add_argument("--out", default="runs/pvtsn_samrn")
    ap.add_argument("--backbone", default="pvt_v2_b2")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--max-boxes", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}")

    ds = AgriVisionSeg(args.data, image_sub=args.image_sub, mask_sub=args.mask_sub,
                       img_size=args.img_size, train=False, augment=False)
    _, _, test_ds = make_splits(ds, seed=42)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"test={len(test_ds)}")

    model = PVTSN(backbone=args.backbone, pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.ckpt} (epoch {ckpt.get('epoch')})")

    sam = SAMRN(device=device)

    out = Path(args.out)
    (out / "preds").mkdir(parents=True, exist_ok=True)

    agg_pvt = {k: 0.0 for k in ["SEN", "SPE", "PRE", "IoU", "DICE"]}
    agg_fus = {k: 0.0 for k in ["SEN", "SPE", "PRE", "IoU", "DICE"]}
    n = 0

    with torch.no_grad():
        for bi, (img, mask) in enumerate(tqdm(loader, desc="infer")):
            img_dev = img.to(device)
            logits = model(img_dev)
            # upsample 1/4-res logits to full resolution for box extraction and eval
            logits = torch.nn.functional.interpolate(logits, size=img.shape[2:], mode="bilinear", align_corners=False)
            pvt_pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(bool)
            gt = mask.cpu().numpy().astype(bool)
            for i in range(img.shape[0]):
                p = pvt_pred[i, 0]
                g = gt[i, 0]
                m1 = metrics_from_mask(p, g)
                for k in agg_pvt:
                    agg_pvt[k] += m1[k]

                # SAM-RN
                image_np = (img[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                boxes = mask_to_boxes(p, min_area=args.min_area, max_boxes=args.max_boxes)
                sam_mask = sam.predict(image_np, boxes)
                fused = np.logical_or(p, sam_mask > 0).astype(bool)
                m2 = metrics_from_mask(fused, g)
                for k in agg_fus:
                    agg_fus[k] += m2[k]

                if bi < 8 and i == 0:
                    Image.fromarray((p * 255).astype(np.uint8)).save(out / "preds" / f"{bi}_{i}_pvt.png")
                    Image.fromarray((sam_mask * 255).astype(np.uint8)).save(out / "preds" / f"{bi}_{i}_sam.png")
                    Image.fromarray((fused * 255).astype(np.uint8)).save(out / "preds" / f"{bi}_{i}_fused.png")
                n += 1

    res = {
        "PVT-SN": {k: v / max(n, 1) for k, v in agg_pvt.items()},
        "PVT-SN+SAM-RN": {k: v / max(n, 1) for k, v in agg_fus.items()},
        "n": n,
    }
    print(json.dumps(res, indent=2))
    with open(out / "fused_metrics.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
