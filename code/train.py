"""Training script for AgriVision PVT-SN reproduction."""
import os
import sys
import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import AgriVisionSeg, make_splits
from model import PVTSN, WeightedBCE, metrics_from_mask


def evaluate(model, loader, device, thr=0.5):
    model.eval()
    agg = {k: 0.0 for k in ["SEN", "SPE", "PRE", "IoU", "DICE"]}
    n = 0
    with torch.no_grad():
        for img, mask in loader:
            img = img.to(device)
            mask = mask.to(device)
            logits = model(img)
            # model outputs 1/4-res logits; upsample to mask size for eval
            logits = torch.nn.functional.interpolate(logits, size=mask.shape[2:], mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(logits) > thr).cpu().numpy().astype(bool)
            gt = mask.cpu().numpy().astype(bool)
            for p, g in zip(pred, gt):
                m = metrics_from_mask(p[0], g[0])
                for k in agg:
                    agg[k] += m[k]
                n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset root (contains Images/ and Masks/, or Generated Images/ and Generated Masks/)")
    ap.add_argument("--image-sub", default="Images")
    ap.add_argument("--mask-sub", default="Masks")
    ap.add_argument("--out", default="runs/pvtsn")
    ap.add_argument("--backbone", default="pvt_v2_b2")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}")

    ds = AgriVisionSeg(args.data, image_sub=args.image_sub, mask_sub=args.mask_sub,
                       img_size=args.img_size, train=True, augment=True)
    ds_eval = AgriVisionSeg(args.data, image_sub=args.image_sub, mask_sub=args.mask_sub,
                            img_size=args.img_size, train=False, augment=False)
    train_ds, val_ds, test_ds = make_splits(ds, seed=args.seed)
    _, val_ds_eval, test_ds_eval = make_splits(ds_eval, seed=args.seed)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds_eval, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds_eval, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = PVTSN(backbone=args.backbone, pretrained=not args.no_pretrained).to(device)
    loss_fn = WeightedBCE()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0
    patience = 0
    history = []
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("val", {}).get("DICE", 0.0)
        print(f"resumed from {args.resume} (epoch {start_epoch - 1}, best DICE {best_dice:.4f})")
        hist_path = out / "history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history = json.load(f)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for img, mask in tqdm(train_loader, desc=f"epoch {epoch}"):
            img = img.to(device)
            mask = mask.to(device)
            logits = model(img)
            # model outputs 1/4-res logits; downsample mask to match
            mask_ds = torch.nn.functional.interpolate(mask, size=logits.shape[2:], mode="nearest")
            loss = loss_fn(logits, mask_ds)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        train_loss = running / max(1, len(train_loader))

        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"epoch {epoch}: loss={train_loss:.4f} "
              f"val SEN={val_metrics['SEN']:.4f} DICE={val_metrics['DICE']:.4f} IoU={val_metrics['IoU']:.4f} "
              f"({elapsed:.0f}s)")
        history.append({"epoch": epoch, "loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        # de-duplicate by epoch in case of resume overwrite
        seen = set()
        dedup = []
        for e in history:
            if e["epoch"] not in seen:
                dedup.append(e)
                seen.add(e["epoch"])
        history = dedup
        with open(out / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if val_metrics["DICE"] > best_dice:
            best_dice = val_metrics["DICE"]
            patience = 0
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": epoch, "val": val_metrics}, out / "best.pt")
            print(f"  -> saved best (DICE={best_dice:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    # test with best checkpoint
    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device)
    print("TEST:", json.dumps(test_metrics, indent=2))
    with open(out / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
