"""PVT-SN + SAM-RN reproduction for AgriVision dense blueberry segmentation.

- VT-SN  : SegFormer-B0 baseline (paper's VT-SN)
- PVT-SN : PVTv2-B2 backbone (closest to ViT-B/16 spec: 512 embed, 24 blocks, 8 heads)
           + Pyramid Pooling Module (PPM) + concat decoder
- SAM-RN : zero-shot SAM with box prompts derived from PVT-SN mask
- Post-fusion: union of PVT-SN mask and SAM-RN masks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import SamModel, SamProcessor
import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  PPM (Pyramid Pooling Module) — the "PP block" in the paper                 #
# --------------------------------------------------------------------------- #
class PPM(nn.Module):
    def __init__(self, in_dim, pool_sizes=(1, 2, 3, 6), reduction_dim=128):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, 1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True),
            ) for _ in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_dim + len(pool_sizes) * reduction_dim, reduction_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [x]
        for stage, s in zip(self.stages, self.pool_sizes):
            # MPS-friendly: interpolate to exact (s,s) then average (equivalent to adaptive pool)
            out = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
            out = stage(out)
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
            feats.append(out)
        return self.bottleneck(torch.cat(feats, dim=1))


# --------------------------------------------------------------------------- #
#  PVT-SN: PVTv2 backbone + PPM + decoder                                     #
# --------------------------------------------------------------------------- #
class PVTSN(nn.Module):
    """PVTv2 backbone + PPM + concat decoder -> 1 logit map at 1/4 resolution."""

    def __init__(self, backbone="pvt_v2_b2", pretrained=True, num_classes=1,
                 decoder_dim=256, dropout=0.1):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        ch = self.backbone.feature_info.channels()
        self.lat = nn.ModuleList([nn.Conv2d(c, decoder_dim, 1) for c in ch])
        self.ppm = PPM(decoder_dim, (1, 2, 3, 6), decoder_dim // 2)
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4 + decoder_dim // 2, decoder_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        lats = [conv(f) for conv, f in zip(self.lat, feats)]
        top = self.ppm(lats[-1])
        # fuse at 1/4 resolution (lats[0]) to avoid MPS INT_MAX tensor-size overflow
        out_size = lats[0].shape[2:]
        up = [F.interpolate(t, size=out_size, mode="bilinear", align_corners=False) for t in lats]
        up.append(F.interpolate(top, size=out_size, mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(up, dim=1))


# --------------------------------------------------------------------------- #
#  SAM-RN: zero-shot SAM with box prompts from PVT-SN mask                    #
# --------------------------------------------------------------------------- #
class SAMRN:
    def __init__(self, model_name="facebook/sam-vit-base", device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_rgb, boxes):
        """image_rgb: HxWx3 uint8. boxes: list of [x0, y0, x1, y1]."""
        if not boxes:
            return np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        inputs = self.processor(image_rgb, input_boxes=[boxes], return_tensors="pt")
        device_inputs = {}
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                device_inputs[k] = v
            elif v.is_floating_point():
                device_inputs[k] = v.to(self.device, dtype=torch.float32)
            else:
                device_inputs[k] = v.to(self.device)
        outputs = self.model(**device_inputs)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )[0]  # (num_boxes, 3, H, W)
        union = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        scores = outputs.iou_scores.cpu()[0]  # (num_boxes, 3)
        for b in range(masks.shape[0]):
            best = int(scores[b].argmax())
            m = (masks[b, best].numpy() > 0.5).astype(np.uint8)
            union = np.logical_or(union, m).astype(np.uint8)
        return union


def mask_to_boxes(mask, min_area=100, max_boxes=64):
    """mask: HxW {0,1}. Returns list of [x0,y0,x1,y1] padded boxes."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        pad = max(2, int(0.05 * max(w, h)))
        boxes.append([max(0, x - pad), max(0, y - pad),
                      min(mask.shape[1], x + w + pad), min(mask.shape[0], y + h + pad)])
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:max_boxes]
    return boxes


# --------------------------------------------------------------------------- #
#  Weighted BCE loss (paper eq. 1)                                            #
# --------------------------------------------------------------------------- #
class WeightedBCE(nn.Module):
    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        fg = target.sum()
        bg = target.numel() - fg
        alpha = bg / (fg + bg + 1e-6)   # weight for foreground
        beta = fg / (fg + bg + 1e-6)    # weight for background
        loss = -(alpha * target * torch.log(p + 1e-6) + beta * (1 - target) * torch.log(1 - p + 1e-6))
        return loss.mean()


def metrics_from_mask(pred, gt, eps=1e-6):
    """pred, gt: boolean numpy arrays."""
    tp = np.logical_and(pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    sen = tp / (tp + fn + eps)
    spe = tn / (tn + fp + eps)
    pre = tp / (tp + fp + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return dict(SEN=sen, SPE=spe, PRE=pre, IoU=iou, DICE=dice)
