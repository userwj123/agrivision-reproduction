"""Python port of the official AgriVision DB-3 synthetic data generator (mainCode.m).

Reproduces the MATLAB logic 1:1 but vectorized with OpenCV/NumPy.
Outputs uncluttered and cluttered image/mask pairs, same naming and layout.
"""
import os
import glob
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def load_bmp(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.bmp")) +
                   glob.glob(os.path.join(folder, "*.png")) +
                   glob.glob(os.path.join(folder, "*.jpg")))
    return files


def generate(background_folder, fruit_folder, out_root,
             num_per_background=5,
             berry_count_range=(10, 50),
             clutter_percentage=0.3,
             clutter_ratio_range=(0.2, 0.5),
             max_berry_size_factor=0.07,
             size_variation_range=(0.6, 0.9),
             rotation_range=(0, 360),
             gaussian_blur_radius=1.5,
             edge_dilate_radius=2,
             edge_blur_radius=2,
             leaf_density_scale=1.5,
             green_hue_range=(0.25, 0.45),
             green_sat_min=0.3,
             alpha=0.9,
             seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    out_img = Path(out_root) / "Generated Images"
    out_mask = Path(out_root) / "Generated Masks"
    out_img_c = Path(out_root) / "Generated Images Cluttered"
    out_mask_c = Path(out_root) / "Generated Masks Cluttered"
    for d in (out_img, out_mask, out_img_c, out_mask_c):
        d.mkdir(parents=True, exist_ok=True)

    bg_files = load_bmp(background_folder)
    fruit_files = load_bmp(fruit_folder)
    if not bg_files or not fruit_files:
        raise FileNotFoundError("No background or fruit images found")

    # MATLAB does dir(...);dir(...) twice -> same list duplicated, effectively just the list
    total = num_per_background * len(bg_files)
    idx = 0
    for i in tqdm(range(total), desc="Generating"):
        idx += 1
        base_name = f"synthetic_image_{idx}"

        bg_path = bg_files[rng.integers(0, len(bg_files))]
        bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        if bg is None:
            continue
        bg_h, bg_w = bg.shape[:2]

        # green mask in HSV (OpenCV H: 0-179 -> map 0.25-0.45 to ~45-81)
        hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
        h_lo = int(round(green_hue_range[0] * 179))
        h_hi = int(round(green_hue_range[1] * 179))
        s_lo = int(round(green_sat_min * 255))
        green_mask = cv2.inRange(hsv, (h_lo, s_lo, 0), (h_hi, 255, 255)) > 0
        green_density = float(green_mask.sum()) / green_mask.size

        mask_img = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
        cluttered_mask_img = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
        cluttered_image = bg.copy()

        num_berries = int(rng.integers(berry_count_range[0], berry_count_range[1] + 1))
        n_cluttered = int(round(clutter_percentage * num_berries))
        cluttered_set = set(rng.choice(num_berries, size=min(n_cluttered, num_berries), replace=False).tolist())

        valid_yx = np.argwhere(green_mask)
        max_berry_size = max_berry_size_factor * min(bg_h, bg_w)

        for j in range(num_berries):
            fruit_path = fruit_files[rng.integers(0, len(fruit_files))]
            fruit = cv2.imread(fruit_path, cv2.IMREAD_COLOR)
            if fruit is None:
                continue
            fruit[fruit < 10] = 0
            fh, fw = fruit.shape[:2]

            scale_factor = (size_variation_range[0] + rng.random() * (size_variation_range[1] - size_variation_range[0])) \
                           * max_berry_size * (1 + green_density * leaf_density_scale) / max(fh, fw)
            new_w = max(1, int(round(fw * scale_factor)))
            new_h = max(1, int(round(fh * scale_factor)))
            fruit = cv2.resize(fruit, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            fh, fw = fruit.shape[:2]

            angle = float(rng.uniform(rotation_range[0], rotation_range[1]))
            M = cv2.getRotationMatrix2D((fw / 2, fh / 2), angle, 1.0)
            fruit = cv2.warpAffine(fruit, M, (fw, fh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            # gaussian blur with sigma = gaussian_blur_radius (MATLAB imgaussfilt uses sigma)
            fruit = cv2.GaussianBlur(fruit, (0, 0), sigmaX=gaussian_blur_radius, sigmaY=gaussian_blur_radius)

            cx, cy = fw // 2, fh // 2
            radius = min(fh, fw) // 2
            yy, xx = np.ogrid[:fh, :fw]
            circular_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2

            if len(valid_yx) == 0:
                continue
            y0, x0 = valid_yx[rng.integers(0, len(valid_yx))]
            x0 = max(0, min(bg_w - fw, int(x0)))
            y0 = max(0, min(bg_h - fh, int(y0)))

            is_cluttered = j in cluttered_set
            if is_cluttered:
                cluttered_mask = circular_mask.copy()
                side = int(rng.integers(1, 5))
                ratio = clutter_ratio_range[0] + (clutter_ratio_range[1] - clutter_ratio_range[0]) * rng.random()
                if side == 1:      # left
                    cluttered_mask[:, :int(round(ratio * fw))] = False
                elif side == 2:    # right
                    cluttered_mask[:, fw - int(round(ratio * fw)):] = False
                elif side == 3:    # top
                    cluttered_mask[:int(round(ratio * fh)), :] = False
                else:              # bottom
                    cluttered_mask[fh - int(round(ratio * fh)):, :] = False
            else:
                cluttered_mask = circular_mask

            roi_bg = bg[y0:y0 + fh, x0:x0 + fw]
            roi_cl = cluttered_image[y0:y0 + fh, x0:x0 + fw]

            cm = circular_mask[..., None]
            bg[y0:y0 + fh, x0:x0 + fw] = np.where(
                cm, (alpha * fruit + (1 - alpha) * roi_bg).astype(np.uint8), roi_bg)

            km = cluttered_mask[..., None]
            cluttered_image[y0:y0 + fh, x0:x0 + fw] = np.where(
                km, (alpha * fruit + (1 - alpha) * roi_cl).astype(np.uint8), roi_cl)

            mask_roi = mask_img[y0:y0 + fh, x0:x0 + fw]
            mask_img[y0:y0 + fh, x0:x0 + fw] = np.where(cm, 255, mask_roi)

            cmask_roi = cluttered_mask_img[y0:y0 + fh, x0:x0 + fw]
            cluttered_mask_img[y0:y0 + fh, x0:x0 + fw] = np.where(km, 255, cmask_roi)

        # edge smoothing (Canny -> dilate -> Gaussian blur)
        gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_mask, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * edge_dilate_radius + 1, 2 * edge_dilate_radius + 1))
        edges_d = cv2.dilate(edges, kernel)
        edge_blur = cv2.GaussianBlur(edges_d.astype(np.float32), (0, 0),
                                     sigmaX=edge_blur_radius, sigmaY=edge_blur_radius)
        eb = edge_blur[..., None]

        bg = (eb * bg + (1 - eb) * bg).astype(np.uint8)  # no-op like MATLAB, but kept for parity
        cluttered_image = (eb * cluttered_image + (1 - eb) * cluttered_image).astype(np.uint8)

        cv2.imwrite(str(out_img / f"{base_name}.bmp"), bg)
        cv2.imwrite(str(out_mask / f"{base_name}.png"), mask_img)
        cv2.imwrite(str(out_img_c / f"{base_name}.bmp"), cluttered_image)
        cv2.imwrite(str(out_mask_c / f"{base_name}.png"), cluttered_mask_img)

    print(f"Done. {idx} image/mask pairs written to {out_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backgrounds", required=True)
    ap.add_argument("--fruits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-per-bg", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.backgrounds, args.fruits, args.out,
             num_per_background=args.num_per_bg, seed=args.seed)
