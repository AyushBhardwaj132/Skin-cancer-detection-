import os
import sys
import random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import resolve_image_path, ensure_dir


def generate_crop_visualizations(n_samples=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    dev_train_path = PROJECT_ROOT / "data" / "dev_train.csv"
    train_meta_path = PROJECT_ROOT / "data" / "train-metadata.csv"
    image_dir = PROJECT_ROOT / "data" / "train-image"

    if dev_train_path.exists():
        df = pd.read_csv(dev_train_path)
    else:
        df = pd.read_csv(train_meta_path)

    sample_df = df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)

    out_dir = PROJECT_ROOT / "outputs" / "evaluation" / "exp3_preprocessing" / "crop_visualization"
    ensure_dir(out_dir)

    successful_crops = 0
    total_processed = len(sample_df)
    results = []

    margin = 0.20

    for idx, row in sample_df.iterrows():
        image_id = str(row["isic_id"])
        try:
            img_path = resolve_image_path(image_dir, image_id)
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
        except Exception:
            continue

        # Step 1: Grayscale & Gaussian Blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Step 2: Otsu Thresholding
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Step 3: Contour Detection
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        is_valid_crop = False
        cropped_rgb = img_rgb.copy()

        if contours:
            c_max = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c_max)
            if (h * w * 0.01) <= area <= (h * w * 0.95):
                x, y, bw, bh = cv2.boundingRect(c_max)
                cx, cy = x + bw / 2.0, y + bh / 2.0
                side = max(bw, bh) * (1.0 + margin)

                x1 = max(0, int(cx - side / 2.0))
                y1 = max(0, int(cy - side / 2.0))
                x2 = min(w, int(cx + side / 2.0))
                y2 = min(h, int(cy + side / 2.0))

                cropped_rgb = img_rgb[y1:y2, x1:x2]
                if cropped_rgb.size > 0 and cropped_rgb.shape[0] >= 10 and cropped_rgb.shape[1] >= 10:
                    is_valid_crop = True

        if is_valid_crop:
            successful_crops += 1

        # Resize components for side-by-side visualization
        vis_h = 300
        orig_vis = cv2.resize(img_rgb, (vis_h, vis_h))
        thresh_vis = cv2.resize(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB), (vis_h, vis_h))
        crop_vis = cv2.resize(cropped_rgb, (vis_h, vis_h))

        # Annotate each panel
        cv2.putText(orig_vis, f"ORIGINAL: {image_id}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(thresh_vis, "OTSU MASK", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        status_color = (0, 255, 0) if is_valid_crop else (255, 0, 0)
        cv2.putText(crop_vis, f"CROP: {'OK' if is_valid_crop else 'FALLBACK'}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

        # Stitch LEFT, CENTER, RIGHT horizontally
        triptych = np.hstack([orig_vis, thresh_vis, crop_vis])
        triptych_bgr = cv2.cvtColor(triptych, cv2.COLOR_RGB2BGR)

        out_filename = out_dir / f"crop_{idx+1:02d}_{image_id}.jpg"
        cv2.imwrite(str(out_filename), triptych_bgr)

        results.append({
            "sample_index": idx + 1,
            "image_id": image_id,
            "crop_success": is_valid_crop,
            "visualization_path": str(out_filename),
        })

    success_rate = (successful_crops / total_processed) * 100.0 if total_processed > 0 else 0.0

    print(f"\n" + "=" * 80)
    print(f"CROP VISUALIZATION SUMMARY ({total_processed} Samples)")
    print(f"=" * 80)
    print(f"  Successful Crops:  {successful_crops} / {total_processed}")
    print(f"  Successful Rate:   {success_rate:.1f}%")
    print(f"  Output Directory:  {out_dir}")
    print(f"=" * 80 + "\n")

    return {
        "successful_crops": successful_crops,
        "total_samples": total_processed,
        "success_rate_percent": success_rate,
        "results": results,
    }

if __name__ == "__main__":
    generate_crop_visualizations(50)
