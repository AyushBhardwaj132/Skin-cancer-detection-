import os
import sys
import random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import resolve_image_path, ensure_dir

def run_image_quality_audit(sample_size=300, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    dev_train_path = PROJECT_ROOT / "data" / "dev_train.csv"
    train_meta_path = PROJECT_ROOT / "data" / "train-metadata.csv"
    image_dir = PROJECT_ROOT / "data" / "train-image"

    if dev_train_path.exists():
        df = pd.read_csv(dev_train_path)
    else:
        df = pd.read_csv(train_meta_path)

    sample_df = df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True)

    audit_out_dir = PROJECT_ROOT / "outputs" / "audit"
    examples_dir = audit_out_dir / "examples"
    ensure_dir(examples_dir)

    results = {
        "total_audited": len(sample_df),
        "black_borders": [],
        "vignetting": [],
        "background_artifacts": [],
        "ruler_markings": [],
        "hair_occlusion": [],
        "blurry_images": [],
        "illumination_variation": [],
        "off_center_lesions": [],
        "lesion_sizes": {"small": 0, "medium": 0, "large": 0},
        "color_distribution": {"light": 0, "medium": 0, "dark": 0},
    }

    # Save example flags to limit 3 examples per category
    example_counts = {
        "black_border": 0,
        "vignetting": 0,
        "background_artifact": 0,
        "ruler_marking": 0,
        "hair_occlusion": 0,
        "blurry": 0,
        "illumination": 0,
        "off_center": 0,
        "color_var": 0,
        "lesion_size": 0,
    }

    for idx, row in sample_df.iterrows():
        image_id = str(row["isic_id"])
        try:
            img_path = resolve_image_path(image_dir, image_id)
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
        except Exception as e:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Black Borders Detection
        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_thresh = 15
        top = np.mean(gray[: int(h * 0.05), :]) < border_thresh
        bottom = np.mean(gray[int(h * 0.95) :, :]) < border_thresh
        left = np.mean(gray[:, : int(w * 0.05)]) < border_thresh
        right = np.mean(gray[:, int(w * 0.95) :]) < border_thresh
        has_black_border = top or bottom or left or right
        if has_black_border:
            results["black_borders"].append(image_id)
            if example_counts["black_border"] < 3:
                cv2.imwrite(str(examples_dir / f"black_border_{image_id}.jpg"), img)
                example_counts["black_border"] += 1

        # 2. Vignetting Detection (corner brightness vs center brightness)
        center_region = gray[int(h * 0.35) : int(h * 0.65), int(w * 0.35) : int(w * 0.65)]
        corners = [
            gray[: int(h * 0.15), : int(w * 0.15)],
            gray[: int(h * 0.15), int(w * 0.85) :],
            gray[int(h * 0.85) :, : int(w * 0.15)],
            gray[int(h * 0.85) :, int(w * 0.85) :],
        ]
        mean_center = np.mean(center_region)
        mean_corners = np.mean([np.mean(c) for c in corners])
        has_vignetting = (mean_center - mean_corners) > 40
        if has_vignetting:
            results["vignetting"].append(image_id)
            if example_counts["vignetting"] < 3:
                cv2.imwrite(str(examples_dir / f"vignetting_{image_id}.jpg"), img)
                example_counts["vignetting"] += 1

        # 3. Background Artifacts (tape/gel bubbles/stickers)
        # Detected via connected components in non-central peripheral regions
        blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
        has_artifact = False
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            # Check if artifact is near boundary and disconnected from center
            if area > (h * w * 0.01) and (cx < w * 0.15 or cx > w * 0.85 or cy < h * 0.15 or cy > h * 0.85):
                has_artifact = True
                break
        if has_artifact:
            results["background_artifacts"].append(image_id)
            if example_counts["background_artifact"] < 3:
                cv2.imwrite(str(examples_dir / f"background_artifact_{image_id}.jpg"), img)
                example_counts["background_artifact"] += 1

        # 4. Ruler Markings (Hough lines in peripheral regions)
        edges = cv2.Canny(gray, 100, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=5)
        has_ruler = False
        if lines is not None:
            for line in lines:
                line_pts = line.ravel()
                if len(line_pts) == 4:
                    x1, y1, x2, y2 = line_pts
                    length = np.hypot(x2 - x1, y2 - y1)
                    # Check line near edges
                    if length > 40 and (x1 < w * 0.2 or x2 > w * 0.8 or y1 < h * 0.2 or y2 > h * 0.8):
                        has_ruler = True
                        break
        if has_ruler:
            results["ruler_markings"].append(image_id)
            if example_counts["ruler_marking"] < 3:
                cv2.imwrite(str(examples_dir / f"ruler_marking_{image_id}.jpg"), img)
                example_counts["ruler_marking"] += 1

        # 5. Hair Occlusion (Black Top-Hat Morphological Filter)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        hair_density = np.mean(blackhat > 30)
        has_hair = hair_density > 0.015
        if has_hair:
            results["hair_occlusion"].append(image_id)
            if example_counts["hair_occlusion"] < 3:
                cv2.imwrite(str(examples_dir / f"hair_occlusion_{image_id}.jpg"), img)
                example_counts["hair_occlusion"] += 1

        # 6. Blurry Images (Laplacian Variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < 80.0
        if is_blurry:
            results["blurry_images"].append(image_id)
            if example_counts["blurry"] < 3:
                cv2.imwrite(str(examples_dir / f"blurry_{image_id}.jpg"), img)
                example_counts["blurry"] += 1

        # 7. Illumination Variation across 4 quadrants
        quad1 = np.mean(gray[: h // 2, : w // 2])
        quad2 = np.mean(gray[: h // 2, w // 2 :])
        quad3 = np.mean(gray[h // 2 :, : w // 2])
        quad4 = np.mean(gray[h // 2 :, w // 2 :])
        illum_diff = max(quad1, quad2, quad3, quad4) - min(quad1, quad2, quad3, quad4)
        has_illum_var = illum_diff > 35.0
        if has_illum_var:
            results["illumination_variation"].append(image_id)
            if example_counts["illumination"] < 3:
                cv2.imwrite(str(examples_dir / f"illumination_var_{image_id}.jpg"), img)
                example_counts["illumination"] += 1

        # 8. Lesion Size & Centroid Position Estimation (Otsu segmentation on inverted gray/saturation)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        _, lesion_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        lesion_pixel_ratio = np.mean(lesion_mask > 0)
        if lesion_pixel_ratio < 0.08:
            results["lesion_sizes"]["small"] += 1
        elif lesion_pixel_ratio > 0.35:
            results["lesion_sizes"]["large"] += 1
        else:
            results["lesion_sizes"]["medium"] += 1

        M = cv2.moments(lesion_mask)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist_from_center = np.hypot(cx - w / 2, cy - h / 2) / (np.hypot(w, h) / 2)
            if dist_from_center > 0.18:
                results["off_center_lesions"].append(image_id)
                if example_counts["off_center"] < 3:
                    cv2.imwrite(str(examples_dir / f"off_center_{image_id}.jpg"), img)
                    example_counts["off_center"] += 1

        # 9. Color Variation (L*a*b* Luminance & Skin tone classification)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        mean_l = np.mean(l_channel)
        if mean_l > 170:
            results["color_distribution"]["light"] += 1
        elif mean_l < 100:
            results["color_distribution"]["dark"] += 1
        else:
            results["color_distribution"]["medium"] += 1

    audit_summary_path = audit_out_dir / "audit_summary.json"
    with open(audit_summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Audit completed on {len(sample_df)} images!")
    print(f"  Black borders:          {len(results['black_borders'])} ({len(results['black_borders'])/len(sample_df)*100:.1f}%)")
    print(f"  Vignetting:             {len(results['vignetting'])} ({len(results['vignetting'])/len(sample_df)*100:.1f}%)")
    print(f"  Background artifacts:   {len(results['background_artifacts'])} ({len(results['background_artifacts'])/len(sample_df)*100:.1f}%)")
    print(f"  Ruler markings:         {len(results['ruler_markings'])} ({len(results['ruler_markings'])/len(sample_df)*100:.1f}%)")
    print(f"  Hair occlusion:         {len(results['hair_occlusion'])} ({len(results['hair_occlusion'])/len(sample_df)*100:.1f}%)")
    print(f"  Blurry images:          {len(results['blurry_images'])} ({len(results['blurry_images'])/len(sample_df)*100:.1f}%)")
    print(f"  Illumination variation: {len(results['illumination_variation'])} ({len(results['illumination_variation'])/len(sample_df)*100:.1f}%)")
    print(f"  Off-center lesions:     {len(results['off_center_lesions'])} ({len(results['off_center_lesions'])/len(sample_df)*100:.1f}%)")
    print(f"  Lesion sizes:           {results['lesion_sizes']}")
    print(f"  Color distribution:     {results['color_distribution']}")
    print(f"  Examples saved to:      {examples_dir}")

    return results

if __name__ == "__main__":
    run_image_quality_audit(300)
