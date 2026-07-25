# ISIC 2024 Dataset Verification Report

**Generated**: 2026-07-26  
**Dataset Source**: Official ISIC 2024 Kaggle Competition  
**Location**: `data/`

---

## Summary

| Metric | Value |
|---|---|
| Total training metadata rows | 37,724 |
| Total training images on disk | 37,724 |
| Image-to-metadata mapping | **100% (0 missing)** |
| Benign samples (target=0) | 35,864 (95.07%) |
| Malignant samples (target=1) | 1,860 (4.93%) |
| Unique patients | 11,978 |
| Unique malignant patients | 1,726 |
| Class imbalance ratio | 19.28 : 1 |

---

## File Inventory

| File | Size | Status |
|---|---|---|
| `data/train-metadata.csv` | ~25 MB | ✅ Present |
| `data/train-image/image/` | 37,724 images | ✅ All present |
| `data/train-image.hdf5` | ~1.2 GB | ✅ Present |
| `data/test-metadata.csv` | ~2 KB | ✅ Present |
| `data/test-image/` | test images | ✅ Present |
| `data/test-image.hdf5` | ~10 KB | ✅ Present |
| `data/sample_submission.csv` | 66 B | ✅ Present |

---

## Metadata Columns (42 total)

| Column | Type | Description |
|---|---|---|
| `isic_id` | string | Unique image identifier |
| `patient_id` | string | Patient identifier for grouping |
| `age_approx` | numeric | Approximate patient age |
| `sex` | categorical | Patient sex |
| `anatom_site_general` | categorical | Anatomical location |
| `clin_size_long_diam_mm` | numeric | Clinical lesion diameter |
| `target` | binary | 0=benign, 1=malignant |
| `image_type` | categorical | Imaging type |
| `tbp_tile_type` | categorical | Total body photo tile type |
| `tbp_lv_location_simple` | categorical | Simplified location |
| `tbp_lv_*` | numeric (32 cols) | Total body photo lesion features |

---

## Class Distribution

```
Benign (0):    ████████████████████████████████████████████████ 35,864 (95.07%)
Malignant (1): ██                                               1,860  (4.93%)
```

---

## Patient Distribution

- **11,978** unique patients total
- **1,726** unique patients with at least one malignant lesion
- Most malignant patients have **1 malignant image** (1,596 patients)
- Maximum malignant images per patient: **3**

---

## Image-Metadata Integrity Check

Every `isic_id` in `train-metadata.csv` maps to an existing image file in `data/train-image/image/`.

- **37,724 / 37,724** images verified ✅
- **0 missing** images
- Image format: JPEG (`.jpg`)

---

## Verification Method

Verification performed by iterating all 37,724 `isic_id` values and checking file existence against `data/train-image/image/{isic_id}.jpg`. All lookups succeeded.
