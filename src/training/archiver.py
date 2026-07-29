from __future__ import annotations

import sys
import shutil
import zipfile
from pathlib import Path


def create_fold_artifact_zip(output_dir: str | Path, fold_idx: int) -> Path | None:
    """Packages all key fold training artifacts into fold_{fold_idx}_artifacts.zip."""
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    eval_dir = output_dir / "evaluation"

    zip_filename = f"fold_{fold_idx}_artifacts.zip"
    zip_path = checkpoints_dir / zip_filename

    files_to_pack: list[Path] = []

    # 1. Checkpoints
    for ckpt_name in [f"best_model_fold{fold_idx}.pt", f"last_checkpoint_fold{fold_idx}.pt"]:
        # Check fold subfolder and root checkpoints dir
        found = list(checkpoints_dir.rglob(ckpt_name))
        for f in found:
            if f.exists() and f.stat().st_size > 0:
                files_to_pack.append(f)
                break

    # 2. History CSV
    for csv_file in logs_dir.glob(f"*fold{fold_idx}*.csv"):
        if csv_file.exists():
            files_to_pack.append(csv_file)

    # 3. Training Curves PNG
    for fig_file in figures_dir.glob(f"*fold{fold_idx}*.png"):
        if fig_file.exists():
            files_to_pack.append(fig_file)

    # 4. Evaluation Metrics / Diagnostics JSON
    for json_file in eval_dir.glob(f"*fold{fold_idx}*.json"):
        if json_file.exists():
            files_to_pack.append(json_file)

    # 5. Training State & Status
    for meta_file in [output_dir / "training_state.json", output_dir / "status.json"]:
        if meta_file.exists():
            files_to_pack.append(meta_file)

    if not files_to_pack:
        print(f"  [ARCHIVER WARN] No artifact files found for Fold {fold_idx}. Skipping zip creation.", flush=True)
        return None

    try:
        print(f"\n  [ARCHIVER] Creating {zip_filename} with {len(files_to_pack)} files...", flush=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files_to_pack:
                arcname = f"fold_{fold_idx}/{file_path.name}"
                zipf.write(file_path, arcname=arcname)

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"  [ARCHIVER SUCCESS] Fold {fold_idx} artifact zip created: {zip_path.name} ({size_mb:.2f} MB) [OK]", flush=True)
        return zip_path
    except Exception as e:
        print(f"  [ARCHIVER ERROR] Failed to create artifact zip for Fold {fold_idx}: {e}", flush=True)
        return None
