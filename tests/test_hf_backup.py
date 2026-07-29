from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.hf_backup import HuggingFaceBackup
from src.config.config import Config


def test_hf_token_resolution():
    print("\n" + "=" * 80)
    print("TEST 1: Hugging Face Token Resolution & Initialization")
    print("=" * 80)

    config = Config()
    print(f"  Configured repo_id: {config.hf_repo_id}")
    assert config.hf_repo_id == "ayushbhar/isic-2024-checkpoints", f"Unexpected repo_id: {config.hf_repo_id}"

    hf_backup = HuggingFaceBackup(repo_id=config.hf_repo_id)
    print(f"  HF Token Resolved: {'YES' if hf_backup.token else 'NO'}")
    print(f"  HF Client Available: {'YES' if hf_backup.is_available else 'NO'}")
    if hf_backup.username:
        print(f"  HF Authenticated User: {hf_backup.username}")


def test_hf_self_test_and_upload():
    print("\n" + "=" * 80)
    print("TEST 2: Hugging Face Self-Test & File Upload")
    print("=" * 80)

    config = Config()
    hf_backup = HuggingFaceBackup(repo_id=config.hf_repo_id)

    if not hf_backup.is_available:
        print("  [NOTICE] HF_TOKEN is not set in local environment. Self-test will report missing token.")
        print("  To enable live uploads locally, run: $env:HF_TOKEN='your_hf_token'")
        return

    # Run self-test with upload
    success = hf_backup.perform_self_test(test_upload=True)
    assert success, "Hugging Face self-test failed!"

    # Test uploading a mock model checkpoint
    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_ckpt = Path(tmp_dir) / "best_model_fold0.pt"
        with open(mock_ckpt, "wb") as f:
            f.write(b"MOCK_MODEL_WEIGHTS_FOR_TESTING")

        print(f"\n  Testing synchronous upload of mock checkpoint {mock_ckpt.name}...")
        uploaded = hf_backup.upload_checkpoint_sync(
            local_path=mock_ckpt,
            fold_idx=0,
            model_name="tf_efficientnetv2_s",
        )
        assert uploaded, "Synchronous checkpoint upload failed!"
        print("  [PASS] Synchronous mock checkpoint upload succeeded!")


def main():
    print("=" * 80)
    print("ISIC 2024 — HUGGING FACE BACKUP TEST SUITE")
    print("=" * 80)

    test_hf_token_resolution()
    test_hf_self_test_and_upload()

    print("\n" + "=" * 80)
    print("[PASS] HUGGING FACE BACKUP TEST SUITE FINISHED CLEANLY")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
