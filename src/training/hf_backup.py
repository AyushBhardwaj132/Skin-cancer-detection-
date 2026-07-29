from __future__ import annotations

import os
import time
import threading
from pathlib import Path


class HuggingFaceBackup:
    """Asynchronously uploads model checkpoints to Hugging Face Hub in a background thread."""

    def __init__(self, repo_id: str = "isic-2024-models", token: str | None = None):
        self.repo_id = repo_id
        self.token = token or os.getenv("HF_TOKEN")

        if not self.token:
            try:
                from kaggle_secrets import UserSecretsClient
                secrets = UserSecretsClient()
                self.token = secrets.get_secret("HF_TOKEN")
            except Exception:
                pass

        self._api = None
        self.is_available = False
        self._init_client()

    def _init_client(self) -> None:
        if not self.token:
            return
        try:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token)
            self.is_available = True
        except ImportError:
            print("  [HF BACKUP NOTICE] 'huggingface_hub' package not installed. Skipping HF upload.")
        except Exception as e:
            print(f"  [HF BACKUP NOTICE] Could not initialize HuggingFace API client: {e}")

    def upload_checkpoint_async(
        self,
        local_path: str | Path,
        fold_idx: int,
        model_name: str = "model",
        max_retries: int = 3,
    ) -> None:
        """Launches a non-blocking background thread to upload checkpoint to Hugging Face."""
        local_path = Path(local_path)
        if not local_path.exists():
            return

        if not self.is_available or self._api is None:
            print("  [HF BACKUP] Token missing or HF API client unavailable. Upload skipped.")
            return

        thread = threading.Thread(
            target=self._upload_worker,
            args=(local_path, fold_idx, model_name, max_retries),
            daemon=True,
            name=f"HFUpload-Fold{fold_idx}",
        )
        thread.start()

    def _upload_worker(
        self,
        local_path: Path,
        fold_idx: int,
        model_name: str,
        max_retries: int,
    ) -> None:
        path_in_repo = f"{model_name}/fold_{fold_idx}/{local_path.name}"
        
        for attempt in range(1, max_retries + 1):
            try:
                # Ensure repo exists
                try:
                    self._api.create_repo(repo_id=self.repo_id, exist_ok=True, private=True)
                except Exception:
                    pass

                self._api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                    repo_type="model",
                )
                print(f"\n  [HF BACKUP SUCCESS] Uploaded {local_path.name} to HF:{self.repo_id}/{path_in_repo} [OK]", flush=True)
                return
            except Exception as e:
                print(f"\n  [HF BACKUP WARNING] Attempt {attempt}/{max_retries} failed for {local_path.name}: {e}", flush=True)
                if attempt < max_retries:
                    time.sleep(2.0 * attempt)

        print(f"\n  [HF BACKUP FAILED] Could not upload {local_path.name} after {max_retries} retries. Continuing training.", flush=True)
