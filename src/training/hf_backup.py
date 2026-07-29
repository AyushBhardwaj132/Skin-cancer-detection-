from __future__ import annotations

import os
import sys
import time
import traceback
import threading
from pathlib import Path


class HuggingFaceBackup:
    """Asynchronously uploads model checkpoints and fold artifacts to Hugging Face Hub."""

    def __init__(
        self,
        repo_id: str = "ayushbhar/isic-2024-checkpoints",
        token: str | None = None,
        max_retries: int = 5,
    ):
        self.repo_id = repo_id
        self.token = self._resolve_token(token)
        self.max_retries = max_retries
        self._api = None
        self.is_available = False
        self.username = None

        self._init_client()

    @staticmethod
    def _resolve_token(explicit_token: str | None) -> str | None:
        """Resolves HF_TOKEN from explicit args, environment variables, HF cache, or Kaggle secrets."""
        if explicit_token:
            return explicit_token

        for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            tok = os.getenv(env_var)
            if tok and tok.strip():
                return tok.strip()

        try:
            from huggingface_hub import HfFolder
            cache_tok = HfFolder.get_token()
            if cache_tok and cache_tok.strip():
                return cache_tok.strip()
        except Exception:
            pass

        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            k_tok = secrets.get_secret("HF_TOKEN")
            if k_tok and k_tok.strip():
                return k_tok.strip()
        except Exception:
            pass

        return None

    def _init_client(self) -> None:
        if not self.token:
            print("[HF WARNING] HF_TOKEN is not set or empty. Hugging Face uploads disabled.", flush=True)
            return

        try:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token)
            
            try:
                user_info = self._api.whoami()
                self.username = user_info.get("name") or user_info.get("fullname") or "authenticated_user"
                print(f"[HF] Authentication successful (User: {self.username})", flush=True)
                self.is_available = True
            except Exception as auth_err:
                print(f"[HF ERROR] Authentication failed with provided token: {auth_err}", flush=True)
                traceback.print_exc()
                self.is_available = False

        except ImportError:
            print("[HF WARNING] 'huggingface_hub' library not installed. Run 'pip install huggingface_hub'.", flush=True)
        except Exception as e:
            print(f"[HF ERROR] Could not initialize Hugging Face API client: {e}", flush=True)
            traceback.print_exc()

    def perform_self_test(self, test_upload: bool = True) -> bool:
        """Runs startup self-test confirming token validity, repo accessibility, and write permission."""
        print(f"\n{'='*60}", flush=True)
        print("RUNNING HUGGING FACE STARTUP SELF-TEST", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Target Repository: {self.repo_id}", flush=True)

        if not self.token or not self.is_available or self._api is None:
            print("[HF ERROR] Self-test FAILED: HF_TOKEN missing or authentication failed.", flush=True)
            print(f"{'='*60}\n", flush=True)
            return False

        try:
            user_info = self._api.whoami()
            print(f"[HF] Authentication verified for user: {user_info.get('name')}", flush=True)

            try:
                self._api.create_repo(repo_id=self.repo_id, exist_ok=True, private=True)
                print(f"[HF] Repository '{self.repo_id}' checked / created [OK]", flush=True)
            except Exception as repo_err:
                print(f"[HF ERROR] Repository check/creation failed: {repo_err}", flush=True)
                traceback.print_exc()
                return False

            if test_upload:
                print(f"[HF] Uploading startup test file to '{self.repo_id}'...", flush=True)
                test_content = f"ISIC 2024 Pipeline Startup Test - {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                test_file_path = Path("hf_startup_test.tmp")
                with open(test_file_path, "w", encoding="utf-8") as f:
                    f.write(test_content)

                try:
                    self._api.upload_file(
                        path_or_fileobj=str(test_file_path),
                        path_in_repo="hf_startup_test.txt",
                        repo_id=self.repo_id,
                        repo_type="model",
                    )
                    print(f"[HF BACKUP SUCCESS] Self-test file verified at HF:{self.repo_id}/hf_startup_test.txt [OK]", flush=True)
                finally:
                    if test_file_path.exists():
                        test_file_path.unlink()

            print(f"[HF BACKUP SUCCESS] Startup self-test PASSED! Repo '{self.repo_id}' ready for backups.", flush=True)
            print(f"{'='*60}\n", flush=True)
            return True

        except Exception as e:
            print(f"[HF BACKUP FAILED] Startup self-test FAILED with exception: {e}", flush=True)
            traceback.print_exc()
            print(f"{'='*60}\n", flush=True)
            return False

    def upload_checkpoint_async(
        self,
        local_path: str | Path,
        fold_idx: int,
        model_name: str = "model",
        subfolder: str | None = None,
    ) -> None:
        """Launches a non-blocking background thread to upload checkpoint to Hugging Face."""
        local_path = Path(local_path)
        if not local_path.exists():
            print(f"[HF WARNING] File to upload does not exist: {local_path}", flush=True)
            return

        if not self.is_available or self._api is None:
            print(f"[HF WARNING] Upload skipped for {local_path.name}: HF client not authenticated.", flush=True)
            return

        thread = threading.Thread(
            target=self._upload_worker,
            args=(local_path, fold_idx, model_name, subfolder),
            daemon=True,
            name=f"HFUpload-Fold{fold_idx}-{local_path.name}",
        )
        thread.start()

    def upload_fold_artifacts_async(
        self,
        output_dir: str | Path,
        fold_idx: int,
        model_name: str = "model",
        zip_path: str | Path | None = None,
    ) -> None:
        """Asynchronously uploads all fold artifacts (checkpoints, state, zip) to Hugging Face."""
        output_dir = Path(output_dir)
        checkpoints_dir = output_dir / "checkpoints"
        logs_dir = output_dir / "logs"
        figures_dir = output_dir / "figures"
        eval_dir = output_dir / "evaluation"

        files_to_upload: list[tuple[Path, str]] = []

        # 1. Best model & Last checkpoint
        for candidate_name in [f"best_model_fold{fold_idx}.pt", f"last_checkpoint_fold{fold_idx}.pt"]:
            found = list(checkpoints_dir.rglob(candidate_name))
            for f in found:
                if f.exists() and f.stat().st_size > 0:
                    clean_model = model_name.replace("tf_", "").replace("-", "_")
                    sub = f"{clean_model}/fold_{fold_idx}"
                    files_to_upload.append((f, sub))
                    break

        # 2. State & Status
        for state_file in [output_dir / "training_state.json", output_dir / "status.json"]:
            if state_file.exists():
                files_to_upload.append((state_file, "meta"))

        # 3. CSV logs & training curves
        for csv_file in logs_dir.glob(f"*fold{fold_idx}*.csv"):
            if csv_file.exists():
                files_to_upload.append((csv_file, f"logs/fold_{fold_idx}"))
        for fig_file in figures_dir.glob(f"*fold{fold_idx}*.png"):
            if fig_file.exists():
                files_to_upload.append((fig_file, f"figures/fold_{fold_idx}"))

        # 4. Diagnostic JSON
        for diag_file in eval_dir.glob(f"*fold{fold_idx}*.json"):
            if diag_file.exists():
                files_to_upload.append((diag_file, f"evaluation/fold_{fold_idx}"))

        # 5. Zip artifact
        if zip_path and Path(zip_path).exists():
            files_to_upload.append((Path(zip_path), f"archives/fold_{fold_idx}"))

        # Launch non-blocking background uploads
        for local_file, sub_dir in files_to_upload:
            self.upload_checkpoint_async(
                local_path=local_file,
                fold_idx=fold_idx,
                model_name=model_name,
                subfolder=sub_dir,
            )

    def _upload_worker(
        self,
        local_path: Path,
        fold_idx: int,
        model_name: str,
        subfolder: str | None = None,
    ) -> bool:
        clean_model_name = model_name.replace("tf_", "").replace("-", "_")
        if subfolder:
            path_in_repo = f"{subfolder}/{local_path.name}"
        else:
            path_in_repo = f"{clean_model_name}/fold_{fold_idx}/{local_path.name}"
        
        print(f"\n[HF] Uploading {local_path.name} to {self.repo_id} ({path_in_repo})...", flush=True)

        for attempt in range(1, self.max_retries + 1):
            try:
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
                print(f"[HF BACKUP SUCCESS] Uploaded {local_path.name} to HF:{self.repo_id}/{path_in_repo} [OK]", flush=True)
                return True

            except Exception as e:
                print(f"\n[HF BACKUP FAILED] Attempt {attempt}/{self.max_retries} failed to upload {local_path.name}:", flush=True)
                traceback.print_exc()
                if attempt < self.max_retries:
                    backoff = 2.0 ** attempt  # 2s, 4s, 8s, 16s, 32s
                    time.sleep(backoff)

        print(f"[HF BACKUP FAILED] All {self.max_retries} upload attempts failed for {local_path.name}. Continuing training.", flush=True)
        return False
