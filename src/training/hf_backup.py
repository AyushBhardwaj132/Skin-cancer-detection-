from __future__ import annotations

import os
import sys
import time
import traceback
import threading
from pathlib import Path


class HuggingFaceBackup:
    """Asynchronously uploads model checkpoints to Hugging Face Hub in a background thread."""

    def __init__(self, repo_id: str = "ayushbhar/isic-2024-checkpoints", token: str | None = None):
        self.repo_id = repo_id
        self.token = self._resolve_token(token)
        self._api = None
        self.is_available = False
        self.username = None

        self._init_client()

    @staticmethod
    def _resolve_token(explicit_token: str | None) -> str | None:
        """Resolves HF_TOKEN from explicit args, environment variables, HF cache, or Kaggle secrets."""
        if explicit_token:
            return explicit_token

        # Check environment variables
        for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            tok = os.getenv(env_var)
            if tok and tok.strip():
                return tok.strip()

        # Check huggingface_hub cache token
        try:
            from huggingface_hub import HfFolder
            cache_tok = HfFolder.get_token()
            if cache_tok and cache_tok.strip():
                return cache_tok.strip()
        except Exception:
            pass

        # Check Kaggle UserSecretsClient
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
            
            # Verify token via whoami()
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
            # 1. Verify whoami
            user_info = self._api.whoami()
            print(f"[HF] Authentication verified for user: {user_info.get('name')}", flush=True)

            # 2. Verify or create repository
            try:
                self._api.create_repo(repo_id=self.repo_id, exist_ok=True, private=True)
                print(f"[HF] Repository '{self.repo_id}' checked / created [OK]", flush=True)
            except Exception as repo_err:
                print(f"[HF ERROR] Repository check/creation failed: {repo_err}", flush=True)
                traceback.print_exc()
                return False

            # 3. Test upload if requested
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
                    print(f"[HF] Upload complete. Self-test file verified at HF:{self.repo_id}/hf_startup_test.txt [OK]", flush=True)
                finally:
                    if test_file_path.exists():
                        test_file_path.unlink()

            print(f"[HF] Startup self-test PASSED! Repository '{self.repo_id}' is ready for model backups.", flush=True)
            print(f"{'='*60}\n", flush=True)
            return True

        except Exception as e:
            print(f"[HF ERROR] Startup self-test FAILED with exception: {e}", flush=True)
            traceback.print_exc()
            print(f"{'='*60}\n", flush=True)
            return False

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
            print(f"[HF WARNING] File to upload does not exist: {local_path}", flush=True)
            return

        if not self.is_available or self._api is None:
            print(f"[HF WARNING] Upload skipped for {local_path.name}: HF client not authenticated.", flush=True)
            return

        thread = threading.Thread(
            target=self._upload_worker,
            args=(local_path, fold_idx, model_name, max_retries),
            daemon=True,
            name=f"HFUpload-Fold{fold_idx}-{local_path.name}",
        )
        thread.start()

    def upload_checkpoint_sync(
        self,
        local_path: str | Path,
        fold_idx: int,
        model_name: str = "model",
        max_retries: int = 3,
    ) -> bool:
        """Synchronous version of upload for testing or explicit blocking requirements."""
        local_path = Path(local_path)
        if not local_path.exists() or not self.is_available or self._api is None:
            return False
        return self._upload_worker(local_path, fold_idx, model_name, max_retries)

    def _upload_worker(
        self,
        local_path: Path,
        fold_idx: int,
        model_name: str,
        max_retries: int,
    ) -> bool:
        clean_model_name = model_name.replace("tf_", "").replace("-", "_")
        path_in_repo = f"{clean_model_name}/fold_{fold_idx}/{local_path.name}"
        
        print(f"\n[HF] Uploading {local_path.name} to {self.repo_id} ({path_in_repo})...", flush=True)

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
                print(f"[HF] Upload complete: {local_path.name} -> HF:{self.repo_id}/{path_in_repo} [OK]", flush=True)
                return True

            except Exception as e:
                print(f"\n[HF ERROR] Attempt {attempt}/{max_retries} failed to upload {local_path.name}:", flush=True)
                traceback.print_exc()
                if attempt < max_retries:
                    time.sleep(2.0 * attempt)

        print(f"[HF ERROR] All {max_retries} upload attempts failed for {local_path.name}. Continuing training.", flush=True)
        return False
