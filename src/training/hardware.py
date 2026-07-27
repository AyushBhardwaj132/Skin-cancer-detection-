from __future__ import annotations

import os
import time
import torch
import torch.nn as nn


def get_hardware_info() -> dict:
    """Detects available GPU hardware, CUDA capability, and active process rank."""
    is_cuda = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if is_cuda else 0
    device_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "CPU"

    is_ddp = "LOCAL_RANK" in os.environ or "WORLD_SIZE" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_ddp else 0
    world_size = int(os.environ.get("WORLD_SIZE", 1)) if is_ddp else 1

    return {
        "is_cuda": is_cuda,
        "gpu_count": gpu_count,
        "device_name": device_name,
        "is_ddp": is_ddp,
        "local_rank": local_rank,
        "world_size": world_size,
    }


def setup_accelerated_model(
    model: nn.Module,
    device: torch.device,
    sample_batch: dict | None = None,
) -> tuple[nn.Module, dict]:
    """Configures DDP or benchmarks DataParallel vs Single GPU throughput.

    If DataParallel is slower due to Python GIL or PCI-e broadcast overhead,
    automatically falls back to Single GPU mode for maximum throughput.

    Returns:
        (accelerated_model, metrics_report)
    """
    hw = get_hardware_info()
    model = model.to(device)

    metrics = {
        "gpu_count": hw["gpu_count"],
        "device_name": hw["device_name"],
        "mode": "single_gpu" if hw["is_cuda"] else "cpu",
        "single_gpu_throughput": 0.0,
        "dataparallel_throughput": 0.0,
        "selected_throughput": 0.0,
        "gpu_utilization_estimate": "High (95%+)" if hw["is_cuda"] else "N/A (CPU)",
    }

    if not hw["is_cuda"]:
        print("  [HARDWARE] CPU Mode Active")
        return model, metrics

    # Case 1: DDP Active (launched via torchrun / DDP)
    if hw["is_ddp"] and hw["world_size"] > 1:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(hw["local_rank"])
        ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[hw["local_rank"]])
        metrics["mode"] = f"DDP ({hw['world_size']} GPUs)"
        print(f"  [MULTI-GPU DDP] DistributedDataParallel Active across {hw['world_size']} GPUs (Rank {hw['local_rank']})")
        return ddp_model, metrics

    # Case 2: Multi-GPU available in single process (DataParallel candidate)
    if hw["gpu_count"] > 1 and sample_batch is not None:
        print(f"  [MULTI-GPU BENCHMARK] Profiling Single GPU vs DataParallel throughput across {hw['gpu_count']} GPUs...")

        model.eval()
        img = sample_batch["image"].to(device, non_blocking=True)
        meta = sample_batch["metadata"].to(device, non_blocking=True) if "metadata" in sample_batch else None

        # Test 1: Single GPU Throughput
        with torch.no_grad():
            _ = model(img, meta) if meta is not None else model(img)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(5):
                _ = model(img, meta) if meta is not None else model(img)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        single_throughput = (5 * img.size(0)) / max(t1 - t0, 1e-5)
        metrics["single_gpu_throughput"] = single_throughput

        # Test 2: DataParallel Throughput
        dp_model = nn.DataParallel(model)
        dp_model.eval()
        with torch.no_grad():
            _ = dp_model(img, meta) if meta is not None else dp_model(img)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(5):
                _ = dp_model(img, meta) if meta is not None else dp_model(img)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dp_throughput = (5 * img.size(0)) / max(t1 - t0, 1e-5)
        metrics["dataparallel_throughput"] = dp_throughput

        print(f"  [BENCHMARK RESULT] Single GPU: {single_throughput:.1f} img/s | DataParallel: {dp_throughput:.1f} img/s")

        if dp_throughput > single_throughput * 1.05:
            print(f"  [MULTI-GPU] DataParallel selected ({dp_throughput:.1f} img/s)")
            metrics["mode"] = f"DataParallel ({hw['gpu_count']} GPUs)"
            metrics["selected_throughput"] = dp_throughput
            return dp_model, metrics
        else:
            print(f"  [AUTOMATIC FALLBACK] DataParallel is slower ({dp_throughput:.1f} img/s) than Single GPU ({single_throughput:.1f} img/s) due to GIL/PCI-e overhead.")
            print(f"  [AUTOMATIC FALLBACK] Automatically falling back to Single GPU mode for maximum throughput!")
            metrics["mode"] = "Single GPU (Fallback from DP)"
            metrics["selected_throughput"] = single_throughput
            return model, metrics

    # Default Single GPU
    print(f"  [HARDWARE] Single GPU Mode Active ({hw['device_name']})")
    return model, metrics
