#!/usr/bin/env python3
"""Smoke test pySigLib on CUDA and through this repo's signature kernel wrapper.

This intentionally does only tiny kernel Gram computations.  It is meant to
answer: "Can this Python environment import pySigLib and execute the signature
kernel on the visible CUDA device without crashing or falling back?"
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
import traceback
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def version_of(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "<not installed>"


def print_env() -> None:
    print("=== Environment ===")
    print("python", sys.version.replace("\n", " "))
    print("executable", sys.executable)
    print("platform", platform.platform())
    print("torch", torch.__version__)
    print("torch_cuda", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    print("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    print("pysiglib_version", version_of("pysiglib"))
    print("SIGNATURE_KERNEL_STRICT", os.environ.get("SIGNATURE_KERNEL_STRICT", "<unset>"))
    print("SIGNATURE_KERNEL_FORCE_CPU", os.environ.get("SIGNATURE_KERNEL_FORCE_CPU", "<unset>"))
    print("SIGNATURE_KERNEL_BACKEND", os.environ.get("SIGNATURE_KERNEL_BACKEND", "<unset>"))
    if torch.cuda.is_available():
        print("cuda_device_count", torch.cuda.device_count())
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(
                f"cuda:{idx}",
                props.name,
                f"capability={props.major}.{props.minor}",
                f"memory_gb={props.total_memory / 1024 ** 3:.2f}",
            )
    print()


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    finite = torch.isfinite(tensor).all().item()
    print(f"{name}_shape", tuple(tensor.shape))
    print(f"{name}_device", tensor.device)
    print(f"{name}_dtype", tensor.dtype)
    print(f"{name}_finite", finite)
    print(f"{name}_minmax", float(tensor.min()), float(tensor.max()))
    assert finite, f"{name} has non-finite entries"


def direct_pysiglib_smoke(device: torch.device, batch: int, length: int, channels: int) -> None:
    print(f"=== Direct pySigLib Smoke: {device} ===")
    from pysiglib.static_kernels import RBFKernel as SigRBFKernel
    from pysiglib.torch_api import sig_kernel_gram

    torch.manual_seed(0)
    x = torch.randn(batch, length, channels, device=device)
    y = torch.randn(batch + 1, length, channels, device=device)
    static_kernel = SigRBFKernel(1.0)
    kxy = sig_kernel_gram(
        x,
        y,
        dyadic_order=1,
        time_aug=True,
        static_kernel=static_kernel,
        max_batch=max(1, min(batch, 4)),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    assert_finite("direct_Kxy", kxy)
    assert kxy.shape == (batch, batch + 1), kxy.shape
    assert kxy.device.type == device.type, (kxy.device, device)
    print("direct_pysiglib_ok", True)
    print()


def repo_wrapper_smoke(device: torch.device, batch: int, length: int, channels: int) -> None:
    print(f"=== Repo SignatureKernel Smoke: {device} ===")
    from optimal_transport import create_kernel

    os.environ["SIGNATURE_KERNEL_STRICT"] = "1"
    kernel = create_kernel(
        "signature",
        time_aug=True,
        lead_lag=True,
        dyadic_order=1,
        static_kernel_type="rbf",
        static_kernel_sigma=1.0,
        add_basepoint=True,
        normalize=True,
        max_seq_len=length,
        max_batch=max(1, min(batch, 4)),
    )
    assert getattr(kernel, "_pysiglib_available", False), "repo wrapper did not detect pySigLib"
    assert getattr(kernel, "backend", None) == "pysiglib", f"unexpected backend={kernel.backend}"

    torch.manual_seed(1)
    x = torch.randn(batch, channels, length, device=device)
    y = torch.randn(batch + 1, channels, length, device=device)
    kxy = kernel(x, y)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    assert_finite("repo_Kxy", kxy)
    assert kxy.shape == (batch, batch + 1), kxy.shape
    assert kxy.device.type == device.type, (kxy.device, device)
    print("repo_signature_kernel_ok", True)
    print()


def sinkhorn_cost_smoke(device: torch.device, batch: int, length: int, channels: int) -> None:
    print(f"=== Repo Kernel-Cost Smoke: {device} ===")
    from optimal_transport import KernelOTPlanSampler, create_kernel

    kernel = create_kernel(
        "signature",
        time_aug=True,
        lead_lag=False,
        dyadic_order=1,
        static_kernel_type="rbf",
        static_kernel_sigma=1.0,
        add_basepoint=True,
        normalize=True,
        max_seq_len=length,
        max_batch=max(1, min(batch, 4)),
    )
    torch.manual_seed(2)
    x = torch.randn(batch, channels, length, device=device)
    y = torch.randn(batch, channels, length, device=device)
    sampler = KernelOTPlanSampler(method="sinkhorn", reg=0.1)
    cost = sampler.compute_kernel_cost(x, y, kernel)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    assert_finite("repo_cost", cost)
    assert cost.shape == (batch, batch), cost.shape
    assert cost.device.type == device.type, (cost.device, device)
    print("repo_kernel_cost_ok", True)
    print()


def run_for_device(device: torch.device, args: argparse.Namespace) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"selected_device cuda:{device.index} {torch.cuda.get_device_name(device)}")
    direct_pysiglib_smoke(device, args.batch, args.length, args.channels)
    repo_wrapper_smoke(device, args.batch, args.length, args.channels)
    sinkhorn_cost_smoke(device, args.batch, args.length, args.channels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--all-visible", action="store_true", help="Test every visible CUDA device.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--channels", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("SIGNATURE_KERNEL_STRICT", "1")
    print_env()

    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available"
        count = torch.cuda.device_count()
        assert count > 0, "No CUDA devices visible"
        device_indices = range(count) if args.all_visible else [0]
        for idx in device_indices:
            run_for_device(torch.device(f"cuda:{idx}"), args)
    else:
        run_for_device(torch.device("cpu"), args)

    print("PYSIGLIB_CUDA_SMOKE_OK")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
