"""Benchmark script for Fourier Neural Operator (FNO) throughput and memory.

Measures:
    - Forward pass latency across spatial resolutions
    - Peak GPU memory (if CUDA) or RAM
    - Throughput (samples per second)
    - Zero-shot resolution scalability
"""

from __future__ import annotations

import argparse
import logging
import time

import torch

from operatorlab.models.fno import FNO2d

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def benchmark_resolution(
    model: FNO2d,
    resolution: int,
    batch_size: int = 16,
    n_warmup: int = 5,
    n_runs: int = 20,
    device: str = "cpu",
) -> dict[str, float]:
    """Benchmark model at a specific resolution."""
    x = torch.randn(batch_size, resolution, resolution, 1, device=device)
    grid = model.make_grid((resolution, resolution), torch.device(device))
    grid = grid.expand(batch_size, -1, -1, -1)

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x, grid)
            if device.startswith("cuda"):
                torch.cuda.synchronize()

    # Timed runs
    latencies: list[float] = []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x, grid)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    mean_latency = sum(latencies) / len(latencies)
    throughput = (batch_size * 1000.0) / mean_latency  # samples/sec

    peak_mem_mb = 0.0
    if device.startswith("cuda"):
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "resolution": resolution,
        "mean_latency_ms": mean_latency,
        "throughput_sps": throughput,
        "peak_mem_mb": peak_mem_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FNO Throughput Benchmark")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cpu', 'cuda', 'auto'")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--modes", type=int, default=12, help="Number of Fourier modes")
    parser.add_argument("--width", type=int, default=64, help="Channel width")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--n-warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--n-runs", type=int, default=20, help="Number of timed runs")
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="Resolutions to benchmark",
    )
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Running FNO benchmark on device: %s", device)
    model = FNO2d(
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
        n_layers=args.n_layers,
    ).to(device)

    params = model.count_parameters()
    logger.info("FNO2d parameters: %s", f"{params:,}")

    print(f"\n{'Resolution':<12} | {'Latency (ms)':<14} | {'Throughput (samp/s)':<22} | {'Peak Mem (MB)':<14}")
    print("-" * 68)

    for res in args.resolutions:
        try:
            res_metrics = benchmark_resolution(
                model=model,
                resolution=res,
                batch_size=args.batch_size,
                n_warmup=args.n_warmup,
                n_runs=args.n_runs,
                device=device,
            )
            print(
                f"{res}x{res:<9} | "
                f"{res_metrics['mean_latency_ms']:<14.2f} | "
                f"{res_metrics['throughput_sps']:<22.1f} | "
                f"{res_metrics['peak_mem_mb']:<14.2f}"
            )
        except Exception as exc:
            logger.warning("Failed benchmark at resolution %d: %s", res, exc)


if __name__ == "__main__":
    main()
