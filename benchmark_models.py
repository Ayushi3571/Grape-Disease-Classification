"""
Benchmark preset backbones for CPU-oriented edge deployment.

Usage:
    python benchmark_models.py --presets base edge edge_plus --runs 30 --warmup 5
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from train import LeafDiseaseTransformer, MODEL_PRESETS, resolve_model_config


class LogitsOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        logits, _ = self.model(pixel_values)
        return logits


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def benchmark_model(wrapper, example, warmup_runs: int, benchmark_runs: int) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(max(0, warmup_runs)):
            _ = wrapper(example)

        start = time.perf_counter()
        for _ in range(max(1, benchmark_runs)):
            logits = wrapper(example)
        end = time.perf_counter()

    avg_latency_ms = ((end - start) / max(1, benchmark_runs)) * 1000.0
    throughput = max(1, benchmark_runs) / max(end - start, 1e-9)
    return avg_latency_ms, throughput


def benchmark_preset(preset_name: str, num_labels: int, artifact_dir: Path, warmup_runs: int, benchmark_runs: int):
    config = resolve_model_config(preset_name, None, None)
    model = LeafDiseaseTransformer(config["model_name"], num_labels=num_labels).cpu().eval()
    wrapper = LogitsOnlyWrapper(model).eval()
    example = torch.randn(1, 3, config["image_size"], config["image_size"], dtype=torch.float32)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = artifact_dir / f"{preset_name}_state_dict.pt"
    torchscript_path = artifact_dir / f"{preset_name}.ts"
    quantized_path = artifact_dir / f"{preset_name}_int8.ts"

    torch.save(model.state_dict(), state_dict_path)
    traced = torch.jit.trace(wrapper, example)
    optimized = torch.jit.optimize_for_inference(traced)
    optimized.save(str(torchscript_path))

    quantized_wrapper = LogitsOnlyWrapper(
        torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    ).eval()
    quantized_traced = torch.jit.trace(quantized_wrapper, example)
    quantized_optimized = torch.jit.optimize_for_inference(quantized_traced)
    quantized_optimized.save(str(quantized_path))

    eager_latency_ms, eager_throughput = benchmark_model(wrapper, example, warmup_runs, benchmark_runs)
    quantized_latency_ms, quantized_throughput = benchmark_model(
        quantized_wrapper, example, warmup_runs, benchmark_runs
    )

    return {
        "preset": preset_name,
        "model_name": config["model_name"],
        "image_size": config["image_size"],
        "state_dict_mb": round(file_size_mb(state_dict_path), 2),
        "torchscript_mb": round(file_size_mb(torchscript_path), 2),
        "torchscript_int8_mb": round(file_size_mb(quantized_path), 2),
        "eager_latency_ms": round(eager_latency_ms, 2),
        "eager_images_per_sec": round(eager_throughput, 2),
        "int8_latency_ms": round(quantized_latency_ms, 2),
        "int8_images_per_sec": round(quantized_throughput, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--presets", nargs="+", default=["base", "edge", "edge_plus"])
    parser.add_argument("--num_labels", type=int, default=6)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=None, help="Override torch CPU thread count")
    parser.add_argument("--artifact_dir", default="./benchmark_artifacts")
    parser.add_argument("--output_json", default="./benchmark_artifacts/benchmark_results.json")
    args = parser.parse_args()

    unknown = [preset for preset in args.presets if preset not in MODEL_PRESETS]
    if unknown:
        raise ValueError(f"Unknown presets: {unknown}. Choose from {sorted(MODEL_PRESETS)}")

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    artifact_dir = Path(args.artifact_dir)
    results = []
    for preset_name in args.presets:
        print(f"Benchmarking {preset_name}...")
        results.append(
            benchmark_preset(
                preset_name,
                num_labels=args.num_labels,
                artifact_dir=artifact_dir,
                warmup_runs=args.warmup,
                benchmark_runs=args.runs,
            )
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\nResults:")
    for result in results:
        print(json.dumps(result, indent=2))
    print(f"\nSaved benchmark JSON to {output_path}")


if __name__ == "__main__":
    main()
