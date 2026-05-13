"""
Run inference with a TorchScript edge export.

Usage:
    python edge_infer.py --artifact ./output_edge/edge/model_edge.ts --image ./leaf.jpg
"""

import argparse
import json
import time
from pathlib import Path

import torch

from train import normalize_image, preprocess_to_rgb


def load_metadata(artifact_path: Path, metadata_path: str | None):
    if metadata_path:
        metadata_file = Path(metadata_path)
    else:
        metadata_file = artifact_path.with_suffix(artifact_path.suffix + ".json")
    with open(metadata_file, "r", encoding="utf-8") as handle:
        return json.load(handle), metadata_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, help="Path to a TorchScript export created by train.py export")
    parser.add_argument("--image", required=True, help="Image to classify")
    parser.add_argument("--metadata", default=None, help="Optional metadata JSON path")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--warmup_runs", type=int, default=3)
    parser.add_argument("--benchmark_runs", type=int, default=20)
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    metadata, metadata_file = load_metadata(artifact_path, args.metadata)
    model = torch.jit.load(str(artifact_path), map_location="cpu")
    model.eval()

    image = preprocess_to_rgb(args.image, metadata["image_size"])
    pixel_values = normalize_image(image, metadata["image_mean"], metadata["image_std"]).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max(0, args.warmup_runs)):
            _ = model(pixel_values)

        start = time.perf_counter()
        for _ in range(max(1, args.benchmark_runs)):
            logits = model(pixel_values)
        end = time.perf_counter()

    probs = torch.softmax(logits, dim=-1).squeeze(0)
    top_k = max(1, min(args.top_k, probs.numel()))
    top_probs, top_indices = torch.topk(probs, k=top_k)
    avg_latency_ms = ((end - start) / max(1, args.benchmark_runs)) * 1000.0

    id2label = {int(k): v for k, v in metadata["id2label"].items()}
    best_index = int(top_indices[0].item())
    print(f"Artifact:      {artifact_path}")
    print(f"Metadata:      {metadata_file}")
    print(f"Prediction:    {id2label[best_index]}")
    print(f"Avg latency:   {avg_latency_ms:.2f} ms/image on CPU")
    print("Top classes:")
    for score, index in zip(top_probs.tolist(), top_indices.tolist()):
        print(f"  {id2label[int(index)]:35s}: {score:.4f}")


if __name__ == "__main__":
    main()
