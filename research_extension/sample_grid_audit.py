from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .classifier_eval import _Classifier, _require_torch, _train_or_load_classifier


def _extract_tiles(image_path: Path, grid_size: int = 8, threshold: int = 250) -> list[np.ndarray]:
    image = np.array(Image.open(image_path).convert("L"))
    ys, xs = np.where(image < threshold)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError(f"No non-background pixels found in {image_path}")
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = image[y0:y1, x0:x1]
    height, width = cropped.shape

    tiles: list[np.ndarray] = []
    for row in range(grid_size):
        for col in range(grid_size):
            y_start = int(round(row * height / grid_size))
            y_end = int(round((row + 1) * height / grid_size))
            x_start = int(round(col * width / grid_size))
            x_end = int(round((col + 1) * width / grid_size))
            tile = cropped[y_start:y_end, x_start:x_end]
            tile = np.array(Image.fromarray(tile).resize((28, 28), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            tiles.append(tile)
    return tiles


def _load_class_prototypes(artifacts_root: Path, device):
    torch, _, _, torchvision, transforms = _require_torch()
    prototype_path = artifacts_root / "research_extension" / "classifier" / "mnist_class_prototypes.pt"
    prototype_path.parent.mkdir(parents=True, exist_ok=True)

    if prototype_path.exists():
        payload = torch.load(prototype_path, map_location=device)
        return payload["prototypes"].to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(
        root=str(artifacts_root / "research_extension" / "data"),
        train=True,
        download=True,
        transform=transform,
    )
    sums = torch.zeros((10, 28, 28), dtype=torch.float32)
    counts = torch.zeros(10, dtype=torch.float32)
    for image, label in dataset:
        sums[label] += image.squeeze(0)
        counts[label] += 1
    prototypes = sums / counts.view(-1, 1, 1).clamp_min(1.0)
    torch.save({"prototypes": prototypes.cpu()}, prototype_path)
    return prototypes.to(device)


def _classifier_forward(model, batch, torch):
    features = model.features(batch)
    flattened = model.head[0](features)
    hidden = model.head[1](flattened)
    hidden = model.head[2](hidden)
    logits = model.head[3](hidden)
    return hidden, logits


def _entropy_from_histogram(histogram: list[int]) -> float:
    counts = np.asarray(histogram, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum())


def _pairwise_l2_mean(embeddings, torch) -> float:
    if embeddings.shape[0] < 2:
        return 0.0
    distances = torch.cdist(embeddings, embeddings, p=2)
    mask = torch.triu(torch.ones_like(distances), diagonal=1) > 0
    values = distances[mask]
    return float(values.mean().item()) if values.numel() else 0.0


def _sharpness_score(tiles: list[np.ndarray]) -> float:
    scores = []
    for tile in tiles:
        gx = np.abs(np.diff(tile, axis=1)).mean()
        gy = np.abs(np.diff(tile, axis=0)).mean()
        scores.append(float(gx + gy))
    return float(np.mean(scores)) if scores else 0.0


def _foreground_ratio(tiles: list[np.ndarray], threshold: float = 0.92) -> float:
    ratios = [float((tile < threshold).mean()) for tile in tiles]
    return float(np.mean(ratios)) if ratios else 0.0


def _top_classes(histogram: list[int]) -> list[dict[str, int]]:
    order = sorted(range(len(histogram)), key=lambda idx: histogram[idx], reverse=True)
    return [{"digit": digit, "count": histogram[digit]} for digit in order[:3]]


def _run_saved_grid_metrics(artifacts_root: Path, classifier, prototypes, device, torch) -> dict[str, Any]:
    results: dict[str, Any] = {}
    sample_dir = artifacts_root / "samples"
    for image_path in sorted(sample_dir.glob("*_final.png")):
        noise_type = image_path.stem.replace("_final", "")
        tiles = _extract_tiles(image_path)
        batch = torch.from_numpy(np.stack(tiles)).unsqueeze(1).to(device)
        normalized = (batch - 0.5) / 0.5

        with torch.no_grad():
            features, logits = _classifier_forward(classifier, normalized, torch)
            probabilities = logits.softmax(dim=1)
            confidences, predictions = probabilities.max(dim=1)

        histogram = torch.bincount(predictions, minlength=10).cpu().tolist()
        predicted_proto = prototypes[predictions].unsqueeze(1)
        prototype_mse = float(((batch - predicted_proto) ** 2).mean().item())
        diversity_l2 = _pairwise_l2_mean(features, torch)

        results[noise_type] = {
            "sample_grid": str(image_path.resolve()),
            "num_tiles": int(batch.shape[0]),
            "avg_confidence": float(confidences.mean().item()),
            "std_confidence": float(confidences.std().item()),
            "predicted_class_histogram": histogram,
            "class_coverage": int(sum(1 for count in histogram if count > 0)),
            "prediction_entropy": _entropy_from_histogram(histogram),
            "top_classes": _top_classes(histogram),
            "prototype_mse": prototype_mse,
            "feature_diversity_l2": diversity_l2,
            "foreground_ratio": _foreground_ratio(tiles),
            "sharpness_score": _sharpness_score(tiles),
        }
    return results


def _ranking_block(runs: dict[str, Any], metric: str, ascending: bool, label: str) -> list[dict[str, Any]]:
    ordered = sorted(runs.items(), key=lambda item: item[1][metric], reverse=not ascending)
    return [
        {"rank": index + 1, "noise_type": noise_type, "value": float(payload[metric]), "label": label}
        for index, (noise_type, payload) in enumerate(ordered)
    ]


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Saved Sample Grid Audit",
        "",
        "This report summarizes classifier-based and image-based diagnostics computed directly from the final saved sample grids.",
        "",
        "## Summary table",
        "",
        "| Distribution | Avg confidence | Class coverage | Entropy | Prototype MSE | Feature diversity | Foreground ratio | Sharpness |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for noise_type, run in payload["runs"].items():
        lines.append(
            "| {name} | {conf:.4f} | {coverage} | {entropy:.4f} | {pmse:.5f} | {div:.4f} | {fg:.4f} | {sharp:.4f} |".format(
                name=noise_type.title(),
                conf=run["avg_confidence"],
                coverage=run["class_coverage"],
                entropy=run["prediction_entropy"],
                pmse=run["prototype_mse"],
                div=run["feature_diversity_l2"],
                fg=run["foreground_ratio"],
                sharp=run["sharpness_score"],
            )
        )
    lines.extend(["", "## Recorded class histograms", ""])
    for noise_type, run in payload["runs"].items():
        lines.append(f"### {noise_type.title()}")
        lines.append("")
        lines.append(f"- Top predicted classes: `{run['top_classes']}`")
        lines.append(f"- Histogram: `{run['predicted_class_histogram']}`")
        lines.append("")
    lines.extend(["## Metric orderings", ""])
    for metric, ranking in payload["rankings"].items():
        lines.append(f"### {ranking[0]['label']}")
        lines.append("")
        lines.append("| Rank | Distribution | Value |")
        lines.append("| ---: | --- | ---: |")
        for row in ranking:
            lines.append(f"| {row['rank']} | {row['noise_type'].title()} | {row['value']:.5f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def run_saved_sample_audit(
    artifacts_root: str | Path,
    output_path: str | Path,
    markdown_output: str | Path | None = None,
) -> Path:
    torch, nn, _, _, _ = _require_torch()
    artifacts = Path(artifacts_root).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, classifier_checkpoint = _train_or_load_classifier(artifacts, device, batch_size=128, epochs=1)
    classifier.eval()
    prototypes = _load_class_prototypes(artifacts, device)

    runs = _run_saved_grid_metrics(artifacts, classifier, prototypes, device, torch)
    payload: dict[str, Any] = {
        "device": str(device),
        "classifier_checkpoint": str(classifier_checkpoint),
        "artifacts_root": str(artifacts),
        "runs": runs,
        "rankings": {
            "avg_confidence": _ranking_block(runs, "avg_confidence", ascending=False, label="Average classifier confidence"),
            "class_coverage": _ranking_block(runs, "class_coverage", ascending=False, label="Predicted class coverage"),
            "prediction_entropy": _ranking_block(runs, "prediction_entropy", ascending=False, label="Prediction entropy"),
            "prototype_mse": _ranking_block(runs, "prototype_mse", ascending=True, label="Prototype MSE"),
            "feature_diversity_l2": _ranking_block(runs, "feature_diversity_l2", ascending=False, label="Feature diversity"),
            "sharpness_score": _ranking_block(runs, "sharpness_score", ascending=False, label="Sharpness score"),
        },
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if markdown_output is not None:
        markdown_path = Path(markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_markdown_report(payload), encoding="utf-8")

    return output
