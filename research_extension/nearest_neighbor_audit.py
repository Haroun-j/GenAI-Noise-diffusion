from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .classifier_eval import _require_torch, _train_or_load_classifier
from .sample_grid_audit import _classifier_forward, _extract_tiles, _top_classes


def _load_feature_bank(artifacts_root: Path, classifier, device):
    torch, _, _, torchvision, transforms = _require_torch()
    cache = artifacts_root / 'research_extension' / 'classifier' / 'mnist_feature_bank.pt'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        payload = torch.load(cache, map_location='cpu')
        return payload['features'], payload['labels'], payload['images']

    dataset = torchvision.datasets.MNIST(
        root=str(artifacts_root / 'research_extension' / 'data'),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
    features, labels = [], []
    for images, batch_labels in loader:
        batch = ((images.to(device) - 0.5) / 0.5)
        with torch.no_grad():
            hidden, _ = _classifier_forward(classifier, batch, torch)
        features.append(hidden.cpu())
        labels.append(batch_labels.cpu())
    payload = {
        'features': torch.cat(features, dim=0),
        'labels': torch.cat(labels, dim=0),
        'images': dataset.data.clone(),
    }
    torch.save(payload, cache)
    return payload['features'], payload['labels'], payload['images']


def _entropy(histogram: list[int]) -> float:
    counts = np.asarray(histogram, dtype=np.float64)
    probs = counts / max(counts.sum(), 1.0)
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum())


def _make_montage(gen_tiles, ref_tiles, ref_labels, pred_labels, distances, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    cols, rows = 4, 3
    pad, cell_w, cell_h = 12, 180, 150
    canvas = Image.new('RGB', (cols * cell_w + (cols + 1) * pad, rows * cell_h + (rows + 1) * pad), 'white')
    draw = ImageDraw.Draw(canvas)
    for idx, (gen_tile, ref_tile, ref_label, pred_label, distance) in enumerate(zip(gen_tiles, ref_tiles, ref_labels, pred_labels, distances)):
        if idx >= cols * rows:
            break
        row, col = divmod(idx, cols)
        x0 = pad + col * (cell_w + pad)
        y0 = pad + row * (cell_h + pad)
        gen_img = Image.fromarray((gen_tile * 255).astype('uint8')).convert('L').resize((70, 70), Image.Resampling.NEAREST).convert('RGB')
        ref_img = Image.fromarray(ref_tile.astype('uint8')).convert('L').resize((70, 70), Image.Resampling.NEAREST).convert('RGB')
        canvas.paste(gen_img, (x0, y0 + 20))
        canvas.paste(ref_img, (x0 + 88, y0 + 20))
        draw.text((x0, y0), f'gen->{pred_label}', fill='black')
        draw.text((x0 + 88, y0), f'nn:{ref_label}', fill='black')
        draw.text((x0, y0 + 100), f'd={distance:.2f}', fill='black')
    canvas.save(target)


def _overview_plot(payload: dict[str, Any], target: Path):
    import matplotlib.pyplot as plt

    target.parent.mkdir(parents=True, exist_ok=True)
    names = [name.title() for name in payload['runs']]
    conf = [payload['runs'][name]['avg_confidence'] for name in payload['runs']]
    dist = [payload['runs'][name]['avg_nn_distance'] for name in payload['runs']]
    cov = [payload['runs'][name]['matched_label_coverage'] for name in payload['runs']]
    consistency = [payload['runs'][name]['prediction_match_rate'] for name in payload['runs']]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    plots = [
        ('Average classifier confidence', conf),
        ('Average nearest-neighbor distance', dist),
        ('Matched-label coverage', cov),
        ('Prediction / neighbor label agreement', consistency),
    ]
    for ax, (title, values) in zip(axes.ravel(), plots):
        ax.bar(names, values, color=['#355c7d', '#6c8ead', '#c06c84'])
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=200, bbox_inches='tight')
    plt.close(fig)


def run_nearest_neighbor_audit(artifacts_root: str | Path, output_path: str | Path, markdown_output: str | Path | None = None) -> Path:
    torch, nn, _, _, _ = _require_torch()
    artifacts = Path(artifacts_root).resolve()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier, classifier_checkpoint = _train_or_load_classifier(artifacts, device, batch_size=128, epochs=1)
    classifier.eval()
    bank_features, bank_labels, bank_images = _load_feature_bank(artifacts, classifier, device)
    bank_features = bank_features.float()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure_dir = output.parent / 'figures'
    runs = {}

    for image_path in sorted((artifacts / 'samples').glob('*_final.png')):
        noise_type = image_path.stem.replace('_final', '')
        tiles = _extract_tiles(image_path)
        batch = torch.from_numpy(np.stack(tiles)).unsqueeze(1).to(device)
        normalized = (batch - 0.5) / 0.5
        with torch.no_grad():
            features, logits = _classifier_forward(classifier, normalized, torch)
            probs = logits.softmax(dim=1)
            confs, preds = probs.max(dim=1)
        features_cpu = features.cpu().float()
        dists = torch.cdist(features_cpu, bank_features)
        nn_dist, nn_idx = dists.min(dim=1)
        nn_labels = bank_labels[nn_idx].cpu().tolist()
        pred_labels = preds.cpu().tolist()
        hist = np.bincount(nn_labels, minlength=10).tolist()
        match_rate = float(np.mean([int(a == b) for a, b in zip(pred_labels, nn_labels)]))
        montage_path = figure_dir / f'nearest_neighbor_{noise_type}.png'
        _make_montage(
            tiles[:12],
            [bank_images[idx].numpy() for idx in nn_idx[:12]],
            nn_labels[:12],
            pred_labels[:12],
            nn_dist[:12].cpu().tolist(),
            montage_path,
        )
        runs[noise_type] = {
            'sample_grid': str(image_path),
            'avg_confidence': float(confs.mean().item()),
            'avg_nn_distance': float(nn_dist.mean().item()),
            'matched_label_histogram': hist,
            'matched_label_coverage': int(sum(1 for count in hist if count > 0)),
            'matched_label_entropy': _entropy(hist),
            'prediction_match_rate': match_rate,
            'top_matched_labels': _top_classes(hist),
            'montage_path': str(montage_path),
        }

    payload = {
        'device': str(device),
        'classifier_checkpoint': str(classifier_checkpoint),
        'feature_bank_size': int(bank_features.shape[0]),
        'runs': runs,
    }
    overview_path = figure_dir / 'nearest_neighbor_overview.png'
    _overview_plot(payload, overview_path)
    payload['overview_figure'] = str(overview_path)
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    if markdown_output is not None:
        lines = [
            '# Nearest-Neighbor Audit',
            '',
            '| Distribution | Avg confidence | Avg NN distance | Label coverage | Label entropy | Match rate |',
            '| --- | ---: | ---: | ---: | ---: | ---: |',
        ]
        for noise_type, run in runs.items():
            lines.append(
                f"| {noise_type.title()} | {run['avg_confidence']:.4f} | {run['avg_nn_distance']:.4f} | {run['matched_label_coverage']} | {run['matched_label_entropy']:.4f} | {run['prediction_match_rate']:.4f} |"
            )
        lines.extend(['', f"Overview figure: `{overview_path}`", ''])
        Path(markdown_output).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output
