from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_loss_rows(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            'epoch': int(row['epoch']),
            'step': int(row['step']),
            'loss': float(row['loss']),
            'epoch_avg_loss': float(row['epoch_avg_loss']),
            'elapsed_s': float(row['elapsed_s']),
        }
        for row in rows
    ]


_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def _relative_improvement(initial: float, final: float) -> float:
    if initial == 0:
        return 0.0
    return float((initial - final) / initial)


def _late_stage_std(values: list[float], tail: int = 5) -> float:
    subset = values[-tail:] if len(values) >= tail else values
    return float(np.std(subset)) if subset else 0.0


def _slope(values: list[float], start: int, end: int) -> float:
    if len(values) <= end or end == start:
        return 0.0
    return float((values[end] - values[start]) / (end - start))


def _plot_training_dynamics(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], target: Path) -> None:
    import matplotlib.pyplot as plt

    target.parent.mkdir(parents=True, exist_ok=True)
    colors = {'gaussian': '#355c7d', 'uniform': '#6c8ead', 'laplace': '#c06c84'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for noise_type in sorted({row['noise_type'] for row in rows}):
        subset = [row for row in rows if row['noise_type'] == noise_type]
        axes[0, 0].plot([row['epoch'] for row in subset], [row['epoch_avg_loss'] for row in subset], marker='o', label=noise_type.title(), color=colors[noise_type])
    axes[0, 0].set_title('Epoch-average loss trajectories')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    names = [row['noise_type'].title() for row in summary_rows]
    axes[0, 1].bar(names, [row['relative_improvement_pct'] for row in summary_rows], color=[colors[row['noise_type']] for row in summary_rows])
    axes[0, 1].set_title('Relative loss improvement (%)')
    axes[0, 1].grid(axis='y', alpha=0.25)

    axes[1, 0].bar(names, [row['late_stage_std'] for row in summary_rows], color=[colors[row['noise_type']] for row in summary_rows])
    axes[1, 0].set_title('Late-stage loss volatility')
    axes[1, 0].grid(axis='y', alpha=0.25)

    axes[1, 1].bar(names, [row['best_epoch'] for row in summary_rows], color=[colors[row['noise_type']] for row in summary_rows])
    axes[1, 1].set_title('Best logged epoch')
    axes[1, 1].grid(axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(target, dpi=200, bbox_inches='tight')
    plt.close(fig)


def run_training_dynamics_audit(
    artifacts_root: str | Path,
    output_path: str | Path,
    markdown_output: str | Path | None = None,
) -> Path:
    artifacts = Path(artifacts_root).resolve()
    logs_dir = artifacts / 'logs'
    with (logs_dir / 'training_campaign_summary.json').open('r', encoding='utf-8') as handle:
        campaign = json.load(handle)

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for noise_type in campaign['noise_types']:
        loss_rows = _load_loss_rows(logs_dir / f'{noise_type}_loss.csv')
        values = [row['epoch_avg_loss'] for row in loss_rows]
        best_row = min(loss_rows, key=lambda row: row['epoch_avg_loss'])
        improvement = _relative_improvement(values[0], values[-1])
        summary = {
            'noise_type': noise_type,
            'initial_epoch_loss': values[0],
            'final_epoch_loss': values[-1],
            'absolute_improvement': float(values[0] - values[-1]),
            'relative_improvement_pct': float(improvement * 100.0),
            'best_epoch': int(best_row['epoch']),
            'best_epoch_loss': float(best_row['epoch_avg_loss']),
            'late_stage_std': _late_stage_std(values),
            'early_slope': _slope(values, 0, min(4, len(values) - 1)),
            'late_slope': _slope(values, max(0, len(values) - 5), len(values) - 1),
            'auc_epoch_loss': float(_trapezoid(values, dx=1.0)),
        }
        summary_rows.append(summary)
        for row in loss_rows:
            rows.append({'noise_type': noise_type, **row})

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure_path = output.parent / 'figures' / 'training_dynamics_audit.png'
    _plot_training_dynamics(rows, summary_rows, figure_path)

    payload = {
        'artifacts_root': str(artifacts),
        'summary_rows': summary_rows,
        'rows': rows,
        'figure': str(figure_path),
    }
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    if markdown_output is not None:
        lines = [
            '# Training Dynamics Audit',
            '',
            '| Distribution | Initial loss | Final loss | Relative improvement (%) | Best epoch | Late-stage volatility |',
            '| --- | ---: | ---: | ---: | ---: | ---: |',
        ]
        for row in summary_rows:
            lines.append(
                f"| {row['noise_type'].title()} | {row['initial_epoch_loss']:.5f} | {row['final_epoch_loss']:.5f} | {row['relative_improvement_pct']:.2f} | {row['best_epoch']} | {row['late_stage_std']:.5f} |"
            )
        lines.extend(['', f"Figure: `{figure_path}`", ''])
        Path(markdown_output).write_text('\n'.join(lines) + '\n', encoding='utf-8')

    return output
