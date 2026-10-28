from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACTS_ROOT = Path("diffusion_noise_project")

FIGURE_CAPTIONS = {
    "beta_schedule.png": "Linear beta schedule used across the controlled runs.",
    "campaign_loss_overview.png": "Campaign-level training-loss overview aggregating the three executed runs.",
    "campaign_samples.png": "Campaign-level sample grid summarizing the saved outputs of the three runs.",
    "forward_diffusion_comparison.png": "Forward-process corruption examples across distributions and timesteps.",
    "kurtosis_diagnostic.png": "Kurtosis diagnostic comparing the Gaussian closed-form shortcut with iterative accumulation under Gaussian, Uniform, and Laplace noise.",
    "loss_comparison.png": "Training-loss trajectories across the three saved runs.",
    "metric_overview.png": "Overview of final loss, denoising MSE, and sample variance from the saved evaluation record.",
    "mnist_samples.png": "Reference MNIST samples used for the study setup.",
    "mse_vs_timestep.png": "Denoising MSE measured across diffusion timesteps for each trained run.",
    "noise_distributions.png": "Standardized Gaussian, Uniform, and Laplace noise distributions used in the forward-process study.",
    "noise_variance_vs_t.png": "Variance profile across timesteps for the configured forward schedules.",
    "samples_comparison.png": "Final generated-sample grids for Gaussian, Uniform, and Laplace training runs.",
    "summary_figure.png": "Composite dashboard combining the principal comparative outputs of the study.",
    "trajectories.png": "Reverse-sampling trajectory comparison under the shared Gaussian reverse sampler.",
}

METRIC_SPECS = [
    {
        "key": "final_epoch_avg_loss",
        "label": "Final epoch loss",
        "source": "training",
        "direction": "ascending",
        "note": "Recorded optimization loss at the end of the 30-epoch training budget.",
    },
    {
        "key": "avg_denoising_mse",
        "label": "Average denoising MSE",
        "source": "evaluation",
        "direction": "ascending",
        "note": "Average denoising error measured in Notebook 5 across the evaluation sweep.",
    },
    {
        "key": "mse_t500",
        "label": "Denoising MSE at t=500",
        "source": "evaluation",
        "direction": "ascending",
        "note": "Mid-trajectory denoising error recorded at timestep 500.",
    },
    {
        "key": "mse_t999",
        "label": "Denoising MSE at t=999",
        "source": "evaluation",
        "direction": "ascending",
        "note": "Late-timestep denoising error recorded at timestep 999.",
    },
    {
        "key": "sample_variance",
        "label": "Sample variance",
        "source": "evaluation",
        "direction": "descending",
        "note": "Descriptive dispersion statistic of the saved sample grids; it is reported as a recorded attribute rather than a standalone quality score.",
    },
]


@dataclass
class RunArtifacts:
    noise_type: str
    label: str
    training: dict[str, Any]
    evaluation: dict[str, Any]
    loss_history: list[dict[str, Any]]
    files: dict[str, Any]


def resolve_artifacts_root(artifacts_root: str | Path | None = None) -> Path:
    root = Path(artifacts_root) if artifacts_root is not None else DEFAULT_ARTIFACTS_ROOT
    return root.expanduser().resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append(
            {
                "epoch": int(row["epoch"]),
                "step": int(row["step"]),
                "loss": float(row["loss"]),
                "epoch_avg_loss": float(row["epoch_avg_loss"]),
                "elapsed_s": float(row["elapsed_s"]),
            }
        )
    return parsed


def _relpath(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _discover_named_files(directory: Path, repo_root: Path, captions: dict[str, str] | None = None) -> dict[str, dict[str, str]]:
    if not directory.exists():
        return {}
    discovered: dict[str, dict[str, str]] = {}
    for file_path in sorted(directory.iterdir()):
        if file_path.is_file():
            payload = {
                "absolute": str(file_path.resolve()),
                "relative": _relpath(file_path.resolve(), repo_root),
            }
            if captions and file_path.name in captions:
                payload["caption"] = captions[file_path.name]
            discovered[file_path.name] = payload
    return discovered


def _discover_recursive_files(directory: Path, repo_root: Path) -> list[dict[str, str]]:
    if not directory.exists():
        return []
    files: list[dict[str, str]] = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            files.append(
                {
                    "name": file_path.name,
                    "absolute": str(file_path.resolve()),
                    "relative": _relpath(file_path.resolve(), repo_root),
                }
            )
    return files


def _metric_value(run_payload: dict[str, Any], spec: dict[str, str]) -> float | None:
    section = run_payload.get(spec["source"], {})
    value = section.get(spec["key"])
    return float(value) if value is not None else None


def _best_epoch(loss_history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not loss_history:
        return None
    return min(loss_history, key=lambda row: row["epoch_avg_loss"])


def _metric_rankings(runs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rankings: dict[str, dict[str, Any]] = {}
    for spec in METRIC_SPECS:
        ordered_rows = []
        for noise_type, payload in runs.items():
            value = _metric_value(payload, spec)
            if value is None:
                continue
            ordered_rows.append(
                {
                    "noise_type": noise_type,
                    "label": payload["label"],
                    "value": value,
                }
            )

        reverse = spec["direction"] == "descending"
        ordered_rows.sort(key=lambda row: row["value"], reverse=reverse)
        ranked_rows = []
        for rank, row in enumerate(ordered_rows, start=1):
            ranked_rows.append({"rank": rank, **row})

        rankings[spec["key"]] = {
            "label": spec["label"],
            "direction": spec["direction"],
            "note": spec["note"],
            "ordered": ranked_rows,
        }
    return rankings


def build_manifest(artifacts_root: str | Path | None = None) -> dict[str, Any]:
    root = resolve_artifacts_root(artifacts_root)
    repo_root = root.parent

    logs_dir = root / "logs"
    figures_dir = root / "figures"
    samples_dir = root / "samples"
    checkpoints_dir = root / "checkpoints"
    tensorboard_dir = root / "tensorboard"

    training_summary = _load_json(logs_dir / "training_campaign_summary.json")
    evaluation_summary = _load_json(logs_dir / "evaluation_summary.json")

    noise_types = training_summary.get("noise_types", [])
    figures = _discover_named_files(figures_dir, repo_root, FIGURE_CAPTIONS)
    samples = _discover_named_files(samples_dir, repo_root)

    runs: dict[str, RunArtifacts] = {}
    total_checkpoint_files = 0
    total_tensorboard_files = 0

    for noise_type in noise_types:
        training = training_summary.get("results", {}).get(noise_type, {})
        evaluation = evaluation_summary.get("metrics", {}).get(noise_type, {})
        loss_history = _load_csv_rows(logs_dir / f"{noise_type}_loss.csv")
        checkpoint_files = _discover_recursive_files(checkpoints_dir / noise_type, repo_root)
        tensorboard_files = _discover_recursive_files(tensorboard_dir / noise_type, repo_root)
        total_checkpoint_files += len(checkpoint_files)
        total_tensorboard_files += len(tensorboard_files)

        best_logged = _best_epoch(loss_history)

        run_files: dict[str, Any] = {
            "checkpoint_dir": _relpath((checkpoints_dir / noise_type).resolve(), repo_root),
            "checkpoint_files": checkpoint_files,
            "checkpoint_count": len(checkpoint_files),
            "tensorboard_dir": _relpath((tensorboard_dir / noise_type).resolve(), repo_root),
            "tensorboard_files": tensorboard_files,
            "tensorboard_file_count": len(tensorboard_files),
        }
        sample_name = f"{noise_type}_final.png"
        if sample_name in samples:
            run_files["sample_image"] = samples[sample_name]["relative"]
            run_files["sample_image_absolute"] = samples[sample_name]["absolute"]

        run_files["best_logged_epoch"] = best_logged

        runs[noise_type] = RunArtifacts(
            noise_type=noise_type,
            label=training.get("run_label", evaluation.get("run_label", noise_type.title())),
            training=training,
            evaluation=evaluation,
            loss_history=loss_history,
            files=run_files,
        )

    run_payloads = {name: asdict(run) for name, run in runs.items()}
    rankings = _metric_rankings(run_payloads)

    manifest = {
        "project_title": "Projet GenAI : diffusion et Battery Sentinel",
        "objective": "Étude contrôlée sur les bruits de diffusion, puis extension vers Battery Sentinel, un prototype de routage d'incertitude pour le suivi batterie construit à partir des régimes Gaussian, Uniform et Laplace.",
        "repo_root": str(repo_root),
        "artifacts_root": str(root),
        "artifact_root_relative": _relpath(root, repo_root),
        "noise_types": noise_types,
        "experimental_controls": training_summary.get("experimental_controls"),
        "config": training_summary.get("config", {}),
        "shared_controls_verified": evaluation_summary.get("shared_controls_verified"),
        "full_comparison_ready": evaluation_summary.get("full_comparison_ready"),
        "sampler_note": evaluation_summary.get("sampler_note"),
        "runs": run_payloads,
        "metric_rankings": rankings,
        "figures": figures,
        "samples": samples,
        "logs": _discover_named_files(logs_dir, repo_root),
        "artifact_counts": {
            "figures": len(figures),
            "logs": len(_discover_named_files(logs_dir, repo_root)),
            "samples": len(samples),
            "checkpoints": total_checkpoint_files,
            "tensorboard_files": total_tensorboard_files,
        },
    }
    return manifest


def save_manifest(output_path: str | Path, artifacts_root: str | Path | None = None) -> Path:
    manifest = build_manifest(artifacts_root)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
