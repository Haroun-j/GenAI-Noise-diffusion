from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision
        import torchvision.transforms as transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Torch and torchvision are required for classifier-based evaluation. Install them with `pip install -e .[eval]`."
        ) from exc
    return torch, nn, F, torchvision, transforms


def _load_unet_class(artifacts_root: Path):
    module_path = artifacts_root / "unet.py"
    spec = importlib.util.spec_from_file_location("saved_unet_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load UNet definition from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UNet


def _make_beta_schedule(torch, schedule: str, T: int, beta_start: float, beta_end: float):
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, T)
    elif schedule == "cosine":
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + 0.008) / 1.008 * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(0, 0.999)
    else:
        raise ValueError(f"Unsupported schedule: {schedule}")
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
    }


class _ForwardDiffusion:
    def __init__(self, torch, schedule, device):
        self.torch = torch
        self.device = device
        self.T = len(schedule["betas"])
        for key, value in schedule.items():
            setattr(self, key, value.to(device))


class _GaussianDiffusion(_ForwardDiffusion):
    pass


class _UniformDiffusion(_ForwardDiffusion):
    pass


class _LaplaceDiffusion(_ForwardDiffusion):
    pass


def _build_diffusion(torch, noise_type: str, schedule: dict[str, Any], device):
    return {
        "gaussian": _GaussianDiffusion,
        "uniform": _UniformDiffusion,
        "laplace": _LaplaceDiffusion,
    }[noise_type](torch, schedule, device)


def _sample(model, diffusion, noise_type: str, n_samples: int, config: dict[str, Any], device):
    torch = diffusion.torch
    model.eval()
    x = torch.randn((n_samples, config["channels"], config["image_size"], config["image_size"]), device=device)
    for t_val in reversed(range(diffusion.T)):
        t_batch = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
        eps_pred = model(x, t_batch)
        alpha_t = diffusion.alphas[t_val]
        alpha_bar_t = diffusion.alphas_cumprod[t_val]
        beta_t = diffusion.betas[t_val]
        coeff = (1 - alpha_t) / (1 - alpha_bar_t).sqrt()
        mean = (1 / alpha_t.sqrt()) * (x - coeff * eps_pred)
        x = mean + (beta_t.sqrt() * torch.randn_like(x) if t_val > 0 else 0)
    return x.clamp(-1, 1)


class _Classifier:
    def __init__(self, nn):
        self.nn = nn

    def build(self):
        nn = self.nn

        class SmallMNISTClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 7 * 7, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 10),
                )

            def forward(self, x):
                return self.head(self.features(x))

        return SmallMNISTClassifier()


def _train_or_load_classifier(artifacts_root: Path, device, batch_size: int, epochs: int):
    torch, nn, F, torchvision, transforms = _require_torch()
    classifier_dir = artifacts_root / "research_extension" / "classifier"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = classifier_dir / "mnist_classifier.pt"

    model = _Classifier(nn).build().to(device)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        return model, checkpoint_path

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root=str(artifacts_root / "research_extension" / "data"), train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

    torch.save({"model": model.state_dict()}, checkpoint_path)
    return model, checkpoint_path


def run_classifier_evaluation(
    artifacts_root: str | Path,
    output_path: str | Path,
    num_samples_per_run: int = 512,
    classifier_epochs: int = 3,
    batch_size: int = 128,
) -> Path:
    torch, _, _, _, _ = _require_torch()
    artifacts = Path(artifacts_root).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, classifier_checkpoint = _train_or_load_classifier(artifacts, device, batch_size, classifier_epochs)
    classifier.eval()

    with (artifacts / "logs" / "training_campaign_summary.json").open("r", encoding="utf-8") as handle:
        training_summary = json.load(handle)

    UNet = _load_unet_class(artifacts)
    schedule = _make_beta_schedule(
        torch,
        training_summary["config"]["schedule"],
        training_summary["config"]["T"],
        training_summary["config"]["beta_start"],
        training_summary["config"]["beta_end"],
    )

    payload: dict[str, Any] = {
        "device": str(device),
        "classifier_checkpoint": str(classifier_checkpoint),
        "num_samples_per_run": num_samples_per_run,
        "runs": {},
    }

    for noise_type in training_summary["noise_types"]:
        checkpoint_dir = artifacts / "checkpoints" / noise_type
        checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found for {noise_type} in {checkpoint_dir}")
        checkpoint_path = checkpoints[-1]
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = UNet(
            in_channels=training_summary["config"]["channels"],
            model_channels=training_summary["config"]["model_channels"],
            channel_mults=tuple(training_summary["config"]["channel_mults"]),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        diffusion = _build_diffusion(torch, noise_type, schedule, device)
        samples = _sample(model, diffusion, noise_type, num_samples_per_run, training_summary["config"], device)
        display_samples = (samples * 0.5 + 0.5).clamp(0, 1)
        logits = classifier(display_samples)
        probabilities = logits.softmax(dim=1)
        confidences, predictions = probabilities.max(dim=1)
        histogram = torch.bincount(predictions, minlength=10).cpu().tolist()
        class_probs = torch.tensor(histogram, dtype=torch.float32)
        class_probs = class_probs / class_probs.sum().clamp_min(1.0)
        entropy = float(-(class_probs * class_probs.clamp_min(1e-12).log()).sum().item())

        payload["runs"][noise_type] = {
            "avg_confidence": float(confidences.mean().item()),
            "std_confidence": float(confidences.std().item()),
            "predicted_class_histogram": histogram,
            "class_coverage": int(sum(1 for count in histogram if count > 0)),
            "prediction_entropy": entropy,
        }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
