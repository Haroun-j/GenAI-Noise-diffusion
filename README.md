# Diffusion noise study and Battery Sentinel

Do diffusion models need Gaussian noise? This is my project for the Generative AI module at CY Tech (ING3,
2025-2026, team of three: Haroun Joudi, Mouad Ouad, Amine Filali; Mouad and Amine wrote the setup and U-Net
notebooks, I wrote the training campaign, the evaluation, the forward-process diagnostic and all of Battery
Sentinel). Part 1 is a controlled experiment on MNIST: same data, same U-Net, same budget and schedule, only
the noise of the forward process changes (Gaussian, Uniform, Laplace). Part 2 reuses the three noise families
as three "uncertainty regimes" in Battery Sentinel, a small battery-monitoring prototype where the shape of a
residual says whether telemetry is normal, quantised, or hit by an impulsive event.

## 1. Three noise families, one DDPM

MNIST 28x28, a compact U-Net with sinusoidal time embedding (6.54 M parameters, channels 64/128/256) that
predicts epsilon with an MSE loss, T = 1000 linear beta schedule (1e-4 to 0.02), 30 epochs, batch 256, AdamW
lr 2e-4, gradient clipping 1.0, mixed precision, seed 42 for initialisation and data order. Gaussian is the
exact DDPM baseline; for Uniform and Laplace I keep the closed form `x_t = sqrt(a_t) x_0 + sqrt(1 - a_t) eps`
and only change the law of `eps`, standardised to unit variance (`U(-sqrt(3), sqrt(3))`,
`Laplace(0, 1/sqrt(2))`). Numbers come from `diffusion_noise_project/logs/` (evaluation summary, loss CSVs).

| Noise    | Final epoch loss | Mean denoising MSE (test, 11 timesteps) | MSE at t=500 | Train time (T4) | Peak GPU |
|----------|-----------------:|----------------------------------------:|-------------:|----------------:|---------:|
| Gaussian | 0.0220           | 0.0226                                  | 0.0134       | 23.1 min        | 2.28 GB  |
| Uniform  | 0.0186           | 0.0195                                  | 0.0074       | 22.5 min        | 2.16 GB  |
| Laplace  | 0.0205           | 0.0209                                  | 0.0109       | 22.5 min        | 2.16 GB  |

Uniform has the lowest loss and the lowest denoising error at every timestep, and the worst images: Gaussian
gives clean digits, Laplace plausible but thick digits, Uniform noisy blobs. The three runs share the standard
Gaussian reverse sampler, so only the Gaussian model is sampled with an update that matches its training law.
A bounded noise is also an easier regression target, so a lower loss does not mean a better generative model.

![Samples from the three runs](diffusion_noise_project/figures/samples_comparison.png)

Notebook 2 adds a diagnostic: under the closed-form shortcut the excess kurtosis of the injected noise stays at
-1.2 (Uniform) and +2.8 (Laplace) at every timestep, while an iterative chain with the same per-step noise goes
to 0 within a few steps, so the two runs are matched-variance surrogates rather than a non-Gaussian chain.

![Loss curves and kurtosis diagnostic](diffusion_noise_project/figures/summary_figure.png)

`research_extension/` audits the saved outputs; its nearest-neighbour check against a small MNIST CNN puts
Gaussian tiles closest to real digits (mean distance 11.7, against 13.5 for Laplace and 14.7 for Uniform).

## 2. Battery Sentinel

A simulator (notebook 7) produces 1440 charging sessions of 128 steps (SOC, voltage, current, temperature),
480 per regime: Gaussian sensor noise (nominal), quantised telemetry (Uniform) and Laplace noise plus current
and temperature shocks (Laplace); split 70/15/15 (1007/217/216). A predictive twin (notebook 8), a 2-layer GRU
with 64 hidden units, reads 24 observed steps and predicts the next SOC, voltage and temperature. Trained on
nominal windows only (8840 windows, 12 epochs), its test RMSE grows with the regime: 0.108 (Gaussian), 0.119
(Uniform), 0.244 (Laplace), which is what makes the residuals informative.

The router (notebook 9) maps residuals to a regime. A likelihood-only router reaches macro-F1 0.351, barely
above chance. The supervised router builds features from the residuals and the window (absolute differences,
distance to a quantisation grid, flat-segment ratio); a random forest trained on the 11 284 validation windows
reaches accuracy 0.9990 and macro-F1 0.9990 on the 11 232 test windows (confusion matrix, Gaussian / Uniform /
Laplace: 3943 / 0 / 9, 0 / 3640 / 0, 2 / 0 / 3638). Each regime maps to an action: monitor, request
higher-resolution telemetry, raise an inspection alert. A score this high mostly says the simulated regimes are
separable by construction.

![Router results](battery_sentinel/figures/battery_router_confusion.png)

## How to run

```
pip install -r requirements.txt
jupyter lab notebooks/
pip install -e .[eval]                                    # audit CLI, works from the saved files
python -m research_extension.cli audit-training-dynamics
python -m research_extension.cli audit-saved-samples      # trains a small MNIST CNN once
python -m research_extension.cli audit-nearest-neighbors
python -m research_extension.cli app                      # Gradio explorer, needs .[app]
```

Notebooks 1 to 6 ran on Google Colab (Tesla T4) and start by mounting Google Drive; elsewhere, replace the
`drive.mount` cell and set `PROJECT_DIR`. The three trainings take about 70 minutes on a T4; the checkpoints
(6 x 78 MB per run) are not in the repository, the logs, figures and sample grids in `diffusion_noise_project/`
are. Notebooks 7 to 10 run on CPU in a few minutes, and the simulated dataset is in `battery_sentinel/data/`,
so you can start at notebook 8. Slides are in `docs/`.

## Limitations

- The reverse sampler is the Gaussian DDPM update for all three runs, so the Uniform and Laplace images come
  from a mismatched sampler; a fair comparison needs a reverse process derived from each forward law.
- The non-Gaussian runs change the marginal noise law, not the chain (see the kurtosis note).
- One seed and one run per family, no FID or IS: loss differences of 0.002 have no error bars, and sample
  quality is judged visually plus a crude classifier audit.
- Battery Sentinel is simulated end to end. The quantisation steps and the shock injections are hard-coded,
  and the router's grid-distance features use the same step sizes, so the 0.999 macro-F1 is close to a sanity
  check. The twin also predicts the simulator's noise-free signal, which does not exist in a real system; the
  next test would be a public cycling dataset (NASA or CALCE) where the regimes are not known in advance.
