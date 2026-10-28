from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import FIGURE_CAPTIONS, build_manifest


def _require_gradio():
    try:
        import gradio as gr  # type: ignore
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "L explorateur local demande l environnement du projet. Installez le avec `pip install -e .[app]` depuis la racine du depot."
        ) from exc
    return gr, pd


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _results_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for noise_type in manifest['noise_types']:
        run = manifest['runs'][noise_type]
        training = run['training']
        evaluation = run['evaluation']
        rows.append(
            {
                'distribution': noise_type.title(),
                'role': run['label'],
                'final_epoch_loss': round(training['final_epoch_avg_loss'], 5),
                'avg_denoising_mse': round(evaluation['avg_denoising_mse'], 5),
                'mse_t500': round(evaluation['mse_t500'], 5),
                'mse_t999': round(evaluation['mse_t999'], 5),
                'sample_variance': round(evaluation['sample_variance'], 5),
                'runtime_min': round(training['duration_minutes'], 2),
                'peak_gpu_gb': round(training['peak_memory_gb'], 2),
            }
        )
    return rows


def _controls_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    config = manifest['config']
    order = [
        'dataset', 'image_size', 'channels', 'T', 'schedule', 'num_epochs', 'batch_size',
        'lr', 'optimizer', 'weight_decay', 'model_channels', 'channel_mults', 'num_workers',
        'seed', 'sample_count'
    ]
    return [{'parameter': key, 'value': config.get(key)} for key in order if key in config]


def _loss_history_rows(manifest: dict[str, Any], noise_type: str) -> list[dict[str, Any]]:
    return [
        {
            'epoch': row['epoch'],
            'step': row['step'],
            'batch_loss': round(row['loss'], 6),
            'epoch_avg_loss': round(row['epoch_avg_loss'], 6),
            'elapsed_s': round(row['elapsed_s'], 1),
        }
        for row in manifest['runs'][noise_type]['loss_history']
    ]


def _checkpoint_rows(manifest: dict[str, Any], noise_type: str) -> list[dict[str, Any]]:
    return [
        {'name': payload['name'], 'path': payload['relative']}
        for payload in manifest['runs'][noise_type]['files'].get('checkpoint_files', [])
    ]


def _figure_payload(manifest: dict[str, Any], figure_name: str) -> tuple[str, str | None]:
    payload = manifest['figures'][figure_name]
    caption = payload.get('caption', FIGURE_CAPTIONS.get(figure_name, 'Figure enregistree a partir des notebooks executes.'))
    note = '\n'.join([f"## {figure_name}", '', caption, '', f"Chemin de sortie : `{payload['relative']}`"])
    return note, payload['absolute']


def _training_dynamics_rows(td_audit: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not td_audit:
        return []
    rows = []
    for row in td_audit['summary_rows']:
        rows.append(
            {
                'distribution': row['noise_type'].title(),
                'initial_loss': round(row['initial_epoch_loss'], 5),
                'final_loss': round(row['final_epoch_loss'], 5),
                'relative_improvement_pct': round(row['relative_improvement_pct'], 2),
                'best_epoch': row['best_epoch'],
                'late_stage_std': round(row['late_stage_std'], 5),
                'auc_epoch_loss': round(row['auc_epoch_loss'], 4),
            }
        )
    return rows


def build_app(artifacts_root=None):
    gr, pd = _require_gradio()
    manifest = build_manifest(artifacts_root)
    repo_root = Path(manifest['repo_root'])
    output_dir = repo_root / 'research_extension_output'
    saved_sample_audit = _load_optional_json(output_dir / 'saved_sample_audit.json')
    nearest_neighbor_audit = _load_optional_json(output_dir / 'nearest_neighbor_audit.json')
    training_dynamics_audit = _load_optional_json(output_dir / 'training_dynamics_audit.json')

    battery_root = repo_root / 'battery_sentinel'
    battery_logs_dir = battery_root / 'logs'
    battery_figures_dir = battery_root / 'figures'
    battery_simulation = _load_optional_json(battery_logs_dir / 'simulation_summary.json')
    battery_twin = _load_optional_json(battery_logs_dir / 'twin_training_summary.json')
    battery_router = _load_optional_json(battery_logs_dir / 'router_metrics.json')

    noise_choices = manifest['noise_types']
    default_run = noise_choices[0]
    results_df = pd.DataFrame(_results_rows(manifest))
    controls_df = pd.DataFrame(_controls_rows(manifest))
    td_df = pd.DataFrame(_training_dynamics_rows(training_dynamics_audit))
    battery_metrics_df = pd.DataFrame(battery_twin.get('test_metrics', [])) if battery_twin else pd.DataFrame([])
    battery_actions_df = pd.DataFrame([
        {'action': key, 'count': value}
        for key, value in (battery_router.get('action_counts', {}) if battery_router else {}).items()
    ])

    figure_choices = [
        name for name in [
            'summary_figure.png',
            'loss_comparison.png',
            'metric_overview.png',
            'samples_comparison.png',
            'trajectories.png',
            'kurtosis_diagnostic.png',
        ]
        if name in manifest['figures']
    ]
    default_figure = figure_choices[0] if figure_choices else None

    battery_figure_map: dict[str, str] = {}
    for name in [
        'battery_sentinel_dashboard.png',
        'battery_regime_examples.png',
        'battery_twin_training_curves.png',
        'battery_router_confusion.png',
        'battery_router_case_studies.png',
    ]:
        candidate = battery_figures_dir / name
        if candidate.exists():
            battery_figure_map[name] = str(candidate.resolve())
    battery_figure_choices = list(battery_figure_map.keys())
    default_battery_figure = battery_figure_choices[0] if battery_figure_choices else None

    overview_text = '\n'.join([
        f"# {manifest['project_title']}",
        '',
        manifest['objective'],
        '',
        f"- Controles communs verifies : `{manifest['shared_controls_verified']}`",
        f"- Comparaison complete disponible : `{manifest['full_comparison_ready']}`",
        f"- Battery Sentinel disponible : `{battery_router is not None}`",
    ])

    def _saved_run(noise_type: str):
        if not saved_sample_audit:
            return None
        return saved_sample_audit.get('runs', {}).get(noise_type)

    def _nn_run(noise_type: str):
        if not nearest_neighbor_audit:
            return None
        return nearest_neighbor_audit.get('runs', {}).get(noise_type)

    def show_run(noise_type: str):
        run = manifest['runs'][noise_type]
        training = run['training']
        evaluation = run['evaluation']
        sample_path = run['files'].get('sample_image_absolute')
        saved_run = _saved_run(noise_type)
        nn_run = _nn_run(noise_type)
        lines = [
            f"## {run['label']}",
            '',
            training.get('training_objective', 'Objectif d entrainement non renseigne.'),
            '',
            f"- Perte finale : `{training['final_epoch_avg_loss']:.5f}`",
            f"- MSE moyenne : `{evaluation['avg_denoising_mse']:.5f}`",
            f"- Temps total : `{training['duration_minutes']:.2f}` minutes",
            f"- Memoire GPU max : `{training['peak_memory_gb']:.2f}` GB",
        ]
        if saved_run:
            lines.extend([
                f"- Confiance moyenne sur les images sauvegardees : `{saved_run['avg_confidence']:.4f}`",
                f"- Couverture des classes : `{saved_run['class_coverage']}`",
                f"- Nettete : `{saved_run['sharpness_score']:.4f}`",
            ])
        if nn_run:
            lines.extend([
                f"- Distance moyenne au voisin proche : `{nn_run['avg_nn_distance']:.4f}`",
                f"- Accord prediction / voisin proche : `{nn_run['prediction_match_rate']:.4f}`",
            ])
        return (
            '\n'.join(lines),
            sample_path,
            pd.DataFrame(_loss_history_rows(manifest, noise_type)),
            pd.DataFrame(_checkpoint_rows(manifest, noise_type)),
        )

    def show_figure(figure_name: str):
        if not figure_name:
            return '', None
        return _figure_payload(manifest, figure_name)

    def show_battery_figure(figure_name: str):
        if not figure_name:
            return '', None
        return f"## {figure_name}\n\nFigure issue de la partie Battery Sentinel.", battery_figure_map.get(figure_name)

    default_run_md, default_run_sample, default_loss_df, default_checkpoint_df = show_run(default_run)
    default_figure_note, default_figure_path = show_figure(default_figure) if default_figure else ('', None)
    default_battery_note, default_battery_path = show_battery_figure(default_battery_figure) if default_battery_figure else ('', None)

    battery_lines = [
        '# Battery Sentinel',
        '',
        'Extension appliquee de l etude tri-bruit pour le suivi batterie.',
    ]
    if battery_simulation:
        battery_lines.append(f"- Sessions simulees : `{battery_simulation.get('num_sessions', 'n/a')}`")
        battery_lines.append(f"- Longueur des sequences : `{battery_simulation.get('sequence_length', 'n/a')}`")
    if battery_router:
        battery_lines.append(f"- Macro-F1 du routeur : `{battery_router.get('macro_f1', 0.0):.4f}`")
        battery_lines.append(f"- F1 de la baseline simple : `{battery_router.get('baseline_binary_f1', 0.0):.4f}`")
    battery_text = '\n'.join(battery_lines)

    with gr.Blocks(title=manifest['project_title']) as demo:
        gr.Markdown(overview_text)

        with gr.Tab('Vue d ensemble'):
            gr.Dataframe(value=results_df, label='Tableau comparatif', interactive=False)
            with gr.Accordion('Configuration controlee', open=False):
                gr.Dataframe(value=controls_df, label='Configuration', interactive=False)
            summary_path = manifest['figures'].get('summary_figure.png', {}).get('absolute')
            if summary_path:
                gr.Image(value=summary_path, label='Figure de synthese', type='filepath', height=420)
            if not td_df.empty:
                with gr.Accordion('Dynamique d entrainement', open=False):
                    gr.Dataframe(value=td_df, label='Resume de l entrainement', interactive=False)

        with gr.Tab('Comparateur diffusion'):
            run_selector = gr.Dropdown(choices=noise_choices, value=default_run, label='Distribution')
            run_md = gr.Markdown(value=default_run_md)
            run_sample = gr.Image(value=default_run_sample, label='Echantillon final', type='filepath', height=420)
            with gr.Accordion('Historique et checkpoints', open=False):
                loss_table = gr.Dataframe(value=default_loss_df, label='Historique des pertes', interactive=False)
                checkpoint_table = gr.Dataframe(value=default_checkpoint_df, label='Checkpoints', interactive=False)
            run_selector.change(show_run, inputs=run_selector, outputs=[run_md, run_sample, loss_table, checkpoint_table])

            figure_selector = gr.Dropdown(choices=figure_choices, value=default_figure, label='Figure globale')
            figure_note = gr.Markdown(value=default_figure_note)
            figure_view = gr.Image(value=default_figure_path, label='Figure', type='filepath', height=420)
            figure_selector.change(show_figure, inputs=figure_selector, outputs=[figure_note, figure_view])

        with gr.Tab('Battery Sentinel'):
            gr.Markdown(value=battery_text)
            if not battery_metrics_df.empty:
                gr.Dataframe(value=battery_metrics_df, label='Evaluation du predictive twin', interactive=False)
            if not battery_actions_df.empty:
                gr.Dataframe(value=battery_actions_df, label='Actions recommandees', interactive=False)
            if battery_figure_choices:
                battery_selector = gr.Dropdown(choices=battery_figure_choices, value=default_battery_figure, label='Figure batterie')
                battery_note = gr.Markdown(value=default_battery_note)
                battery_view = gr.Image(value=default_battery_path, label='Figure batterie', type='filepath', height=480)
                battery_selector.change(show_battery_figure, inputs=battery_selector, outputs=[battery_note, battery_view])

    return demo


def launch_app(artifacts_root=None, host: str = '127.0.0.1', port: int = 9000, share: bool = False) -> None:
    manifest = build_manifest(artifacts_root)
    demo = build_app(artifacts_root)
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        allowed_paths=[str(Path(manifest['repo_root']).resolve())],
    )

