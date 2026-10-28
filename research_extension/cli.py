from __future__ import annotations

import argparse

from .app import launch_app
from .artifacts import save_manifest
from .nearest_neighbor_audit import run_nearest_neighbor_audit
from .sample_grid_audit import run_saved_sample_audit
from .training_dynamics_audit import run_training_dynamics_audit


def _manifest_command(args: argparse.Namespace) -> None:
    output = save_manifest(args.output, args.artifacts_root)
    print(f"Manifest saved to {output}")


def _app_command(args: argparse.Namespace) -> None:
    launch_app(args.artifacts_root, host=args.host, port=args.port, share=args.share)


def _saved_sample_audit_command(args: argparse.Namespace) -> None:
    output = run_saved_sample_audit(
        artifacts_root=args.artifacts_root,
        output_path=args.output,
        markdown_output=args.markdown_output,
    )
    print(f"Saved sample audit saved to {output}")
    if args.markdown_output:
        print(f"Markdown summary saved to {args.markdown_output}")


def _nearest_neighbor_command(args: argparse.Namespace) -> None:
    output = run_nearest_neighbor_audit(
        artifacts_root=args.artifacts_root,
        output_path=args.output,
        markdown_output=args.markdown_output,
    )
    print(f"Nearest-neighbor audit saved to {output}")
    if args.markdown_output:
        print(f"Markdown summary saved to {args.markdown_output}")


def _training_dynamics_command(args: argparse.Namespace) -> None:
    output = run_training_dynamics_audit(
        artifacts_root=args.artifacts_root,
        output_path=args.output,
        markdown_output=args.markdown_output,
    )
    print(f"Training-dynamics audit saved to {output}")
    if args.markdown_output:
        print(f"Markdown summary saved to {args.markdown_output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research-layer tooling for the diffusion noise sensitivity project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="Build a machine-readable manifest from the saved experiment artifacts.")
    manifest.add_argument("--artifacts-root", default="diffusion_noise_project")
    manifest.add_argument("--output", default="research_extension_output/manifest.json")
    manifest.set_defaults(func=_manifest_command)

    app = subparsers.add_parser("app", help="Launch the interactive Gradio explorer.")
    app.add_argument("--artifacts-root", default="diffusion_noise_project")
    app.add_argument("--host", default="127.0.0.1")
    app.add_argument("--port", type=int, default=9000)
    app.add_argument("--share", action="store_true")
    app.set_defaults(func=_app_command)

    saved_audit = subparsers.add_parser("audit-saved-samples", help="Audit the final saved sample grids with classifier and image-based diagnostics.")
    saved_audit.add_argument("--artifacts-root", default="diffusion_noise_project")
    saved_audit.add_argument("--output", default="research_extension_output/saved_sample_audit.json")
    saved_audit.add_argument("--markdown-output")
    saved_audit.set_defaults(func=_saved_sample_audit_command)

    nn_audit = subparsers.add_parser("audit-nearest-neighbors", help="Compare saved sample grids against an MNIST feature bank.")
    nn_audit.add_argument("--artifacts-root", default="diffusion_noise_project")
    nn_audit.add_argument("--output", default="research_extension_output/nearest_neighbor_audit.json")
    nn_audit.add_argument("--markdown-output")
    nn_audit.set_defaults(func=_nearest_neighbor_command)

    td_audit = subparsers.add_parser("audit-training-dynamics", help="Analyze convergence and stability from the saved loss logs.")
    td_audit.add_argument("--artifacts-root", default="diffusion_noise_project")
    td_audit.add_argument("--output", default="research_extension_output/training_dynamics_audit.json")
    td_audit.add_argument("--markdown-output")
    td_audit.set_defaults(func=_training_dynamics_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
