"""CubeMind CLI — entry point for `python -m cubemind` and `cubemind` command.

Usage:
    cubemind demo           — Live brain demo (webcam + audio + keyboard teaching)
    cubemind forward TEXT   — Single forward pass on text input
    cubemind api            — Start FastAPI server
    cubemind version        — Print version
"""

from __future__ import annotations

import sys

import click

from cubemind import __version__


@click.group()
@click.version_option(__version__, prog_name="cubemind")
def cli() -> None:
    """CubeMind — neuro-vector-symbolic cognitive architecture."""


@cli.command()
@click.option("--camera", default=0, help="Camera index.")
@click.option("--width", default=1920, help="Capture width.")
@click.option("--height", default=1080, help="Capture height.")
@click.option("--k", default=8, help="VSA block count.")
@click.option("--l", "l_block", default=64, help="VSA block length.")
@click.option("--d-hidden", default=64, help="Hidden dimension.")
@click.option("--llm", default=None, help="Path to GGUF model.")
def demo(camera, width, height, k, l_block, d_hidden, llm) -> None:
    """Live brain demo — webcam + audio + keyboard teaching."""
    try:
        import cv2 # pyright: ignore[reportMissingImports]
    except ImportError:
        click.echo("Error: opencv-python required. Install with: pip install opencv-python")
        sys.exit(1)

    import time
    from cubemind import create_cubemind

    click.echo("Initializing CubeMind brain...")
    brain = create_cubemind(
        k=k, l=l_block, d_hidden=d_hidden,
        n_gif_levels=8, snn_timesteps=2, snn_ratio=0.3,
        enable_stdp=True,
        n_place_cells=500, n_time_cells=50, n_grid_cells=100,
        max_memories=50000, initial_neurons=32, max_neurons=2000,
        growth_threshold=0.3, enable_neurochemistry=True,
    )

    if llm:
        click.echo(f"Attaching LLM: {llm}")
        brain.attach_llm(model_path=llm, n_ctx=2048, n_gpu_layers=-1)

    click.echo(f"Brain ready: d_vsa={brain.d_vsa}, d_hidden={brain.d_hidden}")

    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    click.echo(
        f"Camera {camera}: "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )
    click.echo("\nControls: T=teach  R=recall  S=stats  Q=quit  SPACE=freeze")
    click.echo("=" * 50)

    fps_counter = 0
    fps_time = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        small = cv2.resize(frame, (160, 120))
        result = brain.forward(image=small)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time = time.time()

        display = _draw_overlay(frame, result, brain, cv2)
        cv2.putText(
            display, f"{fps_display:.0f} FPS | {elapsed_ms:.0f}ms",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2,
        )
        cv2.imshow("CubeMind Brain", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            label = input("\n  Teach label: ").strip()
            if label:
                result = brain.forward(text=label, image=small)
                ng = result.get("neurogenesis", {})
                click.echo(
                    f"  Stored: '{label}' (conf={result['confidence']:.3f}, "
                    f"neurons={ng.get('neuron_count', '?')})"
                )
        elif key == ord("r"):
            results = brain.recall(brain.bc.to_flat(result["input_hv"]), k=5)
            click.echo("\n  Recall top-5:")
            for mid, score in results:
                click.echo(f"    {mid}: {score:.3f}")
        elif key == ord("s"):
            _print_stats(brain)
        elif key == ord(" "):
            ng = result.get("neurogenesis", {})
            click.echo(f"\n  Frozen frame -- Step {result['step']}")
            click.echo(f"  Confidence: {result['confidence']:.3f}")
            click.echo(
                f"  Memories: {result['memories_retrieved']} retrieved, "
                f"{brain.hippocampus.memory_count} total"
            )
            click.echo(
                f"  Neurons: {ng.get('neuron_count', '?')} "
                f"(grew={ng.get('grew', False)}, pruned={ng.get('pruned', 0)})"
            )
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
    click.echo(
        f"\nSession complete: {brain.stats()['step']} steps, "
        f"{brain.hippocampus.memory_count} memories, "
        f"{brain.neurogenesis.neuron_count} neurons"
    )


@cli.command()
@click.argument("text")
@click.option("--k", default=8, help="VSA block count.")
@click.option("--l", "l_block", default=64, help="VSA block length.")
@click.option("--d-hidden", default=64, help="Hidden dimension.")
def forward(text, k, l_block, d_hidden) -> None:
    """Single forward pass on text input."""
    from cubemind import create_cubemind

    brain = create_cubemind(k=k, l=l_block, d_hidden=d_hidden)
    result = brain.forward(text=text)
    click.echo(f"Step: {result['step']}")
    click.echo(f"Confidence: {result['confidence']:.4f}")
    click.echo(f"Memories retrieved: {result['memories_retrieved']}")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host.")
@click.option("--port", default=8000, help="Bind port.")
def api(host, port) -> None:
    """Start the FastAPI server."""
    import uvicorn
    from cubemind.cloud.api import app
    uvicorn.run(app, host=host, port=port)


@cli.group()
def train() -> None:
    """Training subcommands."""


@train.command("vsa-lm")
@click.option("--steps", default=10000, help="Training steps.")
@click.option("--layers", default=6, help="Number of VSA layers.")
@click.option("--d-model", default=256, help="Model dimension.")
@click.option("--seq-len", default=64, help="Sequence length.")
@click.option("--data-dir", default="sandbox/vsa_lm/data", help="Tokenized data directory.")
def train_vsa_lm(steps, layers, d_model, seq_len, data_dir) -> None:
    """Train VSA-LM on TinyStories."""
    from cubemind.training.vsa_lm import main as vsa_lm_main
    vsa_lm_main()


@train.command("harrier")
@click.option("--steps", default=50000, help="Training steps.")
@click.option("--seq-len", default=64, help="Sequence length.")
@click.option("--lr", type=float, default=1e-3, help="Learning rate.")
@click.option("--save-every", default=5000, help="Save checkpoint every N steps.")
@click.option("--log-every", default=100, help="Log every N steps.")
def train_harrier(steps, seq_len, lr, save_every, log_every) -> None:
    """Pre-train MindForge on Harrier 0.6B teacher logits."""
    from cubemind.training.harrier_pretrain import HarrierPretrainConfig, train
    config = HarrierPretrainConfig(
        train_steps=steps, seq_len=seq_len, lr=lr,
        save_every=save_every, log_every=log_every,
    )
    train(config)


@cli.command()
def version() -> None:
    """Print version."""
    click.echo(f"cubemind {__version__}")


# ── Helpers ──────────────────────────────────────────────────────────────


def _print_stats(brain) -> None:
    stats = brain.stats()
    click.echo("\n  Brain Stats:")
    for k, v in stats.items():
        if isinstance(v, dict):
            click.echo(f"    {k}:")
            for k2, v2 in v.items():
                click.echo(f"      {k2}: {v2}")
        else:
            click.echo(f"    {k}: {v}")


def _draw_overlay(frame, result, brain, cv2):
    """Draw brain state overlay on the video frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    panel_w = 320
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    x0 = w - panel_w + 10
    y = 25
    dy = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45
    white, green, cyan = (255, 255, 255), (0, 255, 128), (255, 200, 0)
    yellow, red = (0, 220, 255), (80, 80, 255)

    cv2.putText(frame, "CubeMind Brain", (x0, y), font, 0.6, cyan, 2)
    y += dy + 5
    cv2.putText(frame, f"Step: {result['step']}", (x0, y), font, fs, white, 1)
    y += dy

    conf = result.get("confidence", 0)
    color = green if conf > 0.5 else yellow if conf > 0.2 else red
    cv2.putText(frame, f"Confidence: {conf:.3f}", (x0, y), font, fs, color, 1)
    y += dy
    cv2.putText(
        frame, f"Memories: {result.get('memories_retrieved', 0)}", (x0, y), font, fs, white, 1,
    )
    y += dy
    cv2.putText(
        frame, f"Total: {brain.hippocampus.memory_count}", (x0, y), font, fs, white, 1,
    )
    y += dy + 5

    ng = result.get("neurogenesis", {})
    cv2.putText(frame, "-- Neurogenesis --", (x0, y), font, fs, cyan, 1)
    y += dy
    grew = ng.get("grew", False)
    grow_str = f"Neurons: {ng.get('neuron_count', '?')}"
    if grew:
        grow_str += " [GREW!]"
    cv2.putText(frame, grow_str, (x0, y), font, fs, green if grew else white, 1)
    y += dy
    if ng.get("pruned", 0):
        cv2.putText(frame, f"Pruned: {ng['pruned']}", (x0, y), font, fs, red, 1)
        y += dy
    cv2.putText(
        frame, f"Residual EMA: {ng.get('residual_ema', 0):.4f}", (x0, y), font, fs, white, 1,
    )
    y += dy + 5

    nc = result.get("neurochemistry", {})
    if nc:
        cv2.putText(frame, "-- Neurochemistry --", (x0, y), font, fs, cyan, 1)
        y += dy
        cmap = {
            "dopamine": (0, 200, 255), "serotonin": (255, 200, 0),
            "cortisol": (80, 80, 255), "noradrenaline": (0, 255, 0),
            "oxytocin": (255, 100, 255),
        }
        for name in cmap:
            val = nc.get(name, 0)
            c = cmap[name]
            cv2.putText(frame, f"{name[:4]}: {val:.2f}", (x0, y), font, 0.4, c, 1)
            cv2.rectangle(frame, (x0 + 80, y - 8), (x0 + 80 + int(val * 150), y + 2), c, -1)
            y += dy - 2
        y += 5

    ctx = result.get("spatial_context", {})
    loc = ctx.get("current_location", [0, 0])
    cv2.putText(frame, f"Location: ({loc[0]:.1f}, {loc[1]:.1f})", (x0, y), font, fs, white, 1)
    return frame


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
