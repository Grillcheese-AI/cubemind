"""CubeMind DI Container — wires all modules via python-dependency-injector.

Each module is independent and optional. The container resolves dependencies
from config, falling back gracefully when modules are unavailable.

Usage:
    container = CubeMindContainer()
    container.config.from_dict({"k": 8, "l": 64, "d_hidden": 64})
    container.init_resources()

    brain = container.cubemind()
    result = brain.forward(text="hello")

Override any component:
    container.text_encoder.override(providers.Singleton(MyCustomEncoder, k=8, l=64))
"""

from __future__ import annotations

from dependency_injector import containers, providers

from cubemind.core.constants import K_BLOCKS, L_BLOCK


def _try_import(factory, **kwargs):
    """Try to instantiate a module; return None if it fails."""
    try:
        return factory(**kwargs)
    except Exception:
        return None


class CubeMindContainer(containers.DeclarativeContainer):
    """Top-level DI container for CubeMind.

    All providers are lazy — nothing is instantiated until called.
    Optional modules (vision, audio, neurochemistry, llm) return None
    if their dependencies are missing.
    """

    config = providers.Configuration(default={
        "k": K_BLOCKS,
        "l": L_BLOCK,
        "d_hidden": 128,
        "seed": 42,
        # SNN
        "snn_ratio": 0.5,
        "snn_timesteps": 4,
        "n_gif_levels": 16,
        "gif_tau": 10.0,
        "gif_threshold": 1.0,
        "gif_alpha": 0.01,
        "enable_stdp": True,
        # Memory
        "n_place_cells": 2000,
        "n_time_cells": 100,
        "n_grid_cells": 200,
        "max_memories": 100_000,
        # Neurogenesis
        "initial_neurons": 64,
        "max_neurons": 10000,
        "growth_threshold": 0.35,
        # Flags
        "enable_neurochemistry": True,
        "enable_vision": True,
        "enable_audio": True,
    })

    # ── Core (always available) ──────────────────────────────────────────

    block_codes = providers.Singleton(
        lambda k, l: __import__("cubemind.ops.block_codes", fromlist=["BlockCodes"]).BlockCodes(
            k=k, l=l
        ),
        k=config.k,
        l=config.l,
    )

    text_encoder = providers.Singleton(
        lambda k, l: __import__(
            "cubemind.perception.encoder", fromlist=["Encoder"]
        ).Encoder(k=k, l=l),
        k=config.k,
        l=config.l,
    )

    # ── Optional perception modules ──────────────────────────────────────

    harrier_encoder = providers.Singleton(
        lambda k, l: _try_import(
            __import__(
                "cubemind.perception.harrier_encoder", fromlist=["HarrierEncoder"]
            ).HarrierEncoder,
            k=k, l=l,
        ),
        k=config.k,
        l=config.l,
    )

    vision_encoder = providers.Singleton(
        lambda k, l, enable: _try_import(
            __import__(
                "cubemind.perception.bio_vision", fromlist=["BioVisionEncoder"]
            ).BioVisionEncoder,
            k=k, l=l,
        ) if enable else None,
        k=config.k,
        l=config.l,
        enable=config.enable_vision,
    )

    audio_encoder = providers.Singleton(
        lambda d_vsa, d_hidden, seed, enable: _try_import(
            __import__(
                "cubemind.perception.audio", fromlist=["AudioEncoder"]
            ).AudioEncoder,
            sample_rate=16000, n_mels=40,
            snn_neurons=d_hidden, d_vsa=d_vsa, seed=seed,
        ) if enable else None,
        d_vsa=providers.Callable(lambda k, l: k * l, config.k, config.l),
        d_hidden=config.d_hidden,
        seed=config.seed,
        enable=config.enable_audio,
    )

    # ── Processing ───────────────────────────────────────────────────────

    snn_ffn = providers.Singleton(
        lambda d_hidden, snn_ratio, snn_timesteps, n_gif_levels,
               gif_tau, gif_threshold, gif_alpha, enable_stdp, seed:
            __import__(
                "cubemind.brain.snn_ffn", fromlist=["HybridFFN"]
            ).HybridFFN(
                input_dim=d_hidden, hidden_dim=d_hidden * 2,
                snn_ratio=snn_ratio, num_timesteps=snn_timesteps,
                L=n_gif_levels, tau=gif_tau, threshold=gif_threshold,
                alpha=gif_alpha, enable_stdp=enable_stdp, seed=seed,
            ),
        d_hidden=config.d_hidden,
        snn_ratio=config.snn_ratio,
        snn_timesteps=config.snn_timesteps,
        n_gif_levels=config.n_gif_levels,
        gif_tau=config.gif_tau,
        gif_threshold=config.gif_threshold,
        gif_alpha=config.gif_alpha,
        enable_stdp=config.enable_stdp,
        seed=config.seed,
    )

    # ── Memory ───────────────────────────────────────────────────────────

    hippocampus = providers.Singleton(
        lambda n_place_cells, n_time_cells, n_grid_cells,
               max_memories, d_hidden, seed:
            __import__(
                "cubemind.memory.formation", fromlist=["HippocampalFormation"]
            ).HippocampalFormation(
                spatial_dimensions=2,
                n_place_cells=n_place_cells,
                n_time_cells=n_time_cells,
                n_grid_cells=n_grid_cells,
                max_memories=max_memories,
                feature_dim=d_hidden,
                seed=seed,
            ),
        n_place_cells=config.n_place_cells,
        n_time_cells=config.n_time_cells,
        n_grid_cells=config.n_grid_cells,
        max_memories=config.max_memories,
        d_hidden=config.d_hidden,
        seed=config.seed,
    )

    # ── Brain modules (optional) ─────────────────────────────────────────

    neurochemistry = providers.Singleton(
        lambda enable: _try_import(
            __import__(
                "cubemind.brain.neurochemistry", fromlist=["Neurochemistry"]
            ).Neurochemistry,
        ) if enable else None,
        enable=config.enable_neurochemistry,
    )

    neurogenesis = providers.Singleton(
        lambda initial_neurons, max_neurons, d_hidden, growth_threshold, seed:
            __import__(
                "cubemind.brain.neurogenesis", fromlist=["NeurogenesisController"]
            ).NeurogenesisController(
                initial_neurons=initial_neurons,
                max_neurons=max_neurons,
                feature_dim=d_hidden,
                growth_threshold=growth_threshold,
                seed=seed,
            ),
        initial_neurons=config.initial_neurons,
        max_neurons=config.max_neurons,
        d_hidden=config.d_hidden,
        growth_threshold=config.growth_threshold,
        seed=config.seed,
    )

    spike_bridge = providers.Singleton(
        lambda k, l, seed: __import__(
            "cubemind.brain.spike_vsa_bridge", fromlist=["SpikeVSABridge"]
        ).SpikeVSABridge(k=k, l=l, seed=seed),
        k=config.k,
        l=config.l,
        seed=config.seed,
    )

    # ── Top-level orchestrator ───────────────────────────────────────────

    cubemind = providers.Factory(
        lambda bc, text_enc, harrier, vision, audio, snn, hippo, neuro_chem,
               neuro_gen, spike_br, k, l, d_hidden, seed:
            __import__("cubemind.model", fromlist=["CubeMind"]).CubeMind(
                bc=bc, text_encoder=text_enc,
                harrier_encoder=harrier, vision_encoder=vision,
                audio_encoder=audio, snn_ffn=snn,
                hippocampus=hippo, neurochemistry=neuro_chem,
                neurogenesis=neuro_gen, spike_bridge=spike_br,
                k=k, l=l, d_hidden=d_hidden, seed=seed,
            ),
        bc=block_codes,
        text_enc=text_encoder,
        harrier=harrier_encoder,
        vision=vision_encoder,
        audio=audio_encoder,
        snn=snn_ffn,
        hippo=hippocampus,
        neuro_chem=neurochemistry,
        neuro_gen=neurogenesis,
        spike_br=spike_bridge,
        k=config.k,
        l=config.l,
        d_hidden=config.d_hidden,
        seed=config.seed,
    )
