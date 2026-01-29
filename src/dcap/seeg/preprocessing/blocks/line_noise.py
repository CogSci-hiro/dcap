# =============================================================================
# =============================================================================
#                     ########################################
#                     #    BLOCK 3: LINE-NOISE REMOVAL       #
#                     ########################################
# =============================================================================
# =============================================================================

from dataclasses import asdict
from typing import Tuple

import numpy as np
import mne

from dcap.seeg.preprocessing.configs import LineNoiseConfig
from dcap.seeg.preprocessing.types import BlockArtifact, PreprocContext


def _compute_line_freqs(freq_base: float, max_freq: float) -> list[float]:
    n_harmonics = int(max_freq // freq_base)
    return [freq_base * k for k in range(1, n_harmonics + 1)]


def remove_line_noise(
    raw: mne.io.BaseRaw,
    cfg: LineNoiseConfig,
    ctx: PreprocContext,
) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    """
    Remove line noise using notch or zapline.

    Parameters
    ----------
    raw
        MNE Raw object.
    cfg
        Line-noise configuration.
    ctx
        Preprocessing context.

    Returns
    -------
    raw_out
        Raw object (copy) with line-noise removal applied.
    artifact
        Block artifact.

    Usage example
    -------------
        ctx = PreprocContext()
        raw_out, artifact = remove_line_noise(
            raw=raw,
            cfg=LineNoiseConfig(method="notch", freq_base=50.0, max_harmonic_hz=250.0),
            ctx=ctx,
        )
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("remove_line_noise expects an mne.io.BaseRaw.")

    ctx.add_record("line_noise", asdict(cfg))

    freqs = _compute_line_freqs(cfg.freq_base, cfg.max_harmonic_hz)
    raw_out = raw.copy()

    warnings: list[str] = []

    if cfg.method == "notch":
        raw_out.notch_filter(
            freqs=freqs,
            picks=cfg.picks,
            method="spectrum_fit",
            phase="zero",
            verbose=False,
        )
    elif cfg.method == "zapline":
        try:
            from meegkit import dss
        except ImportError as exc:
            raise ImportError("Zapline requested but meegkit is not installed.") from exc

        data = raw_out.get_data(picks=cfg.picks)
        cleaned, _ = dss.line(data, fline=cfg.freq_base, sfreq=float(raw_out.info["sfreq"]), nremove=1, show=False)
        # Write back into raw._data for selected channels
        sel = raw_out._pick_indices(cfg.picks)
        raw_out._data[sel] = cleaned
        warnings.append("Zapline applied using meegkit.dss.line (nremove=1).")
    else:
        raise ValueError(f"Unknown line-noise method: {cfg.method!r}")

    artifact = BlockArtifact(
        name="line_noise",
        parameters={**asdict(cfg), "freqs_applied": freqs},
        summary_metrics={"n_freqs": float(len(freqs))},
        warnings=warnings,
        figures=[],
    )
    return raw_out, artifact
