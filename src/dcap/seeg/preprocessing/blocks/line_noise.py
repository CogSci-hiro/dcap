from dataclasses import asdict
from typing import Tuple

import mne

from dcap.seeg.preprocessing.configs import LineNoiseConfig
from dcap.seeg.preprocessing.types import BlockArtifact, PreprocContext


def _compute_line_freqs(freq_base: float, max_freq: float) -> list[float]:
    n = int(max_freq // freq_base)
    return [freq_base * k for k in range(1, n + 1)]


def remove_line_noise(
    raw: mne.io.BaseRaw,
    cfg: LineNoiseConfig,
    ctx: PreprocContext,
) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    """
    Remove line noise using notch filtering (MNE) or Zapline (meegkit).

    Notes
    -----
    - This function is logic-only: no file I/O.
    - A copy of Raw is returned to avoid in-place surprises.
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("remove_line_noise expects an mne.io.BaseRaw.")

    ctx.add_record("line_noise", asdict(cfg))

    raw_out = raw.copy()

    freqs = _compute_line_freqs(cfg.freq_base, cfg.max_harmonic_hz)
    picks = cfg.picks

    warnings: list[str] = []

    if cfg.method == "notch":
        raw_out.notch_filter(
            freqs=freqs,
            picks=picks,
            method="spectrum_fit",
            phase="zero",
            verbose=False,
        )

    elif cfg.method == "zapline":
        try:
            from meegkit import dss
        except ImportError as exc:
            raise ImportError(
                "Zapline requested but meegkit is not installed."
            ) from exc

        data = raw_out.get_data(picks=picks)
        sfreq = raw_out.info["sfreq"]

        cleaned, _ = dss.line(
            data,
            fline=cfg.freq_base,
            sfreq=sfreq,
            nremove=1,
            show=False,
        )

        raw_out._data[raw_out._pick_indices(picks)] = cleaned
        warnings.append("Zapline applied using meegkit.dss.line (nremove=1).")

    else:
        raise ValueError(f"Unknown line-noise method: {cfg.method!r}")

    artifact = BlockArtifact(
        name="line_noise",
        parameters={
            **asdict(cfg),
            "freqs_applied": freqs,
        },
        summary_metrics={
            "n_freqs": len(freqs),
            "method": cfg.method,
        },
        warnings=warnings,
        figures=[],
    )

    return raw_out, artifact
