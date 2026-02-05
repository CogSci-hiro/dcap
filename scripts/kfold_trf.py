# =============================================================================
#                     ########################################
#                     #     DCAP SIMPLE TRF PIPELINE (CV)     #
#                     ########################################
# =============================================================================
"""
Simple TRF pipeline (single, non-nested CV) on a BIDS root.

Per subject:
  Per run:
    iEEG:
      - notch 50 Hz + harmonics (capped below Nyquist)
      - broadband: 1–30 Hz
      - high-gamma: 70–150 Hz -> Hilbert envelope
      - downsample to 128 Hz
      - z-score

    Audio (stereo):
      - envelopes for: both (mean), self (L), other (R)
      - Hilbert envelope
      - downsample to 128 Hz
      - z-score

Then:
  - Stack along runs: X_ep (time, run, 1), Y_ep (time, run, channels)
  - Single CV across runs (k folds):
      * compute R² for each alpha, fold, channel
      * select best alpha per channel using mean R² across folds
      * report CV R² per channel using that selected alpha
  - Save:
      * CSV of per-channel CV R² + best alpha
      * 3D electrode plots for 2 (seeg) x 3 (stim) conditions

Usage example
-------------
    python dev/scripts/simple_pipeline.py \
      --bids_root /Volumes/work-4T/diapix-bids \
      --out_dir /Volumes/work-4T/diapix-bids/derivatives/trf_simple_cv \
      --subjects sub-003 sub-004
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)

import mne

try:
    import mne_bids
    from mne_bids import BIDSPath
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires mne-bids. Install it or replace the BIDS loader."
    ) from exc

try:
    import soundfile as sf
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires soundfile to read WAV audio.") from exc

from scipy.signal import hilbert

from dcap.analysis.trf.design_matrix import LagConfig
from dcap.analysis.trf.fit import fit_trf_ridge, predict_trf
from dcap.viz.electrodes.electrodes_3d import plot_electrodes_3d
from sklearn.model_selection import KFold



# =============================================================================
#                     ########################################
#                     #               CONSTANTS               #
#                     ########################################
# =============================================================================

SFREQ_OUT_HZ: float = 128.0

NOTCH_BASE_HZ: float = 50.0
NOTCH_N_HARMONICS: int = 6
NOTCH_NYQUIST_MARGIN_HZ: float = 0.5

BROADBAND_LFREQ_HZ: float = 1.0
BROADBAND_HFREQ_HZ: float = 30.0

HIGHGAMMA_LFREQ_HZ: float = 70.0
HIGHGAMMA_HFREQ_HZ: float = 150.0

CV_ALONG_TIME: bool = True

TIME_K_FOLDS: int = 10          # number of folds over time blocks
TIME_N_BLOCKS: int = 10        # split the full time series into this many contiguous blocks
TIME_SHUFFLE: bool = False     # MUST be False for contiguous behavior
TIME_RANDOM_STATE: int = 0


ALPHAS: np.ndarray = np.logspace(-6, 6, 10)  # np.array([1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4], dtype=float)

TMIN_MS: float = -500.0
TMAX_MS: float = 500.0
LAG_STEP_MS: float = 1000.0 / SFREQ_OUT_HZ  # 1 sample at 128 Hz

SCORE_METRIC: Literal["r2"] = "r2"

SeegCond = Literal["broadband", "highgamma"]
StimCond = Literal["both", "self", "other"]

CROP_EVENT_START: str = "conversation_start"
CROP_EVENT_END: str = "conversation_end"
CROP_EXPECTED_DURATION_S: float = 240.0  # 4 minutes
CROP_TOLERANCE_S: float = .01            # allow small annotation jitter

THRESHOLD = 0.0005

SPEECH_ENV_LP_HZ: float = 15.0          # common choice: 8 Hz (syllabic-rate-ish)
SPEECH_ENV_LP_TRANS_BW_HZ: float = 2.0 # transition band

# Gap to prevent TRF lag leakage (seconds). Use >= max(|tmin|, |tmax|); conservative is max lag.
CV_GAP_S: float = max(abs(TMIN_MS), abs(TMAX_MS)) / 1000.0


# =============================================================================
#                     ########################################
#                     #                CONFIG                 #
#                     ########################################
# =============================================================================

@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Pipeline configuration.

    Parameters
    ----------
    bids_root
        BIDS dataset root.
    out_dir
        Output directory for CSV + plots.
    task
        If provided, forces a BIDS task; else we infer per run from filenames.

    Usage example
    -------------
        cfg = PipelineConfig(
            bids_root=Path("/data/bids"),
            out_dir=Path("./out"),
            task=None,
        )
    """

    bids_root: Path
    out_dir: Path
    task: Optional[str] = None


# =============================================================================
#                     ########################################
#                     #             BIDS HELPERS              #
#                     ########################################
# =============================================================================

def list_subjects(bids_root: Path) -> List[str]:
    """List subject folders like ['sub-001', ...]."""
    return sorted([p.name for p in bids_root.glob("sub-*") if p.is_dir()])


def list_runs_for_subject(bids_root: Path, subject: str) -> List[str]:
    """Return run IDs as strings. Robust to mne-bids version differences."""
    subject_id = subject.replace("sub-", "")

    # Try mne-bids entity listing
    try:
        try:
            runs = mne_bids.get_entity_vals(root=str(bids_root), entity_key="run", subject=subject_id)
        except TypeError:
            runs = mne_bids.get_entity_vals(bids_root=str(bids_root), entity_key="run", subject=subject_id)

        runs = [r for r in runs if r is not None]
        out = sorted({str(int(r)) if str(r).isdigit() else str(r) for r in runs})
        if out:
            return out
    except Exception:
        pass

    # Fallback: glob filenames under sub-XXX/ieeg and parse run-*
    sub_dir = bids_root / subject
    found: set[str] = set()
    for p in sub_dir.glob("**/*run-*_ieeg.*"):
        name = p.name
        if "run-" not in name:
            continue
        token = name.split("run-", 1)[1].split("_", 1)[0].split(".", 1)[0]
        if token:
            found.add(str(int(token)) if token.isdigit() else token)

    out = sorted(found)
    if not out:
        raise RuntimeError(f"No runs found for {subject} under {sub_dir}")
    return out


def _infer_task_for_run(bids_root: Path, subject: str, run: str) -> str:
    """Infer BIDS task from the ieeg filename."""
    sub_dir = bids_root / subject / "ieeg"
    run_candidates = _run_candidates(run)

    matches: List[Path] = []
    for r in run_candidates:
        matches.extend(sub_dir.glob(f"{subject}_task-*_run-{r}_ieeg.*"))
        matches.extend(sub_dir.glob(f"{subject}_task-*_run-{r}_ieeg.*.*"))

    matches = [m for m in matches if m.is_file()]
    if not matches:
        raise RuntimeError(f"Could not infer task for {subject} run-{run}; no matching ieeg files in {sub_dir}")

    name = sorted(matches, key=lambda p: len(p.name))[0].name
    task_part = name.split("task-", 1)[1]
    task = task_part.split("_", 1)[0]
    if not task:
        raise RuntimeError(f"Failed to parse task from: {name}")
    return task


def _run_candidates(run: str) -> List[str]:
    """Return run variants: ['4','04'] etc, preserving uniqueness."""
    out: List[str] = [str(run)]
    try:
        r_int = int(str(run))
        out.extend([str(r_int), f"{r_int:02d}"])
    except Exception:
        pass
    seen: set[str] = set()
    return [r for r in out if not (r in seen or seen.add(r))]


def load_raw_ieeg_bids(bids_root: Path, subject: str, run: str, *, task: Optional[str]) -> mne.io.BaseRaw:
    """Load an iEEG Raw from BIDS with robust run formatting."""
    subject_id = subject.replace("sub-", "")
    task_used = task if task is not None else _infer_task_for_run(bids_root, subject, run)

    last_exc: Optional[Exception] = None
    for run_try in _run_candidates(run):
        bids_path = BIDSPath(
            root=str(bids_root),
            subject=subject_id,
            task=task_used,
            run=run_try,
            datatype="ieeg",
            suffix="ieeg",
        )
        try:
            raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
            raw.load_data()
            return raw
        except Exception as exc:
            last_exc = exc
            continue

    raise FileNotFoundError(
        f"Could not load iEEG for {subject} task-{task_used} run={run}. Last error: {last_exc}"
    )


def find_audio_wav_for_run(bids_root: Path, subject: str, run: str) -> Path:
    """Audio path: sub-XXX/audio/sub-XXX_task-*_run-<run>.wav"""
    audio_dir = bids_root / subject / "audio"
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")

    matches: List[Path] = []
    for r in _run_candidates(run):
        matches.extend(audio_dir.glob(f"{subject}_task-*_run-{r}.wav"))

    matches = [m for m in matches if m.is_file()]
    if not matches:
        raise FileNotFoundError(f"No audio WAV found for {subject} run={run} in {audio_dir}")
    return sorted(matches, key=lambda p: len(p.name))[0]


def load_audio_stereo(wav_path: Path) -> Tuple[np.ndarray, float]:
    """Load stereo WAV -> (n_samples, 2) float64."""
    audio, sfreq = sf.read(str(wav_path), always_2d=True)
    if audio.shape[1] < 2:
        raise ValueError(f"Audio is not stereo: {wav_path} shape={audio.shape}")
    return audio[:, :2].astype(np.float64), float(sfreq)


def load_electrodes_df(bids_root: Path, subject: str) -> pd.DataFrame:
    """Load electrodes TSV from derivatives/elec_recon/sub-XXX."""
    base = bids_root / "derivatives" / "elec_recon" / subject
    matches = sorted(base.glob(f"{subject}_*electrodes.tsv"))
    if not matches:
        raise FileNotFoundError(f"Could not find electrodes TSV under {base}")
    df = pd.read_csv(matches[0], sep="\t")
    if df.empty:
        raise ValueError(f"Electrodes TSV is empty: {matches[0]}")
    return df


# =============================================================================
#                     ########################################
#                     #           DSP + PREPROCESS            #
#                     ########################################
# =============================================================================

def zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Z-score along axis."""
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, keepdims=True) + 1e-12
    return (x - mu) / sd


def hilbert_envelope_1d(x: np.ndarray) -> np.ndarray:
    """Hilbert amplitude envelope of a 1D signal."""
    return np.abs(hilbert(x)).astype(np.float64)


def _valid_notch_freqs(*, sfreq: float) -> np.ndarray:
    """Compute 50 Hz harmonics capped below Nyquist."""
    nyq = sfreq / 2.0
    freqs_all = np.arange(1, NOTCH_N_HARMONICS + 1, dtype=float) * NOTCH_BASE_HZ
    freqs = freqs_all[freqs_all < (nyq - NOTCH_NYQUIST_MARGIN_HZ)]
    return freqs.astype(float)


def preprocess_seeg_branch(raw: mne.io.BaseRaw, *, branch: SeegCond) -> Tuple[np.ndarray, List[str]]:
    """
    Preprocess iEEG into (n_times, n_channels) at SFREQ_OUT_HZ.

    Usage example
    -------------
        y, ch = preprocess_seeg_branch(raw, branch="highgamma")
    """
    r = raw.copy()

    # Notch (safely below Nyquist)
    freqs = _valid_notch_freqs(sfreq=float(r.info["sfreq"]))
    if freqs.size > 0:
        r.notch_filter(freqs=freqs, verbose=False)

    if branch == "broadband":
        r.filter(l_freq=BROADBAND_LFREQ_HZ, h_freq=BROADBAND_HFREQ_HZ, verbose=False)
        r.resample(SFREQ_OUT_HZ, npad="auto", verbose=False)
        data = r.get_data().T
        return zscore(data, axis=0), list(r.ch_names)

    if branch == "highgamma":
        r.filter(l_freq=HIGHGAMMA_LFREQ_HZ, h_freq=HIGHGAMMA_HFREQ_HZ, verbose=False)
        r.resample(SFREQ_OUT_HZ, npad="auto", verbose=False)
        data = r.get_data().T
        env = np.zeros_like(data, dtype=np.float64)
        for ci in range(data.shape[1]):
            env[:, ci] = hilbert_envelope_1d(data[:, ci])
        return zscore(env, axis=0), list(r.ch_names)

    raise ValueError(f"Unknown branch: {branch}")


def preprocess_audio_envelopes(audio_stereo: np.ndarray, *, sfreq_in: float) -> Dict[StimCond, np.ndarray]:
    """
    Create 3 envelopes (both/self/other), resample to SFREQ_OUT_HZ, z-score.

    Returns
    -------
    envs : dict
        Keys: 'both', 'self', 'other'. Values: (n_times,) arrays.
    """
    left = audio_stereo[:, 0]
    right = audio_stereo[:, 1]

    signals: Dict[StimCond, np.ndarray] = {
        "both": 0.5 * (left + right),
        "self": left,
        "other": right,
    }

    envs: Dict[StimCond, np.ndarray] = {}
    for cond, sig in signals.items():
        env = hilbert_envelope_1d(sig.astype(np.float64))

        # Low-pass the envelope (before resampling)
        env_lp = lowpass_1d(env, sfreq=sfreq_in, lpf_hz=SPEECH_ENV_LP_HZ)

        # Downsample to TRF rate
        env_rs = mne.filter.resample(env_lp, down=sfreq_in, up=SFREQ_OUT_HZ, npad="auto")

        # Z-score per run
        envs[cond] = zscore(env_rs.reshape(-1, 1), axis=0).ravel()

    return envs


def trim_all_to_min_len(arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Trim 1D/2D arrays along time to the minimum length."""
    nmin = min(a.shape[0] for a in arrays)
    return [a[:nmin, ...] for a in arrays]


# =============================================================================
#                     ########################################
#                     #             CV + SCORING              #
#                     ########################################
# =============================================================================

def r2_per_channel(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """
    R² per channel.

    Parameters
    ----------
    y, y_hat : ndarray
        Shape (n_times, n_channels)

    Returns
    -------
    r2 : ndarray
        Shape (n_channels,)
    """
    ss_res = np.sum((y - y_hat) ** 2, axis=0)
    y_mean = np.mean(y, axis=0, keepdims=True)
    ss_tot = np.sum((y - y_mean) ** 2, axis=0) + 1e-12
    return (1.0 - ss_res / ss_tot).astype(np.float64)


def make_run_folds(n_runs: int, k_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create CV folds over run indices.

    Behavior
    --------
    - If k_folds >= n_runs: Leave-One-Run-Out (LORO)
    - Else: deterministic contiguous splits (no randomness)

    Usage example
    -------------
        folds = make_run_folds(n_runs=4, k_folds=4)
    """
    idx = np.arange(n_runs)

    if k_folds >= n_runs:
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        for t in range(n_runs):
            test = np.array([t], dtype=int)
            train = idx[idx != t]
            folds.append((train, test))
        return folds

    # Deterministic contiguous split
    splits = np.array_split(idx, k_folds)
    folds = []
    for test in splits:
        train = np.setdiff1d(idx, test)
        folds.append((train.astype(int), test.astype(int)))
    return folds


def cv_r2_alpha_search_time_kfold(
    X: np.ndarray,            # (n_time, 1)
    Y: np.ndarray,            # (n_time, n_ch)
    *,
    sfreq: float,
    lag_config: LagConfig,
    alphas: np.ndarray,
    k_folds: int,
    n_blocks: int,
    gap_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single CV along TIME using sklearn KFold over contiguous time blocks.
    Alpha selection is per-channel using mean CV R² across folds.

    Returns
    -------
    r2_by_fold_by_ch : (k_folds, n_ch)
    best_alpha_by_ch : (n_ch,)
    r2_grid : (n_alphas, k_folds, n_ch)
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X must be (time, features) and Y must be (time, channels).")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same n_time.")

    n_time = X.shape[0]
    n_ch = Y.shape[1]
    n_alphas = int(alphas.size)

    block_id = make_time_blocks(n_samples=n_time, n_blocks=n_blocks)
    unique_blocks = np.unique(block_id)

    if k_folds > unique_blocks.size:
        raise ValueError(f"k_folds={k_folds} > n_blocks={unique_blocks.size}. Reduce k or increase blocks.")

    kf = KFold(n_splits=k_folds, shuffle=TIME_SHUFFLE, random_state=TIME_RANDOM_STATE if TIME_SHUFFLE else None)

    r2_grid = np.zeros((n_alphas, k_folds, n_ch), dtype=np.float64)

    for fold_i, (train_blocks_idx, test_blocks_idx) in enumerate(kf.split(unique_blocks)):
        train_blocks = set(unique_blocks[train_blocks_idx].tolist())
        test_blocks = set(unique_blocks[test_blocks_idx].tolist())

        train_mask = np.array([b in train_blocks for b in block_id], dtype=bool)
        test_mask = np.array([b in test_blocks for b in block_id], dtype=bool)

        # IMPORTANT: gap to prevent TRF lag leakage across boundaries
        train_mask = apply_gap_to_train_mask(train_mask=train_mask, test_mask=test_mask, gap_s=gap_s, sfreq=sfreq)

        X_train = X[train_mask, :]
        Y_train = Y[train_mask, :]
        X_test = X[test_mask, :]
        Y_test = Y[test_mask, :]

        for ai, a in enumerate(alphas):
            fit = fit_trf_ridge(X_train, Y_train, sfreq=sfreq, lag_config=lag_config, alpha=float(a))
            Y_hat = predict_trf(X_test, fit)
            r2_grid[ai, fold_i, :] = r2_per_channel(Y_test, Y_hat)

    mean_r2_by_alpha_by_ch = np.mean(r2_grid, axis=1)  # (n_alphas, n_ch)
    best_idx = np.argmax(mean_r2_by_alpha_by_ch, axis=0)
    best_alpha_by_ch = alphas[best_idx].astype(np.float64)

    r2_folds = np.zeros((k_folds, n_ch), dtype=np.float64)
    for ch_i in range(n_ch):
        r2_folds[:, ch_i] = r2_grid[best_idx[ch_i], :, ch_i]

    return r2_folds, best_alpha_by_ch, r2_grid


# =============================================================================
#                     ########################################
#                     #               PLOTTING                #
#                     ########################################
# =============================================================================

def save_electrode_plot(
    *,
    out_png: Path,
    electrodes_df: pd.DataFrame,
    values_by_channel: Dict[str, float],
    title: str,
    coords_space: Optional[str] = None,
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = None,
) -> None:
    """
    Wrapper for plot_electrodes_3d() with your current signature.

    Notes
    -----
    plot_electrodes_3d expects color_values aligned to electrodes_df row order.
    We align using electrodes_df['name'].

    Usage example
    -------------
        save_electrode_plot(
            out_png=Path("out.png"),
            electrodes_df=edf,
            values_by_channel={"LA01": 0.12},
            title="sub-001 | broadband x both",
        )
    """
    if "name" not in electrodes_df.columns:
        raise ValueError("electrodes_df must contain a 'name' column.")

    names = electrodes_df["name"].astype(str).to_numpy()
    color_values = np.full((names.size,), np.nan, dtype=np.float64)
    for i, n in enumerate(names):
        if n in values_by_channel:
            color_values[i] = float(values_by_channel[n])

    finite = color_values[np.isfinite(color_values)]
    vmax_used = float(np.nanmax(finite)) if (vmax is None and finite.size) else vmax

    plot_electrodes_3d(
        electrodes_df=electrodes_df,
        out_path=out_png,
        coords_space=coords_space,
        title=title,
        color_values=color_values,
        size_values=np.abs(color_values),
        vmin=THRESHOLD,
        vmax=vmax_used,
        threshold=THRESHOLD
    )


# =============================================================================
#                     ########################################
#                     #              SUBJECT RUN              #
#                     ########################################
# =============================================================================

def run_subject(cfg: PipelineConfig, subject: str) -> pd.DataFrame:
    """
    Run the pipeline for one subject and return a results DataFrame.

    Returns
    -------
    df : DataFrame
        One row per (seeg_cond, stim_cond, channel) with CV R² + alpha.

    DataFrame format example
    ------------------------
    sub_id | seeg_cond  | stim_cond | channel | best_alpha | r2_mean | r2_fold0 | ...
    """
    runs = list_runs_for_subject(cfg.bids_root, subject)
    if not runs:
        raise RuntimeError(f"No runs found for {subject}")

    electrodes_df = load_electrodes_df(cfg.bids_root, subject)

    lag_config = LagConfig(
        tmin_ms=TMIN_MS,
        tmax_ms=TMAX_MS,
        step_ms=LAG_STEP_MS,
        mode="same",
    )

    seeg_by_cond: Dict[SeegCond, List[np.ndarray]] = {"broadband": [], "highgamma": []}
    env_by_cond: Dict[StimCond, List[np.ndarray]] = {"both": [], "self": [], "other": []}
    ch_names_ref: Optional[List[str]] = None

    for run in runs:
        raw = load_raw_ieeg_bids(cfg.bids_root, subject, run, task=cfg.task)
        raw = crop_raw_to_conversation(raw)

        y_bb, ch_names = preprocess_seeg_branch(raw, branch="broadband")
        y_hg, ch_names_hg = preprocess_seeg_branch(raw, branch="highgamma")

        if ch_names_ref is None:
            ch_names_ref = ch_names
        else:
            if ch_names != ch_names_ref or ch_names_hg != ch_names_ref:
                raise ValueError(f"Channel list differs across runs for {subject}. Align channels first.")

        wav = find_audio_wav_for_run(cfg.bids_root, subject, run)
        audio, sf_audio = load_audio_stereo(wav)
        envs = preprocess_audio_envelopes(audio, sfreq_in=sf_audio)

        # Trim all arrays to common time length for this run
        arrays_to_trim: List[np.ndarray] = [y_bb, y_hg]
        arrays_to_trim.extend([envs["both"].reshape(-1, 1), envs["self"].reshape(-1, 1), envs["other"].reshape(-1, 1)])
        trimmed = trim_all_to_min_len(arrays_to_trim)

        y_bb_t = trimmed[0]
        y_hg_t = trimmed[1]
        env_both = trimmed[2].ravel()
        env_self = trimmed[3].ravel()
        env_other = trimmed[4].ravel()

        seeg_by_cond["broadband"].append(y_bb_t)
        seeg_by_cond["highgamma"].append(y_hg_t)
        env_by_cond["both"].append(env_both)
        env_by_cond["self"].append(env_self)
        env_by_cond["other"].append(env_other)

    if ch_names_ref is None:
        raise RuntimeError("No channels found after loading.")

    fig_dir = cfg.out_dir / "figures" / subject
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    n_runs = len(runs)
    n_ch = len(ch_names_ref)

    for seeg_cond in ("broadband", "highgamma"):
        # unify time length across runs
        y_runs = seeg_by_cond[seeg_cond]
        y_runs = trim_all_to_min_len(y_runs)
        n_time = y_runs[0].shape[0]
        Y_ep = np.stack(y_runs, axis=1)  # (time, run, ch)

        for stim_cond in ("both", "self", "other"):
            x_runs = env_by_cond[stim_cond]
            x_runs = trim_all_to_min_len([x.reshape(-1, 1) for x in x_runs])
            n_time2 = min(n_time, x_runs[0].shape[0])

            X_cat = concatenate_runs_1d(env_by_cond[stim_cond]).reshape(-1, 1)  # (time_total, 1)
            Y_cat = concatenate_runs_2d(seeg_by_cond[seeg_cond])  # (time_total, n_ch)

            # Make sure they match
            nmin = min(X_cat.shape[0], Y_cat.shape[0])
            X_cat = X_cat[:nmin, :]
            Y_cat = Y_cat[:nmin, :]

            r2_folds, alpha_by_ch, _grid = cv_r2_alpha_search_time_kfold(
                X=X_cat,
                Y=Y_cat,
                sfreq=SFREQ_OUT_HZ,
                lag_config=lag_config,
                alphas=ALPHAS,
                k_folds=TIME_K_FOLDS,
                n_blocks=TIME_N_BLOCKS,
                gap_s=CV_GAP_S,
            )
            r2_mean = np.mean(r2_folds, axis=0)

            # Plot electrodes (mean CV R²)
            values_map = {ch: float(r2_mean[i]) for i, ch in enumerate(ch_names_ref)}
            out_png = fig_dir / f"{subject}_{seeg_cond}_x_{stim_cond}_r2.png"
            title = f"{subject} | {seeg_cond} x {stim_cond} | CV R² (single CV)"
            try:
                save_electrode_plot(
                    out_png=out_png,
                    electrodes_df=electrodes_df,
                    values_by_channel=values_map,
                    title=title,
                    vmin=0.0,
                    vmax=None,
                )
            except Exception as exc:
                print(f"[WARN] Plot failed for {subject} {seeg_cond} {stim_cond}: {exc}")

            # Rows (one per channel)
            for ci, ch in enumerate(ch_names_ref):
                row: Dict[str, Any] = {
                    "sub_id": subject,
                    "seeg_cond": seeg_cond,
                    "stim_cond": stim_cond,
                    "channel": ch,
                    "best_alpha": float(alpha_by_ch[ci]),
                    "r2_mean": float(r2_mean[ci]),
                }
                for fi in range(r2_folds.shape[0]):
                    row[f"r2_fold{fi}"] = float(r2_folds[fi, ci])
                rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
#                     ########################################
#                     #                 CLI                  #
#                     ########################################
# =============================================================================

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bids_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--task", type=str, default=None, help="Optional BIDS task (e.g., diapix). Auto-infer if omitted.")
    p.add_argument("--subjects", type=str, nargs="*", default=None, help="Optional list: sub-003 sub-004 ...")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    cfg = PipelineConfig(
        bids_root=Path(args.bids_root),
        out_dir=Path(args.out_dir),
        task=args.task,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects if args.subjects else list_subjects(cfg.bids_root)

    all_dfs: List[pd.DataFrame] = []
    for sub in subjects:
        print(f"[INFO] Processing {sub}")
        df_sub = run_subject(cfg, sub)
        all_dfs.append(df_sub)

    df = pd.concat(all_dfs, axis=0, ignore_index=True)

    out_csv = cfg.out_dir / "trf_cv_r2.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote CSV: {out_csv}")


def crop_raw_to_conversation(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Crop raw using annotations: conversation_start -> conversation_end.

    Requires annotations with descriptions matching CROP_EVENT_START/CROP_EVENT_END.

    Returns
    -------
    raw_cropped : Raw
        Cropped copy of raw.

    Raises
    ------
    ValueError
        If required events are missing or duration is not ~4 minutes.
    """
    ann = raw.annotations
    if ann is None or len(ann) == 0:
        raise ValueError("Raw has no annotations; cannot crop to conversation window.")

    desc = np.asarray(ann.description, dtype=str)
    onsets = np.asarray(ann.onset, dtype=float)

    start_mask = desc == CROP_EVENT_START
    end_mask = desc == CROP_EVENT_END

    if not np.any(start_mask):
        raise ValueError(f"Missing annotation: {CROP_EVENT_START}")
    if not np.any(end_mask):
        raise ValueError(f"Missing annotation: {CROP_EVENT_END}")

    # Use the first start after time 0, and the first end after that start
    start_t = float(np.min(onsets[start_mask]))
    end_candidates = onsets[end_mask]
    end_after = end_candidates[end_candidates > start_t]
    if end_after.size == 0:
        raise ValueError(f"Found {CROP_EVENT_END} but none occur after {CROP_EVENT_START}.")

    end_t = float(np.min(end_after))

    dur = end_t - start_t
    if not (CROP_EXPECTED_DURATION_S - CROP_TOLERANCE_S <= dur <= CROP_EXPECTED_DURATION_S + CROP_TOLERANCE_S):
        raise ValueError(
            f"Conversation crop duration is {dur:.3f}s, expected ~{CROP_EXPECTED_DURATION_S}s "
            f"(±{CROP_TOLERANCE_S}s). start={start_t:.3f}, end={end_t:.3f}"
        )

    # Crop on the raw time axis (seconds)
    raw_cropped = raw.copy().crop(tmin=start_t, tmax=end_t, include_tmax=False)
    return raw_cropped


def lowpass_1d(x: np.ndarray, *, sfreq: float, lpf_hz: float) -> np.ndarray:
    """
    Zero-phase low-pass using MNE FIR filter.

    Parameters
    ----------
    x
        Shape (n_times,)
    sfreq
        Sampling rate of x (Hz)
    lpf_hz
        Low-pass cutoff (Hz)

    Returns
    -------
    x_filt
        Shape (n_times,)
    """
    if lpf_hz <= 0:
        return x
    nyq = sfreq / 2.0
    if lpf_hz >= nyq:
        return x
    return mne.filter.filter_data(
        x.astype(np.float64),
        sfreq=float(sfreq),
        l_freq=None,
        h_freq=float(lpf_hz),
        verbose=False,
    ).astype(np.float64)


def concatenate_runs_1d(x_runs: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([x.reshape(-1) for x in x_runs], axis=0)

def concatenate_runs_2d(y_runs: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([y for y in y_runs], axis=0)


def make_time_blocks(n_samples: int, n_blocks: int) -> np.ndarray:
    """
    Assign each sample to a contiguous block id in [0, n_blocks-1].
    """
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2")
    # contiguous split points
    edges = np.linspace(0, n_samples, n_blocks + 1, dtype=int)
    block_id = np.empty(n_samples, dtype=int)
    for bi in range(n_blocks):
        block_id[edges[bi]:edges[bi + 1]] = bi
    return block_id


def apply_gap_to_train_mask(
    *,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    gap_s: float,
    sfreq: float,
) -> np.ndarray:
    """
    Remove samples within +/- gap_s of the test region from the training mask.
    """
    gap_n = int(round(gap_s * sfreq))
    if gap_n <= 0:
        return train_mask

    test_idx = np.flatnonzero(test_mask)
    if test_idx.size == 0:
        return train_mask

    # Expand test region by gap on both sides
    start = max(int(test_idx.min()) - gap_n, 0)
    end = min(int(test_idx.max()) + gap_n + 1, train_mask.size)
    train_mask2 = train_mask.copy()
    train_mask2[start:end] = False
    return train_mask2


if __name__ == "__main__":
    main()
