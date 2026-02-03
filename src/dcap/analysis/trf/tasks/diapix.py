# =============================================================================
#                         TRF task adapter: Diapix
# =============================================================================


from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, List, Literal

import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, filtfilt, hilbert, resample_poly

from dcap.analysis.trf.alignment import event_time_to_sample
from dcap.analysis.trf.prep import stack_time_epoch_feature
from dcap.analysis.trf.tasks.base import TrfEpochedData


EnvelopeChannelMode = Literal["mean", "self", "other", "all"]


@dataclass(frozen=True, slots=True)
class EnvelopeFromWavConfig:
    lowpass_hz: float = 20.0
    channel_mode: EnvelopeChannelMode = "all"   # NEW default
    target_sfreq: float = 100.0


@dataclass(frozen=True, slots=True)
class EnvelopeFromWavConfig:
    """
    Configuration for computing amplitude envelope feature(s) from a WAV file.

    Parameters
    ----------
    lowpass_hz
        Low-pass cutoff applied to the Hilbert amplitude envelope (Hz).
    channel_mode
        How to map WAV channels to envelope feature(s):

        - "mean": average all channels -> 1 feature
        - "self": channel 0 -> 1 feature
        - "other": channel 1 -> 1 feature (requires >=2 channels)
        - "all": 3 features [mean, ch0, ch1] (requires >=2 channels)
    target_sfreq
        Sampling frequency of returned envelope feature(s) (Hz).

    Usage example
    -------------
        cfg = EnvelopeFromWavConfig(channel_mode="all", target_sfreq=100.0)
    """
    lowpass_hz: float = 20.0
    channel_mode: EnvelopeChannelMode = "all"
    target_sfreq: float = 100.0


@dataclass(frozen=True, slots=True)
class DiapixConfig:
    """
    Diapix TRF extraction configuration.

    Parameters
    ----------
    task_name : str
        BIDS task name for diapix runs.
    expected_runs : int
        Expected number of runs (epochs).
    event_name : str
        BIDS event marking audio onset in the neural timeline.
    target_sfreq : float
        TRF sampling frequency for both X and Y (Hz).
    envelope : EnvelopeFromWavConfig
        How to compute the envelope from WAV.

    Usage example
    -------------
        cfg = DiapixConfig(target_sfreq=100.0)
    """

    task_name: str = "diapix"
    expected_runs: int = 4
    event_name: str = "conversation_start"
    target_sfreq: float = 100.0
    envelope: EnvelopeFromWavConfig = EnvelopeFromWavConfig()


class DiapixTrfAdapter:
    """
    TRF task adapter for Diapix.

    What it does
    ------------
    - Discovers per-run events.tsv + WAV
    - Computes envelope from WAV (Hilbert amplitude + lowpass)
    - Loads neural data for each run
    - Aligns neural time series using the `conversation_start` onset
    - Crops to shared overlap
    - Stacks 4 runs into epochs dimension for MNE ReceptiveField:
        X: (n_times, n_epochs, n_features)
        Y: (n_times, n_epochs, n_outputs)

    Notes
    -----
    File discovery uses a small set of glob patterns to be robust to project layout.
    If something cannot be found, the raised error includes what was attempted.

    Usage example
    -------------
        adapter = DiapixTrfAdapter(DiapixConfig(target_sfreq=100.0))
        data = adapter.load_epoched(Path("/path/to/bids"), subject="sub-001", session=None)
    """

    name = "diapix"

    def __init__(self, config: DiapixConfig) -> None:
        self._config = config

    # =============================================================================
    # Public API
    # =============================================================================

    def load_epoched(self, bids_root: Path, *, subject: str, session: str | None) -> TrfEpochedData:
        run_ids = [f"run-{i}" for i in range(1, self._config.expected_runs + 1)]

        X_runs: List[np.ndarray] = []
        Y_runs: List[np.ndarray] = []

        for run_id in run_ids:
            events_path = self._find_events_tsv(bids_root, subject=subject, session=session, run_id=run_id)
            wav_path = self._find_wav(bids_root, subject=subject, session=session, run_id=run_id)
            neural_path_or_bids = self._find_neural(bids_root, subject=subject, session=session, run_id=run_id)

            onset_s = self._load_conversation_start_onset_s(events_path, event_name=self._config.event_name)

            X_env, sfreq_x = self._compute_envelope_from_wav(wav_path, self._config.envelope)
            raw = self._load_neural_raw(neural_path_or_bids)
            Y, sfreq_y = self._raw_to_Y(raw)

            # Resample to target TRF sfreq (both X and Y)
            target = float(self._config.target_sfreq)
            X_env = self._resample_time_feature(X_env, sfreq_in=sfreq_x, sfreq_out=target)
            Y = self._resample_time_feature(Y, sfreq_in=sfreq_y, sfreq_out=target)

            # Align using conversation_start: it marks the audio onset in the neural timeline.
            y_onset_samp = event_time_to_sample(onset_s=onset_s, sfreq=target, first_samp=0)
            if y_onset_samp < 0 or y_onset_samp >= Y.shape[0]:
                raise ValueError(
                    f"{subject} {run_id}: conversation_start sample {y_onset_samp} out of bounds for Y length {Y.shape[0]}."
                )

            # Convention: X_env[0] is audio onset. Crop Y from onset; X starts at 0.
            Y_aligned = Y[y_onset_samp:, :]
            X_aligned = X_env

            # Crop to overlap
            n = min(X_aligned.shape[0], Y_aligned.shape[0])
            if n <= 0:
                raise ValueError(f"{subject} {run_id}: no overlap after alignment.")
            X_runs.append(X_aligned[:n, :])
            Y_runs.append(Y_aligned[:n, :])

        # Make all runs equal length (crop to minimum)
        min_len = min(arr.shape[0] for arr in X_runs + Y_runs)
        X_runs = [arr[:min_len, :] for arr in X_runs]
        Y_runs = [arr[:min_len, :] for arr in Y_runs]

        X_epoched = stack_time_epoch_feature(X_runs)  # (time, epoch, feature)
        Y_epoched = stack_time_epoch_feature(Y_runs)  # (time, epoch, channel)

        return TrfEpochedData(X=X_epoched, Y=Y_epoched, sfreq=float(self._config.target_sfreq), epoch_ids=run_ids)

    # =============================================================================
    # Discovery helpers (task-specific)
    # =============================================================================

    def _find_events_tsv(self, bids_root: Path, *, subject: str, session: str | None, run_id: str) -> Path:
        task = self._config.task_name
        subj_dir = bids_root / subject
        ses_part = f"{session}/" if session else ""
        patterns = [
            subj_dir / ses_part / "ieeg" / f"{subject}_{session + '_' if session else ''}task-{task}_{run_id}_events.tsv",
            subj_dir / ses_part / "ieeg" / f"{subject}_task-{task}_{run_id}_events.tsv",
            subj_dir / ses_part / "eeg" / f"{subject}_{session + '_' if session else ''}task-{task}_{run_id}_events.tsv",
            subj_dir / ses_part / "eeg" / f"{subject}_task-{task}_{run_id}_events.tsv",
        ]
        found = self._first_existing(patterns)
        if found is None:
            attempted = "\n".join(str(p) for p in patterns)
            raise FileNotFoundError(f"Could not find events.tsv for {subject} {run_id}. Tried:\n{attempted}")
        return found

    def _find_wav(self, bids_root: Path, *, subject: str, session: str | None, run_id: str) -> Path:
        """
        Locate the run WAV file.

        We try a few common places:
        - BIDS stim/ (if you keep audio there)
        - derivatives (if audio is stored as derivative)
        - a broad fallback search under the subject directory
        """
        task = self._config.task_name
        subj_dir = bids_root / subject
        ses_dir = subj_dir / session if session else subj_dir

        patterns = [
            ses_dir / "stim" / f"{subject}_{session + '_' if session else ''}task-{task}_{run_id}_stim.wav",
            ses_dir / "stim" / f"{subject}_task-{task}_{run_id}_stim.wav",
            bids_root / "derivatives" / "dcap" / subject / (session or "") / "audio" / f"{subject}_task-{task}_{run_id}.wav",
            bids_root / "derivatives" / "dcap" / subject / (session or "") / "audio" / f"{subject}_task-{task}_{run_id}_audio.wav",
        ]

        found = self._first_existing(patterns)
        if found is not None:
            return found

        # Broad fallback: search under subject dir for anything with task/run and .wav
        query_root = ses_dir
        candidates = sorted(query_root.rglob(f"*task-{task}*{run_id}*.wav"))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            # choose the shortest path (often the most canonical) but show ambiguity
            candidates = sorted(candidates, key=lambda p: len(str(p)))
            return candidates[0]

        raise FileNotFoundError(
            f"Could not find WAV for {subject} {run_id}. "
            f"Tried common patterns and searched under {query_root} for '*task-{task}*{run_id}*.wav'."
        )

    def _find_neural(self, bids_root: Path, *, subject: str, session: str | None, run_id: str) -> Any:
        """
        Locate neural data for this run.

        Strategy:
        1) Prefer preprocessed FIF in derivatives/dcap/... if present
        2) Otherwise return a BIDS locator dict that _load_neural_raw() can use with mne-bids if installed
        """
        task = self._config.task_name
        deriv_root = bids_root / "derivatives" / "dcap" / subject / (session or "")
        patterns = [
            deriv_root / "ieeg" / f"{subject}_task-{task}_{run_id}_desc-preproc_raw.fif",
            deriv_root / "ieeg" / f"{subject}_task-{task}_{run_id}_desc-clean_raw.fif",
            deriv_root / "ieeg" / f"{subject}_task-{task}_{run_id}_raw.fif",
            deriv_root / "eeg" / f"{subject}_task-{task}_{run_id}_desc-preproc_raw.fif",
        ]
        found = self._first_existing(patterns)
        if found is not None:
            return found

        # Fallback to BIDS locator (requires mne-bids)
        return {
            "bids_root": bids_root,
            "subject": subject.replace("sub-", ""),
            "session": (session.replace("ses-", "") if session else None),
            "task": task,
            "run": run_id.replace("run-", ""),
            "datatype": "ieeg",
        }

    # =============================================================================
    # Loading / computation helpers
    # =============================================================================

    def _load_conversation_start_onset_s(self, events_tsv: Path, *, event_name: str) -> float:
        df = pd.read_csv(events_tsv, sep="\t")

        # Common BIDS event column is "trial_type" but some people use "event_type" or "value"
        candidate_cols = [c for c in ["trial_type", "event_type", "value"] if c in df.columns]
        if "onset" not in df.columns:
            raise ValueError(f"{events_tsv} has no 'onset' column.")
        if not candidate_cols:
            raise ValueError(f"{events_tsv} has no trial_type/event_type/value column to match {event_name!r}.")

        for col in candidate_cols:
            hits = df[df[col].astype(str) == str(event_name)]
            if len(hits) > 0:
                onset_s = float(hits.iloc[0]["onset"])
                return onset_s

        raise ValueError(f"{events_tsv}: could not find event {event_name!r} in columns {candidate_cols}.")

    def _compute_envelope_from_wav(self, wav_path: Path, cfg: EnvelopeFromWavConfig) -> tuple[np.ndarray, float]:
        """
        Compute envelope feature(s) X from a WAV file.

        Returns
        -------
        X : ndarray, shape (n_times, n_features)
            n_features == 1 for {"mean","self","other"}
            n_features == 3 for "all" -> [mean, ch0, ch1]
        sfreq : float
            Sampling frequency of X (Hz), after resampling to cfg.target_sfreq.
        """
        try:
            import soundfile as sf  # type: ignore
        except Exception as exc:
            raise ImportError("Reading WAV requires 'soundfile' (pip install soundfile).") from exc

        audio, sfreq_in = sf.read(str(wav_path), always_2d=True)  # (n_samples, n_channels)
        n_channels = int(audio.shape[1])

        mode = cfg.channel_mode
        if mode == "mean":
            waves = [np.mean(audio, axis=1)]
        elif mode == "self":
            waves = [audio[:, 0]]
        elif mode == "other":
            if n_channels < 2:
                raise ValueError(f"{wav_path}: channel_mode='other' requires >=2 channels, got {n_channels}.")
            waves = [audio[:, 1]]
        elif mode == "all":
            if n_channels < 2:
                raise ValueError(f"{wav_path}: channel_mode='all' requires >=2 channels, got {n_channels}.")
            waves = [np.mean(audio, axis=1), audio[:, 0], audio[:, 1]]
        else:
            raise ValueError(f"Unsupported channel_mode={mode!r}.")

        # Envelope per waveform
        env_feats = []
        for x in waves:
            x = x.astype(np.float64, copy=False)
            x = x - float(np.mean(x))

            amp = np.abs(hilbert(x))

            nyq = 0.5 * float(sfreq_in)
            cutoff = float(cfg.lowpass_hz)
            if cutoff <= 0 or cutoff >= nyq:
                raise ValueError(f"Envelope lowpass_hz must be in (0, {nyq}). Got {cutoff}.")

            b, a = butter(N=4, Wn=cutoff / nyq, btype="low")  # noqa
            env = filtfilt(b, a, amp).astype(np.float64, copy=False)  # noqa
            env_feats.append(env)

        X = np.stack(env_feats, axis=1)  # (time, feature)

        sfreq_out = float(cfg.target_sfreq)
        if sfreq_out <= 0:
            raise ValueError("Envelope target_sfreq must be > 0.")

        if sfreq_out != float(sfreq_in):
            X = self._resample_time_feature(X, sfreq_in=float(sfreq_in), sfreq_out=sfreq_out)

        return X, sfreq_out

    def _load_neural_raw(self, neural_ref: Any) -> mne.io.BaseRaw:
        """
        Load neural raw data.

        - If `neural_ref` is a Path: read it as FIF.
        - If it's a dict: attempt to use mne-bids to read from BIDS.
        """
        if isinstance(neural_ref, Path):
            return mne.io.read_raw_fif(neural_ref, preload=True, verbose="ERROR")

        if isinstance(neural_ref, dict):
            try:
                from mne_bids import BIDSPath, read_raw_bids  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "Could not find a preprocessed FIF and mne-bids is not installed. "
                    "Install mne-bids or write a project-specific loader for your neural files."
                ) from exc

            bids_path = BIDSPath(
                root=neural_ref["bids_root"],
                subject=neural_ref["subject"],
                session=neural_ref["session"],
                task=neural_ref["task"],
                run=neural_ref["run"],
                datatype=neural_ref.get("datatype", "ieeg"),
            )
            raw = read_raw_bids(bids_path=bids_path, verbose="ERROR")
            raw.load_data()
            return raw

        raise TypeError(f"Unsupported neural reference type: {type(neural_ref)}")

    def _raw_to_Y(self, raw: mne.io.BaseRaw) -> tuple[np.ndarray, float]:
        """
        Convert MNE Raw to a Y array (time, channels).
        """
        Y = raw.get_data(picks="all").T  # (time, channels)
        sfreq = float(raw.info["sfreq"])
        return Y.astype(np.float64, copy=False), sfreq

    def _resample_time_feature(self, x: np.ndarray, *, sfreq_in: float, sfreq_out: float) -> np.ndarray:
        """
        Polyphase resampling along time axis=0 for 2D arrays (time, feature).
        """
        if sfreq_in == sfreq_out:
            return x
        ratio = float(sfreq_out) / float(sfreq_in)
        frac = Fraction(ratio).limit_denominator(1000)
        up = int(frac.numerator)
        down = int(frac.denominator)

        # Resample each feature independently
        feats = []
        for j in range(x.shape[1]):
            feats.append(resample_poly(x[:, j], up=up, down=down))
        return np.vstack(feats).T

    def _first_existing(self, paths: Iterable[Path]) -> Path | None:
        for p in paths:
            if p.exists():
                return p
        return None
