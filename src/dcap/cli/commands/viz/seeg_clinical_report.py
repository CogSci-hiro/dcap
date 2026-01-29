# =============================================================================
# =============================================================================
#                         ###############################
#                         #   CLI COMMAND: clinical-report #
#                         ###############################
# =============================================================================
# =============================================================================

import argparse
from pathlib import Path
from typing import Any, Optional

from dcap.seeg.clinical.configs import ClinicalAnalysisConfig
from dcap.seeg.clinical.run_from_bids import run_clinical_report_from_bids
from dcap.seeg.preprocessing.configs import ClinicalPreprocConfig, GammaEnvelopeConfig


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "clinical-report",
        help="Run BIDS -> preprocessing -> (optional envelope/TRF) -> clinical report",
    )

    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--subject", type=str, required=True, help="BIDS subject (sub-001 or 001)")
    parser.add_argument("--session", type=str, default=None, help="BIDS session (ses-01 or 01)")
    parser.add_argument("--task", type=str, required=True, help="BIDS task (conversation)")
    parser.add_argument("--run", type=str, default=None, help="BIDS run (run-1 or 1)")

    parser.add_argument(
        "--analysis-view",
        type=str,
        default="original",
        choices=["original", "car", "bipolar", "laplacian", "wm_ref"],
    )

    parser.add_argument("--compute-gamma-envelope", action="store_true")
    parser.add_argument("--gamma-low", type=float, default=70.0)
    parser.add_argument("--gamma-high", type=float, default=150.0)
    parser.add_argument("--gamma-smoothing-sec", type=float, default=0.1)

    parser.set_defaults(_dcap_command="seeg_clinical_report")
    return parser


def run(args: argparse.Namespace, cfg: Any) -> Path:
    analysis_cfg = ClinicalAnalysisConfig(analysis_view=str(args.analysis_view))

    envelope_cfg: Optional[GammaEnvelopeConfig] = None
    if bool(args.compute_gamma_envelope):
        envelope_cfg = GammaEnvelopeConfig(
            band_hz=(float(args.gamma_low), float(args.gamma_high)),
            method="hilbert",
            smoothing_sec=float(args.gamma_smoothing_sec),
        )

    preproc_cfg = ClinicalPreprocConfig()

    report_path = run_clinical_report_from_bids(
        bids_root=Path(args.bids_root),
        out_dir=Path(args.out_dir),
        subject_id=str(args.subject),
        session_id=(None if args.session is None else str(args.session)),
        task=str(args.task),
        run_id=(None if args.run is None else str(args.run)),
        preproc_cfg=preproc_cfg,
        analysis_cfg=analysis_cfg,
        envelope_cfg=envelope_cfg,
        trf_cfg=None,
        trf_runner=None,
    )
    return report_path
