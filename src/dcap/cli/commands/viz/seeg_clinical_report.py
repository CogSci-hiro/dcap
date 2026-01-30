# dcap/cli/commands/seeg_clinical_report.py

import argparse
from pathlib import Path
from typing import Any

from dcap.seeg.clinical.run_from_bids import run_clinical_report_from_bids
from dcap.seeg.preprocessing.configs import ClinicalPreprocConfig, GammaEnvelopeConfig
from dcap.seeg.clinical.configs import ClinicalAnalysisConfig
from dcap.seeg.preprocessing.configs import LineNoiseConfig, RereferenceConfig
from dcap.seeg.trf.contracts import TRFConfig


def add_subparser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "seeg-clinical-report",
        help="Run BIDS → preprocessing → TRF → clinical report",
    )

    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run", default=None)
    parser.add_argument("--session", default=None)

    parser.add_argument(
        "--analysis-view",
        default="original",
        choices=["original", "car", "bipolar", "laplacian", "wm_ref"],
    )

    parser.add_argument(
        "--line-noise-method",
        choices=["notch", "zapline"],
        default="notch",
        help="Line-noise removal method",
    )

    parser.add_argument(
        "--line-noise-freq",
        type=float,
        default=50.0,
        help="Base line frequency (50 or 60 Hz)",
    )


def run(args: argparse.Namespace) -> None:
    reref_cfg = RereferenceConfig(
        methods=(args.analysis_view,),
    )

    preproc_cfg = ClinicalPreprocConfig(
        rereference=reref_cfg,
        line_noise=LineNoiseConfig(
            method=str(args.line_noise_method).lower(),
            freq_base=float(args.line_noise_freq),
        ))
    analysis_cfg = ClinicalAnalysisConfig(analysis_view=args.analysis_view)

    report_path = run_clinical_report_from_bids(
        bids_root=args.bids_root,
        out_dir=args.out_dir,
        subject_id=args.subject,
        session_id=args.session,
        task=args.task,
        run_id=args.run,
        preproc_cfg=preproc_cfg,
        analysis_cfg=analysis_cfg,
    )

    print(report_path)
