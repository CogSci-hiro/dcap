# src/dcap/cli/commands/bids_anat.py
# =============================================================================
#                              CLI: bids-anat
# =============================================================================

import argparse
from pathlib import Path
from typing import Any

from dcap.bids.core.anat import AnatWriteConfig, write_anat_and_derivatives
from dcap.bids.core.subject_mapping import load_subject_mapping_entry


def add_subparser(subparsers: Any) -> None:
    """
    Register the bids-anat subcommand.

    Usage example
    -------------
        dcap bids-anat --bids-root bids --bids-subject sub-001 --subjects-dir /path/to/SUBJECTS_DIR \
          --original-id Nic-Ele --overwrite
    """
    parser = subparsers.add_parser(
        "bids-anat",
        help="Write subject-level anatomy (T1w) and copy recon derivatives into BIDS.",
    )

    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--bids-subject", type=str, required=True, help='BIDS subject label, e.g. "sub-001" or "001".')
    parser.add_argument("--session", type=str, default=None, help='Optional session label, e.g. "01" or "ses-01".')
    parser.add_argument("--dataset-id", type=str, required=True,
                        help="Dataset identifier used in the subject mapping YAML.")
    parser.add_argument("--mapping-yaml", type=Path, required=True,
                        help="YAML mapping file with bids_subject -> original_id.")

    parser.add_argument("--subjects-dir", type=Path, required=True, help="FreeSurfer SUBJECTS_DIR")

    parser.add_argument("--deface", action="store_true", help="Deface T1w during write.")
    parser.add_argument("--no-copy-elec-recon", action="store_true", help="Do not copy elec_recon to derivatives.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs where supported.")


def run(args: argparse.Namespace) -> None:
    """
    Run the bids-anat command.

    Parameters
    ----------
    args
        Parsed CLI args.

    Returns
    -------
    None

    Usage example
    -------------
        run(argparse.Namespace(
            bids_root=Path("bids"),
            bids_subject="sub-001",
            session=None,
            subjects_dir=Path("/fs/SUBJECTS_DIR"),
            original_id="Nic-Ele",
            deface=False,
            no_copy_elec_recon=False,
            overwrite=True,
        ))
    """
    bids_subject_bare = _strip_prefix(str(args.bids_subject), "sub")
    session_bare = _strip_prefix(str(args.session), "ses") if args.session is not None else None

    entry = load_subject_mapping_entry(
        mapping_yaml=Path(args.mapping_yaml),
        dataset_id=str(args.dataset_id).strip(),
        bids_subject=str(args.bids_subject),
    )

    cfg = AnatWriteConfig(
        subjects_dir=Path(args.subjects_dir).expanduser().resolve(),
        original_id=entry.original_id,
        bids_root=Path(args.bids_root).expanduser().resolve(),
        bids_subject=bids_subject_bare,
        session=session_bare,
        deface=bool(args.deface),
        overwrite=bool(args.overwrite),
        copy_elec_recon=not bool(args.no_copy_elec_recon),
    )
    write_anat_and_derivatives(cfg)


def _strip_prefix(value: str, prefix: str) -> str:
    s = str(value).strip()
    token = f"{prefix}-"
    return s[len(token):] if s.startswith(token) else s
