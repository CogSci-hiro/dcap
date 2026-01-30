# tests/cli/commands/test_bids_anat.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Optional, cast

import pytest


# =============================================================================
#                               Helper builders
# =============================================================================


def _build_root_parser() -> argparse.ArgumentParser:
    """
    Build a minimal root parser that can host dcap subcommands.

    Notes
    -----
    This avoids importing `dcap.cli.main` (which might register many commands).
    We keep the test focused specifically on bids-anat behavior.
    """
    parser = argparse.ArgumentParser(prog="dcap")
    parser.add_subparsers(dest="command", required=True)
    return parser


def _get_subparsers(parser: argparse.ArgumentParser) -> Any:
    """
    Return the argparse subparsers action from a root parser.

    Raises
    ------
    AssertionError
        If the parser does not have a subparsers action (programming error in test).
    """
    for action in parser._actions:  # noqa: SLF001 (ok in tests)
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            return action
    raise AssertionError("Root parser has no subparsers action.")


def _extract_handler(args: argparse.Namespace) -> Optional[Callable[..., Any]]:
    """
    Extract a handler callable from parsed args.

    We support common CLI patterns:
    - args.func
    - args.handler
    - args.run
    """
    for attr in ("func", "handler", "run"):
        value = getattr(args, attr, None)
        if callable(value):
            return cast(Callable[..., Any], value)
    return None


# =============================================================================
#                              Tests: registration
# =============================================================================


def test_bids_anat_command_registers_subparser() -> None:
    """
    Ensure the module registers the 'bids-anat' subcommand without error.
    """
    from dcap.cli.commands import bids_anat as cmd_bids_anat

    parser = _build_root_parser()
    subparsers = _get_subparsers(parser)

    cmd_bids_anat.add_subparser(subparsers)

    # Parse a minimal command to ensure the name exists (will fail later if args missing).
    # We intentionally do NOT assert on required args here.
    with pytest.raises(SystemExit):
        parser.parse_args(["bids-anat", "--help"])


# =============================================================================
#                           Tests: argument contract
# =============================================================================


def test_bids_anat_requires_expected_minimum_args(tmp_path: Path) -> None:
    """
    If bids-anat defines required args, argparse should enforce them.

    This test is intentionally flexible: it only checks that parsing *without any*
    additional flags fails, which is true for most real commands.
    """
    from dcap.cli.commands import bids_anat as cmd_bids_anat

    parser = _build_root_parser()
    subparsers = _get_subparsers(parser)
    cmd_bids_anat.add_subparser(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(["bids-anat"])


@pytest.mark.parametrize(
    "argv",
    [
        # These are *typical* bids-anat flags; adapt names to your actual CLI.
        # The goal is to verify parsing works and we can reach the handler.
        #
        # If your CLI uses different flag names, change them here (fast).
        ["bids-anat", "--bids-root", "BIDS", "--bids-subject", "sub-001"],
        ["bids-anat", "--bids-root", "BIDS", "--bids-subject", "001"],
    ],
)
def test_bids_anat_parses_common_args(argv: list[str]) -> None:
    """
    Parse a typical invocation and ensure we get a handler attached.

    This assumes the command uses the standard pattern:
        parser.set_defaults(func=run_from_args)
    """
    from dcap.cli.commands import bids_anat as cmd_bids_anat

    parser = _build_root_parser()
    subparsers = _get_subparsers(parser)
    cmd_bids_anat.add_subparser(subparsers)

    args = parser.parse_args(argv)
    handler = _extract_handler(args)

    # If this fails, it usually means:
    # - the command doesn't attach args.func (fine; adjust _extract_handler), or
    # - your test argv doesn't include required flags (adjust argv above).
    assert handler is not None, "bids-anat did not attach a callable handler (func/handler/run)."


# =============================================================================
#                        Tests: wiring to library logic
# =============================================================================


def test_bids_anat_handler_calls_library_function(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Verify the bids-anat CLI handler delegates to the underlying library function.

    This is the key test: "CLI is thin; logic is in the library".
    """
    from dcap.cli.commands import bids_anat as cmd_bids_anat

    # -------------------------------------------------------------------------
    # Build parser + register command
    # -------------------------------------------------------------------------
    parser = _build_root_parser()
    subparsers = _get_subparsers(parser)
    cmd_bids_anat.add_subparser(subparsers)

    bids_root = tmp_path / "bids"
    bids_root.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Monkeypatch the library entrypoint the command is expected to call
    # -------------------------------------------------------------------------
    calls: dict[str, Any] = {}

    def _fake_library_entrypoint(**kwargs: Any) -> None:
        calls["called"] = True
        calls["kwargs"] = kwargs

    # >>> IMPORTANT <<<
    # Update this patch target to match the real function your command calls.
    #
    # Common patterns:
    # - dcap.bids.anat.convert_anat(...)
    # - dcap.bids.core.anat.convert_anat(...)
    # - dcap.seeg.io.anat.export_anat(...)
    #
    # Search in your bids_anat.py for the import it uses and patch *that symbol*.
    monkeypatch.setattr(cmd_bids_anat, "convert_anat", _fake_library_entrypoint, raising=False)

    # If your command imports like:
    #   from dcap.bids.anat import convert_anat
    # then monkeypatching cmd_bids_anat.convert_anat is correct.
    #
    # If it imports a module and calls module.convert_anat, patch that module attribute instead.

    # -------------------------------------------------------------------------
    # Act: parse args and call handler
    # -------------------------------------------------------------------------
    argv = ["bids-anat", "--bids-root", str(bids_root), "--bids-subject", "sub-001"]
    args = parser.parse_args(argv)

    handler = _extract_handler(args)
    assert handler is not None, "bids-anat did not attach a callable handler."

    # Handler calling convention varies; support the most common ones:
    # - handler(args)
    # - handler(args=args)
    try:
        handler(args)
    except TypeError:
        handler(args=args)

    # -------------------------------------------------------------------------
    # Assert: library was called
    # -------------------------------------------------------------------------
    assert calls.get("called", False), (
        "Expected bids-anat handler to call convert_anat (or equivalent). "
        "Update monkeypatch target to the actual entrypoint used in bids_anat.py."
    )

    # Optional: sanity-check that key fields are forwarded
    forwarded = cast(dict[str, Any], calls.get("kwargs", {}))
    # We check weakly to avoid coupling to exact param names.
    assert any(str(bids_root) == str(v) for v in forwarded.values()), "bids_root was not forwarded."
    assert any("sub-001" == v or "001" == v for v in forwarded.values()), "bids_subject was not forwarded."
