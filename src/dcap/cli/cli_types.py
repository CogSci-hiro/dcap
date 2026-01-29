# =============================================================================
#                              CLI: Types
# =============================================================================

from typing import Protocol


class CliCommand(Protocol):
    """
    CLI subcommand module protocol.

    Usage example
    -------------
        # A module implementing this protocol must define:
        # - add_subparser(subparsers)
        # - run(args)
        ...
    """

    def add_subparser(self, subparsers) -> None:  # noqa: ANN001
        ...

    def run(self, args) -> None:  # noqa: ANN001
        ...
