from dcap.cli.main import main


def test_cli_help_exits_successfully() -> None:
    exit_code = main(["--help"])
    assert exit_code == 0
