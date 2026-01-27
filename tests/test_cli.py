from dcap.cli.main import main


def test_cli_help_exits_successfully() -> None:
    exit_code = main(["--help"])
    assert exit_code == 0


def test_cli_registry_init_templates_writes_files(tmp_path) -> None:
    public_out = tmp_path / "registry_public.csv"
    private_out = tmp_path / "registry_private.csv"

    exit_code = main(
        [
            "registry",
            "init-templates",
            "--public-out",
            str(public_out),
            "--private-out",
            str(private_out),
        ]
    )
    assert exit_code == 0
    assert public_out.exists()
    assert private_out.exists()


def test_cli_registry_make_public_creates_sanitized_public_registry(tmp_path) -> None:
    private_path = tmp_path / "registry_private.csv"
    out_path = tmp_path / "registry_public.csv"
    subject_map = tmp_path / "subject_map.csv"

    # Private registry uses subject_key rather than subject.
    private_path.write_text(
        "subject_key,session,task,run,clinician_notes\n"
        "H1,ses-01,conversation,1,VERY_SENSITIVE\n",
        encoding="utf-8",
    )
    subject_map.write_text(
        "subject_key,subject\n"
        "H1,sub-001\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "registry",
            "make-public",
            "--private",
            str(private_path),
            "--out",
            str(out_path),
            "--bids-root",
            "/data/bids/conversation",
            "--subject-map",
            str(subject_map),
        ]
    )
    assert exit_code == 0
    assert out_path.exists()

    text = out_path.read_text("utf-8")
    assert "clinician_notes" not in text
    assert "subject_key" not in text
    assert "sub-001" in text
