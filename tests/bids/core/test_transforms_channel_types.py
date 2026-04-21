from dcap.bids.core.transforms import build_default_channel_types


def test_build_default_channel_types_uses_eeg_for_eeg_datatype() -> None:
    mapping = build_default_channel_types(
        channel_names=["Fz", "Cz", "ECG"],
        datatype="eeg",
        ecg_channel_name="ECG",
    )

    assert mapping == {"Fz": "eeg", "Cz": "eeg", "ECG": "ecg"}


def test_build_default_channel_types_preserves_seeg_default_for_ieeg() -> None:
    mapping = build_default_channel_types(
        channel_names=["A1", "A2", "ECG"],
        datatype="ieeg",
        ecg_channel_name="ECG",
    )

    assert mapping == {"A1": "seeg", "A2": "seeg", "ECG": "ecg"}
