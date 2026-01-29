from typing import Mapping, Tuple
import mne

from dcap.preprocessing.clinical.configs import AnalysisView

def choose_analysis_view(
    raw_views: Mapping[str, mne.io.BaseRaw],
    requested: AnalysisView,
) -> Tuple[str, mne.io.BaseRaw]:
    """
    Return (view_name_used, raw_used).
    Falls back to 'original' if requested isn't present.
    """
    if requested in raw_views:
        return requested, raw_views[requested]

    # Always available by contract
    return "original", raw_views["original"]
