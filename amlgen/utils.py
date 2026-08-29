"""Small shared helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def epoch_seconds(series: pd.Series) -> np.ndarray:
    """Datetime column -> int64 epoch seconds, independent of datetime resolution.

    pandas may store datetime64 at ns, us, ms or s resolution depending on
    version and construction path, so a hard-coded // 10**9 is a silent bug.
    """
    return series.to_numpy(dtype="datetime64[s]").astype(np.int64)
