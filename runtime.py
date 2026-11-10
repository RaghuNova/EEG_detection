"""Shared runtime configuration for quiet, reproducible CLI execution."""

from __future__ import annotations

import logging
import os
import warnings


def configure_quiet_runtime() -> None:
    """Suppress library chatter without hiding application exceptions.

    Third-party scientific libraries are intentionally noisy when reading EDF
    files and fitting models. The project keeps errors as exceptions, while
    suppressing repetitive warnings/progress output. Set EEG_VERBOSE=1 to
    restore informational logs during debugging.
    """

    verbose = os.getenv("EEG_VERBOSE", "0") == "1"
    logging.basicConfig(
        level=logging.INFO if verbose else logging.ERROR,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        force=True,
    )
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"mne|xgboost|sklearn")

    try:
        import mne

        mne.set_log_level("INFO" if verbose else "ERROR")
    except Exception:
        # Optional at import time; dependency errors are raised when the
        # preprocessing path is actually used.
        pass


configure_quiet_runtime()
