"""Fetch the UCI zip into data/ and cache the tidy table (not committed). Same as `make data`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main(["data"])
