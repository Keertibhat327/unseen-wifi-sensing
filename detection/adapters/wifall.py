"""Verified against RS2002/WiFall.zip CSV files, not an inferred tensor format."""

import csv
import json
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from detection.adapters.base import CSIAdapter


class WiFallAdapter(CSIAdapter):
    def convert(self, raw_frame: Mapping[str, str]) -> NDArray[np.float64]:
        """WiFall stores a JSON signed-byte array in the CSV `data` field."""
        data = json.loads(raw_frame["data"])
        if not isinstance(data, list) or int(raw_frame["len"]) != len(data):
            raise ValueError("WiFall len does not match data")
        flag = raw_frame["first_word_invalid"]
        if flag not in ("0", "1"):
            raise ValueError("unknown first_word_invalid value")
        return self.decode(data, first_word_invalid=flag == "1")

    @staticmethod
    def timestamp(raw_frame: Mapping[str, str]) -> datetime:
        """The dataset card specifies UTC+8 for naive collection timestamps."""
        ts = datetime.fromisoformat(raw_frame["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone(timedelta(hours=8)))
        return ts.astimezone(UTC)

    def read(self, path: Path) -> Iterator[tuple[datetime, NDArray[np.float64]]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                yield self.timestamp(row), self.convert(row)
