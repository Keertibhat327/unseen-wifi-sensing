"""TODO: bind a real firmware header and packet-layout map before enabling serial."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from detection.adapters.base import CSIAdapter


class EspCSIAdapter(CSIAdapter):
    def convert(self, raw_frame: Any) -> NDArray[np.float64]:
        """Expected: CSV metadata plus a quoted JSON array, varying with firmware.

        TODO: obtain actual header + lines and firmware/IDF version. Parse with csv,
        check declared length, first_word_invalid, channel bandwidth and packet type;
        map LLTF/HT-LTF/STBC-HT-LTF ranges without guessing offsets. The shared decode
        helper then applies config. Do not invent a working parser from a sample.
        """
        raise NotImplementedError(
            "TODO: ESP-CSI serial requires a verified real header and packet-layout map"
        )
