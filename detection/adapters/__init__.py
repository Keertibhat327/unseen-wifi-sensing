"""All adapters return a two-dimensional time × subcarriers array."""

from detection.adapters.base import AdapterConfig, CSIAdapter
from detection.adapters.wifall import WiFallAdapter

__all__ = ["AdapterConfig", "CSIAdapter", "WiFallAdapter"]
