from datetime import datetime

import numpy as np
import pytest
from test_ml_data import small_config

from contracts.events import Event
from identity.store import CONSENT_TEXT, IdentityStore


def test_disabled_consent_delete_and_unknown(tmp_path):
    path = tmp_path / "local.db"
    with pytest.raises(PermissionError):
        IdentityStore(path)
    assert not path.exists()
    config = small_config()
    windows = np.random.default_rng(5).normal(10, 0.1, (8, 8, 3))
    with IdentityStore(path, enabled=True) as store:
        with pytest.raises(PermissionError):
            store.enroll("person", windows, config, consent="yes")
        assert store.list_subjects() == []
        store.enroll("person", windows, config, consent=CONSENT_TEXT)
        assert (
            datetime.fromisoformat(store.list_subjects()[0]["consented_at"])
            .utcoffset()
            .total_seconds()
            == 0
        )
        assert store.match(windows[0], config, 8)["status"] == "candidate"
        assert store.match(windows[0] + 1000, config, 8)["status"] == "unknown"
        with pytest.raises(ValueError, match="rate"):
            store.match(windows[0], config, 9)
        changed = config.model_copy(update={"subcarrier_indices": [2, 1, 0]})
        with pytest.raises(ValueError, match="compatibility"):
            store.match(windows[0], changed, 8)
    with IdentityStore(path, enabled=True) as store:
        assert store.delete("person")
        assert not store.delete("person")
        assert store.list_subjects() == []
        assert store.match(windows[0], config, 8)["subject"] is None
    assert CONSENT_TEXT.encode() not in path.read_bytes()


def test_cloud_rejects_nested_identity():
    with pytest.raises(ValueError, match="local-only"):
        Event.json_payload({"nested": [{"Identity_Prediction": {"subject": "alias"}}]})
    assert Event.json_payload({"activity_prediction": {"label": "fall"}})
