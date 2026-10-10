"""UTC boundary semantics and offline schedule expiry."""

from datetime import datetime

import httpx
import pytest

from contracts.zones import Zone
from gateway.zones import ZoneFeed, parse_zones


def test_overnight_utc_schedule_and_absolute_interval():
    zone = Zone.model_validate(
        {
            "zone_id": "night",
            "room_id": "studio",
            "armed": True,
            "weekly": [{"weekdays": [4], "start": "22:00", "end": "06:00"}],
        }
    )
    assert zone.active_at(datetime.fromisoformat("2026-10-09T22:00:00Z"))
    assert zone.active_at(datetime.fromisoformat("2026-10-10T05:59:59Z"))
    assert not zone.active_at(datetime.fromisoformat("2026-10-10T06:00:00Z"))
    assert not zone.active_at(datetime.fromisoformat("2026-10-10T22:00:00Z"))
    assert zone.active_at(datetime.fromisoformat("2026-10-10T04:00:00+05:30"))
    bounded = zone.model_copy(update={"ends_at": datetime.fromisoformat("2026-10-10T00:00:00Z")})
    assert not bounded.active_at(datetime.fromisoformat("2026-10-10T01:00:00Z"))


def test_same_day_schedule_and_bad_weekdays():
    body = {
        "zone_id": "work",
        "room_id": "studio",
        "armed": True,
        "weekly": [{"weekdays": [0], "start": "09:00", "end": "17:00"}],
    }
    zone = Zone.model_validate(body)
    assert zone.active_at(datetime.fromisoformat("2026-10-12T09:00:00Z"))
    assert not zone.active_at(datetime.fromisoformat("2026-10-12T17:00:00Z"))
    body["weekly"][0]["weekdays"] = [7]
    with pytest.raises(ValueError):
        Zone.model_validate(body)
    with pytest.raises(ValueError, match="unique"):
        parse_zones([zone.model_dump(), zone.model_dump()])


def test_remote_sync_expiry_and_disarm():
    data = [{"zone_id": "entry", "room_id": "studio", "armed": True}]
    online = True

    def handle(_):
        return httpx.Response(200, json=data) if online else httpx.Response(503)

    feed = ZoneFeed(
        api_url="https://example.test", read_key="read", refresh_seconds=5, max_age_seconds=10
    )
    assert feed.snapshot(now=0) == ([], "unavailable")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        feed.refresh(client, now=0)
        assert feed.snapshot(now=2)[0][0].armed
        online = False
        feed.refresh(client, now=5)
        assert feed.snapshot(now=6)[1] == "synced"
        assert feed.snapshot(now=11) == ([], "expired")
        online = True
        data[0]["armed"] = False
        feed.refresh(client, now=12)
        assert feed.snapshot(now=12)[0][0].armed is False


def test_local_schedule_reload(tmp_path):
    path = tmp_path / "zones.json"
    path.write_text('[{"zone_id":"z","room_id":"studio","armed":true}]')
    feed = ZoneFeed(path=path)
    assert feed.snapshot()[0][0].armed
    path.write_text("[]")
    assert feed.snapshot() == ([], "local")
