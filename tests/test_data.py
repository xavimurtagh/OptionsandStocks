"""Data-layer robustness: retry/backoff and FRED CSV parsing (no network)."""
import pytest

import src.data as data


def test_retry_succeeds_after_transient_failures(monkeypatch):
    monkeypatch.setattr(data.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    assert data._retry(flaky, attempts=5) == "ok"
    assert calls["n"] == 3


def test_retry_raises_after_exhaustion(monkeypatch):
    monkeypatch.setattr(data.time, "sleep", lambda *_: None)

    def always_fail():
        raise TimeoutError("down")

    with pytest.raises(TimeoutError):
        data._retry(always_fail, attempts=3)


def test_fetch_fred_parses_and_drops_missing(monkeypatch):
    monkeypatch.setattr(data.time, "sleep", lambda *_: None)
    csv = "observation_date,DGS10\n2020-01-02,1.88\n2020-01-03,.\n2020-01-06,1.81\n"

    class _Resp:
        text = csv

        def raise_for_status(self):
            pass

    monkeypatch.setattr(data.requests, "get", lambda *a, **k: _Resp())
    s = data._fetch_fred_series("DGS10", "2020-01-01", None)
    # The "." missing marker is coerced to NaN and dropped.
    assert list(s.values) == [1.88, 1.81]
    assert str(s.index[0].date()) == "2020-01-02"
    assert s.index.tz is None


def test_fetch_fred_respects_start_end(monkeypatch):
    monkeypatch.setattr(data.time, "sleep", lambda *_: None)
    csv = ("observation_date,DGS10\n2020-01-02,1.0\n2020-02-02,2.0\n"
           "2020-03-02,3.0\n")

    class _Resp:
        text = csv

        def raise_for_status(self):
            pass

    monkeypatch.setattr(data.requests, "get", lambda *a, **k: _Resp())
    s = data._fetch_fred_series("DGS10", "2020-02-01", "2020-02-28")
    assert list(s.values) == [2.0]
