"""Data-layer robustness: retry/backoff and FRED CSV parsing (no network)."""
import pandas as pd
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


def test_load_fred_per_series_cache_degrades_gracefully(monkeypatch, tmp_path):
    """Per-series caching: an outage (or a newly added series) must not drop the
    series that already have caches."""
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data.time, "sleep", lambda *_: None)
    idx = pd.bdate_range("2020-01-01", periods=8)

    def ok(code, start, end, timeout=60):
        return pd.Series(range(8), index=idx, dtype=float)

    monkeypatch.setattr(data, "_fetch_fred_series", ok)
    df = data.load_fred({"a": "AAA", "b": "BBB"}, "2020-01-01", "2020-01-12")
    assert set(df.columns) == {"a", "b"}
    assert (tmp_path / "fred_series_AAA.parquet").exists()

    # Network down + a brand-new series with no cache: a, b survive from their
    # stale per-series caches; c is dropped, never orphaning the others.
    def boom(*a, **k):
        raise TimeoutError("down")

    monkeypatch.setattr(data, "_fetch_fred_series", boom)
    df2 = data.load_fred({"a": "AAA", "b": "BBB", "c": "CCC"}, "2020-01-01", None)
    assert set(df2.columns) == {"a", "b"}


def test_load_fred_recovers_legacy_combined_cache(monkeypatch, tmp_path):
    """A pre-upgrade combined cache (columns by label) is recovered when FRED is
    down, so switching to per-series caching never loses existing data."""
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data.time, "sleep", lambda *_: None)
    idx = pd.bdate_range("2020-01-01", periods=5)
    pd.DataFrame({"a": range(5), "b": range(5)}, index=idx).to_parquet(
        tmp_path / "fred_AAA_BBB.parquet")

    monkeypatch.setattr(data, "_fetch_fred_series",
                        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("x")))
    df = data.load_fred({"a": "AAA", "b": "BBB"}, "2020-01-01", None)
    assert set(df.columns) == {"a", "b"}
