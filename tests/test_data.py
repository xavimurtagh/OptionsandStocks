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

    def ok(code, start, end, timeout=60, attempts=4):
        return pd.Series(range(8), index=idx, dtype=float)

    monkeypatch.setattr(data, "_fetch_fred_series", ok)
    df = data.load_fred({"a": "AAA", "b": "BBB"}, "2020-01-01", "2020-01-12")
    assert set(df.columns) == {"a", "b"}
    assert (tmp_path / "fred_series_AAA.parquet").exists()

    # A stale cache must fail fast (1 short attempt), not burn 4x60s, when FRED
    # is down - and still serve the stale copy.
    seen = {}

    def rec(code, start, end, timeout=60, attempts=4):
        seen["timeout"], seen["attempts"] = timeout, attempts
        raise TimeoutError("down")

    monkeypatch.setattr(data, "_fetch_fred_series", rec)
    old = pd.bdate_range("2016-01-01", periods=4)   # far older than horizon
    pd.DataFrame({"AAA": range(4)}, index=old).to_parquet(
        tmp_path / "fred_series_AAA.parquet")
    got = data.load_fred({"a": "AAA"}, "2016-01-01", None)
    assert set(got.columns) == {"a"}                # stale cache still served
    assert seen == {"timeout": 15, "attempts": 1}   # fast-fail, not 4x60s

    # Network down + a brand-new series with no cache: a, b survive from their
    # stale per-series caches; c is dropped, never orphaning the others.
    def boom(*a, **k):
        raise TimeoutError("down")

    monkeypatch.setattr(data, "_fetch_fred_series", boom)
    df2 = data.load_fred({"a": "AAA", "b": "BBB", "c": "CCC"}, "2020-01-01", None)
    assert set(df2.columns) == {"a", "b"}


def test_load_prices_refetches_when_cache_misses_earlier_start(monkeypatch, tmp_path):
    """Extending the start (e.g. back to 2005 for the GFC) must refetch, not
    silently return the shorter cached window."""
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data.time, "sleep", lambda *_: None)
    cache = tmp_path / "prices_SPY.parquet"
    short = pd.DataFrame({"SPY_close": [1.0, 2.0]},
                         index=pd.bdate_range("2010-01-01", periods=2))
    short.to_parquet(cache)

    calls = {"n": 0}

    def fake_dl(*a, **k):
        calls["n"] += 1
        idx = pd.bdate_range("2005-01-03", "2010-01-05")
        return pd.DataFrame({("SPY", "Open"): 1.0, ("SPY", "High"): 1.0,
                             ("SPY", "Low"): 1.0, ("SPY", "Close"): 1.0,
                             ("SPY", "Volume"): 1.0}, index=idx)

    monkeypatch.setattr(data.yf, "download", fake_dl)
    # Recent cache, but it starts in 2010 and we ask back to 2005 -> must refetch.
    out = data.load_prices(["SPY"], "2005-01-01", "2010-01-05")
    assert calls["n"] == 1
    assert out.index.min().year == 2005
    # A second call now fully covered by the cache does not refetch.
    data.load_prices(["SPY"], "2005-01-01", "2010-01-05")
    assert calls["n"] == 1


def test_load_yields_yf_rescales_and_labels(monkeypatch):
    """Yahoo yields are mapped to carry's labels and x10-quoted history is
    rescaled to percent."""
    idx = pd.bdate_range("2020-01-01", periods=6)
    px = pd.DataFrame({
        "^IRX_close": [1.5, 1.6, 1.5, 1.4, 1.5, 1.6],        # already percent
        "^TNX_close": [25.0, 24.0, 26.0, 25.5, 24.5, 25.0],  # legacy x10 -> ~2.5%
    }, index=idx)
    monkeypatch.setattr(data, "load_prices", lambda *a, **k: px)
    out = data.load_yields_yf(
        {"short_yield_3m": "^IRX", "nominal_yield_10y_yf": "^TNX"}, "2020-01-01")
    assert set(out.columns) == {"short_yield_3m", "nominal_yield_10y_yf"}
    assert out["nominal_yield_10y_yf"].median() < 5      # rescaled from ~25
    assert abs(out["short_yield_3m"].median() - 1.5) < 0.2


def test_load_yields_yf_empty_on_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("yf down")

    monkeypatch.setattr(data, "load_prices", boom)
    assert data.load_yields_yf({"short_yield_3m": "^IRX"}, "2020-01-01").empty
    assert data.load_yields_yf({}, "2020-01-01").empty


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
