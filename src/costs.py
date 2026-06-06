"""Transaction-cost model — the single source of truth for trading costs.

Replaces the old flat ``cfg.cost_bps`` that was duplicated across backtest.py,
rebalance.py and the diagnose scripts (which silently diverged). Costs are
expressed as round-trip basis points per unit of turnover, where turnover is the
absolute day-to-day change in the position (a position held in [-max_leverage,
max_leverage] capital units). A full flip from +1 to -1 is turnover 2.0 and
therefore costs ``2 * round_trip_bps``.

Per-asset costs matter: SPY/IEF trade for ~1-2bps round-trip while thin or
contango-prone ETFs (CPER, UNG, FXY) realistically cost 8-15bps. Using one flat
number flatters the illiquid names and is a classic backtest-vs-live gap.
"""
from __future__ import annotations

import pandas as pd

from .config import RunConfig

# Fallback per-ticker round-trip cost (bps) used when a RunConfig does not list
# the ticker in its own ``cost_model``. Order-of-magnitude estimates of
# half-spread + baseline impact for liquid ETF trading; tune per broker.
_DEFAULT_COST_BPS: dict[str, float] = {
    "SPY": 1.0, "IEF": 2.0, "TLT": 2.0, "GLD": 2.0, "HYG": 3.0,
    "EFA": 3.0, "EEM": 4.0, "SLV": 4.0, "UUP": 4.0, "VNQ": 4.0,
    "GDX": 4.0, "FXE": 5.0, "USO": 6.0, "DBC": 6.0, "GDXJ": 6.0,
    "FXY": 8.0, "CPER": 12.0, "UNG": 12.0,
}


def asset_cost_bps(ticker: str | None, cfg: RunConfig) -> float:
    """Round-trip cost in bps for one ticker.

    Resolution order: the RunConfig's ``cost_model`` override, then the module
    default table, then the scalar ``cfg.cost_bps`` fallback for anything unknown
    (e.g. macro-only tickers that are never traded).
    """
    table = getattr(cfg, "cost_model", None) or {}
    if ticker is not None and ticker in table:
        return float(table[ticker])
    if ticker is not None and ticker in _DEFAULT_COST_BPS:
        return float(_DEFAULT_COST_BPS[ticker])
    return float(getattr(cfg, "cost_bps", 5.0))


def turnover_cost(turnover: pd.Series, ticker: str | None, cfg: RunConfig,
                  adv_proxy: pd.Series | None = None) -> pd.Series:
    """Cost series (in return units) for a turnover series.

    Base cost is ``turnover * round_trip_bps / 1e4``. When an ADV proxy (e.g. the
    asset's own dollar volume) and a non-zero ``cfg.impact_coef`` are supplied, a
    linear market-impact term ``impact_coef * turnover / ADV`` is added so that
    trading a larger fraction of daily volume costs more. Impact is off by
    default (impact_coef = 0) because it needs a capital assumption to calibrate.
    """
    base_bps = asset_cost_bps(ticker, cfg)
    cost = turnover.abs() * (base_bps / 1e4)

    impact_coef = float(getattr(cfg, "impact_coef", 0.0) or 0.0)
    if adv_proxy is not None and impact_coef > 0:
        adv = adv_proxy.reindex(turnover.index).replace(0, pd.NA)
        participation = (turnover.abs() / adv).astype(float).fillna(0.0)
        cost = cost + turnover.abs() * impact_coef * participation
    return cost
