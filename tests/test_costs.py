"""Transaction-cost model behaviour."""
import numpy as np
import pandas as pd

from src.config import RunConfig
from src.costs import asset_cost_bps, turnover_cost


def test_asset_cost_lookup_order():
    cfg = RunConfig()
    cfg.cost_model = {"GLD": 2.5}
    # Override in cfg.cost_model wins.
    assert asset_cost_bps("GLD", cfg) == 2.5
    # Module default table is used for known tickers absent from the override.
    assert asset_cost_bps("CPER", cfg) == 12.0
    # Unknown ticker falls back to the scalar cost_bps.
    assert asset_cost_bps("ZZZZ", cfg) == cfg.cost_bps
    assert asset_cost_bps(None, cfg) == cfg.cost_bps


def test_zero_turnover_zero_cost():
    cfg = RunConfig()
    turnover = pd.Series([0.0, 0.0, 0.0])
    assert (turnover_cost(turnover, "GLD", cfg) == 0.0).all()


def test_cost_monotonic_in_turnover():
    cfg = RunConfig()
    small = turnover_cost(pd.Series([0.1]), "GLD", cfg).iloc[0]
    large = turnover_cost(pd.Series([0.5]), "GLD", cfg).iloc[0]
    assert large > small


def test_full_flip_costs_two_round_trips():
    cfg = RunConfig()
    # Turnover of 2.0 (a +1 -> -1 flip) costs 2 * round_trip_bps.
    rt = asset_cost_bps("GLD", cfg) / 1e4
    cost = turnover_cost(pd.Series([2.0]), "GLD", cfg).iloc[0]
    assert np.isclose(cost, 2.0 * rt)


def test_illiquid_costs_more_than_liquid():
    cfg = RunConfig()
    t = pd.Series([0.3])
    assert turnover_cost(t, "UNG", cfg).iloc[0] > turnover_cost(t, "SPY", cfg).iloc[0]


def test_impact_adds_cost_when_enabled():
    cfg = RunConfig()
    cfg.impact_coef = 1.0
    turnover = pd.Series([0.5])
    adv = pd.Series([0.5])  # 100% participation
    base = turnover_cost(turnover, "GLD", RunConfig()).iloc[0]
    with_impact = turnover_cost(turnover, "GLD", cfg, adv_proxy=adv).iloc[0]
    assert with_impact > base
