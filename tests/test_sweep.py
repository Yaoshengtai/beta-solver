"""The height sweep (step 6): the sweep climber and the threshold bisection."""

import pytest

from betasolver import sweep as S
from betasolver.model import AVA, CAL


def test_grid_is_19_heights_from_150_to_195():
    assert len(S.GRID) == 19 and S.GRID[0] == 1.50 and S.GRID[-1] == 1.95


def test_sweep_climber_is_halfway_between_cal_and_ava():
    c = S.climber_at(1.72)
    assert c.H == c.span == 1.72 and c.key == "S_1720" and c.name == "1.72 m"
    for g in c.F:
        assert c.F[g] == pytest.approx((AVA.F[g] + CAL.F[g]) / 2)
    assert c.flex == pytest.approx((AVA.flex + CAL.flex) / 2)


def _fake(monkeypatch, ok):
    monkeypatch.setattr(S, "_row", lambda route, H: {"H": H, "ok": ok(H)})


def test_one_switch_is_bisected_to_tolerance(monkeypatch):
    _fake(monkeypatch, lambda H: H >= 1.6612)
    rows = [S._row(None, H) for H in S.GRID]
    sw = S.thresholds(None, rows)
    assert len(sw) == 1 and sw[0]["direction"] == "up"
    assert sw[0]["lo"] < 1.6612 <= sw[0]["hi"] and sw[0]["hi"] - sw[0]["lo"] <= S.TOL


def test_every_switch_is_reported(monkeypatch):
    _fake(monkeypatch, lambda H: 1.60 <= H < 1.80)
    rows = [S._row(None, H) for H in S.GRID]
    sw = S.thresholds(None, rows)
    assert [s["direction"] for s in sw] == ["up", "down"]
    assert sw[0]["H"] == pytest.approx(1.60, abs=S.TOL) and sw[1]["H"] == pytest.approx(1.80, abs=S.TOL)
