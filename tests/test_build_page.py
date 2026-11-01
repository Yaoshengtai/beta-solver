import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_page", ROOT / "site" / "build_page.py")
build_page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_page)

CHAPTERS = ["intro", "data", "body", "check", "move", "search",
            *[f"ex{i}" for i in range(1, 11)], "limits", "appendix"]


@pytest.fixture(scope="module")
def html(tmp_path_factory):
    out = tmp_path_factory.mktemp("site") / "index.html"
    return build_page.build(out)


def test_every_chapter_present_in_order(html):
    found = re.findall(r'<section class="chapter" id="([^"]+)"', html)
    assert found == CHAPTERS


def test_anchors_are_plain_tokens_and_unique(html):
    # The claude.ai preview only passes hashes made of [A-Za-z0-9._~-].
    ids = re.findall(r'\sid="([^"]+)"', html)
    assert len(ids) == len(set(ids))
    for i in ids:
        assert re.fullmatch(r"[A-Za-z0-9._~-]+", i), i


def test_embedded_data_parses(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    data = json.loads(raw)
    assert data["meta"]["repo"].startswith("https://github.com/")


def test_json_cannot_close_script():
    assert "</" not in build_page.embed_json({"x": "</script>"})


def test_no_chinese_in_page(html):
    assert build_page.find_cjk(html) == []


def test_cjk_detector_catches_chinese():
    assert build_page.find_cjk("ok\n体检\n") == [(2, "体检")]


def test_mathjax_pinned(html):
    assert "cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js" in html


def test_preview_has_no_document_wrapper(html):
    frag = build_page.to_preview(html)
    for tag in ("<!doctype", "<html", "<head>", "<body>", 'name="viewport"', "charset="):
        assert tag not in frag.lower()
    assert frag.lstrip().startswith("<title>")


def test_legend_embedded_and_slots_present(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    leg = json.loads(raw)["figures"]["legend"]
    assert len(leg["hold_types"]) == 8
    for slot in ("wall", "hold-types", "markings", "climber", "climber-key"):
        assert f'data-draw="{slot}"' in html
    assert "var Draw = (function ()" in html


def test_hold_types_and_route_example_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    data = json.loads(raw)
    assert [t["name"] for t in data["model"]["hold_types"]][:2] == ["jug", "crimp"]
    ex = data["figures"]["route_example"]
    assert ex["finish"] in {h["id"] for h in ex["holds"]}
    assert 'data-draw="hold-table"' in html and 'data-draw="route-example"' in html


def test_climbers_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    data = json.loads(raw)
    assert [c["name"] for c in data["model"]["climbers"]] == ["Ava", "Ben", "Cal", "Ava (injured)"]
    assert all(v["base"] == "A" for v in data["model"]["variants"])
    assert 'data-draw="climber-cards"' in html


def test_body_data_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    data = json.loads(raw)
    assert data["model"]["body"]["leg"] == 0.5
    assert len(data["figures"]["bodies"]) == 4
    for slot in ("anchor-table", "anchors", "length-table", "to-scale"):
        assert f'data-draw="{slot}"' in html


def test_stance_overview_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    data = json.loads(raw)
    assert data["stance"]["sizes"]["four"] == 17
    assert len(data["stance"]["constraints"]) == 6
    assert len(data["figures"]["state_example"]["poses"]) == 3


def test_reach_figure_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    R = json.loads(raw)["figures"]["reach"]
    assert len(R["region"]) > 10 and set(R["presets"]) == {"good", "low"}
    assert 'data-draw="reach"' in html


def test_direction_figure_slots(html):
    assert 'data-draw="direction"' in html and 'data-draw="direction-reset"' in html


def test_strength_figure_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    data = json.loads(raw)
    S = data["figures"]["strength"]
    assert [r["climber"] for r in S["runs"]] == ["A", "B", "C"]
    assert [b["key"] for b in data["stance"]["effort_bands"]] == ["easy", "hard", "limit", "fail"]
    for slot in ("strength", "strength-bars", "strength-switch", "effort-table"):
        assert f'data-draw="{slot}"' in html
    assert "Planned · unit 2.3" not in html


def test_balance_figure_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    B = json.loads(raw)["figures"]["balance"]
    assert [p["key"] for p in B["panels"]] == ["hang", "off", "sidepull"]
    assert 'data-draw="balance"' in html and 'data-text="balance-strength"' in html
    assert "Planned · unit 2.4" not in html


def test_tipping_figure_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    T = json.loads(raw)["figures"]["tipping"]
    assert len(T["views"]) == 2 and len(T["sweep"]) >= 10 and T["max_error"] < 0.01
    for slot in ('data-draw="tipping"', 'data-draw="tipping-chart"', 'data-text="tip-strength"'):
        assert slot in html
    assert "Planned · unit 2.5" not in html


def test_barndoor_figure_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    D = json.loads(raw)["figures"]["barndoor"]
    assert [p["ok"] for p in D["panels"]] == [True, False]
    assert 'data-draw="barndoor"' in html and "Planned · unit 2.6" not in html


def test_flex_figure_embedded(html):
    assert 'data-draw="flex"' in html and 'data-draw="flex-table"' in html and "Planned · unit 2.7" not in html


def test_full_problem_and_cards(html):
    assert 'data-math="full-problem"' in html and 'data-draw="result-cards"' in html and "Planned · unit 2.8" not in html



def test_solver_section_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    S = json.loads(raw)["figures"]["solver"]
    assert S["summary"]["n"] == 300 and S["summary"]["disagree"] == 0
    assert S["size"]["variables"] == 17 and S["size"]["cones"] == 8
    assert 'data-draw="solver-stats"' in html
    assert "Planned · unit 2." not in html  # chapter 2 is complete



def test_move_figure_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    M = json.loads(raw)["figures"]["move"]
    works, fails = M["moves"]
    assert works["ok"] and abs(works["difficulty"] - max(works["t_release"], works["t_catch"])) < 2e-3
    assert not fails["ok"] and not fails["release_ok"] and fails["catch_ok"]
    assert 'data-draw="move"' in html and "Planned · unit 3.1" not in html
