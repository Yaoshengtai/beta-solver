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
            *[f"ex{i}" for i in range(1, 7)], "limits", "appendix"]


@pytest.fixture(scope="module")
def html(tmp_path_factory):
    out = tmp_path_factory.mktemp("site") / "index.html"
    return build_page.build(out)


@pytest.fixture(scope="module")
def markup(html):
    """The page without its <script> blocks: a slot only counts if the markup has it."""
    return re.sub(r"<script\b.*?</script>", "", html, flags=re.S)


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


def test_legend_embedded_and_slots_present(html, markup):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    leg = json.loads(raw)["figures"]["legend"]
    assert len(leg["hold_types"]) == 8
    for slot in ("hold-types", "climber", "climber-key"):
        assert f'data-draw="{slot}"' in markup
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
    assert {v["key"]: v["base"] for v in data["model"]["variants"]} == {"A_stiff": "A", "A_cross0": "A", "A_cross4": "A", "C_grip": "C"}
    assert 'data-draw="climber-cards"' in html


def test_body_data_embedded(html, markup):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    data = json.loads(raw)
    assert data["model"]["body"]["leg"] == 0.5
    assert len(data["figures"]["bodies"]) == 4
    for slot in ("length-table", "to-scale"):
        assert f'data-draw="{slot}"' in markup
    assert 'data-text="anchors-text"' in markup and 'data-math="limb-lengths"' in markup


def test_stance_overview_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    data = json.loads(raw)
    assert data["stance"]["sizes"]["four"] == 17
    assert len(data["stance"]["constraints"]) == 6


def test_reach_figure_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    R = json.loads(raw)["figures"]["reach"]
    assert len(R["region"]) > 10 and set(R["presets"]) == {"good", "low"}
    assert 'data-draw="reach"' in html


def test_check_sections(markup):
    # chapter 2, slimmed: reach and flexibility together, one balance section
    ids = re.findall(r'<h3 id="(check-[a-z]+)"', markup)
    assert ids == ["check-unknowns", "check-reach", "check-direction", "check-strength", "check-balance",
                   "check-problem", "check-solve"]
    for slot in ('data-draw="reach"', 'data-math="flex-row"', 'data-draw="flex-table"', 'data-draw="balance"',
                 'data-text="tip-facts"', 'data-text="door-facts"', 'data-math="full-problem"', 'data-draw="result-cards"'):
        assert slot in markup


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
    assert "Planned · unit 2.4" not in html


def test_tipping_figure_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    T = json.loads(raw)["figures"]["tipping"]
    assert len(T["views"]) == 2 and len(T["sweep"]) >= 10 and T["max_error"] < 0.01
    assert "Planned · unit 2.5" not in html


def test_barndoor_figure_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    D = json.loads(raw)["figures"]["barndoor"]
    assert [p["ok"] for p in D["panels"]] == [True, False]
    assert "Planned · unit 2.6" not in html


def test_flex_figure_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    X = json.loads(raw)["figures"]["flex"]
    assert [p["ok"] for p in X["panels"]] == [True, False]


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



def test_convex_figure_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    X = json.loads(raw)["figures"]["convex"]
    assert len(X["points"]) == 63 and all(p["ok"] for p in X["points"])
    assert max(p["t_min"] for p in X["points"]) <= X["difficulty"] + 2e-3
    assert X["sweep"]["all_feasible"] and X["sweep"]["moves"] == 100
    assert "Planned · unit 3.2" not in html



def test_twochecks_figure_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    T = json.loads(raw)["figures"]["twochecks"]
    works, fails = T["panels"]
    assert works["ok"] and works["x1"] and works["x2"] and len(works["S3"]) == 72
    assert not fails["ok"] and fails["x1"] is None and fails["x2"]
    assert 'data-draw="twochecks"' in html and "Planned · unit 3.3" not in html



def test_checker_figure_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    K = json.loads(raw)["figures"]["checker"]
    moves = [m for p in K["presets"] for m in p["moves"]]
    assert {m["status"] for m in moves} == {"ok", "fail", "pruned"}
    assert any(m.get("cost") and m["cost"]["cross"] > 0 for m in moves)   # the crossing penalty shows up
    assert any(m.get("cost") and m["cost"]["cut"] > 0 for m in moves)     # and the cut penalty
    assert 'data-draw="checker"' in html and "Planned · unit 3.4" not in html
    assert "Planned · unit 3." not in html                                 # chapter 3 is complete



def test_graph_figure_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    G = json.loads(raw)["figures"]["graph"]
    assert G["stats"]["states"]["reachable"] <= G["stats"]["states"]["held"]
    assert 'data-draw="graph-funnel"' in html and "Planned · unit 4.1" not in html



def test_dijkstra_figure_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    D = json.loads(raw)["figures"]["dijkstra"]
    assert D["status"] == "ok" and D["steps"] and sum(1 for s in D["steps"] if s["crux"]) == 1
    assert all(a <= b + 1e-9 for a, b in zip(D["stage1"], D["stage1"][1:]))
    assert 'data-draw="dijkstra-charts"' in html and "Planned · unit 4.2" not in html



def test_player_embedded(html):
    raw = re.search(r'<script type="application/json" id="site-data">(.*?)</script>', html, re.S).group(1)
    X = json.loads(raw)["figures"]["player"]
    assert [r["climber"] for r in X["runs"]] == ["A", "B", "C"]
    assert all(c["valid"] for c in X["smoothing"])
    for r in X["runs"]:
        assert r["status"] == "ok" and all(len(s["frames"]) == 4 for s in r["steps"])
    assert 'data-draw="player"' in html and "Planned · unit 4." not in html   # chapter 4 is complete



def test_move_sections(markup):
    # chapter 3, slimmed
    assert re.findall(r'<h3 id="(move-[a-z]+)"', markup) == ["move-phases", "move-why", "move-cost"]
    for slot in ('data-draw="move"', 'data-draw="twochecks"', 'data-text="convex-facts"', 'data-draw="checker"'):
        assert slot in markup
