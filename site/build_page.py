"""Build the single-file website from site/template.html.

Everything the page shows that comes from the solver (results, figures, test
report) is embedded as JSON in <script id="site-data">; the page computes nothing.

Usage:
    python site/build_page.py                 # writes docs/index.html
    python site/build_page.py --preview PATH  # also writes the claude.ai preview
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "site" / "template.html"
OUT = ROOT / "docs" / "index.html"
REPO_URL = "https://github.com/Yaoshengtai/beta-solver"

sys.path.insert(0, str(ROOT / "src"))
import betasolver  # noqa: E402
from betasolver import body, model, stance  # noqa: E402

sys.path.insert(0, str(ROOT / "site"))
import legend  # noqa: E402

# CJK ideographs, CJK punctuation and full-width forms. The site is English only.
CJK = re.compile(r"[　-〿㐀-䶿一-鿿豈-﫿＀-￯]")


def collect_data() -> dict:
    return {
        "meta": {
            "version": betasolver.__version__,
            "built": dt.date.today().isoformat(),
            "repo": REPO_URL,
        },
        "model": {
            "wall": {"w": model.WALL_W, "h": model.WALL_H},
            "hold_types": model.hold_types_table(),
            "foot_on_handhold": model.FOOT_ON_HANDHOLD,
            "climbers": [c.to_dict() for c in model.PROFILES.values()],
            "variants": [{**c.to_dict(), "base": c.key.split("_")[0]} for c in model.VARIANTS.values()],
            "body": {
                "shoulder": list(body.SHOULDER), "hip": list(body.HIP), "head": list(body.HEAD),
                "leg": body.LEG, "grip_inset": body.GRIP_INSET,
                "shoulder_width_in_span": body.SHOULDER_WIDTH_IN_SPAN,
                "upper_arm": body.UPPER_ARM, "thigh": body.THIGH, "d_wall": model.D_WALL,
            },
        },
        "stance": {
            "unknowns": stance.UNKNOWNS,
            "constraints": stance.CONSTRAINTS,
            "reasons": stance.REASONS,
            "effort_bands": stance.EFFORT_BANDS,
            "sizes": {
                "four": stance.layout(("LH", "RH", "LF", "RF")).size,
                "one_foot": stance.layout(("LH", "RH", "LF")).size,
                "no_feet": stance.layout(("LH", "RH")).size,
            },
        },
        "results": load_results(),
        "figures": {
            "legend": legend.legend(),
            "route_example": legend.route_example(),
            "bodies": legend.body_figures(),
            "reach": legend.reach_figure(),
            "strength": legend.strength_figure(),
            "balance": legend.balance_figure(),
            "tipping": legend.tipping_figure(),
            "barndoor": legend.barndoor_figure(),
            "flex": legend.flex_figure(),
            "cards": legend.result_cards(),
            "solver": legend.solver_figure(),
            "move": legend.move_figure(),
            "convex": legend.convex_figure(),
            "twochecks": legend.twochecks_figure(),
            "checker": legend.checker_figure(),
            "graph": legend.graph_figure(),
            "dijkstra": legend.search_figure(),
            "player": legend.player_figure(),
        },
    }


RESULTS = ROOT / "site" / "results.json"
N_EXAMPLES = 6


def load_results() -> dict:
    """The solved examples, written by `python -m betasolver.run_all`; empty if not run yet."""
    if not RESULTS.exists():
        return {"meta": {}, "examples": []}
    return json.loads(RESULTS.read_text(encoding="utf-8"))


NAMES = {"LH": "left hand", "RH": "right hand", "LF": "left foot", "RF": "right foot"}


def fill(text: str, runs: dict, sweep: dict | None = None) -> str:
    """Replace {KEY.field} placeholders with numbers from the runs: crux, steps, total, label, fail;
    and {sweep.hstar}, the height threshold of the height sweep."""
    def one(m):
        key, f = m.group(1), m.group(2)
        if key == "sweep" and f == "hstar":
            return f"{sweep['H_star']:.2f} m"
        r = runs[key]
        if f == "crux":
            return f"{r['crux']:.2f}" if r["crux"] is not None else "none"
        if f == "steps":
            return str(len(r["steps"]))
        if f == "total":
            return f"{r['total']:.2f}"
        if f == "label":
            return r["label"]
        if f == "fail":
            x = r["fail"] or {}
            why = {"reach": "out of reach", "strength": "not strong enough", "balance": "direction or balance does not work",
                   "unusable": "a hold this climber cannot use"}.get(x.get("reason"), x.get("reason"))
            return (f"stuck at move {x.get('step')}, {NAMES.get(x.get('limb'), 'a limb')} to {x.get('hold')}, "
                    f"which fails the {x.get('phase')} check: {why}")
        raise KeyError(f)
    return re.sub(r"\{([A-Za-z0-9_]+)\.([a-z]+)\}", one, text)


def html_escape(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def examples_html(results: dict) -> str:
    """The example chapters: one per solved example, then placeholders up to N_EXAMPLES."""
    out = []
    exs = results.get("examples", [])
    for i in range(N_EXAMPLES):
        n = i + 1
        if i < len(exs):
            e = exs[i]
            runs = {r["key"]: r for r in e["runs"]}
            paras = "\n".join(f"  <p>{html_escape(fill(t, runs, e.get('sweep')))}</p>" for t in e["text"])
            sweep_fig = (f"""  <figure class="fig"><div data-example-sweep="{i}"></div>
    <figcaption><b>Crux against height.</b> One climber, 19 heights; every length of the body scales with the height. Click a ringed point to watch that height in the player below.</figcaption></figure>\n"""
                         if e.get("sweep") else "")
            out.append(f"""<section class="chapter" id="ex{n}" data-group="Examples" data-short="{n}. {html_escape(e['title'])}" hidden>
  <header><p class="eyebrow">Example {n}</p><h2>{html_escape(e['title'])}</h2>
  <p class="lede">{html_escape(e['lesson'])}</p></header>
  <div class="tbl-wrap"><table class="tbl" data-example-outcomes="{i}"></table></div>
{sweep_fig}  <figure class="fig"><div class="player" data-example-player="{i}"></div>
    <figcaption><b>Every climber's beta on this route.</b> Pick a climber, then play or step through the moves.</figcaption></figure>
{paras}
  <details class="impl"><summary>Acceptance checks</summary><ul data-example-checks="{i}"></ul></details>
</section>""")
        else:
            out.append(f"""<section class="chapter" id="ex{n}" data-group="Examples" data-short="{n}. Example" hidden>
  <header><p class="eyebrow">Example {n}</p><h2>Example {n}</h2></header>
  <div class="planned"><span class="chip">Planned · unit 5.{n}</span></div>
</section>""")
    return "\n".join(out)


def embed_json(data: dict) -> str:
    # "</" would close the <script> element early.
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render(template: str, data: dict) -> str:
    if template.count("__DATA__") != 1:
        raise ValueError("template must contain exactly one __DATA__ placeholder")
    if "__EXAMPLES__" in template:
        template = template.replace("__EXAMPLES__", examples_html(data.get("results", {})))
    return template.replace("__DATA__", embed_json(data))


def find_cjk(html: str) -> list[tuple[int, str]]:
    """Return (line number, line) for every line containing CJK characters."""
    return [(i, line.strip()) for i, line in enumerate(html.splitlines(), 1) if CJK.search(line)]


def to_preview(html: str) -> str:
    """Strip the document wrapper for the claude.ai preview, which adds its own.

    Keeps everything in <head> except the charset and viewport metas, then the
    body content.
    """
    head = re.search(r"<head>(.*?)</head>", html, re.S).group(1)
    body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
    head = re.sub(r'\s*<meta (charset|name="viewport")[^>]*>', "", head)
    return head.strip() + "\n" + body.strip() + "\n"


def build(out: Path = OUT, preview: Path | None = None) -> str:
    html = render(TEMPLATE.read_text(encoding="utf-8"), collect_data())
    bad = find_cjk(html)
    if bad:
        lines = "\n".join(f"  line {n}: {text[:80]}" for n, text in bad[:10])
        raise SystemExit(f"Chinese text found in the built page:\n{lines}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        preview.write_text(to_preview(html), encoding="utf-8")
    return html


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--preview", type=Path, default=None)
    args = ap.parse_args()
    html = build(args.out, args.preview)
    print(f"wrote {args.out} ({len(html.encode()) / 1024:.1f} KB)")
    if args.preview:
        print(f"wrote {args.preview}")


if __name__ == "__main__":
    main()
