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
        "figures": {
            "legend": legend.legend(),
        },
    }


def embed_json(data: dict) -> str:
    # "</" would close the <script> element early.
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render(template: str, data: dict) -> str:
    if template.count("__DATA__") != 1:
        raise ValueError("template must contain exactly one __DATA__ placeholder")
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
