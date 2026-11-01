# Beta Solver

Finding a climbing beta with optimization.

A climber's position on a wall is modeled as a *state* (which hold each hand and foot is on). Each state gets a small second-order cone program — the **state check**, solved with Clarabel — that asks whether the climber can hold that position (reach, hold direction, strength, in-plane and out-of-wall balance, flexibility) and how hard the most loaded limb is working. A two-stage Dijkstra search then finds the beta whose hardest move (the crux) is as easy as possible, and among those the cheapest overall.

All solving happens offline in Python; the website only reads the exported results.

**Website:** https://yaoshengtai.github.io/beta-solver/

## Layout

```
src/betasolver/     model, body, stance, moves, search, sweep, routes, export
tests/              pytest tests
site/               template.html, build_page.py
docs/index.html     built site, published by GitHub Pages
scripts/            first_push.sh, sync.sh
.github/workflows/  tests.yml (run tests), pages.yml (deploy site)
```

## Running

```bash
pip install -e ".[dev,exact]"
python -m betasolver.run_all             # solve every example -> site/results.json
pytest --report site/test_report.json    # run the tests and write the report shown on the site
python site/build_page.py                # build the site -> docs/index.html
```

The solver is deterministic: `run_all` reproduces the committed `results.json` apart from timings and the date. Plain `pytest` runs the tests without writing a report.

The `exact` extra (cvxpy) is only for the cross-check: the tests and the site build compare the hand-built cone program against the same problem written in cvxpy.

## Status

Under construction, one unit at a time. Each accepted unit is tagged `unit-X.Y`.
