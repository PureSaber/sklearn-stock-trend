# sklearn-stock-trend — TaskSolver workflow

## Start

Run the TaskSolver workflow for `goals/sklearn-complete.yaml`.

Ensure `state/active-goal.json` points to that goal file (see `state/active-goal.example.json`).

TaskSolver install: `C:\tools\React_TaskSolver` (Stop hook in `.cursor/hooks.json`).

## Rules

- Work on one pending task whose `depends_on` are all complete.
- Do not declare tasks complete yourself; Stop hook verifies `done_when`.
- Before finishing each turn: `ruff check src tests` and `pytest -q`.
- Do not commit `.env`, `outputs/*` artifacts, or `state/active-goal.json`.

## CLI (after T02)

```bash
st-train --config configs/default.yaml
st-evaluate --config configs/default.yaml
st-backtest --config configs/default.yaml
st-walkforward --config configs/default.yaml
st-multisymbol --config configs/multisymbol.yaml --symbols configs/symbols.yaml
st-experiment --configs-dir configs/experiments
```

## Recovery

From TaskSolver directory:

```bash
cd C:\tools\React_TaskSolver
npm run task:status
npm run task:resume
```
