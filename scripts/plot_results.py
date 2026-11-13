"""Learning curves and final-return tables from the returns CSVs in results/.

  uv run --with matplotlib python -m scripts.plot_results

Inputs (written by src/train.py: one row per finished episode, "step,return", where
step = agent steps summed over all envs):
  results/walker_walk_v1/returns_seed{0,1,2}.csv          H = 15, 500k agent steps
  results/horizon_sweep/returns_walker_h{5,15,30}_seed{0,1,2}.csv   300k agent steps

Outputs: results/walker_walk.png, results/horizon_ablation.png, and the Markdown
tables printed to stdout.

Conventions:
  - x axis is env frames = step * action_repeat (2 for walker).
  - all envs finish their episodes on the same step (lockstep), so each step has one
    batch of num_envs episodes; a seed's curve is the batch mean at each step.
  - curves show the mean over seeds, with the band spanning the min and max seed.
  - "final" return = mean of a seed's last FINAL_EPISODES episodes.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402

CFG = load_config(ROOT / "configs" / "dmc_walker_walk.yaml")
ACTION_REPEAT = CFG.env.action_repeat
NUM_ENVS = CFG.env.num_envs
FINAL_EPISODES = 3 * NUM_ENVS          # the last 3 batches of episodes = 48 per seed

# Validated categorical slots 1-3 (dataviz reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def load_returns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    return data[:, 0], data[:, 1]


PAPER_BUDGET_FRAMES = 500_000          # DreamerV3's Proprio Control budget: 500K environment steps


def final_up_to(path: Path, max_frames: int) -> float:
    """Mean of the last FINAL_EPISODES episodes that finished within max_frames env frames."""
    steps, rets = load_returns(path)
    keep = steps * ACTION_REPEAT <= max_frames
    return rets[keep][-FINAL_EPISODES:].mean()


def batch_curve(steps: np.ndarray, rets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean return of each batch of episodes that finished on the same step."""
    xs = np.unique(steps)
    return xs, np.array([rets[steps == x].mean() for x in xs])


def load_group(paths: list[Path]):
    """-> frames (n,), per-seed batch means (seeds, n), per-seed final returns (seeds,)."""
    curves, finals, xs_ref = [], [], None
    for p in paths:
        steps, rets = load_returns(p)
        xs, ys = batch_curve(steps, rets)
        if xs_ref is None:
            xs_ref = xs
        n = min(len(xs_ref), len(xs))          # align seeds on their common steps
        xs_ref, curves = xs_ref[:n], [c[:n] for c in curves] + [ys[:n]]
        finals.append(rets[-FINAL_EPISODES:].mean())
    return xs_ref * ACTION_REPEAT, np.stack(curves), np.array(finals)


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel("environment frames", color=INK_2)
    ax.set_ylabel("episode return", color=INK_2)
    ax.set_ylim(0, 1000)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}k"))


def plot(groups: list[tuple[str, np.ndarray, np.ndarray]], title: str, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax, title)
    for (label, frames, curves), color in zip(groups, SERIES):
        mean = curves.mean(0)
        ax.fill_between(frames, curves.min(0), curves.max(0), color=color, alpha=0.15, linewidth=0)
        ax.plot(frames, mean, color=color, linewidth=2, solid_capstyle="round", label=label)
        if len(groups) > 1:                    # direct label at the line end, in text ink
            ax.annotate(label, (frames[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=9)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    ax.margins(x=0.08 if len(groups) > 1 else 0.02)
    fig.text(0.99, 0.01, "line: mean over 3 seeds · band: min to max seed", ha="right",
             color=INK_2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def table(rows: list[tuple[str, np.ndarray]]) -> str:
    lines = ["| Run | Seed 0 | Seed 1 | Seed 2 | Mean ± std (over seeds) |", "|---|---|---|---|---|"]
    for name, finals in rows:
        cells = " | ".join(f"{v:.1f}" for v in finals)
        lines.append(f"| {name} | {cells} | {finals.mean():.1f} ± {finals.std(ddof=0):.1f} |")
    return "\n".join(lines)


def main():
    res = ROOT / "results"

    base = sorted((res / "walker_walk_v1").glob("returns_seed*.csv"))
    frames, curves, finals = load_group(base)
    plot([("H = 15", frames, curves)], "walker-walk (proprio), 3 seeds", res / "walker_walk.png")
    print(f"walker_walk_v1: {len(base)} seeds, last frame {int(frames[-1]):,}, "
          f"final = mean of last {FINAL_EPISODES} episodes")
    at_budget = np.array([final_up_to(p, PAPER_BUDGET_FRAMES) for p in base])
    print(table([("walker-walk, H = 15, 1M frames", finals),
                 (f"walker-walk, H = 15, at {PAPER_BUDGET_FRAMES // 1000}k frames", at_budget)]), "\n")

    groups, rows = [], []
    for h in (5, 15, 30):
        paths = sorted((res / "horizon_sweep").glob(f"returns_walker_h{h}_seed*.csv"))
        if len(paths) < 3:
            print(f"H = {h}: only {len(paths)} seed files found")
        if not paths:
            continue
        frames, curves, finals = load_group(paths)
        groups.append((f"H = {h}", frames, curves))
        rows.append((f"H = {h}", finals))
        print(f"H = {h}: {len(paths)} seeds, last frame {int(frames[-1]):,}")
    plot(groups, "Imagination horizon, walker-walk, 3 seeds each", res / "horizon_ablation.png")
    print(table(rows))


if __name__ == "__main__":
    main()
