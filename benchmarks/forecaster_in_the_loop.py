"""Forecaster-in-the-loop study (Tier 3, item 10).

The paper's central selling point is interpretability *for action*: a forecaster
can inspect the historical promotions behind a prediction and edit their
contribution. That claim was never tested -- the paper measured only automatic
accuracy. This script turns the claim into a falsifiable experiment by
simulating a human forecaster with a tunable skill level and comparing three
interaction modes:

  * raw        -- the model's automatic neighbour-weighted forecast (no human);
  * accept/reject -- a black-box UI: the forecaster can only *veto* a forecast
                  and fall back to a blunt baseline (typical sales), but cannot
                  fix it. This is all an uninterpretable model affords;
  * edit       -- the neighbour UI: the forecaster can down-weight the
                  individual historical promotions they believe are misleading,
                  then the forecast is recomputed. This is what NextDoor affords.

Both human modes draw on the *same* latent knowledge -- whether each retrieved
neighbour belongs to the query's true (partially hidden) demand regime --
observed through a noisy channel whose fidelity is the skill ``s``. At s=0 the
human adds nothing; at s=1 they have an oracle's view of regime membership.

The experiment's thesis: the value of the editable neighbour interface is the
accuracy gap between the *edit* and *accept/reject* curves. If that gap is zero,
the interpretability story buys nothing operationally; if it is large and grows
with skill, the interface is doing real work.

Run with:  uv run python benchmarks/forecaster_in_the_loop.py [--seeds 20]
"""

from __future__ import annotations

import argparse

import numpy as np

from nextdoor import NextDoorForecaster


def generate(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Promotions with a partially hidden demand regime.

    Four observed features drive baseline sales. A binary latent regime adds a
    large level shift and is only *weakly* signalled by a noisy proxy feature,
    so a feature-space neighbour search will sometimes retrieve promotions from
    the wrong regime -- exactly the case a knowledgeable forecaster could catch.
    """
    x = rng.uniform(0, 1, size=(n, 4))
    regime = rng.binomial(1, 0.5, n)  # hidden high/low-demand regime
    # noisy observed proxy of the regime (the only feature that hints at it)
    proxy = 0.5 * regime + 0.5 * rng.uniform(0, 1, n)
    feats = np.column_stack([x, proxy])

    base = 200.0 + 600.0 * x[:, 0] + 300.0 * x[:, 1]
    y = base + 1500.0 * regime + rng.normal(0, 60.0, n)
    y = np.maximum(y, 1.0)
    return {"feats": feats, "regime": regime, "y": y}


def neighbour_belief(
    true_misleading: np.ndarray, skill: float, rng: np.random.Generator
) -> np.ndarray:
    """Forecaster's belief 'is this neighbour misleading?' through a noisy channel.

    Correct with probability ``skill``, flipped otherwise.
    """
    correct = rng.random(true_misleading.shape) < skill
    return np.where(correct, true_misleading, ~true_misleading)


def out50p(y: np.ndarray, y_hat: np.ndarray) -> float:
    rel = np.abs(y - y_hat) / np.abs(y)
    return float(100 * y[rel > 0.5].sum() / y.sum())


def run_once(n: int, skills: list[float], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = generate(n, rng)
    feats, regime, y = data["feats"], data["regime"], data["y"]

    i_tr, i_va = int(0.55 * n), int(0.75 * n)
    f = NextDoorForecaster(random_state=seed, k_neighbours=15)
    f.train_and_validate(feats[:i_tr], y[:i_tr], x_val=feats[i_tr:i_va], y_val=y[i_tr:i_va])
    regime_train = regime[:i_tr]  # aligned with the forecaster's internal order

    x_te, y_te, r_te = feats[i_va:], y[i_va:], regime[i_va:]
    fallback = float(np.median(y[:i_tr]))  # blunt "typical sales" baseline

    raw_pred = np.empty(len(y_te))
    edit_pred = {s: np.empty(len(y_te)) for s in skills}
    ar_pred = {s: np.empty(len(y_te)) for s in skills}

    for j in range(len(y_te)):
        expl = f.explain(x_te[j])
        idx, w, tgt = expl.indices, expl.weights.copy(), expl.targets
        nbr_regime = regime_train[idx]
        misleading = nbr_regime != r_te[j]  # truly from the wrong regime

        raw_pred[j] = w @ tgt

        for s in skills:
            believed_mis = neighbour_belief(misleading, s, rng)

            # edit: drop neighbours believed misleading, renormalise, recompute
            keep = ~believed_mis
            if keep.any() and w[keep].sum() > 0:
                we = w * keep
                edit_pred[s][j] = (we @ tgt) / we.sum()
            else:
                edit_pred[s][j] = raw_pred[j]

            # accept/reject: veto if the forecaster believes most of the
            # forecast's weight rests on misleading neighbours
            believed_ok_weight = w[~believed_mis].sum()
            if believed_ok_weight < 0.5:
                ar_pred[s][j] = fallback
            else:
                ar_pred[s][j] = raw_pred[j]

    result = {
        "raw_mae": np.full(len(skills), np.mean(np.abs(raw_pred - y_te))),
        "raw_out50": np.full(len(skills), out50p(y_te, raw_pred)),
        "edit_mae": np.array([np.mean(np.abs(edit_pred[s] - y_te)) for s in skills]),
        "edit_out50": np.array([out50p(y_te, edit_pred[s]) for s in skills]),
        "ar_mae": np.array([np.mean(np.abs(ar_pred[s] - y_te)) for s in skills]),
        "ar_out50": np.array([out50p(y_te, ar_pred[s]) for s in skills]),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n", type=int, default=600)
    args = parser.parse_args()

    skills = [0.0, 0.25, 0.5, 0.75, 1.0]
    agg: dict[str, list[np.ndarray]] = {}
    for seed in range(args.seeds):
        res = run_once(args.n, skills, seed)
        for k, v in res.items():
            agg.setdefault(k, []).append(v)
    mean = {k: np.mean(v, axis=0) for k, v in agg.items()}

    print(f"\nForecaster-in-the-loop study ({args.seeds} seeds, n={args.n})")
    print("Mean absolute error by interaction mode and forecaster skill\n")
    print(f"{'skill':>7}{'raw':>10}{'accept/reject':>16}{'edit neighbours':>18}")
    print("-" * 51)
    for i, s in enumerate(skills):
        print(
            f"{s:>7.2f}{mean['raw_mae'][i]:>10.0f}"
            f"{mean['ar_mae'][i]:>16.0f}{mean['edit_mae'][i]:>18.0f}"
        )

    print("\nVolume forecast off by >50% (%), by mode and skill\n")
    print(f"{'skill':>7}{'raw':>10}{'accept/reject':>16}{'edit neighbours':>18}")
    print("-" * 51)
    for i, s in enumerate(skills):
        print(
            f"{s:>7.2f}{mean['raw_out50'][i]:>10.1f}"
            f"{mean['ar_out50'][i]:>16.1f}{mean['edit_out50'][i]:>18.1f}"
        )

    gap = mean["ar_mae"][-1] - mean["edit_mae"][-1]
    raw_mae = mean["raw_mae"][-1]
    print(
        f"\nAt full skill, editing beats accept/reject by {gap:.0f} MAE "
        f"({100 * gap / raw_mae:.0f}% of the raw error) -- the operational "
        f"value of the editable neighbour interface."
    )


if __name__ == "__main__":
    main()
