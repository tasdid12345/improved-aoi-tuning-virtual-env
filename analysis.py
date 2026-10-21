"""Interest-management radius tuning: GA-tuned single radius (baseline) vs LightGBM cost model (improved).

Avatar workloads are simulated in code (random-waypoint movement). For each workload, a grid of AOI radii
is scored with cost = 0.1 * mean neighbours per avatar per step + 10 * fraction of interacting pairs
(true distance < 15) missed because AOI membership uses positions that are 3 steps stale.
"""
import json
import os
import sys

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupKFold

SEED = 42
SEEDS = [42, 43, 44]                 # seeds for the randomized models
AREA, STEPS, LAG = 500.0, 60, 3      # arena side, simulated steps, update lag (steps)
INTERACT_D = 15.0                    # pairs closer than this interact
RADII = np.linspace(5, 200, 30)      # AOI radius grid (candidate actions)
N_TRAIN, N_TEST = 300, 100           # workloads; test is shifted (more avatars, faster)
GA_POP, GA_GEN = 8, 10
PARAM_GRID = [dict(num_leaves=7, n_estimators=300),   # small CV grid, selected on training folds only
              dict(num_leaves=15, n_estimators=300),
              dict(num_leaves=31, n_estimators=200)]


def pdist(p):
    """Pairwise Euclidean distances for an (n, 2) array of positions."""
    return np.linalg.norm(p[:, None] - p[None], axis=-1)


def simulate(n, speed, seed):
    """Random-waypoint simulation. Returns per-radius (cost, neighbours/avatar/step, missed fraction)."""
    rng = np.random.default_rng(seed)
    cur = rng.uniform(0, AREA, (n, 2))
    wp = rng.uniform(0, AREA, (n, 2))
    traj = np.empty((STEPS, n, 2))
    for t in range(STEPS):
        traj[t] = cur
        vec = wp - cur
        dist = np.maximum(np.linalg.norm(vec, axis=1), 1e-9)
        arrive = dist <= speed                      # reached waypoint: snap, then pick a new one
        step = np.where(arrive[:, None], vec, vec / dist[:, None] * speed)
        cur = cur + step
        wp[arrive] = rng.uniform(0, AREA, (arrive.sum(), 2))
    off = ~np.eye(n, dtype=bool)                    # exclude self-pairs
    neigh = np.zeros(len(RADII))
    missed = np.zeros(len(RADII))
    inter_total = 0
    for t in range(LAG, STEPS):
        d_true = pdist(traj[t])
        d_stale = pdist(traj[t - LAG])
        within = (d_stale[None] <= RADII[:, None, None]) & off   # AOI membership from stale positions
        interacting = (d_true < INTERACT_D) & off
        neigh += within.sum((1, 2))
        missed += (interacting[None] & ~within).sum((1, 2))
        inter_total += interacting.sum()
    neigh_pa = neigh / (n * (STEPS - LAG))
    miss_frac = missed / inter_total if inter_total > 0 else np.zeros(len(RADII))
    return 0.1 * neigh_pa + 10 * miss_frac, neigh_pa, miss_frac


def make_workloads(n_w, n_range, s_range, rng):
    """Draw (avatar count, speed, simulation seed) for each workload."""
    return [(int(rng.integers(n_range[0], n_range[1] + 1)), float(rng.uniform(*s_range)),
             int(rng.integers(0, 2**31 - 1))) for _ in range(n_w)]


def build(workloads):
    """Simulate every workload; returns features (W,2) and per-radius matrices (W,R)."""
    X, cost, neigh, miss = [], [], [], []
    for n, s, sd in workloads:
        c, nb, ms = simulate(n, s, sd)
        X.append((n, s)); cost.append(c); neigh.append(nb); miss.append(ms)
    return np.array(X, float), np.array(cost), np.array(neigh), np.array(miss)


def run_ga(cost_train, seed):
    """Evolve one radius index minimising mean training cost (population 8, 10 generations, mutation)."""
    rng = np.random.default_rng(seed)
    mean_cost = cost_train.mean(0)
    pop = rng.integers(0, len(RADII), GA_POP)
    best = None
    for gen in range(GA_GEN + 1):
        fit = mean_cost[pop]
        i = np.argmin(fit)
        if best is None or fit[i] < mean_cost[best]:
            best = pop[i]
        if gen == GA_GEN:
            break
        parents = pop[np.argsort(fit)[:GA_POP // 2]]          # truncation selection
        n_kids = GA_POP - len(parents)
        kids = parents[rng.integers(0, len(parents), n_kids)] + rng.integers(-3, 4, n_kids)  # mutation
        pop = np.concatenate([parents, np.clip(kids, 0, len(RADII) - 1)])
    return int(best)


def score(choice, cost, neigh, miss):
    """Mean regret (test cost minus best grid cost), bandwidth and missed fraction of chosen radii."""
    w = np.arange(len(cost))
    regret = cost[w, choice] - cost.min(1)
    return regret.mean(), neigh[w, choice].mean(), miss[w, choice].mean()


def rows(X):
    """Long table: one row per (workload, radius) with features [avatars, speed, radius]."""
    return np.column_stack([np.repeat(X, len(RADII), axis=0), np.tile(RADII, len(X))])


def fit_lgb(Xl, y, params, seed):
    """LightGBM cost regressor; single thread for determinism."""
    m = lgb.LGBMRegressor(learning_rate=0.05, subsample=0.8, subsample_freq=1,
                          random_state=seed, n_jobs=1, verbose=-1, **params)
    return m.fit(Xl, y)


def pick_radius(model, X):
    """For each workload, the radius index with the lowest predicted cost."""
    return model.predict(rows(X)).reshape(len(X), len(RADII)).argmin(1)


def main():
    rng = np.random.default_rng(SEED)
    print("Simulating training workloads (10-80 avatars, speed 1-5)...")
    X_tr, cost_tr, _, _ = build(make_workloads(N_TRAIN, (10, 80), (1, 5), rng))
    print("Simulating shifted test workloads (60-120 avatars, speed 3-8)...")
    X_te, cost_te, neigh_te, miss_te = build(make_workloads(N_TEST, (60, 120), (3, 8), rng))

    # Improved model: choose LightGBM hyper-parameters by grouped CV on training workloads only
    print("Selecting LightGBM parameters by 3-fold grouped CV on training workloads...")
    Xl, y = rows(X_tr), cost_tr.ravel()
    groups = np.repeat(np.arange(len(X_tr)), len(RADII))
    best_params, best_cv = None, np.inf
    for p in PARAM_GRID:
        regs = []
        for tr, va in GroupKFold(n_splits=3).split(Xl, y, groups):
            va_w = np.unique(groups[va])
            m = fit_lgb(Xl[tr], y[tr], p, SEED)
            ch = pick_radius(m, X_tr[va_w])
            regs.append((cost_tr[va_w, ch] - cost_tr[va_w].min(1)).mean())
        cv = float(np.mean(regs))
        print(f"  {p}: CV mean regret = {cv:.5f}")
        if cv < best_cv:
            best_params, best_cv = p, cv

    # Baseline: GA single radius per seed; Improved: LightGBM per seed
    print("Evaluating baseline (GA single radius) and improved (LightGBM) on test workloads...")
    base, imp = [], []
    for s in SEEDS:
        idx = run_ga(cost_tr, s)
        base.append(score(np.full(N_TEST, idx), cost_te, neigh_te, miss_te))
        m = fit_lgb(Xl, y, best_params, s)
        imp.append(score(pick_radius(m, X_te), cost_te, neigh_te, miss_te))
        print(f"  seed {s}: GA radius={RADII[idx]:.1f} regret={base[-1][0]:.5f} | "
              f"LightGBM regret={imp[-1][0]:.5f}")
    base, imp = np.array(base), np.array(imp)
    b_mean, i_mean = base.mean(0), imp.mean(0)
    b_std, i_std = base.std(0)[0], imp.std(0)[0]

    metrics = {
        "baseline_score": float(b_mean[0]), "improved_score": float(i_mean[0]),
        "improvement": float(b_mean[0] - i_mean[0]),
        "baseline_score_std": float(b_std), "improved_score_std": float(i_std),
        "baseline_bandwidth": float(b_mean[1]), "improved_bandwidth": float(i_mean[1]),
        "baseline_missed_fraction": float(b_mean[2]), "improved_missed_fraction": float(i_mean[2]),
        "best_grid_cost_test_mean": float(cost_te.min(1).mean()),
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        print("Non-finite metrics; exiting.")
        sys.exit(1)

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["GA single radius", "LightGBM cost model"], [b_mean[0], i_mean[0]],
           yerr=[b_std, i_std], capsize=5, color=["grey", "tab:blue"])
    ax.set_ylabel("Test mean regret (lower is better)")
    ax.set_title("Interest-management radius choice")
    fig.tight_layout()
    fig.savefig("figures/regret_comparison.png", dpi=120)

    results = {
        "question": "Can a LightGBM cost model that conditions the interest-management radius on user count "
                    "and movement speed beat a single evolutionary-tuned radius on unseen, shifted avatar "
                    "workloads in a simulated virtual environment?",
        "method": "Seeded random-waypoint avatar simulation; 30-radius grid scored by cost; GA-tuned single "
                  "radius vs LightGBM regressor over (avatars, speed, radius) choosing argmin predicted cost; "
                  "test on shifted workloads; seeds 42-44.",
        "dataset": {
            "name": "Simulated random-waypoint avatar workloads (numpy, seeded)",
            "source_url": "generated in code (numpy); no download",
            "n_rows": N_TRAIN + N_TEST,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "mean regret (test cost of chosen radius minus best grid cost)",
        "higher_is_better": False,
        "baseline": "Genetic algorithm (population 8, 10 generations, mutation over radius grid) finding one "
                    "radius minimising mean training cost, applied to all test workloads.",
        "improved": "LightGBM regressor on (avatars, speed, radius) -> cost, trained on training workloads; "
                    "per test workload picks the radius with the lowest predicted cost.",
        "metrics": metrics,
        "notes": "Synthetic workloads; the split is by design (training 10-80 avatars, speed 1-5; test 60-120 "
                 "avatars, speed 3-8), not random. Tree models extrapolate flat outside the training range. "
                 "Cost is noisy for small workloads with few interacting pairs. Regret uses the best of the "
                 "grid as oracle. Std is population std over three seeds (GA seed or LightGBM seed). "
                 "LightGBM parameters chosen by grouped CV on training workloads only.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json and figures/regret_comparison.png")


if __name__ == "__main__":
    main()
