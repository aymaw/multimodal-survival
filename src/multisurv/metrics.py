"""Small, auditable estimators with explicit censoring conventions."""

import numpy as np


def concordance(time, event, risk):
    t, e, r = np.asarray(time), np.asarray(event, dtype=bool), np.asarray(risk)
    comparable = (t[:, None] < t[None, :]) & e[:, None]
    total = int(comparable.sum())
    if not total:
        return None
    score = (r[:, None] > r[None, :]) + 0.5 * (r[:, None] == r[None, :])
    return float(score[comparable].mean())


def censor_survival(train_time, train_event, at, left=False):
    """Reverse KM; event wins a same-bin event/censor tie in discrete data."""
    t, e = np.asarray(train_time), np.asarray(train_event, dtype=bool)
    g = 1.0
    for u in np.unique(t):
        if u > at or (left and u == at):
            break
        # Events occur within bin, then censoring at its end.
        denom = (t >= u).sum() - ((t == u) & e).sum()
        censored = ((t == u) & ~e).sum()
        if denom:
            g *= 1 - censored / denom
    return float(g)


def ipcw_brier(train_time, train_event, time, event, s, horizon):
    time, event, s = np.asarray(time), np.asarray(event), np.asarray(s)
    g_h = censor_survival(train_time, train_event, horizon, left=True)
    if g_h <= 0.05:
        return None  # insufficient censoring support, never silently clip weights
    score = np.zeros(len(time))
    for i, (t, e) in enumerate(zip(time, event)):
        if t <= horizon and e:
            g = censor_survival(train_time, train_event, t, left=True)
            if g <= 0.05:
                return None
            score[i] = s[i] ** 2 / g
        elif t > horizon or (t == horizon and not e):
            # Survival through horizon is known, including end-of-bin censoring.
            # Such end-of-bin observations are weighted by G(horizon-).
            g = censor_survival(train_time, train_event, horizon, left=True)
            score[i] = (1 - s[i]) ** 2 / g
    return float(score.mean())


def bootstrap_cindex(time, event, risk, seed=99, repeats=200):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        idx = rng.integers(0, len(time), len(time))
        value = concordance(np.asarray(time)[idx], np.asarray(event)[idx], np.asarray(risk)[idx])
        if value is not None:
            scores.append(value)
    return np.quantile(scores, [0.025, 0.975]).tolist() if scores else None
