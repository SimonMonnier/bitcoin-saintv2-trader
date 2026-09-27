# -*- coding: utf-8 -*-
"""LE TEST DE L'EXPERT SEUL — un essai, une configuration fixee d'avance.

2026-09-27. Les modeles PPO du jeu perdent sur le test du fold 1 (btc05
-2.82 $/jour, btc04 -1.26 $/jour) apres avoir paru rentables en
validation. Le seul candidat restant est l'expert joue seul.

LA CONFIGURATION EST FIXEE AVANT DE REGARDER, sur la validation du fold 1
(`mesure_expert_val.py`) : sommet 2 % du rang glissant de l'expert, objectif
16 ATR, stop 2 ATR, 3 jetons, -3 R de vie, cout reel. C'est la ligne
PRINCIPALE, et la seule qui compte pour le verdict. La moyenne des 16
barrieres au meme sommet est donnee en second, pour voir si le resultat
tient a une barriere particuliere.

L'EXPERT n'apprend que sur le train du fold 1. Tout ce qui suit la
validation du fold 1 lui est vierge : les trois fenetres de test du
walk-forward, prises separement puis ensemble.

    python test_expert_seul.py
"""
import sys
import time

import numpy as np
import pandas as pd

import jeu_kairos as J
from saint_core import FEATURE_COLS, safe_normalize

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SOMMET = 0.02
OBJ, STOP = 16.0, 2.0


def joue_regle(jours, pred, rang, R, D, S, fin, i, j, cfg):
    H, L = cfg.horizon_max, cfg.lookback
    scores, coups = [], []
    for g, (a0, b0) in enumerate(jours):
        t, jet, sc = max(int(a0), L - 1), cfg.jetons, 0.0
        while t < b0 and jet > 0 and sc > -cfg.vie_R:
            if (t + 1 + H < fin and np.isfinite(rang[t]) and rang[t] >= 1 - SOMMET):
                s_ = int(np.nanargmax(pred[t]))
                r = float(R[t, s_, i, j])
                dd = int(D[t, s_, i, j])
                coups.append((g, t, s_, i, j, r, dd, int(S[t, s_, i, j])))
                sc += r
                jet -= 1
                t += dd
            else:
                t += 1
        scores.append(sc)
    return np.asarray(scores), coups


def main() -> int:
    cfg = J.JeuConfig()
    t0 = time.time()
    d = pd.read_pickle(cfg.cache)
    cols = list(dict.fromkeys(["time", "open", "high", "low", "close", "atr_14",
                               "spread_bar"] + list(FEATURE_COLS)))
    d = d[cols].reset_index(drop=True)
    N = len(d)
    o, h, l, c = (d[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    atr = J.atr_effectif(d["atr_14"].to_numpy(np.float64), c, cfg)
    sp = d["spread_bar"].to_numpy(np.float64)
    n_tr, n_va, n_te = (int(N * x) for x in (cfg.part_train, cfg.part_val, cfg.part_test))
    X = d[list(FEATURE_COLS)].to_numpy(np.float32)
    st = {"mean": X[:n_tr].mean(0), "std": X[:n_tr].std(0)}
    Xn = safe_normalize(X, st).astype(np.float32)
    del X
    R1, D1, S1 = J.table_coups(o, h, l, sp, atr, cfg, 1.0)
    y = J.cibles_expert(R1)
    pred, _ = J.expert_realiste(Xn, y, n_tr, N, cfg)
    vmax = np.nanmax(pred, axis=1)
    rang = pd.Series(vmax).rolling(10_000, min_periods=2_000).rank(pct=True).to_numpy()
    print(f"preparation {time.time() - t0:.0f} s ; expert appris sur [0 : {n_tr:,})", flush=True)

    i0 = list(cfg.tp_atr).index(OBJ)
    j0 = list(cfg.sl_atr).index(STOP)
    debut = n_tr + n_va
    fenetres = [(debut + k * n_te, min(debut + (k + 1) * n_te, N)) for k in range(3)]
    fenetres = [(a, b) for a, b in fenetres if b - a > 1440]
    print(f"\nREGLE FIXEE D'AVANCE : sommet {100 * SOMMET:g} %, objectif {OBJ:g} ATR, "
          f"stop {STOP:g} ATR, {cfg.jetons} jetons, cout reel\n")
    tout_s, tout_c, n_jours = [], [], 0
    for k, (a, b) in enumerate(fenetres):
        jr = J.journees(d["time"], a, b)
        sc, cp = joue_regle(jr, pred, rang, R1, D1, S1, b, i0, j0, cfg)
        bl = J.bilan(sc, cp, c, atr, sp, cfg)
        print(f"TEST {k + 1}  {d['time'].iloc[a]:%Y-%m-%d} -> {d['time'].iloc[b - 1]:%Y-%m-%d}  "
              f"{bl['score'] * cfg.risque_dollars:+.2f} $/jour  |  {J.ligne_bilan(bl, cfg)}",
              flush=True)
        tout_s.append(sc)
        tout_c += [(q[0] + n_jours,) + tuple(q[1:]) for q in cp]
        n_jours += len(jr)
    sc = np.concatenate(tout_s)
    bl = J.bilan(sc, tout_c, c, atr, sp, cfg)
    r = np.array([q[5] for q in tout_c])
    t_r = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else float("nan")
    print(f"\nENSEMBLE  {n_jours} journees  {bl['score'] * cfg.risque_dollars:+.2f} $/jour  "
          f"(t du R par coup {t_r:+.2f})  |  {J.ligne_bilan(bl, cfg)}", flush=True)

    print("\nEN SECOND - la moyenne des 16 barrieres au meme sommet, sur l'ensemble :")
    moy = []
    for i in range(len(cfg.tp_atr)):
        for j in range(len(cfg.sl_atr)):
            ss = []
            for a, b in fenetres:
                jr = J.journees(d["time"], a, b)
                s_, _ = joue_regle(jr, pred, rang, R1, D1, S1, b, i, j, cfg)
                ss.append(s_)
            v = float(np.concatenate(ss).mean() * cfg.risque_dollars)
            moy.append(v)
            print(f"  objectif {cfg.tp_atr[i]:>4g} / stop {cfg.sl_atr[j]:<4g}  {v:+.2f} $/jour", flush=True)
    print(f"  moyenne des 16 : {np.mean(moy):+.2f} $/jour, positives : "
          f"{sum(v > 0 for v in moy)}/16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
