# -*- coding: utf-8 -*-
"""L'expert realiste est-il rentable en validation, sur ses signaux les plus forts ?

2026-09-26, run kairos_jeu_btc03. Le PPO glisse vers « ne jamais trader » :
l'expert qu'il imite ne predit le R d'un coup qu'a +0.04 de correlation, et
une politique jouee a l'argmax n'est presque jamais sure a plus de 50 %. La
question devient : L'EXPERT LUI-MEME est-il rentable en ne jouant que ses
signaux les plus forts ? Sinon, aucun apprentissage ne le sera avec cette
information.

LE SEUIL NE DEPEND PAS DE L'ECHELLE DES PREDICTIONS. Le premier essai
(`joue_expert` dans le jeu) prenait un seuil en R regle sur les predictions
croisees du train ; le modele de validation, appris sur tout le train, rend
des predictions moins extremes, et l'expert n'a joue que 10 coups. Ici le
signal de la minute est compare aux 10 000 minutes PRECEDENTES (rang
glissant, causal) : l'expert ouvre quand il est dans le sommet q de son
propre passe recent.

LES REGLES DU JEU : 3 jetons par jour, -3 R de vie, un coup a la fois ; le
coup est joue a chacune des barrieres fixes du menu, pour voir lesquelles
tiennent. VALIDATION SEULEMENT — rien n'est predit ni lu au-dela.

    python mesure_expert_val.py
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


def main() -> int:
    cfg = J.JeuConfig()
    d = pd.read_pickle(cfg.cache)
    cols = list(dict.fromkeys(["time", "open", "high", "low", "close", "atr_14",
                               "spread_bar"] + list(FEATURE_COLS)))
    d = d[cols].reset_index(drop=True)
    N = len(d)
    o, h, l, c = (d[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    atr = J.atr_effectif(d["atr_14"].to_numpy(np.float64), c, cfg)
    sp = d["spread_bar"].to_numpy(np.float64)
    n_tr, n_va = int(N * cfg.part_train), int(N * cfg.part_val)
    a_va, a_te = n_tr, n_tr + n_va
    X = d[list(FEATURE_COLS)].to_numpy(np.float32)
    st = {"mean": X[:n_tr].mean(0), "std": X[:n_tr].std(0)}
    Xn = safe_normalize(X, st).astype(np.float32)
    del X
    t0 = time.time()
    R1, D1, S1 = J.table_coups(o, h, l, sp, atr, cfg, 1.0)
    y = J.cibles_expert(R1)
    pred = J.expert_realiste(Xn, y, a_va, a_te, cfg)
    print(f"table et expert : {time.time() - t0:.0f} s ; validation "
          f"[{a_va:,} : {a_te:,}), le test n'est pas lu", flush=True)

    j_va = J.journees(d["time"], a_va, a_te)
    vmax = np.nanmax(pred, axis=1)
    # LE RANG GLISSANT, causal : la minute contre les 10 000 precedentes.
    s = pd.Series(vmax[:a_te])
    rang = s.rolling(10_000, min_periods=2_000).rank(pct=True).to_numpy()
    rang = np.concatenate([rang, np.full(N - a_te, np.nan)])
    H, L = cfg.horizon_max, cfg.lookback
    print(f"\n{len(j_va)} journees de validation, 3 jetons, cout reel\n")
    print("  sommet   barrieres (obj/stop ATR)   coups  $/jour  gagnees   net bps  brut bps (t)   PF")
    for q in (0.002, 0.005, 0.01, 0.02):
        for i in range(len(cfg.tp_atr)):
            for j in range(len(cfg.sl_atr)):
                scores, coups = [], []
                for g, (a0, b0) in enumerate(j_va):
                    t, jet, sc = max(int(a0), L - 1), cfg.jetons, 0.0
                    while t < b0 and jet > 0 and sc > -cfg.vie_R:
                        if (t + 1 + H < a_te and np.isfinite(rang[t])
                                and rang[t] >= 1 - q):
                            s_ = int(np.nanargmax(pred[t]))
                            r = float(R1[t, s_, i, j])
                            dd = int(D1[t, s_, i, j])
                            coups.append((g, t, s_, i, j, r, dd, int(S1[t, s_, i, j])))
                            sc += r
                            jet -= 1
                            t += dd
                        else:
                            t += 1
                    scores.append(sc)
                b = J.bilan(np.asarray(scores), coups, c, atr, sp, cfg)
                if b["coups"] == 0:
                    continue
                print(f"  {100*q:4.1f} %   {cfg.tp_atr[i]:>4g} / {cfg.sl_atr[j]:<4g}"
                      f"             {b['coups']:5d}  {b['score'] * cfg.risque_dollars:+6.2f}"
                      f"   {100*b['gagnees']:4.0f}%   {b['net']:+7.2f}   "
                      f"{b['brut']:+6.2f} ({b['t_brut']:+.1f})  {b['pf']:5.2f}", flush=True)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
