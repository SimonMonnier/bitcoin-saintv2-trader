# -*- coding: utf-8 -*-
"""Jusqu'ou elargir la porte de l'expert en M5 ?

2026-09-28, demande du proprietaire : « maintenant qu'on a de meilleurs
signaux, on va augmenter le nombre de trades, toujours dix jetons, mais une
selectivite plus large ». La porte (`porte_rang_expert` = 0.90) ne laisse
entrer que les bougies ou le rang glissant de l'expert est dans ses 10 % du
sommet. La regle 0.90 venait du M1 (run btc04), ou l'expert etait bien plus
faible. Ici on la remesure EN M5.

CE QUI EST MESURE, pour les blocs 1 et 2 du run kairos_jeu_m5_01 (memes
blocs, meme purge, meme expert que `main_blocs`) : par tranche de rang de
l'expert dans le sens joue, le R net MOYEN de tous les coups de ce sens (la
cible de l'expert, cout plein) et celui du coup de reference (2 ATR / 2 ATR),
sa duree, et combien de bougies par jour la tranche laisse entrer.

  - sur la VALIDATION de chaque bloc : l'expert ne l'a pas apprise ;
  - sur l'ENTRAINEMENT : predictions croisees, hors echantillon aussi.
LE BLOC DE TEST N'EST PAS LU.

    python mesure_porte_m5.py > mesure_porte_m5.txt
"""
import sys

import numpy as np
import pandas as pd

import jeu_kairos as J

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TRANCHES = [(0.98, 1.01), (0.95, 0.98), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85),
            (0.70, 0.80), (0.50, 0.70), (0.00, 0.50)]


def tableau(nom, idx, rangs, y, Rref, Dref, jour, cfg):
    n_jours = len(np.unique(jour[idx]))
    print(f"\n  {nom} : {len(idx):,} bougies, {n_jours} jours")
    print("    rang (sens joue)   bougies/jour  R moyen (16 coups)  t (jours)   "
          "coup ref R   duree")
    for a, b in TRANCHES:
        vals, ref, dur, jr = [], [], [], []
        for s_ in range(2):
            m = idx[(rangs[idx, s_] >= a) & (rangs[idx, s_] < b)
                    & np.isfinite(y[idx, s_]) & np.isfinite(Rref[idx, s_])]
            vals.append(y[m, s_])
            ref.append(Rref[m, s_])
            dur.append(Dref[m, s_])
            jr.append(jour[m])
        v, r_, d_, j_ = (np.concatenate(x) for x in (vals, ref, dur, jr))
        if len(v) == 0:
            continue
        pj = pd.Series(v).groupby(j_).mean()
        t = pj.mean() / (pj.std(ddof=1) / np.sqrt(len(pj))) if len(pj) > 2 else float("nan")
        print(f"    {a:.2f} - {min(b, 1.0):.2f}       {len(v) / n_jours:7.1f}      "
              f"{v.mean():+8.4f}        {t:+6.1f}     {r_.mean():+8.4f}   "
              f"{np.median(d_) * cfg.minutes_par_barre:5.0f} min")


def main() -> int:
    cfg = J.JeuConfig()
    d = pd.read_pickle(cfg.cache)
    cols = list(dict.fromkeys(["time", "open", "high", "low", "close", "atr_14",
                               "spread_bar"] + J.colonnes_jeu(cfg)))
    d = d[cols].reset_index(drop=True)
    N = len(d)
    o, h, l, c = (d[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    atr = J.atr_effectif(d["atr_14"].to_numpy(np.float64), c, cfg)
    sp = d["spread_bar"].to_numpy(np.float64)
    t_ns = d["time"].values.astype("int64")
    jour = d["time"].dt.floor("D").values.astype("int64")
    X = d[J.colonnes_jeu(cfg)].to_numpy(np.float32)
    R1, D1, _ = J.table_coups(o, h, l, sp, atr, cfg, 1.0)
    y = J.cibles_expert(R1, cfg)
    ri, rj = J._ref(cfg)
    Rref, Dref = R1[:, :, ri, rj], D1[:, :, ri, rj].astype(np.float64)
    purge = int(cfg.horizon_max + cfg.lookback + cfg.purge_semaines * cfg.barres_par_partie)
    fmt = lambda i: pd.Timestamp(d["time"].iloc[min(i, N - 1)]).strftime("%Y-%m-%d")
    print(f"MESURE DE LA PORTE EN M5 — {N:,} bougies ; coup de reference : objectif "
          f"{cfg.tp_atr[ri]:g} ATR, stop {cfg.sl_atr[rj]:g} ATR ; porte actuelle "
          f"{cfg.porte_rang_expert:g}")
    for k in (0, 1):
        permis, (te0, te1), (va0, va1) = J.masque_blocs(N, cfg.n_blocs, k, purge)
        fin_tr = J.prochain_exclu(permis)
        st_m = X[permis].astype(np.float64).mean(0)
        st_s = X[permis].astype(np.float64).std(0)
        Xn = J.safe_normalize(X, {"mean": st_m.astype(np.float32),
                                  "std": st_s.astype(np.float32)}).astype(np.float32)
        pred, _ = J.expert_multi(Xn, y, t_ns, permis, fin_tr, cfg)
        rangs = J.features_expert(pred, fenetre=10_000 // int(cfg.minutes_par_barre))[:, 2:4]
        print(f"\n=== BLOC {k + 1} (test {fmt(te0)} -> {fmt(te1 - 1)} NON LU) ===")
        tableau(f"VALIDATION {fmt(va0)} -> {fmt(va1 - 1)}", np.arange(va0, va1),
                rangs, y, Rref, Dref, jour, cfg)
        tableau("ENTRAINEMENT (predictions croisees)", np.flatnonzero(permis),
                rangs, y, Rref, Dref, jour, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
