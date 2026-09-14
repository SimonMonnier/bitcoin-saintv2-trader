"""Combien de bougies d'historique, avec le jeu a 30 features ?

LA QUESTION. Le jeu a 30 colonnes contient DEJA des resumes d'historique : RSI
sur 14 bougies, rang de volatilite sur 1440, range_norm rapporte a sa moyenne
sur 1440. Empiler en plus des dizaines de pas de temps peut donc etre redondant
— ou apporter ce que ces resumes ecrasent, a savoir la FORME de la trajectoire
recente.

Aucune mesure ne tranche aujourd'hui. Celle sur le jeu minimal (11 colonnes
brutes) donnait 3 bougies ~ 1 bougie, mais elle ne portait pas sur ce jeu-ci.

CE QUE CA COUTE, et pourquoi ca merite d'etre mesure plutot que suppose : le
tenseur d'observation est en 4 dimensions (batch, temps, features, 80). Passer
de 25 a 54 pas de temps multiplie par 2.2 le travail GPU de chaque forward, sur
une machine deja bridee a 10 % de sa frequence.

La sonde est LINEAIRE : elle ne voit pas ce qu'un transformeur compose. Un
historique qui ne lui sert a rien peut encore servir au modele. Mais un
historique qui l'aide ICI aide a coup sur.

    python mesure_lookback.py
"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from saint_core import ATR_PLANCHER_FRAC

CACHE = "data_cache_BTCUSD_20221215.pkl"
MAX_HOLD, PAS = 240, 10
SL_MULT, RR = 2.0, 1.4
SPREAD_BPS, SLIP_S_BPS, SLIP_E_BPS = 2.61, 2.0, 1.0

JEU_30 = ["rsi_14", "returns", "vol_20", "range_norm", "open_rel", "high_rel",
          "low_rel", "close_ema_dev", "mom_5", "rsi_ok", "vol_rank",
          "high_vol_regime",
          "rsi_14_h1", "returns_h1", "vol_20_h1", "range_norm_h1",
          "open_rel_h1", "high_rel_h1", "low_rel_h1", "close_ema_dev_h1",
          "mom_5_h1", "rsi_ok_h1", "vol_rank_h1", "high_vol_regime_h1",
          "close_h1_dev",
          "taker_ratio", "ls_ratio_top",
          "spread_rel", "heure_sin", "heure_cos"]

# Profondeurs testees. Au-dela de quelques pas, une regression logistique sur
# 30 x N colonnes surapprend : on echantillonne donc les pas au lieu de les
# prendre consecutifs — [t, t-2, t-5, t-12, t-25, t-53] couvre la meme etendue
# temporelle avec six fois moins de colonnes.
PROFONDEURS = {
    "1 bougie":            [0],
    "3 consecutives":      [0, 1, 2],
    "5 espacees (0-12)":   [0, 1, 3, 6, 12],
    "6 espacees (0-25)":   [0, 1, 3, 6, 12, 25],
    "7 espacees (0-53)":   [0, 1, 3, 6, 12, 25, 53],
}


def barrieres(hi, lo, cl, atr, idx, sens, demi_sp, slip_e, slip_s):
    n = len(cl)
    dist = SL_MULT * atr[idx]
    entree = cl[idx] + sens * (demi_sp[idx] + slip_e[idx])
    tp, sl = entree + sens * RR * dist, entree - sens * dist
    sortie = np.full(len(idx), np.nan)
    motif = np.zeros(len(idx), np.int8)
    ouvert = np.ones(len(idx), bool)
    for h in range(1, MAX_HOLD + 1):
        j = idx + h
        valide = ouvert & (j < n)
        if not valide.any():
            break
        jj = j[valide]
        if sens > 0:
            t_sl, t_tp = lo[jj] <= sl[valide], hi[jj] >= tp[valide]
        else:
            t_sl, t_tp = hi[jj] >= sl[valide], lo[jj] <= tp[valide]
        pos = np.where(valide)[0]
        k_sl, k_tp = pos[t_sl], pos[t_tp & ~t_sl]
        sortie[k_sl], motif[k_sl] = sl[k_sl], -1
        sortie[k_tp], motif[k_tp] = tp[k_tp], 1
        ouvert[k_sl] = ouvert[k_tp] = False
    reste = np.where(ouvert)[0]
    if len(reste):
        sortie[reste] = cl[np.minimum(idx[reste] + MAX_HOLD, n - 1)]
    contre = motif <= 0
    sortie = sortie + sens * np.where(contre, -slip_s[idx], slip_s[idx])
    # Pas de commission : ce courtier n'en facture pas sur BTCUSD CFD.
    return (sens * (sortie - entree) - demi_sp[idx]) / dist


def main() -> int:
    df = pd.read_pickle(CACHE)
    n = len(df)
    hi, lo, cl = (df[c].to_numpy(np.float64) for c in ("high", "low", "close"))
    atr = np.maximum(df["atr_14"].to_numpy(np.float64), ATR_PLANCHER_FRAC * cl)
    demi_sp = (SPREAD_BPS / 1e4) * cl / 2.0
    slip_e = (SLIP_E_BPS / 1e4) * cl
    slip_s = (SLIP_S_BPS / 1e4) * cl

    a_va, b_va = int(n * 0.55), int(n * 0.70)
    i_tr = np.arange(60, a_va - MAX_HOLD, PAS)
    i_va = np.arange(a_va, b_va - MAX_HOLD, PAS)
    ok = lambda i: i[np.isfinite(atr[i]) & (atr[i] > 0)]
    i_tr, i_va = ok(i_tr), ok(i_va)
    print(f"{n:,} bougies | train {len(i_tr):,} | val {len(i_va):,} | "
          f"SL {SL_MULT}xATR R:R {RR} | test INTOUCHE\n")

    r_tr = [barrieres(hi, lo, cl, atr, i_tr, s, demi_sp, slip_e, slip_s) for s in (1, -1)]
    r_va = [barrieres(hi, lo, cl, atr, i_va, s, demi_sp, slip_e, slip_s) for s in (1, -1)]

    brut = {c: df[c].to_numpy(np.float32) for c in JEU_30}
    stats = {c: (float(np.nanmean(brut[c])), float(np.nanstd(brut[c])) + 1e-8)
             for c in JEU_30}

    def mat(decalages, idx):
        blocs = []
        for d in decalages:
            X = np.empty((len(idx), len(JEU_30)), np.float32)
            for k, c in enumerate(JEU_30):
                m, s = stats[c]
                X[:, k] = np.clip((brut[c][idx - d] - m) / s, -5, 5)
            blocs.append(X)
        return np.nan_to_num(np.concatenate(blocs, axis=1))

    print(f"{'profondeur':<22} {'colonnes':>9} {'AUC':>8} "
          f"{'E[R] top1%':>12} {'t':>7} {'E[R] top5%':>12} {'t':>7}")
    print("-" * 82)
    for nom, dec in PROFONDEURS.items():
        t0 = time.time()
        Xtr, Xva = mat(dec, i_tr), mat(dec, i_va)
        auc, p_va = [], []
        for k in (0, 1):
            lr = LogisticRegression(max_iter=600, n_jobs=-1)
            lr.fit(Xtr, r_tr[k] > 0)
            p = lr.predict_proba(Xva)[:, 1]
            p_va.append(p)
            auc.append(roc_auc_score(r_va[k] > 0, p))
        bouts = ""
        for q in (0.01, 0.05):
            sel = np.concatenate([r_va[k][p_va[k] >= np.quantile(p_va[k], 1 - q)]
                                  for k in (0, 1)])
            t = sel.mean() / (sel.std() / np.sqrt(len(sel)) + 1e-12)
            bouts += f" {sel.mean():>+11.4f} {t:>+7.1f}"
        print(f"{nom:<22} {Xtr.shape[1]:>9} {np.mean(auc):>8.4f}{bouts}"
              f"   ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

    print()
    print("Un historique qui aide la sonde LINEAIRE aidera le transformeur.")
    print("L'inverse n'est pas vrai : il peut composer ce qu'elle ne voit pas.")
    print("Le cout GPU croit lineairement avec la profondeur retenue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
