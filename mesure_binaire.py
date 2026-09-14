"""Deux features reduisent momentum et volatilite a UN BIT. Combien ca coute ?

    mom_5           = (close > close.shift(5))          -> 0 ou 1
    high_vol_regime = (vol_rank > 0.65)                 -> 0 ou 1

Le modele apprend donc que le prix est monte, sans savoir de combien, et que la
volatilite est haute, sans savoir a quel point. Toute la magnitude est jetee au
seuil. Sur un jeu de dix colonnes, deux d'entre elles ne portent qu'un bit.

Le lookback n'y change rien : cinquante-quatre bits successifs donnent
l'HISTORIQUE DU SIGNE, pas l'amplitude.

VARIANTES CONTINUES TESTEES
    ret_5    = close / close.shift(5) - 1     (rendement 5 bougies, signe + amplitude)
    vol_rank = rang glissant de vol_20        (deja calcule, 0 a 1, continu)

`vol_rank` existe deja dans le cache : high_vol_regime en est le seuillage. On
compare donc la colonne source a son propre drapeau.

    python mesure_binaire.py
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
SPREAD_BPS, SLIP_S_BPS, SLIP_E_BPS, BRUIT_BPS = 2.61, 2.0, 1.0, 0.6

BASE = ["close_ema_dev", "returns", "range_norm",
        "close_h1_dev", "rsi_14_h1", "returns_h1",
        "taker_ratio", "ls_ratio_top"]

JEUX = {
    "10 actuel (2 bits)":      BASE + ["mom_5", "high_vol_regime"],
    "10 momentum continu":     BASE + ["ret_5", "high_vol_regime"],
    "10 volatilite continue":  BASE + ["mom_5", "vol_rank"],
    "10 les deux continus":    BASE + ["ret_5", "vol_rank"],
    "12 bits + continus":      BASE + ["mom_5", "high_vol_regime", "ret_5", "vol_rank"],
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
        h_e = hi[jj] * (1.0 + BRUIT_BPS / 1e4)
        l_e = lo[jj] * (1.0 - BRUIT_BPS / 1e4)
        if sens > 0:
            t_sl, t_tp = l_e <= sl[valide], h_e >= tp[valide]
        else:
            t_sl, t_tp = h_e >= sl[valide], l_e <= tp[valide]
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
    return (sens * (sortie - entree) - demi_sp[idx]) / dist


def main() -> int:
    df = pd.read_pickle(CACHE)
    n = len(df)
    hi, lo, cl = (df[c].to_numpy(np.float64) for c in ("high", "low", "close"))
    atr = np.maximum(df["atr_14"].to_numpy(np.float64), ATR_PLANCHER_FRAC * cl)
    demi_sp = (SPREAD_BPS / 1e4) * cl / 2.0
    slip_e = (SLIP_E_BPS / 1e4) * cl
    slip_s = (SLIP_S_BPS / 1e4) * cl

    # ret_5 calcule a la volee : inutile de reconstruire le cache pour mesurer.
    df["ret_5"] = df["close"] / df["close"].shift(5) - 1.0

    print(f"{n:,} bougies | SL {SL_MULT}xATR R:R {RR}")
    print(f"  mom_5           : {df['mom_5'].nunique()} valeurs distinctes, "
          f"moy {df['mom_5'].mean():.4f}")
    print(f"  high_vol_regime : {df['high_vol_regime'].nunique()} valeurs distinctes, "
          f"moy {df['high_vol_regime'].mean():.4f}")
    print(f"  ret_5           : continu, ecart-type {df['ret_5'].std():.6f}")
    print(f"  vol_rank        : continu, ecart-type {df['vol_rank'].std():.6f}\n")

    a_va, b_va = int(n * 0.55), int(n * 0.70)
    i_tr = np.arange(1440, a_va - MAX_HOLD, PAS)
    i_va = np.arange(a_va, b_va - MAX_HOLD, PAS)
    ok = lambda i: i[np.isfinite(atr[i]) & (atr[i] > 0)]
    i_tr, i_va = ok(i_tr), ok(i_va)

    r_tr = [barrieres(hi, lo, cl, atr, i_tr, s, demi_sp, slip_e, slip_s) for s in (1, -1)]
    r_va = [barrieres(hi, lo, cl, atr, i_va, s, demi_sp, slip_e, slip_s) for s in (1, -1)]

    cols_all = sorted({c for cs in JEUX.values() for c in cs})
    brut = {c: df[c].to_numpy(np.float32) for c in cols_all}
    stats = {c: (float(np.nanmean(brut[c])), float(np.nanstd(brut[c])) + 1e-8)
             for c in cols_all}

    def mat(cols, idx):
        X = np.empty((len(idx), len(cols)), np.float32)
        for k, c in enumerate(cols):
            m, s = stats[c]
            X[:, k] = np.clip((brut[c][idx] - m) / s, -5, 5)
        return np.nan_to_num(X)

    print(f"{'jeu':<24} {'AUC':>8} {'E[R] top1%':>12} {'t':>7} "
          f"{'E[R] top5%':>12} {'t':>7}")
    print("-" * 74)
    for nom, cols in JEUX.items():
        t0 = time.time()
        Xtr, Xva = mat(cols, i_tr), mat(cols, i_va)
        auc, p_va = [], []
        for k in (0, 1):
            lr = LogisticRegression(max_iter=500, n_jobs=-1)
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
        print(f"{nom:<24} {np.mean(auc):>8.4f}{bouts}   ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

    print()
    print("t au-dela de +2 : distinguable de zero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
