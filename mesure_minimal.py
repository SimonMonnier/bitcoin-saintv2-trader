"""Le jeu MINIMAL porte-t-il encore du signal ?

L'hypothese testee : les deux colonnes Binance portent l'essentiel, et les
indicateurs derives du prix (RSI, ATR, rangs de volatilite, ecarts aux moyennes
mobiles) ne sont que du bruit qui coute cher. Si elle est vraie, onze colonnes
sur trois bougies doivent tenir la comparaison face a trente colonnes sur
vingt-cinq.

CE QUE LA SONDE MESURE, ET CE QU'ELLE NE MESURE PAS. C'est une regression
logistique : elle ne voit que des relations lineaires entre les colonnes et le
resultat. Le transformeur, lui, peut composer — former un RSI a partir des
bougies, par exemple. Un jeu qui perd ici peut donc encore gagner a
l'entrainement. En revanche, un jeu qui GAGNE ici a forcement du signal
accessible, et c'est ce qu'on cherche a etablir avant d'engager des heures de
GPU sur une machine qui se bride.

Pour rendre justice aux « trois bougies », on donne a la sonde la concatenation
des trois derniers pas, pas seulement le dernier.

    python mesure_minimal.py
"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from saint_core import ATR_PLANCHER_FRAC

CACHE = "data_cache_BTCUSD_20221215.pkl"
MAX_HOLD = 240
PAS = 10
SL_MULT, RR = 2.0, 1.4          # retenu par mesure_features.py

SPREAD_BPS, SLIP_S_BPS, SLIP_E_BPS, BRUIT_BPS = 2.61, 2.0, 1.0, 0.6

M1 = ["open_rel", "high_rel", "low_rel", "returns"]
H1 = ["open_rel_h1", "high_rel_h1", "low_rel_h1", "returns_h1", "close_h1_dev"]
EXT = ["taker_ratio", "ls_ratio_top"]
LARGE_M1 = ["rsi_14", "returns", "vol_20", "range_norm", "open_rel", "high_rel",
            "low_rel", "close_ema_dev", "mom_5", "rsi_ok", "vol_rank",
            "high_vol_regime"]
LARGE_H1 = ["rsi_14_h1", "returns_h1", "vol_20_h1", "range_norm_h1",
            "open_rel_h1", "high_rel_h1", "low_rel_h1", "close_ema_dev_h1",
            "mom_5_h1", "rsi_ok_h1", "vol_rank_h1", "high_vol_regime_h1",
            "close_h1_dev"]
LIQ = ["spread_rel", "heure_sin", "heure_cos"]

# (nom, colonnes, nombre de bougies concatenees)
JEUX = [
    ("2 Binance seules",        EXT,                          1),
    ("9 bougies sans Binance",  M1 + H1,                      1),
    ("11 minimal, 1 bougie",    M1 + H1 + EXT,                1),
    ("11 minimal, 3 bougies",   M1 + H1 + EXT,                3),
    ("30 large, 1 bougie",      LARGE_M1 + LARGE_H1 + EXT + LIQ, 1),
    ("30 large, 3 bougies",     LARGE_M1 + LARGE_H1 + EXT + LIQ, 3),
]


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

    a_va = int(n * 0.55)
    i_tr = np.arange(3, a_va - MAX_HOLD, PAS)
    i_va = np.arange(a_va, int(n * 0.70) - MAX_HOLD, PAS)
    i_tr = i_tr[np.isfinite(atr[i_tr]) & (atr[i_tr] > 0)]
    i_va = i_va[np.isfinite(atr[i_va]) & (atr[i_va] > 0)]
    print(f"{n:,} bougies | train {len(i_tr):,} | val {len(i_va):,} | "
          f"SL {SL_MULT}xATR R:R {RR} | test INTOUCHE\n")

    # Barrieres : independantes du jeu de features, calculees une fois.
    r_tr = [barrieres(hi, lo, cl, atr, i_tr, s, demi_sp, slip_e, slip_s) for s in (1, -1)]
    r_va = [barrieres(hi, lo, cl, atr, i_va, s, demi_sp, slip_e, slip_s) for s in (1, -1)]

    cols_all = sorted({c for _, cs, _ in JEUX for c in cs})
    stats = {c: (float(np.nanmean(df[c].to_numpy(np.float32))),
                 float(np.nanstd(df[c].to_numpy(np.float32))) + 1e-8)
             for c in cols_all}
    brut = {c: df[c].to_numpy(np.float32) for c in cols_all}

    def mat(cols, idx, n_bougies):
        """Concatene les `n_bougies` derniers pas : [t-2, t-1, t]."""
        blocs = []
        for d in range(n_bougies - 1, -1, -1):
            X = np.empty((len(idx), len(cols)), np.float32)
            for k, c in enumerate(cols):
                m, s = stats[c]
                X[:, k] = np.clip((brut[c][idx - d] - m) / s, -5, 5)
            blocs.append(X)
        return np.nan_to_num(np.concatenate(blocs, axis=1))

    print(f"{'jeu':<26} {'colonnes':>9} {'AUC':>8} {'E[R] top1%':>12} {'t':>7} "
          f"{'E[R] top5%':>12} {'t':>7}")
    print("-" * 88)
    for nom, cols, nb in JEUX:
        t0 = time.time()
        Xtr, Xva = mat(cols, i_tr, nb), mat(cols, i_va, nb)
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
        print(f"{nom:<26} {Xtr.shape[1]:>9} {np.mean(auc):>8.4f}{bouts}"
              f"   ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

    print()
    print("t au-dela de +2 : distinguable de zero. En deca, c'est du bruit,")
    print("quelle que soit la valeur de E[R].")
    return 0


if __name__ == "__main__":
    sys.exit(main())
