"""Les quatre colonnes Binance inutilisees apportent-elles quelque chose ?

`binance_features_BTCUSD.pkl` contient six colonnes ; FEATURE_COLS n'en
consomme que deux. Les quatre autres — ls_ratio_retail, oi_change,
funding_rate, funding_cum24 — sont deja telechargees et alignees.

On mesure leur APPORT MARGINAL sur la meme sonde logistique que
mesure_features.py, aux barrieres de la config courante (SL 2.0xATR,
R:R 1.4), avec la friction complete de l'environnement.

DEUX PRECAUTIONS.

  - Les apports marginaux NE SE COMPOSENT PAS. Quatre features a +0.000 prises
    une a une peuvent valoir +0.003 ensemble, et l'inverse est vrai aussi.
    D'ou la ligne "les 4" en plus des lignes individuelles.

  - Biais de selection : on lit le meilleur d'une petite grille sur LA MEME
    fenetre de validation. Les ecarts sous ~0.002 d'AUC ne sont pas separables
    du bruit. Seule la fenetre de test, intouchee, tranche vraiment.

    python mesure_binance_extra.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from saint_core import FEATURE_COLS, ATR_PLANCHER_FRAC, SOURCE_EXT_FICHIER
from mesure_features import (barrieres, SPREAD_BPS, SLIP_ENTREE_BPS,
                             SLIP_SORTIE_BPS, CACHE, PAS)

SL_MULT, RR = 2.0, 2.8 / 2.0        # config courante : atr_sl_mult / atr_tp_mult
EXTRAS = ["ls_ratio_retail", "oi_change", "funding_rate", "funding_cum24"]


def main() -> int:
    df = pd.read_pickle(CACHE)
    ext = pd.read_pickle(SOURCE_EXT_FICHIER)

    # Jointure sur l'horodatage, comme merge_m1_h1 : pas de reindexation
    # positionnelle, qui decalerait silencieusement les deux sources.
    avant = len(df)
    df = df.join(ext[EXTRAS], on="time")
    couv = {c: df[c].notna().mean() for c in EXTRAS}
    print(f"{avant:,} bougies  |  couverture des colonnes jointes : "
          + "  ".join(f"{c} {100*v:.1f}%" for c, v in couv.items()))

    manquant = [c for c in EXTRAS if couv[c] < 0.90]
    if manquant:
        print(f"ABANDON : couverture insuffisante sur {manquant}")
        return 1

    df = df.dropna(subset=FEATURE_COLS + EXTRAS + ["atr_14"]).reset_index(drop=True)
    n = len(df)
    hi, lo, cl = (df[c].to_numpy(np.float64) for c in ("high", "low", "close"))
    atr = np.maximum(df["atr_14"].to_numpy(np.float64), ATR_PLANCHER_FRAC * cl)
    demi_sp = (SPREAD_BPS / 1e4) * cl / 2.0
    slip_e = (SLIP_ENTREE_BPS / 1e4) * cl
    slip_s = (SLIP_SORTIE_BPS / 1e4) * cl

    b_tr, b_va = int(n * 0.55), int(n * 0.70)
    i_tr = np.arange(0, b_tr - 240, PAS)
    i_va = np.arange(b_tr, b_va - 240, PAS)
    for nom, i in (("tr", i_tr), ("va", i_va)):
        pass
    i_tr = i_tr[np.isfinite(atr[i_tr]) & (atr[i_tr] > 0)]
    i_va = i_va[np.isfinite(atr[i_va]) & (atr[i_va] > 0)]
    print(f"{n:,} bougies apres dropna  |  train {len(i_tr):,}  val {len(i_va):,}  "
          f"|  test au-dela de {int(n*0.70):,}, INTOUCHE")
    print(f"barrieres SL {SL_MULT}xATR  TP {SL_MULT*RR:.1f}xATR  (R:R 1:{RR:.1f})\n")

    stats = {}
    for c in FEATURE_COLS + EXTRAS:
        v = df[c].to_numpy(np.float32)
        stats[c] = (float(np.nanmean(v)), float(np.nanstd(v)) + 1e-8)

    def mat(cols, idx):
        X = np.empty((len(idx), len(cols)), np.float32)
        for k, c in enumerate(cols):
            m, s = stats[c]
            X[:, k] = np.clip((df[c].to_numpy(np.float32)[idx] - m) / s, -5, 5)
        return np.nan_to_num(X)

    # Les barrieres ne dependent pas du jeu de features : une seule fois.
    cache_b = {}
    for sens in (+1, -1):
        for nom, idx in (("tr", i_tr), ("va", i_va)):
            cache_b[(sens, nom)] = barrieres(hi, lo, cl, atr, idx, SL_MULT, RR,
                                             sens, demi_sp, slip_e, slip_s)

    def auc_du_jeu(cols):
        Xtr, Xva = mat(cols, i_tr), mat(cols, i_va)
        out = []
        for sens in (+1, -1):
            r_t, _ = cache_b[(sens, "tr")]
            r_v, _ = cache_b[(sens, "va")]
            # n_jobs=1 : un entrainement GPU tourne en parallele, inutile de
            # lui prendre tous les coeurs.
            lr = LogisticRegression(max_iter=500, n_jobs=1)
            lr.fit(Xtr, r_t > 0)
            p = lr.predict_proba(Xva)[:, 1]
            out.append(roc_auc_score(r_v > 0, p) if len(np.unique(r_v > 0)) > 1 else np.nan)
        return float(np.nanmean(out)), out

    base = list(FEATURE_COLS)
    auc_base, d_base = auc_du_jeu(base)
    print(f"{'jeu':<34} {'AUC':>7} {'ecart':>8}   {'LONG':>7} {'SHORT':>7}")
    print("-" * 70)
    print(f"{'30 actuel (reference)':<34} {auc_base:7.4f} {'':>8}   "
          f"{d_base[0]:7.4f} {d_base[1]:7.4f}")

    for c in EXTRAS:
        a, d = auc_du_jeu(base + [c])
        print(f"{'  + ' + c:<34} {a:7.4f} {a-auc_base:+8.4f}   {d[0]:7.4f} {d[1]:7.4f}")

    a, d = auc_du_jeu(base + EXTRAS)
    print("-" * 70)
    print(f"{'34 = 30 + les 4':<34} {a:7.4f} {a-auc_base:+8.4f}   {d[0]:7.4f} {d[1]:7.4f}")

    # Contre-epreuve : ces colonnes seules valent-elles quelque chose ?
    a, d = auc_du_jeu(EXTRAS)
    print(f"{'les 4 SEULES (contre-epreuve)':<34} {a:7.4f} {'':>8}   {d[0]:7.4f} {d[1]:7.4f}")
    print("\nEcarts sous ~0.002 : non separables du bruit sur cette fenetre.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
