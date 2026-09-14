"""Le flux Binance a la MINUTE apporte-t-il quelque chose aux 30 features ?

Cinq colonnes produites par telecharge_flux_1m.py, mesurees en apport marginal
d'AUC sur la meme sonde logistique que mesure_features.py, aux barrieres de la
config courante (SL 2.0xATR, R:R 1.4) et avec la friction complete.

ALIGNEMENT HORAIRE — le point ou tout peut casser en silence. Les archives
Binance sont en UTC, le cache MT5 en heure du courtier, qui suit le DST
americain (+2 h l'hiver, +3 h l'ete). On reutilise la table d'offsets deja
detectee par build_binance_features, puis on VERIFIE : la nouvelle serie doit
correler fortement avec le taker_ratio 5 min deja en service, qui mesure la
meme chose depuis une autre source. Une correlation faible signifie un
decalage, pas une feature inutile — le script s'arrete dans ce cas.

    python mesure_flux_1m.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from saint_core import FEATURE_COLS, ATR_PLANCHER_FRAC
from build_binance_features import vers_heure_broker
from mesure_features import (barrieres, SPREAD_BPS, SLIP_ENTREE_BPS,
                             SLIP_SORTIE_BPS, CACHE, PAS)

FLUX = "binance_flux_1m_BTCUSD.pkl"
OFFSETS = ".cache_binance/offsets.pkl"
SL_MULT, RR = 2.0, 1.4
CORR_MIN = 0.30          # sous ce seuil, l'alignement est suspect
NOUVELLES = ["taker_1m", "taker_1m_ma5", "volume_z", "nb_trades_z", "taille_trade_z"]


def main() -> int:
    flux = pd.read_pickle(FLUX)
    tab = pd.read_pickle(OFFSETS)
    flux.index = vers_heure_broker(pd.Series(flux.index), tab).values
    flux = flux[~flux.index.duplicated(keep="last")].sort_index()

    df = pd.read_pickle(CACHE)
    avant = len(df)
    df = df.join(flux, on="time")
    couv = {c: df[c].notna().mean() for c in NOUVELLES}
    print(f"{avant:,} bougies  |  couverture : "
          + "  ".join(f"{c} {100*v:.1f}%" for c, v in couv.items()))

    # --- Verification de l'alignement -----------------------------------
    ok = df[["taker_ratio", "taker_1m_ma5"]].dropna()
    corr = float(np.corrcoef(ok["taker_ratio"], ok["taker_1m_ma5"])[0, 1])
    print(f"\ncontrole d'alignement : corr(taker_ratio 5min, taker_1m_ma5) = {corr:+.4f}")
    if abs(corr) < CORR_MIN:
        print(f"ABANDON : sous {CORR_MIN}, les deux sources ne se superposent pas. "
              f"L'offset horaire est faux — mesurer dans cet etat ne dirait rien.")
        return 1
    print("  les deux sources se superposent, l'alignement tient.\n")

    df = df.dropna(subset=FEATURE_COLS + NOUVELLES + ["atr_14"]).reset_index(drop=True)
    n = len(df)
    hi, lo, cl = (df[c].to_numpy(np.float64) for c in ("high", "low", "close"))
    atr = np.maximum(df["atr_14"].to_numpy(np.float64), ATR_PLANCHER_FRAC * cl)
    ds = (SPREAD_BPS / 1e4) * cl / 2.0
    se = (SLIP_ENTREE_BPS / 1e4) * cl
    ss = (SLIP_SORTIE_BPS / 1e4) * cl

    b_tr, b_va = int(n * 0.55), int(n * 0.70)
    i_tr = np.arange(0, b_tr - 240, PAS)
    i_va = np.arange(b_tr, b_va - 240, PAS)
    i_tr = i_tr[np.isfinite(atr[i_tr]) & (atr[i_tr] > 0)]
    i_va = i_va[np.isfinite(atr[i_va]) & (atr[i_va] > 0)]
    print(f"{n:,} bougies apres dropna  |  train {len(i_tr):,}  val {len(i_va):,}  "
          f"|  test au-dela de {int(n*0.70):,}, INTOUCHE\n")

    cols_tous = FEATURE_COLS + NOUVELLES
    stats = {}
    for c in cols_tous:
        v = df[c].to_numpy(np.float32)
        stats[c] = (float(np.nanmean(v)), float(np.nanstd(v)) + 1e-8)

    def mat(cols, idx):
        X = np.empty((len(idx), len(cols)), np.float32)
        for k, c in enumerate(cols):
            m, s = stats[c]
            X[:, k] = np.clip((df[c].to_numpy(np.float32)[idx] - m) / s, -5, 5)
        return np.nan_to_num(X)

    cache_b = {(sens, nom): barrieres(hi, lo, cl, atr, idx, SL_MULT, RR, sens, ds, se, ss)
               for sens in (1, -1) for nom, idx in (("tr", i_tr), ("va", i_va))}

    def auc(cols):
        Xtr, Xva = mat(cols, i_tr), mat(cols, i_va)
        o = []
        for sens in (1, -1):
            r_t, _ = cache_b[(sens, "tr")]
            r_v, _ = cache_b[(sens, "va")]
            lr = LogisticRegression(max_iter=500, n_jobs=1).fit(Xtr, r_t > 0)
            o.append(roc_auc_score(r_v > 0, lr.predict_proba(Xva)[:, 1]))
        return float(np.mean(o)), o

    base = list(FEATURE_COLS)
    a0, d0 = auc(base)
    print(f"{'jeu':<36} {'AUC':>7} {'ecart':>8}   {'LONG':>7} {'SHORT':>7}")
    print("-" * 72)
    print(f"{'30 actuel (reference)':<36} {a0:7.4f} {'':>8}   {d0[0]:7.4f} {d0[1]:7.4f}")

    for c in NOUVELLES:
        a, d = auc(base + [c])
        print(f"{'  + ' + c:<36} {a:7.4f} {a-a0:+8.4f}   {d[0]:7.4f} {d[1]:7.4f}")

    print("-" * 72)
    a, d = auc(base + NOUVELLES)
    print(f"{'35 = 30 + les 5':<36} {a:7.4f} {a-a0:+8.4f}   {d[0]:7.4f} {d[1]:7.4f}")

    # ls_ratio_top vaut +0.0000 : la remplacer plutot que l'ajouter garde le
    # nombre de features constant, donc le cout GPU aussi.
    sans_ls = [c for c in base if c != "ls_ratio_top"]
    a, d = auc(sans_ls + NOUVELLES)
    print(f"{'34 = sans ls_ratio_top + les 5':<36} {a:7.4f} {a-a0:+8.4f}   {d[0]:7.4f} {d[1]:7.4f}")

    a, d = auc(NOUVELLES)
    print(f"{'les 5 SEULES (contre-epreuve)':<36} {a:7.4f} {'':>8}   {d[0]:7.4f} {d[1]:7.4f}")
    print("\nEcarts sous ~0.002 : non separables du bruit sur cette fenetre.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
