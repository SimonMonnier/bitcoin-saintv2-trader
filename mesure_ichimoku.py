"""Ichimoku apporte-t-il quelque chose aux 30 colonnes ? Mesure, pas avis.

DEUX SUBTILITES DE FUITE, opposees, qu'il faut traiter differemment.

  LE KUMO "DANS LE FUTUR" N'EST PAS DU FUTUR. Senkou A et B se calculent sur
  des donnees disponibles MAINTENANT, puis sont DESSINEES 26 periodes plus
  loin. A l'instant t, le nuage affiche en t+26 est deja connu. Le decalage
  va donc dans le sens sur : la valeur affichee en t a ete calculee en t-26.
  Aucune information future — c'est meme l'inverse, c'est de l'information
  RETARDEE.

  LA CHIKOU "DANS LE PASSE" EST UN PIEGE. Elle vaut la cloture de t, dessinee
  en t-26. Batir une colonne alignee sur sa position AFFICHEE mettrait
  close[t] dans la ligne t-26, soit 26 barres de futur dans chaque
  observation. C'est le defaut classique des backtests Ichimoku, et il produit
  des courbes magnifiques qui ne survivent pas une minute en live.
  On l'exprime donc sous sa forme causale : close[t] / close[t-26] - 1, ce
  qui est exactement le momentum sur 26 barres.

TROIS JEUX DE PERIODES, parce que la detention mediane est de 7 barres et que
le critere etabli cette session est qu'une feature doit VARIER A L'ECHELLE OU
LA DECISION SE PREND.

    python mesure_ichimoku.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from saint_core import FEATURE_COLS, ATR_PLANCHER_FRAC
from mesure_features import (barrieres, SPREAD_BPS, SLIP_ENTREE_BPS,
                             SLIP_SORTIE_BPS, CACHE, PAS)

SL_MULT, RR = 2.0, 1.4

# (tenkan, kijun, senkou_b, decalage)
JEUX = {
    "rapide":    (5, 15, 30, 15),
    "classique": (9, 26, 52, 26),
    "lent":      (60, 180, 360, 180),   # ~ Ichimoku H1 vu en M1
}


def _milieu(h, l, p):
    """Milieu du canal haut/bas sur p periodes — la brique d'Ichimoku.

    Statistique DIFFERENTE d'une moyenne mobile : elle ne depend que des
    extremes, donc elle ignore la trajectoire entre les deux.
    """
    return (h.rolling(p).max() + l.rolling(p).min()) / 2.0


def ajoute_ichimoku(df, nom, t, k, b, dec):
    h, l, c = df["high"], df["low"], df["close"]
    tenkan, kijun = _milieu(h, l, t), _milieu(h, l, k)
    # shift(+dec) : la valeur affichee en t a ete CALCULEE en t-dec. Decaler
    # vers l'avant retarde l'information, il ne l'anticipe pas.
    senkou_a = ((tenkan + kijun) / 2.0).shift(dec)
    senkou_b = _milieu(h, l, b).shift(dec)

    cols = {}
    cols[f"ich_{nom}_tenkan"] = c / tenkan - 1.0
    cols[f"ich_{nom}_kijun"] = c / kijun - 1.0
    cols[f"ich_{nom}_tk"] = tenkan / kijun - 1.0
    cols[f"ich_{nom}_kumo_pos"] = (c - (senkou_a + senkou_b) / 2.0) / c
    cols[f"ich_{nom}_kumo_ep"] = (senkou_a - senkou_b) / c
    # Chikou sous forme CAUSALE : jamais alignee sur sa position affichee.
    cols[f"ich_{nom}_chikou"] = c / c.shift(dec) - 1.0
    for n, s in cols.items():
        df[n] = s.replace([np.inf, -np.inf], np.nan)
    return list(cols)


def main() -> int:
    df = pd.read_pickle(CACHE)
    groupes = {}
    for nom, (t, k, b, dec) in JEUX.items():
        groupes[nom] = ajoute_ichimoku(df, nom, t, k, b, dec)
    toutes = [c for g in groupes.values() for c in g]

    print("AUTOCORRELATION A 1 MINUTE — le critere etabli : une feature doit")
    print("varier a l'echelle ou la decision se prend (detention mediane 7 barres).")
    print(f"  repere : taker_1m_ma5 0.86 (apport +0.0087) | "
          f"ls_ratio_top 0.99998 (apport 0.0000)\n")
    for nom, cols in groupes.items():
        bouts = []
        for c in cols:
            s = df[c].dropna().to_numpy()
            bouts.append(f"{c.split('_', 2)[2]:>9} {np.corrcoef(s[:-1], s[1:])[0,1]:.4f}")
        print(f"  {nom:<10} " + "  ".join(bouts))

    df = df.dropna(subset=FEATURE_COLS + toutes + ["atr_14"]).reset_index(drop=True)
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
    print(f"\n{n:,} bougies  |  train {len(i_tr):,}  val {len(i_va):,}  "
          f"|  test au-dela de {int(n*0.70):,}, INTOUCHE\n")

    stats = {}
    for c in FEATURE_COLS + toutes:
        v = df[c].to_numpy(np.float32)
        stats[c] = (float(np.nanmean(v)), float(np.nanstd(v)) + 1e-8)

    def mat(cols, idx):
        X = np.empty((len(idx), len(cols)), np.float32)
        for k, c in enumerate(cols):
            m, s = stats[c]
            X[:, k] = np.clip((df[c].to_numpy(np.float32)[idx] - m) / s, -5, 5)
        return np.nan_to_num(X)

    cb = {(sens, nm): barrieres(hi, lo, cl, atr, ix, SL_MULT, RR, sens, ds, se, ss)
          for sens in (1, -1) for nm, ix in (("tr", i_tr), ("va", i_va))}

    def auc(cols):
        Xtr, Xva = mat(cols, i_tr), mat(cols, i_va)
        o = []
        for sens in (1, -1):
            r_t, _ = cb[(sens, "tr")]
            r_v, _ = cb[(sens, "va")]
            lr = LogisticRegression(max_iter=500, n_jobs=1).fit(Xtr, r_t > 0)
            o.append(roc_auc_score(r_v > 0, lr.predict_proba(Xva)[:, 1]))
        return float(np.mean(o)), o

    base = list(FEATURE_COLS)
    a0, d0 = auc(base)
    print(f"{'jeu':<38} {'AUC':>7} {'ecart':>8}   {'LONG':>7} {'SHORT':>7}")
    print("-" * 74)
    print(f"{'30 actuel (reference)':<38} {a0:7.4f} {'':>8}   {d0[0]:7.4f} {d0[1]:7.4f}")

    for nom, cols in groupes.items():
        a, d = auc(base + cols)
        print(f"{'  + Ichimoku ' + nom + f' ({len(cols)} col.)':<38} "
              f"{a:7.4f} {a-a0:+8.4f}   {d[0]:7.4f} {d[1]:7.4f}")

    print("-" * 74)
    a, d = auc(base + toutes)
    print(f"{'  + les trois jeux (18 col.)':<38} {a:7.4f} {a-a0:+8.4f}   "
          f"{d[0]:7.4f} {d[1]:7.4f}")

    # Composante par composante, sur le jeu classique : Ichimoku n'est pas un
    # bloc, et une seule de ses lignes peut porter tout l'apport.
    print(f"\n  detail du jeu classique, une colonne a la fois :")
    for c in groupes["classique"]:
        a, _ = auc(base + [c])
        print(f"{'    + ' + c.split('_', 2)[2]:<38} {a:7.4f} {a-a0:+8.4f}")

    a, d = auc(toutes)
    print(f"\n{'  les 18 SEULES (contre-epreuve)':<38} {a:7.4f} {'':>8}   "
          f"{d[0]:7.4f} {d[1]:7.4f}")
    print("\nEcarts sous ~0.002 : non separables du bruit sur cette fenetre.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
