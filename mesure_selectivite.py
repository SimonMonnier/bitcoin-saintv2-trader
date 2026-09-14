"""A quelle SELECTIVITE ce jeu de features devient-il rentable ?

POURQUOI CETTE MESURE PLUTOT QU'UNE AUTRE. Quatre ameliorations mesurees de
l'entree — flux 1 min, detention de reference, architecture complete, Ichimoku
— n'ont rien change au PnL. L'AUC de la sonde vaut 0.628 quand aucune politique
entrainee n'a depasse 0.571. L'information EST dans les colonnes ; ce qui ne
marche pas, c'est sa conversion en decisions.

Une hypothese testable : l'entrainement vise 5 % de selectivite. Si l'avantage
de ces features n'existe qu'au sommet de la distribution — 1 % ou moins — alors
aucune feature supplementaire ne sauvera un filtre regle trop large, et la
reponse n'est pas d'ajouter une colonne mais de trader beaucoup moins souvent.

Ce script ne repond QUE a cette question, sur la sonde, sans PPO.

    python mesure_selectivite.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from saint_core import FEATURE_COLS, ATR_PLANCHER_FRAC
from mesure_features import (barrieres, SPREAD_BPS, SLIP_ENTREE_BPS,
                             SLIP_SORTIE_BPS, CACHE, PAS)

SL_MULT, RR = 2.0, 1.4
SELECTIVITES = (0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 1.00)


def main() -> int:
    df = pd.read_pickle(CACHE).dropna(
        subset=FEATURE_COLS + ["atr_14"]).reset_index(drop=True)
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
    print(f"{n:,} bougies  |  train {len(i_tr):,}  val {len(i_va):,}  "
          f"|  test au-dela de {int(n*0.70):,}, INTOUCHE")
    print(f"barrieres SL {SL_MULT}xATR  TP {SL_MULT*RR:.1f}xATR\n")

    stats = {c: (float(np.nanmean(df[c].to_numpy(np.float32))),
                 float(np.nanstd(df[c].to_numpy(np.float32))) + 1e-8)
             for c in FEATURE_COLS}

    def mat(idx, brut=False):
        X = np.empty((len(idx), len(FEATURE_COLS)), np.float32)
        for k, c in enumerate(FEATURE_COLS):
            v = df[c].to_numpy(np.float32)[idx]
            if not brut:
                m, s = stats[c]
                v = np.clip((v - m) / s, -5, 5)
            X[:, k] = v
        return np.nan_to_num(X)

    cb = {(sens, nm): barrieres(hi, lo, cl, atr, ix, SL_MULT, RR, sens, ds, se, ss)
          for sens in (1, -1) for nm, ix in (("tr", i_tr), ("va", i_va))}

    for nom_modele, fabrique, brut in (
            ("sonde logistique", lambda: LogisticRegression(max_iter=500, n_jobs=1), False),
            ("arbres (interactions)",
             lambda: HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                                   learning_rate=0.08,
                                                   random_state=0), True)):
        Xtr, Xva = mat(i_tr, brut), mat(i_va, brut)
        probs, gains, aucs = {}, {}, []
        for sens in (1, -1):
            r_t, _ = cb[(sens, "tr")]
            r_v, _ = cb[(sens, "va")]
            m = fabrique().fit(Xtr, r_t > 0)
            p = m.predict_proba(Xva)[:, 1]
            probs[sens], gains[sens] = p, r_v
            aucs.append(roc_auc_score(r_v > 0, p))

        # Regle IDENTIQUE au live : meilleur cote, puis barre par cote.
        print(f"--- {nom_modele}  (AUC {np.mean(aucs):.4f}) ---")
        print(f"{'selectivite':>12} {'trades':>8} {'WR':>7} {'E[R]/trade':>12} "
              f"{'t':>7} {'verdict':>22}")
        print("-" * 74)
        for q in SELECTIVITES:
            pris, r_pris = [], []
            for sens in (1, -1):
                p, r = probs[sens], gains[sens]
                # q/2 par cote : la selectivite totale visee est q.
                seuil = np.quantile(p, 1.0 - q / 2.0)
                sel = p >= seuil
                pris.append(sel)
                r_pris.append(r[sel])
            r_all = np.concatenate(r_pris)
            if len(r_all) < 30:
                continue
            wr = 100.0 * float((r_all > 0).mean())
            esp = float(r_all.mean())
            t = esp / (r_all.std(ddof=1) / np.sqrt(len(r_all)) + 1e-12)
            # Le point mort se lit sur les gains REELS, pas sur le R:R nominal.
            g = r_all[r_all > 0].mean() if (r_all > 0).any() else 0.0
            pe = abs(r_all[r_all <= 0].mean()) if (r_all <= 0).any() else 0.0
            eq = 100.0 * pe / (g + pe) if (g + pe) > 0 else float("nan")
            if esp > 0 and t > 2.0:
                v = "RENTABLE (t > 2)"
            elif esp > 0:
                v = "positif, non significatif"
            else:
                v = "perdant"
            print(f"{100*q:11.1f}% {len(r_all):8,} {wr:6.1f}% {esp:+11.4f} "
                  f"{t:+7.2f} {v:>22}   (equilibre {eq:.1f}%)")
        print()

    print("E[R] est en UNITES DE RISQUE : +1.0 = un gain egal au montant risque.")
    print("t > 2 : l'esperance est distinguable de zero sur cette fenetre.")
    print("BIAIS : on lit le meilleur d'une grille sur LA MEME validation.")
    print("Seule la fenetre de test, jamais touchee, peut confirmer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
