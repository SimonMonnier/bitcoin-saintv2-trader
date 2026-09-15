"""Choix des features, refait sous protocole PURGE et aux barrieres courantes.

POURQUOI REFAIRE. Deux raisons independantes, chacune suffisante.

  1. Le protocole. Toutes les mesures de features de ce projet utilisent
     PAS = 10 : une entree toutes les 10 barres, alors que les barrieres
     mettent jusqu'a 240 barres a se resoudre. Les fenetres de resultat se
     recouvrent presque entierement. Mesure de l'ecart que ca produit : a 5 %
     de selectivite, un modele a arbres passe de 58.0 % de reussite avec
     chevauchement a 39.7 % sans.

  2. La cible. "Une feature n'est pas bonne dans l'absolu, elle l'est pour une
     CIBLE donnee" — les colonnes horaires coutaient -0.0022 d'AUC contre
     SL 3xATR / R:R 1.4 et rapportaient +0.0018 contre SL 5xATR / R:R 2.0.
     Le R:R vient de passer de 1.4 a 2.0 : l'elagage precedent ne vaut plus.

PROTOCOLE : 8 blocs successifs, entrees espacees de 240 barres (aucun
chevauchement), purge de 240 barres entre le train et le bloc evalue, chaque
bloc juge par un modele entraine uniquement sur ce qui le precede. Fenetre de
test (au-dela de 70 %) JAMAIS touchee.

METRIQUE PRINCIPALE : l'AUC, sur ~7 400 candidats. L'esperance apres selection
est reportee mais elle ne porte que 149 trades a 2 %, donc elle est bruyante —
on la lit comme un signe, pas comme une preuve.

    python mesure_features_purgee.py

Sortie : apport de chaque GROUPE, puis de chaque COLONNE une a une.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from saint_core import (FEATURE_COLS, FEATURE_COLS_M1, FEATURE_COLS_H1,
                        FEATURE_COLS_EXT, FEATURE_COLS_LIQ_TEMPS,
                        ATR_PLANCHER_FRAC)
from mesure_features import (barrieres, MAX_HOLD, SPREAD_BPS, SLIP_ENTREE_BPS,
                             SLIP_SORTIE_BPS, CACHE)
import mesure_ichimoku as MI

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SL_MULT, RR = 2.0, 2.0          # barrieres COURANTES
N_BLOCS = 8
SELECTIVITE = 0.02


def main() -> int:
    df = pd.read_pickle(CACHE)
    # Candidat a l'ajout : le seul jeu Ichimoku qui ait montre quelque chose
    # (+0.0030 avec interactions, jeux classique et lent a zero ou negatifs).
    ich = MI.ajoute_ichimoku(df, "rapide", *MI.JEUX["rapide"])
    df = df.dropna(subset=FEATURE_COLS + ich + ["atr_14"]).reset_index(drop=True)

    n = len(df)
    fin = int(n * 0.70)
    hi, lo, cl = (df[c].to_numpy(np.float64) for c in ("high", "low", "close"))
    atr = np.maximum(df["atr_14"].to_numpy(np.float64), ATR_PLANCHER_FRAC * cl)
    ds = (SPREAD_BPS / 1e4) * cl / 2.0
    se = (SLIP_ENTREE_BPS / 1e4) * cl
    ss = (SLIP_SORTIE_BPS / 1e4) * cl

    colonnes = FEATURE_COLS + ich
    idx_col = {c: k for k, c in enumerate(colonnes)}
    X = np.nan_to_num(np.column_stack(
        [df[c].to_numpy(np.float32) for c in colonnes]))
    # Normalisation sur le TRAIN de chaque bloc : la faire globalement ferait
    # remonter des statistiques futures dans le passe.

    bornes = np.linspace(int(fin * 0.30), fin, N_BLOCS + 1).astype(int)

    # Les barrieres ne dependent PAS des features : une seule fois pour tous
    # les jeux testes, sinon on recalculerait 40 fois la meme chose.
    plan = []
    for b in range(N_BLOCS):
        a_va, b_va = bornes[b], bornes[b + 1]
        i_tr = np.arange(0, a_va - 2 * MAX_HOLD, MAX_HOLD)
        i_va = np.arange(a_va, b_va - MAX_HOLD, MAX_HOLD)
        i_tr = i_tr[np.isfinite(atr[i_tr]) & (atr[i_tr] > 0)]
        i_va = i_va[np.isfinite(atr[i_va]) & (atr[i_va] > 0)]
        if len(i_tr) < 500 or len(i_va) < 40:
            continue
        for sens in (1, -1):
            r_t, _ = barrieres(hi, lo, cl, atr, i_tr, SL_MULT, RR, sens, ds, se, ss)
            r_v, _ = barrieres(hi, lo, cl, atr, i_va, SL_MULT, RR, sens, ds, se, ss)
            plan.append((i_tr, i_va, r_t, r_v))
    print(f"{n:,} bougies  |  {len(plan)//2} blocs x 2 cotes  |  "
          f"barrieres SL {SL_MULT}xATR R:R {RR}")
    print(f"candidats de validation : "
          f"{sum(len(p[1]) for p in plan):,}  (entrees independantes)\n")

    def evalue(cols):
        k = [idx_col[c] for c in cols]
        ps, rs, ys = [], [], []
        for i_tr, i_va, r_t, r_v in plan:
            xt, xv = X[i_tr][:, k], X[i_va][:, k]
            m = xt.mean(0)
            s = xt.std(0) + 1e-8
            lr = LogisticRegression(max_iter=1000, n_jobs=1).fit(
                np.clip((xt - m) / s, -5, 5), r_t > 0)
            ps.append(lr.predict_proba(np.clip((xv - m) / s, -5, 5))[:, 1])
            rs.append(r_v)
            ys.append(r_v > 0)
        p, r, y = np.concatenate(ps), np.concatenate(rs), np.concatenate(ys)
        auc = roc_auc_score(y, p)
        sel = p >= np.quantile(p, 1.0 - SELECTIVITE)
        esp = float(r[sel].mean())
        return auc, esp, int(sel.sum())

    base = list(FEATURE_COLS)
    a0, e0, n0 = evalue(base)
    print(f"{'jeu':<34} {'AUC':>7} {'d AUC':>8} {'E[R] 2%':>9} {'d E[R]':>8}")
    print("-" * 70)
    print(f"{'30 actuel (reference)':<34} {a0:7.4f} {'':>8} {e0:+9.4f} {'':>8}")

    print(f"\n  -- retrait d'un GROUPE --")
    groupes = {"M1 (12 col.)": FEATURE_COLS_M1,
               "H1 (13 col.)": FEATURE_COLS_H1,
               "Binance (2 col.)": FEATURE_COLS_EXT,
               "liquidite/temps (3 col.)": FEATURE_COLS_LIQ_TEMPS}
    for nom, g in groupes.items():
        reste = [c for c in base if c not in g]
        if len(reste) < 2:
            continue
        a, e, _ = evalue(reste)
        print(f"{'  sans ' + nom:<34} {a:7.4f} {a-a0:+8.4f} {e:+9.4f} {e-e0:+8.4f}")

    print(f"\n  -- ajout du candidat --")
    a, e, _ = evalue(base + ich)
    print(f"{'  + Ichimoku rapide (6 col.)':<34} {a:7.4f} {a-a0:+8.4f} "
          f"{e:+9.4f} {e-e0:+8.4f}")

    print(f"\n  -- retrait d'UNE colonne (apport marginal) --")
    lignes = []
    for c in base:
        a, e, _ = evalue([x for x in base if x != c])
        lignes.append((a0 - a, c, a, e - e0))
    lignes.sort(reverse=True)
    for cout, c, a, de in lignes:
        marque = "  <- indispensable" if cout > 0.004 else (
            "  <- inerte" if abs(cout) < 0.0005 else "")
        print(f"{'  sans ' + c:<34} {a:7.4f} {-cout:+8.4f} {'':>9} "
              f"{de:+8.4f}{marque}")

    print(f"\nRAPPEL : les apports marginaux NE SE COMPOSENT PAS. Retirer "
          f"ensemble\ntoutes les colonnes marquees inertes peut couter plus "
          f"que la meilleure\nn'apporte — c'est ce qui avait fait annuler "
          f"l'elagage a 10 colonnes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
