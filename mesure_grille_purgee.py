"""Existe-t-il une configuration de barrieres dont l'avantage SURVIT a un
protocole honnete ?

POURQUOI REFAIRE CETTE GRILLE. Le SL/TP actuel a ete choisi sur des mesures a
PAS = 10 : les entrees etaient espacees de 10 barres alors que les barrieres
mettent jusqu'a 240 barres a se resoudre. Les resultats voisins se recouvrent
donc presque entierement, ce qui fait compter 1 368 trades la ou il y en a 58
d'independants, et gonfle l'esperance comme le t. Mesure de l'ecart : a 5 % de
selectivite, un modele a arbres passait de 58.0 % de reussite avec
chevauchement a 39.7 % sans.

PROTOCOLE ICI :
  - entrees espacees de MAX_HOLD barres : aucune fenetre de resultat ne
    recouvre la suivante, quelle que soit la largeur du stop, puisque
    `barrieres` cloture au marche au plafond ;
  - walk-forward en blocs successifs, chaque bloc evalue par un modele
    entraine UNIQUEMENT sur ce qui le precede ;
  - purge de MAX_HOLD barres entre le train et le bloc evalue, pour qu'aucune
    barriere du train ne deborde sur la periode jugee ;
  - fenetre de test (au-dela de 70 %) JAMAIS touchee.

Modele : regression logistique. Les arbres ont ete ecartes apres mesure — leur
AUC tombe de 0.628 a 0.563 en passant au walk-forward, ils apprenaient une
structure propre au regime voisin.

    python mesure_grille_purgee.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from saint_core import FEATURE_COLS, ATR_PLANCHER_FRAC
from mesure_features import (barrieres, MAX_HOLD, SPREAD_BPS, SLIP_ENTREE_BPS,
                             SLIP_SORTIE_BPS, CACHE)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_BLOCS = 8
SELECTIVITES = (0.01, 0.02, 0.05)
SL_MULTS = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
RR_VALS = (1.0, 1.4, 2.0, 3.0)


def main() -> int:
    df = pd.read_pickle(CACHE).dropna(
        subset=FEATURE_COLS + ["atr_14"]).reset_index(drop=True)
    n = len(df)
    fin = int(n * 0.70)
    hi, lo, cl = (df[c].to_numpy(np.float64) for c in ("high", "low", "close"))
    atr = np.maximum(df["atr_14"].to_numpy(np.float64), ATR_PLANCHER_FRAC * cl)
    ds = (SPREAD_BPS / 1e4) * cl / 2.0
    se = (SLIP_ENTREE_BPS / 1e4) * cl
    ss = (SLIP_SORTIE_BPS / 1e4) * cl
    X = np.nan_to_num(np.column_stack(
        [df[c].to_numpy(np.float32) for c in FEATURE_COLS]))

    bornes = np.linspace(int(fin * 0.30), fin, N_BLOCS + 1).astype(int)
    print(f"{n:,} bougies, seules les {fin:,} premieres (70 %) sont utilisees.")
    print(f"{N_BLOCS} blocs  |  entrees tous les {MAX_HOLD} barres  |  "
          f"purge {MAX_HOLD} barres  |  friction complete\n")

    def evalue(sl_mult, rr):
        p_all, r_all, y_all, temps = [], [], [], []
        for b in range(N_BLOCS):
            a_va, b_va = bornes[b], bornes[b + 1]
            i_tr = np.arange(0, a_va - MAX_HOLD - MAX_HOLD, MAX_HOLD)
            i_va = np.arange(a_va, b_va - MAX_HOLD, MAX_HOLD)
            i_tr = i_tr[np.isfinite(atr[i_tr]) & (atr[i_tr] > 0)]
            i_va = i_va[np.isfinite(atr[i_va]) & (atr[i_va] > 0)]
            if len(i_tr) < 500 or len(i_va) < 40:
                continue
            for sens in (1, -1):
                r_t, m_t = barrieres(hi, lo, cl, atr, i_tr, sl_mult, rr, sens, ds, se, ss)
                r_v, m_v = barrieres(hi, lo, cl, atr, i_va, sl_mult, rr, sens, ds, se, ss)
                lr = LogisticRegression(max_iter=1000, n_jobs=1).fit(X[i_tr], r_t > 0)
                p_all.append(lr.predict_proba(X[i_va])[:, 1])
                r_all.append(r_v)
                y_all.append(r_v > 0)
                temps.append(m_v == 0)          # sorties par le TEMPS
        if not p_all:
            return None
        return (np.concatenate(p_all), np.concatenate(r_all),
                np.concatenate(y_all), np.concatenate(temps))

    print(f"{'SL':>5} {'R:R':>5} {'AUC':>7} {'tps':>6} | "
          + " | ".join(f"{'sel ' + str(int(100*q)) + '%':^28}" for q in SELECTIVITES))
    print(f"{'':>5} {'':>5} {'':>7} {'':>6} | "
          + " | ".join(f"{'trades':>6} {'WR':>6} {'equil':>6} {'t':>6}"
                       for _ in SELECTIVITES))
    print("-" * 120)

    gagnants = []
    for sl_mult in SL_MULTS:
        for rr in RR_VALS:
            out = evalue(sl_mult, rr)
            if out is None:
                continue
            p, r, y, par_temps = out
            auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
            cells = []
            for q in SELECTIVITES:
                sel = p >= np.quantile(p, 1.0 - q)
                rr_ = r[sel]
                if len(rr_) < 20:
                    cells.append(f"{'-':>6} {'-':>6} {'-':>6} {'-':>6}")
                    continue
                wr = 100 * float((rr_ > 0).mean())
                esp = float(rr_.mean())
                t = esp / (rr_.std(ddof=1) / np.sqrt(len(rr_)) + 1e-12)
                g = rr_[rr_ > 0].mean() if (rr_ > 0).any() else 0.0
                pe = abs(rr_[rr_ <= 0].mean()) if (rr_ <= 0).any() else 0.0
                eq = 100 * pe / (g + pe) if (g + pe) > 0 else float("nan")
                cells.append(f"{len(rr_):>6} {wr:5.1f}% {eq:5.1f}% {t:+6.2f}")
                if t > 2.0 and esp > 0:
                    gagnants.append((sl_mult, rr, q, len(rr_), wr, eq, esp, t))
            print(f"{sl_mult:5.1f} {rr:5.1f} {auc:7.4f} {100*par_temps.mean():5.0f}% | "
                  + " | ".join(cells))

    print("\n'tps' = part des trades sortis par le TEMPS (plafond a "
          f"{MAX_HOLD} barres), pas par SL/TP.")
    print("Au-dela de ~30 %, la configuration n'est plus une strategie a "
          "barrieres : c'est une detention fixe.")
    if gagnants:
        print(f"\n{len(gagnants)} case(s) avec esperance positive et t > 2 :")
        for sl_mult, rr, q, nn, wr, eq, esp, t in gagnants:
            print(f"   SL {sl_mult}xATR  R:R {rr}  sel {100*q:.0f}%  ->  "
                  f"{nn} trades  WR {wr:.1f}% (equilibre {eq:.1f}%)  "
                  f"E[R] {esp:+.4f}  t {t:+.2f}")
        print("\nBIAIS DE SELECTION : on lit le meilleur de "
              f"{len(SL_MULTS)*len(RR_VALS)*len(SELECTIVITES)} cases sur LES MEMES "
              "blocs.\nLe maximum d'estimations bruitees est biaise vers le haut. "
              "Une case isolee\na t > 2 par hasard environ une fois sur vingt. "
              "Seule la fenetre de test tranche.")
    else:
        print("\nAUCUNE case n'atteint une esperance positive significative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
