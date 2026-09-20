# -*- coding: utf-8 -*-
"""OU FAUT-IL CESSER D'ACHETER QUAND LA TENDANCE FAIBLIT ?

`mesure_tendance.py` a montre que le momentum sur un mois separe les deux sens
(+0.456 R a 2.1 sigma) la ou aucun detecteur Ichimoku n'y parvient, et que la
separation croit monotonement avec l'horizon — ce qui convainc plus que le
sigma, parce que du bruit ne se range pas par horizon.

Il a aussi montre ce que le regime NE dit PAS : meme en tendance baissiere,
l'achat reste meilleur que la vente (+0.049 contre -0.244 R). Ce n'est donc
pas un aiguillage de sens mais une JAUGE DE FORCE pour les achats.

LA QUESTION QUI RESTE est celle du seuil : a partir de quel momentum l'achat
cesse-t-il de valoir la peine ? Le tercile etait un decoupage commode, pas une
reponse.

CE QUE LA TABLE DOIT MONTRER, et le piege qu'elle doit eviter. Couper haut
fait monter le rendement PAR TRADE mecaniquement — on ne garde que le dessus.
Mais on garde aussi de moins en moins d'occasions, et une occasion non prise
ne rapporte rien. On affiche donc les deux : ce que rapporte une occasion
retenue, ET ce que la regle rapporte RAMENE A TOUTES LES OCCASIONS, prises ou
non. La seconde colonne est celle qui decide.

LA FENETRE DE TEST N'EST PAS LUE.

    python mesure_seuil_tendance.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

import cibles as C
import instruments as I
import training as T

PAS = 12
COLONNE = "tend_mom_mois"


def main() -> int:
    cfg = T.PPOConfig()
    df = pd.read_pickle(I.INSTRUMENTS[cfg.symbol]["cache"])
    n_fin = int(len(df) * 0.55) + int(len(df) * 0.15)
    idx = np.arange(cfg.lookback + 1, n_fin - C.BORNE_DEFAUT - 2, PAS)

    ra, _, ta, _ = C.rendements(df, idx, cfg, indicateur=False, durees=True)
    a = np.asarray(ra, float)
    duree = float(np.nanmedian(np.asarray(ta, float)))
    n_indep = (n_fin - cfg.lookback) / max(duree, 1.0)
    x = df[COLONNE].to_numpy(float)[idx]
    ok = np.isfinite(a) & np.isfinite(x)
    a, x = a[ok], x[ok]
    base = float(a.mean())

    print(f"{len(a):,} occasions, ~{n_indep:,.0f} independantes")
    print(f"sans aucun filtre, un ACHAT rapporte {base:+.3f} R")
    print(f"La fenetre de test n'est pas lue.\n")
    print(f"{'seuil sur ' + COLONNE:<26} {'gardees':>9} {'part':>7} "
          f"{'R par trade':>12} {'R par occasion':>15} {'sigma':>7}")
    print("-" * 82)
    for s in (-np.inf, -0.05, -0.02, 0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        m = x >= s
        if m.sum() < 300:
            continue
        v = a[m]
        par_trade = float(v.mean())
        # RAMENE A TOUTES LES OCCASIONS : une occasion refusee rapporte zero.
        # C'est ce que la regle produit reellement sur la meme periode.
        par_occasion = float(v.sum() / len(a))
        k = max(m.sum() * n_indep / len(a), 1.0)
        se = float(v.std()) / np.sqrt(k)
        sg = (par_trade - base) / se if se > 0 else 0.0
        nom = "aucun filtre" if s == -np.inf else f"momentum >= {100*s:+.0f} %"
        print(f"{nom:<26} {int(m.sum()):>9,} {100*m.mean():>6.1f}% "
              f"{par_trade:>+12.3f} {par_occasion:>+15.3f} {sg:>+7.1f}")
    print("-" * 82)
    print("LECTURE. `R par trade` monte mecaniquement quand on coupe haut :")
    print("on ne garde que le dessus. `R par occasion` tient compte de ce")
    print("qu'on laisse passer, et c'est la seule colonne qui compare des")
    print("regles entre elles a periode egale. Un seuil qui fait monter la")
    print("premiere en faisant baisser la seconde appauvrit le systeme.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
