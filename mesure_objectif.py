# -*- coding: utf-8 -*-
"""QUEL OBJECTIF CHOISIT QUEL LEVIER ?

LE CONSTAT QUI MOTIVE CETTE MESURE. exec31 : le modele gagne de l'argent en
validation et traverse des creux de 63 a 94 %. Aucune valeur de
`penalite_creux` ne l'a jamais fait baisser, et il y a une raison de principe :
la penalite est a POTENTIEL, de la forme Phi(s') - Phi(s). Ng, Harada &
Russell (1999) ont montre que cette classe de shaping ne change PAS la
politique optimale — elle redistribue le meme total dans le temps. On a donc
ajoute une penalite de creux qui, par construction, ne peut pas modifier le
comportement.

L'AUTRE MOITIE DU PROBLEME est que la recompense contient un terme LINEAIRE en
levier : `realized_trade / risk_amount`, le resultat du trade en R. Deux
positions a +1 R rapportent +2, quatre en rapportent +4. Rien n'y est concave,
donc plus de levier est toujours mieux — jusqu'a la ruine, qui est le seul
frein.

CE QU'ON MESURE ICI. Pour chaque palier de budget, on joue les memes episodes
et on note ce que chaque objectif candidat en pense :

  SOMME DES R          l'objectif actuel, lineaire en levier
  LOG-RICHESSE         somme des log(equite_t / equite_t-1) : le critere de
                       croissance optimale. Il est CONCAVE en levier — c'est
                       le critere de Kelly — et la ruine y vaut moins l'infini,
                       donc il l'evite sans qu'on ait a la penaliser.
  LOG MOINS LE TEMPS   log-richesse moins un cout par barre passee en creux.
  PASSE EN CREUX       Contrairement a la penalite a potentiel, ce cout n'est
                       PAS telescopique : il change vraiment l'optimum.

SI LES TROIS DESIGNENT LE MEME BUDGET, changer la recompense ne servira a
rien et il faudra chercher ailleurs. S'ils divergent, l'ecart dit exactement
combien de levier l'objectif actuel achete en trop.

LA FENETRE DE TEST N'EST PAS LUE.

    python mesure_objectif.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

import cibles as C
import instruments as I
import training as T
from saint_core import BUDGETS_PART, FEATURE_COLS

LONGUEUR = 8000
GRAINES = tuple(range(1, 13))
TAUX = 0.25          # entrees neutres, identiques d'un budget a l'autre
LAMBDA_TEMPS = 0.5   # cout par barre passee en creux, x d^2


def _equite(env):
    p = float(env.data.close[env.idx - 1]) if env.idx > 0 else 0.0
    return env.capital + env._latent_at_bid(p)


def main() -> int:
    cfg = T.PPOConfig()
    cfg.episode_length = LONGUEUR + 10
    cfg.veto_tendance = ""          # on isole le levier, pas l'entree
    df = pd.read_pickle(I.INSTRUMENTS[cfg.symbol]["cache"])
    va = int(len(df) * 0.55)
    stats = T.compute_and_save_global_norm_stats(df.iloc[:va], FEATURE_COLS,
                                                 path=None)
    data = T.MarketData(df.iloc[:va].reset_index(drop=True), FEATURE_COLS,
                        stats)
    print(f"{len(GRAINES)} episodes de {LONGUEUR:,} barres par palier, "
          f"entrees neutres a {100*TAUX:g} %")
    print(f"La fenetre de test n'est pas lue.\n")
    print(f"{'budget':>8} {'somme R':>10} {'log-richesse':>14} "
          f"{'log - temps':>13} {'creux max':>11} {'creux moyen':>12} "
          f"{'ruines':>8}")
    print("-" * 82)

    res = {}
    for b in BUDGETS_PART:
        sR, sLog, sPen, dmax, dmoy, morts = [], [], [], [], [], 0
        for g in GRAINES:
            np.random.seed(g)
            env = T.BTCTradingEnvDiscrete(data, cfg)
            T.reset_au_depart(env, 3000 + g * 2500)
            env.set_budget_part(b)
            rng = np.random.default_rng(g)
            e0 = _equite(env)
            courbe = [e0]
            somme_r = 0.0
            _vus = len(env.trades_pnl)
            k = 0
            while k < LONGUEUR:
                a = 0 if (rng.random() < TAUX and env.peut_entrer()) else 2
                _, _, done, _, info = env.step(a)
                k += 1
                courbe.append(_equite(env))
                # LE TERME EN R DE LA RECOMPENSE ACTUELLE. `info` ne le porte
                # pas ; l'environnement le tient dans `trades_pnl`, un par
                # trade ferme. On somme les nouveaux depuis la barre
                # precedente, exactement comme la recompense le fait.
                _n = len(env.trades_pnl)
                for _pnl in env.trades_pnl[_vus:_n]:
                    somme_r += float(np.clip(
                        _pnl / max(getattr(env, "risk_amount", 1e-8), 1e-8),
                        -3.0, 3.0))
                _vus = _n
                if done:
                    morts += 1
                    break
            c = np.asarray(courbe, float)
            pic = np.maximum.accumulate(np.maximum(c, 1e-9))
            d = np.clip(1.0 - c / pic, 0.0, 1.0)
            # LOG-RICHESSE : la ruine y vaut moins l'infini, on la borne au
            # plancher qu'un compte reel atteint (equite <= 0 = fin).
            fin = max(float(c[-1]), 1e-9)
            sLog.append(float(np.log(fin / max(e0, 1e-9))))
            sPen.append(sLog[-1] - LAMBDA_TEMPS * float(np.mean(d * d)) * len(d) / 1000.0)
            sR.append(somme_r)
            dmax.append(float(d.max()))
            dmoy.append(float(d.mean()))
        res[b] = (np.mean(sR), np.mean(sLog), np.mean(sPen),
                  np.mean(dmax), np.mean(dmoy), morts)
        print(f"{100*b:>6.0f} % {res[b][0]:>10.1f} {res[b][1]:>+14.3f} "
              f"{res[b][2]:>+13.3f} {100*res[b][3]:>10.1f}% "
              f"{100*res[b][4]:>11.1f}% {morts:>4}/{len(GRAINES)}")
    print("-" * 82)
    noms = ("SOMME DES R (objectif actuel)", "LOG-RICHESSE",
            "LOG MOINS TEMPS EN CREUX")
    print()
    for i, nom in enumerate(noms):
        best = max(res, key=lambda b: res[b][i])
        print(f"{nom:<32} prefere le budget {100*best:>3.0f} %  "
              f"(creux max {100*res[best][3]:.0f} %)")
    print()
    print("LECTURE. Si `SOMME DES R` prefere un budget plus large que")
    print("`LOG-RICHESSE`, c'est l'objectif actuel qui achete ce levier — pas")
    print("le modele qui se trompe. Aucune penalite A POTENTIEL ne corrigera")
    print("cela, puisqu'elle ne deplace pas l'optimum.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
