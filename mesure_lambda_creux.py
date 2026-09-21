# -*- coding: utf-8 -*-
"""QUELLE VALEUR DE `penalite_creux` CHOISIT QUEL LEVIER ?

POURQUOI IL FAUT LA REMESURER. La penalite valait `lambda (d_t^2 - d_{t-1}^2)`,
une difference de potentiel : sa magnitude PAR BARRE etait donc minuscule, et
6.5 y etait un reglage raisonnable. Elle vaut desormais `lambda d^2` A CHAQUE
BARRE — un cout d'etat, qui ne telescope pas et deplace donc reellement
l'optimum. A 25 % de creux, 6.5 couterait 0.41 par barre, quand le terme de
log-rendement en vaut une fraction. Garder 6.5 reviendrait a interdire toute
position.

CE QUE LA MESURE FAIT. Pour chaque lambda et chaque palier de budget, on joue
les MEMES episodes avec les MEMES entrees et on somme la recompense que
l'environnement rend reellement — pas une formule recopiee ici, qui pourrait
diverger de celle qui tourne. Le budget qui maximise ce total est celui qu'un
agent maximisant la recompense finirait par choisir.

ON CHERCHE le plus petit lambda qui fasse descendre ce choix sous 15 %, parce
qu'un lambda plus grand que necessaire achete de la prudence avec du
rendement.

LA FENETRE DE TEST N'EST PAS LUE.

    python mesure_lambda_creux.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

import instruments as I
import training as T
from saint_core import BUDGETS_PART, FEATURE_COLS

LONGUEUR = 4000
GRAINES = tuple(range(1, 7))
TAUX = 0.25
LAMBDAS = (0.0, 0.005, 0.015, 0.05, 0.15)


def _equite(env):
    p = float(env.data.close[env.idx - 1]) if env.idx > 0 else 0.0
    return env.capital + env._latent_at_bid(p)


def main() -> int:
    cfg0 = T.PPOConfig()
    cfg0.episode_length = LONGUEUR + 10
    cfg0.veto_tendance = ""
    df = pd.read_pickle(I.INSTRUMENTS[cfg0.symbol]["cache"])
    va = int(len(df) * 0.55)
    stats = T.compute_and_save_global_norm_stats(df.iloc[:va], FEATURE_COLS,
                                                 path=None)
    data = T.MarketData(df.iloc[:va].reset_index(drop=True), FEATURE_COLS,
                        stats)
    print(f"{len(GRAINES)} episodes de {LONGUEUR:,} barres par (lambda, budget)")
    print(f"recompense sommee TELLE QUE L'ENVIRONNEMENT LA REND")
    print(f"La fenetre de test n'est pas lue.\n")
    print(f"{'lambda':>8} " + "".join(f"{100*b:>9.0f}%" for b in BUDGETS_PART)
          + f" {'choisi':>8} {'creux max':>11} {'gagnants punis':>15}")
    print("-" * 84)

    for lam in LAMBDAS:
        tot, dmax, pun = {}, {}, {}
        for b in BUDGETS_PART:
            r_s, d_s, punis, gagn = [], [], 0, 0
            for g in GRAINES:
                np.random.seed(g)
                cfg = T.PPOConfig()
                cfg.episode_length = LONGUEUR + 10
                cfg.veto_tendance = ""
                cfg.penalite_creux = lam
                env = T.BTCTradingEnvDiscrete(data, cfg)
                T.reset_au_depart(env, 3000 + g * 2500)
                env.set_budget_part(b)
                rng = np.random.default_rng(g)
                c = [_equite(env)]
                somme = 0.0
                # CE QUI COMPTE VRAIMENT : un trade GAGNANT qui finit avec une
                # recompense NEGATIVE apprend a l'acteur que gagner est
                # mauvais. C'est l'invariant de signe de `test_concurrence`, et
                # c'est lui que le cout de chemin peut retourner.
                _vus, _cum = len(env.trades_pnl), 0.0
                for _ in range(LONGUEUR):
                    a = 0 if (rng.random() < TAUX and env.peut_entrer()) else 2
                    _, r, done, _, _ = env.step(a)
                    somme += float(r)
                    _cum += float(r)
                    if len(env.trades_pnl) > _vus:
                        for _pnl in env.trades_pnl[_vus:]:
                            if _pnl > 0:
                                gagn += 1
                                punis += (_cum < 0)
                        _vus = len(env.trades_pnl)
                        _cum = 0.0
                    c.append(_equite(env))
                    if done:
                        break
                arr = np.asarray(c, float)
                pic = np.maximum.accumulate(np.maximum(arr, 1e-9))
                r_s.append(somme)
                d_s.append(float(np.clip(1.0 - arr / pic, 0, 1).max()))
            tot[b] = float(np.mean(r_s))
            dmax[b] = float(np.mean(d_s))
            pun[b] = 100.0 * punis / max(gagn, 1)
        best = max(tot, key=tot.get)
        print(f"{lam:>8.3f} " + "".join(f"{tot[b]:>10.1f}" for b in BUDGETS_PART)
              + f" {100*best:>7.0f}% {100*dmax[best]:>10.1f}% "
              f"{pun[best]:>14.1f}%")
    print("-" * 84)
    print("LECTURE. Chaque ligne donne la recompense totale par palier, et le")
    print("palier qu'un agent maximisant la recompense finirait par choisir.")
    print("`gagnants punis` est la part des trades GAGNANTS dont la recompense")
    print("cumulee est NEGATIVE. Au-dela de quelques pourcents, le cout de")
    print("chemin noie le resultat du trade et l'acteur apprend que gagner est")
    print("mauvais : c'est la limite haute de lambda, et elle prime sur le")
    print("choix du budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
