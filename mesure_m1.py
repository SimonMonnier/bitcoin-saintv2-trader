# -*- coding: utf-8 -*-
"""LE M1 PEUT-IL PAYER SA FRICTION ?

LA QUESTION N'EST PAS DE MODELISATION, ELLE EST ARITHMETIQUE. La friction d'un
aller-retour est un montant FIXE en dollars — ecart, glissement, commission.
Exprimee dans l'unite qui compte, le R, elle vaut :

    friction / R = cout fixe en dollars / largeur du stop en dollars
                 = cout fixe / (N x ATR)

Le numerateur ne depend pas de l'unite de temps. Le denominateur, si :
l'ATR d'une bougie M1 est bien plus petit que celui d'une M5. Raccourcir les
trades RETRECIT donc le stop, et la meme friction pese plus lourd. Il n'y a
aucun reglage, aucune architecture et aucune recompense qui change cela.

CE QUE LA MESURE FAIT. Elle estime l'ATR relatif a chaque echelle depuis le
cache — M5, M15, H1, H4 — ajuste la loi d'echelle qui les relie, et en deduit
l'ATR du M1. Puis elle calcule, pour plusieurs largeurs de stop, la friction
par R et donc l'AVANTAGE MINIMAL qu'un modele devrait avoir pour seulement
rentrer dans ses frais.

LE REPERE A GARDER EN TETE : l'avantage le plus fort jamais mesure dans ce
depot vaut +0.3 a +0.4 R au-dessus du hasard, sur un fold sur trois.

    python mesure_m1.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

import instruments as I
import training as T


def main() -> int:
    cfg = T.PPOConfig()
    ps = I.INSTRUMENTS[cfg.symbol]
    df = pd.read_pickle(ps["cache"])
    n = int(len(df) * 0.70)
    d = df.iloc[:n]

    # ATR RELATIF PAR ECHELLE, mesure et non suppose.
    ech = [("M5", 1), ("M15", 3), ("H1", 12), ("H4", 48)]
    rel = {}
    o = d[["time", "open", "high", "low", "close"]].set_index("time")
    for nom, k in ech:
        if k == 1:
            r = float(np.median(d["atr_14"] / d["close"]))
        else:
            g = o.resample(f"{5*k}min").agg(
                {"high": "max", "low": "min", "close": "last"}).dropna()
            tr = (g["high"] - g["low"]).rolling(14).mean()
            r = float(np.nanmedian(tr / g["close"]))
        rel[nom] = r
        print(f"  ATR relatif {nom:>3} : {1e4*r:>7.2f} points de base")

    # LOI D'ECHELLE. Si la volatilite suit une racine du temps, l'ATR relatif
    # vaut c x minutes^alpha. On ajuste alpha plutot que de supposer 0.5 : le
    # marche n'a aucune obligation de s'y conformer.
    mins = np.array([5, 15, 60, 240], float)
    y = np.array([rel[n_] for n_, _ in ech], float)
    alpha, logc = np.polyfit(np.log(mins), np.log(y), 1)
    atr_m1 = float(np.exp(logc + alpha * np.log(1.0)))
    print(f"\n  loi d'echelle ajustee : ATR ~ minutes^{alpha:.3f}  "
          f"(racine du temps = 0.500)")
    print(f"  ATR relatif M1 extrapole : {1e4*atr_m1:>7.2f} points de base "
          f"({rel['M5']/atr_m1:.2f}x plus petit qu'en M5)")

    # LA FRICTION, dans les memes termes que `prepare_m5` : ecart + glissement
    # + commission, en points de base du prix.
    cout_bps = float(ps["spread_bps"]) + 3.0
    print(f"\n  friction d'un aller-retour : {cout_bps:.2f} points de base "
          f"du prix (ecart {ps['spread_bps']:.2f} + 3.0 de glissement/commission)")

    print(f"\n{'='*78}")
    print(f"{'echelle':>8} {'stop':>8} {'stop en bps':>12} {'friction/R':>12} "
          f"{'avantage requis':>17}")
    print("-" * 78)
    for nom, a in (("M5", rel["M5"]), ("M1", atr_m1)):
        for mult in (2.0, 4.0, 8.0, 10.0, 20.0):
            largeur = mult * a
            f = cout_bps / (1e4 * largeur)
            marque = "  <- en vigueur" if (nom == "M5" and mult == 10.0) else ""
            print(f"{nom:>8} {mult:>6.0f}x {1e4*largeur:>11.1f} "
                  f"{f:>12.3f} {f:>+17.3f}{marque}")
        print("-" * 78)
    print(f"REPERE : le meilleur avantage jamais mesure ici vaut +0.3 a +0.4 R")
    print(f"au-dessus du hasard, sur un fold sur trois.")
    print(f"\nLECTURE. `avantage requis` est ce qu'il faut gagner PAR TRADE,")
    print(f"en R, uniquement pour couvrir l'aller-retour. Un stop deux fois")
    print(f"plus etroit double cette exigence — c'est une division, pas une")
    print(f"hypothese. Et un stop assez large pour ramener la friction du M1")
    print(f"au niveau du M5 donne un trade qui dure aussi longtemps qu'en M5 :")
    print(f"on aurait change d'echelle sans raccourcir quoi que ce soit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
