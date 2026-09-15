"""La strategie "bornes d'un range" simulee telle qu'elle est ecrite.

CE QUE CE FICHIER TESTE, ET POURQUOI IL FALLAIT UN FICHIER A PART. Mesurer les
features de range contre une cible SL 2xATR / TP 4xATR ne teste pas la
strategie : elle repose sur une cible STRUCTURELLE, pas sur des multiples
d'ATR. Les regles simulees ici sont celles des documents :

  ENTREE    en cloture du chandelier qui SUIT le ressaut ou le soulevement.
  SL        quelques points au-dela de la meche du ressaut / soulevement.
  TP        quelques points avant la borne OPPOSEE du range.
  FILTRE    ratio au moins 2/1, sinon on ne prend pas.
  SORTIE    en cas de cassure de la borne, meme minime, meme si le stop n'est
            pas touche, on sort.
  ABSTENTION  pas de ressaut ni de soulevement : pas de trade. C'est le point
            que le filtre par rang ne savait pas faire.

LE CALCUL QUI DECIDE DE TOUT. La friction est un MONTANT FIXE : demi-spread
aller-retour plus slippage d'entree et de sortie, soit environ 37 $ sur BTCUSD
a 66 000 $. L'unite de risque, elle, est la distance au stop.

    stop structurel serre (~1 ATR = 31 $)  ->  friction = 1.2 R
    stop de 2 ATR (61 $)                   ->  friction = 0.60 R
    meme stop sur H1 (ATR ~8x plus grand)  ->  friction = 0.15 R

Sur M1 un stop structurel serre est donc devore avant meme d'avoir raison.
C'est pour cela que les documents disent : "court terme, chercher des ranges en
UT 15 min ; day trading, 15 min a 1H ; swing, H4 au daily". La strategie n'est
pas concue pour la minute.

Ce script mesure donc la meme strategie sur PLUSIEURS unites de temps, pour
voir a partir de laquelle la friction cesse de la tuer.

    python strategie_range.py
"""

import numpy as np
import pandas as pd

from features_range import ajoute_features_range, FENETRE_RANGE, TOL_BORNE
from mesure_features import (SPREAD_BPS, SLIP_ENTREE_BPS, SLIP_SORTIE_BPS,
                             CACHE)

# Marge au-dela de la meche pour le stop, et en deca de la borne pour l'objectif,
# en fraction de la largeur du range. "quelques points" dans les documents.
MARGE_SL = 0.02
MARGE_TP = 0.05
RATIO_MIN = 2.0
MAX_BARRES = 240          # au-dela, on cloture au marche


def _atr(d):
    """ATR(14) recalcule A L'ECHELLE de la serie fournie.

    Indispensable : reutiliser l'ATR M1 sur une serie H1 rapporterait la
    friction a la mauvaise unite de risque, et c'est precisement ce rapport
    que ce script cherche a mesurer.
    """
    h, l, c = d["high"], d["low"], d["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    d["atr_14"] = tr.rolling(14).mean()
    return d


def resample(df, regle):
    """Reagrege en OHLC. La colonne `time` doit etre un datetime."""
    if regle is None:
        return _atr(df.copy())
    r = (df.set_index("time")[["open", "high", "low", "close"]]
           .resample(regle)
           .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
           .dropna().reset_index())
    return _atr(r)


def simule(df, nom_tf):
    """Rend les rendements NETS, en unites de risque, des trades pris."""
    df = df.copy()
    df, _ = ajoute_features_range(df)
    c = df["close"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    atr = df["atr_14"].to_numpy(np.float64)

    haute = df["close"].rolling(FENETRE_RANGE).max().to_numpy(np.float64)
    basse = df["close"].rolling(FENETRE_RANGE).min().to_numpy(np.float64)
    largeur = haute - basse
    ress = df["rng_ressaut"].to_numpy() > 0
    soul = df["rng_soulevement"].to_numpy() > 0

    # Friction, en prix. Elle ne depend PAS du stop : c'est un montant fixe
    # que l'on paie a l'aller et au retour.
    demi_sp = (SPREAD_BPS / 1e4) * c / 2.0
    slip_e = (SLIP_ENTREE_BPS / 1e4) * c
    slip_s = (SLIP_SORTIE_BPS / 1e4) * c

    n = len(c)
    resultats, refus_ratio, occasions = [], 0, 0
    # UNE POSITION A LA FOIS. Deux raisons, et la seconde est decisive :
    #   - c'est ce que fait le systeme reel, qui ne cumule pas les entrees ;
    #   - tester une entree a CHAQUE barre alors qu'un trade dure jusqu'a 240
    #     barres fait se recouvrir les resultats et gonfle le t. Mesure de ce
    #     biais cette nuit : un modele passait de 39.7 % a 58.0 % de reussite
    #     apparente. On saute donc jusqu'a la sortie avant de chercher le
    #     signal suivant.
    libre_a = 0

    for i in range(FENETRE_RANGE + 1, n - MAX_BARRES - 2):
        if i < libre_a:
            continue
        if not np.isfinite(largeur[i]) or largeur[i] <= 0:
            continue
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        sens = 1 if ress[i] else (-1 if soul[i] else 0)
        if sens == 0:
            continue
        occasions += 1

        # Entree en CLOTURE DU CHANDELIER SUIVANT, degradee par la friction.
        j = i + 1
        entree = c[j] + sens * (demi_sp[j] + slip_e[j])

        if sens == 1:
            sl = l[i] - MARGE_SL * largeur[i]          # sous la meche du ressaut
            tp = haute[i] - MARGE_TP * largeur[i]      # avant la borne opposee
        else:
            sl = h[i] + MARGE_SL * largeur[i]          # au-dessus de la meche
            tp = basse[i] + MARGE_TP * largeur[i]

        risque = abs(entree - sl)
        gain_vise = abs(tp - entree)
        if risque <= 0 or gain_vise / risque < RATIO_MIN:
            refus_ratio += 1
            continue

        # Course des barrieres, a partir de la barre suivant l'entree.
        sortie, motif = None, "temps"
        for k in range(j + 1, min(j + 1 + MAX_BARRES, n)):
            if sens == 1:
                if l[k] <= sl:
                    sortie, motif = sl, "sl"; break
                if h[k] >= tp:
                    sortie, motif = tp, "tp"; break
                # "En cas de cassure de la borne, meme minime, il faut sortir."
                if c[k] < basse[i]:
                    sortie, motif = c[k], "cassure"; break
            else:
                if h[k] >= sl:
                    sortie, motif = sl, "sl"; break
                if l[k] <= tp:
                    sortie, motif = tp, "tp"; break
                if c[k] > haute[i]:
                    sortie, motif = c[k], "cassure"; break
        if sortie is None:
            sortie, k = c[min(j + MAX_BARRES, n - 1)], min(j + MAX_BARRES, n - 1)

        # Slippage de sortie : defavorable sauf si l'objectif limite est touche.
        if motif != "tp":
            sortie -= sens * slip_s[k]
        sortie -= sens * demi_sp[k]

        resultats.append(sens * (sortie - entree) / risque)
        libre_a = k + 1          # on ne rouvre qu'apres la sortie

    r = np.array(resultats, float)
    return r, occasions, refus_ratio


def main() -> int:
    brut = pd.read_pickle(CACHE)[["time", "open", "high", "low", "close"]].copy()
    brut["time"] = pd.to_datetime(brut["time"])
    # Fenetre d'ENTRAINEMENT uniquement : la fenetre de test reste intouchee.
    brut = brut.iloc[:int(len(brut) * 0.70)]

    print(f"{len(brut):,} bougies M1 (70 % initiaux, test intouche)")
    print(f"regles : entree en cloture du chandelier suivant, SL au-dela de la "
          f"meche,\n         TP avant la borne opposee, ratio minimum "
          f"{RATIO_MIN}/1, sortie sur cassure\n")
    print(f"{'UT':>6} {'bougies':>10} {'occasions':>10} {'refus ratio':>12} "
          f"{'trades':>8} {'WR':>7} {'E[R]':>9} {'t':>7} {'friction':>9}")
    print("-" * 88)

    for nom, regle in (("M1", None), ("M5", "5min"), ("M15", "15min"),
                       ("H1", "1h"), ("H4", "4h")):
        d = resample(brut, regle)
        if len(d) < 5000:
            continue
        r, occ, refus = simule(d, nom)
        if len(r) < 30:
            print(f"{nom:>6} {len(d):10,} {occ:10,} {refus:12,} "
                  f"{len(r):8,}   trop peu de trades")
            continue
        wr = 100 * float((r > 0).mean())
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)) + 1e-12)
        atr_med = float(np.nanmedian(d["atr_14"]))
        prix_med = float(np.nanmedian(d["close"]))
        fric = (2 * SPREAD_BPS / 2 + SLIP_ENTREE_BPS + SLIP_SORTIE_BPS) / 1e4 * prix_med
        print(f"{nom:>6} {len(d):10,} {occ:10,} {refus:12,} {len(r):8,} "
              f"{wr:6.1f}% {r.mean():+9.4f} {t:+7.2f} {fric/atr_med:8.2f}R")

    print("\n'friction' = cout fixe rapporte a UN ATR de l'unite de temps.")
    print("C'est le chiffre qui decide : un stop structurel vaut quelques ATR,")
    print("donc une friction de 1.2 R par ATR rend la strategie injouable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
