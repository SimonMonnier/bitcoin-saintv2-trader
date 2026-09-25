# -*- coding: utf-8 -*-
"""Y a-t-il, sur le BTC en minute, un avantage PREDICTIBLE qui paie le spread ?
Et a quel horizon ?

POURQUOI CE FICHIER. Le 2026-09-25, cinq runs de suite ont bute au meme
endroit : le classement des entrees ne bat pas le hasard une fois le spread
paye — a 60 minutes comme a 15. Avant de regler encore la sortie, les cotes
ou le loyer, il faut savoir si l'avantage EXISTE dans ces features a ces
horizons. Si un modele simple n'en trouve pas, le reseau n'en trouvera pas
non plus ; s'il en trouve, c'est l'entrainement du reseau qu'il faut revoir.

LE PROTOCOLE, et il reprend exactement ce que les tetes apprennent :

  - les MEMES features (`FEATURE_COLS`), lues a la barre qui precede
    l'occasion, comme l'observation de l'environnement ;
  - la MEME cible : `cibles_m1.rendements_entree`, rendement net du spread
    REEL de chaque barre, en points de base, pour l'achat et pour la vente ;
  - un LightGBM par cote et par horizon, appris sur le TRAIN du fold 1,
    arrete sur la CALIBRATION, juge sur la VALIDATION ;
  - LE TEST N'EST JAMAIS CHARGE : les donnees sont coupees a la fin de la
    validation du fold 1, avant toute autre operation.

CE QUE LE TABLEAU DIT, en points de base NETS du spread (comparables d'un
horizon a l'autre, contrairement aux R, qui se mettent a l'echelle du
mouvement attendu) :

  IC        correlation de rang entre la prediction et le rendement realise
  plancher  l'IC qu'on obtient par hasard : meme prediction, cible decalee
            dans le temps (rotation), 95e centile de |IC| sur 20 decalages
  hasard    ce que rapporte une occasion prise au hasard
  sommet    ce que rapportent les 5 % d'occasions que le modele prefere,
            en choisissant pour chacune le cote qu'il prefere
  ecart     sommet - hasard : ce que le tri apporte
  gagnant   part du sommet qui finit positive

    python mesure_horizon_btc.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

import cibles_m1 as CM
from saint_core import FEATURE_COLS

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# LES BORNES DU FOLD 1, telles que `training` les imprime :
#   train=538579, calib=51409, val=95476, test=97923 (start=0)
FIN_TRAIN = 538_579
FIN_CALIB = FIN_TRAIN + 51_409
FIN_VAL = FIN_CALIB + 95_476          # 685 464 : RIEN au-dela n'est lu
HORIZONS = (5, 10, 15, 30, 60, 120, 240)
PAS = 4                               # une occasion toutes les 4 barres
Q = 0.05                              # la selectivite de la validation
SYMBOLE = "BTCUSD"


def _occasions(debut, fin, h):
    """Occasions dont la cible se resout AVANT la fin de la fenetre."""
    return np.arange(max(debut, 200), fin - h - 2, PAS)


def _ic(a, b):
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m]).correlation) if m.sum() > 100 else float("nan")


def main() -> int:
    import lightgbm as lgb

    t0 = time.time()
    d = pd.read_pickle("data_cache_BTCUSD_M1.pkl")
    n_total = len(d)
    d = d.iloc[:FIN_VAL].reset_index(drop=True)
    assert len(d) == FIN_VAL, "le cache est plus court que le fold 1"
    print(f"cache {n_total:,} barres, coupe a {FIN_VAL:,} : le TEST n'est pas lu")
    print(f"train [0, {FIN_TRAIN:,})   calibration [{FIN_TRAIN:,}, "
          f"{FIN_CALIB:,})   validation [{FIN_CALIB:,}, {FIN_VAL:,})")
    sp = d["spread_bar"].to_numpy(np.float64) if "spread_bar" in d else None
    if sp is not None:
        manque = ~(np.isfinite(sp) & (sp > 0))
        print(f"barres sans spread reel (repli sur la constante) : "
              f"{100 * manque.mean():.2f} %")
    X = d[list(FEATURE_COLS)].to_numpy(np.float32)
    print(f"{X.shape[1]} features, chargees en {time.time() - t0:.0f} s\n")

    lignes = []
    entete = ("  %-7s %8s %8s %8s %9s %9s %8s %8s"
              % ("horizon", "IC ach", "IC ven", "plancher", "hasard",
                 "sommet", "ecart", "gagnant"))
    print(entete)
    print("  " + "-" * 72)
    rng = np.random.default_rng(0)
    for h in HORIZONS:
        itr = _occasions(0, FIN_TRAIN, h)
        ica = _occasions(FIN_TRAIN, FIN_CALIB, h)
        iva = _occasions(FIN_CALIB, FIN_VAL, h)
        res = {}
        preds = {}
        for cote in ("achat", "vente"):
            cibles = {}
            for nom, ii in (("tr", itr), ("ca", ica), ("va", iva)):
                a, v, _, _ = CM.rendements_entree(d, ii, horizon=h,
                                                  symbole=SYMBOLE)
                cibles[nom] = a if cote == "achat" else v
            # LA CIBLE D'APPRENTISSAGE EST ECRETEE a 1-99 % : quelques
            # minutes de krach ne doivent pas dicter tout le modele. Le
            # jugement, lui, se fait sur le rendement brut.
            ytr = cibles["tr"]
            ok = np.isfinite(ytr)
            lo, hi = np.nanpercentile(ytr, [1, 99])
            m = lgb.LGBMRegressor(
                n_estimators=600, learning_rate=0.03, num_leaves=31,
                min_child_samples=400, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.5, reg_lambda=5.0, verbose=-1)
            okc = np.isfinite(cibles["ca"])
            m.fit(X[itr[ok] - 1], np.clip(ytr[ok], lo, hi),
                  eval_set=[(X[ica[okc] - 1], cibles["ca"][okc])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            p = m.predict(X[iva - 1])
            preds[cote] = p
            res[cote] = cibles["va"]
            res["arbres_" + cote] = int(m.best_iteration_ or m.n_estimators)

        ya, yv = res["achat"], res["vente"]
        pa, pv = preds["achat"], preds["vente"]
        ic_a, ic_v = _ic(pa, ya), _ic(pv, yv)
        # LE PLANCHER : la meme prediction contre une cible decalee.
        pl = []
        for _ in range(20):
            k = int(rng.integers(2_000, len(ya) - 2_000))
            pl.append(abs(_ic(pa, np.roll(ya, k))))
            pl.append(abs(_ic(pv, np.roll(yv, k))))
        plancher = float(np.nanpercentile(pl, 95))
        # LE TRI BILATERAL : pour chaque occasion le cote prefere, puis les
        # 5 % d'occasions ou cette preference est la plus forte.
        ok = np.isfinite(ya) & np.isfinite(yv)
        choix = np.where(pa >= pv, ya, yv)[ok]
        conv = np.maximum(pa, pv)[ok]
        k = max(int(round(Q * ok.sum())), 50)
        top = np.argsort(conv)[-k:]
        hasard = float(np.mean(np.concatenate([ya[ok], yv[ok]])))
        sommet = float(np.mean(choix[top]))
        gagnant = float(np.mean(choix[top] > 0))
        l = ("  %-7s %+8.4f %+8.4f %8.4f %+9.2f %+9.2f %+8.2f %7.1f%%"
             % (f"{h} min", ic_a, ic_v, plancher, hasard, sommet,
                sommet - hasard, 100 * gagnant))
        print(l + f"   ({res['arbres_achat']}/{res['arbres_vente']} arbres,"
                  f" {ok.sum():,} occasions de validation)", flush=True)
        lignes.append(l)
    print(f"\n{time.time() - t0:.0f} s au total")
    print("\nLecture : un IC au-dessus du plancher dit qu'il y a de "
          "l'information ; un sommet POSITIF dit qu'elle paie le spread.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
