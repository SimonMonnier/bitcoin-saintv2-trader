# -*- coding: utf-8 -*-
"""Quelle regle de sortie donne assez d'occasions INDEPENDANTES pour mesurer ?

LE PROBLEME, ET IL PRIME SUR TOUS LES AUTRES. Une position tient 396 barres
en mediane. Sur une fenetre de validation de 47 908 barres, le sommet a 5 %
retient 163 occasions dont seulement ~22 DISJOINTES. La barre de bruit du
`sommet` vaut alors 0.63 R — mesuree par rotation circulaire de la cible
sous le score reel d'une colonne, donc en conservant son groupement
temporel.

    TOUT ce qu'exec69 a mesure tient sous UN ecart-type de bruit :
    exces +0.299 (0.48 sigma), meilleur quart +0.594 (0.95 sigma),
    rho +0.044 (0.74 sigma), rhoAux +0.075 (1.27 sigma).

On ne peut donc ni choisir un point de reprise, ni comparer deux
architectures, ni selectionner des colonnes. Avant de toucher au modele, il
faut rendre la mesure capable de voir quelque chose.

LE LEVIER LE PLUS FORT EST LA DUREE DE DETENTION. Le nombre d'occasions
independantes vaut duree_de_marche / duree_d_un_trade : diviser la seconde
par quatre multiplie le premier par quatre, et divise la barre de bruit par
deux.

CE QUE CE FICHIER BALAIE, et ce qu'il refuse de regarder isolement :

  - la duree, evidemment ;
  - le RENDEMENT moyen, car raccourcir ne sert a rien si l'avantage part ;
  - la FRICTION par R, qui monte quand le stop se resserre : le spread est
    fixe, donc un stop deux fois plus proche double son poids relatif ;
  - et la BARRE DE BRUIT qui en resulte, qui est le but de l'exercice.

La fenetre de TEST n'est pas lue.

    python mesure_duree_sortie.py
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

import cibles as CIB
import training as T

# Fenetre d'ENTRAINEMENT du fold 1 : on regle la geometrie sur elle seule.
FENETRE = (0, 270248)
PAS = 12
Q = 0.05

# (nom, atr_sl_mult, trail_R, objectif en R ou None)
REGLES = [
    ("actuelle        10xATR trail 2.0R", 10.0, 2.0, None),
    ("trailing serre  10xATR trail 1.0R", 10.0, 1.0, None),
    ("trailing tres serre  trail 0.5R",   10.0, 0.5, None),
    ("stop moitie      5xATR trail 2.0R",  5.0, 2.0, None),
    ("stop moitie + trail serre",          5.0, 1.0, None),
    ("stop quart     2.5xATR trail 1.0R",  2.5, 1.0, None),
    ("objectif 2R      10xATR",           10.0, 2.0, 2.0),
    ("objectif 1R       5xATR",            5.0, 1.0, 1.0),
]


def applique(cfg, sl, trail, tp):
    c = copy.copy(cfg)
    c.atr_sl_mult = float(sl)
    c.atr_trail_mult = float(trail) * float(sl)
    c.atr_trail_dist = float(trail) * float(sl)
    c.use_be_trail = True
    if tp is None:
        c.use_tp = False
    else:
        c.use_tp = True
        c.atr_tp_mult = float(tp) * float(sl)
    return c


def barre_de_bruit(idx, r, duree_med, n_tirages=300, graine=3):
    """L'ecart-type d'un sommet obtenu sans information.

    On ne tire PAS un score uniforme : il disperserait ses 5 % dans le temps
    et donnerait une barre trois fois trop basse. On tire un score LISSE,
    dont l'autocorrelation imite celle d'une colonne reelle, et on garde le
    meme groupement que ce que le modele produirait.
    """
    n = len(r)
    if n < 200:
        return float("nan")
    k = max(int(round(Q * n)), 20)
    moy = float(np.mean(r))
    # Un lissage sur la duree typique d'un trade : c'est l'echelle a
    # laquelle un score de marche varie.
    w = max(int(round(duree_med / PAS)), 1)
    rng = np.random.default_rng(graine)
    ex = np.empty(n_tirages)
    noyau = np.ones(w) / w
    for t in range(n_tirages):
        b = np.convolve(rng.standard_normal(n + w), noyau, mode="valid")[:n]
        ex[t] = float(np.mean(r[np.argsort(b)[-k:]])) - moy
    return float(np.std(ex))


def main() -> int:
    cfg0 = T.PPOConfig()
    df = pd.read_pickle("data_cache_XAUUSD_M5.pkl")
    sous = df.iloc[FENETRE[0]:FENETRE[1]].reset_index(drop=True)
    idx = np.arange(cfg0.lookback, len(sous) - CIB.BORNE_DEFAUT - 2, PAS)
    print(f"\nfenetre d'entrainement du fold 1 : {len(sous):,} barres, "
          f"{len(idx):,} occasions de grille")
    print("La fenetre de TEST n'est pas lue.\n")

    import instruments as I
    p = I.INSTRUMENTS["XAUUSD"]
    atr_rel = float(p.get("atr_rel_bps", 0.0)) or 0.0
    spread = float(p["spread_bps"])

    print(f"  {'regle':<36} {'duree':>7} {'occ.':>6} {'indep':>6} "
          f"{'R moyen':>8} {'fric':>6} {'bruit':>7}")
    print("  " + "-" * 82)
    for nom, sl, trail, tp in REGLES:
        cfg = applique(cfg0, sl, trail, tp)
        ra, _, ta, _ = CIB.rendements(sous, idx, cfg, durees=True)
        ok = np.isfinite(ra) & np.isfinite(ta)
        if ok.sum() < 500:
            print(f"  {nom:<36} {'—':>7} {ok.sum():>6}")
            continue
        r, t, i = ra[ok], ta[ok], idx[ok]
        med = float(np.median(t))
        # Occasions DISJOINTES : placement glouton sur les indices.
        n_ind, fin = 0, -1
        for x in np.sort(i):
            if x >= fin:
                n_ind += 1
                fin = x + med
        fric = (spread + 3.0) / (sl * atr_rel) if atr_rel > 0 else float("nan")
        bruit = barre_de_bruit(i, r, med)
        print(f"  {nom:<36} {med:>7.0f} {len(r):>6} {n_ind:>6} "
              f"{np.mean(r):>+8.3f} {fric:>6.3f} {bruit:>7.3f}")

    print("\n  duree  : barres M5 tenues, mediane")
    print("  indep  : occasions DISJOINTES sur la fenetre entiere")
    print("  R moyen: ce que rapporte une entree au hasard sous cette regle")
    print("  fric   : cout d'aller-retour en fraction d'un R")
    print("  bruit  : ecart-type d'un `sommet` obtenu SANS information")
    print("\n  CE QU'ON CHERCHE : le bruit le plus BAS, sans perdre le")
    print("  R moyen ni laisser la friction devorer l'avantage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
