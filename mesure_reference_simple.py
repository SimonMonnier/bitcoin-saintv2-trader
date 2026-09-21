# -*- coding: utf-8 -*-
"""Le reseau de 46 496 parametres bat-il UNE COLONNE ?

POURQUOI CETTE MESURE, ET POURQUOI ELLE EST DECISIVE.

Le reseau obtient un exces sur le hasard de -0.033 au fold 2 — c'est-a-dire
RIEN — avec 46 496 parametres, 259 colonnes et 78 epochs de calcul. La
question qui tranche : **quel `sommet` obtient-on en classant les occasions
par UNE SEULE COLONNE, sans rien apprendre ?**

  - Si une colonne seule fait aussi bien, le reseau n'apporte rien.
  - Si elle fait MIEUX, le reseau detruit un signal qu'on lui donne.
  - S'il fait nettement mieux, il apprend quelque chose et le probleme est
    ailleurs.

DEUX PIEGES QUE CE FICHIER A DU CORRIGER, ET ILS VALENT D'ETRE ECRITS.

  1. LA FENETRE. La premiere version mesurait les colonnes sur la fenetre
     d'ENTRAINEMENT et les comparait au reseau mesure sur la VALIDATION. Les
     deux fenetres n'ont pas le meme hasard — +0.154 R contre +0.569 R — donc
     la comparaison ne voulait rien dire. On mesure ici sur les MEMES
     fenetres de validation que le journal.

  2. |rho| NE PREDIT PAS `sommet`. `mesure_colonnes.py` classe les colonnes
     par correlation de rang avec la cible, et `vol_20` sort en tete a 0.51.
     Or trier par `vol_20` donne un exces de -0.33 : elle choisit activement
     les MAUVAISES occasions. La cible est un rendement en unites de risque
     et le stop vaut 10 x ATR — une part de cette correlation est
     MECANIQUE, un effet du denominateur, pas un avantage.

     CONSEQUENCE : selectionner des colonnes sur |rho| est un mauvais
     critere. Ce qu'il faut classer, c'est ce que chaque colonne rend AU
     SOMMET — et c'est ce que ce fichier mesure, colonne par colonne.

La fenetre de TEST n'est pas lue.

    python mesure_reference_simple.py           # les candidates
    python mesure_reference_simple.py toutes    # les 259, classees par sommet
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import cibles as CIB
import training as T
from saint_core import FEATURE_COLS

# LES FENETRES DE VALIDATION, telles que le walk-forward les decoupe et
# telles que le journal les mesure. Lues dans l'en-tete du run :
#   Fold 1 : train=270248, calib=25796, val=47908, test=49136, start=0
#   Fold 2 : idem, start=49136
VAL = {
    "fold 1": (0 + 270248 + 25796, 0 + 270248 + 25796 + 47908),
    "fold 2": (49136 + 270248 + 25796, 49136 + 270248 + 25796 + 47908),
}
SELECTIVITE = 0.05

CANDIDATES = ("vol_20", "vol_rank", "range_norm", "high_vol_regime",
              "range_norm_h1", "heure_sin", "heure_cos", "low_rel_h1",
              "high_rel_h1", "vol_20_h4", "rng_epaisseur_kumo",
              "rng_cassure_recente", "ich_plat_kijun_h1")


def prepare(df, debut, fin, cfg):
    sous = df.iloc[debut:fin].reset_index(drop=True)
    idx = np.arange(cfg.lookback, len(sous) - CIB.BORNE_DEFAUT - 2,
                    int(cfg.pas_grille_rang))
    # LE RENDEMENT REELLEMENT ENCAISSE — pas l'indicateur que la tete apprend
    # a classer. C'est lui que `sommet` mesure dans le journal.
    rr, _ = CIB.rendements(sous, idx, cfg)
    ok = np.isfinite(rr)
    return sous, idx[ok], rr[ok]


def sommet(score, gain, q=SELECTIVITE):
    bon = np.isfinite(score) & np.isfinite(gain)
    s, g = score[bon], gain[bon]
    if len(s) < 100:
        return float("nan"), float("nan")
    n_top = max(int(round(q * len(s))), 20)
    top = np.argsort(s)[-n_top:]
    return float(np.mean(g[top])), float(np.mean(g))


def main() -> int:
    toutes = len(sys.argv) > 1 and sys.argv[1].startswith("tout")
    cfg = T.PPOConfig()
    df = pd.read_pickle("data_cache_XAUUSD_M5.pkl")
    print(f"\nselectivite {100*SELECTIVITE:.0f} %   "
          f"pas de grille {cfg.pas_grille_rang} barres")
    print("Mesure sur les fenetres de VALIDATION, les memes que le journal.")
    print("La fenetre de TEST n'est pas lue.")

    jeux = {}
    for nom, (a, b) in VAL.items():
        sous, idx, rr = prepare(df, a, b, cfg)
        jeux[nom] = (sous, idx, rr)
        print(f"  {nom} : barres [{a:,} : {b:,}]   {len(rr):,} occasions   "
              f"hasard {np.mean(rr):+.3f} R")

    cols = ([c for c in FEATURE_COLS if c in jeux["fold 1"][0].columns]
            if toutes else
            [c for c in CANDIDATES if c in jeux["fold 1"][0].columns])

    res = []
    for c in cols:
        ligne = {"col": c}
        for nom in VAL:
            sous, idx, rr = jeux[nom]
            v = sous[c].to_numpy(np.float64)[np.maximum(idx - 1, 0)]
            # LE SENS DE TRI SE FIXE SUR LE FOLD 1, et s'applique tel quel au
            # fold 2 : un seul bit ajuste, et jamais sur la fenetre jugee.
            if nom == "fold 1":
                bon = np.isfinite(v) & np.isfinite(rr)
                sens = 1.0
                if bon.sum() > 200:
                    s_top, _ = sommet(v, rr)
                    s_bot, _ = sommet(-v, rr)
                    if np.isfinite(s_bot) and (not np.isfinite(s_top)
                                               or s_bot > s_top):
                        sens = -1.0
                ligne["sens"] = sens
            s, h = sommet(ligne["sens"] * v, rr)
            ligne[nom] = s - h
        res.append(ligne)

    res.sort(key=lambda d: (-d["fold 2"] if np.isfinite(d["fold 2"])
                            else 1e9))
    print("\n" + "=" * 74)
    print("UNE SEULE COLONNE, AUCUN APPRENTISSAGE   (exces sur le hasard)")
    print("=" * 74)
    print(f"  {'colonne':<32} {'sens':>5} {'fold 1':>10} {'fold 2':>10}")
    for d in (res[:25] if toutes else res):
        print(f"  {d['col']:<32} {d['sens']:>+5.0f} "
              f"{d['fold 1']:>+10.3f} {d['fold 2']:>+10.3f}")

    print("\n" + "=" * 74)
    print("CE QUE LE RESEAU OBTIENT, SUR LES MEMES FENETRES")
    print("=" * 74)
    print("  exec69 : 46 496 parametres, 259 colonnes, 45 a 90 epochs")
    print("    fold 1   exces +0.299 en moyenne (+0.594 au meilleur quart)")
    print("    fold 2   exces -0.033 en moyenne sur 78 epochs")
    bons = [d for d in res if np.isfinite(d["fold 2"])]
    if bons:
        b = max(bons, key=lambda d: d["fold 2"])
        print(f"\n  meilleure colonne SEULE au fold 2 : {b['col']} "
              f"a {b['fold 2']:+.3f}")
        # LE VERDICT SE LIT SUR LE FOLD 2 : le fold 1 est la fenetre ou le
        # reseau s'entraine, donc il y est avantage par construction.
        print("  -> " + ("UNE COLONNE SEULE FAIT MIEUX QUE LE RESEAU"
                         if b["fold 2"] > -0.033 else
                         "le reseau fait mieux qu'une colonne seule"))
        st = [d for d in bons if d["fold 1"] > 0 and d["fold 2"] > 0]
        print(f"  colonnes positives sur LES DEUX folds : {len(st)} / "
              f"{len(bons)}")
        for d in sorted(st, key=lambda x: -x["fold 2"])[:8]:
            print(f"     {d['col']:<30} {d['fold 1']:>+8.3f} "
                  f"{d['fold 2']:>+8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
