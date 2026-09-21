# -*- coding: utf-8 -*-
"""Quelles colonnes portent vraiment le signal, et lesquelles sont du bruit ?

POURQUOI CETTE MESURE. Le reseau porte 46 496 parametres et lit 259 colonnes
de marche sur 4 barres, soit 1 060 entrees. En face, une fenetre
d'entrainement contient 438 a 682 occasions INDEPENDANTES — la duree mediane
d'un trade est de 396 barres, mesuree sur 39 997 trades, donc deux occasions
espacees de moins que ca decrivent le meme mouvement.

    68 a 106 PARAMETRES PAR OCCASION INDEPENDANTE.

En apprentissage supervise on vise moins d'un parametre pour dix exemples.
C'est un facteur mille du mauvais cote, et ca se lit dans les resultats
d'exec69 :

  - le fold 1 s'effondre passe l'epoch 45 (exces +0.594 puis -0.008) ;
  - le fold 2 ordonne l'ENSEMBLE des occasions mieux que le fold 1
    (rhoAux +0.075 contre +0.040) mais sa selection du HAUT ne vaut rien
    (exces -0.033, net negatif 78 fois sur 78).

Cette asymetrie est la signature du sur-apprentissage. Le centre de la
distribution demande un signal grossier que le reseau capte sans effort ;
la queue demande une discrimination fine qu'il ne peut obtenir qu'en
MEMORISANT quelles barres precises ont explose. Ces barres-la ne reviennent
pas.

CE QUE CE FICHIER FAIT.

  1. Il classe les colonnes par association avec la cible, SUR LA FENETRE
     D'ENTRAINEMENT DU FOLD 1 UNIQUEMENT. Regarder la validation ou le test
     pour choisir des colonnes deplacerait la fuite au lieu de la supprimer.

  2. Il elague la redondance : deux colonnes correlees a plus de 0.90 ne
     comptent qu'une fois. Les familles ichimoku/range existent en M5, H1 et
     H4, et beaucoup sont des copies decalees.

  3. IL VERIFIE LA STABILITE, et c'est le controle qui decide. Il refait le
     classement sur la fenetre d'entrainement du fold 2 et compare. Si les
     colonnes utiles ne sont pas les memes d'une fenetre a l'autre, alors
     meme la SELECTION ne transfere pas — et reduire le nombre de colonnes
     ne suffira pas.

    python mesure_colonnes.py            # 30 colonnes retenues
    python mesure_colonnes.py 20         # un autre nombre
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import cibles as CIB
import training as T
from saint_core import FEATURE_COLS

# Les deux fenetres d'ENTRAINEMENT, telles que le walk-forward les decoupe.
# Aucune ne touche a la validation ni au test.
TRAIN_1 = (0, 270248)
TRAIN_2 = (49136, 49136 + 270248)
REDONDANCE = 0.90


def cible_et_features(df, debut, fin, cfg):
    """La cible du fold et la valeur de chaque colonne, sur sa grille dense."""
    sous = df.iloc[debut:fin].reset_index(drop=True)
    idx = np.arange(cfg.lookback,
                    len(sous) - CIB.BORNE_DEFAUT - 2,
                    int(cfg.pas_grille_rang))
    ra, rv = CIB.rendements(sous, idx, cfg, indicateur=True)
    # LONG-ONLY : la cible est le cote achat, comme `cotes_permises` le pose.
    y = ra
    ok = np.isfinite(y)
    idx, y = idx[ok], y[ok]
    # LA COLONNE EST LUE A LA BARRE PRECEDENTE, comme l'observation la lit :
    # prendre la barre de decision ferait entrer une information que le
    # modele n'a pas au moment de decider.
    cols = [c for c in FEATURE_COLS if c in sous.columns]
    X = sous[cols].to_numpy(np.float64)[np.maximum(idx - 1, 0)]
    return cols, X, y


def classe(cols, X, y):
    """|rho de Spearman| entre chaque colonne et la cible."""
    out = {}
    for j, c in enumerate(cols):
        v = X[:, j]
        bon = np.isfinite(v)
        # Une colonne quasi constante n'a pas de rang : rho vaut nan et la
        # compter comme nulle est la bonne lecture, pas une approximation.
        if bon.sum() < 200 or np.nanstd(v[bon]) < 1e-12:
            out[c] = 0.0
            continue
        r = spearmanr(v[bon], y[bon]).statistic
        out[c] = abs(float(r)) if np.isfinite(r) else 0.0
    return out


def main() -> int:
    n_garde = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    cfg = T.PPOConfig()
    df = pd.read_pickle("data_cache_XAUUSD_M5.pkl")
    print(f"\n{len(df):,} barres   {len(FEATURE_COLS)} colonnes de marche   "
          f"pas de grille {cfg.pas_grille_rang} barres")
    print("La fenetre de TEST n'est pas lue.")

    print("\nclassement sur la fenetre d'ENTRAINEMENT du fold 1...", flush=True)
    cols, X1, y1 = cible_et_features(df, *TRAIN_1, cfg)
    s1 = classe(cols, X1, y1)
    print(f"  {len(y1):,} occasions, cible mediane {np.median(y1):+.3f}")

    print("classement sur la fenetre d'ENTRAINEMENT du fold 2...", flush=True)
    _, X2, y2 = cible_et_features(df, *TRAIN_2, cfg)
    s2 = classe(cols, X2, y2)
    print(f"  {len(y2):,} occasions, cible mediane {np.median(y2):+.3f}")

    # ---------- LE CONTROLE QUI DECIDE ----------
    print("\n" + "=" * 78)
    print("LES COLONNES UTILES SONT-ELLES LES MEMES D'UNE FENETRE A L'AUTRE ?")
    print("=" * 78)
    a = np.array([s1[c] for c in cols])
    b = np.array([s2[c] for c in cols])
    rr = spearmanr(a, b).statistic
    print(f"  correlation des deux classements : {rr:+.3f}")
    print("  (a +1 les memes colonnes portent le signal dans les deux")
    print("   fenetres ; vers 0 le classement du fold 1 ne dit rien du 2)")
    k = 30
    t1 = set(np.array(cols)[np.argsort(-a)[:k]])
    t2 = set(np.array(cols)[np.argsort(-b)[:k]])
    print(f"  colonnes communes aux {k} premieres de chaque fenetre : "
          f"{len(t1 & t2)} / {k}")
    print(f"  attendu par le hasard : {k*k/len(cols):.1f}")

    # ---------- SELECTION GLOUTONNE, SUR LE FOLD 1 SEUL ----------
    print("\n" + "=" * 78)
    print(f"LES {n_garde} COLONNES RETENUES   (classees sur le fold 1, "
          f"redondance elaguee a {REDONDANCE:.2f})")
    print("=" * 78)
    ordre = [cols[j] for j in np.argsort(-a)]
    j_de = {c: j for j, c in enumerate(cols)}
    gardees = []
    for c in ordre:
        if s1[c] <= 0.0:
            break
        v = X1[:, j_de[c]]
        double = False
        for g in gardees:
            w = X1[:, j_de[g]]
            bon = np.isfinite(v) & np.isfinite(w)
            if bon.sum() < 200:
                continue
            r = spearmanr(v[bon], w[bon]).statistic
            if np.isfinite(r) and abs(r) >= REDONDANCE:
                double = True
                break
        if not double:
            gardees.append(c)
        if len(gardees) >= n_garde:
            break

    print(f"  {'colonne':<34} {'|rho| fold 1':>12} {'|rho| fold 2':>12}")
    for c in gardees:
        print(f"  {c:<34} {s1[c]:>12.4f} {s2[c]:>12.4f}")

    tenues = sum(1 for c in gardees if s2[c] >= 0.02)
    print(f"\n  colonnes qui tiennent au fold 2 (|rho| >= 0.02) : "
          f"{tenues} / {len(gardees)}")
    print(f"  |rho| moyen : fold 1 {np.mean([s1[c] for c in gardees]):.4f}   "
          f"fold 2 {np.mean([s2[c] for c in gardees]):.4f}")

    print("\n" + "=" * 78)
    print("CE QUE CELA CHANGE A LA DISPROPORTION")
    print("=" * 78)
    for n, nom in ((len(cols), "aujourd'hui"), (len(gardees), "retenues")):
        entree = cfg.lookback * (n + 6)
        print(f"  {nom:<12} {n:>4} colonnes  ->  {entree:>5} entrees "
              f"(lookback {cfg.lookback})")
    print("\n  Reduire les colonnes ne suffit pas : c'est le RESEAU qui porte")
    print("  les 46 496 parametres. Cette mesure dit seulement combien")
    print("  d'entrees il est legitime de lui donner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
