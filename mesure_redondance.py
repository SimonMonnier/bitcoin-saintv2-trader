# -*- coding: utf-8 -*-
"""Combien de directions INDEPENDANTES y a-t-il dans les 259 colonnes ?

POURQUOI ON NE PEUT PAS CHOISIR LES COLONNES SUR LEUR PERFORMANCE.

Mesure du 2026-09-21, sur les deux fenetres de validation, sous la nouvelle
geometrie de sortie :

    correlation entre l'exces du fold 1 et celui du fold 2 : -0.049
    les 20 meilleures du fold 1 rendent -0.101 au fold 2
    (la moyenne de TOUTES les colonnes vaut -0.033)

Selectionner sur une fenetre donne donc un resultat PIRE que la moyenne sur
la suivante. Ce n'etait pas un accident de l'ancienne geometrie : sous
`trail 2.0R` la correlation valait -0.426, sous `0.5R` elle vaut -0.049.
Dans les deux cas, choisir des colonnes d'apres ce qu'elles ont rapporte est
au mieux inutile.

CE QUI RESTE LEGITIME. La redondance est une propriete des colonnes
ELLES-MEMES : deux colonnes correlees a 0.97 portent la meme information,
que la cible existe ou non. L'elaguer ne peut pas sur-apprendre, puisque la
cible n'intervient jamais dans le calcul.

CE QUE CE FICHIER MESURE.

  1. Le nombre de directions reellement independantes, par decomposition en
     valeurs propres de la matrice de correlation : combien en faut-il pour
     porter 90, 95 et 99 % de la variance.

  2. Un jeu reduit par elagage glouton de la redondance, sans regarder la
     cible une seule fois.

  3. Ce que cela permet comme taille de reseau, rapporte aux occasions
     independantes disponibles.

La fenetre de TEST n'est pas lue.

    python mesure_redondance.py [seuil]      # seuil de correlation, defaut 0.85
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import cibles as CIB
import training as T
from saint_core import FEATURE_COLS

# Fenetre d'ENTRAINEMENT du fold 1. La redondance ne depend pas de la cible,
# mais on reste sur la fenetre d'entrainement par discipline.
FENETRE = (0, 270248)
PAS = 12


def main() -> int:
    seuil = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
    cfg = T.PPOConfig()
    df = pd.read_pickle("data_cache_XAUUSD_M5.pkl")
    s = df.iloc[FENETRE[0]:FENETRE[1]].reset_index(drop=True)
    idx = np.arange(cfg.lookback, len(s) - CIB.BORNE_DEFAUT - 2, PAS)

    cols = [c for c in FEATURE_COLS if c in s.columns]
    X = s[cols].to_numpy(np.float64)[np.maximum(idx - 1, 0)]

    # Les colonnes mortes partent d'abord : une constante n'informe de rien
    # et gonfle le compte de parametres pour rien.
    sd = np.nanstd(X, axis=0)
    vivantes = [j for j in range(len(cols)) if np.isfinite(sd[j]) and sd[j] > 1e-12]
    mortes = [cols[j] for j in range(len(cols)) if j not in set(vivantes)]
    print(f"\n{len(cols)} colonnes   {len(idx):,} occasions")
    if mortes:
        print(f"  {len(mortes)} colonnes CONSTANTES, retirees : "
              f"{', '.join(mortes[:6])}{' ...' if len(mortes) > 6 else ''}")
    X = X[:, vivantes]
    cols = [cols[j] for j in vivantes]
    X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))

    # ---------- 1. Les directions independantes ----------
    Z = (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)
    C = np.corrcoef(Z, rowvar=False)
    C = np.where(np.isfinite(C), C, 0.0)
    vp = np.sort(np.linalg.eigvalsh(C))[::-1]
    vp = np.maximum(vp, 0.0)
    cum = np.cumsum(vp) / vp.sum()
    print("\n" + "=" * 70)
    print("COMBIEN DE DIRECTIONS INDEPENDANTES ?")
    print("=" * 70)
    for p in (0.80, 0.90, 0.95, 0.99):
        k = int(np.searchsorted(cum, p) + 1)
        print(f"  {100*p:.0f} % de la variance tient en {k:>3} directions "
              f"sur {len(cols)}")
    print(f"\n  Les {len(cols)} colonnes ne portent donc pas {len(cols)}")
    print("  informations distinctes — loin de la. Le reste est de la copie.")

    # ---------- 2. L'elagage glouton ----------
    print("\n" + "=" * 70)
    print(f"ELAGAGE DE LA REDONDANCE   (|correlation| >= {seuil:.2f})")
    print("=" * 70)
    # On garde en priorite les colonnes les plus VARIEES — pas les plus
    # performantes : la cible n'entre pas dans ce choix.
    ordre = np.argsort(-np.abs(Z).mean(0))
    gardees = []
    for j in ordre:
        if any(abs(C[j, g]) >= seuil for g in gardees):
            continue
        gardees.append(j)
    noms = [cols[j] for j in gardees]
    print(f"  {len(cols)} -> {len(noms)} colonnes")
    for i in range(0, min(len(noms), 60), 3):
        print("    " + "  ".join(f"{n:<24}" for n in noms[i:i+3]))
    if len(noms) > 60:
        print(f"    ... et {len(noms)-60} autres")

    # ---------- 3. Ce que ca permet ----------
    print("\n" + "=" * 70)
    print("CE QUE CELA PERMET COMME TAILLE DE RESEAU")
    print("=" * 70)
    duree = 100        # barres, sous la nouvelle regle de sortie
    occ = (FENETRE[1] - FENETRE[0]) / duree
    print(f"  occasions independantes par fenetre d'entrainement : {occ:.0f}")
    print(f"  (duree de detention {duree} barres sous trail 0.5R)\n")
    print(f"  {'jeu de colonnes':<22} {'entrees':>8} {'params vises':>14} "
          f"{'par occasion':>13}")
    for nom, n in (("aujourd'hui", len(cols)), ("elague", len(noms)),
                   ("directions a 95 %",
                    int(np.searchsorted(cum, 0.95) + 1))):
        entree = cfg.lookback * (n + 6)
        # Une regle de pouce honnete : un parametre pour dix occasions
        # independantes. C'est deja genereux pour des donnees de marche.
        vise = occ / 10.0
        print(f"  {nom:<22} {entree:>8} {vise:>14.0f} "
              f"{46496/occ:>13.1f}")
    print(f"\n  Le reseau porte 46 496 parametres, soit {46496/occ:.0f} par")
    print(f"  occasion independante. La cible raisonnable est {occ/10:.0f}.")
    print(f"  Il faut donc le diviser par ~{46496/(occ/10):.0f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
