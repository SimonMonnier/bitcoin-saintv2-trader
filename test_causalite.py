"""Aucune colonne ne doit changer quand on coupe le futur.

CE QUE CE TEST FAIT. Il calcule chaque feature deux fois : une fois sur la
serie entiere, une fois sur la serie TRONQUEE a l'instant t. Si la valeur en t
differe entre les deux, c'est que le calcul a lu une barre posterieure a t.
Aucun autre symptome n'est necessaire, et surtout aucun n'est suffisant.

POURQUOI IL EXISTE, ecrit le 2026-09-15 le jour meme ou il aurait servi. Cinq
des 47 nouvelles colonnes Ichimoku calculaient leur pente avec une difference
CENTREE : (x[i+1] - x[i-1]) / 2, qui lit la barre suivante.
`ich_kumo_futur_pente` allait jusqu'a contenir le plus haut et le plus bas du
LENDEMAIN, puisque ssb_futur[i+1] est le milieu des 52 periodes finissant en
i+1.

Le resultat sur la fenetre de test : +25.5 points au-dessus du point mort,
profit factor 2.85, la ou le jeu precedent donnait +1.6 point. La colonne
gardait son nom, sa forme, son ordre de grandeur et sa plage de valeurs. Rien
dans les statistiques descriptives ne la distinguait. Seule l'INVRAISEMBLANCE
du resultat final l'a trahie — autrement dit, la chance.

Une fuite plus discrete, qui n'aurait donne que deux ou trois points de trop,
serait passee inapercue et aurait fini en production. Ce test la trouverait.

CE QU'IL COUVRE. La chaine ENTIERE de prepare_h1.construit : indicateurs H1,
contexte H4 avec son merge_asof, features de flux, range et Ichimoku. Le bloc
H4 est l'endroit le plus expose du depot — merge_asof choisit par defaut la
bougie H4 qui CONTIENT l'instant courant, donc une bougie encore en formation,
et injecterait jusqu'a quatre heures de futur par ligne. Le shift(1) qui
l'evite tient en un mot, et son absence ne leve aucune erreur. Il est
desormais verifie, pas seulement commente.

CE QU'IL NE COUVRE PAS. Une colonne peut etre parfaitement causale et quand
meme inutile ou trompeuse ; ce fichier ne dit rien de sa valeur. Et il s'arrete
aux colonnes : la normalisation calculee sur la mauvaise fenetre, une cible mal
decalee ou un seuil calibre sur les donnees qui le jugent lui echappent.

    python test_causalite.py [nb_points]
"""

import sys

import numpy as np
import pandas as pd

from prepare_h1 import construit

SOURCE = "klines_h1_spot_BTCUSDT.pkl"
# Assez de barres avant le point teste pour que toutes les fenetres soient
# pleines : la plus longue est FENETRE_EXT = 200 dans features_ichimoku.
MARGE = 800
N_POINTS = 12
# Tolerance relative. Les deux calculs passent par des chemins numeriques
# identiques, donc l'ecart attendu est nul ; on laisse la marge du flottant.
TOL = 1e-9


def colonnes_de(df):
    """La chaine ENTIERE, indicateurs H1 et contexte H4 compris.

    Tester seulement les modules de features laisserait dehors le bloc H4,
    c'est-a-dire l'endroit le plus expose : merge_asof choisit par defaut la
    bougie H4 qui CONTIENT l'instant courant, donc une bougie encore en
    formation, et injecte jusqu'a quatre heures de futur par ligne. Le shift(1)
    qui l'evite est une ligne, et son absence ne leve aucune erreur.
    """
    d, cols, _ = construit(df.copy())
    return d, cols


def main() -> int:
    n_points = int(sys.argv[1]) if len(sys.argv) > 1 else N_POINTS
    brut = pd.read_pickle(SOURCE)
    brut["time"] = pd.to_datetime(brut["time"])
    base = brut.sort_values("time").reset_index(drop=True)

    complet, cols = colonnes_de(base)
    print(f"{len(cols)} colonnes testees sur {n_points} instants tires dans "
          f"{len(base):,} barres")
    print("methode : valeur en t sur la serie ENTIERE contre la meme valeur "
          "sur la serie coupee en t\n")

    rng = np.random.default_rng(0)
    points = sorted(rng.integers(MARGE, len(base) - 5, size=n_points).tolist())

    fautives = {}
    for t in points:
        tronque, _ = colonnes_de(base.iloc[:t + 1].copy())
        a = complet.loc[t, cols].to_numpy(np.float64)
        b = tronque.iloc[-1][cols].to_numpy(np.float64)
        ecart = np.abs(a - b)
        # NaN des deux cotes = accord ; NaN d'un seul cote = desaccord.
        na, nb = np.isnan(a), np.isnan(b)
        ecart = np.where(na & nb, 0.0, ecart)
        ecart = np.where(na ^ nb, np.inf, ecart)
        echelle = np.maximum(np.abs(np.where(na, 0.0, a)), 1.0)
        mauvais = ecart > TOL * echelle
        for k in np.flatnonzero(mauvais):
            fautives.setdefault(cols[k], []).append(
                (t, float(a[k]), float(b[k])))

    if not fautives:
        print(f"AUCUNE FUITE : les {len(cols)} colonnes rendent la meme valeur")
        print("avec et sans le futur, sur tous les instants testes.")
        return 0

    print(f"{len(fautives)} COLONNES LISENT LE FUTUR :\n")
    print(f"{'colonne':>26} {'points':>7}  {'exemple : avec futur -> sans':>40}")
    print("-" * 78)
    for k, v in sorted(fautives.items(), key=lambda x: -len(x[1])):
        t, a, b = v[0]
        print(f"{k:>26} {len(v):4d}/{n_points:<3d}  t={t:<7d} "
              f"{a:+12.6f} -> {b:+12.6f}")
    print("\nUne difference, meme minuscule, veut dire que le calcul a lu une")
    print("barre posterieure. Il n'y a pas de fuite acceptable.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
