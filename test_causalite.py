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

CE QU'IL COUVRE. La chaine ENTIERE de construit() : les indicateurs de
l'echelle de decision, puis CHAQUE contexte superieur avec son merge_asof, features de flux, range et Ichimoku. Le bloc
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

# L'echelle de decision decide de quelle chaine on teste. saint_core.TIMEFRAME
# est la source de verite ; la faire choisir ici la dedoublerait.
import saint_core as S

if S.TIMEFRAME == "M5":
    from prepare_m5 import construit
    SOURCE = "klines_5m_spot_BTCUSDT.pkl"
else:
    from prepare_h1 import construit
    SOURCE = "klines_h1_spot_BTCUSDT.pkl"
# Assez de barres avant le point teste pour que toutes les fenetres soient
# pleines : la plus longue est FENETRE_EXT = 200 dans features_ichimoku.
# LA MARGE SE COMPTE A L'ECHELLE LA PLUS LENTE, pas a celle des decisions. En
# M5 le bloc H4 reclame FENETRE_EXT = 200 bougies H4, plus SENKOU_B = 52 et son
# decalage de 26 : ~280 bougies H4, soit 13 400 barres M5.
#
# UNE MARGE TROP COURTE NE FAIT PAS ECHOUER LE TEST — elle le fait MENTIR. Les
# deux calculs rendent NaN, NaN contre NaN compte comme un accord, et les 85
# colonnes H4 sont declarees saines sans avoir jamais ete evaluees. C'est la
# pire defaillance possible pour un test : celle qui rassure a tort. Le compte
# de colonnes reellement evaluees, affiche a la fin, existe pour que cela se
# voie au lieu de se supposer.
MARGE = 16000 if S.TIMEFRAME == "M5" else 800
N_POINTS = 12
# Barres conservees APRES l'instant teste dans la version "avec futur".
SUITE = 600 if S.TIMEFRAME == "M5" else 120
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
    r = construit(df.copy())
    return r[0], r[1]


def main() -> int:
    n_points = int(sys.argv[1]) if len(sys.argv) > 1 else N_POINTS
    brut = pd.read_pickle(SOURCE)
    brut["time"] = pd.to_datetime(brut["time"])
    base = brut.sort_values("time").reset_index(drop=True)

    rng = np.random.default_rng(0)
    points = sorted(rng.integers(MARGE, len(base) - SUITE - 5,
                                 size=n_points).tolist())

    # ON NE RECONSTRUIT JAMAIS LA SERIE ENTIERE. Les deux calculs partent du
    # MEME debut de fenetre, t - MARGE, donc ils ont exactement la meme
    # histoire : la seule difference est la presence de SUITE barres APRES t.
    # C'est precisement la propriete a tester, et cela borne la memoire a
    # quelques milliers de lignes au lieu de neuf ans en double.
    #
    # SUITE doit couvrir la plus longue avance qu'une fuite puisse lire : la
    # difference centree en lit une, merge_asof en lit une a l'echelle
    # superieure, soit 48 barres M5 pour le H4. 600 laissent un facteur douze.
    fautives = {}
    jamais = {}
    cols = None
    for t in points:
        deb = t - MARGE
        avec, cols = colonnes_de(base.iloc[deb:t + 1 + SUITE].copy())
        sans, _ = colonnes_de(base.iloc[deb:t + 1].copy())
        a = avec.iloc[MARGE][cols].to_numpy(np.float64)
        b = sans.iloc[-1][cols].to_numpy(np.float64)
        ecart = np.abs(a - b)
        # NaN des deux cotes = accord ; NaN d'un seul cote = desaccord.
        na, nb = np.isnan(a), np.isnan(b)
        ecart = np.where(na & nb, 0.0, ecart)
        ecart = np.where(na ^ nb, np.inf, ecart)
        # Une colonne NaN des deux cotes n'a pas ete testee : elle s'est
        # seulement trouvee d'accord sur son absence. On les compte pour que
        # ce silence ne puisse pas passer pour un succes.
        for k in np.flatnonzero(na & nb):
            jamais[cols[k]] = jamais.get(cols[k], 0) + 1
        echelle = np.maximum(np.abs(np.where(na, 0.0, a)), 1.0)
        mauvais = ecart > TOL * echelle
        for k in np.flatnonzero(mauvais):
            fautives.setdefault(cols[k], []).append(
                (t, float(a[k]), float(b[k])))

    print(f"{len(cols)} colonnes testees sur {n_points} instants tires dans "
          f"{len(base):,} barres")
    print(f"methode : valeur en t calculee sur [t-{MARGE}, t+{SUITE}] contre "
          f"la meme valeur sur [t-{MARGE}, t]")
    print()

    muettes = [c for c, n in jamais.items() if n == len(points)]
    if muettes:
        print(f"ATTENTION : {len(muettes)} colonnes sont restees NaN des deux "
              f"cotes sur TOUS les points.")
        print("  Elles ne se sont pas montrees saines, elles se sont tues.")
        print(f"  Augmenter MARGE, actuellement {MARGE}.")
        print(f"  exemples : {muettes[:6]}")
        print()
    print(f"colonnes reellement evaluees : {len(cols) - len(muettes)} / "
          f"{len(cols)}")
    print()

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
