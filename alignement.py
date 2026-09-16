"""Deux instruments au MEME instant — sans quoi le compte partage est une fiction.

LE PROBLEME. Chaque jeu compte les barres de SA propre serie : la barre 50 000
du BTC est en 2018, celle de l'or en 2020. Deux environnements qui partagent
un compte mais avancent chacun dans sa serie simulent un portefeuille qui n'a
jamais existe — un Bitcoin de 2018 finançant un or de 2020. Toute mesure a
deux instruments est fausse tant que ce n'est pas regle.

LA SOLUTION RETENUE : L'INTERSECTION DES HORODATAGES. On ne garde que les
instants ou les DEUX marches cotent, et la ligne i de chaque jeu designe alors
le meme moment. C'est exact par construction, et ca correspond a ce qu'un
compte unique peut faire : ouvrir sur l'or quand l'or est ouvert.

CE QUE CA COUTE. Le Bitcoin cote en continu, l'or cinq jours sur sept avec une
pause quotidienne. L'intersection est donc le calendrier de l'or, et le BTC
perd ses week-ends. Le compte exact est affiche : c'est le prix de l'operation.

CE QUE CA NE CASSE PAS. Les indicateurs sont calcules AVANT, sur chaque serie
continue. Reindexer ensuite ne les recalcule pas et ne les corrompt donc pas —
une moyenne mobile du BTC reste celle du BTC sur ses propres barres, on se
contente de ne regarder que certaines lignes.

CE QUI RESTE FAUX, ET QU'IL FAUT SAVOIR. Les trades du BTC ouverts le vendredi
traversent le week-end dans la realite ; ici la ligne suivante est le lundi,
donc le stop suiveur ne voit pas ce qui s'est passe entre les deux. Le
franchissement de stop pendant un week-end BTC devient invisible. C'est une
sous-estimation du risque, du meme ordre que celle des sauts de l'or —
mesuree a 0.6 % des trades sur le BTC, ou le M5 resout 99.4 % des sorties.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import instruments as I


def jeux_alignes(symboles=("BTCUSD", "XAUUSD")):
    """Rend {symbole: df} restreints aux horodatages COMMUNS, meme longueur.

    La ligne i de chaque jeu rendu designe le meme instant dans tous.
    """
    bruts = {}
    for s in symboles:
        d = pd.read_pickle(I.INSTRUMENTS[s]["cache"])
        d["time"] = pd.to_datetime(d["time"])
        bruts[s] = d

    commun = None
    for s, d in bruts.items():
        t = pd.Index(d["time"])
        commun = t if commun is None else commun.intersection(t)
    commun = commun.sort_values()

    out = {}
    for s, d in bruts.items():
        m = d.set_index("time").loc[commun].reset_index()
        out[s] = m
    return out, commun


def main() -> int:
    symboles = sys.argv[1:] or ["BTCUSD", "XAUUSD"]
    bruts = {}
    for s in symboles:
        d = pd.read_pickle(I.INSTRUMENTS[s]["cache"])
        d["time"] = pd.to_datetime(d["time"])
        bruts[s] = d
        print(f"{s:<8} {len(d):>9,} barres  "
              f"{d['time'].iloc[0]:%Y-%m-%d} -> {d['time'].iloc[-1]:%Y-%m-%d}")

    jeux, commun = jeux_alignes(tuple(symboles))
    ans = (commun[-1] - commun[0]).days / 365.25
    print(f"\ncommun   {len(commun):>9,} barres  "
          f"{commun[0]:%Y-%m-%d} -> {commun[-1]:%Y-%m-%d}  ({ans:.1f} ans)")
    for s, d in bruts.items():
        # Part conservee, comptee sur la periode commune uniquement : compter
        # sur l'historique entier melangerait la perte due au calendrier et
        # celle due aux dates ou l'autre instrument n'existait pas encore.
        dans = d[(d["time"] >= commun[0]) & (d["time"] <= commun[-1])]
        print(f"  {s:<8} garde {len(commun)/max(len(dans),1):>5.1%} de ses barres "
              f"sur la periode commune ({len(dans):,} -> {len(commun):,})")

    # Verification : meme longueur, et meme instant ligne a ligne.
    lg = {s: len(d) for s, d in jeux.items()}
    assert len(set(lg.values())) == 1, f"longueurs differentes : {lg}"
    ref = jeux[symboles[0]]["time"].to_numpy()
    for s in symboles[1:]:
        assert np.array_equal(ref, jeux[s]["time"].to_numpy()), \
            f"{s} n'est pas aligne"
    print(f"\naligne : {len(ref):,} lignes, meme horodatage ligne a ligne")

    for s, d in jeux.items():
        sortie = I.INSTRUMENTS[s]["cache"].replace(".pkl", "_aligne.pkl")
        d.to_pickle(sortie)
        print(f"  ecrit {sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
