# -*- coding: utf-8 -*-
"""La veille dit-elle QUEL modele a ete sauve, et SUR QUEL critere ?

POURQUOI CE FICHIER. La veille recopiait la ligne `NEW BEST` telle quelle.
On y lisait `sommet +0.842 R/occasion` — et depuis le 2026-09-20 ce n'est
plus `sommet` qui decide, c'est le SCORE NET. On croyait donc savoir sur quoi
le modele avait ete retenu, et c'etait faux : deux descriptions du meme
evenement, qui devaient s'accorder par convention. C'est la faute que ce
depot passe son temps a payer.

Trois evenements doivent apparaitre, et chacun a son test :

  RETENU   — le critere, sa decomposition, ce qu'il a battu, le fichier ecrit.
  NON RETENU — la normale, quatre-vingts epochs sur quatre-vingt-dix. Elle
             passait en SILENCE : on regardait defiler un run entier en
             croyant qu'un meilleur modele etait garde alors que rien ne
             l'etait.
  REFUSE   — le cas grave : meilleur score, mais le portefeuille l'a
             disqualifie.

Les lignes de test sont fabriquees par les MEMES f-strings que `training.py`.
Si elles changent la sans changer ici, le test tombe — c'est tout son
interet.

    python test_veille_retenue.py
"""
from __future__ import annotations

import sys

# LA SORTIE DE CE TEST EST SOUVENT CAPTUREE PAR UN TUBE, qui vaut cp1252 sous
# Windows et ne sait pas ecrire une etoile. Cela n'a rien a voir avec la
# console de la veille — on ne veut pas qu'un test tombe pour son propre
# affichage.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import veille_epochs as V

echecs = []

# Les couleurs, telles que `_Journal` les ecrit dans le fichier. La veille
# les retire elle-meme ; les omettre testerait un chemin qui n'existe pas.
MAG = "\033[35m\033[1m"
FIN = "\033[0m"
ROUGE = "\033[31m\033[1m"
GRIS = "\033[90m"


def ligne_best(net, gain, baisse, bat, sommet, hasard, trades, fichier):
    """LA MEME f-string QUE `training.py`, couleurs comprises.

    `bud` ET `abst` EN SONT SORTIS LE 2026-09-21, avec le budget. Ils
    disaient la mise moyenne et la part d'abstention ; sans paliers ils
    valaient zero par construction, et un champ mort qui garde l'air
    vivant est pire qu'un champ absent.
    """
    return (f"  {MAG}★{FIN} {MAG}NEW BEST{FIN}  "
            f"retenu sur le SCORE NET "
            f"{MAG}{net:+.3f} R{FIN} "
            f"par occasion "
            f"(gain {gain:+.3f} R - baisse "
            f"{baisse:.3f} R)  bat {bat}  "
            f"[portillons : sommet {sommet:+.3f}R > hasard "
            f"{hasard:+.3f}R, {trades} trades, compte intact]"
            f"  -> {fichier}")


def ligne_garde(motif):
    return f"  {GRIS}garde{FIN}  le modele en place reste le meilleur — {motif}"


def ligne_refus(net, record, motif):
    return (f"  {ROUGE}REFUSE{FIN}  score net "
            f"{net:+.3f} R par occasion, meilleur que "
            f"{record:+.3f} R, "
            f"mais {motif} — non retenu, et non transmis au fold suivant.")


def rendu(ligne):
    """Ce que la veille AFFICHERAIT, couleurs retirees."""
    v = V.Veilleur()
    blocs = v.avale(ligne + "\n")
    texte = "\n".join("\n".join(b[0]) for b in blocs)
    return V.ANSI.sub("", texte)


def verifie(nom, condition, detail=""):
    if condition:
        print(f"  ok    {nom:<52} {detail}")
    else:
        echecs.append(nom)
        print(f"  ECHEC {nom:<52} {detail}")


def main() -> int:
    print("\n1. LE MODELE RETENU, ET SUR QUEL CRITERE")
    t = rendu(ligne_best(net=0.412, gain=0.687, baisse=0.275,
                         bat="+0.318 R",                          sommet=0.842, hasard=0.623, trades=56,
                         fichier="best_or_exec46_long_wf1.pth"))
    print("\n".join("      " + l for l in t.strip().split("\n")))
    print()
    verifie("le bloc s'affiche", "MODELE RETENU" in t)
    verifie("le critere est NOMME", "SCORE NET" in t)
    verifie("sa valeur y est", "+0.412 R" in t)
    verifie("sa decomposition aussi",
            "gain +0.687 R" in t and "baisse 0.275 R" in t)
    verifie("ce qu'il a battu", "+0.318 R" in t)
    verifie("de combien", "+0.094 R de mieux" in t)
    verifie("le fichier ecrit", "best_or_exec46_long_wf1.pth" in t)
    # LE PORTILLON RESTE VISIBLE MAIS N'EST PLUS PRESENTE COMME LE CRITERE.
    verifie("le sommet est montre comme PORTILLON",
            "portillons" in t and "+0.842 R" in t)
    verifie("le sommet n'est PAS presente comme le critere",
            "SCORE NET" in t.split("portillons")[0]
            and "0.842" not in t.split("portillons")[0])

    print("\n2. LE PREMIER RETENU D'UN FOLD N'A RIEN BATTU, ET LE DIT")
    t = rendu(ligne_best(net=0.412, gain=0.687, baisse=0.275,
                         bat="premier retenu du fold",                          sommet=0.842, hasard=0.623, trades=56,
                         fichier="best_wf1.pth"))
    verifie("pas de faux record battu", "premier retenu de ce fold" in t,
            "au lieu d'un `bat -1000000000.000 %`")
    verifie("aucun `de mieux` invente", "de mieux" not in t)

    print("\n3. LE REFUS ORDINAIRE NE PASSE PLUS EN SILENCE")
    t = rendu(ligne_garde("score net +0.301 R sous le record +0.412 R"))
    verifie("il s'affiche", "non retenu" in t)
    verifie("avec son motif", "+0.301 R" in t and "+0.412 R" in t)

    print("\n4. LE REFUS GRAVE SE DISTINGUE DE L'ORDINAIRE")
    t = rendu(ligne_refus(0.910, 0.410,
                          "compte detruit sur 2 episode(s) de validation"))
    print("\n".join("      " + l for l in t.strip().split("\n")))
    print()
    verifie("il est signale comme tel", "REFUSE MALGRE UN MEILLEUR SCORE" in t)
    verifie("le score et le record y sont",
            "+0.910 R" in t and "+0.410 R" in t)
    verifie("la cause est nommee", "compte detruit" in t)
    verifie("la consequence est dite", "fold suivant" in t)

    print("\n5. LE TEMOIN DE PROFIT NE PASSE PAS POUR LE CRITERE")
    t = rendu("  ☆ NEW BEST PROFIT  ValPNL/trade=    +7.27$  trades=56  "
              "(temoin seulement — ne decide pas du deploiement)")
    verifie("il s'affiche", "NEW BEST PROFIT" in t)
    verifie("marque comme temoin", "ne decide pas" in t)
    verifie("pas de bloc MODELE RETENU", "MODELE RETENU" not in t)

    print("\n6. UNE LIGNE ILLISIBLE N'EST PAS AVALEE EN SILENCE")
    print("   C'est la ligne qui dit quel modele sera deploye : la perdre")
    print("   serait la pire panne de ce fichier.")
    t = rendu("  ★ NEW BEST  un format que personne n'a prevu")
    verifie("la ligne brute survit", "un format que personne n'a prevu" in t)
    verifie("et la veille le signale", "illisible" in t)

    print()
    if echecs:
        print(f"{len(echecs)} ECHEC(S) : {', '.join(echecs)}")
        return 1
    print("La veille dit quel modele est sauve, sur quel critere, ce qu'il a")
    print("battu et dans quel fichier — et elle dit aussi quand rien n'est")
    print("sauve, ce qu'elle taisait.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
