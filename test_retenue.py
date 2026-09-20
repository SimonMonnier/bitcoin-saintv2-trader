# -*- coding: utf-8 -*-
"""Un checkpoint dont le portefeuille a coule n'est ni retenu ni transmis.

POURQUOI CE FICHIER. Le critere de selection est le rendement du SOMMET —
ce que rapportent, en unites de risque, les occasions que le checkpoint
mettrait en position. Il mesure les occasions UNE A UNE et ne sait rien du
nombre tenu simultanement. Depuis que le plafond de risque est retire, un
modele peut donc avoir un classement excellent ET vider le compte en ouvrant
des dizaines de positions correlees : `gain_top` le noterait au sommet.

Une garde refuse ces checkpoints. Elle n'avait jamais ete exercee — aucune
epoch n'avait encore rempli la condition — donc rien ne disait qu'elle
tirait. C'est exactement le genre de chemin que ce depot paie : il tourne,
il rend des nombres, et le jour ou il compte il ne fait rien.

    python test_retenue.py
"""

from __future__ import annotations

import math

import training as T

MIN_TRADES = 20
# Le critere n'est plus un pourcentage de creux mais la DESTRUCTION du
# compte : appel de marge, lot minimum infinancable, equite a zero. Le seuil
# de 40 % a ete retire le 2026-09-19 — il tuait l'episode au quart de sa
# tranche, et un tiers des episodes ainsi tues auraient fini gagnants.

echecs = []


def cas(nom, attendu_retenu, attendu_motif=None, **kw):
    params = dict(gain_top=0.50, score_rang=0.08, val_num_trades=60,
                  val_ruine=0, best_metric=0.20, min_trades=MIN_TRADES,
                  gain_tous=0.10)
    params.update(kw)
    retenu, motif = T.retient_checkpoint(**params)
    # SOUS-CHAINE, PAS PREFIXE. Le motif du garde contre le hasard nomme
    # d'abord le sommet ("sommet +1.401 R ne bat pas le hasard +1.499 R"),
    # parce que c'est le chiffre qu'on veut lire en premier dans le journal.
    # Le seul endroit qui exige vraiment un PREFIXE est le refus pour compte
    # detruit — la veille le compte ainsi — et il est verifie a part, plus bas.
    ok = (retenu == attendu_retenu) and (
        attendu_motif is None or attendu_motif in motif)
    print(("  ok   " if ok else "  RATE ") + f"{nom:<52}"
          + ("retenu" if retenu else f"refuse : {motif}"))
    if not ok:
        echecs.append(nom)


def main() -> int:
    print("CE QUI DOIT ETRE RETENU")
    cas("meilleur sommet, classement positif, compte sain", True)
    cas("creux profond mais compte vivant", True, val_ruine=0)
    # LE RHO NE DECIDE PLUS. Il pesait les 21 000 occasions a egalite quand
    # le deploiement ne regarde que les 5 % du haut ; depuis que la tete est
    # entrainee sur le critere de deploiement lui-meme, les deux se separent.
    # Mesure : exec26 ep.1 affiche rho -0.0101 pour un sommet a +2.255 R
    # contre +1.229 au hasard. L'ancien portillon jetait ce checkpoint.
    cas("rho nul, mais le sommet bat le hasard", True, score_rang=0.0)
    cas("rho negatif, mais le sommet bat le hasard", True, score_rang=-0.05)
    cas("rho non mesurable, mais le sommet bat le hasard", True,
        score_rang=float("nan"))
    cas("le cas exec26 ep.1 en entier", True, gain_top=2.255,
        score_rang=-0.0101, gain_tous=1.229, best_metric=-9e9,
        val_num_trades=105)

    print("\nCE QUI DOIT ETRE REFUSE — le compte detruit est le cas qui compte")
    cas("un compte detruit, sommet pourtant meilleur", False, "compte detruit",
        val_ruine=1, gain_top=9.99)
    cas("plusieurs comptes detruits, sommet excellent", False, "compte detruit",
        val_ruine=4, gain_top=99.0)
    cas("pas assez de trades", False, "seulement",
        val_num_trades=MIN_TRADES - 1)
    cas("sommet egal au hasard", False, "ne bat pas le hasard",
        gain_tous=0.50)
    cas("sommet SOUS le hasard", False, "ne bat pas le hasard",
        gain_tous=0.60)
    cas("le cas exec23 ep.5 : rho bon, sommet sous le hasard", False,
        "ne bat pas le hasard", gain_top=1.401, score_rang=0.072,
        gain_tous=1.499, best_metric=-9e9)
    cas("sommet egal au record", False, "sommet", gain_top=0.20)
    cas("sommet sous le record", False, "sommet", gain_top=0.19)
    cas("sommet non mesurable", False, "sommet", gain_top=float("nan"))
    cas("hasard non mesurable : le garde s'efface, il ne refuse pas", True,
        gain_tous=float("nan"))

    print("\nL'ORDRE DES REFUS : le compte prime sur tout le reste")
    cas("detruit ET trop peu de trades -> le compte d'abord", False, "seulement",
        val_ruine=1, val_num_trades=3)

    print("\nUN REFUS POUR CREUX EST RECONNAISSABLE, c'est lui qui est compte")
    _, motif = T.retient_checkpoint(
        gain_top=9.9, score_rang=0.08, val_num_trades=60, val_ruine=1,
        best_metric=0.2, min_trades=MIN_TRADES)
    ok = motif.startswith("compte detruit")
    print(("  ok   " if ok else "  RATE ")
          + f"{'le motif commence par compte detruit':<52}{motif}")
    if not ok:
        echecs.append("motif compte detruit")

    print()
    if echecs:
        print(f"{len(echecs)} ECHEC(S) : " + ", ".join(echecs))
        return 1
    print("La garde tire : un portefeuille qui a creve le garde-fou ne peut")
    print("pas etre retenu, quel que soit son classement — donc pas transmis")
    print("au fold suivant, donc jamais deploye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
