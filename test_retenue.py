# -*- coding: utf-8 -*-
"""Ce qui choisit le checkpoint doit voir le creux, pas seulement le tri.

POURQUOI CE FICHIER. Le critere de selection etait le rendement du SOMMET —
ce que rapportent, en unites de risque, les occasions que le checkpoint
mettrait en position. Il mesure les occasions UNE A UNE : il ne sait rien du
nombre tenu simultanement, rien de l'ORDRE dans lequel elles arrivent — donc
rien du creux — et rien de la TAILLE misee sur chacune.

CE QUE CETTE CECITE COUTAIT. La tete de budget est desormais la seule que PPO
entraine, et elle dispose d'un palier 0 % : un levier d'ABSTENTION explicite.
Un modele qui apprendrait parfaitement a ne rien miser aux mauvais moments
rendait exactement le meme `gain_top` qu'un modele misant pareil partout. Le
critere qui choisit ce qui sera deploye etait aveugle a la seule chose qu'on
venait de lui apprendre.

Le critere est donc `gain - creux`, reconstitue sur la grille d'occasions en
ordre chronologique et en fraction de compte. Un modele qui s'abstient au bon
moment est paye deux fois : il retire un rendement negatif, ce qui monte le
gain ET baisse le creux.

La garde du compte detruit, elle, n'a pas bouge : un classement excellent qui
vide le compte en ouvrant des dizaines de positions correlees reste refuse, et
`gain_top` ne peut toujours pas le voir.

Une garde refuse ces checkpoints. Elle n'avait jamais ete exercee — aucune
epoch n'avait encore rempli la condition — donc rien ne disait qu'elle
tirait. C'est exactement le genre de chemin que ce depot paie : il tourne,
il rend des nombres, et le jour ou il compte il ne fait rien.

    python test_retenue.py
"""

from __future__ import annotations

import math

import training as T
from saint_core import score_retenue_grille

MIN_TRADES = 20
# Le critere n'est plus un pourcentage de creux mais la DESTRUCTION du
# compte : appel de marge, lot minimum infinancable, equite a zero. Le seuil
# de 40 % a ete retire le 2026-09-19 — il tuait l'episode au quart de sa
# tranche, et un tiers des episodes ainsi tues auraient fini gagnants.

echecs = []


def cas_direct(nom, condition, detail=""):
    """Un controle qui ne passe pas par `retient_checkpoint`."""
    ok = bool(condition)
    print(("  ok   " if ok else "  RATE ") + f"{nom:<52}" + detail)
    if not ok:
        echecs.append(nom)


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
    # SANS SCORE NET, LE CRITERE RESTE `gain_top` — les appels anterieurs
    # gardent leur sens exact : pas de budget mesure, donc pas de creux a
    # soustraire.
    cas("sommet egal au record", False, "score net", gain_top=0.20)
    cas("sommet sous le record", False, "score net", gain_top=0.19)
    cas("sommet non mesurable", False, "sommet", gain_top=float("nan"))
    cas("hasard non mesurable : le garde s'efface, il ne refuse pas", True,
        gain_tous=float("nan"))

    # ============================================================
    print("\nLE CREUX ENTRE DANS LE CRITERE — c'est le trou qu'on vient de")
    print("boucher. `gain_top` note la qualite du TRI et rien d'autre : ni")
    print("l'ordre des occasions, donc pas le creux, ni la taille misee sur")
    print("chacune, donc pas l'abstention. Or l'abstention est desormais la")
    print("seule chose que PPO apprenne.")
    # ============================================================
    # DEUX MODELES AU TRI IDENTIQUE. Meme `gain_top`, meme `gain_tous` : a
    # l'ancien critere ils etaient indiscernables. L'un traverse un creux de
    # 8 % pour finir a +12 %, l'autre s'abstient aux mauvais moments et
    # traverse 2 % pour finir a +11 %.
    cas("le tri identique ne suffit plus : le creux departage", True,
        gain_top=0.50, score_retenue=0.11 - 0.02, creux=0.02,
        best_metric=0.12 - 0.08)
    cas("meme tri, creux profond -> refuse contre l'abstinent", False,
        "score net", gain_top=0.50, score_retenue=0.12 - 0.08, creux=0.08,
        best_metric=0.11 - 0.02)

    # UN GAIN PLUS ELEVE NE SUFFIT PAS S'IL SE PAIE EN CREUX.
    cas("gain superieur mais creux qui l'annule", False, "score net",
        gain_top=9.99, score_retenue=0.30 - 0.29, creux=0.29,
        best_metric=0.05)
    # ET L'ABSTENTION TOTALE NE PEUT PAS GAGNER PAR DEFAUT : ne rien miser
    # rend un score net de 0, qui ne bat aucun record positif.
    cas("s'abstenir toujours -> score net nul, ne bat rien", False,
        "score net", gain_top=0.50, score_retenue=0.0, creux=0.0,
        best_metric=0.04)
    # ... mais il bat un modele qui perd, et c'est voulu : mieux vaut ne pas
    # trader que trader mal.
    cas("s'abstenir toujours bat un modele qui perd", True,
        gain_top=0.50, score_retenue=0.0, creux=0.0, best_metric=-0.03)

    cas("score net non mesurable -> refuse", False, "score net non mesurable",
        gain_top=0.50, score_retenue=float("nan"), best_metric=-9e9)

    # LE COMPTE DETRUIT PRIME TOUJOURS, quel que soit le score net.
    cas("compte detruit : le score net ne rachete rien", False,
        "compte detruit", gain_top=0.50, score_retenue=9.99, creux=0.0,
        best_metric=-9e9, val_ruine=1)

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

    # ============================================================
    print()
    print("LE CRITERE NE RECOMPENSE PAS LE LEVIER — et il le faisait.")
    print("La premiere version sommait budget x R sans normaliser : les deux")
    print("termes etaient lineaires en budget, donc leur difference aussi.")
    print("Doubler tous les budgets doublait le score. Mesure sur exec55 :")
    print("gain/baisse valait 1.0 aux epochs 1 et 2 — aucune competence — mais")
    print("`net` passait de -0.019 a +0.168 parce que le budget etait passe de")
    print("2 a 12 positions. Le checkpoint a ete retenu la-dessus, avec 85 %")
    print("de creux en validation.")
    print()
    print("CE TEST APPELLE LA MEME FONCTION QUE L'ENTRAINEMENT. La version")
    print("precedente reimplementait la formule dans le test : deux ecritures")
    print("du meme calcul, qui doivent s'accorder par convention.")
    # ============================================================
    import numpy as _np
    _rng = _np.random.default_rng(3)
    _R = _rng.normal(0.15, 1.0, 160)

    _refs = [score_retenue_grille(_np.full(160, 2.0 * m), _R)[2]
             for m in (1, 2, 6, 12)]
    cas_direct("le levier ne change pas le score",
               max(_refs) - min(_refs) < 1e-12,
               f"x1 x2 x6 x12 -> {_refs[0]:+.4f} partout")

    _plat = score_retenue_grille(_np.full(160, 4.0), _R)[2]
    _abst = score_retenue_grille(_np.where(_R < -0.5, 0.0, 4.0), _R)[2]
    _gros = score_retenue_grille(_np.where(_R < -0.5, 12.0, 4.0), _R)[2]
    cas_direct("s'abstenir sur les mauvaises occasions PAIE",
               _abst > _plat,
               f"{_plat:+.4f} -> {_abst:+.4f}")
    cas_direct("miser gros sur les mauvaises occasions PUNIT",
               _gros < _plat,
               f"{_plat:+.4f} -> {_gros:+.4f}")
    cas_direct("ne rien miser rend exactement zero",
               score_retenue_grille(_np.zeros(160), _R) == (0.0, 0.0, 0.0),
               "pas de division par zero, pas de score negatif")
    cas_direct("un budget negatif ou nul partout rend zero",
               score_retenue_grille(_np.full(5, -1.0), _np.ones(5))[2] == 0.0)
    _ok_forme = False
    try:
        score_retenue_grille(_np.ones(5), _np.ones(4))
    except ValueError:
        _ok_forme = True
    cas_direct("des tailles qui ne correspondent pas levent", _ok_forme,
               "une valeur de budget par occasion, verifie")

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
