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
from saint_core import score_retenue

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
    # ------------------------------------------------------------------
    # LE PLANCHER ABSOLU, AJOUTE LE 2026-09-21.
    #
    # CE QUE CE BLOC TESTAIT AVANT : « s'abstenir bat un modele qui perd »,
    # avec `best_metric=-0.03`. L'intention etait juste — mieux vaut ne pas
    # trader que trader mal — mais la regle qui la portait etait purement
    # RELATIVE, et elle a produit exactement le contraire sur exec69 fold 2 :
    # huit checkpoints retenus d'affilee, tous a score negatif, de -0.215 a
    # -0.083. Le « meilleur modele du fold » etait une configuration
    # perdante, et c'est elle qui aurait ete deployee.
    #
    # LA SITUATION QUE CE CAS DECRIVAIT EST DESORMAIS INATTEIGNABLE :
    # `best_metric` ne peut plus valoir -0.03, puisqu'un modele a -0.03
    # n'aurait jamais ete retenu. On teste donc le contrat REEL.
    # ------------------------------------------------------------------
    cas("un modele perdant n'est jamais retenu, meme sans record a battre",
        False, "PERDANTE",
        gain_top=0.50, score_retenue=-0.03, creux=0.10, best_metric=-9e9)
    cas("l'abstention totale non plus : elle ne trade rien",
        False, "PERDANTE",
        gain_top=0.50, score_retenue=0.0, creux=0.0, best_metric=-9e9)
    cas("un modele gagnant, lui, passe des le premier", True,
        gain_top=0.50, score_retenue=0.02, creux=0.10, best_metric=-9e9)

    # ------------------------------------------------------------------
    # LE PORTILLON DU HASARD EXIGE UNE MARGE.
    #
    # Sur exec69 fold 2, le sommet a franchi le hasard de +0.020 R et le
    # checkpoint a ete retenu — alors que l'erreur-type de cette moyenne
    # vaut ~0.24 R, les occasions du haut se chevauchant trente-trois fois.
    # ------------------------------------------------------------------
    cas("un ecart au hasard sous la marge ne passe pas", False, "marge",
        gain_top=0.523, gain_tous=0.503, marge_hasard=0.24,
        score_retenue=0.50, best_metric=-9e9)
    cas("le meme ecart passe si la marge est nulle", True,
        gain_top=0.523, gain_tous=0.503, marge_hasard=0.0,
        score_retenue=0.50, best_metric=-9e9)
    cas("un ecart franc passe malgre la marge", True,
        gain_top=1.200, gain_tous=0.503, marge_hasard=0.24,
        score_retenue=0.50, best_metric=-9e9)

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

    # CE QUE CES CAS TESTAIENT AVANT, ET POURQUOI ILS ONT CHANGE.
    #
    # `score_retenue_grille(budgets, rendements)` ponderait chaque occasion
    # par la mise du modele, et les cas verifiaient l'invariance au levier,
    # le gain de l'abstention et la punition d'une grosse mise au mauvais
    # moment. Le budget a ete retire le 2026-09-21 : la taille posee valait
    # 1.0000 unite, p5 0.9999, p95 1.0001, sur 7 363 trades — les six
    # paliers rendaient tous la meme taille, parce que la taille voulue par
    # le risque tombe sous le lot minimum du courtier.
    #
    # `score_retenue(rendements)` ne pondere plus. L'abstention n'a pas
    # disparu pour autant : elle est passee de « miser zero sur une
    # occasion retenue » a « ne pas la retenir », et c'est `_top` qui la
    # porte. Les cas ci-dessous testent donc la MEME propriete economique,
    # au bon endroit.
    _gain, _baisse, _net = score_retenue(_R)
    cas_direct("gain = moyenne des rendements retenus",
               abs(_gain - float(_R.mean())) < 1e-12,
               f"{_gain:+.4f}")
    cas_direct("baisse = moyenne quadratique des seules PERTES",
               abs(_baisse - float(_np.sqrt(
                   (_np.minimum(_R, 0.0) ** 2).mean()))) < 1e-12,
               f"{_baisse:.4f}")
    cas_direct("net = gain - baisse", abs(_net - (_gain - _baisse)) < 1e-12,
               f"{_gain:+.4f} - {_baisse:.4f} = {_net:+.4f}")

    # ECARTER LES MAUVAISES OCCASIONS PAIE, et c'est tout l'objet du critere.
    _tout = score_retenue(_R)[2]
    _trie = score_retenue(_R[_R > -0.5])[2]
    cas_direct("ecarter les occasions perdantes monte le score",
               _trie > _tout,
               f"{_tout:+.4f} -> {_trie:+.4f} en jetant "
               f"{int((_R <= -0.5).sum())} occasions sur {len(_R)}")

    # SANS AUCUNE PERTE, LA BAISSE EST NULLE et le net vaut le gain.
    _g2, _b2, _n2 = score_retenue(_np.abs(_R))
    cas_direct("aucune perte -> baisse nulle, net = gain",
               _b2 == 0.0 and abs(_n2 - _g2) < 1e-12,
               f"gain {_g2:+.4f}  baisse {_b2:.4f}")

    # LE SCORE EST DANS L'UNITE DU RENDEMENT, et ce n'est plus une
    # invariance mais une propriete voulue : sans budget il n'y a plus de
    # levier a neutraliser, et le critere doit dire combien on gagne.
    _k = score_retenue(3.0 * _R)[2]
    cas_direct("le score suit l'echelle des rendements",
               abs(_k - 3.0 * _net) < 1e-9,
               f"x3 -> {_net:+.4f} devient {_k:+.4f}")

    # RIEN A NOTER REND EXACTEMENT ZERO, pas une division par zero.
    cas_direct("une liste vide rend zero",
               score_retenue(_np.array([])) == (0.0, 0.0, 0.0))
    cas_direct("des rendements tous non finis rendent zero",
               score_retenue(_np.full(5, _np.nan)) == (0.0, 0.0, 0.0),
               "les NaN sont ecartes, pas comptes pour zero")

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
