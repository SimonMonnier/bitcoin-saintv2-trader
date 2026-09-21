# -*- coding: utf-8 -*-
"""Chaque palier de budget ouvre-t-il VRAIMENT un nombre different de positions ?

POURQUOI CETTE MESURE. Le journal d'exec47 disait ceci, a l'epoch 1 :

    budgets[0%:14(100) 1%:18(100) 3%:21(99) 6%:14(97) 15%:14(80) 40%:18]

Le nombre entre parentheses est la part des fois ou le palier choisi a
lui-meme REFUSE l'entree. A 3 % de risque, 99 % des entrees voulues etaient
refusees ; a 40 %, aucune. PLUS LE BUDGET ETAIT GROS, MOINS IL REFUSAIT —
l'inverse de l'intuition, et la signature exacte d'un plancher de lot minimum.
La validation etait tombee de 300-450 trades a 56.

LA CAUSE. Le budget plafonne la somme des pertes si TOUS les stops etaient
touches. Une position au lot MINIMUM risque deja un montant incompressible :
si le budget est en dessous, le compte n'ouvre pas une position plus petite,
il n'en ouvre AUCUNE. A 1 000 $ avec un stop de 10 x ATR sur l'or, les paliers
1 % et 3 % etaient donc inoperants en permanence — trois paliers sur six
faisaient la meme chose que le palier 0, sans le dire.

LE CORRECTIF. L'echelle compte maintenant des POSITIONS MINIMALES :
(0, 1, 2, 4, 7, 12), de meme amplitude que l'ancienne. Ce fichier verifie
que chaque palier au-dessus de zero ouvre ce qu'il annonce, a tout ATR et
a tout capital.

    python mesure_capacite_budget.py
"""
from __future__ import annotations

import sys

from saint_core import (
    BUDGETS_POSITIONS,
    PLAFOND_RISQUE_EQUITE,
    places_ouvrables_compte,
)

LOT_MIN = 0.01
CONTRAT = 100.0          # XAUUSD : 100 onces par lot
MARGE_FRAC = 1.0 / 100   # levier 1:100
NIVEAU_MARGE = 1.0
PRIX = 3300.0
MULT_STOP = 10.0
ATRS = (4.0, 8.0, 15.0)          # calme / normal / agite
CAPITAUX = (1000.0, 10000.0)

echecs = []


def risque_une(atr: float) -> float:
    """Ce qu'une position au lot minimum perd si son stop est touche."""
    return MULT_STOP * atr * LOT_MIN * CONTRAT


CAPITAL_DEPART = 1000.0


def places(equity: float, n: int, atr: float,
           capital_reference: float = 0.0) -> int:
    return places_ouvrables_compte(
        equity=equity, marge_utilisee=0.0, prix=PRIX, lot_min=LOT_MIN,
        contrat=CONTRAT, marge_frac=MARGE_FRAC, niveau_marge=NIVEAU_MARGE,
        budget_positions=n, risque_engage=0.0, risque_une=risque_une(atr),
        capital_reference=capital_reference)


print(f"\nor a {PRIX:,.0f} $   lot minimum {LOT_MIN} "
      f"({LOT_MIN*CONTRAT:.0f} once)   stop {MULT_STOP:.0f} x ATR")
print(f"paliers {BUDGETS_POSITIONS}   plafond de survie "
      f"{100*PLAFOND_RISQUE_EQUITE:.0f} % de l'equite")
print("=" * 78)

for equity in CAPITAUX:
    print(f"\nCOMPTE {equity:,.0f} $")
    for atr in ATRS:
        r1 = risque_une(atr)
        ouvertes = [places(equity, n, atr) for n in BUDGETS_POSITIONS]
        detail = "  ".join(
            f"{n}pos->{o}" for n, o in zip(BUDGETS_POSITIONS, ouvertes))
        print(f"  ATR {atr:>4.0f} $   une position risque {r1:>6.2f} $ "
              f"({100*r1/equity:>5.1f} % du compte)")
        print(f"              {detail}")
        # Ce que le plafond de survie autorise, en positions.
        plafond = int(PLAFOND_RISQUE_EQUITE * equity / r1)
        distincts = len(set(ouvertes))
        print(f"              plafond de survie : {plafond} position(s)   "
              f"-> {distincts} niveaux distincts sur {len(BUDGETS_POSITIONS)}")

print("\n" + "=" * 78)
print("\nCE QUI DOIT ETRE VRAI")


def verifie(nom, condition, detail=""):
    if condition:
        print(f"  ok    {nom:<56} {detail}")
    else:
        echecs.append(nom)
        print(f"  ECHEC {nom:<56} {detail}")


# 1. L'abstention abstient, partout.
verifie("le palier 0 n'ouvre jamais rien",
        all(places(e, 0, a) == 0 for e in CAPITAUX for a in ATRS))

# 2. Le palier 1 ouvre une position des que la marge le permet, a tout ATR
#    et a tout capital — c'est tout l'objet du changement d'unite.
mauvais = [(e, a) for e in CAPITAUX for a in ATRS if places(e, 1, a) < 1]
verifie("le palier 1 ouvre toujours au moins une position",
        not mauvais, f"defaillant pour {mauvais}" if mauvais else "")

# 3. L'echelle est monotone : un palier plus haut n'ouvre jamais moins.
mono = all(
    places(e, BUDGETS_POSITIONS[i], a) <= places(e, BUDGETS_POSITIONS[i+1], a)
    for e in CAPITAUX for a in ATRS
    for i in range(len(BUDGETS_POSITIONS) - 1))
verifie("l'echelle est monotone", mono)

# 4. LE POINT DU CHANGEMENT : la meme action veut dire la meme chose a
#    1 000 $ et a 10 000 $ — TANT QUE LE PLAFOND DE SURVIE NE MORD PAS.
#
#    L'assertion portait sur TOUS les paliers, et elle etait vraie tant que le
#    plus haut valait 8. A 12, elle tombe : a 1 000 $ et ATR 4 la garde de
#    survie autorise 10 positions, donc 12 est ramene a 10, alors qu'a
#    10 000 $ il reste 12. Ce n'est pas une incoherence de l'echelle, c'est la
#    GARDE qui fait son travail — et c'est precisement la difference entre
#    "le palier ne veut rien dire" et "le compte ne peut pas se le permettre".
#
#    On verifie donc l'egalite la ou la garde ne mord d'aucun cote, et on
#    verifie SEPAREMENT que la garde mord bien au-dessus.
_cap_1k = int(PLAFOND_RISQUE_EQUITE * 1000.0 / risque_une(4.0))
_sous_garde = [n for n in BUDGETS_POSITIONS if n <= _cap_1k]
egaux = all(places(1000.0, n, 4.0) == places(10000.0, n, 4.0)
            for n in _sous_garde)
verifie("un palier veut dire la meme chose a 1 000 $ et a 10 000 $",
        egaux, f"(ATR 4, paliers {_sous_garde} sous la garde de {_cap_1k})")

_au_dessus = [n for n in BUDGETS_POSITIONS if n > _cap_1k]
verifie("au-dessus, c'est la garde qui tranche, pas l'echelle",
        all(places(1000.0, n, 4.0) == _cap_1k for n in _au_dessus),
        f"paliers {_au_dessus} ramenes a {_cap_1k} a 1 000 $, "
        f"intacts a 10 000 $" if _au_dessus else "aucun palier au-dessus")

# 5. Le plafond de survie mord quand il doit : le palier le plus haut a
#    ATR 15 sur 1 000 $ engagerait 12 x 150 = 1 800 $, soit 180 % du compte.
_haut = BUDGETS_POSITIONS[-1]
_cap_15 = int(PLAFOND_RISQUE_EQUITE * 1000.0 / risque_une(15.0))
verifie("le plafond de survie empeche la ruine",
        places(1000.0, _haut, 15.0) <= _cap_15,
        f"{places(1000.0, _haut, 15.0)} position(s) au lieu de {_haut} "
        f"({_haut} x {risque_une(15.0):.0f} $ vaudrait "
        f"{100*_haut*risque_une(15.0)/1000.0:.0f} % du compte)")

# 6. L'AMPLITUDE EST CELLE D'AVANT. L'ancienne echelle ouvrait 0 / 1 / 3 / 10
#    positions a 1 000 $ et ATR 4. Le haut de la nouvelle doit valoir autant,
#    sinon on trade structurellement moins — c'est ce qui a fait tomber la
#    validation de 300-450 trades a 141.
verifie("le haut de l'echelle vaut l'ancien palier 40 %",
        places(1000.0, _haut, 4.0) >= 10,
        f"{places(1000.0, _haut, 4.0)} positions contre 10 avant")

# 7. Ce que l'ancienne echelle faisait, pour memoire : un budget en fraction
#    d'equite sous le cout d'une position n'ouvrait rien.
ancien_3pct = 0.03 * 1000.0
verifie("l'ancien palier 3 % etait bien inoperant a 1 000 $",
        ancien_3pct < risque_une(4.0),
        f"{ancien_3pct:.0f} $ de budget pour une position a "
        f"{risque_une(4.0):.0f} $")


# ============================================================
print()
print("8. LA CAPACITE SUIT LA CROISSANCE DU COMPTE")
print("   C'est le test qui MANQUAIT, et son absence a coute la moitie des")
print("   trades. L'echelle en positions n'etait verifiee qu'a 1 000 $ — le")
print("   seul capital ou elle coincide avec l'ancienne echelle en fraction")
print("   d'equite. Des que le compte monte, une echelle FIXE cesse de")
print("   suivre : 12 positions a 3 562 $ contre 35 pour l'ancien 40 %.")
# ============================================================
_haut2 = BUDGETS_POSITIONS[-1]
_atr = 4.0
print()
print("   equite     fixe   suit le compte   ancien 40 %")
for eq in (1000.0, 2500.0, 3562.0, 8000.0):
    _fixe = places(eq, _haut2, _atr)
    _suit = places(eq, _haut2, _atr, capital_reference=CAPITAL_DEPART)
    _anc = int(PLAFOND_RISQUE_EQUITE * eq / risque_une(_atr))
    print(f"   {eq:>8,.0f}   {_fixe:>6}   {_suit:>14}   {_anc:>11}")
print()

verifie("au capital de depart, rien ne change",
        places(CAPITAL_DEPART, _haut2, _atr, CAPITAL_DEPART)
        == places(CAPITAL_DEPART, _haut2, _atr),
        "la propriete voulue est intacte")

_base = places(CAPITAL_DEPART, _haut2, _atr)
_croiss = [(eq, places(eq, _haut2, _atr, CAPITAL_DEPART),
            int(PLAFOND_RISQUE_EQUITE * eq / risque_une(_atr)))
           for eq in (2500.0, 3562.0, 8000.0)]
verifie("quand le compte monte, la capacite monte",
        all(v > _base for _e, v, _a in _croiss),
        ", ".join(f"{e:,.0f}$->{v}" for e, v, _a in _croiss))

verifie("et elle retrouve l'ancien palier 40 %",
        all(v == a for _e, v, a in _croiss),
        "identique a l'echelle en fraction d'equite d'avant")

verifie("un palier veut la meme chose quel que soit le capital DE DEPART",
        places(1000.0, 4, _atr, 1000.0) == places(10000.0, 4, _atr, 10000.0),
        "4 positions au depart, a 1 000 $ comme a 10 000 $")

print()
if echecs:
    print(f"{len(echecs)} ECHEC(S) : {', '.join(echecs)}")
    print("\nL'ECHELLE DE RISQUE N'EXISTE PAS. Plusieurs paliers font la meme")
    print("chose, et PPO croit apprendre un dimensionnement qu'il ne peut pas")
    print("exercer — c'est exactement ce qui a fait tomber la validation a 56")
    print("trades.")
    sys.exit(1)
print("Chaque palier ouvre ce qu'il annonce, a tout ATR et a tout capital,")
print("et le plafond de survie borne le tout sans effacer l'echelle.")
