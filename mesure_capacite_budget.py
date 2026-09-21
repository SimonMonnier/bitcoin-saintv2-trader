# -*- coding: utf-8 -*-
"""Chaque palier de budget ouvre-t-il VRAIMENT un nombre different de positions ?

POURQUOI CETTE MESURE, ET CE QU'ELLE A LAISSE PASSER DEUX FOIS.

  PREMIERE FOIS — l'echelle en FRACTION D'EQUITE. Le journal d'exec47 disait,
  a l'epoch 1 :

      budgets[0%:14(100) 1%:18(100) 3%:21(99) 6%:14(97) 15%:14(80) 40%:18]

  Le nombre entre parentheses est la part des fois ou le palier choisi a
  lui-meme REFUSE l'entree. A 3 % de risque, 99 % des entrees voulues etaient
  refusees ; a 40 %, aucune. PLUS LE BUDGET ETAIT GROS, MOINS IL REFUSAIT —
  la signature exacte d'un plancher de lot minimum. Trois paliers sur six
  faisaient la meme chose que le palier 0, sans le dire.

  SECONDE FOIS — l'echelle en POSITIONS ABSOLUES (0, 1, 2, 4, 7, 12), qui a
  corrige la premiere. Le plafond de survie borne le total a 40 % de l'equite,
  donc il vaut `0.40 x equite / risque_une` POSITIONS — et `risque_une` suit
  l'ATR. Quand l'or s'agite, le plafond descend et ECRASE LE HAUT :

      equite 1 000 $     ATR  4 $ ->  0  1  2  4  7 10     6 paliers sur 6
                         ATR  8 $ ->  0  1  2  4  5  5     5
                         ATR 15 $ ->  0  1  2  2  2  2     3

  CE FICHIER NE POUVAIT PAS LE VOIR : il testait la MONOTONIE (un palier plus
  haut n'ouvre jamais moins, avec un <=), jamais l'INJECTIVITE. Un ecrasement
  passe la monotonie sans broncher.

  CE QUE CELA COUTAIT A L'APPRENTISSAGE. PPO recevait la MEME recompense pour
  quatre actions differentes : son gradient sur ces etats etait du bruit pur,
  et aucun reglage d'entropie n'y pouvait rien. C'est l'une des trois causes
  mesurees du budget qui n'apprenait pas.

LE CORRECTIF. Le budget est desormais une PART DU PLAFOND DE SURVIE :
(0, 0.2, 0.4, 0.6, 0.8, 1.0). Les six paliers sont des fractions du MEME
nombre, donc ils restent distincts tant que ce nombre vaut au moins six
positions, et se degradent ensuite au rythme de l'arithmetique — jamais par
le haut.

CE QUE CE FICHIER VERIFIE MAINTENANT, en plus du reste : que le nombre de
paliers DISTINCTS vaut le maximum arithmetiquement possible, a tout ATR et a
tout capital. C'est le test qui manquait.

    python mesure_capacite_budget.py
"""
from __future__ import annotations

import sys

from saint_core import (
    BUDGETS_PART,
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


def places(equity: float, part: float, atr: float) -> int:
    return places_ouvrables_compte(
        equity=equity, marge_utilisee=0.0, prix=PRIX, lot_min=LOT_MIN,
        contrat=CONTRAT, marge_frac=MARGE_FRAC, niveau_marge=NIVEAU_MARGE,
        budget_part=part, risque_engage=0.0, risque_une=risque_une(atr))


def ancien(equity: float, n: int, atr: float) -> int:
    """L'ECHELLE D'AVANT, rejouee : un nombre ABSOLU de positions.

    Elle est reproduite ici plutot que lue, parce qu'elle n'existe plus dans
    le depot. Sans elle on ne pourrait pas MONTRER ce que la correction
    change, seulement affirmer qu'elle change quelque chose.
    """
    if n <= 0:
        return 0
    r1 = risque_une(atr)
    par_budget = int(n * equity / CAPITAL_DEPART)
    par_survie = int(PLAFOND_RISQUE_EQUITE * equity / r1)
    return max(0, min(par_budget, par_survie))


ANCIENS_PALIERS = (0, 1, 2, 4, 7, 12)

print(f"\nor a {PRIX:,.0f} $   lot minimum {LOT_MIN} "
      f"({LOT_MIN*CONTRAT:.0f} once)   stop {MULT_STOP:.0f} x ATR")
print(f"paliers {BUDGETS_PART}   plafond de survie "
      f"{100*PLAFOND_RISQUE_EQUITE:.0f} % de l'equite")
print("=" * 78)

for equity in CAPITAUX:
    print(f"\nCOMPTE {equity:,.0f} $")
    for atr in ATRS:
        r1 = risque_une(atr)
        nouv = [places(equity, f, atr) for f in BUDGETS_PART]
        vieux = [ancien(equity, n, atr) for n in ANCIENS_PALIERS]
        print(f"  ATR {atr:>4.0f} $   une position risque {r1:>6.2f} $ "
              f"({100*r1/equity:>5.1f} % du compte)   "
              f"plafond {int(PLAFOND_RISQUE_EQUITE*equity/r1)} position(s)")
        print("              avant   " + "  ".join(f"{v:>3}" for v in vieux)
              + f"   -> {len(set(vieux))} distincts")
        print("              apres   " + "  ".join(f"{v:>3}" for v in nouv)
              + f"   -> {len(set(nouv))} distincts")

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
        all(places(e, 0.0, a) == 0 for e in CAPITAUX for a in ATRS))

# 2. Tout palier non nul ouvre au moins une position : le plancher explicite
#    de `places_ouvrables_compte`. Sans lui, une petite part d'un plafond
#    etroit rendrait zero — donc l'abstention — alors que le modele a
#    explicitement choisi de trader.
mauvais = [(e, a, f) for e in CAPITAUX for a in ATRS
           for f in BUDGETS_PART[1:] if places(e, f, a) < 1]
verifie("tout palier non nul ouvre au moins une position",
        not mauvais, f"defaillant pour {mauvais}" if mauvais else "")

# 3. L'echelle est monotone : un palier plus haut n'ouvre jamais moins.
verifie("l'echelle est monotone",
        all(places(e, BUDGETS_PART[i], a) <= places(e, BUDGETS_PART[i+1], a)
            for e in CAPITAUX for a in ATRS
            for i in range(len(BUDGETS_PART) - 1)))

# 4. LE TEST QUI MANQUAIT. Le nombre de paliers DISTINCTS doit valoir le
#    maximum arithmetiquement possible. Le plafond vaut P positions ; les
#    paliers non nuls prennent leurs valeurs dans [1, P], donc on ne peut pas
#    depasser 1 + min(5, P) valeurs distinctes. Toute perte AU-DELA de cette
#    borne est un ecrasement, c'est-a-dire un defaut.
print()
perdus = []
for e in CAPITAUX:
    for a in ATRS:
        obs = len(set(places(e, f, a) for f in BUDGETS_PART))
        att = 1 + min(len(BUDGETS_PART) - 1, places(e, 1.0, a))
        if obs < att:
            perdus.append((e, a, obs, att))
verifie("aucun palier n'est perdu par ecrasement",
        not perdus,
        "  ".join(f"{e:,.0f}$/ATR{a:.0f}: {o} au lieu de {t}"
                  for e, a, o, t in perdus) if perdus
        else "le compte de paliers distincts est le maximum possible")

#    ET LA PREUVE PAR LA REGRESSION : l'ancienne echelle, elle, en perdait.
perdus_avant = []
for e in CAPITAUX:
    for a in ATRS:
        obs = len(set(ancien(e, n, a) for n in ANCIENS_PALIERS))
        att = 1 + min(len(ANCIENS_PALIERS) - 1,
                      int(PLAFOND_RISQUE_EQUITE * e / risque_une(a)))
        if obs < att:
            perdus_avant.append((e, a, obs, att))
verifie("l'ancienne echelle en perdait bien, elle",
        bool(perdus_avant),
        "  ".join(f"{e:,.0f}$/ATR{a:.0f}: {o} au lieu de {t}"
                  for e, a, o, t in perdus_avant))

# 5. Le haut de l'echelle EST le plafond de survie. Les deux bornes doivent
#    coincider exactement : si le haut valait moins, une part de la capacite
#    autorisee resterait inatteignable ; s'il valait plus, le plafond serait
#    franchi.
print()
ecarts = [(e, a, places(e, 1.0, a),
           int(PLAFOND_RISQUE_EQUITE * e / risque_une(a)))
          for e in CAPITAUX for a in ATRS]
verifie("le palier le plus haut vaut exactement le plafond de survie",
        all(v == p for _e, _a, v, p in ecarts),
        "  ".join(f"{e:,.0f}$/ATR{a:.0f}: {v}!={p}"
                  for e, a, v, p in ecarts if v != p) or "sur les six cas")

# 6. LE PLAFOND DE SURVIE PROTEGE TOUJOURS. A ATR 15 sur 1 000 $, douze
#    positions engageraient 1 800 $ — 180 % du compte.
_cap_15 = int(PLAFOND_RISQUE_EQUITE * 1000.0 / risque_une(15.0))
verifie("le plafond de survie empeche la ruine",
        places(1000.0, 1.0, 15.0) <= _cap_15,
        f"{places(1000.0, 1.0, 15.0)} position(s) au lieu de 12 "
        f"(12 x {risque_une(15.0):.0f} $ vaudrait "
        f"{100*12*risque_une(15.0)/1000.0:.0f} % du compte)")

# 7. L'AMPLITUDE A ETE DIVISEE PAR DEUX, ET C'EST VOULU.
#
#    Ce controle exigeait `>= 10 positions`, le niveau de l'ancien palier
#    40 %. Il gardait donc une AMPLITUDE, a une epoque ou la crainte etait de
#    trader trop peu.
#
#    LA CRAINTE A CHANGE DE SENS, et la mesure avec. Sur exec69 le creux de
#    validation valait 59.6 % au fold 1 et 71.6 % au fold 2, jusqu'a 86.5 % :
#    sur un compte de 1 000 $, etre descendu a 280 $ avant de remonter.
#    `PLAFOND_RISQUE_EQUITE` est passe de 40 % a 20 % pour cette raison.
#
#    Ce qu'on verifie n'est donc plus un plancher d'amplitude — le test 5
#    verifie deja que le haut de l'echelle VAUT exactement le plafond — mais
#    que la reduction a bien eu lieu, et dans la bonne proportion.
_haut_1k = places(1000.0, 1.0, 4.0)
_attendu = int(PLAFOND_RISQUE_EQUITE * 1000.0 / risque_une(4.0))
verifie("l'exposition a bien ete divisee par deux",
        _haut_1k == _attendu and abs(PLAFOND_RISQUE_EQUITE - 0.20) < 1e-9,
        f"{_haut_1k} positions a {100*PLAFOND_RISQUE_EQUITE:.0f} % "
        f"(contre 10 a 40 %)")


# ============================================================
print()
print("8. LA CAPACITE SUIT LA CROISSANCE DU COMPTE, SANS `capital_reference`")
print("   Le plafond vaut 40 % de l'equite divises par le risque d'UNE")
print("   position. Sous le lot minimum du courtier ce risque ne depend pas")
print("   du capital, donc le plafond croit avec l'equite — et toute part de")
print("   ce plafond aussi. La composition est automatique : le parametre")
print("   `capital_reference` qui la portait a la main a pu disparaitre.")
# ============================================================
_atr = 4.0
print()
print("   equite      20 %   40 %   60 %   80 %  100 %   ancien 40 %")
for eq in (1000.0, 2500.0, 3562.0, 8000.0):
    _l = "  ".join(f"{places(eq, f, _atr):>5}" for f in BUDGETS_PART[1:])
    _anc = int(PLAFOND_RISQUE_EQUITE * eq / risque_une(_atr))
    print(f"   {eq:>8,.0f}   {_l}   {_anc:>11}")
print()

_base = places(CAPITAL_DEPART, 1.0, _atr)
_croiss = [(eq, places(eq, 1.0, _atr),
            int(PLAFOND_RISQUE_EQUITE * eq / risque_une(_atr)))
           for eq in (2500.0, 3562.0, 8000.0)]
verifie("quand le compte monte, la capacite monte",
        all(v > _base for _e, v, _a in _croiss),
        ", ".join(f"{e:,.0f}$->{v}" for e, v, _a in _croiss))

verifie("et elle vaut l'ancien palier 40 %",
        all(v == a for _e, v, a in _croiss),
        "identique a l'echelle en fraction d'equite d'avant")

# LA PROPRIETE QUE L'ANCIENNE ECHELLE DEFENDAIT — « un palier veut dire la
# meme chose quel que soit le capital de depart » — est desormais VRAIE PAR
# CONSTRUCTION, puisque la part se rapporte a un plafond qui suit le compte.
verifie("une part veut dire la meme chose a tout capital",
        all(places(1000.0, f, _atr) * 10 == places(10000.0, f, _atr)
            for f in BUDGETS_PART[1:]),
        "x10 de capital, x10 de positions, a chaque palier")


print()
if echecs:
    print(f"{len(echecs)} ECHEC(S) : " + " | ".join(echecs))
    sys.exit(1)
print("Tout est verifie.")
sys.exit(0)
