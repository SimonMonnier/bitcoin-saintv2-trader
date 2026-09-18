# -*- coding: utf-8 -*-
"""Un run long-only ne doit produire AUCUNE vente. Nulle part.

CE QUE CE TEST DEFEND, et pourquoi il existe. `or_exec02` a tourne 273 epochs
sur trois folds en `side="long"`. L'entrainement affichait `S(0W/0L) +0.00$` a
chaque epoch : le rollout respectait bien le masque d'actions. La VALIDATION,
elle, comptait 31 a 68 % de ventes selon le fold.

    fold   LONG        SHORT       part des trades vendus
    wf1    +87.9 $     -286.9 $    68 %
    wf2    +421.1 $    -108.5 $    31 %
    wf3    +517.1 $    -435.8 $    64 %

Les ventes ont emporte 81 % du gain des achats, et le "meilleur modele" a ete
choisi sur ce net. Le test de la fenetre reservee a ete consomme avec la meme
regle.

LA CAUSE. Validation, calibration et test calculent d'abord les probabilites a
partir des logits MASQUES, puis — quand `tri_par_tete_aux` est vrai — les
ECRASENT par la sortie de la tete auxiliaire :

    probs_np = sigmoid(policy.rendement(st))

La tete auxiliaire predit les deux sens sans rien savoir du cote autorise. Le
masque disparaissait avec le tableau qu'il avait produit.

DEUX VERROUS PLUTOT QU'UN, parce qu'un seul avait deja cede :
  1. la REGLE de decision connait le cote et refuse l'autre ;
  2. `training.py` leve si le cote interdit compte un seul trade.

    python test_cote.py
"""

from __future__ import annotations

import numpy as np

from saint_core import (EntryDecisionPolicy, cotes_permises,
                        decide_avec_barres, rolling_decision_spec)

ACHAT, VENTE, ATTENDRE = 0, 1, 2


def _cas(nom, obtenu, attendu, echecs):
    ok = obtenu == attendu
    print(("  ok   " if ok else "  RATE ") + nom
          + ("" if ok else f"   (obtenu {obtenu}, attendu {attendu})"))
    if not ok:
        echecs.append(nom)


def main() -> int:
    echecs = []
    barres = (0.5, 0.5)

    # La vente est PLUS NETTE que l'achat : c'est exactement la configuration
    # qui fuyait, puisque `decide` retient le meilleur des deux cotes.
    print("regle de decision, vente plus convaincante que l'achat :")
    _cas("both  retient la vente",
         decide_avec_barres(0.60, 0.99, barres, "both"), VENTE, echecs)
    _cas("long  retient l'achat quand meme",
         decide_avec_barres(0.60, 0.99, barres, "long"), ACHAT, echecs)
    _cas("long  attend si l'achat ne passe pas sa barre",
         decide_avec_barres(0.40, 0.99, barres, "long"), ATTENDRE, echecs)
    _cas("short retient la vente",
         decide_avec_barres(0.99, 0.60, barres, "short"), VENTE, echecs)
    _cas("short attend si la vente ne passe pas sa barre",
         decide_avec_barres(0.99, 0.40, barres, "short"), ATTENDRE, echecs)

    # Le rang glissant est le mode REELLEMENT utilise en validation et en
    # test : le verifier sur la regle nue ne suffirait pas.
    print("\npolitique a rang glissant, 4 000 occasions tirees au hasard :")
    rng = np.random.default_rng(0)
    amorce = [list(rng.random(500)), list(rng.random(500))]
    for cote, interdit, nom_interdit in (("long", VENTE, "ventes"),
                                         ("short", ACHAT, "achats")):
        pol = EntryDecisionPolicy(rolling_decision_spec(0.05, 500, amorce, cote))
        compte = {ACHAT: 0, VENTE: 0, ATTENDRE: 0}
        for _ in range(4000):
            compte[pol.decide(float(rng.random()), float(rng.random()))] += 1
        _cas(f"{cote:<5} : zero {nom_interdit} "
             f"({compte[1 - interdit]} trades du cote permis)",
             compte[interdit], 0, echecs)
    pol = EntryDecisionPolicy(rolling_decision_spec(0.05, 500, amorce, "both"))
    compte = {ACHAT: 0, VENTE: 0, ATTENDRE: 0}
    for _ in range(4000):
        compte[pol.decide(float(rng.random()), float(rng.random()))] += 1
    _cas("both  : les deux cotes tradent",
         compte[ACHAT] > 0 and compte[VENTE] > 0, True, echecs)

    # Le cote voyage DANS le checkpoint : le live recharge la specification et
    # ne peut pas la contredire par oubli d'argument.
    print("\nle cote survit a la specification :")
    _cas("aller-retour de spec",
         EntryDecisionPolicy(
             rolling_decision_spec(0.05, 500, amorce, "long")).side,
         "long", echecs)
    _cas("spec anterieure sans cle 'side' -> both",
         EntryDecisionPolicy({'version': 1, 'mode': 'fixed',
                              'thresholds': [0.5, 0.5]}).side, "both", echecs)

    # La selectivite se divise par le nombre de cotes OUVERTS. Diviser par
    # deux en long-only faisait trader a la moitie de la frequence demandee.
    print("\nnombre de cotes ouverts :")
    _cas("long", sum(cotes_permises("long")), 1, echecs)
    _cas("short", sum(cotes_permises("short")), 1, echecs)
    _cas("both", sum(cotes_permises("both")), 2, echecs)

    print()
    if echecs:
        print(f"{len(echecs)} ECHEC(S) : " + ", ".join(echecs))
        return 1
    print("tout passe — le cote interdit ne trade nulle part.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
