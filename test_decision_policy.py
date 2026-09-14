"""Tests de regression du filtre d'entree.

Trois defauts ont ete trouves dans la premiere version du filtre par rang
glissant. Chacun etait invisible : aucune exception, aucun journal, juste des
resultats de validation qui ne voulaient rien dire. Ce fichier les fige.

    python test_decision_policy.py

Sortie attendue : "8/8 OK". Tout echec ici invalide les mesures de validation
d'un run, pas seulement une ligne de code.
"""

import sys

import numpy as np

from saint_core import (SeuilRang, EntryDecisionPolicy, rolling_decision_spec,
                        load_decision_policy)

_ok = _total = 0


def verifie(nom, condition, detail=""):
    global _ok, _total
    _total += 1
    if condition:
        _ok += 1
        print(f"  OK    {nom}")
    else:
        print(f"  ECHEC {nom}   {detail}")


def _echantillon(n=1000, graine=0):
    return list(np.random.default_rng(graine).normal(0.33, 0.01, n))


# ============================================================
# DEFAUT 1 — historiques partages entre episodes
#
# Deux SeuilRang etaient partages par les 48 environnements de validation,
# pourtant situes a des dates differentes. Une decision dependait donc des
# probabilites des autres episodes ET de leur ordre d'execution.
# ============================================================

def test_independance():
    spec = rolling_decision_spec(0.025, 500, [_echantillon(300), _echantillon(300, 1)])
    a, b = EntryDecisionPolicy(spec), EntryDecisionPolicy(spec)
    reference = b.thresholds
    for _ in range(400):
        a.decide(0.95, 0.95)          # on sature l'historique de A
    verifie("deux instances n'echangent rien",
            b.thresholds == reference,
            f"B a bouge : {reference} -> {b.thresholds}")
    verifie("l'instance saturee a bien bouge, elle",
            a.thresholds != reference)


def test_copie_profonde():
    """Muter la spec apres coup ne doit pas atteindre une instance vivante."""
    spec = rolling_decision_spec(0.025, 500, [_echantillon(300), _echantillon(300, 1)])
    p = EntryDecisionPolicy(spec)
    avant = p.thresholds
    spec['bootstrap'][0].append(99.0)
    spec['fraction_per_side'] = 0.9
    verifie("la spec est copiee en profondeur", p.thresholds == avant)


# ============================================================
# DEFAUT 2 — egalites
#
# `p >= quantile` acceptait TOUT quand la distribution est degeneree : pour
# 1000 probabilites identiques et une cible de 2.5 %, 1000 entrees etaient
# ouvertes. C'est exactement l'etat d'une politique en debut d'entrainement,
# ou l'entropie vaut ln(3) et les probabilites sont quasi uniformes.
# ============================================================

def test_egalites():
    for valeur in (0.3333, 0.5, 0.12345, 1e-6):
        s = SeuilRang(0.025, 1000, amorce=[valeur] * 1000)
        acceptes = sum(1 for _ in range(1000) if valeur >= s.seuil())
        verifie(f"p constante {valeur:<9} -> aucune entree",
                acceptes == 0, f"{acceptes}/1000 acceptees")


def test_cible_tenue():
    """La correction des egalites ne doit pas casser le cas normal."""
    ech = _echantillon(2000)
    s = SeuilRang(0.025, 2000, amorce=ech)
    part = float(np.mean(np.asarray(ech) >= s.seuil()))
    verifie("distribution reelle : cible 2.5 % tenue",
            abs(part - 0.025) < 0.005, f"{100*part:.2f} %")


def test_abstention_avant_amorce():
    """Sous le minimum d'echantillons, on s'abstient au lieu de tirer au sort."""
    s = SeuilRang(0.025, 1000)
    verifie("fenetre vide -> seuil infini -> HOLD",
            not np.isfinite(s.seuil()))


# ============================================================
# DEFAUT 3 — regle differente entre selection et execution
#
# La validation utilisait le rang glissant ; le test final et le live
# gardaient des seuils figes. Le checkpoint etait donc choisi sous une regle
# et execute sous une autre. Mesure de ce que coute cet ecart : l'etendue des
# convictions ayant ete multipliee par 26 entre deux epochs, le nombre de
# trades etait tombe de 1461 a 20, dont zero vente.
# ============================================================

def test_meme_objet_partout():
    spec = rolling_decision_spec(0.025, 500, [_echantillon(300), _echantillon(300, 1)])
    p = EntryDecisionPolicy(spec)
    verifie("le mode par defaut est le rang glissant",
            p.mode == 'rolling_rank', p.mode)


def test_repli_checkpoint_ancien():
    """Un checkpoint sans spec doit tomber en mode 'fixed', pas planter."""
    import json, tempfile, os
    with tempfile.TemporaryDirectory() as d:
        pth = os.path.join(d, "modele.pth")
        with open(pth.replace(".pth", "_calib.json"), "w", encoding="utf-8") as f:
            json.dump({"calib_thr_buy": 0.4, "calib_thr_sell": 0.35,
                       "calib_thr": 0.4}, f)
        p = load_decision_policy(pth)
    verifie("checkpoint ancien -> mode fixed",
            p.mode == 'fixed' and p.thresholds == (0.4, 0.35),
            f"{p.mode} {p.thresholds}")


if __name__ == "__main__":
    print("Filtre d'entree — tests de regression\n")
    test_independance()
    test_copie_profonde()
    test_egalites()
    test_cible_tenue()
    test_abstention_avant_amorce()
    test_meme_objet_partout()
    test_repli_checkpoint_ancien()
    print(f"\n{_ok}/{_total} OK")
    sys.exit(0 if _ok == _total else 1)
