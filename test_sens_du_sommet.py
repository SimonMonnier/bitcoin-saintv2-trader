# -*- coding: utf-8 -*-
"""Le sommet se mesure sur le cote QUE LE MODELE CHOISIT.

CE QUI A RENDU CE TEST NECESSAIRE, le 2026-09-21. Le diagnostic calculait
le gain du « cote joue » par `np.maximum(achat, vente)` des que les deux
cotes etaient permis : le rendement du MEILLEUR cote, choisi apres coup.
Ce tableau entre droit dans le critere de retenue — `score_retenue_grille`
et le portillon du hasard le lisent tous les deux.

La branche dormait derriere le long-only, ou il n'y a pas de choix a
faire ; elle s'est reveillee en ouvrant la vente. Et la selection avait le
defaut symetrique : `_top` prenait les scores les plus HAUTS, donc les
plus fortes convictions d'ACHAT, les ventes ayant un score negatif par
construction. On selectionnait des achats et on les payait au tarif du
meilleur des deux cotes.

CE QUE CA OFFRAIT, mesure par ce test sur un modele qui ne sait RIEN :
+0.510 sigma par occasion. L'exces reel annonce a l'epoch 1 valait +0.026.

    python test_sens_du_sommet.py
"""
import numpy as np
from saint_core import cotes_permises

rng = np.random.default_rng(0)
n = 20000
ra = rng.normal(-0.18, 0.85, n)
rv = rng.normal(-0.24, 0.85, n)
# Un modele qui voit un peu : le score suit la part symetrique, bruitee.
vrai = (ra - rv) / 2.0
scores = 0.35 * vrai + rng.normal(0, 1.0, n)

_ok = _ko = 0


def verifie(nom, cond, detail=""):
    global _ok, _ko
    if cond:
        _ok += 1
        print("  ok   %-56s %s" % (nom, detail))
    else:
        _ko += 1
        print("  ECHEC %-55s %s" % (nom, detail))


def selection(side, q=0.05):
    ca, cv = cotes_permises(side)
    if ca and cv:
        sens = np.where(scores >= 0.0, 1.0, -1.0)
        conv = np.abs(scores)
    elif cv:
        sens = np.full(n, -1.0)
        conv = -scores
    else:
        sens = np.full(n, 1.0)
        conv = scores
    gain = np.where(sens > 0.0, ra, rv)
    n_top = max(int(round(q * n)), 20)
    top = np.argsort(conv)[-n_top:]
    return sens, conv, gain, top


print("\n1. LE LONG-ONLY EST INCHANGE AU BIT PRES")
_, _, g_l, t_l = selection("long")
verifie("le tri est celui du score brut",
        np.array_equal(np.sort(t_l), np.sort(np.argsort(scores)[-1000:])))
verifie("le gain est la colonne achat",
        np.array_equal(g_l, ra))

print("\n2. LE BILATERAL NE REGARDE PLUS EN AVANT")
s_b, c_b, g_b, t_b = selection("both")
g_triche = np.maximum(ra, rv)
verifie("le gain n'est PAS le maximum des deux cotes",
        not np.allclose(g_b, g_triche),
        "sommet honnete %+.4f  contre %+.4f en trichant"
        % (g_b[t_b].mean(), g_triche[t_b].mean()))
verifie("le sens suit le signe du score",
        np.array_equal(s_b > 0, scores >= 0))
verifie("les ventes sont retenues, pas seulement les achats",
        0.2 < np.mean(s_b[t_b] < 0) < 0.8,
        "%.0f %% de ventes dans le sommet" % (100 * np.mean(s_b[t_b] < 0)))

print("\n3. CE QUE LA TRICHE VALAIT, sur un modele qui ne sait RIEN")
s0 = rng.normal(0, 1.0, n)
sens0 = np.where(s0 >= 0, 1.0, -1.0)
top0 = np.argsort(np.abs(s0))[-1000:]
honnete = np.where(sens0 > 0, ra, rv)[top0].mean()
triche = np.maximum(ra, rv)[top0].mean()
verifie("un modele aveugle n'a plus d'avantage",
        abs(honnete - np.mean(np.where(sens0 > 0, ra, rv))) < 0.1,
        "honnete %+.3f   en trichant %+.3f   ecart offert %+.3f"
        % (honnete, triche, triche - honnete))

print("\n4. `rho` NE PUNIT PLUS UNE VENTE REUSSIE")
from scipy.stats import spearmanr
budget = np.zeros(n)
budget[t_b] = c_b[t_b] / c_b[t_b].max()
rho_avant = spearmanr(budget, vrai).correlation
rho_apres = spearmanr(budget, s_b * vrai).correlation
verifie("rho monte une fois le sens pris en compte",
        rho_apres > rho_avant,
        "avant %+.4f -> apres %+.4f" % (rho_avant, rho_apres))

print("\n5. L'HORIZON DE DISJONCTION")
idx = np.sort(rng.choice(85000, 1000, replace=False))


def paquet(h):
    nn, fin = 0, -1
    for i in idx:
        if i >= fin:
            nn += 1
            fin = i + h
    return nn


a202, a30 = paquet(202), paquet(30)
verifie("trente barres comptent plus d'occasions disjointes que 202",
        a30 > a202,
        "h=202 -> %d   h=30 -> %d   marge divisee par %.2f"
        % (a202, a30, np.sqrt(a30 / a202)))

print("\n%d/%d OK" % (_ok, _ok + _ko))
raise SystemExit(1 if _ko else 0)
