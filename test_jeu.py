# -*- coding: utf-8 -*-
"""Le jeu KAIROS : resolution des coups, regles des parties, expert, tetes.

    python test_jeu.py
"""
import sys
from dataclasses import replace

import numpy as np
import torch

import jeu_kairos as J
from saint_core import ACHETER, ATTENDRE, N_BASE_FEATURES, VENDRE

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

N_OK = N_KO = 0


def verifie(nom, cond, detail=""):
    global N_OK, N_KO
    if cond:
        N_OK += 1
        print(f"  ok   {nom:<58} {detail}")
    else:
        N_KO += 1
        print(f"  ECHEC {nom:<57} {detail}")


# Un marche plat a 100, ATR 1, sans cout : chaque coup se lit a la main.
cfg0 = replace(J.JeuConfig(), horizon_max=10, tp_atr=(2.0, 4.0),
               sl_atr=(2.0, 4.0), glissement_entree_bps=0.0,
               glissement_sortie_bps=0.0)
N = 60


def marche():
    o = np.full(N, 100.0)
    return o.copy(), o.copy(), o.copy(), np.zeros(N), np.ones(N)


print("\n1. LA RESOLUTION DES COUPS")
o, h, l, sp, atr = marche()
t0 = 10
h[t0 + 1 + 3] = 103.0            # une pointe a +3, trois barres apres l'entree
R, D, S = J.table_coups(o, h, l, sp, atr, cfg0, 1.0)
verifie("achat : objectif 2 ATR touche -> +1 R", abs(R[t0, 0, 0, 0] - 1.0) < 1e-6,
        f"R {R[t0, 0, 0, 0]:+.3f}")
verifie("achat : duree = barre de sortie + 1", D[t0, 0, 0, 0] == 4 and S[t0, 0, 0, 0] == 0)
verifie("vente : la meme pointe touche son stop -> -1 R",
        abs(R[t0, 1, 0, 0] + 1.0) < 1e-6 and S[t0, 1, 0, 0] == 1,
        f"R {R[t0, 1, 0, 0]:+.3f}")
verifie("achat objectif 4 ATR non touche : sortie au temps, 0 R",
        abs(R[t0, 0, 1, 0]) < 1e-6 and S[t0, 0, 1, 0] == 2
        and D[t0, 0, 1, 0] == cfg0.horizon_max + 1)
verifie("un stop plus large divise le resultat en R",
        abs(R[t0, 0, 0, 1] - 0.5) < 1e-6, f"R {R[t0, 0, 0, 1]:+.3f}")

o, h, l, sp, atr = marche()
h[t0 + 2], l[t0 + 2] = 103.0, 97.0
R, D, S = J.table_coups(o, h, l, sp, atr, cfg0, 1.0)
verifie("objectif et stop dans la meme barre : le stop d'abord",
        abs(R[t0, 0, 0, 0] + 1.0) < 1e-6 and S[t0, 0, 0, 0] == 1)

o, h, l, sp, atr = marche()
o[t0 + 3:], h[t0 + 3:], l[t0 + 3:] = 96.0, 96.0, 96.0
R, D, S = J.table_coups(o, h, l, sp, atr, cfg0, 1.0)
verifie("un stop saute par un trou : sortie a l'ouverture, -2 R",
        abs(R[t0, 0, 0, 0] + 2.0) < 1e-6, f"R {R[t0, 0, 0, 0]:+.3f}")

o, h, l, sp, atr = marche()
sp[:] = 2.0
cfgc = replace(cfg0, glissement_entree_bps=0.5, glissement_sortie_bps=1.0)
R1, _, _ = J.table_coups(o, h, l, sp, atr, cfgc, 1.0)
R0, _, _ = J.table_coups(o, h, l, sp, atr, cfgc, 0.0)
_attendu = -100.0 * (2.0 + 0.5 + 1.0) / 1e4 / 2.0
verifie("achat au temps sur marche plat : paie spread + glissements",
        abs(R1[t0, 0, 1, 0] - _attendu) < 1e-4, f"{R1[t0, 0, 1, 0]:+.5f} / {_attendu:+.5f}")
verifie("vente : le meme cout, lu sur l'ask", R1[t0, 1, 1, 0] < -0.0015)
verifie("sans cout (niveau 1 de la rampe) : 0 R exactement",
        abs(R0[t0, 0, 1, 0]) < 1e-9 and abs(R0[t0, 1, 1, 0]) < 1e-9)
verifie("aucun coup ne lit au-dela des donnees",
        np.isnan(R1[N - 1 - cfg0.horizon_max:]).all()
        and np.isfinite(R1[:N - 1 - cfg0.horizon_max]).all())

_cp = J.JeuConfig()
_ae = J.atr_effectif(np.array([0.0002, 0.02]), np.array([100.0, 100.0]), _cp)
verifie("plancher : un ATR calme est releve a atr_min_bps",
        abs(_ae[0] - _cp.atr_min_bps * 1e-4 * 100.0) < 1e-12, str(_ae))
verifie("plancher : un ATR agite reste tel quel", abs(_ae[1] - 0.02) < 1e-12)
verifie("plancher : le cout ne depasse jamais ~0.2 R au plus petit stop",
        3.5 / (min(_cp.sl_atr) * _cp.atr_min_bps) <= 0.25)

print("\n2. LES REGLES DES PARTIES")
torch.manual_seed(0)
cfgj = replace(J.JeuConfig(), horizon_max=10, jetons=3, lookback=4)
pol = J.PolitiqueJeu(cfgj)
g = pol.groupes_jeu()
verifie("six groupes, dont les quatre tetes", sorted(g) ==
        ["achat", "gain", "perte", "tronc", "valeur", "vente"])
ids = [id(q) for v in g.values() for q in v]
verifie("les groupes sont disjoints", len(ids) == len(set(ids)))
opt = J.optimiseurs(pol, cfgj)
verifie("un optimiseur par groupe", sorted(opt) == sorted(g))
rng = np.random.default_rng(0)
M = 3000
o = 100.0 + np.cumsum(rng.normal(0, 0.05, M))
h, l = o + 0.08, o - 0.08
Xn = rng.normal(0, 1, (M, N_BASE_FEATURES)).astype(np.float32)
R, D, S = J.table_coups(o, h, l, np.zeros(M), np.full(M, 0.05), cfgj, 1.0)
jours = np.array([[100, 1000], [1000, 1900], [1900, 2800]])
with torch.no_grad():
    pol.tete_achat.bias.fill_(20.0)           # la politique achete toujours
sc, cp, tr = J.joue(pol, jours, Xn, R, D, S, 2800, cfgj, "cpu", explore=False,
                    collecte=True)
par = np.bincount([c[0] for c in cp], minlength=3)
verifie("une politique qui achete toujours epuise ses jetons, pas plus",
        (par == cfgj.jetons).all(), str(par))
ok_suite = True
for gg in range(3):
    cg = [c for c in cp if c[0] == gg]
    for a_, b_ in zip(cg, cg[1:]):
        ok_suite &= b_[1] == a_[1] + a_[6]
verifie("un coup ne commence qu'a la fin du precedent", ok_suite)
verifie("le score d'une partie est la somme de ses coups",
        all(abs(sc[gg] - sum(c[5] for c in cp if c[0] == gg)) < 1e-9 for gg in range(3)))
verifie("chaque partie finit sur une transition terminale",
        all(t_[-1][12] for t_ in tr if t_))
sc2, cp2, _ = J.joue(pol, np.array([[2790, 2800]]), Xn, R, D, S, 2800, cfgj,
                     "cpu", explore=False)
verifie("pas de coup qui se resoudrait au-dela de la fenetre", len(cp2) == 0)
cfgv = replace(cfgj, vie_R=0.5, jetons=50)
sc3, cp3, _ = J.joue(pol, jours, Xn, R, D, S, 2800, cfgv, "cpu", explore=False)
fin_vie = True
for gg in range(3):
    part = np.cumsum([c[5] for c in cp3 if c[0] == gg])
    fin_vie &= bool((part[:-1] > -0.5).all()) if len(part) > 1 else True
verifie("la partie s'arrete quand la vie est perdue", fin_vie,
        " ".join(f"{x:+.2f}" for x in sc3))

print("\n3. L'OBSERVATION NE LIT QUE LE PASSE")
et = J.etat_jeu(np.array([3]), np.array([0.0]), np.array([500]), cfgj)
a = J.observations(Xn, np.array([500]), et, 4)
Xm = Xn.copy()
Xm[501:] = 99.0
b = J.observations(Xm, np.array([500]), et, 4)
verifie("changer l'avenir ne change pas l'observation", np.array_equal(a, b))
verifie("la derniere barre vue est t", np.array_equal(a[0, -1, :N_BASE_FEATURES], Xn[500]))

print("\n4. L'EXPERT")
cfge = replace(cfgj, expert_k=4, expert_R_min=0.5)
ex = J.coups_expert(jours, R, D, 2800, cfge)
ok_k = all(sum(1 for e in ex if jours[q, 0] <= e[0] < jours[q, 1]) <= 4
           for q in range(3))
verifie("au plus expert_k coups par journee", ok_k, f"{len(ex)} coups")
verifie("chaque coup rapporte au moins expert_R_min", all(e[4] >= 0.5 for e in ex))
chev = False
for q in range(3):
    iv = sorted((e[0], e[0] + int(D[e[0], e[1], e[2], e[3]])) for e in ex
                if jours[q, 0] <= e[0] < jours[q, 1])
    chev |= any(b_[0] <= a_[1] for a_, b_ in zip(iv, iv[1:]))
verifie("les coups de l'expert ne se chevauchent pas", not chev)
verifie("l'expert ne lit pas au-dela de la fenetre",
        all(e[0] + 1 + cfge.horizon_max < 2800 for e in ex))

print("\n5. UNE MISE A JOUR PPO ET UNE PASSE D'IMITATION")
pol2 = J.PolitiqueJeu(cfgj)
opt2 = J.optimiseurs(pol2, cfgj)
avant = {k: [q.detach().clone() for q in v] for k, v in pol2.groupes_jeu().items()}
_, _, tr2 = J.joue(pol2, jours, Xn, R, D, S, 2800, cfgj, "cpu", explore=True,
                   collecte=True)
lg = J.avantages(tr2, cfgj)
st = J.maj_ppo(pol2, opt2, lg, Xn, replace(cfgj, ppo_epochs=1, minibatch=256),
               "cpu", rng)
verifie("PPO tourne et rend ses mesures", np.isfinite(st["kl"]) and st["n"] > 0,
        f"{st['n']} decisions, {st['n_coups']} coups")
bouge = {k: any(not torch.equal(a_, b_) for a_, b_ in zip(avant[k], v))
         for k, v in pol2.groupes_jeu().items()}
verifie("les six groupes ont avance", all(bouge.values()), str(bouge))
cfgi = replace(cfge, expert_epochs=1, expert_pas_neg=20)
bar_av = [q.detach().clone() for k in ("gain", "perte")
          for q in pol2.groupes_jeu()[k]]
ent_av = [q.detach().clone() for k in ("achat", "vente")
          for q in pol2.groupes_jeu()[k]]
J.imite_expert(pol2, opt2, jours, R, D, Xn, 2800, cfgi, "cpu", rng)
bar_ap = [q for k in ("gain", "perte") for q in pol2.groupes_jeu()[k]]
ent_ap = [q for k in ("achat", "vente") for q in pol2.groupes_jeu()[k]]
verifie("l'imitation n'enseigne pas les barrieres",
        all(torch.equal(a_, b_) for a_, b_ in zip(bar_av, bar_ap)))
verifie("elle enseigne l'entree",
        any(not torch.equal(a_, b_) for a_, b_ in zip(ent_av, ent_ap)))
dt_ = J.departs_tires(jours, np.random.default_rng(1))
verifie("les departs tires restent dans la journee, avant la marge",
        bool(((dt_[:, 0] >= jours[:, 0]) & (dt_[:, 0] <= jours[:, 1] - 240)).all()
             and (dt_[:, 1] == jours[:, 1]).all()), str(dt_[:, 0]))

print(f"\n{N_OK}/{N_OK + N_KO} OK")
raise SystemExit(1 if N_KO else 0)
