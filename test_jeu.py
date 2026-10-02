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
               glissement_sortie_bps=0.0, minutes_par_barre=1, partie="jour",
               swap_achat_bps_jour=0.0, swap_vente_bps_jour=0.0, positions_max=1)
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
_ae = J.atr_effectif(np.array([0.0002, 0.2]), np.array([100.0, 100.0]), _cp)
verifie("plancher : un ATR calme est releve a atr_min_bps",
        abs(_ae[0] - _cp.atr_min_bps * 1e-4 * 100.0) < 1e-12, str(_ae))
verifie("plancher : un ATR agite (20 bps) reste tel quel", abs(_ae[1] - 0.2) < 1e-12)
verifie("plancher : le cout ne depasse jamais ~0.2 R au plus petit stop",
        3.5 / (min(_cp.sl_atr) * _cp.atr_min_bps) <= 0.25)

print("\n2. LES REGLES DES PARTIES")
torch.manual_seed(0)
# LES TESTS DES REGLES JOUENT EN M1 : leurs donnees synthetiques ont les
# 71 colonnes du M1. Le M15 a les siennes (voir la derniere section).
# Mise de 1 % (10 $ par R) : les comptes des tests de bilan s'y referent,
# independamment de la mise du jeu (0.5 % depuis m15_05).
cfgj = replace(J.JeuConfig(), horizon_max=10, jetons=3, lookback=4, n_expert=0,
               minutes_par_barre=1, risque_pct=1.0, marches=(), partie="jour",
               swap_achat_bps_jour=0.0, swap_vente_bps_jour=0.0, positions_max=1,
               jeu_version=1)
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

# PLUSIEURS POSITIONS A LA FOIS (run h1_02).
cfgm = replace(cfgj, positions_max=3, jetons=12)
scm, cpm, trm = J.joue(pol, jours, Xn, R, D, S, 2800, cfgm, "cpu", explore=False,
                       collecte=True)
ok_places, ok_bougie = True, True
for gg in range(3):
    cg = sorted((c[1], c[1] + c[6]) for c in cpm if c[0] == gg)
    for k_, (a_, _) in enumerate(cg):
        ouverts = sum(1 for (x, y) in cg[:k_] if y > a_)
        ok_places &= ouverts < 3
    ok_bougie &= len({a_ for a_, _ in cg}) == len(cg)
verifie("positions multiples : jamais plus de 3 coups ouverts a la fois", ok_places)
verifie("positions multiples : au plus un coup ouvert par bougie", ok_bougie)
verifie("positions multiples : des coups se chevauchent vraiment",
        any(b_[1] < a_[1] + a_[6] for gg in range(3)
            for a_, b_ in zip(sorted([c for c in cpm if c[0] == gg], key=lambda z: z[1]),
                              sorted([c for c in cpm if c[0] == gg], key=lambda z: z[1])[1:])))
verifie("positions multiples : chaque decision dure une bougie pour le PPO",
        all(x[11] == 1 for tr_ in trm for x in tr_))
verifie("positions multiples : le score final compte tous les coups",
        all(abs(scm[gg] - sum(c[5] for c in cpm if c[0] == gg)) < 1e-9 for gg in range(3)))
# Le score VU ne compte que les coups resolus : a la decision qui suit
# l'ouverture du premier coup, l'etat montre encore un score nul.
tr0 = trm[0]
t_premier = [c for c in cpm if c[0] == 0][0]
res_min = min(c[1] + c[6] for c in cpm if c[0] == 0)
suiv = [x for x in tr0 if t_premier[1] < x[0] < res_min]
verifie("positions multiples : un coup ouvert ne revele pas son issue",
        len(suiv) > 0 and all(abs(x[1][1]) < 1e-9 for x in suiv))
verifie("positions multiples : l'etat dit combien de places sont prises",
        len(suiv) > 0 and suiv[0][1][4] > 0)
bm = J.bilan(np.array([0.0]), [(0, 0, 0, 0, 0, -1.0, 50, 1), (0, 10, 0, 0, 0, -1.0, 5, 1),
                              (0, 20, 0, 0, 0, +3.0, 5, 0)],
             np.full(80, 100.0), np.full(80, 1.0), np.zeros(80), cfgm)
verifie("positions multiples : le drawdown suit l'ordre des resolutions",
        abs(bm["dd_dollars"] + 10.0) < 1e-9, "%.2f" % bm["dd_dollars"])

# L'expert enseigne des coups qui se chevauchent quand le jeu le permet.
_pred = np.full((M, 2), -1.0)
_pred[500:520, 0] = np.linspace(2.0, 1.0, 20)        # vingt bougies d'achat de suite
_j1 = np.array([[400, 900]])
_e1 = J.coups_expert_predits(_j1, _pred, D, 2800, replace(cfgj, expert_k=10, positions_max=1))
_e10 = J.coups_expert_predits(_j1, _pred, D, 2800, replace(cfgj, expert_k=10, positions_max=10))
_ouv = lambda e: max(sum(1 for (a, *_r) in e if a <= t_ < a + int(D[a, 0, J._ref(cfgj)[0], J._ref(cfgj)[1]]) + 1)
                     for t_ in range(400, 900))
verifie("l'expert a une position : jamais deux coups ouverts ensemble", _ouv(_e1) == 1, str(len(_e1)))
verifie("l'expert a dix positions : il enseigne ses dix coups, qui se chevauchent",
        len(_e10) == 10 and _ouv(_e10) > 1 and _ouv(_e10) <= 10, f"{len(_e10)} coups, {_ouv(_e10)} ouverts")

# COMME EN LIVE (run h1_04) : la marge limite les positions ouvertes.
cfgmg = replace(cfgj, positions_max=10, jetons=12)
_mg = np.full((M, len(cfgmg.sl_atr)), 0.4)          # 40 % de l'equite par coup
_, cpg, _ = J.joue(pol, jours, Xn, R, D, S, 2800, cfgmg, "cpu", explore=False, marge=_mg)
_ouv_g = max(sum(1 for c in cpg if c[0] == gg and c[1] <= t_ < c[1] + c[6])
             for gg in range(3) for t_ in range(100, 2800, 7))
verifie("comme en live : jamais plus de coups que la marge n'en couvre (2 a 40 %)",
        _ouv_g <= 2 and len(cpg) > 0, f"{_ouv_g} ouverts")
_, cpg0, _ = J.joue(pol, jours, Xn, R, D, S, 2800, cfgmg, "cpu", explore=False)
verifie("sans marge fournie, la contrainte ne joue pas", len(cpg0) >= len(cpg))
cfl = replace(cfgj, capital=1000.0, risque_pct=1.0, sl_atr=(1.0, 2.0, 4.0, 8.0),
              lot_min=0.01, pas_lot=0.01, contrat=1.0, levier=500.0)
_cl = J.compte_live(np.array([0, 10]), np.array([0, 3]), np.array([2.0, -1.0]),
                    np.array([5, 5]), np.full(20, 50_000.0), np.full(20, 500.0), cfl)
verifie("comme en live : 0.02 lot au stop serre, 0.01 minimum au stop large",
        abs(_cl["live_total"] - (20.0 - 40.0)) < 1e-6, "%.2f" % _cl["live_total"])
verifie("comme en live : le lot minimum fait risquer plus que 1 %",
        abs(_cl["live_risque_max"] - 40.0 / 1020.0) < 1e-9, "%.4f" % _cl["live_risque_max"])
_cl2 = J.compte_live(np.array([0, 10]), np.array([0, 0]), np.array([10.0, 2.0]),
                     np.array([5, 5]), np.full(20, 50_000.0), np.full(20, 100.0), cfl)
verifie("comme en live : la mise suit l'equite (interets composes)",
        abs(_cl2["live_total"] - 122.0) < 1e-6, "%.2f" % _cl2["live_total"])
cfl3 = replace(cfl, levier=0.5)
_cl3 = J.compte_live(np.array([0]), np.array([0]), np.array([1.0]),
                     np.array([5]), np.full(20, 50_000.0), np.full(20, 500.0), cfl3)
verifie("comme en live : un coup dont la marge depasse l'equite est refuse",
        _cl3["live_marge"] == 1 and _cl3["live_pris"] == 0)

cfgp = replace(cfgj, porte_rang_expert=0.9)
rg = np.zeros((M, 2), np.float32)
rg[[150, 400, 1100], 0] = 0.95          # trois minutes ou l'achat est permis
_, cpp, trp = J.joue(pol, jours, Xn, R, D, S, 2800, cfgp, "cpu", explore=False,
                     collecte=True, rangs=rg)
verifie("la porte : on n'achete qu'aux minutes ou le rang le permet",
        sorted(c_[1] for c_ in cpp) == [150, 400, 1100], str([c_[1] for c_ in cpp]))
verifie("les portes sont gardees pour la mise a jour (bit 0 achat, bit 1 vente)",
        all(x[2] in (0, 1) for tr_ in trp for x in tr_)
        and any(x[2] == 1 for tr_ in trp for x in tr_))
_, cp0, _ = J.joue(pol, jours, Xn, R, D, S, 2800, cfgp, "cpu", explore=False, rangs=None)
verifie("sans rangs, pas de porte", len(cp0) == 9)
lg_ = J._masque_logits(torch.zeros(2, 3), torch.tensor([True, False]),
                       torch.tensor([False, True]))
verifie("le masque traite l'achat et la vente separement",
        lg_[0, 0] == 0 and lg_[0, 1] < -1e8 and lg_[1, 0] < -1e8 and lg_[1, 1] == 0)

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

print("\n4b. L'EXPERT REALISTE")
yx = J.cibles_expert(R, cfge)
verifie("la cible expert moyenne les 16 barrieres, comme demande",
        abs(yx[500, 0] - np.clip(np.nanmean(R[500, 0]), -3, 5)) < 1e-5)
pred_s = np.full((M, 2), np.nan, np.float32)
pred_s[:, 0] = -1.0
pred_s[:, 1] = -1.0
pred_s[300, 0] = 0.5        # une seule minute d'achat predite positive
pred_s[1200, 1] = 0.7       # une de vente
cx = J.coups_expert_predits(jours, pred_s, D, 2800, cfgj)
verifie("l'expert n'enseigne que les minutes predites positives",
        sorted((c_[0], c_[1]) for c_ in cx) == [(300, 0), (1200, 1)], str(cx))
sx, kx = J.joue_expert(jours, pred_s, R, D, S, 2800, 0.1, cfgj)
verifie("l'expert joue seul, au seuil, sans voir la suite du jour",
        sorted((c_[1], c_[2]) for c_ in kx) == [(300, 0), (1200, 1)])
verifie("son score est la somme de ses coups",
        abs(sx.sum() - sum(c_[5] for c_ in kx)) < 1e-9)
Xg = np.random.default_rng(3).normal(0, 1, (M, N_BASE_FEATURES)).astype(np.float32)
cfgx = replace(cfgj, expert_arbres=20, expert_blocs=2, expert_pas_app=3)
px, _mx = J.expert_realiste(Xg, yx, 2000, 2400, cfgx)
verifie("l'expert predit le train et la validation, rien au-dela",
        np.isfinite(px[:2400]).all() and np.isnan(px[2400:]).all())

fe = J.features_expert(px, fenetre=500)
verifie("les colonnes de l'expert n'ont aucun trou", np.isfinite(fe).all())
px2 = px.copy()
px2[1500:] = 99.0
fe2 = J.features_expert(px2, fenetre=500)
verifie("le rang de l'expert ne lit que le passe",
        np.array_equal(fe[:1500], fe2[:1500]))
pj = J.PolitiqueJeu(replace(cfgj, n_expert=4))
xj = torch.zeros(2, 4, N_BASE_FEATURES + 4 + J.N_ETAT)
verifie("le modele lit les 71 colonnes, les 4 de l'expert et l'etat",
        pj.jeu(xj)[0].shape == (2, 3))

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
J.imite_expert(pol2, opt2, J.coups_expert(jours, R, D, 2800, cfgi), jours,
               Xn, 2800, cfgi, "cpu", rng)
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

print("\n5b. LE BILAN DETAILLE")
# Quatre coups : +1, -1, +2, -1 R, dans cet ordre ; deux longs, deux shorts.
cz = [(0, 10, 0, 0, 0, 1.0, 1, 0), (0, 20, 1, 0, 0, -1.0, 1, 1),
      (1, 30, 0, 0, 0, 2.0, 1, 0), (1, 40, 1, 0, 0, -1.0, 1, 1)]
bz = J.bilan(np.array([0.0, 1.0]), cz, np.full(60, 100.0), np.full(60, 1.0),
             np.zeros(60), cfgj)
verifie("gagnants, perdants, longs, shorts",
        (bz["gagnants"], bz["perdants"], bz["longs"], bz["shorts"]) == (2, 2, 2, 2))
verifie("win rate et profit factor", abs(bz["win_rate"] - 0.5) < 1e-12
        and abs(bz["pf"] - 1.5) < 1e-12)
verifie("drawdown : la pire baisse du compte, dans l'ordre du temps",
        abs(bz["dd_dollars"] + 10.0) < 1e-9, "%.2f $" % bz["dd_dollars"])
verifie("total en dollars", abs(bz["total_dollars"] - 10.0) < 1e-9)
verifie("la ligne lisible dit tout",
        all(k in J.ligne_detail(bz, cfgj) for k in
            ("gagnants", "perdants", "win rate", "longs", "shorts",
             "profit factor", "drawdown")))

print("\n5c. LE DRAWDOWN")
cd = replace(cfgj, seuil_baisse=0.05, mise_en_baisse=0.5)
# 1000 $, 10 $ par R : apres -5 R le compte est a -5 %, au seuil ; les coups
# suivants ne risquent plus que 5 $, jusqu'au retour au plus haut.
ep = J.compte_prudent(np.array([-1.0] * 6 + [-1.0, +2.0, +20.0, -1.0]), cd)
verifie("les cinq premieres pertes a mise entiere", abs(ep[5] - 950.0) < 1e-9, str(ep[:8]))
verifie("a 5 % sous le plus haut, la mise est reduite de moitie",
        abs(ep[6] - 945.0) < 1e-9 and abs(ep[7] - 940.0) < 1e-9 and abs(ep[8] - 950.0) < 1e-9)
verifie("elle revient entiere au nouveau plus haut",
        abs(ep[9] - 1050.0) < 1e-9 and abs(ep[10] - 1040.0) < 1e-9, str(ep[8:]))
bz2 = J.bilan(np.array([0.0, 1.0]), cz, np.full(60, 100.0), np.full(60, 1.0),
              np.zeros(60), cfgj)
verifie("le bilan porte le compte prudent et son drawdown",
        "prudent_total" in bz2 and bz2["prudent_dd_dollars"] <= 0
        and "gain / drawdown" in J.ligne_prudente(bz2, cfgj))
trp = [[(5, None, 3, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, -1.0, 1, True)]]
a1 = J.avantages(trp, replace(cfgj, poids_pertes=1.0))[0][13]
a2 = J.avantages(trp, replace(cfgj, poids_pertes=1.5))[0][13]
verifie("les pertes pesent 1.5 fois dans la recompense du PPO",
        abs(a1 + 1.0) < 1e-12 and abs(a2 + 1.5) < 1e-12, f"{a1:+.2f} / {a2:+.2f}")
trg = [[(5, None, 3, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, +1.0, 1, True)]]
verifie("les gains ne changent pas", abs(J.avantages(trg, cfgj)[0][13] - 1.0) < 1e-12)

print("\n6. LE JEU EN BOUGIES DE 15 MINUTES")
c15 = replace(J.JeuConfig(), marches=(), minutes_par_barre=15, horizon_max=32,
              partie="jour", cache="data_cache_BTCUSD_M15.pkl")
verifie("le jeu d'un marche en M15 : 96 bougies par jour",
        c15.minutes_par_barre == 15 and c15.barres_par_jour == 96)
import prepare_btc_m15 as P15
verifie("il lit les features du M15", J.colonnes_jeu(c15) == list(P15.FEATURE_COLS_M15))
verifie("ses OFI portent leurs horizons reels (15, 60, 240 min)",
        all(f"ofi_{k}" in P15.FEATURE_COLS_M15 for k in (15, 60, 240))
        and "ofi_5" not in P15.FEATURE_COLS_M15)
verifie("ses contextes sont le H1 et le H4",
        "close_h1_dev" in P15.FEATURE_COLS_M15 and "close_h4_dev" in P15.FEATURE_COLS_M15
        and not any(c.endswith("_m5") for c in P15.FEATURE_COLS_M15))
p15 = J.PolitiqueJeu(c15)
x15 = torch.zeros(2, c15.lookback, len(P15.FEATURE_COLS_M15) + c15.n_expert + J.N_ETAT)
verifie("le modele M15 lit ses colonnes, celles de l'expert et l'etat",
        p15.jeu(x15)[0].shape == (2, 3))
et15 = J.etat_jeu(np.array([3]), np.array([0.0]), np.array([48]), c15)
verifie("la moitie de la journee restante vaut 0.5 en M15", abs(et15[0, 2] - 0.5) < 1e-6)
verifie("l'horizon est de 32 bougies, huit heures", c15.horizon_max == 32)

print("\n7. LE JEU MULTI-MARCHES")
import prepare_multi_m15 as PM
cm = replace(J.JeuConfig(), marches=tuple(PM.MARCHES), cache="data_cache_MULTI_M15.pkl",
             minutes_par_barre=15, horizon_max=32, partie="jour")
verifie("le multi-marches M15 reste disponible a cote du multi H1 par defaut",
        tuple(cm.marches) == tuple(PM.MARCHES))
verifie("il lit les features communes, avec une colonne par marche",
        J.colonnes_jeu(cm) == list(PM.FEATURE_COLS_MULTI)
        and all(f"m_{m}" in PM.FEATURE_COLS_MULTI for m in PM.MARCHES)
        and not any(c.startswith(("ofi_", "prime_cb", "taker")) for c in PM.FEATURE_COLS_MULTI))
pm_ = J.PolitiqueJeu(cm)
xm_ = torch.zeros(2, cm.lookback, len(PM.FEATURE_COLS_MULTI) + cm.n_expert + J.N_ETAT)
verifie("le modele multi lit ses colonnes, l'expert et l'etat", pm_.jeu(xm_)[0].shape == (2, 3))
mk = np.array(["A"] * 5 + ["B"] * 4)
bl = J.blocs_marches(mk)
verifie("les blocs de marches", bl == [("A", 0, 5), ("B", 5, 9)], str(bl))
tn = np.array([0, 10, 20, 30, 40, 0, 10, 20, 30], np.int64)
fs = J.fin_segment(tn, bl, 25)
verifie("la fin du segment reste dans le bloc du marche",
        list(fs[:5]) == [3] * 5 and list(fs[5:]) == [8] * 4, str(fs))
verifie("une borne par ligne dans la regle des coups",
        J._lim(fs, np.array([0, 6])).tolist() == [3, 8] and J._lim(7, 3) == 7)
pj_ = J.par_jour(np.array([1.0, -0.5, 2.0]), np.array([[0, 2], [5, 7], [2, 4]]),
                 np.array(["j1", "j1", "j2", "j2", "j2", "j1", "j1", "j1", "j1"]))
verifie("le score du jour est la somme des marches", sorted(pj_.tolist()) == [0.5, 2.0],
        str(pj_))
cz2 = [(0, 3, 0, 0, 0, 1.0, 1, 0), (1, 6, 1, 0, 0, -1.0, 1, 1)]
verifie("le bilan par marche", "A 1 trades" in J.ligne_marches(cz2, mk, cfgj)
        and "B 1 trades" in J.ligne_marches(cz2, mk, cfgj))
bo = J.bilan(np.array([0.0]), [(0, 6, 0, 0, 0, -1.0, 1, 1), (0, 1, 0, 0, 0, 2.0, 1, 0)],
             np.full(9, 100.0), np.full(9, 1.0), np.zeros(9), cfgj,
             ordre=np.array([0, 5, 0, 0, 0, 0, 1, 0, 0], np.int64))
import pandas as _pd
_tb = _pd.Series(_pd.to_datetime(["2020-01-01 00:00", "2025-06-02 10:15"]))
_eb = J.extras_btc(_tb)
verifie("les sources du BTC s'alignent sur ses bougies, NaN avant leur debut",
        _eb is not None and _eb.shape[0] == 2 and np.isnan(_eb[0]).all()
        and np.isfinite(_eb[1]).any(), "" if _eb is None else str(_eb.shape))
verifie("le drawdown suit l'ordre du TEMPS, pas celui des lignes",
        abs(bo["dd_dollars"] + 10.0) < 1e-9, "%.2f" % bo["dd_dollars"])

print("\n8. LE JEU EN H1, PARTIES D'UNE SEMAINE")
import prepare_btc_h1_binance as PH
ch = replace(J.JeuConfig(), marches=(), cache=PH.SORTIE, minutes_par_barre=60,
             partie="semaine", horizon_max=72, expert_k=6, expert_pas_neg=1)
verifie("le jeu H1 : une semaine par partie, 168 bougies",
        ch.minutes_par_barre == 60 and ch.partie == "semaine" and ch.barres_par_partie == 168)
verifie("il lit les colonnes du H1 Binance", J.colonnes_jeu(ch) == list(PH.FEATURE_COLS_H1))
_t = _pd.Series(_pd.date_range("2024-01-01 00:00", periods=24 * 21, freq="1h"))
_s = J.semaines(_t, 0, len(_t), 50)
verifie("trois semaines entieres, du lundi au lundi",
        _s.shape == (3, 2) and _s[0].tolist() == [0, 168] and _s[2].tolist() == [336, 504],
        str(_s.tolist()))
verifie("une semaine coupee par la borne n'est pas une partie",
        J.semaines(_t, 0, 400, 50).shape[0] == 2)
verifie("parties() : des semaines en H1, des journees en M15",
        J.parties(_t, 0, len(_t), ch).shape[0] == 3
        and J.parties(_t, 0, len(_t), replace(ch, partie="jour", minutes_par_barre=60)).shape[0] == 21)
eh = J.etat_jeu(np.array([3]), np.array([0.0]), np.array([84]), ch)
verifie("la moitie de la semaine restante vaut 0.5", abs(eh[0, 2] - 0.5) < 1e-6)
# Le swap : marche plat a 100, ATR 1, sortie au temps apres H+1 = 11 bougies H1.
chs = replace(ch, horizon_max=10, tp_atr=(50.0,), sl_atr=(50.0,), glissement_entree_bps=0.0,
              glissement_sortie_bps=0.0, atr_min_bps=0.0)
_n = 40
_o = np.full(_n, 100.0)
Rh, Dh, Sh = J.table_coups(_o, _o, _o, np.zeros(_n), np.ones(_n), chs, 1.0)
_att = -(20.0 / 365 * 100) / 1e4 * 100.0 * 11 * 60 / 1440.0 / 50.0
verifie("le swap acheteur se paie au prorata de la duree",
        abs(float(Rh[0, 0, 0, 0]) - _att) < 1e-6 and Sh[0, 0, 0, 0] == 2,
        "%.6f contre %.6f" % (float(Rh[0, 0, 0, 0]), _att))
verifie("aucun swap a la vente (0 chez Vantage)", abs(float(Rh[0, 1, 0, 0])) < 1e-9)
Rz, _, _ = J.table_coups(_o, _o, _o, np.zeros(_n), np.ones(_n), chs, 0.0)
verifie("sans cout (rampe a 0), pas de swap non plus", abs(float(Rz[0, 0, 0, 0])) < 1e-9)
verifie("36 colonnes, dont le flux et le financement",
        len(PH.FEATURE_COLS_H1) == 36 and "flux_4" in PH.FEATURE_COLS_H1
        and "funding_der" in PH.FEATURE_COLS_H1)

print("\n9. LE MULTI-MARCHES H1 (le BTC et les indices)")
import prepare_multi_h1 as PMH
cmh = replace(J.JeuConfig(), marches=tuple(PMH.MARCHES), cache=PMH.SORTIE,
              minutes_par_barre=60, partie="semaine", horizon_max=72, expert_k=6,
              expert_pas_neg=1, porte_rang_expert=0.90)
import prepare_btc_m5 as P5
_dft = replace(J.JeuConfig(), marches=(), cache=P5.SORTIE)
verifie("le BTC seul en M5 (run m5_02) : une journee par partie, 10 jetons, une position",
        _dft.positions_max == 1
        and _dft.jetons == 10 and _dft.expert_k == 10 and _dft.partie == "jour"
        and _dft.porte_rang_expert == 0.85
        and _dft.minutes_par_barre == 5 and _dft.horizon_max == 96
        and _dft.barres_par_partie == 288)
verifie("le M5 lit ses colonnes, dont l'ecart Coinbase / Binance",
        J.colonnes_jeu(_dft) == list(P5.FEATURE_COLS_M5) and "prime_cb" in P5.FEATURE_COLS_M5
        and "flux_3" in P5.FEATURE_COLS_M5)
_p5 = J.PolitiqueJeu(_dft)
_x5 = torch.zeros(2, _dft.lookback, len(P5.FEATURE_COLS_M5) + _dft.n_expert + J.N_ETAT)
verifie("le modele M5 lit ses colonnes, l'expert et l'etat", _p5.jeu(_x5)[0].shape == (2, 3))
verifie("le multi H1 : le BTC et les indices, une semaine par partie",
        tuple(cmh.marches) == tuple(PMH.MARCHES) and len(cmh.marches) == len(PMH.MARCHES) >= 13
        and cmh.minutes_par_barre == 60 and cmh.partie == "semaine")
verifie("la configuration du run h1_01 : une position, 10 jetons, l'expert a 6 coups",
        cmh.positions_max == 1 and cmh.jetons == 10 and cmh.expert_k == 6
        and cmh.porte_rang_expert == 0.90 and cmh.risque_pct == 1.0)
verifie("il lit les colonnes communes, avec une colonne par marche",
        J.colonnes_jeu(cmh) == list(PMH.FEATURE_COLS_MULTI_H1)
        and all(f"m_{m}" in PMH.FEATURE_COLS_MULTI_H1 for m in PMH.MARCHES)
        and not any(c in PMH.FEATURE_COLS_MULTI_H1 for c in PMH.EXTRAS_BTC_H1))
pmh = J.PolitiqueJeu(cmh)
xmh = torch.zeros(2, cmh.lookback, len(PMH.FEATURE_COLS_MULTI_H1) + cmh.n_expert + J.N_ETAT)
verifie("le modele multi H1 lit ses colonnes, l'expert et l'etat", pmh.jeu(xmh)[0].shape == (2, 3))
_nm = len(PMH.MARCHES)
verifie("un tronc commun, un jeu de tetes par marche",
        pmh.n_marches == _nm and len(pmh.tete_achat_m) == _nm and len(pmh.critic_m) == _nm)
gmh = pmh.groupes_jeu()
_ids = [id(q) for v in gmh.values() for q in v]
verifie("les groupes restent disjoints et portent les tetes par marche",
        len(_ids) == len(set(_ids)) and sorted(gmh) == ["achat", "gain", "perte", "tronc", "valeur", "vente"]
        and len(gmh["achat"]) == _nm * len(list(pmh.mlp_achat.parameters()) + list(pmh.tete_achat.parameters())))
_colm = J.colonnes_jeu(cmh)
_xa = torch.zeros(2, cmh.lookback, len(_colm) + cmh.n_expert + J.N_ETAT)
_xa[0, :, _colm.index("m_BTCUSD")] = 1.0
_xa[1, :, _colm.index("m_GER40")] = 1.0
verifie("le marche se lit dans ses colonnes", pmh.marche_de(_xa).tolist() == [0, 2])
pmh.eval()                                       # sans dropout : sorties exactes
with torch.no_grad():
    _av = pmh.jeu(_xa)[0].clone()
    pmh.tete_achat_m[2].bias.add_(5.0)            # on ne touche que la tete du GER40
    _ap = pmh.jeu(_xa)[0]
verifie("changer la tete d'un marche ne change que ce marche",
        torch.allclose(_av[0], _ap[0]) and abs(float(_ap[1, 0] - _av[1, 0]) - 5.0) < 1e-4)
_c1 = replace(cmh, tetes_par_marche=False)
verifie("sans tetes par marche : le modele commun d'avant", J.PolitiqueJeu(_c1).n_marches == 1)
verifie("le BTC seul n'a qu'un jeu de tetes", J.PolitiqueJeu(ch).n_marches == 1)
_u = PMH.serveur_vers_utc(_pd.Series(_pd.to_datetime(["2024-07-08 16:30", "2024-01-08 16:30"])))
verifie("l'heure du serveur MT5 passe en UTC (ouverture de New York 13h30 l'ete, 14h30 l'hiver)",
        [x.strftime("%H:%M") for x in _u] == ["13:30", "14:30"])
_sa, _sv = PMH.swap_bps_jour({"contrat": 1.0, "swap_long": -6.0, "swap_short": 1.0}, 20_000.0)
verifie("le swap d'un indice : un cout a l'achat, un credit a la vente",
        abs(_sa - 3.0) < 1e-9 and abs(_sv + 0.5) < 1e-9, f"{_sa} {_sv}")
_tm = _pd.Series(list(_pd.date_range("2024-01-01", periods=24 * 14, freq="1h"))
                 + list(_pd.date_range("2024-01-01", periods=24 * 14, freq="1h")))
_bl = [("A", 0, 336), ("B", 336, 672)]
_jm = J.journees_multi(_tm, _bl, _pd.Timestamp("2024-01-01").value,
                       _pd.Timestamp("2024-01-15").value, 60, cmh)
verifie("deux marches, deux semaines chacun : quatre parties, chacune dans son marche",
        _jm.shape == (4, 2) and _jm[:2].max() <= 336 and _jm[2:].min() >= 336, str(_jm.tolist()))

print("\n10. LA VALIDATION CROISEE PURGEE")
_pm, _te, _va = J.masque_blocs(1000, 10, 0, 30)
verifie("le bloc de test et la validation (le bloc le plus eloigne)",
        _te == (0, 100) and _va == (500, 600))
verifie("la zone tampon est retiree de l'entrainement, des deux cotes",
        not _pm[:130].any() and _pm[130:470].all() and not _pm[470:630].any() and _pm[630:].all())
_px = J.prochain_exclu(_pm)
verifie("un coup d'entrainement ne peut pas lire la zone exclue suivante",
        _px[200] == 470 and _px[700] == 1000 and _px[50] == 50)
_pm9, _te9, _va9 = J.masque_blocs(1000, 10, 9, 30)
verifie("le dernier bloc aussi : test a la fin, validation au milieu",
        _te9 == (900, 1000) and _va9 == (400, 500) and not _pm9[870:].any())

print("\n11. LE MULTI-MARCHES M5 (BTC, ETH, ZEC)")
import prepare_multi_m5 as PM5
verifie("par defaut : le BTC seul en M5 (run m5_03)",
        J.JeuConfig().marches == () and J.JeuConfig().cache == P5.SORTIE)
_mm = replace(J.JeuConfig(), marches=("BTCUSD", "ETHUSD"), cache=PM5.SORTIE)
verifie("par defaut : BTC et ETH en M5 (sans le ZEC), validation en blocs",
        tuple(_mm.marches) == ("BTCUSD", "ETHUSD") and set(_mm.marches) <= set(PM5.MARCHES)
        and _mm.cache == PM5.SORTIE and _mm.minutes_par_barre == 5 and _mm.partie == "jour"
        and _mm.validation == "blocs" and _mm.tetes_par_marche)
verifie("les memes regles que le BTC seul du run m5_02",
        _mm.jetons == 10 and _mm.positions_max == 1 and _mm.expert_k == 10
        and _mm.porte_rang_expert == 0.85 and _mm.horizon_max == 96 and _mm.risque_pct == 1.0)
verifie("EXACTEMENT les colonnes du BTC, plus une colonne par marche",
        J.colonnes_jeu(_mm) == list(P5.FEATURE_COLS_M5) + ["m_BTCUSD", "m_ETHUSD", "m_ZECUSD"])
_mm3 = replace(_mm, marches=("BTCUSD", "ETHUSD", "ZECUSD"))
_pm5 = J.PolitiqueJeu(_mm3)
_xm = torch.zeros(3, _mm.lookback, len(J.colonnes_jeu(_mm3)) + _mm.n_expert + J.N_ETAT)
_ic = J.colonnes_jeu(_mm).index("m_ETHUSD")
_xm[1, :, _ic] = 1.0
_ic = J.colonnes_jeu(_mm).index("m_ZECUSD")
_xm[2, :, _ic] = 1.0
_xm[0, :, J.colonnes_jeu(_mm).index("m_BTCUSD")] = 1.0
_pm5.eval()
verifie("trois petits cerveaux, chaque ligne lit le sien",
        _pm5.n_marches == 3 and _pm5.marche_de(_xm).tolist() == [0, 1, 2]
        and _pm5.jeu(_xm)[0].shape == (3, 3))
verifie("le swap des cryptos Vantage : -25 %/an -> 6.85 bps par jour, 0 -> rien",
        abs(PM5.swap_bps_jour(-25) - 6.849) < 1e-2 and PM5.swap_bps_jour(0) == 0.0)
# deux marches empiles : le premier a 1000 bougies de 5 min, le second
# commence plus tard (a la bougie 300 du premier)
_t_a = _pd.Timestamp("2020-01-01").value + np.arange(1000, dtype=np.int64) * 300 * 10**9
_t_b = _t_a[300:]
_tt = np.concatenate([_t_a, _t_b])
_pmd, _ted, _vad = J.masque_blocs_dates(_tt, _t_a, 10, 0, 30 * 300 * 10**9)
_pmi, _tei, _vai = J.masque_blocs(1000, 10, 0, 30)
verifie("les blocs en dates sont ceux du BTC seul, bougie pour bougie",
        (_pmd[:1000] == _pmi).all() and _ted == (int(_t_a[0]), int(_t_a[100]))
        and _vad == (int(_t_a[500]), int(_t_a[600])))
verifie("l'autre marche perd les memes DATES (test, validation et tampon)",
        (_pmd[1000:] == _pmi[300:]).all())
_dc = _pd.DataFrame({"marche": ["A"] * 5 + ["B"] * 3,
                    "time": _pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03",
                                             "2020-01-04", "2020-01-05",
                                             "2020-01-03", "2020-01-04", "2020-01-05"])})
_dcc = J.coupe_debut_commun(_dc)
verifie("le meme point de depart : chaque marche commence a la premiere bougie du plus recent",
        _mm.debut_commun and len(_dcc) == 6
        and (_dcc.groupby("marche")["time"].min() == _pd.Timestamp("2020-01-03")).all())
_pmd9, _ted9, _ = J.masque_blocs_dates(_tt, _t_a, 10, 9, 30 * 300 * 10**9)
verifie("le dernier bloc va jusqu'a la derniere bougie comprise",
        _ted9[1] == int(_t_a[-1]) + 1 and not _pmd9[870:1000].any() and not _pmd9[-130:].any())

print("\n12. TROIS MODELES INDEPENDANTS, UN COMPTE COMMUN")
verifie("par defaut : un modele par marche", _mm.modeles_par_marche and _mm.marche_seul == "")
_cz = J.config_marche(_mm, "ZECUSD")
verifie("un modele, un bloc : config_marche(m, 3) ne joue que le bloc 3",
        J.config_marche(_mm, "BTCUSD", 3).bloc_seul == 3 and _cz.bloc_seul == 0)
verifie("le modele du ZEC : le jeu du BTC seul, ses colonnes, son nom, son dossier d'echange",
        _cz.marches == () and _cz.marche_seul == "ZECUSD" and not _cz.modeles_par_marche
        and _cz.prefixe == _mm.prefixe + "_ZECUSD" and _cz.echanges == f"echanges_{_mm.prefixe}"
        and J.colonnes_jeu(_cz) == list(P5.FEATURE_COLS_M5) and _cz.porte_rang_expert == 0.85)
import os as _os
if _os.path.exists(J.MT5_CRYPTO):
    verifie("le ZEC garde son contrat Vantage (100), le BTC le sien (1)",
            _cz.contrat == 100 and J.config_marche(_mm, "BTCUSD").contrat == 1)
_dm = _pd.DataFrame({"marche": ["A"] * 4 + ["B"] * 2,
                     "time": _pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04",
                                              "2020-01-03", "2020-01-04"]),
                     "swap_achat_bps_jour": [5.0] * 4 + [13.7] * 2,
                     "swap_vente_bps_jour": [0.0] * 4 + [13.7] * 2})
_dA, _cA = J.donnees_marche_seul(_dm, replace(_cz, marche_seul="A", marches_communs=("A", "B")))
_dA1, _ = J.donnees_marche_seul(_dm, replace(_cz, marche_seul="A", marches_communs=("A",)))
verifie("le debut commun ne regarde que les marches du compte (sans B : tout A)",
        len(_dA1) == 4 and J.config_marche(_mm, "BTCUSD").marches_communs == ("BTCUSD", "ETHUSD"))
verifie("un marche seul : ses bougies depuis le debut commun, son swap",
        len(_dA) == 2 and (_dA["marche"] == "A").all() and _cA.swap_achat_bps_jour == 5.0
        and _dA["time"].min() == _pd.Timestamp("2020-01-03"))
_tn = np.arange(100, dtype=np.int64) * 10
_tr = J.table_trades([(0, 5, 1, 0, 2, 1.5, 7, 0)], _tn, np.full(100, 200.0), np.full(100, 2.0), _cz)
verifie("un coup devient un trade : entree, sortie, R, sens, distance du stop, prix",
        _tr["t_entree"].iat[0] == 50 and _tr["t_sortie"].iat[0] == 120 and _tr["r"].iat[0] == 1.5
        and _tr["sens"].iat[0] == 1 and _tr["dist"].iat[0] == _cz.sl_atr[2] * 2.0
        and _tr["prix"].iat[0] == 200.0)
_ta = _pd.DataFrame({"t_entree": [0, 20], "t_sortie": [10, 30], "r": [1.0, -1.0], "sens": [0, 1],
                     "dist": [10.0, 10.0], "prix": [1000.0, 1000.0]})
_tb = _pd.DataFrame({"t_entree": [5], "t_sortie": [15], "r": [2.0], "sens": [0],
                     "dist": [10.0], "prix": [1000.0]})
_sp = {"A": {"contrat": 1.0, "lot_min": 0.01, "pas_lot": 0.01},
       "B": {"contrat": 1.0, "lot_min": 0.01, "pas_lot": 0.01}}
_bc = J.bilan_commun({"A": _ta, "B": _tb}, _sp, 3, _cz)
verifie("le compte commun : 3 trades, win rate 2/3, PF 3, +20 $, drawdown -10 $",
        _bc["coups"] == 3 and abs(_bc["win_rate"] - 2 / 3) < 1e-9 and abs(_bc["pf"] - 3.0) < 1e-9
        and abs(_bc["total_dollars"] - 20.0) < 1e-9 and abs(_bc["dd_dollars"] + 10.0) < 1e-9
        and _bc["longs"] == 2 and _bc["shorts"] == 1, str(_bc))
verifie("comme en live : la mise suit le SOLDE COMMUN (le 3e coup mise 10.30 $ apres +30 $)",
        abs(_bc["live_total"] - 19.70) < 1e-6 and _bc["live_pris"] == 3, f"{_bc['live_total']}")
verifie("le lot grossit avec le solde : 1.00 lot a 1000 $, 1.03 lot a 1030 $",
        abs(_bc["par_marche"]["A"]["lot_max"] - 1.03) < 1e-9
        and abs(_bc["live_solde"] - 1019.70) < 1e-6 and abs(_bc["live_win_rate"] - 2 / 3) < 1e-9,
        str(_bc["par_marche"]["A"]))
_bm = J.bilan_commun({"A": _ta, "B": _tb}, _sp, 3, replace(_cz, levier=1.0))
verifie("la marge est partagee : le coup du B n'entre pas pendant celui du A",
        _bm["live_pris"] == 2 and _bm["live_marge"] == 1, str(_bm["live_pris"]))
verifie("par marche : les trades et les dollars de chacun",
        _bc["par_marche"]["A"]["coups"] == 2 and abs(_bc["par_marche"]["B"]["total_dollars"] - 20) < 1e-9)

print(f"\n{N_OK}/{N_OK + N_KO} OK")
raise SystemExit(1 if N_KO else 0)
