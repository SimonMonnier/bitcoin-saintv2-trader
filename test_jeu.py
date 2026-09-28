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
               swap_achat_bps_jour=0.0, swap_vente_bps_jour=0.0, positions_max=1)
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
yx = J.cibles_expert(R)
verifie("la cible est le R moyen de tous les coups d'un sens",
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
verifie("le jeu par defaut est le BTC seul ; le multi-marches reste disponible",
        tuple(J.JeuConfig().marches) == () and tuple(cm.marches) == tuple(PM.MARCHES))
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
ch = J.JeuConfig()
verifie("le jeu par defaut : H1, une semaine par partie, 168 bougies",
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

print(f"\n{N_OK}/{N_OK + N_KO} OK")
raise SystemExit(1 if N_KO else 0)
