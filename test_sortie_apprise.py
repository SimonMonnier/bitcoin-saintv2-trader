# -*- coding: utf-8 -*-
"""LA SORTIE EST UNE DECISION APPRISE, plus un seuil ecrit a la main.

CE QUI A MENE ICI, ET C'EST QUATRE ECHECS DE LA MEME NATURE. Le
2026-09-22, la sortie a ete confiee a deux tetes supervisees :
`tete_cloture` predit le risque, `tete_profit` ce qu'il reste a prendre,
et une REGLE transformait chaque prediction en decision.

Les deux tetes ont bien appris — rho +0.355 et +0.272, au-dessus de la
mesure hors ligne. La regle, elle, a echoue quatre fois :

    seuil 1.00 ATR absolu       tenue[G 3/15 P 52/88 x0.2] — INVERSE
    + condition latent > 0      x0.5, toujours inverse
    seuil relatif 0.25 x latent `fermerait 0.0 %` sur 51 epochs sur 53
    redressement d'echelle      74 % des GAGNANTS soldes par la fin
                                d'episode, par aucune tete

Chaque correction etait juste et aucune n'a suffi, parce que le defaut
n'etait pas dans la calibration : il etait dans l'idee de seuiller une
amplitude predite.

ET RIEN NE S'OPPOSAIT A PPO. Le journal du depot est explicite : « toute
conclusion tiree d'un ecart de validation inferieur a 0.2 R par trade est
du bruit, y compris les "PPO degrade la validation" accumules depuis
exec24. Ces runs n'ont pas montre que PPO nuit ; ils n'ont rien montre. »

    python test_sortie_apprise.py
"""
import io
import sys

import numpy as np
import torch

import training as T
from saint_core import (
    FERMER,
    N_ACTIONS,
    N_ACTIONS_SORTIE,
    N_BASE_FEATURES,
    N_PROFIT_FEATURES,
    N_SORTIE_FEATURES,
    COL_SENS_SORTIE,
    IDX_SENS_POS,
    OBS_N_FEATURES,
    SAINTPolicySingleHead,
    TENIR,
)

_ok = _ko = 0


def verifie(nom, cond, detail=""):
    global _ok, _ko
    if cond:
        _ok += 1
        print("  ok   %-56s %s" % (nom, detail))
    else:
        _ko += 1
        print("  ECHEC %-55s %s" % (nom, detail))


src = io.open("training.py", encoding="utf-8").read()

# ============================================================
print("\n1. LA POLITIQUE DECIDE, DANS LES TROIS BOUCLES")
# ============================================================
verifie("les trois boucles appellent `decide_sortie`",
        src.count("decide_sortie(") == 4,
        "1 definition + %d appels" % (src.count("decide_sortie(") - 1))
verifie("le rollout EXPLORE",
        "explore=True" in src,
        "PPO exige que l'action jouee vienne de la loi mise a jour")
# LE MOTIF PORTE LA PARENTHESE FERMANTE : sans elle il comptait aussi
# l'occurrence du COMMENTAIRE qui explique le choix, et le test echouait
# sur du texte au lieu de code.
# LA VALIDATION ET LE TEST JOUENT L'ACTION LA PLUS PROBABLE : pas de
# hasard en live, donc pas de hasard dans ce qui juge le modele — decision
# du proprietaire, 2026-09-25. Le tirage a graine fixe reste disponible
# par `evaluation_stochastique`, mais il n'est pas la regle.
verifie("validation et test jouent l'action la plus probable",
        src.count("generateur=_gen_v)") == 2
        and src.count("generateur=_gen_t)") == 2
        and not T.PPOConfig().evaluation_stochastique,
        "entree ET sortie, validation ET test")


# LES DEUX ANCIENNES REGLES NE DECIDENT PLUS RIEN. Les garder en
# parallele a ete explicitement refuse : deux organes qui repondent a la
# meme question finissent par diverger.
for _m, _n in (("_f[_bi]", "demande_cloture"), ("_fp[_bi]", "demande_profit")):
    verifie("`%s` ne decide plus dans le rollout" % _n, _m not in src)
verifie("aucune des deux n'est consultee ailleurs",
        "_vf[_bi]" not in src and "_tf[_bi]" not in src)

# ============================================================
print("\n2. LES TETES SUPERVISEES SONT DEBRANCHEES, PAS SUPPRIMEES")
# ============================================================
c = T.PPOConfig()
verifie("leur passe d'entrainement est eteinte",
        int(c.pas_cloture_par_epoch) == 0 and int(c.pas_profit_par_epoch) == 0,
        "cloture %d pas, profit %d pas"
        % (c.pas_cloture_par_epoch, c.pas_profit_par_epoch))
torch.manual_seed(0)
pol = SAINTPolicySingleHead(n_features=OBS_N_FEATURES, d_model=8,
                            num_blocks=1, heads=1, max_len=4, n_freq=2,
                            n_actions=N_ACTIONS).eval()
verifie("leurs tenseurs restent dans le reseau",
        hasattr(pol, "tete_cloture") and hasattr(pol, "tete_profit"),
        "les points de reprise anterieurs restent chargeables")

# ============================================================
print("\n3. ELLE NE LIT QUE LES QUATRE COLONNES DE LA POSITION")
# ============================================================
verifie("deux actions, pas quatre", N_ACTIONS_SORTIE == 2,
        "TENIR=%d FERMER=%d — elle n'entre jamais" % (TENIR, FERMER))
x = np.random.randn(11, 4, OBS_N_FEATURES).astype(np.float32)
a, lp, v = T.decide_sortie(pol, x, "cpu", N_BASE_FEATURES, explore=True)
verifie("elle rend une action par etat", a.shape == (11,))
verifie("un log-vraisemblance par action", lp.shape == (11,) and np.isfinite(lp).all())
verifie("une valeur par etat", v.shape == (11,) and np.isfinite(v).all())
verifie("les actions sont dans {TENIR, FERMER}",
        bool(np.isin(a, [TENIR, FERMER]).all()))
verifie("elle passe par `entree_sortie`",
        "entree_sortie(etats, n_base)" in src,
        "le SEUL endroit qui sache extraire ses colonnes")
p4 = T.entree_profit(x, N_BASE_FEATURES)
verifie("la tete de profit garde ses quatre colonnes",
        p4.shape == (11, N_PROFIT_FEATURES) and N_PROFIT_FEATURES == 4)
# LE SENS : les shorts ouvrent depuis le 2026-09-25, et un meme etat de
# marche ne dit pas la meme chose selon le cote tenu.
p5 = T.entree_sortie(x, N_BASE_FEATURES)
verifie("la sortie lit cinq colonnes : les quatre, plus le sens",
        p5.shape == (11, N_SORTIE_FEATURES) and N_SORTIE_FEATURES == 5
        and np.array_equal(p5[:, :4], p4))
verifie("le sens est celui du bloc position",
        np.array_equal(p5[:, COL_SENS_SORTIE],
                       x[:, -1, N_BASE_FEATURES + IDX_SENS_POS]))
verifie("le tampon PPO lit la meme extraction",
        "_pin = entree_sortie(_ep, N_BASE_FEATURES)" in src)

# L'ARGMAX EST DETERMINISTE, L'ECHANTILLONNAGE NE L'EST PAS.
a1, _, _ = T.decide_sortie(pol, x, "cpu", N_BASE_FEATURES, explore=False)
a2, _, _ = T.decide_sortie(pol, x, "cpu", N_BASE_FEATURES, explore=False)
verifie("sans exploration, la decision est reproductible",
        bool((a1 == a2).all()))
_xg = np.random.randn(64, 4, OBS_N_FEATURES).astype(np.float32)
_g1 = torch.Generator().manual_seed(7)
_g2 = torch.Generator().manual_seed(7)
_t1, _, _ = T.decide_sortie(pol, _xg, "cpu", N_BASE_FEATURES,
                            explore=True, generateur=_g1)
_t2, _, _ = T.decide_sortie(pol, _xg, "cpu", N_BASE_FEATURES,
                            explore=True, generateur=_g2)
verifie("a graine egale, le tirage est identique", bool((_t1 == _t2).all()),
        "deux epochs se jugent sur les memes aleas")
_m64 = np.ones((64, 3), bool)
_e1, _, _ = T.decide_entree(pol, _xg, _m64, "cpu", explore=True,
                            generateur=torch.Generator().manual_seed(7))
_e2, _, _ = T.decide_entree(pol, _xg, _m64, "cpu", explore=True,
                            generateur=torch.Generator().manual_seed(7))
verifie("pour l'entree aussi", bool((_e1 == _e2).all()))
verifie("et il ne se reduit pas a l'argmax",
        len(set(_e1.tolist())) > 1,
        "une politique presque uniforme tire les trois actions")

# ============================================================
print("\n4. LE LOYER DU TEMPS EST CE QUI FAIT LE SCALPING")
# ============================================================
verifie("le loyer est non nul",
        float(c.loyer_temps_atr) > 0.0,
        "a zero la politique reapprend a tenir des jours — c'est ce que "
        "le run du 2026-09-22 a fait")
_ar = 0.79   # aller-retour reel, en ATR : 4.18 bps sur un ATR de 5.3 bps
verifie("tenir trente barres coute une fraction d'aller-retour",
        0.05 < 30 * c.loyer_temps_atr / _ar < 0.60,
        "%.1f %% d'un aller-retour" % (100 * 30 * c.loyer_temps_atr / _ar))
_derive = 0.00073   # +0.231 bps/heure, en ATR par barre
verifie("il domine largement la derive du marche",
        c.loyer_temps_atr > 3 * _derive,
        "loyer %.5f contre derive %.5f ATR/barre — tenir sans raison coute"
        % (c.loyer_temps_atr, _derive))
verifie("la recompense telescope",
        '_lat - float(lat_prec[_k])' in src,
        "la somme d'un trade vaut le net realise moins le loyer")

# ============================================================
print("\n4b. UNE DECISION A CHAQUE BARRE, PARTOUT")
# ============================================================
# LE PROBLEME ETAIT LA FREQUENCE, PAS LE TARIF. Mesure du 2026-09-22 :
# |delta latent| vaut 0.650 ATR par barre en moyenne, le loyer 0.0066 —
# 98 fois plus petit. PPO ne pouvait pas le voir, et en quatre epochs
# l'entropie est tombee de 0.572 a 0.192 pendant que le taux de fermeture
# passait de 40 % a 10.9 %.
#
# LA CADENCE EST REPASSEE A UNE BARRE, et c'est un choix du proprietaire
# du 2026-09-25. Elle avait ete portee a 15 pour rendre le loyer visible ;
# le LOYER ZOMBIE — loyer multiplie sur les positions en perte — rend le
# signal visible du cote des pertes sans espacer les decisions. Ce test
# gardait l'ancienne valeur ; il garde maintenant la nouvelle, et surtout
# qu'elle soit LA MEME PARTOUT.
_K = int(c.pas_decision_sortie)
verifie("la sortie decide a chaque barre",
        _K == 1, "pas_decision_sortie = %d" % _K)
verifie("la meme cadence au rollout, en validation et au test",
        src.count("cadence_sortie(") == 4,
        "une seule regle, `cadence_sortie`, lue aux trois boucles")
_bruit = 0.650
_zomb = float(getattr(c, "loyer_zombie_mult", 1.0))
_rp = c.loyer_temps_atr * _zomb / _bruit
verifie("du cote des PERTES, le loyer zombie rend le signal visible",
        _rp > 0.10,
        "%.1f %% du bruit d'une barre, contre %.1f %% sans lui"
        % (100 * _rp, 100 * c.loyer_temps_atr / _bruit))
# LE MOTIF NE NOMME PLUS LE LOYER, SEULEMENT SA MULTIPLICATION PAR LA
# DUREE. Il exigeait `(_loyer + _derive) * _dt` et a casse le 2026-09-25
# quand le loyer est devenu `_loyer_eff` — un changement VOULU, fait entre
# deux sessions : le loyer zombie ci-dessous. Le test gardait une ecriture,
# pas une propriete ; c'est la propriete qui compte.
verifie("la duree entre decisions entre dans la recompense",
        "+ _sens_k * _derive) * _dt" in src,
        "sans quoi une decision couvrant 15 barres n'en paierait qu'une")

# LE LOYER ZOMBIE. Le loyer est multiplie quand la position PERD, et
# seulement alors : la politique apprend a couper les perdants qui
# trainent sans qu'on touche aux gagnants, et sans horloge. C'est la
# reponse a l'effondrement du 2026-09-22 — la politique avait appris a ne
# jamais fermer, parce que rien ne rendait la perte qui dure couteuse.
verifie("le loyer zombie existe",
        float(getattr(c, "loyer_zombie_mult", 1.0)) > 1.0,
        "x%.1f sur les positions en perte" % getattr(c, "loyer_zombie_mult", 1.0))
verifie("il ne s'applique que SOUS LE POINT MORT",
        "if _lat < _mort - _marge_z" in src
        and T.PPOConfig().marge_zombie_atr == 2.0
        and "_mort = -float(envs[_k].cout_entree_atr)" in src,
        "le spread seul ne fait pas un zombie : 66 % des positions en "
        "auraient ete a leur premiere decision")

# LA DERIVE EST RETIREE. Sans cela, tenir une position longue dans un BTC
# qui monte rapporte en moyenne, et la politique apprend a ne jamais
# fermer — ce qu'elle a fait.
verifie("la derive du marche est retiree de la recompense",
        float(c.derive_atr_barre) > 0.0 and "_derive" in src,
        "%.5f ATR/barre — on veut le TIMING, pas le beta" % c.derive_atr_barre)

# ET L'ENTROPIE NE DOIT PAS S'EFFONDRER AVANT D'AVOIR APPRIS.
verifie("la sortie a son propre poids d'entropie",
        float(c.entropie_sortie) > float(getattr(c, "entropy_coef", 0.003)),
        "%.3f contre %.3f pour l'entree — a 0.003 elle est tombee a 0.192"
        % (c.entropie_sortie, getattr(c, "entropy_coef", 0.003)))

# ============================================================
print("\n5. LA GEOMETRIE EST REVENUE AU SCALPING")
# ============================================================
c1 = T.PPOConfig()
c1.timeframe_entrainement = "M1"
verifie("l'horizon d'etiquetage tient dans l'heure",
        int(c1.horizon_cloture) <= 60,
        "horizon %d min — il a valu 480, et personne n'avait verifie la "
        "direction cumulee" % c1.horizon_cloture)
verifie("aucune horloge ne ferme",
        T.plafond_detention(c1) == 0,
        "la duree n'est plus bornee, elle est TARIFEE")
verifie("l'echantillon couvre plus que l'horizon",
        int(c1.tenue_max_cloture) > int(c1.horizon_cloture),
        "tenue_max %d > horizon %d" % (c1.tenue_max_cloture, c1.horizon_cloture))

# ============================================================
print("\n6. QUATRE TETES, CHACUNE SON OPTIMISEUR")
# ============================================================
# LE 2026-09-25, LE PROPRIETAIRE : une tete pour l'achat, une pour la
# vente, une pour la coupure des pertes, une pour la coupure des gains,
# CHACUNE AVEC SON OPTIMISEUR, toutes en PPO, toutes sur toutes les
# features.
verifie("les optimiseurs sont crees par groupe",
        src.count("optims_ppo = optimiseurs_ppo(policy, cfg)") == 1,
        "un par tete, plus le tronc et le critique d'entree")
verifie("il est cree une fois par fold, pas a chaque epoch",
        "_optim_s = optim.Adam(" not in src,
        "reconstruit a chaque epoch, il jetait l'etat Adam")
_gr = pol.groupes_ppo()
verifie("quatre tetes, chacune son groupe",
        all(k in _gr for k in ("achat", "vente", "gain", "perte")),
        ", ".join(sorted(_gr)))
_vus = {}
_dup = [k for k, v in _gr.items() for q in v
        if _vus.setdefault(id(q), k) != k]
verifie("aucun poids n'appartient a deux optimiseurs", not _dup, str(_dup))
verifie("l'entree est entrainee par PPO",
        "maj_ppo_entree(policy, ep_buf, optims_ppo, cfg, device)" in src)
verifie("la sortie aussi",
        "maj_ppo_sortie(policy, sortie_buf, optims_ppo, cfg, device)" in src)
verifie("l'avantage est normalise PAR TETE",
        "_ADV[_mq] = (_x - _x.mean()) / (_x.std() + 1e-8)" in src,
        "ensemble, les pertes gonflees par le loyer zombie fixeraient "
        "l'echelle et les gains deviendraient du bruit")
verifie("la derniere transition est terminale",
        'b[-1]["done"] = True' in src,
        "l'episode s'arrete et l'environnement solde : bootstrapper "
        "au-dela espererait une suite qui n'existe pas")
verifie("`lambda_gae` est lu au bon nom",
        "float(cfg.lambda_gae)" in src,
        "`gae_lambda` n'existe pas — un getattr aurait pris 0.95 en silence")

# ============================================================
print("\n7. LE JOURNAL MONTRE CE QUE LA POLITIQUE FAIT")
# ============================================================
for _m, _q in (("PPO entree", "la ligne de l'entree"),
               ("achat {_pa_:.1f}%", "la repartition des entrees"),
               ("PPO sortie", "la ligne de la sortie"),
               ("PPO sortie {_cote}", "une ligne PAR TETE de sortie"),
               ("ferme {100*_ferme", "le taux de fermeture de chaque tete"),
               ("KL {_st_s['kl']", "la divergence"),
               ("clip {100*_st_s['clip']", "la part ecretee")):
    verifie("le journal porte %s" % _q, _m in src)
verifie("il dit aussi quand le lot est trop petit",
        src.count("trop peu de") >= 2,
        "un PPO muet ressemble a un PPO qui marche")

# ============================================================
print("\n8. UN MODELE DE SORTIE, DEUX TETES, SEPARE DES ENTREES")
# ============================================================
# CE QUE LE PROPRIETAIRE A DEMANDE, le 2026-09-25 : une tete pour fermer
# les GAINS, une pour fermer les PERTES, sur le MEME modele PPO, et ce
# modele distinct des tetes d'ouverture. La tete unique d'avant devait
# donner les deux reponses avec les memes poids de sortie.
for _nom in ("acteur_sortie_gain", "critique_sortie_gain",
             "acteur_sortie_perte", "critique_sortie_perte"):
    verifie("la tete `%s` existe" % _nom, hasattr(pol, _nom))

_x = torch.randn(8, N_SORTIE_FEATURES)
_x[:4, 0] = 2.0          # en gain
_x[4:, 0] = -2.0         # en perte
pol.zero_grad(set_to_none=True)
_lg, _v = pol.sortie(_x)
(_lg[:4].sum() + _v[:4].sum()).backward()


def _g(mod):
    return sum(float(q.grad.abs().sum()) for q in mod.parameters()
               if q.grad is not None)


verifie("une decision EN GAIN entraine la tete de gain",
        _g(pol.acteur_sortie_gain) > 0.0)
verifie("et le corps partage",
        _g(pol.mlp_sortie) > 0.0, "une seule representation de la situation")
verifie("mais JAMAIS la tete de perte",
        _g(pol.acteur_sortie_perte) == 0.0
        and _g(pol.critique_sortie_perte) == 0.0)
verifie("ni les tetes d'ouverture",
        _g(pol.tete_achat) == 0.0 and _g(pol.tete_vente) == 0.0)

# L'EQUILIBRE EXACT VA A LA TETE DE PERTE : aucun gain a proteger, et le
# spread deja paye.
_z = torch.zeros(2, N_SORTIE_FEATURES)
pol.zero_grad(set_to_none=True)
_lz, _ = pol.sortie(_z)
_lz.sum().backward()
verifie("l'equilibre exact va a la tete de perte",
        _g(pol.acteur_sortie_perte) > 0.0 and _g(pol.acteur_sortie_gain) == 0.0)
pol.zero_grad(set_to_none=True)

# ============================================================
print("\n9b. LE POINT MORT EST CELUI QUE L'ENVIRONNEMENT A FACTURE")
# ============================================================
import pandas as _pd
from saint_core import FEATURE_COLS as _FC
_df = _pd.read_pickle("data_cache_BTCUSD_M1.pkl").iloc[:60_000]
_df = _df.reset_index(drop=True)
_st = T.compute_and_save_global_norm_stats(_df, _FC, path=None)
_md = T.MarketData(_df, _FC, _st)
_cf = T.PPOConfig()
_cf.episode_length = 5_000
for _act, _nom in ((0, "long"), (1, "short")):
    _e = T.BTCTradingEnvDiscrete(_md, _cf)
    T.reset_au_depart(_e, 20_000)
    verifie("%s : point mort nul a plat" % _nom, _e.cout_entree_atr == 0.0)
    _e.step(_act)
    _ci = int(_e.entry_idx)
    # LE POINT MORT EST LA PERTE LATENTE A MARCHE INCHANGE : le short paie
    # le spread a la sortie, et son point mort doit le compter.
    _o = float(_md.open[_ci])
    _sortie0 = _o * (1.0 + float(_e._p_spread[0]) / 1e4) if _act == 1 else _o
    _attendu = abs(_sortie0 - _e.entry_price) / _e.entry_atr
    verifie("%s : point mort = perte latente a marche inchange" % _nom,
            _e.n_positions == 1 and _e.cout_entree_atr > 0.0
            and abs(_e.cout_entree_atr - _attendu) < 1e-9,
            "%.3f ATR" % _e.cout_entree_atr)
    # LA CLOTURE DU MODELE PAIE LE PRIX EXECUTABLE. Le short rachetait au
    # BID : il ne payait jamais le spread, et paraissait donc meilleur.
    _sp = float(_e._p_spread[0])
    _e.step(3)
    _tm = _e.trades_meta[-1]
    _bid = float(_md.open[_tm["exit_idx"]])
    _attendu_sortie = _bid * (1.0 + _sp / 1e4) if _act == 1 else _bid
    verifie("%s : la cloture du modele paie le prix executable" % _nom,
            abs(_tm["exit_price"] - _attendu_sortie) < 1e-9 * _bid,
            "rachat a l'ask" if _act == 1 else "revente au bid")
    verifie("%s : remis a zero a la fermeture" % _nom,
            _e.n_positions == 0 and _e.cout_entree_atr == 0.0)

print("\n9c. LA COLLECTE VISE DES TRADES, PAS DES CONSULTATIONS")
# Le 2026-09-25 le regulateur comptait les attentes : episodes joues 7, 4,
# 2, 1, 1, et ~1 000 trades d'entrainement pour 4 400 en validation.
verifie("le regulateur divise la cible de TRADES par les ouvertures",
        "_vise = int(round(cfg.cible_trades / _par_ep))" in src
        and '_ouv = max(int(sampling_audit.get("ouvertures", 0)), 1)' in src)
verifie("les ouvertures sont comptees a l'ouverture",
        'sampling_audit["ouvertures"] += 1' in src)
verifie("l'entrainement vise au moins la validation",
        T.PPOConfig().cible_trades >= 4_400, str(T.PPOConfig().cible_trades))
# 1.0 DEPUIS LE PPO COMPLET : les attentes sont des actions de la
# politique d'entree, et l'avantage se calcule sur la suite complete.
verifie("toutes les attentes sont gardees",
        T.PPOConfig().garde_attentes == 1.0
        and src.count("if _garde:") == 2,
        "%.0f %% gardees" % (100 * T.PPOConfig().garde_attentes))
verifie("les ouvertures, elles, sont toutes gardees",
        "_garde = (nouveau is not None and j_ouvert < 0" in src)

print("\n9d. LES DEPARTS DU CURRICULUM SONT PARTAGES, PAS RECOPIES")
# 160 environnements recopiaient chacun 2.6 Mo de departs identiques.
_e1 = T.BTCTradingEnvDiscrete(_md, _cf)
_e2 = T.BTCTradingEnvDiscrete(_md, _cf)
if _e1.low_vol_starts is not None:
    verifie("deux environnements lisent le MEME tableau",
            _e1.low_vol_starts is _e2.low_vol_starts
            and _e1.high_vol_starts is _e2.high_vol_starts)
    verifie("il est en lecture seule",
            not _e1.low_vol_starts.flags.writeable)
_e3 = T.BTCTradingEnvDiscrete(_md, _cf)
_ref = _e3.low_vol_starts
del _md._departs_vol
_e4 = T.BTCTradingEnvDiscrete(_md, _cf)
verifie("recalcule a neuf, il donne les memes departs",
        (_ref is None and _e4.low_vol_starts is None)
        or np.array_equal(_ref, _e4.low_vol_starts))

print("\n9e. L'AGE, LA TENUE ET L'HORIZON RESTENT ALIGNES")
# Trois reglages qui n'ont de sens qu'ensemble. L'horizon est passe de 60 a
# 15 le 2026-09-25 ; si l'un des trois ne suit pas, la colonne d'age sature
# trop tot ou reste ecrasee, sans qu'aucune erreur ne se leve.
from saint_core import SCALPING_MAX_HOLDING as _SMH
_c9 = T.PPOConfig()
verifie("l'age se normalise sur l'horizon",
        _SMH == int(_c9.horizon_cloture) == int(_c9.scalping_max_holding),
        "SCALPING_MAX_HOLDING %d, horizon %d" % (_SMH, _c9.horizon_cloture))
verifie("la tenue rejouee va jusqu'a la saturation de l'age",
        int(_c9.tenue_max_cloture) == 3 * _SMH,
        "tenue_max_cloture %d = 3 x %d" % (_c9.tenue_max_cloture, _SMH))
verifie("l'horizon est celui du scalping choisi", int(_c9.horizon_cloture) == 15)

print("\n10. PPO COMPLET : QUATRE TETES SUR TOUTES LES FEATURES")
# ============================================================
from saint_core import ACHETER, VENDRE, ATTENDRE, N_ACTIONS_ENTREE
torch.manual_seed(1)
np.random.seed(1)
_p10 = SAINTPolicySingleHead(n_features=OBS_N_FEATURES, d_model=8,
                             num_blocks=2, heads=1, n_freq=16, mlp_dim=4,
                             lecture="colonnes", max_len=4,
                             n_actions=N_ACTIONS)
_x10 = torch.randn(6, 4, OBS_N_FEATURES)
_lg10, _v10 = _p10.entree(_x10)
verifie("l'entree rend trois logits et une valeur",
        tuple(_lg10.shape) == (6, N_ACTIONS_ENTREE) and tuple(_v10.shape) == (6,))
verifie("le logit d'ATTENDRE est fixe a zero",
        bool((_lg10[:, ATTENDRE] == 0).all()),
        "chaque tete dit combien elle prefere ouvrir a attendre")
verifie("la politique d'entree part presque uniforme",
        float(_lg10[:, :2].abs().max()) < 0.1,
        "acteurs initialises a gain 0.01")

# L'achat n'entraine jamais la vente, et reciproquement.
_p10.zero_grad(set_to_none=True)
_p10.entree(_x10)[0][:, ACHETER].sum().backward()
verifie("un gradient d'ACHAT ne touche pas la tete de vente",
        _g(_p10.tete_achat) > 0 and _g(_p10.tete_vente) == 0
        and _g(_p10.mlp_vente) == 0)
verifie("il remonte dans le tronc", _g(_p10.embed) > 0,
        "la tete lit toutes les features")
_p10.zero_grad(set_to_none=True)

# La sortie lit le tronc, donc toutes les features.
_ps = torch.randn(6, N_SORTIE_FEATURES)
_ps[:3, 0] = 2.0
_ps[3:, 0] = -2.0
_lgs, _vs = _p10.sortie_complete(_x10, _ps)
(_lgs[:3].sum() + _vs[:3].sum()).backward()
verifie("une decision en GAIN entraine la tete de gain",
        _g(_p10.lecteur_gain) > 0 and _g(_p10.acteur_sortie_gain) > 0)
verifie("jamais la tete de perte",
        _g(_p10.lecteur_perte) == 0 and _g(_p10.acteur_sortie_perte) == 0)
verifie("ni les tetes d'ouverture",
        _g(_p10.tete_achat) == 0 and _g(_p10.tete_vente) == 0)
verifie("la sortie lit le tronc : toutes les features",
        _g(_p10.embed) > 0, "elle ne lisait que cinq colonnes")
_p10.zero_grad(set_to_none=True)

# Les deux mises a jour, sur des tampons factices : chacune ne fait bouger
# que les groupes qu'elle doit faire bouger.
_c10 = T.PPOConfig()
_c10.batch_size = 64
_c10.max_transitions_ppo = 0
_o10 = T.optimiseurs_ppo(_p10, _c10)
_g10 = _p10.groupes_ppo()


def _photo():
    return {k: [q.detach().clone() for q in v] for k, v in _g10.items()}


def _bouge(a, b):
    return {k for k in a
            if any(float((x - y).abs().max()) > 0 for x, y in zip(a[k], b[k]))}


_n = 300
_eb = []
for _ in range(2):
    _ac = np.random.randint(0, 3, _n)
    _eb.append({
        "states": [np.random.randn(4, OBS_N_FEATURES).astype(np.float32)
                   for _ in range(_n)],
        "masques": [np.ones(3, bool) for _ in range(_n)],
        "actions": list(_ac),
        "rewards": list(np.random.randn(_n) * (_ac != 2)),
        "dts": list(np.where(_ac != 2, 5, 1)),
        "dones": [False] * (_n - 1) + [True],
        "lps": [float(np.log(1 / 3))] * _n, "vals": [0.0] * _n,
        "barres": [0] * _n, "positions": [0] * _n})
_a = _photo()
_st10 = T.maj_ppo_entree(_p10, _eb, _o10, _c10, "cpu")
verifie("la mise a jour d'ENTREE fait bouger achat, vente, tronc, critique",
        _bouge(_a, _photo()) == {"achat", "vente", "tronc", "valeur_entree"},
        str(sorted(_bouge(_a, _photo()))))
_sb = []
for _ in range(2):
    _sb.append([{"o": np.random.randn(4, OBS_N_FEATURES).astype(np.float32),
                 "p": np.random.randn(N_SORTIE_FEATURES).astype(np.float32),
                 "a": int(np.random.randint(0, 2)), "lp": float(np.log(0.5)),
                 "v": 0.0, "r": float(np.random.randn()),
                 "done": bool(np.random.rand() < 0.1)} for _ in range(200)])
_a = _photo()
T.maj_ppo_sortie(_p10, _sb, _o10, _c10, "cpu")
verifie("la mise a jour de SORTIE fait bouger gain, perte, tronc",
        _bouge(_a, _photo()) == {"gain", "perte", "tronc"},
        str(sorted(_bouge(_a, _photo()))))

# Le masque est respecte, et le solde decide AVANT le tirage.
_ae, _, _ = T.decide_entree(_p10, [np.random.randn(4, OBS_N_FEATURES)
                                   .astype(np.float32) for _ in range(50)],
                            np.array([[True, False, True]] * 50), "cpu")
verifie("une action masquee n'est jamais tiree", bool((_ae != VENDRE).all()))

# L'avantage semi-MDP, calcule a la main.
_adv, _ret = T.avantages_semi_mdp([1.0, 0.0, 2.0], [3, 1, 2], [0.5, 0.2, 0.1],
                                  [False, False, True], 0.9, 0.95)
verifie("l'avantage actualise chaque decision par sa duree",
        np.allclose(_adv, [1.6947, 1.5145, 1.9], atol=1e-4),
        str(np.round(_adv, 4)))

# UNE REGLE, TROIS LECTEURS.
verifie("collecte, validation et test decident par `decide_entree`",
        src.count("decide_entree(") == 4, "1 definition + 3 appels")
verifie("et masquent par `masque_entree`",
        src.count("masque_entree(") == 4, "1 definition + 3 appels")
verifie("le classement ne tourne plus",
        T.PPOConfig().pas_rang_par_epoch == 0 and not T.PPOConfig().diag_rang)

print("\n9. LE COTE SE DECLARE A UN SEUL ENDROIT")
# ============================================================
# Le 2026-09-25 la configuration disait "both" et le run est parti en LONG
# seul : le bloc __main__ redeclarait `cfg_long.side = "long"`.
_main = src[src.index('if __name__ == "__main__":'):]
_actives = [l for l in _main.split("\n")
            if ".side = " in l and not l.strip().startswith("#")]
verifie("le bloc __main__ ne redeclare pas le cote",
        not _actives, "; ".join(l.strip() for l in _actives))
verifie("la configuration ouvre les deux cotes",
        T.PPOConfig().side == "both", T.PPOConfig().side)

print("\n%d/%d OK" % (_ok, _ok + _ko))
if _ko:
    print("\nLA SORTIE EST REDEVENUE UN SEUIL. Quatre calibrations ont deja")
    print("echoue sur ce chemin — le defaut n'etait pas la valeur du seuil,")
    print("c'etait l'idee de seuiller une amplitude predite.")
sys.exit(1 if _ko else 0)
