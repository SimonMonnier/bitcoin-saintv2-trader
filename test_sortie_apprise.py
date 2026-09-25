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
verifie("validation et test prennent l'argmax",
        src.count("explore=False)") == 2,
        "on juge la decision, pas son bruit")

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
verifie("elle passe par `entree_profit`",
        "entree_profit(etats, n_base)" in src,
        "le SEUL endroit qui sache extraire les quatre colonnes")
p4 = T.entree_profit(x, N_BASE_FEATURES)
verifie("quatre colonnes, pas une de plus",
        p4.shape == (11, N_PROFIT_FEATURES) and N_PROFIT_FEATURES == 4)

# L'ARGMAX EST DETERMINISTE, L'ECHANTILLONNAGE NE L'EST PAS.
a1, _, _ = T.decide_sortie(pol, x, "cpu", N_BASE_FEATURES, explore=False)
a2, _, _ = T.decide_sortie(pol, x, "cpu", N_BASE_FEATURES, explore=False)
verifie("sans exploration, la decision est reproductible",
        bool((a1 == a2).all()))

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
print("\n4b. LA DECISION EST SEMI-MDP, ET C'EST CE QUI REND LE LOYER VISIBLE")
# ============================================================
# LE PROBLEME ETAIT LA FREQUENCE, PAS LE TARIF. Mesure du 2026-09-22 :
# |delta latent| vaut 0.650 ATR par barre en moyenne, le loyer 0.0066 —
# 98 fois plus petit. PPO ne pouvait pas le voir, et en quatre epochs
# l'entropie est tombee de 0.572 a 0.192 pendant que le taux de fermeture
# passait de 40 % a 10.9 %.
#
# En decidant tous les K barres, le cout croit en K et le bruit en RACINE
# de K : le rapport s'ameliore en racine de K.
_K = int(c.pas_decision_sortie)
verifie("la sortie ne decide plus a chaque barre",
        _K > 1, "une decision tous les %d barres" % _K)
verifie("mais assez souvent pour couper vite",
        _K <= 30, "%d barres — au-dela le scalping perd son sens" % _K)
_bruit = 0.650
_r1 = c.loyer_temps_atr / _bruit
_rK = (c.loyer_temps_atr + c.derive_atr_barre) * _K / (_bruit * np.sqrt(_K))
verifie("le rapport signal/bruit du loyer a gagne un facteur",
        _rK > 3 * _r1,
        "%.1f %% par decision contre %.1f %% par barre" % (100 * _rK, 100 * _r1))
# LE MOTIF NE NOMME PLUS LE LOYER, SEULEMENT SA MULTIPLICATION PAR LA
# DUREE. Il exigeait `(_loyer + _derive) * _dt` et a casse le 2026-09-25
# quand le loyer est devenu `_loyer_eff` — un changement VOULU, fait entre
# deux sessions : le loyer zombie ci-dessous. Le test gardait une ecriture,
# pas une propriete ; c'est la propriete qui compte.
verifie("la duree entre decisions entre dans la recompense",
        "+ _derive) * _dt" in src,
        "sans quoi une decision couvrant 15 barres n'en paierait qu'une")

# LE LOYER ZOMBIE. Le loyer est multiplie quand la position PERD, et
# seulement alors : la politique apprend a couper les perdants qui
# trainent sans qu'on touche aux gagnants, et sans horloge. C'est la
# reponse a l'effondrement du 2026-09-22 — la politique avait appris a ne
# jamais fermer, parce que rien ne rendait la perte qui dure couteuse.
verifie("le loyer zombie existe",
        float(getattr(c, "loyer_zombie_mult", 1.0)) > 1.0,
        "x%.1f sur les positions en perte" % getattr(c, "loyer_zombie_mult", 1.0))
verifie("il ne s'applique qu'en PERTE",
        "_loyer * _loyer_z_mult if _lat < 0.0" in src,
        "un gagnant paie le loyer normal — on ne veut pas le faire fermer")

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
print("\n6. PPO N'ENTRAINE QUE LA SORTIE")
# ============================================================
verifie("l'optimiseur ne porte que les poids de sortie",
        "policy.mlp_sortie.parameters()" in src
        and "policy.acteur_sortie.parameters()" in src
        and "policy.critique_sortie.parameters()" in src)
verifie("l'acteur d'ENTREE reste gele",
        "g[actor 0.00e+00" not in src or "actor.parameters()" not in
        src.split("PPO SUR LA SORTIE")[-1].split("print(f\"  {_col('phase'")[0],
        "seuls trois modules recoivent un gradient ici")
verifie("l'avantage est centre reduit",
        "_ADV = (_ADV - _ADV.mean())" in src,
        "sinon `clip_eps` mordrait selon le loyer, qui n'a rien a voir")
verifie("la derniere transition est terminale",
        '_b[-1]["done"] = True' in src,
        "l'episode s'arrete et l'environnement solde : bootstrapper "
        "au-dela espererait une suite qui n'existe pas")
verifie("`lambda_gae` est lu au bon nom",
        "float(cfg.lambda_gae)" in src,
        "`gae_lambda` n'existe pas — un getattr aurait pris 0.95 en silence")

# ============================================================
print("\n7. LE JOURNAL MONTRE CE QUE LA POLITIQUE FAIT")
# ============================================================
for _m, _q in (("PPO sortie", "la ligne dediee"),
               ("ferme {100*_ppo['ferme']", "le taux de fermeture"),
               ("KL {_ppo['kl']", "la divergence"),
               ("clip {100*_ppo['clip']", "la part ecretee")):
    verifie("le journal porte %s" % _q, _m in src)
verifie("il dit aussi quand le lot est trop petit",
        "trop peu pour une mise a jour" in src,
        "un PPO muet ressemble a un PPO qui marche")

print("\n%d/%d OK" % (_ok, _ok + _ko))
if _ko:
    print("\nLA SORTIE EST REDEVENUE UN SEUIL. Quatre calibrations ont deja")
    print("echoue sur ce chemin — le defaut n'etait pas la valeur du seuil,")
    print("c'etait l'idee de seuiller une amplitude predite.")
sys.exit(1 if _ko else 0)
