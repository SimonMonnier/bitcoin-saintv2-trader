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
verifie("le modele de sortie a SON optimiseur",
        "optimizer_sortie = optim.Adam(\n        policy.params_sortie()" in src,
        "separe de `optimizer_rang`, qui porte le tronc et les entrees")
verifie("il est cree une fois par fold, pas a chaque epoch",
        "_optim_s = optim.Adam(" not in src,
        "reconstruit a chaque epoch, il jetait l'etat Adam")
_ids_s = {id(q) for q in pol.params_sortie()}
_ent = list(pol.tete_achat.parameters()) + list(pol.tete_vente.parameters())
verifie("il ne porte AUCUN poids des tetes d'ouverture",
        not any(id(q) in _ids_s for q in _ent),
        "ouvrir et fermer sont deux metiers")
verifie("l'acteur d'ENTREE reste gele",
        "g[actor 0.00e+00" not in src or "actor.parameters()" not in
        src.split("PPO SUR LA SORTIE")[-1].split("print(f\"  {_col('phase'")[0],
        "seuls trois modules recoivent un gradient ici")
verifie("l'avantage est normalise PAR TETE",
        "_ADV[_mq] = (_x - _x.mean()) / (_x.std() + 1e-8)" in src,
        "ensemble, les pertes gonflees par le loyer zombie fixeraient "
        "l'echelle et les gains deviendraient du bruit")
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
               ("PPO sortie {_cote}", "une ligne PAR TETE"),
               ("ferme {100*_ferme", "le taux de fermeture de chaque tete"),
               ("KL {np.mean(_lk)", "la divergence"),
               ("clip {100*np.mean(_lcl)", "la part ecretee")):
    verifie("le journal porte %s" % _q, _m in src)
verifie("il dit aussi quand le lot est trop petit",
        "trop peu pour une mise a jour" in src,
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
