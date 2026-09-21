# -*- coding: utf-8 -*-
"""La tete de cloture DECIDE, et une position ne peut pas devenir immortelle.

CE QUI A RENDU CE TEST NECESSAIRE, le 2026-09-21. `tete_cloture` a ete
construite, sa cible ecrite, sa passe supervisee branchee, et personne ne
l'appelait : `policy.cloture` n'apparaissait qu'a l'entrainement de la tete
elle-meme.

Sans stop, sans objectif, sans trailing et sans plafond de detention, la
seule sortie etait l'action 3 — que rien ne produisait, parce que
`peut_decider()` ne rend vrai que si l'environnement peut ENTRER et qu'a
une position a la fois un env en position n'y figure jamais. Le journal du
fold 1 :

    12 episodes -> 12 decisions -> 1 trade par episode, tenu 5 759 barres.

« Scalping M1 » decrivait quatre jours de detention.

    python test_cloture_branchee.py
"""
import io
import sys

import numpy as np
import torch

import training as T
from saint_core import (
    N_ACTIONS,
    N_BASE_FEATURES,
    N_POS_FEATURES,
    SAINTPolicySingleHead,
    build_mask_from_pos_scalar,
)

_ok = _ko = 0


def verifie(nom, cond, detail=""):
    global _ok, _ko
    if cond:
        _ok += 1
        print("  ok   %-54s %s" % (nom, detail))
    else:
        _ko += 1
        print("  ECHEC %-53s %s" % (nom, detail))


# ============================================================
print("\n1. LA TETE EST CONSULTEE, ET PAS SEULEMENT ENTRAINEE")
# ============================================================
src = io.open("training.py", encoding="utf-8").read()
verifie("les trois boucles appellent `demande_cloture`",
        src.count("demande_cloture(") == 4,
        "1 definition + %d appels" % (src.count("demande_cloture(") - 1))
for _b, _n in (("actions_env[_k] = 3", "rollout"),
               ("v_actions[_k] = 3", "validation"),
               ("t_actions[_k] = 3", "test")):
    verifie("la boucle de %s peut emettre CLOTURER" % _n, _b in src)

# ============================================================
print("\n2. LE SIGNE DU LOGIT EST LA DECISION")
# ============================================================
torch.manual_seed(0)
pol = SAINTPolicySingleHead(n_features=N_BASE_FEATURES + N_POS_FEATURES,
                            d_model=8, num_blocks=1, heads=1, max_len=2,
                            n_freq=2, n_actions=N_ACTIONS)
pol.eval()
etats = np.random.randn(64, 2, N_BASE_FEATURES + N_POS_FEATURES).astype(np.float32)


class _Faux:
    """Une tete dont on impose le RISQUE predit, en ATR."""

    def __init__(self, risque):
        # softplus(z) = r  ->  z = log(exp(r) - 1)
        self._z = float(np.log(np.expm1(max(float(risque), 1e-6))))

    def cloture(self, x):
        return torch.full((x.shape[0], 1), self._z)


def _etat(latent, n=4):
    """Un etat EN POSITION, avec le latent voulu en ATR d'entree."""
    x = np.zeros((n, 2, N_BASE_FEATURES + N_POS_FEATURES), np.float32)
    x[:, :, N_BASE_FEATURES + 0] = 1.0        # sens long
    x[:, :, N_BASE_FEATURES + 1] = latent     # latent, colonne 1
    return x


# C'EST LA TETE QUI COMMANDE : le seuil vaut `k x risque_predit`, sans
# stop fixe. Voir `PPOConfig.coupe_risque` pour le balayage.
verifie("une perte au-dela du risque predit ferme",
        T.demande_cloture(_Faux(4.0), _etat(-1.5), "cpu", N_BASE_FEATURES,
                          coupe=0.25, marge=0.0).all(),
        "seuil 0.25 x 4.0 = 1.0, latent -1.5 -> ferme")
verifie("une perte en deca tient",
        not T.demande_cloture(_Faux(4.0), _etat(-0.5), "cpu", N_BASE_FEATURES,
                              coupe=0.25, marge=0.0).any(),
        "latent -0.5 sous un seuil de 1.0 : c'est du bruit")

# LE SEUIL SUIT LE MARCHE, ET C'EST TOUT L'INTERET. Meme latent, deux
# regimes : la position tient dans la tempete et sort dans le calme.
_lat = -1.5
verifie("marche AGITE : le seuil s'elargit, on tient",
        not T.demande_cloture(_Faux(12.0), _etat(_lat), "cpu",
                              N_BASE_FEATURES, coupe=0.25, marge=0.0).any(),
        "risque 12 ATR -> seuil 3.0, latent -1.5 tient")
verifie("marche CALME : le seuil se resserre, on sort",
        T.demande_cloture(_Faux(2.0), _etat(_lat), "cpu", N_BASE_FEATURES,
                          coupe=0.25, marge=0.0).all(),
        "risque 2 ATR -> seuil 0.5, latent -1.5 ferme")
verifie("un gain tient, quel que soit le risque",
        not T.demande_cloture(_Faux(50.0), _etat(+2.0), "cpu",
                              N_BASE_FEATURES, coupe=0.25, marge=0.0).any())

# AUCUN STOP FIXE N'EST POSE. Si `marge_sortie` cessait d'etre nulle, la
# regle redeviendrait un stop et la tete un ornement.
import training as _T
verifie("la configuration ne pose AUCUN stop fixe",
        float(_T.PPOConfig().marge_sortie) == 0.0
        and not _T.PPOConfig().use_sl,
        "marge_sortie 0.0, use_sl False — le seuil vient de la tete")
verifie("et la tete commande vraiment",
        float(_T.PPOConfig().coupe_risque) > 0.0,
        "coupe_risque %.2f : sa sortie entre dans la decision"
        % _T.PPOConfig().coupe_risque)

verifie("le seuil ne peut pas devenir negatif",
        not T.demande_cloture(_Faux(1e-9), _etat(+0.01), "cpu",
                              N_BASE_FEATURES, coupe=0.0, marge=0.0).any(),
        "plancher a 0.05 ATR, meme a marge nulle")

# ============================================================
print("\n3. LES COLONNES VUES PAR LA TETE SONT CELLES DE SON ENTRAINEMENT")
# ============================================================
_vu = {}


class _Espion:
    def cloture(self, x):
        _vu["x"] = x.numpy().copy()
        return torch.zeros((x.shape[0], 1)) - 1.0


_e = etats.copy()
_e[:, :, N_BASE_FEATURES + 3:] = 7.0      # capacite, creux, budget : non nuls
T.demande_cloture(_Espion(), _e, "cpu", N_BASE_FEATURES)
verifie("capacite et creux sont remis a zero",
        np.all(_vu["x"][:, :, N_BASE_FEATURES + 3:] == 0.0),
        "l'echantillon d'entrainement ne remplit que sens, latent et age")
verifie("le sens, le latent et l'age passent intacts",
        np.array_equal(_vu["x"][:, :, :N_BASE_FEATURES + 3],
                       _e[:, :, :N_BASE_FEATURES + 3]))
verifie("l'appelant n'est pas modifie",
        np.all(_e[:, :, N_BASE_FEATURES + 3:] == 7.0),
        "`demande_cloture` copie, elle n'ecrit pas dans l'etat de l'env")

# ============================================================
print("\n4. UNE POSITION NE PEUT PAS DEVENIR IMMORTELLE")
# ============================================================
c1 = T.PPOConfig()
c1.timeframe_entrainement = "M1"
c1.tenue_max_cloture = 30
c1.max_holding_bars = 0
verifie("en M1 le plafond vient de `tenue_max_cloture`",
        T.plafond_detention(c1) == 30,
        "et non de `max_holding_bars`, qui vaut %d" % c1.max_holding_bars)
verifie("aucune sortie automatique n'est desactivee par accident",
        not (c1.use_sl or c1.use_tp or c1.use_be_trail)
        and T.plafond_detention(c1) > 0,
        "sans stop, sans objectif, sans trailing -> le plafond est la SEULE "
        "sortie qui ne depende pas du modele")

c5 = T.PPOConfig()
c5.timeframe_entrainement = "M5"
c5.max_holding_bars = 0
verifie("en M5 rien ne change", T.plafond_detention(c5) == 0)

# UN REGLAGE POSE A LA MAIN NE DOIT PAS ETRE REMPLACE EN SILENCE. La
# premiere version prenait `tenue_max_cloture` des que le pas etait M1 et
# ignorait un plafond explicite : `test_economic_learning` en fixe un a 20,
# l'environnement sortait a 30, et la cible calculee a 20 ne concordait
# plus avec ce que l'environnement jouait.
cx = T.PPOConfig()
cx.timeframe_entrainement = "M1"
cx.tenue_max_cloture = 30
cx.max_holding_bars = 20
verifie("un plafond explicite gagne sur la valeur deduite",
        T.plafond_detention(cx) == 20,
        "20 pose a la main, 30 deduit de l'echantillon")

verifie("le plafond suit l'echantillon, il ne le double pas",
        T.plafond_detention(c1) == c1.tenue_max_cloture,
        "la tete n'a jamais vu d'age au-dela, sa colonne y sature")

# ============================================================
print("\n5. CLOTURER RESTE INTERDIT HORS POSITION")
# ============================================================
_mp = build_mask_from_pos_scalar(0, "cpu", "both")
_mi = build_mask_from_pos_scalar(1, "cpu", "both")
verifie("a plat : acheter, vendre, attendre — pas clore",
        [bool(x) for x in _mp] == [True, True, True, False])
verifie("en position : attendre ou clore — pas ouvrir",
        [bool(x) for x in _mi] == [False, False, True, True])

print("\n%d/%d OK" % (_ok, _ok + _ko))
if _ko:
    print("\nLA SORTIE EST A NOUVEAU DEBRANCHEE. Une position qui ne se ferme")
    print("pas ne fait pas un trade long : elle fait UN SEUL trade par")
    print("episode, et le reste de la fenetre n'est jamais joue.")
sys.exit(1 if _ko else 0)
