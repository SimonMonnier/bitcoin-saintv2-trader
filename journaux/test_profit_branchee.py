# -*- coding: utf-8 -*-
"""La tete de PROFIT decide, et elle sait fermer une position EN GAIN.

CE QUI REND CE TEST NECESSAIRE, et ce n'est pas une precaution abstraite.
Le 2026-09-21, `tete_cloture` a ete construite, sa cible ecrite, sa passe
supervisee branchee — et personne ne l'appelait. Elle a vecu des jours en
ornement : sa perte descendait, son rho s'affichait, et elle ne decidait
rien. Aucune erreur ne se declenche dans ce cas.

CE QUE LA TETE DE PROFIT APPORTE, ET QUE RIEN D'AUTRE NE PEUT FAIRE. La
regle de cloture est :

    seuil = marge + coupe x risque          (toujours >= 0.05)
    fermer si  latent < -max(seuil, 0.05)

Le seuil est toujours positif et la condition exige un latent NEGATIF :
une position EN GAIN ne peut mathematiquement jamais declencher de
cloture. Mesure du 2026-09-22 sur la regle telle qu'elle est codee :
mediane ET moyenne de tenue des gagnants EGALES au plafond, sans une
exception. « Gagnant » ne voulait pas dire « la tete a decide de sortir
avec un gain », mais « le trade a survecu au chronometre ».

    python test_profit_branchee.py
"""
import io
import sys

import numpy as np
import torch

import training as T
from saint_core import (
    IDX_PROFIT_MARCHE,
    IDX_PROFIT_POS,
    N_ACTIONS,
    N_BASE_FEATURES,
    N_POS_FEATURES,
    N_PROFIT_FEATURES,
    OBS_N_FEATURES,
    SAINTPolicySingleHead,
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
print("\n1. LA TETE EST CONSULTEE, ET PAS SEULEMENT ENTRAINEE")
# ============================================================
verifie("les trois boucles appellent `demande_profit`",
        src.count("demande_profit(") == 4,
        "1 definition + %d appels" % (src.count("demande_profit(") - 1))
for _b, _n in (("if bool(_f[_bi]) or bool(_fp[_bi]):", "rollout"),
               ("if bool(_vf[_bi]) or bool(_vfp[_bi]):", "validation"),
               ("if bool(_tf[_bi]) or bool(_tfp[_bi]):", "test")):
    verifie("la boucle de %s lit sa sortie" % _n, _b in src)
verifie("elle a sa propre passe supervisee",
        "policy.profit(_pin)" in src and "_perte_p" in src)
verifie("sa passe a un optimiseur DEDIE",
        "_optimiseur_profit(policy, cfg)" in src,
        "sinon son gradient remonterait dans un tronc qu'elle ne lit pas")

# ============================================================
print("\n2. ELLE NE LIT QUE CE QU'ELLE A APPRIS A LIRE")
# ============================================================
verifie("quatre colonnes, pas une de plus", N_PROFIT_FEATURES == 4,
        "latent, age, creux_rang, flux_rang")
x = np.random.randn(9, 4, OBS_N_FEATURES).astype(np.float32)
p = T.entree_profit(x, N_BASE_FEATURES)
verifie("`entree_profit` rend (lot, 4)", p.shape == (9, 4))
verifie("colonne 0 = latent",
        np.allclose(p[:, 0], x[:, -1, N_BASE_FEATURES + IDX_PROFIT_POS[0]]))
verifie("colonne 1 = age",
        np.allclose(p[:, 1], x[:, -1, N_BASE_FEATURES + IDX_PROFIT_POS[1]]))
verifie("colonne 2 = creux_rang (bloc NORMALISE)",
        np.allclose(p[:, 2], x[:, -1, IDX_PROFIT_MARCHE[0]]))
verifie("colonne 3 = flux_rang (bloc NORMALISE)",
        np.allclose(p[:, 3], x[:, -1, IDX_PROFIT_MARCHE[1]]))
verifie("seule la DERNIERE barre est lue",
        np.allclose(T.entree_profit(x, N_BASE_FEATURES),
                    T.entree_profit(x[:, -1:, :], N_BASE_FEATURES)),
        "la mesure qui l'a justifiee portait sur l'etat courant")
_y = x.copy()
T.entree_profit(_y, N_BASE_FEATURES)
verifie("l'appelant n'est pas modifie", np.array_equal(_y, x))

# ============================================================
print("\n3. LA DECISION SUIT LE SEUIL")
# ============================================================
torch.manual_seed(0)
pol = SAINTPolicySingleHead(n_features=OBS_N_FEATURES, d_model=8,
                            num_blocks=1, heads=1, max_len=4, n_freq=2,
                            n_actions=N_ACTIONS).eval()


class _Faux:
    """Une tete dont on impose le RESTE predit, en ATR — et son rho.

    `_rho_profit` est pose par la passe supervisee sur la vraie politique.
    Une tete qui ne l'a pas est traitee comme jamais entrainee, donc
    muette : c'est le garde d'armement, et il se teste ici aussi.
    """

    def __init__(self, reste, rho=1.0, echelle=1.0):
        self._z = float(np.log(np.expm1(max(float(reste), 1e-6))))
        self._rho_profit = float(rho)
        self._echelle_profit = float(echelle)

    def profit(self, p):
        return torch.full((p.shape[0], 1), self._z)


class _FauxSansRho:
    """Une politique qui n'a jamais vu la passe supervisee."""

    def __init__(self, reste):
        self._z = float(np.log(np.expm1(max(float(reste), 1e-6))))

    def profit(self, p):
        return torch.full((p.shape[0], 1), self._z)


def _etat(latent, n=5):
    """Un etat EN POSITION, avec le latent voulu."""
    a = np.zeros((n, 4, OBS_N_FEATURES), np.float32)
    a[:, :, N_BASE_FEATURES + 0] = 1.0
    a[:, :, N_BASE_FEATURES + 1] = latent
    return a


# LA REGLE EST RELATIVE AU GAIN DEJA ACQUIS. Avec coupe = 0.25 et un
# latent de +3.0 ATR, le seuil effectif vaut 0.75 ATR.
verifie("plus rien a prendre -> on ferme",
        T.demande_profit(_Faux(0.2), _etat(+3.0), "cpu",
                         N_BASE_FEATURES, 0.25).all(),
        "reste 0.2 ATR sous 0.25 x 3.0 = 0.75")
verifie("il reste a prendre -> on tient",
        not T.demande_profit(_Faux(9.0), _etat(+3.0), "cpu",
                             N_BASE_FEATURES, 0.25).any(),
        "reste 9.0 ATR bien au-dessus de 0.75")
verifie("a coupe NULLE la regle ne mord jamais",
        not T.demande_profit(_Faux(0.0), _etat(+3.0), "cpu",
                             N_BASE_FEATURES, 0.0).any(),
        "un modele sans tete entrainee ne doit pas fermer au hasard")

# LE SEUIL SUIT LE GAIN, ET C'EST TOUT L'INTERET. Meme reste predit, deux
# positions : celle qui a beaucoup gagne se solde, celle qui commence
# tient. Une constante en ATR ne sait pas faire cette difference — et
# c'est ce qui a inverse le rapport des tenues le 2026-09-22.
verifie("gros gain acquis, meme reste -> on solde",
        T.demande_profit(_Faux(1.0), _etat(+8.0), "cpu",
                         N_BASE_FEATURES, 0.25).all(),
        "1.0 ATR de reste contre 0.25 x 8.0 = 2.0")
verifie("petit gain acquis, meme reste -> on tient",
        not T.demande_profit(_Faux(1.0), _etat(+1.0), "cpu",
                             N_BASE_FEATURES, 0.25).any(),
        "1.0 ATR de reste contre 0.25 x 1.0 = 0.25 — il reste tout a faire")

# LE GARDE D'ARMEMENT. A l'epoch 1 la tete est a rho +0.066 et predit
# `reste median 4.39 ATR` la ou une tete ajustee en predit 13.5 : elle
# declenchait parce qu'elle SE TROMPAIT.
verifie("une tete non entrainee ne decide RIEN",
        not T.demande_profit(_Faux(0.01, rho=0.02), _etat(+5.0), "cpu",
                             N_BASE_FEATURES, 0.25, rho_min=0.10).any(),
        "rho 0.02 sous le minimum de 0.10")
verifie("une tete entrainee decide",
        T.demande_profit(_Faux(0.01, rho=0.20), _etat(+5.0), "cpu",
                         N_BASE_FEATURES, 0.25, rho_min=0.10).all(),
        "rho 0.20 au-dessus du minimum")
verifie("sans rho du tout, elle est muette",
        not T.demande_profit(_FauxSansRho(0.01), _etat(+5.0), "cpu",
                             N_BASE_FEATURES, 0.25, rho_min=0.10).any(),
        "une politique qui n'a jamais vu la passe supervisee")

# ELLE NE FERME QU'UN GAIN, ET CETTE LIGNE A COUTE UN RUN.
#
# La premiere version n'avait pas la condition sur le latent. Resultat en
# une epoch : tenue[G 1/11 P 1/10 x1.1] — le rapport valait 19.6 en
# simulation — ENV C 27.9 %, PF 0.46, sommet -0.420 contre -0.154 au
# hasard. A l'entree le latent est NEGATIF (on paie le spread) et une tete
# encore incertaine predit volontiers moins d'un ATR de reste : la regle
# fermait tout de suite, sur une perte.
verifie("elle NE ferme PAS une position en perte",
        not T.demande_profit(_Faux(0.01), _etat(-2.0), "cpu",
                             N_BASE_FEATURES, 0.25).any(),
        "reste 0.01 ATR — mais la position perd, et 0.25 x (-2) est negatif")
verifie("ni une position a l'equilibre exact",
        not T.demande_profit(_Faux(0.01), _etat(0.0), "cpu",
                             N_BASE_FEATURES, 0.25).any(),
        "latent nul : il n'y a aucun profit a prendre")
verifie("un gain, meme minuscule, suffit",
        T.demande_profit(_Faux(0.001), _etat(+0.01), "cpu",
                         N_BASE_FEATURES, 0.25).all(),
        "le cout de sortie sera paye de toute facon, il n'entre pas au seuil")
verifie("la condition est ECRITE dans le code",
        "latent > 0.0" in src,
        "une prise de profit qui ferme une perte est un second mauvais stop")
verifie("la regle est RELATIVE dans le code",
        "float(coupe) * latent" in src,
        "une constante en ATR ne suit pas le marche")
verifie("la passe supervisee pose le rho",
        "policy._rho_profit = " in src,
        "sans quoi le garde laisserait passer une tete non entrainee")

# ============================================================
# L'ECHELLE EST REDRESSEE, ET CE N'EST PAS UN DETAIL.
#
# `rho` mesure l'ORDRE, pas la MAGNITUDE. Le 2026-09-22 la tete atteint
# rho +0.150 — exactement la valeur visee hors ligne — en predisant
# `reste median 3.57 ATR` la ou la cible en vaut 13 : un classement juste,
# une echelle quatre fois trop basse. La regle compare `reste` a
# `coupe x latent`, donc elle est sensible a cette erreur d'un seul cote :
# calibree pour mordre sur 0.4 % des trades, elle en declenchait 46.9 %.
verifie("sans redressement, une tete sous-estimante ferme",
        T.demande_profit(_Faux(1.0, echelle=1.0), _etat(+8.0), "cpu",
                         N_BASE_FEATURES, 0.25).all(),
        "1.00 ATR annonce contre un seuil de 0.25 x 8.0 = 2.0")
verifie("avec le redressement, elle tient",
        not T.demande_profit(_Faux(1.0, echelle=4.0), _etat(+8.0), "cpu",
                             N_BASE_FEATURES, 0.25).any(),
        "la meme prediction redressee vaut 4.00 ATR, au-dessus de 2.0")
verifie("la passe supervisee pose l'echelle",
        "policy._echelle_profit = " in src,
        "sinon le redressement vaudrait 1.0 pour toujours")
verifie("la decision l'applique",
        '_echelle_profit", 1.0)' in src,
        "mesuree d'un cote, appliquee de l'autre")

print("\n4. CE QUE `demande_cloture` NE PEUT PAS FAIRE")
# ============================================================


class _FauxRisque:
    def __init__(self, r):
        self._z = float(np.log(np.expm1(max(float(r), 1e-6))))

    def cloture(self, x):
        return torch.full((x.shape[0], 1), self._z)


_gain = _etat(+4.0)
verifie("la cloture ne ferme JAMAIS une position en gain",
        not T.demande_cloture(_FauxRisque(0.001), _gain, "cpu",
                              N_BASE_FEATURES, coupe=0.0, marge=0.0).any(),
        "le seuil a un plancher a 0.05 et exige un latent negatif")
verifie("la tete de profit, elle, le peut",
        T.demande_profit(_Faux(0.1), _gain, "cpu",
                         N_BASE_FEATURES, 0.25).all(),
        "c'est le seul organe capable de solder un gagnant")

# ============================================================
print("\n5. LA CONFIGURATION NE LA LAISSE PAS ORNEMENTALE")
# ============================================================
c = T.PPOConfig()
verifie("la prise de profit est active",
        float(c.coupe_profit) > 0.0,
        "coupe_profit %.2f x latent — a zero la tete ne deciderait rien"
        % c.coupe_profit)
verifie("elle reste sous le seuil ou la mesure la dit destructrice",
        float(c.coupe_profit) <= 0.5,
        "au-dela de 0.5 le balayage donne avantage -2.55 puis -22.06")
verifie("le garde d'armement est pose",
        float(getattr(c, "rho_profit_min", 0.0)) > 0.0,
        "rho_profit_min %.2f" % getattr(c, "rho_profit_min", 0.0))
verifie("sa passe supervisee tourne",
        int(c.pas_profit_par_epoch) > 0,
        "%d pas par epoch" % c.pas_profit_par_epoch)
verifie("le stop a ete desserre",
        float(c.coupe_risque) >= 1.0,
        "coupe_risque %.2f — a 0.25 il coupait 76 %% des trades"
        % c.coupe_risque)

# ============================================================
print("\n6. LA CIBLE EST UNE AMPLITUDE, JAMAIS UN SIGNE")
# ============================================================
import pandas as pd

import cibles_m1 as C

_n = 4000
_rng = np.random.default_rng(3)
_px = 100.0 * np.exp(np.cumsum(_rng.normal(0, 0.001, _n)))
_df = pd.DataFrame({"close": _px, "open": _px, "high": _px * 1.001,
                    "low": _px * 0.999,
                    "atr_14": np.full(_n, 0.1)})
_idx = np.arange(100, _n - 600, 7)
_cp = C.cible_profit(_df, _idx, np.ones(len(_idx)), np.full(len(_idx), 0.1),
                     horizon=480)
verifie("elle est positive ou nulle", bool((_cp >= 0).all()),
        "min %.3f" % _cp.min())
verifie("elle est finie partout", bool(np.isfinite(_cp).all()))
verifie("elle n'est pas constante", float(_cp.std()) > 1e-6,
        "ecart-type %.3f ATR" % _cp.std())
_cs = C.cible_profit(_df, _idx, -np.ones(len(_idx)),
                     np.full(len(_idx), 0.1), horizon=480)
verifie("le cote VENTE est le miroir, pas une copie",
        bool((_cs >= 0).all()) and not np.allclose(_cs, _cp),
        "une vente gagne quand le prix baisse")

print("\n%d/%d OK" % (_ok, _ok + _ko))
if _ko:
    print("\nLA PRISE DE PROFIT EST A NOUVEAU DEBRANCHEE. Sans elle, le seul")
    print("moyen de sortir d'un gagnant est le plafond de detention — et la")
    print("mesure dit que le sommet du rebond varie de 49 a 2 770 minutes.")
sys.exit(1 if _ko else 0)
