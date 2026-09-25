# -*- coding: utf-8 -*-
"""LE CRITERE DE SAUVEGARDE JUGE LA STRATEGIE QUE LE MODELE JOUE.

CE QUI ETAIT FAUX, mesure le 2026-09-25. Le run precedent a tourne 138
epochs et n'a retenu AUCUN modele. En cherchant pourquoi, deux defauts :

  LE CRITERE NE JOUAIT PAS LA SORTIE. `sommet` et `score_retenue` lisaient
  le rendement d'une position tenue EXACTEMENT `horizon_cloture` barres.
  La politique de sortie PPO n'entrait nulle part dans la decision de
  sauvegarde : elle aurait pu apprendre des sorties parfaites sans que le
  checkpoint le voie.

  LA VALIDATION NE DECIDAIT PAS AU RYTHME DE L'ENTRAINEMENT. Le rollout
  consultait la politique tous les 15 barres, la validation et le test A
  CHAQUE BARRE. En argmax, quinze fois plus d'occasions de fermer que ce
  qu'elle avait appris.

LA PREUVE CENTRALE est la section 2 : une politique qui tient toujours,
bornee a l'horizon, doit rendre l'ancien critere AU BIT PRES. Si ce test
casse, quelque chose d'autre que la barre de sortie a change.

    python test_critere_sortie.py
"""

import os
import sys
import warnings

sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

import training as T
from saint_core import FERMER, TENIR

ok = ko = 0


def verifie(nom, cond, detail=""):
    global ok, ko
    ok, ko = (ok + 1, ko) if cond else (ok, ko + 1)
    print("  %s %-54s %s" % ("ok   " if cond else "ECHEC", nom, detail))


# ---------------------------------------------------------------- cadence
print("\n1. LA CADENCE EST EXACTE")
dep = np.zeros(1, np.int64)
fr = np.array([True])
quand = []
for t in range(1, 50):
    dec, ec = T.cadence_sortie([0], fr, dep, 15)
    if dec:
        quand.append(t)
        fr[0] = False
verifie("decisions aux barres 1, 16, 31, 46", quand == [1, 16, 31, 46],
        str(quand) + "  (c'etait 1, 17, 33 : cadence 16)")

# ---------------------------------------------------------------- donnees
cfg = T.PPOConfig()
cfg.timeframe_entrainement = "M1"
d = pd.read_pickle("data_cache_BTCUSD_M1.pkl").iloc[:200_000].reset_index(drop=True)


class _Data:
    """Juste ce que le simulateur lit : le cadre et les features normalisees."""
    def __init__(self, df):
        self.df = df
        from saint_core import FEATURE_COLS
        self.features = df[list(FEATURE_COLS)].to_numpy(np.float32)


data = _Data(d)
idx = np.arange(cfg.lookback, len(d) - 400, 97)


class _Politique:
    """Une politique qui rend toujours la meme action."""
    def __init__(self, action):
        self.a = action

    def sortie(self, x):
        lg = torch.full((x.shape[0], 2), -5.0)
        lg[:, self.a] = 5.0
        return lg, torch.zeros(x.shape[0])


# ---------------------------------------------------------------- equivalence
print("\n2. TOUJOURS TENIR, BORNE A L'HORIZON = L'ANCIEN CRITERE, EXACTEMENT")
c2 = T.PPOConfig()
c2.timeframe_entrainement = "M1"
c2.tenue_max_cloture = c2.horizon_cloture
ra_n, rv_n, ten = T.rendements_sortie_ppo(_Politique(TENIR), data, idx, c2, "cpu")
ra_o, rv_o = T.rendements_du_systeme(d, idx, c2)
m = np.isfinite(ra_o) & np.isfinite(ra_n)
verifie("toutes les positions tenues jusqu'a l'horizon",
        bool((ten == c2.horizon_cloture).all()),
        "tenue %d partout" % c2.horizon_cloture)
e_a = float(np.max(np.abs(ra_n[m] - ra_o[m])))
e_v = float(np.max(np.abs(rv_n[m] - rv_o[m])))
verifie("achat : identique a l'ancien calcul", e_a < 1e-9, "ecart max %.2e" % e_a)
verifie("vente : identique a l'ancien calcul", e_v < 1e-9, "ecart max %.2e" % e_v)

# ---------------------------------------------------------------- bornes
print("\n3. LA SORTIE EST BIEN CELLE DE LA POLITIQUE")
_, _, t1 = T.rendements_sortie_ppo(_Politique(FERMER), data, idx, cfg, "cpu")
verifie("toujours FERMER -> sortie a la premiere barre",
        bool((t1 == 1).all()), "tenue 1 partout")
_, _, t2 = T.rendements_sortie_ppo(_Politique(TENIR), data, idx, cfg, "cpu")
verifie("toujours TENIR -> sortie a la borne",
        bool((t2 == cfg.tenue_max_cloture).all()),
        "tenue %d partout" % cfg.tenue_max_cloture)

print("\n4. UNE VRAIE POLITIQUE (initialisee au hasard)")
from saint_core import (N_ACTIONS, OBS_N_FEATURES, SAINTPolicySingleHead)
torch.manual_seed(0)
pol = SAINTPolicySingleHead(n_features=OBS_N_FEATURES, d_model=8, num_blocks=1,
                            heads=1, max_len=4, n_freq=2,
                            n_actions=N_ACTIONS).eval()
ra, rv, tt = T.rendements_sortie_ppo(pol, data, idx, cfg, "cpu")
verifie("rendements finis", bool(np.isfinite(ra[m]).all() and np.isfinite(rv[m]).all()))
verifie("des tenues variees, pas figees",
        len(np.unique(tt)) > 1,
        "mediane %.0f, de %d a %d" % (np.median(tt), tt.min(), tt.max()))

print("\n4b. LA SORTIE REJOUEE CONNAIT LE SENS DE LA POSITION")
# Une politique qui ne ferme QUE les shorts. Si le sens n'arrivait pas a
# la politique, les deux cotes auraient la meme tenue.


class _FermeShorts:
    def sortie(self, x):
        a = (x[:, -1] < 0).long()
        lg = torch.full((x.shape[0], 2), -5.0)
        lg[torch.arange(x.shape[0]), a] = 5.0
        return lg, torch.zeros(x.shape[0])


_, _, t3 = T.rendements_sortie_ppo(_FermeShorts(), data, idx, cfg, "cpu")
verifie("deux tenues, une par sens", t3.shape == (len(idx), 2))
verifie("le long tient jusqu'a la borne",
        bool((t3[:, 0] == cfg.tenue_max_cloture).all()))
verifie("le short ferme des la premiere barre", bool((t3[:, 1] == 1).all()))

print("\n5. LES QUATRE ENDROITS LISENT LA MEME CADENCE")
src = open("training.py", encoding="utf-8").read()
verifie("rollout, validation et test appellent `cadence_sortie`",
        src.count("cadence_sortie(") == 4, "1 definition + %d appels"
        % (src.count("cadence_sortie(") - 1))
verifie("le critere est recalcule a chaque epoch",
        "rendements_sortie_ppo(\n                policy, val_data, _rang_idx" in src)
verifie("l'ancienne cadence decalee a disparu",
        "depuis_dec[_k] - 1" not in src)

print("\n%d/%d OK" % (ok, ok + ko))
sys.exit(1 if ko else 0)
