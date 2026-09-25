# ======================================================================
# PPO + SAINTv2 — SCALPING BTCUSD M1 (SINGLE-HEAD + ACTION MASK + H1)
# Version "Loup Ω" LONG / SHORT / CLOSE
# ======================================================================

import os
import re
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# `expandable_segments` N'EXISTE PAS SOUS WINDOWS. L'allocateur CUDA le
# refuse et emet un avertissement a chaque transfert de tenseur — du rouge
# plein la fenetre, pour une option qui n'a jamais rien fait ici. On ne la
# pose donc que la ou elle est supportee.
if os.name != "nt":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# DEUX AVERTISSEMENTS PYTORCH, CONNUS ET SANS CONSEQUENCE, TUS NOMMEMENT.
#
# Ils sortaient en rouge a chaque lancement et a chaque construction de
# reseau, au milieu des chiffres qu'on vient lire — au point de faire croire
# a une panne. Ils sont filtres UN PAR UN, sur leur texte : un
# `filterwarnings("ignore")` global cacherait aussi ceux qu'on veut voir, et
# c'est precisement le genre de silence que ce depot paie cher.
#
#   enable_nested_tensor : `nn.TransformerEncoder` renonce a son chemin
#   optimise parce que nos blocs sont en pre-norm. C'est notre choix
#   d'architecture, pas un defaut, et le resultat est identique.
#
#   flash attention : la roue installee n'embarque pas le noyau flash, donc
#   `scaled_dot_product_attention` retombe sur le noyau mathematique. Plus
#   lent, exact au meme resultat, et rien ici ne peut le changer.
warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True.*")
warnings.filterwarnings("ignore", message=".*not compiled with flash attention.*")

import math
from economic_learning import downside_score
from execution_quotes import execution_quote
import copy
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import MetaTrader5 as mt5
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.distributions import Categorical

# Optimisations PyTorch
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ============================================================
# LOG HELPERS — ANSI couleurs pour les prints d'entraînement
# ============================================================

try:
    import colorama
    colorama.just_fix_windows_console()
except ImportError:
    pass


class _C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[32m"
    RED     = "\033[31m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    GREY    = "\033[90m"
    WHITE   = "\033[97m"


def _col(text: str, color: str) -> str:
    return f"{color}{text}{_C.RESET}"


def _money(x: float, width: int = 9) -> str:
    """+1234.56$ ou -123.45$ coloré selon signe."""
    s = f"{x:+.2f}$".rjust(width)
    if x > 0:
        return _col(s, _C.GREEN)
    if x < 0:
        return _col(s, _C.RED)
    return _col(s, _C.GREY)


def _pct(x: float, width: int = 5) -> str:
    return f"{x*100:>{width-1}.1f}%"


def _med(v):
    """La mediane, ou zero si personne. Le journal ne doit jamais lever."""
    import numpy as _np
    return float(_np.median(v)) if len(v) else 0.0


def _moy(v):
    import numpy as _np
    return float(_np.mean(v)) if len(v) else 0.0


def _max(v):
    import numpy as _np
    return float(_np.max(v)) if len(v) else 0.0


def _split_by_side(trades_pnl, trades_side):
    """Renvoie (pnl_long, pnl_short) avec chacun (wins, losses, total_pnl)."""
    long_w  = [p for p, s in zip(trades_pnl, trades_side) if s == 1 and p > 0]
    long_l  = [p for p, s in zip(trades_pnl, trades_side) if s == 1 and p <= 0]
    short_w = [p for p, s in zip(trades_pnl, trades_side) if s == -1 and p > 0]
    short_l = [p for p, s in zip(trades_pnl, trades_side) if s == -1 and p <= 0]
    return {
        "long":  {"wins": long_w,  "losses": long_l,  "pnl": float(sum(long_w + long_l))},
        "short": {"wins": short_w, "losses": short_l, "pnl": float(sum(short_w + short_l))},
    }

# ============================================================
# SEED GLOBAL
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

from saint_core import (
    PolitiqueEnsemble,
    N_POS_FEATURES,
    SCALPING_MAX_HOLDING,
    # LES DEUX ACTIONS DE SORTIE, importees et non recopiees. Un `1` en dur
    # a la place de `FERMER` ne leverait aucune erreur : il ferait seulement
    # fermer quand il faut tenir, si l'ordre changeait un jour.
    TENIR,
    FERMER,
    N_ACTIONS_SORTIE,
    # LES TROIS ACTIONS D'ENTREE DE PPO, dans l'ordre de l'environnement.
    ACHETER,
    VENDRE,
    ATTENDRE,
    N_ACTIONS_ENTREE,
    # LES TROIS CONSTANTES DE LA TETE DE PROFIT. Elles sont importees
    # plutot que recopiees : `entree_profit` s'en sert pour extraire les
    # quatre colonnes, et un indice recopie ici finirait par diverger de
    # `FEATURE_COLS` a la premiere colonne ajoutee.
    IDX_PROFIT_POS,
    IDX_PROFIT_MARCHE,
    N_PROFIT_FEATURES,
    # LE SENS DE LA POSITION, que le modele de sortie lit en plus.
    IDX_SENS_POS,
    N_SORTIE_FEATURES,
    COL_SENS_SORTIE,
    N_ACTIONS,
    MASK_VALUE,
    NORM_STATS_PATH,
    FEATURE_COLS,
    N_BASE_FEATURES,
    OBS_N_FEATURES,
    merge_m1_h1,
    ATR_PLANCHER_FRAC,
    charge_source_externe,
    SOURCE_EXT_NOM,
    safe_normalize,
    SeuilRang,
    N_BLOCS_DEFAUT,
    EntryDecisionPolicy,
    cotes_permises,
    decide_avec_barres,
    places_ouvrables_compte,
    score_retenue,
    rolling_decision_spec,
    SAINTPolicySingleHead,
    PatchTSTPolicy,
    build_policy,
    build_mask_from_pos_scalar,
)

# ============================================================
# CONSTANTES
# ============================================================

CONF_THRESHOLD = 0.40  # Ancien seuil ABSOLU de production. Conservé pour les
                       # checkpoints antérieurs ; remplacé par une calibration
                       # par quantile, voir SELECTIVITE ci-dessous.

# ============================================================
# SÉLECTIVITÉ — remplace le seuil absolu
# ============================================================
#
# Un seuil absolu sur p(BUY)/p(SELL) est ININTERPRÉTABLE : il suppose de
# connaître à l'avance l'échelle de conviction du modèle. Mesuré à l'epoch 6 sur
# deux configurations successives, la policy converge vers p ≈ (0.31, 0.31, 0.38)
# — HOLD est l'argmax partout, et la validation renvoyait 0 trade MÊME avec un
# seuil à zéro. Ce n'était donc pas le seuil qui mordait, mais la règle argmax.
#
# Or exiger p(BUY) > p(HOLD) est une contrainte inutile : ce qui décide de
# l'ouverture n'est pas que trader soit l'action la PLUS PROBABLE, mais que son
# espérance soit positive sur la fraction sélectionnée. Et la mesure hors
# échantillon (barrière triple, 1.9 M bougies) situe précisément l'edge dans la
# queue : winrate 35 % en moyenne, 44 % au top 10 %, 46.6 % au top 0.5 %.
#
# On raisonne donc en SÉLECTIVITÉ : « trader les q % d'instants les plus
# favorables ». Le seuil absolu correspondant est recalculé à chaque epoch comme
# le quantile (1−q) de max(p_BUY, p_SELL) observé sur les états flat. Il suit
# automatiquement l'échelle du modèle, et il est stocké avec le checkpoint pour
# que live / backtests / MQL5 appliquent exactement la même barre.
SELECTIVITE_START = 0.50   # epoch 1 : on trade la moitié des occasions (feedback)
# CE QUE COUTE D'ELARGIR, mesure le 2026-09-19 sur le checkpoint d'exec06 a
# l'epoch 7, fenetre de validation du fold 1 (3 273 occasions, 8.2 mois).
#
# ESSAYE A 16.18 %, PUIS REVENU A 5 % le meme jour, sur demande — le
# tableau reste parce qu'il repond a la question « et si on elargissait »,
# et qu'elle reviendra.
#
#     select.  occasions  trades/mois  gain/occ  avantage  total
#         5%        164         20.1    +1.798    +1.192   +196
#        10%        327         40.1    +1.611    +1.005   +329
#     16.18%        530         65.0    ~+1.35    ~+0.74   ~+392
#        20%        655         80.3    +1.181    +0.575   +376
#        50%      1,636        200.6    +0.897    +0.291   +475
#       100%      3,273        401.3    +0.606    +0.000      +0
#
# TROIS FOIS PLUS DE TRADES POUR 38 % D'AVANTAGE EN MOINS par occasion, donc
# environ le DOUBLE d'avantage total sur la fenetre. La ligne a 100 % dit
# pourquoi la question a une reponse : prendre toutes les occasions, c'est
# gagner la moyenne de toutes les occasions, donc zero avantage. Elargir
# n'est gratuit nulle part.
#
# DEUX RESERVES, ET ELLES NE SONT PAS LEVEES.
#
#   La colonne « total » additionne des R en supposant les occasions
#   INDEPENDANTES. Ce depot a deja vu cette somme mentir : l'optimum
#   4xATR/6R qu'elle designait s'est effondre des qu'on l'a rejoue en equite
#   continue. A 65 trades par mois les positions se chevauchent bien plus
#   qu'a 20, donc la correlation mord davantage. A verifier en equite.
#
#   La courbe depend de la QUALITE du classement. Rejouee le meme jour sur
#   un checkpoint non entraine, elle devient PLATE et non monotone —
#   +0.336 a 5 %, -0.001 a 10 %, +0.096 a 16 % — c'est-a-dire qu'elle ne
#   dit plus rien. Celle du tableau vient d'un modele a rhoAux +0.16. Toute
#   decision de selectivite prise sur un modele qui ne classe pas encore est
#   prise sur du bruit.
SELECTIVITE_FIN = 0.05     # régime : les 5 % les plus favorables
# 40 -> 10, LE 2026-09-16 : la rampe passait tout le run dans la zone perdante.
#
# COURBE MESUREE sur la validation, 12 phases, seuils calibres sur le train.
# E[R] des occasions retenues selon la selectivite :
#
#     select.   trades/ph      E[R]   err-type   en points de winrate
#        5 %           5    +0.5591     0.4430        +18.6
#       10 %           9    +0.2023     0.1508         +6.7
#       15 %          15    +0.0226     0.1243         +0.8
#       20 %          21    +0.0598     0.0929         +2.0
#       30 %          50    -0.1221     0.0382         -4.1   <- t = -3.2
#       50 %         117    -0.0992     0.0287         -3.3
#      100 %         195    -0.0877     0.0167         -2.9
#
# L'avantage est CONCENTRE au sommet et meurt vers 15 %. Au-dela de 30 % il
# est significativement NEGATIF : les occasions peu convaincantes ne sont pas
# neutres, elles perdent.
#
# CE QUE LA RAMPE FAISAIT. Partant de 50 % et arrivant a 5 % en QUARANTE
# epochs — la duree totale du run — la politique passait la quasi-totalite de
# son apprentissage entre 50 et 15 %, c'est-a-dire dans la zone mesuree
# perdante, et n'atteignait le regime utile qu'a la toute derniere epoch.
#
# PIRE : LA MOYENNE DES POIDS PORTE SUR LES DIX DERNIERES EPOCHS, donc sur la
# plage 17 % -> 5 %, dont l'esperance mesuree tourne autour de zero. Le modele
# DEPLOYE etait la moyenne de politiques entrainees dans un regime neutre,
# jamais dans celui ou l'avantage se trouve.
#
# A 10 epochs, le regime utile est atteint a l'epoch 11 et les trente
# suivantes — les dix moyennees comprises — s'y deroulent entierement. La
# rampe garde sa raison d'etre, donner du retour au critic au demarrage, mais
# cesse d'occuper le run.
SELECTIVITE_RAMP_EPOCHS = 10


def selective_threshold(samples, fraction: float) -> float:
    """Seuil >= conservateur, y compris si les convictions sont identiques."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("Convictions vides ou non finies")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("La fraction doit etre dans ]0, 1]")
    threshold = float(np.quantile(samples, 1.0 - fraction))
    if np.mean(samples >= threshold) > fraction + 1.0 / len(samples):
        threshold = float(np.nextafter(threshold, np.inf))
    return threshold


def rollout_action_probabilities(model_probs, force_prob, threshold, side):
    """Distribution effective, apres curriculum et remplacement BUY/SELL -> HOLD."""
    raw = np.asarray(model_probs, dtype=np.float64)
    effective = raw / raw.sum()
    for action in (0, 1):
        if raw[action] < threshold:
            effective[2] += effective[action]
            effective[action] = 0.0
    forced = np.array([1., 0., 0.] if side == 'long' else
                      [0., 1., 0.] if side == 'short' else [.5, .5, 0.])
    return (1.0-force_prob) * effective + force_prob * forced


def perte_top_k(score, y, masque_cote, s_sel, tau_frac=0.25):
    """Le rendement moyen des `s_sel` % les mieux notes, rendu DERIVABLE.

    C'EST LE CRITERE DE SELECTION LUI-MEME. `gain_top` — le `sommet` du
    journal — mesure exactement cela : ce que rapportent les occasions que le
    checkpoint mettrait en position. C'est lui qui decide quel modele part au
    fold suivant, donc lui qu'il faut optimiser.

    POURQUOI LA MOINDRE CARRE NE SUFFIT PAS, mesure sur exec35. Elle minimise
    l'erreur sur TOUTES les occasions, donc elle est dominee par le gros de la
    distribution. Le deploiement, lui, ne garde que les 5 % du haut. Les deux
    grandeurs ont diverge, et la mesure le montre sur la serie LISSEE sur cinq
    epochs, dont l'ecart-type vaut 0.120 :

        AuxL (moindres carres)   0.9595 -> 0.9269 -> 0.9074 -> 0.9112  plafonne
        sommet lisse             1.126  -> ... -> 0.864   = 2.2 ecarts-types

    Le reseau devenait meilleur sur l'occasion moyenne et moins bon sur celles
    qui decident. RESERVE DE LECTURE : deux baisses precedentes de `sommet`
    m'avaient trompe parce que je les lisais sur la serie BRUTE, dont
    l'ecart-type vaut 0.31 — a ce niveau, trois points ne disent rien. C'est
    la serie lissee qui tranche, et elle seule.

    LE SEUIL EST DETACHE, LA PONDERATION NE L'EST PAS. `topk` donne le k-ieme
    meilleur score du lot ; la sigmoide autour de ce seuil rend une selection
    douce dont le gradient remonte aux scores. A temperature nulle on
    retrouverait exactement la moyenne des k retenus.

    LA TEMPERATURE SUIT L'ECHELLE DES SCORES — une fraction de leur ecart-type
    — et non une constante : rien ne fixe l'unite d'un score, et un tau absolu
    deviendrait soit un seuil dur, soit une moyenne uniforme des que le reseau
    change d'echelle.

    VIVIER UNIQUE ENTRE LES COTES PERMIS. La regle deployee classe
    max(achat, vente) et retient les 5 % des OCCASIONS ; prendre le sommet de
    chaque colonne separement garantirait un quota a chaque sens. En
    long-only la question ne se pose pas, mais elle se reposera.
    """
    m = (masque_cote > 0)
    sc = score[:, m]
    yy = y[:, m]
    if sc.numel() == 0:
        return score.sum() * 0.0
    sc = sc.reshape(-1)
    yy = yy.reshape(-1)
    k = max(int(round(sc.numel() * float(s_sel))), 1)
    seuil = torch.topk(sc.detach(), k).values[-1]
    tau = (tau_frac * sc.detach().std()).clamp_min(1e-6)
    w = torch.sigmoid((sc - seuil) / tau)
    return -((w * yy).sum() / w.sum().clamp_min(1e-6))


def comprime(x: float, borne: float) -> float:
    """Comprime sans jamais aplatir : `borne * signe(x) * log1p(|x|/borne)`.

    POURQUOI PAS UN ECRETAGE. `clip(r, -3.5, 3.5)` rend IDENTIQUES toutes les
    barres au-dela de la borne. Une barre qui perd 91 % de l'equite et une qui
    en perd 35 % recoivent exactement la meme recompense : -3.5. Le modele ne
    peut donc pas apprendre que la premiere est pire — l'information est
    detruite avant d'atteindre le gradient, et c'est precisement la queue
    qu'on veut lui enseigner.

    CETTE FORME EST STRICTEMENT CROISSANTE, donc elle conserve l'ORDRE : plus
    mauvais reste plus mauvais, indefiniment. Elle borne la magnitude sans
    borner la distinction. Au voisinage de zero elle vaut x (elle ne deforme
    pas le regime ordinaire) ; loin, elle croit comme un logarithme.

    L'ECRETAGE RESTE, mais tres au large, comme garde-fou contre une valeur
    aberrante — un gap, un ATR degenere — et non comme rabot sur le resultat.
    """
    if not np.isfinite(x):
        return 0.0
    b = max(float(borne), 1e-9)
    return float(b * math.copysign(math.log1p(abs(x) / b), x))


def retient_checkpoint(gain_top, score_rang, val_num_trades, val_ruine,
                       best_metric, min_trades, gain_tous=None,
                       score_retenue=None, creux=None, marge_hasard=0.0):
    """Ce checkpoint remplace-t-il le meilleur ? Rend (oui, raison).

    UNE FONCTION PLUTOT QU'UN `if` DANS LA BOUCLE, parce que c'est la regle
    qui decide ce qui part au fold suivant et donc ce qui sera deploye. Elle
    etait ecrite en ligne et n'avait jamais ete exercee : aucune epoch
    n'avait encore rempli la condition de refus, donc personne ne savait si
    elle tirait.

    QUATRE CONDITIONS, toutes necessaires.

      ASSEZ DE TRADES. Un resultat sur trois trades ne dit rien.

      LE SOMMET BAT LE HASARD SUR LES MEMES OCCASIONS. C'est le garde
      contre le tirage : un sommet rentable qui ne fait pas mieux que
      prendre tout n'a rien trie, il a profite d'une fenetre porteuse.

      CE GARDE ETAIT `rho > 0`, ET IL A COUTE UN RUN ENTIER. Mesure sur
      exec31 (2026-09-20) : 263 epochs, 263 refus pour "classement", UN SEUL
      checkpoint retenu. Les folds 1 et 2 n'en ont produit aucun. `rhoAux`
      etait negatif en permanence — -0.055, -0.145, -0.037 selon le fold —
      et le portillon lui obeissait, pendant que le fold 3 triait
      reellement : sommet +0.95 a +1.10 R contre +0.673 au hasard, stable
      sur dix epochs consecutives.

      POURQUOI rho ET LE SOMMET SE SEPARENT. Le rho pese TOUTES les
      occasions a egalite ; le deploiement ne regarde que les 5 % du haut.
      Une tete qui ordonne parfaitement le sommet et au hasard le reste a
      rho nul, et c'est exactement la tete qu'on veut. Les deux sens de
      l'erreur ont ete observes :

          exec23 ep.5   rho +0.0720  sommet +1.401 R  contre +1.499 au hasard
          exec26 ep.1   rho -0.0101  sommet +2.255 R  contre +1.229 au hasard

      La premiere "classe nettement" et perd contre le hasard ; la seconde
      "n'ordonne rien" et le bat de +1.026 R.

      `score_rang` reste calcule et affiche : il diagnostique, il ne decide
      plus.

      LE SCORE NET BAT LE MEILLEUR CONNU. C'est le critere proprement
      dit, et il a change le 2026-09-20 : il valait `gain_top`, le
      rendement moyen en unites de risque des occasions que le checkpoint
      mettrait en position.

      CE QUE `gain_top` NE POUVAIT PAS VOIR, ET POURQUOI C'ETAIT GRAVE.
      Il note la QUALITE DU TRI, occasion par occasion, et rien d'autre.
      Il ignore l'ORDRE dans lequel elles arrivent, donc le creux ; il
      ignore la TAILLE misee sur chacune, donc tout le travail de la tete
      de budget. Or c'est desormais la seule tete que PPO entraine, et on
      vient de lui donner un palier 0 % — un levier d'abstention explicite.
      Un modele qui apprendrait parfaitement a ne rien miser aux mauvais
      moments aurait produit exactement le meme `gain_top` qu'un modele qui
      mise pareil partout. Le critere de selection etait aveugle a la seule
      chose qu'on venait de lui apprendre.

      CE QU'ON PREND A LA PLACE. Sur la MEME grille d'occasions, en ordre
      CHRONOLOGIQUE, on reconstitue ce que le compte aurait vecu :

          r_i    = budget_i x R_i        pour les occasions retenues, 0 sinon
          equite = somme cumulee des r_i
          gain   = equite finale
          creux  = plus grand recul de cette equite, depuis son sommet

      `budget_i` est le palier que la tete de budget prendrait a cet
      instant — par ARGMAX, la regle deployee — donc r_i se lit en fraction
      de compte : un budget de 3 % sur une occasion a +2 R rapporte 6 %.

          critere = gain - creux

      UN MODELE QUI S'ABSTIENT AU BON MOMENT EST PAYE DEUX FOIS : il retire
      un r_i negatif, ce qui monte le gain ET baisse le creux. Un modele qui
      mise gros au mauvais moment est puni deux fois. `gain_top` ne
      distinguait ni l'un ni l'autre.

      POURQUOI PAS LE CREUX DU SIMULATEUR, qui existe pourtant deja
      (`val_max_dd`). Il porte sur ~55 trades d'un seul chemin, avec la
      variance que le simulateur ajoute par les emplacements et le budget :
      c'est precisement pour cela que le PnL de validation avait ete ecarte
      comme critere, et le creux qui en sort a la meme faiblesse. Le creux
      de grille, lui, se lit sur ~21 000 points, soit ~930 occasions
      independantes.

      CE QUE CETTE MESURE APPROXIME. Les occasions sont sommees comme si
      elles se succedaient sans se chevaucher et sans composition. C'est
      faux au sens strict — plusieurs positions coexistent — mais c'est
      l'approximation que `gain_top` faisait deja en les jugeant une a une,
      et celle-ci lui ajoute l'ordre et la taille. Elle ne remplace pas la
      mesure du simulateur : `val_max_dd` reste au journal.

      LE SOMMET BAT TOUJOURS LE HASARD : la condition ci-dessus n'a pas
      bouge, elle garde le tri honnete pendant que le score net juge la
      rentabilite.

      LE COMPTE N'A PAS ETE DETRUIT. Aucun episode de validation ne doit
      s'etre termine par une fin que le courtier aurait imposee — appel de
      marge, lot minimum infinancable, equite sous zero. Sans cette
      condition, un modele au classement excellent qui vide le compte en
      ouvrant soixante positions correlees serait retenu, et transmis au
      fold suivant. `gain_top` ne peut pas le voir : il mesure les occasions
      UNE A UNE et ne sait rien du nombre tenu simultanement.

      CE CRITERE ETAIT UN POURCENTAGE, et il ne pouvait plus l'etre. Il
      refusait tout checkpoint dont un episode depassait 40 % de creux —
      ce qui etait le seuil qui TUAIT l'episode, donc la condition se
      confondait avec « un episode est mort ». Depuis que l'episode va au
      bout de sa tranche, un creux de 60 % peut etre suivi d'une remontee :
      garder 40 % refuserait presque tout, et refuserait sur un chiffre
      qu'aucun courtier n'applique. Mesure du 2026-09-19 : un tiers des
      episodes tues a 40 % auraient fini AU-DESSUS du capital de depart.

    La raison rendue n'est pas decorative : elle est affichee quand un
    candidat est ecarte, sans quoi on regarde quatre-vingt-dix epochs en
    croyant qu'un meilleur modele est garde alors que rien ne l'est.
    """
    if val_num_trades < min_trades:
        return False, f"seulement {val_num_trades} trades de validation"
    if gain_tous is not None and np.isfinite(gain_tous):
        # LE PORTILLON DU HASARD EXIGE UN ECART SIGNIFICATIF, PAS UN ECART.
        #
        # CE QU'IL LAISSAIT PASSER, mesure sur exec69 fold 2 : le sommet a
        # franchi le hasard de +0.020 R a l'epoch 18 et le checkpoint a ete
        # retenu. Or l'erreur-type sur cette moyenne vaut ~0.24 R — les
        # occasions du haut se CHEVAUCHENT, donc il y en a ~32 independantes
        # derriere les 1 074 comptees. Un ecart de 0.020 R vaut 0.08 sigma :
        # le portillon cense proteger du tirage venait d'en laisser passer un.
        #
        # `marge_hasard` est calculee par l'appelant, qui a les rendements
        # sous la main. A zero — anciens appels et cas de test — la regle
        # redevient exactement celle d'avant.
        _m = float(marge_hasard) if np.isfinite(marge_hasard) else 0.0
        if not np.isfinite(gain_top) or gain_top <= gain_tous + _m:
            _dit = "" if _m <= 0 else f" + {_m:.3f} de marge"
            return False, (f"sommet {gain_top:+.3f} R ne bat pas le hasard "
                           f"{gain_tous:+.3f} R{_dit} sur les memes occasions")
    if not np.isfinite(gain_top):
        return False, "sommet non mesurable"
    # LE SCORE COMPARE AU RECORD EST LE SCORE NET quand il est fourni.
    # Le repli sur `gain_top` garde les anciens appels — et les cas de test
    # historiques — exacts : sans budget mesure, il n'y a pas de creux a
    # soustraire, et le critere se reduit a ce qu'il etait.
    _sc = gain_top if score_retenue is None else score_retenue
    if not np.isfinite(_sc):
        return False, "score net non mesurable"
    # UN MODELE PERDANT N'EST JAMAIS RETENU, MEME S'IL EST LE MOINS MAUVAIS.
    #
    # LE DEFAUT QUE CECI CORRIGE, mesure sur exec69 fold 2. `best_metric`
    # part a -1e9 et la regle ne demandait que `_sc > best_metric` : elle
    # repondait « lequel est le meilleur » et jamais « celui-ci vaut-il la
    # peine d'etre joue ». Le fold 2 a donc retenu HUIT checkpoints de suite,
    # tous a score negatif :
    #
    #     -0.215 -> -0.166 -> -0.146 -> -0.124 -> -0.121 -> -0.110
    #            -> -0.099 -> -0.083
    #
    # Sur 78 epochs, PAS UNE SEULE n'a eu un score net positif. Le modele
    # « meilleur du fold » etait donc une configuration perdante — et c'est
    # elle qui chainait vers le fold suivant et qu'on aurait deployee.
    #
    # CE QUE LE SCORE NEGATIF VEUT DIRE, concretement : sur les occasions
    # qu'il choisit et mise, le rendement moyen est INFERIEUR a la taille
    # typique de ses pertes. Il n'y a pas de reglage de taille qui rende ca
    # rentable.
    #
    # LE COUT ASSUME : un fold peut ne retenir AUCUN modele. C'est la bonne
    # reponse — « ce fold n'a rien produit » est une information, « voici le
    # moins mauvais perdant » n'en est pas une. Le depot applique deja ce
    # principe : un score non mesurable ferme le portillon au lieu de
    # l'ouvrir, deux lignes plus haut.
    if _sc <= 0.0:
        return False, (f"score net {_sc:+.4f} : configuration PERDANTE, "
                       f"le rendement des occasions retenues ne couvre pas "
                       f"la taille de ses pertes")
    if _sc <= best_metric:
        # `baisse`, ET EN R — PAS UN POURCENTAGE.
        #
        # Ce champ affichait `100 x creux` suivi d'un signe %, un reliquat de
        # l'epoque ou ces grandeurs etaient des fractions de compte. Depuis
        # que le critere est normalise par le risque mise, elles sont en R
        # PAR POSITION. Une baisse de 0.946 R s'affichait donc « creux
        # 94.6 % », ce qui se lit comme un compte presque efface — alors que
        # le creux de validation de cette epoch valait 34.5 %.
        #
        # Un chiffre faux dans un message de refus est pire qu'un champ
        # absent : il fait conclure a une catastrophe la ou il n'y en a pas.
        _d = ("" if creux is None or not np.isfinite(creux)
              else f" (baisse {creux:.3f} R/position)")
        return False, (f"score net {_sc:+.4f} sous le record "
                       f"{best_metric:+.4f}{_d}")
    if val_ruine:
        return False, (f"compte detruit sur {val_ruine} episode(s) de "
                       f"validation")
    return True, ""


def selectivite_for_epoch(epoch: int) -> float:
    """Fraction d'occasions tradées à l'epoch donnée (décroissance linéaire).

    Démarrer serré priverait l'agent de tout retour d'expérience ; finir large
    le ferait trader hors de sa zone de compétence.
    """
    t = min(max(epoch - 1, 0) / float(SELECTIVITE_RAMP_EPOCHS), 1.0)
    return SELECTIVITE_START + (SELECTIVITE_FIN - SELECTIVITE_START) * t

# Rampe du seuil pendant l'entraînement.
#
# Sur 3 actions, une policy fraîchement initialisée est quasi uniforme :
# p ≈ 1/3 = 0.3333 pour chaque action (mesuré : Hflat = 1.098 sur un max de
# 1.0986). Appliquer 0.40 dès le départ convertissait donc CHAQUE BUY/SELL en
# HOLD — d'où `val_trades = 0` sur les 148 epochs du log d'origine, et un
# Sortino30 bloqué à 0 qui empêchait toute sélection de "best model".
# Le train tradait quand même, mais uniquement via les ouvertures forcées du
# curriculum, dont la probabilité tombe à 0.05 vers l'epoch 36.
#
# Cette rampe absolue est REMPLACÉE par la calibration par quantile décrite
# plus haut : partir de 0.30 quand la policy uniforme vaut 0.3333 ne laissait que
# 0.028 de marge, que la première mise à jour de l'actor consommait aussitôt.

# Modèles pré-entraînés pour le mode CLOSE
BEST_MODEL_LONG_PATH = "best_saintv2_loup_long_wf1_long_wf1.pth"
BEST_MODEL_SHORT_PATH = "best_saintv2_loup_short_wf1_short_wf1.pth"


# ============================================================
# CONFIG
# ============================================================

@dataclass
class PPOConfig:
    # Données
    # L'OR SEUL. Le Bitcoin est ecarte de l'entrainement : sur les memes
    # dates et la meme geometrie, l'or rend +16.5 a +20.1 R par an contre
    # +12.5 pour le meilleur reglage du BTC, avec un avantage par trade a 3.9
    # a 6.0 ecarts-types contre ~2.0. Son spread est trois fois plus etroit.
    #
    # Sa geometrie lui est propre et ne se copie pas de l'autre : son ATR
    # relatif vaut 6.6 points de base contre 15.9, donc un stop de 6xATR y
    # ferait 40 bps la ou il en fait 95 sur le BTC. `instruments.py` porte les
    # valeurs ; elles sont appliquees ci-dessous.
    # RETOUR AU BTC LE 2026-09-21, ET C'EST LE FLUX D'ORDRES QUI DECIDE.
    #
    # L'argument ci-dessus reste VRAI sur ses propres termes : l'or a un
    # spread trois fois plus etroit, et sur les memes colonnes de PRIX il
    # rendait davantage. Il etait simplement incomplet — il comparait deux
    # instruments sur un jeu de colonnes dont on a mesure, le 2026-09-21,
    # qu'il ne porte AUCUNE direction :
    #
    #   les 266 colonnes une par une   0 au-dessus du plancher, 1 a 480 min
    #   une combinaison (ridge)        aucun horizon au-dessus de son plancher
    #   le modele entraine             +0.004 bps a 0.0 ecart-type
    #
    # UN AVANTAGE NUL PERD TOUJOURS, QUEL QUE SOIT LE SPREAD. Un spread
    # etroit ne sert que s'il y a quelque chose a encaisser. C'est la
    # faute de raisonnement : on choisissait l'instrument le moins cher
    # pour jouer une strategie qui ne gagne rien.
    #
    # CE QUE LE BTC A ET QUE L'OR N'AURA JAMAIS. Binance publie ses
    # TRANSACTIONS — `nb_trades`, `taker_buy_base` — gratuitement et depuis
    # 2017. Un CFD sur l'or n'a pas de marche central, donc pas de bande de
    # transactions : son champ `last` vaut zero, mesure. Les sept colonnes
    # de microstructure ajoutees le meme jour ne portent que des
    # COTATIONS.
    #
    # CE QU'IL EN COUTE, ET IL FAUT L'ECRIRE : la friction passe de 1.68 a
    # 3.71 bps d'aller-retour (spread MT5 reel, 1.853 bps median contre
    # 0.56 sur l'or). Il faut donc capter deux fois plus pour rentrer dans
    # ses frais. Le pari est qu'un avantage REEL sur du flux vaut mieux
    # qu'un avantage NUL sur du prix bon marche.
    symbol: str = "BTCUSD"
    timeframe: int = mt5.TIMEFRAME_M1
    htf_timeframe: int = mt5.TIMEFRAME_H1
    # Fenetre temporelle (UTC). Si date_to est None -> maintenant.
    #
    # Bornee au 2022-12-15 par la disponibilite des colonnes Binance, et ce
    # n'est pas un choix : avant cette date, taker_ratio manque 97 jours et
    # ls_ratio_top ~260 jours sur 2022. Combler par ffill fabriquerait une
    # constante et apprendrait au modele un regime inexistant. A partir du
    # 2022-12-15 la couverture est de 100.000 % sur les quatre colonnes.
    #
    # C'est 1.6x MOINS de donnees que sur l'or (2018-09-11), et sur un probleme
    # a signal faible le nombre d'echantillons est precisement le facteur
    # limitant. C'est le prix des features de positionnement.
    date_from: datetime = field(default_factory=lambda: datetime(2022, 12, 15))
    date_to: Optional[datetime] = None
    # n_bars est garde en fallback uniquement si copy_rates_range echoue.
    # BTCUSD tourne 24/7 : densite mesuree 0.971 bougie par minute calendaire
    # (1 913 925 bougies du 2022-12-14 au 2026-09-13).
    n_bars: int = 2_100_000
    # 25 bougies — MESURE, pas suppose. Le reglage a 54 etait moins bon.
    #
    # Sonde logistique sur le jeu a 30 colonnes, esperance par unite de risque
    # apres selection :
    #
    #     profondeur          AUC     E[R] top1%   t     E[R] top5%    t
    #     1 bougie          0.6094     +0.1753   +3.4     +0.0513    +2.2
    #     3 consecutives    0.6128     +0.1860   +3.6     +0.0847    +3.6
    #     5 espacees (0-12) 0.6120     +0.1860   +3.6     +0.1077    +4.6
    #     6 espacees (0-25) 0.6105     +0.1610   +3.1     +0.1147    +4.8
    #     7 espacees (0-53) 0.6094     +0.1247   +2.4     +0.0811    +3.4
    #
    # L'historique aide jusqu'a ~25 bougies puis NUIT : a 53 l'esperance
    # retombe sous celle d'une bougie unique. La colonne qui decide est celle
    # du top 5 %, puisque c'est la selectivite a laquelle le modele opere
    # desormais — elle culmine a 0-25.
    #
    # Le jeu a 30 colonnes porte deja des resumes d'historique (RSI sur 14,
    # rang de volatilite sur 1440, range_norm rapporte a sa moyenne sur 1440) :
    # empiler cinquante pas de temps en plus etait redondant, et le bruit
    # ajoute l'emportait sur l'information.
    # PatchTST permet un historique bien plus long que SAINT a cout egal :
    # l'attention porte sur ~L/8 segments au lieu de L barres, et les colonnes
    # sont traitees separement. A 96 barres avec des segments de 16 et un pas
    # de 8, cela fait 11 jetons — l'attention coute 75 fois moins qu'a 96.
    # LOOKBACK RAMENE DE 96 A 4. Mesure du 2026-09-16 : le meme modele, le
    # meme protocole, seule change la profondeur de passe empilee en entree.
    #
    #     profondeur  colonnes       PnL   trades    ecart   PF    folds +
    #              1       103   +579.40$     432   +3.9pt  1.18     3/3
    #              4       412   +992.73$     471   +6.1pt  1.30     2/3
    #             16      1648   +670.05$     500   +3.9pt  1.18     3/3
    #
    # Seize fois plus de colonnes rendent exactement le meme chiffre que la
    # ligne seule. Le passe n'apporte rien de mesurable, et c'est coherent avec
    # ce que sont ces colonnes : Tenkan resume deja 9 barres, Kijun 26, SSB 52,
    # Chikou en regarde 26 en arriere. Tout cela est DEJA dans la ligne de
    # l'instant t ; lui redonner les 95 precedentes lui redonne ce qu'il a
    # deja, sous une forme plus difficile — et lui offre 95 barres de plus pour
    # surajuster.
    # 4 -> 2 PUIS 2 -> 4, LE MEME JOUR. Le detour vaut d'etre garde, parce
    # qu'il a mis au jour un chiffre faux.
    #
    # LE LOOKBACK NE CHANGE AUCUN PARAMETRE. Verifie le 2026-09-21 :
    # 28 034 poids a 2 barres comme a 26. La position temporelle est portee
    # par RoPE dans l'attention, pas par une table indexee sur `max_len`.
    # Il ne commande donc QUE du calcul.
    #
    # LE CHIFFRE QUI A JUSTIFIE 4 -> 2 ETAIT FAUX SUR UN POINT. Il annoncait,
    # par lot de 256 :
    #
    #     lookback 2    91 ms    x0.65
    #     lookback 4   141 ms    x1.00
    #     lookback 8   910 ms    x6.47      <- incompatible avec l'architecture
    #
    # Les blocs font de l'attention AXIALE : `attn_temps` sur (B x F) suites
    # de longueur T, `attn_feat` sur (B x T) suites de longueur F. Le cout
    # total vaut donc F.T^2 + T.F^2, et avec F = 265 :
    #
    #     lookback 2      142 572   x1.0
    #     lookback 4      287 264   x2.0
    #     lookback 8      583 008   x4.1
    #     lookback 26   2 018 796  x14.2
    #
    # De 4 a 8 la theorie dit x2.0 et la mesure disait x6.47. Ce point a
    # tres probablement ete pris pendant qu'autre chose occupait la carte —
    # le piege que ce depot connait deja. Le rapport 2 contre 4, lui, tient.
    #
    # POURQUOI LE T^2 NE MORD PAS. A 2 barres, l'attention temporelle ne
    # pese que 1/131 de l'attention sur les colonnes. Elle part de si bas
    # que meme multipliee par 169 elle reste minoritaire : a 26 barres elle
    # vaut encore moins de 10 % du total. Ce qui domine est l'attention sur
    # les 265 colonnes, et celle-ci n'est que LINEAIRE en T.
    #
    # ON REVIENT DONC A 4, la valeur mesuree le 2026-09-16 sur le balayage
    # de profondeur (1 / 4 / 16 barres, meme protocole) :
    #
    #     profondeur  colonnes       PnL   trades   ecart   PF   folds +
    #              1       103   +579.40$     432  +3.9pt  1.18     3/3
    #              4       412   +992.73$     471  +6.1pt  1.30     2/3
    #             16      1648   +670.05$     500  +3.9pt  1.18     3/3
    #
    # RESERVE A GARDER EN TETE, et elle est double. Ce tableau montre 1 et
    # 16 A EGALITE : seul le point du milieu depasse, sur trois points, sans
    # barre d'erreur, avec 471 trades et 2/3 folds positifs contre 3/3 pour
    # les deux autres. Et il decrit un AUTRE SYSTEME — M5/H1, stop et
    # trailing, des trades de plusieurs heures. Le scalping M1 sans stop n'a
    # aucune raison d'avoir la meme profondeur utile. La valeur est reprise
    # parce que c'est la seule qui ait ete mesuree, pas parce que la mesure
    # vaut encore.
    #
    # CE QUE 4 REND AU MODELE par rapport a 2 : une dynamique sur quatre
    # points au lieu d'une variation entre deux. La profondeur longue reste
    # dans les COLONNES — les suffixes `_h1` et `_h4`, et `tend_vs_ma_mois`
    # — mais la dynamique recente de chacune passe par ici.
    lookback: int = 4
    # Le decoupage suit le lookback : des segments de 16 barres n'existent pas
    # dans une fenetre de 4. Segment 2 et pas 1 donnent trois jetons, donc une
    # attention qui a encore quelque chose a faire ; un segment de 4 n'en
    # donnerait qu'un seul et l'encodeur deviendrait un simple MLP.
    taille_patch: int = 2
    pas_patch: int = 1

    # "M1" ou "H1". Choisit le cache et, avec lui, l'echelle de decision.
    # M5 -> M1, LE 2026-09-21 : passage au scalping.
    #
    # CE QUE CE SEUL CHAMP COMMANDE. Il choisit le cache lu
    # (`data_cache_XAUUSD_M1.pkl`, 879 860 barres contre 491 360 en M5) et
    # donc toute la geometrie du walk-forward — qui se decoupe en FRACTIONS,
    # et s'adapte donc seul.
    #
    # CE QU'IL NE COMMANDE PAS, et qu'il a fallu regler a part : la duree
    # d'un trade, l'horizon de la tete de cloture, l'absence de stop. Le
    # timeframe dit a quelle finesse on DECIDE ; il ne dit pas combien de
    # temps on TIENT.
    #
    # L'HISTORIQUE EST PLUS COURT : 2.5 ans en M1 contre 7 en M5, parce que
    # 900 000 barres pesent deja 1.8 Go et que ce depot a tue un processus
    # sur un tampon de 2.5 Go sans message. C'est peu en calendrier, mais
    # pour du scalping ce qui compte est le nombre d'occasions
    # INDEPENDANTES — duree de marche divisee par duree d'un trade — et
    # elles passent de 1 338 par fenetre a plusieurs dizaines de milliers.
    timeframe_entrainement: str = "M1"

    # "patchtst" ou "saint". build_policy DEDUIT l'architecture du checkpoint
    # au chargement, donc ce reglage ne concerne que l'entrainement.
    # SAINT PLUTOT QUE PATCHTST, a budget de parametres EGAL.
    #
    # Ce que la nuit du 2026-09-16 a etabli, dans l'ordre :
    #   - l'axe du TEMPS ne porte rien (profondeur 16 = profondeur 1) ;
    #   - croiser les COLONNES vaut +2 points au banc (ridge +1.8, LightGBM
    #     +2.6, TabM +3.8 sur les memes fenetres) ;
    #   - PatchTST est a canaux INDEPENDANTS : il ne peut pas croiser les
    #     colonnes, par construction. C'est le defaut de Ridge transpose au
    #     modele sequentiel, et Ridge est celui qui perd ;
    #   - reduire le reseau a fait passer PPO de -1.4 a +2.2 points au test.
    #
    # SAINT etait ecarte parce qu'il coutait 19 690 ms par passe et 8.9 Go.
    # Deux choses ont change : le lookback est tombe de 96 a 4, ce qui divise
    # par 24 le produit T x F, et la largeur de sa tete est devenue reglable —
    # elle valait 256 en dur et pesait 67 % du reseau, quand l'attention entre
    # features en coute 2 %.
    #
    # A d_model 8, n_freq 4, mlp_dim 32 : 13 592 parametres contre 15 680 pour
    # PatchTST. SAINT est donc PLUS PETIT que la configuration qui vient de
    # donner +2.2, pour 15.4 ms contre 8.7. Une seule chose change entre les
    # deux runs : croiser les colonnes, ou non.
    # ------------------------------------------------------------------
    # "ensemble" : TOUS LES RESEAUX DANS LE MEME ROLLOUT, le 2026-09-16.
    #
    # Le vote doit exister PENDANT l'apprentissage, pas seulement a
    # l'evaluation. Deux facons de s'y prendre, et une seule qui tient :
    #
    #   les entrainer separement puis voter    l'action executee ne vient
    #                                          d'aucune des politiques mises a
    #                                          jour ; le rapport de PPO perd
    #                                          son sens et demande un poids
    #                                          d'importance non borne.
    #   un melange presente comme UNE          l'action est tiree de ce qui
    #   politique                              est mis a jour. PPO reste exact,
    #                                          et le gradient atteint chaque
    #                                          membre par sa part dans la
    #                                          probabilite de l'action choisie.
    #
    # C'est la seconde. `PolitiqueEnsemble` moyenne les PROBABILITES — une voix
    # par membre — et non les logits, qui laisseraient un membre tres confiant
    # ecraser les autres.
    #
    # Les membres sont opposes par construction sur la question qui separe le
    # mieux les modeles de ce depot : SAINT ne fait que croiser les colonnes,
    # PatchTST ne les croise jamais. Des erreurs correlees ne s'annulent pas ;
    # c'est la condition pour qu'un vote reduise la variance.
    #
    # UN SEUL PASSAGE SUR LES TROIS FOLDS, et non un par architecture : les
    # membres partagent le rollout, donc les memes barres, la meme
    # normalisation et les memes episodes. La comparaison entre eux est
    # APPARIEE sans effort, et le temps de calcul ne double pas.
    # ------------------------------------------------------------------
    # SAINT SEUL DEPUIS LE 2026-09-21. Le vote a deux est retire.
    #
    # LE RAISONNEMENT CI-DESSUS RESTE VRAI EN PRINCIPE — deux modeles qui
    # se trompent differemment reduisent la variance — mais il supposait
    # deux votants qui font ce que leur nom promet. Le commentaire de
    # `membres` le dit lui-meme : a lookback 4, PatchTST n'est plus un
    # decoupage en segments, c'est un MLP par colonne. On payait un second
    # passage avant par barre — le goulot mesure, la carte tournant a 28 %
    # d'utilisation avec `gpu_idle` actif — pour un votant degenere.
    #
    # ET LE VOTE N'A JAMAIS ETE MESURE ICI. Aucun chiffre de ce depot ne
    # compare `saint` seul a `saint + patchtst` sur la geometrie M1. Le
    # garder etait une hypothese, pas un resultat.
    architecture: str = "saint"
    # Membres de l'ensemble : (architecture, lookback, taille_patch, pas).
    # Meme lookback pour tous — la mesure de profondeur utile n'a rien trouve
    # au-dela de 4 barres, et un lookback different ferait varier deux choses a
    # la fois entre les votants.
    # LA GEOMETRIE DE PATCHTST N'EST PAS ARBITRAIRE, elle est CONTRAINTE.
    # `mesure_lookback_utile` n'a rien trouve au-dela de 4 barres de passe, et
    # a lookback 4 il ne reste presque aucun choix :
    #
    #     patch 2 pas 1 -> 3 patchs   <- retenu, le maximum d'information
    #     patch 2 pas 2 -> 2 patchs
    #     patch 3 pas 1 -> 2 patchs
    #     patch 4 quelconque -> 1 patch, degenere
    #
    # Il faut l'assumer : a cette profondeur PatchTST n'est plus un decoupage
    # en segments, c'est un MLP PAR COLONNE. Ce qu'on lui demande dans le vote
    # reste intact — etre le pole qui NE CROISE JAMAIS les colonnes, face a un
    # SAINT qui ne fait que les croiser — mais le nom promet davantage que ce
    # que la profondeur permet.
    # `membres` N'EST PLUS LU tant que `architecture` vaut "saint". Il est
    # garde tel quel : le jour ou le vote sera MESURE plutot que suppose,
    # il suffira de remettre "ensemble".
    membres: tuple = (("saint", 2, 1), ("patchtst", 2, 1))
    # ------------------------------------------------------------------
    # LE TROISIEME VOTANT : TabM, supervise, qui oppose son veto.
    #
    # Il ne peut pas etre un membre du melange — pas de gradient dans cette
    # boucle, et des scores qui dependent de la BARRE et non de la seule
    # observation. Il entre par le MASQUE D'ACTIONS : il ne propose rien, il
    # interdit. C'est la regle que decrit `evalue_ensemble` — un signal n'est
    # pas pris si autre chose le contredit — et c'est aussi la fonction du
    # Chikou-Span dans le systeme Ichimoku.
    #
    # Il est ajuste EN CROISE sur la fenetre d'entrainement : chaque barre
    # recoit le score d'un TabM qui ne l'a pas vue. Sans cela il serait un
    # oracle sur les donnees ou PPO apprend, les deux reseaux apprendraient a
    # lui deferer, et en production le partenaire deviendrait ordinaire.
    #
    # Le mettre a False retire le veto et laisse les deux reseaux voter seuls.
    # MESURE DU 2026-09-16, ET ELLE DIT NON. Balayage du taux de veto sur la
    # validation, 12 phases, comparaison APPARIEE de chaque phase a elle-meme
    # sans veto :
    #
    #     part   trades/ph   ecart au temoin   err-type      t
    #     0.20        24          +0.2321        0.1570    1.48   (7 phases)
    #     0.30        50          -0.0317        0.0343   -0.92
    #     0.40        90          -0.0029        0.0352   -0.08
    #     0.50       117          -0.0097        0.0281   -0.35   <- en place
    #     0.65       148          -0.0150        0.0178   -0.84
    #     0.80       169          -0.0069        0.0126   -0.55
    #
    # Aucun taux n'ajoute quoi que ce soit : |t| < 1 partout ou les douze
    # phases sont exploitables. Le seul positif, 0.20, ne tient que sur sept
    # phases et 24 trades chacune — et il est le meilleur d'un balayage de
    # huit valeurs, donc son t de 1.48 ne vaut rien.
    #
    # CE QUE LA CORRECTION DE PHASE A CHANGE. Sur une seule grille d'entrees,
    # le veto a 0.50 affichait -0.13 d'ecart et les directions REFUSEES
    # rendaient +0.19 : un filtre qui semblait marcher a l'envers. Moyenne sur
    # douze phases, l'ecart tombe a -0.0097 +/- 0.0281. L'inversion etait un
    # artefact de phase — exactement le piege mesure le 15 septembre, ou E[R]
    # allait de +0.158 a -0.071 selon la grille, pour les memes donnees.
    #
    # POURQUOI ON LE COUPE PLUTOT QUE DE LE GARDER "AU CAS OU". Six mecanismes
    # ont deja ete ecartes ici sur ce critere — attention entre groupes,
    # meta-etiquetage, taille par conviction. Garder celui-ci parce qu'il est
    # seduisant reviendrait a laisser une influence non mesuree decider a la
    # place de la mesure, et a rendre illisible le run qui teste le vote a deux.
    #
    # CE QUE LA MESURE NE DIT PAS : elle juge le veto comme filtre AUTONOME sur
    # des barrieres. Son role dans l'ensemble serait d'ecarter les directions
    # que les reseaux PPO prendraient mal — une interaction que cette sonde ne
    # voit pas. L'absence de valeur autonome n'est donc pas une preuve
    # d'inutilite ; c'est simplement la seule preuve disponible, et dans ce
    # depot la charge revient au mecanisme.
    votant_tabm: bool = False

    # ------------------------------------------------------------------
    # TETE AUXILIAIRE SUPERVISEE, le 2026-09-16.
    #
    # LE CONSTAT QUI L'A FAIT NAITRE. Sur deux runs et deux geometries,
    # l'apprentissage PPO DEGRADE la validation de facon significative —
    # -1.95 pt a -2.1 sigma sur exec24, -3.47 a -2.2 sur exec31 — sans que le
    # cote entrainement bouge (-1.5, -0.0, +0.4, +2.5, -0.0). Ce n'est donc pas
    # du sur-ajustement : le gradient pousse la politique vers un endroit qui
    # n'aide ni l'un ni l'autre.
    #
    # LE DIAGNOSTIC. On entraine la politique a AGIR, puis on s'en sert comme
    # CLASSEUR : a l'evaluation on jette 95 % de ses decisions et on garde les
    # 5 % ou sa probabilite est la plus haute. Rien dans l'objectif de PPO ne
    # recompense un bon ORDRE de ces probabilites — seulement une bonne action
    # en moyenne. Cela explique le plus vieux fait non explique du depot : une
    # regression logistique atteint 0.6271 d'AUC, aucune politique entrainee
    # n'a depasse 0.5707. Le modele supervise, lui, REGRESSE le rendement
    # realise : il est entraine exactement a classer.
    #
    # LA CIBLE est le rendement NET en unites de risque d'un achat et d'une
    # vente a cette barre, aux barrieres de l'environnement — les memes
    # etiquettes que celles de TabM, calculees par evalue_tabm_test.
    #
    # LE COEFFICIENT. A 1.0 les deux pertes pesent du meme ordre : l'erreur
    # quadratique sur une cible dans [-1, +2] vaut ~1, la perte d'acteur ~0.05.
    # C'est donc la tete auxiliaire qui mene le tronc, ce qui est VOULU : c'est
    # elle qui porte le signal dense, PPO ne voyant que ~2 000 decisions par
    # epoch. Le mettre a 0.0 retire la tete et redonne exactement le run
    # precedent — c'est le temoin de l'experience.
    aux_coef: float = 1.0
    # LA GRILLE DENSE SUR LAQUELLE LA TETE DE RANG S'ENTRAINE. Un point toutes
    # les 12 barres — une occasion par heure en M5 — tire INDEPENDAMMENT de ce
    # que la politique visite. Voir la construction de `_gr_idx` pour le
    # diagnostic qui l'impose. 0 desactive et rend le comportement d'avant.
    pas_grille_rang: int = 12
    # Pas de gradient supervises par epoch, sur cette grille.
    # 240, POUR GARDER LE MEME BUDGET DE GRADIENT QU'AVANT. La tete
    # recevait 264 pas par epoch depuis la boucle PPO (33 lots x 8 mises a
    # jour) ; ce terme est desormais coupe, et lui donner seulement 40 pas sur
    # la grille reviendrait a diviser son entrainement par 6.6 EN MEME TEMPS
    # qu'on change sa distribution. On ne saurait plus lequel des deux agit.
    # Seule la DONNEE change, pas la quantite.
    #
    # 0 DEPUIS LE 2026-09-25 : LE CLASSEMENT EST ABANDONNE, choix du
    # proprietaire. Les tetes d'achat et de vente ne regressent plus un
    # rendement a horizon fixe ; elles sont des politiques apprises par PPO.
    pas_rang_par_epoch: int = 0

    # LA TETE DE CLOTURE A SA PROPRE PASSE, ET SON PROPRE ECHANTILLON.
    #
    # Elle ne peut pas partager celui des tetes d'entree : la grille dense
    # ne contient que des etats PLATS, et « faut-il fermer ? » n'a de sens
    # que sur un etat ou une position EXISTE. Voir
    # `cibles_m1.echantillon_cloture`, qui les fabrique sans passer par le
    # simulateur — donc independamment de la politique, comme la grille.
    # LES DEUX TETES SUPERVISEES SONT DEBRANCHEES, PAS SUPPRIMEES.
    #
    # `tete_cloture` et `tete_profit` ont bien appris — rho +0.355 et
    # +0.272, au-dessus de la mesure hors ligne. Ce ne sont pas elles qui
    # ont echoue, ce sont les REGLES qui transformaient leurs predictions
    # en decisions : quatre calibrations successives, toutes fausses, et la
    # derniere laissait 74 % des GAGNANTS se faire solder par la fin
    # d'episode.
    #
    # LA SORTIE EST DESORMAIS UNE POLITIQUE PPO sur les memes quatre
    # colonnes. Les garder en parallele « comme diagnostic » a ete
    # explicitement refuse, et c'est defendable : deux organes qui
    # repondent a la meme question finissent par diverger, et on ne saurait
    # plus lequel a decide.
    #
    # A ZERO, LEUR PASSE NE TOURNE PLUS. Les tenseurs restent dans le
    # reseau pour que les points de reprise anterieurs se chargent — meme
    # convention que `actor` et `critic` depuis la suppression de la
    # direction. Remettre 240 les rallume sans rien toucher d'autre.
    pas_cloture_par_epoch: int = 0
    # LA PRISE DE PROFIT : fermer quand il ne reste plus rien a prendre.
    #
    # `tete_profit` predit COMBIEN il reste a gagner d'ici l'epuisement, en
    # ATR d'entree. On ferme quand cette amplitude tombe sous ce seuil.
    #
    # POURQUOI CET ORGANE EXISTE. Le systeme n'avait AUCUNE prise de
    # profit : `demande_cloture` exige un latent NEGATIF, donc une position
    # en gain ne pouvait jamais declencher de cloture. La seule sortie d'un
    # gagnant etait le plafond de detention — mediane ET moyenne de tenue
    # des gagnants egales au plafond, sans une exception.
    #
    # CE QUE LE PLAFOND COUTE. Le sommet du rebond tombe a 49 minutes au
    # dixieme centile et a 2 770 au quatre-vingt-dixieme, ecart-type 991.
    # Un oracle parfait ajouterait +246.84 bps par trade.
    #
    # LA VALEUR EST PRUDENTE, ET ELLE N'EST PAS MESUREE. Le reste median
    # vaut 10.7 ATR ; a 1.0 la regle ne mord que sur les positions
    # vraiment epuisees. C'est deliberement timide : toutes les prises de
    # profit a SEUIL FIXE testees le 2026-09-22 — objectif en ATR, these
    # refermee, objectif adapte au risque — DEGRADENT le resultat. Une
    # tete apprise est autre chose qu'un seuil fixe, mais rien ne le
    # prouve encore. Le journal compte ses declenchements a chaque epoch :
    # c'est ce compteur qui dira s'il faut monter le seuil.
    #
    # A ZERO, LA REGLE EST DESACTIVEE et la tete devient un ornement — le
    # meme etat que `tete_cloture` a connu pendant des jours.
    # LA PRISE DE PROFIT EST SANS ECHELLE, COMME LE STOP.
    #
    #     fermer si   reste_predit  <  coupe_profit x latent
    #
    # « Ce qu'il reste a prendre est petit devant ce que j'ai deja. » Un
    # critere d'EPUISEMENT, pas un objectif de gain.
    #
    # ELLE ETAIT UNE CONSTANTE EN ATR, ET LE RUN L'A PAYE. Avec
    # `reste < 1.00 ATR`, l'epoch 1 du 2026-09-22 rend :
    #
    #     tenue[G 3/15  P 52/88  x0.2]     WR 89.2 %   PF 0.60
    #
    # Le rapport INVERSE : on tenait les perdants six fois plus longtemps
    # que les gagnants. Les deux portes ne parlaient pas la meme langue —
    # le stop compare a un multiple du risque predit, donc il suit le
    # marche ; la prise de profit comparait a une constante, donc non.
    #
    # LE BALAYAGE QUI A FIXE 0.25, entrees validees, aucun plafond,
    # friction reelle, horizon 72 h :
    #
    #     aucune prise de profit        NET +22.25  avantage +12.13  0.0 %
    #     reste < 0.25 x latent         NET +22.89  avantage +12.79  0.4 %
    #     reste < 0.50 x latent         NET +22.90  avantage +11.11 18.9 %
    #     reste < 1.00 x latent         NET  +9.20  avantage  -2.55 53.8 %
    #     reste < 2.00 x latent         NET  -3.91  avantage -22.06 72.7 %
    #
    # ET IL FAUT LIRE CE TABLEAU POUR CE QU'IL EST. Aucune valeur ne
    # RAPPORTE quoi que ce soit de mesurable : le gain de 0.25 vaut
    # +0.64 bps sur sept mois d'evaluation, indiscernable du bruit. Ce que
    # le tableau etablit, c'est qu'au-dela de 0.5 la regle DETRUIT. On
    # prend donc la valeur la plus timide qui morde encore, pour que la
    # porte existe sans pouvoir se deregler.
    #
    # A ZERO, LA REGLE EST DESACTIVEE et la tete redevient un ornement.
    coupe_profit: float = 0.25
    # LA TETE NE DECIDE QU'UNE FOIS QU'ELLE SAIT.
    #
    # A l'epoch 1 elle est a `rho +0.066` et predit `reste median 4.39 ATR`
    # la ou une tete ajustee en predit 13.5 : elle declenchait parce
    # qu'elle se TROMPAIT. A l'epoch 2, `rho +0.099`, et `fermerait`
    # tombait deja de 10.2 % a 1.6 % — elle se corrige seule, mais elle
    # avait le droit de decider pendant qu'elle etait fausse.
    #
    # C'est le SEUL defaut que la mesure etablisse clairement.
    rho_profit_min: float = 0.10
    # LE LOYER DU TEMPS, EN ATR PAR BARRE TENUE.
    #
    # C'EST ICI QUE LE SCALPING S'ECRIT, et nulle part ailleurs. Le plafond
    # de detention a ete retire ; la duree n'est donc plus BORNEE, elle est
    # TARIFEE. Une position qui paie son loyer vit, les autres non.
    #
    # LA VALEUR EST DERIVEE, PAS CHOISIE. On veut qu'une detention de
    # trente minutes coute environ le quart d'un aller-retour :
    #
    #     aller-retour reel      4.18 bps
    #     ATR median d'une barre 5.3 bps   ->  0.79 ATR par aller-retour
    #     le quart, sur 30 barres           ->  0.0066 ATR par barre
    #
    # A ZERO, LA POLITIQUE REAPPREND A TENIR DES JOURS — c'est exactement
    # ce que le run du 2026-09-22 a fait quand plus rien ne tarifait le
    # temps. Le monter raccourcit les trades, le baisser les allonge : ce
    # reglage EST le curseur scalping/swing, et il est explicite.
    #
    # A COMPARER A LA DERIVE DU MARCHE : +0.231 bps par heure, soit
    # 0.00073 ATR par barre. Le loyer vaut NEUF FOIS la derive — tenir
    # sans raison coute donc reellement quelque chose.
    # LA SORTIE NE DECIDE PLUS A CHAQUE BARRE.
    #
    # C'EST UN PROBLEME DE FREQUENCE, PAS DE TARIF. Mesure du 2026-09-22 :
    #
    #     |delta latent| par barre   moyenne 0.650 ATR   mediane 0.483
    #     loyer du temps             0.0066 ATR par barre
    #
    # Le loyer est 98 FOIS plus petit que le bruit qu'il doit traverser.
    # PPO ne peut pas le voir, donc il n'optimise que le rebond, et tenir
    # gagne toujours. Resultat en quatre epochs : entropie effondree de
    # 0.572 a 0.192, taux de fermeture de 40 % a 10.9 %, et 16 trades tous
    # tenus jusqu'a la fin d'episode.
    #
    # EN DECIDANT TOUS LES K BARRES, le loyer d'une decision vaut K fois
    # plus pendant que le bruit ne croit qu'en RACINE de K. Le rapport
    # s'ameliore donc en racine de K :
    #
    #     K =  1   loyer 0.0066   bruit 0.650   ->  1 %
    #     K = 15   loyer 0.099    bruit 2.52    ->  4 %
    #     K = 60   loyer 0.396    bruit 5.03    ->  8 %
    #
    # C'est exactement la formulation semi-MDP que ce depot utilise deja
    # pour les entrees : une decision couvre plusieurs barres, et sa
    # recompense les accumule.
    #
    # QUINZE, ET PAS SOIXANTE. A 60 la politique ne peut plus couper une
    # perte avant une heure, ce qui contredit le scalping. A 15 elle garde
    # une granularite de quart d'heure et gagne un facteur quatre sur le
    # rapport signal/bruit.
    #
    # UNE DECISION A CHAQUE BARRE — un choix du proprietaire, 2026-09-25.
    #
    # Cette valeur etait passee a 15 pour rendre le loyer visible dans le
    # gradient : a une barre il etait 98 fois plus petit que le bruit. Le
    # contexte a change sur un point qui compte : le LOYER ZOMBIE multiplie
    # le loyer par 15 sur les positions en perte. Du cote des pertes le
    # rapport signal/bruit passe donc de 1 % a 15 %, et c'est la tete de
    # PERTE — distincte depuis le meme jour — qui recoit ce signal.
    #
    # Du cote des GAINS le loyer reste invisible a une barre : la tete de
    # gain apprend surtout du rebond lui-meme. C'est voulu — on ne veut pas
    # la pousser a fermer un gagnant.
    #
    # LA MEME CADENCE PARTOUT, et c'est `cadence_sortie` qui la porte :
    # rollout, validation, test et critere de sauvegarde.
    pas_decision_sortie: int = 1
    loyer_temps_atr: float = 0.0066
    # LOYER RENFORCE SUR LES POSITIONS EN PERTE (zombies).
    #
    # Mesure du 2026-09-22 sur trades_long_wf2 : 333 positions tenues
    # 2000+ barres, perte mediane -23 $, cout total -7772 $. Le loyer
    # ordinaire (0.0066 ATR/barre) est 98x plus petit que le bruit du
    # latent : la tete de sortie ne "sent" pas que rester en perte est
    # cher, donc elle n'apprend pas a couper.
    #
    # Quand latent < 0, on multiplie le loyer. Les gagnants gardent le
    # loyer faible → pas de pression a fermer. Les perdants paient cher
    # chaque barre → l'avantage de FERMER devient fort → la tete apprend
    # a sortir tot. Aucun stop au temps, purement un signal d'apprentissage.
    loyer_zombie_mult: float = 15.0
    # LE ZOMBIE COMMENCE A 2 ATR SOUS LE POINT MORT — choix du proprietaire,
    # 2026-09-25, sur mesure.
    #
    # Au point mort meme, une minute de bruit suffisait a declencher le
    # loyer x15 : la sortie fermait tout a la premiere barre, six epochs sur
    # six. Mais le BTC en minute s'ecarte de 7 a 8 ATR en une heure, et
    # couper vite coupe aussi les futurs gagnants. Mesure sur le train du
    # fold 1, entrees au hasard, deux sens (a 1 point pres) :
    #
    #     marge   coupees   quand (mediane)   gagnants a 60 min coupes avant
    #     0.5 ATR   87 %        3 min               74 %
    #     1.0 ATR   81 %        6 min               62 %
    #     2.0 ATR   68 %       11 min               41 %
    #     3.0 ATR   56 %       17 min               26 %
    #
    # 2 ATR : les vrais perdants restent chers en une dizaine de minutes,
    # sans fenetre de temps, et le bruit d'une minute ne fait plus fermer.
    #
    # REMESURE A 15 MINUTES, quand l'horizon y est passe (meme echantillon) :
    #
    #     marge   coupees   quand (mediane)   gagnants a 15 min coupes avant
    #     1.0 ATR   62 %        4 min               30 %
    #     1.5 ATR   51 %        5 min               18 %
    #     2.0 ATR   41 %        6 min               10 %
    #     3.0 ATR   26 %        8 min                3 %
    #
    # 2 ATR est garde : coupure deux fois plus rapide qu'a 60 minutes, et
    # quatre fois moins de gagnants tues.
    marge_zombie_atr: float = 2.0
    # LA DERIVE DU MARCHE EST RETIREE DE LA RECOMPENSE.
    #
    # Sans cela, tenir une position longue dans un BTC qui monte rapporte
    # en moyenne, et la politique apprend a ne jamais fermer — ce qu'elle
    # a fait. On ne veut pas qu'elle capte le BETA, on veut qu'elle capte
    # le TIMING.
    #
    # C'est la meme correction que l'« ecart au marche du meme mois » qui
    # sert de juge a toutes les mesures d'entree de cette session. La
    # valeur vient de la : +0.231 bps par heure sur 23 mois, soit
    # 0.00073 ATR par barre pour un ATR median de 5.3 bps.
    derive_atr_barre: float = 0.00073
    # L'ENTROPIE NE DOIT PAS S'EFFONDRER AVANT D'AVOIR APPRIS.
    #
    # `entropy_coef` vaut 0.003 pour l'acteur d'entree. Sur une politique a
    # DEUX actions, ce poids n'a pas empeche l'entropie de tomber a 0.192
    # sur 0.693 en quatre epochs — la politique est devenue deterministe
    # avant que le critique n'ait fini de se caler.
    entropie_sortie: float = 0.02
    # LE BONUS D'ENTROPIE DE LA POLITIQUE D'ENTREE. Sans lui, trois actions
    # dont deux coutent le spread s'effondrent vite sur ATTENDRE : c'est la
    # reponse la plus sure tant que rien n'est appris, et c'est celle qui
    # empeche d'apprendre quoi que ce soit.
    entropie_entree: float = 0.01
    pas_profit_par_epoch: int = 0
    # L'HORIZON SUR LEQUEL ELLE JUGE — ET IL FIXE AUSSI CELUI DE L'ENTREE.
    #
    # Cette valeur est la source unique de trois choses : la cible de
    # `tete_cloture`, l'horizon des etiquettes d'entree
    # (`rendements_du_systeme`), et la borne de resolution de la grille
    # (`_borne_syst`). Les separer laisserait l'entree notee sur une duree
    # que l'environnement ne joue pas.
    #
    # ELLE VALAIT TRENTE, ET UNE MESURE L'A DEPLACEE. L'ancien commentaire
    # justifiait 30 par le point mort : « en dessous de dix minutes il faut
    # capturer 16 % du plafond pour rentrer dans ses frais, 9 % a trente ».
    # Le raisonnement portait sur la part d'un mouvement a capturer, jamais
    # sur ce qu'un critere d'entree REEL rapporte une fois la friction
    # payee.
    #
    # LA MESURE DU 2026-09-22. `close_ema_dev` dans son decile le plus bas
    # en rang glissant — acheter le creux — hors echantillon, 2 878
    # occasions sur 194 JOURS distincts, friction comptee honnetement
    # (spread d'entree PLUS spread de sortie, pas deux fois celui
    # d'entree) :
    #
    #      30 min    brut +2.62 bps    NET -2.72    3.28 ecarts-type
    #      60 min    brut +5.00 bps    NET -0.34    3.58
    #     120 min    brut +7.62 bps    NET +2.28    3.13
    #     240 min    brut +9.26 bps    NET +3.92    2.33
    #
    # Le signal est le MEME partout — c'est le meme decile, les memes
    # occasions. Seule la friction change de poids relatif. A trente
    # minutes le systeme etait structurellement incapable de monetiser un
    # signal qu'il avait deja dans ses colonnes : `close_ema_dev` fait
    # partie des 272 depuis le debut.
    #
    # LE PASSAGE A ZERO EST ENTRE 60 ET 120. On prend 120, le premier
    # palier franchement positif, et non 240 : au-dela la fenetre
    # d'evaluation ne contient plus assez d'episodes independants, et
    # l'ecart-type retombe de 3.13 a 2.33.
    #
    # CE QUE CE CHIFFRE NE DIT PAS, et il faut le lire avec : l'avantage
    # est NEGATIF sur l'or a tous les horizons, et la fenetre de mesure est
    # un BTC qui monte. Acheter le creux amplifie la derive. Le rapport au
    # marche est de 3.8 fois (+9.26 contre +2.45 a 240 min), donc ce n'est
    # pas que du beta — mais il y en a dedans.
    #
    # DEUXIEME DEPLACEMENT, LE 2026-09-22 AU SOIR. La valeur 120 a tenu
    # quelques heures. Le test du signe MOIS PAR MOIS — 23 mois, aucun
    # parametre ajuste puisque le rang glissant se calibre seul — montre
    # que l'avantage GRANDIT avec la detention pendant que la friction se
    # paie une fois :
    #
    #     120 min   avantage +3.22 bps   17/23 mois   NET -0.65
    #     240 min   avantage +4.33 bps   17/23 mois   NET +0.93
    #     480 min   avantage +6.48 bps   17/23 mois   NET +4.07
    #     480 min + flux fort  +11.95    20/23 mois   NET +9.53
    #
    # A 120 minutes on payait le cout d'entree d'un mouvement qu'on ne
    # restait pas assez longtemps pour toucher. CE N'EST PLUS DU SCALPING
    # et il faut le savoir : huit heures de detention, une position a la
    # fois, environ 90 trades par mois reellement jouables.
    # ================================================================
    # RETOUR AU SCALPING — ET C'EST UNE DERIVE QU'IL FAUT NOMMER.
    # ================================================================
    #
    # Cette valeur a ete 30, puis 120, puis 480 dans la meme journee. Chaque
    # pas etait justifie par une mesure — le creux paie a 120, l'avantage
    # grandit jusqu'a 480 — et PERSONNE N'A VERIFIE LA DIRECTION CUMULEE.
    # On est parti d'un scalpeur M1 et on est arrive a des positions de
    # TROIS JOURS : `tenue[G 3330/3074 P 379/1086]`, maximum 5 517 minutes.
    #
    # C'est le mode d'echec classique de l'optimisation locale : quarante
    # pas justifies un par un menent ou personne ne voulait aller. Le
    # proprietaire du depot l'a vu avant moi.
    #
    # 60 MINUTES, ET PLUS 480. L'horizon d'etiquetage revient a une heure —
    # une duree ou le mot scalping garde un sens. On sait ce que ca coute :
    # l'avantage du creux mesure sur 23 mois valait +11.95 bps a 480 min et
    # seulement +3.22 a 120. A 60 il sera plus faible encore.
    #
    # ON L'ASSUME, PARCE QUE L'ALTERNATIVE ETAIT DE CHANGER DE PRODUIT sans
    # le decider. Un avantage plus mince sur la geometrie voulue vaut mieux
    # qu'un avantage plus gros sur une strategie que personne n'a demandee.
    #
    # 15 MINUTES — choix du proprietaire, 2026-09-25 : du vrai scalping.
    #
    # A 60, couper les pertes vite et predire le mouvement d'une heure se
    # contredisaient. Le BTC en minute s'ecarte de 7 a 8 ATR en une heure :
    # une coupure a 2 ATR sous le point mort tuait 41 % des trades qui
    # auraient fini gagnants a 60 minutes. A 15, la meme coupure n'en tue
    # plus que 10 % — mesure sur le train du fold 1, voir `marge_zombie_atr`.
    #
    # CE QUE CA COUTE, et il faudra le lire dans la veille : l'aller-retour
    # (4.18 bps median) ne raccourcit pas avec le trade. Il pese environ
    # deux fois plus lourd sur un mouvement de 15 minutes que sur un de 60.
    # Les entrees au hasard gagnent 44.7 % du temps a 15 minutes, contre
    # 47.8 % a 60.
    #
    # CE QUI SUIT L'HORIZON : `SCALPING_MAX_HOLDING` (saint_core), qui
    # normalise l'age, et `tenue_max_cloture` ci-dessous, trois fois
    # l'horizon — la ou la colonne d'age sature.
    horizon_cloture: int = 15
    # La tenue maximale tiree dans l'echantillon. Au-dela, la position est
    # plus vieille que tout ce que la tete verra.
    #
    # ELLE SUIT `horizon_cloture` ET NE DOIT PAS S'EN DETACHER. C'est elle
    # que `plafond_detention` rend en M1 : si l'echantillon de la tete
    # s'arretait a 30 pendant que l'environnement tient 120, la colonne
    # d'age saturerait a un quart de la vie reelle des positions et la tete
    # deciderait sur une valeur qu'elle n'a jamais vue.
    # LA TENUE MAXIMALE TIREE DANS L'ECHANTILLON — ET PLUS LE PLAFOND.
    #
    # Ces deux roles etaient le meme reglage jusqu'au 2026-09-22 :
    # `plafond_detention` rendait cette valeur en M1. Le plafond a ete
    # retire ; il ne reste que le role d'echantillonnage.
    #
    # ELLE MONTE A 1440 POUR SUIVRE. Sans horloge, une position vit bien
    # au-dela de 480 barres. Si l'echantillon s'arretait la, les deux tetes
    # decideraient sur des ages qu'elles n'ont jamais vus — exactement ce
    # que `echantillon_cloture` existe pour empecher.
    # 180 -> 45 LE 2026-09-25, avec l'horizon : trois fois l'horizon, la ou
    # la colonne d'age `min(age / SCALPING_MAX_HOLDING, 3)` sature.
    tenue_max_cloture: int = 45
    # LA REGLE DE SORTIE :  fermer si  latent < -(marge + k x risque)
    #
    # Tout est en ATR D'ENTREE — `latent_atr` est la colonne 1 de l'etat,
    # `cible_risque` est construite dans la meme unite.
    #
    # C'EST LA TETE QUI COMMANDE, ET PAS UN STOP FIXE. `marge_sortie` vaut
    # zero : le seuil est entierement porte par ce que la tete predit, donc
    # il bouge barre par barre avec le marche. Un stop fixe ne bougerait
    # pas, et il a ete refuse — voir plus bas.
    #
    # CE QUE LE BALAYAGE DONNE, hors echantillon, 2 355 positions a entree
    # aleatoire, erreur-type 0.165 bps :
    #
    #     aucune sortie (horizon 30)      moy -3.894  sd 15.77  med 30
    #     stop FIXE 1.0 ATR   (m=1 k=0)   moy -3.536  sd  9.58  med  6
    #     tete seule          (m=0 k=.25) moy -3.549  sd 10.69  med 13
    #     tete seule          (m=0 k=.50) moy -3.801  sd 12.95  med 30
    #
    # Les deux premieres sorties se valent — 0.013 bps d'ecart pour une
    # erreur-type de 0.165 — et toutes deux battent nettement l'absence de
    # sortie. La mesure ne departage donc PAS le stop fixe de la tete, et
    # dans ce cas c'est la conception demandee qui tranche : une tete qui
    # decide, pas un stop.
    #
    # ET CE N'EST PAS UNE CONCESSION GRATUITE. Un seuil qui suit la
    # prediction s'adapte au regime ; un stop fixe en ATR ne s'adapte qu'a
    # la volatilite du JOUR DE L'ENTREE, puisque l'ATR est fige a
    # l'ouverture. La tete, elle, lit le marche courant a chaque barre.
    marge_sortie: float = 0.0
    # `coupe_risque` EST LE SEUIL, EN MULTIPLES DU RISQUE PREDIT.
    #
    # 0.25 est retenu : c'est la valeur qui, sans marge fixe, fait le
    # meilleur resultat du balayage tout en gardant une detention MEDIANE
    # DE TREIZE BARRES. A 0.50 la detention remonte a 30 — la tete ne
    # ferme plus jamais — et la moyenne retombe a -3.801.
    #
    # RESERVE A ECRIRE : ce balayage porte sur des entrees ALEATOIRES, ou
    # l'esperance vaut exactement moins la friction et ou aucune regle de
    # sortie ne peut creer d'avantage. Il classe les regles par la variance
    # et la queue, pas par le gain. Les vraies entrees du modele portent
    # +1.27 bps de brut ; c'est sur elles que ce reglage doit etre
    # remesure, et le journal donne la duree de detention a chaque epoch.
    # LE STOP SE DESSERRE, ET C'EST UNE MESURE QUI L'IMPOSE.
    #
    # Balayage du 2026-09-22 sur les entrees validees (creux + flux fort),
    # plafond 1440, friction reelle a l'entree ET a la sortie, test du
    # signe mois par mois contre le marche du meme mois :
    #
    #     0.25 x  stop  4.91 ATR  avantage  +2.80  12/23 p 0.50  stoppe 76.2 %
    #     0.50 x  stop  9.82 ATR  avantage  +6.05  15/23 p 0.11  stoppe 58.4 %
    #     1.00 x  stop 19.64 ATR  avantage  +8.02  17/23 p 0.017 stoppe 33.7 %
    #     2.00 x  stop 39.28 ATR  avantage +13.00  17/23 p 0.017 stoppe 11.8 %
    #     4.00 x  stop 78.57 ATR  avantage +20.10  17/23 p 0.017 stoppe  2.5 %
    #     aucun                   avantage +25.04  18/23 p 0.005 stoppe  0.0 %
    #
    # Monotone : chaque desserrage ameliore. A 0.25 le stop coupait TROIS
    # TRADES SUR QUATRE et l'avantage n'etait meme pas significatif.
    #
    # ON PREND 2.0, PAS 4.0 NI ZERO. A 2.0 l'avantage est significatif et
    # le stop garde un sens — environ 39 ATR, soit ~2 % de mouvement
    # adverse sur ce BTC. A 4.0 il vaut 78 ATR : il ne protege plus de
    # rien, il decore. Et sans stop du tout, une position traverse toute
    # l'excursion adverse de la fenetre, dont la mediane MESUREE vaut
    # 11.87 ATR et la queue bien davantage.
    #
    # UNE MESURE FAUSSE A FAILLI FAIRE SUPPRIMER LE STOP. Une premiere
    # version de ce balayage annoncait « avantage -3.84, le stop detruit
    # tout » : la tete de risque simulee y etait ajustee sans INTERCEPT —
    # la colonne de uns, standardisee, valait (1-1)/1e-12 = 0 — donc elle
    # predisait 1.64 ATR la ou la cible en vaut 11.87, et le stop balaye
    # etait huit fois trop serre. La vraie tete du depot annonce bien
    # 13.95 ATR dans son journal.
    coupe_risque: float = 2.0
    n_echantillon_cloture: int = 200_000
    # LE POIDS DES MOINDRES CARRES, DESORMAIS EXPLICITE ET BAISSE A 0.2.
    #
    # Il valait 1.0 implicitement — le terme n'avait pas de coefficient. La
    # decomposition de la perte, mesuree pour la premiere fois le 2026-09-21
    # (exec60, epoch 1), montre pourquoi c'etait le mauvais reglage :
    #
    #     AuxL 2.3138  [mse 2.0829   topk -0.1531   ancre 1.9201]
    #
    #   `mse` pese 90 % du total ET IL ECHOUE : RMSE 1.443 pour une cible
    #   dont l'ecart-type vaut au plus 1.406. La tete predit MOINS BIEN
    #   qu'une constante egale a la moyenne, qui donnerait 1.98.
    #
    #   `topk` est negatif — il fait son travail — mais ne pese que 7 %.
    #
    #   `ancre` vaut 1.9201, soit un ecart d'ecarts-types de 1.386 : la tete
    #   produit une dispersion tres inferieure a celle de la cible. Elle
    #   predit presque une constante.
    #
    # POURQUOI CELA EXPLIQUAIT L'EFFONDREMENT. Quand une cible n'est pas
    # predictible en magnitude, le gradient des moindres carres pousse la
    # prediction vers la MOYENNE, c'est-a-dire vers une constante. C'est
    # exactement ce qu'on mesurait : `etendue` a 0.0002, `H` a 0, la barre
    # degeneree. Les deux autres termes tentaient de l'en empecher avec 10 %
    # du poids.
    #
    # ON DEPLOIE UN CLASSEMENT, PAS UNE VALEUR. Les consommateurs passent le
    # score par une sigmoide et le comparent a un quantile ; la magnitude de
    # la cible n'est jamais lue. Mettre 90 % du gradient a la predire est un
    # mauvais emploi.
    #
    # LES MOINDRES CARRES RESTENT, avec leur role documente — tenir l'echelle
    # pour que la sigmoide ne sature pas — mais a 0.2 au lieu de 1.0. Leur
    # contribution passe de 2.083 a 0.417, soit de 90 % a ~44 % du total, et
    # le tri comme l'ancre prennent le reste sans qu'on ait touche a leurs
    # coefficients.
    # 0.2 -> 0.05, LE 2026-09-21. L'ETENDUE DES SCORES S'EFFONDRAIT.
    #
    # CE QU'ON OBSERVE, exec71, fold 1 :
    #
    #     etendue des convictions   ep 1  0.2272   ep 12  0.0274
    #     au meme stade sur exec69  ep 1  0.2222   ep 12  0.1020
    #
    # Huit fois moins en douze epochs, quatre fois moins qu'au run
    # precedent. Une tete qui produit des scores quasi constants ne classe
    # plus rien : la barre de selectivite se pose n'importe ou dans une
    # distribution plate, et `sommet` mesure du hasard. Le depot connait
    # cette panne — voir `SeuilRang` — mais l'avait vue par derive du SEUIL,
    # pas par effondrement du SCORE.
    #
    # ET LE MODELE PLAFONNE EN MEME TEMPS : exces +0.037 de moyenne sur
    # douze epochs, `net` plat a -0.475 depuis l'epoch 2. Il n'apprend plus.
    #
    # CE QUE LA DECOMPOSITION DIT, ET CE QU'ELLE NE DIT PAS.
    #
    #     exec71 ep 12   AuxL -0.1261 [mse 2.2938 topk -0.9547 ancre 1.8494]
    #     exec69 ep 12   AuxL -0.1378 [mse 2.2956 topk -0.9670 ancre 1.8503]
    #
    # IDENTIQUES A LA TROISIEME DECIMALE. La nouvelle geometrie n'a donc
    # RIEN change a la composition de la perte, et le lien de cause entre
    # `coef_mse` et l'effondrement de l'etendue N'EST PAS ETABLI. Ce
    # changement est une tentative raisonnee, pas un diagnostic — et si
    # l'etendue ne remonte pas, c'est ici qu'il faudra revenir le lire.
    #
    # POURQUOI LUI PLUTOT QU'`ancre`. Ponderes, les trois termes valent
    # +0.459 (mse), +0.370 (ancre) et -0.955 (topk). `ancre` est precisement
    # le terme cense empecher la sous-dispersion, et le monter serait le
    # geste direct. Mais le depot l'a mesure : il ne faut pas MULTIPLIER un
    # terme, il faut DIVISER le dominant. Multiplier fait exploser le
    # gradient, l'ecretage le ramene a 0.6, et on deplace le compromis sans
    # donner un seul pas de plus au tri — c'est l'echec de `coef_top_k = 5`.
    #
    # LE RAISONNEMENT D'ORIGINE VAUT ENCORE, EN PLUS FORT. Quand une cible
    # n'est pas predictible en magnitude, le gradient des moindres carres
    # pousse la prediction vers la MOYENNE. Le plafond atteignable est passe
    # de +2.03 a +0.65 R avec le trailing a 0.5R : le signal a extraire est
    # trois fois plus petit, donc la pente vers la constante est
    # relativement plus forte. On divise encore par quatre.
    # 0.05 -> 0.20, RETOUR. L'ESSAI N'A RIEN DONNE.
    #
    # Il avait ete baisse pour empecher la tete de se refugier dans une
    # constante — l'etendue des scores tombait de 0.2272 a 0.0274 en douze
    # epochs. Mesure sur exec72, six epochs, contre exec71 au meme stade :
    #
    #              exces moyen      net moyen    etendue ep 5
    #   exec71        +0.032          -0.47         0.0494
    #   exec72        -0.044          -0.61         0.0290
    #
    # PIRE SUR LES TROIS COLONNES, et l'etendue oscille davantage
    # (0.118 -> 0.363 -> 0.063 -> 0.029) au lieu de se stabiliser. Six
    # epochs ne prouvent pas que 0.05 soit mauvais, mais rien n'indique
    # qu'il soit meilleur, et 0.20 a douze epochs derriere lui.
    #
    # ON REVIENT A LA VALEUR CONNUE pour ne changer qu'UNE chose a la fois :
    # le changement suivant porte sur la geometrie de sortie, et le
    # confondre avec un reglage de perte rendrait son effet illisible.
    coef_mse: float = 0.2

    # LE CRITERE DE SELECTION, AJOUTE A LA PERTE DE LA TETE DE RANG.
    #
    # REVENU A 1.0 APRES MESURE. Il avait ete porte a 5.0 pour faire dominer
    # le tri ; l'experience a echoue, et son echec est instructif.
    #
    #     temoin (1.0)   epoch 1   sommet +1.059   net +0.366   etendue 0.0678
    #     essai  (5.0)   epoch 1   sommet +0.817   net +0.160   etendue 0.0217
    #                              gnorm 203.5 contre 45.0
    #
    # Et a l'epoch 2 le run est MORT : la distribution des scores avait
    # assez bouge pour que la barre calibree a l'epoch 1 rejette 99.95 % des
    # entrees. Le modele est reste a plat, le compte de decisions est passe de
    # 2 103 a 43 050, et le tampon de collecte a epuise la memoire.
    #
    # CE QUE CELA APPREND, ET QUI VAUT PLUS QUE LE REGLAGE : il ne faut pas
    # MULTIPLIER un terme, il faut DIVISER le terme dominant. Multiplier fait
    # exploser le gradient (203 contre 45) ; l'ecretage le ramene ensuite a
    # 0.6, donc la DIRECTION change mais le PAS non — on deplace le compromis
    # sans donner plus de moyens au tri, et l'echelle des scores part. Diviser
    # le terme dominant fait baisser le gradient total, donc l'ecretage mord
    # MOINS, et le poids relatif du tri monte quand meme.
    #
    # C'est `coef_mse` ci-dessus qui porte le reequilibrage, pas celui-ci.
    coef_top_k: float = 1.0
    # 0.20 ET NON 0.05, ET C'EST UN ECART ASSUME AVEC LE POINT DE COUPE
    # DEPLOYE. Le raisonnement d'origine etait qu'entrainer sur une autre
    # selectivite que celle qu'on deploie revient a optimiser une autre regle.
    # C'est vrai, et ce n'est pas ce qui a casse.
    #
    # CE QUE LA MESURE MONTRE (2026-09-20, `rhoAux` par epoch) :
    #
    #   exec35 ni top-k ni ancre   +0.074 ... +0.020  |  +0.009 ... +0.031
    #   exec38 top-k + ancre 0.2   -0.025 ... +0.135  |  -0.011 ... -0.169
    #                              ^ acteur gele      ^ acteur vivant
    #
    # exec38 avait le MEILLEUR rhoAux de tous les runs tant que l'acteur
    # dormait — jusqu'a +0.135, quand aucun autre ne depasse +0.074. La
    # rupture tombe a l'epoch 7, soit UNE EPOCH APRES le reveil de l'acteur
    # (`critic_warmup_epochs = 5`). Et exec35, qui traverse le meme reveil
    # sans top-k, ne se degrade pas. Ce n'est donc ni l'acteur seul, ni
    # l'ancre : c'est le top-k qui ne survit pas au reveil.
    #
    # LE MECANISME. PPO depose un gradient de 1.47 a 1.81 sur le TRONC
    # PARTAGE, mesure dans le journal. Le terme top-k, lui, concentre tout son
    # gradient sur environ SIX occasions par lot de 128 — les 5 % retenus.
    # Signal faible et a forte variance contre une moindre carre qui utilise
    # les 128. Tant que l'acteur dort il suffit ; des que PPO remodele le
    # socle, il ne tient pas.
    #
    # A 20 %, le terme porte sur ~26 occasions au lieu de 6 : quatre fois plus
    # de signal, et une variance divisee par deux. BAISSER `coef_top_k`
    # ferait l'inverse — affaiblir un signal deja trop faible.
    #
    # LE PRIX : on entraine le classement sur le haut des 20 % et on deploie
    # sur le haut des 5 %. Les deux ensembles sont emboites, donc l'ordre
    # appris reste le bon ; c'est la ponderation qui differe legerement.
    selectivite_top_k: float = 0.05
    # Empeche l'etendue des scores de se resserrer sous celle de la cible.
    # A SENS UNIQUE : nul quand la moindre carre suffit.
    #
    # 0.2 ET NON 1.0. A pleine force l'ancre fait bien son travail — etendue
    # de validation tenue a 0.55-0.64 contre 0.027-0.093 sans elle — mais elle
    # coute sur le critere qui SELECTIONNE les checkpoints. Comparaison a
    # epochs identiques, acteur gele dans les deux cas (2026-09-20) :
    #
    #                          exec36 sans ancre   exec37 avec (coef 1.0)
    #     `sommet` moyen 1-5               1.116                    0.705
    #     `etendue[val]`            0.027 - 0.093            0.55 - 0.64
    #     `AuxL` a l'epoch 5              0.3842                   0.1113
    #
    # L'ancre gagne sur ce pour quoi elle est faite, et `AuxL` est plus basse
    # ALORS QU'ELLE PORTE UN TERME DE PLUS — donc moindre carre et top-k
    # seuls sont strictement mieux ajustes avec elle. Mais 0.705 contre 1.116
    # sur `sommet` fait nominalement 2 sigma, moins en verite puisque les
    # epochs d'un meme run sont correlees entre elles.
    #
    # CE QUI FAIT PENCHER : sans ancre, l'etendue REMONTAIT toute seule
    # (0.027 -> 0.078 en cinq epochs). Le probleme qu'elle corrige ne se
    # produisait donc pas vraiment. On garde une protection contre un
    # effondrement — le garde-fou `pbs_etendue_min` est a 0.01 — sans en payer
    # le plein prix sur le classement.
    # REMIS A 0.2 LE 2026-09-21 : LA PREMISSE DU RETRAIT EST TOMBEE.
    #
    # Le raisonnement ci-dessus concluait que « le probleme qu'elle corrige ne
    # se produisait pas vraiment », parce que l'etendue remontait toute seule
    # (0.027 -> 0.078 en cinq epochs sur exec36). Cette mesure a ete prise
    # quand l'ACTEUR decidait encore des entrees : l'echelle de la tete de
    # rang ne servait alors qu'a la validation.
    #
    # DEPUIS QUE LA DIRECTION EST SUPPRIMEE, le score de cette tete choisit
    # aussi les etats que le rollout COLLECTE — donc ses propres donnees
    # d'entrainement. C'est une boucle de retroaction qui n'existait pas, et
    # elle change le verdict.
    #
    # MESURE, exec52, quatre epochs :
    #
    #     epoch        1        2        3        4
    #     etendue[tr]  0.1948   0.0114   0.0003   0.0214
    #     etendue[val] 0.0145   0.0004   0.0144   0.0854
    #     thr[tr]      0.000    0.556    0.514    0.000
    #     H            0.581    0.000    0.000    0.570
    #     AuxL         1.7600   1.9572   2.1381   2.2088
    #     rhoAux      +0.0232  +0.0404  +0.0149  -0.0737
    #
    # L'etendue ne remonte pas : elle tombe a 0.0003, un ordre de grandeur
    # SOUS le plancher de 0.027 qui avait servi a conclure. La tete donne alors
    # le meme score a toutes les occasions ; sous `pbs_etendue_min` la barre est
    # forcee a zero, tout passe, et `H` tombe a 0 — le modele prend la meme
    # decision partout. `AuxL` monte a chaque epoch et `rhoAux` finit a
    # -0.0737, soit 3 sigma A L'ENVERS pour une incertitude de 0.024.
    #
    # 0.2 ET NON 1.0 : c'est le reglage qui tient l'echelle sans payer le plein
    # prix sur `sommet` (0.705 contre 1.116 a pleine force). Le garde-fou
    # `pbs_etendue_min` reste en place — il n'a pas suffi, il n'etait pas fait
    # pour ca.
    coef_ancre: float = 0.2

    # ------------------------------------------------------------------
    # DIAGNOSTIC DE CLASSEMENT. Purement informatif : rien ne s'en sert pour
    # selectionner un checkpoint, decider ou arreter. Le mettre a False
    # supprime la ligne `rho` du journal et la passe avant qui la produit.
    # LE TRI SE FAIT PAR LA TETE AUXILIAIRE, pas par la politique.
    #
    # LE CONSTAT, mesure le 2026-09-16 sur 11 094 occasions de validation,
    # entrainement sur le train, test intouche :
    #
    #   apprenant                        rho valid.     +/-  sigma   E[R] sommet 5%
    #   moindres carres (rendement)        +0.0432  0.0216   +2.0          +0.4082
    #   logistique sklearn                 +0.0504  0.0232   +2.2          +0.3044
    #   TEMOIN poids au hasard             +0.0029  0.0194   +0.1          +0.0791
    #
    # Une regression LINEAIRE classe a 2 sigma pendant que le rho de la
    # politique oscille dans le bruit. L'information est donc dans les
    # features et PPO ne l'extrait pas : il optimise le rendement de ses
    # ACTIONS, et personne ne lui demande que l'ORDRE de ses probabilites soit
    # juste — or c'est tout ce dont la selectivite a 5 % se sert.
    #
    # C'est le plus vieux chiffre du depot qui se confirme : une logistique
    # obtenait 0.6271 d'AUC contre 0.5707 pour la politique.
    #
    # La tete auxiliaire, elle, est entrainee a PREDIRE le rendement — donc a
    # l'ordonner — sur tous les etats collectes et non sur les seuls etats de
    # decision. Mesure sur exec32 : rho +0.0686 pour elle contre +0.0696 pour
    # la politique, a egalite, alors qu'elle n'est qu'une couche lineaire de
    # deux sorties. Un modele de quelques dizaines de parametres egale tout le
    # reseau : le reseau n'apporte rien au classement.
    #
    # CE QUE CELA NE CHANGE PAS. Le rollout continue d'echantillonner les
    # actions de la politique, donc PPO reste exact et on-policy. Seule la
    # SELECTION — le seuil de conviction applique en validation, au test et en
    # production — lit desormais la tete. La perte de la tete ne change pas
    # non plus : mesure faite, le carre sur le rendement continu bat la cible
    # binaire sur le rendement du sommet (+0.41 contre +0.28), qui est le seul
    # chiffre qui compte.
    #
    # `aux_coef` n'est deliberement PAS touche dans le meme run : deux
    # changements simultanes rendraient le resultat inattribuable.

    # LA VALIDATION COMPLETE NE TOURNE PLUS A CHAQUE EPOCH.
    #
    # Decomposition mesuree du temps d'une epoch :
    #
    #   temps[collecte 105s  maj PPO 112s  calibration 38s  validation 182s]
    #
    # La validation est le premier poste, 45 % du total, et elle ne depend pas
    # du nombre de decisions : c'est un cout fixe de parcours de la fenetre.
    #
    # Or on a mesure le meme jour que son PnL est ILLISIBLE : ~150 trades,
    # erreur-type de 0.11 R sur leur moyenne, donc incapable de distinguer un
    # modele qui ajoute 0.10 R d'un modele qui n'ajoute rien. On payait 45 %
    # de chaque epoch pour un chiffre dont on a demontre qu'il ne dit rien.
    #
    # Le `rho` de classement, lui, coute DEUX forwards — 30 millisecondes —
    # et c'est la metrique qu'on lit reellement. Il reste a chaque epoch.
    #
    # Ce choix ne selectionne rien : le modele deploye est la MOYENNE DES
    # POIDS des dernieres epochs, qui ne depend d'aucune validation. Les
    # familles "best" et "bestprofit" en dependent, elles, et sont mises a
    # jour moins souvent — ce sont de toute facon des choix faits sur la
    # validation, que ce depot mesure a -3.3 points contre la moyenne.
    #
    # La DERNIERE epoch valide toujours, pour que le run se termine sur une
    # mesure complete.
    #
    # ------------------------------------------------------------------
    # 3 -> 1 LE 2026-09-19 : LA VALIDATION TOURNE A CHAQUE EPOCH.
    #
    # Le raisonnement ci-dessus tenait tant que la validation ne servait
    # qu'a produire un PnL qu'on ne lisait pas. Deux choses l'ont perime.
    #
    # LE CRITERE DE SELECTION A CHANGE. Ce n'est plus le Sortino mais le
    # rendement du SOMMET, sous garde que le PORTEFEUILLE ait survecu. Cette
    # garde lit le creux de la derniere mesure reelle : une epoch sur trois,
    # elle jugeait donc un modele sur un portefeuille vieux de deux epochs.
    #
    # LE PLAFOND DE RISQUE A ETE RETIRE. Le compte peut desormais porter des
    # dizaines de positions correlees, et c'est precisement ce que le
    # critere ne voit pas : `gain_top` mesure les occasions UNE A UNE. Un
    # modele au classement excellent qui vide le compte serait retenu — et
    # transmis au fold suivant — si le creux qui le disqualifie date de deux
    # epochs.
    #
    # CE QUE CELA COUTE, et ce n'est pas negligeable : ~145 s par epoch,
    # soit environ SEPT HEURES sur un run de trois folds a quatre-vingt-dix
    # epochs. C'est le prix pour que le garde-fou de survie regarde le
    # portefeuille de l'epoch qu'il juge, et non celui d'avant.
    # ------------------------------------------------------------------
    validation_tous_les: int = 1

    #
    # FALSE DEPUIS LE PPO COMPLET, 2026-09-25. La grille jugeait un
    # CLASSEMENT — le sommet du tri contre le hasard — et il n'y a plus de
    # classement. Le critere juge desormais les trades joues en validation.
    diag_rang: bool = False
    diag_rang_pas: int = 12    # une decision suivie toutes les heures
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    # Largeur du reseau. 64/256 donnait 1.19 M de parametres pour 16 708
    # barres d'entrainement, soit 71 parametres par exemple — et le modele
    # memorisait. 32/128 ramene a 298 k.
    # D_MODEL 32 -> 8. Le relevé de l'architecture d'exec16 a montré ce que
    # personne n'avait regardé : la TETE pese 92 % des parametres et
    # l'encodeur 8 %. Les jetons sont moyennes avant la tete, donc reduire le
    # lookback de 96 a 4 n'a rien enleve au modele — il a seulement moins a
    # lire. Six essais d'architecture ont donc porte sur 8 % du reseau.
    #
    # d_model est le seul reglage qui agisse sur les 92 % : il multiplie les
    # 107 entrees de la tete. 32 -> 8 fait tomber le compte de 493 796 a
    # 129 404 parametres, soit 2.3 par barre d'entrainement au lieu de 8.9.
    #
    # Les tetes d'attention restent a 4, donc 2 dimensions chacune. C'est
    # etroit, mais l'encodeur ne represente plus que 2 % du reseau : ce n'est
    # plus lui qu'on regle.
    # D_MODEL 8 -> 4 ET MLP_DIM 128 -> 32. Les deux agissent sur la taille de
    # la TETE, qui pese 95 a 99 % du reseau : c'est le meme levier pris par ses
    # deux bouts, pas deux variables.
    #
    # LA COLONNE QUI COMMANDE n'est pas le nombre de parametres par barre mais
    # par OCCASION INDEPENDANTE. La fenetre d'entrainement fait 55 538 barres,
    # mais seulement ~2 314 occasions sans chevauchement — une par jour, la
    # duree d'un trade. C'est ce nombre-la qui borne ce qu'on peut apprendre.
    #
    #   d_model  mlp_dim   params   par barre   par occasion
    #         8      128   129 404       2.33          55.9   exec17
    #         8       32    31 292       0.56          13.5
    #         4      128    72 704       1.31          31.4
    #         4       32    15 680       0.28           6.8   ici
    #
    # Trois runs successifs designent la meme direction sur le meme fold :
    # -3.08 (d_model 32, lookback 96), -1.62 (d_model 32, lookback 4), -0.73
    # (d_model 8). C'est la seule tendance monotone que ce depot ait produite.
    #
    # CE QU'IL FAUT SAVOIR EN ARRIVANT LA. A d_model 4 avec quatre tetes,
    # chaque tete d'attention a UNE dimension : l'attention devient une
    # similarite scalaire, c'est-a-dire a peu pres rien, et l'encodeur ne pese
    # plus que 5 % des poids. Ce qu'on entraine est un petit MLP sur des
    # features moyennees, avec un transformer decoratif. C'est la description
    # de TabM, qui rend +3.8 points en quelques minutes.
    d_model_patch: int = 4
    # 32 -> 8, meme raison : voir saint_mlp_dim. PatchTST porte le tiers du
    # budget, il doit suivre la meme reduction sous peine de devenir le membre
    # dominant par accident.
    mlp_dim: int = 8

    # PPO Training
    # BUDGET FIXE, PAS D'ARRET PRECOCE. Mesure du 2026-09-15 : le checkpoint
    # choisi sur la validation rend -3.3 points au test quand le dernier epoch
    # en rend -0.2, et faire choisir la selectivite de TabM par la validation
    # le fait passer de +1.6 a -2.1. Deux methodes sans rapport, meme verdict :
    # ce qu'on regle sur cette fenetre se transfere NEGATIVEMENT. Validation et
    # test font onze mois chacune et se suivent ; un reglage ajuste sur onze
    # mois de marche est anti-correle aux onze suivants.
    #
    # L'arret precoce EST une selection sur la validation : il choisit quand
    # s'arreter d'apres elle. On fixe donc le budget a l'avance. 40 epochs est
    # choisi parce que les trois folds d'exec12 avaient plateau entre 30 et 35
    # — c'est un budget lu sur la DYNAMIQUE d'entrainement, jamais sur le test.
    # 40 -> 90. Deux raisons, et aucune n'est un reglage.
    #
    # Un episode peut desormais s'arreter sur RUINE — solde insuffisant pour
    # le lot minimum, ou appel de marge. Les episodes courts ne couvrent plus
    # la fenetre, donc une epoch voit moins de marche qu'avant a budget egal.
    #
    # Et la capacite depend de l'equity, donc la politique doit apprendre une
    # boucle : decider, gagner ou perdre, voir sa capacite changer, decider a
    # nouveau. Une boucle s'apprend sur des passages repetes, pas sur
    # quarante.
    #
    # La moyenne des poids porte sur les dernieres epochs (`n_moyenne_poids`),
    # donc allonger le run ne selectionne rien sur la validation : il n'y a
    # pas de choix a faire, seulement plus de passages.
    # 90 -> 45. LA SECONDE MOITIE DETRUISAIT LE MODELE.
    #
    # MESURE SUR exec69, fold 1, 91 epochs, selectivite de validation
    # CONSTANTE a 5 % — donc `sommet` compare bien la meme chose d'un bout
    # a l'autre, et l'effondrement n'est pas un artefact de mesure :
    #
    #     ep  1-22   exces +0.514   net +0.450
    #     ep 23-45   exces +0.594   net +0.425     <- le sommet
    #     ep 46-68   exces +0.103   net -0.114
    #     ep 69-91   exces -0.008   net -0.300     <- au niveau du hasard
    #
    # Le dernier quart ne vaut PLUS RIEN. Le modele apprend jusqu'a ~45 puis
    # se degrade jusqu'a ne plus battre un tirage au sort.
    #
    # POURQUOI, ET CE N'EST PAS UN REGLAGE. Le reseau porte 46 496
    # parametres pour 438 a 682 occasions INDEPENDANTES par fenetre
    # d'entrainement — soit 68 a 106 parametres par occasion, quand on vise
    # couramment moins d'un parametre pour dix exemples. Passe un certain
    # point, chaque epoch supplementaire memorise la fenetre au lieu
    # d'apprendre quoi que ce soit de transferable. Reduire `epochs` ne
    # corrige pas cette disproportion, ca en limite les degats.
    #
    # ON NE MET PAS D'ARRET PRECOCE AUTOMATIQUE A LA PLACE : `patience`
    # surveillerait le score de validation, qui est precisement la grandeur
    # que le sur-apprentissage fait monter avant de la faire chuter. Un
    # plafond mesure est plus honnete qu'une regle qui lit le symptome.
    epochs: int = 45
    # Nombre d'epochs finales dont les poids sont moyennes. La moyenne remplace
    # le "meilleur checkpoint" : elle ne depend d'aucun tirage particulier.
    n_moyenne_poids: int = 10
    # Épisodes joués EN PARALLÈLE (un seul forward batché par pas) : le
    # profilage montre qu'un batch 16 coûte le même temps qu'un batch 1, donc
    # multiplier les épisodes est quasi gratuit en temps de rollout.
    #
    # Monté de 4 à 32 parce que l'actor n'apprend que sur les états FLAT, et
    # qu'avec des détentions de ~5h l'agent est en position 99.9% du temps :
    # à 4 épisodes il n'y avait qu'une quinzaine de décisions dans tout le
    # buffer d'une epoch, soit 0.26 par minibatch de 256 — la plupart des mises
    # à jour voyaient zéro décision et le gradient de l'actor était nul.
    # 128 envs × 6000 bougies. Le nombre de DÉCISIONS (états flat) est le seul
    # volume qui compte pour l'actor : il valait 113-156 par epoch, soit un
    # unique minibatch, soit 2 pas de gradient par epoch. Le semi-MDP a rendu
    # le rollout peu coûteux (la policy n'est plus appelée qu'aux décisions) :
    # on convertit cette vitesse en volume. Attendu : ~1700 décisions/epoch.
    # DIMENSIONNEMENT H1. Le jeu fait 30 379 barres contre 1.8 M en M1 :
    # un episode de 6 000 barres couvrirait 28 % du train a lui seul, et les
    # episodes se recouvriraient presque entierement.
    # 96 episodes de 2 016 barres couvrent 193 536 barres, soit 29 % des
    # 666 421 barres d'entrainement a chaque epoch. En H1 le meme reglage en
    # couvrait 69 % — mais le jeu H1 etait douze fois plus petit, et ce qui
    # compte n'est pas la part du jeu vue par epoch, c'est le nombre de trades
    # que la collecte produit. A ~46 trades par episode, une epoch en voit
    # environ 4 400, contre 1 400 en H1.
    #
    # 96 -> 336, le 2026-09-16, POUR REPARER UNE ERREUR INTRODUITE ICI MEME.
    #
    # Le paragraphe ci-dessus decrit 96 episodes de 2 016 barres = 193 536
    # barres par epoch. En ramenant episode_length a 576, j'ai divise ce volume
    # par 3.5 sans toucher au nombre d'episodes : la collecte est tombee a
    # 55 296 barres, soit 10 % de la fenetre d'entrainement, et les decisions
    # PPO de ~2 050 a ~850 par epoch.
    #
    # L'ARGUMENT QUI M'AVAIT TROMPE. J'avais mesure qu'un episode cinq fois
    # plus long rendait le meme nombre de decisions — 2 054 en M5 contre 2 195
    # en H1 — et conclu que la longueur ne servait a rien. Mais ces deux
    # chiffres viennent de DEUX ECHELLES DIFFERENTES. A l'interieur du M5 les
    # decisions sont proportionnelles aux barres collectees, et la mesure ne
    # disait rien de cela. Comparer entre echelles ce qu'il fallait comparer
    # a echelle constante : c'est la meme faute que la mesure a 30 barres qui
    # avait recommande un stop de 8xATR inexistant.
    #
    # CE QUE CA COUTAIT. Le jeu M5 offre ~15 100 occasions independantes, mais
    # chaque epoch n'en echantillonnait que 850 — 6 %, tirees ailleurs a chaque
    # fois. Apres dix epochs le modele avait vu 0.7 passage sur ses donnees. En
    # H1 il voyait 2 195 decisions pour 2 314 occasions, soit la quasi-totalite
    # du jeu A CHAQUE EPOCH. Le gain d'occasions du M5 n'avait donc jamais
    # atteint l'optimiseur : on avait achete des donnees et laisse le budget de
    # collecte a sa valeur d'avant.
    #
    # LE SYMPTOME QUI L'A TRAHI : sur exec23, le PF d'ENTRAINEMENT plafonne a
    # 0.87-0.95 pendant dix epochs. Un modele qui n'arrive pas a gagner sur les
    # donnees qu'il optimise n'est pas en train de sur-apprendre — il n'apprend
    # pas du tout, faute d'echantillons.
    #
    # POURQUOI PLUS D'EPISODES ET PAS DES EPISODES PLUS LONGS. 336 x 576
    # redonne exactement les 193 536 barres d'origine, mais avec 336 departs
    # independants au lieu de 96 : moins de correlation a l'interieur d'un lot,
    # donc un gradient moins bruite a volume egal. Les episodes tournent EN
    # PARALLELE avec un seul forward batche par pas, et la politique fait
    # 45 000 parametres — le surcout GPU d'un batch 336 contre 96 est nul.
    # 336 -> 84 : le produit avec `episode_length` est conserve a 483 840
    # barres. On allonge les episodes sans changer ce qu'une epoch consomme.
    # PLUSIEURS POSITIONS SIMULTANEES, mesure du 2026-09-16.
    #
    # LE CONSTAT. L'acteur ne recoit de gradient que sur les barres ou il PEUT
    # entrer : en position, le masque ne laisse que HOLD et log pi(HOLD) vaut
    # exactement 0. Releve dans le journal : `dec` vaut ~500 par epoch pour
    # 483 840 barres collectees, soit 0.1 %. Entre deux trades enchaines la
    # politique n'est plate qu'UNE barre. Le plafond configure vaut 40 000 et
    # n'est jamais approche.
    #
    # C'est le diagnostic que la veille repete a chaque epoch : "APPREND MAIS
    # RESTE PLAT — le gradient passe mais l'entropie tient a 1.093 sur 1.099".
    # Avec 500 echantillons par epoch, une politique ne peut pas se
    # differencier.
    #
    # CE QUE LA CONCURRENCE ACHETE, ET CE QU'ELLE N'ACHETE PAS.
    #
    #     K   trades   N effectif   decisions vues par l'acteur
    #     1       49         49.0                           45
    #     4      182         65.0                          182
    #     8      420         77.2                          383
    #    20     1245        104.3                        1 154
    #
    # Le gradient est multiplie par K, exactement. L'INFORMATION, non : deux
    # trades de meme sens ouverts a une heure d'ecart correlent a 0.72, et
    # encore a 0.31 apres 24 h. Multiplier les trades par 25 ne multiplie les
    # occasions independantes que par 2.1, donc la detectabilite par 1.46.
    #
    # C'est donc un levier D'APPRENTISSAGE, pas de rentabilite. Et c'est le
    # seul qui soit gratuit en calcul : allonger la collecte donnerait le meme
    # gradient en payant K fois plus de barres, alors qu'ici le nombre de
    # barres ne bouge pas.
    #
    # LE RISQUE NE SUIT PAS LE NOMBRE. K positions de meme sens subissent le
    # meme mouvement : la variance du portefeuille vaut K(1+(K-1)rho) fois
    # celle d'une seule, soit ~30 fois a K=8 avec rho 0.45. `_compute_dynamic_size`
    # divise donc la taille par racine de ce facteur, pour que le risque total
    # reste `risk_per_trade` quel que soit K.
    #
    # A K=1 tout ce chemin doit rendre EXACTEMENT l'environnement d'avant.
    # `test_concurrence.py` le verifie sur 613 trades et six scenarios, en
    # comparant recompense par recompense et centime par centime.
    # 1 -> 8. PLAFOND DE TABLEAU, pas nombre de positions : le nombre reel
    # est ce que le solde permet, recalcule a chaque barre par
    # `places_ouvrables`, et le modele le voit dans son observation.
    #
    # 8 est le GENOU DU COUT, mesure : jusque-la les operations sur
    # emplacements restent vectorielles et la barre ne coute pas plus cher.
    #
    #     K   trades   decisions   x dec   secondes
    #     1      613       5,364    1.0x       14.0
    #     8    2,480      21,768    4.1x       14.1   <- gratuit
    #    16    2,760      24,257    4.5x       16.5
    #    64    3,337      29,313    5.5x       29.4
    #
    # Le gradient de l'acteur est multiplie par K ; l'information, elle, ne
    # l'est pas — 25 fois plus de trades ne valent que 2.1 fois plus
    # d'occasions independantes, parce que deux positions de meme sens
    # ouvertes a une heure d'ecart correlent a 0.72. C'est donc un levier
    # d'APPRENTISSAGE, et c'est comme tel qu'il faut le juger.
    # 8 -> 32. LE PLAFOND NE DOIT PLUS MORDRE, sinon il masque le cercle
    # vertueux : a 8, l'equity pouvait doubler sans qu'une seule position de
    # plus soit permise. C'est desormais le budget de risque qui decide, et
    # 32 est simplement au-dessus de ce qu'il autorise a l'equity de depart
    # (9 positions a 1 000 EUR). Le cout suit l'occupation reelle, pas le
    # plafond : les operations sont vectorielles sur un tableau de 32.
    # IL N'Y A PLUS DE PLAFOND. `positions_max` n'est qu'une ALLOCATION
    # INITIALE : les tableaux d'emplacements doublent de taille quand ils se
    # remplissent, donc le nombre de positions n'est borne que par ce que le
    # solde permet.
    #
    # Tout plafond fixe masquait le cercle vertueux. A 32 la capacite saturait
    # des 2 000 $ d'equity, a 64 des 4 000 $ : au-dela, gagner n'ouvrait plus
    # rien et le modele n'avait plus rien a apprendre de sa reussite. La
    # limite doit venir du solde, et seulement de lui.
    positions_max: int = 64

    # ------------------------------------------------------------------
    # LE COURTIER, TEL QU'IL EST. Releve sur Vantage BTCUSD le 2026-09-16 :
    #
    #     contrat 1 BTC | lot min 0.01 | pas 0.01 | levier compte 500
    #     marge pour 1 lot : 131.17 EUR a 75 654, soit 0.173 % du notionnel
    #
    # `positions_max` n'est donc PAS le nombre de positions : c'est la taille
    # du tableau, un plafond de memoire. Le nombre REEL est ce que le solde
    # permet, et il change a chaque barre — c'est ce que le modele doit
    # apprendre, pas une constante qu'on lui impose.
    #
    # DEUX CONTRAINTES, ET CE N'EST PAS CELLE QU'ON CROIT QUI MORD.
    #
    # La marge est large : a 0.173 % du notionnel, 1 000 EUR autorisent des
    # centaines de lots minimums. Ce qui borne vraiment, c'est le LOT MINIMUM.
    # A 12xATR la taille visee vaut 0.0045 BTC, soit moins de la moitie du
    # minimum de 0.01 : le courtier impose donc 2.2 fois la taille voulue, et
    # le risque reel est de 1.19 % par trade au lieu des 0.53 % configures.
    #
    # L'entrainement dimensionnait en continu et ne voyait jamais cela. Il
    # apprenait sur un courtier qui n'existe pas.
    # LE CONTRAT DE L'OR VAUT CENT ONCES. Un lot minimum y represente donc
    # 4 265 $ de notionnel contre 762 pour le BTC, et risque 2.81 % d'un
    # compte de 1 000 EUR contre 0.73 %. C'est ce chiffre, et non la
    # volatilite, qui fixe le nombre de positions tenables.
    lot_min: float = 0.01
    # TAILLE DU CONTRAT, en unites par lot. Elle vaut 1 sur le BTC et 100 sur
    # l'or : un lot d'or, c'est cent onces. `instruments.py` la fixe par
    # symbole ; l'ignorer ferait trader cent fois trop petit sur l'or.
    contrat: float = 100.0
    lot_pas: float = 0.01
    marge_frac: float = 0.001734        # marge / notionnel, mesuree
    # Niveau de marge (equity / marge utilisee) sous lequel on n'ouvre plus.
    # Les courtiers appellent la marge vers 100 % et liquident vers 50 %.
    niveau_marge_ouverture: float = 3.0
    niveau_marge_liquidation: float = 0.5
    # Mettre a False redonne le dimensionnement continu d'avant, qui sert de
    # temoin : c'est la seule facon de mesurer ce que la contrainte coute.
    marge_realiste: bool = True

    # LE CERCLE VERTUEUX, et pourquoi la marge seule ne le produit pas.
    #
    # Gagner augmente l'equity, donc la capacite, donc le nombre de positions
    # tenables — et perdre la reduit. C'est la boucle que le modele doit
    # apprendre. Encore faut-il qu'une contrainte la porte.
    #
    # Ce n'est PAS la marge. A 1:500 et 0.173 % du notionnel, 1 000 EUR
    # autorisent deja 254 lots minimums : le plafond du tableau mordrait
    # toujours en premier, l'equity pourrait doubler sans rien changer, et la
    # boucle serait invisible. Mesure faite : plafond 32, marge 254, donc la
    # marge n'a jamais decide.
    #
    # Ce qui borne vraiment, c'est LE RISQUE ENGAGE. La somme des montants a
    # perdre si tous les stops etaient touches ne doit pas depasser une part
    # de l'equity. Cette part est un montant, donc elle grandit quand on gagne
    # et retrecit quand on perd — exactement la boucle recherchee, et sur une
    # echelle economique plutot que sur un artefact de levier.
    #
    # A 1 000 EUR : 10 % font 100 EUR, un lot minimum risque ~11 EUR a
    # 12xATR, donc 9 positions. A 2 000 EUR, 18. A 500 EUR, 4. La capacite
    # suit l'equity lineairement, dans les deux sens.
    #
    # La valeur 0.10 n'est pas mesuree — c'est une limite de prudence, pas un
    # optimum. Ce qui est mesure, c'est qu'elle mord la ou la marge ne mord
    # pas. Le modele apprend a l'interieur ; il ne la choisit pas.
    # ------------------------------------------------------------------
    # PLUS AUCUN PLAFOND DE RISQUE : SEUL LE COURTIER BORNE, sur demande
    # explicite. Le modele doit apprendre a ouvrir autant de positions que le
    # compte en permet, et a gerer ce risque lui-meme.
    #
    # CE QUE CETTE VALEUR FAISAIT. `places_ouvrables` prenait le minimum de
    # deux bornes : ce que la MARGE du courtier autorise, et ce que NOTRE
    # budget de risque tolere. La seconde est une politique, pas une
    # contrainte de marche — et a 3 % elle etait de loin la plus serree :
    # elle n'autorisait qu'UNE position d'or a 1 000 EUR, ce qui rendait
    # toute la machinerie de concurrence inerte.
    #
    # A ZERO, la borne de risque disparait (`par_risque` vaut l'infini) et il
    # ne reste que la marge.
    #
    # CE QUE CELA OUVRE, ET CE QU'IL FAUT SAVOIR. La marge autorise environ
    # QUARANTE-QUATRE lots minimums d'or a 1 000 EUR — niveau de marge 3,
    # 7.44 $ immobilises par lot. Chacun risque 2.81 % de l'equite au stop,
    # donc quarante-quatre positions de meme sens exposent ~124 % du compte.
    # Un mouvement adverse coordonne le vide. Ce n'est pas un effet de bord :
    # c'est la situation que le modele doit apprendre a ne pas provoquer.
    #
    # CE QUI RESTE POUR L'EN EMPECHER, et ce sont les seules barrieres :
    #   - l'appel de marge, quand l'equite tombe sous la moitie de la marge ;
    #   - le garde-fou de creux a 40 %, qui termine l'episode ;
    #   - le plancher de capital a 20 % ;
    #   - la penalite de creux, PORTEE PAR CHAQUE POSITION et non partagee,
    #     donc croissante avec l'exposition — c'est le gradient qui distingue
    #     la premiere position de la quarantieme.
    #
    # Les episodes mourront plus souvent. Un episode qui meurt enseigne sa
    # mort ; c'est le prix a payer pour que le modele decouvre la limite au
    # lieu de la recevoir.
    # LE BUDGET DE RISQUE EST DESORMAIS CHOISI PAR LE MODELE, a chaque
    # decision, parmi `BUDGETS_PART`. Cette valeur n'est plus que le point
    # de DEPART d'un episode, avant la premiere decision.
    #
    # Elle vaut le palier le PLUS BAS, et c'est deliberement un a priori
    # prudent : le modele doit choisir activement de s'exposer davantage. La
    # mettre a zero — le reglage du 2026-09-19 — ne donnait pas le controle au
    # modele, cela le lui retirait : plus rien ne bornait, et rien dans son
    # espace d'action ne le remplacait.
    # TOUS LES COMBIEN LE BUDGET PEUT-IL ETRE REVISE, en barres.
    #
    # Le budget est choisi a chaque decision, mais une decision n'existait que
    # si le budget COURANT laissait de la place : un modele qui choisit 3 % et
    # ouvre une position se verrouillait pour la vie de cette position, soit
    # ~1 140 barres avec un stop a 10xATR. Mesure : l'ancien gating bloquait
    # 99 % des barres, et le modele n'etait consulte que 44 fois par episode.
    #
    # Rouvrir la revision a CHAQUE barre leve le verrou mais coute cher :
    # 33 307 decisions au lieu de 530, et l'epoch passe de 3 a 12 minutes —
    # collecte 26 s -> 181 s, validation 96 s -> 293 s. Sur 160 epochs et
    # trois folds, 96 heures au lieu de 24.
    #
    # 12 barres de M5 = UNE HEURE. Le modele garde le moyen de sortir d'un
    # budget serre sans attendre la fermeture d'une position, et la lecture
    # est simple : c'est la frequence a laquelle un gerant revisite son budget
    # de risque. A zero, la revision est rouverte a chaque barre.
    # `revision_budget_barres` a ete retire le 2026-09-21. Il rendait la
    # parole au modele toutes les douze barres pour qu'il puisse DESSERRER
    # son budget sans attendre qu'une position se ferme. Sans budget, il
    # n'y a plus rien a desserrer : `peut_decider` se reduit a
    # `peut_entrer`, et la tete de cloture s'occupe des etats en position.
    _budget_retire_le: str = "2026-09-21"

    # LE BUDGET PAR DEFAUT, EN NOMBRE DE POSITIONS MINIMALES.
    #
    # Il valait 0.03 — 3 % de l'equite — et cette valeur etait INOPERANTE a
    # 1 000 $ de capital : une position au lot minimum y risque deja 4 a 15 %
    # du compte selon l'ATR, donc un budget de 3 % n'en ouvrait aucune. Voir
    # `BUDGETS_PART` dans saint_core pour la mesure.
    #
    # Il sert de valeur de repli quand la tete de budget n'a pas encore
    # choisi : UNE position, le plus petit palier qui trade.



    # `duree_trade_barres` A ETE RETIREE LE 2026-09-21.
    #
    # Elle comptait les occasions DISJOINTES derriere le sommet, donc elle
    # fixait la marge du portillon du hasard. Son propre commentaire
    # avertissait : « c'est exactement le genre de constante qui survit a
    # la mesure qui l'a produite ». C'est ce qui s'est passe.
    #
    # Elle valait 202 barres, mesure sous un stop suiveur a 1.0 R. Le
    # scalping M1 n'a ni stop ni trailing et borne la detention a
    # `horizon_cloture`. La constante a survecu au trailing
    # qui la justifiait, et le placement glouton comptait 126 occasions
    # independantes la ou il y en a ~6.7 fois plus : marge sur-estimee d'un
    # facteur racine(202/30) = 2.6, donc des points de reprise valables
    # refuses sans que rien ne le signale.
    #
    # LA DUREE SE LIT DESORMAIS DANS `_borne_syst`, la meme fonction qui
    # borne la grille et le diagnostic. Elle ne peut plus diverger de la
    # geometrie parce qu'elle EST la geometrie — c'est le seul reglage
    # qu'une mesure ne peut pas perimer.

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    # 84 -> 20, ET C'EST UN GAIN, pas une reduction.
    #
    # Le budget d'une epoch ne se compte pas en barres mais en DECISIONS :
    # c'est le seul endroit ou l'acteur recoit du gradient. Avant la
    # concurrence il fallait 483 840 barres pour en produire 580, parce que
    # la politique etait en position 99.9 % du temps. Avec plusieurs
    # positions, un episode de 5 760 barres en produit ~200.
    #
    #     84 episodes -> ~17 000 decisions, mise a jour PPO 29 fois plus
    #                    lourde qu'a K=1, soit 4 a 12 minutes par epoch
    #     20 episodes -> ~4 000 decisions, sept fois le gradient d'avant
    #                    pour le meme temps d'epoch
    #
    # On garde donc sept fois plus de gradient qu'a K=1 en divisant par
    # quatre les barres parcourues. Le cout d'une epoch reste celui d'avant ;
    # ce qui change est ce qu'on en tire.
    #
    # La couverture du marche par epoch baisse, mais le run compte 90 epochs
    # et les departs sont tires au hasard : la couverture cumulee augmente.
    # PLAFOND d'episodes, pas une consigne : le nombre reellement joue se
    # deduit de `cible_decisions` a chaque epoch. Voir ci-dessous.
    #
    # 40 -> 160, PARCE QU'IL MORDAIT. Le chiffre de 40 avait ete calibre sur
    # le BTC, dont un episode produit ~200 decisions : la cible de 4 000 etait
    # alors atteinte en 20 episodes et le plafond ne servait que de garde-fou.
    #
    # L'OR N'A PAS CETTE DENSITE. Son lot minimum vaut 4 265 $ de notionnel
    # contre 762 pour le BTC, donc le budget de risque n'autorise qu'une a
    # deux positions a 1 000 EUR ; le compte est alors en position ~99 % du
    # temps et un episode de 5 760 barres ne produit que ~25 decisions. Mesure
    # du 2026-09-17, entrees neutres tentees a CHAQUE barre possible, quatre
    # graines, episode d'entrainement complet :
    #
    #   budget   pos med   pos max   dec/episode   creux med   survie
    #       3%       2         3          16           11%      4/4
    #       6%       3         5          19           17%      4/4
    #       9%       4         6          22           19%      4/4
    #      12%       5         6          23           16%      4/4
    #      18%       6        10          29           27%      3/4
    #      30%       9        19          39           40%      3/4
    #
    # ELARGIR LE COMPTE NE REGLE DONC RIEN : dix fois le budget de risque ne
    # multiplie les decisions que par 2.4, pour quatre fois le creux et une
    # graine sur quatre qui meurt. La densite de decisions est bornee par la
    # DUREE des trades (28 h), pas par la capacite du compte.
    #
    # Le seul levier propre est donc le nombre d'episodes, et son cout est
    # lineaire : 40 episodes plafonnes donnaient ~1 000 decisions, soit le
    # QUART de la cible. L'acteur recevait quatre fois moins de gradient que
    # prevu — ce qui n'etait pas un choix, juste un plafond herite du BTC.
    episodes_per_epoch: int = 160

    # ------------------------------------------------------------------
    # LE BUDGET D'UNE EPOCH SE COMPTE EN DECISIONS, ET IL S'AJUSTE SEUL.
    #
    # Une decision est le seul endroit ou l'acteur recoit du gradient, et son
    # nombre par episode depend de la GEOMETRIE : a 12xATR un trade dure 34 h
    # et un episode de 20 jours en produit ~180 ; a 6xATR il dure 10 h et en
    # produit ~600. Fixer le nombre d'episodes revient donc a laisser le cout
    # d'une epoch varier d'un facteur trois a chaque changement de stop.
    #
    # C'est arrive : exec46 sortait une epoch en 306 s avec 3 620 decisions ;
    # le meme reglage a 6xATR en demandait ~12 000, soit ~17 minutes par
    # epoch, donc 76 heures pour 90 epochs et trois folds. Rien n'etait casse
    # — le budget etait exprime dans la mauvaise unite.
    #
    # Ici on vise un nombre de DECISIONS et on en deduit les episodes, a
    # partir de ce que l'epoch precedente a reellement produit. Le cout d'une
    # epoch devient constant, quelle que soit la geometrie, et aucun
    # changement futur ne le refera deriver.
    cible_decisions: int = 4000
    #
    # CETTE CIBLE COMPTAIT LES MAUVAISES CHOSES, ET LE RUN L'A MONTRE —
    # 2026-09-25.
    #
    # Une « transition versee » est toute decision prise a plat, ATTENTES
    # COMPRISES. Tant qu'un trade durait des heures, l'agent etait rarement
    # a plat et les deux comptes se ressemblaient. Des que la sortie a ferme
    # a la premiere minute, l'agent a ete a plat presque tout le temps : un
    # seul episode fournissait ~3 800 « decisions », et le regulateur a
    # conclu qu'un episode suffisait. Episodes joues : 7, 4, 2, 1, 1. Trades
    # d'entrainement : 8 028, 4 023, 2 428, 1 416, 1 014 — et vers 200 a la
    # selectivite de regime, pendant que la validation en jouait 4 400.
    #
    # LE REGULATEUR VISE DONC DES TRADES. C'est ce dont l'apprentissage se
    # nourrit : chaque trade ouvert est une trajectoire pour le modele de
    # sortie. `cible_decisions` ne dimensionne plus que l'estimation de
    # depart, avant toute mesure.
    #
    # 5 000, un peu plus que la validation (~4 400) : l'entrainement ne doit
    # pas voir moins de marche que la mesure qui le juge. Le budget de temps
    # ci-dessous reste la borne finale.
    cible_trades: int = 5_000

    # LE BUDGET DE TEMPS DE LA COLLECTE, EN SECONDES — et il prime sur la
    # cible de decisions.
    #
    # POURQUOI IL EXISTE. Le regulateur d'episodes vise `cible_decisions` en
    # divisant par le nombre de TRANSITIONS qu'un episode verse. Une decision
    # qui n'ouvre rien n'est pas versee : quand le budget de risque refuse
    # trois entrees sur quatre, un episode ne verse presque plus rien et le
    # regulateur reclame le plafond.
    #
    # MESURE, exec48 : l'epoch 1 joue 12 episodes en 31 s de collecte et verse
    # peu. Le regulateur demande alors 160 episodes, lisses a 86 — 7.2 fois
    # plus — et l'epoch 2 depasse quinze minutes contre deux. Une epoch qui
    # septuple d'un coup n'est pas un reglage, c'est une panne de cadence :
    # on ne peut plus rien lire, plus rien comparer, et la machine chauffe
    # pour simuler un agent qui attend.
    #
    # CE QUE LE PLAFOND FAIT. La cible de decisions reste la consigne, mais
    # elle est atteinte sur PLUSIEURS epochs au lieu d'une seule. C'est le bon
    # arbitrage : le gradient d'une epoch est moins riche, la cadence reste
    # lisible, et rien ne se decide sur une seule epoch de toute facon —
    # l'ecart-type de `sommet` d'une epoch a l'autre vaut 0.31.
    # 90 -> 300 -> 90 -> 40. LE DETOUR EST INSTRUCTIF, DONC IL RESTE ECRIT.
    #
    # Il avait ete porte a 300 pour nourrir `ActorL`, estime sur 23 a 49
    # positions fermees par epoch. Cette raison a disparu le meme jour :
    # `ppo_actif` est passe a False, il n'y a plus d'acteur a nourrir.
    #
    # CE QUE LA COLLECTE SERT ENCORE, ET RIEN D'AUTRE : produire
    # l'echantillon `pbs_epoch` qui calibre la barre d'entree, et les
    # statistiques de trades du journal. Elle n'entraine plus aucun poids.
    #
    # MESURE SUR exec70. Le trailing a 0.5R triple le nombre de positions
    # ouvertes et fermees par episode — 2 662 trades d'entrainement a
    # l'epoch 1 contre 0 a 95 sous 2.0R — donc la collecte est passee de
    # 32-59 s a 78-96 s, et l'epoch de ~3 a ~4.3 minutes.
    #
    # 40 -> 200, LE 2026-09-21, ET C'EST LA TETE DE CLOTURE QUI L'IMPOSE.
    #
    # « 40 s suffisent » etait vrai quand les positions duraient UNE barre.
    # Depuis que la tete de cloture decide, elles durent dix barres en
    # moyenne — et elle est consultee a CHAQUE barre en position. Il y a
    # donc desormais DEUX passes avant par barre au lieu d'une : celle des
    # tetes d'entree sur les environnements plats, celle de la cloture sur
    # ceux en position. Un episode est passe de 10.5 s a 56 s.
    #
    # CE QUE 40 s DONNAIENT ALORS, au journal :
    #
    #     epoch 2   4 episodes en 92 s   -> le temps en autorise 1
    #     epoch 3   1 episode  en 56 s   -> le temps en autorise 1
    #     epoch 3   567 trades d'entrainement contre 2 258 en validation
    #
    # Le regulateur ne pouvait plus finir un seul episode dans le budget.
    #
    # POURQUOI CA COMPTE, alors que la collecte n'entraine aucun poids.
    # Elle produit `pbs_epoch`, l'echantillon qui CALIBRE LA BARRE
    # D'ENTREE. A 603 decisions, un quantile a 5 % repose sur trente
    # observations. Et `sommet` s'est degrade en meme temps que la
    # collecte s'effondrait — +0.087, +0.023, -0.061 sur trois epochs —
    # pendant que le profit factor, lui, montait. Les deux ne mesurent pas
    # la meme chose : le second porte sur les trades joues, le premier sur
    # le classement, qui se calibre sur ce que la collecte rend.
    #
    # 200 s VISENT TROIS A QUATRE EPISODES. L'epoch passe d'environ 4 a 7
    # minutes ; c'est le prix d'une barre calibree sur quelques milliers
    # de decisions plutot que sur six cents.
    #
    # LE RETRAIT DU PLAFOND DE DETENTION A TOUT CHANGE. Sans horloge, une
    # position vit des heures, et comme l'environnement n'en tient qu'UNE
    # a la fois, il passe l'essentiel de son temps a ne pas pouvoir
    # decider. Mesure a l'epoch 1 du 2026-09-22 :
    #
    #     ENV [B 0.1 %  S 0.0 %  H 99.9 %  C 0.1 %]
    #     524 decisions sur 7 episodes, contre 7 468 avant le retrait
    #
    # Ce n'est pas un defaut, c'est l'arithmetique : le nombre d'occasions
    # vaut la duree de marche divisee par la duree d'un trade, et cette
    # derniere a ete multipliee par vingt. `sommet` se mesurait sur 66
    # occasions independantes — du bruit.
    #
    # LA REPONSE EST PLUS D'EPISODES EN PARALLELE, chacun tenant sa propre
    # position. Le budget passe a 600 s : la collecte est desormais
    # dominee par l'attente, pas par le calcul, donc un episode de plus
    # coute peu.
    secondes_collecte_max: int = 600
    # ET LA CROISSANCE EST BORNEE. Sans cela le premier ajustement saute au
    # plafond avant que la mesure de temps n'ait servi une seule fois.
    #
    # ET ELLE MONTE A 3.0 AVEC LE RETRAIT DU PLAFOND. A 1.5, partant de 7
    # episodes, il en faut huit epochs pour atteindre la centaine que la
    # detention longue reclame — huit epochs pendant lesquelles la tete de
    # rang se calibre sur du bruit. La borne existe pour empecher un saut
    # AVANT la premiere mesure de temps ; a 3.0 elle le fait encore.
    croissance_episodes_max: float = 3.0
    # ------------------------------------------------------------------
    # Idem pour la validation : 21-32 trades donnaient un Sortino purement
    # bruité (PF 3.10 puis 0.51 d'une epoch à l'autre), donc une sélection du
    # "best model" au hasard.
    # 48 au lieu de 16 : a selectivite figee a 5 %, le nombre de trades par
    # epoch tombait a 38 — soit +-7.9 points d'incertitude sur le winrate, ou
    # plus rien n'est distinguable de rien. Trois fois plus d'episodes rendent
    # la mesure exploitable sans toucher a la REGLE de decision.
    val_episodes: int = 40
    # 400 barres valaient 17 jours en H1 ; en M5 elles n'en font que 33
    # heures, soit moins qu'un seul trade median de 44 barres suivi de son
    # successeur. Un episode doit contenir assez de trades pour que le retour
    # cumule signifie quelque chose : 2 016 barres font une semaine de M5, donc
    # une quarantaine de trades medians.
    # 2016 -> 576. Mesure : des episodes cinq fois plus longs rendent
    # EXACTEMENT le meme nombre de decisions PPO (2 054 contre 2 195 en H1),
    # parce qu'une decision ne se cloture qu'a la fermeture de la position.
    # Les 2 016 barres coutaient donc 19 s de collecte par epoch pour rien.
    # 576 barres font 48 heures de M5, soit une quinzaine de trades medians par
    # episode — assez pour que le retour cumule ait un sens.
    # 576 -> 1 440, CONSEQUENCE MECANIQUE DU STOP, pas une seconde hypothese.
    # Le trade median passe de 44 a 147 barres : a episodes constants, les
    # decisions PPO tomberaient de 3 073 a ~900, et on relirait le manque
    # d'echantillons deja diagnostique. 336 x 1 440 = 483 840 barres, soit 73 %
    # de la fenetre d'entrainement a chaque epoch et ~2 300 decisions — le
    # regime H1, ou le modele voyait presque tout son jeu a chaque passage.
    # 1 440 -> 5 760, mesure du 2026-09-16, et c'est une CORRECTION DE DEFAUT,
    # pas un reglage.
    #
    # A la fin d'un episode l'environnement FERME LA POSITION AU MARCHE
    # (`done_reason = "episode_end"`). Avec la sortie au stop suiveur a
    # 12xATR, la duree d'un trade vaut 26 h en mediane mais 59 h en moyenne,
    # p90 a 148 h et une queue jusqu'a 30 jours. Un episode de 1 440 barres
    # fait 5 jours : 13.1 % des trades durent plus longtemps qu'un episode
    # ENTIER, et pour une entree uniforme dans l'episode 35.5 % sont coupes
    # avant leur sortie.
    #
    # Ce ne serait qu'un biais si la coupe frappait au hasard. Elle frappe
    # exactement l'inverse :
    #
    #     trades coupes     E[R] +0.5192   duree 127 h   n=31 253
    #     trades non coupes E[R] -0.2752   duree  22 h   n=56 780
    #
    # TOUT LE PROFIT DE CETTE GEOMETRIE EST DANS LES TRADES LONGS, et
    # l'environnement les fermait a cinq jours. Les trades de plus de cinq
    # jours, 13 % du total, rendent +0.905 R chacun. C'est pour les laisser
    # courir que l'objectif a ete retire le matin meme ; l'episode les coupait
    # quand meme, une barriere plus loin.
    #
    # C'est aussi un ECART TRAIN/LIVE. `max_holding_bars` vaut 0 avec ce
    # commentaire : "l'environnement ne simule pas une sortie que l'execution
    # ne sait pas faire". Or `episode_end` fait precisement cela — une
    # fermeture au marche dont le live est incapable. Le commentaire etait
    # contredit par le code trente lignes plus bas.
    #
    # 5 760 barres = 20 jours couvre le p95 (225 h) et ramene la coupe de
    # 35.5 % a environ 13 %. Le budget de collecte ne bouge pas : 84 x 5 760
    # fait les memes 483 840 barres que 336 x 1 440, donc le meme nombre de
    # decisions PPO et le meme nombre de trades par epoch. Ce qui change est
    # qu'ils vont au bout.
    episode_length: int = 5760      # 20 jours de M5
    # Fraction des états EN POSITION conservée pour la mise à jour PPO.
    # Ils sont masqués à HOLD donc sans gradient d'actor ; les garder tous
    # faisait passer la mise à jour de 40s à 7 minutes pour rien.
    in_position_keep_frac: float = 0.10
    # PLAFOND sur le nombre de DECISIONS utilisees par mise a jour PPO.
    #
    # Sans lui, le cout d'une epoch croit sans borne, et pour une raison
    # inscrite dans la conception : quand la selectivite descend, l'agent
    # refuse plus d'occasions, reste donc FLAT plus longtemps, et le nombre
    # d'instants ou il peut decider explose. Mesure sur un run reel, la
    # selectivite passant de 50 % a 23 % :
    #
    #     epoch 1    dec   7 438    maj PPO   16 s
    #     epoch 10   dec  10 964    maj PPO  125 s
    #     epoch 15   dec  31 582    maj PPO  429 s
    #     epoch 22   dec 129 429    maj PPO 1522 s
    #
    # Dix-sept fois plus de decisions, et la selectivite devait encore
    # descendre de 23 % a 5 %. Le run n'aurait jamais atteint la fin.
    #
    # Ce n'est pas qu'une question de vitesse : la mise a jour consomme TOUTES
    # les decisions collectees, donc elle faisait 232 pas de gradient a l'epoch
    # 1 et 4 044 a l'epoch 22. Le taux d'apprentissage et le coefficient
    # d'entropie suivent un calendrier PAR EPOCH — ils ne decrivaient plus ce
    # qui se passait, et cela explique la volatilite observee (+12.6 puis -1.3
    # entre deux epochs consecutives).
    #
    # Le sous-echantillonnage est uniforme et NON BIAISE : les avantages GAE
    # sont calcules sur les trajectoires completes avant d'arriver ici, donc
    # en retirer une partie ne fausse aucun calcul — c'est le meme argument
    # qui justifie deja `in_position_keep_frac`.
    # Un episode H1 de 400 barres donne au plus 400 decisions ; 96 episodes
    # en donnent ~38 000, dont la plupart en position. Le plafond garde son
    # role : rendre les epochs comparables entre elles.
    # Plafond du nombre de decisions retenues pour la mise a jour PPO. Les
    # episodes M5 etant cinq fois plus longs, la collecte en produit bien
    # davantage ; on releve le plafond en proportion pour ne pas jeter
    # l'essentiel de ce que la nouvelle echelle apporte.
    # LE PLAFOND QUI BORNE LA MISE A JOUR PPO, et il etait regle trop haut.
    #
    # MESURE, exec49 : l'epoch 1 produit 1 240 transitions et met a jour en
    # 9 s. Le cout est lineaire. A l'epoch 2 la barre de selectivite s'active
    # et chaque entree refusee verse une transition « attente » au tampon ;
    # le compte monte vers ce plafond, et la mise a jour a depasse HUIT
    # MINUTES contre neuf secondes — pour une epoch qui en prenait deux au
    # total. A 40 000, ce plafond ne bornait rien : il valait 32 fois ce
    # qu'une epoch saine produit.
    #
    # 6 000 tient la mise a jour sous la minute tout en laissant cinq fois
    # plus de transitions qu'une epoch de reference. Le sous-echantillonnage
    # porte sur les etats FLAT, tires au hasard : il reduit le volume, pas la
    # nature du signal.
    # 6 000 -> 3 000. C'EST CETTE BORNE QUI MORD, PAS LE TEMPS.
    #
    # MESURE SUR exec70 : des l'epoch 4 la collecte s'arrete a 6 000
    # decisions en 44 s, bien avant les 90 s autorisees. Baisser le plafond
    # de temps seul n'aurait donc plus eu aucun effet — c'est le genre de
    # reglage qu'on croit tourner et qui ne commande rien.
    #
    # COMBIEN EN FAUT-IL VRAIMENT. L'unique consommateur est le calibrage
    # de la barre : un quantile a 5 % sur `pbs_epoch`. A 3 000 decisions il
    # reste ~150 observations dans la queue, ce qui suffit largement pour
    # estimer ce quantile — `SeuilRang` lui-meme n'exige que 50 observations
    # avant de trancher.
    #
    # CE QUE CELA NE FAIT PAS : accelerer beaucoup. L'epoch dure ~258 s dont
    # ~70 de collecte ; la diviser par deux rend ~14 %. Le reste est la
    # validation (~70 s) et la passe de rang (~97 s), et les raccourcir
    # degraderait la mesure — ce qu'on vient precisement de passer la
    # journee a ameliorer.
    max_decisions_per_epoch: int = 3_000

    # LE PLAFOND DU LOT PPO ENTIER, etats plats ET etats en position.
    #
    # `max_decisions_per_epoch` ci-dessus ne borne que la part plate ; la part
    # en position vient de `in_position_keep_frac` et grandit avec le volume
    # collecte. Ce plafond-ci est le seul qui borne le TEMPS de la mise a
    # jour, puisque ce temps suit la taille du lot.
    #
    # ETALONNAGE : 1 753 transitions -> 13 s. 8 000 -> environ 60 s, ce qui
    # tient l'epoch sous quatre minutes avec le reste des phases.
    max_transitions_ppo: int = 8_000

    # LA SOUPAPE DE LA COLLECTE, ET POURQUOI ELLE N'EST PAS UN REGULATEUR.
    #
    # `max_decisions_per_epoch` et `max_transitions_ppo` plafonnent ce qui
    # ENTRE DANS LA MISE A JOUR. Ils ne plafonnent RIEN en amont : la collecte
    # empile tous les etats, et le sous-echantillonnage n'arrive qu'apres.
    #
    # CE QUE CELA A COUTE (exec61, epoch 2). Le modele a cesse d'entrer —
    # 43 029 refus sur 43 050 decisions — donc il est reste a plat, donc on
    # l'a interroge a chaque barre, donc le compte de decisions est passe de
    # 2 103 a 43 050. A 54 barres x 265 colonnes x 4 octets, cela fait 2.5 Go
    # rien que pour les etats. Le processus a ete tue : stderr vide, aucune
    # trace Python, la signature d'un manque de memoire.
    #
    # N'IMPORTE QUELLE EPOCH OU LE MODELE CESSE DE TRADER fait donc tomber le
    # run, quel que soit le reglage. Ce n'est pas propre a l'experience qui
    # l'a declenche.
    #
    # 20 000 EST UNE SOUPAPE, PAS UN REGLAGE. Une epoch saine en produit
    # 2 000 a 6 000 : le plafond ne mord jamais. Il ne sert qu'a empecher la
    # mort du processus dans le cas pathologique, et il se signale bruyamment
    # quand il tire — parce qu'une collecte tronquee biaise l'echantillon vers
    # le DEBUT des episodes, et qu'un biais silencieux serait pire que le
    # crash qu'il evite.
    #
    # 20 000 -> 400 000 LE 2026-09-25, et c'est `garde_attentes` qui le
    # permet. La soupape comptait des DECISIONS parce que chacune stockait
    # son etat. Desormais seule une attente sur vingt en stocke un : 400 000
    # decisions pesent ~20 000 attentes plus les trades, soit ~0.1 Go —
    # moins que les 20 000 d'avant a plein. Elle reste une soupape : elle ne
    # doit mordre que si l'agent cesse d'entrer.
    #
    # 400 000 -> 50 000 LE 2026-09-25, avec le PPO complet : chaque decision
    # garde de nouveau son etat (~5 Ko), donc 50 000 pesent ~0.25 Go — ce
    # que la machine peut encore donner pendant l'entrainement.
    plafond_collecte: int = 50_000

    # LA PART DES ATTENTES DONT ON GARDE L'ETAT. Une decision qui n'ouvre
    # rien stockait son observation entiere, ~5 Ko ; elles font 95 % des
    # decisions a la selectivite de regime. Leur SEUL lecteur, la mise a
    # jour PPO des entrees, a ete supprime le 2026-09-21 : le lot est
    # ramene a `max_decisions_per_epoch` puis n'est plus lu. Viser plus de
    # trades en stockant toutes les attentes aurait demande ~0.5 Go sur une
    # machine qui n'en a plus que 0.46 de libre — la mort memoire d'exec61.
    #
    # UN TIRAGE UNIFORME, donc la composition de ce qui reste ne change pas.
    # Les decisions qui OUVRENT sont toutes gardees.
    #
    # 1.0 DEPUIS LE PASSAGE AU PPO COMPLET, 2026-09-25. Les attentes sont
    # redevenues des ACTIONS de la politique d'entree : PPO en a besoin, et
    # l'avantage se calcule sur la suite COMPLETE des decisions de chaque
    # episode. En jeter dix-neuf sur vingt casserait cette suite. La memoire
    # est tenue par `plafond_collecte`, ramene en consequence.
    garde_attentes: float = 1.0
    # Nombre de passes PPO sur les données collectées.
    # Testé à 8 pour tenter de débloquer le KL (0.0004 contre un target de
    # 0.03) : sans effet sur le KL, resté à 0.0000, et le critique a divergé
    # (CriticL 9.6 → 1.7e6, ~500 pas d'optimisation sur le même batch).
    # Ce n'est donc pas le nombre de passes qui bride l'actor.
    # 4 -> 8 passes sur le meme lot. Avec seulement ~2 050 echantillons par
    # epoch, doubler les passes double les pas de gradient sans collecter une
    # seule barre de plus. Le sur-ajustement au lot est borne par l'arret sur
    # KL, qui coupe l'epoch des que la politique s'eloigne trop.
    updates_per_epoch: int = 8
    tp_shrink: float = 1.0  # pas de shrink (formule explicite : atr_tp_mult contient déjà le facteur final)

    # Batch reduit : avec ~1700 decisions par epoch, 256 ne donnait que 6
    # minibatches. A 128 on double le nombre de pas de gradient a donnees egales.
    batch_size: int = 128
    # GAMMA — a ablater, PAS a changer a l'aveugle.
    #
    # Le pas vaut UNE MINUTE, donc 0.97 correspond a une demi-vie de 23 minutes.
    # Poids restant d'une recompense selon l'horizon :
    #
    #     gamma     30min    60min   120min   240min   demi-vie
    #     0.970    40.10%   16.08%    2.59%    0.07%     23 min
    #     0.990    73.97%   54.72%   29.94%    8.96%     69 min
    #     0.995    86.04%   74.03%   54.80%   30.03%    138 min
    #
    # LE CRITERE N'EST PAS LA DUREE D'UN TRADE. L'ancienne justification de
    # cette ligne s'appuyait sur max_holding_bars = 240, qui vaut desormais 0
    # — et qui ne concernait de toute facon que 0.09 % des trades. Elle
    # generalisait un cas de queue a toute la distribution.
    #
    # Detention reellement mesuree sur BTCUSD a 2.0xATR : MEDIANE 7 barres,
    # moyenne 12. Poids restant du resultat d'un trade a l'entree :
    #
    #     gamma   7 barres  12 barres  32 barres   horizon   cycles
    #     0.970      80.8%      69.4%      37.7%     33 min      1.0
    #     0.990      93.2%      88.6%      72.5%    100 min      3.1
    #     0.995      96.6%      94.2%      85.2%    200 min      6.2
    #     0.999      99.3%      98.8%      96.8%   1000 min     31.2
    #
    # Meme a 0.97 le trade median garde 81 % de son poids : la duree n'est pas
    # la contrainte qui mord. Le semi-MDP la traite deja par son bootstrap
    # gamma^dt, et la recompense n'attend pas la cloture — log_ret est verse a
    # CHAQUE barre.
    #
    # Ce que gamma gouverne ici, c'est l'HORIZON D'OPPORTUNITE ENTRE TRADES.
    # L'agent est flat 91 % du temps et toute la strategie repose sur la
    # selectivite : la valeur d'attendre doit refleter les occasions futures.
    # Un cycle complet fait ~32 barres (~20 d'attente a 5 % de selectivite,
    # ~12 de detention). A 0.995 l'agent voit ~6 cycles ; le descendre pour
    # "coller" aux 7 barres d'un trade le ramenerait a un seul cycle et le
    # rendrait myope sur precisement ce dont la strategie depend.
    #
    # CHOIX : 0.995, par raisonnement sur l'horizon d'opportunite, TOUJOURS PAS
    # par mesure — aucune ablation n'a encore compare les valeurs entre elles.
    #
    # A SURVEILLER : un gamma plus haut augmente la variance des avantages et
    # l'echelle des cibles du critique. Si CriticL s'envole ou si quasi0
    # remonte, c'est le premier suspect.
    # ------------------------------------------------------------------
    # GAMMA EST EXPRIME PAR BARRE, et on a change deux fois ce que vaut une
    # barre sans y toucher : H1 -> M5, puis SL 4 -> 8xATR. Corrige le
    # 2026-09-16.
    #
    # CE QUE GAMMA FAIT ICI. La recompense est une plus-value latente payee a
    # CHAQUE barre, accumulee en `p["R"] += gamma**dt * reward`. Une barre
    # tardive du trade compte donc moins qu'une barre precoce — alors que rien
    # dans la strategie ne justifie de preferer un gain tot dans le trade : ce
    # sont les BARRIERES qui decident, pas la date.
    #
    # LA MESURE, 11 971 courses resolues sur le M5 a SL 8xATR / R:R 2.0 :
    #
    #             n    duree med   R moyen
    #     GAINS   3993     254 b    +2.00 R
    #     PERTES  7978     146 b    -1.02 R
    #
    # UN GAIN DURE 1.74 FOIS PLUS LONGTEMPS QU'UNE PERTE — c'est mecanique, le
    # take-profit est deux fois plus loin que le stop. L'escompte frappe donc
    # les gains plus fort que les pertes, et le R:R que l'agent optimise n'est
    # pas celui que l'environnement paie :
    #
    #     gamma      poids gain   poids perte   R:R effectif
    #     0.995        0.567         0.711          1.56      -20.2 %
    #     0.999        0.883         0.931          1.86       -5.1 %
    #     0.9999       0.987         0.993          1.95       -0.5 %
    #
    # A 0.995 le point mort VU PAR L'AGENT monte a 39.1 % quand celui que
    # l'environnement applique vaut 33.8 % : on lui demandait 5.3 points de
    # winrate de plus que ce qu'il fallait vraiment, et on le poussait a fuir
    # les configurations lentes a se resoudre — c'est-a-dire les gagnantes.
    #
    # (Le poids est calcule en supposant la plus-value accumulee uniformement
    # sur la duree, soit (1-g^dt)/(dt(1-g)). C'est une approximation au premier
    # ordre ; elle ne change ni le signe ni l'ordre de grandeur, qui tiennent au
    # seul rapport 254/146.)
    #
    # POURQUOI 0.9999 ET PAS 1.0. Gamma garde un role legitime : l'escompte du
    # bootstrap gamma^dt, qui dit jusqu'ou l'agent planifie au-dela du trade en
    # cours. A 0.9999 il vaut 0.982 sur un trade median et 0.865 sur un episode
    # de 1 440 barres — l'agent prefere encore, faiblement, gagner tot dans
    # l'episode. A 1.0 il n'y aurait plus aucune preference temporelle et le
    # critique perdrait son ancrage.
    #
    # EN H1 CE REGLAGE ETAIT SAIN : un trade durait 13 barres, 0.995^13 = 0.937,
    # la distorsion etait sous les 2 %. Ce n'est pas la valeur qui etait fausse,
    # c'est qu'elle n'a pas suivi l'echelle.
    # ------------------------------------------------------------------
    gamma: float = 0.9999
    lambda_gae: float = 0.95
    clip_eps: float = 0.18
    # ------------------------------------------------------------------
    # REGLAGES PPO RECALCULES POUR LE M5, le 2026-09-16.
    #
    # LE DIAGNOSTIC, lu dans le journal d'exec21 sur onze epochs entrainees :
    #
    #     KL        median  0.0002   pour une cible de 0.030
    #     clipfrac  median  0.0 %    aucune mise a jour n'atteint sa borne
    #     lratio    median  0.015    le ratio de politique bouge a peine
    #     advStd    median  1.34     contre ~2.24 en H1
    #     dec       median  2 054    identique au H1 malgre des episodes 5x
    #                                plus longs
    #
    # Le run utilisait 0.7 % du mouvement de politique qu'il s'autorise. Le
    # modele ne refusait pas d'apprendre : on ne le laissait pas bouger.
    # L'entropie tenait a 1.094 sur 1.099 apres quinze epochs, et l'ecart au
    # point mort etait DESCENDU a 3 points sous la politique gelee.
    #
    # DEUX CAUSES, toutes deux propres au changement d'echelle :
    #
    #   1. Le signal d'avantage est 40 % plus faible (advStd 1.34 contre 2.24).
    #      Une barre M5 porte douze fois moins de mouvement qu'une barre H1.
    #      Or `entropy_coef` est un coefficient ABSOLU : le bonus d'entropie
    #      pese donc 1.7 fois plus lourd, relativement, qu'il ne pesait en H1.
    #
    #   2. Les episodes cinq fois plus longs ne produisent PAS plus
    #      d'echantillons. Le semi-MDP ne cloture une decision qu'a la
    #      fermeture de la position, donc le nombre d'echantillons suit le
    #      nombre de CYCLES DE TRADE, pas le nombre de barres. Allonger les
    #      episodes a coute du temps de collecte sans rien apporter a
    #      l'apprentissage.
    #
    # LES QUATRE CHANGEMENTS SERVENT LE MEME BUT — laisser la politique bouger
    # — et c'est pourquoi ils partent ensemble malgre la regle habituelle d'une
    # variable a la fois. Les separer demanderait quatre runs pour corriger un
    # seul defaut, et le garde-fou ci-dessous borne le risque :
    #
    #   L'ARRET PRECOCE SUR KL REND CES CHANGEMENTS SURS PAR CONSTRUCTION. Si
    #   la divergence depasse 1.5 x target_kl, l'epoch s'interrompt d'elle-meme
    #   en plein milieu. On ne peut donc pas "trop" augmenter le pas : on peut
    #   seulement gaspiller du calcul, jamais casser la politique.
    # ------------------------------------------------------------------
    # 3e-4 -> 1e-3. La divergence KL croit a peu pres comme le CARRE du pas,
    # donc tripler le pas multiplie la KL par ~11 : elle passerait de 0.0002 a
    # 0.002, soit encore quinze fois sous la cible.
    lr: float = 1e-3
    target_kl: float = 0.03
    value_coef: float = 0.5
    # Coefficient d'entropie — plage usuelle PPO (1e-3 à 1e-2).
    # Effectif = entropy_coef × (1.5 → 0.5) sur 120 epochs, soit 0.012 → 0.004.
    # Relation utile pour 3 actions : Hflat = 1.0889 ⟺ p_max = 0.40, la valeur
    # que le modèle DOIT dépasser pour franchir son propre filtre de conviction.
    # Releve de 0.008 a 0.030. Mesure sur exec10 : a 0.008 l'entropie passait
    # de 1.099 a 0.495 en quinze epochs, et le resultat se degradait en
    # parallele. Le bonus ne retenait pas la politique, qui se figeait sur ce
    # qu'elle avait memorise d'un jeu de 16 708 barres.
    # 0.030 -> 0.015. Ce coefficient avait ete DOUBLE sur exec11 pour empecher
    # l'entropie de s'effondrer en H1 — et il avait marche. En M5 le terme de
    # politique-gradient est 40 % plus faible tandis que celui-ci reste absolu,
    # donc le meme reglage pousse 1.7 fois plus fort vers l'uniformite. Le
    # ramener de moitie retablit l'equilibre qu'il avait en H1.
    # 0.015 -> 0.003. MESURE SUR exec67, EPOCH 17 :
    #
    #     entropy_coef            0.015   x1.367 de rampe  ->  0.0205 effectif
    #     Hbudget                 1.698 / 1.792   = 95 % du plafond
    #     bonus d'entropie        0.0205 x 1.698  = 0.0348
    #     ActorL                  0.0118
    #
    # LE BONUS VAUT TROIS FOIS LE GRADIENT DE POLITIQUE. La perte de la tete
    # de budget est donc dominee par un terme dont le seul but est de MAINTENIR
    # LA DISTRIBUTION UNIFORME — et elle l'est, a 95 % de son plafond apres
    # douze epochs de PPO.
    #
    # LA REFERENCE QUI DIT QUE C'EST TROP. Sur exec40, la meme tete descendait
    # de 1.386 (son plafond a quatre paliers) a 1.054 — soit 76 % — et le creux
    # de validation tombait de 90 % a 23-50 %. Le budget APPREND sous PPO quand
    # le gradient lui parvient ; ici il ne lui parvient pas.
    #
    # POURQUOI DIVISER PAR CINQ ET NON PAR DEUX. Il faut que le terme de
    # politique DOMINE, pas qu'il egale : a 0.003 le bonus effectif vaut
    # 0.0041 x 1.698 = 0.0070, soit 0.6 fois ActorL au lieu de 3 fois. C'est
    # le rapport qu'il avait en H1, ou il fonctionnait.
    #
    # LE GARDE-FOU RESTE : le facteur x5 sous `entropy_budget < 0.1` rattrape
    # un effondrement, et c'est lui qui borne le risque de ce changement.
    entropy_coef: float = 0.003

    # Nombre d'epochs sans amelioration avant arret. Voir le commentaire de
    # `patience` dans run_training_on_split.
    # patience <= 0 desactive l'arret precoce. Voir la note sur `epochs`.
    patience: int = 0
    # (A) Clip très serré : 0.5 a laissé passer des gradients qui ont explosé
    # avec AMP (clip appliqué sur valeurs scaled). Maintenant 0.3 + unscale fix.
    # 0.3 -> 0.6. La norme observee vaut 0.90, donc chaque mise a jour etait
    # divisee par trois AVANT meme d'etre appliquee. Avec un pas triple, ce
    # plafond aurait absorbe l'essentiel du changement et l'aurait rendu
    # invisible — on aurait conclu que le pas ne sert a rien.
    max_grad_norm: float = 0.6

    # LE PLAFOND D'ECRETAGE DE LA PASSE SUPERVISEE, distinct de celui de PPO.
    #
    # POURQUOI IL EXISTE. `max_grad_norm = 0.6` est calibre pour des gradients
    # de POLITIQUE. La passe de rang, elle, produit une norme de 25.8 en
    # moyenne — 43 fois le plafond — donc l'ecretage mordait sur 100 % des
    # pas. Mesure, exec57 epoch 1 :
    #
    #     phase rang 80 s   AuxL 2.0148   gnorm 25.832   ecrete 100%
    #
    # CE QU'UN ECRETAGE PERMANENT CASSE. La norme du gradient cesse de
    # compter : chaque pas fait exactement 0.6, qu'on soit loin de l'optimum
    # ou dessus. L'optimiseur perd sa propriete fondamentale — les pas
    # retrecissent quand on approche — donc il ne converge pas, il tourne
    # autour. Une perte qui monte lentement est le symptome attendu, et c'est
    # ce qu'on observait : AuxL 2.0148 -> 2.3332 -> 2.4232 sur trois epochs,
    # avec 240 pas d'entrainement par epoch.
    #
    # Cela expliquait aussi l'effondrement de `etendue` : des pas de taille
    # constante dans un objectif invariant d'echelle font deriver les scores
    # vers une constante sans que la perte le sanctionne.
    #
    # 40 EST POSE AU-DESSUS DE LA NORME OBSERVEE, pas en dessous. Le role de
    # l'ecretage redevient celui d'un GARDE-FOU contre une pointe, et non d'un
    # regulateur permanent : Adam voit le vrai gradient et sa normalisation
    # fait son travail. `ecrete %` est au journal pour verifier qu'il tombe a
    # quelques pour cent — s'il reste a 100, le plafond est encore trop bas.
    # MESURE, exec58 : RELACHER L'ECRETAGE A EMPIRE LES CHOSES.
    #
    #     plafond 0.6    AuxL 2.0148   gnorm  25.832   ecrete 100%
    #     plafond  40    AuxL 2.5222   gnorm 149.360   ecrete  46%
    #
    # Le gradient a ete multiplie par 5.8 et la perte a monte. C'est de la
    # divergence au sens propre : des pas plus grands poussent les parametres
    # dans une region ou la pente est plus forte encore. L'ecretage a 0.6 ne
    # bridait donc pas l'apprentissage — il empechait l'explosion, et je
    # prenais l'equilibre d'un systeme RETENU pour un systeme etouffe.
    #
    # On revient donc au plafond serre, et on agit sur le PAS, qui est le
    # vrai coupable. Le reglage reste separe de celui de PPO : les deux
    # objectifs n'ont aucune raison de partager un plafond.
    max_grad_norm_rang: float = 0.6

    # LE PAS DE LA PASSE SUPERVISEE, ET SON PROPRE ADAM.
    #
    # DEUX PROBLEMES, UN SEUL CORRECTIF. La passe de rang partageait
    # l'optimiseur de PPO, donc son etat Adam : `m` et `v` sont estimes sur
    # des gradients de POLITIQUE (CriticL jusqu'a 221) puis appliques a un
    # objectif supervise qui n'a rien a voir (AuxL ~2). Les deux se
    # whipsawent — le gradient du tronc variait d'un facteur 85 d'une epoch a
    # l'autre.
    #
    # Un optimiseur dedie leur donne chacun leur etat, et permet de poser un
    # pas adapte : 1e-4 contre 1e-3 pour PPO. 240 pas par epoch sur un reseau
    # de 46 000 parametres, c'est beaucoup — le pas doit etre petit.
    #
    # CE QUI DOIT SE VOIR AU JOURNAL : `AuxL` qui DESCEND au lieu de monter
    # (2.0148 -> 2.3332 -> 2.4232 sur exec56), et `ecrete %` qui tombe sous
    # 100 sans qu'on ait touche au plafond — parce que le gradient se calme
    # de lui-meme quand les pas cessent de depasser.
    lr_rang: float = 1e-4

    # SAINT
    # 80 -> 8. Avec une seule tete, head_dim vaut 8, ce que
    # saint_core._verifie_dim_tete exige pour SDPA.
    d_model: int = 8
    # Frequences de l'embedding numerique par colonne. 16 en faisait le second
    # poste du reseau (29 %) ; 4 le ramene a un niveau comparable aux blocs.
    # 4 -> 2, LE 2026-09-21. C'EST LE POSTE PRINCIPAL DU RESEAU.
    #
    # LA REPARTITION, comptee module par module a 37 034 parametres :
    #
    #     embed     22 260   60.1 %   <- commande par `saint_n_freq`
    #     mlp        8 504   23.0 %
    #     col_emb    2 128    5.7 %
    #     norm       2 120    5.7 %
    #     blocks     1 984    5.4 %
    #     actor + tete_aux + critic   30   0.1 %
    #
    # L'embedding numerique par colonne pese a lui seul plus que tout le
    # reste. `n_freq` 4 -> 2 retire 9 010 parametres, soit 24 % du reseau —
    # aucun autre bouton n'en retire autant :
    #
    #     n_freq 2        28 024   x0.76
    #     mlp_dim 2       32 766   x0.88
    #     num_blocks 1    36 042   x0.97
    #     lookback 2      37 034   x1.00   (aucun parametre, que du calcul)
    #
    # POURQUOI IL FAUT REDUIRE. 1 338 occasions INDEPENDANTES par fenetre
    # d'entrainement sous `trail 1.0R`, pour 37 034 parametres : 28 par
    # occasion, quand on vise moins d'un pour dix. Et la memorisation est
    # mesuree, pas supposee — sur exec69 fold 1, `AuxL` descend de +0.022 a
    # -1.340 pendant que l'exces de validation s'effondre de +0.594 a
    # -0.008. C'est la courbe de sur-apprentissage dans sa forme classique.
    #
    # `d_model` NE PEUT PAS SERVIR. Avec une seule tete, `head_dim` vaut
    # `d_model`, et `_verifie_dim_tete` exige un multiple de 8 pour SDPA :
    # la construction REFUSE a 4 comme a 2. Ce n'est pas une precaution
    # theorique, c'est verifie.
    #
    # ON S'ARRETE A DEUX CHANGEMENTS. Empiler `n_freq 1`, `num_blocks 1` et
    # `mlp_dim 2` descendrait a 18 259 — mais on ne saurait plus lequel a
    # agi, et ce depot a deja paye ce genre de lot.
    saint_n_freq: int = 2
    saint_heads: int = 1
    # 16 -> 4, LE 2026-09-16 : la capacite avait double sans mesure.
    #
    # La tete pese 92 a 98 % du reseau et lit `n_features x d_model`, donc
    # passer de 103 a 260 colonnes l'a multipliee par 2.5 — et ajouter un
    # second reseau encore par 1.6. Budget mesure, pour 4 516 occasions
    # independantes :
    #
    #     mlp SAINT / patch    SAINT    PatchTST    total   par occasion
    #          16 / 32        62 548     35 776    98 324       21.8
    #           8 / 16        45 412     18 016    63 428       14.0
    #           4 /  8        36 892      9 328    46 220       10.2   <-
    #
    # Le seul ancrage empirique est le regime H1 qui avait rendu +2.2 en test :
    # 26 752 parametres pour 2 314 occasions, soit 11.6. A 21.8 on etait a
    # presque le double, sans qu'aucune mesure ne justifie que ce soit payable
    # — alors que la seule chose qui ait jamais deplace un resultat cote modele
    # dans ce depot est la REDUCTION de taille (493 796 -> 15 680 avait fait
    # passer PPO de -1.4 a +2.2).
    #
    # 10.2 encadre 11.6 par en dessous, du cote qui a marche. C'est un ancrage
    # herite, pas une mesure : la capacite ne se tranche pas sur une sonde
    # supervisee, il faut un run.
    saint_mlp_dim: int = 4
    # LECTURE PAR COLONNES plutot que par jeton CLS. exec19 a montre le
    # goulot : en lecture CLS, la tete recevait SEIZE nombres pour resumer 107
    # colonnes, contre 428 chez PatchTST. L'etendue des convictions plafonnait
    # a 0.028 apres 23 epochs — vingt fois moins que PatchTST au meme stade —
    # et l'entropie ne descendait pas sous 1.089 sur 1.099 : le modele n'avait
    # pas la place d'exprimer des convictions differentes selon les situations.
    #
    # En lisant les representations PAR COLONNE de la derniere bougie, la tete
    # recoit 856 nombres, et les blocs d'attention — qui ne coutent que 1 984
    # parametres — continuent de croiser les features. 26 752 parametres au
    # total, soit 11.6 par occasion independante contre 6.8 pour exec18.
    saint_lecture: str = "colonnes"

    # Profondeur du tronc — source unique : saint_core.N_BLOCS_DEFAUT.
    # 3 d'apres la configuration par defaut du FT-Transformer (Gorishniy 2021).
    # A ne PAS recopier en dur ailleurs : build_policy la deduit du checkpoint.
    # Deux blocs plutot que trois : a ce budget, chaque bloc pese 15 % du
    # reseau, donc la profondeur est un choix de taille autant que de capacite.
    num_blocks: int = 2

    # INTERSAMPLE ATTENTION, forme deployable — voir saint_core.ReferenceMemory.
    # Nombre d'observations de reference tirees de la fenetre de TRAIN, rangees
    # dans le checkpoint et consultees a chaque decision. 0 desactive la memoire.
    #
    # La version litterale de SAINT fait regarder au modele les autres lignes du
    # LOT. En live le lot vaut 1 : l'operation degenere et les poids appris sous
    # un lot de 128 se comporteraient autrement. Une banque figee donne la meme
    # capacite — comparer l'instant present a des situations historiques — en
    # restant identique a l'entrainement et en production.
    # MEMOIRE INTER-ECHANTILLONS DESACTIVEE POUR CE RUN. Elle est la seconde
    # chose que SAINT apporte, et une seule question a la fois : ce run teste
    # l'attention entre COLONNES contre l'independance des canaux. Ajouter la
    # banque de references melangerait deux effets, ce qui a deja coute une
    # nuit sur exec11.
    n_ref: int = 0

    # Trading
    initial_capital: float = 1000.0
    position_size: float = 0.06
    # Levier = contrainte de MARGE uniquement, jamais un multiplicateur de PnL.
    # MT5 calcule le PnL = delta_price × volume × contract_size.
    # 6.0 était un héritage du Bitcoin. Sur l'or il rendait la taille par le
    # risque IMPOSSIBLE : risquer 1.2 % de 1000 $ avec un stop à 5xATR (2.50 $)
    # demande 4.8 onces, soit 11 520 $ de notionnel — donc 1 920 $ de marge à
    # levier 6, au-dessus du capital. Le plafond de marge aurait rogné la
    # position d'un facteur 5.5 et annulé silencieusement la correction
    # d'échelle. Les courtiers offrent 100 à 500 sur XAUUSD ; à 100 la marge
    # requise tombe à 115 $, soit 11.5 % du capital, sous le plafond de 35 %.
    # Le levier ne multiplie toujours PAS le PnL : il borne le notionnel.
    leverage: float = 100.0
    # COMMISSION NULLE — ce courtier n'en facture pas sur BTCUSD CFD.
    #
    # A 0.0004 (4 bps du notionnel) le simulateur prelevait un cout qui n'existe
    # pas. Et il le prelevait au pire endroit : la commission etant
    # proportionnelle au NOTIONNEL, et le notionnel variant comme l'inverse de
    # la distance de stop a risque constant (size = risque / sl_dist), elle
    # penalisait surtout les stops serres :
    #
    #     SL       notionnel   commission   WR d'equilibre (R:R 1.4)
    #     2xATR     11 583 $      4.63 $         57.8 %
    #     5xATR      4 633 $      1.85 $         48.1 %
    #    10xATR      2 317 $      0.93 $         44.9 %
    #                         sans commission :  41.7 %
    #
    # Sur un risque de 12 $, 4.63 $ representent 39 % — l'agent apprenait donc
    # une economie bien plus dure que la realite, ce qui peut expliquer qu'il
    # atteigne presque systematiquement sa limite de pertes.
    #
    # Le cout reel de ce courtier est ENTIEREMENT dans le spread, mesure a
    # 2.61 bps et deja facture. Le slippage reste actif : c'est un cout distinct.
    #
    # A REVERIFIER si le compte change : un historique de deals MT5 donne la
    # commission reelle par lot (le champ `commission` de history_deals_get).
    fee_rate: float = 0.0
                                     # config validée +135 EUR MT5 tester
    # `max_drawdown` et `min_capital_frac` ONT ETE RETIRES le 2026-09-19.
    #
    # Ils terminaient l'episode a 40 % de creux et sous 20 % du capital de
    # depart. Aucun courtier n'applique l'un ni l'autre : ce qui termine un
    # compte reel est l'appel de marge, l'impossibilite de financer le lot
    # minimum, et l'equite a zero — les trois sont modelises dans `step`.
    #
    # Ce qu'ils coutaient, mesure sur 29 episodes a la selectivite de
    # validation : 26 % seulement des barres etaient jouees, et UN TIERS des
    # episodes tues auraient fini AU-DESSUS du capital de depart. On refusait
    # donc un checkpoint sur une observation tronquee, sans savoir ce qu'il
    # valait — et le modele, ne voyant jamais l'au-dela, ne pouvait pas
    # apprendre a se remettre d'un creux profond.
    #
    # La prudence n'est plus enseignee par une coupure mais par la PENALITE
    # DE CREUX, qui est graduee et que le modele voit venir.
    # ------------------------------------------------------------------
    # LA PENALITE DE CREUX EST UNE RAMPE, PLUS UNE FALAISE.
    #
    # Elle valait `-0.2 si creux > max_drawdown, sinon 0` : rien entre 0 et
    # 40 %, puis une marche a l'instant meme ou l'episode est tue. Aucun
    # gradient n'indiquait au modele qu'il approchait.
    #
    # CE QUE CELA PRODUISAIT, mesure le 2026-09-19 sur six episodes a
    # entrees neutres : CINQ SUR SIX meurent du garde-fou, au quart de leur
    # longueur. Et surtout —
    #
    #     recompense moyenne par barre sur les 200 dernieres : -0.0417
    #     recompense moyenne sur tout l'episode              : -0.0144
    #
    # — la politique PERDAIT au moment de mourir, et de plus en plus vite.
    # Couper l'episode lui evitait donc des milliers de barres negatives :
    # dans le mecanisme RL standard la terminaison coute le futur, mais ici
    # le futur etait negatif. LA MORT ETAIT UN SOULAGEMENT, et la marche de
    # -0.2 le seul contrepoids.
    #
    # LA FORME : `penalite_creux x d^2`, ou d est la distance au garde-fou
    # bornee a 1. Le carre la garde hors du chemin tant que le compte va
    # bien et la fait mordre pres de la coupure.
    #
    # LAMBDA SE RECALE SUR L'ECHELLE REELLE DE LA RECOMPENSE, et c'est le
    # deuxieme recalibrage de la journee. Le premier etait errone.
    #
    # HISTORIQUE, parce que l'erreur est instructive. La penalite valait
    # `-0.2 si creux > 40 %`. En la transformant en potentiel j'ai voulu
    # garder cette magnitude et j'ai ancre lambda dessus : d'abord 0.2 quand
    # `d` saturait a 40 % de creux, puis 1.25 quand `d` a ete etendu jusqu'a
    # la ruine (`lambda x 0.40^2 = 0.2`). Les deux fois j'ai ancre sur un
    # CHIFFRE, sans verifier a quelle echelle de recompense ce chiffre avait
    # ete regle.
    #
    # CE QUE LA MESURE A MONTRE, exec09 epochs 1 a 7 :
    #
    #     exec04 (budget de risque 3 %)          |R| ~ 2.8
    #     exec07 (budget 0, garde-fou a 40 %)    |R|   1.36   <- l'ancrage
    #     exec09 (budget 0, sans garde-fou)      |R|   6.77   (mediane)
    #
    # Le levier illimite a multiplie le signal monetaire par CINQ pendant que
    # la penalite restait callee sur l'ancienne echelle. Une position
    # traversant un creux de 0.5 a 0.7 payait 1.25 x 0.24 = 0.30 contre un
    # |R| de 7.6 : quatre pour cent du signal qu'elle devait contrebalancer.
    # Sur sept epochs le creux de validation n'a pas bouge d'un point —
    # 104.9, 100.3, 102.8, 100.4, 108.3, 103.8, 112.0 % — et le PnL MEDIAN
    # par episode est reste a -950 $ sur 1 000 $, epoch apres epoch.
    #
    # LE NOUVEL ANCRAGE EST UN RAPPORT, PLUS UN CHIFFRE : lambda suit |R|.
    # 1.25 x (6.77 / 1.36) = 6.2, arrondi a 6.5.
    #
    #     creux  25 % -> 0.41
    #     creux  40 % -> 1.04
    #     creux  70 % -> 3.19
    #     creux 100 % -> 6.50   (ruine)
    #
    # Une position traversant un creux de 0.5 a 0.7 paie desormais 1.56
    # contre |R| 7.6, soit ~21 % du signal. Senti, sans dominer.
    #
    # RESERVE, ET ELLE EST SERIEUSE. C'est du shaping, et ce depot l'a deja
    # paye : l'ancien shaping — quatre termes qui recompensaient le fait
    # d'ETRE en position — a enseigne une strategie perdante pendant
    # soixante-quinze epochs, et il a ete retire au profit d'un principe :
    # on ne garde que ce qui correspond a de l'argent reel, symetriquement.
    # Ce qui justifie l'exception ici est que le creux EST de l'argent reel,
    # et que la falaise creait une incitation manifestement perverse.
    # A surveiller : si le modele cesse d'ouvrir plutot que d'ouvrir mieux,
    # c'est ce terme qu'il faut soupconner en premier.
    # 0.05, ET NON 6.5. Le reglage herite valait 6.5 pour la forme A
    # POTENTIEL, ou le terme est une DIFFERENCE, donc minuscule par barre.
    # C'est desormais un COUT D'ETAT paye a chaque barre PENDANT L'EXPOSITION.
    # Un trade dure ~372 barres, donc la penalite effective par trade vaut
    # 372 x lambda x d^2 : a 6.5 elle depasserait mille, contre un terme en R
    # borne a +/-3. L'agent n'ouvrirait jamais.
    #
    # BALAYAGE DU 2026-09-20 (`mesure_lambda_creux.py`, recompense sommee
    # telle que l'environnement la rend, 6 episodes de 4 000 barres, entrees
    # neutres identiques d'un palier a l'autre) :
    #
    #   lambda     3 %      6 %     15 %     40 %  choisi  creux  gagnants punis
    #    0.000    +7.2     +5.6     -0.0     -4.7     3 %   9.7 %          3.7 %
    #    0.005    +7.2     +5.4     -1.2     -8.0     3 %   9.7 %          5.6 %
    #    0.015    +7.0     +5.1     -3.5    -14.7     3 %   9.7 %          5.6 %
    #    0.050    +6.6     +3.8    -11.7    -38.0     3 %   9.7 %          5.6 %
    #    0.150    +5.5     +0.0    -35.2   -104.6     3 %   9.7 %          7.4 %
    #
    # LE RESULTAT INATTENDU : a lambda = 0, SANS aucune penalite, l'objectif
    # choisit deja 3 % et l'ordre est monotone decroissant en levier. C'est la
    # correction du TERME EN R — moyenne au lieu de somme — qui a regle le
    # probleme a elle seule : 40 % passe de +113.9 a -4.7. La penalite de
    # creux ne decide plus du levier, et AUCUN lambda ne change le budget
    # choisi ni le creux.
    #
    # CE QUI FIXE LA LIMITE HAUTE n'est donc pas le budget mais le SIGNE DES
    # GAGNANTS : la part des trades gagnants dont la recompense cumulee est
    # negative. A 0.15 elle double (7.4 % contre 3.7 % sans penalite) — le
    # cout de chemin noie alors le resultat du trade, et l'acteur apprend que
    # gagner est mauvais. C'est l'invariant de signe de `test_concurrence`,
    # et il prime sur tout le reste.
    #
    # 0.05 EST LE COMPROMIS : frein six fois plus fort sur le levier maximum
    # que sans penalite (-38.0 contre -4.7), 8 % de recompense en moins au
    # palier retenu, et pas un gagnant puni de plus qu'a 0.005.
    penalite_creux: float = 0.05
    # ------------------------------------------------------------------
                                 # config validée +135 EUR MT5 tester

    # Risk management / position sizing
    # 1.2 % -> 0.53 %, mesure du 2026-09-16 sur l'enchainement CHRONOLOGIQUE.
    #
    # Avec la sortie au stop suiveur, 3 423 trades sur 7.7 ans en risque
    # fractionnaire compose :
    #
    #     risque/trade   creux max   gain final
    #         1.20 %       49.0 %      +1146 %
    #         0.60 %       27.7 %       +338 %
    #         0.53 %       25.0 %       +280 %   <- retenu
    #         0.30 %       14.8 %       +121 %
    #
    # 0.53 % est le risque qui ramene le creux maximal a 25 %. Au-dela, le gain
    # nominal grimpe vite mais le creux aussi : a 1.2 % il faut encaisser la
    # moitie du capital, ce qu'aucun dimensionnement raisonnable ne suppose.
    #
    # LE CREUX SE LIT DANS L'ORDRE CHRONOLOGIQUE, jamais melange. Un tirage
    # melange repartit les pertes au hasard alors qu'elles s'agglutinent dans
    # les regimes qui ne conviennent pas a la strategie : la premiere version
    # de cette mesure annoncait 135 R de creux la ou l'ordre reel en donne 52.
    risk_per_trade: float = 0.0053
    # max_position_frac : SUPPRIME. Il ne servait qu au plafond de notionnel,
    # qui est desormais exprime directement par max_notional_mult. Le laisser
    # aurait laisse croire a un reglage actif alors que plus rien ne le lisait.
    # Notionnel maximum, en multiples du capital. Independant du levier : voir
    # PPOEnv._compute_dynamic_size.
    max_notional_mult: float = 30.0
    position_vol_penalty: float = 1e-3

    # StopLoss / TakeProfit — SL=10×ATR, TP=14×ATR (R:R 1:1.4 conservé).
    #
    # Choisi sur mesure, pas au jugé. Un test supervisé hors RL (régression
    # logistique sur les 16 features) donne, pour la course TP/SL :
    #   SL  2×ATR (≈62$)  : AUC 0.5043, friction = 46.9% du gain visé
    #   SL  5×ATR (≈156$) : AUC 0.5213, friction = 18.8%
    #   SL 10×ATR (≈311$) : AUC 0.5292, friction =  9.4%   ← retenu
    #   SL 20×ATR (≈623$) : AUC 0.5338, friction =  4.7%
    # Le signal directionnel de ces features vit à 1-4h (AUC 0.529 à 1h, 0.535
    # à 4h). Un stop à 2×ATR(M1) est touché par le bruit bien avant que cette
    # tendance ne se réalise : la structure du trade détruisait l'information.
    # 10×ATR est le compromis — friction maîtrisée et détention ~5h, donc assez
    # de trades par épisode pour apprendre (≈13 contre ≈3 à 20×ATR).
    #
    # RECALIBRE POUR L'OR. La mesure precedente (3xATR / R:R 1.4) venait de
    # BTCUSD ; le cout relatif de l'or est different (0.250 x ATR contre 0.416)
    # et son optimum se deplace. Mesure sur 1.32 M bougies, PnL par trade en
    # unites d'ATR, non-resolus CLOTURES AU MARCHE (jamais jetes) :
    #
    #   SL | RR0.7   RR1.0   RR1.4   RR2.0    resolution
    #  1.0x| -0.021  -0.016  -0.013  +0.069     100.0%
    #  2.0x| +0.021  +0.053  +0.044  -0.034      99.9%
    #  3.0x| +0.009  +0.085  -0.013  +0.008      99.0%
    #  5.0x| +0.132  +0.027  +0.093  +0.317      92.3%   <- retenu
    # 10.0x| -0.268  -0.081  -0.220  -0.471      63.2%   <- rejete
    #
    # On ne retient PAS la case maximale : avec ~790 trades selectionnes et un
    # ecart-type de PnL d'environ 6 ATR, l'erreur type vaut +/-0.21 et le +0.317
    # n'est qu'a 1.5 sigma. Ce qui fait preuve, c'est que la ligne 5xATR est
    # positive sur SES QUATRE ratios — une ligne entiere du bon cote vaut bien
    # mieux qu'une case isolee. C'est l'erreur commise sur BTCUSD, ou le 10xATR
    # paraissait le meilleur avant que la clotures des non-resolus ne le rende
    # perdant.
    #
    # Le 10xATR est elimine sans ambiguite : negatif partout et seulement 63 %
    # de resolution — l'or n'a pas assez d'amplitude pour atteindre des
    # barrieres aussi lointaines en 240 min.
    # MESURE sur BTCUSD, plancher d'ATR corrige, friction complete, 28 653
    # candidats de validation, non-resolus clotures au marche, esperance par
    # unite de risque apres selection du meilleur 1 % :
    #
    #     SL   R:R    WR     requis    E[R]      t
    #     2.0  1.4   50.0%   47.3%   +0.1536   +3.0   <- retenu
    #     2.0  2.0   40.9%   39.0%   +0.1523   +2.5
    #     3.0  1.4   49.5%   45.6%   +0.1322   +2.7
    #     5.0  2.0   47.2%   35.7%   +0.0516   +1.1
    #
    # La derniere ligne dit pourquoi il ne faut PAS choisir sur l'ecart au
    # seuil d'equilibre : +11.5 points d'ecart, et presque rien au bout, parce
    # qu'a stop large la plupart des « gains » sont des sorties au temps
    # minuscules. L'esperance par unite de risque est le seul critere.
    #
    # Les trois premieres lignes sont a portee de bruit l'une de l'autre : le
    # choix du couple exact est moins solide que le choix du jeu de features,
    # dont les six premieres places du classement etaient toutes occupees par
    # les jeux larges.
    # STOP 2 -> 4 xATR, impose par le changement d'echelle. Mesure du
    # 2026-09-16, sortie sur barriere UNIQUEMENT comme l'environnement et le
    # live :
    #
    #     UT   SL   friction/R   E[R] hasard   duree med   occasions   requis
    #     M5  2.0      0.198R       -0.1684         12b      21 308   +0.1684R
    #     M5  4.0      0.099R       -0.0892         44b       5 811   +0.0892R
    #     M5  8.0      0.049R       -0.0425        147b       1 739   +0.0425R
    #
    # La friction est un montant FIXE en dollars ; l'unite de risque est la
    # distance au stop. En M5 l'ATR vaut 54.58 $ contre 223 $ en H1, donc un
    # stop de 2xATR y ferait payer 0.198 R par aller-retour — quatre fois plus
    # que l'avantage que le modele sait produire. Le 4xATR ramene ce cout a
    # 0.075 R tout en gardant 5 811 occasions independantes sur 2.5 ans, soit
    # ~15 000 sur les neuf ans telecharges.
    # ------------------------------------------------------------------
    # GEOMETRIE DES BARRIERES — SL 8xATR, corrige le 2026-09-16.
    #
    # L'ERREUR QUE CECI REPARE. La table qui avait fait choisir 4xATR comptait
    # les occasions de DEUX facons selon la ligne. Les mesures viennent du cache
    # M1, soit 3.6 ans ; le jeu M5 en couvre 9.05. La ligne retenue a ete mise a
    # l'echelle des neuf ans (5 811 -> 15 089), la ligne ecartee est restee a
    # celle des 3.6 ans (1 739), et c'est ce 1 739 qui l'a fait tomber sous le
    # seuil de ~4 900. A la meme echelle elle en vaut 4 516.
    #
    #   config         friction   horizon   occasions 9 ans   avantage requis
    #   H1  SL 2xATR    0.047 R     13.0 h            2 314        +0.0233 R
    #   M5  SL 4xATR    0.099 R      3.7 h           15 089        +0.0892 R
    #   M5  SL 8xATR    0.049 R     12.2 h            4 516        +0.0425 R
    #
    # CE QUI DECIDE : LA DETECTABILITE, (avantage - friction) x racine(N).
    #
    #   avantage brut   SL 4     SL 8
    #        0.09 R     0.10     3.19
    #        0.11 R     2.56     4.54
    #        0.13 R     5.01     5.88
    #        0.15 R     7.47     7.22
    #
    # Le croisement tombe a 0.1456 R. L'avantage mesure du modele vaut +0.09 a
    # +0.13 R : il est TOUT ENTIER du cote ou 8xATR gagne. A 0.09 R, le 4xATR
    # ne laisse meme pas 1 % de son avantage brut survivre a la friction.
    #
    # ET SURTOUT, L'HORIZON REDEVIENT CELUI DE LA MESURE. Les +0.09 a +0.13 R
    # ont ete mesures en H1, ou le trade median dure 13 heures. Le 4xATR en M5
    # dure 3 h 40 : on avait mesure un avantage sur un horizon et on l'a deploye
    # sur un autre, 3.5 fois plus court, en supposant qu'il suivrait. Rien ne le
    # garantissait — predire quatre heures et predire douze heures sont deux
    # problemes differents. Le 8xATR dure 12.2 h et rend la question identique a
    # celle qu'on savait resoudre.
    #
    # Le M5 garde alors son seul avantage reel sur le H1 : deux fois plus
    # d'occasions independantes, 4 516 contre 2 314, a friction et horizon
    # inchanges. Il ne reste aucun axe sur lequel le H1 lui soit superieur.
    # ------------------------------------------------------------------
    # 12 -> 6xATR. LA QUESTION A CHANGE AVEC LA CONCURRENCE, et la reponse
    # avec elle.
    #
    # Le balayage du matin comparait le rendement PAR TRADE, une position a
    # la fois. Le nombre de trades y etait plafonne par l'occupation — a 5 %
    # de selectivite une position etait ouverte 91 % du temps — donc
    # raccourcir la duree n'achetait rien et seule l'esperance comptait. 12x
    # gagnait, et c'etait juste sous cette contrainte.
    #
    # Les positions simultanees font sauter ce plafond. Ce qui compte n'est
    # plus le rendement par trade mais le rendement PAR AN :
    #
    #   stop  friction  duree  trades/an  E[R] sym      +/-   R/an  lot min
    #     3x    0.078R   2.8h        874   +0.0476  0.0160  +41.6    0.64%
    #     4x    0.059R   4.7h        418   +0.0658  0.0237  +27.5    0.85%
    #     6x    0.039R   9.8h        144   +0.0865  0.0417  +12.5    1.28%
    #     8x    0.029R  16.4h         75   +0.1447  0.0582  +10.8    1.70%
    #    12x    0.020R  34.1h         40   +0.2106  0.0748   +8.4    2.55%
    #    16x    0.015R  55.2h         30   +0.2660  0.0843   +8.1    3.40%
    #
    # Le rendement par trade monte toujours avec la largeur — la mesure du
    # matin n'etait pas fausse. Mais le rendement annuel fait l'inverse : on
    # perd un tiers par trade et on en gagne vingt fois plus.
    #
    # POURQUOI PAS 3x, QUI RAPPORTE CINQ FOIS PLUS. Parce que c'est un pari
    # sur le spread. A 3x la friction pese 0.078 R par trade, quatre fois
    # plus qu'a 12x, et l'avantage annuel s'effondre avec elle :
    #
    #   stop      x1    x1.5      x2      x3   (multiplicateur de friction)
    #     3x   +41.6    +8.5   -24.6   -90.9
    #     4x   +27.5   +15.6    +3.7   -20.0
    #     6x   +12.5    +9.7    +7.0    +1.5
    #     8x   +10.8    +9.8    +8.7    +6.6
    #
    # Or la friction supposee est probablement optimiste : 2.24 points de
    # base de spread releves sur MT5 contre 1.85 supposes, et les 3 points de
    # slippage sont une hypothese, pas une mesure. 6x est positif sous TOUTES
    # les hypotheses testees, jusqu'a trois fois la friction supposee.
    #
    # CE QU'ON GAGNE EN PLUS. La duree passe de 34 h a 9.8 h, donc plus aucun
    # trade n'est ampute par l'episode de 20 jours. Et le lot minimum du
    # courtier n'impose plus que 1.28 % de risque au lieu de 2.55 % — a
    # 1 000 EUR de capital, c'est la difference entre jouable et pas.
    #
    # RESERVE. Ces lignes ne sont PAS APPARIEES : chaque largeur utilise son
    # propre echantillon non chevauchant. La comparaison appariee du matin
    # donnait l'inverse par trade, ce qui est coherent — par trade le large
    # gagne, par an le court gagne — mais les deux ne se deduisent pas l'une
    # de l'autre.
    # 10xATR, RETABLI apres un detour instructif.
    #
    # Le balayage conjoint stop x trailing designait 4xATR avec un trailing de
    # 6 R : +42.5 R par an sous friction majoree contre +16.2 pour 10x/2R,
    # deux fois et demie mieux. Applique, puis mesure sur le CHEMIN DE
    # L'EQUITE — lot minimum, marge, appel de marge, ordre chronologique :
    #
    #   stop  trail  budget  pos  trades  tenue   gain  creux           fin
    #     4x     6R      3%    5     890  0.77a   +76%   40%   max_drawdown
    #     6x     3R      3%    3     546  0.94a   +33%   40%   max_drawdown
    #     6x     2R      3%    3     287  0.36a   -15%   40%   max_drawdown
    #    10x     2R      3%    2     257  1.14a   +39%   28%    fenetre finie
    #    10x   1.5R      3%    2     345  1.14a   +45%   20%    fenetre finie
    #
    # SEUL LE 10x SURVIT A LA FENETRE. Le 4x/6R gagne +76 % en neuf mois puis
    # rend 40 % et s'arrete ; son profil est "beaucoup de pertes a -1 R,
    # quelques gains tres gros", et la serie de pertes arrive avant les gains.
    #
    # L'ERREUR DE METHODE QUI A CONDUIT LA, et elle touchait tous les
    # balayages de geometrie de ce depot : le "R par an sur entrees non
    # chevauchantes" ADDITIONNE des R comme si leur ordre n'importait pas. Il
    # importe — une serie de pertes tue un compte avant que les gains
    # n'arrivent, et une somme ne peut pas le voir. Toute geometrie doit
    # desormais passer par `mesure_portefeuille` avant d'etre retenue.
    #
    # Le trailing reste a 2 R : le balayage le place sur un plateau de cinq
    # declenchements, ce qui est plus de donnees qu'un seul chemin d'equite.
    # Mais 1.5 R y donne +45 % pour 20 % de creux contre +39 % pour 28 % —
    # a departager sur plusieurs graines, ce qui n'est pas fait.
    atr_sl_mult: float = 10.0
    # R:R 1:2.0. Le 1.4 precedent venait de la grille de mesure_features.py,
    # qui echantillonne une entree toutes les 10 barres alors que les
    # barrieres mettent jusqu'a 240 barres a se resoudre : les fenetres de
    # resultat se recouvrent et gonflent l'esperance.
    #
    # Grille refaite en walk-forward PURGE (entrees espacees de 240 barres,
    # purge de 240 entre train et bloc, 8 blocs), esperance en unites de
    # risque a 2 % de selectivite :
    #
    #     SL 2.0 R:R 1.4   +0.055   5 blocs positifs sur 8
    #     SL 2.0 R:R 2.0   +0.267   6 blocs sur 8
    #
    # Effet principal : le point mort passe de 44.0 % a 35.7 %. Les runs
    # atteignaient 33.1 % — il leur manquait 11 points, il leur en manque 3.
    # R:R maintenu a 2 : l'objectif vaut le double du stop, donc 8xATR.
    # R:R 2.0 inchange : c'est le stop qu'on fait varier, pas le rapport.
    atr_tp_mult: float = 60.0   # inutilise tant que use_tp vaut False
    # L'OBJECTIF QUE LE MODELE APPREND A ATTEINDRE, en multiples d'ATR.
    # 16 = 2 R : la largeur sur laquelle la courbe de selectivite etait
    # franchement monotone, donc celle ou le classement a du sens.
    aux_tp_mult: float = 20.0   # 2 R, la largeur ou le classement a du sens

    # ------------------------------------------------------------------
    # LA REGLE DE SORTIE, mesuree le 2026-09-16 et changee pour cette raison.
    #
    # LE CONSTAT. Sur 7.7 ans de BTCUSD, entrees NON CHEVAUCHANTES espacees du
    # p75 des durees, sens alterne achat/vente, aucun modele :
    #
    #     regle                        E[R]   creux max   gain total
    #     SL/TP fixes (en place)    -0.0396      213 R       -156 R
    #     sans TP, trail 1.5/1.5    +0.0829       52 R       +284 R
    #
    # La geometrie en place PERD sur des entrees neutres. Le modele devait donc
    # compenser -0.0396 R avant de commencer a gagner. La regle sans objectif
    # rapporte +0.0829 R toute seule — les deux tiers de l'avantage que ce
    # depot cherchait a faire produire par un reseau.
    #
    # Verifie sur 12 phases d'entrees non chevauchantes, 81 700 trades :
    # +0.0718 +/- 0.0034 R, soit 21 ecarts-types, 12 phases positives sur 12.
    # Et la part SYMETRIQUE — derive du sous-jacent deduite — vaut +0.0638,
    # donc ce n'est pas la hausse du Bitcoin : le short seul est positif
    # (+0.0088) malgre un sous-jacent multiplie par 20 sur la periode.
    #
    # POURQUOI LE TP COUTAIT SI CHER. A 2 R il vend au douzieme du chemin les
    # trades qui vont loin — precisement ceux qui paient. La progression est
    # monotone : TP 2 R -> -0.029, TP 4 R -> -0.0005, pas de TP -> +0.003 a
    # +0.067 selon la distance du trailing.
    #
    # CE QUE LA MESURE NE DIT PAS. Elle porte sur une course aux barrieres en
    # numpy, pas sur cet environnement : l'ordre des extremes intra-barre, la
    # calibration et la taille par le risque n'y sont pas. Et c'est un suivi de
    # tendance — il vit et meurt avec l'existence de tendances. La part
    # symetrique vaut +0.0752 sur la premiere moitie de l'historique et +0.0294
    # sur la seconde.
    #
    # `use_tp = False` retire l'objectif ; le stop suiveur devient la seule
    # sortie. Le remettre a True redonne exactement la geometrie precedente,
    # et c'est le temoin de l'experience.
    # OBJECTIF REMIS, MAIS A 6 R — mesure du 2026-09-16, stop ADAPTATIF 8xATR,
    # entrees neutres, moyenne par annee sur neuf ans :
    #
    #     TP  2 R (ancien)     -0.0732 +/- 0.0287   1 annee positive sur 9
    #     TP  4 R              -0.0310 +/- 0.0208
    #     TP  6 R              -0.0042 +/- 0.0176   <- retenu
    #     TP 10 R              +0.0073 +/- 0.0171
    #     sans objectif        +0.0051 +/- 0.0171
    #
    # LA NEUTRALITE EST MONOTONE EN LA LARGEUR, et atteinte des 6 R. Au-dela,
    # 6 R, 10 R et l'absence d'objectif sont indistinguables : rien a gagner a
    # elargir davantage. Le coupable n'etait donc pas le take-profit en soi
    # mais sa LARGEUR — a 2 R il vend au douzieme du chemin.
    #
    # POURQUOI PAS "SANS OBJECTIF", QUI MESURE PAREIL. Parce qu'une cible
    # BORNEE garde la question prédictive bien posee. "Le prix ira-t-il 6 R en
    # haut avant 1 R en bas" a une reponse binaire a horizon defini, que les
    # features savent traiter. "Jusqu'ou ira la tendance avant de se retourner"
    # a une queue droite enorme dominee par la persistance du mouvement — et
    # c'est ce qui a APLATI la courbe de classement du modele : elle passait de
    # +0.56 au sommet a -0.12 au tout-venant sous l'ancienne geometrie, elle est
    # plate a +0.02 partout sans objectif.
    #
    # On cherche donc les deux proprietes ensemble : la neutralite de la
    # nouvelle geometrie et la predictibilite de l'ancienne.
    #
    # CE QUI N'EST PAS MESURE : que 6 R restaure effectivement le classement.
    # Le balayage de pente a manque d'echantillons par phase et n'a rien rendu
    # d'exploitable. C'est le pari de ce run, pas son acquis.
    # LA SORTIE N'A PAS D'OBJECTIF ; l'objectif sert de CIBLE a predire.
    #
    # exec35 a mesure que les deux proprietes recherchees — une geometrie
    # neutre et une cible classable — ne se reunissent pas en jouant sur la
    # largeur du take-profit : a 6 R la geometrie devient neutre (-0.0042
    # contre -0.0732 a 2 R) mais le gain d'apprentissage tombe a exactement
    # zero. La predictibilite de l'ancienne geometrie venait de ce qui la
    # faisait perdre.
    #
    # Elles se reunissent des qu'on cesse de les confondre : la POSITION court
    # jusqu'au stop suiveur, sans objectif, et le MODELE apprend a predire si
    # le trade aurait touche 2 R. Voir `cibles._regle_cible` pour la mesure qui
    # rend ce decouplage legitime — correlation de rang +0.892 entre les deux.
    use_tp: bool = False
    # LE STOP N'EST PLUS POSE — scalping M1, la sortie vient du modele.
    #
    # `sl_dist` reste calcule : il dimensionne la position et donne son
    # echelle a la recompense. Ce qui disparait est l'ORDRE STOP chez le
    # courtier, donc la borne automatique de la perte. `tete_cloture` la
    # remplace, et elle seule.
    use_sl: bool = False
    # ------------------------------------------------------------------

    # Microstructure — v2 (palier intermédiaire validé, 2026-05-20)
    # v3 stress était trop dur : modèle convergeait vers HOLD-always (degenerate).
    # v2 = friction réaliste qui permet l'apprentissage tout en restant exigeant.
    # MESURE sur 1 913 925 bougies BTCUSD (2022-12-14 -> 2026-09-13) :
    #   moyenne   2.607 bps  = 14.33 $ a un prix median de 66 583 $
    #   mediane   2.169 bps
    #   p90       4.437 bps
    #   p99       7.385 bps
    # BTCUSD coute donc 3.6x le spread de l'or (0.72 bps). C'est le handicap
    # structurel de ce symbole, et il pese directement sur le choix du SL :
    # aller-retour = 28.65 $, soit 27.7 % d'un stop a 3xATR et 8.3 % d'un stop
    # a 10xATR.
    # 2.61 -> 0.68 : le spread de l'OR, releve chez le courtier contre 2.23
    # pour le BTC. Garder celui du Bitcoin aurait facture a l'or une friction
    # trois fois trop chere — et comme elle est un montant FIXE, son poids en
    # unites de risque est ce qui decide de la largeur du stop et de la
    # rentabilite. Une erreur ici ne leve rien : elle rend seulement le
    # resultat faux.
    spread_bps: float = 0.68
    slippage_bps: float = 2.0    # sortie

    # Distribution bimodale du spread. Le facteur vient du rapport p99/moyenne
    # mesure ci-dessus (7.385 / 2.607 = 2.83), et non d'un choix : a 5.0 il
    # aurait simule un spread large de 13 bps, jamais observe.
    spread_wide_prob: float = 0.30
    spread_bps_wide_factor: float = 2.83

    entry_slippage_bps: float = 1.0  # entree
    # Extension bar.high/low pour capturer les wicks intra-minute.
    # PLAFOND CRITIQUE : doit rester tres en-dessous de atr_sl_mult x ATR, sinon
    # le bruit synthetique declenche le SL avant tout mouvement de marche reel.
    #
    # BTCUSD : ATR(14) M1 mesure = 34.46 $ a un prix median de 66 583 $, soit
    # 5.18 bps — nettement plus large, en relatif, que les 2.9 bps de l'or. Un
    # SL a 5xATR vaut donc 25.9 bps et un bruit moyen de 0.6 bps n'en represente
    # que 2.3 %, contre 4.2 % sur l'or. La valeur en bps est conservee telle
    # quelle : c'est deja une unite relative.
    # BRUIT DE TICK — DESACTIVE PAR DEFAUT.
    #
    # Il etendait artificiellement high et low de 0 a 1.2 bps, avec ce
    # commentaire : « simule les wicks intra-minute non capturees par
    # l'agregation M1 ». C'est faux. Une bougie MT5 porte DEJA le prix maximum
    # et minimum atteints sur la periode ; l'agregation perd l'ORDRE des
    # mouvements, pas les extremes. Il n'y a donc aucune meche a recuperer.
    #
    # Ce que le bruit faisait reellement, a 66 000 $ : ajouter jusqu'a 7.92 $ a
    # chaque extreme, donc declencher des stops et des objectifs qui n'auraient
    # pas ete touches. Et comme le moteur compte PERDANTE une bougie qui touche
    # les deux barrieres, l'effet net est un biais PESSIMISTE — sur un stop a
    # 2xATR (69 $), 7.92 $ representent 11 % de la distance.
    #
    # On le garde disponible pour les TESTS DE ROBUSTESSE (« le resultat
    # survit-il a une execution degradee ? »), jamais dans la reference. Le
    # spread et le slippage restent actifs : ce sont des couts d'execution
    # reels, distincts et mesures.
    tick_noise_bps: float = 0.0

    # Scalp
    # Detention de reference pour normaliser bars_held_norm — doit rester egale
    # a saint_core.SCALPING_MAX_HOLDING, que le live utilise pour construire la
    # meme feature. Les changer separement decale l'observation entre les deux.
    #
    # 120 venait des barrieres larges de l'or (5xATR : ~25 min pour toucher le
    # SL, une centaine pour le TP). A 2.0xATR sur BTCUSD la detention mediane
    # mesuree est de 7 barres : bars_held_norm valait ~0.06 pour un trade
    # typique, soit une entree d'observation qui ne portait presque rien.
    #
    # A 30 : ~0.23 pour un trade median, saturation au-dela de 90 barres
    # (environ 1 % des trades, 98.2 % se resolvant en 60 barres).
    #
    # ELLE N'EST PLUS ECRITE ICI. Elle vit dans `saint_core`, que le live
    # lit aussi — voir la mesure de saturation qui l'a portee a 480. Le
    # commentaire ci-dessus avertissait deja qu'un reglage duplique finit
    # par diverger ; il avait raison, et il manquait la TROISIEME copie,
    # en dur dans `cibles_m1`.
    scalping_max_holding: int = SCALPING_MAX_HOLDING

    # SORTIE PAR LE TEMPS — DESACTIVEE (0 = pas de plafond).
    #
    # Le live n'a aucun chemin de fermeture au marche : une position n'y sort
    # que par SL/TP chez le courtier. Un plafond present ici et absent la-bas
    # fait mesurer une strategie qu'on ne peut pas executer.
    #
    # Le retirer ne coute presque rien sur BTCUSD aux barrieres actuelles.
    # Mesure sur 250 000 entrees de la fenetre d'entrainement, SL 2.0xATR /
    # TP 2.8xATR : detention MEDIANE de 7 barres, 99.91 % des trades resolus
    # en 240 barres, 100 % en 1440. Le plafond interceptait moins d'un trade
    # sur mille, et ceux-la finissaient 52 % TP / 48 % SL — aucun biais a
    # preserver.
    #
    # Le motif d'origine venait de l'OR a SL 5xATR / TP 10xATR, ou l'agent
    # restait 99.9 % du temps en position (574 decisions par epoch contre
    # ~9500 sur BTC, critic loss 65 352 contre 46). A 2.0xATR ce regime
    # n'existe pas. Si on revient un jour a des barrieres larges, remesurer
    # AVANT de laisser ce champ a 0.
    #
    # Le biais de survie reste couvert : la fin d'episode liquide toute
    # position encore ouverte (voir _close_position / terminal_reason).
    max_holding_bars: int = 0

    # Break-even & trailing stop — DÉSACTIVÉ pour aligner sur le backtest no_be_trail
    # et sur le MQL5 (SaintV2_WF3 sans BE/trail).
    # LE TRAILING EST DESORMAIS LA SORTIE PRINCIPALE, pas un ajustement.
    # Sans lui ET sans take-profit, un trade n'aurait plus aucune sortie autre
    # que son stop initial : chaque position irait a -1 R ou courrait jusqu'a
    # la fin de l'episode. Les deux reglages vont ensemble.
    #
    # L'ancienne mesure du depot — "avec BE/trail PF 0.98, sans PF 1.85" —
    # declenchait a 1.0 et 1.5 ATR, dimensionnes pour un stop de 5xATR. Avec un
    # stop de 8xATR cela declenche apres 12 % du chemin vers le stop : le
    # moindre bruit scratche la position. Ce n'etait pas une mesure du
    # trailing, c'etait une mesure d'un trailing mal dimensionne.
    # LE STOP SUIVEUR EST COUPE AVEC LE STOP. Scalping M1 : la sortie est
    # une DECISION de `tete_cloture`, pas une geometrie. Laisser le trailing
    # actif ferait fermer des positions avant que la tete n'ait son mot a
    # dire — et on n'apprendrait jamais si elle sait le faire.
    use_be_trail: bool    = False
    # PAS DE BREAK-EVEN. Mesure : avec un trailing large il fait passer la part
    # symetrique de +0.0638 a +0.0167, et surtout il la rend instable — +0.0280
    # sur la premiere moitie de l'historique, -0.0262 sur la seconde. Un seuil
    # hors d'atteinte le neutralise sans toucher au code.
    # BREAK-EVEN DESACTIVE, ET CE N'EST PAS UN OUBLI.
    #
    # 1e9 est un seuil jamais atteint. Il a ete mesure avant d'etre laisse la
    # (2026-09-20, 28 099 occasions, train + validation, la fenetre de test
    # n'est pas lue) :
    #
    #   armement   R moyen   perte moy      p10   ecart-type   Sortino
    #   aucun       +0.228      -1.051    -1.10        2.418    +0.267
    #   0.25 R      +0.033      -0.283    -1.06        1.360    +0.070
    #   0.75 R      +0.143      -0.608    -1.09        1.995    +0.208
    #   1 R         +0.171      -0.726    -1.09        2.134    +0.231
    #   1.5 R       +0.212      -0.915    -1.10        2.327    +0.262
    #   2 R         +0.228      -1.051    -1.10        2.418    +0.267
    #
    # AUCUN NIVEAU NE RAPPORTE, et le Sortino — l'arbitrage exact qu'un
    # break-even propose, rendement contre risque de BAISSE — decroit de
    # facon monotone. A 2 R et au-dela il devient neutre, parce que le
    # trailing s'arme deja la et remonte le stop au-dessus de l'entree.
    #
    # LE CHIFFRE QUI EXPLIQUE TOUT EST LE p10. Le break-even divise la perte
    # MOYENNE par presque quatre, mais le dixieme percentile ne bouge pas :
    # -1.10 devient -1.06. Un trade qui part contre soi n'atteint jamais le
    # seuil d'armement et prend son -1 R entier. Le break-even ne coupe donc
    # QUE les trades qui sont d'abord alles dans le bon sens — les meilleurs.
    # A 0.25 R il en coupe 64 % a l'entree et fait tomber le p99 de +11.05 a
    # +6.47. Il ne protege pas des mauvais trades, il tue les bons.
    #
    # `mesure_break_even.py` rejoue la table si quelqu'un veut revenir dessus.
    atr_be_mult: float    = 1e9
    # 1.5 R -> 2.0 R, ET LE TRAILING N'AVAIT JAMAIS ETE CALIBRE.
    #
    # Les valeurs 1.5 / 1.5 venaient de `mesure_trailing`, dont le resultat a
    # ete RETIRE le meme jour : elle comparait un stop FIXE de 94 points de
    # base a un environnement a stop ADAPTATIF, et le +0.0829 R annonce
    # valait +0.0001 une fois corrige. La mesure est tombee, les parametres
    # etaient restes.
    #
    # Balayage a 6xATR, entrees non chevauchantes, hors test, rendement PAR
    # AN en unites de risque :
    #
    #   decl.   d=0.5  d=1.0  d=1.5  d=2.0
    #    0.5R   +22.8  +12.2   +9.8   +7.7
    #    1.0R   +10.9   +5.9   +7.6   +7.7
    #    1.5R    +7.3   +2.5   +7.3  +10.1   <- l'ancien reglage, dans un creux
    #    2.0R    +7.9   +7.1  +12.0  +12.2
    #    3.0R    +2.0   +7.5  +12.1  +11.7
    #
    # LE PIC A +22.8 EST UN MIRAGE. Il fait 783 trades par an de 3.5 h
    # chacun, donc la friction le devore. Avec une friction une fois et
    # demie plus chere — hypothese raisonnable, le spread releve sur MT5
    # valant 2.24 points de base contre 1.85 supposes :
    #
    #   decl.   d=0.5  d=1.0  d=1.5  d=2.0
    #    0.5R    +7.9   +5.9   +6.6   +5.8
    #    1.5R    +3.6   -0.5   +4.8   +8.3
    #    2.0R    +5.1   +4.7   +9.8  +10.5
    #    3.0R    +0.0   +5.8  +10.4  +10.3
    #
    # Le vrai optimum est un PLATEAU, pas un pic : quatre cases voisines —
    # declenchement 2.0 a 3.0 R, distance 1.5 a 2.0 R — s'accordent a +12 de
    # base et +10 sous friction majoree. Choisir dans un plateau concordant
    # n'est pas selectionner du bruit ; choisir le +22.8 isole l'aurait ete.
    #
    # Detail du reglage retenu : 91 trades par an, E[R] +0.1349 +/- 0.0590,
    # duree mediane 11.4 h. L'ancien reglage donnait +4.8 sous la meme
    # hypothese de friction, donc on double.
    #
    # La case voisine de l'ancien reglage (1.5 R / 1.0 R) vaut -0.5 : cette
    # zone de la surface est instable, ce qui explique qu'un reglage non
    # mesure y ait atterri sans que rien ne le signale.
    atr_trail_mult: float = 20.0   # 2.0 R (le stop vaut 10 ATR)   # gain en ATR pour déclencher le trailing
    atr_trail_dist: float = 20.0   # 2.0 R   # distance du trailing (en ATR)

    # Warmup critique : N epochs où seul le critique est mis à jour
    critic_warmup_epochs: int = 5
    # LE DRAPEAU `gel_direction` A DISPARU AVEC LA DIRECTION.
    #
    # Il detachait le terme de direction du ratio de PPO pour que seul le
    # budget apprenne. Le geler laissait la tete DECIDER encore, en rollout,
    # pendant que le deploiement decidait par la tete de rang : le desaccord
    # entre entrainement et deploiement restait entier, c'est meme lui qui
    # rendait le gel presque sans effet visible. La tete est maintenant hors
    # circuit des deux cotes, et il n'y a plus rien a geler.


    # Seuil minimum de trades en val pour sauvegarder le meilleur modèle
    min_val_trades_save: int = 20

    # Diagnostic d'amplitude faible. Le train conserve son exploration;
    # la validation garde son budget d'entrees sauf comparaison legacy explicite.
    pbs_etendue_min: float = 0.01
    # Comparaisons experimentales : ne pas ouvrir tout lorsque le signal est plat.
    # SELECTIVITE DE VALIDATION — FIGEE.
    #
    # A None, elle suivait celle du training, qui descend de 50 % a 5 % sur 40
    # epochs. Mesure sur un run : 50 % a l'epoch 1, 37.6 % a l'epoch 12. Une
    # amelioration du PnL melangeait donc l'apprentissage du modele et le
    # resserrement du filtre — deux causes qu'on ne peut pas separer apres coup.
    #
    # Figer les fenetres ne suffisait pas : c'est la REGLE de decision qui
    # devait l'etre aussi pour que deux checkpoints soient comparables.
    #
    # Valeur : la cible finale du calendrier d'entrainement, donc le regime
    # qu'on cherche reellement a atteindre. Elle doit suivre
    # `SELECTIVITE_FIN` — si les deux divergent, on entraine vers un regime
    # et on mesure dans un autre, et la comparaison entre checkpoints ne
    # veut plus rien dire.
    #
    # Repassee a 5 % apres un essai a 16.18 % : voir le tableau sous
    # `SELECTIVITE_FIN`. Elle doit rester egale a `SELECTIVITE_FIN`.
    validation_selectivity: Optional[float] = 0.05
    # Part de la fenetre de validation reservee a la CALIBRATION des seuils.
    #
    # Les seuils etaient calibres sur les MEMES episodes que la passe 2 evaluait
    # ensuite — l'etat du generateur aleatoire etait meme sauvegarde et restaure
    # pour que les deux passes portent sur des fenetres identiques. Le quantile
    # de conviction resumait donc toute la periode AVANT que ses trades ne
    # soient simules. En live, cette information n'existe pas : on calibre sur
    # le passe et on trade la suite.
    #
    # On decoupe donc la fenetre en deux dans l'ordre du temps : la premiere
    # partie calibre, la seconde evalue, et les seuils sont figes entre les deux.
    calib_frac: float = 0.35
    # Taille de la fenetre glissante du filtre par RANG, en nombre d'occasions.
    #
    # Mesure du taux d'acceptation reel pour une cible de 5 %, sous trois
    # regimes de derive de l'echelle des convictions :
    #
    #     fenetre    stable   derive x26   resserrement
    #        250      5.1%       6.6%          4.8%
    #        500      5.2%       7.9%          4.4%
    #       2000      4.8%      15.0%          6.3%
    #       5000      4.8%      24.6%         11.1%
    #
    # Les grandes fenetres retardent sur la distribution courante. 500 garde
    # 25 echantillons au-dessus de la barre — assez pour estimer un quantile —
    # tout en restant a moins de trois points de la cible sous une derive
    # bien plus violente que celle observee.
    # En OCCASIONS, pas en barres. A 5 % de selectivite sur H1, 500 occasions
    # representent des mois : on raccourcit pour que le rang suive le regime.
    # Fenetre du filtre par rang glissant, en BARRES. 200 barres valaient huit
    # jours en H1 ; en M5 elles ne font que 17 heures, trop court pour que le
    # quantile decrive autre chose que la seance en cours. 2 016 barres — une
    # semaine — rendent la meme portee temporelle qu'en H1.
    rang_fenetre: int = 2016
    # Graine dediee aux fenetres de validation. Elles etaient tirees au sort a
    # chaque epoch : une amelioration pouvait venir d'un scenario plus facile
    # plutot que d'un meilleur modele. Fixees, les epochs deviennent comparables.
    val_seed: int = 20260914
    legacy_flat_validation: bool = False
    evaluate_test: bool = True
    # LA VALIDATION ET LE TEST TIRENT LES ACTIONS DE LA POLITIQUE —
    # choix du proprietaire, 2026-09-25, au vu des deux premieres epochs du
    # PPO complet.
    #
    # En argmax, une politique encore indecise — un tiers par action a
    # l'entree, 50/50 a la sortie — se jouait comme une regle absolue : la
    # vente legerement devant, la validation vendait a CHAQUE minute a
    # plat ; la fermeture legerement devant, elle fermait a la minute
    # suivante. 46 046 shorts d'une minute, zero long, pendant que la
    # politique entrainee achetait 28 % du temps. La strategie jugee
    # n'etait pas celle que PPO apprend.
    #
    # EN TIRANT, la validation mesure exactement ce que PPO optimise :
    # l'esperance sous la politique. Le tirage suit un GENERATEUR A GRAINE
    # FIXE, le meme a chaque epoch : deux epochs se comparent sur les memes
    # aleas, et le generateur global de l'entrainement n'est pas touche.
    # Quand la politique deviendra confiante, tirage et argmax convergent.
    #
    # FALSE — L'ACTION LA PLUS PROBABLE, decision du proprietaire le meme
    # soir : pas de hasard en live, donc pas de hasard en validation ni au
    # test, qui doivent juger la regle qu'on deploierait. On laisse la
    # politique apprendre a se decider, longs et shorts. Le tirage reste
    # disponible ici, mais il n'est plus la regle.
    evaluation_stochastique: bool = False
    # PPO requiert les probabilites de la politique qui a tire les actions.
    # L'ancien curriculum forcait BUY/SELL ou remappait en HOLD sans corriger
    # logprob. Conserve uniquement pour reproduire les anciens diagnostics.
    legacy_off_policy_curriculum: bool = False

    # Curriculum vol
    use_vol_curriculum: bool = True

    # Cache disque du dataframe fusionné (évite ~15 min de recalcul d'indicateurs
    # à chaque lancement). Invalidé si le cache accuse plus de N heures de retard.
    use_data_cache: bool = True
    # 48 HEURES, la valeur d'origine, retablie le 2026-09-19.
    #
    # Elle avait ete relevee le temps que le terminal MetaTrader — fraichement
    # installe — rapatrie son historique. Il etait alors plafonne en nombre de
    # barres et ne rendait rien au-dela de 20 000 en M5, contre 493 497 dans
    # `data_cache_XAUUSD_M5.pkl` : jeter le cache a ce moment-la aurait
    # remplace sept ans d'historique par trois semaines, sans autre signe
    # qu'un run plus rapide. Le plafond du terminal est passe a ILLIMITE, donc
    # le garde-fou peut reprendre son role.
    #
    # Ce qu'il protege : un ENTRAINEMENT lance en croyant porter les barres du
    # jour alors que le cache en a des anciennes. Ce qu'il ne protege pas : le
    # LIVE, qui ne passe pas par ce cache — `kairos_live` lit `flux_live` et a
    # sa propre exigence de fraicheur.
    #
    # LE RAFRAICHISSEMENT DE L'OR PASSE PAR `prepare_or.py`, qui lit du M5.
    # Le chemin MT5 de `load_mt5_data` demande du M1 : c'est l'heritage du
    # BTC, et il ne sait pas reconstruire le jeu de l'or.
    #
    # Pour travailler hors ligne sans toucher a cette valeur, relever
    # `cfg.data_cache_max_lag_hours` sur l'INSTANCE au lancement du script
    # concerne : le garde-fou reste entier pour tous les autres.
    # RELEVE DE 48 A 240 HEURES LE 2026-09-21, et ce n'est pas du confort.
    #
    # A 48 h, le cache a expire pendant une session de travail et le
    # rechargement depuis MT5 a leve une exception : le chemin de
    # rechargement ne produit PLUS les ~230 colonnes Ichimoku, range et H4 —
    # le journal n'affiche meme aucune ligne H4. Plus aucun run ne pouvait
    # demarrer, et la panne ne correlait avec aucun changement de code
    # puisqu'elle attendait l'expiration d'un cache pour se declarer.
    #
    # POURQUOI 240 H EST SANS CONSEQUENCE POUR L'ENTRAINEMENT. Les donnees
    # couvrent 2022-12-15 a aujourd'hui, soit ~3.8 ans. Dix jours de queue
    # manquants, c'est 0.7 % du jeu, et ils tombent APRES la fenetre de test —
    # donc ils n'entrent ni dans l'entrainement, ni dans la validation, ni
    # dans la mesure. Le LIVE, lui, ne lit pas ce cache : il interroge MT5 a
    # chaque bougie.
    #
    # CE QUE CELA NE CORRIGE PAS : le chemin de rechargement reste casse. Il
    # faudra le reparer avant que ce cache-ci ne devienne vraiment vieux.
    data_cache_max_lag_hours: float = 240.0

    # Device
    force_cpu: bool = False
    # AMP réactivé. Je l'avais coupé en soupçonnant la fp16 d'écraser le
    # gradient de l'actor : l'écart venait en réalité du warmup critique, qui
    # exclut volontairement actor_loss de la loss pendant 5 epochs. Mesuré
    # après warmup : gActor passe de 2.45e-04 à 5.36e-01, soit la valeur
    # théorique attendue. La fp16 n'y était pour rien.
    use_amp: bool = False  # FP32 reference: avoid the observed FP16 gradient overflows.
    resume_weights: bool = False  # Explicit warm start, not a full optimizer resume.

    # Spécialisation d'agent (mode "close" supprimé)
    # "both"  → BUY + SELL + HOLD
    # "long"  → seulement BUY1 / HOLD
    # "short" → seulement SELL1 / HOLD
    #
    # C'EST ICI QUE LE COTE SE DECLARE, et nulle part ailleurs. Il valait
    # "both" pendant que le bloc __main__ posait `cfg_long.side = "long"` :
    # tout ce qui lisait `PPOConfig()` sans passer par ce bloc — le resolveur
    # de checkpoints, le live, l'interface — croyait donc le pipeline
    # bilateral. Le resolveur ne cherchait que des fichiers `*_both_*` et
    # proposait le dernier run BTC bilateral a un live long-only sur l'or.
    #
    # Un reglage duplique finit toujours par diverger ; celui-ci l'avait
    # deja fait sans lever d'erreur. Le bloc __main__ le repete desormais a
    # l'identique, ce qui est une redondance visible et non un second
    # reglage.
    # LE VETO DE TENDANCE : on n'achete que si le marche monte sur un mois.
    #
    # MESURE DU 2026-09-20 (`mesure_seuil_tendance.py`, 27 354 occasions,
    # train + validation, la fenetre de test n'est pas lue) :
    #
    #   seuil                gardees   R par trade   R par occasion   sigma
    #   aucun filtre          100.0%        +0.247           +0.247
    #   momentum >=  0 %       54.0%        +0.412           +0.223    +1.5
    #   momentum >= +1 %       46.3%        +0.468           +0.217    +1.8
    #   momentum >= +3 %       33.6%        +0.483           +0.162    +1.6
    #   momentum >= +8 %        9.0%        +0.102           +0.009    -0.7
    #
    # L'ARBITRAGE, ET POURQUOI IL PENCHE DU COTE DU FILTRE. Le rendement PAR
    # TRADE passe de +0.247 a +0.412, presque le double. Le rendement PAR
    # OCCASION baisse, parce qu'on laisse passer la moitie des occasions et
    # qu'une occasion refusee ne rapporte rien. C'est le premier qui compte
    # ici : le budget borne le nombre de positions SIMULTANEES et la regle
    # deployee ne retient deja que 5 % des occasions. On ne manque pas
    # d'occasions, on manque de places — et remplir une place avec un trade a
    # +0.41 R plutot qu'a +0.25 R est un gain net.
    #
    # ZERO PLUTOT QUE +1 %, bien que +1 % mesure un peu mieux (+0.468 a 1.8
    # sigma contre +0.412 a 1.5). Zero ne se regle pas : c'est "le marche
    # monte-t-il sur un mois, oui ou non". +1 % serait un parametre de plus a
    # sur-ajuster pour 0.3 sigma.
    #
    # CE QUE CE VETO N'EST PAS. Il ne designe pas un cote. Meme en tendance
    # baissiere, l'achat reste meilleur que la vente sur cette fenetre
    # (+0.049 contre -0.244 R) : le regime gradue la FORCE de l'avantage a
    # l'achat, il ne le renverse jamais. C'est pourquoi il s'applique ici en
    # ABSTENTION, pas en bascule vers la vente.
    # LE VETO EST RETIRE : C'EST LA TETE DE BUDGET QUI S'ABSTIENT MAINTENANT.
    #
    # Il refusait toute entree en momentum mensuel negatif — 46 % des
    # occasions — et il etait ecrit a la main, avec un seuil fixe. Depuis que
    # `BUDGETS_PART` contient 0, le modele APPREND quand ne pas trader, par
    # la meme tete qui apprend combien risquer, et sur le meme signal.
    #
    # Les trois colonnes de tendance restent dans l'observation : le modele
    # peut donc toujours conditionner son budget sur le regime, mais c'est
    # LUI qui decide du seuil et non plus nous.
    veto_tendance: str = ""   # "" pour desactiver
    veto_tendance_seuil: float = 0.0
    # LES DEUX COTES DEPUIS LE 2026-09-21, et c'est le scalping qui
    # l'impose. Le long-only venait de l'or tenu en tendance ; sur trente
    # minutes de M1 il n'y a pas de tendance a suivre, et il coutait deux
    # choses.
    #
    # LA TETE DE VENTE ETAIT MORTE A LA NAISSANCE. `cotes_permises("long")`
    # rend (True, False), et ce masque multiplie la moindre carre de la
    # passe supervisee : la colonne VENDRE recevait exactement zero
    # gradient, et `build_mask_from_pos_scalar` la fermait aussi au
    # deploiement. La tete demandee existait dans le reseau et n'etait ni
    # entrainee ni lue.
    #
    # ET LA MOITIE DES OCCASIONS ETAIT JETEE. Mesure du 2026-09-21 sur la
    # fenetre de validation du fold 1, 7 146 occasions, cible nette a 30
    # minutes : achat p99 +1.92 sigma, vente p99 +1.94 sigma. Les deux
    # queues sont de meme taille. En n'en gardant qu'une, on divisait par
    # deux le nombre de trades possibles sans aucune raison mesuree.
    # L'ACHAT SEUL, ET C'EST UNE MESURE QUI L'IMPOSE, PAS UNE PRUDENCE.
    #
    # Le 2026-09-22, les deux cotes ont ete cherches separement sur 23 mois
    # de BTC et 30 mois d'or, a 120, 240 et 480 minutes, avec et sans
    # conditionnement par le flux. Resultat, horizon 480, BTC :
    #
    #     achat du creux   (decile bas)   avantage  +6.48 bps   17/23 mois
    #     vente du sommet  (decile haut)  avantage  -2.58 bps    aucun
    #
    # LE DECILE HAUT MONTE AUSSI. Ce n'est donc pas de la reversion — sinon
    # il baisserait. C'est un marche a demande permanente : les creux sont
    # rachetes, les exces ne sont pas vendus. Il n'y a rien a vendre, a
    # aucun horizon, dans aucune condition testee.
    #
    # ET JOUER LES DEUX COTES COUTE. Une tete sans avantage entre quand
    # meme, paie l'aller-retour — 4.36 bps mesures — et occupe la seule
    # position disponible pendant que le creux suivant passe.
    #
    # LES DEUX COTES A NOUVEAU — une decision du proprietaire, 2026-09-25.
    #
    # La mesure ci-dessus tient toujours, et elle est a 120-480 minutes :
    # vendre le sommet n'y rapportait rien. Mais le run est passe au
    # scalping — horizon 60 barres, tenue jugee par une sortie apprise — et
    # a cet horizon la mesure du 2026-09-21 trouvait deux queues de meme
    # taille (achat p99 +1.92 sigma, vente +1.94). C'est la validation, et
    # elle seule, qui dira laquelle des deux a raison ici.
    #
    # CHAQUE COTE A SA TETE : `tete_achat` ouvre les longs, `tete_vente` les
    # shorts, chacune avec son propre lecteur du tronc (`mlp_achat`,
    # `mlp_vente`). En `long`, le masque de `cotes_permises` annulait tout
    # gradient de la seconde ; en `both`, elle apprend.
    #
    # CE QUE CE CHANGEMENT A DEMANDE AILLEURS, et qui aurait ete faux sans :
    #   - le modele de sortie lit le SENS (`entree_sortie`) ;
    #   - la derive se retire dans le sens de la position ;
    #   - le critere rejoue la sortie avec le sens, et rend les deux tenues.
    side: str = "both"

    # Préfixe pour nommer les fichiers de modèle
    model_prefix: str = "saintv2_btc_m1_flux01"
    def __post_init__(self):
        """Les constantes de l'instrument viennent de `instruments.py`.

        POURQUOI ICI ET PAS CHEZ L'APPELANT. Elles etaient recopiees dans ce
        dataclass ET dans le registre, et il fallait que chaque consommateur
        pense a appeler `config_instrument`. Personne ne le faisait :
        l'entrainement, les trois backtests et le live lisaient tous les
        valeurs du dataclass, et le registre ne servait a rien.

        ELLES AVAIENT DEJA DIVERGE : la marge valait 0.001734 ici contre
        0.001744 dans le registre, ou elle est mesuree par
        `order_calc_margin`. Assez pour que l'entrainement compte 129 places
        ouvrables la ou le live en comptait 128 — trouve par le test
        d'alignement de capacite, pas par une relecture.

        En le faisant a la construction, tout objet `PPOConfig` porte les
        memes constantes, quel que soit le fichier qui le cree. Les valeurs
        ecrites plus haut dans ce dataclass ne sont plus que des defauts
        pour un symbole inconnu du registre.
        """
        try:
            import instruments as _I
        except Exception:
            return
        p = _I.INSTRUMENTS.get(getattr(self, "symbol", None))
        if p is None:
            return
        self.atr_sl_mult = float(p["atr_sl_mult"])
        self.atr_trail_mult = float(p["trail_R"]) * self.atr_sl_mult
        self.atr_trail_dist = float(p["trail_R"]) * self.atr_sl_mult
        self.atr_tp_mult = 6.0 * self.atr_sl_mult      # inutilise, use_tp False
        self.aux_tp_mult = 2.0 * self.atr_sl_mult      # cible du classement
        self.spread_bps = float(p["spread_bps"])
        # LA LOI DU SPREAD ET LE GLISSEMENT VIENNENT DU REGISTRE AUSSI.
        #
        # Ils vivaient dans `PPOConfig` seulement, donc `cibles_m1` ne
        # pouvait pas les lire et recopiait une approximation — 4.36 bps
        # contre 2.27 reellement preleves. Les valeurs sont IDENTIQUES a
        # celles d'avant : ce n'est pas un changement de simulateur, c'est
        # la suppression du second exemplaire.
        self.spread_wide_prob = float(p["spread_wide_prob"])
        self.spread_bps_wide_factor = float(p["spread_bps_wide_factor"])
        self.entry_slippage_bps = float(p["entry_slippage_bps"])
        self.lot_min = float(p["lot_min"])
        self.lot_pas = float(p["lot_pas"])
        self.marge_frac = float(p["marge_frac"])
        self.contrat = float(p["contrat"])




# ============================================================
# CHARGEMENT M1 + H1
# ============================================================

def _fetch_paginated(symbol: str, timeframe: int,
                     date_from: datetime, date_to: datetime,
                     chunk: int = 100_000) -> Optional[np.ndarray]:
    """
    Récupère les bougies entre date_from et date_to en paginant par chunks
    de fin → début (copy_rates_from accepte un date_to + count).
    MT5 limite copy_rates_range quand le cache local est trop court.
    Cette fonction "remonte" pas à pas pour tirer l'historique manquant.
    """
    all_chunks = []
    cursor = date_to
    seen_oldest = None
    safety_iter = 0
    while safety_iter < 200:  # garde-fou : max 200 × chunk = 20M bougies
        safety_iter += 1
        rates = mt5.copy_rates_from(symbol, timeframe, cursor, chunk)
        if rates is None or len(rates) == 0:
            break

        # Filtrer ce qui est avant date_from (fin de la pagination)
        oldest_ts = int(rates[0]["time"])
        oldest_dt = datetime.utcfromtimestamp(oldest_ts)

        # On garde tout ce chunk pour l'instant, le filtrage final se fait après
        all_chunks.append(rates)

        # Critère d'arrêt 1 : on a dépassé date_from
        if oldest_dt <= date_from:
            break
        # Critère d'arrêt 2 : on n'avance plus
        if seen_oldest is not None and oldest_ts >= seen_oldest:
            break
        seen_oldest = oldest_ts

        # On positionne le curseur juste AVANT la bougie la plus ancienne
        cursor = oldest_dt - pd.Timedelta(seconds=1)

    if not all_chunks:
        return None

    # Concat + dédoublon + filtre date_from
    rates_all = np.concatenate(all_chunks)
    rates_all = np.unique(rates_all)  # numpy structuré : trie + dédoublonne
    ts_from = int(date_from.timestamp())
    ts_to   = int(date_to.timestamp())
    rates_all = rates_all[(rates_all["time"] >= ts_from) & (rates_all["time"] <= ts_to)]
    return rates_all


def _data_cache_path(cfg: PPOConfig) -> str:
    """Le cache H1 est PRECALCULE par prepare_h1.py, pas reconstruit ici.

    Recalculer les indicateurs a l'echelle H1 depuis MT5 exigerait de dupliquer
    les fenetres corrigees de prepare_h1.py — et c'est precisement la
    duplication qui avait laisse diverger l'alignement H1 et le bruit de ticks
    dans ce projet. Une seule source.
    """
    # LE CACHE SUIT LE SYMBOLE. Il etait code en dur sur BTCUSD : basculer
    # `cfg.symbol` sur XAUUSD aurait donc charge le Bitcoin en silence, et
    # l'entrainement aurait tourne sur le mauvais instrument sans qu'aucune
    # erreur soit levee. `instruments.py` nomme le fichier de chacun.
    tf = getattr(cfg, "timeframe_entrainement", "M1")
    sym = getattr(cfg, "symbol", "BTCUSD")
    if tf == "M5":
        try:
            import instruments as _I
            return _I.INSTRUMENTS[sym]["cache"]
        except Exception:
            return f"data_cache_{sym}_M5.pkl"
    if tf == "H1":
        return f"data_cache_{sym}_H1.pkl"
    # LE M1 EST ARRIVE AVEC LE SCALPING, le 2026-09-21. Il tombait
    # jusqu'ici dans le repli date, qui aurait cherche un fichier
    # inexistant et relance un telechargement de six heures sans le dire.
    if tf == "M1":
        return f"data_cache_{sym}_M1.pkl"
    return f"data_cache_{cfg.symbol}_{cfg.date_from:%Y%m%d}.pkl"


# QUI CONSTRUIT LE CACHE M5, PAR SYMBOLE.
#
# Le chemin de rechargement de `load_mt5_data` recupere du M1 et du H1 BRUTS.
# Il ne calcule ni les features Ichimoku, ni les features de range, ni les
# echelles superieures : ces ~230 colonnes sont produites par un script dedie,
# qui resample le M5 en H1 et en H4 avec les fenetres corrigees a l'echelle.
#
# CE QUE COUTAIT L'ABSENCE DE CETTE TABLE. Quand le cache M5 expirait, le code
# tombait dans le chemin MT5, fusionnait ce qu'il pouvait, puis mourait sur un
# `dropna(subset=...)` listant DEUX CENT TRENTE noms de colonnes introuvables.
# Rien dans ce message ne disait quoi faire, et la panne ne correlait avec
# aucun changement de code puisqu'elle attendait l'expiration d'un cache. Le
# chemin de rechargement ne pouvait PAS marcher : il n'a jamais su produire ces
# colonnes.
PREPARATEURS_M5 = {
    "XAUUSD": "prepare_or.py",
    "BTCUSD": "prepare_m5.py",
}
# LE M1 A SON PROPRE PREPARATEUR, et un fichier separe plutot qu'un drapeau
# : les deux caches coexistent, et un preparateur qui ecrirait tantot l'un
# tantot l'autre finirait par ecraser le mauvais. Un cache ecrase ne se
# rattrape qu'en le retelechargeant.
PREPARATEURS_M1 = {
    "XAUUSD": "prepare_or_m1.py",
    "BTCUSD": "prepare_btc_m1.py",
}


def plafond_detention(cfg) -> int:
    """Combien de barres une position peut vivre, au maximum. 0 = sans borne.

    EN M1 C'EST `tenue_max_cloture`, ET CE N'EST PAS UN CHOIX DE CONFORT.
    L'echantillon qui entraine `tete_cloture` tire des tenues de 1 a
    `tenue_max_cloture` barres, et la colonne d'age vaut
    `min(barres / scalping_max_holding, 3.0)`. Au-dela de cette borne la
    tete voit une valeur qu'elle n'a jamais rencontree a l'entrainement :
    elle ne distingue plus une heure de quatre jours.

    Le plafond est donc pose LA OU L'OBSERVATION CESSE D'INFORMER, et il se
    lit dans le meme champ que l'echantillon. Deux reglages auraient
    diverge — c'est la faute que ce fichier documente le plus souvent.

    UN `max_holding_bars` EXPLICITE GAGNE TOUJOURS, et c'est un test qui
    l'a impose. La premiere version prenait `tenue_max_cloture` des que le
    pas etait M1, donc elle IGNORAIT un plafond pose a la main :
    `test_economic_learning` fixe 20, l'environnement sortait a 30, et la
    cible calculee a 20 ne concordait plus. Un reglage qu'on pose et que le
    code remplace en silence est pire qu'un reglage absent.

    EN M5 rien ne change : `max_holding_bars` vaut 0, la fin d'episode
    liquide, et c'est la geometrie mesuree de ce cote-la.
    """
    _exp = int(getattr(cfg, "max_holding_bars", 0))
    if _exp > 0:
        return _exp
    # PLUS AUCUN PLAFOND DEDUIT, ET C'EST UNE DECISION ASSUMEE.
    #
    # En M1 cette fonction rendait `tenue_max_cloture`, donc une sortie
    # FORCEE PAR L'HORLOGE. Mesure du 2026-09-22 : elle decidait de
    # 100 % des sorties gagnantes — mediane ET moyenne de tenue des
    # gagnants egales au plafond, sans une exception. « Gagnant » ne
    # voulait pas dire « une tete a decide de sortir avec un gain » mais
    # « le trade a survecu au chronometre ».
    #
    # LES DEUX PORTES SONT DESORMAIS DES TETES : `tete_cloture` coupe la
    # perte, `tete_profit` prend le gain. Aucune horloge ne tranche a leur
    # place. C'est ce que le proprietaire du depot a demande, deux fois, et
    # c'est ce que la mesure du sommet soutient : il tombe a 49 minutes au
    # dixieme centile et a 2 770 au quatre-vingt-dixieme, ecart-type 991.
    # Un plafond constant ferme tous les gagnants au meme instant.
    #
    # CE QUE CA COUTE, ET IL FAUT LE LIRE AVEC. L'avantage de la strategie
    # PLAFONNE entre 48 et 96 heures puis s'effondre — +25.38 bps a 2 880
    # minutes, +17.10 a 5 760, et le test du signe n'y est plus
    # significatif. Sans horloge, c'est `tete_profit` SEULE qui empeche une
    # position de pourrir au-dela de l'epuisement. Elle est donc devenue
    # portante, et son seuil n'est pas mesure.
    #
    # LE FILET QUI RESTE est la fin d'episode, pas une regle : le
    # simulateur solde ce qui est encore ouvert. Le journal affiche la
    # tenue MAXIMALE a chaque epoch — une position immortelle s'y verra.
    #
    # `max_holding_bars`, pose explicitement, reste prioritaire : c'est la
    # porte de sortie pour qui veut retablir un plafond sans toucher au
    # code.
    return 0


def _optimiseur_profit(policy, cfg):
    """Un Adam qui ne porte QUE les poids de la tete de profit.

    POURQUOI PAS `optimizer_rang`. Celui-la porte tout le reseau. La tete
    de profit ne lit PAS le tronc — c'est le seul organe qui l'ignore, et
    une mesure l'impose. Lui donner un optimiseur global ferait remonter
    son gradient dans un tronc qu'elle n'utilise pas : le tronc bougerait
    au service d'un organe qui ne le lit pas, et les tetes d'entree en
    paieraient le prix sans que rien ne le signale.

    L'ETAT ADAM EST RECONSTRUIT A CHAQUE EPOCH, et c'est assumé : la tete
    fait 1 249 parametres et 240 pas par epoch. Le moment perdu au
    redemarrage coute moins que le fil a tirer pour le conserver.
    """
    ps = list(policy.mlp_profit.parameters()) + \
        list(policy.tete_profit.parameters())
    return optim.Adam(ps, lr=float(getattr(cfg, "lr_rang", cfg.lr)), eps=1e-8)


def entree_profit(x, n_base: int):
    """Les QUATRE colonnes que `tete_profit` lit, dans CET ordre exact.

    UN SEUL ENDROIT SAIT LES EXTRAIRE, et c'est tout l'objet de cette
    fonction. La passe supervisee et la decision doivent presenter les
    memes colonnes dans le meme ordre ; ce depot documente sous « deux
    ecritures de la meme regle » ce qui arrive quand elles se separent —
    la tete apprend sur une distribution et decide sur une autre, sans
    qu'aucune erreur ne se declenche.

        0  latent    gain non realise, en ATR d'entree      bloc position 1
        1  age       tenue normalisee                        bloc position 2
        2  creux_rang  rang glissant de l'ecart a l'EMA      bloc marche
        3  flux_rang   rang glissant du volume agressif      bloc marche

    LES DEUX DERNIERES SONT PRISES DANS LE BLOC DE FEATURES, donc
    NORMALISEES par les statistiques figees du fold 1 — et c'est
    volontaire. La fabrique d'echantillons pourrait rendre le rang brut
    entre 0 et 1 ; lire le bloc de l'etat des deux cotes garantit que la
    tete voit la meme chose a l'entrainement et en decision.

    LA DERNIERE BARRE SEULEMENT. La tete ne regarde pas l'historique : la
    mesure qui l'a justifiee portait sur l'etat courant, et lui donner une
    fenetre serait lui promettre une information qu'elle n'a pas apprise.
    """
    import numpy as _np
    a = _np.asarray(x, dtype=_np.float32)
    if a.ndim == 2:
        a = a[None, ...]
    ip, im = IDX_PROFIT_POS, IDX_PROFIT_MARCHE
    return _np.stack([
        a[:, -1, n_base + ip[0]],
        a[:, -1, n_base + ip[1]],
        a[:, -1, im[0]],
        a[:, -1, im[1]],
    ], axis=1).astype(_np.float32)


def entree_sortie(x, n_base: int):
    """Les CINQ colonnes que le modele de sortie lit, dans CET ordre exact.

        0-3  celles de `entree_profit` : latent, age, creux_rang, flux_rang
        4    sens     +1 long, -1 short                  bloc position 0

    LE SENS EST LA DEPUIS QUE LES SHORTS OUVRENT, le 2026-09-25. Voir
    `N_SORTIE_FEATURES` dans `saint_core` : le meme etat de marche ne dit
    pas la meme chose selon le cote tenu, et le flux ne se retourne pas.

    UNE SEULE ECRITURE, trois lecteurs : la decision (`decide_sortie`), le
    tampon PPO de la collecte, et — par la meme convention, vectorisee —
    `rendements_sortie_ppo`.
    """
    import numpy as _np
    a = _np.asarray(x, dtype=_np.float32)
    if a.ndim == 2:
        a = a[None, ...]
    return _np.concatenate(
        [entree_profit(a, n_base), a[:, -1, n_base + IDX_SENS_POS][:, None]],
        axis=1).astype(_np.float32)


def cadence_sortie(en_pos, frais, depuis, pas: int):
    """QUI decide de sortir a cette barre. Une seule ecriture, quatre lecteurs.

    POURQUOI CETTE FONCTION EXISTE. Le 2026-09-25, trois boucles ne
    consultaient pas la politique de sortie au meme rythme :

        rollout      tous les 15 barres    (la politique APPREND a ce rythme)
        validation   A CHAQUE BARRE
        test         A CHAQUE BARRE

    En `argmax`, une politique interrogee chaque barre ferme a la premiere
    ou elle penche vers FERMER — quinze fois plus d'occasions de le faire
    que ce qu'elle a appris. La validation jouait donc une autre strategie
    que l'entrainement, et c'est elle qui decide du checkpoint. Le critere
    de sauvegarde, lui, ne jouait meme pas de sortie du tout.

    LA REGLE : une position FRAICHE decide tout de suite ; ensuite une
    decision tous les `pas` barres exactement. `depuis` compte les barres
    ecoulees depuis la derniere decision, par position.

    L'ANCIENNE CADENCE ETAIT DE 16, PAS DE 15. Le test precedait
    l'increment : apres une decision a la barre 1, la suivante tombait a
    la barre 17. Et le loyer facturait 15 barres sur un intervalle de 16.
    On incremente desormais AVANT de tester : decisions aux barres 1, 16,
    31, et `ecoule` vaut exactement l'intervalle facture.

    Rend (decideurs, ecoule), ou `ecoule[k]` est le nombre de barres
    couvertes par la decision precedente de k — celui que le loyer doit
    facturer.
    """
    for k in en_pos:
        depuis[k] += 1
    dec = [k for k in en_pos if bool(frais[k]) or depuis[k] >= pas]
    ecoule = {k: int(depuis[k]) for k in dec}
    for k in dec:
        depuis[k] = 0
    return dec, ecoule


def rendements_sortie_ppo(policy, data, idx, cfg, device):
    """Ce que rapporte chaque occasion SOUS LA SORTIE QUE LE MODELE JOUE.

    CE QUI ETAIT FAUX, ET CA DECIDAIT DU DEPLOIEMENT. `sommet` et
    `score_retenue` lisaient `rendements_du_systeme`, qui rend le
    rendement d'une position tenue EXACTEMENT `horizon_cloture` barres.
    Depuis que la sortie est une politique PPO, aucun trade ne se joue
    ainsi : le critere de sauvegarde jugeait une strategie que personne ne
    jouait, et la politique de sortie n'y entrait nulle part. Elle aurait
    pu apprendre des sorties parfaites sans que le checkpoint le voie.

    ON DEROULE LA POLITIQUE le long de chaque chemin de prix de la grille,
    en `argmax`, a la cadence de `cadence_sortie` — la meme qu'au rollout,
    en validation et au test. La politique change a chaque epoch, donc ce
    calcul aussi : il est refait a chaque epoch, plus une fois par fold.

    UNE SEULE CHOSE CHANGE PAR RAPPORT A L'ANCIEN CRITERE : la barre de
    sortie. Le prix d'entree, le cout et l'echelle sont ceux de
    `rendements_du_systeme`, a l'identique — sinon deux choses auraient
    bouge a la fois et on ne saurait pas laquelle agit.

    L'ENTREE DE LA POLITIQUE, ELLE, SUIT LA CONVENTION DE L'ENVIRONNEMENT :
    execution a l'OUVERTURE avec le spread paye, ATR de la barre d'entree.
    `cibles_m1.echantillon_cloture` a mesure qu'une convention differente
    decale le latent de 0.61 ATR — assez pour changer la decision.

    LA DETENTION EST BORNEE PAR `tenue_max_cloture`, et par la fin des
    donnees : une position encore ouverte la est soldee, comme
    l'environnement le fait en fin d'episode.
    """
    import cibles_m1 as _CM
    import instruments as _IN
    df = data.df
    feats = data.features
    c = df["close"].to_numpy(np.float64)
    o = df["open"].to_numpy(np.float64) if "open" in df.columns else c
    atr = np.maximum(df["atr_14"].to_numpy(np.float64), 1e-9)
    if "spread_bar" in df.columns:
        spb = df["spread_bar"].to_numpy(np.float64)
        spb = np.where(np.isfinite(spb) & (spb > 0), spb,
                       float(np.nanmedian(spb[spb > 0])))
    else:
        spb = np.full(len(c), float(_IN.spread_espere(cfg.symbol)))
    n = len(c)
    h = int(getattr(cfg, "horizon_cloture", 30))
    T = int(getattr(cfg, "tenue_max_cloture", h))
    pas = max(1, int(getattr(cfg, "pas_decision_sortie", 1)))
    smh = float(max(SCALPING_MAX_HOLDING, 1))
    gl = float(_IN.INSTRUMENTS[cfg.symbol]["entry_slippage_bps"]) / 2.0
    im = IDX_PROFIT_MARCHE
    idx = np.asarray(idx, dtype=np.int64)
    m = len(idx)
    ip = np.maximum(idx - 1, 0)

    def _un_cote(sens):
        p_ent = o[idx] * (1.0 + sens * (spb[ip] + gl) / 1e4)
        a_ent = atr[idx]
        t_sortie = np.full(m, T, dtype=np.int64)
        ouvert = np.ones(m, dtype=bool)
        frais = np.ones(m, dtype=bool)
        depuis = np.zeros(m, dtype=np.int64)
        for t in range(1, T + 1):
            j = idx + t
            bout = ouvert & (j >= n - 1)
            if bout.any():
                t_sortie[bout] = np.maximum(n - 2 - idx[bout], 1)
                ouvert[bout] = False
            if not ouvert.any():
                break
            # LA MEME REGLE QUE `cadence_sortie`, vectorisee.
            depuis[ouvert] += 1
            dec = ouvert & (frais | (depuis >= pas))
            depuis[dec] = 0
            frais[dec] = False
            if not dec.any():
                continue
            r = np.flatnonzero(dec)
            jj = j[r]
            x = np.stack([
                sens * (c[jj] - p_ent[r]) / a_ent[r],
                np.full(len(r), min(t / smh, 3.0)),
                feats[jj - 1, im[0]],
                feats[jj - 1, im[1]],
                # LE SENS, dans la colonne ou `entree_sortie` le met.
                np.full(len(r), sens),
            ], axis=1).astype(np.float32)
            with torch.no_grad():
                lg, _ = policy.sortie(torch.from_numpy(x).to(device))
            ferme = r[(lg.argmax(-1) == FERMER).cpu().numpy()]
            t_sortie[ferme] = t
            ouvert[ferme] = False
        je = np.minimum(idx + t_sortie, n - 1)
        mv = (c[je] - c[idx]) / c[idx] * 1e4
        return sens * mv - _CM.cout_par_barre(df, idx, cfg.symbol), t_sortie

    ach, t_a = _un_cote(+1.0)
    ven, t_v = _un_cote(-1.0)
    # L'ECHELLE EST CELLE DE `rendements_du_systeme`, A L'IDENTIQUE.
    _px = c[ip]
    _sig = atr[ip] / np.maximum(_px, 1e-9) * 1e4 * np.sqrt(float(h))
    _sig = np.where(np.isfinite(_sig) & (_sig > 1e-6), _sig, np.nan)
    # LES DEUX TENUES, une colonne par sens : depuis que les shorts ouvrent,
    # celle de l'achat seul ne decrivait plus que la moitie des positions.
    return ach / _sig, ven / _sig, np.stack([t_a, t_v], axis=1)


def optimiseurs_ppo(policy, cfg):
    """UN OPTIMISEUR PAR GROUPE de `policy.groupes_ppo()`.

    Choix du proprietaire, 2026-09-25 : une tete pour l'achat, une pour la
    vente, une pour la coupure des gains, une pour la coupure des pertes,
    CHACUNE AVEC SON OPTIMISEUR. S'y ajoutent le tronc, partage et nourri
    par les quatre, et le critique de l'etat plat. Les groupes sont
    disjoints : aucun optimiseur ne touche les poids d'un autre.
    """
    return {nom: optim.Adam(ps, lr=float(cfg.lr), eps=1e-8)
            for nom, ps in policy.groupes_ppo().items()}


def masque_entree(masque_plat, envs_):
    """Les actions d'entree permises, pour chaque environnement qui decide.

    Rend (masque (B, 3) bool, sans_place (B,) bool).

    LE COTE vient du masque plat (`cotes_permises`, et le veto eventuel).
    LE SOLDE vient ensuite : une ouverture qu'il refuserait est masquee
    AVANT le tirage. L'action enregistree est alors celle qui est jouee, et
    sa probabilite celle de la loi qui l'a tiree — c'est ce qui garde le
    rapport de PPO exact. ATTENDRE est toujours permis.

    UNE ECRITURE, TROIS LECTEURS : collecte, validation, test.
    """
    m = np.asarray(masque_plat, dtype=bool)[:, :N_ACTIONS_ENTREE].copy()
    m[:, ATTENDRE] = True
    sans = np.zeros(len(envs_), dtype=bool)
    for i, e in enumerate(envs_):
        _px = float(e.data.close[min(e.idx, e.data.length - 1)])
        if e.places_ouvrables(_px) <= 0:
            m[i, ACHETER] = False
            m[i, VENDRE] = False
            sans[i] = True
    return m, sans


def _tire(logits, explore: bool, generateur=None):
    """L'action : tiree de la loi, ou son argmax si `explore` est faux.

    AVEC `generateur`, LE TIRAGE EST REPRODUCTIBLE : la validation et le
    test en passent un a graine fixe, pour que deux epochs se jugent sur
    les memes aleas sans toucher au generateur global de l'entrainement.
    """
    if not explore:
        return logits.argmax(-1)
    if generateur is None:
        return torch.distributions.Categorical(logits=logits).sample()
    probs = torch.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, 1, generator=generateur).squeeze(-1)


def decide_entree(policy, etats, masques, device, explore: bool = True,
                  generateur=None):
    """Acheter, vendre ou attendre ? Rend (actions, logprobs, valeurs).

    On ECHANTILLONNE a l'entrainement — PPO a besoin que l'action jouee
    vienne de la loi qu'il met a jour — et on prend l'argmax en validation
    et au test, comme pour la sortie.
    """
    x = torch.as_tensor(np.asarray(etats, dtype=np.float32), device=device)
    m = torch.as_tensor(np.asarray(masques, dtype=bool), device=device)
    with torch.no_grad():
        logits, valeur = policy.entree(x)
        logits = logits.masked_fill(~m, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        a = _tire(logits, explore, generateur)
        lp = dist.log_prob(a)
    return (a.cpu().numpy().astype(np.int64),
            lp.cpu().numpy().astype(np.float32),
            valeur.cpu().numpy().astype(np.float32))


def avantages_semi_mdp(r, dt, v, fin, gamma: float, lam: float):
    """GAE sur une suite de decisions qui durent chacune `dt` barres.

    Une decision d'entree couvre toute la vie du trade qu'elle ouvre — ou
    une seule barre si elle attend. Sa recompense `r` est deja la somme
    actualisee de ce que le trade a rapporte ; le suivant s'actualise donc
    de `gamma ** dt`, et la trace de `(gamma ** dt) * lam`.

    La derniere decision de la suite n'a pas de suivante : sa valeur de
    depart est prise a zero, comme a la fin d'un episode.
    """
    r = np.asarray(r, np.float64)
    dt = np.maximum(np.asarray(dt, np.float64), 1.0)
    v = np.asarray(v, np.float64)
    fin = np.asarray(fin, bool)
    n = len(r)
    adv = np.zeros(n)
    a = 0.0
    for t in range(n - 1, -1, -1):
        suite = 0.0 if (fin[t] or t == n - 1) else 1.0
        vn = v[t + 1] if t + 1 < n else 0.0
        g = gamma ** dt[t]
        d = r[t] + g * vn * suite - v[t]
        a = d + g * lam * suite * a
        adv[t] = a
    return adv, adv + v


def _boucle_ppo(avant, A, LP, RET, ADV, optims, groupes, noms, cfg,
                coef_h, device):
    """Les passes PPO communes aux deux mises a jour.

    UN PAS PAR OPTIMISEUR ET PAR LOT, chacun sur ses poids et son propre
    ecretage de gradient : une tete qui recoit un gradient fort ne reduit
    pas le pas des autres.
    """
    eps = float(getattr(cfg, "clip_eps", 0.2))
    cv = float(getattr(cfg, "value_coef", 0.5))
    n = len(A)
    nlot = max(1, n // max(1, int(cfg.batch_size)))
    st = {"actor": [], "critic": [], "kl": [], "clip": []}
    for _ in range(int(getattr(cfg, "passes_ppo", 4))):
        perm = torch.randperm(n, device=device)
        for i in range(nlot):
            m = perm[i::nlot]
            lg, vv = avant(m)
            dist = torch.distributions.Categorical(logits=lg)
            lpn = dist.log_prob(A[m])
            rt = torch.exp(lpn - LP[m])
            pa = -torch.min(rt * ADV[m],
                            torch.clamp(rt, 1 - eps, 1 + eps) * ADV[m]).mean()
            pc = torch.nn.functional.mse_loss(vv, RET[m])
            perte = pa + cv * pc - coef_h * dist.entropy().mean()
            for nom in noms:
                optims[nom].zero_grad(set_to_none=True)
            perte.backward()
            for nom in noms:
                torch.nn.utils.clip_grad_norm_(groupes[nom],
                                               float(cfg.max_grad_norm))
                optims[nom].step()
            st["actor"].append(float(pa))
            st["critic"].append(float(pc))
            st["kl"].append(float((LP[m] - lpn).mean()))
            st["clip"].append(float(((rt - 1).abs() > eps).float().mean()))
    return {k: float(np.mean(v)) if v else float("nan") for k, v in st.items()}


def _sous_lot(n, cfg):
    """Les indices gardes pour la mise a jour, APRES le calcul de l'avantage."""
    idx = np.arange(n)
    n_max = int(getattr(cfg, "max_transitions_ppo", 0) or 0)
    if 0 < n_max < n:
        idx = np.sort(np.random.choice(idx, n_max, replace=False))
    return idx


def maj_ppo_entree(policy, ep_buf, optims, cfg, device):
    """La mise a jour PPO des tetes d'ACHAT et de VENTE, et du tronc.

    Rend un dict de diagnostics, ou None s'il y a trop peu de decisions.
    """
    gam, lam = float(getattr(cfg, "gamma", 0.99)), float(cfg.lambda_gae)
    O, M, A, LP, RET, ADV = [], [], [], [], [], []
    for b in ep_buf:
        if not b["actions"]:
            continue
        adv, ret = avantages_semi_mdp(b["rewards"], b["dts"], b["vals"],
                                      b["dones"], gam, lam)
        O.extend(b["states"]); M.extend(b["masques"]); A.extend(b["actions"])
        LP.extend(b["lps"]); ADV.extend(adv); RET.extend(ret)
    n_total = len(A)
    if n_total < 256:
        return None
    idx = _sous_lot(n_total, cfg)
    O = torch.as_tensor(np.stack([O[i] for i in idx]).astype(np.float32),
                        device=device)
    M = torch.as_tensor(np.stack([M[i] for i in idx]).astype(bool),
                        device=device)
    A = torch.as_tensor(np.asarray(A, np.int64)[idx], device=device)
    LP = torch.as_tensor(np.asarray(LP, np.float32)[idx], device=device)
    RET = torch.as_tensor(np.asarray(RET, np.float32)[idx], device=device)
    ADV = torch.as_tensor(np.asarray(ADV, np.float32)[idx], device=device)
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    groupes = policy.groupes_ppo()
    noms = ("tronc", "achat", "vente", "valeur_entree")

    def avant(m):
        lg, vv = policy.entree(O[m])
        return lg.masked_fill(~M[m], -1e9), vv

    st = _boucle_ppo(avant, A, LP, RET, ADV, optims, groupes, noms, cfg,
                     float(getattr(cfg, "entropie_entree", 0.01)), device)
    with torch.no_grad():
        hs = []
        for d in range(0, len(A), 2048):
            mm = torch.arange(d, min(d + 2048, len(A)), device=device)
            hs.append(torch.distributions.Categorical(
                logits=avant(mm)[0]).entropy())
        st["H"] = float(torch.cat(hs).mean())
    st["n"], st["n_total"] = len(A), n_total
    st["parts"] = [float((A == a).float().mean()) for a in range(N_ACTIONS_ENTREE)]
    return st


def maj_ppo_sortie(policy, sortie_buf, optims, cfg, device):
    """La mise a jour PPO des tetes de COUPURE DES GAINS et DES PERTES.

    Une decision par barre de chaque position. L'avantage se calcule sur la
    trajectoire entiere de la position — elle peut passer de la perte au
    gain et revenir — puis chaque decision entraine la tete du cote ou elle
    a ete prise.
    """
    gam, lam = float(getattr(cfg, "gamma", 0.99)), float(cfg.lambda_gae)
    O, P, A, LP, RET, ADV = [], [], [], [], [], []
    for b in sortie_buf:
        if not b:
            continue
        b[-1]["done"] = True
        n = len(b)
        av = [0.0] * n
        adv = 0.0
        for t in range(n - 1, -1, -1):
            nt = 0.0 if b[t]["done"] else 1.0
            vn = b[t + 1]["v"] if (t + 1 < n) else 0.0
            d = b[t]["r"] + gam * vn * nt - b[t]["v"]
            adv = d + gam * lam * nt * adv
            av[t] = adv
        for t in range(n):
            O.append(b[t]["o"]); P.append(b[t]["p"]); A.append(b[t]["a"])
            LP.append(b[t]["lp"]); ADV.append(av[t])
            RET.append(av[t] + b[t]["v"])
    n_total = len(A)
    if n_total < 256:
        return None
    idx = _sous_lot(n_total, cfg)
    O = torch.as_tensor(np.stack([O[i] for i in idx]).astype(np.float32),
                        device=device)
    _P = torch.as_tensor(np.asarray(P, np.float32)[idx], device=device)
    A = torch.as_tensor(np.asarray(A, np.int64)[idx], device=device)
    LP = torch.as_tensor(np.asarray(LP, np.float32)[idx], device=device)
    RET = torch.as_tensor(np.asarray(RET, np.float32)[idx], device=device)
    _ADV = torch.as_tensor(np.asarray(ADV, np.float32)[idx], device=device)
    _en_gain = _P[:, 0] > 0.0
    # L'AVANTAGE EST NORMALISE PAR TETE : ensemble, celui des pertes —
    # gonfle par le loyer zombie — fixerait l'echelle, et celui des gains
    # deviendrait du bruit.
    for _mq in (_en_gain, ~_en_gain):
        if int(_mq.sum()) > 1:
            _x = _ADV[_mq]
            _ADV[_mq] = (_x - _x.mean()) / (_x.std() + 1e-8)
    groupes = policy.groupes_ppo()
    noms = ("tronc", "gain", "perte")

    def avant(m):
        return policy.sortie_complete(O[m], _P[m])

    st = _boucle_ppo(avant, A, LP, RET, _ADV, optims, groupes, noms, cfg,
                     float(getattr(cfg, "entropie_sortie", 0.02)), device)
    with torch.no_grad():
        hs = []
        for d in range(0, len(A), 2048):
            mm = torch.arange(d, min(d + 2048, len(A)), device=device)
            hs.append(torch.distributions.Categorical(
                logits=avant(mm)[0]).entropy())
        h = torch.cat(hs)
    st["n"], st["n_total"] = len(A), n_total
    st["tetes"] = {}
    for nom, mq in (("GAIN ", _en_gain), ("PERTE", ~_en_gain)):
        k = int(mq.sum())
        st["tetes"][nom] = (
            k,
            float(h[mq].mean()) if k else float("nan"),
            float((A[mq] == FERMER).float().mean()) if k else float("nan"))
    return st


def decide_sortie(policy, etats, device, n_base: int, explore: bool = True,
                  generateur=None):
    """Tenir ou fermer ? Rend (actions, logprobs, valeurs), un par etat.

    ELLE REMPLACE DEUX REGLES ECRITES A LA MAIN, et c'est tout son objet.
    `demande_cloture` seuillait un risque predit, `demande_profit` un
    reste predit. Les deux tetes apprenaient bien — rho +0.355 et +0.272 —
    et les quatre calibrations successives de leurs seuils ont echoue :

        seuil absolu       tenue[G 3/15 P 52/88 x0.2], rapport INVERSE
        + latent > 0       x0.5, toujours inverse
        seuil relatif      `fermerait 0.0 %` sur 51 epochs sur 53
        echelle redressee  74 % des GAGNANTS soldes par la fin d'episode

    Le defaut n'etait pas dans la calibration mais dans l'idee de seuiller
    une amplitude. Ici il n'y a plus de seuil : la politique SORT la
    decision.

    ON ECHANTILLONNE A L'ENTRAINEMENT, on prend l'argmax ailleurs. PPO a
    besoin que l'action jouee vienne de la distribution qu'il met a jour —
    sinon le rapport de vraisemblance perd son sens, et le depot documente
    deja cette faute sous « l'action executee ne vient d'AUCUNE des
    politiques mises a jour ». En validation et en test on veut la
    decision, pas son bruit.
    """
    # LA TETE LIT TOUTES LES FEATURES : l'observation entiere par le tronc,
    # plus les cinq colonnes de position. Voir `sortie_complete`.
    p = torch.from_numpy(entree_sortie(etats, n_base)).to(device)
    x = torch.as_tensor(np.asarray(etats, dtype=np.float32), device=device)
    with torch.no_grad():
        logits, valeur = policy.sortie_complete(x, p)
        dist = torch.distributions.Categorical(logits=logits)
        a = _tire(logits, explore, generateur)
        lp = dist.log_prob(a)
    return (a.cpu().numpy().astype(np.int64),
            lp.cpu().numpy().astype(np.float32),
            valeur.cpu().numpy().astype(np.float32))


def demande_profit(policy, etats, device, n_base: int, coupe: float = 0.0,
                   rho_min: float = 0.0):
    """Faut-il PRENDRE LE PROFIT ? Un booleen par etat en position.

    LA REGLE EST SYMETRIQUE DU STOP, et se lit de la meme facon :

        fermer si  reste_predit  <  coupe x latent

    SANS ECHELLE, exactement comme le stop compare le latent a un multiple
    du risque predit. Une constante en ATR ne suit pas le marche et se
    deregle des que la tete se trompe d'amplitude — c'est ce qui a inverse
    le rapport des tenues le 2026-09-22, gagnants tenus 3 minutes et
    perdants 52.

    `tete_profit` rend une amplitude — ce qu'il reste a prendre d'ici
    l'epuisement, en ATR d'entree — apres softplus, donc positive. Quand
    elle tombe sous le seuil, il n'y a plus rien a attendre et le trade a
    fait son travail.

    A SEUIL NUL LA REGLE NE MORD JAMAIS, et c'est le repli voulu : un
    modele sans tete de profit entrainee ne doit pas se mettre a fermer au
    hasard.
    """
    import numpy as _np
    # DEUX VERROUS AVANT MEME DE REGARDER L'ETAT.
    #
    # `coupe` a zero desactive la regle. Et tant que la tete n'a pas
    # atteint `rho_min`, elle ne decide de rien : `_rho_profit` est pose
    # par la passe supervisee a chaque epoch et vaut zero avant la
    # premiere, donc une tete jamais entrainee est muette par defaut.
    _rho = float(getattr(policy, "_rho_profit", 0.0))
    if float(coupe) <= 0.0 or not _np.isfinite(_rho) or _rho < float(rho_min):
        n = 1 if _np.asarray(etats).ndim == 2 else len(etats)
        return _np.zeros(n, dtype=bool)
    p = entree_profit(etats, n_base)
    with torch.no_grad():
        z = policy.profit(torch.from_numpy(p).to(device)).squeeze(-1)
    reste = torch.nn.functional.softplus(z).float().cpu().numpy()
    # LE REDRESSEMENT D'ECHELLE, mesure par la passe supervisee a chaque
    # epoch. Sans lui, une tete dont l'ORDRE est juste mais la MAGNITUDE
    # quatre fois trop basse declenche cent fois trop souvent — c'est
    # exactement ce qui s'est produit le 2026-09-22.
    reste = reste * float(getattr(policy, "_echelle_profit", 1.0))

    # ELLE NE FERME QU'UNE POSITION EN GAIN, ET C'EST VRAI PAR DEFINITION.
    #
    # LA PREMIERE VERSION N'AVAIT PAS CETTE CONDITION, et le run du
    # 2026-09-22 l'a payee en une epoch :
    #
    #     tenue[G 1/11  P 1/10  x1.1]     le rapport valait 19.6 en simulation
    #     ENV [B 28.0%  H 44.1%  C 27.9%] ouvrir et refermer a la barre suivante
    #     PF 0.46, sommet -0.420 contre -0.154 au hasard
    #
    # A l'entree le latent est NEGATIF — on vient de payer le spread — et
    # une tete encore incertaine predit volontiers moins d'un ATR de reste.
    # La regle fermait donc immediatement, sur une perte.
    #
    # UNE PRISE DE PROFIT QUI FERME UNE PERTE N'EST PAS UNE PRISE DE
    # PROFIT, C'EST UN SECOND MAUVAIS STOP. Le vrai stop, lui, a toujours
    # exige un latent negatif ; son symetrique doit exiger un latent
    # positif. J'ai construit l'un sans donner a l'autre la condition
    # miroir.
    #
    # LE COUT DE SORTIE N'ENTRE PAS DANS LE SEUIL. Il sera paye de toute
    # facon, donc il s'annule entre « fermer » et « tenir » — meme
    # raisonnement que `cibles_m1.cible_cloture`, et le facturer ici ferait
    # tenir trop longtemps.
    latent = p[:, 0]
    return (reste < float(coupe) * latent) & (latent > 0.0)


def demande_cloture(policy, etats, device, n_base: int,
                    coupe: float = 0.0, marge: float = 1.0):
    """Faut-il fermer ? Un booleen par etat EN POSITION fourni.

    LA TETE NE PREDIT PLUS LE SENS, ELLE PREDIT LE RISQUE. C'est un
    changement de fond, impose par la mesure du 2026-09-21.

    CE QUI NE MARCHAIT PAS. La tete apprenait le SIGNE de « gagne-t-on
    encore en tenant trente minutes ». Quatre epochs au hasard — juste
    50.8 %, 49.2 %, 53.1 %, 46.9 % — perte collee a ln(2). Et la regle
    fermait des que le logit passait sous zero : une piece lancee a chaque
    barre, donc une detention mediane d'UNE minute et 88 % des trades a une
    seule barre. Ce n'etait pas un defaut d'apprentissage, la cible elle-
    meme est a 50.0 % de positifs.

    CE QUI MARCHE. Le SENS du prix a trente minutes n'est pas dans ces
    donnees — trois tests independants le disent. L'AMPLITUDE y est, et
    fortement : `vol_20` predit l'etendue a venir a rho 0.625, 0.637 hors
    echantillon, contre un plancher par rotation de 0.136. La tete apprend
    donc « combien ca peut me couter de rester », et non « ou va le prix ».

    LA REGLE, ET ELLE DIT EXACTEMENT CE QU'ON LUI DEMANDE :

        fermer si  latent_atr  <  -(marge + coupe x risque_predit)

    Une position tres en gain devant un marche calme attend — son latent
    absorbe le risque. La meme en perte devant un marche qui s'ouvre sort
    — le risque depasse ce qu'elle peut encaisser. Patienter quand
    patienter paie, couper vite quand ca coute.

    LES DEUX GRANDEURS SONT EN ATR D'ENTREE, sans quoi la comparaison
    n'aurait aucun sens. `latent_atr` est la colonne 1 de l'etat, remplie
    par `_get_obs` ; `cible_risque` est construite dans la meme unite.

    LE PLAFOND DE DETENTION RESTE, en second rideau : voir
    `plafond_detention`. Une regle qui depend d'une prediction ne doit
    jamais etre la SEULE borne.

    LES COLONNES DE COMPTE SONT REMISES A ZERO, parce que c'est leur
    valeur A L'ENTRAINEMENT — l'echantillon ne remplit que le sens, le
    latent et l'age. La borne vient de `N_POS_FEATURES`.
    """
    import numpy as _np
    x = _np.array(etats, dtype=_np.float32, copy=True)
    if x.ndim == 2:
        x = x[None, ...]
    x[:, :, n_base + 3:n_base + N_POS_FEATURES] = 0.0
    # LE LATENT SE LIT DANS L'ETAT, sur la derniere barre — c'est celle ou
    # la decision se prend. Le reseau le voit aussi ; on le relit ici parce
    # que la REGLE en a besoin, pas seulement la tete.
    latent = x[:, -1, n_base + 1]
    with torch.no_grad():
        z = policy.cloture(torch.from_numpy(x).to(device)).squeeze(-1)
    # LA TETE REND UNE AMPLITUDE, DONC POSITIVE. `softplus` l'impose sans
    # couper le gradient a zero comme le ferait un `relu` : une prediction
    # negative serait un non-sens physique, et un risque nul ferait tenir
    # indefiniment.
    risque = torch.nn.functional.softplus(z).float().cpu().numpy()
    # LA MARGE TIENT LA POSITION, LE RISQUE LA MODULE. `coupe` vaut zero
    # par defaut : le balayage ne distingue pas son effet du bruit. Voir
    # `PPOConfig.coupe_risque`.
    seuil = float(marge) + float(coupe) * risque
    return latent < -np.maximum(seuil, 0.05)


def _borne_syst(cfg) -> int:
    """De combien de barres une occasion a besoin pour se resoudre.

    En M1 c'est `horizon_cloture`. En M5 c'est la borne de securite de
    `cibles`, six jours. Garder la seconde en M1 jetterait 8 640 barres au
    bout de chaque fenetre pour des trades qui durent deux heures : la
    grille perdrait six jours d'occasions pour rien.
    """
    if getattr(cfg, "timeframe_entrainement", "M5") == "M1":
        return int(getattr(cfg, "horizon_cloture", 30)) + 2
    import cibles as _CI
    return int(_CI.BORNE_DEFAUT)


def rendements_du_systeme(df, idx, cfg, indicateur: bool = False):
    """Ce que rapporte une entree SOUS LA REGLE QUE LE SYSTEME JOUE.

    POURQUOI CETTE FONCTION EXISTE — elle est nee d'un defaut mesure.
    Le 2026-09-21 le systeme est passe au scalping M1 : plus de stop, plus
    de trailing, sortie decidee par `tete_cloture` a trente minutes. Les
    trois tetes ont ete branchees sur `cibles_m1`. Le DIAGNOSTIC, lui, a
    continue de lire `cibles.py`, la geometrie au stop.

    CE QUE CA DONNAIT, mesure sur la fenetre de validation du fold 1 :

        duree mediane              297 barres = 5 HEURES
        part qui finit a -1 R      100.0 %
        hasard  -1.099             sommet  -1.088

    CENT POUR CENT DES OCCASIONS AU STOP : toutes identiques, donc rien a
    classer. `rho`, `rhoAux`, `sommet`, `net` et le portillon du hasard
    notaient une strategie que personne ne joue, et le critere de retenue
    choisissait ses points de reprise la-dessus.

    LA REGLE SE LIT DESORMAIS A UN SEUL ENDROIT. En M1 on prend `cibles_m1`
    a l'horizon de la tete de cloture ; ailleurs `cibles`, inchange. Un
    appelant qui oublierait de suivre un changement de geometrie n'existe
    plus, parce qu'il n'y a plus qu'un appel.

    `indicateur` est ignore en M1 et c'est voulu. Il demandait a `cibles`
    l'ATTEINTE de 2 R plutot que le rendement de sortie, parce que le
    second, borne en bas par le stop et ouvert en haut par le trailing,
    n'etait pas classable. Sans stop ni objectif la question ne se pose
    plus : `cibles_m1` rend directement le rendement net a l'horizon, en
    points de base, et c'est exactement ce que le systeme encaisse.
    """
    if getattr(cfg, "timeframe_entrainement", "M5") == "M1":
        import cibles_m1 as _CM
        h = int(getattr(cfg, "horizon_cloture", 30))
        ach, ven, _, _ = _CM.rendements_entree(df, idx, horizon=h)
        # --- MISE A L'ECHELLE DU RISQUE, et elle n'est pas cosmetique. ---
        #
        # `rendements_entree` rend des POINTS DE BASE. Tout l'aval a ete
        # bati pour une cible en unites de risque, d'ecart-type ~1.4 :
        # `coef_mse` y a ete regle, l'ancre d'etendue y compare
        # l'ecart-type du score, et les consommateurs — validation, test,
        # live — passent la conviction dans une SIGMOIDE avant de la
        # comparer a un quantile.
        #
        # LA CIBLE BRUTE CASSE LES TROIS. Mesure du 2026-09-21 sur la
        # fenetre de validation du fold 1, 7 146 occasions :
        #
        #     ecart-type                        20.78 bps
        #     part saturee par la sigmoide      64.5 %
        #
        # Deux tiers des occasions rendues indiscernables : la selectivite
        # disparait avant meme que le modele ait appris quoi que ce soit.
        #
        # LE DIVISEUR EST LA VOLATILITE DE L'HORIZON, pas une constante
        # choisie pour tomber juste. Un trade de `h` minutes se joue contre
        # le mouvement typique de `h` minutes, soit l'ATR d'une barre fois
        # racine de `h`. En divisant, la cible repond a « combien de sigmas
        # ce trade a-t-il pris », qui est sans unite et comparable d'un
        # regime a l'autre.
        #
        # ET C'EST AUSSI CE QUE LE COMPTE ENCAISSE. Le budget dimensionne
        # la position par le risque, donc la taille varie comme 1/ATR et le
        # gain en euros comme rendement/ATR — exactement cette quantite.
        # C'etait deja la raison d'etre du R.
        #
        # CE QUE CA DONNE, sur trois fenetres :
        #
        #     validation fold 1   sd 0.807   sature 0.00 %   rho(bps) +0.964
        #     train fold 1        sd 0.857   sature 0.00 %   rho(bps) +0.944
        #     fold 3 validation   sd 0.901   sature 0.03 %   rho(bps) +0.956
        #
        # L'echelle tient d'une fenetre a l'autre sans etre ajustee, la
        # sigmoide ne sature plus, et le rang avec le profit brut reste a
        # +0.95 : on classe encore les memes trades, a la ponderation par
        # le risque pres.
        #
        # LE DIVISEUR NE LIT QUE LE PASSE : `close` et `atr_14` a la barre
        # `idx-1`, celle qui precede la decision. L'entree, elle, se fait a
        # l'ouverture de `idx`.
        _i = np.asarray(idx, dtype=np.int64)
        _j = np.maximum(_i - 1, 0)
        _px = df["close"].to_numpy(np.float64)[_j]
        _atr = np.maximum(df["atr_14"].to_numpy(np.float64)[_j], 1e-9)
        _sigma = _atr / np.maximum(_px, 1e-9) * 1e4 * np.sqrt(float(h))
        _sigma = np.where(np.isfinite(_sigma) & (_sigma > 1e-6), _sigma, np.nan)
        return ach / _sigma, ven / _sigma
    import cibles as _CI
    return _CI.rendements(df, idx, cfg, indicateur=indicateur)


def _try_load_data_cache(cfg: PPOConfig, date_to: datetime) -> Optional[pd.DataFrame]:
    """Relit le dataframe fusionné mis en cache si sa couverture est suffisante.

    Le calcul des indicateurs coûte ~15 min sur 2.3M bougies, dominé par le
    rang glissant `vol_20.rolling(1440).rank(pct=True)` (≈3.4 milliards d'ops).
    Comme `date_to` vaut `now()` à chaque lancement, on ne compare pas les dates
    à l'identique : un cache qui s'arrête quelques heures avant la fin demandée
    reste valable pour un entraînement sur 4+ ans.
    """
    path = _data_cache_path(cfg)
    h1 = getattr(cfg, "timeframe_entrainement", "M1") == "H1"

    if not os.path.exists(path):
        if h1:
            # En H1 le cache est PRECALCULE par prepare_h1.py. Le reconstruire
            # depuis MT5 exigerait de dupliquer ici les fenetres corrigees a
            # l'echelle, et c'est exactement la duplication qui a deja fait
            # diverger l'alignement H1 dans ce projet.
            raise FileNotFoundError(
                f"{path} absent. Lancer `python prepare_h1.py` : en mode H1 "
                f"le jeu n'est pas reconstruit depuis MT5.")
        _prep = PREPARATEURS_M5.get(cfg.symbol)
        if _prep:
            raise FileNotFoundError(
                f"{path} absent. Lancer `python {_prep}` : le chemin de "
                f"rechargement MT5 ne recupere que du M1/H1 brut et ne "
                f"produit aucune des features Ichimoku, range et H4 — il "
                f"echouerait sur un dropna de ~230 colonnes introuvables.")
        return None
    try:
        df = pd.read_pickle(path)
    except Exception as e:
        print(f"[CACHE] Illisible ({e}) → rechargement depuis MT5.")
        return None

    if "time" not in df.columns or len(df) == 0:
        return None
    manquantes = set(FEATURE_COLS) - set(df.columns)
    if manquantes:
        _prep = PREPARATEURS_M5.get(cfg.symbol) if not h1 else None
        if _prep:
            raise RuntimeError(
                f"{path} : {len(manquantes)} feature(s) absente(s), dont "
                f"{', '.join(sorted(manquantes)[:5])}. Relancer "
                f"`python {_prep}` — le rechargement MT5 ne sait pas "
                f"produire ces colonnes.")
        print(f"[CACHE] Features absentes ({', '.join(sorted(manquantes))}) → "
              f"rechargement depuis MT5.")
        return None

    last = pd.to_datetime(df["time"].iloc[-1])
    retard_h = (date_to - last).total_seconds() / 3600.0
    if h1:
        # Pas de rejet sur la fraicheur : le cache H1 ne peut pas etre
        # reconstruit ici, donc le rejeter ne ferait que tomber dans le chemin
        # M1 et echouer sur des colonnes H4 introuvables. On informe et on
        # continue — c'est a l'operateur de relancer prepare_h1.py s'il veut
        # des barres plus recentes.
        print(f"[CACHE H1] {len(df):,} bougies, {retard_h:.1f}h de retard "
              f"(precalcule par prepare_h1.py, pas de rechargement MT5).")
    elif retard_h > cfg.data_cache_max_lag_hours:
        _prep = PREPARATEURS_M5.get(cfg.symbol)
        if _prep:
            raise RuntimeError(
                f"{path} : {retard_h:.1f} h de retard, au-dela des "
                f"{cfg.data_cache_max_lag_hours:.0f} h tolerees. Relancer "
                f"`python {_prep}` pour le reconstruire.\n"
                f"Le rechargement depuis MT5 n'est PAS une solution de "
                f"repli : il ne recupere que du M1/H1 brut et ne produit "
                f"aucune des features Ichimoku, range et H4.\n"
                f"Pour entrainer sans reconstruire, relever "
                f"`PPOConfig.data_cache_max_lag_hours` — les barres "
                f"manquantes sont en QUEUE d'historique, donc apres la "
                f"fenetre de test, et n'entrent ni dans l'entrainement ni "
                f"dans la mesure.")
        print(f"[CACHE] Trop ancien ({retard_h:.1f}h de retard) → rechargement depuis MT5.")
        return None

    # DROPNA APRES RELECTURE — indispensable, et son absence a coute un crash.
    #
    # Le cache est ecrit avec un dropna portant sur le FEATURE_COLS du moment.
    # Elargir ensuite le jeu de features ne change ni le nom des colonnes (donc
    # le controle ci-dessus passe) ni la date (donc celui du retard passe), mais
    # les lignes ou les NOUVELLES colonnes sont NaN sont toujours la. Elles
    # traversent la normalisation, arrivent au reseau, et ressortent en
    # probabilites NaN : « ValueError: probabilities contain NaN », a mille
    # lieues de sa cause.
    #
    # Exemple vecu : vol_rank_h1 n'est couvert qu'a 95.47 % (glissant de 1440
    # bougies H1, soit 60 jours d'amorcage). Le cache construit pour 10 features
    # relu pour 30 en contenait 4.5 % de lignes empoisonnees.
    avant = len(df)
    df = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns] + ["atr_14"])
    if len(df) < avant:
        print(f"[CACHE] {avant - len(df):,} lignes retirees "
              f"({100*(avant-len(df))/avant:.2f} %) : NaN sur des features que le "
              f"cache ne filtrait pas.")
    if len(df) == 0:
        print("[CACHE] Vide apres filtrage → rechargement depuis MT5.")
        return None

    print(f"[CACHE] {len(df):,} bougies relues depuis {path} "
          f"(retard {retard_h:.1f}h) — chargement MT5 évité.")
    return df.reset_index(drop=True)


def load_mt5_data(cfg: PPOConfig) -> pd.DataFrame:
    date_to_req = cfg.date_to or datetime.now()
    if cfg.use_data_cache:
        cached = _try_load_data_cache(cfg, date_to_req)
        if cached is not None:
            return cached

    print("Connexion MT5…")
    if not mt5.initialize():
        raise RuntimeError("Erreur MT5.init()")

    date_from = cfg.date_from
    date_to = date_to_req
    print(
        f"[DATA] Plage demandée : {date_from:%Y-%m-%d %H:%M} → "
        f"{date_to:%Y-%m-%d %H:%M}"
    )

    # Force MT5 à charger le symbol dans le Market Watch
    mt5.symbol_select(cfg.symbol, True)

    # 1) Tentative directe avec copy_rates_range (rapide quand cache OK)
    rates_m1 = mt5.copy_rates_range(cfg.symbol, cfg.timeframe, date_from, date_to)
    rates_h1 = mt5.copy_rates_range(cfg.symbol, cfg.htf_timeframe, date_from, date_to)

    # Fraction des minutes CALENDAIRES qu'on s'attend a trouver. Le seuil de
    # 0.7 etait cale sur BTCUSD, qui cote 24h/24 et 7j/7. L'or ferme le week-end
    # et fait une pause quotidienne : son maximum atteignable est de
    # (5/7) x (23/24) = 68.5 %, donc 0.7 declenchait une pagination inutile sur
    # un historique pourtant complet (2.82 M bougies recuperees d'un coup).
    n_m1_target = int((date_to - date_from).total_seconds() // 60 * 0.95)

    # 2) Si insuffisant → pagination par copy_rates_from
    if rates_m1 is None or len(rates_m1) < n_m1_target:
        nb = 0 if rates_m1 is None else len(rates_m1)
        print(f"[DATA] copy_rates_range M1 insuffisant ({nb:,} bougies), pagination en cours…")
        rates_m1 = _fetch_paginated(cfg.symbol, cfg.timeframe, date_from, date_to, chunk=100_000)

    if rates_h1 is None or len(rates_h1) < 100:
        nb = 0 if rates_h1 is None else len(rates_h1)
        print(f"[DATA] copy_rates_range H1 insuffisant ({nb:,} bougies), pagination en cours…")
        rates_h1 = _fetch_paginated(cfg.symbol, cfg.htf_timeframe, date_from, date_to, chunk=20_000)

    # 3) Dernier fallback : copy_rates_from_pos avec n_bars (peut être petit)
    if rates_m1 is None or len(rates_m1) == 0:
        print(f"[DATA] Pagination M1 vide, dernier fallback copy_rates_from_pos(0, {cfg.n_bars}).")
        rates_m1 = mt5.copy_rates_from_pos(cfg.symbol, cfg.timeframe, 0, cfg.n_bars)
    if rates_h1 is None or len(rates_h1) == 0:
        n_h1 = max(cfg.n_bars // 5, 5000)
        print(f"[DATA] Pagination H1 vide, dernier fallback copy_rates_from_pos(0, {n_h1}).")
        rates_h1 = mt5.copy_rates_from_pos(cfg.symbol, cfg.htf_timeframe, 0, n_h1)

    # Les features Binance viennent d'un FICHIER, pas de MT5 : elles sont
    # construites hors ligne par build_binance_features.py, qui detecte le
    # decalage horaire du broker empiriquement. Le fichier est deja exprime
    # en heure broker, donc la jointure est exacte a la minute.
    feats_ext = charge_source_externe(date_from, date_to)
    # `point` sert à convertir le spread des bougies (en POINTS) vers le prix,
    # pour que spread_rel ait la même unité qu'en live où il vaut ask − bid.
    _si = mt5.symbol_info(cfg.symbol)
    point = float(_si.point) if _si is not None else 1.0

    mt5.shutdown()

    if rates_m1 is None or rates_h1 is None:
        raise RuntimeError("MT5 n'a renvoyé aucune donnée M1 ou H1")

    print(f"[DATA] M1 bougies brutes : {len(rates_m1):,}  |  H1 : {len(rates_h1):,}")

    if feats_ext is None or len(feats_ext) == 0:
        raise RuntimeError(
            f"Source externe absente : {SOURCE_EXT_NOM}.\n"
            f"Deux des dix features en dépendent. Lancer "
            f"build_binance_features.py d'abord."
        )
    print(f"[{SOURCE_EXT_NOM}] {len(feats_ext):,} minutes")

    # dropna_subset EXPLICITE. Sans lui, merge_m1_h1 fait un dropna() sur TOUTES
    # les colonnes — y compris vol_rank_h1 et high_vol_regime_h1, qui reposent
    # sur un glissant de 1440 bougies H1, soit 60 JOURS d'amorcage. Aucune des
    # deux n'est dans FEATURE_COLS : on jetait donc deux mois de donnees a cause
    # de colonnes que le modele ne voit jamais. Sur une fenetre de 3.75 ans deja
    # bornee par la couverture Binance, c'est 4.6 % du jeu.
    # Les backtests passaient deja ce sous-ensemble ; l'entrainement, non.
    merged = merge_m1_h1(rates_m1, rates_h1, feats_ext=feats_ext, point=point,
                         dropna_subset=FEATURE_COLS + ["atr_14"])
    print(f"{len(merged)} bougies M1 alignées avec H1 + {SOURCE_EXT_NOM} "
          f"après indicateurs.")

    if cfg.use_data_cache:
        path = _data_cache_path(cfg)
        try:
            merged.to_pickle(path)
            print(f"[CACHE] Sauvegardé → {path} "
                  f"({os.path.getsize(path) / 1e6:.0f} Mo) : les prochains lancements "
                  f"sauteront le calcul des indicateurs.")
        except Exception as e:
            print(f"[CACHE] Écriture impossible ({e}) — sans conséquence.")

    return merged


# ============================================================
# FEATURES / NORMALISATION
# ============================================================



def compute_and_save_global_norm_stats(df: pd.DataFrame, feature_cols: List[str], path=NORM_STATS_PATH) -> Dict[str, np.ndarray]:
    X = df[feature_cols].values.astype(np.float32)
    mean, std = X.mean(0), X.std(0)

    # GARDE-FOU : une seule valeur non finie dans une colonne suffit a rendre sa
    # moyenne et son ecart-type NaN, et la normalisation empoisonne alors
    # TOUTES les lignes de cette colonne — pas seulement celles qui etaient
    # mauvaises. Le symptome final est « probabilities contain NaN » au tirage
    # de l'action, sans rapport visible avec la cause.
    #
    # Ecrire un fichier de stats NaN est le pire cas : il est relu ensuite sans
    # broncher a chaque lancement. On refuse de l'ecrire.
    mauvais = [feature_cols[k] for k in range(len(feature_cols))
               if not (np.isfinite(mean[k]) and np.isfinite(std[k]))]
    if mauvais:
        raise ValueError(
            f"Statistiques non finies pour : {', '.join(mauvais)}. Le dataframe "
            f"contient des NaN sur ces colonnes — il n'a pas ete filtre sur le "
            f"jeu de features courant. Refus d'ecrire {NORM_STATS_PATH}, qui "
            f"serait relu en silence a chaque lancement."
        )

    stats = {"mean": mean, "std": std}
    # Les NOMS sont stockés avec les stats : ne vérifier que leur nombre laissait
    # passer un changement de définition à cardinalité constante (remplacer les
    # prix bruts par des ratios, par exemple) — le fichier obsolète était alors
    # rechargé en silence et normalisait avec des mean/std sans rapport.
    if path is not None:
        np.savez(path, mean=mean, std=std, features=np.array(feature_cols))
        print(f"Stats de normalisation TRAIN sauvegardées → {path}")
    return stats


def load_global_norm_stats() -> Optional[Dict[str, np.ndarray]]:
    if not os.path.exists(NORM_STATS_PATH):
        return None
    data = np.load(NORM_STATS_PATH, allow_pickle=True)
    stats = {"mean": data["mean"], "std": data["std"]}

    if stats["mean"].shape[0] != len(FEATURE_COLS):
        print(f"[NORM] Mismatch cardinalité : {stats['mean'].shape[0]} vs "
              f"{len(FEATURE_COLS)} actuelles → recompute.")
        return None

    if "features" not in data:
        print("[NORM] Fichier sans noms de features (format antérieur) → recompute.")
        return None

    cached = [str(x) for x in data["features"]]
    if cached != list(FEATURE_COLS):
        changed = set(cached) ^ set(FEATURE_COLS)
        print(f"[NORM] Features modifiées ({', '.join(sorted(changed))}) → recompute.")
        return None

    print(f"Stats de normalisation GLOBALes chargées depuis → {NORM_STATS_PATH}")
    return stats


class MarketData:
    def __init__(self, df: pd.DataFrame, feature_cols: List[str],
                 stats: Optional[Dict[str, np.ndarray]] = None):
        X = df[feature_cols].values.astype(np.float32)

        if stats is not None:
            # Passe par le noyau : diviser par (std + 1e-8) plutôt que par un std
            # corrigé donnait un écart de ~2e-5 sur l'obs entre training et live.
            X = safe_normalize(X, stats)

        # GARDE-FOU : un NaN ici traverse le reseau sans bruit et ne se
        # manifeste qu'au tirage de l'action, sous la forme
        # « ValueError: probabilities contain NaN » — un message qui ne dit ni
        # quelle colonne ni quelle ligne. On echoue ici, en nommant le coupable.
        if not np.isfinite(X).all():
            mauvais = np.where(~np.isfinite(X).all(axis=0))[0]
            noms = ", ".join(f"{feature_cols[k]} ({int((~np.isfinite(X[:, k])).sum())} lignes)"
                             for k in mauvais)
            raise ValueError(
                f"Valeurs non finies dans les features apres normalisation : {noms}. "
                f"Le dataframe n'a pas ete filtre sur le jeu de features courant "
                f"— cache obsolete, ou dropna_subset incomplet."
            )

        self.features = X
        # Reference au dataframe source, pas une copie : le votant TabM a
        # besoin des colonnes brutes (high, low, atr) pour recalculer ses
        # cibles de barrieres sur exactement la meme fenetre que le rollout.
        self.df = df
        self.close = df["close"].values.astype(np.float32)
        self.open = df["open"].values.astype(np.float32) if "open" in df.columns else self.close.copy()
        self.length = len(df)

        self.atr14 = df["atr_14"].values.astype(np.float32) if "atr_14" in df.columns else np.zeros(len(df), np.float32)
        # LE SPREAD REELLEMENT COTE A CHAQUE BARRE, en points de base.
        #
        # `copy_rates` le renvoie depuis toujours et le preparateur le
        # jetait. Il remplace une loi bimodale inventee — voir
        # `_sample_trade_spread_bps`. Absent du cache, on rend un tableau
        # de zeros et l'environnement retombe sur l'ancien tirage : les
        # caches anterieurs restent lisibles.
        self.spread_bar = (df["spread_bar"].values.astype(np.float32)
                           if "spread_bar" in df.columns
                           else np.zeros(len(df), np.float32))
        self.ema20_h1 = df["ema_20_h1"].values.astype(np.float32) if "ema_20_h1" in df.columns else np.zeros(len(df), np.float32)
        self.high = df["high"].values.astype(np.float32) if "high" in df.columns else np.zeros(len(df), np.float32)
        self.low = df["low"].values.astype(np.float32) if "low" in df.columns else np.zeros(len(df), np.float32)

    def __len__(self):
        return self.length


def create_datasets(df: pd.DataFrame, feature_cols: List[str],
                    stats: Dict[str, np.ndarray], calib_frac: float = 0.35):
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    df_train = df[:train_end].reset_index(drop=True)
    df_val_tout = df[train_end:val_end].reset_index(drop=True)
    df_test = df[val_end:].reset_index(drop=True)

    # Meme decoupe chronologique que create_datasets_from_slices : calibrer sur
    # la periode qu'on evalue ensuite revient a la resumer avant de la trader.
    n_cal = int(len(df_val_tout) * calib_frac)
    df_calib = df_val_tout[:n_cal].reset_index(drop=True)
    df_val = df_val_tout[n_cal:].reset_index(drop=True)

    train_data = MarketData(df_train, feature_cols, stats)
    calib_data = MarketData(df_calib, feature_cols, stats)
    val_data   = MarketData(df_val,   feature_cols, stats)
    test_data  = MarketData(df_test,  feature_cols, stats)

    print(f"SPLIT simple : train={len(df_train)}, calib={len(df_calib)}, "
          f"val={len(df_val)}, test={len(df_test)}")
    return train_data, calib_data, val_data, test_data


def departs_disjoints(longueur: int, lookback: int, pas: int) -> List[int]:
    """Departs d'episodes couvrant la fenetre UNE FOIS, sans chevauchement.

    POURQUOI CETTE FONCTION EXISTE, mesure du 2026-09-15. La validation et le
    test tiraient leurs departs au hasard via `reset()`, avec deux
    consequences qui ont fausse tout le pilotage d'exec10 a exec12 :

      - COUVERTURE PARTIELLE. Le test jouait 5 episodes de 400 barres, soit
        2 000 barres sur 7 934. Trois quarts de la fenetre n'etaient jamais
        joues. Sur 132 trades l'erreur-type du winrate valait 4.1 points,
        quand l'effet cherche en vaut 3 : les trois folds se contredisaient
        (+3.9 / -9.8 / -3.2) par pur bruit d'echantillon. Reevalues sur la
        fenetre entiere, ils se sont alignes a environ -3 points.

      - ECHANTILLON BIAISE. `reset()` passe par le curriculum de volatilite,
        qui tire parmi les 30 % de barres les moins volatiles et les 30 % les
        plus volatiles. Le milieu de la distribution n'etait jamais mesure. Le
        curriculum est un outil d'ENTRAINEMENT ; l'appliquer a la mesure fait
        evaluer autre chose que ce qu'on croit.

    Le train garde ses departs aleatoires : c'est la que le curriculum sert.
    """
    if longueur <= lookback + pas + 2:
        return [lookback]
    return list(range(lookback, longueur - pas - 2, pas))


def reset_au_depart(env, depart: int):
    """reset() puis depart IMPOSE, avec l'observation recalculee.

    `reset()` tire son propre depart au hasard ; on le remplace ensuite. Le
    point delicat est la derniere ligne : l'observation rendue par reset()
    correspond au depart tire, pas a celui qu'on impose. L'oublier ne leve
    aucune erreur — le premier pas de l'episode part simplement d'ailleurs.
    """
    _, info = env.reset()
    env.start_idx = depart
    env.end_idx = depart + env.cfg.episode_length
    env.idx = depart
    return env._get_obs(), info


def create_datasets_from_slices(
    df: pd.DataFrame,
    feature_cols: List[str],
    start: int,
    train_len: int,
    val_len: int,
    test_len: int,
    stats: Dict[str, np.ndarray],
    calib_frac: float = 0.35
):
    n = len(df)
    end = start + train_len + val_len + test_len
    assert end <= n, "Fenêtre walk-forward hors limites"

    df_train = df[start:start + train_len].reset_index(drop=True)
    df_val_tout = df[start + train_len:start + train_len + val_len].reset_index(drop=True)
    df_test  = df[start + train_len + val_len:end].reset_index(drop=True)

    # La fenetre de validation se coupe en deux DANS L'ORDRE DU TEMPS : la
    # premiere partie sert a calibrer les seuils de conviction, la seconde a
    # mesurer. Voir cfg.calib_frac — calibrer sur les episodes qu'on evalue
    # ensuite revient a resumer la periode avant de la trader.
    n_cal = int(len(df_val_tout) * calib_frac)
    df_calib = df_val_tout[:n_cal].reset_index(drop=True)
    df_val   = df_val_tout[n_cal:].reset_index(drop=True)

    train_data = MarketData(df_train, feature_cols, stats)
    calib_data = MarketData(df_calib, feature_cols, stats)
    val_data   = MarketData(df_val,   feature_cols, stats)
    test_data  = MarketData(df_test,  feature_cols, stats)

    print(f"  • Fenêtre WF : train={len(df_train)}, calib={len(df_calib)}, "
          f"val={len(df_val)}, test={len(df_test)} (start={start}, end={end})")
    return train_data, calib_data, val_data, test_data


# ======================================================================
# ENVIRONNEMENT
# ======================================================================

class Portefeuille:
    """UN COMPTE. Plusieurs instruments peuvent le partager.

    POURQUOI IL EXISTE. En production il n'y a qu'un compte MetaTrader : si le
    Bitcoin tient quinze positions, l'or en voit moins de disponibles, parce
    que les deux puisent dans la MEME equite et la MEME marge. Un
    environnement par instrument avec chacun son capital simulerait deux
    comptes qui n'existent pas, et le risque reel vaudrait le double de ce
    qu'on croit.

    L'equite, la marge utilisee et le risque engage se somment donc sur TOUS
    les environnements inscrits, et la capacite d'ouverture en decoule.

    A UN SEUL INSTRUMENT ce compte ne contient qu'un environnement, et chacune
    de ces sommes se reduit a son terme unique : le comportement est alors
    rigoureusement celui d'avant, ce que `test_concurrence.py` verifie.
    """

    def __init__(self, capital: float):
        self.capital = float(capital)
        self.envs: List = []

    def inscrire(self, env) -> None:
        if env not in self.envs:
            self.envs.append(env)

    # A UN SEUL INSTRUMENT, LA SOMME EST SON TERME UNIQUE. Ces quatre
    # fonctions sont appelees plusieurs fois par barre et par environnement ;
    # a un instrument, la boucle et le generateur coutent plus que le calcul
    # qu'ils portent. Le raccourci rend exactement la meme valeur — `0.0 + x`
    # vaut `x` — et le chemin general reste la, inchange, pour le jour ou le
    # Bitcoin revient dans le compte.

    def equity(self) -> float:
        """Capital plus le latent de TOUS les instruments."""
        envs = self.envs
        if len(envs) == 1:
            e = envs[0]
            bid = e.data.close[min(max(e.idx - 1, 0), e.data.length - 1)]
            return self.capital + e._latent_at_bid(bid)
        lat = 0.0
        for e in envs:
            bid = e.data.close[min(max(e.idx - 1, 0), e.data.length - 1)]
            lat += e._latent_at_bid(bid)
        return self.capital + lat

    def marge_utilisee(self) -> float:
        envs = self.envs
        if len(envs) == 1:
            return envs[0].marge_utilisee
        return sum(e.marge_utilisee for e in envs)

    def risque_engage(self) -> float:
        # `risque_engage_local` est la MEME somme qu'avant, gardee tant que
        # les emplacements ne bougent pas. Elle etait recalculee deux fois par
        # barre — masque booleen, indexation avancee, reduction — pour un
        # tableau qui ne change qu'a l'ouverture et a la fermeture.
        envs = self.envs
        if len(envs) == 1:
            return envs[0].risque_engage_local
        return sum(e.risque_engage_local for e in envs)

    def n_positions(self) -> int:
        envs = self.envs
        if len(envs) == 1:
            return envs[0].n_positions
        return sum(e.n_positions for e in envs)


BORNE_LOG = 0.05


def comprime_log(x, borne: float = BORNE_LOG):
    """Borne un log-rendement SANS en detruire l'ordre.

    CE QU'ELLE REMPLACE, et pourquoi la troncature etait un piege.

    La recompense valait `10 x clip(log(E'/E), -0.05, +0.05)`. Le logarithme
    de l'equite est le bon objectif pour un compte qui compose : maximiser
    E[log W] est exactement le critere qui evite la ruine, parce que log(0)
    diverge. C'est ce qui donne a une politique une raison de ne pas tout
    miser, sans qu'on ait a le lui dire.

    LA TRONCATURE COUPAIT PRECISEMENT CETTE PARTIE. Mesure du 2026-09-19 sur
    13 491 barres : la pire barre emportait 91.3 % de l'equite d'un coup, et
    rapportait -0.50 — exactement comme une barre a -4.9 %. Cent quatre-vingts
    barres perdaient plus de 10 %, cinq plus de 50 %, toutes indiscernables.
    Le clip effacait jusqu'a 23.87 de signal sur UNE barre, quand la penalite
    de creux calibree le meme jour vaut au plus 6.5 sur TOUT un episode : on
    soignait le symptome en amont du mal.

    MAIS BORNER RESTE NECESSAIRE, et ce n'est pas negociable. Sans borne,
    l'equite ecretee a 1e-8 donne log(1e-8/950) = -25.3, donc -253 de
    recompense. Mesure a l'epoch 1 : |R| 338, advStd 28168, quasi0 100 % —
    les avantages etant normalises par leur ecart-type, une poignee de
    valeurs geantes ecrase tout le reste sous 0.05 et l'acteur ne recoit
    PLUS AUCUN GRADIENT. Cela a ete essaye et cela detruit l'apprentissage.

    LE DEFAUT N'ETAIT DONC PAS DE BORNER, C'ETAIT DE TRONQUER. Cette forme
    borne en croissance logarithmique tout en restant strictement monotone :

        x = -0.05  (-4.9 %)   ->  -0.035
        x = -0.11  ( -10 %)   ->  -0.055
        x = -0.29  ( -25 %)   ->  -0.089
        x = -2.44  ( -91 %)   ->  -0.195
        x = -25.3  (ruine)    ->  -0.311

    Un ecart de 5.6x entre une perte de 5 % et une perte de 91 %, la ou la
    troncature en donnait 1.0x. Et le cas pathologique rend -0.31 au lieu de
    -25.3 : le probleme d'echelle qui avait impose le clip ne revient pas.

    Elle vaut x au premier ordre (f'(0) = 1), donc les barres ordinaires —
    l'immense majorite — sont inchangees.
    """
    if isinstance(x, np.ndarray):
        return borne * np.sign(x) * np.log1p(np.abs(x) / borne)
    return borne * math.copysign(math.log1p(abs(x) / borne), x)


class BTCTradingEnvDiscrete(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data: MarketData, cfg: PPOConfig):
        super().__init__()
        self.data = data
        self.cfg = cfg
        self.lookback = cfg.lookback

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback, OBS_N_FEATURES),
            dtype=np.float32
        )

        # LE VETO DE TENDANCE, lu une fois. Voir `PPOConfig.veto_tendance`
        # pour la mesure. Une colonne absente du cache desactive le veto
        # bruyamment plutot qu'en silence.
        _vt = getattr(cfg, "veto_tendance", "")
        if _vt and _vt not in data.df.columns:
            raise ValueError(
                f"veto_tendance='{_vt}' absent du cache. Colonnes de tendance "
                f"disponibles : {[c for c in data.df.columns if c.startswith('tend_')]}")
        self._tendance = (data.df[_vt].to_numpy(np.float64) if _vt else None)
        self._seuil_tendance = float(getattr(cfg, "veto_tendance_seuil", 0.0))

        if self.cfg.use_vol_curriculum:
            self._init_vol_curriculum()
        else:
            self.low_vol_starts = None
            self.high_vol_starts = None

        self.risk_scale = 1.0
        self.last_risk_scale = 1.0
        self.reset()

    # ---------- Curriculum vol ----------

    def _init_vol_curriculum(self):
        """Les departs de faible et de forte volatilite, UNE FOIS par donnees.

        CHAQUE ENVIRONNEMENT LES RECALCULAIT ET LES GARDAIT, alors qu'ils ne
        dependent que des donnees, de `lookback` et de `episode_length` —
        les memes pour tous les environnements d'un pool. Mesure du
        2026-09-25 : 2.6 Mo par environnement, 160 environnements crees
        d'avance, donc 0.42 Go, sur une machine de 15.7 Go ou l'entrainement
        en occupait 13.6 et ou une fenetre a cesse de repondre.

        LES TABLEAUX SONT PARTAGES EN LECTURE SEULE : `reset` n'en fait que
        tirer un indice. Le verrou d'ecriture le garantit — une ecriture
        leverait une erreur au lieu de modifier les departs de tous.
        """
        _cle = (int(self.lookback), int(self.cfg.episode_length))
        _cache = getattr(self.data, "_departs_vol", None)
        if _cache is None:
            _cache = {}
            try:
                self.data._departs_vol = _cache
            except AttributeError:
                _cache = None
        if _cache is not None and _cle in _cache:
            self.low_vol_starts, self.high_vol_starts = _cache[_cle]
            return
        self._calcule_departs_vol()
        for _a in (self.low_vol_starts, self.high_vol_starts):
            if _a is not None:
                _a.setflags(write=False)
        if _cache is not None:
            _cache[_cle] = (self.low_vol_starts, self.high_vol_starts)

    def _calcule_departs_vol(self):
        close = self.data.close
        ret = np.diff(close) / (close[:-1] + 1e-8)
        vol20 = pd.Series(ret).rolling(20).std().to_numpy()
        vol20 = np.concatenate([[np.nan], vol20])

        valid = ~np.isnan(vol20)
        if valid.sum() < 30:
            self.low_vol_starts = None
            self.high_vol_starts = None
            return

        q_low, q_high = np.quantile(vol20[valid], [0.3, 0.7])

        candidate_low = np.where((vol20 <= q_low) & valid)[0]
        candidate_high = np.where((vol20 >= q_high) & valid)[0]

        max_start = self.data.length - self.cfg.episode_length - 2
        low = candidate_low[
            (candidate_low >= self.lookback) &
            (candidate_low <= max_start)
        ]
        high = candidate_high[
            (candidate_high >= self.lookback) &
            (candidate_high <= max_start)
        ]

        self.low_vol_starts = low if len(low) > 0 else None
        self.high_vol_starts = high if len(high) > 0 else None

    # `set_budget_part` A ETE RETIREE LE 2026-09-21.
    #
    # Elle posait la part du plafond que la position allait occuper. Trois
    # mesures l'ont condamnee, et aucune ne porte sur le budget lui-meme :
    #
    #   la regle de capacite demandait deja la capacite a PLEIN depuis le
    #   passage a une position unique (`_b = 1.0`) ;
    #
    #   la taille posee valait 1.0000 unite, p5 0.9999, p95 1.0001, sur
    #   7 363 trades — les six paliers rendaient tous la meme taille ;
    #
    #   la taille voulue par le risque vaut 0.003 lot a 1 000 $, soit trois
    #   fois moins que le lot minimum du courtier. Le plancher s'impose, et
    #   aucun mecanisme de dimensionnement ne peut rien y changer sous
    #   3 377 $ de capital.
    #
    # LA CAPITALISATION N'EN DEPEND PAS : `_compute_dynamic_size` calcule
    # `capital x risk_per_trade / distance_de_stop`, donc la taille suivra
    # l'equite des que le compte depassera le plancher.
    def set_risk_scale(self, scale: float):
        self.risk_scale = float(max(scale, 0.0))
        if self.risk_scale <= 0.0:
            self.risk_scale = 1.0
        self.last_risk_scale = float(self.risk_scale)

    # ---------- Gym API ----------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = self.data.length - self.cfg.episode_length - 2

        start_idx = None
        if self.cfg.use_vol_curriculum and self.low_vol_starts is not None and self.high_vol_starts is not None:
            if np.random.rand() < 0.5 and len(self.low_vol_starts) > 0:
                start_idx = int(np.random.choice(self.low_vol_starts))
            elif len(self.high_vol_starts) > 0:
                start_idx = int(np.random.choice(self.high_vol_starts))

        if start_idx is None:
            start_idx = np.random.randint(self.lookback, max_start)

        self.start_idx = start_idx
        self.end_idx = self.start_idx + self.cfg.episode_length
        self.idx = self.start_idx

        # LE CAPITAL APPARTIENT AU COMPTE, pas a l'instrument. Sans compte
        # explicite, chaque environnement en cree un pour lui seul — ce qui
        # reproduit exactement le comportement d'avant.
        if getattr(self, "portefeuille", None) is None:
            self.portefeuille = Portefeuille(self.cfg.initial_capital)
        self.portefeuille.inscrire(self)
        self.portefeuille.capital = self.cfg.initial_capital

        # LES POSITIONS SONT DES EMPLACEMENTS, pas des scalaires. Un tableau
        # par champ plutot qu'une liste d'objets : les barrieres se testent
        # alors en une operation vectorielle sur tous les emplacements, ce qui
        # rend le cout par barre insensible a K.
        #
        # `_p_sens` a 0 marque un emplacement LIBRE. C'est la seule source de
        # verite sur l'occupation — aucun compteur separe, qui finirait par
        # diverger sans lever d'erreur.
        k = max(int(getattr(self.cfg, "positions_max", 1)), 1)
        self._K = k
        self._p_sens = np.zeros(k, dtype=np.int64)
        self._p_entree = np.zeros(k, dtype=np.float64)
        self._p_taille = np.zeros(k, dtype=np.float64)
        self._p_sl = np.zeros(k, dtype=np.float64)
        self._p_tp = np.zeros(k, dtype=np.float64)
        self._p_atr = np.zeros(k, dtype=np.float64)
        # LE COUT D'ENTREE, EN ATR D'ENTREE : l'ecart entre le prix paye
        # et l'ouverture de la barre. C'est le latent d'une position dont
        # le marche n'a pas bouge — son POINT MORT. Voir `cout_entree_atr`.
        self._p_cout = np.zeros(k, dtype=np.float64)
        self._p_idx = np.full(k, -1, dtype=np.int64)
        self._p_risque = np.zeros(k, dtype=np.float64)
        self._p_spread = np.full(k, float(self.cfg.spread_bps), dtype=np.float64)
        self._p_be = np.zeros(k, dtype=bool)
        self._p_trail = np.zeros(k, dtype=bool)
        # Recompense attribuee a chaque emplacement pendant la barre courante.
        self._r_slots = np.zeros(k, dtype=np.float64)
        self._realise_slots = np.zeros(k, dtype=np.float64)
        self._latent_prec = np.zeros(k, dtype=np.float64)
        self._slots_fermes: List[int] = []
        self._slot_ouvert = -1
        self._notionnel = 0.0
        self._cap_cle, self._cap_val = None, {}
        self._ver = 0
        self._c_actifs = None
        self._c_latent = None
        self._c_risque = None
        self._c_agr = None
        self._c_n = -1
        self._touche()

        # LES CONSTANTES DU COURTIER, LUES UNE FOIS. `_places_ouvrables` et
        # `_taille_quantifiee` les relisaient par `getattr(self.cfg, ...,
        # defaut)` a chaque appel : releve au profileur, 696 764 `getattr` et
        # 618 471 `max` pour 20 000 barres, soit trente-cinq de chaque par
        # barre. Elles ne changent pas pendant un episode — l'environnement
        # est reconstruit ou remis a zero entre deux — donc les relire est du
        # pur cout d'appel.
        #
        # LES NOMS ET LES DEFAUTS SONT CEUX D'AVANT, terme pour terme : c'est
        # la meme lecture, faite une fois.
        c = self.cfg
        self._k_marge_frac = float(getattr(c, "marge_frac", 0.001734))
        self._k_seuil_marge = float(getattr(c, "niveau_marge_ouverture", 3.0))
        self._k_lot_min = float(getattr(c, "lot_min", 0.01))
        self._k_lot_pas = float(getattr(c, "lot_pas", 0.01))
        self._k_contrat = float(getattr(c, "contrat", 1.0))
        self._k_realiste = bool(getattr(c, "marge_realiste", False))

        self.risk_amount = 0.0
        # Spread du trade en cours — réé-échantillonné à chaque ouverture
        self.current_trade_spread_bps = float(self.cfg.spread_bps)

        self.last_realized_pnl = 0.0
        self.peak_capital = self.capital
        self.trades_pnl: List[float] = []
        # Side (+1 long, -1 short) du trade fermé — index aligné avec trades_pnl
        self.trades_side: List[int] = []
        # Métadonnées trade-by-trade : dict par trade fermé
        # {entry_idx, exit_idx, side, entry_price, exit_price, pnl, hit_sl, hit_tp, hold_bars}
        self.trades_meta: List[Dict] = []

        self.risk_scale = 1.0
        self.last_risk_scale = 1.0

        self.max_dd = 0.0
        # Le potentiel de creux de la barre precedente. A zero au depart :
        # l'episode commence a son sommet, donc rien n'est du. C'est ce qui
        # fait que le total facture sur l'episode vaut exactement le creux
        # FINAL, et non la somme des creux traverses.
        self._d_prec = 0.0

        obs = self._get_obs()
        return obs, {
            "capital": self.capital,
            "position": self.position,
            "drawdown": 0.0,
            "done_reason": None
        }

    # ------------------------------------------------------------------
    # VUES SCALAIRES SUR LES EMPLACEMENTS. Une trentaine d'endroits — le live,
    # les evaluateurs, le journal, l'observation — lisent `env.position` ou
    # `env.current_size`. Les exposer en lecture seule, derivees des
    # emplacements, evite deux etats a tenir d'accord : c'est la faute que ce
    # depot paie en boucle des qu'une valeur est recopiee au lieu d'etre lue.
    #
    # A K=1 chacune rend exactement ce que l'attribut rendait avant.
    # ------------------------------------------------------------------
    # UNE VERSION D'ETAT, ET TOUT CE QUI EN DEPEND CESSE D'ETRE REFAIT.
    #
    # Releve au profileur sur 20 000 barres, geometrie de l'or, entrees
    # neutres : 819 548 reductions numpy pour 20 000 pas — QUARANTE ET UNE
    # par barre — dont 479 595 `.sum()` et 319 961 `.any()`, toutes sur des
    # tableaux de 64 emplacements. A cette taille une reduction numpy ne
    # calcule presque rien : elle paie son cout d'appel. Le poste suivant,
    # `_latent_at_bid`, etait appele QUATRE fois par barre — 1.44 s sur 7.74,
    # soit 19 % du pas — pour un etat qui ne change qu'a l'ouverture et a la
    # fermeture.
    #
    # ON NE CHANGE AUCUN CALCUL. `_ver` s'incremente a chaque mutation des
    # emplacements, et ce qui s'en deduit est garde tant qu'elle ne bouge
    # pas : memes tableaux, memes ordres de sommation, mêmes valeurs au bit
    # pres. `test_concurrence.py` le verifie sur 613 trades et six scenarios
    # a geometrie figee, et `verifie_caches()` ci-dessous recalcule tout a
    # froid pour le confronter au cache.
    #
    # LE PIEGE A EVITER EST CONNU : un compteur qui derive silencieusement de
    # ce qu'il resume. D'ou une seule porte — `_touche()` — appelee aux
    # quatre endroits ou `_p_sens` change : reset, agrandissement, ouverture,
    # fermeture.
    # ------------------------------------------------------------------
    def _touche(self) -> None:
        """Les emplacements ont change : ce qui en derive est perime."""
        self._ver += 1
        self._c_actifs = None
        self._c_latent = None
        self._c_risque = None
        self._c_agr = None
        self._c_n = -1

    def verifie_caches(self) -> None:
        """Le cache dit-il la meme chose qu'un calcul a froid ? Leve sinon."""
        a = self._p_sens != 0
        assert np.array_equal(self._actifs, a), "cache _actifs perime"
        assert self.n_positions == int(np.count_nonzero(self._p_sens)), \
            "cache n_positions perime"
        assert self.risque_engage_local == float(self._p_risque[a].sum()), \
            "cache risque_engage perime"
        agr, self._c_agr = self._agregats(), None
        assert agr == self._agregats(), "cache des agregats perime"

    @property
    def _actifs(self):
        c = self._c_actifs
        if c is None:
            c = self._p_sens != 0
            self._c_actifs = c
        return c

    @property
    def n_positions(self) -> int:
        n = self._c_n
        if n < 0:
            n = int(np.count_nonzero(self._p_sens))
            self._c_n = n
        return n

    @property
    def risque_engage_local(self) -> float:
        """Somme des montants a perdre si tous les stops etaient touches."""
        r = self._c_risque
        if r is None:
            r = float(self._p_risque[self._actifs].sum())
            self._c_risque = r
        return r

    @property
    def notionnel(self) -> float:
        """Somme des notionnels ouverts, tenue a jour a l'ouverture et a la
        fermeture plutot que recalculee : elle est lue plusieurs fois par
        barre et ne change que deux fois par trade."""
        return float(self._notionnel)

    @property
    def marge_utilisee(self) -> float:
        return self.notionnel * self._k_marge_frac

    def _equity_courante(self) -> float:
        """L'equite du COMPTE, tous instruments confondus."""
        return self.portefeuille.equity()

    def _taille_quantifiee(self, taille: float) -> float:
        """Ce que le courtier accepterait reellement.

        Le lot minimum n'est pas un detail d'execution : a 12xATR la taille
        visee vaut 0.0045 BTC contre un minimum de 0.01, donc le courtier
        impose 2.2 fois la position voulue. L'ignorer, c'est s'entrainer sur
        un courtier qui n'existe pas.
        """
        if not self._k_realiste:
            return taille
        # LE LOT N'EST PAS L'UNITE. `taille` est en unites de l'instrument —
        # bitcoins ou onces — alors que le courtier quantifie en LOTS, et un
        # lot d'or vaut CENT onces. Quantifier a 0.01 sans passer par la
        # taille du contrat ferait trader un centieme d'once la ou le minimum
        # reel est une once : cent fois trop petit, et le risque annonce
        # n'aurait aucun rapport avec le risque joue.
        contrat = self._k_contrat
        pas = self._k_lot_pas * contrat
        mini = self._k_lot_min * contrat
        if taille < mini:
            # Le courtier ne sait pas faire plus petit. On prend le minimum,
            # comme le fait `kairos_live`, et le risque reel depasse la cible.
            return mini
        return math.floor(taille / pas + 1e-9) * pas

    def places_ouvrables(self, prix: float, budget=None) -> int:
        # Memorisee pour la barre : la boucle de collecte la demande via
        # `peut_entrer()` puis via `_get_obs()`, et rien entre les deux ne
        # peut la changer. La cle porte l'etat dont elle depend, donc une
        # ouverture ou une fermeture l'invalide d'elle-meme.
        # La cle porte l'etat du COMPTE : une ouverture sur l'autre
        # instrument change la capacite de celui-ci, et la memo doit s'en
        # apercevoir. Sans cela, l'or continuerait d'ouvrir sur une capacite
        # calculee avant que le Bitcoin ne consomme le budget.
        # LE PRIX MANQUAIT A LA CLE, et ce n'etait pas un detail de cache.
        # `peut_entrer()` interroge au CLOSE de la barre, `step()` a son
        # OUVERTURE : deux prix differents dans la meme barre, qui
        # renvoyaient pourtant la valeur memorisee du premier. Le prix entre
        # dans `m_une`, dans le plancher d'ATR et dans la taille quantifiee,
        # donc la capacite annoncee n'etait pas celle du prix demande.
        #
        # L'etat du COMPTE est desormais resume par les versions de ses
        # environnements : une ouverture ou une fermeture, ici ou sur l'autre
        # instrument, les fait bouger. C'est exact et cela ne coute aucune
        # reduction — la cle precedente en demandait quatre.
        # DEUX PRIX PAR BARRE, DEUX ENTREES. La cloture et l'ouverture sont
        # interrogees en alternance dans la meme barre : une memo a une seule
        # case les chasserait l'une l'autre et recalculerait tout a chaque
        # fois. L'etat du COMPTE est resume par les versions de ses
        # environnements — une ouverture ou une fermeture, ici ou sur l'autre
        # instrument, les fait bouger.
        # LE BUDGET EST SORTI DE LA CLE le 2026-09-21. Il y etait entre
        # parce que le modele le choisissait a chaque decision et que
        # `peut_entrer(au_plafond=True)` interrogeait la capacite sous un
        # budget qui n'etait pas celui en vigueur — les deux reponses se
        # chassaient. Les deux raisons ont disparu ensemble : il n'y a plus
        # de budget, et `peut_decider` ne demande plus la capacite au
        # plafond.
        cle = (self.idx, self.capital,
               tuple(e._ver for e in self.portefeuille.envs))
        if self._cap_cle != cle:
            self._cap_cle, self._cap_val = cle, {}
        val = self._cap_val.get(prix)
        if val is None:
            val = self._places_ouvrables(prix)
            self._cap_val[prix] = val
        return val

    def _places_ouvrables(self, prix: float, budget=None) -> int:
        """Peut-on ouvrir, ici et maintenant ? Rend 0 ou 1.

        UNE SEULE POSITION A LA FOIS, depuis le 2026-09-21. La fonction
        rendait un NOMBRE de places — c'est le K du portefeuille, celui que
        le budget arbitrait. Il n'y a plus de portefeuille : il y a une
        position, ou aucune.

        LA REGLE EST POSEE ICI ET NULLE PART AILLEURS, et c'est le point
        important. `peut_entrer` la consulte pour savoir s'il faut
        interroger le modele ; `step` la consulte avant d'ouvrir. Un
        premier essai ne l'avait mise que dans `peut_entrer` — `step`
        ouvrait sans demander, et la mesure a compte VINGT-HUIT positions
        simultanees sur un episode. Les deux chemins passent par cette
        fonction ; elle seule doit porter la regle.

        LA MARGE GARDE SON MOT A DIRE : si le compte ne peut pas financer
        meme une position, la reponse reste zero.
        """
        if self.n_positions > 0:
            return 0
        # Le nombre d'emplacements libres n'est PAS une limite : les
        # tableaux doublent a la demande. Ce qui limite, c'est le solde, et
        # cette fonction doit donc dire ce que le SOLDE permet — pas ce que
        # l'allocation courante contient. La borner par la taille du tableau
        # la ferait mentir des que l'equity depasse ce qui a ete alloue, et
        # c'est exactement ce que le modele doit voir changer quand il gagne.
        # LE CALCUL VIT DANS `saint_core`, pas ici. L'environnement et
        # `kairos_live` appellent la MEME fonction : c'est la seule facon
        # qu'ils aient de rester d'accord sur le nombre de positions
        # ouvrables. Ici on ne fait que lui fournir l'etat du compte.
        if not self._k_realiste:
            return max(self._K - self.n_positions, 0)

        # Le risque d'une position de plus, a la taille que le courtier
        # imposerait reellement — lot minimum compris. C'est l'UNITE dans
        # laquelle le modele exprime son budget depuis le 2026-09-20 : il
        # choisit un NOMBRE de positions minimales, pas une fraction
        # d'equite.
        # LE BUDGET A DISPARU, D'ABORD AVEC LES POSITIONS MULTIPLES puis
        # tout a fait le 2026-09-21. C'est la marge et le plafond de
        # survie, eux seuls, qui disent si l'unique position peut s'ouvrir.

        # L'ABSTENTION N'EST PLUS UN PALIER ZERO. Elle est l'appartenance
        # au sommet du classement : hors des `q %` du haut, on n'entre pas.
        # Le garde-fou vivait a deux
        # endroits, l'env et le live : deux endroits ou se tromper, pour une
        # regle qui doit etre la meme. Il est maintenant dans la fonction que
        # les deux appellent.

        atr_raw = (float(self.data.atr14[self.idx - 1])
                   if self.idx - 1 >= 0 else 0.0)
        atr = max(atr_raw, ATR_PLANCHER_FRAC * prix, 1e-8)
        taille = self._taille_quantifiee(self._compute_dynamic_size(prix))
        r_une = max(self.cfg.atr_sl_mult * atr * taille, 1e-12)

        return places_ouvrables_compte(
            equity=self._equity_courante(),
            marge_utilisee=self.portefeuille.marge_utilisee(),
            prix=prix,
            lot_min=self._k_lot_min,
            contrat=self._k_contrat,
            marge_frac=self._k_marge_frac,
            niveau_marge=self._k_seuil_marge,
            risque_engage=self.portefeuille.risque_engage(),
            risque_une=r_une)

    def _agrandir(self) -> None:
        """Double le nombre d'emplacements. Il n'y a pas de plafond.

        Les tableaux sont une allocation, pas une limite : quand ils se
        remplissent alors que le solde autorise davantage, ils doublent. Un
        plafond fixe masquerait le cercle vertueux des que l'equity depasse
        ce qu'il permet.
        """
        k = self._K
        self._K = k * 2
        def _e(a, v=0):
            return np.concatenate([a, np.full(k, v, dtype=a.dtype)])
        self._p_sens = _e(self._p_sens)
        self._p_entree = _e(self._p_entree)
        self._p_taille = _e(self._p_taille)
        self._p_sl = _e(self._p_sl)
        self._p_tp = _e(self._p_tp)
        self._p_atr = _e(self._p_atr)
        self._p_cout = _e(self._p_cout)
        self._p_idx = _e(self._p_idx, -1)
        self._p_risque = _e(self._p_risque)
        self._p_spread = _e(self._p_spread, float(self.cfg.spread_bps))
        self._p_be = _e(self._p_be, False)
        self._p_trail = _e(self._p_trail, False)
        self._r_slots = _e(self._r_slots)
        self._realise_slots = _e(self._realise_slots)
        self._touche()
        self._latent_prec = _e(self._latent_prec)

    def peut_financer_un_lot(self, prix: float) -> bool:
        """Le SOLDE permet-il encore un lot minimum ? La ruine, c'est cela.

        A NE PAS CONFONDRE AVEC `places_ouvrables`. Celle-ci applique aussi le
        budget de risque, qui est NOTRE politique : quand un pic de volatilite
        fait qu'un lot minimum depasse 1 % de l'equity, elle rend zero alors
        que le compte se porte tres bien. Premiere version de la regle de
        ruine : un compte a +16 % avec 4 % de creux etait declare mort parce
        qu'il ne pouvait pas ouvrir A CET INSTANT.

        La ruine est une contrainte de COURTIER, pas de politique : c'est
        quand la marge ne suffit plus au plus petit ordre possible. Ne pas
        vouloir trader n'est pas mourir.
        """
        if not self._k_realiste:
            return True
        eq = self._equity_courante()
        if eq <= 0:
            return False
        frac = self._k_marge_frac
        seuil = self._k_seuil_marge
        m_une = max(prix * self._k_lot_min * self._k_contrat * frac, 1e-12)
        # La marge deja immobilisee est celle du COMPTE : les positions de
        # l'autre instrument la consomment aussi.
        return (eq / max(seuil, 1e-9)
                - self.portefeuille.marge_utilisee()) >= m_une

    def peut_decider(self) -> bool:
        """Le reseau doit-il etre consulte a cette barre ?

UNE SEULE SOURCE pour les trois boucles — rollout, validation, test.
        Les avoir ecrites separement, c'est exactement comment ce depot
        s'est retrouve avec une validation qui jouait une autre strategie
        que l'entrainement.

        ELLE AVAIT UNE SECONDE RAISON, RETIREE LE 2026-09-21. Toutes les
        `revision_budget_barres` barres, on rendait la parole au modele
        pour qu'il puisse DESSERRER son budget sans attendre qu'une
        position se ferme. Sans budget il n'y a plus rien a desserrer.

        ATTENTION A CE QU'ELLE NE DIT PAS. Elle ne rend vrai que si
        l'environnement peut ENTRER — donc jamais en position. Les etats en
        position sont traites a part, par `demande_cloture`, et c'est un
        oubli de ce point qui a produit la position immortelle du meme
        jour : un seul trade par episode de 5 760 barres.
        """
        return self.peut_entrer()

    def peut_entrer(self, au_plafond: bool = False) -> bool:
        """Une DECISION existe quand le solde permet encore une position.

        `au_plafond` REND LE BUDGET REVISABLE, et c'est tout l'objet du
        parametre.

        Le budget etait choisi a chaque decision, mais une decision n'existait
        que si le budget COURANT laissait de la place. Un modele qui choisit
        3 % et ouvre une position remplit son budget, sort de `deciding`, et
        n'est plus consulte jusqu'a ce que la position se ferme — environ
        1 140 barres avec un stop a 10xATR. Il pouvait donc se verrouiller
        dans un budget serre sans pouvoir le relever. Mesure du 2026-09-19 :
        44 decisions par episode de 5 760 barres, soit une tous les 130 barres.

        En interrogeant la capacite au budget le PLUS LARGE, l'environnement
        est consulte des qu'une position serait possible A UN BUDGET
        QUELCONQUE. Le modele choisit alors son budget, et c'est ce choix qui
        ouvre ou refuse l'entree — au lieu qu'un choix passe la lui interdise.
        """
        if not self.tendance_favorable():
            return False
        # ==============================================================
        # UNE SEULE POSITION A LA FOIS, LE 2026-09-21.
        #
        # C'est la regle, et elle est posee ICI parce que `peut_entrer` est
        # la source unique de la decision d'entree — rollout, validation,
        # test et live y passent tous. L'ecrire ailleurs en plus ferait
        # deux endroits ou se tromper, et ce depot l'a deja paye.
        #
        # CE QUE CELA SUPPRIME. Le BUDGET n'avait de sens que pour arbitrer
        # entre PLUSIEURS positions simultanees : combien en ouvrir a la
        # fois. Avec une seule, la question disparait entierement — et avec
        # elle `BUDGETS_PART`, `part_du_rang`, `set_budget_part` et le terme
        # de budget de `places_ouvrables_compte`. Toute la journee du
        # 2026-09-21 a ete passee a regler cet arbitrage ; il n'existe plus.
        #
        # CE QUI RESTE A ARBITRER, ET C'EST AUTRE CHOSE : la TAILLE de
        # l'unique position. Sans stop, elle ne peut plus se deduire d'une
        # distance de stop ; elle doit se borner au creux que le compte
        # supporte. Voir `cibles_m1.rendements_entree`, qui rend ce creux
        # pour chaque occasion — une position sur cent en traverse 80 bps.
        #
        # `au_plafond` N'A PLUS D'OBJET : il servait a interroger la
        # capacite au budget le plus large pour qu'un budget serre ne
        # verrouille pas le modele. Sans budget, il n'y a rien a relever.
        # ==============================================================
        if self.n_positions > 0:
            return False
        prix = float(self.data.close[min(self.idx, self.data.length - 1)])
        return self.places_ouvrables(prix) > 0

    def tendance_favorable(self) -> bool:
        """Le regime autorise-t-il une entree a cette barre ?

        POSE ICI ET NULLE PART AILLEURS. `peut_entrer` est la source unique
        de la decision d'entree — rollout, validation et test y passent tous.
        Un veto ecrit en trois endroits finirait par differer entre eux, et
        c'est la faute que ce depot a deja payee plusieurs fois.

        Le seuil et la colonne viennent de la configuration : voir
        `PPOConfig.veto_tendance` pour la mesure qui les justifie. Une colonne
        vide desactive le veto et l'environnement retrouve exactement son
        comportement d'avant.
        """
        if self._tendance is None:
            return True
        # LA COLONNE BRUTE, PAS CELLE DE `features`. Cette derniere est
        # NORMALISEE : comparer un seuil de 0 a une valeur centree-reduite
        # reviendrait a couper a la MOYENNE du momentum, pas a zero — soit un
        # tout autre filtre, et sans que rien ne le signale. `MarketData`
        # garde le dataframe source, on y lit donc la valeur telle que la
        # mesure l'a vue.
        i = min(max(self.idx - 1, 0), len(self._tendance) - 1)
        v = float(self._tendance[i])
        return bool(np.isfinite(v) and v >= self._seuil_tendance)

    # CINQ AGREGATS, UNE SEULE PASSE, ET ILS NE CHANGENT QU'AVEC LES
    # EMPLACEMENTS. `_get_obs` lit `position` deux fois, `entry_price` et
    # `entry_atr` chacun deux fois, `bars_in_position` une : releve au
    # profileur, cinq lectures de `position` par barre, chacune refaisant un
    # masque booleen, deux indexations avancees et une reduction. Les
    # expressions sont recopiees terme pour terme — meme ordre de sommation,
    # meme division — seule leur frequence change.
    def _agregats(self):
        c = self._c_agr
        if c is not None:
            return c
        a = self._actifs
        if not a.any():
            c = (0, 0.0, 0.0, 0.0, -1)
        else:
            t = self._p_taille[a]
            st = t.sum()
            net = float((self._p_sens[a] * t).sum())
            pos = 0 if net == 0.0 else (1 if net > 0 else -1)
            taille = float(np.abs(t).sum())
            if st > 0:
                entree = float((self._p_entree[a] * t).sum() / st)
                atr = float((self._p_atr[a] * t).sum() / st)
            else:
                entree = atr = 0.0
            c = (pos, taille, entree, atr, int(self._p_idx[a].min()))
        self._c_agr = c
        return c

    @property
    def position(self) -> int:
        """Sens NET. A K=1, le sens de l'unique position."""
        return self._agregats()[0]

    @property
    def current_size(self) -> float:
        return self._agregats()[1]

    @property
    def entry_price(self) -> float:
        """Prix d'entree moyen PONDERE PAR LA TAILLE. A K=1, le prix exact."""
        return self._agregats()[2]

    @property
    def entry_atr(self) -> float:
        return self._agregats()[3]

    @property
    def cout_entree_atr(self) -> float:
        """Le POINT MORT de la position, en ATR d'entree : ce que l'entree a
        coute — spread et glissement — avant que le marche bouge.

        UNE POSITION DONT LE LATENT VAUT `-cout_entree_atr` N'A RIEN PERDU
        SUR LE MARCHE : elle a paye son ticket. Mesure du 2026-09-25 sur le
        train du fold 1 : ce cout vaut 0.389 ATR en mediane, et il met 66 %
        des positions EN PERTE a leur premiere decision, longs comme shorts.

        Moyenne ponderee par la taille, comme `entry_atr` ; a K=1, le cout
        exact de l'unique position. Zero a plat.
        """
        a = self._actifs
        if not a.any():
            return 0.0
        t = self._p_taille[a]
        st = t.sum()
        return float((self._p_cout[a] * t).sum() / st) if st > 0 else 0.0

    @property
    def entry_idx(self) -> int:
        return self._agregats()[4]

    @property
    def bars_in_position(self) -> int:
        """Detention de la position la PLUS ANCIENNE encore ouverte."""
        e = self._agregats()[4]
        return int(self.idx - e) if e >= 0 else 0

    @property
    def sl_price(self) -> float:
        a = np.flatnonzero(self._actifs)
        return float(self._p_sl[a[0]]) if len(a) else 0.0

    @property
    def tp_price(self) -> float:
        a = np.flatnonzero(self._actifs)
        return float(self._p_tp[a[0]]) if len(a) else 0.0

    @property
    def break_even_done(self) -> bool:
        a = np.flatnonzero(self._actifs)
        return bool(self._p_be[a[0]]) if len(a) else False

    @property
    def trail_active(self) -> bool:
        a = np.flatnonzero(self._actifs)
        return bool(self._p_trail[a[0]]) if len(a) else False

    def _latents_par_slot(self, bid) -> np.ndarray:
        """Le latent de CHAQUE emplacement, en une operation.

        `execution_quote` n'est qu'une multiplication — un long sort au BID,
        un short paie l'ASK — donc elle se met en tableau sans rien changer au
        resultat. Les emplacements libres rendent zero, leur sens valant 0.

        RESTREINDRE LE CALCUL AUX EMPLACEMENTS OUVERTS A ETE ESSAYE, ET C'EST
        PLUS LENT : 4 400 barres/s contre 5 076, mesure trois fois. A cette
        taille l'indexation booleenne et l'allocation du tampon coutent plus
        que la passe complete sur soixante-quatre elements contigus. On garde
        donc la forme la plus simple — c'est aussi la plus rapide ici.
        """
        sens = self._p_sens.astype(np.float64)
        q = float(bid) * np.where(self._p_sens == -1,
                                  1.0 + np.maximum(self._p_spread, 0.0) / 1e4,
                                  1.0)
        return ((sens * (q - self._p_entree) - self.cfg.fee_rate * q)
                * self._p_taille)

    def _latent_slot(self, bid, i: int) -> float:
        if self._p_sens[i] == 0:
            return 0.0
        q = execution_quote(bid, -int(self._p_sens[i]), float(self._p_spread[i]))
        return (self._p_sens[i] * (q - self._p_entree[i])
                - self.cfg.fee_rate * q) * self._p_taille[i]
    # ------------------------------------------------------------------

    def _get_obs(self):
        start = self.idx - self.lookback
        base = self.data.features[start:self.idx]

        if self.idx > 0:
            current_price = float(self.data.close[self.idx - 1])
        else:
            current_price = 0.0

        # PnL latent en unités d'ATR (scale-invariant, typiquement dans [-5, 5])
        if self.position != 0 and self.entry_atr > 1e-8 and self.entry_price > 0.0:
            unrealized_atr = float(
                self.position * (current_price - self.entry_price) / self.entry_atr
            )
        else:
            unrealized_atr = 0.0

        # Durée de détention normalisée (0=flat, 1=max_holding, >1=overtime)
        bars_held_norm = float(
            min(self.bars_in_position / max(self.cfg.scalping_max_holding, 1), 3.0)
        )

        pos_feature  = float(self.position)
        # LA QUATRIEME COLONNE PORTE LA CAPACITE RESTANTE, plus l'echelle de
        # risque. Celle-ci valait 1.0 en permanence — `set_risk_scale(1.0)` a
        # chaque pas — donc elle n'apportait rien : une constante ne porte pas
        # d'information, elle agit comme un biais.
        #
        # A la place, la part des places encore ouvrables. Le modele ne peut
        # apprendre une contrainte qu'il ne voit pas : sans cette colonne il
        # proposerait des entrees que le solde refuse, et n'aurait aucun moyen
        # de distinguer un refus d'une occasion manquee.
        # LA PART DE CAPACITE ENCORE LIBRE, et non un compte divise par un
        # plafond : le plafond n'existe plus, et diviser par une taille de
        # tableau qui double ferait changer l'echelle de l'observation en
        # cours d'episode — le reseau verrait la meme situation sous deux
        # valeurs differentes.
        #
        # `libres / (libres + tenues)` vaut 1 quand rien n'est ouvert et tend
        # vers 0 quand le solde sature. Elle ne depend d'aucune constante.
        if getattr(self.cfg, "marge_realiste", False):
            _libres = self.places_ouvrables(max(current_price, 1e-8))
            _tenues = self.n_positions
            risk_feature = _libres / max(_libres + _tenues, 1)
        else:
            risk_feature = float(self.last_risk_scale)

        # UNE ALLOCATION AU LIEU DE TROIS. `np.repeat` construisait un bloc
        # (lookback, 4), `np.concatenate` un (lookback, 260), puis `.astype`
        # en recopiait un troisieme. Les features sont deja en float32, donc
        # l'affectation dans un tampon neuf ecrit exactement les memes
        # octets. `_get_obs` pesait 35 % du pas au profileur.
        #
        # LE TAMPON EST NEUF A CHAQUE APPEL, et ce n'est pas negociable : la
        # collecte garde l'observation dans `en_attente[k]["state"]` jusqu'a
        # la mise a jour PPO. Un tampon reutilise ferait apprendre le reseau
        # sur l'etat de la DERNIERE barre pour toutes les decisions.
        # LA DISTANCE AU GARDE-FOU DE CREUX. Elle vaut 0 au sommet et 1.0
        # quand l'episode est coupe. Sans elle le modele ne voyait pas la
        # limite qu'on lui demande de respecter : le creux se deduisait a
        # 67 % de la capacite restante, mais le PIC manquait, et deux
        # observations identiques pouvaient correspondre a douze points de
        # creux d'ecart.
        #
        # ELLE SE MESURE PAR RAPPORT A LA RUINE, PLUS PAR RAPPORT A 40 %.
        #
        # Elle valait `creux / 0.40`, bornee a 1 : elle saturait donc des
        # 40 % de creux, et au-dela le modele voyait la meme valeur pour un
        # compte a -40 % et un compte a -90 %. Elle decrivait une convention
        # de laboratoire, pas la realite du compte.
        #
        # Elle vaut maintenant le creux lui-meme : 0 au sommet, 1 quand il ne
        # reste plus rien. C'est « combien de creux il reste avant de ne plus
        # pouvoir trader », et c'est la meme grandeur que celle qui tue
        # desormais l'episode.
        #
        # L'EQUITE INCLUT TOUTES LES POSITIONS OUVERTES — `_latent_at_bid`
        # les somme. Le compte est juge sur ce qu'il vaudrait s'il fermait
        # tout maintenant, ce qui est exactement ce que le courtier regarde.
        _eq = self.capital + self._latent_at_bid(current_price)
        _pic = max(self.peak_capital, _eq)
        _creux = (_pic - _eq) / (_pic + 1e-8)
        distance_creux = float(min(max(_creux, 0.0), 1.0))

        # LE BUDGET COURANT, rapporte au plus large des paliers. La
        # quatrieme colonne porte deja la capacite restante, mais deux budgets
        # differents peuvent donner la meme capacite : sans cette colonne le
        # modele ne saurait pas lequel il a choisi, ni donc quoi corriger.
        # DEJA UNE PART DANS [0, 1] depuis le 2026-09-21 : la division par
        # le haut de l'echelle n'a plus d'objet. Voir `BUDGETS_PART`.
        # LA LARGEUR VIENT DE `N_POS_FEATURES`, PAS D'UN NOMBRE EN DUR.
        #
        # Elle etait ecrite `n_base + 6`. La sixieme colonne portait le
        # budget, retire le 2026-09-21 ; un nombre en dur aurait laisse une
        # colonne de zeros que le reseau aurait continue de lire, et que
        # `demande_cloture` aurait continue de remettre a zero. Le meme
        # piege que `action_counts_env = np.zeros(3)`, qui a survecu au
        # passage a quatre actions jusqu'au premier CLOTURER emis.
        n_base = base.shape[1]
        obs = np.empty((self.lookback, n_base + N_POS_FEATURES),
                       dtype=np.float32)
        obs[:, :n_base] = base
        obs[:, n_base] = pos_feature
        obs[:, n_base + 1] = unrealized_atr
        obs[:, n_base + 2] = bars_held_norm
        obs[:, n_base + 3] = risk_feature
        obs[:, n_base + 4] = distance_creux
        return obs

    def _apply_micro(self, price: float, side: int, is_entry: bool = True) -> float:
        """Convertit une bougie BID en prix BUY/SELL exécutable."""
        price = execution_quote(price, side, self.current_trade_spread_bps)
        if is_entry and self.cfg.entry_slippage_bps > 0:
            extra = np.random.uniform(0.0, self.cfg.entry_slippage_bps) / 10_000.0
            price *= (1 + side * extra)
        return price

    def _sample_trade_spread_bps(self) -> float:
        """Le spread de ce trade. LU sur la barre, plus tire d'une loi.

        CE QU'IL Y AVAIT AVANT, ET POURQUOI IL EST PARTI. Une loi bimodale
        « calibree sur les logs MT5 Vantage BTCUSD » : 70 % du temps la
        valeur centrale a +-2 %, 30 % du temps un tirage uniforme entre
        1.2 et 2.83 fois cette valeur. Deux problemes.

        ELLE ETAIT CALIBREE SUR LE BTC, ET APPLIQUEE A L'OR. Le facteur
        d'elargissement et la probabilite venaient d'un autre instrument.

        ET ELLE ETAIT BEAUCOUP TROP LARGE. Mesure du 2026-09-21 sur les
        900 000 barres M1 de l'or, spread REELLEMENT cote par le
        courtier :

            median  0.563 bps      p90  0.770      p99  0.860

        La loi inventee donnait 0.887 bps en esperance et une queue
        jusqu'a 1.92. Le simulateur facturait donc environ 1.6 fois le
        spread reel, et les etiquettes avec lui. Un modele entraine sous
        une friction majoree apprend a s'abstenir plus qu'il ne faut —
        c'est la meme faute que `GLISSEMENT_BPS`, corrigee le matin meme,
        et elle etait encore la, dans l'autre moitie du calcul.

        LA VALEUR EST CELLE DE LA BARRE PRECEDENTE, pas de la barre
        courante : c'est ce qu'un ordre passe a l'ouverture rencontre, et
        c'est causal.

        LE REPLI EXISTE POUR LES CACHES ANTERIEURS. Sans colonne
        `spread_bar`, on retombe sur l'ancien tirage — sinon un cache
        d'avant le 2026-09-21 rendrait un spread nul, donc une friction
        nulle, et le run paraitrait rentable pour une raison purement
        comptable.
        """
        i = min(max(self.idx - 1, 0), len(self.data.spread_bar) - 1)
        reel = float(self.data.spread_bar[i])
        if reel > 0.0:
            return reel
        base = float(self.cfg.spread_bps)
        if base <= 0.0:
            return 0.0
        if np.random.rand() < self.cfg.spread_wide_prob:
            return float(np.random.uniform(
                base * 1.2, base * self.cfg.spread_bps_wide_factor))
        return float(np.random.uniform(base * 0.98, base * 1.02))

    def _compute_dynamic_size(self, price: float) -> float:
        """Taille dictée par le RISQUE, jamais par une constante absolue.

        L'ancienne version renvoyait `position_size` (0.06, moitié pendant 40
        epochs) quels que soient le prix et l'ATR. Une constante en unités de
        l'instrument ne veut rien dire d'un symbole à l'autre :

          BTC, SL 3xATR ~= 270 $  -> perte de 8.100 $ = 0.8100 % du capital
          OR,  SL 5xATR ~= 2.50 $ -> perte de 0.075 $ = 0.0075 % du capital

        soit un facteur 108 pour un risque économique censé être IDENTIQUE. Sur
        l'or le modèle jouait donc une position si petite que le terme de PnL
        réalisé de la récompense valait 0.007 : noyé sous le bruit de marquage
        au marché, et incapable de faire déclencher le moindre garde-fou de
        drawdown. `risk_per_trade = 1.2 %` était dans la config depuis le début
        mais n'était lu nulle part.

        On le lit. size = risque_voulu / distance_de_stop : une perte au stop
        coûte exactement `risk_per_trade` du capital, sur n'importe quel
        symbole, à n'importe quel prix, sous n'importe quelle volatilité.
        """
        scale = float(max(self.risk_scale, 0.0))
        if scale <= 0.0:
            scale = 1.0

        # MEME ATR que celui qui posera le stop quelques lignes plus bas : s'ils
        # divergeaient, la perte au stop ne vaudrait plus le risque visé et
        # toute l'échelle de récompense repartirait à la dérive.
        atr_raw = float(self.data.atr14[self.idx - 1]) if self.idx - 1 >= 0 else 0.0
        atr = max(atr_raw, ATR_PLANCHER_FRAC * price, 1e-8)
        sl_dist = self.cfg.atr_sl_mult * atr
        if not np.isfinite(sl_dist) or sl_dist <= 0.0:
            return 0.0

        # LE RISQUE VISE EST CELUI DU PORTEFEUILLE, pas celui d'une ligne.
        #
        # K positions de meme sens subissent le MEME mouvement : leur variance
        # commune vaut K(1 + (K-1)rho) fois celle d'une seule. Mesure du
        # 2026-09-16 : deux trades de meme sens ouverts a une heure d'ecart
        # correlent a 0.72, et encore a 0.31 apres 24 h — soit rho de l'ordre
        # de 0.45 sur la duree de vie typique d'une position.
        #
        # Garder `risk_per_trade` comme risque TOTAL impose donc de diviser
        # chaque ligne par la racine de ce facteur. Sans cela, passer K de 1 a
        # 8 multiplierait le risque reel par 5.8 en silence, et le run suivant
        # paraitrait meilleur ou pire pour une raison qui n'a rien a voir avec
        # l'apprentissage.
        #
        # A K=1 le facteur vaut exactement 1 et l'expression est celle d'avant.
        k = int(getattr(self, "_K", 1))
        if k > 1:
            rho = float(getattr(self.cfg, "correlation_positions", 0.45))
            scale = scale / math.sqrt(k * (1.0 + (k - 1) * rho))

        size = (self.capital * self.cfg.risk_per_trade * scale) / sl_dist

        # Plafond de NOTIONNEL, exprime directement en multiples du capital.
        #
        # Il valait `capital x max_position_frac x leverage`. Faire dependre un
        # garde-fou du levier est un piege : en portant le levier de 6 a 100
        # pour rendre la taille par le risque possible, le plafond est passe de
        # 2.1x a 35x le capital SANS QUE RIEN NE LE SIGNALE. Un plafond ne doit
        # dependre que de lui-meme.
        #
        # 30x est haut, et c'est assume : un stop a 5xATR sur l'or vaut 0.073 %
        # du prix, donc risquer 1.2 % demande ~16x de notionnel. C'est le prix
        # d'un stop serre, et le stop borne la perte. Le plafond ne sert qu'a
        # bloquer le cas degenere ou l'ATR s'effondre et ou la taille partirait
        # a l'infini.
        notion_max = self.capital * self.cfg.max_notional_mult
        size = min(size, notion_max / max(price, 1e-8))

        return float(max(size, 0.0))

    @property
    def capital(self) -> float:
        """Vue sur le solde du COMPTE, partage entre instruments."""
        return self.portefeuille.capital

    @capital.setter
    def capital(self, v) -> None:
        self.portefeuille.capital = float(v)

    def _latent_at_bid(self, bid):
        """Somme des latents de TOUS les emplacements ouverts.

        EN UNE OPERATION VECTORIELLE, parce que c'est le chemin chaud : la
        premiere version bouclait en Python sur les emplacements, et comme
        `peut_entrer()` et `_get_obs()` l'appellent chacun a chaque barre et
        pour chaque environnement, la collecte s'est effondree des K=8.
        `execution_quote` n'est qu'une multiplication — un long sort au BID,
        un short paie l'ASK — donc elle se met en tableau sans rien changer
        au resultat.

        A K=1 c'est le calcul d'avant, terme pour terme.
        """
        # MEMORISE SUR (bid, version). Le meme prix revient plusieurs fois
        # par barre — `peut_entrer`, `_get_obs`, le marquage d'entree et celui
        # de sortie lisent tous `close[idx-1]` ou `close[idx]` — et entre deux
        # de ces appels rien ne peut avoir change si les emplacements n'ont
        # pas bouge. Le calcul rendu est celui d'avant, terme pour terme.
        if self.n_positions == 0:
            return 0.0
        c = self._c_latent
        if c is not None and c[0] == bid:
            return c[1]
        a = self._actifs
        sens = self._p_sens[a]
        # execution_quote(bid, -sens, spread) : le cote vendu paie le spread.
        q = float(bid) * np.where(sens == -1,
                                  1.0 + np.maximum(self._p_spread[a], 0.0) / 1e4,
                                  1.0)
        v = float(((sens * (q - self._p_entree[a])
                    - self.cfg.fee_rate * q) * self._p_taille[a]).sum())
        self._c_latent = (bid, v)
        return v

    def _close_position(self, exit_price, hit_sl=False, hit_tp=False,
                        hit_temps=False, terminal_reason=None, slot=None):
        """Ferme UN emplacement. `slot=None` prend le plus ancien ouvert.

        Le defaut sur le plus ancien garde le comportement d'avant a K=1, ou
        il n'y en a qu'un — et donne une regle explicite plutot qu'un ordre
        accidentel quand il y en a plusieurs.
        """
        if slot is None:
            a = np.flatnonzero(self._p_sens != 0)
            if not len(a):
                return 0.0
            slot = int(a[np.argmin(self._p_idx[a])])
        sens = int(self._p_sens[slot])
        if sens == 0:
            return 0.0
        taille = float(self._p_taille[slot])
        pnl = sens * (exit_price - self._p_entree[slot]) * taille
        fee = self.cfg.fee_rate * exit_price * taille
        realized = pnl - fee

        self.capital += realized
        self.last_realized_pnl = realized
        self.trades_pnl.append(realized)
        self.trades_side.append(sens)
        self.trades_meta.append({
            "entry_idx": int(self._p_idx[slot]),
            "exit_idx": int(self.idx),
            "side": sens,
            "entry_price": float(self._p_entree[slot]),
            "exit_price": float(exit_price),
            "pnl": float(realized),
            # EN UNITES DE RISQUE DE CE TRADE : c'est ce que le critere de
            # sauvegarde juge depuis le passage au PPO complet.
            "r": float(realized / max(float(self._p_risque[slot]), 1e-8)),
            "hit_sl": bool(hit_sl),
            "hit_tp": bool(hit_tp),
            "hit_temps": bool(hit_temps),
            "terminal_reason": terminal_reason,
            "hold_bars": int(self.idx - self._p_idx[slot]),
        })

        # Le terme de recompense du trade realise s'exprime en unites du
        # risque DE CE TRADE, pas d'un risque global : sans cela un trade
        # ouvert petit et un trade ouvert grand n'auraient pas la meme echelle.
        self._r_slots[slot] += float(np.clip(
            realized / max(float(self._p_risque[slot]), 1e-8), -3.0, 3.0))
        # Le montant brut sert au terme de marquage : l'argent realise a quitte
        # le latent de l'emplacement pour rejoindre le capital, donc il doit
        # etre recompte dans la variation d'equity DE CET EMPLACEMENT.
        self._realise_slots[slot] += realized
        self._slots_fermes.append(int(slot))

        self._notionnel = max(0.0, self._notionnel
                              - float(self._p_entree[slot]) * taille)
        self._p_sens[slot] = 0
        self._p_taille[slot] = 0.0
        self._p_entree[slot] = 0.0
        self._p_sl[slot] = 0.0
        self._p_tp[slot] = 0.0
        self._p_idx[slot] = -1
        self._p_atr[slot] = 0.0
        self._p_cout[slot] = 0.0
        self._p_risque[slot] = 0.0
        self._p_be[slot] = False
        self._p_trail[slot] = False
        self._touche()
        if self.n_positions == 0:
            self.risk_scale = 1.0
            self.last_risk_scale = 1.0
        return realized

    def step(self, action: int):
        # DEUX PRIX DISTINCTS, et les confondre etait une erreur de timing.
        #
        # `_get_obs` ne montre au modele que les features jusqu'a l'indice
        # idx-1 : il decide donc a la CHARNIERE entre la bougie idx-1 close et
        # la bougie idx qui s'ouvre. Son ordre au marche part a cet instant, et
        # se remplit a l'OUVERTURE de la bougie idx.
        #
        # L'ancienne version executait au CLOSE de la bougie idx. Le simulateur
        # accordait donc au modele une minute entiere de mouvement qu'il n'avait
        # pas vue — ce n'est pas de l'anticipation (il ne la voit toujours pas),
        # mais une execution qui ne correspond a aucun ordre reel. Personne ne
        # peut decider a l'ouverture et obtenir le close.
        prix_execution = self.data.open[self.idx]     # entree : a l'ouverture
        price = self.data.close[self.idx]             # marquage : a la cloture
        high_bar = self.data.high[self.idx]
        low_bar = self.data.low[self.idx]

        # Bruit de tick — nul par defaut, cf. cfg.tick_noise_bps. Ne sert qu'aux
        # tests de robustesse : les bougies MT5 portent deja les extremes reels.
        if self.cfg.tick_noise_bps > 0:
            noise_h = np.random.uniform(0.0, self.cfg.tick_noise_bps) / 10_000.0
            noise_l = np.random.uniform(0.0, self.cfg.tick_noise_bps) / 10_000.0
            high_bar = high_bar * (1.0 + noise_h)
            low_bar  = low_bar  * (1.0 - noise_l)

        old_pos = self.position

        # Même marquage au close exécutable aux deux bornes; les mèches
        # bruitées servent uniquement aux triggers, jamais au reward latent.
        bid_prec = self.data.close[self.idx - 1] if self.idx > 0 else price
        prev_equity = self.capital + self._latent_at_bid(bid_prec)

        # LATENT DE DEPART PAR EMPLACEMENT. Il sert a repartir la recompense :
        # chaque decision ne doit recevoir que ce que SA position a produit,
        # sinon un trade gagnant crediterait le trade perdant ouvert a cote et
        # l'apprentissage porterait sur une moyenne que personne ne joue.
        # VECTORISE : c'est une fois par barre et par environnement, donc une
        # boucle Python sur K y coute directement K fois plus cher. La
        # premiere version en faisait une, et la collecte s'effondrait des que
        # le plafond montait.
        self._latent_prec = self._latents_par_slot(bid_prec)
        self._r_slots[:] = 0.0
        self._realise_slots[:] = 0.0
        self._slots_fermes = []
        self._slot_ouvert = -1

        realized_trade = 0.0
        # COMBIEN DE POSITIONS SE FERMENT SUR CETTE BARRE. Le terme en R de la
        # recompense en avait besoin : voir plus bas, il en prend la MOYENNE
        # et non la somme.
        n_fermes = 0
        hit_sl = hit_tp = hit_temps = False

        # LA CLOTURE EST UNE ACTION, DEPUIS LE 2026-09-21.
        #
        # Ce drapeau existait, cable sur `False`, avec le commentaire « plus
        # de mode close : fermeture uniquement par SL/TP/break-even/
        # trailing ». Toute la plomberie qui en depend — `if not
        # manual_close and ...` devant le stop, le trailing et l'objectif —
        # etait donc du code mort qui attendait qu'on la rallume.
        #
        # On la rallume. L'action 3 ferme la position au marche, et court-
        # circuite du meme geste les sorties geometriques : quand le modele
        # decide, la regle ne decide plus.
        manual_close = (action == 3) and bool(self._p_sens.any())

        # --------- CLOTURE SUR DECISION DU MODELE ---------
        # Elle passe AVANT l'ouverture : une meme barre peut donc fermer une
        # position et en ouvrir une autre, ce qui est exactement ce qu'un
        # scalpeur fait quand le signal s'inverse.
        #
        # `_close_position` est la MEME fonction que celle qu'appelaient le
        # stop et l'objectif. La sortie change de DECIDEUR, pas de mecanique
        # — donc pas de seconde ecriture de la comptabilite de sortie, qui
        # est l'endroit ou deux chemins finissent toujours par diverger.
        #
        # LA CLOTURE PAIE LE PRIX EXECUTABLE — bug trouve le 2026-09-25.
        #
        # Elle passait `prix_execution`, l'ouverture BID brute. Correct pour
        # un long, qui revend au bid ; FAUX pour un short, qui doit racheter
        # a l'ASK. Le short avait vendu au bid et rachetait au bid : il ne
        # payait JAMAIS le spread, seulement le glissement d'entree. Mesure
        # sur les trades d'une barre, ou le marche ne bouge pas en moyenne :
        #
        #                  long         short
        #   validation   -2.66 bps    -0.81 bps
        #   entrainement -2.11 bps    -0.49 bps
        #
        # Un spread d'ecart. C'est ce qui faisait les shorts « meilleurs »
        # dans tous les runs depuis leur retour : 41-50 % de gagnants contre
        # 18-30 % pour les longs. Les autres sorties — stop, objectif, fin
        # d'episode — passaient deja par `execution_quote` ; seule celle que
        # le modele decide l'oubliait.
        #
        # LE SPREAD EST CELUI DU TRADE, `_p_spread` : c'est lui qui valorise
        # deja la position latente (`_latent_slot`). Sans cela, la recompense
        # voyait le spread pendant la tenue et le rendait a la cloture —
        # fermer un short en devenait paye.
        if manual_close:
            _a = np.flatnonzero(self._p_sens != 0)
            _j = int(_a[np.argmin(self._p_idx[_a])])
            _q = execution_quote(prix_execution, -int(self._p_sens[_j]),
                                 float(self._p_spread[_j]))
            self._close_position(_q, terminal_reason="modele", slot=_j)
            n_fermes += 1

        # --------- OUVERTURE DIRECTE (pas de confirmation dans l'env) ---------
        # La confirmation signal→pause→re-signal est gérée dans kairos_live.py uniquement.
        # En training, le reward shaping pénalise déjà les mauvaises entrées.
        # UN EMPLACEMENT LIBRE SUFFIT, la ou l'ancienne condition exigeait
        # d'etre entierement plat. A K=1 les deux coincident exactement.
        # Plus de place dans le tableau alors que le solde en permet ? On
        # agrandit. Le solde reste seul juge.
        if (not manual_close and action in (0, 1)
                and self.n_positions >= self._K
                and self.places_ouvrables(prix_execution) > 0):
            self._agrandir()
        # LA RECHERCHE D'UN EMPLACEMENT LIBRE NE SE FAIT QUE SI ON OUVRE.
        #
        # Elle etait faite a CHAQUE barre : une comparaison sur tout le
        # tableau d'emplacements, puis `flatnonzero` qui alloue la liste
        # complete des libres — pour n'en lire que le premier, et seulement
        # dans les 0.5 % de barres ou l'action est un achat. A 256
        # emplacements et 870 000 barres par epoch, c'est 223 millions de
        # comparaisons dont 99.5 % sont jetees.
        #
        # `argmax` sur le masque booleen rend le PREMIER vrai, donc
        # exactement ce que `flatnonzero(...)[0]` rendait. La branche est
        # gardee par `.any()`, qui s'arrete au premier libre trouve.
        _ouvre = (not manual_close) and action in (0, 1)
        _masque_libre = (self._p_sens == 0) if _ouvre else None
        if _ouvre and _masque_libre.any():
            side = 1 if action == 0 else -1
            size = self._taille_quantifiee(self._compute_dynamic_size(prix_execution))
            # LE SOLDE PEUT REFUSER L'ORDRE. C'est un refus, pas une erreur :
            # en live l'ordre part et le serveur le rejette faute de marge.
            if size > 0.0 and self.places_ouvrables(prix_execution) <= 0:
                size = 0.0
            if size > 0.0:
                j = int(np.argmax(_masque_libre))
                self._p_taille[j] = size
                self._p_sens[j] = side
                # Échantillonne le spread pour ce trade (variabilité réaliste)
                spread = self._sample_trade_spread_bps()
                self._p_spread[j] = spread
                self.current_trade_spread_bps = spread
                exec_price = self._apply_micro(prix_execution, side, is_entry=True)
                self._p_entree[j] = exec_price
                self._p_idx[j] = self.idx

                atr_raw = float(self.data.atr14[self.idx - 1]) if self.idx - 1 >= 0 else 0.0
                # MEME plancher que saint_core.effective_atr — le dupliquer ici
                # avec une autre valeur ferait diverger l'entrainement du live.
                fallback = ATR_PLANCHER_FRAC * exec_price
                entry_atr = max(atr_raw, fallback, 1e-8)
                self._p_atr[j] = entry_atr
                # LE POINT MORT EST LA PERTE LATENTE D'UNE POSITION DONT LE
                # MARCHE N'A PAS BOUGE : ce que coute l'aller-retour tel que
                # `_latent_slot` le valorise. Il valait `|prix paye -
                # ouverture|`, juste pour un long — qui paie le spread a
                # l'entree — mais pas pour un short, qui le paie a la SORTIE :
                # son point mort ne comptait que le glissement (0.010 ATR
                # contre 0.103 pour le long), alors que son latent inclut deja
                # le rachat a l'ask. Corrige avec la cloture, 2026-09-25.
                self._p_cout[j] = max(0.0, -side * (
                    execution_quote(prix_execution, -side, spread)
                    - exec_price)) / entry_atr

                sl_dist = self.cfg.atr_sl_mult * entry_atr
                tp_dist = self.cfg.atr_tp_mult * entry_atr * self.cfg.tp_shrink

                # Montant réellement en jeu si le stop est touché. C'est l'unité
                # dans laquelle la récompense du trade sera exprimée : un stop
                # vaut -1, une cible à R:R 2 vaut +2, partout et toujours.
                self._p_risque[j] = float(sl_dist * size)
                self.risk_amount = float(sl_dist * size)

                # UN tp_price NUL VEUT DIRE "PAS D'OBJECTIF". Le test des
                # barrieres plus bas est deja garde par `tp > 0`, donc il
                # suffit de ne pas le poser — aucune branche a ajouter, et le
                # chemin du stop reste rigoureusement le meme.
                if not getattr(self.cfg, "use_tp", True):
                    tp_dist = 0.0
                # UN sl_price NUL VEUT DIRE "PAS DE STOP", exactement
                # comme un tp nul veut dire "pas d'objectif" : le test des
                # barrieres est deja garde par `sl > 0`, donc il suffit de
                # ne pas le poser. Aucune branche a ajouter, et le chemin
                # de sortie reste rigoureusement le meme.
                #
                # `sl_dist` CONTINUE DE SERVIR, et c'est important : il
                # dimensionne la position — `_compute_dynamic_size` divise
                # par lui — et il donne son denominateur a `risk_amount`,
                # donc a l'echelle de la recompense. Ne pas POSER le stop
                # n'est pas la meme chose que ne pas savoir ce qu'on risque.
                #
                # CE QUE CELA RETIRE, ET IL FAUT LE DIRE : plus rien ne
                # borne la perte. Une position sur cent traverse 80 bps de
                # creux latent avant de se resoudre (`cibles_m1`), et c'est
                # desormais `tete_cloture` qui doit s'en charger. Si elle
                # se trompe, il n'y a personne derriere.
                _pose_sl = bool(getattr(self.cfg, "use_sl", True))
                if side == 1:
                    self._p_sl[j] = (max(1e-8, exec_price - sl_dist)
                                     if _pose_sl else 0.0)
                    self._p_tp[j] = (max(1e-8, exec_price + tp_dist)
                                     if tp_dist > 0 else 0.0)
                else:
                    self._p_sl[j] = (max(1e-8, exec_price + sl_dist)
                                     if _pose_sl else 0.0)
                    self._p_tp[j] = (max(1e-8, exec_price - tp_dist)
                                     if tp_dist > 0 else 0.0)

                self._p_be[j] = False
                self._p_trail[j] = False
                self._notionnel += exec_price * size
                self._slot_ouvert = j
                self._touche()

        # --------- BREAK-EVEN + TRAILING STOP ---------
        # Désactivé par défaut (cfg.use_be_trail=False) pour aligner sur le
        # backtest "no_be_trail" et sur le MQL5 SaintV2_WF3 (sans BE/trail).
        # Permet d'isoler la qualité du signal d'entrée pur (SL/TP fixes uniquement).
        # CHAQUE EMPLACEMENT SUIT SON PROPRE STOP. Deux positions ouvertes a
        # des prix differents n'ont ni le meme point mort ni le meme
        # declenchement ; un stop commun melangerait les deux et suivrait le
        # mouvement d'une position que l'autre n'a pas vecu.
        # EN UNE PASSE VECTORIELLE. La version en boucle appelait
        # `execution_quote` une fois par position ouverte et par barre —
        # releve au profileur : 98 appels par barre a 35 positions, et c'est
        # le premier poste de cout de l'environnement. La fonction n'est
        # qu'une multiplication, donc tout se met en tableau sans changer un
        # seul resultat : le temoin a K=1 le verifie au centime.
        #
        # L'ordre est conserve : break-even d'abord, puis trailing, qui lit
        # donc le stop DEJA remonte par le break-even.
        if self.cfg.use_be_trail and not manual_close and self._p_sens.any():
            sens = self._p_sens
            vif = ((sens != 0) & (self._p_taille > 0) & (self._p_atr > 1e-8)
                   & (self.idx > self._p_idx))
            if vif.any():
                # Le SL de cette bougie ne peut utiliser que les bougies déjà closes.
                haut = float(self.data.high[self.idx - 1])
                bas = float(self.data.low[self.idx - 1])
                fav_bid = np.where(sens == 1, haut, bas)
                # execution_quote(fav_bid, -sens, spread) : le cote vendu paie.
                fav_price = fav_bid * np.where(
                    sens == -1, 1.0 + np.maximum(self._p_spread, 0.0) / 1e4, 1.0)
                fav_move = sens * (fav_price - self._p_entree)

                # Break-even : déplace SL à l'entrée quand gain >= atr_be_mult × ATR
                m_be = vif & (~self._p_be) & (
                    fav_move >= self.cfg.atr_be_mult * self._p_atr)
                if m_be.any():
                    self._p_sl = np.where(
                        m_be & (sens == 1),
                        np.maximum(self._p_sl, self._p_entree), self._p_sl)
                    self._p_sl = np.where(
                        m_be & (sens == -1),
                        np.minimum(self._p_sl, self._p_entree), self._p_sl)
                    self._p_be |= m_be

                # Trailing stop : suit le prix à atr_trail_dist × ATR quand gain >= atr_trail_mult × ATR
                m_tr = vif & (fav_move >= self.cfg.atr_trail_mult * self._p_atr)
                if m_tr.any():
                    trail_sl = fav_price - sens * self.cfg.atr_trail_dist * self._p_atr
                    self._p_sl = np.where(
                        m_tr & (sens == 1),
                        np.maximum(self._p_sl, trail_sl), self._p_sl)
                    self._p_sl = np.where(
                        m_tr & (sens == -1),
                        np.minimum(self._p_sl, trail_sl), self._p_sl)
                    self._p_trail |= m_tr

        # --------- SL/TP AUTO ---------
        # LONG ferme au BID, SHORT à l'ASK. SL/TP sont déjà des prix
        # exécutables: aucun spread supplémentaire sur leur prix de sortie.
        # La bougie d'ENTREE est desormais incluse dans le test des barrieres.
        #
        # Elle en etait exemptee (`self.idx > self.entry_idx`) parce que l'entree
        # se faisait au CLOSE : tester la bougie contre un stop pose a sa propre
        # cloture n'aurait eu aucun sens. Maintenant que l'entree est a
        # l'OUVERTURE, la position vit pendant toute la bougie et son stop doit
        # pouvoir etre touche — c'est le cas en reel.
        #
        # Cela supprime au passage un angle mort : une position a fort notionnel
        # restait sans protection pendant sa premiere minute.
        # LES EMPLACEMENTS SONT TESTES DANS L'ORDRE, du plus ancien au plus
        # recent. L'ordre compte parce que chaque fermeture tire un slippage :
        # le fixer rend la trajectoire reproductible, et a K=1 il reproduit
        # exactement la sequence d'avant.
        if not manual_close and self._p_sens.any():
            open_bid = getattr(self.data, 'open', self.data.close)[self.idx]
            # LA DETECTION EST VECTORIELLE, la fermeture ne boucle que sur
            # les emplacements qui TOUCHENT — une poignee par barre, la ou
            # l'ancienne version parcourait toutes les positions ouvertes et
            # appelait `execution_quote` pour chacune.
            sens_a = self._p_sens
            vivant = (sens_a != 0) & (self._p_taille > 0) & (self._p_entree > 0)
            spread_f = 1.0 + np.maximum(self._p_spread, 0.0) / 1e4
            est_long = sens_a == 1
            sl_pose = self._p_sl > 0
            tp_pose = self._p_tp > 0

            touche_sl = vivant & sl_pose & np.where(
                est_long, low_bar <= self._p_sl,
                high_bar * spread_f >= self._p_sl)
            touche_tp = vivant & tp_pose & (~touche_sl) & np.where(
                est_long, high_bar >= self._p_tp,
                low_bar * spread_f <= self._p_tp)
            # LE PLAFOND SE LIT DANS `plafond_detention`, une seule
            # source pour l'environnement et pour l'echantillon qui
            # entraine la tete de cloture. Il valait `max_holding_bars`,
            # fige a 0 : sans stop ni objectif ni trailing, plus AUCUNE
            # sortie automatique n'existait.
            _plaf_det = plafond_detention(self.cfg)
            touche_temps = (vivant & (~touche_sl) & (~touche_tp)
                            & (_plaf_det > 0)
                            & ((self.idx - self._p_idx) >= _plaf_det))

            a_fermer = np.flatnonzero(touche_sl | touche_tp | touche_temps)
            # L'ordre reste celui des entrees : chaque fermeture tire un
            # slippage, donc l'ordre fixe la trajectoire aleatoire.
            a_fermer = a_fermer[np.argsort(self._p_idx[a_fermer], kind="stable")]
            for j in a_fermer:
                j = int(j)
                sens = int(sens_a[j])
                h_sl = bool(touche_sl[j])
                h_tp = bool(touche_tp[j])
                h_temps = bool(touche_temps[j])
                open_quote = open_bid * (float(spread_f[j]) if sens == -1 else 1.0)
                if h_sl:
                    exit_price = (min(float(self._p_sl[j]), open_quote) if sens == 1
                                  else max(float(self._p_sl[j]), open_quote))
                elif h_tp:
                    exit_price = float(self._p_tp[j])
                else:
                    exit_price = self._apply_micro(price, -sens, is_entry=False)

                if True:
                    # SL/sortie au marché: slippage adverse. TP: niveau cible,
                    # sans amélioration favorable systématique inventée.
                    slip_max = self.cfg.slippage_bps / 10_000.0
                    if slip_max > 0 and not h_tp:
                        slip_amount = exit_price * np.random.uniform(0.0, slip_max)
                        # Une sortie au marche (temps ecoule) traverse le
                        # spread : elle est DEFAVORABLE comme un SL, jamais
                        # favorable comme un TP ou le momentum joue pour nous.
                        contre = h_sl or h_temps
                        if sens == 1:   # LONG
                            exit_price += (-slip_amount if contre else slip_amount)
                        else:           # SHORT
                            exit_price += (slip_amount if contre else -slip_amount)

                    realized_trade += self._close_position(
                        exit_price, h_sl, h_tp, h_temps, slot=j)
                    n_fermes += 1
                    hit_sl = hit_sl or h_sl
                    hit_tp = hit_tp or h_tp
                    hit_temps = hit_temps or h_temps

        # Les fenêtres sont des épisodes FINIS: toute position restante est
        # liquidée et comptée avant de calculer la dernière récompense.
        marked_equity = self.capital + self._latent_at_bid(price)
        marked_peak = max(self.peak_capital, marked_equity)
        marked_dd = (marked_peak - marked_equity) / (marked_peak + 1e-8)
        done_reason = None
        # L'APPEL DE MARGE EXISTE, et il ne previent pas. Quand l'equity
        # tombe sous une fraction de la marge immobilisee, le courtier
        # liquide — il ne demande pas l'avis du modele. Ne pas le simuler
        # laisserait l'agent apprendre qu'il peut porter n'importe quelle
        # exposition tant que le prix finit par revenir.
        _mu = self.marge_utilisee
        _niveau = (marked_equity / _mu) if _mu > 1e-12 else float("inf")
        if self.idx + 1 >= self.end_idx:
            done_reason = "episode_end"
        elif _niveau < float(getattr(self.cfg, "niveau_marge_liquidation", 0.5)):
            done_reason = "appel_de_marge"
        elif (getattr(self.cfg, "marge_realiste", False)
              and not self._p_sens.any()
              and not self.peut_financer_un_lot(price)):
            # LA RUINE. Aucune position ouverte, et le solde n'en permet plus
            # aucune : le compte ne peut plus rien faire. Laisser tourner
            # l'episode enseignerait qu'on survit a la ruine en attendant, ce
            # qui est faux — un compte qui ne peut plus prendre le lot
            # minimum est termine.
            done_reason = "solde_insuffisant"
        elif marked_equity <= 0.0:
            # LA SEULE FIN QUI EXISTE VRAIMENT : le compte n'a plus rien.
            #
            # Les deux fins qui etaient ici — creux > 40 %, capital sous 20 %
            # du depart — n'existent pas chez un courtier. C'etaient des
            # conventions de laboratoire, et elles coutaient cher :
            #
            #   - EN MESURE. Un episode tue au quart de sa tranche laisse les
            #     trois quarts non joues. Mesure du 2026-09-19, 29 episodes a
            #     la selectivite de validation : 26 % des barres seulement
            #     etaient jouees, et UN TIERS des episodes tues auraient fini
            #     AU-DESSUS du capital de depart. On refusait un checkpoint
            #     sur une observation tronquee, sans savoir ce qu'il valait.
            #
            #   - EN APPRENTISSAGE. Le modele ne voyait jamais l'au-dela du
            #     garde-fou, donc il ne pouvait pas apprendre a se remettre
            #     d'un creux profond : ces trajectoires n'existaient pas dans
            #     son vecu.
            #
            # CE QUI RESTE EST CE QUE LE COURTIER FAIT REELLEMENT, et qui
            # etait deja modelise juste au-dessus : l'appel de marge quand le
            # niveau passe sous 50 %, et l'impossibilite de financer le lot
            # minimum. Cette branche-ci est le filet : une bougie qui saute
            # peut emporter l'equite sous zero avant que le niveau de marge
            # ne soit relu.
            done_reason = "ruine"
        if done_reason is not None and self._p_sens.any():
            restants = np.flatnonzero(self._p_sens != 0)
            restants = restants[np.argsort(self._p_idx[restants], kind="stable")]
            for j in restants:
                j = int(j)
                sens = int(self._p_sens[j])
                exit_price = self._apply_micro(price, -sens, is_entry=False)
                slip = (np.random.uniform(0.0, self.cfg.slippage_bps) / 10000.0
                        if self.cfg.slippage_bps > 0 else 0.0)
                exit_price *= 1.0 - sens * slip
                realized_trade += self._close_position(
                    exit_price, terminal_reason=done_reason, slot=j)
                n_fermes += 1

        # ==================================================================
        # REWARD SHAPING Ω — LONG + SHORT AVEC BONUS MOMENTUM CONFIRMÉ
        # ==================================================================
        equity = self.capital + self._latent_at_bid(price)
        equity_clamped = max(equity, 1e-8)
        prev_equity_clamped = max(prev_equity, 1e-8)
        log_ret = math.log(equity_clamped / prev_equity_clamped)

        # BORNE PAR BOUGIE. Le terme de marquage au marche n'est qu'un shaping :
        # il densifie le signal entre l'entree et la sortie. Une seule bougie ne
        # doit jamais pouvoir peser plus que le resultat du trade lui-meme.
        #
        # Sans borne, il le peut : l'equity est ecretee a 1e-8 quand une position
        # a fort notionnel est balayee, et log(1e-8 / 950) = -25.3 produit une
        # recompense de -253. Mesure a l'epoch 1 : |R| 338, advStd 28168, et
        # surtout quasi0 100 % — les avantages etant normalises par leur
        # ecart-type, une poignee de valeurs geantes ecrase TOUS les autres sous
        # 0.05 et l'acteur ne recoit plus aucun gradient (g[actor 1.5e-04]).
        log_ret = comprime_log(log_ret)

        # ==================================================================
        # REWARD ALIGNÉ SUR LA RENTABILITÉ RÉELLE
        #
        # L'ancien shaping additionnait quatre termes qui récompensaient le fait
        # d'ÊTRE EN POSITION indépendamment du résultat :
        #   hit_tp +1.0 / hit_sl -0.5   → un gain valait 2× une perte
        #   +0.4 si trade gagnant        → bonus sans contrepartie
        #   -move × 2.5 si flat          → punissait le fait d'attendre
        #   +|unrealized| × 2.8          → gain latent payé, perte latente non
        # Avec WR 31% et avgW/avgL ≈ 1.0 (mesurés), l'espérance en reward valait
        # 0.31 × 1.4 - 0.69 × 0.5 = +0.089 : le modèle était donc RÉCOMPENSÉ
        # pour une stratégie perdante. Il a appris exactement ça — en position
        # 97% du temps, PnL de validation plat sur 75 epochs.
        #
        # On ne garde que ce qui correspond à de l'argent réel, symétriquement.
        # ==================================================================
        reward = log_ret * 10.0

        # Shaping sur trade réalisé : proportionnel au PnL et SYMÉTRIQUE, pour
        # densifier le signal sans réintroduire de biais directionnel.
        # Échelle : ±1% du capital initial sature le terme.
        # =================================================================
        # LE TERME EN R EST UNE MOYENNE, PLUS UNE SOMME — et c'est ce qui le
        # rend neutre au levier.
        #
        # CE QUE LA SOMME COUTAIT, mesure du 2026-09-20 (`mesure_objectif.py`,
        # 12 episodes de 8 000 barres, entrees neutres identiques d'un palier
        # a l'autre) :
        #
        #   budget   somme des R   log-richesse   creux max
        #     3 %            5.7         +0.133      11.8 %
        #     6 %           10.1         +0.206      21.2 %
        #    15 %            7.3         +0.444      45.3 %
        #    40 %          113.9         +0.472      76.5 %
        #
        # 113.9 CONTRE 5.7 : l'objectif payait onze a vingt fois plus pour le
        # levier maximum. Ce n'est pas le modele qui se trompait en choisissant
        # 40 % — c'est la recompense qui le lui achetait. Quatre positions a
        # +1 R rapportaient +4, sans aucune concavite pour freiner.
        #
        # POURQUOI PAS SUPPRIMER LE TERME. Il densifie le signal : un trade
        # ferme donne un +/-1 a +/-3 net, la ou `log_ret` sur une barre vaut
        # une fraction. Le retirer rendrait l'apprentissage beaucoup plus lent.
        # Mais son unite — le R — est DELIBEREMENT independante de la taille,
        # et c'est precisement ce qui le rendait lineaire en nombre de
        # positions. La moyenne garde l'unite et la densite, et retire la
        # prime au nombre.
        #
        # LE PORTEFEUILLE RESTE COMPTE, mais par `log_ret`, qui est
        # equity-relatif : deux positions qui gagnent font monter l'equite
        # deux fois plus, et ce terme-la le voit. Le R, lui, dit desormais
        # "mes trades sont-ils bons", pas "combien j'en ai ouvert".
        # =================================================================
        if realized_trade != 0.0:
            # Exprimé en R : le montant risqué sur CE trade sert d'unité. Un
            # stop touché vaut -1, la cible (R:R 2.0) vaut +2, une sortie au
            # temps vaut ce qu'elle vaut entre les deux. Diviser par une
            # constante en dollars, comme avant, rendait ce terme dépendant du
            # prix du symbole : 0.81 sur Bitcoin, 0.007 sur l'or.
            reward += float(np.clip(
                realized_trade
                / max(getattr(self, "risk_amount", 0.0), 1e-8)
                / max(n_fermes, 1),
                -3.0, 3.0
            ))

        # Le "Momentum-Confirmed Entry Bonus" (+1.8 à l'ouverture quand mom_5 /
        # rsi_ok / high_vol_regime s'alignent) est retiré : il payait le fait
        # d'OUVRIR, sans jamais regarder le résultat du trade. C'était un a
        # priori codé en dur qui poussait au sur-trading, et sa magnitude (1.8)
        # écrasait le terme de PnL réel. Si ces conditions ont de la valeur, le
        # modèle peut la retrouver — elles sont dans ses features d'entrée.

        # Clip SYMÉTRIQUE. L'ancien [-0.7, +1.4] signalait qu'un gain vaut deux
        # fois une perte, alors que le winrate d'équilibre mesuré est de 49.6% —
        # soit un rapport de 1:1. Cette asymétrie à elle seule rendait une
        # stratégie perdante attractive.
        # Borne de sécurité contre les valeurs aberrantes (gap, ATR dégénéré),
        # PAS un rabot sur le résultat des trades : à +-1 elle ramènerait une
        # cible à +2 R au niveau d'un stop à -1 R, c'est-a-dire qu'elle
        # enseignerait un R:R de 1.0 alors que la configuration en vise 2.0 —
        # et ferait passer le winrate d'équilibre de 33 % à 50 %.
        # L'ECRETAGE DUR EST REMPLACE PAR UNE COMPRESSION QUI GARDE L'ORDRE.
        #
        # `clip(reward, -3.5, 3.5)` rendait identiques toutes les barres
        # au-dela de la borne : -35 % et -91 % d'equite recevaient la meme
        # recompense. Or la ruine est exactement ce qu'on veut enseigner, et
        # la penalite de creux ne peut pas la porter seule — a d proche de 1,
        # son terme est deja dans la zone ecretee.
        #
        # `comprime` est strictement croissante : elle borne la magnitude sans
        # borner la DISTINCTION. Un ecretage large subsiste derriere, contre
        # les valeurs aberrantes, mais a un niveau ou il ne mord plus sur des
        # resultats de trade plausibles.
        if hasattr(self.cfg, "current_epoch") and self.cfg.current_epoch >= 10:
            reward = comprime(reward, 3.5)
            reward = float(np.clip(reward, -12.0, 12.0))

        # ==================================================================
        # LA RECOMPENSE SE REPARTIT ENTRE LES EMPLACEMENTS.
        #
        # POURQUOI. Une decision PPO couvre la vie d'une position : elle
        # accumule les recompenses de l'ouverture a la fermeture. Avec
        # plusieurs positions ouvertes, une recompense scalaire crediterait
        # chaque decision de ce que les AUTRES ont produit — un trade gagnant
        # paierait le trade perdant ouvert a cote, et l'acteur apprendrait une
        # moyenne que personne ne joue.
        #
        # La part de chaque emplacement est son propre mouvement de marquage,
        # plus son propre resultat realise (ajoute dans `_close_position`, en
        # unites du risque DE CE TRADE). Une correction additive repartit
        # ensuite le reste — plafonnements, penalite de drawdown, ecart entre
        # rendement logarithmique et arithmetique — a parts egales entre les
        # emplacements concernes.
        #
        # A K=1 c'est une identite : un seul emplacement, donc il recoit
        # `reward` exactement, quel que soit le detail du calcul ci-dessus.
        # ==================================================================
        # Finalisation
        # ==================================================================
        self.peak_capital = max(self.peak_capital, equity)
        dd = (self.peak_capital - equity) / (self.peak_capital + 1e-8)
        self.max_dd = max(self.max_dd, dd)

        # ON FACTURE LE CREUSEMENT, PAS LE CREUX.
        #
        # J'ai d'abord ecrit `-lambda x d^2`, la rampe sur le NIVEAU. Le pas
        # 18 du scenario `avec objectif` l'a condamnee : compte plat, aucune
        # position, rien d'ouvert ni de ferme — et -4.19e-05 preleve quand
        # meme, a chaque barre, pour un creux hérité d'un trade deja clos.
        # Une taxe sur un etat est une INTEGRALE SUR LE TEMPS : a d = 0.5 elle
        # coute 0.05 par barre, soit -43 500 sur une epoch de 870 000 barres,
        # ce qui noie tout le signal de trading. Et surtout elle ne s'arrete
        # qu'en refaisant un sommet — ou en MOURANT. Je rendais la mort plus
        # attirante en croyant l'en dissuader.
        #
        # LA FORME JUSTE EST CELLE DU POTENTIEL. Avec Phi(s) = -lambda x d^2,
        # le terme `Phi(s') - Phi(s)` ne facture que la VARIATION du creux :
        #
        #   - creux qui se creuse   -> cout
        #   - creux qui se resorbe  -> credit, exactement egal
        #   - compte plat, d constant -> RIEN
        #
        # Il telescope : sur tout un episode le total vaut Phi(fin) - Phi(0),
        # soit -lambda x d_fin^2. Un episode tue au garde-fou paie donc
        # EXACTEMENT -0.2 au total — l'ancienne falaise, au centime — mais
        # etalee le long du chemin, ce qui donne enfin un gradient. Un episode
        # qui stationne a 20 % de creux pendant cent mille barres paie 0.05
        # UNE FOIS, et non cent mille fois.
        #
        # C'est aussi la seule classe de shaping dont on sache qu'elle ne
        # change pas la politique optimale (Ng, Harada & Russell, 1999) :
        # elle redistribue le meme total dans le temps. C'est ce qui leve ma
        # reserve — le shaping a enseigne une strategie perdante ici pendant
        # soixante-quinze epochs, mais c'etait un shaping qui ajoutait de la
        # valeur la ou il n'y avait pas d'argent. Celui-ci n'en ajoute aucune.
        #
        # gamma vaut 1 dans le terme, et non gamma^dt : l'exactitude du
        # telescopage compte ici plus que l'escompte d'une penalite qui, sur
        # une barre, vaut deja presque son actualisation.
        # MEME `d` QUE CELUI QUE LE MODELE OBSERVE : le creux lui-meme, 0 au
        # sommet et 1 a la ruine. Plus de division par 40 % — la penalite
        # saturait alors des 40 % de creux, donc elle cessait de mordre
        # exactement quand la situation empirait.
        # =================================================================
        # LA PENALITE DE CREUX N'EST PLUS A POTENTIEL, ET C'EST TOUT LE POINT.
        #
        # Elle valait `lambda (d_t^2 - d_{t-1}^2)`, une difference de potentiel.
        # Ng, Harada & Russell (1999) etablissent que cette classe de shaping
        # ne change PAS la politique optimale : elle telescope, donc elle
        # redistribue le meme total dans le temps. Le commentaire precedent
        # s'en felicitait. C'etait une erreur de but : on ne cherchait pas a
        # accelerer la convergence vers la meme politique, on cherchait a
        # obtenir une AUTRE politique — une qui ne traverse pas 90 % de creux.
        #
        # Aucune valeur de `penalite_creux` ne pouvait y parvenir. exec31 le
        # montre : creux de validation de 63 a 94 % avec la penalite active.
        #
        # CE QUI LA REMPLACE : un COUT D'ETAT, paye a chaque barre passee en
        # creux, proportionnel a d^2. Il ne telescope pas, donc il deplace
        # reellement l'optimum — un chemin qui plonge et remonte coute
        # desormais plus cher qu'un chemin plat, a resultat final egal.
        #
        # MESURE DU 2026-09-20 (`mesure_objectif.py`, 12 episodes de 8 000
        # barres par palier, entrees neutres identiques d'un palier a l'autre) :
        #
        #   budget   log-richesse   log - temps en creux   creux max
        #     3 %         +0.133                  +0.119       11.8 %
        #     6 %         +0.206                  +0.150       21.2 %
        #    15 %         +0.444                  +0.130       45.3 %
        #    40 %         +0.472                  -0.695       76.5 %
        #
        # LA LOG-RICHESSE SEULE CHOISIT 40 %, le levier maximum — passer a un
        # critere de croissance n'aurait donc rien change. Kelly ne s'auto-
        # limite que si la variance mord assez ; sur cette fenetre l'or monte,
        # aucun episode ne ruine, et 40 % reste en deca du retournement.
        #
        # C'EST LE COUT DU TEMPS PASSE EN CREUX qui deplace l'optimum : il le
        # ramene a 6 % avec 21 % de creux maximum, et rend 40 % franchement
        # negatif. C'est la seule des trois formes testees qui fasse ce qu'on
        # lui demande.
        #
        # LE PRIX A PAYER, ET IL EST REEL : cette penalite N'EST PAS neutre
        # vis-a-vis de la politique optimale. C'est voulu, mais cela signifie
        # que le modele n'optimise plus le rendement — il optimise le
        # rendement SOUS CONTRAINTE DE CHEMIN. Les deux ne coincident pas, et
        # `penalite_creux` regle desormais l'arbitrage entre eux au lieu de
        # n'etre qu'une vitesse de convergence.
        # =================================================================
        # ELLE N'EST FACTUREE QUE PENDANT L'EXPOSITION, et c'est ce qui rend
        # le cout d'etat utilisable.
        #
        # LE COMMENTAIRE CI-DESSUS DECRIT UNE EXPERIENCE DEJA FAITE, et j'ai
        # failli la refaire a l'identique. `-lambda d^2` facture a chaque
        # barre avait ete essaye puis rejete pour deux raisons mesurees :
        #
        #   UNE TAXE SUR UN CREUX HERITE. Compte plat, rien d'ouvert ni de
        #   ferme, et -4.19e-05 preleve quand meme pour un creux laisse par un
        #   trade deja clos. C'est l'invariant de repartition de
        #   `test_concurrence` qui l'attrape — il l'a attrape de nouveau.
        #
        #   ELLE RENDAIT LA MORT ATTIRANTE. Si le seul moyen d'arreter de
        #   payer est de refaire un sommet ou de terminer l'episode, terminer
        #   devient l'option bon marche. On dissuade la ruine en la
        #   recompensant.
        #
        # LA CONDITION D'EXPOSITION LEVE LES DEUX D'UN COUP. Un compte plat ne
        # paie RIEN : le creux herite ne se facture plus, l'invariant est
        # respecte, et surtout FERMER SES POSITIONS SUFFIT A ARRETER LE COUT.
        # La mort n'est plus la seule issue — elle n'est meme plus une issue
        # avantageuse, puisque se mettre a plat coute strictement moins cher.
        #
        # Et le sens economique suit : on paie le creux tant qu'on y est
        # EXPOSE, pas tant qu'on s'en souvient.
        #
        # CE QUI RESTE VRAI DU REPROCHE : ce terme n'est PAS neutre vis-a-vis
        # de la politique optimale. C'est voulu — c'etait tout l'objet du
        # changement — mais cela signifie que le modele optimise desormais le
        # rendement SOUS CONTRAINTE DE CHEMIN, et que `penalite_creux` arbitre
        # entre les deux au lieu de n'etre qu'une vitesse de convergence.
        _dist_creux = min(max(dd, 0.0), 1.0)
        _expose = bool(np.any(self._p_sens != 0))
        _pen = (float(getattr(self.cfg, "penalite_creux", 0.2))
                * _dist_creux * _dist_creux) if _expose else 0.0
        self._d_prec = _dist_creux
        reward -= _pen

        # CHAQUE POSITION CALCULE SA PROPRE RECOMPENSE. Elle n'est pas une
        # part d'un total.
        #
        # La premiere version repartissait : chaque emplacement recevait son
        # marquage, puis une correction additive egale ramenait la somme a la
        # recompense globale. Le controle de signe l'a prise en defaut — un
        # trade gagnant de +0.91 $ accumulait -0.118 de recompense, parce que
        # la correction lui faisait porter les pertes des positions ouvertes a
        # cote. Une decision aurait appris le resultat de ses voisines.
        #
        # Ici la formule est celle de la recompense globale, appliquee a la
        # variation d'equity DE CET EMPLACEMENT seul. A K=1 l'emplacement
        # unique porte toute la variation, donc sa part vaut `reward`
        # EXACTEMENT, terme pour terme — c'est ce que `test_concurrence.py`
        # verifie sur 613 trades.
        #
        # La somme sur les emplacements ne vaut alors plus exactement la
        # recompense globale, et c'est voulu : celle-ci est une grandeur de
        # PORTEFEUILLE, avec ses propres plafonnements. Les deux ne decrivent
        # pas la meme chose et n'ont aucune raison de coincider a plusieurs.
        _concernes = set(np.flatnonzero(self._actifs).tolist())
        _concernes.update(self._slots_fermes)
        if self._slot_ouvert >= 0:
            _concernes.add(int(self._slot_ouvert))
        if _concernes:
            # LA PENALITE DE DRAWDOWN N'EST PAS PARTAGEE, ELLE EST PORTEE
            # PAR CHAQUE POSITION.
            #
            # Je l'avais divisee par le nombre de positions concernees, pour
            # ne pas la compter soixante fois. J'ai obtenu l'inverse du but :
            # a K=1 elle valait -0.2, a 60 positions -0.003. Elle disparaissait
            # exactement quand elle devait mordre le plus.
            #
            # Et ce n'est pas un detail de reglage. Mesure du 2026-09-16 : la
            # part de marquage au marche a une mediane de 0.0007 par barre
            # quand le terme de trade realise atteint 3. Sur la vie d'un
            # trade le marquage cumule ~0.09 contre 1 a 3 — donc la concavite
            # du logarithme, seule chose qui facture le risque, pese 3 a 10 %
            # du signal. Tout le reste est LINEAIRE dans le nombre de
            # positions et aveugle a leur correlation.
            #
            # Le pricing du risque reposait donc entierement sur cette
            # penalite, que j'avais annulee. Non partagee, elle totalise
            # 0.2 x K : elle croit avec l'exposition, ce qui est precisement
            # le gradient qui manquait pour distinguer la 1re position de la
            # 40e. Chaque position ouverte a contribue au creux ; chaque
            # decision doit le sentir.
            # `_pen` EST CELUI DE LA PENALITE GLOBALE, calcule UNE SEULE FOIS
            # plus haut — le recalculer ici le doublerait, puisque le terme
            # de potentiel avance `_d_prec` au passage. Chaque emplacement
            # porte la meme variation de creux, non divisee par leur nombre,
            # pour la raison exposee ci-dessus.
            _clip = (hasattr(self.cfg, "current_epoch")
                     and self.cfg.current_epoch >= 10)
            # ON NE PARCOURT QUE LES EMPLACEMENTS CONCERNES. La boucle
            # portait sur `range(self._K)` — SOIXANTE-QUATRE tours de boucle
            # Python par barre, qui doublent avec le tableau, pour n'en
            # traiter qu'un ou deux. A 870 000 barres par epoch cela faisait
            # 56 millions d'iterations dont 55 ne servaient qu'a ecrire un
            # zero sur un zero.
            #
            # CAR C'ETAIT BIEN UN ZERO SUR UN ZERO : `_r_slots` est remis a
            # zero en tete de pas, et les seuls emplacements qu'on y ajoute
            # ensuite — ceux que `_close_position` crediteur, celui qui
            # vient d'ouvrir, ceux encore ouverts — sont exactement
            # `_concernes`. La branche supprimee n'ecrivait donc jamais rien
            # d'autre que la valeur deja en place.
            #
            # EN TABLEAU, ET C'EST MAINTENANT QUE CA COMPTE. La boucle
            # Python tournait une fois par emplacement CONCERNE et par
            # barre : deux tours quand le budget de risque n'autorisait
            # qu'une position, mais soixante depuis qu'il est retire. A
            # 870 000 barres par epoch, c'est 52 millions d'iterations.
            #
            # `math.log` ETAIT LE VERROU, et il a saute a la mesure. Le
            # remplacer par `np.log` avait ete refuse faute de garantie sur
            # l'egalite des bits. Mesure du 2026-09-19, trois millions de
            # valeurs tirees sur toute la plage utile — rapports proches de
            # 1, rapports larges, rapports a 1e-9 pres — : IDENTIQUES AU BIT
            # dans les trois cas. Le reste suit terme pour terme,
            # `np.clip(x, -a, a)` etant exactement `max(min(x, a), -a)`.
            _lat = self._latents_par_slot(price)
            _prec, _real = self._latent_prec, self._realise_slots
            _idx = np.fromiter(_concernes, np.int64, len(_concernes))
            _d = _lat[_idx] - _prec[_idx] + _real[_idx]
            _lr = np.log(np.maximum(prev_equity_clamped + _d, 1e-8)
                         / prev_equity_clamped)
            # LE PARENTHESAGE EST CELUI DE LA BOUCLE, et ce n'est pas une
            # coquetterie. Elle ecrivait `out[j] += 10*c - pen`, donc
            # `out + (10*c - pen)`. Ecrire `(out + 10*c) - pen` donne un
            # resultat different de 4.4e-16 des que la penalite est active :
            # l'addition flottante n'est pas associative. Le test bit a bit
            # l'a montre — identique a penalite nulle, different sinon.
            _v = self._r_slots[_idx] + (10.0 * comprime_log(_lr)
                                        - _pen)
            self._r_slots[_idx] = np.clip(_v, -3.5, 3.5) if _clip else _v
        else:
            self._r_slots[:] = 0.0

        # A UNE SEULE POSITION, LA PART EST LA RECOMPENSE PAR AFFECTATION.
        #
        # Le calcul par emplacement est algebriquement le meme que le calcul
        # global, mais il n'emprunte pas le meme chemin de flottants : l'ecart
        # est de l'ordre de 1e-16 par barre. Mesure du 2026-09-16 — il suffit
        # a faire diverger la perte auxiliaire (2.0666 contre 2.0662), donc le
        # tirage des mini-lots, donc le seuil de calibration, donc quatre
        # trades de validation. Le lot collecte etait pourtant identique :
        # 580 decisions et un ecart-type d'avantage de 1.67 des deux cotes.
        #
        # Une affectation supprime la question. Elle ne cache rien : a K=1 il
        # n'y a qu'une position, donc sa part EST le total, et c'est la
        # definition, pas une approximation.
        if self._K == 1:
            self._r_slots[0] = reward


        self.idx += 1
        done = done_reason is not None

        obs = self._get_obs()

        return obs, float(reward), done, False, {
            # La repartition par emplacement, pour que la boucle de collecte
            # attribue a chaque decision ce que SA position a produit. A K=1
            # ce tableau a un seul element, egal a `reward`.
            "r_slots": self._r_slots.copy(),
            "slots_fermes": list(self._slots_fermes),
            "slot_ouvert": int(self._slot_ouvert),
            "n_positions": self.n_positions,
            "capital": self.capital,
            "drawdown": self.max_dd,
            "done_reason": done_reason,
            "position": self.position
        }



# ======================================================================
# PPO UTILITAIRES
# ======================================================================

def etat_gpu() -> str:
    """Temperature et frequence du GPU, pour le log d'epoch.

    POURQUOI CETTE LIGNE EXISTE. Les epochs ralentissaient regulierement — 36 s,
    50 s, 92 s, 112 s — et toutes les phases ensemble. On a d'abord cherche une
    liste qui s'allonge, puis une inefficacite d'architecture. La cause etait
    ailleurs : mesure en cours de run, GPU a 90 degres, frequence tombee a
    210 MHz sur un maximum de 2100, raison de bridage 0x20 = SwThermalSlowdown.
    La machine se bride a 10 % de sa vitesse, et rien dans le log ne le disait.

    Un ralentissement thermique ressemble a un probleme de code tant qu'on ne
    regarde pas la frequence. Maintenant on la regarde.
    """
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,clocks.sm,clocks.max.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        t, sm, mx = [int(x.strip()) for x in r.stdout.strip().split(",")]
        pct = 100 * sm / max(mx, 1)
        alerte = "  BRIDE" if pct < 70 else ""
        return f"gpu[{t}C {sm}/{mx}MHz {pct:.0f}%{alerte}]"
    except Exception:
        return "gpu[?]"


def get_device(cfg: PPOConfig):
    if cfg.force_cpu:
        print("CPU forcé.")
        return torch.device("cpu")
    if torch.cuda.is_available():
        print("CUDA détecté — utilisation GPU.")
        return torch.device("cuda")
    print("Pas de CUDA — utilisation CPU.")
    return torch.device("cpu")


class RewardNormalizer:
    """Normalise les rewards par l'écart-type glissant (algorithme de Welford)."""
    def __init__(self):
        self._mean = 0.0
        self._var  = 1.0
        self._count = 0

    def update(self, x: float):
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        self._var  += (delta * (x - self._mean) - self._var) / self._count

    @property
    def std(self) -> float:
        return max(math.sqrt(abs(self._var)), 1e-8)

    def normalize(self, x: float) -> float:
        return x / self.std


def compute_gae(rewards, values, dones, gamma, lam, last_value=0.0):
    values = values + [last_value]
    gae = 0.0
    adv = []

    for t in reversed(range(len(rewards))):
        mask = 1 - int(dones[t])
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv.insert(0, gae)

    returns = [adv[i] + values[i] for i in range(len(adv))]
    return adv, returns


def compute_gae_semi_mdp(rewards, values, dones, dts, gamma, lam, last_value=0.0):
    """GAE pour des actions de DURÉE VARIABLE.

    Chaque transition k va d'une décision à la suivante et couvre Δt bougies :
    `rewards[k]` est déjà la somme actualisée Σ γ^i r_i sur cette durée. Le
    bootstrap doit donc être escompté de γ^Δt et non de γ, sinon une position
    tenue 1000 bougies serait valorisée comme une attente d'une seule bougie.
    """
    values = list(values) + [last_value]
    gae = 0.0
    adv = []

    for t in reversed(range(len(rewards))):
        mask = 1 - int(dones[t])
        g_dt = gamma ** max(int(dts[t]), 1)
        delta = rewards[t] + g_dt * values[t + 1] * mask - values[t]
        gae = delta + g_dt * lam * mask * gae
        adv.insert(0, gae)

    returns = [adv[i] + values[i] for i in range(len(adv))]
    return adv, returns


_VEILLEUR = None


def _veilleur():
    """L'analyste des epochs, construit une fois et garde pour tout le run.

    Son etat est par FOLD — PnL cumule, epoch precedente, epochs a actor gele
    qui servent de reference au hasard — donc il doit traverser les trois
    folds. Un objet par epoch afficherait "premiere epoch" quatre-vingt-dix
    fois de suite.
    """
    global _VEILLEUR
    if _VEILLEUR is None:
        import veille_epochs
        _VEILLEUR = veille_epochs.Veilleur()
    return _VEILLEUR


class ScoresAPlat:
    """Les scores de decision de CHAQUE barre, calcules en une seule passe.

    POURQUOI. Profilage de la validation, 2026-09-19 : 131 s au total, dont
    **124.6 s dans le forward du reseau** — 5 760 appels de lot 10, a 21.6 ms
    chacun. `env.step` n'en represente que 6.7. Un reseau de 45 000
    parametres coute la meme chose a lot 1 qu'a lot 64 : ce n'est pas le
    calcul qui coute, c'est l'appel. Vectoriser du numpy n'y changerait rien.

    CE QUI REND LE GROUPAGE POSSIBLE. L'observation vaut 256 colonnes de
    marche plus QUATRE colonnes d'etat : sens, latent en ATR, duree de
    detention, capacite restante. A une barre ou l'environnement est PLAT —
    et la validation ne decide que la — les trois premieres valent exactement
    zero par construction, et la quatrieme vaut `libres / (libres + tenues)`
    avec `tenues = 0`, donc 1.0 des qu'une place est ouvrable.

    L'observation d'un instant de decision ne depend donc que du MARCHE. Elle
    se calcule pour toutes les barres a l'avance, par lots de 8 192, ce que
    le diagnostic de rang fait deja depuis toujours.

    CE QUI N'EST PAS SUPPOSE. Le cas `libres = 0` donne 0.0 et non 1.0. On ne
    parie pas dessus : `pour()` COMPARE les quatre colonnes d'etat reelles a
    celles du cache et ne sert la valeur memorisee que si elles coincident,
    au bit pres. Les autres repassent par un forward. C'est exact par
    construction, et le compteur `secours` dit combien de fois c'est arrive.
    """

    ETAT_A_PLAT = (0.0, 0.0, 0.0, 1.0, 0.0)

    def __init__(self, policy, data, cfg, device, masque, lot=8192,
                 aux=True):
        self.policy = policy
        self.data = data
        self.cfg = cfg
        self.device = device
        self.masque = masque
        self.aux = aux
        self.lot = lot
        self.secours = 0
        self.servis = 0
        self._table = None

    def _construit(self):
        lb = self.cfg.lookback
        n = self.data.length
        idx = np.arange(lb, n, dtype=np.int64)
        n_f = self.data.features.shape[1]
        table = np.full((n, N_ACTIONS), np.nan, np.float32)
        masque_t = torch.from_numpy(
            np.repeat(self.masque[None, :], self.lot, axis=0)).to(self.device)

        # LE MODE DU RESEAU EST RESTAURE, PAS FORCE. Une premiere version
        # finissait par `policy.train()` : appelee depuis la calibration,
        # elle rallumait le DROPOUT au milieu de la validation, qui tourne en
        # `eval()` depuis toujours. Le test de non-regression l'a prise en
        # defaut — ecart maximal 0.23 la ou on attendait 1e-6 — et sans lui
        # la validation aurait mesure une politique bruitee sans que rien ne
        # leve.
        etait_en_train = self.policy.training
        self.policy.eval()

        # LES FENETRES SE PRENNENT EN VUE GLISSANTE, sans boucle Python. La
        # premiere version recopiait `features[i-lb:i]` ligne a ligne : 63 000
        # tours de boucle, 34 s pour une fenetre de validation. La vue ne
        # copie rien, c'est le passage au tenseur qui materialise le lot.
        fen = np.lib.stride_tricks.sliding_window_view(
            self.data.features, lb, axis=0)          # (n-lb+1, n_f, lb)
        try:
            with torch.no_grad():
                for d0 in range(0, len(idx), self.lot):
                    b = idx[d0:d0 + self.lot]
                    obs = np.empty((len(b), lb, n_f + 5), np.float32)
                    # `fen[i - lb]` porte les barres [i-lb, i), axes (n_f, lb)
                    obs[:, :, :n_f] = np.swapaxes(fen[b - lb], 1, 2)
                    obs[:, :, n_f:n_f + 3] = 0.0
                    obs[:, :, n_f + 3] = 1.0
                    obs[:, :, n_f + 4] = 0.0
                    st = torch.from_numpy(obs).to(self.device)
                    lg, _ = self.policy(st)
                    pr = torch.softmax(
                        lg.masked_fill(~masque_t[:len(b)], MASK_VALUE),
                        dim=-1).cpu().numpy()
                    if self.aux:
                        try:
                            a = self.policy.rendement(st).float().cpu().numpy()
                            pr = pr.copy()
                            pr[:, :2] = 1.0 / (1.0 + np.exp(-np.clip(a, -30, 30)))
                        except Exception:
                            self.aux = False
                    table[b] = pr
        finally:
            if etait_en_train:
                self.policy.train()
        self._table = table

    def pour(self, etats, indices):
        """Rend le tableau de scores pour ces observations, dans cet ordre.

        `etats[i]` est l'observation deja construite, `indices[i]` la barre
        qu'elle decrit. Une observation dont les colonnes d'etat ne sont pas
        celles du cache repasse par le reseau.
        """
        if self._table is None:
            self._construit()
        n_f = self.data.features.shape[1]
        attendu = np.asarray(self.ETAT_A_PLAT, np.float32)
        out = np.empty((len(etats), N_ACTIONS), np.float32)
        manquants = []
        for i, (o, bar) in enumerate(zip(etats, indices)):
            if (0 <= bar < self._table.shape[0]
                    and np.isfinite(self._table[bar, 0])
                    and np.array_equal(o[-1, n_f:], attendu)):
                out[i] = self._table[bar]
                self.servis += 1
            else:
                manquants.append(i)
        if manquants:
            self.secours += len(manquants)
            vb = np.stack([etats[i] for i in manquants], axis=0)
            st = torch.as_tensor(vb, dtype=torch.float32, device=self.device)
            mk = torch.from_numpy(
                np.repeat(self.masque[None, :], len(manquants), axis=0)
            ).to(self.device)
            etait = self.policy.training
            self.policy.eval()
            with torch.no_grad():
                lg, _ = self.policy(st)
                pr = torch.softmax(lg.masked_fill(~mk, MASK_VALUE),
                                   dim=-1).cpu().numpy()
                if self.aux:
                    try:
                        a = self.policy.rendement(st).float().cpu().numpy()
                        pr[:, :2] = 1.0 / (1.0 + np.exp(-np.clip(a, -30, 30)))
                    except Exception:
                        pass
            if etait:
                self.policy.train()
            for j, i in enumerate(manquants):
                out[i] = pr[j]
        return out


def build_action_mask_from_positions(positions: torch.Tensor, side: str) -> torch.Tensor:
    device = positions.device
    B = positions.shape[0]
    mask = torch.zeros(B, N_ACTIONS, dtype=torch.bool, device=device)

    flat = (positions == 0)
    inpos = ~flat

    # 0=BUY  1=SELL  2=HOLD
    if flat.any():
        if side == "both":
            mask[flat, 0] = True
            mask[flat, 1] = True
            mask[flat, 2] = True
        elif side == "long":
            mask[flat, 0] = True
            mask[flat, 2] = True
        elif side == "short":
            mask[flat, 1] = True
            mask[flat, 2] = True
        else:
            mask[flat, 0] = True
            mask[flat, 1] = True
            mask[flat, 2] = True

    if inpos.any():
        mask[inpos, 2] = True  # HOLD = nouvelle index 2

    return mask


def map_agent_action_to_env_action(
    a: int,
    pos: int,
    cfg: PPOConfig,
    device: torch.device,
    state_tensor: torch.Tensor,
    policy_long: Optional[nn.Module] = None,
    policy_short: Optional[nn.Module] = None,
    epoch: int = 1,
) -> Tuple[int, float]:
    env_action = 2
    risk_scale = 1.0

    if state_tensor.dim() == 2:
        state_tensor = state_tensor.unsqueeze(0)
    elif state_tensor.dim() == 3 and state_tensor.size(0) > 1:
        state_tensor = state_tensor[:1]
    elif state_tensor.dim() != 3:
        raise ValueError(f"state_tensor doit être (B,T,F) ou (T,F), reçu {state_tensor.shape}")

    # Mode "close" supprimé : seuls les sides "both" / "long" / "short" sont supportés.
    # Modes both / long / short — espace agent réduit à 3 actions
    #   a=0 → BUY     (env_action=0)
    #   a=1 → SELL    (env_action=1)
    #   a=2 → HOLD    (env_action=2)
    if pos == 0:
        if a == 0:    # BUY
            env_action = 0
            risk_scale = 1.0
        elif a == 1:  # SELL
            env_action = 1
            risk_scale = 1.0
        else:         # HOLD ou tout autre
            env_action = 2
            risk_scale = 1.0
    else:
        env_action = 2
        risk_scale = 1.0

    if epoch <= 35:
        risk_scale = 1.0

    return env_action, risk_scale


# ======================================================================
# TRAINING PPO (sur un split donné)
# ======================================================================

def _correlation_rang(a: np.ndarray, b: np.ndarray) -> float:
    """Rho de Spearman, egalites traitees par rangs moyens.

    Les egalites ne sont pas un detail : une politique saturee rend la meme
    probabilite sur des milliers de barres, et leur donner des rangs
    arbitraires fabriquerait de la correlation a partir de l'ordre du tableau.
    """
    def _rg(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), float)
        r[o] = np.arange(len(x), dtype=float)
        xs = x[o]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = _rg(np.asarray(a, float)), _rg(np.asarray(b, float))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = ra.std() * rb.std() * len(ra)
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def run_training_on_split(
    train_data: MarketData,
    calib_data: MarketData,
    val_data: MarketData,
    test_data: MarketData,
    stats: Dict[str, np.ndarray],
    cfg: PPOConfig,
    suffix: str = "",
    poids_initiaux: Optional[str] = None,
):
    import csv as _csv

    device = get_device(cfg)

    # Persist the effective configuration: source defaults may change after a run.
    import json
    import hashlib
    from pathlib import Path
    manifest_path = Path(f"run_{cfg.model_prefix}_{cfg.side}{suffix}.json")
    if manifest_path.exists() and not cfg.resume_weights:
        raise FileExistsError(f"Run existant: {manifest_path}. Choisir un nouveau model_prefix ou dossier.")
    manifest = {"config": vars(cfg), "seed": SEED, "features": list(FEATURE_COLS),
                "splits": {"train": len(train_data), "calibration": len(calib_data),
                           "validation": len(val_data), "test_reserved": len(test_data)},
                "source_sha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                  for name in ("training.py", "saint_core.py", "execution_quotes.py", "economic_learning.py")}}
    manifest_path.write_text(json.dumps(manifest, default=str, indent=2), encoding="utf-8")

    # CSV de métriques
    csv_path = f"training_log_{cfg.side}{suffix}.csv"
    _csv_fields = [
        "epoch",
        "train_pnl", "train_trades", "train_win", "train_loss",
        "train_totalW", "train_totalL", "train_avgW", "train_avgL", "train_nbL",
        "train_pf", "train_dd",
        # Split LONG / SHORT côté training
        "train_long_w", "train_long_l", "train_long_pnl",
        "train_short_w", "train_short_l", "train_short_pnl",
        "val_pnl", "val_trades", "val_win", "val_loss",
        "val_totalW", "val_totalL", "val_avgW", "val_avgL", "val_nbL",
        "val_pf", "val_dd",
        # Split LONG / SHORT côté validation
        "val_long_w", "val_long_l", "val_long_pnl",
        "val_short_w", "val_short_l", "val_short_pnl",
        "sortino", "sortino30",
        "actor_loss", "critic_loss", "entropy", "entropy_flat", "kl", "grad_norm",
        "buy_ratio", "sell_ratio", "hold_ratio", "close_ratio",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as _f:
        _csv.DictWriter(_f, fieldnames=_csv_fields).writeheader()

    # CSV trade-by-trade (TRAIN + VAL) : 1 ligne par trade fermé
    trades_csv_path = f"trades_{cfg.side}{suffix}.csv"
    _trades_fields = [
        "epoch", "phase", "episode", "entry_idx", "exit_idx",
        "side", "entry_price", "exit_price", "pnl",
        "hit_sl", "hit_tp", "hold_bars",
    ]
    with open(trades_csv_path, "w", newline="", encoding="utf-8") as _ft:
        _csv.DictWriter(_ft, fieldnames=_trades_fields).writeheader()

    def _param_grad_norm(params) -> float:
        """Norme L2 des gradients d'un groupe, en UNE synchronisation.

        L'ancienne version appelait `.item()` sur chaque tenseur. Avec 82
        tenseurs de parametres et trois groupes evalues a chaque pas
        d'optimisation, cela faisait ~41 000 synchronisations GPU->CPU par
        epoch, pour du code purement DIAGNOSTIQUE — il ne sert qu'a afficher
        g[actor … critic … tronc …] dans la ligne META.

        `_foreach_norm` calcule toutes les normes en un seul lancement fusionne,
        et on ne rapatrie qu'un scalaire.
        """
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return 0.0
        return float(torch.stack(torch._foreach_norm(grads)).norm(2).item())

    # Lignes de masque précalculées en numpy, dérivées de l'unique source de
    # vérité (build_mask_from_pos_scalar) pour rester cohérentes avec le live.
    # Seul le masque FLAT sert désormais : depuis la réécriture semi-MDP, la
    # policy n'est jamais interrogée en position (l'action y est forcée).
    MASK_FLAT = build_mask_from_pos_scalar(0, torch.device("cpu"), cfg.side).numpy()

    # ---- Le troisieme votant, ajuste en croise sur CETTE fenetre ----
    votant = None
    if getattr(cfg, "votant_tabm", False):
        try:
            import banc_rendement_net as _B
            import evalue_tabm_test as _E
            import tabm_votant as _TV
            t0 = time.time()
            votant = _TV.ajuste(
                train_data.df, 0, train_data.length, _B.modele_tabm(),
                _E.cibles_brutes, FEATURE_COLS, stats,
                pas_train=_E.PAS_TRAIN)
            print(f"  • Troisieme votant TabM ajuste en croise "
                  f"({_TV.K_BLOCS} blocs, {time.time()-t0:.0f}s) : "
                  f"{100*votant.couverture():.0f} % des barres notees, "
                  f"seuil {votant.seuil:+.4f} R")
        except Exception as e:
            print(f"  • Votant TabM INDISPONIBLE ({type(e).__name__}: {e}) — "
                  f"le vote se fait a deux. Ce n'est pas la configuration "
                  f"decrite, et il faut le savoir avant de lire le resultat.")
            votant = None

    # Pools d'environnements créés UNE SEULE FOIS : le constructeur calcule les
    # quantiles de volatilité du curriculum sur tout le split (np.quantile sur
    # ~1.2M lignes), donc les instancier à chaque epoch coûterait plusieurs
    # minutes par fold pour rien. Ils sont simplement reset() à chaque epoch.
    # Tous partagent le même objet cfg, donc cfg.current_epoch les pilote tous.
    train_envs = [
        BTCTradingEnvDiscrete(train_data, cfg) for _ in range(cfg.episodes_per_epoch)
    ]
    # Nombre d'episodes REELLEMENT joues. Il part d'une estimation — la duree
    # mediane d'un trade donne l'ordre de grandeur des decisions par episode —
    # puis se corrige des la premiere epoch avec le compte reel.
    #
    # Partir du plafond ferait payer plein tarif la seule epoch qui n'a pas
    # encore de mesure, et c'est justement celle qu'on regarde le plus.
    _dec_par_ep = max(cfg.episode_length / max(cfg.atr_sl_mult * 1.7, 1.0), 1.0)
    n_episodes_courant = [max(1, min(cfg.episodes_per_epoch,
                                     int(round(cfg.cible_trades / _dec_par_ep))))]

    # MESURE EXHAUSTIVE, pas échantillonnée. Voir departs_disjoints : la
    # validation, la calibration et le test couvrent désormais leur fenêtre
    # entière une fois, au lieu de tirer des départs au hasard dans un pool
    # biaisé par le curriculum de volatilité. `cfg.val_episodes` ne dimensionne
    # donc plus rien ; le nombre d'épisodes est celui qu'exige la fenêtre.
    departs_val = departs_disjoints(val_data.length, cfg.lookback,
                                    cfg.episode_length)
    departs_calib = departs_disjoints(calib_data.length, cfg.lookback,
                                      cfg.episode_length)
    val_envs = [BTCTradingEnvDiscrete(val_data, cfg) for _ in departs_val]
    # Environnements de CALIBRATION, sur la fenetre qui PRECEDE celle de mesure.
    calib_envs = [BTCTradingEnvDiscrete(calib_data, cfg) for _ in departs_calib]
    print(f"  • Mesure exhaustive : {len(departs_val)} épisodes de validation "
          f"({100*len(departs_val)*cfg.episode_length/max(val_data.length,1):.0f} % "
          f"de la fenêtre), {len(departs_calib)} de calibration")

    # ------------------------------------------------------------------
    # LE CLASSEMENT, mesure de diagnostic ajoutee le 2026-09-16.
    #
    # POURQUOI. La validation rend ~150 trades par epoch et l'erreur-type sur
    # leur moyenne vaut 0.11 R. Elle ne peut donc pas distinguer un modele qui
    # ajoute 0.10 R d'un modele qui n'ajoute rien, et le PnL affiche plus haut
    # oscille en consequence : sur exec37, -7, 0, +44, +11, +117, +1, +58, -71
    # dollars d'une epoch a l'autre, sans que rien n'ait appris ni desappris.
    #
    # La selectivite jette 95 % de l'information. Le reseau produit une
    # opinion a CHAQUE barre ; n'en regarder que le sommet, c'est mesurer la
    # moyenne d'un echantillon de 150 quand on en a 11 000. On ne demande donc
    # plus "combien gagne-t-il" mais "CLASSE-T-IL" — la seule chose dont la
    # selectivite a besoin, et une question a laquelle un grand echantillon
    # repond. Mesure sur exec32 : rho +0.0696 +/- 0.0238, soit 2.9 ecarts-
    # types, la ou le PnL de validation du meme run ne depassait pas 1.3.
    #
    # CE BLOC NE CALCULE PLUS QUE LA GRILLE ET SON FILTRE.
    #
    # Il affirmait deux choses devenues fausses. « Le rendement reel ne
    # depend pas du modele : il se calcule une fois ici » — vrai tant que la
    # sortie etait un horizon fixe, faux depuis qu'elle est une politique
    # PPO qui change a chaque epoch. Et « rien ne s'en sert pour
    # selectionner » — faux aussi : `_rang_gain` alimente `score_retenue`,
    # qui DECIDE du checkpoint. Les rendements sont donc recalcules a chaque
    # epoch par `rendements_sortie_ppo` ; ceux d'ici ne servent plus qu'a
    # filtrer les occasions valides.
    _rang_idx = _rang_reel = None
    _rang_ra = _rang_rv = None
    # LES MEMES OCCASIONS TENUES `horizon_cloture` BARRES, gardees pour le
    # fold : `_rang_ra` est ecrase a chaque epoch par la sortie PPO. Elles
    # ne decident de rien — voir `_g60_top`.
    _rang_ra60 = _rang_rv60 = None
    _g60_top = _g60_tous = float("nan")
    if getattr(cfg, "diag_rang", True):
        _pas = max(int(getattr(cfg, "diag_rang_pas", 12)), 1)
        _i = np.arange(cfg.lookback,
                       val_data.length - _borne_syst(cfg) - 2, _pas)
        if len(_i) >= 500:
            _ra, _rv = rendements_du_systeme(val_data.df, _i, cfg)
            _ok = np.isfinite(_ra) & np.isfinite(_rv)
            # ON NE JUGE LA REGLE QUE SUR CE QU'ELLE PEUT PRENDRE.
            #
            # `rhoAux` et `sommet` se calculaient sur TOUTE la grille, y
            # compris les occasions que le veto de tendance interdit. Or la
            # tete n'y est jamais entrainee — aucune decision n'y existe — et
            # le systeme deploye n'y entrera jamais. On la notait donc sur un
            # domaine hors de sa question.
            #
            # MESURE DU 2026-09-20, grille horaire, veto `tend_mom_mois >= 0` :
            #
            #                  entrainee sur   jugee sur   R la ou elle apprend
            #   entrainement          51.6 %       100 %   +0.412  (ailleurs -0.083)
            #   validation            64.3 %       100 %   +0.543  (ailleurs +0.822)
            #
            # En validation, 36 % de la grille lui est INCONNUE et ces
            # occasions rapportent PLUS (+0.822 contre +0.543). `sommet`
            # piochait donc ses 5 % dans un vivier dont une part n'est pas
            # deployable, et `rhoAux` la notait sur des questions qu'on ne lui
            # posera jamais. C'est ce qui explique un rho negatif (-0.135
            # mesure sur le checkpoint d'exec31) alors que la CIBLE, elle,
            # transfere a +0.921.
            #
            # Le veto est lu sur la colonne BRUTE, comme dans l'environnement.
            _vt = getattr(cfg, "veto_tendance", "")
            if _vt and _vt in val_data.df.columns:
                _mv = val_data.df[_vt].to_numpy(np.float64)
                _seuil = float(getattr(cfg, "veto_tendance_seuil", 0.0))
                _elig = np.zeros(len(_i), bool)
                _j = np.maximum(_i - 1, 0)
                _v = _mv[_j]
                _elig = np.isfinite(_v) & (_v >= _seuil)
                _ok = _ok & _elig
                print(f"  • Classement : veto `{_vt}` applique a la grille — "
                      f"{100*_elig.mean():.0f} % des occasions sont "
                      f"deployables, les autres ne sont ni apprises ni jugees")
            if _ok.sum() >= 500:
                _rang_idx = _i[_ok]
                # Part SYMETRIQUE : (achat - vente) / 2. La derive du
                # sous-jacent s'annule, donc un rho positif dit que le modele
                # distingue les moments, pas qu'il a profite d'une hausse.
                _rang_reel = ((_ra - _rv) / 2.0)[_ok]
                # LE GAIN DU COTE REELLEMENT JOUE, en unites de risque.
                #
                # `_rang_reel` est la part SYMETRIQUE, (achat - vente) / 2 :
                # elle annule la derive du sous-jacent, donc un rho positif
                # dit que le modele distingue les MOMENTS et non qu'il a
                # profite d'une hausse. C'est la bonne cible pour juger un
                # CLASSEMENT.
                #
                # Ce n'est pas ce que la strategie ENCAISSE. Un run long-only
                # prend des achats : ce qu'il gagne, c'est `_ra`, derive
                # comprise. Pour juger la RENTABILITE du tri il faut donc le
                # rendement du cote qu'on joue vraiment, et les deux
                # grandeurs doivent coexister — l'une dit si l'ordre est bon,
                # l'autre si le sommet paie.
                # LE COTE JOUE EST CELUI QUE LE MODELE CHOISIT, ET IL
                # NE PEUT DONC PAS SE CALCULER ICI.
                #
                # Cette ligne valait `np.maximum(_ra, _rv)` des que les deux
                # cotes etaient permis : le rendement du MEILLEUR cote,
                # choisi apres coup. C'est un regard en avant, et il entrait
                # droit dans le critere de retenue — `score_retenue_grille`
                # et le portillon du hasard lisent tous deux ce tableau.
                #
                # IL DORMAIT DERRIERE LE LONG-ONLY. Avec un seul cote permis
                # il n'y a pas de choix a faire et la branche etait morte ;
                # elle s'est reveillee le 2026-09-21 en ouvrant la vente.
                # Un modele incapable de distinguer l'achat de la vente
                # aurait ete credite du bon cote a chaque occasion.
                #
                # ET LA SELECTION AVAIT LE DEFAUT SYMETRIQUE : `_top`
                # prenait les scores les PLUS HAUTS, c'est-a-dire les plus
                # fortes convictions d'ACHAT. Les ventes, dont le score est
                # negatif par construction (`achat - vente`), n'etaient
                # jamais retenues. On selectionnait donc des achats et on
                # les payait au tarif du meilleur des deux cotes.
                #
                # Les deux colonnes sont desormais gardees telles quelles ;
                # le sens, le gain et le rho se derivent par epoch du score
                # du modele, plus bas.
                _rang_ra = _ra[_ok]
                _rang_rv = _rv[_ok]
                _rang_ra60 = np.nan_to_num(_rang_ra.copy(), nan=0.0)
                _rang_rv60 = np.nan_to_num(_rang_rv.copy(), nan=0.0)
                _ca_d, _cv_d = cotes_permises(cfg.side)
                print(f"  • Classement : {len(_rang_idx):,} decisions de "
                      f"validation suivies (une toutes les {_pas} barres), "
                      f"gain median achat {float(np.median(_rang_ra)):+.3f} / "
                      f"vente {float(np.median(_rang_rv)):+.3f} — le cote "
                      f"joue est celui que le modele choisit "
                      f"({'les deux permis' if _ca_d and _cv_d else cfg.side})")

    def _un_membre(archi, taille_patch, pas):
        """Un reseau seul, a l'architecture demandee."""
        if archi == "patchtst":
            return build_policy(
                device, lookback=cfg.lookback, n_features=OBS_N_FEATURES,
                archi="patchtst", n_ref=cfg.n_ref, num_blocks=cfg.num_blocks,
                d_model_patch=cfg.d_model_patch, mlp_dim=cfg.mlp_dim,
                taille_patch=taille_patch, pas=pas)
        # Memes arguments que la branche mono-reseau plus bas : les recopier
        # partiellement produirait deux SAINT differents selon le chemin pris.
        return SAINTPolicySingleHead(
            n_features=OBS_N_FEATURES, d_model=cfg.d_model,
            num_blocks=cfg.num_blocks, heads=cfg.saint_heads,
            n_freq=cfg.saint_n_freq, mlp_dim=cfg.saint_mlp_dim,
            lecture=cfg.saint_lecture, dropout=0.05, ff_mult=2,
            max_len=cfg.lookback, n_actions=N_ACTIONS,
            n_ref=cfg.n_ref).to(device)

    if cfg.architecture == "ensemble":
        policy = PolitiqueEnsemble(
            [_un_membre(a, tp, pp) for a, tp, pp in cfg.membres]).to(device)
        print(f"  • Ensemble de {len(cfg.membres)} reseaux qui votent DANS le "
              f"rollout : {', '.join(a for a, _, _ in cfg.membres)}")
        for (a, _, _), m in zip(cfg.membres, policy.membres):
            print(f"      {a:<10} {sum(q.numel() for q in m.parameters()):>8,} "
                  f"parametres")
    else:
        policy = build_policy(
        device, lookback=cfg.lookback, n_features=OBS_N_FEATURES,
        archi=cfg.architecture, n_ref=cfg.n_ref, num_blocks=cfg.num_blocks,
        d_model_patch=cfg.d_model_patch, mlp_dim=cfg.mlp_dim,
        taille_patch=cfg.taille_patch, pas=cfg.pas_patch,
    ) if cfg.architecture == "patchtst" else SAINTPolicySingleHead(
        n_features=OBS_N_FEATURES,
        d_model=cfg.d_model,
        num_blocks=cfg.num_blocks,
        # head_dim = d_model / heads doit etre un multiple de 8 : voir
        # saint_core._verifie_dim_tete, qui leve a la construction plutot que
        # de laisser CUDA rendre une erreur illisible au premier forward.
        heads=cfg.saint_heads,
        n_freq=cfg.saint_n_freq,
        mlp_dim=cfg.saint_mlp_dim,
        lecture=cfg.saint_lecture,
        dropout=0.05,
        ff_mult=2,
        max_len=cfg.lookback,
        n_actions=N_ACTIONS,
        n_ref=cfg.n_ref,
    ).to(device)

    # ------------------------------------------------------------------
    # LES POIDS DU FOLD PRECEDENT, quand le walk-forward est CHAINE.
    #
    # Sans cela chaque fold repart de zero et le run produit trois modeles
    # qui n'ont vu qu'un tiers de l'histoire chacun. Chaine, c'est UN modele
    # qui traverse les fenetres dans l'ordre du temps : le fold 1 apprend sur
    # les barres les plus anciennes, le 2 reprend ses poids et continue sur
    # une fenetre decalee, le 3 fait de meme.
    #
    # CE QUI NE FUIT PAS. La fenetre d'entrainement du fold k s'arrete avant
    # sa fenetre de validation, qui precede son test : un modele entre au
    # test de son fold sans avoir vu une seule de ses barres. Que le fold 3
    # s'entraine plus tard sur ce qui fut le test du fold 1 ne change rien au
    # chiffre deja mesure du fold 1 — il a ete releve avant.
    #
    # L'OPTIMISEUR, LUI, REPART A NEUF, et c'est voulu : ses moments sont
    # ceux d'une distribution de gradients qui vient de changer de fenetre.
    # Le warmup du critic recommence pour la meme raison.
    # ------------------------------------------------------------------
    if poids_initiaux is not None:
        etat = torch.load(poids_initiaux, map_location=device)
        policy.load_state_dict(etat, strict=True)
        print(f"[{cfg.side.upper()}{suffix}]  • POIDS HERITES de "
              f"{poids_initiaux} — ce fold CONTINUE le precedent, il ne "
              f"repart pas de zero.")
        print(f"    Les epochs a actor gele ne mesurent donc plus le hasard "
              f"mais la politique heritee.")

    # =================================================================
    # LA TETE DE RANG S'ENTRAINE SUR UNE GRILLE, PLUS SUR LA TRAJECTOIRE.
    #
    # LE DIAGNOSTIC QUI L'IMPOSE (2026-09-20, checkpoint d'exec31) :
    #
    #   fenetre          rho(tete, cible)  rho(tete, reel)  rho(cible, reel)
    #   ENTRAINEMENT             -0.0969          -0.0743          +0.9541
    #   VALIDATION               -0.1748          -0.1354          +0.9210
    #
    # LA CIBLE TRANSFERE PARFAITEMENT — +0.95 et +0.92, mieux que le +0.892
    # que le code annoncait — et la tete la classe A L'ENVERS. Sur les DEUX
    # fenetres, entrainement compris : ce n'est donc PAS du surapprentissage,
    # elle aurait alors bien predit la ou elle apprend.
    #
    # LA CAUSE EST UN DECALAGE DE DISTRIBUTION. La perte auxiliaire ne voyait
    # que les etats du TAMPON PPO — environ 4 100 decisions par epoch, celles
    # que la politique visite, filtrees par le veto de tendance et par la
    # capacite restante. `rhoAux` et `sommet`, eux, se mesurent sur une grille
    # UNIFORME. Mesure : la tete ne s'entrainait que sur 51.6 % des occasions
    # de la fenetre d'entrainement et 64.3 % de celles de validation.
    #
    # Une grille horaire dense, tiree independamment de ce que la politique
    # choisit, supprime l'ecart : elle apprend exactement sur le domaine ou on
    # la note, et ou la regle deployee l'interrogera.
    #
    # LE VETO Y EST APPLIQUE AUSSI, pour la meme raison : le systeme deploye
    # n'entre jamais en tendance baissiere, donc entrainer la tete a y classer
    # serait apprendre une question qu'on ne lui posera pas.
    # =================================================================
    # ---------- L'ECHANTILLON DE LA TETE DE CLOTURE ----------
    # Construit UNE FOIS par fold, comme la grille dense, et pour la meme
    # raison : il ne depend pas de la politique, donc le recalculer a
    # chaque epoch ne changerait rien qu'une depense.
    _clot = None
    if getattr(cfg, "pas_cloture_par_epoch", 0) > 0:
        try:
            import cibles_m1 as _CM1
            # LA MEME FABRIQUE REND LES DEUX CIBLES. Le septieme element
            # est `cible_profit` — ce qu'il reste a prendre, en ATR
            # d'entree. Meme echantillon, memes unites, meme ATR d'entree :
            # les deux tetes voient exactement les memes positions, et
            # aucune seconde fabrique ne peut diverger de la premiere.
            _ech = _CM1.echantillon_cloture(
                train_data.df,
                int(cfg.n_echantillon_cloture),
                horizon=int(cfg.horizon_cloture),
                tenue_max=int(cfg.tenue_max_cloture))
            _ci, _cs, _cl, _ct, _cy = _ech[:5]
            _cyp = _ech[6] if len(_ech) > 6 else None
            _cok = (np.isfinite(_cy) & (_ci >= cfg.lookback)
                    & (_ci < train_data.length))
            if _cok.sum() >= 1000:
                _clot = {
                    "idx": _ci[_cok].astype(np.int64),
                    "sens": _cs[_cok].astype(np.float32),
                    "lat": _cl[_cok].astype(np.float32),
                    "tenue": _ct[_cok].astype(np.float32),
                    "y": torch.tensor(_cy[_cok].astype(np.float32),
                                      device=device),
                    "yp": (torch.tensor(_cyp[_cok].astype(np.float32),
                                        device=device)
                           if _cyp is not None else None),
                }
                print(f"  • Tete de cloture : {int(_cok.sum()):,} etats EN "
                      f"POSITION, horizon {cfg.horizon_cloture} min, "
                      f"INDEPENDANTS de la politique")
        except Exception as _e:
            print(f"  • Tete de cloture : echantillon indisponible ({_e})")

    _gr_idx = _gr_y = None
    if getattr(cfg, "pas_grille_rang", 0) > 0:
        _gi = np.arange(cfg.lookback,
                        train_data.length - _borne_syst(cfg) - 2,
                        int(cfg.pas_grille_rang))
        if len(_gi) >= 500:
            _ga, _gv = rendements_du_systeme(train_data.df, _gi, cfg,
                                             indicateur=True)
            _gy = np.stack([_ga, _gv], axis=1).astype(np.float32)
            _mc0 = np.array(cotes_permises(cfg.side), bool)
            _gok = np.isfinite(_gy)[:, _mc0].all(axis=1)
            _vt0 = getattr(cfg, "veto_tendance", "")
            if _vt0 and _vt0 in train_data.df.columns:
                _vv = train_data.df[_vt0].to_numpy(np.float64)[
                    np.maximum(_gi - 1, 0)]
                _gok &= np.isfinite(_vv) & (
                    _vv >= float(getattr(cfg, "veto_tendance_seuil", 0.0)))
            _gr_idx = _gi[_gok]
            _gr_y = torch.tensor(np.nan_to_num(_gy[_gok]), device=device)
            print(f"  • Tete de rang : grille dense de {len(_gr_idx):,} "
                  f"occasions (une toutes les {cfg.pas_grille_rang} barres), "
                  f"INDEPENDANTE de la politique")

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-8)
    # L'OPTIMISEUR DE LA PASSE SUPERVISEE, SEPARE. Voir `lr_rang` pour la
    # mesure qui l'impose. Il porte les MEMES parametres — le tronc doit
    # apprendre des deux objectifs — mais son propre etat Adam et son propre
    # pas.
    # LE MODELE DE SORTIE A SON OPTIMISEUR, SEPARE DE CELUI DES ENTREES.
    #
    # `optimizer_rang` porte le tronc et les tetes d'ouverture ; celui-ci
    # ne porte que `policy.params_sortie()` — le corps de sortie et ses
    # deux tetes. Aucun des deux ne touche les poids de l'autre.
    #
    # IL VIT TOUT LE FOLD. La version precedente reconstruisait
    # l'optimiseur a chaque epoch : l'etat Adam — les moments qui lissent
    # le pas — etait jete toutes les quelques minutes.
    # UN OPTIMISEUR PAR TETE, PLUS LE TRONC ET LE CRITIQUE D'ENTREE — voir
    # `optimiseurs_ppo`. Crees une fois par fold : l'etat Adam survit d'une
    # epoch a l'autre.
    optims_ppo = optimiseurs_ppo(policy, cfg)

    optimizer_rang = optim.Adam(
        policy.parameters(),
        lr=float(getattr(cfg, "lr_rang", cfg.lr)), eps=1e-8)

    # Groupes de paramètres pour diagnostiquer d'où vient le gradient.
    #
    # LE PREFIXE D'ENSEMBLE EST RETIRE AVANT DE TRIER. Dans un ensemble les
    # noms valent `membres.0.actor.*` : tester `startswith("actor.")` rendrait
    # trois listes VIDES, donc un gradient d'acteur affiche a zero. La veille
    # lit precisement ce chiffre pour distinguer une politique gelee d'une
    # politique qui apprend lentement — elle aurait annonce "ACTOR GELE" sur
    # tout un run en train d'apprendre.
    def _sans_membre(n):
        return re.sub(r"^membres\.\d+\.", "", n)

    # `g[actor ...]` DESIGNE LA TETE QUI PORTE LA POLITIQUE, et celle-ci est
    # desormais la tete de BUDGET. La veille lit ce chiffre pour distinguer une
    # politique gelee d'une politique qui apprend lentement ; le laisser
    # pointer sur `actor.` aurait affiche 0.00e+00 a chaque epoch et fait
    # annoncer « ACTOR GELE » sur un run parfaitement sain.
    # IL DESIGNE LA TETE DE RANG, seule tete entrainee depuis que le budget
    # se deduit du classement et que PPO ne met plus rien a jour. Il a
    # pointe successivement `actor.` puis `tete_budget.`, et a chaque fois
    # l'oubli de le deplacer a fait afficher 0.00e+00 sur un run sain.
    # IL DESIGNE LES TROIS TETES ENTRAINEES : achat, vente, cloture.
    #
    # Il a pointe successivement `actor.`, `tete_budget.`, puis `tete_aux.`,
    # et a CHAQUE fois l'oubli de le deplacer a coute quelque chose — un
    # « ACTOR GELE » annonce sur un run sain, puis ce plantage-ci. C'est
    # l'assertion elle-meme qui l'a rattrape les deux dernieres fois : sans
    # elle le diagnostic aurait affiche 0.00e+00 en silence.
    #
    # ON NOMME LES TROIS, pas un prefixe commun : si une tete disparait ou
    # se renomme, l'assertion tombe au demarrage plutot que de laisser un
    # chiffre faux au journal pendant quatre-vingt-dix epochs.
    _tetes_vivantes = ("tete_achat.", "tete_vente.", "tete_cloture.")
    actor_head_params = [p for n, p in policy.named_parameters()
                         if _sans_membre(n).startswith(_tetes_vivantes)]
    _manquantes = [t for t in _tetes_vivantes
                   if not any(_sans_membre(n).startswith(t)
                              for n, _ in policy.named_parameters())]
    assert not _manquantes, (
        f"tetes introuvables : {_manquantes} — le diagnostic de gradient "
        f"designerait des tetes inexistantes")

    # ---- LA TETE DE DIRECTION EST MISE HORS CIRCUIT ----
    #
    # Plus rien ne lit ses sorties : ni le rollout, ni la validation, ni le
    # test, ni le live, ni la perte. Elle ne recevrait donc deja aucun gradient
    # par le graphe d'autograd. On le rend EXPLICITE plutot que de le laisser
    # dependre d'une absence — si un jour quelqu'un relit ces logits, il verra
    # une tete gelee et se posera la question, au lieu de reentrainer en
    # silence une direction qui ne decide rien.
    #
    # ELLE N'EST PAS SUPPRIMEE DU RESEAU, et c'est deliberé : les points de
    # reprise du depot contiennent ses poids, et `torch.load(strict=True)` les
    # refuserait tous. Elle occupe 771 parametres sur ~46 000 et ne coute qu'un
    # produit matriciel par lot.
    _n_gel = 0
    for _n, _pm in policy.named_parameters():
        if _sans_membre(_n).startswith("actor."):
            _pm.requires_grad_(False)
            _n_gel += 1
    print(f"  direction supprimee : {_n_gel} tenseurs de la tete `actor` "
          f"geles, aucune sortie lue")
    critic_head_params = [p for n, p in policy.named_parameters()
                          if _sans_membre(n).startswith("critic.")]
    trunk_params = [
        p for n, p in policy.named_parameters()
        if not _sans_membre(n).startswith(("actor.", "critic."))
    ]

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.05
    )
    # LE RECUIT DE LA PASSE SUPERVISEE, QUI MANQUAIT.
    #
    # PPO a le sien depuis toujours ; `optimizer_rang`, cree le 2026-09-21,
    # n'en avait aucun. Son pas restait donc a 1e-4 pendant 240 pas par epoch
    # x ~90 epochs, soit 21 600 pas au MEME taux.
    #
    # POURQUOI C'EST LE VRAI DEFAUT. Avec Adam, le pas vaut ~`lr` par
    # parametre quelle que soit la norme du gradient — c'est sa
    # normalisation par la variance courante qui le decide, pas l'ecretage.
    # A pas constant, la perte se stabilise donc sur un PLANCHER DE BRUIT
    # fixe par `lr` : elle ne s'installe jamais, elle oscille autour de
    # l'optimum. C'est exactement ce qu'on mesurait — AuxL 2.31 -> 2.16 ->
    # 2.39 — et je l'attribuais a l'ecretage a 100 %, qui n'y est pour
    # presque rien.
    #
    # MEME FORME QUE CELUI DE PPO : recuit cosinus sur la duree du run,
    # plancher a 5 % du pas initial, un pas d'ordonnanceur par epoch.
    scheduler_rang = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_rang, T_max=cfg.epochs,
        eta_min=float(getattr(cfg, "lr_rang", cfg.lr)) * 0.05
    )

    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=(cfg.use_amp and device.type == "cuda")
    )

    def save_checkpoint(state, path):
        # Sidecar portant le même nom: chaque modèle conserve SON scaler.
        np.savez(path.replace('.pth', '_norm.npz'), **stats,
                 features=np.array(FEATURE_COLS), execution_version=2,
                 fee_rate=cfg.fee_rate)
        torch.save(state, path)

    best_path = f"best_{cfg.model_prefix}_{cfg.side}{suffix}.pth"
    best_profit_path = f"bestprofit_{cfg.model_prefix}_{cfg.side}{suffix}.pth"

    if os.path.exists(best_path):
        if not cfg.resume_weights:
            raise ValueError("Poids existants: nouveau model_prefix requis, ou resume_weights=True explicitement.")
        from saint_core import load_model_norm_stats
        norm_path = best_path.replace('.pth', '_norm.npz')
        if not os.path.exists(norm_path):
            raise ValueError('Reprise interdite sans scaler associé au checkpoint')
        previous_stats = load_model_norm_stats(best_path)
        if any(not np.array_equal(previous_stats[k], stats[k]) for k in ('mean','std')):
            raise ValueError('Reprise interdite avec un autre scaler')
        print(f"→ Chargement du modèle existant ({best_path}) pour continuation…")
        norm_path = best_path.replace('.pth', '_norm.npz')
        if not os.path.exists(norm_path):
            raise ValueError("Checkpoint historique sans normalisation associée: utiliser un nouveau model_prefix.")
        checkpoint_stats = np.load(norm_path)
        if any(not np.array_equal(checkpoint_stats[k], stats[k]) for k in ("mean", "std")):
            raise ValueError("Normalisation différente: repartir de zéro avec un nouveau model_prefix.")
        policy.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    # Mode "close" supprimé : pas de modèles gelés à charger.
    policy_long = None
    policy_short = None

    best_val_profit = -1e9
    # Le meilleur CLASSEMENT vu jusqu'ici — un rho, donc dans [-1, 1].
    # Voir plus bas pourquoi ce n'est plus le Sortino qui selectionne.
    best_metric = -1e9
    best_state = None
    _best_epoch = -1        # l'epoch dont les poids partent au fold suivant
    _refus_creux = 0        # candidats ecartes parce que le compte a ete DETRUIT
    best_thresholds = None
    best_decision_spec = None
    # L'AMORCE DU RANG GLISSANT DU ROLLOUT, transmise d'une epoch a l'autre.
    # Vide a la premiere : le rang demarre alors sur ses propres observations.
    pbs_rollout_prec: list = [[], []]
    epochs_no_improve = 0
    # ARRET PRECOCE. L'ancienne regle — 60 % du total, soit 144 epochs —
    # laissait le run se degrader indefiniment. Mesure sur exec10 H1 : sommet
    # a l'epoch 11 (ecart +5.1 pt, PF 1.16), puis chute reguliere jusqu'a
    # -8.2 pt a l'epoch 20 pendant que l'entropie tombait de 1.10 a 0.50.
    # Dix epochs de plus n'ont produit que de la degradation.
    #
    # Sur un jeu H1 de 16 708 barres, la patience se compte en dizaines
    # d'epochs, pas en centaines.
    patience = cfg.patience
    metric_history: List[float] = []

    reward_normalizer = RewardNormalizer()

    # Seuil absolu réalisant la sélectivité visée, recalibré à chaque epoch sur
    # la distribution de max(p_BUY, p_SELL). 0.0 à l'epoch 1 = aucun filtre :
    # inventer une valeur avant d'avoir mesuré la moindre conviction n'aurait
    # aucun sens.
    calib_thr_courant = 0.0

    # Seuil appliqué EN VALIDATION, calibré sur la distribution de conviction de
    # la fenêtre de validation elle-même (celle de l'epoch précédente).
    #
    # Calibrer sur le train et appliquer au val ne marche pas : mesuré à
    # l'epoch 8, le quantile train montait à 0.417 alors que toute la
    # distribution val restait en dessous — 0 trade. La conviction du modèle
    # dépend du régime, donc « trader les q % des meilleurs instants » doit se
    # calibrer sur la fenêtre où l'on trade.
    # UNE BARRE PAR CÔTÉ. Une barre unique sur max(p_BUY, p_SELL) dégénère :
    # un écart de l'ordre du millième entre les deux côtés — du bruit — suffit à
    # verrouiller la sélection sur un seul, et le côté verrouillé change à chaque
    # epoch (mesuré : epoch 4 tout LONG, epoch 7 tout SHORT, zéro trade de
    # l'autre côté dans les deux cas). Calibrer chaque côté sur SA propre
    # distribution rend les deux accessibles quel que soit un biais constant.
    calib_thr_val = [0.0, 0.0]      # [BUY, SELL]

    # Derniere validation COMPLETE, reprise telle quelle les epochs ou elle
    # est sautee. Voir `validation_tous_les`.
    _val_prec = None
    # LE RENDEMENT EN R DE CHAQUE TRADE DE LA DERNIERE VALIDATION MESUREE,
    # repris aux epochs ou elle est sautee. Voir le critere PPO.
    _val_r_prec: List[float] = []
    _val_epoch = 0

    # Somme courante des poids des dernieres epochs, et son compteur.
    somme_poids, n_moyennes = None, 0
    stats_precedentes = None

    # Un tour SUPPLEMENTAIRE, numerote cfg.epochs + 1 : il charge la moyenne
    # des poids, saute la mise a jour PPO, et laisse le reste de la boucle
    # calibrer puis mesurer le modele moyenne. Faire passer la moyenne par le
    # chemin normal evite d'ecrire un second calibrage qui pourrait diverger du
    # premier sans que rien ne le signale.
    for epoch in range(1, cfg.epochs + 2):
        cfg.current_epoch = epoch
        epoch_moyenne = (epoch == cfg.epochs + 1)
        if epoch_moyenne:
            if somme_poids is None:
                break
            ref = policy.state_dict()
            moyenne = {k: (v / n_moyennes).to(dtype=ref[k].dtype)
                       for k, v in somme_poids.items()}
            policy.load_state_dict(moyenne)
            print(f"[{cfg.side.upper()}{suffix}] MOYENNE DES POIDS sur les "
                  f"{n_moyennes} dernieres epochs — le calibrage et la mesure "
                  f"ci-dessous portent sur elle.")

        batch_states = []
        # Le masque EFFECTIVEMENT applique au moment de la decision. Il ne se
        # reconstruit plus a l'update : le veto de TabM depend de la barre, pas
        # de la position, donc un masque refabrique ne serait pas le meme.
        batch_masques = []
        batch_barres = []
        batch_actions = []
        batch_positions = []

        total_reward_epoch = 0.0
        epoch_pnl = []
        epoch_dd = []
        epoch_trades_pnl: List[float] = []
        epoch_trades_side: List[int] = []  # +1 long, -1 short — index aligné avec epoch_trades_pnl

        # LA TAILLE VIENT DE `N_ACTIONS`, ET NON D'UN 3 EN DUR.
        #
        # Elle etait ecrite en dur et a survecu au passage a quatre actions
        # du 2026-09-21. Rien ne l'a signale pendant tout ce temps : tant
        # que la sortie etait debranchee, aucune action 3 n'etait produite
        # et l'index n'etait jamais atteint. Le premier CLOTURER emis a
        # fait tomber le run sur `IndexError: index 3 is out of bounds`.
        #
        # C'est la meme faute que le bloc d'etat a coutee plus haut, et la
        # meme chance : un tableau trop COURT leve, la ou un compteur trop
        # long se serait tu en comptant des zeros.
        action_counts_env = np.zeros(N_ACTIONS, dtype=np.int64)

        # eval() pour TOUTE l'epoch — collecte ET mise à jour.
        #
        # En train(), le dropout (p=0.05) était actif pendant la collecte : les
        # log-probs stockés venaient d'un réseau échantillonné au hasard. Pire,
        # la mise à jour tirait d'autres masques de dropout, donc le ratio PPO
        # exp(new_log - old_log) ne valait pas 1 au premier pas alors qu'il le
        # devrait par construction — le clipping s'appliquait à du bruit.
        #
        # Ne désactiver le dropout que sur la collecte aurait déplacé le biais
        # sans le supprimer : il faut que π_old et π_new soient la MÊME fonction.
        # eval() ne touche pas à autograd, les gradients circulent normalement.
        policy.eval()

        # CHRONOMETRAGE PAR PHASE. Sans lui, on extrapole le cout d'une epoch a
        # partir de micro-bancs d'essai — et un micro-banc lance PENDANT
        # l'entrainement mesure la contention GPU, pas le code. C'est l'erreur
        # qui m'a fait attribuer 11 minutes aux forwards alors qu'ils en valent
        # 2. On mesure donc les phases la ou elles s'executent.
        _t_phase = time.time()
        _chrono = {}

        # --------- collecte expériences (épisodes VECTORISÉS) ---------
        # Les épisodes avancent en lockstep : un seul forward batché par pas au
        # lieu d'un forward batch-1 par épisode. Le profilage donne 6.56 ms pour
        # un forward batch 1 contre 0.028 ms pour env.step() — 99.6% du coût est
        # du lancement de kernels à vide, et un batch 16 coûte le même temps
        # qu'un batch 1. Batcher les épisodes divise donc le temps par ~N.
        # On ne joue que le nombre d'episodes qu'exige la cible de
        # decisions. Les autres environnements existent — les instancier
        # coute des minutes — mais ne sont pas parcourus.
        envs = train_envs[:max(1, min(n_episodes_courant[0], len(train_envs)))]
        n_envs = len(envs)

        states: List[np.ndarray] = []
        infos: List[Dict] = []
        for e in envs:
            s0, i0 = e.reset()
            states.append(s0)
            infos.append(i0)

        ep_buf = [
            {"states": [], "masques": [], "barres": [], "actions": [],
             "rewards": [], "dones": [], "positions": [],
             "dts": [], "lps": [], "vals": []}
            for _ in range(n_envs)
        ]
        last_reason = [None] * n_envs
        active = list(range(n_envs))

        # ============================================================
        # LE TAMPON DE LA POLITIQUE DE SORTIE
        # ============================================================
        #
        # UNE TRANSITION PAR BARRE TENUE, et c'est ce qui rend ce probleme
        # traitable la ou l'entree ne l'est pas :
        #
        #     decision d'ENTREE   ~55 occasions independantes par fenetre
        #     decision de SORTIE  une par barre de chaque trade
        #
        # LA RECOMPENSE TELESCOPE, et c'est le point du schema :
        #
        #     a chaque barre tenue   latent_t - latent_{t-1} - loyer
        #     a la fermeture         - loyer, et fin
        #
        # La somme sur un trade vaut exactement `latent_final -
        # latent_entree - loyer x duree`. Comme `latent_entree` porte deja
        # le spread paye a l'ouverture, ce total EST le net realise en ATR,
        # moins le loyer du temps. Aucun terme de friction a ajouter a la
        # main : il est deja dans le latent, et le cout de SORTIE sera paye
        # de toute facon — il s'annule donc entre « fermer » et « tenir »,
        # meme raisonnement que `cibles_m1.cible_cloture`.
        #
        # UNE APPROXIMATION ASSUMEE. Le latent est lu a la CLOTURE de la
        # barre, l'environnement solde a l'OUVERTURE de la suivante : il
        # manque un pas au dernier terme. C'est la meme convention que
        # `cibles_m1.echantillon_cloture` documente pour le prix d'entree,
        # et du meme ordre de grandeur.
        sortie_buf = [[] for _ in range(n_envs)]
        # Le latent de la DERNIERE DECISION, par environnement. NaN veut
        # dire « pas de position en cours » — on ne peut alors pas calculer
        # de difference.
        lat_prec = np.full(n_envs, np.nan, dtype=np.float64)
        # Barres ecoulees depuis la derniere decision de sortie. La
        # politique n'est interrogee que tous les `pas_decision_sortie` ;
        # entre deux, la position tient et la recompense s'accumule.
        depuis_dec = np.zeros(n_envs, dtype=np.int64)
        _loyer = float(getattr(cfg, "loyer_temps_atr", 0.0))
        _derive = float(getattr(cfg, "derive_atr_barre", 0.0))
        _loyer_z_mult = float(getattr(cfg, "loyer_zombie_mult", 15.0))
        _marge_z = float(getattr(cfg, "marge_zombie_atr", 0.0))
        _pas_s = max(1, int(getattr(cfg, "pas_decision_sortie", 1)))

        # Sélectivité visée cette epoch, et seuil absolu qui la réalise.
        # Le seuil vient du quantile mesuré à l'epoch PRÉCÉDENTE : à l'epoch 1 il
        # n'existe pas encore, on laisse alors passer toutes les décisions plutôt
        # que d'inventer une valeur.
        selectivite = selectivite_for_epoch(epoch)
        conf_thr = calib_thr_courant
        pbs_epoch: List[float] = []   # max(p_BUY, p_SELL) sur les états flat
        pbs_rollout: List[List[float]] = [[], []]   # BUY / SELL, pour l'amorce

        # ============================================================
        # LE ROLLOUT DECIDE PAR RANG GLISSANT, COMME LE DEPLOIEMENT.
        #
        # LE DEFAUT QUE CELA SUPPRIME, ET IL ETAIT DEJA DOCUMENTE ICI. La
        # barre du rollout valait `calib_thr_courant` : un NIVEAU calibre sur
        # l'epoch precedente et applique tel quel a la suivante. C'est
        # exactement ce que la docstring de `SeuilRang` decrit comme le
        # probleme qu'il resout :
        #
        #   « L'etendue des convictions est passee de 0.0035 a 0.0914 entre
        #     les epochs 6 et 12 — vingt-six fois plus — pendant que le seuil
        #     herite restait autour de 0.24. La barre s'est retrouvee tres
        #     haut dans la distribution courante, et le nombre de trades s'est
        #     effondre de 1 461 a 20. »
        #
        # `SeuilRang` a ete ecrit pour ca — et il n'etait branche qu'en
        # validation, au test et en live. En supprimant la direction, il a
        # fallu redonner une barre au rollout, et c'est le NIVEAU herite qui a
        # ete remis : le defaut que ce module existe pour empecher.
        #
        # CE QUE CELA COUTAIT, mesure le 2026-09-21. A l'epoch 2 la barre de
        # l'epoch 1 rejette presque tout, l'agent cesse d'entrer, reste a plat,
        # et se fait interroger a chaque barre : le compte de decisions passe
        # de 2 103 a plus de 20 000. Le run exec61 est mort par epuisement
        # memoire ; exec64 n'a survecu que parce que la soupape de collecte a
        # tire.
        #
        # ON PARTAGEAIT DEJA `decide_avec_barres` ENTRE LES DEUX COTES, mais
        # pas la facon de CALCULER la barre. Une regle partagee dont le seuil
        # se calcule autrement des deux cotes n'est pas une regle partagee.
        #
        # UN FLUX PAR ENVIRONNEMENT, jamais partage : la fenetre glissante est
        # l'historique d'UNE suite de decisions, et les melanger ferait juger
        # une occasion par les convictions d'un autre episode.
        #
        # L'AMORCE VIENT DE L'EPOCH PRECEDENTE quand elle existe. A la
        # premiere, elle est vide : `SeuilRang` rend alors un seuil infini
        # jusqu'a ses cinquante premieres observations, donc les cinquante
        # premieres decisions de chaque flux sont des attentes. Sur plusieurs
        # milliers, c'est un amorcage, pas un biais — et c'est CAUSAL, comme
        # en production ou le bot ne connait que ce qu'il a deja vu.
        # ============================================================
        _n_cotes_tr = max(sum(cotes_permises(cfg.side)), 1)
        _frac_tr = max(selectivite / _n_cotes_tr, 1e-4)
        _spec_tr = rolling_decision_spec(
            _frac_tr, cfg.rang_fenetre, pbs_rollout_prec, cfg.side)
        train_decisions = {k: EntryDecisionPolicy(_spec_tr)
                           for k in range(len(envs))}

        # Probabilité d'ouverture forcée (curriculum), constante sur l'epoch
        if epoch <= 15:
            force_prob = max(0.15, 0.92 - epoch * 0.05)
        elif epoch <= 35:
            force_prob = 0.25
        else:
            force_prob = 0.05

        if not cfg.legacy_off_policy_curriculum:
            force_prob = 0.0
            # LE SEUIL N'EST PLUS MIS A ZERO. Il l'etait parce que la
            # selectivite venait alors du p(ATTENDRE) de l'acteur : la barre
            # aurait filtre DEUX fois. La direction supprimee, plus rien ne
            # retient l'entree — sans barre, le rollout entrerait a CHAQUE
            # occasion pendant que la validation n'en retient que 5 %, et on
            # aurait reconstruit le desaccord qu'on vient de supprimer, dans
            # l'autre sens. `calib_thr_courant` vaut 0 a l'epoch 1, ou la
            # distribution des scores est encore plate : le premier passage
            # reste donc sans filtre, par construction et non par exception.

        # ============ ROLLOUT SEMI-MDP ============
        # Une DÉCISION n'existe que lorsque l'agent est flat : en position le
        # masque ne laisse que HOLD, donc appeler le réseau y est inutile — son
        # résultat est jeté. Avec des stops à 10×ATR une position dure ~1140
        # bougies, donc 98.5% des forwards ne servaient à rien.
        #
        # Ici on ne sollicite la policy QUE sur les envs flat. Les envs en
        # position avancent d'un simple env.step(HOLD) à 0.028 ms, et leurs
        # récompenses s'accumulent (actualisées) dans la décision qui a ouvert
        # la position. Chaque transition devient donc :
        #     (état de décision, action, R = Σ γ^i r_i, Δt = durée)
        # ce qui est la formulation semi-MDP standard pour des actions de durée
        # variable. Le GAE en tient compte via γ^Δt.
        # UNE DECISION PAR EMPLACEMENT, et non plus une par environnement.
        #
        # Une decision PPO couvre la vie d'une position : elle nait quand
        # l'agent peut entrer, et se ferme quand SA position se ferme. Avec
        # plusieurs positions ouvertes il y a donc plusieurs decisions en
        # cours dans le meme environnement, chacune accumulant la recompense
        # de SON emplacement — `info["r_slots"]`, calculee sur la variation
        # d'equity de cette position seule.
        #
        # A K=1 le dictionnaire n'a jamais plus d'une entree et la sequence
        # est celle d'avant : meme etat de decision, meme recompense
        # accumulee, meme drapeau de fin.
        # SANS CETTE REMISE A ZERO, une position fermee puis rouverte
        # verrait sa premiere difference de latent calculee contre le
        # latent de la position PRECEDENTE — une recompense fabriquee de
        # toutes pieces, et qu'aucune erreur ne signalerait.
        def _oublie_sortie(k):
            lat_prec[k] = np.nan

        pending: List[Dict[int, Dict]] = [dict() for _ in range(n_envs)]
        # Decision prise a cette barre, pas encore rattachee : on ne connait
        # son emplacement qu'APRES le pas, puisque c'est l'environnement qui
        # l'attribue.
        en_attente: Dict[int, Dict] = {}
        # Repartition des paliers de budget retenus sur l'epoch, et combien
        # de fois le palier choisi a lui-meme refuse l'entree.
        # LA TABLE DES PALIERS, pour classer la part deduite du rang
        # dans le compteur d'affichage. `searchsorted` sur un tuple
        # croissant rend l'indice du palier exact.
        # CE QUE LA REGLE D'ENTREE A REELLEMENT DECIDE, par categorie.
        #
        # `H` etait l'entropie des logits de l'acteur. L'acteur ne decide plus
        # rien, donc ce nombre ne mesurerait plus que la derive d'une tete
        # morte. On garde le champ — la veille le lit, et son plafond
        # ln(cotes+1) reste exact — mais on lui donne un contenu qui existe
        # encore : l'entropie de la REPARTITION DES DECISIONS d'entree.
        #
        # Elle dit ce que `H` disait : a 0, la regle s'est figee (elle
        # n'entre jamais, ou elle entre toujours) ; au plafond, elle est a
        # pile ou face. La difference est qu'elle porte maintenant sur la
        # regle qu'on DEPLOIE.
        # TROIS, ET PAS `N_ACTIONS` : NE PAS « CORRIGER » CETTE LIGNE.
        #
        # Elle compte les decisions d'ENTREE — acheter, vendre, attendre —
        # et c'est d'elle que sort `H`, dont le plafond affiche par META
        # vaut ln(cotes permis + 1). CLOTURER n'est jamais une entree :
        # `build_mask_from_pos_scalar` le ferme a plat, et l'action 3 est
        # posee directement dans `actions_env`, sans passer par `a`.
        #
        # L'allonger a quatre ajouterait une case toujours nulle, qui ne
        # changerait pas `H` mais ferait mentir le plafond — et
        # `test_direction_supprimee` verifie exactement ce plafond.
        entrees_epoch = [0, 0, 0]
        # L'AUDIT DE COLLECTE. Ce qu'il compte est bien vivant ; seuls
        # son nom et un de ses champs dataient de PPO.
        #
        #   decisions          barres ou le modele a pu ENTRER. C'est ce
        #                      chiffre qui a revele la position immortelle
        #                      du 2026-09-21 : 12 sur 12 episodes.
        #   forced_actions     ouvertures imposees par le curriculum.
        #   remapped_actions   entrees que le solde a refusees.
        #   cote_interdit      une VENTE passee en long-only, ou l'inverse.
        #                      Ce champ s'appelait `max_logprob_error` : il
        #                      verifiait que la log-probabilite stockee
        #                      correspondait a la loi qui avait tire
        #                      l'action. Plus rien n'est tire — la decision
        #                      est deterministe — donc il a ete rebranche
        #                      sur ce qui peut encore casser, SANS etre
        #                      renomme. Un champ qui mesure autre chose que
        #                      son nom est pire qu'un champ absent.
        # `ouvertures` : les trades REELLEMENT ouverts par la collecte.
        # C'est lui que le regulateur d'episodes vise — voir `cible_trades`.
        sampling_audit = {"decisions": 0, "ouvertures": 0,
                          "forced_actions": 0,
                          "remapped_actions": 0, "cote_interdit": 0.0,
                          "episodes_joues": len(envs)}
        _p_attente = float(getattr(cfg, "garde_attentes", 1.0))

        def _verse(k: int, p: Dict, done_flag: bool) -> None:
            """Verse une décision terminée au buffer de l'env k."""
            buf = ep_buf[k]
            buf["positions"].append(0)          # une décision est toujours flat
            buf["states"].append(p["state"])
            buf["masques"].append(p["masque"])
            buf["barres"].append(p["barre"])
            buf["actions"].append(p["action"])
            buf["rewards"].append(p["R"])
            buf["dones"].append(done_flag)
            buf["dts"].append(max(p["dt"], 1))
            # CE QUE PPO RELIRA : la probabilite de l'action sous la loi
            # qui l'a tiree, et la valeur que le critique donnait a l'etat.
            buf["lps"].append(float(p.get("lp", 0.0)))
            buf["vals"].append(float(p.get("v", 0.0)))

        def _cloture(k: int, slot: int, done_flag: bool) -> None:
            p = pending[k].pop(slot, None)
            if p is not None:
                _verse(k, p, done_flag)

        _collecte_tronquee = False
        while active:
            # LA SOUPAPE. Voir `plafond_collecte` pour ce que son absence a
            # coute. On s'arrete entre deux pas, jamais au milieu d'un, pour
            # que les decisions en attente soient versees normalement par la
            # cloture d'episode qui suit.
            _plaf_col = int(getattr(cfg, "plafond_collecte", 0) or 0)
            if _plaf_col and sampling_audit["decisions"] >= _plaf_col:
                _collecte_tronquee = True
                break
            # 1) Un environnement decide des qu'il lui reste un emplacement
            #    libre — c'est la definition d'une decision. A K=1 cela revient
            #    exactement a "etre plat".
            deciding = [k for k in active if envs[k].peut_decider()]
            actions_env = {k: 2 for k in active}

            # ---- LA SORTIE, ET ELLE MANQUAIT ENTIEREMENT ----
            #
            # `deciding` ne contient que les envs qui peuvent ENTRER. A une
            # position a la fois, un env en position n'y figure jamais :
            # le reseau ne le voyait plus, aucune action 3 n'etait produite,
            # et la position vivait jusqu'a la fin de l'episode. Voir
            # `demande_cloture` pour la mesure.
            _en_pos = [k for k in active
                       if k not in deciding and envs[k].n_positions > 0]
            if _en_pos:
                # LA SORTIE EST DECIDEE PAR LA POLITIQUE, plus par deux
                # seuils ecrits a la main. Voir `decide_sortie` pour les
                # quatre calibrations successives qui ont echoue.
                # QUI DECIDE A CETTE BARRE — par `cadence_sortie`, la meme
                # regle qu'en validation, au test et dans le critere.
                _a_decider, _ecoule = cadence_sortie(
                    _en_pos, ~np.isfinite(lat_prec), depuis_dec, _pas_s)
                if _a_decider:
                    _ep = [states[k] for k in _a_decider]
                    _sa, _slp, _sv = decide_sortie(
                        policy, _ep, device, N_BASE_FEATURES, explore=True)
                    _pin = entree_sortie(_ep, N_BASE_FEATURES)
                    for _bi, _k in enumerate(_a_decider):
                        _lat = float(_pin[_bi, 0])
                        # LA RECOMPENSE DE LA DECISION PRECEDENTE se solde
                        # ICI : elle couvre les `depuis_dec` barres ecoulees
                        # depuis elle. Le loyer ET la derive se comptent
                        # PAR BARRE, donc multiplies par cette duree.
                        if sortie_buf[_k] and np.isfinite(lat_prec[_k]):
                            _dt = float(max(_ecoule[_k], 1))
                            # Loyer normal si en gain ; x loyer_zombie_mult
                            # si en perte → la tete apprend a couper les
                            # zombies sans stop au temps et sans toucher
                            # aux gagnants.
                            #
                            # LE ZOMBIE SE COMPTE DEPUIS LE POINT MORT, PLUS
                            # DEPUIS ZERO — 2026-09-25, accord du proprietaire.
                            #
                            # A zero, le cout d'entree (0.389 ATR median)
                            # mettait 66 % des positions EN PERTE avant que
                            # le marche ait bouge : loyer x15 des la
                            # premiere barre, et la politique a appris a
                            # tout fermer a la premiere minute — tenue
                            # mediane 1, quatre epochs sur quatre. Fermer ne
                            # rendait pas le spread, deja paye ; cela le
                            # figeait. Est zombie desormais la position que
                            # le MARCHE a mise en perte, au-dela du ticket.
                            # ET 2 ATR PLUS BAS : voir `marge_zombie_atr`.
                            _mort = -float(envs[_k].cout_entree_atr)
                            _loyer_eff = (
                                _loyer * _loyer_z_mult
                                if _lat < _mort - _marge_z
                                else _loyer)
                            # LA DERIVE SE RETIRE DANS LE SENS DE LA
                            # POSITION. Elle est la pour que le long ne
                            # soit pas paye a tenir un BTC qui monte. Le
                            # short, lui, PAIE deja cette derive dans son
                            # latent : la lui retirer encore la compterait
                            # deux fois, et le pousserait a fermer tout
                            # short pour une raison qui n'a rien a voir
                            # avec son timing. Signee, elle neutralise le
                            # beta des deux cotes.
                            _sens_k = float(_pin[_bi, COL_SENS_SORTIE])
                            sortie_buf[_k][-1]["r"] = (
                                _lat - float(lat_prec[_k])
                                - (_loyer_eff + _sens_k * _derive) * _dt)
                        _ferme = int(_sa[_bi]) == FERMER
                        sortie_buf[_k].append({
                            # L'OBSERVATION ENTIERE : la tete de sortie lit
                            # le tronc, donc toutes les features.
                            "o": _ep[_bi],
                            "p": _pin[_bi].copy(), "a": int(_sa[_bi]),
                            "lp": float(_slp[_bi]), "v": float(_sv[_bi]),
                            # A la fermeture il ne reste rien a accumuler :
                            # le gain l'a deja ete, decision apres decision.
                            "r": 0.0,
                            "done": bool(_ferme),
                        })
                        lat_prec[_k] = np.nan if _ferme else _lat
                        if _ferme:
                            actions_env[_k] = 3

            if deciding:
                # ---- L'ENTREE EST UNE DECISION PPO — 2026-09-25 ----
                #
                # Plus de classement, plus de barre de selectivite, plus de
                # quantile : `tete_achat` et `tete_vente` sortent les logits
                # d'ACHETER et de VENDRE, celui d'ATTENDRE vaut zero, et
                # l'action est TIREE de cette loi. PPO l'entraine sur ce que
                # le trade ouvert a rapporte, sortie comprise.
                masks_np = np.repeat(MASK_FLAT[None, :], len(deciding), axis=0)
                if votant is not None:
                    for bi, k in enumerate(deciding):
                        pa, pv = votant.veto(envs[k].idx - 1)
                        if not pa:
                            masks_np[bi, 0] = False
                        if not pv:
                            masks_np[bi, 1] = False
                _m3, _sans = masque_entree(masks_np, [envs[k] for k in deciding])
                _ea, _elp, _ev = decide_entree(
                    policy, [states[k] for k in deciding], _m3, device,
                    explore=True)
                # `remapped_actions` COMPTE DESORMAIS LES DECISIONS OU LE
                # SOLDE INTERDISAIT D'OUVRIR — masquees avant le tirage.
                sampling_audit["remapped_actions"] += int(_sans.sum())
                for bi, k in enumerate(deciding):
                    a = int(_ea[bi])
                    sampling_audit["decisions"] += 1
                    entrees_epoch[a] += 1
                    if a != ATTENDRE and not cotes_permises(cfg.side)[a]:
                        sampling_audit["cote_interdit"] = 1.0
                    en_attente[k] = {
                        "state": states[k],
                        "masque": _m3[bi].copy(),
                        "barre": envs[k].idx - 1,
                        "action": a,
                        "lp": float(_elp[bi]),
                        "v": float(_ev[bi]),
                        "R": 0.0,
                        "dt": 0,
                    }
                    actions_env[k] = a

            # 3) Un pas pour tous les envs actifs.
            still_active = []
            for k in active:
                env_k = envs[k]
                env_action = actions_env[k]
                # L'ECHELLE N'EST PLUS REMISE A 1.0 ICI. Elle etait imposee
                # a chaque pas, ce qui rendait la tete de taille sans effet :
                # la decision la posait juste avant, cette ligne l'effacait
                # juste apres. Elle est desormais fixee au moment de la
                # decision et ne sert qu'a l'ouverture, donc la conserver
                # entre deux decisions est sans consequence.
                action_counts_env[env_action] += 1

                ns, reward, done, _, info = env_k.step(env_action)
                total_reward_epoch += reward
                # NORMALISATION DE RECOMPENSE RETIREE.
                #
                # Elle divisait par un ecart-type glissant, ce qui n'avait de
                # sens que tant que la recompense etait libellee en dollars et
                # donc d'echelle arbitraire. Elle est maintenant en unites de
                # RISQUE (un stop = -1, la cible = +2) : bornee, comparable d'un
                # symbole a l'autre, et deja a la bonne echelle.
                #
                # La conserver nuisait deux fois. D'abord son initialisation est
                # fausse : _var part a 1.0 puis tombe a 0.0 des le premier
                # echantillon, donc std = 1e-8 et les premieres recompenses sont
                # multipliees par 1e8. Ensuite, sur une recompense creuse (99 %
                # des pas a zero), diviser par l'ecart-type revient a multiplier
                # les rares evenements par 1/sqrt(densite) — une amplification
                # qui n'apporte aucune information et deplace la cible du
                # critique a chaque epoch.
                reward_normalizer.update(reward)   # garde la statistique pour le log

                # CHAQUE DECISION ACCUMULE LA RECOMPENSE DE SON EMPLACEMENT.
                # Lui donner la recompense globale la crediterait de ce que
                # les autres positions ont produit. A K=1 les deux coincident
                # exactement — `r_slots[0]` vaut `reward`, verifie par
                # `test_concurrence.py`.
                rs = info.get("r_slots")
                for slot, p in pending[k].items():
                    r_p = float(rs[slot]) if rs is not None else reward
                    p["R"] += (cfg.gamma ** p["dt"]) * r_p
                    p["dt"] += 1

                # La decision prise avant ce pas se rattache maintenant a
                # l'emplacement que l'environnement lui a donne. Sans
                # emplacement — action d'attente, ou ouverture refusee faute
                # de taille — elle ne dure qu'une barre et ne rapporte rien :
                # la crediter du mouvement des positions ouvertes a cote lui
                # ferait apprendre leur resultat.
                nouveau = en_attente.pop(k, None)
                j_ouvert = int(info.get("slot_ouvert", -1))
                if nouveau is not None and j_ouvert >= 0:
                    sampling_audit["ouvertures"] += 1
                # UNE ATTENTE SUR VINGT GARDE SON ETAT. Voir `garde_attentes`.
                _garde = (nouveau is not None and j_ouvert < 0
                          and np.random.rand() < _p_attente)
                if nouveau is not None:
                    nouveau["dt"] = 1
                    if j_ouvert >= 0 and rs is not None:
                        nouveau["R"] += float(rs[j_ouvert])
                        pending[k][j_ouvert] = nouveau
                    elif j_ouvert >= 0:
                        nouveau["R"] += reward
                        pending[k][j_ouvert] = nouveau

                states[k] = ns
                infos[k] = info

                if done:
                    last_reason[k] = info.get("done_reason")
                    for slot in list(pending[k]):
                        _cloture(k, slot, True)
                    if _garde:
                        _verse(k, nouveau, True)
                else:
                    # Les emplacements fermes a cette barre terminent leur
                    # decision. La version d'avant le faisait en tete de la
                    # boucle suivante ; le faire ici ne change ni le contenu
                    # ni le drapeau, et evite de relire `infos`.
                    for slot in info.get("slots_fermes", []):
                        _cloture(k, int(slot), False)
                    if _garde:
                        _verse(k, nouveau, False)
                    still_active.append(k)

            active = still_active

        # --------- clôture des épisodes ---------
        for k, env_k in enumerate(envs):
            info_k = infos[k]
            epoch_dd.append(info_k["drawdown"])

            final_close = env_k.data.close[env_k.idx - 1]
            latent = 0.0
            if env_k.position != 0 and env_k.current_size > 0 and env_k.entry_price > 0:
                latent = (
                    env_k.position *
                    (final_close - env_k.entry_price) *
                    env_k.current_size
                )
            final_equity = env_k.capital + latent
            epoch_pnl.append(final_equity - cfg.initial_capital)
            epoch_trades_pnl.extend(env_k.trades_pnl)
            epoch_trades_side.extend(env_k.trades_side)

            # Trade-by-trade CSV (TRAIN)
            with open(trades_csv_path, "a", newline="", encoding="utf-8") as _ft:
                _w = _csv.DictWriter(_ft, fieldnames=_trades_fields)
                for tm in env_k.trades_meta:
                    _w.writerow({
                        "epoch": epoch, "phase": "train", "episode": k + 1,
                        "entry_idx": tm["entry_idx"], "exit_idx": tm["exit_idx"],
                        "side": tm["side"],
                        "entry_price": round(tm["entry_price"], 4),
                        "exit_price": round(tm["exit_price"], 4),
                        "pnl": round(tm["pnl"], 4),
                        "hit_sl": int(tm["hit_sl"]), "hit_tp": int(tm["hit_tp"]),
                        "hold_bars": tm["hold_bars"],
                    })

            # Toutes les fins sont terminales et les positions sont liquidées.
            last_value = 0.0

            # LE GAE A ETE RETIRE LE 2026-09-21.
            #
            # `compute_gae_semi_mdp` produisait `adv` et `ret`, verses dans
            # `batch_adv` et `batch_returns`, convertis en tenseurs
            # `advantages` et `returns`... et jamais relus. Leur seul
            # consommateur etait la mise a jour PPO, retiree le matin meme.
            # Verifie par recherche : aucune lecture apres la ligne qui les
            # normalise.
            #
            # CE QUE CA ENTRAINAIT PLUS HAUT. `values` venait de `critic`,
            # donc de `self.mlp` — 8 472 parametres, 15.9 % du reseau,
            # traverses a CHAQUE decision pour alimenter trois tableaux que
            # personne n'ouvrait.
            buf = ep_buf[k]

            batch_states.extend(buf["states"])
            batch_masques.extend(buf["masques"])
            batch_barres.extend(buf["barres"])
            batch_actions.extend(buf["actions"])
            batch_positions.extend(buf["positions"])

        # --------- sous-échantillonnage des états SANS décision ---------
        # Avec un stop à 10×ATR l'agent tient ~300 bougies, donc ~99% des états
        # collectés sont "en position" : masqués à HOLD, ils ne portent aucun
        # gradient d'actor. On collectait 128 000 états pour ~120 décisions.
        # Les avantages GAE sont déjà calculés sur les trajectoires COMPLÈTES,
        # donc en retirer une partie ici ne fausse rien : on garde tous les
        # états flat (le signal de l'actor) et un échantillon des autres, assez
        # pour que le critique reste calibré.
        _pos_arr = np.asarray(batch_positions)
        _flat_mask = _pos_arr == 0

        # Plafond sur les DECISIONS (etats flat). Voir cfg.max_decisions_per_epoch :
        # sans lui, leur nombre croit avec la selectivite jusqu'a rendre une
        # epoch interminable, et fait varier d'un facteur 17 le nombre de pas de
        # gradient entre le debut et la fin du run.
        _flat_idx = np.flatnonzero(_flat_mask)
        _dec_collectees = len(_flat_idx)
        if 0 < cfg.max_decisions_per_epoch < _dec_collectees:
            _tire = np.random.choice(_flat_idx, cfg.max_decisions_per_epoch,
                                     replace=False)
            _flat_retenu = np.zeros(len(_pos_arr), bool)
            _flat_retenu[_tire] = True
        else:
            _flat_retenu = _flat_mask.copy()

        _keep = _flat_retenu.copy()
        if cfg.in_position_keep_frac < 1.0:
            _tirage = np.random.rand(len(_pos_arr)) < cfg.in_position_keep_frac
            _keep |= (~_flat_mask) & _tirage
        else:
            _keep[:] = True
        if _keep.sum() < cfg.batch_size:      # garde-fou : jamais moins d'un batch
            _keep[:] = True

        # ---- LE LOT ENTIER EST PLAFONNE, PAS SEULEMENT SA PART PLATE ----
        #
        # `max_decisions_per_epoch` ne borne que les etats FLAT. A cote,
        # `in_position_keep_frac` en rajoute 10 % des etats EN POSITION — et
        # ces 10 % grandissent avec le total, donc le plafond ne plafonnait
        # qu'une moitie.
        #
        # MESURE, exec51 epoch 2 : 47 660 transitions collectees. La part
        # plate est ramenee a 6 000, mais 10 % des 47 660 en ajoutent ~4 700 :
        # lot reel ~10 700, mise a jour en 143 s contre 13 s pour 1 753.
        # L'epoch passe de 3 a 6 minutes, et cela empire a chaque epoch
        # puisque la selectivite se resserre et gonfle les attentes.
        #
        # ON TIRE UNIFORMEMENT DANS CE QUI A ETE RETENU, donc sans changer la
        # proportion entre etats plats et etats en position : on reduit le
        # VOLUME, pas la composition du signal.
        _budget_lot = int(getattr(cfg, "max_transitions_ppo", 0) or 0)
        if 0 < _budget_lot < int(_keep.sum()):
            _idx_keep = np.flatnonzero(_keep)
            _garde = np.random.choice(_idx_keep, _budget_lot, replace=False)
            _keep = np.zeros(len(_pos_arr), bool)
            _keep[_garde] = True
        # Le nombre de transitions REELLEMENT PRODUITES par la collecte,
        # fige avant le sous-echantillonnage : c'est lui qui dimensionne
        # le budget d'episodes de l'epoch suivante, pas la taille du lot
        # PPO qui suit, ni le nombre de consultations du reseau.
        _n_transitions = len(batch_actions)
        _sel = np.flatnonzero(_keep)

        # LE DIAGNOSTIC DES AVANTAGES A ETE RETIRE LE 2026-09-21.
        #
        # Il demandait si la normalisation melangeait des echelles
        # incompatibles — une decision de trade accumule sa recompense sur
        # des centaines de bougies, une attente sur une seule. La question
        # etait juste, et elle n'a plus d'objet : sans PPO il n'y a plus
        # d'avantages, plus de normalisation, et plus de gradient de
        # politique dont l'ecrasement pourrait inquieter.

        batch_states = [batch_states[i] for i in _sel]
        batch_masques = [batch_masques[i] for i in _sel]
        batch_barres = [batch_barres[i] for i in _sel]
        batch_actions = [batch_actions[i] for i in _sel]
        batch_positions = [batch_positions[i] for i in _sel]

        # --------- tenseurs batch ---------
        states_np = np.stack(batch_states, axis=0)
        states = torch.tensor(states_np, dtype=torch.float32, device=device)
        # Les masques qui ont AGI, alignes sur les etats. L'update les reprend
        # tels quels au lieu de les refabriquer depuis la position : voir le
        # commentaire au point d'usage.
        masques = (torch.tensor(np.stack(batch_masques, axis=0),
                                dtype=torch.bool, device=device)
                   if batch_masques else None)

        # LES ETIQUETTES DE LA TETE AUXILIAIRE ONT ETE SUPPRIMEES ICI.
        #
        # Ce bloc calculait `cibles_aux` a chaque epoch — un appel a
        # `cibles.rendements` sur les ~3 000 barres decidees — et PERSONNE
        # NE LE LISAIT. Le seul consommateur etait la mise a jour PPO,
        # retiree le 2026-09-21 ; la variable lui a survecu.
        #
        # DEUX RAISONS DE L'OTER PLUTOT QUE DE LA REBRANCHER. La tete
        # auxiliaire n'existe plus : `tete_aux` a ete remplacee par
        # `tete_achat` et `tete_vente`, qui apprennent sur la grille dense,
        # sur leur propre echantillon, plus haut dans cette fonction. Et le
        # calcul se faisait sur l'ancienne geometrie au stop — celle dont
        # 100 % des occasions finissent a -1 R — donc meme rebranche il
        # n'aurait rien appris a personne.

        # --------- banque de reference (intersample attention) ---------
        # Constituee UNE fois, a partir d'observations reelles de la fenetre de
        # TRAIN — donc aucune fuite : en validation et en test, le modele
        # consulte des situations anterieures a la periode evaluee, au meme
        # titre que ses poids.
        #
        # On la tire des etats deja collectes plutot que de reconstruire des
        # observations a la main : c'est la garantie qu'elles sont baties par le
        # meme chemin de code que celles vues en production.
        if policy.memoire is not None and float(policy.memoire.bank_pret) == 0.0:
            k = min(cfg.n_ref, states.shape[0])
            idx = torch.randperm(states.shape[0], device=device)[:k]
            policy.definit_banque(states[idx])
            print(f"  [MEMOIRE] banque de {k} references figee "
                  f"(fenetre train, epoch {epoch})")

        actions = torch.tensor(batch_actions, dtype=torch.long, device=device)
        positions = torch.tensor(batch_positions, dtype=torch.long, device=device)

        assert states.size(0) == actions.size(0) == positions.size(0)

        # LES AVANTAGES ET LEUR NORMALISATION SONT PARTIS AVEC LE GAE.
        #
        # La normalisation ne portait que sur les etats FLAT, pour que les
        # 97.5 % d'etats en position — masques a HOLD, donc sans gradient
        # de politique — ne dictent pas la calibration des 2.5 % qui
        # decident. Le raisonnement etait juste ; il n'a plus d'objet sans
        # PPO pour consommer le resultat.
        epoch_actor_loss = []
        epoch_critic_loss = []
        epoch_aux_loss = []
        epoch_entropy = []
        epoch_entropy_flat = []
        # Entropie de la TETE DE TAILLE, tenue a part et hors des tuples
        # de stats reportees : elle n'a pas a decaler des indices lus par
        # position ailleurs. Son plafond est ln(4) = 1.386. Si elle tombe
        # vers 0, le modele a fige sa taille — c'est le symptome a
        # surveiller en premier.
        epoch_grad_norm = []
        epoch_flat_frac = []
        g_actor_hist = []
        g_critic_hist = []
        g_trunk_hist = []
        logratio_hist = []
        clipfrac_hist = []
        epoch_kl = []

        n_samples = states.size(0)
        idx = np.arange(n_samples)

        _chrono["collecte"] = time.time() - _t_phase; _t_phase = time.time()
        # LE CHRONO S'ANNONCE AU FIL DE L'EPOCH, PLUS SEULEMENT A LA FIN.
        #
        # Il n'etait ecrit que dans la ligne META, donc a la toute fin. Quand
        # une epoch est passee de 2 a plus de 15 minutes, il n'y avait RIEN a
        # lire pendant ces quinze minutes : ni la phase en cours, ni le nombre
        # d'episodes joues, ni un debit. J'ai propose deux explications, les
        # deux fausses, faute de mesure — exactement le travers que ce depot
        # documente partout ailleurs.
        #
        # Une ligne par phase longue coute un `print` et supprime la question.
        if _collecte_tronquee:
            print(f"  {_col('SOUPAPE', _C.RED + _C.BOLD)}  collecte "
                  f"interrompue a {sampling_audit['decisions']} decisions "
                  f"(plafond {cfg.plafond_collecte}). Le modele n'entre "
                  f"presque plus, donc il reste a plat et on l'interroge a "
                  f"chaque barre. L'echantillon de cette epoch est BIAISE "
                  f"vers le debut des episodes — ne pas lire ses chiffres "
                  f"comme les autres.")
        print(f"  {_col('phase', _C.GREY)}  collecte {_chrono['collecte']:.0f} s "
              f"sur {len(envs)} episodes "
              f"({_chrono['collecte']/max(len(envs),1):.1f} s/episode)")

        # ZERO PASSE QUAND PPO EST COUPE. Le mecanisme existait deja pour
        # `epoch_moyenne` ; on s'y branche plutot que d'ajouter un second
        # chemin de saut. Toutes les pertes — acteur, critic, entropie — et
        # tous les pas d'optimiseur vivent dans cette boucle, donc il n'y a
        # rien d'autre a neutraliser.
        # ==================================================================
        # LA MISE A JOUR PPO A ETE SUPPRIMEE LE 2026-09-21.
        #
        # ELLE N'AVAIT PLUS D'ACTION A OPTIMISER. La tete de direction a ete
        # retiree de la decision le 2026-09-20, la tete de budget du reseau
        # le lendemain. L'acteur n'avait donc plus aucune sortie lue, et le
        # critic n'existait que pour lui fournir un avantage.
        #
        # LES DEUX VERDICTS, chacun mesure sur un run complet :
        #
        #   L'ENTREE (exec40, 23 epochs). Comparee a sa propre politique
        #   GELEE, la politique entrainee rend +0.27 point en moyenne pour
        #   une reference de bruit a +0.7. C'est du bruit.
        #
        #   LE BUDGET (exec67, 22 epochs). PPO l'entraine reellement — le
        #   gradient saute de 7e-05 a 1.1e-01 a l'epoch 6 pile — mais ce
        #   qu'il apprend est de MISER LE MINIMUM, et `rho` se degrade avec :
        #   negatif 10 fois sur 11 sur la seconde moitie du run.
        #
        # LA RAISON COMMUNE, ET ELLE EST STRUCTURELLE : PPO optimise le
        # rendement de ses ACTIONS ; rien dans son objectif ne recompense un
        # bon ORDRE de ses sorties. Or la selectivite ne consomme QU'UN
        # ORDRE, et `rho` est une correlation de rang.
        #
        # POURQUOI SUPPRIMER PLUTOT QUE DESACTIVER. Le bloc restait derriere
        # `ppo_actif = False` et deballait quatre valeurs de `policy.sorties`,
        # qui n'en rend plus que trois. Il ne pouvait donc PLUS fonctionner —
        # un drapeau qui promet de rallumer quelque chose de casse est pire
        # qu'une absence. Son histoire est ici ; le code, dans git.
        #
        # CE QUI RESTE. Le ROLLOUT tourne toujours : il calibre la barre
        # d'entree et produit les statistiques de trades. Le TRONC n'apprend
        # plus que par la tete de rang, en supervise, sur la grille dense.
        # ==================================================================

        # `scheduler_rang.step()` N'EST PAS ICI : voir apres la passe `rang`.
        import json as _json
        with open(f"audit_collecte_{cfg.side}{suffix}.jsonl", "a", encoding="utf-8") as _fa:
            _fa.write(_json.dumps({"epoch": epoch,
                                  "on_policy": not cfg.legacy_off_policy_curriculum,
                                  **sampling_audit}) + "\n")
        # `[PPO SAMPLING]` ETAIT LE NOM D'AVANT. PPO a ete supprime le
        # 2026-09-21 ; la ligne, elle, compte toujours quelque chose de
        # reel. On la renomme plutot que de laisser croire qu'une mise a
        # jour de politique tourne encore.
        print(f"[COLLECTE] epoch={epoch} {sampling_audit}")

        # AJUSTEMENT DU BUDGET. La geometrie decide du nombre de decisions
        # qu'un episode produit ; on en deduit combien d'episodes il faut
        # pour atteindre la cible. Borne entre 1 et le plafond, et lissee de
        # moitie pour qu'une epoch atypique ne fasse pas osciller le budget.
        _joues = max(sampling_audit.get("episodes_joues", 1), 1)
        # ON COMPTE LES TRANSITIONS VERSEES, PLUS LES DECISIONS.
        #
        # Les deux se confondaient tant qu'une decision n'existait que si une
        # entree etait possible : presque chacune ouvrait une position, donc
        # produisait une transition. Depuis que `deciding` interroge la
        # capacite AU PLAFOND, l'environnement est consulte sur presque toutes
        # les barres et la plupart des decisions finissent en « attendre » —
        # or une decision qui n'ouvre rien N'EST PAS versee au tampon.
        #
        # Mesure de l'epoch 1 d'exec14 : 33 307 decisions pour 12 episodes,
        # soit 2 776 par episode. Viser `cible_decisions / 2776` aurait ramene
        # le budget a UN episode par epoch — une seule fenetre de marche par
        # gradient. Le compteur mesurait la consultation, pas l'apprentissage.
        #
        # ET DESORMAIS ON COMPTE LES TRADES. Les transitions versees
        # comprenaient les attentes : un agent presque toujours a plat en
        # versait ~3 800 par episode, et le regulateur est descendu a UN
        # episode par epoch. Voir `cible_trades`.
        _ouv = max(int(sampling_audit.get("ouvertures", 0)), 1)
        _par_ep = max(_ouv / _joues, 1e-9)
        _vise = int(round(cfg.cible_trades / _par_ep))
        _vise = max(1, min(_vise, cfg.episodes_per_epoch))
        _avant = n_episodes_courant[0]
        _lisse = max(1, (_avant + _vise) // 2)

        # ---- LE TEMPS BORNE LE BUDGET, ET IL LE BORNE EN DERNIER ----
        #
        # On sait ce que les episodes de CETTE epoch ont coute : `collecte`
        # secondes pour `_joues` episodes. Le nombre qui tient dans le budget
        # est donc une regle de trois sur une mesure, pas une estimation.
        #
        # LA CROISSANCE EST BORNEE EN PLUS. Au premier ajustement le
        # regulateur veut sauter au plafond ; sans borne il y saute AVANT que
        # la mesure de temps n'ait pu servir, et l'epoch suivante est deja
        # perdue. On monte donc par paliers, et la cible de decisions est
        # atteinte en quelques epochs au lieu d'une.
        _t_col = float(_chrono.get("collecte", 0.0))
        _budget_s = float(getattr(cfg, "secondes_collecte_max", 0) or 0)
        _plafond_tps = _lisse
        if _t_col > 1e-6 and _budget_s > 0:
            _plafond_tps = max(1, int(_joues * _budget_s / _t_col))
        _plafond_croi = max(
            1, int(_avant * float(getattr(cfg, "croissance_episodes_max",
                                          1.5))))
        n_episodes_courant[0] = max(1, min(_lisse, _plafond_tps,
                                           _plafond_croi))
        if n_episodes_courant[0] < _lisse:
            _quoi = ("le temps" if _plafond_tps <= _plafond_croi
                     else "la croissance")
            print(f"  {_col('cadence', _C.GREY)}  {_joues} episodes en "
                  f"{_t_col:.0f} s -> le regulateur en veut {_lisse}, "
                  f"{_quoi} en autorise {n_episodes_courant[0]} "
                  f"(budget {_budget_s:.0f} s)")


        # ---- Recalibration du seuil sur la conviction réellement observée ----
        # Le quantile (1 − sélectivité) de max(p_BUY, p_SELL) est, par
        # construction, la barre que franchissent exactement les `sélectivité` %
        # d'instants les plus favorables. Il suit donc l'échelle du modèle au
        # lieu de la supposer — c'est tout l'intérêt par rapport à un seuil fixe.
        # GARDE-FOU sur l'étendue. Mesuré à l'epoch 1 : la policy initiale donne
        # med 0.352 / max 0.354, soit 0.002 d'amplitude. Sur une distribution
        # aussi plate, un quantile est un pile ou face — le seuil calculé sur le
        # train tombait au-dessus de TOUTES les valeurs du val, d'où 0 trade.
        # Tant que le modèle ne différencie pas les instants, filtrer n'a aucun
        # sens : on laisse tout passer pour disposer d'une mesure de référence.
        # L'AMORCE DE L'EPOCH SUIVANTE. On garde la fin de la fenetre, comme
        # `rolling_decision_spec` le fait pour la validation.
        if pbs_rollout[0] or pbs_rollout[1]:
            pbs_rollout_prec = [c[-cfg.rang_fenetre:] for c in pbs_rollout]

        if pbs_epoch:
            pbs_med = float(np.median(pbs_epoch))
            pbs_max = float(np.max(pbs_epoch))
            pbs_etendue = float(np.quantile(pbs_epoch, 0.99)
                                - np.quantile(pbs_epoch, 0.01))
            if pbs_etendue < cfg.pbs_etendue_min:
                calib_thr_courant = 0.0
            else:
                calib_thr_courant = float(
                    np.quantile(pbs_epoch, 1.0 - selectivite))
        else:
            pbs_med = pbs_max = pbs_etendue = 0.0

        profit_epoch = float(sum(epoch_pnl))
        num_trades_epoch = len(epoch_trades_pnl)
        winrate_epoch = (
            float(np.mean([p > 0 for p in epoch_trades_pnl]))
            if num_trades_epoch > 0 else 0.0
        )
        max_dd_epoch = float(max(epoch_dd) if epoch_dd else 0.0)

        wins_train   = [p for p in epoch_trades_pnl if p > 0]
        losses_train = [p for p in epoch_trades_pnl if p <= 0]
        avg_win_train  = float(np.mean(wins_train))   if wins_train  else 0.0
        avg_loss_train = float(np.mean(losses_train)) if losses_train else 0.0
        num_loss_train = len(losses_train)
        total_profit_train  = float(sum(wins_train))
        total_loss_train    = float(sum(losses_train))
        profit_factor_train = total_profit_train / (abs(total_loss_train) + 1e-8)

        total_actions_env = int(action_counts_env.sum()) if action_counts_env.sum() > 0 else 1
        buy_count, sell_count, hold_count, close_count = action_counts_env
        buy_ratio = buy_count / total_actions_env
        sell_ratio = sell_count / total_actions_env
        hold_ratio = hold_count / total_actions_env
        # LA PART DE CLOTURES EST LE TEMOIN QUI MANQUAIT.
        #
        # C'est exactement le chiffre qui aurait montre la position
        # immortelle sans avoir a lire `decisions` : a zero, la sortie est
        # debranchee ou la tete ne ferme jamais, et un episode de 5 760
        # barres ne fait qu'un seul trade.
        close_ratio = close_count / total_actions_env

        # --------- validation (greedy) ---------
        policy.eval()

        # L'EPOCH 1 VALIDE TOUJOURS. Sans elle il n'y a aucune mesure a
        # reprendre les epochs suivantes, et le journal afficherait une
        # validation vide — 0 trade, PnL nul — qu'on lirait comme un
        # effondrement. Premiere version de ce patch : c'est exactement ce
        # qu'elle faisait.
        #
        # Calcule ICI et non plus bas : la passe de CALIBRATION le lit, et
        # elle vient avant.
        _valide = (cfg.validation_tous_les <= 1 or epoch == 1
                   or (epoch % cfg.validation_tous_les) == 0
                   or epoch >= cfg.epochs)
        pbs_val: List[List[float]] = [[], []]   # convictions BUY / SELL sur le val
        val_pnl = []
        val_dd = []
        # LES DEUX PORTILLONS DE LA VALIDATION, comptes separement. Voir leur
        # incrementation dans la boucle pour ce que leur absence a coute.
        _portillons = {"occasions": 0, "barre": 0, "capacite": 0, "entrees": 0}
        # COMMENT CHAQUE EPISODE DE VALIDATION S'EST TERMINE. Le creux ne
        # suffit plus a juger : depuis que l'episode va au bout de sa tranche,
        # un creux de 60 % peut etre suivi d'une remontee. Ce qui disqualifie
        # un checkpoint n'est plus un pourcentage, c'est un compte DETRUIT.
        val_fins: List[str] = []
        val_trades = []
        # LA TENUE PAR ISSUE, ET ELLE MANQUAIT AU JOURNAL.
        #
        # `trades_meta` porte `hold_bars` depuis toujours et il ne sortait que
        # dans le CSV trade-par-trade. La question « combien de temps tient-on
        # un gagnant, combien de temps coupe-t-on un perdant » a du etre
        # posee quatre fois et resolue quatre fois par simulation hors ligne,
        # alors que la boucle de validation avait le chiffre sous la main.
        #
        # CE QU'IL REVELE, mesure le 2026-09-22 au plafond de 480 : les
        # gagnants tiennent TOUS jusqu'au plafond, mediane ET moyenne a 480.
        # La regle n'a aucune prise de profit — elle ne sait que couper les
        # pertes. « Gagnant » ne veut pas dire « la tete a decide de sortir
        # avec un gain », ca veut dire « le trade a survecu au chronometre ».
        # Un journal qui ne montre que AvgW et AvgL ne peut pas faire voir ca.
        val_tenues_g: List[int] = []
        val_tenues_p: List[int] = []
        val_trades_side: List[int] = []

        # Épisodes de validation VECTORISÉS : même principe que le rollout de
        # train. La val pesait 64% du coût (7 × 4000 pas contre 4 × 4000), et
        # elle ne termine jamais en avance puisqu'elle ne déclenche pas le
        # garde-fou DD. La batcher la rend ~7× moins chère, ce qui permet de
        # garder val_episodes=7 (nécessaire à la stabilité du Sortino30).
        n_val = len(val_envs)


        _chrono["maj PPO"] = time.time() - _t_phase; _t_phase = time.time()
        # ==============================================================
        # PPO COMPLET — LES QUATRE TETES APPRENNENT ICI (2026-09-25)
        # ==============================================================
        #
        # D'abord l'ENTREE — achat et vente, sur les decisions prises a
        # plat —, puis la SORTIE — coupure des gains et des pertes, sur les
        # decisions prises en position. Chaque tete avance par SON
        # optimiseur ; le tronc, lu par les quatre, avance par le sien a
        # chacune des deux mises a jour.
        #
        # EN MODE EVAL : le tronc a du dropout, et le rapport de PPO n'est
        # exact que si l'ancienne et la nouvelle politique sont la meme
        # fonction. `eval()` ne coupe pas le gradient.
        policy.eval()
        _st_e = maj_ppo_entree(policy, ep_buf, optims_ppo, cfg, device)
        if _st_e is not None:
            epoch_actor_loss.append(_st_e["actor"])
            epoch_critic_loss.append(_st_e["critic"])
            epoch_kl.append(_st_e["kl"])
            clipfrac_hist.append(_st_e["clip"])
            _pa_, _pv_, _ph_ = (100 * x for x in _st_e["parts"])
            print(f"  {_col('phase', _C.GREY)}  PPO entree  "
                  f"{_st_e['n']:,} decisions (sur {_st_e['n_total']:,})  "
                  f"ActorL {_st_e['actor']:+.4f}  CriticL {_st_e['critic']:.4f}  "
                  f"KL {_st_e['kl']:+.4f}  clip {100*_st_e['clip']:.0f}%  "
                  f"H {_st_e['H']:.3f}/{np.log(N_ACTIONS_ENTREE):.3f}  "
                  f"achat {_pa_:.1f}%  vente {_pv_:.1f}%  attendre {_ph_:.1f}%")
        else:
            print(f"  {_col('phase', _C.GREY)}  PPO entree  trop peu de "
                  f"decisions pour une mise a jour")
        _st_s = maj_ppo_sortie(policy, sortie_buf, optims_ppo, cfg, device)
        if _st_s is not None:
            print(f"  {_col('phase', _C.GREY)}  PPO sortie  "
                  f"{_st_s['n']:,} transitions (sur {_st_s['n_total']:,})  "
                  f"ActorL {_st_s['actor']:+.4f}  CriticL {_st_s['critic']:.4f}  "
                  f"KL {_st_s['kl']:+.4f}  clip {100*_st_s['clip']:.0f}%")
            for _cote, (_nd, _hh, _ferme) in _st_s["tetes"].items():
                print(f"  {_col('phase', _C.GREY)}  PPO sortie {_cote}  "
                      f"{_nd:,} decisions  "
                      f"H {_hh:.3f}/{np.log(N_ACTIONS_SORTIE):.3f}  "
                      f"ferme {100*_ferme:.1f}%")
        else:
            print(f"  {_col('phase', _C.GREY)}  PPO sortie  trop peu de "
                  f"transitions pour une mise a jour")
        _chrono["maj PPO"] = time.time() - _t_phase; _t_phase = time.time()

        # LA PASSE SUPERVISEE TOURNE APRES PPO, ET NON AVANT.
        #
        # `tete_aux` est posee sur le TRONC PARTAGE — `tete_aux(mlp(norm(encode(x))))`.
        # Le journal d'exec34 donne les gradients que PPO y depose :
        #
        #     g[actor 1.72e-06   critic 2.87e-01   tronc 1.47e+00]
        #     g[actor 5.45e-06   critic 1.17e+00   tronc 1.81e+00]
        #
        # LE TRONC BOUGE DE 1.5 A 1.8 A CHAQUE MISE A JOUR, six ordres de
        # grandeur au-dessus de l'acteur, essentiellement sous la perte du
        # critic. La tete de rang ne se bat plus contre une autre perte sur
        # elle-meme — ce terme a ete coupe — mais contre un socle qui se
        # deforme sous elle.
        #
        # ET L'ORDRE AGGRAVAIT TOUT. La passe tournait AVANT les 264 pas de
        # PPO, et la validation APRES : la tete etait donc notee 264 pas
        # apres son dernier pas d'entrainement. `AuxL` remontait d'une epoch
        # a l'autre — 2.0787 puis 2.1426 — alors que plus rien ne la
        # contrariait directement.
        #
        # Posee ici, elle est la DERNIERE a toucher le tronc avant que la
        # validation ne le lise, et avant que la collecte suivante ne s'en
        # serve pour decider des entrees. Ca ne supprime pas le partage du
        # tronc — ce serait un autre chantier — mais ca cesse de mesurer la
        # tete juste apres l'avoir defaite.

        # ---------- PASSE SUPERVISEE SUR LA GRILLE DENSE ----------
        # Elle tourne sur son PROPRE lot : la tete apprend donc sur un
        # echantillon que la politique n'a pas choisi. La perte reste
        # la MEME qu'avant — moindres carres sur l'indicateur, masquee par le
        # cote — pour que le seul changement soit la DONNEE, et que l'effet
        # soit attribuable.
        _rang_perte = float("nan")
        # DECLARES AVANT LA BRANCHE, parce que le journal les lit APRES
        # elle. Sans grille dense — fenetre trop courte, `pas_grille_rang`
        # trop large — la branche ne s'execute pas et le `print` tombait
        # sur un `UnboundLocalError`, APRES tout le travail de l'epoch.
        _gg = []
        _rang_mse = _rang_topk = _rang_ancre = float("nan")
        _rang_gnorm = _rang_clip = float("nan")
        _plaf_rang = float(getattr(cfg, "max_grad_norm_rang", 0.6))
        if _gr_idx is not None and getattr(cfg, "pas_rang_par_epoch", 0) > 0:
            policy.train()
            _xg = np.zeros((cfg.lookback, N_POS_FEATURES), np.float32)
            _xg[:, 3] = 1.0
            _mcg = torch.tensor(
                np.array(cotes_permises(cfg.side), np.float32), device=device)
            _pg = []
            _gg = []
            _pm = []   # moindres carres
            _pk = []   # top-k
            _pa = []   # ancre d'etendue
            _ng = min(cfg.batch_size, len(_gr_idx))
            for _ in range(int(cfg.pas_rang_par_epoch)):
                _b = np.random.choice(len(_gr_idx), _ng, replace=False)
                _o = np.stack([
                    np.concatenate([train_data.features[i - cfg.lookback:i],
                                    _xg], axis=-1) for i in _gr_idx[_b]])
                _pred = policy.sorties(
                    torch.from_numpy(_o).to(device))
                # LES TROIS TERMES SONT GARDES SEPAREMENT.
                #
                # `AuxL` sommait moindres carres, top-k et ancre en un seul
                # nombre. Les trois ne mesurent pas la meme chose : le premier
                # predit la VALEUR de la cible, le deuxieme ordonne le SOMMET,
                # le troisieme tient l'ECHELLE. L'un peut monter pendant qu'un
                # autre descend, et la somme ne dit alors rien — on constate
                # que « quelque chose se casse » sans savoir quoi.
                #
                # C'est exactement ce qu'on a vecu : `AuxL` oscillait sans
                # tendance pendant que `sommet` tenait a +0.6 au-dessus du
                # hasard. Les deux ne se contredisaient pas, ils ne parlaient
                # pas de la meme chose.
                #
                # RAPPEL DE CE QUI JUGE VRAIMENT LE CLASSEMENT : c'est
                # `sommet` mesure sur la VALIDATION, pas cette perte mesuree
                # sur la grille d'ENTRAINEMENT. Une perte qui descend pendant
                # que `sommet` s'effondre, c'est du surapprentissage, et ce
                # depot l'a deja paye. Ces trois nombres servent a savoir CE
                # QUI derive, pas a decider.
                _err = (_pred - _gr_y[_b]).pow(2) * _mcg
                _t_mse = (_err.sum(dim=1) / _mcg.sum()).mean()
                _t_topk = torch.zeros((), device=_pred.device)
                _t_ancre = torch.zeros((), device=_pred.device)
                _pr = float(getattr(cfg, "coef_mse", 1.0)) * _t_mse
                # LE CRITERE DE DEPLOIEMENT S'AJOUTE A LA MOINDRE CARRE, il
                # ne la remplace pas. La moindre carre garde le score dans
                # l'unite de la cible — c'est elle qui l'empeche de deriver,
                # et les consommateurs lui appliquent une sigmoide avant de le
                # comparer a un quantile : un score parti a +-20 saturerait.
                # Le terme top-k, lui, est exactement invariant par
                # transformation affine du score, donc incapable de fixer une
                # echelle. Les deux se completent au lieu de se concurrencer.
                #
                # LA CIBLE EST L'INDICATEUR BORNE, pas le R brut. Mesure du
                # 2026-09-20 : l'indicateur va de -1.41 a +1.99 et se classe
                # avec le rendement reellement encaisse a rho +0.921, quand le
                # R brut monte a +18.93. Sur du R brut, la moyenne des k
                # meilleurs est dominee par une poignee d'evenements extremes
                # et devient instable — c'est ce qui avait impose un ecretage
                # ailleurs. Borne, le probleme ne se pose pas.
                if float(getattr(cfg, "coef_top_k", 0.0)) > 0.0:
                    _t_topk = perte_top_k(
                        _pred, _gr_y[_b], _mcg,
                        float(getattr(cfg, "selectivite_top_k", 0.05)))
                    _pr = _pr + float(cfg.coef_top_k) * _t_topk
                    # L'ANCRE D'ETENDUE, ET POURQUOI ELLE EST A SENS UNIQUE.
                    #
                    # `perte_top_k` est exactement invariante par
                    # transformation affine du score : elle ne juge que
                    # l'ORDRE. Elle ne peut donc fixer aucune echelle, et rien
                    # ne l'empeche de resserrer les scores les uns sur les
                    # autres. Or les consommateurs — validation, test, live —
                    # appliquent une SIGMOIDE au score avant de le comparer a
                    # un quantile : des convictions trop serrees s'egalisent
                    # a la precision machine et la selectivite disparait.
                    #
                    # MESURE DU 2026-09-20, exec36 contre exec35 aux MEMES
                    # epochs (acteur gele dans les deux, tout le reste
                    # identique) — etendue des convictions en validation :
                    #
                    #     sans top-k   0.35  0.15  0.23  0.22  0.22
                    #     avec top-k   0.027 0.044 0.037 0.093 0.078
                    #
                    # Un facteur QUATRE. Rien n'etait casse — le seuil separait
                    # encore, les trades se faisaient — mais la marge fondait.
                    #
                    # RESERVE SUR MA PREMIERE LECTURE : j'avais annonce un
                    # facteur 1 500 en comparant `etendue[tr]`. C'etait faux.
                    # Cette colonne mesure les probabilites de la POLITIQUE,
                    # quasi uniformes tant que l'acteur est gele, et je
                    # comparais des epochs de phases differentes.
                    #
                    # L'ANCRE NE MORD QUE VERS LE BAS. Celle de l'autre
                    # variante forcait moyenne 0 et ecart-type 1 ; ici elle se
                    # battrait contre la moindre carre, qui vise une cible
                    # d'ecart-type 1.44. On penalise donc uniquement
                    # l'etendue MANQUANTE par rapport a celle de la cible :
                    # quand la moindre carre fait son travail, ce terme vaut
                    # exactement zero et ne contraint rien.
                    _sd_y = _gr_y[_b][:, _mcg > 0].std()
                    _sd_p = _pred[:, _mcg > 0].std()
                    _t_ancre = torch.relu(_sd_y - _sd_p).pow(2)
                    _pr = _pr + float(getattr(cfg, "coef_ancre", 1.0)) * _t_ancre
                optimizer_rang.zero_grad(set_to_none=True)
                (cfg.aux_coef * _pr).backward()
                # LA NORME DU GRADIENT DE CETTE PASSE, QUI MANQUAIT.
                #
                # `AuxL` monte a chaque epoch alors qu'on entraine cette tete
                # 240 pas par epoch : 2.0148 -> 2.3332 -> 2.4232 sur exec56.
                # Deux causes possibles, et rien ne permettait de les separer.
                #
                #   AMORTISSEMENT : cette passe partage l'optimiseur Adam avec
                #   PPO, dont les gradients sont d'un autre ordre de grandeur
                #   (CriticL 221 contre AuxL 2.4). Adam divise par la racine
                #   de `v`, gonfle par PPO : les pas seraient alors etouffes et
                #   la tete n'apprendrait pas. Signature : gradient PETIT.
                #
                #   DIVERGENCE : le pas est trop grand pour cet objectif et la
                #   perte monte d'elle-meme. Signature : gradient GRAND, et
                #   l'ecretage qui mord en permanence.
                #
                # La norme AVANT ecretage tranche entre les deux, et
                # `clip_grad_norm_` la rend deja — il suffisait de la lire.
                _gn_rang = float(torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    float(getattr(cfg, "max_grad_norm_rang",
                                  cfg.max_grad_norm))))
                optimizer_rang.step()
                _pg.append(float(_pr.item()))
                _gg.append(_gn_rang)
                _pm.append(float(_t_mse.item()))
                _pk.append(float(_t_topk.item()))
                _pa.append(float(_t_ancre.item()))
            _rang_perte = float(np.mean(_pg)) if _pg else float("nan")
            _rang_gnorm = float(np.mean(_gg)) if _gg else float("nan")
            _rang_mse = float(np.mean(_pm)) if _pm else float("nan")
            _rang_topk = float(np.mean(_pk)) if _pk else float("nan")
            _rang_ancre = float(np.mean(_pa)) if _pa else float("nan")
            _plaf_rang = float(getattr(cfg, "max_grad_norm_rang",
                                       cfg.max_grad_norm))
            _rang_clip = (float(np.mean([g > _plaf_rang for g in _gg]))
                          if _gg else float("nan"))

        # ---------- PASSE SUPERVISEE DE LA TETE DE CLOTURE ----------
        #
        # SON PROPRE ECHANTILLON, ET C'EST TOUT L'INTERET. Les tetes
        # d'entree apprennent sur la grille dense, qui ne contient que des
        # etats PLATS. « Faut-il fermer ? » n'a de sens que sur un etat ou
        # une position existe — avec son sens, son gain latent, son age.
        #
        # L'ECHANTILLON NE VIENT PAS DU SIMULATEUR. Un etat en position se
        # decrit par (barre d'entree, sens, barres tenues) ; le reste se
        # deduit des prix. On les tire, donc la tete apprend sur des
        # positions que la politique courante n'aurait pas prises — elle ne
        # s'enferme pas dans ses propres habitudes. Meme principe, et meme
        # raison, que la grille dense pour les entrees.
        #
        # LA CIBLE EST BRUTE. Le cout de sortie sera paye de toute facon,
        # donc il s'annule entre « fermer » et « tenir ». Le facturer ferait
        # fermer trop tot. Voir `cibles_m1.cible_cloture`.
        _clot_perte = float("nan")
        if _clot is not None and getattr(cfg, "pas_cloture_par_epoch", 0) > 0:
            policy.train()
            _nc = min(cfg.batch_size, len(_clot["idx"]))
            _pc = []
            for _ in range(int(cfg.pas_cloture_par_epoch)):
                _bc = np.random.choice(len(_clot["idx"]), _nc, replace=False)
                _ic = _clot["idx"][_bc]
                # LE BLOC D'ETAT EST REMPLI POSITION PAR POSITION, dans les
                # unites exactes de `_get_obs` — sens, latent en ATR de
                # l'entree, age normalise. Les trois autres colonnes restent
                # a leur valeur d'etat plat : la capacite vaut zero puisque
                # la position occupe la seule place, et le creux du compte
                # n'appartient pas a cette position.
                _xc = np.zeros((_nc, cfg.lookback, N_POS_FEATURES), np.float32)
                _xc[:, :, 0] = _clot["sens"][_bc][:, None]
                _xc[:, :, 1] = _clot["lat"][_bc][:, None]
                _xc[:, :, 2] = _clot["tenue"][_bc][:, None]
                _oc = np.stack([
                    train_data.features[i - cfg.lookback:i] for i in _ic])
                _oc = np.concatenate([_oc, _xc], axis=-1)
                _pred_c = policy.cloture(
                    torch.from_numpy(_oc).to(device)).squeeze(-1)
                # ELLE APPREND UNE AMPLITUDE, PLUS UN SIGNE.
                #
                # DEUX CIBLES ONT DEJA ECHOUE ICI, ET IL FAUT LES DEUX
                # HISTOIRES POUR COMPRENDRE LA TROISIEME.
                #
                # MOINDRES CARRES SUR LE RENDEMENT FUTUR (bps, sd 14.60).
                # Trois epochs : perte 210.94 -> 213.43 -> 217.03 pour une
                # variance de cible de 213.19, et « fermerait » passant de
                # 49 % a 100 %. La tete s'est figee sur une constante.
                # Quand une cible n'est pas predictible EN MAGNITUDE, le
                # gradient des moindres carres pousse vers la moyenne.
                #
                # ENTROPIE CROISEE SUR LE SIGNE. Quatre epochs au hasard —
                # juste 50.8 %, 49.2 %, 53.1 %, 46.9 % — perte collee a
                # 0.6932, soit ln(2). Ce n'etait pas la faute de la perte :
                # la cible elle-meme est a 50.0 % de positifs. Le SENS du
                # prix a trente minutes n'est pas dans ces donnees, et
                # trois tests independants le disent.
                #
                # LA CIBLE EST DESORMAIS LE RISQUE : la pire excursion a
                # venir, en ATR d'entree, POSITIVE. Voir
                # `cibles_m1.cible_risque`. Elle se predit — `vol_20` la
                # correle a 0.63 hors echantillon contre un plancher de
                # 0.14 — donc les moindres carres ont enfin de quoi
                # mordre, et le mode d'echec de la premiere version ne
                # peut pas revenir : il venait de l'absence de signal.
                #
                # EN RACINE, ET C'EST DELIBERE. La cible va de 0.17 a 105
                # ATR, une queue de facteur six cents. En brut, quelques
                # tempetes dicteraient tout le gradient. La racine ramene
                # la distribution a une echelle ou chaque etat compte, et
                # reste monotone — l'ordre, qui est ce que la regle lit,
                # est preserve exactement.
                #
                # `softplus` A LA SORTIE : la tete rend une amplitude, donc
                # positive. Un `relu` couperait le gradient a zero sur les
                # etats calmes, ceux-la memes ou tenir est le bon choix.
                _y_c = _clot["y"][_bc].clamp_min(0.0)
                _perte_c = (torch.nn.functional.softplus(_pred_c).sqrt()
                            - _y_c.sqrt()).pow(2).mean()
                optimizer_rang.zero_grad(set_to_none=True)
                _perte_c.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    float(getattr(cfg, "max_grad_norm_rang",
                                  cfg.max_grad_norm)))
                optimizer_rang.step()
                _pc.append(float(_perte_c.item()))
            _clot_perte = float(np.mean(_pc)) if _pc else float("nan")
            with torch.no_grad():
                # CE QU'ON LIT SUR UNE AMPLITUDE : la correlation de RANG
                # avec la cible. La regle ne consomme qu'un ORDRE — elle
                # compare le latent a un multiple du risque — donc c'est
                # l'ordre qu'il faut surveiller, pas l'erreur absolue.
                #
                # `juste %` et `fermerait %` ont disparu avec la cible
                # binaire : sur une amplitude ils ne veulent rien dire.
                _rq = torch.nn.functional.softplus(_pred_c).cpu().numpy()
                _yr = _y_c.cpu().numpy()
                _n_r = min(len(_rq), 4000)
                try:
                    from scipy.stats import spearmanr as _sr
                    _rho_r = float(_sr(_rq[:_n_r], _yr[:_n_r]).correlation)
                except Exception:
                    _rho_r = float("nan")
                _med_r = float(np.median(_rq))
            print(f"  {_col('phase', _C.GREY)}  cloture  "
                  f"perte {_clot_perte:.4f}  "
                  f"rho {_rho_r:+.3f} (vol_20 seul fait +0.63)  "
                  f"risque median {_med_r:.2f} ATR")

        # ---------- LA TETE DE PROFIT ----------
        #
        # ELLE NE TRAVERSE PAS LE TRONC, et c'est une mesure qui l'impose.
        # La meme cible, les memes occasions, le 2026-09-22 :
        #
        #     274 colonnes + l'etat   IC +0.1401  plancher 0.2806   1 arbre
        #     l'etat SEUL             IC +0.1911  plancher 0.0685 100 arbres
        #
        # Les colonnes de marche n'affaiblissent pas le signal, elles
        # l'EFFACENT. On lui donne donc les quatre colonnes de
        # `entree_profit` et rien d'autre.
        #
        # SON OPTIMISEUR NE PORTE QUE SES PROPRES POIDS. `optimizer_rang`
        # porte tout le reseau : s'en servir ici ferait remonter le
        # gradient de la prise de profit dans le tronc, que cette tete
        # n'utilise justement pas. Le tronc bougerait au service d'un
        # organe qui ne le lit pas.
        #
        # LA PERTE EST EN RACINE, comme pour le risque et pour la meme
        # raison : la cible a une queue de plusieurs ordres de grandeur, et
        # en brut quelques rebonds geants dicteraient tout le gradient. La
        # racine est monotone, donc l'ORDRE — seul consomme par la regle —
        # est preserve exactement.
        _prof_perte = float("nan")
        if (_clot is not None and _clot.get("yp") is not None
                and getattr(cfg, "pas_profit_par_epoch", 0) > 0):
            policy.train()
            _optim_prof = _optimiseur_profit(policy, cfg)
            _np_ = min(cfg.batch_size, len(_clot["idx"]))
            _pp = []
            for _ in range(int(cfg.pas_profit_par_epoch)):
                _bp = np.random.choice(len(_clot["idx"]), _np_, replace=False)
                _ip = _clot["idx"][_bp]
                _xp = np.zeros((_np_, cfg.lookback, N_POS_FEATURES), np.float32)
                _xp[:, :, 0] = _clot["sens"][_bp][:, None]
                _xp[:, :, 1] = _clot["lat"][_bp][:, None]
                _xp[:, :, 2] = _clot["tenue"][_bp][:, None]
                _op = np.stack([
                    train_data.features[i - cfg.lookback:i] for i in _ip])
                _op = np.concatenate([_op, _xp], axis=-1)
                # LA MEME EXTRACTION QU'EN DECISION, par la meme fonction.
                _pin = torch.from_numpy(
                    entree_profit(_op, N_BASE_FEATURES)).to(device)
                _pred_p = policy.profit(_pin).squeeze(-1)
                _y_p = _clot["yp"][_bp].clamp_min(0.0)
                _perte_p = (torch.nn.functional.softplus(_pred_p).sqrt()
                            - _y_p.sqrt()).pow(2).mean()
                _optim_prof.zero_grad(set_to_none=True)
                _perte_p.backward()
                _optim_prof.step()
                _pp.append(float(_perte_p.item()))
            _prof_perte = float(np.mean(_pp)) if _pp else float("nan")
            with torch.no_grad():
                _rp = torch.nn.functional.softplus(_pred_p).cpu().numpy()
                _yp_ = _y_p.cpu().numpy()
                try:
                    from scipy.stats import spearmanr as _sr2
                    _n_p = min(len(_rp), 4000)
                    _rho_p = float(_sr2(_rp[:_n_p], _yp_[:_n_p]).correlation)
                except Exception:
                    _rho_p = float("nan")
                _med_p = float(np.median(_rp))
                # CE QUE LA REGLE FERAIT SUR CET ECHANTILLON. Sans ce
                # compteur, un seuil trop bas ferait de la tete un
                # ornement silencieux — l'etat exact ou `tete_cloture` a
                # vecu pendant des jours.
                # LE COMPTEUR SUIT LA REGLE REELLE, sinon il mesure
                # autre chose que ce que l'environnement joue : la regle
                # est relative au latent ET exige un gain.
                _lat_p = _pin[:, 0].cpu().numpy()
                _part_p = float(np.mean(
                    (_rp < float(getattr(cfg, "coupe_profit", 0.0)) * _lat_p)
                    & (_lat_p > 0.0)))
                # LE GARDE D'ARMEMENT LIT CETTE VALEUR. Elle est posee sur
                # la politique elle-meme pour que `demande_profit` la
                # trouve sans qu'on ait a la faire passer par six appels.
                policy._rho_profit = (float(_rho_p) if np.isfinite(_rho_p)
                                      else 0.0)
                # L'ECHELLE, ET ELLE EST AUSSI IMPORTANTE QUE L'ORDRE.
                #
                # `rho` mesure l'ORDRE, pas la MAGNITUDE. Le run du
                # 2026-09-22 le montre net : la tete atteint rho +0.150 —
                # exactement la valeur visee hors ligne — en predisant
                # `reste median 3.57 ATR` la ou la cible en vaut 13. Un
                # facteur QUATRE d'erreur d'echelle, avec un classement
                # parfaitement correct.
                #
                # ET LA REGLE EST SENSIBLE A L'ECHELLE D'UN COTE. Elle
                # compare `reste_predit` a `coupe x latent` : le latent est
                # une grandeur reelle, la prediction non. Resultat, une
                # regle calibree pour mordre sur 0.4 % des trades en
                # declenchait 46.9 %.
                #
                # ON REDRESSE PAR LE RAPPORT DES MEDIANES. La mediane est
                # robuste aux queues — et cette cible en a de tres longues.
                # Le facteur est borne : une tete degeneree ne doit pas
                # pouvoir fabriquer un redressement de mille.
                _med_y = float(np.median(_yp_))
                _ech = _med_y / max(_med_p, 1e-6)
                policy._echelle_profit = float(np.clip(_ech, 0.1, 10.0))
            print(f"  {_col('phase', _C.GREY)}  profit   "
                  f"perte {_prof_perte:.4f}  "
                  f"rho {_rho_p:+.3f} (etat seul faisait +0.15)  "
                  f"reste median {_med_p:.2f} ATR (cible {float(np.median(_yp_)):.2f}, "
                  f"echelle x{getattr(policy, '_echelle_profit', 1.0):.2f})  "
                  f"fermerait {100*_part_p:.1f}% "
                  f"(coupe {getattr(cfg, 'coupe_profit', 0.0):.2f} x latent, "
                  f"armee {'OUI' if _rho_p >= float(getattr(cfg, 'rho_profit_min', 0.0)) else 'NON'})")
        # `AuxL` RAPPORTE LA PASSE SUPERVISEE quand la grille existe. Il
        # reportait la perte auxiliaire calculee DANS la boucle PPO ; ce n'est
        # plus la qu'elle s'entraine, et laisser l'ancien compteur afficherait
        # une grandeur qui ne pilote plus rien.
        if np.isfinite(_rang_perte):
            epoch_aux_loss.append(_rang_perte)
        _chrono["rang"] = time.time() - _t_phase; _t_phase = time.time()
        # LE RECUIT DE LA PASSE SUPERVISEE AVANCE ICI, PAS AVEC CELUI DE PPO.
        #
        # Je l'avais pose a cote de `scheduler.step()`, sans verifier que la
        # passe `rang` s'execute 313 lignes PLUS LOIN dans la boucle. A
        # l'epoch 1 l'ordonnanceur avancait donc avant que `optimizer_rang`
        # n'ait fait un seul pas, et PyTorch saute alors la premiere valeur du
        # bareme — le recuit etait decale d'une epoch sur tout le run, avec un
        # avertissement a chaque lancement.
        #
        # Un ordonnanceur se fait avancer APRES l'optimiseur qu'il commande.
        # Les deux vivent donc chacun a la fin de LEUR phase.
        scheduler_rang.step()
        print(f"  {_col('phase', _C.GREY)}  rang {_chrono['rang']:.0f} s"
              + (f"   AuxL {_rang_perte:.4f} "
                 f"[mse {_rang_mse:.4f} topk {_rang_topk:.4f} "
                 f"ancre {_rang_ancre:.4f}]  gnorm {_rang_gnorm:.3f}  "
                 f"ecrete {100*_rang_clip:.0f}% (plafond {_plaf_rang:.2f})"
                 if _gg else ""))

        # ---------- PASSE 1 : calibration sur la policy COURANTE ----------
        # Calibrer sur l'epoch précédente ne marche pas : la policy bouge trop
        # vite (KL ~0.04, clipfrac ~9 %). Mesuré à l'epoch 7, la distribution de
        # p_BUY est passée de 0.338 à 0.283 d'une epoch à l'autre — la barre
        # héritée se retrouvait AU-DESSUS de toute la distribution courante, et
        # plus aucun BUY ne passait. D'où zéro trade long, deux runs de suite.
        #
        # On parcourt donc la fenêtre de validation une première fois en forçant
        # HOLD : l'agent reste flat du début à la fin, ce qui donne la
        # distribution complète des occasions sous la policy du moment. Les
        # barres en découlent, puis la passe 2 valide réellement.
        #
        # L'état du générateur est figé et restauré pour que les deux passes
        # portent sur EXACTEMENT les mêmes fenêtres.
        # FENETRES FIXES d'une epoch a l'autre.
        #
        # Les episodes de validation etaient tires au sort a chaque epoch, donc
        # une amelioration pouvait venir d'un scenario plus facile plutot que
        # d'un meilleur modele — sur 4 a 16 episodes, la variance de tirage
        # domine largement les ecarts qu'on cherche a mesurer. En figeant la
        # graine, deux epochs deviennent comparables.
        #
        # Deux graines distinctes pour les deux passes : elles portent sur des
        # fenetres DIFFERENTES (calibration puis mesure), et reutiliser la meme
        # les correlerait sans raison.
        etat_rng = np.random.get_state()
        np.random.seed(cfg.val_seed)

        # LA BANQUE SE RAFRAICHIT ICI, AVANT LA CALIBRATION — elle ne l'etait
        # qu'apres. Les poids du tronc ont bouge pendant l'epoch, donc les
        # representations mises en cache sont perimees. La calibration
        # tournait avec l'ancienne banque et la validation avec la nouvelle :
        # les barres etaient donc calibrees sur un modele legerement
        # different de celui qui les applique. Personne ne l'aurait vu.
        policy.rafraichit_banque()

        # UNE TABLE DE SCORES PRECALCULES A ETE ESSAYEE ICI, ET RETIREE.
        #
        # L'idee : la validation ne decide qu'a plat, ou les quatre colonnes
        # d'etat sont constantes, donc l'observation ne depend que du marche
        # et tous les scores se calculent d'avance par gros lots. Le calcul
        # etait juste — verifie, zero decision basculee a la selectivite
        # jouee — mais il est PLUS LENT : `validation 153 s` contre 142.
        #
        # L'ERREUR DE RAISONNEMENT. « Un reseau de 45 000 parametres coute
        # autant a lot 1 qu'a lot 8 192 » est vrai jusqu'a quelques dizaines
        # d'echantillons, faux a 8 192 : a cette taille le calcul redevient
        # reel. Et la table couvre TOUTE la fenetre — 74 107 barres — la ou
        # la boucle n'en visite que celles ou un environnement est a plat.
        # On payait donc plus de travail pour eviter des appels moins chers
        # qu'on ne le croyait.
        #
        # CE QUI RESTE A ESSAYER, si quelqu'un y revient : un lot
        # intermediaire — 256 ou 1 024 — et la table restreinte aux seules
        # barres que les episodes visitent. Le balayage n'a pas ete mene.
        # `ScoresAPlat` est conservee, inutilisee, avec son test.

        # PASSE 1 sur les environnements de CALIBRATION : la fenetre qui
        # PRECEDE celle de mesure. Les seuils en sortent, puis sont figes.
        v_states = []
        v_infos = []
        for e, d in zip(calib_envs, departs_calib):
            s0, i0 = reset_au_depart(e, d)
            v_states.append(s0)
            v_infos.append(i0)
        # LA CALIBRATION TOURNE A CHAQUE EPOCH, et ce n'est plus du gaspillage.
        #
        # Elle etait sautee quand la validation l'etait — 38 secondes pour un
        # resultat que personne ne lisait. C'etait vrai tant que le meilleur
        # checkpoint se choisissait sur le Sortino, qui n'existe que les
        # epochs validees.
        #
        # CE N'EST PLUS LE CAS. Le critere est desormais le CLASSEMENT, et il
        # est calcule a CHAQUE epoch sur les memes 3 273 decisions. Une epoch
        # non validee peut donc porter le meilleur modele du run — et sans
        # calibration fraiche on ne pourrait pas l'enregistrer, parce que le
        # `_calib.json` ecrit a cote porte les barres de decision : les
        # apparier a des poids d'une autre epoch ferait trader le checkpoint
        # sous une calibration qui n'est pas la sienne.
        #
        # Le cout est borne et connu : ~25 s sur les deux tiers des epochs,
        # soit environ +12 % de duree de run, contre la certitude de pouvoir
        # enregistrer le meilleur modele QUELLE QUE SOIT l'epoch ou il tombe.
        cal_active = list(range(len(calib_envs)))
        # PAS DE CALIBRATION — un quantile n'a pas besoin de chaque bougie.
        #
        # Cette passe ne sert qu'a estimer deux quantiles de la distribution de
        # conviction. Mesure du cout : un forward vaut ~39 ms quel que soit le
        # contenu en dessous de batch 16, et une epoch en lancait 18 000, dont
        # 12 000 pour les DEUX passes de validation — les deux tiers du budget
        # pour un huitieme des donnees. A raison d'une bougie sur cinq, cette
        # passe tombe de 6 000 a 1 200 forwards.
        #
        # L'environnement, lui, avance a CHAQUE bougie : sauter des pas de
        # marche fausserait la traversee. Seul le forward est espace, et il n'a
        # aucun effet sur la trajectoire puisque l'action est forcee a HOLD.
        # test_fenetre_vierge.py calibre deja ainsi, au meme pas.
        PAS_CALIB = 5
        pas_courant = 0
        with torch.no_grad():
            while cal_active:
                if pas_courant % PAS_CALIB == 0:
                    vb = np.stack([v_states[k] for k in cal_active], axis=0)
                    st = torch.as_tensor(vb, dtype=torch.float32, device=device)
                    masks_b = torch.from_numpy(
                        np.repeat(MASK_FLAT[None, :], len(cal_active), axis=0)
                    ).to(device)
                    _rend_c = policy.sorties(st)
                    # LA BARRE SE CALIBRE SUR LE SCORE QUI LA FRANCHIRA.
                    # Calibrer un quantile sur les probabilites de la politique
                    # puis juger la tete auxiliaire avec ne voudrait rien dire :
                    # deux distributions differentes, donc un quantile qui ne
                    # selectionne plus la fraction visee. Meme calcul des deux
                    # cotes, toujours.
                    # PLUS DE BASCULE : LA TETE DE RANG EST LA SEULE SOURCE.
                    # `tri_par_tete_aux` permettait de rebrancher les logits de
                    # l'acteur sur la decision. La direction supprimee, ce drapeau
                    # n'aurait plus offert qu'un moyen de rejouer le defaut qu'on
                    # vient de corriger — un `False` quelque part, et le rollout et
                    # le deploiement rejoueraient deux strategies differentes.
                    _a = _rend_c.float().cpu().numpy()
                    probs_np = 1.0 / (1.0 + np.exp(-np.clip(_a, -30, 30)))
                    for bi, k in enumerate(cal_active):
                        pbs_val[0].append(float(probs_np[bi, 0]))
                        pbs_val[1].append(float(probs_np[bi, 1]))
                pas_courant += 1
                suite = []
                for bi, k in enumerate(cal_active):
                    # La calibration force HOLD a chaque pas : aucune
                    # position n'y est ouverte, donc l'echelle n'y sert a rien.
                    ns, _r, done, _, info = calib_envs[k].step(2)   # HOLD forcé
                    v_states[k] = ns
                    v_infos[k] = info
                    if not done:
                        suite.append(k)
                cal_active = suite

        # LA REPRISE SE FAIT ICI, avant tout calcul qui lit `pbs_val`.
        #
        # Elle etait plus bas, apres la passe de validation — mais les seuils,
        # l'etendue et la specification de decision se calculent AVANT, et
        # `np.quantile` sur une liste vide leve une exception. Le run s'est
        # arrete a l'epoch 2 pour cette raison exacte : un patch qui saute une
        # passe doit fournir ce que cette passe produisait, au moment ou c'est
        # lu, pas plus tard.
        # `pbs_val` N'EST PLUS REPRIS : il vient de la calibration, qui tourne
        # maintenant a chaque epoch. Le reprendre ecraserait une distribution
        # FRAICHE, mesuree sous les poids courants, par celle d'il y a deux
        # epochs — exactement l'inverse de ce qu'on cherche. Seuls les
        # resultats de la passe de TRADING, elle toujours sautee, sont repris.
        if not _valide and _val_prec is not None:
            (val_pnl, val_dd, val_trades, val_trades_side,
             _, _val_epoch, val_fins) = _val_prec
            val_pnl, val_dd, val_fins = list(val_pnl), list(val_dd), list(val_fins)
            val_trades, val_trades_side = list(val_trades), list(val_trades_side)

        # Barres issues de CETTE distribution, une par côté, budget partagé.
        #
        # Une faible amplitude ne justifie pas de supprimer le filtre.
        # Garder un quantile par cote; une distribution constante ne declenche
        # aucune entree. Le mode legacy sert uniquement aux ablations.
        val_selectivity = (selectivite if cfg.validation_selectivity is None
                           else cfg.validation_selectivity)
        etendues = [float(np.quantile(pbs_val[c], 0.99)
                          - np.quantile(pbs_val[c], 0.01)) for c in (0, 1)]
        if cfg.legacy_flat_validation and min(etendues) < cfg.pbs_etendue_min:
            calib_thr_val[0] = calib_thr_val[1] = 0.0
        else:
            for c in (0, 1):
                calib_thr_val[c] = selective_threshold(pbs_val[c], val_selectivity / 2.0)
        val_etendue = float(min(etendues))

        _chrono["calibration"] = time.time() - _t_phase; _t_phase = time.time()

        # FILTRE PAR RANG GLISSANT, et non par niveau fige.
        #
        # Le niveau calibre sur la fenetre precedente ne transferait pas : mesure
        # sur un run reel, l'etendue des convictions est passee de 0.0035 a
        # 0.0914 entre les epochs 6 et 12 pendant que le seuil restait a ~0.24,
        # et le nombre de trades s'est effondre de 1 461 a 20, dont zero vente.
        #
        # Un rang ne depend d'aucune echelle absolue : « cette occasion est-elle
        # dans les 5 % les plus convaincues des 500 dernieres vues ». La fenetre
        # est amorcee avec les convictions de la passe de calibration, qui la
        # precede chronologiquement — donc aucune information future.
        # (La banque a ete rafraichie plus haut, AVANT la calibration.)

        # LA SELECTIVITE SE DIVISE PAR LE NOMBRE DE COTES OUVERTS, pas par
        # deux. `val_selectivity / 2.0` supposait un run bilateral : en
        # long-only il ne reste qu'un cote, et diviser quand meme faisait
        # trader l'agent a la MOITIE de la frequence demandee — 2.5 % la ou
        # la configuration en demande 5. Le cote ferme n'a besoin d'aucune
        # part du budget puisque `EntryDecisionPolicy` le refuse desormais.
        _n_cotes = max(sum(cotes_permises(cfg.side)), 1)
        decision_spec = rolling_decision_spec(
            val_selectivity / _n_cotes, cfg.rang_fenetre, pbs_val, cfg.side)
        val_decisions = [EntryDecisionPolicy(decision_spec) for _ in range(n_val)]

        # ---------- PASSE 2 : validation réelle, rang glissant ----------
        # Fenetre posterieure a celle de calibration, graine distincte mais
        # constante d'une epoch a l'autre.
        #
        # Sautee une epoch sur `validation_tous_les`. Les compteurs gardent
        # alors les valeurs de la derniere mesure : le journal affiche donc la
        # validation la plus recente, jamais des zeros qu'on lirait comme un
        # effondrement.
        np.random.seed(cfg.val_seed + 1)
        # LE TIRAGE DE LA VALIDATION : meme graine a chaque epoch. Voir
        # `evaluation_stochastique`.
        _explore_v = bool(getattr(cfg, "evaluation_stochastique", True))
        _gen_v = torch.Generator(device=device)
        _gen_v.manual_seed(int(cfg.val_seed) + 2)
        v_states = []
        v_infos = []
        for e, d in zip(val_envs, departs_val):
            s0, i0 = reset_au_depart(e, d)
            v_states.append(s0)
            v_infos.append(i0)

        v_active = list(range(n_val)) if _valide else []
        # LA CADENCE DE SORTIE, par environnement de validation. Voir
        # `cadence_sortie` : sans elle la validation interrogeait la
        # politique a chaque barre, quinze fois plus souvent qu'elle
        # n'avait appris a l'etre.
        v_frais = np.ones(max(n_val, 1), dtype=bool)
        v_depuis = np.zeros(max(n_val, 1), dtype=np.int64)
        _pas_v = max(1, int(getattr(cfg, "pas_decision_sortie", 1)))

        # Même principe qu'au rollout : la policy n'est sollicitée que sur les
        # envs flat. En position l'action est forcée, le forward serait jeté.
        with torch.no_grad():
            while v_active:
                # DES QU'UNE PLACE EST LIBRE, comme l'entrainement et le
                # live. La condition etait `position == 0` : le compte ne
                # decidait que s'il etait ENTIEREMENT plat, donc il restait
                # assis pendant toute la vie d'une position — vingt-cinq
                # heures en mediane.
                #
                # CE QUE CELA FAUSSAIT. L'entrainement joue un essaim
                # (`peut_entrer()`, ligne ~4927) pendant que la validation
                # jouait UNE position a la fois. On entrainait donc une
                # strategie et on en mesurait une autre. Le creux le disait
                # sans qu'on l'ecoute : 57 a 76 % a l'entrainement contre 12
                # a 20 % en validation.
                #
                # CE QUE CELA COUTAIT EN TRADES. A 5 % de selectivite la
                # fenetre offre 164 occasions, soit 20 trades par mois ; le
                # run n'en produisait que 4.4. Le facteur 4.5 manquant etait
                # ce temps d'inactivite, et non un choix de geometrie.
                #
                # `peut_entrer()` demande au SOLDE, pas au tableau : c'est la
                # meme regle partagee que `kairos_live` appelle desormais.
                deciding = [k for k in v_active if val_envs[k].peut_decider()]
                v_actions = {k: 2 for k in v_active}
                # LA MEME SORTIE QU'AU ROLLOUT, par la meme fonction. Une
                # validation qui ne fermerait pas mesurerait une strategie
                # que l'entrainement ne joue pas — ce depot l'a deja paye.
                _vp = [k for k in v_active
                       if k not in deciding and val_envs[k].n_positions > 0]
                for _k in v_active:
                    if val_envs[_k].n_positions == 0:
                        v_frais[_k] = True
                _vd, _ = cadence_sortie(_vp, v_frais, v_depuis, _pas_v)
                if _vd:
                    # LA POLITIQUE EST TIREE, graine fixe : voir
                    # `evaluation_stochastique`.
                    _va, _, _ = decide_sortie(
                        policy, [v_states[k] for k in _vd], device,
                        N_BASE_FEATURES, explore=_explore_v,
                        generateur=_gen_v)
                    for _bi, _k in enumerate(_vd):
                        v_frais[_k] = False
                        if int(_va[_bi]) == FERMER:
                            v_actions[_k] = 3

                if deciding:
                    # LA MEME POLITIQUE QU'A L'ENTRAINEMENT, tiree avec
                    # une graine fixe — voir `evaluation_stochastique`.
                    # `barre` compte desormais les attentes que la politique
                    # a CHOISIES, `capacite` celles que le solde imposait.
                    _m3, _sans = masque_entree(
                        np.repeat(MASK_FLAT[None, :], len(deciding), axis=0),
                        [val_envs[k] for k in deciding])
                    _va_e, _, _ = decide_entree(
                        policy, [v_states[k] for k in deciding], _m3, device,
                        explore=_explore_v, generateur=_gen_v)
                    _portillons["occasions"] += len(deciding)
                    _portillons["capacite"] += int(_sans.sum())
                    for bi, k in enumerate(deciding):
                        a = int(_va_e[bi])
                        if a == ATTENDRE:
                            _portillons["barre"] += 1
                        else:
                            _portillons["entrees"] += 1
                        v_actions[k] = a

                still = []
                for k in v_active:
                    ns, _r, done, _, info = val_envs[k].step(v_actions[k])
                    v_states[k] = ns
                    v_infos[k] = info
                    if not done:
                        still.append(k)
                v_active = still

        # Trades groupes par SOUS-PERIODE, pas seulement en un total.
        #
        # Les 19 episodes de validation sont disjoints et couvrent la fenetre
        # dans l'ordre du temps : les regrouper par quatre donne quatre
        # sous-periodes consecutives d'environ trois mois. Un total unique
        # cache ce qui compte le plus ici — un modele qui gagne sur une periode
        # et perd sur les trois autres affiche la meme moyenne qu'un modele
        # regulier, et c'est exactement le piege dans lequel TabM est tombe
        # (tout son avantage apparent venait d'un bloc haussier sur quatre).
        trades_par_bloc = [[] for _ in range(4)]

        for k, ve in enumerate(val_envs):
            trades_par_bloc[min(k * 4 // max(len(val_envs), 1), 3)].extend(
                ve.trades_pnl)
            final_close = ve.data.close[ve.idx - 1] if ve.idx > 0 else ve.data.close[-1]

            latent = 0.0
            if ve.position != 0 and ve.current_size > 0 and ve.entry_price > 0:
                latent = (
                    ve.position *
                    (final_close - ve.entry_price) *
                    ve.current_size
                )

            final_equity = ve.capital + latent
            val_pnl.append(final_equity - cfg.initial_capital)
            val_dd.append(v_infos[k]["drawdown"])
            val_fins.append(str(v_infos[k].get("done_reason") or "episode_end"))
            val_trades.extend(ve.trades_pnl)
            val_trades_side.extend(ve.trades_side)
            for _tm in ve.trades_meta:
                (val_tenues_g if _tm["pnl"] > 0 else val_tenues_p).append(
                    int(_tm["hold_bars"]))

            # Trade-by-trade CSV (VAL)
            with open(trades_csv_path, "a", newline="", encoding="utf-8") as _ft:
                _w = _csv.DictWriter(_ft, fieldnames=_trades_fields)
                for tm in ve.trades_meta:
                    _w.writerow({
                        "epoch": epoch, "phase": "val", "episode": k,
                        "entry_idx": tm["entry_idx"], "exit_idx": tm["exit_idx"],
                        "side": tm["side"],
                        "entry_price": round(tm["entry_price"], 4),
                        "exit_price": round(tm["exit_price"], 4),
                        "pnl": round(tm["pnl"], 4),
                        "hit_sl": int(tm["hit_sl"]), "hit_tp": int(tm["hit_tp"]),
                        "hold_bars": tm["hold_bars"],
                    })

        # QUAND LA VALIDATION EST SAUTEE, on reprend la derniere mesure.
        #
        # Sans cela les compteurs seraient vides et le journal afficherait des
        # zeros — 0 trade, PnL nul, Sortino nul — qu'on lirait comme un
        # effondrement du modele alors que rien n'aurait ete mesure. Afficher
        # la derniere valeur connue est honnete : le marqueur `(mesure a
        # l'epoch N)` dit d'ou elle vient.
        if _valide:
            _val_epoch = epoch
            _val_prec = (list(val_pnl), list(val_dd), list(val_trades),
                         list(val_trades_side), [list(pbs_val[0]),
                                                 list(pbs_val[1])], epoch,
                         list(val_fins))

        # Recalibration pour l'epoch SUIVANTE, sur la fenêtre de validation.
        # Même garde-fou : filtrer une distribution plate revient à tirer au sort.
        val_profit = float(sum(val_pnl))
        val_max_dd = float(max(val_dd) if val_dd else 0.0)
        # UN COMPTE DETRUIT, c'est-a-dire une fin que le COURTIER aurait
        # imposee : appel de marge, impossibilite de financer le lot minimum,
        # ou equite sous zero. Une fin de tranche n'en est pas une.
        _FINS_RUINE = ("ruine", "appel_de_marge", "solde_insuffisant")
        val_ruine = int(sum(1 for f in val_fins if f in _FINS_RUINE))
        val_num_trades = len(val_trades)
        val_winrate = (
            float(np.mean([t > 0 for t in val_trades]))
            if val_num_trades > 0 else 0.0
        )

        wins_val   = [t for t in val_trades if t > 0]
        losses_val = [t for t in val_trades if t <= 0]
        avg_win_val  = float(np.mean(wins_val))   if wins_val  else 0.0
        avg_loss_val = float(np.mean(losses_val)) if losses_val else 0.0
        num_loss_val = len(losses_val)
        total_profit_val  = float(sum(wins_val))
        total_loss_val    = float(sum(losses_val))
        profit_factor_val = total_profit_val / (abs(total_loss_val) + 1e-8)

        if val_num_trades > 10:
            rets = np.array(val_trades, dtype=np.float32) / cfg.initial_capital
            mean_ret = float(rets.mean())
            sortino = downside_score(rets)
        else:
            sortino = 0.0

        metric = sortino
        metric_history.append(metric)
        if len(metric_history) >= 30:
            recent_metric = float(np.mean(metric_history[-30:]))
        else:
            recent_metric = float(np.mean(metric_history))

        train_loss_rate = (1 - winrate_epoch) if num_trades_epoch > 0 else 0.0
        val_loss_rate   = (1 - val_winrate)   if val_num_trades  > 0 else 0.0

        # --------- Split LONG / SHORT ---------
        train_split = _split_by_side(epoch_trades_pnl, epoch_trades_side)
        val_split   = _split_by_side(val_trades,       val_trades_side)

        train_long_w  = len(train_split["long"]["wins"])
        train_long_l  = len(train_split["long"]["losses"])
        train_short_w = len(train_split["short"]["wins"])
        train_short_l = len(train_split["short"]["losses"])
        train_long_pnl  = train_split["long"]["pnl"]
        train_short_pnl = train_split["short"]["pnl"]

        val_long_w  = len(val_split["long"]["wins"])
        val_long_l  = len(val_split["long"]["losses"])
        val_short_w = len(val_split["short"]["wins"])
        val_short_l = len(val_split["short"]["losses"])
        val_long_pnl  = val_split["long"]["pnl"]
        val_short_pnl = val_split["short"]["pnl"]

        # LE COTE INTERDIT DOIT AFFICHER ZERO TRADE, et rien ne le verifiait.
        # or_exec02 a tourne 91 epochs x 3 folds en long-only avec 31 a 68 %
        # de VENTES en validation : l'entrainement montrait bien S(0W/0L),
        # mais personne ne comparait les deux lignes. Le controle coute une
        # comparaison par epoch et aurait economise le run entier.
        _pb, _ps = cotes_permises(cfg.side)
        for _nom, _permis, _n in (("achats", _pb, val_long_w + val_long_l),
                                  ("ventes", _ps, val_short_w + val_short_l)):
            if not _permis and _n:
                raise RuntimeError(
                    f"side={cfg.side!r} interdit les {_nom}, or la validation "
                    f"en compte {_n}. La regle de decision et le masque "
                    f"d'actions ne decrivent plus la meme strategie.")

        # --------- Couleurs / styles ---------
        tag       = _col(f"[{cfg.side.upper()}{suffix}]", _C.MAGENTA + _C.BOLD)
        epoch_str = _col(f"EPOCH {epoch:03d}", _C.CYAN + _C.BOLD)

        # Couleur du Sortino30 selon valeur
        s30 = recent_metric
        if s30 >= 0.05:
            s30_col = _C.GREEN + _C.BOLD
        elif s30 >= 0.0:
            s30_col = _C.GREEN
        elif s30 >= -0.05:
            s30_col = _C.YELLOW
        else:
            s30_col = _C.RED

        # Couleur du DD
        def _dd_col(d):
            if d < 0.2:   return _C.GREEN
            if d < 0.5:   return _C.YELLOW
            return _C.RED

        # ----- Ligne 1 : TRAIN -----
        _ligne_train = (
            f"{tag} {epoch_str}  "
            f"{_col('TRAIN', _C.WHITE + _C.BOLD)}  "
            f"PNL {_money(profit_epoch, width=10)}  "
            f"trades={num_trades_epoch:>4d}  "
            f"WR {_pct(winrate_epoch)}  "
            f"PF {profit_factor_train:>4.2f}  "
            f"DD {_col(_pct(max_dd_epoch), _dd_col(max_dd_epoch))}  "
            f"{_col('L', _C.GREEN)}({_col(str(train_long_w), _C.GREEN)}W/"
            f"{_col(str(train_long_l), _C.RED)}L) {_money(train_long_pnl, width=9)}  "
            f"{_col('S', _C.BLUE)}({_col(str(train_short_w), _C.GREEN)}W/"
            f"{_col(str(train_short_l), _C.RED)}L) {_money(train_short_pnl, width=9)}"
        )
        print(_ligne_train)

        # ----- Ligne 2 : VAL -----
        _ligne_val = (
            f"{tag} {epoch_str}  "
            f"{_col('VAL  ', _C.WHITE + _C.BOLD)}  "
            f"PNL {_money(val_profit, width=10)}  "
            f"trades={val_num_trades:>4d}  "
            f"WR {_pct(val_winrate)}  "
            f"PF {profit_factor_val:>4.2f}  "
            f"DD {_col(_pct(val_max_dd), _dd_col(val_max_dd))}  "
            f"{_col('L', _C.GREEN)}({_col(str(val_long_w), _C.GREEN)}W/"
            f"{_col(str(val_long_l), _C.RED)}L) {_money(val_long_pnl, width=9)}  "
            f"{_col('S', _C.BLUE)}({_col(str(val_short_w), _C.GREEN)}W/"
            f"{_col(str(val_short_l), _C.RED)}L) {_money(val_short_pnl, width=9)}"
        )
        print(_ligne_val)

        # ----- Ligne 3 : METRICS PPO -----
        # On rend au generateur son etat d'avant validation : les graines fixes
        # ci-dessus ne doivent pas rendre deterministes les episodes
        # d'ENTRAINEMENT, qui eux doivent varier.
        np.random.set_state(etat_rng)

        _chrono["validation"] = time.time() - _t_phase

        # L'EPOCH DE MOYENNE N'A PAS DE STATISTIQUES D'OPTIMISATION : aucune
        # mise a jour PPO n'y a eu lieu, donc np.mean([]) vaut nan, et `H nan`
        # ne correspond a aucun motif numerique. La veille laisserait tomber la
        # ligne en silence — c'est-a-dire precisement l'epoch deployee. On
        # reprend donc les colonnes d'optimisation de la derniere epoch
        # entrainee, et le mot MOYENNE en FIN de ligne dit ce qu'il en est ;
        # il est place apres tous les champs lus, pour ne casser aucun motif.
        if epoch_moyenne and stats_precedentes is not None:
            (epoch_actor_loss, epoch_critic_loss, epoch_entropy,
             epoch_entropy_flat, epoch_kl, epoch_grad_norm) = stats_precedentes

        # ---- `H` ET `Hflat` : L'ENTROPIE DE LA REGLE D'ENTREE ----
        #
        # Les deux champs portaient l'entropie des logits de l'ACTEUR. L'acteur
        # ne decide plus rien : le nombre aurait continue de s'afficher, aurait
        # continue de bouger — le tronc change — et n'aurait plus rien mesure
        # de ce qui se passe. Un indicateur mort qui garde l'air vivant est
        # pire qu'un champ absent ; c'est exactement ce qui m'a fait annoncer
        # « ACTOR GELE » sur un run qui apprenait.
        #
        # Ils portent maintenant l'entropie de la REPARTITION DES DECISIONS
        # d'entree sur l'epoch : acheter, vendre, attendre. Elle repond a la
        # meme question que l'ancienne — la regle s'est-elle figee ? — mais sur
        # la regle qu'on DEPLOIE. A 0, le modele fait toujours la meme chose
        # (jamais d'entree, ou une entree a chaque occasion) ; au plafond
        # ln(cotes+1), il est a pile ou face. Le plafond affiche par la ligne
        # META est deja celui-la, il n'a pas eu a changer.
        #
        # LES DEUX CHAMPS SONT EGAUX, et c'est voulu : `Hflat` valait `H`
        # restreint aux etats plats, or dans un semi-MDP toute decision est
        # prise a plat. Ils l'etaient deja a 0.001 pres. On garde les deux
        # parce que la veille lit l'un et l'autre, et qu'un champ retire est
        # une veille muette de plus.
        _tot_e = sum(entrees_epoch)
        if _tot_e > 0:
            _pe = np.array(entrees_epoch, float) / _tot_e
            _pe = _pe[_pe > 0]
            _H_entree = float(-(_pe * np.log(_pe)).sum())
            epoch_entropy = [_H_entree]
            epoch_entropy_flat = [_H_entree]

        # Ecart au point mort, SOUS-PERIODE PAR SOUS-PERIODE. Le point mort est
        # recalcule dans chaque bloc sur ses propres gains et pertes : un bloc
        # ou l'on gagne gros et rarement n'a pas le meme seuil qu'un bloc ou
        # l'on gagne petit et souvent, et comparer les deux a un seuil commun
        # melangerait deux questions.
        blocs_ecart = []
        for _tb in trades_par_bloc:
            _a = np.array(_tb, float)
            if len(_a) < 15:
                blocs_ecart.append(float("nan"))
                continue
            _g, _p = _a[_a > 0], _a[_a <= 0]
            if not len(_g) or not len(_p):
                blocs_ecart.append(float("nan"))
                continue
            _aw, _al = float(_g.mean()), abs(float(_p.mean()))
            blocs_ecart.append(100.0 * len(_g) / len(_a)
                               - 100.0 * _al / (_aw + _al))
        _bl = " ".join("  nan" if np.isnan(x) else f"{x:+5.1f}"
                       for x in blocs_ecart)
        _npos = sum(1 for x in blocs_ecart if np.isfinite(x) and x > 0)
        _nval = sum(1 for x in blocs_ecart if np.isfinite(x))

        # Le rho de l'epoch. Etat PLAT, celui que voit la politique avant
        # d'entrer : position 0, latent 0, detention 0, echelle de risque 1.
        _rho_aux = float(chr(110) + chr(97) + chr(110))
        _gain_top = _gain_tous = float(chr(110) + chr(97) + chr(110))
        # SANS GRILLE, PAS DE SCORE NET — et surtout pas un 0 qui passerait
        # pour un resultat. `retient_checkpoint` refuse un score non
        # mesurable, donc un nan ferme le portillon au lieu de l'ouvrir.
        _gain_net = _creux_grille = _score_net = float(
            chr(110) + chr(97) + chr(110))
        _marge_hasard = 0.0
        _n_eff = 0
        _sa = []
        if _rang_idx is not None:
            policy.eval()
            # Etat A PLAT : aucune position, capacite pleine, aucun creux.
            #
            # LA LARGEUR VIENT DE `N_POS_FEATURES`, ET NON D'UN 5 EN DUR.
            # Elle etait ecrite en dur, et ajouter une sixieme colonne d'etat
            # a fait tomber le run sur un `RuntimeError` de dimension — la
            # seule chance dans l'affaire, car un bloc trop COURT leve, la ou
            # un bloc mal REMPLI se serait tu. Toute colonne future doit donc
            # etre initialisee ici explicitement.
            _extra = np.zeros((cfg.lookback, N_POS_FEATURES), np.float32)
            _extra[:, 3] = 1.0          # capacite pleine
            _s = []
            with torch.no_grad():
                for _d in range(0, len(_rang_idx), 8192):
                    _b = _rang_idx[_d:_d + 8192]
                    _o = np.stack([
                        np.concatenate([val_data.features[i - cfg.lookback:i],
                                        _extra], axis=-1) for i in _b])
                    _t = torch.from_numpy(_o).to(device)
                    # UN SEUL PASSAGE : cette boucle traversait le tronc
                    # deux fois par lot de 8 192, pour la direction puis pour
                    # le rendement.
                    _rend_r = policy.sorties(_t)
                    # `rho` MESURE DESORMAIS LE BUDGET, PAS LA DIRECTION.
                    #
                    # Il portait la correlation de rang entre le score de la
                    # POLITIQUE de direction et le rendement reel. Cette tete
                    # est supprimee : le chiffre aurait continue de s'afficher
                    # et de bouger — le tronc change — sans plus rien mesurer.
                    #
                    # La question qu'il posait — « cette tete classe-t-elle
                    # les occasions ? » — se pose maintenant de la seule tete
                    # que PPO entraine encore : le BUDGET. On correle donc le
                    # risque ESPERE (la moyenne des paliers ponderee par leurs
                    # probabilites, pas l'argmax, qui jetterait toute la
                    # nuance de la distribution) au rendement reel de
                    # l'occasion.
                    #
                    # CE QU'IL DIT. rho > 0 : le modele mise gros quand
                    # l'occasion paie — c'est le seul endroit du journal ou
                    # cela se lit. rho ~ 0 avec un `sommet` positif : la
                    # selection marche et le dimensionnement suit le hasard,
                    # donc tout le gain vient de la tete de rang. rho < 0 : il
                    # mise gros quand il perd, et le creux le dira aussi, mais
                    # plus tard.
                    #
                    # `rhoAux` reste la correlation de la TETE DE RANG, celle
                    # qui decide des entrees. Les deux colonnes restent donc
                    # deux mesures distinctes de deux tetes distinctes, comme
                    # avant — mais toutes deux sur des tetes vivantes.
                    # `_s` PORTAIT L'ESPERANCE DE LA TETE DE BUDGET. Cette
                    # tete n'existe plus : la part se deduit du rang, plus
                    # bas, a partir de `_top`. Rien a collecter ici.
                    # LE PALIER QUI SERAIT REELLEMENT POSE, par ARGMAX.
                    #
                    # C'est la regle du deploiement : le rollout TIRE le
                    # budget, la validation, le test et le live prennent le
                    # maximum. Le score de retenue doit mesurer la strategie
                    # qu'on joue, donc l'argmax — c'est le principe que tout
                    # ce depot applique, et qu'on vient d'appliquer a
                    # l'entree.
                    #
                    # POURQUOI `rho` GARDE L'ESPERANCE, lui. Il est
                    # diagnostique et non decisif : une correlation de rang
                    # sur six paliers discrets serait constante — donc nan —
                    # tant que la tete n'a pas commence a moduler, et un nan
                    # dans la ligne META rend la veille muette. L'esperance
                    # bouge des la premiere epoch et dit la meme chose.

                    # LA TETE AUXILIAIRE CLASSE-T-ELLE MIEUX QUE LA POLITIQUE ?
                    #
                    # Mesure du 2026-09-16 : une regression lineaire simple
                    # obtient rho +0.0434 (2.0 sigma) sur cette cible, quand
                    # le rho de la politique oscille dans le bruit. L'info est
                    # donc dans les features et PPO ne l'extrait pas — il
                    # optimise le rendement de ses ACTIONS, personne ne lui
                    # demande que l'ORDRE de ses probabilites soit juste, et
                    # c'est pourtant tout ce dont la selectivite se sert.
                    #
                    # La tete auxiliaire, elle, est entrainee a PREDIRE le
                    # rendement, sur tous les etats collectes et non sur les
                    # seuls etats de decision. Si son rho monte pendant que
                    # celui de la politique reste plat, c'est elle qui doit
                    # trier en production.
                    try:
                        _a = _rend_r.float().cpu().numpy()
                        _sa.append(_a[:, 0] - _a[:, 1])
                    except Exception:
                        _sa = None
            policy.train()
            # LE TEMOIN : ce que la tete de budget aurait mise. Il ne commande
            # plus rien quand `budget_par_rang` est vrai, mais il reste mesure
            # a chaque epoch — sans quoi on ne saurait pas si la regle qu'on
            # remplace s'etait mise a marcher.
            # `rho` SE MESURE SUR LA PART DEPLOYEE, calculee plus bas a
            # partir de `_top`. Ici il n'y a plus rien a correler : la tete
            # de budget a ete retiree du reseau.
            if _sa:
                _rho_aux = _correlation_rang(np.concatenate(_sa), _rang_reel)

            # ============================================================
            # CE QUE LE SOMMET RAPPORTE, et c'est le critere de selection.
            #
            # POURQUOI PAS LE RHO SEUL. Un rho eleve dit que l'ORDRE est bon
            # sur toute la fenetre. Il ne dit pas que les 5 % retenus
            # gagnent : on peut bien ordonner l'ensemble et avoir un sommet
            # qui ne paie pas. Or c'est le sommet, et lui seul, qui est
            # trade.
            #
            # POURQUOI PAS LE PNL DE VALIDATION. Il porte sur ~55 trades,
            # soit +-2.81 $ d'erreur-type pour un gain mesure de +5.30 $ —
            # 1.9 ecart-type. Selectionner la-dessus quatre-vingt-dix fois,
            # c'est attraper un pic de chance, et ce depot l'a chiffre :
            # -3.3 points au test.
            #
            # CE QU'ON PREND. Le rendement MOYEN, en unites de risque, des
            # occasions que ce checkpoint mettrait effectivement en position
            # — les `selectivite` % les mieux notees. C'est de la
            # rentabilite, sur l'ensemble reellement joue, mesuree sur ~164
            # observations au lieu de 55, et sans la variance que le
            # simulateur ajoute par les emplacements et le budget.
            #
            # Mesure anterieure du depot, meme grandeur : le sommet a 5 %
            # rapporte +0.41 R par occasion contre +0.079 au hasard.
            # ============================================================
            # PAS DE REPLI. `_s` porte maintenant le budget espere ;
            # classer les occasions avec lui mesurerait un `sommet` que
            # personne ne trade, et il serait retenu comme critere de
            # selection du meilleur modele. Sans tete de rang, il n'y a pas
            # de strategie a mesurer — on le dit au lieu de le maquiller.
            if not _sa:
                raise RuntimeError(
                    "la tete de rang n'a rien produit : c'est elle qui decide "
                    "des entrees depuis la suppression de la direction, et le "
                    "`sommet` qui selectionne le meilleur modele se calcule "
                    "sur son classement.")
            _scores_sel = np.concatenate(_sa)
            # LE SENS ET LA CONVICTION SE SEPARENT, parce que le score est
            # signe : `_sa` porte `achat - vente`, donc son SIGNE dit quel
            # cote prendre et sa VALEUR ABSOLUE dit avec quelle force.
            #
            # Trier sur le score brut revenait a ne retenir que des achats.
            # On trie sur la conviction, et le gain suit le sens choisi.
            #
            # UN SEUL COTE PERMIS : le comportement d'origine est garde au
            # bit pres — le sens est constant, et la conviction est le score
            # oriente dans ce sens, donc `argsort` retient exactement les
            # memes occasions qu'avant.
            _ca_s, _cv_s = cotes_permises(cfg.side)
            if _ca_s and _cv_s:
                _sens_sel = np.where(_scores_sel >= 0.0, 1.0, -1.0)
                _conv_sel = np.abs(_scores_sel)
            elif _cv_s:
                _sens_sel = np.full(len(_scores_sel), -1.0)
                _conv_sel = -_scores_sel
            else:
                _sens_sel = np.full(len(_scores_sel), 1.0)
                _conv_sel = _scores_sel
            # CE QUE LA POSITION ENCAISSE, une fois le sens choisi.
            # LE RENDEMENT DE CHAQUE OCCASION SOUS LA SORTIE QUE LE MODELE
            # JOUE A CETTE EPOCH. Voir `rendements_sortie_ppo` : l'ancien
            # calcul tenait chaque position `horizon_cloture` barres, une
            # strategie que personne ne joue depuis que la sortie est PPO.
            # Le « hasard » est la moyenne du MEME tableau, donc les deux
            # termes de la comparaison changent ensemble.
            _rang_ra, _rang_rv, _rang_tenue = rendements_sortie_ppo(
                policy, val_data, _rang_idx, cfg, device)
            _rang_ra = np.nan_to_num(_rang_ra, nan=0.0)
            _rang_rv = np.nan_to_num(_rang_rv, nan=0.0)
            # LA TENUE DU SENS QUE LE TRI CHOISIT pour chaque occasion.
            _rang_tenue = np.where(_sens_sel > 0.0,
                                   _rang_tenue[:, 0], _rang_tenue[:, 1])
            print(f"  {_col('phase', _C.GREY)}  critere  sortie PPO rejouee sur "
                  f"{len(_rang_idx):,} occasions — tenue mediane "
                  f"{float(np.median(_rang_tenue)):.0f} barres, "
                  f"{100*float(np.mean(_rang_tenue < int(getattr(cfg, 'tenue_max_cloture', 0)))):.0f} % "
                  f"fermees par la politique")
            _rang_gain = np.where(_sens_sel > 0.0, _rang_ra, _rang_rv)
            # LA MEME SELECTIVITE QUE LA VALIDATION APPLIQUE, sans quoi on
            # jugerait un sommet que personne ne trade.
            _q = float(np.clip(val_selectivity, 0.01, 1.0))
            _n_top = max(int(round(_q * len(_scores_sel))), 20)
            _top = np.argsort(_conv_sel)[-_n_top:]

            # ============================================================
            # LA PART DEDUITE DU RANG — l'option C, sur la grille dense.
            #
            # LE RANG EST CELUI DE L'OCCASION DANS LA GRILLE ENTIERE, pas
            # dans les seules retenues : c'est la grandeur que la fenetre
            # glissante estime en validation et en live, et les deux doivent
            # mesurer la meme chose.
            #
            # `argsort(argsort(x))` rend le rang 0-base ; le `+0.5` centre
            # chaque occasion dans son intervalle, donc aucune ne tombe
            # exactement sur la barre par accident d'indexation.
            # ============================================================
            if True:   # la part vient du rang, et de nulle part d'ailleurs
                # `rho` A ETE RETIRE LE 2026-09-21, AVEC LE BUDGET.
                #
                # Il correlait la PART DEPLOYEE au rendement de l'occasion :
                # « le modele mise-t-il plus la ou ca paie ? ». Sans paliers
                # il n'y a plus de part a correler — toute occasion retenue
                # est prise a la meme taille, celle du lot minimum.
                #
                # IL NE MESURAIT DEJA PLUS RIEN. Sur six epochs du fold 1 :
                # +0.0058, +0.0041, +0.0184, -0.0117, +0.0071, +0.0009, pour
                # un bruit de +/- 0.024. La veille ecrivait « n'ordonne
                # rien » a chaque epoch, et elle avait raison.
                #
                # CE QUI RESTE POUR JUGER LE CLASSEMENT : `rhoAux`, qui
                # correle le SCORE de la tete au rendement, et `sommet`, qui
                # dit ce que rapporte le haut du tri. Ni l'un ni l'autre ne
                # depend d'une taille de mise.
                _gain_top = float(np.mean(_rang_gain[_top]))
            _gain_tous = float(np.mean(_rang_gain))
            # LE SOMMET DES ENTREES SEULES, SORTIE FIXE A L'HORIZON DE LA
            # CIBLE. Un DIAGNOSTIC, il ne decide rien : la sauvegarde juge la
            # strategie jouee, sortie PPO comprise.
            #
            # POURQUOI IL EXISTE. Six epochs sur six, la sortie a ferme a la
            # premiere minute ; le sommet ci-dessus mesurait donc le spread
            # plus une minute de bruit, et ne pouvait pas dire si les
            # entrees — entrainees a 60 minutes — valent quelque chose.
            # Celui-ci le dit : les MEMES occasions, le MEME tri, le MEME
            # sens, tenues jusqu'a l'horizon.
            if _rang_ra60 is not None and len(_rang_ra60) == len(_sens_sel):
                _g60 = np.where(_sens_sel > 0.0, _rang_ra60, _rang_rv60)
                _g60_top = float(np.mean(_g60[_top]))
                _g60_tous = float(np.mean(_g60))

            # ============================================================
            # L'INCERTITUDE DU SOMMET, ET ELLE EST GRANDE.
            #
            # `_gain_top` est une moyenne sur ~1 074 occasions retenues —
            # mais elles SE CHEVAUCHENT. Une position tient 396 barres en
            # mediane et la grille pose une occasion toutes les 12 barres :
            # trente-trois occasions consecutives decrivent donc le MEME
            # mouvement. Diviser par racine(1074) surestimerait la precision
            # d'un facteur six.
            #
            # ON COMPTE LES OCCASIONS REELLEMENT DISJOINTES, par un placement
            # glouton sur leurs indices de barre : on prend la premiere, on
            # saute tout ce qui commence avant sa cloture, on recommence.
            # C'est exact, et ca ne suppose rien sur leur repartition.
            _idx_top = np.sort(np.asarray(_rang_idx, dtype=np.int64)[_top])
            # L'HORIZON VIENT DE LA GEOMETRIE, PLUS D'UNE CONSTANTE A PART.
            #
            # `duree_trade_barres` valait 202, mesure sur l'ancienne sortie
            # au stop suiveur. Le scalping M1 borne la detention a
            # `horizon_cloture`, et deux occasions plus eloignees que cette
            # duree ne partagent alors AUCUNE barre.
            #
            # CE QUE LA CONSTANTE PERIMEE COUTAIT : le placement glouton
            # comptait 126 occasions independantes la ou il y en a ~6.7 fois
            # plus, donc une marge sur-estimee d'un facteur racine(202/30) =
            # 2.6. A l'epoch 1, exces +0.026 contre marge 0.047 — refuse
            # alors que la marge juste vaut ~0.018.
            #
            # `_borne_syst` est deja la source unique de cette duree pour la
            # grille et le diagnostic ; elle l'est aussi ici.
            _h = int(max(_borne_syst(cfg), 1))
            _n_eff, _fin = 0, -1
            for _i in _idx_top:
                if _i >= _fin:
                    _n_eff += 1
                    _fin = _i + _h
            _sd_top = (float(np.std(_rang_gain[_top], ddof=1))
                       if _n_top > 1 else float("nan"))
            # UN SIGMA, ET C'EST UN PLANCHER, PAS UN TEST.
            #
            # Le checkpoint retenu est le MAXIMUM sur ~45 epochs : meme a
            # deux sigmas par epoch, le maximum de quarante-cinq tirages
            # depasserait souvent le seuil. Cette marge n'etablit donc pas
            # une signification — elle empeche seulement de retenir un ecart
            # que le bruit explique a lui seul, ce que le portillon a
            # laisse passer huit fois de suite sur le fold 2.
            _marge_hasard = (_sd_top / math.sqrt(max(_n_eff, 1))
                             if np.isfinite(_sd_top) and _n_eff >= 2
                             else 0.0)
            # ============================================================

            # ============================================================
            # CE QUE LE MODELE ENGAGE, ET CE QUE CA LUI COUTE QUAND IL SE
            # TROMPE.
            #
            # LE TROU QUE CECI BOUCHE. `sommet` note la QUALITE DU TRI,
            # occasion par occasion. Il ignore la TAILLE misee sur chacune,
            # donc tout le travail de la tete de budget. Cette tete est
            # desormais la seule que PPO entraine, et elle dispose d'un palier
            # 0 % : un levier d'ABSTENTION explicite. Un modele qui
            # apprendrait parfaitement a ne rien miser aux mauvais moments
            # rendait exactement le meme `sommet` qu'un modele misant pareil
            # partout. Le critere qui choisit le checkpoint etait aveugle a la
            # seule chose qu'on venait de lui apprendre.
            #
            # LA GRANDEUR JUGEE, par occasion retenue :
            #
            #     r_i = positions_i x R_i
            #
            # `positions_i` est le palier que la tete poserait la — par
            # ARGMAX, la regle deployee — compte en POSITIONS MINIMALES, et
            # `R_i` le rendement en unites de risque. Leur produit se lit
            # donc en « R d'une position minimale » : miser trois positions
            # sur une occasion a +2 R rapporte 6.
            #
            # POURQUOI PAS UNE FRACTION DE COMPTE, ce qu'on affichait avant.
            # Il faudrait convertir par l'ATR de chaque barre et par l'equite
            # du moment — deux grandeurs qui bougent — pour un nombre qui ne
            # serait juste qu'a capital fixe. L'unite « position minimale »
            # est celle de l'ACTION du modele, elle ne depend ni du capital ni
            # de la volatilite, et elle est proportionnelle a l'effet sur le
            # compte a tout instant donne.
            #
            # Une occasion ecartee, ou misee a 0 position, rend exactement 0 :
            # c'est par la que l'abstention entre.
            #
            #     moyenne = mean(r_i)                  gain espere par occasion
            #     baisse  = sqrt(mean(min(r_i, 0)^2))  taille typique des pertes
            #     critere = moyenne - baisse
            #
            # UN MODELE QUI S'ABSTIENT AU BON MOMENT EST PAYE DEUX FOIS : il
            # retire un r_i negatif, ce qui monte la moyenne ET vide la queue
            # gauche, donc baisse la `baisse`. Un modele qui mise gros au
            # mauvais moment est puni deux fois.
            #
            # POURQUOI PAS UNE VRAIE COURBE D'EQUITE, ce que j'avais ecrit
            # d'abord et que l'epoch 1 a condamne sur-le-champ. Elle sommait
            # les r_i dans l'ordre chronologique et prenait le plus grand
            # recul. Resultat affiche : gain +404 %, creux +164 %. Un creux de
            # 164 % n'existe pas — le compte meurt a 100. La cause : 160
            # occasions retenues sur une fenetre qui n'en tient que ~34 SANS
            # CHEVAUCHEMENT (38 424 barres pour ~1 140 barres de detention).
            # La somme comptait donc le meme capital cinq fois. Et comme les
            # deux termes grandissaient avec le NOMBRE d'occasions, le creux
            # ne pouvait pas mordre : il suffisait d'en ajouter pour gonfler
            # les deux.
            #
            # UNE MOYENNE ET UNE DEMI-VARIANCE, ELLES, SUPPORTENT LE
            # CHEVAUCHEMENT. Ce sont des grandeurs de DISTRIBUTION : les
            # occasions qui se recouvrent les rendent moins PRECISES — il y a
            # ~34 observations independantes derriere 160 — mais ne les
            # faussent pas. Une trajectoire cumulee, elle, exige que les
            # occasions se suivent, et elle n'a aucun sens sans cela.
            #
            # POURQUOI PAS `val_max_dd`, LE CREUX DU SIMULATEUR, qui est un
            # vrai creux. Il porte sur ~55 trades d'un seul chemin, avec la
            # variance que le simulateur ajoute par les emplacements. C'est
            # exactement ce qui avait disqualifie le PnL de validation comme
            # critere. Il reste au journal, il ne decide pas.
            #
            # LE COEFFICIENT 1 DEVANT LA BAISSE EST UNE CONVENTION, pas une
            # mesure : « moyenne moins un ecart a la baisse » est la forme
            # standard d'une moyenne corrigee du risque. Je n'ai pas de quoi
            # le calibrer sans toucher a la fenetre de test, et je ne le
            # ferai pas.
            # ============================================================
            # RAPPORTE AU RISQUE MISE, ET C'EST UNE CORRECTION DE FOND.
            #
            # LE DEFAUT. La premiere version sommait `budget_i x R_i` puis en
            # prenait la moyenne, et retranchait un demi-ecart-type calcule sur
            # les memes produits. Les deux termes sont alors LINEAIRES en
            # budget, donc leur difference aussi : doubler tous les budgets
            # double le score. Le critere recompensait le LEVIER, alors qu'il
            # avait ete construit pour ne pas le faire — je l'ai meme ecrit
            # noir sur blanc, et c'etait faux.
            #
            # CE QUE LA MESURE A MONTRE (exec55, deux epochs) :
            #
            #     epoch   bud    gain    baisse   gain/baisse    net
            #       1     1.94   1.417    1.435      0.988      -0.019
            #       2    12.00   9.518    9.350      1.018      +0.168
            #
            # `gain/baisse` vaut 1.0 aux deux epochs — AUCUNE competence ni
            # dans un cas ni dans l'autre. Mais `net` passe de negatif a
            # positif parce que `bud` a ete multiplie par six, et le
            # checkpoint a ete retenu la-dessus, avec un creux de validation a
            # 85 %. Le critere selectionnait le levier.
            #
            # LA NORMALISATION. On pondere par le risque reellement mise :
            #
            #     g = somme(b_i x R_i) / somme(b_i)      R par POSITION misee
            #     d = racine( somme(b_i x min(R_i,0)^2) / somme(b_i) )
            #     net = g - d
            #
            # Doubler tous les budgets ne change plus rien : numerateur et
            # denominateur doublent ensemble.
            #
            # ET L'ABSTENTION PAIE TOUJOURS, qui etait tout l'objet du
            # critere. Miser zero sur une occasion la retire des DEUX sommes :
            # elle ne pese plus dans la moyenne, donc une mauvaise occasion
            # ecartee monte `g`, et elle ne pese plus dans la queue gauche,
            # donc elle baisse `d`. Paye deux fois, comme avant.
            #
            # NE RIEN MISER DU TOUT rend un score de zero — pas une division
            # par zero, et pas un score negatif. C'est voulu : mieux vaut ne
            # pas trader que trader mal, et le portillon du hasard reste
            # devant pour empecher qu'on retienne un modele inerte.
            # ============================================================
            # LE CALCUL VIT DANS `saint_core`, pas ici : son test appelle
            # la MEME fonction, au lieu de la reimplementer.
            _gain_net, _creux_grille, _score_net = score_retenue(
                _rang_gain[_top])
            # LE BUDGET MOYEN REELLEMENT POSE SUR LES OCCASIONS RETENUES, et
            # la part d'entre elles ou le modele s'abstient tout a fait. Sans
            # ces deux nombres, un score qui monte ne dit pas SI c'est
            # l'abstention qui l'a fait monter.
            # CE QUE LA TETE AURAIT MISE SUR LES MEMES OCCASIONS. Sans
            # ce temoin on ne saurait pas si la regle qu'on a remplacee
            # s'etait mise a marcher — et on ne pourrait plus revenir en
            # arriere sur autre chose qu'une impression.
            # `bud` ET `abst` ONT DISPARU AVEC LES PALIERS. La premiere
            # disait la mise moyenne sur les occasions retenues, la seconde
            # la part de celles ou le modele misait zero. Toute occasion
            # retenue est desormais prise a la meme taille, et l'abstention
            # est l'appartenance au sommet : hors des `q %` du haut, on
            # n'entre pas. Les deux champs valaient respectivement 60 % et
            # 0 % a chaque epoch — deux constantes affichees comme des
            # mesures.
            # RIEN A POSER : `bud` et `abst` ne sont plus affiches.

        # LA BARRE REELLE DU ROLLOUT, lue sur un flux qui en a une. Calculee
        # ici et non dans la f-string : une lambda au milieu d'une
        # concatenation de cinquante champs est exactement le genre de ligne
        # qu'on relit mal — elle a d'ailleurs plante ce run sur un plus
        # unaire.
        # LA BARRE QUI A REELLEMENT COMMANDE, et non celle qu'on croyait.
        #
        # Cette ligne valait `float(conf_thr)` sous un commentaire qui
        # affirmait le contraire — « on affiche celle qu'un flux porte
        # reellement, pas `conf_thr` qui ne commande plus rien ». Elle
        # affichait `conf_thr`, et `conf_thr` commandait. Trois affirmations
        # fausses dans six lignes, decouvertes le 2026-09-21 parce que le
        # journal montrait `S 0.0 %` au rollout contre 45 % de ventes en
        # validation.
        #
        # DEUX BARRES, COMME EN VALIDATION. Les flux ont chacun leur
        # fenetre ; on prend celle de l'environnement 0, comme la
        # validation prend `val_decisions[0]`. Un flux vaut l'autre :
        # ils voient le meme nombre d'occasions.
        # LE CRITERE JUGE LES TRADES JOUES EN VALIDATION — PPO complet,
        # 2026-09-25.
        #
        # Il jugeait un CLASSEMENT : les 5 % d'occasions preferees d'une
        # grille, contre le hasard. Il n'y a plus de classement ; la
        # strategie est ce que la politique JOUE. Le score net se calcule
        # donc sur les trades de validation, en R de chacun — gain moyen
        # moins taille typique des pertes, la meme formule
        # (`score_retenue`). Le portillon du hasard n'a plus d'objet : il
        # comparait un tri a un tirage.
        if _rang_idx is None:
            if _valide:
                _val_r_prec = [float(tm.get("r", 0.0))
                               for ve in val_envs for tm in ve.trades_meta]
            if _val_r_prec:
                _gain_net, _creux_grille, _score_net = score_retenue(
                    np.asarray(_val_r_prec, dtype=np.float64))

        _thr_roll = train_decisions[0].thresholds
        _ligne_meta = (
            f"{tag} {epoch_str}  "
            f"{_col('META ', _C.GREY + _C.BOLD)}  "
            # `rho` A QUITTE LE BANDEAU LE 2026-09-21, avec le budget.
            # Il correlait la PART DEPLOYEE au rendement ; sans paliers il
            # n'y a plus de part, et le champ n'affichait plus que `+nan`.
            # `rhoAux` reste : il correle le SCORE de la tete au rendement,
            # et ne depend d'aucune taille de mise.
            f"rhoAux {_rho_aux:>+6.4f}  "
            + ("" if _valide else f"[val ep{_val_epoch}] ") +
            f"sommet {_gain_top:>+6.3f}R/{_gain_tous:>+6.3f}R  "
            f"horizon {int(getattr(cfg, 'horizon_cloture', 60))}m "
            f"{_g60_top:>+6.3f}R/{_g60_tous:>+6.3f}R  "
            # LE CRITERE DE RETENUE, ET CE QUI LE COMPOSE.
            #
            # `net` est ce qui choisit le checkpoint depuis le 2026-09-20 :
            # ce que rapporte une occasion retenue, en R d'une POSITION
            # MINIMALE, MOINS la taille typique de ses pertes. `sommet`,
            # au-dessus, ne juge plus que la qualite du tri — il garde le
            # portillon du hasard, il ne classe plus.
            #
            # LES TROIS SONT EN R PAR POSITION MISEE, pondere par le
            # budget. Ils ne dependent donc PAS du levier : doubler tous les
            # budgets les laisse identiques. Un `net` de +0.30 veut dire que
            # chaque position misee rapporte 0.30 R de plus que la taille
            # typique de ses pertes — quel que soit le nombre de positions.
            #
            # LES LIRE COMME UN TOTAL DE FENETRE serait une erreur d'un
            # facteur 160 : il y a ~160 occasions retenues.
            #
            # `bud` et `abst` disent POURQUOI `net` a bouge : le palier moyen
            # pose sur les occasions retenues, et la part d'entre elles ou le
            # modele s'abstient completement. Un `net` qui monte avec `abst`
            # qui monte, c'est l'abstention qui paie ; un `net` qui monte avec
            # `bud` qui monte, c'est le levier. Deux histoires opposees que le
            # seul `net` ne permettait pas de separer.
            f"net {_score_net:>+6.3f}R "
            f"(gain {_gain_net:>+6.3f}R baisse {_creux_grille:>5.3f}R)  "
            # LE SEUIL A BATTRE POUR ETRE SAUVEGARDE, ecrit ici pour que la
            # veille n'ait pas a le deviner. `retient_checkpoint` exige un
            # score STRICTEMENT positif — sinon « configuration PERDANTE » —
            # et STRICTEMENT au-dessus du record du fold. D'ou le max.
            #
            # IL N'APPARAISSAIT QUE DANS LE MESSAGE DE REFUS, et seulement
            # quand le record etait la raison du refus. On voyait le score
            # epoch apres epoch sans jamais voir ce qu'il lui manquait.
            f"a_battre {max(0.0, float(best_metric)):+.3f}R  "
            # `bud` ET `abst` ONT QUITTE LE BANDEAU LE 2026-09-21.
            #
            # Ils disaient la mise moyenne sur les occasions retenues et la
            # part de celles ou le modele misait zero. Sans paliers les deux
            # valent zero par construction — et un indicateur mort qui garde
            # l'air vivant est pire qu'un champ absent. C'est la lecon que
            # `H` a deja coutee a ce depot : j'ai annonce « ACTOR GELE » sur
            # un run qui apprenait.
            #
            # CE QUI LES REMPLACE POUR LA MEME QUESTION : `sel[...]` dit
            # quelle part des occasions passe la barre, et c'est la seule
            # abstention qui existe encore.
            # LE TEMOIN, cote a cote avec la regle deployee : `rhoTete` est
            # ce que la tete de budget aurait obtenu, `rho` ce que le rang
            # obtient. La comparaison se lit sur une seule ligne.
            # LA MARGE EXIGEE ET LE NOMBRE D'OCCASIONS DISJOINTES derriere
            # le sommet. Sans ces deux nombres au journal, un refus « ne bat
            # pas le hasard + marge » serait invérifiable.
            + f"[indep {int(_n_eff):>3d} marge {_marge_hasard:.3f}]  "
            f"Sortino {metric:>+6.3f}  "
            f"{_col(f'Sortino30 {s30:>+6.3f}', s30_col)}  "
            f"AvgW {_money(avg_win_train, width=8)}  AvgL {_money(avg_loss_train, width=8)}  "
            # LA TENUE MAXIMALE Y EST DEPUIS LE RETRAIT DU PLAFOND. Sans
            # horloge, rien n'empeche structurellement une position de
            # courir jusqu'a la fin de l'episode ; ce chiffre est le seul
            # endroit ou ca se verrait. Le fold 1 a deja vecu un trade
            # tenu 5 759 barres sans que rien ne le signale.
            f"tenue[G {_med(val_tenues_g):.0f}/{_moy(val_tenues_g):.0f} "
            f"P {_med(val_tenues_p):.0f}/{_moy(val_tenues_p):.0f} "
            f"x{(_moy(val_tenues_g) / max(_moy(val_tenues_p), 1e-9)):.1f} "
            f"max {max(_max(val_tenues_g), _max(val_tenues_p)):.0f}]  "
            f"ActorL {np.mean(epoch_actor_loss):>+7.4f}  "
            f"AuxL {np.mean(epoch_aux_loss) if epoch_aux_loss else float(chr(110)+chr(97)+chr(110)):>7.4f}  "
            f"CriticL {np.mean(epoch_critic_loss):>7.4f}  "
            f"H {np.mean(epoch_entropy):>5.3f}  "
            # LE PLAFOND EST CELUI DU COTE, pas 1.099 en dur. En
            # long-only la politique choisit entre ACHETER et ATTENDRE :
            # son maximum vaut ln 2 = 0.693, pas ln 3. Ecrit en dur, il
            # faisait passer une politique a PILE OU FACE pour une
            # politique differenciee a 63 %.
            f"Hflat {np.mean(epoch_entropy_flat):>5.3f}/"
            f"{math.log(sum(cotes_permises(cfg.side)) + 1):>5.3f}  "
            # L'HISTOGRAMME DES PALIERS A ETE RETIRE LE 2026-09-21 : il n'y
            # a plus qu'une taille, celle du lot minimum, mesuree a 1.0000
            # unite (p5 0.9999, p95 1.0001) sur 7 363 trades.
            f"sel[train {100*selectivite:>4.1f}% val {100*val_selectivity:>4.1f}%] "
            # LEQUEL DES DEUX PORTILLONS FERME. `occ` est le nombre
            # d'occasions ou la validation a pu decider ; `barre` celles que
            # le score n'a pas fait passer ; `cap` celles que le score a
            # fait passer mais que le budget a refusees ; `ent` les entrees
            # reellement ouvertes. Sans ces quatre nombres, une chute du
            # nombre de trades ne dit pas si c'est la SELECTION qui se
            # resserre ou la CAPACITE qui manque — et les deux appellent des
            # corrections opposees.
            + (f"portillons[occ {_portillons['occasions']} barre "
               f"{_portillons['barre']} cap {_portillons['capacite']} "
               f"ent {_portillons['entrees']}] "
               if _portillons["occasions"] else "")
            # LA BARRE DU ROLLOUT EST DESORMAIS UN RANG GLISSANT : on affiche
            # celle qu'un flux porte reellement en fin d'epoch, pas
            # `conf_thr` qui ne commande plus rien.
            + f"thr[trB {_thr_roll[0]:.3f} trS {_thr_roll[1]:.3f} "
            + f"valB {val_decisions[0].thresholds[0]:.3f} valS {val_decisions[0].thresholds[1]:.3f}] "
            f"etendue[tr {pbs_etendue:.4f} val {val_etendue:.4f}]  "
            f"blocs[{_bl}] {_npos}/{_nval}  "
            f"KL {np.mean(epoch_kl):>+6.4f}  "
            f"dec {n_samples:>6d}"
            + (f"/{_dec_collectees}" if _dec_collectees > n_samples else "") + "  "
            # `|R|`, `advStd`, `clip` et `quasi0` ont quitte le bandeau avec
            # les avantages : quatre champs qui decrivaient la calibration
            # d'un gradient qui n'existe plus.
            f"gnorm {np.mean(epoch_grad_norm) if epoch_grad_norm else 0.0:>7.3f}  "
            f"g[actor {np.mean(g_actor_hist) if g_actor_hist else 0.0:.2e} "
            f"critic {np.mean(g_critic_hist) if g_critic_hist else 0.0:.2e} "
            f"tronc {np.mean(g_trunk_hist) if g_trunk_hist else 0.0:.2e}]  "
            f"lratio {np.mean(logratio_hist) if logratio_hist else 0.0:.4f} "
            f"clipfrac {100*np.mean(clipfrac_hist) if clipfrac_hist else 0.0:.1f}%  "
            f"temps[{' '.join(f'{k} {v:.0f}s' for k, v in _chrono.items())}]  "
            f"{etat_gpu()}  "
            f"flat {100*np.mean(epoch_flat_frac) if epoch_flat_frac else 0.0:>4.1f}%  "
            f"ENV [{_col(f'B {buy_ratio:>4.1%}', _C.GREEN)} "
            f"{_col(f'S {sell_ratio:>4.1%}', _C.BLUE)} "
            f"{_col(f'H {hold_ratio:>4.1%}', _C.GREY)} "
            f"{_col(f'C {close_ratio:>4.1%}', _C.MAGENTA)}]"
            + ("  MOYENNE DES POIDS" if epoch_moyenne else "")
        )
        print(_ligne_meta)

        # ---- L'ANALYSE DE LA VEILLE, DANS CE TERMINAL-CI ----
        #
        # Les trois lignes ci-dessus sont des colonnes : elles disent ce qui
        # s'est passe, pas ce que cela vaut. Le point mort, l'ecart a la
        # politique GELEE DU MEME RUN, l'erreur-type de cet ecart et les
        # diagnostics vivaient dans `veille_epochs.py` — donc dans une
        # seconde fenetre, qu'il fallait penser a lancer. Quand elle ne
        # l'etait pas, personne ne lisait ces chiffres.
        #
        # C'est le MEME code qui les produit, pas une copie : on lui donne le
        # texte qu'on vient d'ecrire et il rend les memes blocs qu'au fichier.
        # Une seconde mise en forme aurait diverge de la premiere sans que
        # rien ne le signale, comme le reste de ce depot l'a deja montre.
        try:
            for _console, _brut in _veilleur().avale(
                    "\n".join((_ligne_train, _ligne_val, _ligne_meta))):
                for _l in _console:
                    print(_l)
                print()
        except Exception as _e:
            # L'analyse ne doit JAMAIS interrompre un entrainement : elle
            # commente, elle ne produit rien dont la suite depende. Mais elle
            # se tait bruyamment — trois fois deja un motif casse l'a rendue
            # muette sans que personne ne s'en apercoive.
            print(f"  [veille] analyse indisponible ({type(_e).__name__}: "
                  f"{_e}) — les colonnes ci-dessus restent valides.")

        # Écriture CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as _f:
            _csv.DictWriter(_f, fieldnames=_csv_fields).writerow({
                "epoch": epoch,
                "train_pnl": round(profit_epoch, 4),
                "train_trades": num_trades_epoch,
                "train_win": round(winrate_epoch, 4),
                "train_loss": round(train_loss_rate, 4),
                "train_totalW": round(total_profit_train, 4),
                "train_totalL": round(total_loss_train, 4),
                "train_avgW": round(avg_win_train, 4),
                "train_avgL": round(avg_loss_train, 4),
                "train_nbL": num_loss_train,
                "train_pf": round(profit_factor_train, 4),
                "train_dd": round(max_dd_epoch, 4),
                "train_long_w": train_long_w,
                "train_long_l": train_long_l,
                "train_long_pnl": round(train_long_pnl, 4),
                "train_short_w": train_short_w,
                "train_short_l": train_short_l,
                "train_short_pnl": round(train_short_pnl, 4),
                "val_pnl": round(val_profit, 4),
                "val_trades": val_num_trades,
                "val_win": round(val_winrate, 4),
                "val_loss": round(val_loss_rate, 4),
                "val_totalW": round(total_profit_val, 4),
                "val_totalL": round(total_loss_val, 4),
                "val_avgW": round(avg_win_val, 4),
                "val_avgL": round(avg_loss_val, 4),
                "val_nbL": num_loss_val,
                "val_pf": round(profit_factor_val, 4),
                "val_dd": round(val_max_dd, 4),
                "val_long_w": val_long_w,
                "val_long_l": val_long_l,
                "val_long_pnl": round(val_long_pnl, 4),
                "val_short_w": val_short_w,
                "val_short_l": val_short_l,
                "val_short_pnl": round(val_short_pnl, 4),
                "sortino": round(metric, 6),
                "sortino30": round(recent_metric, 6),
                "actor_loss": round(float(np.mean(epoch_actor_loss)), 6),
                "critic_loss": round(float(np.mean(epoch_critic_loss)), 6),
                "entropy": round(float(np.mean(epoch_entropy)), 6),
                "entropy_flat": round(float(np.mean(epoch_entropy_flat)), 6),
                "grad_norm": round(float(np.mean(epoch_grad_norm)) if epoch_grad_norm else 0.0, 6),
                "kl": round(float(np.mean(epoch_kl)), 6),
                "buy_ratio": round(buy_ratio, 4),
                "sell_ratio": round(sell_ratio, 4),
                "hold_ratio": round(hold_ratio, 4),
                "close_ratio": round(close_ratio, 4),
            })

        def _sauve_seuil(chemin_pth: str) -> None:
            """Écrit le seuil calibré à côté du checkpoint.

            Sans ce fichier, live / backtests / MQL5 appliqueraient une barre
            différente de celle sur laquelle le modèle a été sélectionné — le
            checkpoint et son seuil ne veulent rien dire l'un sans l'autre.
            """
            import json
            with open(chemin_pth.replace(".pth", "_calib.json"), "w",
                      encoding="utf-8") as f:
                json.dump({
                    "decision_policy": decision_spec,
                    "calib_thr_buy": float(calib_thr_val[0]),
                    "calib_thr_sell": float(calib_thr_val[1]),
                    "calib_thr": float(max(calib_thr_val)),
                    "calib_thr_train": float(conf_thr),
                    "legacy_off_policy_curriculum": cfg.legacy_off_policy_curriculum,
                    "selectivite": float(val_selectivity),
                    "etendue_val": round(val_etendue, 6),
                    "epoch": epoch,
                    "val_trades": val_num_trades,
                    "regle": "rolling_rank; historique par flux; egalites conservatrices",
                    # GEOMETRIE DE L'ENTREE. Elle n'est PAS recuperable depuis
                    # les poids : avec un segment de 2, n'importe quel lookback
                    # se reconstruit en ajustant le pas, et le reseau se charge
                    # sans broncher avec un espacement temporel faux. Le seul
                    # symptome serait un modele qui trade mal. On l'ecrit donc
                    # ici, a cote du seuil, puisque c'est deja le fichier qui
                    # dit comment se servir du checkpoint.
                    "lookback": int(cfg.lookback),
                    "taille_patch": int(cfg.taille_patch),
                    "pas_patch": int(cfg.pas_patch),
                    "architecture": cfg.architecture,
                    # Les membres, pour que le live sache ce qu'il recharge :
                    # un ensemble se reconnait a ses cles `membres.N.`, mais
                    # leur ORDRE et leur nature meritent d'etre ecrits en clair.
                    "membres": [a for a, _, _ in cfg.membres]
                    if cfg.architecture == "ensemble" else None,
                    "n_features": int(OBS_N_FEATURES),
                    # Le nombre de tetes est le SEUL element de la geometrie
                    # SAINT qui ne se deduise pas des poids : l'attention a la
                    # meme forme quel que soit le decoupage. Un mauvais choix
                    # se charge sans erreur et calcule autre chose.
                    "saint_heads": int(cfg.saint_heads),
                    "saint_lecture": cfg.saint_lecture,
                    "saint_n_freq": int(cfg.saint_n_freq),
                    "saint_mlp_dim": int(cfg.saint_mlp_dim),
                    "d_model": int(cfg.d_model),
                    "num_blocks": int(cfg.num_blocks),
                }, f, indent=2)

        # Best sur PnL PAR TRADE, et non sur le PnL TOTAL.
        #
        # Le critere comparait des totaux. Or la selectivite se resserre au fil
        # du run, donc les epochs tardives tradent moins et perdent moins EN
        # VALEUR ABSOLUE, quelle que soit leur qualite. Mesure sur un run reel :
        #
        #     epoch 10   -58.49 $ sur 182 trades = -0.32 $/trade   PF 0.96
        #     epoch 16   -32.10 $ sur  38 trades = -0.84 $/trade   PF 0.88
        #
        # L'epoch 16 a ete retenue comme « NEW BEST PROFIT » alors qu'elle est
        # 2.6 fois PIRE par trade et que son profit factor est inferieur. Le
        # critere recompensait le fait de trader moins, pas de trader mieux.
        # ---- MOYENNE DES POIDS des dernieres epochs ----
        # Elle remplace le "meilleur checkpoint". Un checkpoint choisi sur la
        # validation vaut -3.3 points au test la ou le dernier epoch en vaut
        # -0.2 : selectionner, c'est attraper un pic de validation qui ne se
        # reproduit pas. Une moyenne ne depend d'aucun tirage particulier.
        #
        # On accumule en float64 : sommer une dizaine de tenseurs float32 puis
        # diviser perd des chiffres significatifs sur les petits poids, et rien
        # ne le signalerait.
        if not epoch_moyenne:
            stats_precedentes = (epoch_actor_loss, epoch_critic_loss,
                                 epoch_entropy, epoch_entropy_flat,
                                 epoch_kl, epoch_grad_norm)

        if not epoch_moyenne and epoch > cfg.epochs - cfg.n_moyenne_poids:
            etat_courant = policy.state_dict()
            if somme_poids is None:
                somme_poids = {k: v.detach().to(torch.float64).clone()
                               for k, v in etat_courant.items()}
            else:
                for k, v in etat_courant.items():
                    somme_poids[k] += v.detach().to(torch.float64)
            n_moyennes += 1

        val_profit_par_trade = (val_profit / val_num_trades
                                if val_num_trades > 0 else -1e18)
        if (val_num_trades >= cfg.min_val_trades_save
                and val_profit_par_trade > best_val_profit):
            best_val_profit = val_profit_par_trade
            state_profit = policy.state_dict().copy()
            save_checkpoint(state_profit, best_profit_path)
            _sauve_seuil(best_profit_path)
            # CE CHECKPOINT N'EST PAS DEPLOYE, ET LA LIGNE DOIT LE DIRE.
            #
            # Deux etoiles se suivaient dans le journal, l'une jaune l'autre
            # magenta, sans que rien ne dise laquelle comptait. Celle-ci suit
            # le PnL par trade — ~55 trades, erreur-type 2.81 $ pour un gain
            # de 5.30 $ : selectionner la-dessus coute -3.3 points mesures
            # ici. Elle est conservee pour comparaison, elle n'est ni retenue
            # ni transmise au fold suivant.
            print(
                f"  {_col('☆', _C.YELLOW)} "
                f"{_col('NEW BEST PROFIT', _C.YELLOW)}  "
                f"ValPNL/trade={_money(best_val_profit, width=10)}  "
                f"trades={val_num_trades}  "
                f"{_col('(temoin seulement — ne decide pas du deploiement)', _C.GREY)}"
            )

        # ==================================================================
        # LE MEILLEUR CHECKPOINT SE CHOISIT SUR LE CLASSEMENT, PAS SUR LE
        # SORTINO NI SUR LE PNL. Trois raisons, toutes mesurees.
        #
        # 1. LE NOMBRE D'OBSERVATIONS. Le Sortino et le PnL par trade se
        #    calculent sur ~55 trades de validation ; le rho porte sur les
        #    3 273 decisions de la fenetre. La veille annonce elle-meme, a
        #    presque chaque epoch, que le gain par trade est SOUS son
        #    erreur-type. Selectionner sur une grandeur non separable de zero
        #    revient a tirer au sort — et ce tirage coute -3.3 points mesures
        #    ici, avec un fold d'exec18 a +8.14 en validation pour -0.9 en
        #    test.
        #
        # 2. LE RHO EST LE SEUL A ETRE FRAIS A CHAQUE EPOCH. La validation ne
        #    tourne qu'une epoch sur `validation_tous_les` ; entre deux, le
        #    Sortino affiche est CELUI DE LA DERNIERE MESURE. Le comparer a
        #    `best_metric` ne compare alors rien. Le rho, lui, est recalcule
        #    a chaque epoch sur les memes 3 273 etats — il suit la politique
        #    reellement courante.
        #
        # 3. C'EST LA GRANDEUR QUI DECIDE EN PRODUCTION. Avec
        #    `tri_par_tete_aux`, ce sont les scores de la tete auxiliaire qui
        #    trient les occasions en validation, en test et en live. `rhoAux`
        #    mesure exactement la qualite de ce tri. Choisir un checkpoint
        #    sur une autre grandeur, c'est optimiser ce qu'on ne joue pas.
        #
        # CE QUE CE CRITERE N'EST PAS : une mesure de rentabilite. Un rho
        # eleve dit que l'ordre des occasions est bon, pas que la geometrie
        # gagne. Les deux se lisent ensemble, et `bestprofit_` garde sa
        # lecture par le PnL pour qu'on puisse les comparer.
        #
        # LE CRITERE EST LA RENTABILITE DU SOMMET, sous condition que le
        # classement tienne. Deux grandeurs, et il faut les deux :
        #
        #   `_gain_top`  ce que rapportent, en R, les occasions que ce
        #                checkpoint mettrait en position — c'est la
        #                RENTABILITE, mesuree sur l'ensemble reellement joue.
        #   `rho`        la qualite de l'ORDRE sur toute la fenetre. Il sert
        #                de garde : un sommet rentable avec un classement nul
        #                est un tirage, pas un tri, et il ne se reproduira
        #                pas sur une autre fenetre.
        #
        # Prendre le gain SEUL reviendrait a selectionner sur 164
        # observations sans savoir si le tri les a choisies ou si elles sont
        # tombees la. Prendre le rho SEUL — ce que faisait la version d'il y
        # a une heure — mesure un ordre juste sans jamais verifier que le
        # sommet paie. Les deux ensemble disent : il trie, et ce qu'il trie
        # rapporte.
        #
        # TOUTES LES EPOCHS SONT CANDIDATES, et c'est le point.
        #
        # Avant, la selection ne pouvait retenir qu'une epoch sur trois : le
        # Sortino n'existe que les epochs validees, et sur les autres la
        # comparaison portait sur une valeur REPRISE de la derniere mesure —
        # donc jamais superieure au record. Soixante epochs sur quatre-vingt-
        # dix ne pouvaient structurellement pas etre retenues, quelle que
        # soit leur qualite. Personne ne l'avait remarque parce que rien ne
        # le signalait : le code avait l'air de tester chaque epoch.
        #
        # Le classement est calcule a chaque epoch, et la calibration aussi
        # depuis qu'on la fait tourner partout : les poids enregistres et les
        # barres ecrites a cote viennent donc toujours de la MEME epoch.
        # ====================================================================
        # LE GARDE-FOU DE SURVIE, et pourquoi il a fallu l'ajouter.
        #
        # `_gain_top` mesure ce que rapportent les occasions du sommet UNE A
        # UNE. Il ne sait rien du nombre tenu simultanement, de leur
        # correlation, ni de la survie du compte. Tant que le budget de
        # risque n'autorisait qu'une ou deux positions, la distinction etait
        # sans consequence. Depuis qu'il est retire, un modele peut avoir un
        # excellent classement par occasion ET vider le compte en ouvrant
        # soixante positions correlees — et `_gain_top` le noterait au
        # sommet.
        #
        # On refuse donc de retenir un checkpoint dont la DERNIERE mesure de
        # portefeuille a creve le garde-fou de creux. Le seuil n'est pas
        # invente : c'est la destruction du compte, celle qui termine deja
        # un episode.
        #
        # LE CREUX EST CELUI DE L'EPOCH JUGEE. `validation_tous_les` vaut 1
        # depuis le 2026-09-19 : le portefeuille est mesure a chaque epoch,
        # donc la garde lit l'etat du modele qu'elle juge et non celui d'il
        # y a deux epochs. C'est ce qui a ete paye sept heures de run.
        #
        # Si la frequence de validation remonte un jour, cette garde
        # redevient approximative — elle jugerait sur un portefeuille
        # perime — et il faudra le savoir avant de lire ses refus.
        # ====================================================================
        # `_rho_ep` N'EST PLUS UN REPLI : il valait la correlation de la
        # part deployee, retiree avec le budget, donc `nan` par
        # construction. Un repli vers `nan` aurait ferme le portillon en
        # silence le jour ou `rhoAux` manque.
        _score_rang = _rho_aux
        _retenu, _pourquoi = retient_checkpoint(
            # SANS CLASSEMENT, le « sommet » est le gain moyen par trade :
            # le portillon du hasard est saute (`gain_tous` vaut nan).
            gain_top=(_gain_top if _rang_idx is not None else _gain_net),
            score_rang=_score_rang,
            gain_tous=_gain_tous,
            score_retenue=_score_net, creux=_creux_grille,
            marge_hasard=_marge_hasard,
            val_num_trades=val_num_trades, val_ruine=val_ruine,
            best_metric=best_metric,
            min_trades=cfg.min_val_trades_save)
        # TOUT REFUS SE DIT, ET DIT SUR QUOI.
        #
        # Seul le refus pour compte detruit etait affiche. Les autres — un
        # score sous le record, un sommet qui ne bat pas le hasard, trop peu
        # de trades — passaient en silence, et on regardait quatre-vingt-dix
        # epochs en croyant qu'un meilleur modele etait garde alors que rien
        # ne l'etait. Le motif est deja calcule par `retient_checkpoint` ;
        # il ne restait qu'a l'ecrire.
        #
        # LE MOT `REFUSE` RESTE RESERVE au cas grave — le modele etait
        # MEILLEUR et c'est le portefeuille qui l'a disqualifie. Les refus
        # ordinaires sont annonces plus sobrement : ils sont la normale,
        # quatre-vingts epochs sur quatre-vingt-dix.
        if not _retenu:
            if _pourquoi.startswith("compte detruit"):
                _refus_creux += 1
                print(f"  {_col('REFUSE', _C.RED + _C.BOLD)}  score net "
                      f"{_score_net:+.3f} R par occasion, meilleur que "
                      f"{best_metric:+.3f} R, "
                      f"mais {_pourquoi} — non retenu, et non transmis au "
                      f"fold suivant.")
            else:
                print(f"  {_col('garde', _C.GREY)}  le modele en place reste "
                      f"le meilleur — {_pourquoi}")
        if _retenu:
            _ancien = best_metric
            best_metric = _score_net
            best_state = copy.deepcopy(policy.state_dict())
            best_thresholds = list(calib_thr_val)
            best_decision_spec = copy.deepcopy(decision_spec)
            save_checkpoint(best_state, best_path)
            _sauve_seuil(best_path)
            epochs_no_improve = 0
            _quoi = "rhoAux" if np.isfinite(_rho_aux) else "rho"
            _best_epoch = epoch
            # LA LIGNE DIT SUR QUOI LE MODELE A ETE RETENU, ET CE QU'IL A
            # BATTU.
            #
            # Elle annoncait `sommet` — qui n'est plus le critere depuis le
            # 2026-09-20. On lisait donc un chiffre, on croyait que c'etait
            # lui qui avait decide, et c'en etait un autre. C'est exactement
            # la faute que ce depot passe son temps a payer : deux
            # descriptions du meme evenement, qui doivent s'accorder par
            # convention.
            #
            # TROIS CHOSES, ET RIEN D'AUTRE : le critere et sa decomposition,
            # ce qu'il a battu, et le fichier ecrit. Le reste — sommet,
            # rhoAux, Sortino — reste sur la ligne META, ou il diagnostique.
            _av = ("premier retenu du fold" if _ancien <= -1e8
                   else f"{_ancien:+.3f} R")
            print(
                f"  {_col('★', _C.MAGENTA + _C.BOLD)} "
                f"{_col('NEW BEST', _C.MAGENTA + _C.BOLD)}  "
                f"retenu sur le SCORE NET "
                f"{_col(f'{_score_net:+.3f} R', _C.MAGENTA + _C.BOLD)} "
                f"par occasion "
                f"(gain {_gain_net:+.3f} R - baisse "
                f"{_creux_grille:.3f} R)  bat {_av}  "
                f"[portillons : sommet {_gain_top:+.3f}R > hasard "
                f"{_gain_tous:+.3f}R, {val_num_trades} trades, compte intact]"
                f"  -> {best_path}"
            )
        else:
            epochs_no_improve += 1
            # patience <= 0 : aucun arret precoce. C'etait une selection sur la
            # validation comme une autre — elle choisissait la longueur du run
            # d'apres elle. Le budget est desormais fixe a l'avance.
            if patience > 0 and epochs_no_improve >= patience:
                print(f"[{cfg.side.upper()}{suffix}] Early stopping après "
                      f"{epoch} epochs (le CLASSEMENT ne progresse plus — "
                      f"c'est rhoAux qui compte les epochs sans amélioration "
                      f"depuis qu'il sélectionne le checkpoint).")
                break

    last_path = f"last_{cfg.model_prefix}_{cfg.side}{suffix}.pth"
    save_checkpoint(policy.state_dict(), last_path)
    _sauve_seuil(last_path)

    # CE QUI PART AU FOLD SUIVANT, DIT EN CLAIR. Le chainage transmet
    # `best_`, et jusqu'ici rien n'affichait DE QUELLE epoch il venait ni
    # avec quel score : une chaine qui se serait rompue en silence aurait
    # produit trois modeles independants sous le nom d'un seul.
    if best_state is None:
        print(f"[{cfg.side.upper()}{suffix}] AUCUN checkpoint 'best' ecrit sur "
              f"ce fold. Le fold suivant reprendra 'last'.")
        if _refus_creux:
            print(f"    {_refus_creux} candidat(s) ont ete ECARTES parce que "
                  f"le compte a ete DETRUIT en validation — appel de marge, "
                  f"lot minimum infinancable, ou equite a zero. Le "
                  f"classement progressait, le portefeuille mourait : c'est "
                  f"le cas ou il faut valider a CHAQUE epoch "
                  f"(validation_tous_les = 1) pour que le critere voie ce "
                  f"qui tue le compte.")
        else:
            print(f"    Aucune epoch n'a atteint {cfg.min_val_trades_save} "
                  f"trades de validation avec un classement positif.")
    else:
        print(f"[{cfg.side.upper()}{suffix}] MEILLEUR CHECKPOINT : epoch "
              f"{_best_epoch} sur {epoch}, sommet {best_metric:+.3f} R par "
              f"occasion. C'est lui qui ouvre le fold suivant.")
        if _refus_creux:
            print(f"    ({_refus_creux} candidat(s) mieux classes ont ete "
                  f"ecartes pour un compte detruit en validation.)")

    if not cfg.evaluate_test:
        print(f"[{cfg.side.upper()}{suffix}] Entrainement termine. TEST reserve, non consulte.")
        return

    print(f"[{cfg.side.upper()}{suffix}] Entraînement terminé, passage en TEST…")

    # LE TEST PORTE SUR LE MODELE COURANT, qui est la moyenne des poids des
    # dernieres epochs (voir l'epoch cfg.epochs + 1). Les checkpoints "best" et
    # "bestprofit" continuent d'etre ecrits pour comparaison, mais ils ne
    # pilotent plus rien : mesure du 2026-09-15, le "best" choisi sur la
    # validation rend -3.3 points au test quand le dernier epoch en rend -0.2.
    #
    # Les seuils et la regle de decision viennent du calibrage fait sur la
    # moyenne elle-meme, sur la fenetre de calibration — un seuil doit suivre
    # l'echelle des scores du modele qu'il filtre, ce n'est pas une selection.
    if best_state is not None:
        print(f"[{cfg.side.upper()}{suffix}] (le checkpoint 'best' existe mais "
              f"n'est PAS utilise pour le test — voir la note ci-dessus)")
    policy.eval()

    # Le test couvre sa fenetre ENTIERE, en episodes disjoints. Il jouait
    # auparavant 5 episodes tires au hasard, soit un quart de la fenetre :
    # 132 trades pour les trois folds, une erreur-type de 4.1 points, et trois
    # resultats qui se contredisaient. Voir departs_disjoints.
    departs_test = departs_disjoints(test_data.length, cfg.lookback,
                                     cfg.episode_length)
    n_test = len(departs_test)
    print(f"  • TEST exhaustif : {n_test} épisodes disjoints, "
          f"{100*n_test*cfg.episode_length/max(test_data.length,1):.0f} % "
          f"de la fenêtre")
    test_decisions = [EntryDecisionPolicy(decision_spec) for _ in range(n_test)]
    test_envs = [BTCTradingEnvDiscrete(test_data, cfg) for _ in range(n_test)]
    all_trades = []
    all_dd = []
    all_equity = []

    t_states: List[np.ndarray] = []
    t_infos: List[Dict] = []
    for e, d in zip(test_envs, departs_test):
        s0, i0 = reset_au_depart(e, d)
        t_states.append(s0)
        t_infos.append(i0)
    t_active = list(range(n_test))
    # LE TEST TIRE COMME LA VALIDATION, sur sa propre graine fixe.
    _explore_t = bool(getattr(cfg, "evaluation_stochastique", True))
    _gen_t = torch.Generator(device=device)
    _gen_t.manual_seed(int(cfg.val_seed) + 3)
    t_frais = np.ones(max(n_test, 1), dtype=bool)
    t_depuis = np.zeros(max(n_test, 1), dtype=np.int64)
    _pas_t = max(1, int(getattr(cfg, "pas_decision_sortie", 1)))

    with torch.no_grad():
        while t_active:
            # MEME REGLE QU'EN VALIDATION ET QU'A L'ENTRAINEMENT. Mesurer
            # le test sur une strategie a une position quand on en deploie
            # une a plusieurs mesurerait ce que personne ne joue.
            deciding = [k for k in t_active if test_envs[k].peut_decider()]
            t_actions = {k: 2 for k in t_active}
            # LA SORTIE, PAR LA MEME FONCTION QU'AU ROLLOUT ET EN
            # VALIDATION. Sans elle le test jouerait une position immortelle
            # pendant que l'entrainement en ferme : la fenetre de test est a
            # usage unique, et une strategie mesuree de travers l'aurait
            # depensee pour rien.
            _tp = [k for k in t_active
                   if k not in deciding and test_envs[k].n_positions > 0]
            for _k in t_active:
                if test_envs[_k].n_positions == 0:
                    t_frais[_k] = True
            _td, _ = cadence_sortie(_tp, t_frais, t_depuis, _pas_t)
            if _td:
                _ta, _, _ = decide_sortie(
                    policy, [t_states[k] for k in _td], device,
                    N_BASE_FEATURES, explore=_explore_t, generateur=_gen_t)
                for _bi, _k in enumerate(_td):
                    t_frais[_k] = False
                    if int(_ta[_bi]) == FERMER:
                        t_actions[_k] = 3

            if deciding:
                # LA MEME POLITIQUE QU'EN VALIDATION, tiree pareil.
                _m3, _ = masque_entree(
                    np.repeat(MASK_FLAT[None, :], len(deciding), axis=0),
                    [test_envs[k] for k in deciding])
                _ta_e, _, _ = decide_entree(
                    policy, [t_states[k] for k in deciding], _m3, device,
                    explore=_explore_t, generateur=_gen_t)
                for bi, k in enumerate(deciding):
                    t_actions[k] = int(_ta_e[bi])

            still = []
            for k in t_active:
                ns, _r, done, _, info = test_envs[k].step(t_actions[k])
                t_states[k] = ns
                t_infos[k] = info
                if not done:
                    still.append(k)
            t_active = still

    with torch.no_grad():
        for ep, test_env in enumerate(test_envs):
            info = t_infos[ep]
            if test_env.idx > 0:
                final_close = test_env.data.close[test_env.idx - 1]
            else:
                final_close = test_env.data.close[-1]

            latent = 0.0
            if test_env.position != 0 and test_env.current_size > 0 and test_env.entry_price > 0:
                latent = (
                    test_env.position *
                    (final_close - test_env.entry_price) *
                    test_env.current_size
                )

            final_equity = test_env.capital + latent

            dd_ep = info["drawdown"]
            all_dd.append(dd_ep)
            all_trades.extend(test_env.trades_pnl)
            all_equity.append(final_equity - cfg.initial_capital)

    test_profit = float(sum(all_equity))
    test_num_trades = len(all_trades)
    test_winrate = (
        float(np.mean([p > 0 for p in all_trades]))
        if test_num_trades > 0 else 0.0
    )
    test_max_dd = float(max(all_dd) if all_dd else 0.0)

    wins_test   = [p for p in all_trades if p > 0]
    losses_test = [p for p in all_trades if p <= 0]
    avg_win_test     = float(np.mean(wins_test))   if wins_test   else 0.0
    avg_loss_test    = float(np.mean(losses_test)) if losses_test else 0.0
    num_loss_test    = len(losses_test)
    total_profit_test = float(sum(wins_test))
    total_loss_test   = float(sum(losses_test))

    print(
        f"[{cfg.side.upper()}{suffix}][TEST] "
        f"NetPNL={test_profit:+9.2f}$  "
        f"Trades={test_num_trades}  "
        f"Win={test_winrate:2.0%}  Loss={1-test_winrate:2.0%}  "
        f"TotalW={total_profit_test:+8.2f}$  TotalL={total_loss_test:+8.2f}$  "
        f"AvgW={avg_win_test:+.3f}$  AvgL={avg_loss_test:+.3f}$  "
        f"NbLoss={num_loss_test}  "
        f"MaxDD={test_max_dd:.3f}"
    )

    print(f"[{cfg.side.upper()}{suffix}] Fin du split.")


# ======================================================================
# TRAINING SIMPLE (split 70/15/15)
# ======================================================================

def run_training_full(cfg: PPOConfig):
    df = load_mt5_data(cfg)

    stats = compute_and_save_global_norm_stats(df.iloc[:int(len(df) * .70)], FEATURE_COLS, path=None)

    train_data, calib_data, val_data, test_data = create_datasets(
        df, FEATURE_COLS, stats, calib_frac=cfg.calib_frac)
    run_training_on_split(train_data, calib_data, val_data, test_data, stats,
                          cfg, suffix="")


# ======================================================================
# WALK-FORWARD
# ======================================================================

def run_walkforward(
    cfg_base: PPOConfig,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    max_folds: int = 1,
    start_fold: int = 1,
    bootstrap_from_path: Optional[str] = None,
    chaine: bool = True,
):
    """Walk-forward CHAINE : un seul modele, du plus ancien au plus recent.

    `chaine=True` transmet les poids d'un fold au suivant. `start_fold`
    selectionne la premiere fenetre ; `bootstrap_from_path` donne des poids de
    depart au tout premier fold.

    POURQUOI LE CHAINAGE ETAIT REFUSE, ET CE QUI A CHANGE. Le code levait :
    « le bootstrap inter-fold exige une adaptation explicite de
    normalisation ». L'objection est juste — chaque fold calculait ses
    statistiques sur SON train, donc un reseau herite aurait vu ses entrees a
    une autre echelle que celle sous laquelle il a appris — mais elle n'avait
    jamais ete chiffree.

    MESURE DU 2026-09-19, ecart entre les statistiques du fold 1 et celles
    des suivants, exprime en ecarts-types du fold 1 :

        fold   derive de moyenne (med / p95 / max)   sigma_k / sigma_1 (med)
          2         0.008 / 0.063 / 0.125                    1.007
          3         0.013 / 0.092 / 0.162                    1.011

    Treize millio-iemes d'ecart-type en median, seize centiemes au pire, sur
    des entrees qui vivent dans [-3, 3]. La raison tient a la geometrie du
    walk-forward : le pas vaut la longueur du TEST, donc les trains des folds
    1 et 3 partagent 64 % de leurs barres.

    ON NE CORRIGE DONC RIEN, ON SUPPRIME LE PROBLEME : les statistiques sont
    calculees UNE FOIS, sur le train du premier fold, et servent a tous. La
    derive ci-dessus devient le prix a payer — l'echelle vue par le reseau ne
    bouge plus du tout — et le checkpoint porte une seule normalisation, celle
    que le live utilisera.

    « MAIS LE MODELE NE CONNAITRA PAS LES PRIX RECENTS. » L'objection est la
    bonne : l'or vaut 1 494 $ au debut du jeu et 4 463 $ a la fin, et des
    statistiques de 2019-2023 appliquees a 2026 enverraient toute colonne
    portant un NIVEAU hors de la plage apprise. Mesure du 2026-09-19, fenetre
    par fenetre, standardisation par les statistiques du fold 1 :

        fenetre         periode            or            derive med   hors plage
        fold 1 train    2019-09->2023-08   1494-> 1955$      0.000       0.09 %
        fold 1 TEST     2024-08->2025-04   2457-> 3319$      0.067       0.13 %
        fold 2 TEST     2025-04->2026-01   3319-> 4463$      0.060       0.07 %
        fold 3 TEST     2026-01->2026-09   4463-> 4378$      0.020       0.23 %

    Et la comparaison qui tranche, sur la fenetre LA PLUS RECENTE :

        statistiques du fold 1 (figees)    derive med 0.020   |z| p99 3.74
        statistiques du fold 3 (par fold)  derive med 0.019   |z| p99 4.09

    Les statistiques figees font AUSSI BIEN, et un peu mieux au 99e centile.

    LA RAISON EST STRUCTURELLE : aucune des 256 colonnes ne suit le prix.
    Correlation au close sur toute la serie, seuil 0.8 : zero colonne
    retenue. Ce sont des rendements, des rapports et des rangs. Le prix peut
    tripler, l'observation ne bouge pas d'echelle — c'est precisement ce qui
    rend le figeage possible, et ce qui le rendrait impossible si une seule
    colonne portait un niveau. Le jour ou l'on en ajoute une, cette mesure
    est a refaire AVANT de figer quoi que ce soit.

    A NOTER, parce que la question revient : AUCUNE normalisation, figee ou
    par fold, ne peut inclure la fenetre de test. Les statistiques du fold 3
    s'arretent en 2024-12 pour un test qui court de 2026-01 a 2026-09. Les
    calculer sur les barres a juger serait une fuite de futur. « Connaitre
    les prix recents » n'est donc pas ce que la normalisation par fold
    apporte — elle n'apporte ici rien du tout.
    """
    if bootstrap_from_path is not None and start_fold != 1:
        raise ValueError("bootstrap_from_path ne s'applique qu'au premier fold "
                         "joue ; les suivants heritent du precedent.")
    assert train_frac + val_frac + test_frac <= 1.0 + 1e-6, "Les fractions ne peuvent pas dépasser 1.0"

    df_full = load_mt5_data(cfg_base)
    n = len(df_full)


    train_len = int(n * train_frac)
    val_len   = int(n * val_frac)
    test_len  = int(n * test_frac)
    window_len = train_len + val_len + test_len

    if train_len <= 0 or val_len <= 0 or test_len <= 0 or window_len > n:
        raise ValueError("Longueurs de segments invalides (train/val/test) pour le walk-forward.")

    step = test_len

    print(f"\n=== WALK-FORWARD {cfg_base.side.upper()}"
          f"{' CHAINE' if chaine else ' INDEPENDANT'} ===")
    print(f"Total bars={n}, window_len={window_len}, "
          f"train={train_len}, val={val_len}, test={test_len}, step={step}")
    if chaine:
        print("Les folds se transmettent le MEILLEUR checkpoint : UN modele "
              "traverse tout l'historique, du plus ancien au plus recent.")

    # Skip les folds avant start_fold
    fold = start_fold - 1
    start = (start_fold - 1) * step
    if start_fold > 1:
        print(f"[SKIP] Folds 1 → {start_fold - 1} sautés (start_fold={start_fold})")

    # UNE SEULE NORMALISATION POUR TOUT LE RUN, celle du train du fold 1.
    #
    # Il en faut UNE SEULE parce que les folds se transmettent les poids : si
    # l'echelle des entrees changeait d'un fold a l'autre, le reseau herite
    # lirait ses colonnes autrement qu'il ne les a apprises. C'est ce qui
    # interdisait le chainage avant qu'on le mesure.
    #
    # ELLE NE VOIT QUE DU PASSE, et c'est le point. La calculer sur tout
    # l'historique couvrirait aussi les prix recents, mais ferait entrer les
    # fenetres de TEST dans la moyenne et l'ecart-type — une fuite de futur.
    # Elle a ete essayee puis retiree : la mesure ci-dessous montre qu'elle
    # n'apportait rien qui vaille ce prix.
    stats = compute_and_save_global_norm_stats(
        df_full.iloc[start:start + train_len], FEATURE_COLS, path=None)
    print(f"Normalisation figee sur le train du fold {start_fold} "
          f"[{start} : {start + train_len}) — la meme pour tous les folds, "
          f"et elle ne voit que du passe.")

    precedent = bootstrap_from_path
    while start + window_len <= n and fold < max_folds:
        fold += 1
        print(f"\n--- Fold {fold} : indices [{start} : {start + window_len}) ---")
        train_data, calib_data, val_data, test_data = create_datasets_from_slices(
            df_full, FEATURE_COLS,
            start=start,
            train_len=train_len,
            val_len=val_len,
            test_len=test_len,
            stats=stats,
            calib_frac=cfg_base.calib_frac
        )

        cfg_fold = PPOConfig(**cfg_base.__dict__)
        cfg_fold.model_prefix = f"{cfg_base.model_prefix}_wf{fold}"

        suffix = f"_wf{fold}"
        run_training_on_split(train_data, calib_data, val_data, test_data, stats,
                              cfg_fold, suffix=suffix,
                              poids_initiaux=precedent)

        # LE FOLD SUIVANT REPREND LE MEILLEUR CHECKPOINT, sur demande
        # explicite. `best_` est celui du meilleur Sortino de validation.
        #
        # CE QUE CE CHOIX COUTE, et il est assume : selectionner un
        # checkpoint sur son resultat de validation vaut -3.3 points mesures
        # dans ce depot, parce que la validation selectionne a l'envers de
        # facon reproductible — le fold 2 d'exec18 affichait +8.14 en
        # validation pour -0.9 en test. La moyenne des derniers jeux de poids
        # (`last_`) ne depend d'aucun tirage, et c'est pour cela qu'elle reste
        # la famille deployee par `checkpoints.py`.
        #
        # Ici le transfert ne sert pas a deployer mais a CONTINUER : ce qui
        # part au fold suivant est le jeu de poids qui a le mieux tenu sur la
        # fenetre precedente, et il sera reentraine sur la suivante.
        #
        # Repli sur `last_` si le meilleur n'existe pas — un fold qui n'aurait
        # jamais ameliore son Sortino n'en ecrit pas. Le repli est annonce,
        # jamais silencieux : sans cela le fold suivant repartirait de zero
        # sans que rien ne le dise.
        if chaine:
            _best = f"best_{cfg_fold.model_prefix}_{cfg_fold.side}{suffix}.pth"
            _last = f"last_{cfg_fold.model_prefix}_{cfg_fold.side}{suffix}.pth"
            if os.path.exists(_best):
                precedent = _best
            elif os.path.exists(_last):
                print(f"  ATTENTION : {_best} absent — le fold {fold} n'a "
                      f"jamais ameliore son Sortino. Le fold suivant reprend "
                      f"{_last} a la place.")
                precedent = _last
            else:
                print(f"  ATTENTION : ni {_best} ni {_last} — le fold suivant "
                      f"REPART DE ZERO, la chaine est rompue.")
                precedent = None
        start += step

    print(f"\n=== FIN WALK-FORWARD {cfg_base.side.upper()} (folds entraînés : {start_fold} → {fold}) ===")


# ======================================================================
# MAIN
# ======================================================================

class _Journal:
    """Ecrit a l'ecran ET dans un fichier que la veille peut lire.

    POURQUOI PAS `Tee-Object`. La commande PowerShell ouvre le fichier en
    acces EXCLUSIF : la veille ne peut alors meme pas l'ouvrir en lecture,
    elle ne lit rien et n'affiche rien — sans erreur visible, puisqu'elle
    n'attrape que `FileNotFoundError`. Une demi-heure a chercher un motif
    casse pour un verrou de fichier.

    Python, lui, ouvre en partage par defaut sous Windows : un autre
    processus peut lire pendant qu'on ecrit. On vide le tampon a chaque
    ligne, sinon la veille lirait avec plusieurs minutes de retard.

    L'ECRAN RESTE LA SOURCE : si l'ecriture du fichier echoue — disque plein,
    fichier verrouille par autre chose — on continue d'afficher. Un journal
    est un confort, l'entrainement n'en depend pas.
    """

    def __init__(self, flux, chemin):
        self.flux = flux
        try:
            self.f = open(chemin, "w", encoding="utf-8", buffering=1,
                          newline="\n")
        except OSError:
            self.f = None

    def write(self, texte):
        self.flux.write(texte)
        if self.f is not None:
            try:
                self.f.write(texte)
            except OSError:
                self.f = None
        return len(texte)

    def flush(self):
        self.flux.flush()
        if self.f is not None:
            try:
                self.f.flush()
            except OSError:
                self.f = None

    def isatty(self):
        return self.flux.isatty()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    # Le journal que lit `veille_epochs.py`. Ecrit ici plutot que par une
    # redirection du shell : voir `_Journal`.
    _jrn = _Journal(sys.stdout, "training_btc.log")
    sys.stdout = _jrn
    sys.stderr = _Journal(sys.stderr, "training_err.log")
    cfg_base = PPOConfig()

    # =======================================================
    # LE PIPELINE. Le cote vient de `PPOConfig.side`, et de nulle part
    # ailleurs.
    # =======================================================
    # `cfg_long` GARDE SON NOM par habitude : il porte le cote que la
    # configuration declare, "both" depuis le 2026-09-25.
    cfg_long = PPOConfig(**cfg_base.__dict__)
    # CETTE LIGNE REPETAIT `cfg_long.side = "long"`, « a l'identique ».
    # Le 2026-09-25 la configuration est passee a "both" pour ouvrir les
    # shorts, et le run est reparti... en LONG seul : `[LONG_wf1]`, « vente
    # INTERDIT », zero short sur 3 600 trades. La repetition devait etre
    # une redondance visible ; elle etait un second reglage, qui gagnait.
    # C'est la panne que le commentaire de `PPOConfig.side` decrit, a
    # l'envers. On LIT la configuration, on ne la recopie pas.

    # LE TITRE ETAIT EN DUR, et il annoncait « BILATERAL » pendant que la
    # configuration disait `side = "long"`. Troisieme fois de la journee
    # qu'une phrase du bandeau affirme ce que le code ne fait pas — apres
    # le nom de l'instrument et la friction. Un bandeau se lit en
    # diagonale : c'est exactement l'endroit ou une affirmation fausse
    # survit le plus longtemps.
    #
    # ET IL SE CALCULE APRES `cfg_long`, PAS AVANT. La premiere version du
    # correctif lisait `cfg_long.side` quinze lignes avant sa creation :
    # `NameError` au demarrage, run mort en trois secondes. Remplacer une
    # phrase en dur par une lecture de configuration deplace la phrase dans
    # le temps, et c'est tout le piege — le bandeau s'imprime en tete, la
    # configuration se construit apres.
    print("\n" + "=" * 70)
    _ca, _cv = cotes_permises(cfg_long.side)
    _quoi = ("BILATERAL (achat + vente + attendre)" if (_ca and _cv)
             else "A L'ACHAT SEUL (achat + attendre)" if _ca
             else "A LA VENTE SEULE (vente + attendre)")
    print("  ENTRAINEMENT %s" % _quoi)
    print("=" * 70)

    # LES CONSTANTES DE L'INSTRUMENT VIENNENT DU REGISTRE, PAS DE LA CONFIG.
    #
    # `instruments.py` a ete ecrit pour etre cette source unique — largeur de
    # stop, trailing, friction, lot minimum, taille du contrat, marge — et il
    # n'etait appele par personne. Les memes constantes vivaient aussi dans
    # `PPOConfig`, recopiees a la main.
    #
    # ELLES AVAIENT DEJA DIVERGE, d'un seul chiffre et sans bruit : la marge
    # valait 0.001734 dans la config contre 0.001744 dans le registre, ou
    # elle est mesuree par `order_calc_margin`. Un ecart de 0.6 % qui ne
    # casse rien — il decale simplement la capacite d'un cran, 129 places
    # ouvrables au lieu de 128 — mais qui suffit a ce que l'entrainement et
    # le live ne comptent pas les memes positions. C'est le test
    # d'alignement de la capacite qui l'a trouve, pas une relecture.
    #
    # Tous les autres champs coincidaient deja. Brancher le registre ne
    # change donc qu'une valeur, et supprime la source du desaccord.
    # `PPOConfig.__post_init__` a deja applique le registre : on se contente
    # de dire ce qui est reellement en vigueur.
    print(f"Constantes lues dans instruments.py pour {cfg_long.symbol} : "
          f"stop {cfg_long.atr_sl_mult:g}xATR, spread "
          f"{cfg_long.spread_bps:g} bps, contrat {cfg_long.contrat:g}, "
          f"marge {cfg_long.marge_frac:.6f}")
    # Un prefixe par jeu d'observation OU par architecture : les poids ne sont
    # jamais interchangeables d'une lignee a l'autre, et le manifeste refuse de
    # reecrire un run existant.
    #
    #   exec3  30 features, scalping_max_holding 120, sortie temps a 240 barres
    #   exec4  sortie par le temps retiree, detention de reference a 30 barres
    #   exec6  ls_ratio_top (+0.0000) remplacee par taker_1m_ma5 (+0.0087)
    #   exec11 CONTRE LE SURAPPRENTISSAGE, les trois leviers ensemble :
    #          modele divise par 4 (298 k au lieu de 1.19 M), coefficient
    #          d'entropie x3.75 (0.030), patience ramenee de 144 a 25 epochs.
    #          exec10 culminait a l'epoch 11 puis se degradait regulierement.
    #   exec10 ECHELLE H1 et features de RANGE. La friction passe de 1.05 a
    #          0.09 unite de risque par ATR — c'est le seul changement de la
    #          session appuye sur une relation mecanique et non sur un ecart
    #          a la limite du bruit. 55 features dont 25 de structure de range.
    #   exec9  RESEAU PatchTST : segments temporels et canaux independants,
    #          lookback 96 au lieu de 25. Mesure a lot 128 : 66.9 ms par pas
    #          contre 285.9 pour SAINT a 25 — quatre fois plus d'historique
    #          pour 4.3 fois moins cher, l'attention portant sur 11 segments
    #          au lieu de 25 x 35 jetons.
    #   exec8  R:R 2.0
    #   exec7  ARCHITECTURE COMPLETE : une attention ET son FFN par axe,
    #          RMSNorm en pre-norm, QK-Norm, RoPE sur l'axe temps, SwiGLU,
    #          LayerScale, plongement numerique periodique, jeton CLS.
    #          Cout mesure : 2.38x le fwd+bwd de la version reduite, a
    #          profondeur egale et etat thermique identique.
    #          AUCUN checkpoint anterieur n'est chargeable.
    # exec12 — MEME reglage qu'exec11, seules les DONNEES changent. exec11
    # avait change trois choses a la fois (arret precoce, modele reduit,
    # entropie remontee) et n'avait rien donne : 0 epoch positif sur 24 apres
    # warmup, entropie tombee de 1.099 a 0.51 malgre le coefficient a 0.030.
    # Le diagnostic etait bon mais le remede portait sur le mauvais terme :
    # 298 k parametres pour 16 708 barres font encore 18 parametres par barre.
    #
    # Le jeu H1 passe de 30 379 a 79 340 barres (2017-08 au lieu de 2023-02,
    # archives Binance spot au lieu de MT5). Donc 5.4 parametres par barre.
    # On ne touche a rien d'autre, sinon exec11 et exec12 ne seraient plus
    # comparables et on ne saurait pas ce qui a agi.
    # exec03 — MEME reglage qu'exec02, UNE SEULE chose change : la validation
    # et le test cessent de vendre dans un run long-only.
    #
    # exec02 a tourne 91 epochs x 3 folds. L'entrainement etait correct — la
    # ligne TRAIN affichait S(0W/0L) +0.00$ a chaque epoch — mais la
    # validation ouvrait des ventes, parce que la substitution par la tete
    # auxiliaire ecrasait le tableau de probabilites ET le masque de cote
    # avec. Ce que cela a coute, par fold et par epoch :
    #
    #     fold   LONG        SHORT       part des trades vendus
    #     wf1    +87.9 $     -286.9 $    68 %
    #     wf2    +421.1 $    -108.5 $    31 %
    #     wf3    +517.1 $    -435.8 $    64 %
    #
    # Les achats rapportaient +1 026 $ par epoch tous folds confondus, les
    # ventes en reprenaient 831. Le "meilleur modele" a donc ete choisi sur
    # un net qui ne mesurait pas la strategie entrainee, et la fenetre de
    # test a ete consommee avec la meme regle : -190 $, +279 $, +525 $, soit
    # +614 $ pour 174 trades, 1.4 ecart-type — indistinguable de zero.
    #
    # LE TEST D'exec03 SERA UNE SECONDE LECTURE de ces memes fenetres. Il
    # faut le lire comme tel : la correction est un defaut repare, pas un
    # reglage choisi sur le test, mais la fenetre n'est plus vierge.
    #
    # Deux verrous plutot qu'un, verifies par `test_cote.py` : la regle de
    # decision connait le cote, et l'entrainement leve si le cote interdit
    # compte un seul trade.
    # exec04 — exec03 plus le PLAFOND D'EPISODES releve de 40 a 160.
    # exec03 confirmait la correction du cote (S(0W/0L) partout) mais tournait
    # encore a 1 107 decisions par epoch pour 4 000 visees : le plafond
    # herite du BTC mordait, et l'acteur recevait le quart du gradient prevu.
    # Seule cette borne change ; le reste est exec03 a l'identique.
    # exec05 — TROIS changements, tous structurels, aucun de reglage.
    #
    #   1. WALK-FORWARD CHAINE. Le fold 2 reprend les poids du fold 1, le 3
    #      ceux du 2 : un seul modele traverse l'histoire du plus ancien au
    #      plus recent, au lieu de trois qui en voient chacun un tiers. La
    #      normalisation est figee sur le train du fold 1, ce qui rend le
    #      transfert exact — la derive mesuree entre folds valait 0.013
    #      ecart-type en median, 0.16 au pire.
    #
    #   2. ENVIRONNEMENT DEUX FOIS PLUS RAPIDE, a resultat identique :
    #      2 584 -> 4 950 barres/s au profileur. Les reductions numpy passent
    #      de 41 a 7 par barre, la boucle de repartition ne parcourt plus les
    #      64 emplacements mais les deux concernes.
    #
    #   3. exec04 est mort a l'epoch 9 (arret de la machine), donc il n'y a
    #      rien a comparer plus loin que cela.
    # exec06 — exec05 sur un cache RAFRAICHI. Le terminal MetaTrader a ete
    # reinstalle et son plafond de barres porte a illimite ; le jeu de l'or
    # va desormais jusqu'au 2026-09-18 (494 052 barres contre 493 497).
    #
    # Les douze annees supplementaires que le terminal propose ont ete
    # ECARTEES apres mesure : 127 barres par semaine contre 1 358, et un ATR
    # relatif de 4.68 points de base contre 6.63 — des fragments epars, pas
    # des seances, et une geometrie qui n'est pas la meme. Le detail est dans
    # `prepare_or.py`.
    # exec07 — NOUVELLE LIGNEE D'OBSERVATION : 261 entrees au lieu de 260.
    #
    # La cinquieme colonne d'etat porte la DISTANCE AU GARDE-FOU DE CREUX.
    # Le modele ne la voyait pas, alors qu'on lui demande de ne pas franchir
    # cette limite : la capacite restante lui en donnait 67 % de la variance,
    # mais le PIC manquait, et deux observations identiques pouvaient
    # correspondre a douze points de creux d'ecart.
    #
    # AUCUN CHECKPOINT ANTERIEUR N'EST CHARGEABLE. `checkpoints.py` filtre
    # sur `n_features` et les refuse au lieu de les charger de travers —
    # verifie.
    # exec08 — LA PENALITE DE CREUX EST UNE RAMPE, PLUS UNE FALAISE.
    #
    # ET C'EST PRECISEMENT POURQUOI LE PREFIXE DOIT CHANGER. L'observation
    # est identique a exec07 — 261 entrees — donc le filtre sur `n_features`
    # de `checkpoints.py` ACCEPTERAIT ces poids sans rien signaler. Ils ont
    # ete entraines contre une recompense differente : le critic a appris a
    # evaluer un monde ou approcher le garde-fou ne coutait rien jusqu'a la
    # marche. Le reprendre ici reviendrait a demarrer avec une fonction de
    # valeur fausse, en silence. La seule barriere est le prefixe.
    # exec09 — L'EPISODE MEURT DE RUINE, PLUS D'UN POURCENTAGE DE CREUX.
    #
    # ET C'EST POURQUOI LE PREFIXE DOIT CHANGER, alors meme que
    # l'observation garde ses 261 entrees : la CINQUIEME A CHANGE
    # D'ECHELLE. Elle valait `creux / 0.40` et saturait a 1 des 40 % de
    # creux ; elle vaut maintenant le creux lui-meme, donc 1 seulement a la
    # ruine. Le filtre sur `n_features` de `checkpoints.py` ne peut pas voir
    # cela — il compte les colonnes, pas ce qu'elles signifient — et
    # accepterait les poids d'exec08 en silence, avec une entree dont le
    # sens a ete divise par 2.5. Le prefixe est la seule barriere.
    # exec10 — lambda passe de 1.25 a 6.5. La recompense change, donc les
    # poids d'exec09 ne sont pas reprenables : leur critic a appris a evaluer
    # un monde ou approcher la ruine coutait cinq fois moins.
    # exec17 — UN SEUL PASSAGE DANS LE TRONC. Changement de COUT, pas de
    # comportement : les sorties sont identiques au bit.
    #
    # Profil de la collecte d'exec16 : le passage avant pese 95 % du temps,
    # le gating 0.3 %, `env.step` 4.6 %. Soit 87 ms par lot pour un reseau de
    # 46 000 parametres — un cout de LANCEMENT, pas de calcul, parce que le
    # tronc etait traverse plusieurs fois par lot :
    #
    #     rollout                4 traversees (direction + budget, x2 membres)
    #     validation / test      6 (+ rendement)
    #     mise a jour PPO        6, et trois graphes d'autograd
    #     passe de classement    4 par lot de 8 192
    #
    # Les tetes ne sont que des couches lineaires sur la MEME representation.
    # `PolitiqueEnsemble.sorties` les rend en un passage, et chaque tete est
    # combinee comme elle l'etait separement — moyenne des probabilites pour
    # la direction et le budget, moyenne directe pour la valeur et le
    # rendement. Ecart maximal verifie : 0.000e+00 sur les quatre.
    #
    # Reprend exec16 : revision horaire du budget, bonus d'entropie sur les
    # deux tetes, `H` ramenee a la direction seule, budget d'episodes compte
    # sur les transitions versees.
    cfg_long.model_prefix = "saintv2_btc_m1_flux01"

    # LE JOURNAL CONSIGNE LA GEOMETRIE, parce que ce depot a deja paye deux
    # fois la meme faute : une regle de sortie changee dans la config pendant
    # que l'etiquette de la tete auxiliaire continuait de decrire l'ancien
    # trade. Les deux tournaient, les deux rendaient des nombres, et rien ne
    # signalait qu'elles ne parlaient plus du meme trade.
    #
    # Les deux lignes se lisent de la MEME source que l'environnement et que
    # l'etiquette — `cibles._regle` et `cibles._regle_cible`. Elles ne peuvent
    # donc pas diverger de ce qui tourne : si elles se contredisent a l'ecran,
    # c'est la configuration qui se contredit.
    # ET ELLE A FAILLI PAYER UNE TROISIEME FOIS. Le 2026-09-21, le passage
    # au scalping M1 a coupe le stop (`use_sl=False`) et le trailing, mais
    # `cibles._regle` lit `cfg.atr_sl_mult` SANS consulter `use_sl` : la
    # banniere aurait annonce un stop de 10xATR devant un systeme qui n'en
    # pose aucun. La ligne censee empecher la divergence l'aurait couverte.
    if getattr(cfg_long, "timeframe_entrainement", "M5") == "M1":
        import cibles_m1 as _CM
        _h = int(getattr(cfg_long, "horizon_cloture", 30))
        # LE BANDEAU NOMME LES DEUX PORTES DE SORTIE, ET IL N'EN NOMMAIT
        # QU'UNE. Quatrieme phrase du meme bandeau, dans la meme soiree,
        # qui affirme ce que le code ne fait plus — apres le nom de
        # l'instrument, la friction et le titre bilateral. Un bandeau se
        # lit en diagonale : c'est precisement la qu'une affirmation
        # perimee survit le plus longtemps.
        _sp_ = float(getattr(cfg_long, "coupe_profit", 0.0))
        _prof_txt = (f"`tete_profit` quand il reste moins de {_sp_:.2f} x le "
                     f"gain deja acquis (et seulement si elle a appris : rho "
                     f">= {getattr(cfg_long, 'rho_profit_min', 0.0):.2f})"
                     if _sp_ > 0.0
                     else "`tete_profit` DESACTIVEE (coupe_profit nul)")
        print(f"SORTIE    AUCUN stop fixe, AUCUN objectif fixe, AUCUN "
              f"trailing — DEUX portes : `tete_cloture` quand la perte "
              f"depasse {getattr(cfg_long, 'coupe_risque', 0.0):.2f} x le "
              f"risque predit, et {_prof_txt}")
        _pl_ = plafond_detention(cfg_long)
        print("          AUCUN plafond de detention — une position vit tant "
              "que les deux tetes la laissent vivre"
              if _pl_ <= 0 else
              f"          plafond de detention {_pl_} minutes "
              f"(`max_holding_bars` pose a la main)")
        # LE BANDEAU DIT LE COUT REELLEMENT FACTURE, pas une constante.
        #
        # Il affichait `cout_aller_retour`, la constante de repli, alors
        # que les etiquettes facturent le spread de CHAQUE barre depuis le
        # 2026-09-21. Un bandeau qui ment sur la geometrie est exactement
        # ce que ce bloc existe pour empecher — il l'a deja fait deux fois
        # dans la meme journee.
        # CE BLOC N'A JAMAIS RIEN AFFICHE D'AUTRE QUE LA CONSTANTE, et il a
        # fallu un mois pour s'en apercevoir. Il appelait `chemin_cache()`,
        # une fonction qui N'EXISTE PAS — la vraie s'appelle
        # `_data_cache_path`. L'`AttributeError` tombait a chaque
        # execution, le `except Exception` nu l'avalait, et le bandeau se
        # rabattait sur la constante en toute discretion.
        #
        # Le commentaire juste au-dessus dit « un bandeau qui ment sur la
        # geometrie est exactement ce que ce bloc existe pour empecher ».
        # Il le faisait depuis le debut.
        #
        # ON N'ATTRAPE DONC PLUS QUE CE QUI EST PREVISIBLE — cache absent,
        # colonne manquante — et on IMPRIME la cause. Un repli muet sur une
        # constante fausse est pire que pas de repli du tout : il donne
        # l'apparence d'un chiffre mesure.
        try:
            import numpy as _np
            import pandas as _pd
            _chem = _data_cache_path(cfg_long)
            _dd = _pd.read_pickle(_chem)
            _sp = _dd["spread_bar"].to_numpy(_np.float64)
            _sp = _sp[_np.isfinite(_sp) & (_sp > 0)]
            if _sp.size == 0:
                raise ValueError("colonne `spread_bar` vide")
            # LE GLISSEMENT VIENT D'`instruments`, il n'est plus en dur :
            # le 0.5 ecrit ici valait pour un seul symbole.
            import instruments as _IN
            _gl = float(_IN.INSTRUMENTS[cfg_long.symbol]["entry_slippage_bps"]) / 2.0
            _ar = 2.0 * _sp + _gl
            _txt = (f"{_np.median(_ar):.2f} bps median (spread REEL de chaque "
                    f"barre, de {_np.percentile(_ar, 5):.2f} a "
                    f"{_np.percentile(_ar, 95):.2f})")
        except (OSError, KeyError, ValueError) as _e:
            _txt = (f"{_CM.cout_aller_retour(cfg_long.symbol):.2f} bps "
                    f"(CONSTANTE — le spread reel n'a pas pu etre lu : {_e})")
        print(f"CIBLE     rendement NET a {_h} minutes, en points de base | "
              f"aller-retour {_txt} deja deduit — c'est exactement ce que "
              f"la position encaisse")
        _ca, _cv = cotes_permises(cfg_long.side)
        print(f"          une seule position a la fois | "
              f"achat {'OUI' if _ca else 'INTERDIT'}, "
              f"vente {'OUI' if _cv else 'INTERDIT'} — "
              f"`tete_achat` et `tete_vente` separees, et un cote interdit "
              f"est une tete SANS GRADIENT")
    else:
        import cibles as _CIB
        _so, _ci = _CIB._regle(cfg_long), _CIB._regle_cible(cfg_long)
        print(f"SORTIE    stop {_so['sl']:g}xATR = 1 R | objectif "
              f"{(str(_so['tp'] / _so['sl']) + ' R') if _so['tp'] else 'AUCUN'} | "
              f"trailing " + (f"depuis {_so['ts'] / _so['sl']:.2f} R, "
                              f"distance {_so['td'] / _so['sl']:.2f} R"
                              if _so["ts"] is not None else "inactif"))
        print(f"CIBLE     stop {_ci['sl']:g}xATR = 1 R | objectif "
              f"{_ci['tp'] / _ci['sl']:.2f} R — ce que la tete auxiliaire apprend "
              f"a CLASSER, pas ce que la position encaisse")

    # Chaque fold repart de zéro avec les statistiques de son train.
    print(f"Walk-forward {cfg_long.model_prefix} {cfg_long.side.upper()} ({cfg_long.symbol} seul) : trois folds CHAINES, le meilleur checkpoint de chacun ouvre le suivant.")
    # ==================================================================
    # DEUX ARCHITECTURES DANS LE MEME RUN, pour que le vote existe.
    #
    # `evalue_ensemble.py` fait voter trois modeles choisis pour se tromper
    # DIFFEREMMENT :
    #
    #     TabM       supervise, regression du rendement net, MLP ensemble
    #     PatchTST   PPO, canaux INDEPENDANTS : ne croise jamais les colonnes
    #     SAINT      PPO, attention sur les features : ne fait que les croiser
    #
    # Les deux reseaux PPO sont opposes par construction sur la question qui
    # separe le mieux les modeles de ce depot. C'est la condition pour qu'un
    # vote reduise la variance : des erreurs correlees ne s'annulent pas.
    #
    # POURQUOI ILS DOIVENT PARTIR ENSEMBLE. Un vote n'a de sens que si les
    # votants lisent la MEME observation. Jusqu'ici l'ensemble pointait sur
    # exec18 (PatchTST) et exec20 (SAINT), tous deux a 107 features en H1,
    # pendant que le run courant en portait 264 : il n'y avait plus aucun
    # PatchTST sur l'observation en service, donc plus de vote possible.
    # Les entrainer dans le meme processus garantit qu'ils partagent le jeu,
    # la geometrie, la normalisation par fold et les fenetres — donc que la
    # comparaison est APPARIEE barre a barre.
    #
    # SEQUENTIEL, PAS PARALLELE. Deux entrainements simultanes sur cette carte
    # ont deja ete essayes : la temperature est montee a 89 C, l'horloge est
    # tombee a 315 MHz, et les deux processus ecrivaient les memes checkpoints.
    # Le GPU est deja bride thermiquement a un seul run.
    # ==================================================================
    # UN SEUL PASSAGE SUR LES TROIS FOLDS. La version precedente entrainait
    # SAINT puis PatchTST a la suite — deux fois trois folds, et surtout deux
    # politiques qui ne s'etaient jamais vues. Le vote n'existait alors qu'a
    # l'evaluation, sur des reseaux entraines chacun a agir seul.
    #
    # Ils partagent maintenant le rollout : l'action executee est celle du
    # melange, et chacun apprend a jouer SA part dans une decision commune.
    run_walkforward(cfg_long, train_frac=0.55, val_frac=0.15, test_frac=0.10,
                    max_folds=3, start_fold=1,
                    bootstrap_from_path=None,
                    chaine=True)

    print("\n" + "=" * 70)
    print("  WALK-FORWARD LONG-ONLY TERMINÉ : 3 folds.")
    # LE BILAN LISAIT UNE LISTE ECRITE EN DUR, et elle mentait.
    #
    # Il annoncait `best_saintv2_or_exec17_long_wf1_long_wf1.pth` et ses
    # deux freres — trois fichiers d'un AUTRE run, sur un autre instrument,
    # qu'aucun des runs recents n'a produits. Le run du 2026-09-22 s'est
    # termine sans RETENIR un seul modele — chaque epoch finissait sur
    # « garde » — et son bilan affirmait pourtant trois sauvegardes.
    #
    # Cinquieme phrase de la meme famille en trois jours : apres le nom de
    # l'instrument, la friction, le titre bilateral et le nombre de portes
    # de sortie. Un bilan de fin de run est lu une fois, en diagonale, et
    # c'est la qu'une affirmation fausse coute le plus — on croit avoir un
    # modele deployable.
    #
    # ON LIT DONC LE DISQUE. Ce qui existe est liste ; ce qui manque est
    # dit en clair.
    import glob as _glob
    _pref = cfg_long.model_prefix
    print("  Modeles RETENUS (sauvegardes sur le critere net) :")
    _vus = sorted(_glob.glob(f"best_{_pref}*.pth"))
    if _vus:
        for _f in _vus:
            print(f"    {_f}")
    else:
        print(f"    AUCUN. Aucune epoch n'a battu le critere de retenue — "
              f"aucun modele de ce run n'est deployable.")
    _tem = sorted(_glob.glob(f"bestprofit_{_pref}*.pth"))
    if _tem:
        print("  Temoins (meilleur PnL/trade, NE decident PAS du deploiement) :")
        for _f in _tem:
            print(f"    {_f}")
    print("=" * 70)

    # ---------------------------------------------------------
    # Alternative : entraîner LONG et SHORT séparément
    # (à dé-commenter si tu veux des modèles spécialisés au lieu du duel)
    # ---------------------------------------------------------
    # cfg_long = PPOConfig(**cfg_base.__dict__)
    # cfg_long.side = "long"
    # cfg_long.model_prefix = "saintv2_loup_long"
    # run_walkforward(cfg_long, train_frac=0.55, val_frac=0.15, test_frac=0.10, max_folds=3)
    #
    # cfg_short = PPOConfig(**cfg_base.__dict__)
    # cfg_short.side = "short"
    # cfg_short.model_prefix = "saintv2_loup_short"
    # run_walkforward(cfg_short, train_frac=0.55, val_frac=0.15, test_frac=0.10, max_folds=3)

    # (Mode "close" retiré : l'agent ne ferme plus manuellement, uniquement via SL/TP/trailing)
