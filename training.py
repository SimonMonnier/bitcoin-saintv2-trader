# ======================================================================
# PPO + SAINTv2 — SCALPING BTCUSD M1 (SINGLE-HEAD + ACTION MASK + H1)
# Version "Loup Ω" LONG / SHORT / CLOSE
# ======================================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    N_ACTIONS,
    MASK_VALUE,
    NORM_STATS_PATH,
    FEATURE_COLS,
    OBS_N_FEATURES,
    merge_m1_h1,
    ATR_PLANCHER_FRAC,
    charge_source_externe,
    SOURCE_EXT_NOM,
    safe_normalize,
    SeuilRang,
    N_BLOCS_DEFAUT,
    EntryDecisionPolicy,
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
SELECTIVITE_FIN = 0.05     # régime : les 5 % les plus favorables
SELECTIVITE_RAMP_EPOCHS = 40


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
    lookback: int = 4
    # Le decoupage suit le lookback : des segments de 16 barres n'existent pas
    # dans une fenetre de 4. Segment 2 et pas 1 donnent trois jetons, donc une
    # attention qui a encore quelque chose a faire ; un segment de 4 n'en
    # donnerait qu'un seul et l'encodeur deviendrait un simple MLP.
    taille_patch: int = 2
    pas_patch: int = 1

    # "M1" ou "H1". Choisit le cache et, avec lui, l'echelle de decision.
    timeframe_entrainement: str = "M5"

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
    architecture: str = "saint"

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
    mlp_dim: int = 32

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
    epochs: int = 40
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
    episodes_per_epoch: int = 336
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
    episode_length: int = 1440      # 5 jours de M5
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
    max_decisions_per_epoch: int = 40_000
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
    gamma: float = 0.995
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
    entropy_coef: float = 0.015

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

    # SAINT
    # 80 -> 8. Avec une seule tete, head_dim vaut 8, ce que
    # saint_core._verifie_dim_tete exige pour SDPA.
    d_model: int = 8
    # Frequences de l'embedding numerique par colonne. 16 en faisait le second
    # poste du reseau (29 %) ; 4 le ramene a un niveau comparable aux blocs.
    saint_n_freq: int = 4
    saint_heads: int = 1
    saint_mlp_dim: int = 16
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
    min_capital_frac: float = 0.20  # garde-fou capital : épisode terminé si capital < 20%
                                     # config validée +135 EUR MT5 tester
    max_drawdown: float = 0.40  # garde-fou DD : épisode terminé si drawdown > 40%
                                 # config validée +135 EUR MT5 tester

    # Risk management / position sizing
    risk_per_trade: float = 0.012
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
    atr_sl_mult: float = 8.0
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
    atr_tp_mult: float = 16.0

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
    spread_bps: float = 2.61
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
    scalping_max_holding: int = 30

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
    use_be_trail: bool    = False  # mettre à True pour réactiver
    atr_be_mult: float    = 1.0   # gain en ATR pour déclencher le break-even
    atr_trail_mult: float = 1.5   # gain en ATR pour déclencher le trailing
    atr_trail_dist: float = 1.0   # distance du trailing (en ATR)

    # Warmup critique : N epochs où seul le critique est mis à jour
    critic_warmup_epochs: int = 5

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
    # Valeur 5 % : la cible finale du calendrier d'entrainement, donc le regime
    # qu'on cherche reellement a atteindre. Assez selectif pour approcher la
    # zone ou la sonde mesurait un avantage, assez large pour garder une
    # centaine de trades par epoch — au-dela, l'estimation du winrate devient
    # trop bruitee pour suivre une progression.
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
    # PPO requiert les probabilites de la politique qui a tire les actions.
    # L'ancien curriculum forcait BUY/SELL ou remappait en HOLD sans corriger
    # logprob. Conserve uniquement pour reproduire les anciens diagnostics.
    legacy_off_policy_curriculum: bool = False

    # Curriculum vol
    use_vol_curriculum: bool = True

    # Cache disque du dataframe fusionné (évite ~15 min de recalcul d'indicateurs
    # à chaque lancement). Invalidé si le cache accuse plus de N heures de retard.
    use_data_cache: bool = True
    data_cache_max_lag_hours: float = 48.0

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
    side: str = "both"

    # Préfixe pour nommer les fichiers de modèle
    model_prefix: str = "saintv2_singlehead_scalping_ohlc_indics_h1_loup"



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
    tf = getattr(cfg, "timeframe_entrainement", "M1")
    if tf == "M5":
        return "data_cache_BTCUSD_M5.pkl"
    if tf == "H1":
        return "data_cache_BTCUSD_H1.pkl"
    return f"data_cache_{cfg.symbol}_{cfg.date_from:%Y%m%d}.pkl"


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
        self.close = df["close"].values.astype(np.float32)
        self.open = df["open"].values.astype(np.float32) if "open" in df.columns else self.close.copy()
        self.length = len(df)

        self.atr14 = df["atr_14"].values.astype(np.float32) if "atr_14" in df.columns else np.zeros(len(df), np.float32)
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

        self.capital = self.cfg.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.current_size = 0.0

        self.sl_price = 0.0
        self.tp_price = 0.0
        self.entry_idx = -1
        self.entry_atr = 0.0
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

        self.bars_in_position = 0
        self.risk_scale = 1.0
        self.last_risk_scale = 1.0

        self.max_dd = 0.0

        # Gestion dynamique du SL
        self.break_even_done = False
        self.trail_active    = False

        obs = self._get_obs()
        return obs, {
            "capital": self.capital,
            "position": self.position,
            "drawdown": 0.0,
            "done_reason": None
        }

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
        risk_feature = float(self.last_risk_scale)

        extra_vec = np.array(
            [pos_feature, unrealized_atr, bars_held_norm, risk_feature],
            dtype=np.float32
        )
        extra_block = np.repeat(extra_vec[None, :], self.lookback, axis=0)

        obs = np.concatenate([base, extra_block], axis=-1).astype(np.float32)
        return obs

    def _apply_micro(self, price: float, side: int, is_entry: bool = True) -> float:
        """Convertit une bougie BID en prix BUY/SELL exécutable."""
        price = execution_quote(price, side, self.current_trade_spread_bps)
        if is_entry and self.cfg.entry_slippage_bps > 0:
            extra = np.random.uniform(0.0, self.cfg.entry_slippage_bps) / 10_000.0
            price *= (1 + side * extra)
        return price

    def _sample_trade_spread_bps(self) -> float:
        """Échantillonne le spread pour le trade qui démarre.

        Distribution bimodale calibrée sur les logs MT5 Vantage BTCUSD :
          - (1 - wide_prob) % : spread tight ≈ valeur centrale ±2% (session active)
                                 ex : 2.45-2.55 bps (16.7-17.3$ sur BTC 68k)
          - wide_prob %        : spread élargi 1.2× à wide_factor× (overnight / news)
                                 ex : 3.0-7.5 bps (20-50$ sur BTC 68k)
        """
        base = float(self.cfg.spread_bps)
        if base <= 0.0:
            return 0.0
        if np.random.rand() < self.cfg.spread_wide_prob:
            low  = base * 1.2
            high = base * self.cfg.spread_bps_wide_factor
            return float(np.random.uniform(low, high))
        else:
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

    def _latent_at_bid(self, bid):
        if self.position == 0:
            return 0.0
        quote = execution_quote(bid, -self.position, self.current_trade_spread_bps)
        return (self.position * (quote - self.entry_price)
                - self.cfg.fee_rate * quote) * self.current_size

    def _close_position(self, exit_price, hit_sl=False, hit_tp=False,
                        hit_temps=False, terminal_reason=None):
        pnl = self.position * (exit_price - self.entry_price) * self.current_size
        fee = self.cfg.fee_rate * exit_price * self.current_size
        realized = pnl - fee

        self.capital += realized
        self.last_realized_pnl = realized
        self.trades_pnl.append(realized)
        self.trades_side.append(int(self.position))
        self.trades_meta.append({
            "entry_idx": int(self.entry_idx),
            "exit_idx": int(self.idx),
            "side": int(self.position),
            "entry_price": float(self.entry_price),
            "exit_price": float(exit_price),
            "pnl": float(realized),
            "hit_sl": bool(hit_sl),
            "hit_tp": bool(hit_tp),
            "hit_temps": bool(hit_temps),
            "terminal_reason": terminal_reason,
            "hold_bars": int(self.idx - self.entry_idx),
        })

        self.position = 0
        self.current_size = 0.0
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.entry_idx = -1
        self.entry_atr = 0.0
        self.risk_scale = 1.0
        self.last_risk_scale = 1.0
        self.break_even_done = False
        self.trail_active    = False
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
        prev_equity = self.capital + self._latent_at_bid(
            self.data.close[self.idx - 1] if self.idx > 0 else price)

        realized_trade = 0.0
        hit_sl = hit_tp = hit_temps = False

        if old_pos != 0:
            self.bars_in_position += 1
        else:
            self.bars_in_position = 0

        # Plus de mode "close" : fermeture uniquement par SL/TP/break-even/trailing
        manual_close = False

        # --------- OUVERTURE DIRECTE (pas de confirmation dans l'env) ---------
        # La confirmation signal→pause→re-signal est gérée dans kairos_live.py uniquement.
        # En training, le reward shaping pénalise déjà les mauvaises entrées.
        if not manual_close and action in (0, 1) and old_pos == 0:
            side = 1 if action == 0 else -1
            size = self._compute_dynamic_size(prix_execution)
            if size > 0.0:
                self.current_size = size
                self.position = side
                # Échantillonne le spread pour ce trade (variabilité réaliste)
                self.current_trade_spread_bps = self._sample_trade_spread_bps()
                exec_price = self._apply_micro(prix_execution, side, is_entry=True)
                self.entry_price = exec_price
                self.entry_idx = self.idx

                atr_raw = float(self.data.atr14[self.idx - 1]) if self.idx - 1 >= 0 else 0.0
                # MEME plancher que saint_core.effective_atr — le dupliquer ici
                # avec une autre valeur ferait diverger l'entrainement du live.
                fallback = ATR_PLANCHER_FRAC * exec_price
                self.entry_atr = max(atr_raw, fallback, 1e-8)

                sl_dist = self.cfg.atr_sl_mult * self.entry_atr
                tp_dist = self.cfg.atr_tp_mult * self.entry_atr * self.cfg.tp_shrink

                # Montant réellement en jeu si le stop est touché. C'est l'unité
                # dans laquelle la récompense du trade sera exprimée : un stop
                # vaut -1, une cible à R:R 2 vaut +2, partout et toujours.
                self.risk_amount = float(sl_dist * self.current_size)

                if side == 1:
                    self.sl_price = max(1e-8, exec_price - sl_dist)
                    self.tp_price = max(1e-8, exec_price + tp_dist)
                else:
                    self.sl_price = max(1e-8, exec_price + sl_dist)
                    self.tp_price = max(1e-8, exec_price - tp_dist)

                self.break_even_done = False
                self.trail_active    = False

        # --------- BREAK-EVEN + TRAILING STOP ---------
        # Désactivé par défaut (cfg.use_be_trail=False) pour aligner sur le
        # backtest "no_be_trail" et sur le MQL5 SaintV2_WF3 (sans BE/trail).
        # Permet d'isoler la qualité du signal d'entrée pur (SL/TP fixes uniquement).
        if (self.cfg.use_be_trail and not manual_close and self.position != 0
                and self.current_size > 0 and self.entry_atr > 1e-8
                and self.idx > self.entry_idx):
            # Le SL de cette bougie ne peut utiliser que les bougies déjà closes.
            fav_bid = self.data.high[self.idx - 1] if self.position == 1 else self.data.low[self.idx - 1]
            fav_price = execution_quote(fav_bid, -self.position, self.current_trade_spread_bps)
            fav_move  = self.position * (fav_price - self.entry_price)

            # Break-even : déplace SL à l'entrée quand gain >= atr_be_mult × ATR
            if not self.break_even_done and fav_move >= self.cfg.atr_be_mult * self.entry_atr:
                if self.position == 1:
                    self.sl_price = max(self.sl_price, self.entry_price)
                else:
                    self.sl_price = min(self.sl_price, self.entry_price)
                self.break_even_done = True

            # Trailing stop : suit le prix à atr_trail_dist × ATR quand gain >= atr_trail_mult × ATR
            if fav_move >= self.cfg.atr_trail_mult * self.entry_atr:
                trail_sl = fav_price - self.position * self.cfg.atr_trail_dist * self.entry_atr
                if self.position == 1:
                    self.sl_price = max(self.sl_price, trail_sl)
                else:
                    self.sl_price = min(self.sl_price, trail_sl)
                self.trail_active = True

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
        if not manual_close and self.position != 0 and self.current_size > 0 and self.entry_price > 0:
            exit_price = None
            open_bid = getattr(self.data, 'open', self.data.close)[self.idx]
            open_quote = execution_quote(open_bid, -self.position, self.current_trade_spread_bps)

            if self.position == 1:  # LONG
                if self.sl_price > 0 and low_bar <= self.sl_price:
                    exit_price = min(self.sl_price, open_quote)
                    hit_sl = True
                elif self.tp_price > 0 and high_bar >= self.tp_price:
                    exit_price = self.tp_price
                    hit_tp = True
            else:                    # SHORT
                if self.sl_price > 0 and execution_quote(high_bar, 1, self.current_trade_spread_bps) >= self.sl_price:
                    exit_price = max(self.sl_price, open_quote)
                    hit_sl = True
                elif self.tp_price > 0 and execution_quote(low_bar, 1, self.current_trade_spread_bps) <= self.tp_price:
                    exit_price = self.tp_price
                    hit_tp = True

            # Ni SL ni TP : cloture AU MARCHE au plafond de detention.
            # INACTIF par defaut (max_holding_bars = 0) : le live ne sait pas
            # fermer au marche, donc l'environnement ne le fait pas non plus.
            if (exit_price is None and self.cfg.max_holding_bars > 0
                    and self.bars_in_position >= self.cfg.max_holding_bars):
                exit_price = self._apply_micro(price, -self.position, is_entry=False)
                hit_temps = True

            if exit_price is not None:
                # SL/sortie au marché: slippage adverse. TP: niveau cible,
                # sans amélioration favorable systématique inventée.
                slip_max = self.cfg.slippage_bps / 10_000.0
                if slip_max > 0 and not hit_tp:
                    slip_amount = exit_price * np.random.uniform(0.0, slip_max)
                    # Une sortie au marche (temps ecoule) traverse le spread :
                    # elle est DEFAVORABLE comme un SL, jamais favorable comme
                    # un TP ou le momentum joue pour nous.
                    contre = hit_sl or hit_temps
                    if self.position == 1:   # LONG
                        exit_price += (-slip_amount if contre else slip_amount)
                    else:                     # SHORT
                        exit_price += (slip_amount if contre else -slip_amount)

                realized_trade = self._close_position(exit_price, hit_sl, hit_tp, hit_temps)

        # Les fenêtres sont des épisodes FINIS: toute position restante est
        # liquidée et comptée avant de calculer la dernière récompense.
        marked_equity = self.capital + self._latent_at_bid(price)
        marked_peak = max(self.peak_capital, marked_equity)
        marked_dd = (marked_peak - marked_equity) / (marked_peak + 1e-8)
        done_reason = None
        if self.idx + 1 >= self.end_idx:
            done_reason = "episode_end"
        elif marked_dd > self.cfg.max_drawdown:
            done_reason = "max_drawdown"
        elif marked_equity < self.cfg.initial_capital * self.cfg.min_capital_frac:
            done_reason = "min_capital"
        if done_reason is not None and self.position != 0:
            exit_price = self._apply_micro(price, -self.position, is_entry=False)
            slip = np.random.uniform(0.0, self.cfg.slippage_bps) / 10000.0 if self.cfg.slippage_bps > 0 else 0.0
            exit_price *= 1.0 - self.position * slip
            realized_trade += self._close_position(exit_price, terminal_reason=done_reason)

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
        log_ret = max(min(log_ret, 0.05), -0.05)

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
        if realized_trade != 0.0:
            # Exprimé en R : le montant risqué sur CE trade sert d'unité. Un
            # stop touché vaut -1, la cible (R:R 2.0) vaut +2, une sortie au
            # temps vaut ce qu'elle vaut entre les deux. Diviser par une
            # constante en dollars, comme avant, rendait ce terme dépendant du
            # prix du symbole : 0.81 sur Bitcoin, 0.007 sur l'or.
            reward += float(np.clip(
                realized_trade / max(getattr(self, "risk_amount", 0.0), 1e-8),
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
        if hasattr(self.cfg, "current_epoch") and self.cfg.current_epoch >= 10:
            reward = float(np.clip(reward, -3.5, 3.5))

        # ==================================================================
        # Finalisation
        # ==================================================================
        self.peak_capital = max(self.peak_capital, equity)
        dd = (self.peak_capital - equity) / (self.peak_capital + 1e-8)
        self.max_dd = max(self.max_dd, dd)

        # Pénalité DD réactivée (max_drawdown=0.40) — punit les trajectoires catastrophiques
        if dd > self.cfg.max_drawdown:
            reward -= 0.2

        self.idx += 1
        done = done_reason is not None

        obs = self._get_obs()

        return obs, float(reward), done, False, {
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

def run_training_on_split(
    train_data: MarketData,
    calib_data: MarketData,
    val_data: MarketData,
    test_data: MarketData,
    stats: Dict[str, np.ndarray],
    cfg: PPOConfig,
    suffix: str = ""
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
        "buy_ratio", "sell_ratio", "hold_ratio",
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

    # Pools d'environnements créés UNE SEULE FOIS : le constructeur calcule les
    # quantiles de volatilité du curriculum sur tout le split (np.quantile sur
    # ~1.2M lignes), donc les instancier à chaque epoch coûterait plusieurs
    # minutes par fold pour rien. Ils sont simplement reset() à chaque epoch.
    # Tous partagent le même objet cfg, donc cfg.current_epoch les pilote tous.
    train_envs = [
        BTCTradingEnvDiscrete(train_data, cfg) for _ in range(cfg.episodes_per_epoch)
    ]

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

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-8)

    # Groupes de paramètres pour diagnostiquer d'où vient le gradient.
    actor_head_params = [p for n, p in policy.named_parameters() if n.startswith("actor.")]
    critic_head_params = [p for n, p in policy.named_parameters() if n.startswith("critic.")]
    trunk_params = [
        p for n, p in policy.named_parameters()
        if not n.startswith(("actor.", "critic."))
    ]

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.05
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
    best_metric = -1e9
    best_state = None
    best_thresholds = None
    best_decision_spec = None
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
        batch_actions = []
        batch_oldlog = []
        batch_adv = []
        batch_returns = []
        batch_values = []
        batch_positions = []

        total_reward_epoch = 0.0
        epoch_pnl = []
        epoch_dd = []
        epoch_trades_pnl: List[float] = []
        epoch_trades_side: List[int] = []  # +1 long, -1 short — index aligné avec epoch_trades_pnl

        action_counts_env = np.zeros(3, dtype=np.int64)

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
        envs = train_envs
        n_envs = len(envs)

        states: List[np.ndarray] = []
        infos: List[Dict] = []
        for e in envs:
            s0, i0 = e.reset()
            states.append(s0)
            infos.append(i0)

        ep_buf = [
            {"states": [], "actions": [], "logprobs": [], "rewards": [],
             "values": [], "dones": [], "positions": [], "dts": []}
            for _ in range(n_envs)
        ]
        last_reason = [None] * n_envs
        active = list(range(n_envs))

        # Sélectivité visée cette epoch, et seuil absolu qui la réalise.
        # Le seuil vient du quantile mesuré à l'epoch PRÉCÉDENTE : à l'epoch 1 il
        # n'existe pas encore, on laisse alors passer toutes les décisions plutôt
        # que d'inventer une valeur.
        selectivite = selectivite_for_epoch(epoch)
        conf_thr = calib_thr_courant
        pbs_epoch: List[float] = []   # max(p_BUY, p_SELL) sur les états flat

        # Probabilité d'ouverture forcée (curriculum), constante sur l'epoch
        if epoch <= 15:
            force_prob = max(0.15, 0.92 - epoch * 0.05)
        elif epoch <= 35:
            force_prob = 0.25
        else:
            force_prob = 0.05

        if not cfg.legacy_off_policy_curriculum:
            force_prob = 0.0
            conf_thr = 0.0

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
        pending: List[Optional[Dict]] = [None] * n_envs
        sampling_audit = {"decisions": 0, "forced_actions": 0,
                          "remapped_actions": 0, "max_logprob_error": 0.0}

        def _cloture(k: int, done_flag: bool) -> None:
            """Ferme la décision en cours de l'env k et la verse au buffer."""
            p = pending[k]
            if p is None:
                return
            buf = ep_buf[k]
            buf["positions"].append(0)          # une décision est toujours flat
            buf["states"].append(p["state"])
            buf["actions"].append(p["action"])
            buf["logprobs"].append(p["logprob"])
            buf["rewards"].append(p["R"])
            buf["values"].append(p["value"])
            buf["dones"].append(done_flag)
            buf["dts"].append(max(p["dt"], 1))
            pending[k] = None

        while active:
            # 1) Les positions qui viennent de se clore terminent leur décision.
            for k in active:
                if pending[k] is not None and infos[k].get("position", 0) == 0:
                    _cloture(k, False)

            # 2) Nouvelle décision pour les envs sans décision en cours.
            #    Ce sont les seuls à nécessiter un forward.
            deciding = [k for k in active if pending[k] is None]
            actions_env = {k: 2 for k in active}

            if deciding:
                batch_np = np.stack([states[k] for k in deciding], axis=0)
                s_tensor = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
                # Tous ces envs sont flat : un seul masque suffit.
                masks_b = torch.from_numpy(
                    np.repeat(MASK_FLAT[None, :], len(deciding), axis=0)
                ).to(device)

                with torch.no_grad():
                    logits_b, values_b = policy(s_tensor)
                    logits_mb = logits_b.masked_fill(~masks_b, MASK_VALUE)
                    logp_b = torch.log_softmax(logits_mb, dim=-1)
                    packed = torch.cat(
                        [logp_b, values_b.reshape(-1, 1)], dim=1
                    ).cpu().numpy()

                logp_np = packed[:, :N_ACTIONS]
                vals_np = packed[:, N_ACTIONS]
                probs_np = np.exp(logp_np)

                for bi, k in enumerate(deciding):
                    # ---- CURRICULUM D'OUVERTURE FORCÉE ----
                    a = None
                    if np.random.rand() < force_prob:
                        sampling_audit["forced_actions"] += 1
                        if cfg.side == "long":
                            a = 0
                        elif cfg.side == "short":
                            a = 1
                        else:
                            # Équilibré : biaiser vers SELL sur un sous-jacent
                            # passé de 25k$ à 89k$ revenait à nourrir le modèle
                            # d'exemples perdants.
                            a = random.choice([0, 1])

                    # Conviction de trade sur cet état, quelle que soit l'action
                    # finalement prise : c'est la distribution de ces valeurs qui
                    # calibre le seuil, elle doit donc être collectée sur TOUS
                    # les états flat, y compris ceux ouverts de force.
                    pbs_epoch.append(float(max(probs_np[bi, 0], probs_np[bi, 1])))

                    if a is None:
                        p = probs_np[bi]
                        a = int(np.random.choice(N_ACTIONS, p=p / p.sum()))
                        if a != 2 and p[a] < conf_thr:
                            sampling_audit["remapped_actions"] += 1
                            a = 2  # conviction insuffisante → attendre

                    sampling_audit["decisions"] += 1
                    normalized_p = rollout_action_probabilities(
                        probs_np[bi], force_prob, conf_thr, cfg.side)
                    sampling_audit["max_logprob_error"] = max(
                        sampling_audit["max_logprob_error"],
                        abs(float(np.log(normalized_p[a])) - float(logp_np[bi, a])))

                    pending[k] = {
                        "state": states[k],
                        "action": a,
                        "logprob": float(logp_np[bi, a]),
                        "value": float(vals_np[bi]),
                        "R": 0.0,
                        "dt": 0,
                    }
                    actions_env[k] = a

            # 3) Un pas pour tous les envs actifs.
            still_active = []
            for k in active:
                env_k = envs[k]
                env_action = actions_env[k]
                env_k.set_risk_scale(1.0)
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

                p = pending[k]
                if p is not None:
                    p["R"] += (cfg.gamma ** p["dt"]) * reward
                    p["dt"] += 1

                states[k] = ns
                infos[k] = info

                if done:
                    last_reason[k] = info.get("done_reason")
                    _cloture(k, True)
                else:
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

            buf = ep_buf[k]
            adv, ret = compute_gae_semi_mdp(
                buf["rewards"], buf["values"], buf["dones"], buf["dts"],
                cfg.gamma, cfg.lambda_gae, last_value
            )

            batch_states.extend(buf["states"])
            batch_actions.extend(buf["actions"])
            batch_oldlog.extend(buf["logprobs"])
            batch_adv.extend(adv)
            batch_returns.extend(ret)
            batch_values.extend(buf["values"])
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
        _sel = np.flatnonzero(_keep)

        # Diagnostic : la normalisation des avantages mélange-t-elle des
        # échelles incompatibles ? Une décision de trade accumule sa récompense
        # sur ~450 bougies, une attente sur 1 seule. Si l'écart-type est dicté
        # par les trades, les attentes tombent à ~0 après normalisation et ne
        # portent plus de gradient.
        _adv_np = np.asarray(batch_adv, dtype=np.float64)
        _adv_std_raw = float(_adv_np.std()) if len(_adv_np) else 0.0
        _adv_norm = (_adv_np - _adv_np.mean()) / (_adv_std_raw + 1e-8) * 1.5
        _frac_clip = float(np.mean(np.abs(_adv_norm) >= 15.0 - 1e-6)) if len(_adv_np) else 0.0
        _frac_nul = float(np.mean(np.abs(_adv_norm) < 0.05)) if len(_adv_np) else 0.0

        batch_states = [batch_states[i] for i in _sel]
        batch_actions = [batch_actions[i] for i in _sel]
        batch_oldlog = [batch_oldlog[i] for i in _sel]
        batch_adv = [batch_adv[i] for i in _sel]
        batch_returns = [batch_returns[i] for i in _sel]
        batch_values = [batch_values[i] for i in _sel]
        batch_positions = [batch_positions[i] for i in _sel]

        # --------- tenseurs batch ---------
        states_np = np.stack(batch_states, axis=0)
        states = torch.tensor(states_np, dtype=torch.float32, device=device)

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
        oldlog = torch.tensor(batch_oldlog, dtype=torch.float32, device=device).view(-1)
        advantages = torch.tensor(batch_adv, dtype=torch.float32, device=device)
        returns = torch.tensor(batch_returns, dtype=torch.float32, device=device)
        values_old = torch.tensor(batch_values, dtype=torch.float32, device=device)
        positions = torch.tensor(batch_positions, dtype=torch.long, device=device)

        assert states.size(0) == oldlog.size(0) == actions.size(0) == values_old.size(0) == positions.size(0)

        # Normalisation des avantages sur les seuls états FLAT.
        #
        # ~97.5% du batch est constitué d'états en position, où le masque ne
        # laisse qu'une action légale : log π(HOLD) = 0 exactement, donc
        # ∇log π = 0 — ils ne portent aucun gradient de politique. Calculer
        # moyenne et écart-type sur l'ensemble laissait ces 97.5% dicter la
        # calibration des 2.5% qui décident réellement.
        # (Mesuré via l'identité H = fraction_flat × Hflat, l'entropie des
        #  états en position étant nulle par construction.)
        _flat_all = (positions == 0)
        if bool(_flat_all.any()):
            _adv_ref = advantages[_flat_all]
            advantages = (advantages - _adv_ref.mean()) / (_adv_ref.std() + 1e-8)
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.clamp(-10, 10)
        advantages = advantages * 1.5

        epoch_actor_loss = []
        epoch_critic_loss = []
        epoch_entropy = []
        epoch_entropy_flat = []
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

        for upd in range(0 if epoch_moyenne else cfg.updates_per_epoch):
            np.random.shuffle(idx)

            for start in range(0, n_samples, cfg.batch_size):
                end = start + cfg.batch_size
                ids = idx[start:end]

                sb = states[ids]
                ab = actions[ids]
                lb_old = oldlog[ids]
                adv_b = advantages[ids]
                ret_b = returns[ids]
                val_old = values_old[ids]
                pos_b = positions[ids]

                with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
                    logits, value = policy(sb)
                    mask_batch = build_action_mask_from_positions(pos_b, cfg.side)
                    logits_masked = logits.masked_fill(~mask_batch, MASK_VALUE)

                    # ──────────────────────────────────────────────────────────
                    # GARDE-FOU 1 : clamp des logits avant softmax pour éviter
                    # overflow numérique → cause directe des NaN observés.
                    # ──────────────────────────────────────────────────────────
                    logits_masked = torch.clamp(logits_masked, min=-30.0, max=30.0)

                    # GARDE-FOU 2 : si NaN/Inf détectés (ex: gradients précédents
                    # ont corrompu les poids), skip ce batch et reset l'optimizer
                    if torch.isnan(logits_masked).any() or torch.isinf(logits_masked).any():
                        print(f"  {_col('⚠ NaN/Inf dans logits, skip batch', _C.RED)}")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    dist = Categorical(logits=logits_masked)
                    new_log = dist.log_prob(ab)
                    entropy_per_state = dist.entropy()
                    entropy = entropy_per_state.mean()

                    # Entropie restreinte aux états FLAT. En position, 2 actions
                    # sur 3 sont masquées à -1e4 donc l'entropie y vaut 0 par
                    # construction et tire la moyenne vers le bas : `entropy`
                    # seul ne permet pas de distinguer « policy saturée » de
                    # « beaucoup d'états en position ». H_flat le dit sans
                    # ambiguïté — max = ln(3) = 1.0986.
                    _flat_b = (pos_b == 0)
                    entropy_flat = (
                        entropy_per_state[_flat_b].mean()
                        if bool(_flat_b.any()) else entropy
                    )

                    # GARDE-FOU 3 : ratio PPO clampé pour éviter exp() explosif
                    log_ratio = new_log - lb_old
                    log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
                    ratio = log_ratio.exp()
                    surr1 = adv_b * ratio
                    surr2 = adv_b * torch.clamp(
                        ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps
                    )
                    # Objectif de l'actor restreint aux états FLAT : seuls ceux-là
                    # portent une décision. En position le masque ne laisse que
                    # HOLD, donc ratio = 1 et gradient nul — les inclure dans la
                    # moyenne divisait le gradient de politique par ~40 (97.5%
                    # du batch) sans rien y ajouter.
                    # Le critique, lui, garde TOUT le batch : un état en position
                    # porte bien de l'information de valeur.
                    _surr = torch.min(surr1, surr2)
                    actor_loss = (
                        -_surr[_flat_b].mean() if bool(_flat_b.any())
                        else -_surr.mean() * 0.0
                    )

                    value_pred = value.squeeze(-1)
                    v_clipped = val_old + (value_pred - val_old).clamp(-0.2, 0.2)
                    unclipped_loss = (value_pred - ret_b).pow(2)
                    clipped_loss = (v_clipped - ret_b).pow(2)
                    critic_loss = torch.max(unclipped_loss, clipped_loss).mean()

                    # Régularisation d'entropie : maintient l'exploration sans
                    # empêcher la conviction.
                    #
                    # L'ancien réglage (coef 0.30 × 2.0 = 0.57 effectif) clouait
                    # Hflat à ~1.09, soit p_max ≈ 0.395 — juste SOUS le seuil de
                    # production de 0.40. Le modèle ne pouvait structurellement
                    # pas passer son propre filtre de conviction. Ce coefficient
                    # avait été monté à 0.30 pour compenser un gradient d'actor
                    # cassé (dilué par les 97% d'états sans décision) ; ce
                    # problème étant corrigé, on revient dans la plage usuelle
                    # des implémentations PPO (1e-3 à 1e-2).
                    t = min((epoch - 1) / 120.0, 1.0)
                    entropy_coef_epoch = cfg.entropy_coef * (1.5 * (1 - t) + 0.5 * t)
                    # La régularisation porte sur entropy_FLAT, pas sur la
                    # moyenne masquée : en position 2 actions sur 3 sont
                    # masquées, donc ces états ont une entropie nulle par
                    # construction et écrasent la moyenne (H=0.046 alors que
                    # Hflat=1.0986, soit le maximum exact de ln(3)).
                    #
                    # Avec `entropy`, le garde-fou anti-collapse ci-dessous se
                    # déclenchait en permanence sur cet artefact : coefficient
                    # effectif 0.30 × 1.99 × 5 = 2.99 contre un actor_loss de
                    # l'ordre de 1e-4. La policy restait clouée à l'uniforme,
                    # donc plafonnée à p = 1/3 = 0.333 — condamnée à repasser
                    # sous le seuil de conviction dès que la rampe atteint 0.40.
                    entropy_bonus = entropy_coef_epoch * entropy_flat

                    # Sur un maximum de ln(3) = 1.0986, passer sous 0.1 est un
                    # vrai effondrement de la policy.
                    if entropy_flat.item() < 0.1:
                        entropy_bonus = entropy_bonus * 5.0

                    # Warmup critique : N premières epochs → pas d'actor_loss.
                    # Le bonus d'entropie est CONSERVÉ : actor et critic partagent
                    # tout le tronc (input_proj → blocks → norm → mlp), donc
                    # optimiser critic_loss seul déforme aussi les logits de
                    # l'actor. Sans ce terme, la policy passait de H=1.0986
                    # (uniforme) à H≈0.05 (déterministe) dès l'epoch 1, et PPO
                    # ne pouvait plus la rouvrir avec clip_eps=0.18.
                    if epoch <= cfg.critic_warmup_epochs:
                        loss = cfg.value_coef * critic_loss - entropy_bonus
                    else:
                        loss = actor_loss + cfg.value_coef * critic_loss - entropy_bonus

                # GARDE-FOU 5 : check la loss finale
                if not torch.isfinite(loss):
                    print(f"  {_col('⚠ Loss non-finite, skip batch', _C.RED)}")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                # ──────────────────────────────────────────────────────────
                # FIX CRITIQUE : unscale AVANT clip_grad_norm sinon le clip
                # opère sur des gradients ×2^16 (AMP) → inefficace
                # ──────────────────────────────────────────────────────────
                scaler.unscale_(optimizer)

                # Décomposition du gradient par tête, une fois par epoch, avant
                # clipping. Répond à : l'actor reçoit-il seulement un gradient ?
                # (KL bloqué à 0.0000 alors que le critique, lui, apprend.)
                # Mesure sur CHAQUE pas et non sur le premier seulement : un
                # unique minibatch n'est pas representatif, et c'est justement
                # ce qu'il faut ecarter avant de conclure sur le gradient.
                _ga = _param_grad_norm(actor_head_params)
                _gc = _param_grad_norm(critic_head_params)
                _gt = _param_grad_norm(trunk_params)
                if all(math.isfinite(v) for v in (_ga, _gc, _gt)):
                    g_actor_hist.append(_ga)
                    g_critic_hist.append(_gc)
                    g_trunk_hist.append(_gt)
                    _lr = float(log_ratio.abs().mean().item())
                    _clipped = float(
                        (log_ratio.abs() > math.log1p(cfg.clip_eps)).float().mean().item()
                    )
                    logratio_hist.append(_lr)
                    clipfrac_hist.append(_clipped)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), cfg.max_grad_norm
                )

                # GARDE-FOU 6 : si grad norm est NaN/Inf, skip la step
                if not torch.isfinite(grad_norm):
                    print(f"  {_col('⚠ grad_norm non-fini, skip step', _C.RED)}")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue

                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    # KL mesuré sur les états FLAT également : en position le
                    # ratio vaut exactement 1 donc la KL y est nulle. Moyennée
                    # sur tout le batch, elle était divisée par ~40, ce qui
                    # rendait l'early-stop (target_kl=0.03) inatteignable :
                    # il aurait fallu une KL réelle de 1.2 pour le déclencher.
                    _kl_all = lb_old - new_log
                    approx_kl = (
                        _kl_all[_flat_b].mean().item() if bool(_flat_b.any())
                        else 0.0
                    )
                    epoch_flat_frac.append(float(_flat_b.float().mean().item()))

                epoch_actor_loss.append(actor_loss.item())
                epoch_critic_loss.append(critic_loss.item())
                epoch_entropy.append(entropy.item())
                epoch_entropy_flat.append(entropy_flat.item())
                epoch_kl.append(approx_kl)
                # Norme du gradient AVANT clipping (clip_grad_norm_ la retourne).
                # Le rapport gnorm / max_grad_norm donne le facteur exact par
                # lequel la mise à jour est réduite : si gnorm=15 pour un clip
                # à 0.3, tout le pas est divisé par 50, actor compris.
                epoch_grad_norm.append(float(grad_norm))

            if np.mean(epoch_kl) > 1.5 * cfg.target_kl:
                print(f"  {_col('⚠ early-stop KL', _C.YELLOW)}  KL={np.mean(epoch_kl):.4f}")
                break

        scheduler.step()
        import json as _json
        with open(f"sampling_audit_{cfg.side}{suffix}.jsonl", "a", encoding="utf-8") as _fa:
            _fa.write(_json.dumps({"epoch": epoch,
                                  "on_policy": not cfg.legacy_off_policy_curriculum,
                                  **sampling_audit}) + "\n")
        print(f"[PPO SAMPLING] epoch={epoch} {sampling_audit}")

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
        buy_count, sell_count, hold_count = action_counts_env
        buy_ratio = buy_count / total_actions_env
        sell_ratio = sell_count / total_actions_env
        hold_ratio = hold_count / total_actions_env

        # --------- validation (greedy) ---------
        policy.eval()
        pbs_val: List[List[float]] = [[], []]   # convictions BUY / SELL sur le val
        val_pnl = []
        val_dd = []
        val_trades = []
        val_trades_side: List[int] = []

        # Épisodes de validation VECTORISÉS : même principe que le rollout de
        # train. La val pesait 64% du coût (7 × 4000 pas contre 4 × 4000), et
        # elle ne termine jamais en avance puisqu'elle ne déclenche pas le
        # garde-fou DD. La batcher la rend ~7× moins chère, ce qui permet de
        # garder val_episodes=7 (nécessaire à la stabilité du Sortino30).
        n_val = len(val_envs)

        _chrono["maj PPO"] = time.time() - _t_phase; _t_phase = time.time()

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

        # PASSE 1 sur les environnements de CALIBRATION : la fenetre qui
        # PRECEDE celle de mesure. Les seuils en sortent, puis sont figes.
        v_states = []
        v_infos = []
        for e, d in zip(calib_envs, departs_calib):
            s0, i0 = reset_au_depart(e, d)
            v_states.append(s0)
            v_infos.append(i0)
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
                    logits_b, _ = policy(st)
                    probs_np = torch.softmax(
                        logits_b.masked_fill(~masks_b, MASK_VALUE), dim=-1
                    ).cpu().numpy()
                    for bi, k in enumerate(cal_active):
                        pbs_val[0].append(float(probs_np[bi, 0]))
                        pbs_val[1].append(float(probs_np[bi, 1]))
                pas_courant += 1
                suite = []
                for bi, k in enumerate(cal_active):
                    calib_envs[k].set_risk_scale(1.0)
                    ns, _r, done, _, info = calib_envs[k].step(2)   # HOLD forcé
                    v_states[k] = ns
                    v_infos[k] = info
                    if not done:
                        suite.append(k)
                cal_active = suite

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
        # Les poids du tronc ont bouge pendant l'epoch : les representations
        # de la banque mises en cache sont perimees. Sans ce rafraichissement,
        # la memoire repondrait avec l'encodage d'il y a N pas de gradient.
        policy.rafraichit_banque()

        decision_spec = rolling_decision_spec(
            val_selectivity / 2.0, cfg.rang_fenetre, pbs_val)
        val_decisions = [EntryDecisionPolicy(decision_spec) for _ in range(n_val)]

        # ---------- PASSE 2 : validation réelle, rang glissant ----------
        # Fenetre posterieure a celle de calibration, graine distincte mais
        # constante d'une epoch a l'autre.
        np.random.seed(cfg.val_seed + 1)
        v_states = []
        v_infos = []
        for e, d in zip(val_envs, departs_val):
            s0, i0 = reset_au_depart(e, d)
            v_states.append(s0)
            v_infos.append(i0)

        v_active = list(range(n_val))

        # Même principe qu'au rollout : la policy n'est sollicitée que sur les
        # envs flat. En position l'action est forcée, le forward serait jeté.
        with torch.no_grad():
            while v_active:
                deciding = [k for k in v_active if v_infos[k].get("position", 0) == 0]
                v_actions = {k: 2 for k in v_active}

                if deciding:
                    vb = np.stack([v_states[k] for k in deciding], axis=0)
                    st = torch.as_tensor(vb, dtype=torch.float32, device=device)
                    masks_b = torch.from_numpy(
                        np.repeat(MASK_FLAT[None, :], len(deciding), axis=0)
                    ).to(device)

                    logits_b, _ = policy(st)
                    logits_mb = logits_b.masked_fill(~masks_b, MASK_VALUE)
                    probs_np = torch.softmax(logits_mb, dim=-1).cpu().numpy()

                    for bi, k in enumerate(deciding):
                        # On retient le MEILLEUR CÔTÉ, puis on exige seulement
                        # que sa conviction franchisse la barre calibrée.
                        #
                        # L'ancienne règle passait par argmax sur les 3 actions,
                        # donc exigeait implicitement p(BUY) > p(HOLD). Mesuré :
                        # la policy converge vers p ≈ (0.31, 0.31, 0.38), HOLD
                        # est l'argmax partout, et la validation renvoyait 0
                        # trade même avec un seuil nul. Ce n'est pas ce qu'on
                        # veut mesurer : une stratégie qui ne trade que 5 % du
                        # temps a forcément p(HOLD) majoritaire en moyenne.
                        pb, ps = float(probs_np[bi, 0]), float(probs_np[bi, 1])
                        # Chaque côté est jugé sur SA barre. Si les deux passent,
                        # on retient le plus net par rapport à la sienne, pas le
                        # plus probable dans l'absolu.
                        # Les niveaux courants sont lus AVANT d'enregistrer :
                        # une occasion ne doit pas participer au quantile qui
                        # la juge.
                        a = val_decisions[k].decide(pb, ps)
                        v_actions[k] = a

                still = []
                for k in v_active:
                    val_envs[k].set_risk_scale(1.0)
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
            val_trades.extend(ve.trades_pnl)
            val_trades_side.extend(ve.trades_side)

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

        # Recalibration pour l'epoch SUIVANTE, sur la fenêtre de validation.
        # Même garde-fou : filtrer une distribution plate revient à tirer au sort.
        val_profit = float(sum(val_pnl))
        val_max_dd = float(max(val_dd) if val_dd else 0.0)
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
        print(
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

        # ----- Ligne 2 : VAL -----
        print(
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

        print(
            f"{tag} {epoch_str}  "
            f"{_col('META ', _C.GREY + _C.BOLD)}  "
            f"Sortino {metric:>+6.3f}  "
            f"{_col(f'Sortino30 {s30:>+6.3f}', s30_col)}  "
            f"AvgW {_money(avg_win_train, width=8)}  AvgL {_money(avg_loss_train, width=8)}  "
            f"ActorL {np.mean(epoch_actor_loss):>+7.4f}  "
            f"CriticL {np.mean(epoch_critic_loss):>7.4f}  "
            f"H {np.mean(epoch_entropy):>5.3f}  "
            f"Hflat {np.mean(epoch_entropy_flat):>5.3f}/1.099  "
            f"sel[train {100*selectivite:>4.1f}% val {100*val_selectivity:>4.1f}%] "
            f"thr[tr {conf_thr:.3f} "
            f"valB {val_decisions[0].thresholds[0]:.3f} valS {val_decisions[0].thresholds[1]:.3f}] "
            f"etendue[tr {pbs_etendue:.4f} val {val_etendue:.4f}]  "
            f"blocs[{_bl}] {_npos}/{_nval}  "
            f"KL {np.mean(epoch_kl):>+6.4f}  "
            f"dec {n_samples:>6d}"
            + (f"/{_dec_collectees}" if _dec_collectees > n_samples else "") + "  "
            f"|R| {np.mean(np.abs(batch_adv)) if batch_adv else 0.0:>6.3f}  "
            f"advStd {_adv_std_raw:>7.2f}  "
            f"clip {100*_frac_clip:>4.1f}%  quasi0 {100*_frac_nul:>4.1f}%  "
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
            f"{_col(f'H {hold_ratio:>4.1%}', _C.GREY)}]"
            + ("  MOYENNE DES POIDS" if epoch_moyenne else "")
        )

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
            print(
                f"  {_col('★', _C.YELLOW + _C.BOLD)} "
                f"{_col(f'NEW BEST PROFIT', _C.YELLOW + _C.BOLD)}  "
                f"ValPNL/trade={_money(best_val_profit, width=10)}  trades={val_num_trades}"
            )

        # Select the CURRENT policy score, not an average of earlier policies.
        if val_num_trades >= cfg.min_val_trades_save and metric > best_metric:
            best_metric = metric
            best_state = copy.deepcopy(policy.state_dict())
            best_thresholds = list(calib_thr_val)
            best_decision_spec = copy.deepcopy(decision_spec)
            save_checkpoint(best_state, best_path)
            _sauve_seuil(best_path)
            epochs_no_improve = 0
            print(
                f"  {_col('★', _C.MAGENTA + _C.BOLD)} "
                f"{_col(f'NEW BEST SORTINO', _C.MAGENTA + _C.BOLD)}  "
                f"Sortino={metric:+.3f}  trades={val_num_trades}"
            )
        else:
            epochs_no_improve += 1
            # patience <= 0 : aucun arret precoce. C'etait une selection sur la
            # validation comme une autre — elle choisissait la longueur du run
            # d'apres elle. Le budget est desormais fixe a l'avance.
            if patience > 0 and epochs_no_improve >= patience:
                print(f"[{cfg.side.upper()}{suffix}] Early stopping après {epoch} epochs (Sortino rolling ne progresse plus).")
                break

    last_path = f"last_{cfg.model_prefix}_{cfg.side}{suffix}.pth"
    save_checkpoint(policy.state_dict(), last_path)
    _sauve_seuil(last_path)

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

    with torch.no_grad():
        while t_active:
            deciding = [k for k in t_active if t_infos[k].get("position", 0) == 0]
            t_actions = {k: 2 for k in t_active}

            if deciding:
                tb = np.stack([t_states[k] for k in deciding], axis=0)
                st = torch.as_tensor(tb, dtype=torch.float32, device=device)
                masks_b = torch.from_numpy(
                    np.repeat(MASK_FLAT[None, :], len(deciding), axis=0)
                ).to(device)

                logits_b, _ = policy(st)
                logits_mb = logits_b.masked_fill(~masks_b, MASK_VALUE)
                probs_np = torch.softmax(logits_mb, dim=-1).cpu().numpy()
                for bi, k in enumerate(deciding):
                    # MÊME règle qu'en validation et qu'en production : meilleur
                    # côté, puis barre calibrée. Utiliser ici le seuil absolu
                    # historique ferait mesurer le TEST sur une autre politique
                    # que celle qu'on déploierait.
                    # MÊME règle qu'en validation : une barre par côté, toutes
                    # deux calibrées sur la VALIDATION et jamais sur le test —
                    # les recalibrer ici reviendrait à régler un paramètre sur
                    # les données censées mesurer la généralisation.
                    pb, ps = float(probs_np[bi, 0]), float(probs_np[bi, 1])
                    a = test_decisions[k].decide(pb, ps)
                    t_actions[k] = a

            still = []
            for k in t_active:
                test_envs[k].set_risk_scale(1.0)
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
    auto_chain: bool = False,
):
    """Walk-forward: scaler calculé sur le train de chaque fold, sans transfert implicite.

    start_fold permet de sélectionner la première fenêtre. Les paramètres de
    bootstrap historiques restent acceptés pour produire une erreur explicite.
    """
    if auto_chain or bootstrap_from_path is not None:
        raise ValueError("Le bootstrap inter-fold exige une adaptation explicite de normalisation; auto_chain doit être False.")
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

    print(f"\n=== WALK-FORWARD {cfg_base.side.upper()} ===")
    print(f"Total bars={n}, window_len={window_len}, "
          f"train={train_len}, val={val_len}, test={test_len}, step={step}")

    # Skip les folds avant start_fold
    fold = start_fold - 1
    start = (start_fold - 1) * step
    if start_fold > 1:
        print(f"[SKIP] Folds 1 → {start_fold - 1} sautés (start_fold={start_fold})")

    while start + window_len <= n and fold < max_folds:
        fold += 1
        print(f"\n--- Fold {fold} : indices [{start} : {start + window_len}) ---")

        stats = compute_and_save_global_norm_stats(
            df_full.iloc[start:start + train_len], FEATURE_COLS, path=None)
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
                              cfg_fold, suffix=suffix)

        start += step

    print(f"\n=== FIN WALK-FORWARD {cfg_base.side.upper()} (folds entraînés : {start_fold} → {fold}) ===")


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    cfg_base = PPOConfig()

    # =======================================================
    # PIPELINE DUEL : un seul modèle qui décide BUY ou SELL
    # =======================================================
    # side="both": BUY, SELL ou HOLD à plat, sans actions forcées.
    # Résultat : un unique fichier .pth qui remplace LONG + SHORT,
    # exactement comme en live "duel".
    print("\n" + "=" * 70)
    print("  ENTRAÎNEMENT DUEL (long + short combinés)")
    print("=" * 70)
    cfg_duel = PPOConfig(**cfg_base.__dict__)
    cfg_duel.side = "both"
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
    cfg_duel.model_prefix = "saintv2_loup_duel_exec25"

    # Chaque fold repart de zéro avec les statistiques de son train.
    print("Walk-forward exec25: trois folds sans bootstrap inter-fold.")
    run_walkforward(cfg_duel, train_frac=0.55, val_frac=0.15, test_frac=0.10,
                    max_folds=3, start_fold=1,
                    bootstrap_from_path=None,
                    auto_chain=False)

    print("\n" + "=" * 70)
    print("  WALK-FORWARD TERMINÉ : 3 folds entraînés, indépendance préservée.")
    print("  Fichiers générés :")
    print("    bestprofit_saintv2_loup_duel_exec2_wf1_both_wf1.pth")
    print("    bestprofit_saintv2_loup_duel_exec2_wf2_both_wf2.pth")
    print("    bestprofit_saintv2_loup_duel_exec2_wf3_both_wf3.pth")
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
