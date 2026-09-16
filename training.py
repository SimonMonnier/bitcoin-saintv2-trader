# ======================================================================
# PPO + SAINTv2 — SCALPING BTCUSD M1 (SINGLE-HEAD + ACTION MASK + H1)
# Version "Loup Ω" LONG / SHORT / CLOSE
# ======================================================================

import os
import re
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
    PolitiqueEnsemble,
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
    symbol: str = "XAUUSD"
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
    architecture: str = "ensemble"
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
    tri_par_tete_aux: bool = True

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
    validation_tous_les: int = 3

    diag_rang: bool = True
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
    epochs: int = 90
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
    # BUDGET DE RISQUE PARTAGE, en part de l'equite du COMPTE.
    #
    # Porte a 1.00 sur demande explicite, pour que LES DEUX instruments
    # tiennent un essaim. Ce qui suit dit ce que cette valeur coute, mesure et
    # non suppose.
    #
    # POURQUOI IL FAUT MONTER SI HAUT POUR L'OR. Son contrat vaut cent onces :
    # a 1 000 EUR son lot MINIMUM risque 2.81 % du compte contre 0.73 % pour
    # le BTC. Un essaim d'or exige donc mecaniquement de risquer la moitie du
    # capital — ce n'est pas un reglage, c'est la taille du contrat.
    #
    # MESURE SUR EQUITE CONTINUE, deux instruments alignes, entrees neutres :
    #
    #   budget   capital   pos BTC   pos OR     gain    creux   barres     fin
    #       3%    1,000$    9/25     0/1         -5%       9%    4,000      ok
    #      10%    1,000$   36/51     0/3         -0%      17%    4,000      ok
    #      30%    1,000$   57/107    7/16       +51%      37%    4,000      ok
    #     100%    1,000$   22/33    36/44       -23%      43%      538  XAUUSD
    #       3%   25,000$   42/93    14/22        -0%       6%    4,000      ok
    #
    # A 100 % L'EPISODE MEURT A LA BARRE 538 sur 4 000. L'entrainement ne peut
    # rien apprendre a cette valeur : les episodes s'arretent avant d'avoir
    # produit des decisions, et un balayage anterieur avait deja montre que le
    # compte meurt sur les deux graines des 10 % en mono-instrument.
    #
    # DEUX VALEURS DONNENT L'ESSAIM SANS TUER LE RUN : 30 % a 1 000 EUR, au
    # bord du garde-fou de drawdown ; ou 3 % a 25 000 EUR, qui produit le meme
    # essaim pour six fois moins de creux. C'est un probleme de CAPITAL, pas
    # de budget : a 25 000 EUR le lot minimum de l'or ne pese plus que 0.11 %.
    # RETENU : 30 %. C'est la valeur la plus haute ou les deux instruments
    # essaiment ET ou l'episode survit — a 100 % il meurt a la barre 538 sur
    # 4 000. Le creux de 37 % reste au bord du garde-fou a 40 %, ce qui est
    # assume : c'est le prix d'un essaim d'or a 1 000 EUR de capital.
    budget_risque: float = 0.30
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
    episodes_per_epoch: int = 40

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
    min_capital_frac: float = 0.20  # garde-fou capital : épisode terminé si capital < 20%
                                     # config validée +135 EUR MT5 tester
    max_drawdown: float = 0.40  # garde-fou DD : épisode terminé si drawdown > 40%
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
    # 6 -> 10xATR : la largeur de l'OR, derivee de son propre balayage.
    # Son ATR relatif vaut 6.6 points de base contre 15.9 pour le BTC, donc
    # 10xATR y fait 66 bps quand 6xATR en faisait 95 sur le Bitcoin. Le
    # balayage donne +20.1 R/an a 6x mais 3.1 % des reouvertures franchissent
    # le stop ; a 10x il n'y en a plus que 1.6 % pour +16.5 R/an, et la
    # friction tombe de 0.090 a 0.054 R.
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
    use_be_trail: bool    = True
    # PAS DE BREAK-EVEN. Mesure : avec un trailing large il fait passer la part
    # symetrique de +0.0638 a +0.0167, et surtout il la rend instable — +0.0280
    # sur la premiere moitie de l'historique, -0.0262 sur la seconde. Un seuil
    # hors d'atteinte le neutralise sans toucher au code.
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
        # Reference au dataframe source, pas une copie : le votant TabM a
        # besoin des colonnes brutes (high, low, atr) pour recalculer ses
        # cibles de barrieres sur exactement la meme fenetre que le rollout.
        self.df = df
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

    def equity(self) -> float:
        """Capital plus le latent de TOUS les instruments."""
        lat = 0.0
        for e in self.envs:
            bid = e.data.close[min(max(e.idx - 1, 0), e.data.length - 1)]
            lat += e._latent_at_bid(bid)
        return self.capital + lat

    def marge_utilisee(self) -> float:
        return sum(e.marge_utilisee for e in self.envs)

    def risque_engage(self) -> float:
        return sum(float(e._p_risque[e._p_sens != 0].sum()) for e in self.envs)

    def n_positions(self) -> int:
        return sum(e.n_positions for e in self.envs)


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
        self._cap_cle, self._cap_val = None, 0

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
    @property
    def _actifs(self):
        return self._p_sens != 0

    @property
    def n_positions(self) -> int:
        return int(np.count_nonzero(self._p_sens))

    @property
    def notionnel(self) -> float:
        """Somme des notionnels ouverts, tenue a jour a l'ouverture et a la
        fermeture plutot que recalculee : elle est lue plusieurs fois par
        barre et ne change que deux fois par trade."""
        return float(self._notionnel)

    @property
    def marge_utilisee(self) -> float:
        return self.notionnel * float(getattr(self.cfg, "marge_frac", 0.001734))

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
        if not getattr(self.cfg, "marge_realiste", False):
            return taille
        # LE LOT N'EST PAS L'UNITE. `taille` est en unites de l'instrument —
        # bitcoins ou onces — alors que le courtier quantifie en LOTS, et un
        # lot d'or vaut CENT onces. Quantifier a 0.01 sans passer par la
        # taille du contrat ferait trader un centieme d'once la ou le minimum
        # reel est une once : cent fois trop petit, et le risque annonce
        # n'aurait aucun rapport avec le risque joue.
        contrat = float(getattr(self.cfg, "contrat", 1.0))
        pas = float(getattr(self.cfg, "lot_pas", 0.01)) * contrat
        mini = float(getattr(self.cfg, "lot_min", 0.01)) * contrat
        if taille < mini:
            # Le courtier ne sait pas faire plus petit. On prend le minimum,
            # comme le fait `kairos_live`, et le risque reel depasse la cible.
            return mini
        return math.floor(taille / pas + 1e-9) * pas

    def places_ouvrables(self, prix: float) -> int:
        # Memorisee pour la barre : la boucle de collecte la demande via
        # `peut_entrer()` puis via `_get_obs()`, et rien entre les deux ne
        # peut la changer. La cle porte l'etat dont elle depend, donc une
        # ouverture ou une fermeture l'invalide d'elle-meme.
        # La cle porte l'etat du COMPTE : une ouverture sur l'autre
        # instrument change la capacite de celui-ci, et la memo doit s'en
        # apercevoir. Sans cela, l'or continuerait d'ouvrir sur une capacite
        # calculee avant que le Bitcoin ne consomme le budget.
        cle = (self.idx, self.portefeuille.n_positions(),
               round(self.portefeuille.marge_utilisee(), 9),
               round(self.portefeuille.risque_engage(), 9), self.capital)
        if getattr(self, "_cap_cle", None) == cle:
            return self._cap_val
        val = self._places_ouvrables(prix)
        self._cap_cle, self._cap_val = cle, val
        return val

    def _places_ouvrables(self, prix: float) -> int:
        """Combien de positions le SOLDE permet encore, ici et maintenant.

        C'est le K reel. Il ne se configure pas, il se calcule : il tombe
        quand l'equity baisse, quand le prix monte, ou quand des positions
        sont deja ouvertes. Le modele doit l'apprendre, donc il le voit
        (quatrieme colonne du bloc d'etat de l'observation).
        """
        # Le nombre d'emplacements libres n'est PAS une limite : les
        # tableaux doublent a la demande. Ce qui limite, c'est le solde, et
        # cette fonction doit donc dire ce que le SOLDE permet — pas ce que
        # l'allocation courante contient. La borner par la taille du tableau
        # la ferait mentir des que l'equity depasse ce qui a ete alloue, et
        # c'est exactement ce que le modele doit voir changer quand il gagne.
        if not getattr(self.cfg, "marge_realiste", False):
            return max(self._K - self.n_positions, 0)
        eq = self._equity_courante()
        if eq <= 0:
            return 0
        frac = float(getattr(self.cfg, "marge_frac", 0.001734))
        seuil = float(getattr(self.cfg, "niveau_marge_ouverture", 3.0))
        # Marge qu'une position minimale consommerait.
        m_une = max(prix * float(getattr(self.cfg, "lot_min", 0.01))
                    * float(getattr(self.cfg, "contrat", 1.0)) * frac, 1e-12)
        # On s'arrete avant que le niveau de marge ne descende sous le seuil.
        marge_max = eq / max(seuil, 1e-9)
        par_marge = math.floor(
            (marge_max - self.portefeuille.marge_utilisee()) / m_une)

        # LE RISQUE ENGAGE, et c'est lui qui porte le cercle vertueux. La
        # somme des montants a perdre si tous les stops etaient touches reste
        # sous une part de l'EQUITY COURANTE : gagner l'augmente donc la
        # capacite, perdre la reduit. La marge, elle, est trop large pour
        # border quoi que ce soit a ce levier.
        budget = float(getattr(self.cfg, "budget_risque", 0.0))
        par_risque = float("inf")
        if budget > 0.0:
            atr_raw = (float(self.data.atr14[self.idx - 1])
                       if self.idx - 1 >= 0 else 0.0)
            atr = max(atr_raw, ATR_PLANCHER_FRAC * prix, 1e-8)
            # Le risque d'une position de plus, a la taille que le courtier
            # imposerait reellement — lot minimum compris.
            taille = self._taille_quantifiee(self._compute_dynamic_size(prix))
            r_une = max(self.cfg.atr_sl_mult * atr * taille, 1e-12)
            # LE RISQUE ENGAGE EST CELUI DU COMPTE. C'est tout l'interet
            # du partage : quinze positions ouvertes sur le Bitcoin
            # consomment le budget, donc l'or en voit moins de disponibles.
            # Compter instrument par instrument autoriserait deux fois le
            # risque voulu sans que rien ne le signale.
            engage = self.portefeuille.risque_engage()
            par_risque = math.floor((budget * eq - engage) / r_une)

        return int(max(0, min(par_marge, par_risque)))

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
        self._p_idx = _e(self._p_idx, -1)
        self._p_risque = _e(self._p_risque)
        self._p_spread = _e(self._p_spread, float(self.cfg.spread_bps))
        self._p_be = _e(self._p_be, False)
        self._p_trail = _e(self._p_trail, False)
        self._r_slots = _e(self._r_slots)
        self._realise_slots = _e(self._realise_slots)
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
        if not getattr(self.cfg, "marge_realiste", False):
            return True
        eq = self._equity_courante()
        if eq <= 0:
            return False
        frac = float(getattr(self.cfg, "marge_frac", 0.001734))
        seuil = float(getattr(self.cfg, "niveau_marge_ouverture", 3.0))
        m_une = max(prix * float(getattr(self.cfg, "lot_min", 0.01))
                    * float(getattr(self.cfg, "contrat", 1.0)) * frac, 1e-12)
        # La marge deja immobilisee est celle du COMPTE : les positions de
        # l'autre instrument la consomment aussi.
        return (eq / max(seuil, 1e-9)
                - self.portefeuille.marge_utilisee()) >= m_une

    def peut_entrer(self) -> bool:
        """Une DECISION existe quand le solde permet encore une position."""
        prix = float(self.data.close[min(self.idx, self.data.length - 1)])
        return self.places_ouvrables(prix) > 0

    @property
    def position(self) -> int:
        """Sens NET. A K=1, le sens de l'unique position."""
        a = self._actifs
        if not a.any():
            return 0
        net = float((self._p_sens[a] * self._p_taille[a]).sum())
        return 0 if net == 0.0 else (1 if net > 0 else -1)

    @property
    def current_size(self) -> float:
        a = self._actifs
        return float(np.abs(self._p_taille[a]).sum()) if a.any() else 0.0

    @property
    def entry_price(self) -> float:
        """Prix d'entree moyen PONDERE PAR LA TAILLE. A K=1, le prix exact."""
        a = self._actifs
        t = self._p_taille[a]
        return float((self._p_entree[a] * t).sum() / t.sum()) if t.sum() > 0 else 0.0

    @property
    def entry_atr(self) -> float:
        a = self._actifs
        t = self._p_taille[a]
        return float((self._p_atr[a] * t).sum() / t.sum()) if t.sum() > 0 else 0.0

    @property
    def entry_idx(self) -> int:
        a = self._actifs
        return int(self._p_idx[a].min()) if a.any() else -1

    @property
    def bars_in_position(self) -> int:
        """Detention de la position la PLUS ANCIENNE encore ouverte."""
        a = self._actifs
        return int(self.idx - self._p_idx[a].min()) if a.any() else 0

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
        a = self._p_sens != 0
        if not a.any():
            return 0.0
        sens = self._p_sens[a]
        # execution_quote(bid, -sens, spread) : le cote vendu paie le spread.
        q = float(bid) * np.where(sens == -1,
                                  1.0 + np.maximum(self._p_spread[a], 0.0) / 1e4,
                                  1.0)
        return float(((sens * (q - self._p_entree[a])
                       - self.cfg.fee_rate * q) * self._p_taille[a]).sum())

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
        self._p_risque[slot] = 0.0
        self._p_be[slot] = False
        self._p_trail[slot] = False
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
        hit_sl = hit_tp = hit_temps = False

        # Plus de mode "close" : fermeture uniquement par SL/TP/break-even/trailing
        manual_close = False

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
        _libres = np.flatnonzero(self._p_sens == 0)
        if not manual_close and action in (0, 1) and len(_libres):
            side = 1 if action == 0 else -1
            size = self._taille_quantifiee(self._compute_dynamic_size(prix_execution))
            # LE SOLDE PEUT REFUSER L'ORDRE. C'est un refus, pas une erreur :
            # en live l'ordre part et le serveur le rejette faute de marge.
            if size > 0.0 and self.places_ouvrables(prix_execution) <= 0:
                size = 0.0
            if size > 0.0:
                j = int(_libres[0])
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
                if side == 1:
                    self._p_sl[j] = max(1e-8, exec_price - sl_dist)
                    self._p_tp[j] = (max(1e-8, exec_price + tp_dist)
                                     if tp_dist > 0 else 0.0)
                else:
                    self._p_sl[j] = max(1e-8, exec_price + sl_dist)
                    self._p_tp[j] = (max(1e-8, exec_price - tp_dist)
                                     if tp_dist > 0 else 0.0)

                self._p_be[j] = False
                self._p_trail[j] = False
                self._notionnel += exec_price * size
                self._slot_ouvert = j

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
            touche_temps = (vivant & (~touche_sl) & (~touche_tp)
                            & (self.cfg.max_holding_bars > 0)
                            & ((self.idx - self._p_idx)
                               >= self.cfg.max_holding_bars))

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
        elif marked_dd > self.cfg.max_drawdown:
            done_reason = "max_drawdown"
        elif marked_equity < self.cfg.initial_capital * self.cfg.min_capital_frac:
            done_reason = "min_capital"
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

        # Pénalité DD réactivée (max_drawdown=0.40) — punit les trajectoires catastrophiques
        if dd > self.cfg.max_drawdown:
            reward -= 0.2

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
        _concernes = set(int(j) for j in np.flatnonzero(self._p_sens != 0))
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
            _pen = 0.2 if dd > self.cfg.max_drawdown else 0.0
            _clip = (hasattr(self.cfg, "current_epoch")
                     and self.cfg.current_epoch >= 10)
            _lat = self._latents_par_slot(price)
            for j in range(self._K):
                if j not in _concernes:
                    self._r_slots[j] = 0.0
                    continue
                d_j = (_lat[j] - self._latent_prec[j]
                       + self._realise_slots[j])
                lr = math.log(max(prev_equity_clamped + d_j, 1e-8)
                              / prev_equity_clamped)
                self._r_slots[j] += 10.0 * max(min(lr, 0.05), -0.05) - _pen
                if _clip:
                    self._r_slots[j] = float(
                        np.clip(self._r_slots[j], -3.5, 3.5))
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
                                     int(round(cfg.cible_decisions / _dec_par_ep))))]

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
    # LE COUT EST NUL PAR EPOCH. Le rendement reel ne depend pas du modele :
    # il se calcule une fois ici, et chaque epoch n'ajoute qu'une passe avant
    # sur 11 000 observations. Ce qui suit est du diagnostic : rien ne s'en
    # sert pour selectionner, decider ou arreter.
    _rang_idx = _rang_reel = None
    if getattr(cfg, "diag_rang", True):
        import cibles as _CIB
        _pas = max(int(getattr(cfg, "diag_rang_pas", 12)), 1)
        _i = np.arange(cfg.lookback,
                       val_data.length - _CIB.BORNE_DEFAUT - 2, _pas)
        if len(_i) >= 500:
            _ra, _rv = _CIB.rendements(val_data.df, _i, cfg)
            _ok = np.isfinite(_ra) & np.isfinite(_rv)
            if _ok.sum() >= 500:
                _rang_idx = _i[_ok]
                # Part SYMETRIQUE : (achat - vente) / 2. La derive du
                # sous-jacent s'annule, donc un rho positif dit que le modele
                # distingue les moments, pas qu'il a profite d'une hausse.
                _rang_reel = ((_ra - _rv) / 2.0)[_ok]
                print(f"  • Classement : {len(_rang_idx):,} decisions de "
                      f"validation suivies (une toutes les {_pas*5} min)")

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

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-8)

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

    actor_head_params = [p for n, p in policy.named_parameters()
                         if _sans_membre(n).startswith("actor.")]
    critic_head_params = [p for n, p in policy.named_parameters()
                          if _sans_membre(n).startswith("critic.")]
    trunk_params = [
        p for n, p in policy.named_parameters()
        if not _sans_membre(n).startswith(("actor.", "critic."))
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

    # Derniere validation COMPLETE, reprise telle quelle les epochs ou elle
    # est sautee. Voir `validation_tous_les`.
    _val_prec = None
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
             "logprobs": [],
             "rewards": [], "values": [], "dones": [], "positions": [],
             "dts": []}
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
        pending: List[Dict[int, Dict]] = [dict() for _ in range(n_envs)]
        # Decision prise a cette barre, pas encore rattachee : on ne connait
        # son emplacement qu'APRES le pas, puisque c'est l'environnement qui
        # l'attribue.
        en_attente: Dict[int, Dict] = {}
        sampling_audit = {"decisions": 0, "forced_actions": 0,
                          "remapped_actions": 0, "max_logprob_error": 0.0,
                          "episodes_joues": len(envs)}

        def _verse(k: int, p: Dict, done_flag: bool) -> None:
            """Verse une décision terminée au buffer de l'env k."""
            buf = ep_buf[k]
            buf["positions"].append(0)          # une décision est toujours flat
            buf["states"].append(p["state"])
            buf["masques"].append(p["masque"])
            buf["barres"].append(p["barre"])
            buf["actions"].append(p["action"])
            buf["logprobs"].append(p["logprob"])
            buf["rewards"].append(p["R"])
            buf["values"].append(p["value"])
            buf["dones"].append(done_flag)
            buf["dts"].append(max(p["dt"], 1))

        def _cloture(k: int, slot: int, done_flag: bool) -> None:
            p = pending[k].pop(slot, None)
            if p is not None:
                _verse(k, p, done_flag)

        while active:
            # 1) Un environnement decide des qu'il lui reste un emplacement
            #    libre — c'est la definition d'une decision. A K=1 cela revient
            #    exactement a "etre plat".
            deciding = [k for k in active if envs[k].peut_entrer()]
            actions_env = {k: 2 for k in active}

            if deciding:
                batch_np = np.stack([states[k] for k in deciding], axis=0)
                s_tensor = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
                # Tous ces envs sont flat : un seul masque suffit.
                # LE TROISIEME VOTANT ENTRE ICI. TabM n'a pas de gradient et
                # ses scores dependent de la BARRE, pas seulement de
                # l'observation : il ne peut donc pas etre un membre du
                # melange differentiable. Il oppose un veto — il ne propose
                # rien, il interdit — ce qui est exactement la regle que decrit
                # `evalue_ensemble` : un signal n'est pas pris si autre chose
                # le contredit.
                masks_np = np.repeat(MASK_FLAT[None, :], len(deciding), axis=0)
                if votant is not None:
                    for bi, k in enumerate(deciding):
                        pa, pv = votant.veto(envs[k].idx - 1)
                        if not pa:
                            masks_np[bi, 0] = False
                        if not pv:
                            masks_np[bi, 1] = False
                masks_b = torch.from_numpy(masks_np).to(device)

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

                    en_attente[k] = {
                        "state": states[k],
                        "masque": masks_np[bi].copy(),
                        # L'indice de barre : la tete auxiliaire a besoin de
                        # l'etiquette de CETTE occasion, et elle se calcule sur
                        # le dataframe, pas sur l'observation normalisee.
                        "barre": envs[k].idx - 1,
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
                    if nouveau is not None and j_ouvert < 0:
                        _verse(k, nouveau, True)
                else:
                    # Les emplacements fermes a cette barre terminent leur
                    # decision. La version d'avant le faisait en tete de la
                    # boucle suivante ; le faire ici ne change ni le contenu
                    # ni le drapeau, et evite de relire `infos`.
                    for slot in info.get("slots_fermes", []):
                        _cloture(k, int(slot), False)
                    if nouveau is not None and j_ouvert < 0:
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

            buf = ep_buf[k]
            adv, ret = compute_gae_semi_mdp(
                buf["rewards"], buf["values"], buf["dones"], buf["dts"],
                cfg.gamma, cfg.lambda_gae, last_value
            )

            batch_states.extend(buf["states"])
            batch_masques.extend(buf["masques"])
            batch_barres.extend(buf["barres"])
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
        batch_masques = [batch_masques[i] for i in _sel]
        batch_barres = [batch_barres[i] for i in _sel]
        batch_actions = [batch_actions[i] for i in _sel]
        batch_oldlog = [batch_oldlog[i] for i in _sel]
        batch_adv = [batch_adv[i] for i in _sel]
        batch_returns = [batch_returns[i] for i in _sel]
        batch_values = [batch_values[i] for i in _sel]
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

        # ETIQUETTES DE LA TETE AUXILIAIRE : le rendement net d'un achat et
        # d'une vente a chaque barre decidee. Calculees ICI et pas une fois
        # pour toute la fenetre — ~2 000 indices par epoch coutent une seconde,
        # les 521 692 de la fenetre en couteraient des centaines.
        #
        # Les occasions dont la course ne se resout pas dans le plafond
        # rendent NaN ; elles sont masquees plutot que remplies par zero, un
        # zero etant une prediction et non une absence.
        cibles_aux = None
        if cfg.aux_coef > 0.0 and batch_barres:
            try:
                # LA CIBLE SE LIT DANS LA CONFIGURATION, elle ne se recopie pas.
                #
                # Elle venait de `evalue_tabm_test.cibles_brutes`, qui calcule un
                # trade a take-profit FIXE de 2 R avec plafond de detention. Le
                # jour ou l'environnement est passe au stop suiveur SANS
                # objectif, l'etiquette est restee la meme : la tete apprenait a
                # predire le resultat d'un trade que personne ne fait. Rien ne
                # l'a signale — les deux fonctions tournaient et rendaient des
                # nombres.
                #
                # `cibles.rendements` lit la regle de sortie dans PPOConfig :
                # changer l'environnement change l'etiquette, sans rien a
                # synchroniser. Les deux distributions n'ont d'ailleurs rien a
                # voir — l'ancienne est bimodale a -1/+2, la nouvelle a une
                # mediane a -1.01 et une queue jusqu'a +13 R, pour une variance
                # de 3.21 au lieu de 1.94.
                # `_CIB` et non `_C` : ce dernier est la classe de couleurs du
                # module, et l'importer sous ce nom dans cette fonction le rend
                # LOCAL — toutes les references a `_C.MAGENTA` plus bas
                # tombaient alors sur le module `cibles`.
                import cibles as _CIB
                idx = np.asarray(batch_barres, dtype=np.int64)
                idx = np.clip(idx, 0, train_data.length - 1)
                # `indicateur=True` : la tete apprend a predire l'atteinte de
                # 2 R, pas le rendement de la sortie au trailing. Les deux se
                # suivent au rang 0.892, et seule la premiere est classable.
                ra, rv = _CIB.rendements(train_data.df, idx, cfg,
                                         indicateur=True)
                y = np.stack([ra, rv], axis=1).astype(np.float32)
                ok = np.isfinite(y).all(axis=1)
                cibles_aux = (
                    torch.tensor(np.nan_to_num(y), device=device),
                    torch.tensor(ok, device=device))
            except Exception as e:
                print(f"  ! tete auxiliaire sans etiquettes ({type(e).__name__}"
                      f": {e}) — cette epoch n'entraine que PPO")
                cibles_aux = None

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
        epoch_aux_loss = []
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
                    # LE MASQUE QUI A AGI, PAS UN MASQUE RECONSTRUIT.
                    #
                    # Il etait refabrique ici depuis la seule position. Tant
                    # que le masque n'en dependait que, les deux coincidaient ;
                    # le veto de TabM depend de la BARRE, donc ils divergent.
                    # Recalculer reviendrait a comparer la probabilite nouvelle
                    # d'une action a son ancienne probabilite SOUS UNE AUTRE
                    # DISTRIBUTION — le rapport de PPO ne mesurerait plus rien,
                    # et aucune erreur ne serait levee.
                    mask_batch = (masques[ids] if masques is not None
                                  else build_action_mask_from_positions(
                                      pos_b, cfg.side))
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

                    # TETE AUXILIAIRE : regression du rendement net realise.
                    #
                    # ELLE S'ENTRAINE DES LE WARMUP, contrairement a l'actor.
                    # Le warmup existe pour que le critic se stabilise avant
                    # que la politique ne bouge ; la tete auxiliaire, elle, ne
                    # depend d'aucune politique — sa cible est ce que le marche
                    # a FAIT, pas ce que l'agent aurait gagne. Rien ne justifie
                    # de la retarder, et cinq epochs de signal dense gratuit
                    # valent d'etre prises.
                    #
                    # Les occasions non resolues sont MASQUEES et non mises a
                    # zero : un zero serait une prediction, pas une absence.
                    aux_loss = torch.zeros((), device=device)
                    if cibles_aux is not None:
                        y_b, ok_b = cibles_aux[0][ids], cibles_aux[1][ids]
                        if ok_b.any():
                            pred = policy.rendement(sb)
                            aux_loss = ((pred - y_b).pow(2).mean(dim=1)
                                        * ok_b).sum() / ok_b.sum()
                            loss = loss + cfg.aux_coef * aux_loss

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
                epoch_aux_loss.append(aux_loss.item())
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

        # AJUSTEMENT DU BUDGET. La geometrie decide du nombre de decisions
        # qu'un episode produit ; on en deduit combien d'episodes il faut
        # pour atteindre la cible. Borne entre 1 et le plafond, et lissee de
        # moitie pour qu'une epoch atypique ne fasse pas osciller le budget.
        _joues = max(sampling_audit.get("episodes_joues", 1), 1)
        _par_ep = max(sampling_audit["decisions"] / _joues, 1e-9)
        _vise = int(round(cfg.cible_decisions / _par_ep))
        _vise = max(1, min(_vise, cfg.episodes_per_epoch))
        n_episodes_courant[0] = max(1, (n_episodes_courant[0] + _vise) // 2)


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
        # LA CALIBRATION NE SERT QU'A LA VALIDATION — elle produit les barres
        # que la passe suivante applique. La faire tourner une epoch ou la
        # validation est sautee coutait 38 secondes pour un resultat que
        # personne ne lit.
        cal_active = list(range(len(calib_envs))) if _valide else []
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
                    # LA BARRE SE CALIBRE SUR LE SCORE QUI LA FRANCHIRA.
                    # Calibrer un quantile sur les probabilites de la politique
                    # puis juger la tete auxiliaire avec ne voudrait rien dire :
                    # deux distributions differentes, donc un quantile qui ne
                    # selectionne plus la fraction visee. Meme calcul des deux
                    # cotes, toujours.
                    if getattr(cfg, "tri_par_tete_aux", False):
                        try:
                            _a = policy.rendement(st).float().cpu().numpy()
                            probs_np = 1.0 / (1.0 + np.exp(-np.clip(_a, -30, 30)))
                        except Exception:
                            pass
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

        # LA REPRISE SE FAIT ICI, avant tout calcul qui lit `pbs_val`.
        #
        # Elle etait plus bas, apres la passe de validation — mais les seuils,
        # l'etendue et la specification de decision se calculent AVANT, et
        # `np.quantile` sur une liste vide leve une exception. Le run s'est
        # arrete a l'epoch 2 pour cette raison exacte : un patch qui saute une
        # passe doit fournir ce que cette passe produisait, au moment ou c'est
        # lu, pas plus tard.
        if not _valide and _val_prec is not None:
            (val_pnl, val_dd, val_trades, val_trades_side,
             pbs_val, _val_epoch) = _val_prec
            val_pnl, val_dd = list(val_pnl), list(val_dd)
            val_trades, val_trades_side = list(val_trades), list(val_trades_side)
            pbs_val = [list(pbs_val[0]), list(pbs_val[1])]

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
        #
        # Sautee une epoch sur `validation_tous_les`. Les compteurs gardent
        # alors les valeurs de la derniere mesure : le journal affiche donc la
        # validation la plus recente, jamais des zeros qu'on lirait comme un
        # effondrement.
        np.random.seed(cfg.val_seed + 1)
        v_states = []
        v_infos = []
        for e, d in zip(val_envs, departs_val):
            s0, i0 = reset_au_depart(e, d)
            v_states.append(s0)
            v_infos.append(i0)

        v_active = list(range(n_val)) if _valide else []

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

                    # LE SCORE DE TRI VIENT DE LA TETE AUXILIAIRE. Les deux
                    # sorties predisent le rendement net d'un achat et d'une
                    # vente ; on les passe par une sigmoide pour qu'elles
                    # vivent dans [0, 1] comme les probabilites, puisque la
                    # barre calibree et `EntryDecisionPolicy` raisonnent sur
                    # des rangs. La monotonie suffit : un rang ne depend pas
                    # de l'echelle.
                    if getattr(cfg, "tri_par_tete_aux", False):
                        try:
                            _aux = policy.rendement(st).float().cpu().numpy()
                            probs_np = 1.0 / (1.0 + np.exp(-np.clip(_aux, -30, 30)))
                        except Exception:
                            pass

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
                        # MEME SOURCE QUE LA CALIBRATION, forcement : la
                        # barre et la valeur jugee doivent sortir du meme
                        # calcul, sinon le quantile ne veut rien dire.
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
                                                 list(pbs_val[1])], epoch)

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

        # Le rho de l'epoch. Etat PLAT, celui que voit la politique avant
        # d'entrer : position 0, latent 0, detention 0, echelle de risque 1.
        _rho_ep = float(chr(110) + chr(97) + chr(110))
        _rho_aux = float(chr(110) + chr(97) + chr(110))
        _sa = []
        if _rang_idx is not None:
            policy.eval()
            _extra = np.zeros((cfg.lookback, 4), np.float32)
            _extra[:, 3] = 1.0
            _s = []
            with torch.no_grad():
                for _d in range(0, len(_rang_idx), 8192):
                    _b = _rang_idx[_d:_d + 8192]
                    _o = np.stack([
                        np.concatenate([val_data.features[i - cfg.lookback:i],
                                        _extra], axis=-1) for i in _b])
                    _t = torch.from_numpy(_o).to(device)
                    _lg = policy(_t)
                    if isinstance(_lg, tuple):
                        _lg = _lg[0]
                    _pr = torch.softmax(_lg, dim=-1).float().cpu().numpy()
                    _s.append(_pr[:, 0] - _pr[:, 1])
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
                        _a = policy.rendement(_t).float().cpu().numpy()
                        _sa.append(_a[:, 0] - _a[:, 1])
                    except Exception:
                        _sa = None
            policy.train()
            _rho_ep = _correlation_rang(np.concatenate(_s), _rang_reel)
            if _sa:
                _rho_aux = _correlation_rang(np.concatenate(_sa), _rang_reel)

        print(
            f"{tag} {epoch_str}  "
            f"{_col('META ', _C.GREY + _C.BOLD)}  "
            f"rho {_rho_ep:>+6.4f}  rhoAux {_rho_aux:>+6.4f}  "
            + ("" if _valide else f"[val ep{_val_epoch}] ") +
            f"Sortino {metric:>+6.3f}  "
            f"{_col(f'Sortino30 {s30:>+6.3f}', s30_col)}  "
            f"AvgW {_money(avg_win_train, width=8)}  AvgL {_money(avg_loss_train, width=8)}  "
            f"ActorL {np.mean(epoch_actor_loss):>+7.4f}  "
            f"AuxL {np.mean(epoch_aux_loss) if epoch_aux_loss else float(chr(110)+chr(97)+chr(110)):>7.4f}  "
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
                # MEME SCORE DE TRI QU'EN VALIDATION. Mesurer le test avec un
                # autre tri que celui qui a servi a calibrer les barres
                # mesurerait une strategie que personne ne deploierait.
                if getattr(cfg, "tri_par_tete_aux", False):
                    try:
                        _a = policy.rendement(st).float().cpu().numpy()
                        probs_np = 1.0 / (1.0 + np.exp(-np.clip(_a, -30, 30)))
                    except Exception:
                        pass
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
    cfg_duel.model_prefix = "saintv2_loup_duel_exec59"

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
    import cibles as _CIB
    _so, _ci = _CIB._regle(cfg_duel), _CIB._regle_cible(cfg_duel)
    print(f"SORTIE    stop {_so['sl']:g}xATR = 1 R | objectif "
          f"{(str(_so['tp'] / _so['sl']) + ' R') if _so['tp'] else 'AUCUN'} | "
          f"trailing " + (f"depuis {_so['ts'] / _so['sl']:.2f} R, "
                          f"distance {_so['td'] / _so['sl']:.2f} R"
                          if _so["ts"] is not None else "inactif"))
    print(f"CIBLE     stop {_ci['sl']:g}xATR = 1 R | objectif "
          f"{_ci['tp'] / _ci['sl']:.2f} R — ce que la tete auxiliaire apprend "
          f"a CLASSER, pas ce que la position encaisse")

    # Chaque fold repart de zéro avec les statistiques de son train.
    print("Walk-forward exec59: trois folds sans bootstrap inter-fold.")
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
    run_walkforward(cfg_duel, train_frac=0.55, val_frac=0.15, test_frac=0.10,
                    max_folds=3, start_fold=1,
                    bootstrap_from_path=None,
                    auto_chain=False)

    print("\n" + "=" * 70)
    print("  WALK-FORWARD TERMINÉ : 3 folds, tous les reseaux dans le meme "
          "rollout.")
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
