"""Noyau partagé KAIROS — modèle, features, normalisation, masque, SL/TP.

Tout ce qui DOIT être strictement identique entre training, backtests, live et
export ONNX vit ici, et nulle part ailleurs. Ces blocs étaient auparavant
recopiés dans 5 fichiers, ce qui avait laissé diverger :
  - l'alignement M1/H1 (fuite temporelle en training, valeurs partielles en live)
  - la calibration du bruit de ticks entre training et backtests
  - le multiplicateur de levier appliqué au PnL du seul training

Règle : ne jamais dupliquer une de ces fonctions dans un fichier appelant.
"""

import os

# Doit précéder l'import de torch/numpy (conflit OpenMP sous Windows/conda).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import collections
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# CONSTANTES
# ============================================================

# 0:BUY  1:SELL  2:HOLD
# 3 -> 4 : ACHETER, VENDRE, ATTENDRE, CLOTURER.
#
# LA QUATRIEME EST NOUVELLE, ET ELLE CHANGE LA NATURE DU SYSTEME. Jusqu'au
# 2026-09-21 une position ne se fermait que par une REGLE GEOMETRIQUE —
# stop, objectif, stop suiveur — et le modele n'avait aucun mot a dire : le
# masque ne lui laissait qu'ATTENDRE des qu'il etait en position.
#
# Le scalping M1 n'a ni stop ni trailing. La sortie devient donc une
# DECISION, portee par `tete_cloture`, et il lui faut une action pour
# s'exprimer.
N_ACTIONS = 4
MASK_VALUE = -1e4  # valeur de masquage compatible float16

NORM_STATS_PATH = "norm_stats_ohlc_indics.npz"

CLIP_SIGMA = 5.0

# Detention de reference pour normaliser bars_held_norm.
# DOIT rester egale a training.PPOConfig.scalping_max_holding : training,
# backtest et live construisent la meme feature avec cette constante, et les
# desynchroniser decale l'observation entre l'apprentissage et l'execution.
#
# 120 venait des barrieres larges de l'or. Mesure sur 250 000 entrees BTCUSD a
# SL 2.0xATR / TP 2.8xATR : detention MEDIANE de 7 barres, 98.2 % des trades
# resolus en 60 barres. A 120 la feature valait ~0.06 pour un trade typique et
# ne portait quasiment aucune information. A 30 elle vaut ~0.23 et sature
# au-dela de 90 barres, ce qui ne concerne qu'environ 1 % des trades.
# L'ECHELLE DE LA COLONNE D'AGE. UNE SEULE SOURCE, ET ELLE EN AVAIT TROIS.
#
# `bars_held_norm = min(bars_in_position / SCALPING_MAX_HOLDING, 3.0)`, donc
# la colonne SATURE a trois fois cette valeur. Au-dela, toutes les positions
# portent exactement le meme nombre et les tetes ne les distinguent plus.
#
# ELLE VALAIT 30, ET C'ETAIT JUSTE — quand le plafond de detention valait
# 30 minutes et que 98.2 % des trades se resolvaient en 60 barres. Le
# plafond a ete retire le 2026-09-22 et la detention se compte desormais en
# heures : une colonne qui sature a une heure et demie ne distingue plus
# rien.
#
# CE QUE LA SATURATION COUTAIT, mesure le 2026-09-22 sur la cible de
# `tete_profit`, memes occasions, seule l'echelle d'age changeant :
#
#     age / 1440 (sans saturation)   IC +0.1523   sature 16.7 %
#     min(age/30, 3.0)   ANCIEN      IC +0.1325   sature 66.7 %
#     min(age/480, 3.0)  RETENU      IC +0.1523   sature 16.7 %
#     sans age du tout               IC +0.0994
#
# DEUX TIERS DES ECHANTILLONS SATURAIENT. L'age vaut environ 0.05 d'IC a
# lui seul ; en faire disparaitre l'essentiel en revenait a le retirer a
# moitie.
#
# TROIS FICHIERS ECRIVAIENT CETTE VALEUR : ici, `PPOConfig` pour
# l'environnement, et `cibles_m1` en dur dans la fabrique d'echantillons.
# Le commentaire de `PPOConfig` en connaissait deux et avertissait deja que
# « les changer separement decale l'observation entre les deux ». Les deux
# autres la LISENT desormais ici.
# TROISIEME VALEUR EN UN JOUR, ET ELLE SUIT LA GEOMETRIE. 30 quand le
# plafond valait 30, 480 quand il valait 480, 60 maintenant que le scalping
# est retabli. La colonne sature a trois fois cette valeur, soit 180 barres
# — exactement `tenue_max_cloture`.
#
# LA LAISSER A 480 AURAIT ETE PIRE QUE DE NE RIEN FAIRE : avec des trades
# de quelques minutes, `min(age/480, 3.0)` reste colle a zero et la colonne
# ne distingue plus rien. Le defaut serait l'inverse de celui de ce matin —
# saturation totale d'un cote, ecrasement total de l'autre — et tout aussi
# muet. C'est `test_cloture_branchee` qui l'a attrape.
#
# 15 DEPUIS LE 2026-09-25, avec `horizon_cloture` : la colonne d'age sature
# a 45 barres, exactement `tenue_max_cloture`. A 60, un trade de quinze
# minutes n'aurait jamais depasse 0.25 sur une colonne qui va jusqu'a 3.
SCALPING_MAX_HOLDING = 15


# ============================================================
# INDICATEURS (M1 & H1)
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Indicateurs M1. LES FENETRES SONT CALIBREES POUR LA MINUTE.

    ATTENTION AVANT DE REUTILISER CETTE FONCTION SUR UN AUTRE TIMEFRAME.
    `rolling(1440)` apparait deux fois — pour `range_norm` et `vol_rank` — et
    signifie UNE JOURNEE sur M1. Sur H1 cela ferait 60 jours, sur H4 240 jours.
    Les colonnes porteraient alors leur nom sans mesurer ce qu'il annonce, et
    le warmup mangerait le debut de l'historique : mesure sur un essai H4,
    14.2 % des bougies perdues contre 4.5 % avec la fenetre a l'echelle.

    Le decalage de warmup deplace les periodes evaluees, donc le point de
    comparaison — assez pour inverser une conclusion. C'est arrive : un premier
    test donnait le bloc H4 legerement favorable ; avec les fenetres corrigees
    il degrade tout (AUC -0.0061, esperance a 5 % de +0.1579 a +0.0542, blocs
    positifs de 7/8 a 5/8).

    Pour un autre timeframe, passer la fenetre "journee" en parametre :
    6 bougies en H4, 24 en H1, 1440 en M1.
    """
    h = df["high"]
    l = df["low"]
    c = df["close"]

    def rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - 100 / (1 + rs)

    df["rsi_14"] = rsi(c, 14)

    prev_close = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    df["returns"] = c.pct_change()
    df["vol_20"] = df["returns"].rolling(20).std()
    # Amplitude de la bougie rapportee au prix, PUIS a sa propre normale
    # recente. Le simple (h-l)/c derive de 1.20 ecart-type sur 2018-2026 : les
    # regimes de volatilite changent lentement, et une feature qui suit cette
    # derive encode l'ANNEE plutot que l'etat du marche. Rapportee a sa moyenne
    # glissante d'une journee, elle repond a la vraie question : cette bougie
    # est-elle grande PAR RAPPORT AUX RECENTES ?
    _rn = (h - l) / (c + 1e-8)
    df["range_norm_abs"] = _rn
    df["range_norm"] = _rn / (_rn.rolling(1440, min_periods=120).mean() + 1e-12)

    # ---------- Prix sous forme STATIONNAIRE ----------
    # Les niveaux bruts (open/high/low/close) ne peuvent pas servir de features :
    # z-scorés sur 2022-2026 (mean 59127, std 32102) ils encodent la POSITION
    # dans l'historique, donc un index temporel. Le modèle y surapprend et, en
    # live, dès que le prix sort de la plage d'entraînement, la feature sature
    # au clip ±5σ et il opère hors distribution.
    # On garde la même information sous forme de ratios, invariants au niveau :
    #   - forme de la bougie (où open/high/low se situent vs la clôture)
    #   - écart à la tendance récente (close vs EMA 60)
    o = df["open"]
    df["open_rel"] = (o - c) / (c + 1e-8)
    df["high_rel"] = (h - c) / (c + 1e-8)
    df["low_rel"] = (l - c) / (c + 1e-8)
    df["close_ema_dev"] = c / (c.ewm(span=60, adjust=False).mean() + 1e-8) - 1.0

    # Momentum / filtre RSI / régime de volatilité.
    # Coulées en float dès ici : ces colonnes entrent telles quelles dans le
    # vecteur d'entrée, et un bool qui traverse un merge_asof devient un dtype
    # `object` — numériquement identique, mais qui fait sortir pandas de ses
    # chemins vectorisés et lever un FutureWarning à chaque fusion.
    df["mom_5"] = (df["close"] > df["close"].shift(5)).astype(np.float64)
    df["rsi_ok"] = ((df["rsi_14"] > 28) & (df["rsi_14"] < 72)).astype(np.float64)
    df["vol_rank"] = df["vol_20"].rolling(1440).rank(pct=True)
    df["high_vol_regime"] = (df["vol_rank"] > 0.65).astype(np.float64)

    return df


# ============================================================
# FEATURES
# ============================================================

# ============================================================
#  TROIS LECONS TRANSVERSALES — elles valaient deja sur XAUUSD
# ============================================================
#
# 1. Une feature n'est pas bonne ou mauvaise dans l'absolu, elle l'est pour une
#    CIBLE donnee. Les colonnes horaires coutaient -0.0022 d'AUC contre une
#    cible SL 3xATR / R:R 1.4 et rapportaient +0.0018 contre SL 5xATR / R:R 2.0.
#    TOUT ELAGAGE DOIT ETRE REFAIT SI LE SL/TP CHANGE.
#
# 2. Les apports marginaux NE SE COMPOSENT PAS. Neuf colonnes d'apport
#    individuel nul ou negatif coutaient 0.0024 une fois retirees ensemble —
#    plus que la meilleure feature du jeu n'en apporte. Elles se couvrent
#    mutuellement. C'est ce qui a fait annuler l'elagage a 10 colonnes.
#
# 3. Une feature doit VARIER A L'ECHELLE OU LA DECISION SE PREND. La detention
#    mediane d'un trade est de 7 barres ; une serie constante sur cette duree ne
#    peut pas departager deux entrees. C'est le critere qui separe taker_ratio
#    (autocorr 1 min 0.82, apport -0.0498 si retiree) du funding et des ratios
#    long/short (autocorr 0.999+, apport 0.0000).
#
# Une source ECARTEE, pour memoire : l'indice dollar, malgre une correlation de
# -0.345 avec l'or. Apport marginal exactement 0.0000, et il limitait la fusion
# a 1.22 M bougies contre 1.32 M — les 8 % de donnees perdues valaient plus que
# la colonne.

# JEU A 30 COLONNES — refait sous protocole PURGE, aux barrieres courantes.
#
# L'ancien tableau inscrit ici (AUC 0.6215, E[R] +0.1751, t +3.4) reposait sur
# PAS = 10 : une entree toutes les 10 barres alors que les barrieres mettent
# jusqu'a 240 barres a se resoudre. Les fenetres de resultat se recouvraient,
# ce qui faisait compter 1 368 trades la ou il y en a 58 d'independants. Mesure
# de l'ecart : a 5 % de selectivite, un modele a arbres passait de 58.0 % de
# reussite avec chevauchement a 39.7 % sans.
#
# PROTOCOLE ACTUEL (mesure_features_purgee.py) : 8 blocs successifs, entrees
# espacees de 240 barres, purge de 240 entre train et bloc, 7 440 candidats
# independants, SL 2.0xATR / R:R 2.0, fenetre de test intouchee.
#
#                              AUC     E[R] a 2 %
#     30 actuel             0.5832       +0.2960
#     sans les 2 Binance    0.5517       +0.0953
#     sans le bloc H1       0.5967       +0.1367
#     + Ichimoku rapide     0.5800       +0.2419
#
# TROIS LECTURES, dont une contre-intuitive.
#
# 1. Binance porte l'avantage. Retirer taker_ratio coute -0.0309 d'AUC et
#    -0.2443 d'esperance ; retirer taker_1m_ma5 coute -0.2521 d'esperance.
#    Ce sont les deux seules colonnes indispensables des trente.
#
# 2. Ichimoku est REJETE. Il donnait +0.0030 avec interactions sur une seule
#    decoupe a R:R 1.4 ; sous protocole purge aux barrieres courantes il vaut
#    -0.0031 d'AUC et -0.054 d'esperance.
#
# 3. LE H1 EST GARDE MALGRE UNE AUC DEFAVORABLE. Le retirer AMELIORE l'AUC de
#    +0.0136 — et degrade l'esperance a toutes les selectivites utilisees :
#
#        selectivite    30 actuel    sans H1
#             1 %        +0.3453     +0.1324
#             2 %        +0.2960     +0.1367
#             5 %        +0.1579     +0.0588
#            10 %        -0.0177     +0.0614   <- il ne gagne qu'ici
#
#    et par bloc a 5 % : 7 blocs positifs sur 8 avec H1, 4 sur 8 sans.
#
#    L'AUC mesure le classement sur TOUTE la distribution ; la strategie ne
#    touche jamais que les 2 a 5 % du sommet. Un groupe peut degrader l'ordre
#    global tout en ameliorant l'extreme — et c'est l'extreme qui est trade.
#    CHOISIR LA METRIQUE QUI CORRESPOND AU POINT DE FONCTIONNEMENT.
FEATURE_COLS_M1 = [
    "rsi_14", "returns", "vol_20", "range_norm", "open_rel", "high_rel",
    "low_rel", "close_ema_dev", "mom_5", "rsi_ok", "vol_rank", "high_vol_regime",
]

FEATURE_COLS_H1 = [
    "rsi_14_h1", "returns_h1", "vol_20_h1", "range_norm_h1", "open_rel_h1",
    "high_rel_h1", "low_rel_h1", "close_ema_dev_h1", "mom_5_h1", "rsi_ok_h1",
    "vol_rank_h1", "high_vol_regime_h1",
    "close_h1_dev",      # ecart du dernier H1 CLOTURE au prix M1 courant
]

# SOURCE EXTERNE — Binance BTCUSDT perpetuel : positionnement des acteurs et
# flux agressif, la seule classe d'information absente du flux CFD.
# Mesure du 2026-09-14 (sonde logistique, SL 2.0xATR / R:R 1.4, meme fenetre
# de validation) : retirer une colonne du jeu de 30 et regarder ce que ca coute.
#
#     sans taker_ratio     0.6178 -> 0.5680   -0.0498
#     sans ls_ratio_top    0.6178 -> 0.6177   -0.0000
#
# ls_ratio_top est INERTE. Son autocorrelation a une minute vaut 0.999985 :
# c'est une variable de REGIME, constante sur les 7 barres d'un trade median,
# donc incapable de departager deux entrees espacees de quelques minutes.
# Meme verdict pour les quatre colonnes Binance jamais branchees (funding,
# oi_change, ls_ratio_retail) : 0.4847 d'AUC a elles seules, sous le hasard.
#
# On la remplace par le meme flux agressif mais a la MINUTE, lisse sur 5 —
# derive des archives klines 1m, que le pipeline n'avait jamais consommees.
# Balayage de la fenetre de lissage, apport marginal sur les 30 :
#
#     1min +0.0016 | 3min +0.0067 | 5min +0.0087 | 10min +0.0050
#     15min +0.0027 | 30min +0.0017 | 60min +0.0004 | 120min -0.0000
#
# Bosse reguliere a sommet unique, pas un pic : le signal a une echelle de
# temps propre de 5 minutes. A 120 min la serie est redevenue une variable de
# regime et vaut exactement zero, comme ls_ratio_top. Le remplacement garde le
# compte a 30 features, donc le meme cout GPU.
# RETIRE LE 2026-09-16 : ces colonnes n'existent QUE sur Binance.
#
# `taker_ratio` et `taker_ma5` viennent du flux agressif acheteur/vendeur, que
# seul un carnet d'ordres crypto publie. Elles n'ont pas d'equivalent sur l'or,
# le Nasdaq ou n'importe quel CFD : les garder condamnait le modele a un seul
# instrument.
#
# CE QUE LEUR RETRAIT COUTE, mesure avant de decider — meme apprenant lineaire,
# meme fenetre, meme cible, entrainement sur le train et mesure sur la
# validation :
#
#   jeu de features               colonnes      rho     +/-   E[R] sommet 5%
#   260 colonnes (avec)                260  +0.0432  0.0216          +0.4082
#   256 colonnes (sans)                256  +0.0426  0.0216          +0.3956
#
# Six dix-millemes de rho, et 0.013 R sur le sommet contre une erreur-type de
# 0.13. Le retrait est gratuit, et il ouvre un second instrument dont
# l'avantage mesure est de 50 % superieur a celui du BTC.
FEATURE_COLS_EXT = []

# LIQUIDITE ET TEMPS. J'avais exclu les features d'heure sur BTCUSD en invoquant
# une mesure a -0.0022 ; la mesure comparative dit le contraire sous cette
# configuration : 27 -> 30 colonnes fait passer l'esperance de +0.1249 a +0.1751.
# LIQUIDITE ET TEMPS. `spread_rel` a ete RETIRE le 2026-09-15, quand le jeu H1
# est passe de 3.6 a 9.1 ans en changeant de source. Le spread venait de MT5,
# qui ne remonte qu'a 2023 ; sur les six annees ajoutees il aurait fallu le
# constanter. Une colonne constante sur 70 % du jeu n'est pas neutre : elle
# apprend au modele a distinguer "avant" de "apres", c'est-a-dire la date, qui
# est la fuite la plus facile a commettre et la plus difficile a voir.
#
# Les deux colonnes qui le remplacent viennent des memes archives que le prix,
# donc elles existent sur toute la periode :
#   - taille moyenne d'un trade  : une heure poussee par de gros ordres ne se
#     comporte pas comme une heure faite de poussiere, a volume egal ;
#   - intensite                  : rang du nombre de trades sur une semaine
#     glissante, soit le regime d'activite, en rang pour rester stationnaire
#     entre 2017 et 2026 ou les volumes absolus n'ont aucune commune mesure.
# `flux_taille_trade` et `flux_intensite` derivent de `quote_vol` et
# `nb_trades`, deux champs que seul Binance publie — retires le 2026-09-16
# pour la meme raison que FEATURE_COLS_EXT. `heure_sin`/`heure_cos` se
# calculent a partir de l'horodatage seul et restent.
# LE REGIME DE TENDANCE, en horizons longs. Le plus long que le modele voyait
# etait `mom_5_h4`, soit vingt heures : il ne pouvait pas savoir dans quel
# regime il se trouvait. Mesure du 2026-09-20 dans `prepare_m5.construit` :
# seuls les horizons LONGS separent les deux sens, et la separation croit
# monotonement avec l'horizon la ou les detecteurs Ichimoku ne donnent rien.
FEATURE_COLS_LIQ_TEMPS = ["tend_mom_sem", "tend_mom_mois", "tend_vs_ma_mois",
                          "heure_sin", "heure_cos"]

# ============================================================
#  ECHELLE DE DECISION — H1 depuis exec10
# ============================================================
#
# La friction est un montant FIXE (~37.84 $ sur BTCUSD) ; l'unite de risque est
# la distance au stop, qui grandit avec l'echelle. Rapport mesure :
#
#        UT    friction/ATR    E[R] de la strategie de range
#        M1        1.05 R              -1.69
#        M5        0.40 R              -0.66
#       M15        0.21 R              -0.46
#        H1        0.09 R              -0.27
#        H4        0.04 R              +0.12
#
# Monotone sans exception. Sur M1 on paie plus que le stop avant d'avoir eu
# raison. H1 est le point d'equilibre : friction basse ET assez de barres pour
# entrainer (30 379), ce que le H4 n'offre pas (271 trades).
#
# Le jeu H1 est produit par prepare_h1.py. Les 12 colonnes de prix sont
# calculees SUR H1 ; le contexte superieur est le H4, decale de shift(1).
FEATURE_COLS_TF = FEATURE_COLS_M1                      # calculees sur H1
# ECHELLE DE DECISION ET CONTEXTE SUPERIEUR. Les deux vont ensemble : le bloc
# de contexte porte le nom de SON echelle, pas de celle des decisions.
#
#     decisions en H1  ->  contexte H4  ->  suffixe _h4
#     decisions en M5  ->  contexte H1  ->  suffixe _h1
#
# Laisser le suffixe fige a "_h4" en changeant d'echelle donnerait des colonnes
# dont le nom ment sur leur contenu, et prepare_m5 echouerait sur un desaccord
# avec cette liste — ce qui est le bon comportement, mais mieux vaut que la
# source de verite sache de quelle echelle on parle.
#
# POURQUOI LE M5 EST DEVENU L'ECHELLE DE DECISION, mesure du 2026-09-16, avec
# sortie sur barriere UNIQUEMENT comme l'environnement et le live :
#
#     UT   SL   friction/R   E[R] hasard   duree med   occasions   requis
#     M1  8.0      0.132R       -0.1136        122b      10 472   +0.1136R
#     M5  4.0      0.099R       -0.0892         44b       5 811   +0.0892R
#     H1  2.0      0.047R       -0.0233         13b       1 640   +0.0233R
#
# Le mur de ce depot n'a jamais ete l'architecture : c'est le nombre
# d'occasions INDEPENDANTES, qui vaut la duree d'historique divisee par la
# duree d'un trade — jamais par le nombre de barres. Il en faut ~4 900 pour
# qu'un avantage de 0.02 R soit lisible ; le H1 en offre 2 314, le M5 ~15 000.
#
# Le M1 reste hors de portee a toute largeur de stop : meme a 8xATR il exige
# +0.114 R d'avantage, davantage que tout ce que ce depot a mesure.
TIMEFRAME = "M5"
# LES ECHELLES SUPERIEURES SONT UNE LISTE, de la plus proche a la plus
# lointaine. En H1 il n'y en avait qu'une, le H4 ; en M5 on lit le H1 ET le H4.
#
# Pourquoi plusieurs. Les periodes Ichimoku sont des nombres de BOUGIES, donc
# chaque echelle raconte une duree differente avec les memes colonnes :
#
#     echelle   Tenkan 9   Kijun 26   Senkou-B 52
#        M5      45 min      2 h 10      4 h 20
#        H1       9 h        26 h        52 h
#        H4      36 h         4.3 j       8.7 j
#
# Un trade M5 dure 44 barres, soit 3 h 40. Le M5 decrit donc ce qui se passe
# PENDANT le trade, le H1 le mouvement qui le contient, le H4 le regime dans
# lequel ce mouvement s'inscrit. C'est exactement la lecture que prescrivent
# les documents Ichimoku : trouver le signal sur son unite de temps, le valider
# en basculant sur les unites superieures.
ECHELLES_SUP = {"H1": ["_h4"], "M5": ["_h1", "_h4"]}[TIMEFRAME]
SUFFIXE_SUP = ECHELLES_SUP[0]       # le contexte immediat, celui du merge_asof
FEATURE_COLS_SUP = [c.replace("_h1", SUFFIXE_SUP) for c in FEATURE_COLS_H1]

# Features de la strategie "bornes d'un range" (Ichimoku / Riguet).
# Voir features_range.py pour les definitions exactes et, surtout, pour la
# raison qui fait que les figures de chandeliers y sont CONDITIONNEES a la
# proximite d'une borne : hors des bornes, les documents les disent
# "strictement inutiles", et une colonne calculee partout n'est que du bruit.
from features_range import GROUPES as _GROUPES_RANGE
FEATURE_COLS_RANGE = [c for g in _GROUPES_RANGE.values() for c in g]

# Features de TOUTES les autres strategies Ichimoku : Chikou-Span (le filtre
# du systeme, dont le modele n'avait rien), replis sur plat Kijun, cassures de
# nouveaux plus hauts, espace tradable pour le ratio 2/1, boussole du nuage, et
# les 14 structures de renversement en chandeliers, cette fois NON conditionnees
# a la proximite d'une borne de range.
#
# MESURE QUI A DECIDE DE LEUR ADOPTION, dans le MOTEUR REEL, sur les memes
# fenetres de test que PPO, avec une selectivite de 5 % fixee avant tout test :
#
#      jeu          PnL    trades   ecart au point mort    PF    folds positifs
#   56 colonnes   +207.69$   387        +1.6 pt           1.07        2/3
#  103 colonnes   +568.31$   432        +3.8 +/- 2.3 pt   1.18        3/3
#
# Les 47 colonnes apportent +2.2 points et font passer de deux folds positifs
# a trois. Ce n'est pas significatif (t = 1.6), mais c'est le meilleur chiffre
# hors-echantillon que ce depot ait produit.
#
# UNE PREMIERE VERSION DE CES COLONNES FUYAIT LE FUTUR, et il faut que ce soit
# ecrit ici. Cinq d'entre elles calculaient leur pente avec une difference
# CENTREE, qui lit la barre suivante ; `ich_kumo_futur_pente` allait jusqu'a
# contenir le plus haut et le plus bas du lendemain. Le test affichait alors
# +25.5 points et un profit factor de 2.85. C'est l'invraisemblance du chiffre
# qui a trahi la fuite, aucune verification ne l'aurait attrapee — la colonne
# gardait son nom, sa forme et son ordre de grandeur. Voir features_ichimoku
# `_pente`. Les mesures faites avant ce correctif ont ete jetees.
from features_ichimoku import COLONNES as FEATURE_COLS_ICHIMOKU

# LES MEMES STRUCTURES, LUES A L'ECHELLE SUPERIEURE.
#
# Les periodes Ichimoku sont des nombres de BOUGIES : Tenkan 9, Kijun 26,
# SSB 52. Calculees en M5 elles couvrent 45 minutes, 2 h et 4 h ; calculees en
# H1, neuf heures, un jour et deux jours. Ce ne sont pas les memes structures
# de marche, malgre des noms identiques.
#
# POURQUOI CE BLOC EXISTE, mesure du 2026-09-16. L'avantage du modele (+0.09 a
# +0.13 R) avait ete mesure en H1, sur des colonnes H1. Passe en M5 pour
# multiplier par 6.5 le nombre d'occasions independantes, exec22 n'apprenait
# plus rien : huit epochs entrainees, zero positive, gain nul sur sa propre
# politique gelee — alors meme que la mecanique d'apprentissage etait reparee
# (entropie 1.099 -> 0.996, etendue 0.53, clipfrac 18 %).
#
# L'hypothese est donc que le gain d'occasions du M5 se payait d'une perte
# d'avantage : les structures qui portaient le signal avaient disparu avec le
# changement d'echelle. Plutot que de revenir au H1 et de rendre les
# occasions, on AJOUTE les colonnes H1 aux lignes M5 — la finesse du M5 pour
# decider, les structures H1 la ou l'avantage a ete mesure. C'est aussi ce que
# prescrit le livre : valider un signal en basculant sur les UT superieures.
#
# LE M5 PAIE LE SURCOUT. 72 colonnes de plus font passer la tete de 856 a
# 1 400 entrees et le modele de 26 752 a ~45 000 parametres — mais avec 15 145
# occasions independantes au lieu de 2 314, cela fait 3 parametres par
# occasion contre 11.6. La nouvelle echelle achete la capacite d'en porter
# davantage.
FEATURE_COLS_RANGE_SUP = [c + SUFFIXE_SUP for c in FEATURE_COLS_RANGE]
FEATURE_COLS_ICHIMOKU_SUP = [c + SUFFIXE_SUP for c in FEATURE_COLS_ICHIMOKU]


def bloc_echelle(sfx):
    """Le jeu COMPLET lu a une echelle superieure : 85 colonnes.

    Douze indicateurs de base, l'ecart du prix courant a la derniere cloture de
    cette echelle, les 25 colonnes de range et les 47 d'Ichimoku. Les memes
    noms qu'en M5, a leur suffixe pres — et des contenus qui n'ont rien a voir,
    puisque les periodes se comptent en bougies.
    """
    return ([c.replace("_h1", sfx) for c in FEATURE_COLS_H1]
            + [c + sfx for c in FEATURE_COLS_RANGE]
            + [c + sfx for c in FEATURE_COLS_ICHIMOKU])


# Les echelles AU-DELA de la premiere. La premiere est deja epelee ci-dessus,
# en trois listes separees, parce que le H1 l'etait avant que le M5 n'existe et
# que d'autres fichiers s'y referent nommement.
FEATURE_COLS_SUP_LOINTAINES = [c for sfx in ECHELLES_SUP[1:]
                               for c in bloc_echelle(sfx)]

# COUT EN PARAMETRES, ET POURQUOI IL EST PAYABLE. La tete lit n_features x
# d_model, soit 260 x 8 = 2 080 entrees contre 1 400 a une seule echelle
# superieure : le modele passe d'environ 45 000 a 58 000 parametres.
# L'entrainement M5 offre ~15 100 occasions INDEPENDANTES, donc 3.8 parametres
# par occasion — quand le run H1 qui avait rendu +2.2 points en test en portait
# 11.6. Ce n'est pas la capacite qui borne ici.
#
# LE VRAI RISQUE EST AILLEURS : chaque colonne ajoutee est une direction de
# plus dans laquelle un gradient de politique nourri par ~1 000 decisions par
# epoch peut se perdre. C'est pourquoi les blocs se mesurent avant de se
# garder, et sur la VALIDATION uniquement.
# LA MICROSTRUCTURE, AJOUTEE LE 2026-09-21, ET C'EST LA PREMIERE FAMILLE
# QUI NE SOIT PAS UNE TRANSFORMATION DU PRIX.
#
# POURQUOI ELLE EXISTE. Les 259 colonnes precedentes sont toutes des
# fonctions de l'OHLC — Ichimoku, ranges, volatilite, momentum. Mesure du
# 2026-09-21, occasions DISJOINTES et plancher par ROTATION : aucune ne
# porte de direction, a aucun horizon de 1 a 60 minutes. Et le modele
# entraine choisissait le mauvais cote — -1.235 bps contre +0.011 pour un
# tirage au sort, soit 5.1 ecarts-types. Une 260e fonction du meme OHLC
# n'y changerait rien.
#
# CE QUE MT5 DONNE ET QUE LE DEPOT JETAIT. `copy_rates` renvoie `spread`
# depuis toujours ; le preparateur selectionnait six colonnes et le
# laissait tomber. Et `copy_ticks_range` rend bid et ask a chaque
# cotation : 257.8 millions de ticks sur l'historique, agreges par minute.
#
# CE QU'ON NE PEUT PAS EN TIRER, et il faut l'ecrire. Le champ `last` vaut
# zero sur ce CFD : ce sont des COTATIONS, pas des transactions. Il n'y a
# donc pas de volume a l'ask contre volume au bid, donc pas de vraie
# classification acheteur/vendeur. `tick_desequilibre` compte l'asymetrie
# des MISES A JOUR du carnet — un proxy plus faible, mais qui n'est pas
# dans le prix.
FEATURE_COLS_MICRO = [
    "spread_bar",         # le spread cote sur la barre, en bps
    "tick_n",             # nombre de cotations dans la minute
    "tick_spread_moy",    # ecart bid-ask moyen sur la minute
    "tick_spread_max",    # son maximum
    "tick_ask_part",      # part des cotations ou SEUL l'ask a bouge
    "tick_bid_part",      # part des cotations ou SEUL le bid a bouge
    "tick_desequilibre",  # ask_part - bid_part : la pression de carnet
]

# LE FLUX D'ORDRES, REMIS LE 2026-09-21 APRES AVOIR ETE RETIRE LE 09-16.
#
# CE QUI L'AVAIT FAIT RETIRER. Le commentaire de `prepare_m5.construit` le
# dit : « elles n'existent QUE sur Binance : aucun CFD ne publie
# `taker_buy_base` ni `nb_trades`. Elles ont ete retirees pour que l'or
# puisse partager exactement le meme jeu de colonnes ». On a donc supprime
# la seule information qui ne fut pas du prix, pour qu'un instrument qui
# n'en a pas puisse suivre.
#
# CE QUI LES FAIT REVENIR. Une journee de mesures sur l'or M1, le
# 2026-09-21, occasions disjointes et plancher par rotation :
#
#   les 266 colonnes une par une   0 au-dessus du plancher, 1 a 480 min
#   une combinaison (ridge)        aucun horizon au-dessus de son plancher
#   le modele entraine             +0.004 bps a 0.0 ecart-type
#
# Ce n'est pas un defaut d'apprentissage : il n'y a rien a apprendre. Les
# sept colonnes de microstructure ajoutees le meme jour ne portent que des
# COTATIONS — le champ `last` d'un CFD vaut zero, il n'y a pas de
# transactions publiques, parce qu'un CFD n'a pas de marche central.
#
# BINANCE PUBLIE SES TRANSACTIONS, gratuitement et depuis 2017 : combien
# d'echanges, quel volume achete a l'AGRESSION. C'est la seule classe
# d'information de ce depot qui ne soit pas une fonction de l'OHLC.
#
# ELLES N'EXISTENT QUE SUR LE BTC, et il faut l'assumer : un cache d'or
# ne les portera jamais. `prepare_btc_m1` les construit ; `MarketData`
# leve si une colonne de `FEATURE_COLS` manque, donc l'or ne peut plus
# tourner avec ce jeu. C'est voulu — le partage force etait justement la
# faute.
FEATURE_COLS_FLUX = [
    "nb_trades",          # nombre de TRANSACTIONS dans la minute
    "taker_buy_base",     # volume achete a l'agression
    "taker_ratio",        # sa part du volume : la pression acheteuse
    "taker_ma5",          # la meme, lissee sur cinq minutes
    "flux_taille_trade",  # taille moyenne d'une transaction
    "flux_intensite",     # transactions par minute, en rang sur la journee
]

# LES RANGS GLISSANTS, ET C'EST UNE MESURE QUI LES IMPOSE.
#
# Ces deux colonnes existent DEJA dans le jeu — `close_ema_dev` et
# `taker_buy_base` — et le modele ne s'en est jamais servi. Voici
# pourquoi.
#
# Le signal ne vit pas dans le NIVEAU de ces colonnes, il vit dans leur
# POSITION LOCALE. « Le cours est dans son dixieme le plus bas des 20 000
# dernieres barres » est un evenement ; « close_ema_dev vaut -0.004 » n'en
# est pas un, parce que la valeur qui correspondait a un creux en 2024 ne
# correspond plus a rien en 2026. La soiree du 2026-09-22 a montre la
# meme derive sur le spread : un seuil au 40e centile calcule deux ans
# plus tot n'attrapait PLUS AUCUNE occasion sur le BTC, et 96 % des
# occasions sur l'or.
#
# ET `training.py` FIGE LA NORMALISATION sur le train du fold 1 — ce qui
# protege de la fuite, et c'est indispensable, mais fait voir au modele la
# DERIVE de la colonne au lieu de sa position. Il recevait l'information
# sous une forme ou elle n'etait pas lisible.
#
# CE QUE LE RANG DEBLOQUE, mesure sur 23 mois, ecart au marche du MEME
# mois, friction reelle, horizon 480 min, test du signe :
#
#     creux seul                    +6.48 bps   17/23 mois   p 0.017
#     creux + flux fort            +11.95 bps   20/23 mois   p 0.0003
#
# LA FENETRE EST DE 20 000 BARRES, soit deux semaines de M1. Assez long
# pour que le rang soit stable, assez court pour suivre le regime. Elle
# ne regarde QUE LE PASSE : `rolling` ferme la fenetre sur la barre
# courante.
FEATURE_COLS_RANGS = [
    "creux_rang",   # rang glissant de `close_ema_dev` : la profondeur du creux
    "flux_rang",    # rang glissant du volume agressif : purge ou erosion
]

FEATURE_COLS = (FEATURE_COLS_TF + FEATURE_COLS_SUP
                + FEATURE_COLS_EXT + FEATURE_COLS_LIQ_TEMPS
                + FEATURE_COLS_RANGE + FEATURE_COLS_ICHIMOKU
                + FEATURE_COLS_RANGE_SUP + FEATURE_COLS_ICHIMOKU_SUP
                + FEATURE_COLS_SUP_LOINTAINES
                + FEATURE_COLS_MICRO + FEATURE_COLS_FLUX
                + FEATURE_COLS_RANGS)

N_BASE_FEATURES = len(FEATURE_COLS)

# Embedding de position, DANS CET ORDRE EXACT — l'entrainement et le live
# remplissent ces colonnes chacun de leur cote, et une inversion ne leverait
# aucune erreur :
#   0  position         -1 / 0 / +1, sens net
#   1  unrealized_atr   latent des positions ouvertes, en unites d'ATR
#   2  bars_held_norm   age de la PLUS ANCIENNE position
#   3  capacite         places ouvrables / (ouvrables + tenues)
#   4  distance_creux   creux courant du compte, 1.0 = il ne reste rien
#
# LA CINQUIEME A ETE AJOUTEE LE 2026-09-19, et voici pourquoi. La penalite de
# creux est portee par chaque position — on demandait donc au modele de gerer
# une grandeur qu'il ne voyait pas.
#
# IL N'ETAIT PAS AVEUGLE POUR AUTANT : la capacite restante correle a -0.60
# avec le creux, puisqu'elle est proportionnelle a l'equite, et une
# regression sur les quatre colonnes en expliquait 67.5 % de la variance.
# MAIS LE PIC MANQUAIT. Le creux vaut (pic - equite) / pic, et le pic n'etait
# nulle part : l'erreur residuelle valait 4.3 points de creux, et deux
# observations IDENTIQUES pouvaient correspondre a 12 points d'ecart. Avec un
# c'est exactement l'incertitude qui compte — le modele ne pouvait pas
# distinguer « je vais bien » de « je suis au bord ».
#
# LA FORME A CHANGE D'ECHELLE LE MEME JOUR, en fin de journee. Elle valait
# `creux / max_drawdown`, donc 1.0 la ou le garde-fou coupait a 40 %. Ce
# garde-fou a ete retire : il tuait l'episode au quart de sa tranche, laissait
# les trois quarts non mesures, et un tiers des episodes ainsi tues auraient
# fini AU-DESSUS du capital de depart. Seule la ruine termine desormais un
# episode — appel de marge, lot minimum infinancable, equite a zero — comme
# chez un courtier.
#
# La colonne porte donc le CREUX LUI-MEME : 0 au sommet, 1.0 quand il ne
# reste plus rien. Bornee a 40 % elle saturait, et un compte a -40 % etait
# indiscernable d'un compte a -90 % — elle cessait d'informer exactement
# quand la situation empirait.
#
# L'EQUITE QU'ELLE MESURE INCLUT TOUTES LES POSITIONS OUVERTES : le compte
# est juge sur ce qu'il vaudrait s'il fermait tout maintenant, ce que le
# courtier regarde. `kairos_live.distance_garde_fou` applique la MEME
# formule ; les deux doivent bouger ensemble.
#
# COUT : l'observation passe de 260 a 261 entrees, donc AUCUN checkpoint
# anterieur n'est chargeable. `checkpoints.py` filtre sur `n_features` et les
# refusera au lieu de les charger de travers.
# LE MODELE CHOISIT SON BUDGET DE RISQUE, donc le NOMBRE de positions.
#
# PREMIERE TENTATIVE, ET SON ECHEC. J'avais d'abord donne au modele une tete
# de TAILLE, qui multipliait la taille de chaque position. Mesure du
# 2026-09-19 : les quatre paliers rendaient EXACTEMENT le meme lot.
#
#     taille demandee a l'echelle 1.00 : 0.0205 once
#     lot minimum du courtier          : 1.0000 once
#
# Le plancher du courtier ecrase la demande d'un facteur 49. A 1 000 $ de
# capital, UN lot minimum sur l'or vaut 1 550 $ de notionnel — 155 % du compte
# — et risque 2.81 % du capital au stop. `risk_per_trade` n'a donc jamais ete
# operant a ce capital, et aucune tete de taille ne peut l'etre : il faudrait
# ~50 000 $ pour que le dimensionnement par le risque reprenne la main.
#
# LE SEUL LEVIER QUI EXISTE REELLEMENT A CE CAPITAL EST LE NOMBRE DE
# POSITIONS. Il est gouverne par `peut_entrer()` -> `places_ouvrables`, donc
# par le BUDGET DE RISQUE du portefeuille — celui qui valait 3 % jusqu'au
# 2026-09-19 et qu'on a mis a zero pour laisser le modele apprendre. Le mettre
# a zero ne lui a pas donne le controle : cela le lui a RETIRE, puisque rien
# dans son espace d'action ne le remplacait.
#
# La tete choisit donc un budget, et le budget commande la capacite. C'est
# exactement « qu'il ouvre autant de trades qu'il peut, afin de gerer le
# risque » — mais avec la main sur le curseur.
#
# LES PALIERS SE DEDUISENT DU RISQUE D'UN LOT MINIMUM (2.81 % a 1 000 $ sur
# l'or) : ils valent approximativement 1, 2, 5 et 14 positions tenables. Le
# plus bas vaut 3 %, exactement le reglage d'exec04 — le seul run de ce depot
# dont le creux de validation soit reste a 14 %.
#
# AUCUN PALIER NE PEUT ETRE NUL. Un budget sous le risque d'un seul lot
# rendrait `peut_entrer()` faux en permanence ; l'environnement ne figurerait
# plus dans `deciding`, et le modele ne serait PLUS JAMAIS consulte pour
# relever son budget. Le plus bas doit donc laisser passer une position.
#
# LA POLITIQUE EST FACTORISEE : la direction garde ses trois actions, donc les
# seuils calibres, le masque de cote, `decide_avec_barres` et la tete de rang
# sont INCHANGES. Le budget est tire a CHAQUE decision, y compris quand elle
# vaut « attendre » — choisir de rester serre en attendant est une decision de
# portefeuille a part entiere, contrairement a une taille qui ne sert qu'a
# l'ouverture.
#
#     log pi(a) = log pi_dir(d) + log pi_budget(b)
# `N_BUDGETS` a ete retire le 2026-09-21 avec la tete de budget, les
# paliers et la colonne d'etat. Le nom est garde en commentaire parce que
# d'anciens points de reprise le portent dans leur manifeste.
# ZERO EST UN PALIER, ET C'EST LE MECANISME D'ABSTENTION DU MODELE.
#
# Le budget etait le seul levier de DOSAGE ; il devient aussi le levier
# d'ARRET. A 0 %, aucune place n'est ouvrable : le modele choisit de ne pas
# trader, et ce choix est appris comme les autres, par la meme tete.
#
# L'ECHELLE EST EN NOMBRE DE POSITIONS MINIMALES, PLUS EN POURCENTAGE
# D'EQUITE — et c'est une mesure qui l'a imposee, pas un gout.
#
# CE QUI NE MARCHAIT PAS. Les paliers valaient (0, 1, 3, 6, 15, 40) % du
# compte. Or le budget de risque plafonne la somme des pertes si TOUS les
# stops etaient touches, et une position au lot MINIMUM risque deja un
# montant incompressible. A 1 000 $ de capital, stop 10 x ATR sur l'or :
#
#     ATR  4 $  ->  une position risque   40 $  =  4 % du compte
#     ATR  8 $  ->  une position risque   80 $  =  8 % du compte
#     ATR 15 $  ->  une position risque  150 $  = 15 % du compte
#
# Un budget SOUS ce montant n'ouvre pas une position plus petite : il n'en
# ouvre AUCUNE. Les paliers 1 % et 3 % etaient donc inoperants en permanence,
# 6 % l'etait des que l'ATR depassait 6 $. Trois paliers sur six faisaient la
# meme chose que le palier 0 — ne pas trader — sans le dire. L'echelle de
# risque que PPO croyait apprendre n'existait pas.
#
# CE QUE LE JOURNAL DISAIT, et que personne n'avait lu ainsi (exec47, ep. 1) :
#
#     budgets[0%:14(100) 1%:18(100) 3%:21(99) 6%:14(97) 15%:14(80) 40%:18]
#
# Le nombre entre parentheses est la part des entrees refusees par le palier
# lui-meme. PLUS LE BUDGET EST GROS, MOINS IL REFUSE — l'inverse de
# l'intuition, et la signature exacte d'un plancher de lot minimum. La
# validation etait tombee de 300-450 trades a 56.
#
# CE QUE L'ECHELLE VEUT DIRE MAINTENANT : « combien de positions minimales
# suis-je pret a avoir en risque a la fois ». Zero reste l'abstention. Chaque
# palier au-dessus ouvre au moins une position des que la marge le permet,
# quels que soient l'ATR ET LE CAPITAL.
# LE BUDGET EST UNE PART DU PLAFOND DE SURVIE, ET NON UN NOMBRE ABSOLU.
#
# CE QUE L'ECHELLE ABSOLUE COUTAIT, mesure le 2026-09-21. Le plafond de
# survie borne le total a `PLAFOND_RISQUE_EQUITE` de l'equite, donc il vaut
# `0.40 x equite / risque_une` POSITIONS — et `risque_une` suit l'ATR. Quand
# l'or s'agite, le plafond descend et ECRASE le haut de l'echelle :
#
#     equite 1 000 $    paliers (0, 1, 2, 4, 7, 12) absolus
#     ATR  4 $   ->  0   1   2   4   7  10      6 paliers distincts sur 6
#     ATR  8 $   ->  0   1   2   4   5   5      5
#     ATR 15 $   ->  0   1   2   2   2   2      3
#
# QUATRE PALIERS SUR SIX FONT LA MEME CHOSE en marche agite. PPO recoit alors
# la MEME recompense pour quatre actions differentes : son gradient sur ces
# etats est du bruit pur, et aucun reglage d'entropie n'y change rien. C'est
# la troisieme cause mesuree du budget qui n'apprend pas — les deux autres
# etant le bonus d'entropie a 3x le gradient de politique et les 23 trades
# fermes par epoch.
#
# L'ASSERTION DE STRICTE CROISSANCE NE L'AVAIT PAS VU : les paliers croissent
# bien, c'est leur IMAGE par la capacite qui s'ecrase. Le test mesurait la
# monotonie (<=), pas l'injectivite.
#
# CE QUE LA PART CORRIGE. Les six paliers sont des fractions du MEME plafond,
# donc ils restent distincts tant que le plafond vaut au moins six positions,
# et se degradent ensuite au rythme de l'arithmetique — jamais par le haut.
#
#     ATR  4 $   ->  0   2   4   6   8  10      6
#     ATR  8 $   ->  0   1   2   3   4   5      6
#     ATR 15 $   ->  0   1   1   1   2   2      3   (le maximum possible)
#
# LINEAIRE, ET NON GEOMETRIQUE. Une echelle geometrique garderait plus de
# finesse en bas, mais elle perd des paliers des que le plafond est petit :
# a cinq positions, (0, 1/12, 1/6, 1/3, 7/12, 1) rend 0, 1, 1, 1, 2, 5. Le
# defaut qu'on corrige etant precisement l'ecrasement, on choisit l'echelle
# qui le minimise.
#
# LA COMPOSITION EST AUTOMATIQUE, et c'est ce qui permet de supprimer
# `capital_reference`. Tant que le courtier impose son lot minimum — jusque
# vers 6 700 $ sur l'or — `risque_une` ne depend pas du capital, donc le
# plafond croit lineairement avec l'equite et une part de ce plafond aussi.
# Au-dela, `_compute_dynamic_size` fait grossir la TAILLE de chaque position
# avec le capital : `risque_une` devient proportionnel a l'equite, le plafond
# se stabilise en NOMBRE, et c'est correct — le compte compose par la taille.
# L'ancienne echelle multipliait en plus le compte de positions par
# `equite / capital_reference` : une croissance en equite au CARRE, que seul
# le plafond de survie empechait de se voir.
# `BUDGETS_PART` A ETE RETIREE LE 2026-09-21.
#
# Elle donnait six paliers de mise, choisis par le rang de la conviction.
# CE QU'ELLE VALAIT, MESURE trade par trade sur 7 363 trades du fold 1 :
# la taille posee valait 1.0000 unite, p5 0.9999, p95 1.0001. Les six
# paliers se projetaient sur UNE seule taille.
#
# DEUX RAISONS, ET AUCUNE N'EST LE BUDGET. D'abord l'environnement est
# passe a UNE position a la fois : la regle de capacite demande desormais
# la capacite a plein (`_b = 1.0`), donc le palier n'entrait plus dans le
# calcul. Ensuite, et c'est le fond, la taille voulue par le risque vaut
# `capital x risk_per_trade / distance_de_stop` = 0.003 lot a 1 000 $,
# soit TROIS FOIS MOINS que le lot minimum du courtier. Le plancher
# s'impose, et aucun mecanisme de dimensionnement ne peut rien y changer
# tant que le capital n'atteint pas 3 377 $.
#
# LA CAPITALISATION SURVIT : `size = capital x risk_per_trade / sl_dist`
# ne depend pas du budget. Quand le compte depassera le plancher, la
# taille suivra l'equite comme avant.

# LE PLAFOND QU'AUCUN CHOIX NE FRANCHIT, exprime lui en fraction d'equite.
#
# Sans lui, 8 positions a ATR 15 engageraient 8 x 150 $ = 120 % d'un compte de
# 1 000 $ : la ruine, choisie par une tete de reseau a l'epoch 1. Ce plafond
# vaut ce que valait l'ancien palier le plus haut — 40 % — donc il ne
# restreint rien de ce qui etait deja permis ; il empeche seulement l'echelle
# en positions de sortir de l'enveloppe qui avait ete mesuree.
#
# EFFET VISIBLE : en marche calme (ATR 4) les huit paliers sont distincts ; en
# marche agite (ATR 15) ils s'ecrasent sur deux positions. C'est voulu — a
# 1 000 $ on ne TIENT pas huit positions sur un or qui bouge de 15 $ — et cela
# se lit dans les taux de refus du journal.
# 0.40 -> 0.20. LE CREUX DE VALIDATION ETAIT INVIVABLE.
#
# MESURE SUR exec69, creux maximal par epoch de validation :
#
#     fold 1    59.6 % en moyenne, jusqu'a 80.1 %
#     fold 2    71.6 % en moyenne, jusqu'a 86.5 %
#
# Sur un compte de 1 000 $, 72 % de creux veut dire etre descendu a 280 $
# avant de remonter. Personne ne tient une telle trajectoire, et un modele
# qu'on ne peut pas suivre ne rapporte rien.
#
# CE QUE `baisse` NE VOYAIT PAS. Le critere de retenue soustrait une
# demi-deviation PAR OCCASION — 0.74 R, stable — qui ne regarde jamais le
# CHEMIN. Elle ignore que les pertes s'enchainent. Le critere a donc retenu
# sans broncher un modele a 80 % de creux.
#
# POURQUOI DIVISER PAR DEUX, ET PAS AUTRE CHOSE. Le creux d'un portefeuille
# est a peu pres proportionnel a l'exposition tant que la regle de sortie ne
# change pas : 72 % x 0.5 ~ 36 %, et la progression naturelle du tri en avait
# deja retire 12 points au fold 1. On vise donc 25-35 %, ce qui est tenable.
#
# CE QUE CELA COUTE : moins de positions simultanees, donc moins de
# rendement en valeur absolue. Les grandeurs qui jugent le MODELE — `rho`,
# `sommet`, `PF`, `net` — sont invariantes d'echelle et ne bougeront pas.
PLAFOND_RISQUE_EQUITE = 0.20

# CINQ COLONNES D'ETAT DEPUIS LE 2026-09-21, et c'etait six.
#
# La sixieme portait `budget_courant`, la part du plafond que le modele
# s'etait donnee. Elle part avec tout l'appareil de budget.
#
# UNE CONSTANTE NE PORTE PAS D'INFORMATION, elle agit comme un biais. Le
# budget ne commandait plus rien depuis que l'environnement est passe a UNE
# position — la regle de capacite demande la capacite a plein — donc la
# colonne ne variait plus qu'au gre d'un chiffre que personne ne lisait.
#
# LES CINQ QUI RESTENT : sens, gain latent en ATR, age normalise, capacite
# restante, distance au garde-fou de creux. Les trois premieres sont celles
# que `cibles_m1.echantillon_cloture` remplit ; les deux dernieres decrivent
# le compte et valent zero dans cet echantillon.
# CE QUE LA TETE DE PROFIT REGARDE, ET RIEN D'AUTRE.
#
# Elle ne passe PAS par le tronc, et c'est une mesure qui l'impose. Le
# 2026-09-22, la meme cible apprise sur les memes occasions :
#
#     274 colonnes de marche + l'etat   IC +0.1401   plancher 0.2806   1 arbre
#     l'etat SEUL                       IC +0.1911   plancher 0.0685 100 arbres
#
# Les colonnes de marche ne diluent pas le signal, elles l'EFFACENT : le
# modele tombe a un arbre et passe sous son propre plancher. Brancher
# cette tete sur le tronc partage reviendrait a la nourrir de bruit.
#
# AVEC LES SEULES QUATRE COLONNES QUE L'ETAT PORTE, le signal tient — et
# c'est une INTERACTION, aucune ne le porte seule :
#
#     les 4 ensemble      IC +0.1523   plancher 0.0581    47 arbres
#     latent + age        IC +0.1038   plancher 0.0554   168 arbres
#     latent seul         IC +0.0436   plancher 0.0459   au plancher
#     age seul            IC -0.0007   plancher 0.0178   au plancher
#
# C'est exactement ce qui justifie une tete APPRISE plutot qu'un seuil.
IDX_PROFIT_POS = (1, 2)      # latent en ATR d'entree, age normalise
COLS_PROFIT_MARCHE = ("creux_rang", "flux_rang")
N_PROFIT_FEATURES = len(IDX_PROFIT_POS) + len(COLS_PROFIT_MARCHE)

# LE MODELE DE SORTIE LIT UNE COLONNE DE PLUS : LE SENS DE LA POSITION.
#
# TANT QUE SEULS LES LONGS OUVRAIENT, le sens etait une constante et le
# modele de sortie n'en avait pas besoin. Avec les shorts, le meme etat de
# marche ne veut plus dire la meme chose selon le cote tenu. `creux_rang`
# bas, c'est un prix tombe sous sa moyenne : pour un long, la situation
# qui l'a mis en perte ; pour un short, celle qui l'a mis en gain — et
# qui s'epuise peut-etre. `flux_rang` compte le volume AGRESSIF A
# L'ACHAT : il pousse dans le sens du long et contre le short.
#
# ON NE PEUT PAS SIMPLEMENT RETOURNER CES COLONNES POUR UN SHORT. Le rang
# du creux se retournerait, celui du flux non : son miroir serait le
# volume agressif a la VENTE, que l'etat ne porte pas. Le sens est donc
# donne tel quel, et le modele apprend lui-meme ce qu'il change.
#
# IL VIENT DU BLOC POSITION, colonne 0 — `float(self.position)`, +1 ou
# -1 — deja dans l'observation. Aucune colonne nouvelle a fabriquer.
#
# `tete_profit` GARDE SES QUATRE COLONNES : c'est un autre organe, et sa
# mesure (IC +0.1523) a ete faite sans le sens.
IDX_SENS_POS = 0
N_SORTIE_FEATURES = N_PROFIT_FEATURES + 1
COL_SENS_SORTIE = N_PROFIT_FEATURES     # le sens est la DERNIERE colonne

# DEUX ACTIONS DE SORTIE, DANS CET ORDRE EXACT : tenir, puis fermer.
# L'ordre est lu par `decide_sortie` et par la boucle de collecte ; une
# inversion ne leverait aucune erreur, elle ferait seulement fermer quand
# il faut tenir.
TENIR, FERMER = 0, 1

# LES TROIS ACTIONS D'ENTREE, dans l'ordre des actions de l'environnement :
# 0 acheter, 1 vendre, 2 attendre. Le logit d'ATTENDRE est fixe a zero ; les
# tetes d'achat et de vente disent chacune combien elles preferent ouvrir a
# attendre. Voir `SAINTPolicySingleHead.entree`.
ACHETER, VENDRE, ATTENDRE = 0, 1, 2
N_ACTIONS_ENTREE = 3
N_ACTIONS_SORTIE = 2

# OU LIRE LES DEUX COLONNES DE LA TETE DE PROFIT, dans le bloc de features.
#
# ELLES SONT LUES NORMALISEES, DES DEUX COTES, ET C'EST LE POINT. La
# fabrique d'echantillons pourrait rendre le rang BRUT entre 0 et 1, tandis
# que l'environnement montre le bloc normalise par les statistiques figees
# du fold 1. Apprendre sur l'un et decider sur l'autre ferait la faute que
# `cibles_m1` documente le plus souvent : deux ecritures de la meme regle,
# qui divergent sans jamais lever d'erreur.
#
# Les deux cotes lisent donc AU MEME ENDROIT — le bloc de features de
# l'etat — et ces indices sont le seul endroit qui sache ou.
IDX_PROFIT_MARCHE = tuple(FEATURE_COLS.index(_c) for _c in COLS_PROFIT_MARCHE)


N_POS_FEATURES = 5
OBS_N_FEATURES = N_BASE_FEATURES + N_POS_FEATURES


# ============================================================
# SEUIL DE CONVICTION CALIBRÉ
# ============================================================

_CALIB_CACHE: Dict[str, float] = {}


def load_calib_threshold(pth: str, repli: Optional[float] = None) -> float:
    """Barre de conviction écrite par le training à côté du checkpoint `pth`.

    Le couple (poids, seuil) est indissociable : le checkpoint a été SÉLECTIONNÉ
    sous ce seuil précis, qui réalise une sélectivité donnée (« trader les q %
    d'instants les plus favorables »). Appliquer une autre valeur ferait tourner
    une politique qui n'a jamais été évaluée.

    La RÈGLE associée, la même partout : retenir le meilleur côté (BUY vs SELL)
    puis exiger que sa probabilité franchisse cette barre. Jamais d'argmax sur
    les trois actions — une stratégie qui ne trade que 5 % du temps a p(HOLD)
    majoritaire presque partout, et l'argmax renverrait HOLD en permanence.

    `repli` sert aux checkpoints antérieurs à cette calibration ; sans repli, on
    lève plutôt que de trader sur une barre inventée.
    """
    b, sll = load_calib_thresholds(pth, repli)
    return max(b, sll)


def load_calib_thresholds(pth: str, repli: Optional[float] = None):
    """Les DEUX barres (BUY, SELL) écrites par le training.

    Une barre unique sur max(p_BUY, p_SELL) dégénère : un écart de l'ordre du
    millième entre les côtés — du bruit — verrouille la sélection sur un seul,
    et le côté verrouillé change d'une epoch à l'autre (mesuré : epoch 4 tout
    LONG, epoch 7 tout SHORT, zéro trade de l'autre côté). Chaque côté est donc
    calibré sur SA propre distribution.

    Règle de décision, identique partout :
        les côtés dont p franchit LEUR barre sont candidats ; à égalité on
        retient celui dont l'écart à sa barre est le plus net ; sinon HOLD.
    """
    if pth in _CALIB_CACHE:
        return _CALIB_CACHE[pth]

    import json

    chemin = pth.replace(".pth", "_calib.json")
    if not os.path.exists(chemin):
        if repli is not None:
            print(f"[CALIB] {os.path.basename(chemin)} absent — repli sur "
                  f"{repli:.3f} (checkpoint antérieur à la calibration).")
            _CALIB_CACHE[pth] = (float(repli), float(repli))
            return _CALIB_CACHE[pth]
        raise FileNotFoundError(
            f"Seuil calibré introuvable : {chemin}\n"
            f"Il est produit par training.py en même temps que {pth}. "
            f"Sans lui, la barre serait arbitraire et le modèle tournerait hors "
            f"des conditions où il a été mesuré."
        )
    with open(chemin, encoding="utf-8") as f:
        d = json.load(f)
    # Les checkpoints antérieurs n'ont qu'une barre commune.
    b = float(d.get("calib_thr_buy", d["calib_thr"]))
    sl = float(d.get("calib_thr_sell", d["calib_thr"]))
    _CALIB_CACHE[pth] = (b, sl)
    print(f"[CALIB] {os.path.basename(pth)} : barres BUY {b:.4f} / SELL {sl:.4f} "
          f"(sélectivité {100 * float(d.get('selectivite', 0)):.1f} %, "
          f"epoch {d.get('epoch', '?')})")
    return b, sl


COTES_PERMISES = {"long": (True, False), "short": (False, True),
                  "both": (True, True), "duel": (True, True)}


def cotes_permises(side: str):
    """(achat permis, vente permise) — MEME table que le masque d'actions."""
    if side not in COTES_PERMISES:
        raise ValueError(f"cote inconnu : {side!r}")
    return COTES_PERMISES[side]


def decide_avec_barres(p_buy: float, p_sell: float, barres,
                       side: str = "both") -> int:
    """0 = BUY, 1 = SELL, 2 = HOLD. SOURCE UNIQUE de la règle de décision.

    `side` INTERDIT un cote, exactement comme `build_mask_from_pos_scalar`
    l'interdit dans les logits. Il fallait les deux, et c'est ce qui manquait.

    CE QUE COUTAIT L'OUBLI, mesure sur or_exec02 (run long-only, trois folds).
    La validation et le test remplacent les probabilites de la politique par
    la sortie de la TETE AUXILIAIRE quand `tri_par_tete_aux` est vrai :

        probs_np = sigmoid(policy.rendement(st))

    Cette substitution ECRASE le tableau issu des logits masques, donc le
    masque de cote avec. La tete auxiliaire predit les deux sens sans rien
    savoir du cote autorise, et `decide` retenait "le meilleur des deux" —
    des ventes, dans un run qui n'en avait jamais entraine une seule.

        fold   LONG        SHORT       part des trades vendus
        wf1    +87.9 $     -286.9 $    68 %
        wf2    +421.1 $    -108.5 $    31 %
        wf3    +517.1 $    -435.8 $    64 %

    L'entrainement affichait S(0W/0L) +0.00 $ a chaque epoch — le rollout,
    lui, respectait le masque. Les ventes ont emporte 81 % du gain des
    achats, et c'est sur ce net que le "meilleur modele" a ete choisi.

    ON NE CORRIGE PAS DANS LES SCORES mais ici, dans la regle : mettre le
    cote interdit a -inf obligerait sa barre calibree a valoir +inf pour que
    rien ne passe, soit deux choses a garder d'accord. Une seule suffit.
    """
    if isinstance(barres, EntryDecisionPolicy):
        return barres.decide(p_buy, p_sell)
    permis_b, permis_s = cotes_permises(side)
    ok_b = permis_b and p_buy >= barres[0]
    ok_s = permis_s and p_sell >= barres[1]
    if ok_b and ok_s:
        return 0 if (p_buy - barres[0]) >= (p_sell - barres[1]) else 1
    if ok_b:
        return 0
    if ok_s:
        return 1
    return 2


SOURCE_EXT_NOM = "Binance BTCUSDT"
SOURCE_EXT_FICHIER = "binance_features_BTCUSD.pkl"

# Colonne servant de garde-fou d'alignement, et seuil.
#
# Mesure sur 171 385 minutes (fev-mai 2026), correlation aux rendements M1 du
# broker selon le decalage applique :
#   taker_ratio      +0 : 0.2074   +1 : 0.1540   +2 : 0.1118   +5 : 0.0025   +60 : -0.0015
#   ls_ratio_top     +0 : -0.0030  ... plat partout
#   oi_change, funding_rate, funding_cum24, ls_ratio_retail : plats partout
#
# `taker_ratio` est donc la SEULE colonne capable de detecter un mauvais
# alignement. Sa decroissance lente sur +1/+2 n'est pas un defaut : ces metriques
# sont publiees toutes les 5 minutes puis reportees a la minute, donc un
# decalage d'une ou deux minutes ne change quasiment rien — et n'est pas nocif
# pour la meme raison. Le seuil vise la classe de decalage qui compte : une
# erreur d'heure d'ete vaut 60 minutes, ou la correlation tombe a zero.
COL_GARDE_EXT = "taker_ratio"
CORR_GARDE_MIN = 0.10


def charge_source_externe(date_from=None, date_to=None, chemin=None):
    """Charge les features de la source externe, indexees par l'heure BROKER.

    Le fichier est produit par build_binance_features.py, qui detecte le
    decalage horaire du broker EMPIRIQUEMENT en correlant les rendements M1 —
    un offset fixe decalerait un bon tiers de l'historique d'une heure entiere
    sans la moindre erreur visible, ce broker suivant le calendrier DST
    americain.

    Renvoie None si le fichier est absent : les colonnes seront alors NaN et le
    dropna videra le jeu, ce qui est bruyant et donc preferable a un silence.
    """
    import os
    chemin = chemin or SOURCE_EXT_FICHIER
    if not os.path.exists(chemin):
        return None
    df = pd.read_pickle(chemin)
    if date_from is not None:
        df = df[df.index >= pd.Timestamp(date_from)]
    if date_to is not None:
        df = df[df.index <= pd.Timestamp(date_to)]
    manquantes = [c for c in FEATURE_COLS_EXT if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"{SOURCE_EXT_FICHIER} ne contient pas {manquantes}. "
            f"Relancer build_binance_features.py."
        )
    return df


def merge_m1_h1(rates_m1, rates_h1,
                feats_ext=None,
                point: float = 1.0,
                corr_min: float = CORR_GARDE_MIN,
                dropna_subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Construit le DataFrame M1 enrichi du H1 et de la source externe.

    Le `shift(1)` sur le bloc H1 est le point critique : merge_asof(backward)
    selectionne le bar H1 qui CONTIENT l'instant M1, donc un bar encore en
    formation. En historique ses valeurs sont deja finalisees, ce qui injectait
    jusqu'a 59 minutes de futur (returns_h1 livrait le rendement complet de
    l'heure en cours) ; en live le meme bar est partiel. Decaler d'un bar donne
    le dernier H1 reellement cloture, identique en training, backtest et live.
    """
    df_m1 = pd.DataFrame(rates_m1)
    df_m1["time"] = pd.to_datetime(df_m1["time"], unit="s")
    df_m1.set_index("time", inplace=True)
    # `spread` est en POINTS entiers dans les bougies MT5 ; on le garde brut ici
    # et on le convertit en prix au moment de calculer spread_rel.
    spread_pts = df_m1["spread"].astype(float) if "spread" in df_m1 else None
    df_m1 = df_m1[["open", "high", "low", "close", "tick_volume"]]
    df_m1 = add_indicators(df_m1)
    if spread_pts is not None:
        df_m1["_spread_pts"] = spread_pts

    df_h1 = pd.DataFrame(rates_h1)
    df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
    df_h1.set_index("time", inplace=True)
    df_h1 = df_h1[["open", "high", "low", "close", "tick_volume"]]
    df_h1 = add_indicators(df_h1)
    df_h1 = df_h1.add_suffix("_h1")
    df_h1 = df_h1.shift(1)

    merged = pd.merge_asof(
        df_m1.reset_index().sort_values("time"),
        df_h1.reset_index().rename(columns={"time_h1": "time"}).sort_values("time"),
        on="time",
        direction="backward",
    )

    # Ecart du dernier H1 cloture au prix M1 courant. Se calcule forcement APRES
    # la fusion puisqu'il croise les deux timeframes.
    merged["close_h1_dev"] = merged["close_h1"] / (merged["close"] + 1e-8) - 1.0

    # ---------- SOURCE EXTERNE ----------
    # Jointure EXACTE a la minute. Le fichier est deja exprime en heure broker
    # (build_binance_features.py detecte le decalage en correlant les rendements
    # M1), donc pas de merge_asof : il masquerait un defaut d'alignement en
    # collant la valeur la plus proche.
    _cols_ext = ([c for c in FEATURE_COLS_EXT if c in feats_ext.columns]
                 if feats_ext is not None else [])
    # AUCUNE COLONNE RETENUE : IL N'Y A RIEN A FUSIONNER NI A GARDER.
    #
    # `FEATURE_COLS_EXT` est VIDE dans la configuration or — les features
    # Binance sont des features BTC, on ne les utilise pas ici. Le bloc
    # s'executait quand meme des que le fichier existait : il fusionnait zero
    # colonne, puis le garde-fou lisait `COL_GARDE_EXT` dans un dataframe qui
    # ne l'avait pas, et levait `KeyError: 'taker_ratio'`.
    #
    # CELA NE S'ETAIT JAMAIS VU parce que le cache M5 court-circuite cette
    # fusion. Le jour ou il a expire — 48 h de retard — le rechargement est
    # tombe ici, et aucun run ne pouvait plus demarrer. Une panne qui attend
    # l'expiration d'un cache pour se declarer est la pire espece : elle ne
    # correle avec aucun changement de code.
    #
    # LE GARDE-FOU N'EST PAS AFFAIBLI : il verifie l'alignement des colonnes
    # QU'ON UTILISE. Quand on n'en utilise aucune, il n'y a pas d'alignement a
    # verifier. Des qu'une seule revient dans `FEATURE_COLS_EXT`, il reprend.
    if feats_ext is not None and len(feats_ext) > 0 and _cols_ext:
        fx = feats_ext[_cols_ext]
        merged = merged.merge(fx, left_on="time", right_index=True, how="left")

        chevauche = merged["time"].between(fx.index.min(), fx.index.max())
        if chevauche.any():
            # GARDE-FOU 1 — jointure totalement ratee. Un decalage d'horodatage
            # laisserait toutes les colonnes a NaN puis le dropna viderait le
            # dataframe SANS erreur. C'est le piege qui s'est deja referme sur
            # les ticks.
            # LA COLONNE TEMOIN DOIT ETRE PARMI CELLES RETENUES. Si elle
            # ne l'est pas, on ne peut pas mesurer l'alignement : on le dit
            # au lieu de lever un `KeyError` a trois niveaux de pandas.
            if COL_GARDE_EXT not in merged.columns:
                raise ValueError(
                    f"la colonne temoin d'alignement `{COL_GARDE_EXT}` n'est "
                    f"pas dans les colonnes retenues {_cols_ext} : "
                    f"l'alignement de {SOURCE_EXT_NOM} ne peut pas etre "
                    f"verifie. Ajouter la colonne a FEATURE_COLS_EXT, ou "
                    f"changer COL_GARDE_EXT pour une colonne utilisee.")
            couv = merged.loc[chevauche, COL_GARDE_EXT].notna().mean()
            if couv < 0.5:
                raise ValueError(
                    f"Jointure {SOURCE_EXT_NOM} quasi vide : seules "
                    f"{100*couv:.1f}% des bougies de la plage couverte ont des "
                    f"donnees. Verifier l'alignement des horodatages "
                    f"(broker : {merged['time'].iloc[0]}, "
                    f"externe : {fx.index[0]})."
                )

            # GARDE-FOU 2 — indispensable, et c'est celui qui manquait la
            # premiere fois. Le precedent ne detecte qu'une jointure TOTALEMENT
            # ratee. Un decalage d'un nombre entier de minutes s'aligne
            # parfaitement sur la grille : la couverture reste a 100 % et le
            # modele recoit les bonnes colonnes sur les MAUVAISES minutes.
            # Panne parfaitement silencieuse.
            #
            # On verifie donc le SENS des donnees. Mesure sur 171 385 minutes :
            # taker_ratio correle a +0.207 aux rendements contemporains, +0.003
            # a 5 minutes de decalage, -0.002 a 60. Un decalage d'heure d'ete,
            # qui est la panne realiste, tombe donc tres au-dessous du seuil.
            ok = merged.loc[chevauche, ["returns", COL_GARDE_EXT]].dropna()
            if len(ok) > 1000:
                c = float(np.corrcoef(ok["returns"], ok[COL_GARDE_EXT])[0, 1])
                if not np.isfinite(c) or c < corr_min:
                    raise ValueError(
                        f"Correlation {COL_GARDE_EXT} / rendements de {c:+.3f}, "
                        f"sous le seuil de {corr_min:.2f} : les deux series sont "
                        f"probablement DECALEES dans le temps. La jointure a "
                        f"reussi mais sur les mauvaises minutes — verifier le "
                        f"fuseau du broker et relancer "
                        f"build_binance_features.py --offsets."
                    )
    else:
        for c in FEATURE_COLS_EXT:
            merged[c] = np.nan

    # ---------- LIQUIDITE ----------
    if "_spread_pts" in merged.columns:
        # `spread` des bougies MT5 est en POINTS entiers ; l'ATR est en PRIX.
        # Sans la conversion par `point` la feature serait 100x trop grande sur
        # l'or — inoffensif apres z-scoring, mais le live calcule son spread en
        # unites de PRIX (ask - bid) et les deux divergeraient silencieusement.
        # Rapporte a sa propre normale d'une journee : la question posee
        # devient « le spread est-il anormalement large EN CE MOMENT ? »,
        # stationnaire par construction. Mesure sur l'or 2018-2026, la
        # version brute derivait de 1.74 ecart-type sur huit ans — elle
        # encodait l'ANNEE, pas l'etat du marche.
        # (ancien commentaire) le spread rapporte a
        # l'ATR passe de 0.91 a 0.11, soit 1.74 ecart-type de derive. Les
        # brokers ont resserre leurs cotations au fil des ans. Brute, cette
        # colonne encode donc l'ANNEE : entrainee sur 2018-2023 (valeurs ~0.5 a
        # 0.9) puis appliquee a 2026 (~0.11), elle saturerait a -0.76 sigma en
        # permanence et deviendrait muette exactement la ou l'on trade.
        #
        # On la rapporte donc a sa propre normale d'une journee. La question
        # posee devient « le spread est-il anormalement large EN CE MOMENT ? »,
        # qui est celle qui portait le signal (+0.0025 d'apport marginal), et
        # elle est stationnaire par construction.
        _sr = (merged["_spread_pts"] * point) / (merged["atr_14"] + 1e-9)
        merged["spread_rel"] = _sr / (
            _sr.rolling(1440, min_periods=120).mean() + 1e-12)
        merged = merged.drop(columns=["_spread_pts"])
    else:
        merged["spread_rel"] = np.nan

    # ---------- TEMPS ----------
    # Phase du jour en sinus/cosinus : 23 h et 0 h doivent etre voisins, ce
    # qu'un numero d'heure ne dit pas.
    h = merged["time"].dt.hour + merged["time"].dt.minute / 60.0
    merged["heure_sin"] = np.sin(2 * np.pi * h / 24.0)
    merged["heure_cos"] = np.cos(2 * np.pi * h / 24.0)

    if dropna_subset is None:
        merged = merged.dropna()
    else:
        merged = merged.dropna(subset=dropna_subset)
    return merged.reset_index(drop=True)



# ============================================================
# NORMALISATION
# ============================================================

def safe_normalize(X, stats, clip_sigma: float = CLIP_SIGMA):
    z = (X - stats["mean"]) / (stats["std"] + 1e-8)
    return np.clip(z, -clip_sigma, clip_sigma)


def load_model_norm_stats(checkpoint_path: str) -> Dict[str, np.ndarray]:
    """Scaler du checkpoint; compatibilité explicite avec les anciens modèles."""
    import warnings
    from pathlib import Path
    checkpoint_path = Path(checkpoint_path)
    path = checkpoint_path.with_name(checkpoint_path.stem + '_norm.npz')
    if not path.exists():
        warnings.warn(f'{checkpoint_path}: ancien modèle sans scaler associé; statistiques historiques utilisées.', RuntimeWarning)
        path = checkpoint_path.parent / NORM_STATS_PATH
    return load_norm_stats(str(path))


def load_shared_model_norm_stats(paths):
    """Refuse une observation commune à des modèles normalisés différemment."""
    stats = [load_model_norm_stats(p) for p in paths]
    if not stats:
        raise ValueError('Aucun modèle sélectionné')
    if any(not all(np.array_equal(s[k], stats[0][k]) for k in ('mean', 'std')) for s in stats[1:]):
        raise ValueError('Scalers différents: utiliser le mode multi_agent (observation par modèle).')
    return stats[0]


def load_norm_stats(path: str = NORM_STATS_PATH) -> Dict[str, np.ndarray]:
    """Charge les stats Z-score en VALIDANT qu'elles correspondent aux features.

    Le fichier embarque les noms des colonnes sur lesquelles il a été calculé.
    Sans cette vérification, changer la définition d'une feature à cardinalité
    constante (remplacer un prix brut par un ratio, par exemple) laissait le
    live et les backtests normaliser avec des mean/std sans aucun rapport —
    en silence, et sans que les shapes ne signalent quoi que ce soit.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stats de normalisation introuvables : {path}")
    data = np.load(path, allow_pickle=True)

    mean = data["mean"]
    if mean.shape[0] != N_BASE_FEATURES:
        raise ValueError(
            f"{path} contient {mean.shape[0]} features, le code en attend "
            f"{N_BASE_FEATURES}. Relancer training.py pour les recalculer."
        )

    if "features" not in data:
        raise ValueError(
            f"{path} est au format antérieur (sans noms de features) et ne peut "
            f"pas être validé. Relancer training.py pour le régénérer."
        )

    cached = [str(x) for x in data["features"]]
    if cached != list(FEATURE_COLS):
        diff = sorted(set(cached) ^ set(FEATURE_COLS))
        raise ValueError(
            f"{path} a été calculé sur d'autres features ({', '.join(diff)}). "
            f"Relancer training.py pour le régénérer."
        )

    std = data["std"]
    if (mean.shape != (N_BASE_FEATURES,) or std.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(std).all()
            or (std < 0).any()):
        raise ValueError(f"Statistiques invalides dans {path}")
    return {"mean": mean, "std": std}


def build_obs(df: pd.DataFrame,
              stats: Dict[str, np.ndarray],
              lookback: int,
              pos: int,
              unrealized_atr: float,
              bars_held_norm: float,
              last_risk_scale: float) -> Optional[np.ndarray]:
    """Assemble l'observation (lookback, OBS_N_FEATURES).

    Le bloc position est répété sur tout le lookback, exactement comme dans
    l'environnement d'entraînement.
    """
    if len(df) < lookback + 1:
        return None

    X = df[FEATURE_COLS].values.astype(np.float32)
    base = safe_normalize(X, stats)[-lookback:]

    extra_vec = np.array(
        [float(pos), float(unrealized_atr), float(bars_held_norm), float(last_risk_scale)],
        dtype=np.float32,
    )
    extra_block = np.repeat(extra_vec[None, :], lookback, axis=0)
    return np.concatenate([base, extra_block], axis=-1).astype(np.float32)


# ============================================================
# SAINT v2 — SINGLE-HEAD (ACTOR + CRITIC)
# ============================================================

# ============================================================
#  ARCHITECTURE — SAINT a double axe, version complete
# ============================================================
#
# Le bloc avait ete reduit a (attention temps, attention features, un FFN) pour
# tenir dans le budget thermique. On revient a la structure COMPLETE — une
# attention ET son FFN par axe — avec les composants qui font l'etat de l'art
# depuis SAINT (2021). Chacun est la pour une raison mesurable, pas par mode.
#
#   Pre-norm + RMSNorm         Xiong et al. 2020 ; Zhang & Sennrich 2019.
#                              La pre-norm rend les blocs profonds entrainables
#                              sans warmup ; RMSNorm retire le recentrage, qui
#                              ne sert a rien apres une projection lineaire.
#
#   QK-Norm                    Henry et al. 2020 ; Dehghani et al. 2023 (ViT-22B).
#                              Normalise Q et K AVANT le produit scalaire. Sans
#                              elle, les logits d'attention grandissent sans
#                              borne et l'entrainement diverge tard, d'un coup.
#                              Pertinent ici : la perte du critique oscillait
#                              entre 24 et 47 d'une epoch a l'autre.
#
#   SDPA / FlashAttention      Dao et al. 2022. torch.nn.functional.
#                              scaled_dot_product_attention dispatche vers le
#                              noyau fusionne : attention EXACTE, mais sans
#                              materialiser la matrice T x T. C'est ce qui paie
#                              le surcout de la structure complete.
#
#   RoPE sur l'axe TEMPS       Su et al. 2021. Position RELATIVE par rotation.
#                              Sur l'axe des features, on garde un plongement
#                              appris : les colonnes n'ont pas d'ordre naturel,
#                              leur imposer une geometrie de position serait
#                              une contrainte fausse.
#
#   SwiGLU                     Shazeer 2020, "GLU Variants Improve Transformer".
#                              Remplace le gating par sigmoide. Largeur interne
#                              ramenee a 2/3 pour garder le meme compte de
#                              parametres qu'un FFN dense equivalent.
#
#   LayerScale                 Touvron et al. 2021 (CaiT). Un gain par canal
#                              sur chaque branche residuelle, initialise a 1e-4 :
#                              le reseau demarre proche de l'identite et ouvre
#                              les branches a mesure qu'elles servent. C'est ce
#                              qui permet d'empiler des blocs sans instabilite.
#
#   Plongement numerique       Gorishniy et al. 2022, "On Embeddings for
#   periodique + lineaire      Numerical Features". Un scalaire projete
#                              lineairement occupe une seule direction ; les
#                              activations periodiques lui donnent une
#                              representation a haute frequence ou l'attention
#                              peut distinguer des valeurs proches. C'est l'un
#                              des gains les mieux repliques sur donnees
#                              tabulaires. Variante PR (sans les bins), qui ne
#                              demande aucune statistique externe au modele.
#
#   Jeton CLS                  Gorishniy et al. 2021 (FT-Transformer). Un jeton
#                              appris sur l'axe des features, que l'attention
#                              remplit. Remplace la moyenne, qui traite toutes
#                              les colonnes a poids egal.
#
# CE QUI A ETE ECARTE, ET POURQUOI — l'INTERSAMPLE ATTENTION de SAINT.
#
# C'est l'innovation qui donne son nom au papier : chaque ligne du LOT regarde
# les autres lignes du lot. Elle n'est pas utilisable ici, et pas pour une
# raison de cout.
#
# En production, l'agent decide sur UNE observation a la fois : le lot vaut 1.
# Un softmax sur un seul element rend 1, donc l'attention inter-echantillons
# degenere en une simple projection de la valeur. Les poids appris sous un lot
# de 128 se comporteraient autrement en live — un ecart entrainement/production
# silencieux, exactement la classe de defaut que ce projet passe son temps a
# eliminer. Sur des series temporelles, elle ferait en plus circuler de
# l'information entre des dates differentes du meme lot.
#
# Les deux axes conserves — TEMPS et FEATURES — sont definis pour un echantillon
# unique et se comportent donc identiquement a l'entrainement et en live.


class RMSNorm(nn.Module):
    """Normalisation par la norme quadratique, sans recentrage ni biais."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * self.weight.float()).to(dtype)


def _rope_tables(longueur: int, dim_tete: int, device, dtype, base: float = 10_000.0):
    """Cosinus/sinus de RoPE. Recalcules a la volee : negligeable devant
    l'attention, et evite un buffer dont la taille figerait le lookback."""
    moitie = dim_tete // 2
    freqs = 1.0 / (base ** (torch.arange(0, moitie, device=device).float() / moitie))
    angles = torch.outer(torch.arange(longueur, device=device).float(), freqs)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _applique_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """x : (N, tetes, L, dim_tete). Rotation par paires de canaux."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def _verifie_dim_tete(d: int, heads: int) -> int:
    """La dimension de tete doit etre un MULTIPLE DE 8. Verifie tot et fort.

    Sous PyTorch 2.5.1 / CUDA 12.4 / SM 8.6, une dimension de tete non
    multiple de 8 ne provoque pas de repli sur le noyau generique : le
    repartiteur de scaled_dot_product_attention lance un noyau fautif et la
    carte remonte "an illegal memory access was encountered". L'erreur est
    ASYNCHRONE, donc elle ressort n'importe ou plus tard — dans notre cas au
    calcul de la norme du gradient, six epochs apres le debut du run.

    Mesure, un processus neuf par cas :
        head_dim  8 OK | 16 OK | 32 OK | 40 OK
        head_dim 10 ECHEC | 20 ECHEC

    d_model=80 avec 4 tetes donne 20. Avec 5 tetes, 16. La largeur du modele
    et le nombre de parametres sont identiques — seul le decoupage change.

    L'ancienne architecture n'etait pas touchee : nn.MultiheadAttention ne
    passe pas par ce chemin.
    """
    if d % heads != 0:
        raise ValueError(f"d_model={d} n'est pas divisible par heads={heads}")
    dim_tete = d // heads
    if dim_tete % 8 != 0:
        raise ValueError(
            f"dimension de tete {dim_tete} (d_model={d} / heads={heads}) : "
            f"elle DOIT etre un multiple de 8, sinon scaled_dot_product_"
            f"attention lance un noyau fautif et la carte tombe en acces "
            f"memoire illegal, de facon asynchrone donc indebogable. "
            f"Choisir heads parmi {[h for h in range(1, d + 1) if d % h == 0 and (d // h) % 8 == 0]}."
        )
    return dim_tete


# Taille maximale du lot passe d'un coup a l'attention. La limite dure est
# 65 535 (dimension de grille CUDA) ; on garde la moitie de marge, le decoupage
# ne coutant rien de mesurable.
LOT_ATTENTION_MAX = 32768


class AxialAttention(nn.Module):
    """Attention multi-tetes sur UN axe, en pre-norm.

    Rend la sortie de la BRANCHE, sans residu : c'est le bloc qui applique le
    LayerScale puis l'addition, pour que le gain par canal porte bien sur la
    branche et non sur la somme.
    """

    def __init__(self, d: int, heads: int, dropout: float, rope: bool = False):
        super().__init__()
        self.heads = heads
        self.dim_tete = _verifie_dim_tete(d, heads)
        self.rope = rope
        self.norm = RMSNorm(d)
        # Sans biais : la pre-norm en amont en produit deja l'equivalent.
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.dim_tete)
        self.k_norm = RMSNorm(self.dim_tete)
        self.p_drop = dropout
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, L, D = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(N, L, self.heads, self.dim_tete).transpose(1, 2)
        k = k.view(N, L, self.heads, self.dim_tete).transpose(1, 2)
        v = v.view(N, L, self.heads, self.dim_tete).transpose(1, 2)

        # QK-Norm : la norme des requetes et des cles ne depend plus de
        # l'echelle des activations, seule leur DIRECTION compte.
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope:
            cos, sin = _rope_tables(L, self.dim_tete, x.device, q.dtype)
            q, k = _applique_rope(q, cos, sin), _applique_rope(k, cos, sin)

        # DECOUPAGE DU LOT : UNE LIMITE CUDA, PAS UNE LIMITE DE MEMOIRE.
        #
        # SAINT replie un axe dans la dimension de lot avant d'appeler
        # l'attention : sur l'axe TEMPS le lot vaut B x F, soit le nombre
        # d'episodes multiplie par le nombre de colonnes. Une dimension de
        # grille CUDA plafonne a 65 535, et le noyau SDPA en indexe une par
        # (lot x tetes). A 260 colonnes, cela casse des 251 episodes joues en
        # parallele — avec un "CUDA error: invalid configuration argument" qui
        # ne nomme ni le lot, ni les colonnes, ni l'attention.
        #
        # Le plafond depend donc du PRODUIT de deux reglages qui se choisissent
        # separement et pour des raisons sans rapport : le nombre de features et
        # le nombre d'episodes collectes par epoch. Les lier par un plantage a
        # l'execution ferait payer chaque enrichissement du jeu de colonnes par
        # une reduction de la collecte, exactement quand on cherche a augmenter
        # les deux. On tranche le lot, la limite disparait, et le resultat est
        # identique au bit pres — l'attention ne melange jamais deux elements du
        # lot entre eux, donc la decouper n'en change aucun.
        if N > LOT_ATTENTION_MAX:
            o = torch.cat([
                torch.nn.functional.scaled_dot_product_attention(
                    q[i:i + LOT_ATTENTION_MAX], k[i:i + LOT_ATTENTION_MAX],
                    v[i:i + LOT_ATTENTION_MAX],
                    dropout_p=self.p_drop if self.training else 0.0)
                for i in range(0, N, LOT_ATTENTION_MAX)], dim=0)
        else:
            o = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=self.p_drop if self.training else 0.0)
        o = o.transpose(1, 2).reshape(N, L, D)
        return self.drop(self.proj(o))


class SwiGLU(nn.Module):
    """FFN a porte SiLU, en pre-norm. Rend la branche, sans residu."""

    def __init__(self, d: int, mult: int = 2, dropout: float = 0.05):
        super().__init__()
        # 2/3 : deux projections d'entree au lieu d'une, on compense pour
        # garder le meme nombre de parametres qu'un FFN dense de largeur d*mult.
        inner = max(8, int(round(2 * mult * d / 3 / 8)) * 8)
        self.norm = RMSNorm(d)
        self.w_in = nn.Linear(d, 2 * inner, bias=False)
        self.w_out = nn.Linear(inner, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, porte = self.w_in(self.norm(x)).chunk(2, dim=-1)
        return self.drop(self.w_out(torch.nn.functional.silu(porte) * a))


class SAINTv2Block(nn.Module):
    """Un tour COMPLET : (attention temps + FFN), puis (attention features + FFN).

    C'est la structure du papier : une attention et SON reseau feed-forward par
    axe. La version reduite partageait un seul FFN pour les deux axes, ce qui
    forcait le melange temporel et le melange inter-colonnes a passer par la
    meme transformation non lineaire.

    Chaque branche est mise a l'echelle par un gain appris par canal
    (LayerScale) initialise a 1e-4 : au premier pas le bloc est quasiment
    l'identite, et il ouvre les branches qui servent.
    """

    def __init__(self, d: int, heads: int, dropout: float, mult: int,
                 drop_path: float = 0.0, ls_init: float = 1e-4):
        super().__init__()
        self.attn_temps = AxialAttention(d, heads, dropout, rope=True)
        self.ff_temps = SwiGLU(d, mult, dropout)
        self.attn_feat = AxialAttention(d, heads, dropout, rope=False)
        self.ff_feat = SwiGLU(d, mult, dropout)
        self.gamma = nn.ParameterList(
            [nn.Parameter(ls_init * torch.ones(d)) for _ in range(4)])
        self.drop_path = float(drop_path)

    def _branche(self, h: torch.Tensor, sortie: torch.Tensor,
                 gamma: torch.Tensor) -> torch.Tensor:
        sortie = gamma * sortie
        if self.training and self.drop_path > 0.0:
            garde = 1.0 - self.drop_path
            forme = (h.shape[0],) + (1,) * (h.dim() - 1)
            masque = torch.rand(forme, device=h.device) < garde
            sortie = sortie * masque / garde
        return h + sortie

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F, D = x.shape

        # ---- axe TEMPS : chaque colonne regarde sa propre histoire ----
        h = x.permute(0, 2, 1, 3).reshape(B * F, T, D)
        h = self._branche(h, self.attn_temps(h), self.gamma[0])
        h = self._branche(h, self.ff_temps(h), self.gamma[1])
        x = h.reshape(B, F, T, D).permute(0, 2, 1, 3)

        # ---- axe FEATURES : chaque instant melange ses colonnes ----
        h = x.reshape(B * T, F, D)
        h = self._branche(h, self.attn_feat(h), self.gamma[2])
        h = self._branche(h, self.ff_feat(h), self.gamma[3])
        return h.reshape(B, T, F, D)


class NumericalEmbedding(nn.Module):
    """Plongement periodique + lineaire, un jeu de poids PAR COLONNE.

    Un scalaire projete par une seule matrice n'occupe qu'une direction de
    l'espace latent : deux valeurs proches donnent deux vecteurs proches, et
    l'attention ne peut pas les separer. Les activations periodiques
    sin/cos(2 pi f x), avec des frequences APPRISES par colonne, donnent une
    representation ou un petit ecart devient une grande distance angulaire.

    La valeur brute est concatenee aux composantes periodiques : sans elle, le
    plongement serait invariant par periode et perdrait l'ordre.
    """

    def __init__(self, n_features: int, d: int, n_freq: int = 16,
                 sigma: float = 0.05):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(n_features, n_freq) * sigma)
        self.weight = nn.Parameter(torch.empty(n_features, 2 * n_freq + 1, d))
        self.bias = nn.Parameter(torch.zeros(n_features, d))
        nn.init.normal_(self.weight, std=(2 * n_freq + 1) ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, F) -> (B, T, F, d)
        v = x.unsqueeze(-1)
        angles = 2.0 * math.pi * v * self.freqs
        traits = torch.cat([angles.sin(), angles.cos(), v], dim=-1)
        return torch.einsum("btfk,fkd->btfd", traits, self.weight) + self.bias


class ReferenceMemory(nn.Module):
    """Intersample attention DEPLOYABLE : le lot de reference est FIGE.

    LE PROBLEME DE LA VERSION LITTERALE. Dans SAINT, chaque ligne regarde les
    autres lignes DU LOT COURANT. En production l'agent decide sur une seule
    observation : le lot vaut 1, le softmax sur un element rend 1, et
    l'operation degenere en projection de la valeur. Les poids appris sous un
    lot de 128 se comporteraient autrement en live. Fabriquer un faux lot en
    live ne reglerait rien : il ne ressemblerait pas a celui de l'entrainement.

    LA CORRECTION. Les « autres echantillons » ne sont pas le lot courant mais
    une BANQUE de K observations reelles, tiree une fois pour toutes de la
    fenetre d'ENTRAINEMENT et rangee dans le checkpoint. Le modele compare
    l'instant present a une bibliotheque de situations historiques — ce que
    l'axe temporel ne donne pas, lui qui ne voit que les 25 dernieres bougies.

    Trois proprietes que cette forme conserve, et que la version littérale perd :

      - INDEPENDANCE AU LOT. La banque est la meme pour tous les echantillons,
        donc aucune information ne circule ENTRE les lignes du lot courant. Un
        lot de 8 donne toujours exactement 8 passes de 1 (verifie par
        test_architecture.py). C'est ce qui la rend deployable.

      - IDENTITE ENTRAINEMENT / LIVE. La banque et ses representations encodees
        voyagent avec les poids. Le live recharge exactement ce que le training
        a utilise, sans rien recalculer.

      - ABSENCE DE FUITE. La banque vient de la fenetre de TRAIN uniquement.
        En validation comme en test, le modele consulte des situations
        anterieures a la periode evaluee — c'est de la connaissance apprise,
        au meme titre que les poids, pas de l'information future.

    COUT. Les representations de la banque sont encodees une fois puis mises en
    cache : la requete ne coute qu'une attention croisee depuis UN vecteur par
    echantillon vers K cles. Le cache est rafraichi apres chaque mise a jour
    PPO (les poids du tronc ayant bouge) et fige a la sauvegarde.

    Parente : Gorishniy et al. 2023, "TabR: Tabular Deep Learning Meets Nearest
    Neighbors" — une tete de recherche sur un jeu de references fige y bat les
    transformers tabulaires purs.
    """

    def __init__(self, d_lecture: int, n_ref: int, heads: int, dropout: float,
                 ls_init: float = 1e-4):
        super().__init__()
        self.n_ref = int(n_ref)
        self.d = d_lecture
        self.heads = heads
        self.dim_tete = _verifie_dim_tete(d_lecture, heads)

        self.norm_q = RMSNorm(d_lecture)
        self.norm_kv = RMSNorm(d_lecture)
        self.to_q = nn.Linear(d_lecture, d_lecture, bias=False)
        self.to_kv = nn.Linear(d_lecture, 2 * d_lecture, bias=False)
        self.proj = nn.Linear(d_lecture, d_lecture, bias=False)
        self.q_norm = RMSNorm(self.dim_tete)
        self.k_norm = RMSNorm(self.dim_tete)
        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(ls_init * torch.ones(d_lecture))

        # Buffers : sauvegardes avec le state_dict, donc transportes vers le
        # live sans traitement particulier.
        self.register_buffer("bank_obs", torch.zeros(0), persistent=True)
        self.register_buffer("bank_repr", torch.zeros(0), persistent=True)
        self.register_buffer("bank_pret", torch.zeros(1), persistent=True)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Redimensionne les buffers AVANT de charger.

        Ils naissent vides — la longueur de fenetre n'est pas connue a la
        construction — et load_state_dict refuse une forme differente. Sans ce
        crochet, un checkpoint portant une banque serait rejete, ou pire,
        charge avec une memoire vide qui rendrait le modele silencieusement
        different de celui qui a ete mesure.
        """
        for nom in ("bank_obs", "bank_repr", "bank_pret"):
            cle = prefix + nom
            if cle in state_dict:
                courant = getattr(self, nom)
                arrivant = state_dict[cle]
                if courant.shape != arrivant.shape:
                    setattr(self, nom, torch.zeros_like(arrivant))
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def definit_banque(self, obs: torch.Tensor):
        """Fixe les observations de reference. A n'appeler qu'avec du TRAIN."""
        if obs.dim() != 3:
            raise ValueError(f"banque attendue en (K, T, F), recue {tuple(obs.shape)}")
        self.bank_obs = obs.detach().clone()
        self.bank_repr = torch.zeros(obs.shape[0], self.d,
                                     device=obs.device, dtype=obs.dtype)
        self.bank_pret = torch.zeros(1, device=obs.device)

    @torch.no_grad()
    def rafraichit(self, encodeur):
        """Re-encode la banque avec les poids courants du tronc.

        Appelee apres chaque mise a jour PPO. Sans cela, les representations
        mises en cache derivent des poids d'il y a N pas de gradient, et la
        requete interroge une memoire perimee.
        """
        if self.bank_obs.numel() == 0:
            return
        morceaux = []
        for i in range(0, self.bank_obs.shape[0], 64):
            morceaux.append(encodeur(self.bank_obs[i:i + 64]))
        self.bank_repr = torch.cat(morceaux, dim=0)
        self.bank_pret = torch.ones(1, device=self.bank_repr.device)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Tant que la banque n'a pas ete encodee, la memoire est inerte :
        # mieux vaut un modele sans memoire qu'un modele qui interroge des
        # zeros et apprend a s'y fier.
        if self.bank_repr.numel() == 0 or float(self.bank_pret) == 0.0:
            return h

        B = h.shape[0]
        q = self.to_q(self.norm_q(h)).view(B, 1, self.heads, self.dim_tete).transpose(1, 2)
        kv = self.to_kv(self.norm_kv(self.bank_repr))
        k, v = kv.chunk(2, dim=-1)
        # Taille REELLE de la banque, pas celle demandee a la construction.
        # training.py tire min(cfg.n_ref, nombre d'etats collectes) : si une
        # epoch collecte moins d'etats que prevu, la banque est plus petite et
        # une vue de taille self.n_ref porterait sur des elements inexistants.
        n = self.bank_repr.shape[0]
        k = k.view(1, n, self.heads, self.dim_tete).transpose(1, 2).expand(B, -1, -1, -1)
        v = v.view(1, n, self.heads, self.dim_tete).transpose(1, 2).expand(B, -1, -1, -1)
        q, k = self.q_norm(q), self.k_norm(k)

        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, self.d)
        return h + self.gamma * self.drop(self.proj(o))


class SAINTPolicySingleHead(nn.Module):
    """actor : logits (N_ACTIONS) — critic : V(s)."""

    def __init__(
        self,
        n_features: int = OBS_N_FEATURES,
        d_model: int = 80,
        num_blocks: int = 2,
        heads: int = 4,
        dropout: float = 0.05,
        ff_mult: int = 2,
        max_len: int = 64,
        n_actions: int = N_ACTIONS,
        n_freq: int = 16,
        drop_path: float = 0.0,
        ls_init: float = 1e-4,
        n_ref: int = 0,
        # LARGEUR DE LA TETE. Elle valait 256 ecrit en dur, et c'etait le poste
        # le plus lourd du reseau : a d_model 8, la tete pesait 70 144
        # parametres sur 104 860, soit 67 %, quand les blocs d'attention entre
        # features — la seule chose que SAINT apporte et que PatchTST ne sait
        # pas faire — en coutaient 1 984, soit 2 %.
        #
        # Laisser 256 en dur imposait un budget de parametres decide une fois
        # pour toutes, sur un jeu qui n'a que 2 314 occasions independantes. Or
        # la reduction de ce budget est precisement ce qui a fait passer PPO de
        # -1.4 a +2.2 points au test dans la nuit du 2026-09-16.
        mlp_dim: int = 256,
        # COMMENT LA TETE LIT LE TRONC. Deux lectures, et le choix decide de
        # tout le comportement du reseau.
        #
        #   "cls"      le jeton CLS, resume par l'attention : 2 x d_model
        #              nombres, quel que soit le nombre de colonnes.
        #   "colonnes" les representations PAR COLONNE de la derniere bougie,
        #              concatenees : n_features x d_model nombres.
        #
        # POURQUOI CE PARAMETRE EXISTE, mesure du 2026-09-16. Avec la lecture
        # CLS et d_model 8, la tete recoit SEIZE nombres pour resumer 107
        # colonnes, quand PatchTST lui en donne 428. Resultat : l'etendue des
        # convictions plafonne a 0.028 apres 23 epochs — vingt fois moins que
        # PatchTST au meme stade — et l'entropie ne descend pas sous 1.089 sur
        # un maximum de 1.099. Le modele n'apprend pas moins bien : il n'a pas
        # la place de dire des choses differentes selon les situations.
        #
        # Pour egaler la largeur de PatchTST en lecture CLS il faudrait
        # d_model = 214, soit un embedding par colonne de 206 000 parametres.
        # La lecture par colonnes donne la meme largeur pour d_model 8, en
        # gardant les blocs d'attention qui croisent les features — lesquels ne
        # coutent que 1 984 parametres. Ce n'etait donc jamais l'attention qui
        # etait trop chere, c'etait la facon de la lire qui etait trop etroite.
        lecture: str = "cls",
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_actions = n_actions

        self.embed = NumericalEmbedding(n_features, d_model, n_freq=n_freq)
        # Position TEMPORELLE : portee par RoPE dans l'attention, pas ici.
        # Identite de COLONNE : apprise, car les features n'ont pas d'ordre.
        self.col_emb = nn.Embedding(n_features + 1, d_model)
        # Jeton de lecture, place en tete de l'axe des features.
        self.cls = nn.Parameter(torch.zeros(1, 1, 1, d_model))

        # Profondeur stochastique croissante : les premiers blocs, qui portent
        # les representations de base, sont conserves plus souvent.
        taux = [drop_path * i / max(num_blocks - 1, 1) for i in range(num_blocks)]
        self.blocks = nn.ModuleList([
            SAINTv2Block(d_model, heads, dropout, ff_mult,
                         drop_path=taux[i], ls_init=ls_init)
            for i in range(num_blocks)
        ])

        if lecture not in ("cls", "colonnes"):
            raise ValueError(f"lecture inconnue : {lecture}")
        self.lecture = lecture
        # Largeur de ce que la tete recoit. En "cls", deux vues concatenees du
        # jeton resume ; en "colonnes", une vue par feature.
        dim_lecture = (2 * d_model if lecture == "cls"
                       else n_features * d_model)
        self.dim_lecture = dim_lecture
        self.norm = RMSNorm(dim_lecture)

        # UN MLP DE LECTURE PAR TETE, DEPUIS LE 2026-09-21.
        #
        # CE QUI A MOTIVE LA SEPARATION. Les « trois tetes » etaient trois
        # couches de CINQ parametres chacune, posees sur un tronc de
        # 28 014 : 99.95 % du reseau leur etait commun. Ce ne sont pas
        # trois reseaux, ce sont trois lectures d'une meme representation,
        # et elles se disputaient la derniere couche.
        #
        # CE QUI RESTE PARTAGE, ET C'EST VOULU : l'encodeur — plongement,
        # blocs d'attention axiale, banque de reference. Il apprend « a
        # quoi ressemble le marche maintenant », une question commune aux
        # trois. Le TRIPLER couterait un passage avant de plus par barre,
        # et le passage avant est deja le goulot : la carte tourne a 28 %
        # d'utilisation avec `gpu_idle` actif, on paie le LANCEMENT.
        #
        # CE QUI CESSE DE L'ETRE : la lecture. « Faut-il acheter »,
        # « faut-il vendre » et « faut-il fermer » sont trois questions
        # differentes posees a la meme representation, et rien n'oblige
        # leur derniere transformation a etre la meme.
        def _lecture():
            return nn.Sequential(
                nn.Linear(dim_lecture, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, mlp_dim),
                nn.GELU(),
            )

        # `self.mlp` RESTE, et sert `actor` et `critic`. Les deux sont
        # gelees ou mortes depuis la suppression de PPO, mais leurs
        # tenseurs doivent survivre pour que les points de reprise
        # anterieurs se chargent encore.
        self.mlp = _lecture()
        self.mlp_achat = _lecture()
        self.mlp_vente = _lecture()
        self.mlp_cloture = _lecture()

        # Intersample attention, forme deployable : voir ReferenceMemory.
        # n_ref = 0 -> aucune memoire, le modele est strictement celui d'avant.
        self.memoire = (ReferenceMemory(dim_lecture, n_ref, heads, dropout,
                                        ls_init=ls_init) if n_ref > 0 else None)

        self.actor = nn.Linear(mlp_dim, n_actions)
        self.critic = nn.Linear(mlp_dim, 1)
        # ------------------------------------------------------------------
        # TETE AUXILIAIRE : le rendement net d'un ACHAT et d'une VENTE.
        #
        # POURQUOI ELLE EXISTE, mesure du 2026-09-16. Sur deux runs et deux
        # geometries, l'apprentissage PPO DEGRADE la validation de facon
        # significative — -1.95 pt a -2.1 sigma sur exec24, -3.47 a -2.2 sur
        # exec31 — sans que le cote entrainement bouge. Ce n'est donc pas du
        # sur-ajustement : le gradient pousse la politique vers un endroit qui
        # n'aide ni l'un ni l'autre.
        #
        # Le diagnostic : ON ENTRAINE LA POLITIQUE A AGIR, PUIS ON S'EN SERT
        # COMME CLASSEUR. PPO maximise le rendement des actions prises ; a
        # l'evaluation on jette 95 % des decisions et on garde les 5 % ou sa
        # probabilite est la plus haute. Rien dans l'objectif de PPO ne
        # recompense un bon ORDRE de ces probabilites — seulement une bonne
        # action en moyenne. Une politique peut etre optimale au sens de PPO
        # avec une confiance dont l'ordre ne veut rien dire.
        #
        # Cela explique le plus vieux fait non explique du depot : une
        # regression logistique atteint 0.6271 d'AUC, aucune politique
        # entrainee n'a depasse 0.5707. Le modele supervise REGRESSE le
        # rendement realise, donc il est entraine exactement a classer.
        #
        # CE QUE CETTE TETE CHANGE. Elle regresse le rendement net en unites de
        # risque, un scalaire par direction. Le gradient devient DENSE — il
        # porte sur toutes les decisions, pas seulement celles qui ont ete
        # tradees — et a FAIBLE VARIANCE, une regression sur cible continue au
        # lieu d'un avantage multiplie par une log-probabilite. Et il optimise
        # exactement la quantite qu'on lit au moment de decider.
        #
        # PPO N'EST PAS RETIRE. Les deux pertes partagent le tronc ; si la tete
        # auxiliaire porte tout, la comparaison entre ses scores et ceux de la
        # politique le dira.
        # TROIS TETES, UNE PAR DECISION. Elles etaient deux : un
        # `Linear(d, 2)` portait l'achat et la vente ensemble. Le calcul
        # est identique — deux applications lineaires independantes de la
        # meme representation — mais la structure ne disait pas ce
        # qu'elle faisait, et un gradient mort d'un SEUL cote n'aurait
        # pas ete lisible dans le diagnostic.
        self.tete_achat = nn.Linear(mlp_dim, 1)
        self.tete_vente = nn.Linear(mlp_dim, 1)
        # LA TROISIEME TETE : elle decide de FERMER. Une sortie, en
        # points de base bruts a venir. Voir `cloture`.
        self.tete_cloture = nn.Linear(mlp_dim, 1)
        # LA QUATRIEME : ELLE DECIDE DE PRENDRE LE PROFIT.
        #
        # Elle est le seul organe du reseau qui ne traverse PAS le tronc.
        # Voir `N_PROFIT_FEATURES` pour la mesure qui l'impose : nourrie
        # des 274 colonnes de marche, la meme cible tombe sous son propre
        # plancher de bruit.
        #
        # ELLE PREDIT UNE AMPLITUDE : ce qu'il reste a prendre, en ATR
        # d'entree, positif. Jamais un signe — ce depot porte l'echec
        # d'une cible signee, a 50.8 / 49.2 / 53.1 / 46.9 % sur quatre
        # epochs. Le softplus est applique par l'appelant, comme pour le
        # risque.
        self.mlp_profit = nn.Sequential(
            nn.Linear(N_PROFIT_FEATURES, 32), nn.GELU(),
            nn.Linear(32, 32), nn.GELU())
        self.tete_profit = nn.Linear(32, 1)

        # ============================================================
        # L'ACTEUR DE SORTIE — LA SORTIE REDEVIENT UNE DECISION APPRISE
        # ============================================================
        #
        # POURQUOI IL EXISTE, ET C'EST UNE HISTOIRE DE MODE D'ECHEC REPETE.
        # Le 2026-09-22, la sortie a ete confiee a DEUX tetes supervisees :
        # `tete_cloture` predit le risque, `tete_profit` ce qu'il reste a
        # prendre, et une REGLE ECRITE A LA MAIN transformait chaque
        # prediction en decision. Les deux tetes ont bien appris —
        # `rho +0.355` et `+0.272`, au-dessus de la mesure hors ligne. La
        # regle, elle, a echoue quatre fois de suite :
        #
        #     seuil 1.00 ATR absolu       tenue[G 3/15 P 52/88 x0.2]
        #     + condition latent > 0      x0.5, toujours inverse
        #     seuil relatif 0.25 x latent `fermerait 0.0 %` sur 51/53 epochs
        #     redressement d'echelle      74 % des GAGNANTS soldes par la
        #                                 fin d'episode, aucune tete
        #
        # Chaque correction etait juste et n'a jamais suffi, parce que le
        # defaut n'etait pas dans la calibration : il etait dans l'idee
        # meme de seuiller une amplitude predite. PPO SUPPRIME LA REGLE —
        # la politique sort la decision, pas un nombre qu'il faut ensuite
        # comparer a quelque chose.
        #
        # ET RIEN NE S'Y OPPOSE. Le journal du depot est explicite sur la
        # suppression de PPO en 2026-09-21 : « toute conclusion tiree d'un
        # ecart de validation inferieur a 0.2 R par trade est du bruit, y
        # compris les "PPO degrade la validation" accumules depuis exec24.
        # Ces runs n'ont pas montre que PPO nuit ; ils n'ont rien montre. »
        #
        # LA SORTIE EST UN BIEN MEILLEUR PROBLEME DE RL QUE L'ENTREE, et
        # c'est quantitatif :
        #
        #     decision d'ENTREE   ~55 occasions independantes par fenetre
        #     decision de SORTIE  une par barre de chaque trade
        #
        # Le plancher de bruit qui a tue toutes les mesures PPO passees
        # vient de la RARETE des occasions d'entree. La sortie n'a pas ce
        # probleme, et sa consequence se realise dans le trade meme, donc
        # l'attribution de credit est courte.
        #
        # IL LIT LES QUATRE COLONNES DE `tete_profit`, PLUS LE SENS :
        # latent, age, creux_rang, flux_rang, sens. La mesure qui impose
        # de ne pas lui donner le marche est la meme — nourrie des 274
        # colonnes de marche, la meme cible tombe sous son plancher de
        # bruit avec UN arbre. Le sens, lui, est la depuis que les shorts
        # ouvrent : voir `N_SORTIE_FEATURES`.
        #
        # DEUX ACTIONS : tenir, fermer. Pas quatre — il ne decide jamais
        # d'entrer, et lui laisser des actions impossibles diluerait son
        # gradient sur des cas qu'il ne voit pas.
        self.mlp_sortie = nn.Sequential(
            nn.Linear(N_SORTIE_FEATURES, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU())
        self.acteur_sortie = nn.Linear(64, N_ACTIONS_SORTIE)
        # LE CRITIQUE PARTAGE LE TRONC DE L'ACTEUR. Sur quatre colonnes
        # d'entree, deux corps separes apprendraient deux fois la meme
        # representation ; et c'est la valeur de l'ETAT qu'il estime, pas
        # celle d'une action.
        self.critique_sortie = nn.Linear(64, 1)
        # ============================================================
        # LE MODELE DE SORTIE : UN CORPS, DEUX TETES
        # ============================================================
        #
        # UN SEUL MODELE PPO pour la sortie, et il est SEPARE DES TETES
        # D'OUVERTURE. Les tetes d'achat et de vente vivent sur le tronc
        # SAINT et apprennent par la tete de rang ; le modele de sortie ne
        # lit pas le tronc, n'a pas le meme optimiseur, et ne recoit aucun
        # gradient des entrees. C'est la frontiere qui compte : ouvrir et
        # fermer sont deux metiers, et l'un ne doit pas deformer l'autre.
        #
        # SUR CE MODELE, DEUX TETES : une pour fermer les GAINS, une pour
        # fermer les PERTES. Fermer un gain et fermer une perte ne sont pas
        # la meme decision — sur un gain on se demande s'il reste quelque
        # chose a prendre, sur une perte si elle va se reprendre ou
        # s'aggraver. Une tete unique devait donner les deux reponses avec
        # les memes poids de sortie.
        #
        # LE CORPS EST PARTAGE, ET C'EST VOULU. Les deux tetes lisent la
        # meme situation — latent, age, creux, flux — et une seule
        # representation de cette situation suffit. Ce qui differe, c'est ce
        # qu'on en DECIDE : chaque tete a son acteur et son critique.
        #
        # `mlp_sortie` EST CE CORPS. Il existait deja : les points de
        # reprise anterieurs y chargent donc leurs poids, qui servent de
        # depart. `acteur_sortie` et `critique_sortie`, eux, ne decident
        # plus rien et restent pour le chargement — meme convention que
        # `actor` et `critic`.
        #
        # LE ROUTAGE EST LE SIGNE DU LATENT au moment de la decision. Voir
        # `sortie`. Le loyer zombie, qui multiplie le loyer du temps sur les
        # positions en perte, alimente donc uniquement la tete de PERTE.
        self.acteur_sortie_gain = nn.Linear(64, N_ACTIONS_SORTIE)
        self.critique_sortie_gain = nn.Linear(64, 1)
        self.acteur_sortie_perte = nn.Linear(64, N_ACTIONS_SORTIE)
        self.critique_sortie_perte = nn.Linear(64, 1)

        # ============================================================
        # QUATRE TETES PPO SUR LE TRONC — demande du proprietaire, 2026-09-25
        # ============================================================
        #
        #   tete_achat    ouvre un LONG             `entree`
        #   tete_vente    ouvre un SHORT            `entree`
        #   tete de gain  ferme une position en GAIN    `sortie_complete`
        #   tete de perte ferme une position en PERTE   `sortie_complete`
        #
        # TOUTES LISENT LE TRONC, donc TOUTES LES FEATURES. Les deux tetes de
        # sortie ne lisaient que cinq colonnes (latent, age, creux, flux,
        # sens) ; elles lisent desormais la representation SAINT complete,
        # plus ces cinq colonnes, concatenees.
        #
        # CHACUNE A SON OPTIMISEUR : voir `groupes_ppo`. Le tronc est
        # partage et a le sien, nourri par les quatre.
        #
        # PLUS DE CLASSEMENT : les tetes d'achat et de vente ne regressent
        # plus un rendement a horizon fixe, elles sortent des LOGITS de
        # politique appris par PPO sur ce que les trades rapportent.
        #
        # LE LECTEUR DE SORTIE EST ETROIT A L'ENTREE — 16 — parce que la
        # lecture « colonnes » fait 2 232 valeurs : un lecteur de 64 en
        # couterait 143 000 par tete, cinq fois le tronc.
        def _lecture_sortie():
            return nn.Sequential(
                nn.Linear(dim_lecture + N_SORTIE_FEATURES, 16), nn.GELU(),
                nn.Linear(16, 64), nn.GELU())
        self.lecteur_gain = _lecture_sortie()
        self.lecteur_perte = _lecture_sortie()

        self._init_poids()

    def _init_poids(self):
        """Initialisation orthogonale, tete d'acteur a gain 0.01.

        Engstrom et al. 2020, "Implementation Matters in Deep Policy Gradients" :
        sur PPO, ce detail pese davantage que la plupart des choix
        algorithmiques. Un acteur initialise a gain 1 sort des logits deja
        marques, donc une politique prematurement piquee, et les premiers pas
        de gradient corrigent un a priori arbitraire au lieu d'apprendre.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
        # LES QUATRE ACTEURS PPO PARTENT PRESQUE UNIFORMES, pour la meme
        # raison que `actor` : un acteur a gain racine de 2 sortirait des
        # logits deja marques, et le premier pas corrigerait un a priori
        # arbitraire. Les critiques restent a gain 1.
        for _m in (self.tete_achat, self.tete_vente,
                   self.acteur_sortie_gain, self.acteur_sortie_perte):
            nn.init.orthogonal_(_m.weight, gain=0.01)
            nn.init.zeros_(_m.bias)
        for _m in (self.critique_sortie_gain, self.critique_sortie_perte):
            nn.init.orthogonal_(_m.weight, gain=1.0)
            nn.init.zeros_(_m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Tronc : (B, T, F) -> lecture a deux vues (B, 2*d_model).

        Separe du forward pour que la banque de reference soit encodee par le
        MEME tronc, sans repasser par la tete ni par la memoire elle-meme.
        """
        assert x.dim() == 3, f"Input x must be (B,T,F), got {x.shape}"
        B, T, F = x.shape

        tok = self.embed(x)                                   # (B, T, F, D)

        # Jeton CLS en tete de l'axe des features : l'attention inter-colonnes
        # y agrege ce qui compte, au lieu d'une moyenne a poids egaux.
        tok = torch.cat([self.cls.expand(B, T, 1, self.d_model), tok], dim=2)

        cols = torch.arange(F + 1, device=x.device)
        tok = tok + self.col_emb(cols).view(1, 1, F + 1, self.d_model)

        for blk in self.blocks:
            tok = blk(tok)

        # LECTURE A DEUX VUES.
        #
        # L'ancienne version calculait cls_time et cls_feat par deux moyennes
        # qui, prises dans l'un ou l'autre ordre, donnent EXACTEMENT le meme
        # tenseur (ecart mesure 4.5e-08) : les deux vues etaient confondues et
        # la moitie de la tete lisait la meme chose.
        #
        # On garde deux vues reellement distinctes, lues sur le jeton CLS :
        #   - le resume de toute la fenetre ;
        #   - la DERNIERE bougie, celle sur laquelle la decision se prend.
        # Concatenees et non additionnees, pour que le MLP puisse les ponderer.
        if self.lecture == "colonnes":
            # Une vue PAR COLONNE, prise sur la derniere bougie — celle sur
            # laquelle la decision se prend. Les colonnes ont deja circule
            # entre elles dans les blocs, donc chaque representation porte le
            # croisement ; on ne jette pas ce croisement en le moyennant.
            #
            # Le jeton CLS est ecarte ici (indice 0) : il resume ce que les
            # colonnes disent deja, et l'ajouter reviendrait a compter deux
            # fois la meme information dans une tete qu'on cherche a garder
            # petite.
            return tok[:, -1, 1:, :].reshape(tok.shape[0], -1)

        cls = tok[:, :, 0, :]                                 # (B, T, D)
        return torch.cat([cls.mean(dim=1), cls[:, -1, :]], dim=-1)

    def forward(self, x: torch.Tensor):
        h = self.encode(x)

        # Attention vers la banque de reference. Identique pour tous les
        # echantillons du lot, donc l'independance au lot est preservee.
        if self.memoire is not None:
            h = self.memoire(h)

        h = self.mlp(self.norm(h))
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def sorties(self, x: torch.Tensor):
        """Les scores d'ENTREE — achat et vente — en un passage. (B, 2).

        ELLE EN RENDAIT TROIS, PUIS DEUX, PUIS UN SEUL TENSEUR. La tete de
        direction a ete supprimee le 2026-09-20, celle de budget le
        2026-09-21, et le meme jour `actor` et `critic` ont cesse d'etre
        CALCULES — leurs tenseurs restent, mais plus rien ne lit leur
        sortie. Ce qui reste est ce que la decision consomme.

        POURQUOI UN SEUL PASSAGE. Le rollout appelait `policy(x)` puis
        `policy.budget(x)`, et la validation y ajoutait `policy.rendement(x)`
        — deux et trois traversees completes du tronc pour des tetes qui ne
        sont que des couches lineaires sur la MEME representation. Profil du
        2026-09-19 : le passage avant pese 95 % de la collecte, un cout de
        LANCEMENT et non de calcul. Le tronc est traverse une fois ici.

        ELLES ETAIENT QUATRE. `tete_budget` a ete retiree le 2026-09-21,
        apres la tete de direction. Le raisonnement est le meme dans les
        deux cas, et il a ete mesure deux fois :

          PPO optimise le rendement de ses ACTIONS ; rien dans son objectif
          ne recompense un bon ORDRE de ses sorties. Or c'est un ORDRE que
          la selectivite consomme, et `rho` est une correlation de rang.

          exec40, 23 epochs, tete de DIRECTION : +0.27 point contre une
          reference de bruit a +0.7. Supprimee.
          exec67, 22 epochs, tete de BUDGET : elle apprend — `Hbudget`
          descend de 1.781 a 1.656 des que le gradient lui parvient — mais
          ce qu'elle apprend est de MISER LE MINIMUM, et `rho` se degrade
          avec : negatif 10 fois sur 11 sur la seconde moitie du run.

        CE QUI LA REMPLACE. La taille se deduit du RANG du score de la tete
        de rang, la meme grandeur qui decide de l'entree. Voir
        `part_du_rang` : plus l'occasion est haut classee, plus on mise, et
        `rho` suit le signe du classement sans qu'aucune tete l'apprenne.

        `actor` RESTE, GELEE. La direction a ete supprimee de la DECISION
        mais ses tenseurs restent dans le reseau — les retirer casserait le
        chargement des points de reprise, et le journal verifie a chaque
        lancement qu'aucune de ses sorties n'est lue.
        """
        h = self.encode(x)
        if self.memoire is not None:
            h = self.memoire(h)
        # L'ENCODEUR EST TRAVERSE UNE FOIS, les lectures sont separees.
        # C'est tout l'objet du compromis : le cout d'un passage avant est
        # celui du LANCEMENT, pas du calcul, et trois MLP de 8 472
        # parametres sur une representation deja calculee ne relancent
        # rien — ils s'ajoutent au meme noyau.
        # `self.mlp`, `actor` ET `critic` NE SONT PLUS CALCULES ICI.
        #
        # CE QU'ILS COUTAIENT. `self.mlp` fait 8 472 parametres — 15.9 %
        # du reseau — traverses a CHAQUE decision. Il ne nourrissait plus
        # que deux tetes mortes : `actor`, gelee et dont aucune sortie
        # n'est lue depuis la suppression de la direction, et `critic`,
        # dont la valeur alimentait `advantages`, `values_old` et
        # `returns` — trois tableaux calcules puis jamais relus depuis la
        # suppression de PPO.
        #
        # LES TENSEURS RESTENT DANS LE RESEAU. On cesse de les CALCULER,
        # on ne les retire pas : les points de reprise anterieurs les
        # portent, et `test_direction_supprimee` verifie que la tete de
        # direction existe encore pour qu'ils restent chargeables.
        zn = self.norm(h)
        return torch.cat([self.tete_achat(self.mlp_achat(zn)),
                          self.tete_vente(self.mlp_vente(zn))], -1)

    def rendement(self, x: torch.Tensor) -> torch.Tensor:
        """Rendement attendu (ACHAT, VENTE). (B, 2).

        DEUX TETES DISTINCTES, `tete_achat` et `tete_vente`, recomposees
        ici en un seul tenseur parce que tous les consommateurs lisent
        `(achat, vente)` ensemble. Les garder separees rend lisible un
        gradient mort d'un seul cote, qu'un `Linear(d, 2)` unique masquait.

        L'UNITE A CHANGE AVEC LE SCALPING M1 : sans stop, `R` n'a plus de
        denominateur. Ces sorties se lisent en POINTS DE BASE de rendement
        net, friction deduite. Voir `cibles_m1`.
        """
        h = self.encode(x)
        if self.memoire is not None:
            h = self.memoire(h)
        zn = self.norm(h)
        return torch.cat([self.tete_achat(self.mlp_achat(zn)),
                          self.tete_vente(self.mlp_vente(zn))], -1)

    def cloture(self, x: torch.Tensor) -> torch.Tensor:
        """Faut-il fermer la position ouverte ? (B, 1), en points de base.

        CE QU'ELLE PREDIT : ce qu'on gagne ENCORE en tenant l'horizon de
        plus, en BRUT. Negatif, il faut fermer ; positif, tenir. Voir
        `cibles_m1.cible_cloture` pour pourquoi la cible est brute — le cout
        de sortie sera paye de toute facon, donc il s'annule entre les deux
        branches de la decision, et le facturer ferait fermer trop tot.

        POURQUOI ELLE N'EST PAS DANS `sorties`. Les tetes d'entree sont
        interrogees sur les etats PLATS, celle-ci sur les etats EN POSITION.
        Les deux ensembles sont disjoints : les reunir dans un seul appel
        calculerait systematiquement une tete pour rien. Le tronc est
        traverse une fois dans chaque cas, ce qui est le but de `sorties`.

        ELLE LIT L'ETAT DE LA POSITION, et c'est ce qui la distingue d'une
        tete d'entree appliquee a l'envers : les six colonnes d'etat portent
        le sens, le gain latent, les barres tenues et la capacite restante.
        Sans elles la question « faut-il fermer » n'a pas de sens.
        """
        h = self.encode(x)
        if self.memoire is not None:
            h = self.memoire(h)
        return self.tete_cloture(self.mlp_cloture(self.norm(h)))

    def sortie(self, p: torch.Tensor):
        """Tenir ou fermer ? Rend (logits, valeur) sur (B, 5) colonnes.

        LES QUATRE COLONNES DE `profit`, PLUS LE SENS DE LA POSITION. Voir
        `entree_sortie` dans `training.py`, le SEUL endroit qui sache les
        extraire, et `N_SORTIE_FEATURES` pour la raison du sens.

        L'ACTEUR REND DES LOGITS, PAS UNE PROBABILITE. PPO a besoin du
        log-rapport entre l'ancienne et la nouvelle politique ; le
        calculer depuis une probabilite deja normalisee perd en precision
        sur les queues, la ou le rapport compte le plus.
        """
        # LE ROUTAGE PAR LE SIGNE DU LATENT. Le corps est calcule une fois ;
        # les deux tetes lisent sa sortie, et `torch.where` choisit ligne a
        # ligne. Le gradient d'une decision en perte remonte donc dans la
        # tete de PERTE et dans le corps partage — jamais dans la tete de
        # gain.
        #
        # L'EQUILIBRE EXACT VA A LA TETE DE PERTE. Un latent nul ne porte
        # aucun gain a proteger ; il porte en revanche le spread deja paye.
        h = self.mlp_sortie(p)
        en_gain = p[:, 0] > 0.0
        logits = torch.where(en_gain.unsqueeze(-1),
                             self.acteur_sortie_gain(h),
                             self.acteur_sortie_perte(h))
        valeur = torch.where(en_gain,
                             self.critique_sortie_gain(h).squeeze(-1),
                             self.critique_sortie_perte(h).squeeze(-1))
        return logits, valeur

    def _lecture_tronc(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        if self.memoire is not None:
            h = self.memoire(h)
        return self.norm(h)

    def entree(self, x: torch.Tensor):
        """Ouvrir un long, un short, ou attendre ? (logits (B, 3), valeur (B,)).

        DEUX TETES, UNE PAR COTE. `tete_achat` rend le logit d'ACHETER,
        `tete_vente` celui de VENDRE, chacune par son propre lecteur du
        tronc. Le logit d'ATTENDRE est fixe a zero : chaque tete dit
        seulement combien elle prefere ouvrir a ne rien faire, et le
        gradient d'une decision d'achat ne touche jamais la tete de vente.

        LA VALEUR de l'etat plat vient de `critic` sur son lecteur `mlp` —
        les deux etaient morts depuis la suppression de PPO, et reprennent
        exactement leur role d'origine.
        """
        zn = self._lecture_tronc(x)
        la = self.tete_achat(self.mlp_achat(zn))
        lv = self.tete_vente(self.mlp_vente(zn))
        logits = torch.cat([la, lv, torch.zeros_like(la)], dim=-1)
        valeur = self.critic(self.mlp(zn)).squeeze(-1)
        return logits, valeur

    def sortie_complete(self, x: torch.Tensor, p: torch.Tensor):
        """Tenir ou fermer, en lisant TOUTES les features. (logits (B, 2), valeur (B,)).

        `x` est l'observation complete, `p` les cinq colonnes de position
        (`entree_sortie` dans `training.py`) : latent, age, creux, flux,
        sens. Le tronc lit la premiere ; les deux sont concatenees pour la
        tete, qui voit donc le marche entier ET l'etat de sa position.

        LE ROUTAGE EST CELUI DE `sortie` : le signe du latent. Une decision
        en gain entraine la tete de gain, une decision en perte — equilibre
        exact compris — la tete de perte, et jamais l'inverse.
        """
        z = torch.cat([self._lecture_tronc(x), p], dim=-1)
        en_gain = p[:, 0] > 0.0
        hg = self.lecteur_gain(z)
        hp = self.lecteur_perte(z)
        logits = torch.where(en_gain.unsqueeze(-1),
                             self.acteur_sortie_gain(hg),
                             self.acteur_sortie_perte(hp))
        valeur = torch.where(en_gain,
                             self.critique_sortie_gain(hg).squeeze(-1),
                             self.critique_sortie_perte(hp).squeeze(-1))
        return logits, valeur

    def groupes_ppo(self):
        """Les poids de chaque optimiseur, DISJOINTS. dict nom -> liste.

            achat          lecteur + tete d'achat
            vente          lecteur + tete de vente
            gain           lecteur + acteur + critique de la coupure des gains
            perte          lecteur + acteur + critique de la coupure des pertes
            valeur_entree  lecteur + critique de l'etat plat
            tronc          plongement, blocs d'attention, normalisation

        Le tronc est PARTAGE : les quatre tetes le lisent, leurs pertes y
        remontent, et c'est son optimiseur a lui qui l'avance.
        """
        def _p(*mods):
            return [q for m in mods for q in m.parameters()]
        g = {
            "achat": _p(self.mlp_achat, self.tete_achat),
            "vente": _p(self.mlp_vente, self.tete_vente),
            "gain": _p(self.lecteur_gain, self.acteur_sortie_gain,
                       self.critique_sortie_gain),
            "perte": _p(self.lecteur_perte, self.acteur_sortie_perte,
                        self.critique_sortie_perte),
            "valeur_entree": _p(self.mlp, self.critic),
            "tronc": _p(self.embed, self.col_emb, *self.blocks, self.norm)
                     + [self.cls],
        }
        if self.memoire is not None:
            g["tronc"] += list(self.memoire.parameters())
        return g

    def params_sortie(self):
        """Tous les poids du MODELE DE SORTIE, et rien d'autre.

        Le corps partage et les deux tetes. Ni le tronc, ni les tetes
        d'ouverture : un seul optimiseur les porte, et il n'en touche
        aucun autre.
        """
        mods = (self.mlp_sortie,
                self.acteur_sortie_gain, self.critique_sortie_gain,
                self.acteur_sortie_perte, self.critique_sortie_perte)
        return [q for m in mods for q in m.parameters()]

    def profit(self, p: torch.Tensor) -> torch.Tensor:
        """Ce qu'il reste a prendre sur la position ouverte. (B, 1).

        ELLE NE PREND PAS L'ETAT COMPLET, mais les QUATRE colonnes de
        `N_PROFIT_FEATURES`, deja extraites par l'appelant : latent, age,
        `creux_rang`, `flux_rang`. C'est le seul chemin du reseau qui
        ignore le tronc, et une mesure l'impose — voir
        `N_PROFIT_FEATURES`.

        LA SORTIE EST UN LOGIT. L'appelant lui applique un softplus pour
        obtenir une amplitude positive, exactement comme pour le risque :
        les deux grandeurs vivent en ATR d'entree et la regle de sortie
        les compare sans conversion.
        """
        return self.tete_profit(self.mlp_profit(p))

    def definit_banque(self, obs: torch.Tensor):
        """Fixe les references. UNIQUEMENT des observations de la fenetre train."""
        if self.memoire is None:
            raise RuntimeError("Modele construit sans memoire (n_ref = 0)")
        self.memoire.definit_banque(obs)
        self.rafraichit_banque()

    def rafraichit_banque(self):
        """A appeler apres chaque mise a jour des poids du tronc."""
        if self.memoire is not None:
            self.memoire.rafraichit(self.encode)


class PatchTSTPolicy(nn.Module):
    """Politique PPO batie sur PatchTST — actor : logits, critic : V(s).

    POURQUOI CETTE ARCHITECTURE ICI. SAINT fait de l'attention sur DEUX axes,
    dont celui des colonnes, ce qui coute cher et limite en pratique la
    fenetre a 25 barres. PatchTST prend le probleme a l'envers : il decoupe la
    serie en SEGMENTS et traite chaque colonne INDEPENDAMMENT, ce qui permet
    d'allonger l'historique sans faire exploser le cout.

    Deux mecanismes, tous deux tires du papier (Nie et al., ICLR 2023) :

      SEGMENTATION. On regroupe les barres par paquets de `taille_patch` avec
      recouvrement. L'attention porte alors sur ~L/pas jetons au lieu de L, et
      chaque jeton represente un MOTIF local plutot qu'une barre isolee. A 96
      barres avec des segments de 16 et un pas de 8, cela fait 11 jetons au
      lieu de 96 — l'attention coute 75 fois moins.

      CANAUX INDEPENDANTS. Le MEME encodeur voit chaque feature separement,
      sans melange inter-colonnes. C'est l'exact oppose de SAINT. Le papier
      defend que ce partage REGULARISE : au lieu d'apprendre une fonction sur
      34 colonnes conjointes, on apprend une fonction sur une serie, vue 34
      fois. Le melange entre colonnes n'a lieu qu'a la toute fin, dans la tete.

    TETE AGREGEE, ET NON "FLATTEN". Le papier aplatit (F x N x d) parce qu'il
    predit une sequence entiere. Ici la sortie est 3 logits et un scalaire : on
    moyenne donc sur les segments avant la tete. A 192 barres, la version
    aplatie ferait 11.4 M de parametres contre 0.6 M ici — pour un
    environnement qui produit quelques milliers de decisions par epoch, le
    surapprentissage serait certain.

    INDEPENDANCE AU LOT. Aucune operation ne croise les echantillons d'un lot :
    un lot de 8 donne exactement 8 passes de 1, ce que verifie
    test_architecture.py. C'est la condition pour que les poids appris sous un
    lot de 128 se comportent pareil en live, ou le lot vaut 1.
    """

    def __init__(
        self,
        n_features: int = OBS_N_FEATURES,
        lookback: int = 96,
        taille_patch: int = 16,
        pas: int = 8,
        d_model: int = 32,
        heads: int = 4,
        num_blocks: int = 3,
        dropout: float = 0.1,
        n_actions: int = N_ACTIONS,
        mlp_dim: int = 128,
    ):
        super().__init__()
        self.n_features = n_features
        self.lookback = lookback
        self.taille_patch = taille_patch
        self.pas = pas
        self.d_model = d_model
        self.n_actions = n_actions
        self.n_patchs = max(1, (lookback - taille_patch) // pas + 1)

        self.proj = nn.Linear(taille_patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patchs, d_model))
        nn.init.normal_(self.pos, std=0.02)

        couche = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.enc = nn.TransformerEncoder(couche, num_layers=num_blocks)
        self.norm = nn.LayerNorm(d_model)

        # LA TETE DOMINE LE COMPTE DE PARAMETRES : n_features x d_model
        # entrees, soit 59 x 64 = 3 776 a l'ancienne largeur. Sur un jeu H1 de
        # 16 708 barres d'entrainement, cela faisait 71 parametres par exemple
        # et le modele memorisait — mesure : entropie de 1.10 a 0.50 en quinze
        # epochs, et l'ecart au point mort passant de -1.3 a -8.2 pendant que
        # l'etendue montait a 0.95. Signature de surapprentissage.
        # UN MLP DE LECTURE PAR TETE, comme dans SAINT — voir la-bas pour
        # la mesure. Les deux membres de l'ensemble doivent lire de la
        # MEME facon, sans quoi leur moyenne fait voter deux modeles qui
        # ne posent pas la meme question.
        def _lecture():
            return nn.Sequential(
                nn.Linear(n_features * d_model, mlp_dim), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, mlp_dim), nn.GELU(),
            )

        self.mlp = _lecture()          # `actor` et `critic`, gelees
        self.mlp_achat = _lecture()
        self.mlp_vente = _lecture()
        self.mlp_cloture = _lecture()
        self.actor = nn.Linear(mlp_dim, n_actions)
        self.critic = nn.Linear(mlp_dim, 1)
        # Meme tete auxiliaire que SAINT : voir le commentaire la-bas. Les
        # deux membres doivent predire la MEME quantite pour que leur
        # moyenne ait un sens.
        # TROIS TETES, UNE PAR DECISION. Elles etaient deux : un
        # `Linear(d, 2)` portait l'achat et la vente ensemble. Le calcul
        # est identique — deux applications lineaires independantes de la
        # meme representation — mais la structure ne disait pas ce
        # qu'elle faisait, et un gradient mort d'un SEUL cote n'aurait
        # pas ete lisible dans le diagnostic.
        self.tete_achat = nn.Linear(mlp_dim, 1)
        self.tete_vente = nn.Linear(mlp_dim, 1)
        # LA TROISIEME TETE : elle decide de FERMER. Une sortie, en
        # points de base bruts a venir. Voir `cloture`.
        self.tete_cloture = nn.Linear(mlp_dim, 1)
        # LA QUATRIEME : ELLE DECIDE DE PRENDRE LE PROFIT.
        #
        # Elle est le seul organe du reseau qui ne traverse PAS le tronc.
        # Voir `N_PROFIT_FEATURES` pour la mesure qui l'impose : nourrie
        # des 274 colonnes de marche, la meme cible tombe sous son propre
        # plancher de bruit.
        #
        # ELLE PREDIT UNE AMPLITUDE : ce qu'il reste a prendre, en ATR
        # d'entree, positif. Jamais un signe — ce depot porte l'echec
        # d'une cible signee, a 50.8 / 49.2 / 53.1 / 46.9 % sur quatre
        # epochs. Le softplus est applique par l'appelant, comme pour le
        # risque.
        self.mlp_profit = nn.Sequential(
            nn.Linear(N_PROFIT_FEATURES, 32), nn.GELU(),
            nn.Linear(32, 32), nn.GELU())
        self.tete_profit = nn.Linear(32, 1)

        # ============================================================
        # L'ACTEUR DE SORTIE — LA SORTIE REDEVIENT UNE DECISION APPRISE
        # ============================================================
        #
        # POURQUOI IL EXISTE, ET C'EST UNE HISTOIRE DE MODE D'ECHEC REPETE.
        # Le 2026-09-22, la sortie a ete confiee a DEUX tetes supervisees :
        # `tete_cloture` predit le risque, `tete_profit` ce qu'il reste a
        # prendre, et une REGLE ECRITE A LA MAIN transformait chaque
        # prediction en decision. Les deux tetes ont bien appris —
        # `rho +0.355` et `+0.272`, au-dessus de la mesure hors ligne. La
        # regle, elle, a echoue quatre fois de suite :
        #
        #     seuil 1.00 ATR absolu       tenue[G 3/15 P 52/88 x0.2]
        #     + condition latent > 0      x0.5, toujours inverse
        #     seuil relatif 0.25 x latent `fermerait 0.0 %` sur 51/53 epochs
        #     redressement d'echelle      74 % des GAGNANTS soldes par la
        #                                 fin d'episode, aucune tete
        #
        # Chaque correction etait juste et n'a jamais suffi, parce que le
        # defaut n'etait pas dans la calibration : il etait dans l'idee
        # meme de seuiller une amplitude predite. PPO SUPPRIME LA REGLE —
        # la politique sort la decision, pas un nombre qu'il faut ensuite
        # comparer a quelque chose.
        #
        # ET RIEN NE S'Y OPPOSE. Le journal du depot est explicite sur la
        # suppression de PPO en 2026-09-21 : « toute conclusion tiree d'un
        # ecart de validation inferieur a 0.2 R par trade est du bruit, y
        # compris les "PPO degrade la validation" accumules depuis exec24.
        # Ces runs n'ont pas montre que PPO nuit ; ils n'ont rien montre. »
        #
        # LA SORTIE EST UN BIEN MEILLEUR PROBLEME DE RL QUE L'ENTREE, et
        # c'est quantitatif :
        #
        #     decision d'ENTREE   ~55 occasions independantes par fenetre
        #     decision de SORTIE  une par barre de chaque trade
        #
        # Le plancher de bruit qui a tue toutes les mesures PPO passees
        # vient de la RARETE des occasions d'entree. La sortie n'a pas ce
        # probleme, et sa consequence se realise dans le trade meme, donc
        # l'attribution de credit est courte.
        #
        # IL LIT LES QUATRE COLONNES DE `tete_profit`, PLUS LE SENS :
        # latent, age, creux_rang, flux_rang, sens. La mesure qui impose
        # de ne pas lui donner le marche est la meme — nourrie des 274
        # colonnes de marche, la meme cible tombe sous son plancher de
        # bruit avec UN arbre. Le sens, lui, est la depuis que les shorts
        # ouvrent : voir `N_SORTIE_FEATURES`.
        #
        # DEUX ACTIONS : tenir, fermer. Pas quatre — il ne decide jamais
        # d'entrer, et lui laisser des actions impossibles diluerait son
        # gradient sur des cas qu'il ne voit pas.
        self.mlp_sortie = nn.Sequential(
            nn.Linear(N_SORTIE_FEATURES, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU())
        self.acteur_sortie = nn.Linear(64, N_ACTIONS_SORTIE)
        # LE CRITIQUE PARTAGE LE TRONC DE L'ACTEUR. Sur quatre colonnes
        # d'entree, deux corps separes apprendraient deux fois la meme
        # representation ; et c'est la valeur de l'ETAT qu'il estime, pas
        # celle d'une action.
        self.critique_sortie = nn.Linear(64, 1)

        # MEME INTERFACE QUE SAINT. L'entrainement interroge `policy.memoire`
        # et appelle `rafraichit_banque()` a chaque epoch : sans ces deux
        # attributs, changer de reseau plante au premier pas. On expose donc
        # une memoire inexistante plutot que de semer des getattr dans la
        # boucle d'entrainement — l'appelant n'a pas a savoir quel reseau il
        # pilote.
        #
        # PatchTST n'a PAS de banque de references : sa regularisation vient
        # du partage de l'encodeur entre colonnes, pas d'une memoire externe.
        self.memoire = None

        self._init_poids()

    def rafraichit_banque(self):
        """Sans objet ici — voir `self.memoire`."""
        return

    def definit_banque(self, obs):
        raise RuntimeError(
            "PatchTSTPolicy n'a pas de banque de references : sa "
            "regularisation vient du partage de l'encodeur entre colonnes.")

    def _init_poids(self):
        """Tete d'acteur a gain 0.01 (Engstrom et al. 2020).

        Sur PPO ce detail pese davantage que la plupart des choix
        algorithmiques : un acteur initialise a gain 1 sort des logits deja
        marques, donc une politique prematurement piquee que les premiers
        gradients doivent defaire au lieu d'apprendre.
        """
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F) -> (B, n_features * d_model)."""
        assert x.dim() == 3, f"Input x must be (B,T,F), got {x.shape}"
        B, T, F = x.shape
        if T < self.taille_patch:
            raise ValueError(
                f"lookback {T} plus court que la taille de segment "
                f"{self.taille_patch}")
        h = x.permute(0, 2, 1)                              # (B, F, T)
        h = h.unfold(dimension=2, size=self.taille_patch, step=self.pas)
        h = h[:, :, -self.n_patchs:, :]                     # segments les plus RECENTS
        h = self.proj(h) + self.pos.unsqueeze(1)
        # On replie F dans le lot : le meme encodeur voit chaque colonne
        # separement, sans qu'aucune information ne circule entre colonnes
        # ni entre echantillons.
        h = h.reshape(B * F, self.n_patchs, self.d_model)
        h = self.norm(self.enc(h))
        h = h.reshape(B, F, self.n_patchs, self.d_model).mean(dim=2)
        return h.reshape(B, F * self.d_model)

    def forward(self, x: torch.Tensor):
        h = self.mlp(self.encode(x))
        return self.actor(h), self.critic(h).squeeze(-1)

    def sorties(self, x: torch.Tensor):
        """Les TROIS tetes en UN SEUL passage dans le tronc.

        Voir `SAINTPolicySingleHead.sorties` : meme raison d'etre, et meme
        histoire — `tete_budget` retiree le 2026-09-21 apres la direction.
        """
        # Voir `SAINTPolicySingleHead.sorties` : ni `mlp`, ni `actor`, ni
        # `critic` ne sont calcules, leurs tenseurs restent chargeables.
        zn = self.encode(x)
        return torch.cat([self.tete_achat(self.mlp_achat(zn)),
                          self.tete_vente(self.mlp_vente(zn))], -1)

    def rendement(self, x: torch.Tensor) -> torch.Tensor:
        """Rendement attendu (ACHAT, VENTE), en bps nets. (B, 2)."""
        zn = self.encode(x)
        return torch.cat([self.tete_achat(self.mlp_achat(zn)),
                          self.tete_vente(self.mlp_vente(zn))], -1)

    def cloture(self, x: torch.Tensor) -> torch.Tensor:
        """Faut-il fermer ? (B, 1), en bps bruts. Voir l'autre reseau."""
        return self.tete_cloture(self.mlp_cloture(self.encode(x)))



N_REF_DEFAUT = 256

# Profondeur du tronc. 3 blocs, d'apres la configuration par defaut du
# FT-Transformer (Gorishniy et al. 2021), presentee dans le papier comme une
# baseline forte sans reglage et la mieux replique sur donnees tabulaires.
#
# La profondeur n'etait pas exploitable avant : les stabilisateurs ajoutes avec
# l'architecture complete — pre-norm, LayerScale a 1e-4, QK-Norm — sont ce qui
# rend un empilement plus profond entrainable (Bjorck et al. 2021, "Towards
# Deeper Deep Reinforcement Learning" : sans normalisation adaptee, un reseau
# profond en RL ne converge simplement pas).
#
# Cout mesure : 3.86x le fwd+bwd de l'ancien bloc reduit, contre 2.38x a 2.
N_BLOCS_DEFAUT = 3


ARCHI_DEFAUT = "patchtst"


class PolitiqueEnsemble(nn.Module):
    """Plusieurs reseaux qui votent, presentes comme UNE politique.

    POURQUOI CETTE FORME ET PAS UNE BOUCLE DANS L'ENTRAINEMENT. Faire voter
    trois modeles pendant l'apprentissage demande que l'action executee vienne
    du VOTE, pas d'un reseau en particulier. Or PPO compare la probabilite
    nouvelle a l'ancienne POUR L'ACTION TIREE : si l'action vient d'ailleurs
    que de la politique mise a jour, le rapport ne veut plus rien dire et il
    faut un poids d'importance non borne pour le rattraper.

    En presentant le melange comme une politique unique, le probleme
    disparait : l'action EST tiree de ce qui est mis a jour. Le gradient
    traverse le melange et atteint chaque membre, pondere par la part qu'il
    prend dans la probabilite de l'action choisie — un membre confiant sur un
    bon coup en recoit davantage. C'est un melange d'experts, pas un
    contournement, et la boucle d'entrainement n'a pas une ligne a changer.

    CE QUE LE MELANGE N'EST PAS. Ce n'est pas la moyenne des logits, qui
    reviendrait a une moyenne geometrique et laisserait un membre tres confiant
    ecraser les autres. C'est la moyenne des PROBABILITES — chaque membre a une
    voix, et un membre qui s'abstient (proche de l'uniforme) ne bloque rien.

    LA VALEUR EST MOYENNEE AUSSI. Chaque critique evalue la meme politique de
    comportement, celle du melange, puisque c'est elle qui genere les donnees.
    Leur moyenne est donc un estimateur de la meme quantite, en moins bruite.
    """

    def __init__(self, membres):
        super().__init__()
        self.membres = nn.ModuleList(membres)

    # ---- LA BOUCLE D'ENTRAINEMENT INTERROGE LA POLITIQUE SUR SA BANQUE ----
    #
    # `memoire`, `definit_banque` et `rafraichit_banque` appartiennent a la
    # ReferenceMemory de SAINT. Un ensemble n'en a pas en propre : il relaie a
    # ses membres, et rend None quand aucun n'en porte. Sans ces trois-la, un
    # `policy.memoire` dans la boucle leve une AttributeError au milieu du
    # premier fold — apres vingt minutes de collecte.

    @property
    def memoire(self):
        for m in self.membres:
            mem = getattr(m, "memoire", None)
            if mem is not None:
                return mem
        return None

    def definit_banque(self, *a, **kw):
        for m in self.membres:
            if hasattr(m, "definit_banque"):
                m.definit_banque(*a, **kw)

    def rafraichit_banque(self, *a, **kw):
        for m in self.membres:
            if hasattr(m, "rafraichit_banque"):
                m.rafraichit_banque(*a, **kw)

    def rendement(self, x):
        """La moyenne des rendements predits par les membres. (B, 2).

        Meme logique que pour la valeur : chaque membre estime la meme
        quantite, leur moyenne l'estime en moins bruite.
        """
        return torch.stack([m.rendement(x) for m in self.membres], 0).mean(0)

    def cloture(self, x):
        """La tete de cloture de l'ensemble. (B, 1).

        MOYENNE DIRECTE, comme la valeur et le rendement : les membres
        estiment la MEME quantite — des points de base a venir — et non des
        logits sur des echelles qui leur seraient propres.
        """
        return torch.stack([m.cloture(x) for m in self.membres], 0).mean(0)

    def sorties(self, x):
        """Les scores d'entree de l'ensemble : moyenne des membres. (B, 2).

        ELLE COMBINAIT TROIS TETES. La direction passait par une moyenne
        des PROBABILITES — moyenner des logits n'a pas de sens, ils ne
        vivent pas sur la meme echelle d'un membre a l'autre — et la
        valeur par une moyenne directe. Les deux ont cesse d'etre
        calculees le 2026-09-21 : plus rien ne lisait leur sortie.

        LE RENDEMENT SE MOYENNE DIRECTEMENT, lui, parce que les membres
        estiment la MEME quantite dans la MEME unite — des points de base
        de rendement net. C'est la condition pour qu'une moyenne ait un
        sens, et elle n'est pas remplie par des logits.
        """
        return torch.stack([m.sorties(x) for m in self.membres], 0).mean(0)

    def forward(self, x):
        probs, valeurs = [], []
        for m in self.membres:
            lo, v = m(x)
            probs.append(torch.softmax(lo, dim=-1))
            valeurs.append(v)
        p = torch.stack(probs, 0).mean(0).clamp_min(1e-9)
        # On rend des LOGITS, pas des probabilites : l'appelant leur applique
        # son masque d'actions puis un log_softmax. log(p) est un jeu de logits
        # valide pour p, a une constante additive pres que le softmax absorbe.
        return torch.log(p), torch.stack(valeurs, 0).mean(0)


def build_policy(device, lookback: int = 25,
                 n_features: int = OBS_N_FEATURES,
                 n_ref: int = N_REF_DEFAUT,
                 num_blocks: int = N_BLOCS_DEFAUT,
                 archi: str = ARCHI_DEFAUT,
                 d_model_patch: int = 32,
                 mlp_dim: int = 128,
                 taille_patch: int = 16,
                 pas: int = 8,
                 heads: int = 0,
                 state_dict=None):
    """Instancie la policy avec les hyperparamètres d'architecture du training.

    Toute divergence ici rend les checkpoints incompatibles : passer par cette
    fabrique plutôt que de recopier les arguments.

    `state_dict` : si fourni, la taille de la banque de références est DÉDUITE
    des poids au lieu d'être supposée. C'est le seul moyen sûr — une constante
    à tenir synchronisée entre training et live finit toujours par diverger, et
    l'erreur serait soit un refus de chargement, soit pire, un modèle construit
    sans mémoire qui trade en ignorant une partie de ce qu'il a appris.
    """
    # UN ENSEMBLE SE RECONNAIT A SES CLES `membres.N.` et se reconstruit membre
    # par membre : chacun garde sa propre architecture, deduite de ses poids.
    if state_dict is not None and any(
            k.startswith("membres.") for k in state_dict):
        n_membres = 1 + max(int(k.split(".")[1]) for k in state_dict
                            if k.startswith("membres."))
        membres = []
        for i in range(n_membres):
            pref = f"membres.{i}."
            sous = {k[len(pref):]: v for k, v in state_dict.items()
                    if k.startswith(pref)}
            membres.append(build_policy(
                device, lookback=lookback, n_features=n_features,
                n_ref=n_ref, num_blocks=num_blocks, d_model_patch=d_model_patch,
                mlp_dim=mlp_dim, taille_patch=taille_patch, pas=pas,
                heads=heads, state_dict=sous))
        return PolitiqueEnsemble(membres).to(device)

    # L'ARCHITECTURE elle-meme se deduit du fichier, DANS LES DEUX SENS. Le
    # parametre `pos` n'existe que dans PatchTST, `embed.weight` que dans
    # SAINT ; le fichier tranche donc seul, et `archi` ne sert plus que quand
    # il n'y a pas de fichier (construction a neuf).
    #
    # POURQUOI DANS LES DEUX SENS. La version precedente ne reconnaissait que
    # PatchTST : un checkpoint SAINT tombait dans la branche par defaut, qui
    # vaut "patchtst" depuis exec10, et le live aurait construit la mauvaise
    # architecture. Le chargement echouait ensuite sur une liste de cles
    # illisible au lieu de dire ce qui n'allait pas. Une deduction qui ne
    # couvre qu'un cas n'est pas une deduction, c'est un defaut par defaut.
    #
    # Le prefixe est aussi devenu une egalite : `startswith("pos")` aurait
    # attrape n'importe quelle future colonne nommee pos_quelque_chose.
    if state_dict is not None:
        est_patch = "pos" in state_dict
        est_saint = any(k.startswith(("embed.", "blocks.")) for k in state_dict)
        if est_patch and est_saint:
            raise ValueError("checkpoint ambigu : cles PatchTST ET SAINT")
        if est_patch:
            # LA TAILLE DE SEGMENT SE LIT DANS LE FICHIER : proj projette un
            # segment sur d_model, donc sa dimension d'entree EST la taille du
            # segment. Le pas, lui, ne s'y trouve pas — on le deduit de la
            # relation n_patchs = (lookback - segment) / pas + 1, avec le
            # lookback fourni par l'appelant.
            #
            # CE QUE CELA REMPLACE. La version precedente ecrivait
            # `(n_patchs - 1) * 8 + 16`, soit le pas et la taille codes en dur.
            # Elle rendait le bon lookback pour la seule configuration 16/8 et
            # un lookback FAUX pour toute autre, sans rien signaler — le meme
            # defaut que la profondeur et la taille de banque, deja corrige
            # ici, et qui a deja coute un checkpoint charge de travers.
            seg = int(state_dict["proj.weight"].shape[1])
            n_p = int(state_dict["pos"].shape[1])
            pas_lu = ((lookback - seg) // (n_p - 1)) if n_p > 1 else pas
            p = PatchTSTPolicy(
                n_features=n_features, lookback=lookback,
                taille_patch=seg, pas=max(1, pas_lu),
                d_model=int(state_dict["pos"].shape[2]),
                mlp_dim=int(state_dict["mlp.0.weight"].shape[0]),
                n_actions=N_ACTIONS).to(device)
            # CE QUE CE CONTROLE NE PEUT PAS FAIRE, et qu'il faut savoir. Le
            # lookback n'est PAS recuperable depuis les poids : le pas etant
            # deduit de la relation ci-dessus, n'importe quel lookback se
            # reconstruit en ajustant le pas, et le reseau se charge alors sans
            # broncher avec un espacement temporel faux. Verifie : un
            # checkpoint de lookback 4 se reconstruit proprement en pretendant
            # 96. Le seul symptome serait un modele qui trade mal.
            #
            # La geometrie est donc ecrite dans le fichier _calib.json a cote
            # du checkpoint, qui est deja celui qui dit comment s'en servir.
            # Le controle ci-dessous n'attrape que les incoherences grossieres.
            if p.n_patchs != n_p:
                raise ValueError(
                    f"lookback {lookback} incompatible avec le checkpoint : "
                    f"{p.n_patchs} segments construits contre {n_p} attendus "
                    f"(segment {seg}, pas {max(1, pas_lu)})")
            return p
        if est_saint:
            archi = "saint"
            # GEOMETRIE SAINT DEDUITE DU FICHIER. Elle etait ecrite en dur —
            # d_model 80, 5 tetes, n_freq 16, tete de 256 — donc un checkpoint
            # entraine a une autre taille etait IMPOSSIBLE a recharger, en live
            # comme en reevaluation. Troisieme exemplaire du meme defaut dans
            # cette fonction, apres la profondeur et la taille de banque.
            #
            # Ce que le fichier donne :
            #   cls              (1,1,1,d)          -> d_model
            #   embed.freqs      (n_features, k)    -> n_freq
            #   blocks.N.*                          -> profondeur
            #   mlp.0.weight     (mlp_dim, entree)  -> largeur de tete ET mode
            #                                          de lecture, car l'entree
            #                                          vaut 2*d en lecture CLS
            #                                          et n_features*d en
            #                                          lecture par colonnes.
            d_saint = int(state_dict["cls"].shape[-1])
            n_freq = int(state_dict["embed.freqs"].shape[1])
            mlp_dim = int(state_dict["mlp.0.weight"].shape[0])
            entree = int(state_dict["mlp.0.weight"].shape[1])
            lecture = "cls" if entree == 2 * d_saint else "colonnes"
            if lecture == "colonnes" and entree != n_features * d_saint:
                raise ValueError(
                    f"entree de tete {entree} incompatible : ni 2*{d_saint} "
                    f"(cls) ni {n_features}*{d_saint} (colonnes)")
            # LE NOMBRE DE TETES SE DEDUIT DE LA QK-NORM. Les poids
            # d'attention (qkv, proj) ont la meme forme quel que soit le
            # decoupage en tetes, donc eux ne disent rien — mais q_norm
            # normalise CHAQUE TETE separement, sa taille EST la dimension par
            # tete. heads = d_model / head_dim.
            #
            # Le repli approximatif que j'avais mis d'abord (d_model // 8)
            # rendait 10 tetes pour un checkpoint qui en avait 5, et le
            # chargement echouait sur la taille des normes de la memoire. Sans
            # memoire il aurait charge SANS ERREUR et calcule autre chose : les
            # poids d'attention auraient ete redecoupes en dix blocs au lieu de
            # cinq. C'est exactement le genre de panne silencieuse que ce depot
            # collectionne.
            if heads <= 0:
                cle_qn = "blocks.0.attn_temps.q_norm.weight"
                if cle_qn in state_dict:
                    heads = max(1, d_saint // int(state_dict[cle_qn].shape[0]))
                else:
                    heads = max(1, d_saint // 8)
            return SAINTPolicySingleHead(
                n_features=n_features, d_model=d_saint,
                num_blocks=(max(int(k.split(".")[1])
                                for k in state_dict
                                if k.startswith("blocks.")) + 1),
                heads=heads, n_freq=n_freq, mlp_dim=mlp_dim,
                lecture=lecture, dropout=0.05, ff_mult=2,
                max_len=lookback, n_actions=N_ACTIONS,
                n_ref=int(state_dict["memoire.bank_repr"].shape[0])
                if "memoire.bank_repr" in state_dict else 0).to(device)

        cle = "memoire.bank_repr"
        n_ref = int(state_dict[cle].shape[0]) if cle in state_dict else 0
        # Profondeur deduite elle aussi : elle etait ecrite en dur ici ET dans
        # training.py, deux endroits a tenir d'accord a la main.
        vus = {int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")}
        if vus:
            num_blocks = max(vus) + 1

    if archi == "patchtst":
        return PatchTSTPolicy(n_features=n_features, lookback=lookback,
                              taille_patch=taille_patch, pas=pas,
                              d_model=d_model_patch, mlp_dim=mlp_dim,
                              n_actions=N_ACTIONS).to(device)

    return SAINTPolicySingleHead(
        n_features=n_features,
        d_model=80,
        num_blocks=num_blocks,
        # 5 tetes et non 4 : d_model 80 / 4 = 20, qui n'est pas un multiple
        # de 8. Voir _verifie_dim_tete.
        heads=5,
        dropout=0.05,
        ff_mult=2,
        max_len=lookback,
        n_actions=N_ACTIONS,
        n_ref=n_ref,
    ).to(device)


def get_device(cfg) -> torch.device:
    if getattr(cfg, "force_cpu", False) or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


# ============================================================
# MASQUE D'ACTIONS
# ============================================================

def build_mask_from_pos_scalar(pos: int, device, side: str) -> torch.Tensor:
    # 0=BUY  1=SELL  2=HOLD
    mask = torch.zeros(N_ACTIONS, dtype=torch.bool, device=device)

    if pos != 0:
        # EN POSITION : ATTENDRE ou CLOTURER, et rien d'autre.
        #
        # Le masque ne laissait qu'ATTENDRE, parce que la sortie venait du
        # stop. Elle vient maintenant du modele — voir `N_ACTIONS`. Ouvrir
        # reste interdit : une seule position a la fois.
        mask[2] = True
        mask[3] = True
        return mask

    if side == "long":
        mask[0] = True
        mask[2] = True
    elif side == "short":
        mask[1] = True
        mask[2] = True
    else:  # "both" / "duel"
        mask[0] = True
        mask[1] = True
        mask[2] = True

    return mask


# ============================================================
# SIZING / SL / TP
# ============================================================

def compute_dynamic_volume(equity: float, max_lot: float = 100.0) -> float:
    """Paliers de 1000$ à partir de 2000$ : 0.10 lot puis +0.10 par tranche."""
    if equity <= 2000.0:
        tier = 1
    else:
        tier = int((equity - 1.0) // 1000.0)
    return round(min(0.10 * tier, max_lot), 2)


class SeuilRang:
    """Filtre de conviction par RANG GLISSANT, et non par niveau fige.

    LE PROBLEME QU'IL RESOUT. Le filtre precedent calibrait un NIVEAU de
    conviction sur une periode, puis l'appliquait telle quelle a la suivante.
    Mesure sur un run reel : l'etendue des convictions sur la fenetre evaluee
    est passee de 0.0035 a 0.0914 entre les epochs 6 et 12 — vingt-six fois
    plus — pendant que le seuil herite restait autour de 0.24. La barre s'est
    donc retrouvee tres haut dans la distribution courante, et le nombre de
    trades s'est effondre de 1 461 a 20, dont zero vente.

    Un niveau ne transfere pas quand la politique derive. Un RANG, si :
    « entrer si cette occasion est dans les q % les plus convaincues des N
    dernieres vues » ne depend d'aucune echelle absolue.

    C'EST AUSSI CE QU'ON FERAIT EN LIVE. Le bot ne peut pas connaitre la
    distribution des convictions a venir ; il ne connait que celles qu'il a
    deja vues. Cette regle est donc causale par construction, et le simulateur
    reproduit enfin la contrainte de production au lieu de la contourner.

    La fenetre glissante est en NOMBRE D'OCCASIONS, pas en minutes : ce qui
    compte est d'avoir assez d'echantillons pour estimer un quantile, pas de
    couvrir une duree.
    """

    def __init__(self, fraction: float, taille_fenetre: int = 2000,
                 amorce=None):
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction hors de ]0, 1]")
        self.fraction = float(fraction)
        self.taille = int(taille_fenetre)
        self._vus = collections.deque(maxlen=self.taille)
        # Minimum d'echantillons avant de trancher. Sous ce seuil, un quantile
        # a 5 % n'a aucun sens : on s'abstient plutot que de tirer au sort.
        self._minimum = max(50, int(2.0 / self.fraction))
        if self.taille < self._minimum:
            raise ValueError('Fenêtre trop courte pour estimer ce quantile')
        if amorce is not None:
            for v in amorce:
                self.observe(v)

    def pret(self) -> bool:
        return len(self._vus) >= self._minimum

    def seuil(self) -> float:
        """Niveau courant correspondant a la fraction visee."""
        if not self.pret():
            return float("inf")
        values = np.asarray(self._vus, dtype=np.float64)
        threshold = float(np.quantile(values, 1.0 - self.fraction))
        # An atom at the quantile must not turn a flat policy into 100% BUY.
        if np.mean(values >= threshold) > self.fraction + 1.0 / len(values):
            threshold = float(np.nextafter(threshold, np.inf))
        return threshold

    def accepte(self, conviction: float) -> bool:
        """Cette conviction est-elle dans le haut du rang glissant ?

        On decide AVANT d'enregistrer : une occasion ne doit pas participer au
        quantile qui la juge.
        """
        if not np.isfinite(conviction):
            raise ValueError('Conviction non finie')
        ok = self.pret() and float(conviction) >= self.seuil()
        self.observe(conviction)
        return bool(ok)

    def rang(self, conviction: float) -> float:
        """OU se situe cette conviction dans la fenetre, entre 0 et 1.

        `accepte` dit SI l'occasion passe ; ceci dit DE COMBIEN. C'est ce qui
        permet de miser gros sur le haut du classement et petit sur le bas,
        sans qu'aucune tete n'ait a l'apprendre — voir `part_du_rang`.

        MEME DISCIPLINE QUE `accepte` : on lit AVANT d'enregistrer, donc une
        occasion ne participe jamais au rang qui la juge. L'appelant doit
        appeler `observe` ensuite, exactement comme `accepte` le fait.

        REND `nan` TANT QUE LA FENETRE EST TROP COURTE, et non 0.5 : un rang
        invente placerait toutes les premieres occasions au milieu de
        l'echelle, et le palier qui en sortirait aurait l'air mesure.
        """
        if not np.isfinite(conviction):
            raise ValueError('Conviction non finie')
        if not self.pret():
            return float('nan')
        vus = np.asarray(self._vus, dtype=np.float64)
        return float(np.mean(vus <= float(conviction)))

    def observe(self, conviction: float) -> None:
        """Enregistre une conviction sans decider (occasions non evaluees)."""
        if not np.isfinite(conviction):
            raise ValueError('Conviction non finie')
        self._vus.append(float(conviction))


class EntryDecisionPolicy:
    """One mutable decision stream. Never share it across episodes or agents.

    Call once per flat opportunity. In-position bars do not enter its history.
    A checkpoint stores the immutable starting specification, not validation's
    final history. A new stream always starts from an independent copy.
    """
    def __init__(self, spec):
        import copy
        self.spec = copy.deepcopy(spec)
        if self.spec.get('version') != 1:
            raise ValueError('Version de politique de décision inconnue')
        self.mode = self.spec['mode']
        # LE COTE FAIT PARTIE DE LA SPECIFICATION, donc du checkpoint : une
        # politique long-only rechargee ailleurs doit rester long-only sans
        # que l'appelant ait a le redire. Les specs anterieures n'ont pas la
        # cle et gardent leur comportement documente.
        self.side = str(self.spec.get('side', 'both'))
        cotes_permises(self.side)          # leve tot si la valeur est fausse
        if self.mode == 'rolling_rank':
            if self.spec.get('ties') != 'conservative':
                raise ValueError('Convention des égalités inconnue')
            bootstrap = self.spec['bootstrap']
            if len(bootstrap) != 2:
                raise ValueError('Deux historiques BUY/SELL sont requis')
            self.ranks = [SeuilRang(self.spec['fraction_per_side'], self.spec['window'], values)
                          for values in bootstrap]
        elif self.mode == 'fixed':
            self.fixed = tuple(float(v) for v in self.spec['thresholds'])
            if len(self.fixed) != 2 or not np.isfinite(self.fixed).all():
                raise ValueError('Seuils fixes invalides')
        else:
            raise ValueError('Mode de décision inconnu')

    @property
    def thresholds(self):
        return tuple(rank.seuil() for rank in self.ranks) if self.mode == 'rolling_rank' else self.fixed

    def decide(self, p_buy, p_sell):
        values = (float(p_buy), float(p_sell))
        if not np.isfinite(values).all():
            raise ValueError('Probabilités non finies')
        thresholds = self.thresholds  # Both read before either history changes.
        action = decide_avec_barres(*values, thresholds, self.side)
        if self.mode == 'rolling_rank':
            # Les DEUX historiques continuent d'observer, meme le cote
            # interdit : sa barre ne sert alors a rien, mais la fenetre reste
            # comparable d'un run a l'autre et le jour ou le cote est rouvert
            # elle n'a pas a etre reamorcee.
            for rank, value in zip(self.ranks, values):
                rank.observe(value)
        return action


def rolling_decision_spec(fraction, window, bootstrap, side='both'):
    spec = {'version': 1, 'mode': 'rolling_rank', 'ties': 'conservative',
            'side': str(side),
            'fraction_per_side': float(fraction), 'window': int(window),
            'bootstrap': [[float(v) for v in side_[-window:]]
                          for side_ in bootstrap]}
    EntryDecisionPolicy(spec)  # Fail before saving an unusable checkpoint.
    return spec


def load_decision_policy(checkpoint, fallback=None, side=None):
    """Fresh state on every load; callers retain one instance per stream."""
    import json
    from pathlib import Path
    path = Path(checkpoint).with_suffix('').with_name(Path(checkpoint).stem + '_calib.json')
    if path.exists():
        with path.open(encoding='utf-8') as f:
            metadata = json.load(f)
        if 'decision_policy' in metadata:
            spec = dict(metadata['decision_policy'])
            # `side` explicite l'emporte : un checkpoint d'avant la correction
            # ne porte pas la cle, et le rechargerait sans restriction.
            if side is not None:
                spec['side'] = str(side)
            return EntryDecisionPolicy(spec)
    # Older checkpoints retain their documented fixed-threshold behavior.
    return EntryDecisionPolicy({'version': 1, 'mode': 'fixed',
                                'side': str(side) if side is not None else 'both',
                                'thresholds': list(load_calib_thresholds(checkpoint, fallback))})


def score_retenue(rendements):
    """Le critere qui CHOISIT le checkpoint. Rend (gain, baisse, net).

    UNE SEULE ECRITURE, et c'est la raison d'etre de cette fonction. Elle
    vivait en ligne dans la boucle d'entrainement, et son test la
    reimplementait — deux descriptions du meme calcul, qui doivent s'accorder
    par convention. C'est la faute que ce depot passe son temps a payer.

    CE QU'ELLE MESURE. Pour chaque occasion RETENUE par la tete de rang :

        b_i   nombre de positions minimales que le modele y miserait
        R_i   rendement de l'occasion, en unites de risque

        gain   = somme(b_i x R_i) / somme(b_i)
        baisse = racine( somme(b_i x min(R_i, 0)^2) / somme(b_i) )
        net    = gain - baisse

    Soit le R moyen par POSITION MISEE, moins la taille typique des pertes,
    la meme ponderation des deux cotes.

    ELLE EST INVARIANTE AU LEVIER, et ca a coute une version pour s'en
    apercevoir. La premiere sommait `b_i x R_i` sans normaliser : les deux
    termes etaient alors lineaires en budget, donc leur difference aussi, et
    doubler tous les budgets doublait le score. Mesure sur exec55 :

        epoch   bud    gain    baisse   gain/baisse    net (ancienne version)
          1     1.94   1.417    1.435      0.988         -0.019
          2    12.00   9.518    9.350      1.018         +0.168

    `gain/baisse` vaut 1.0 aux deux epochs — aucune competence ni dans un cas
    ni dans l'autre — mais l'ancien `net` passait de negatif a positif parce
    que le budget avait ete multiplie par six. Le checkpoint a ete retenu
    la-dessus, avec un creux de validation a 85 %.

    L'ABSTENTION PAIE, et c'est tout l'objet du critere. Miser zero sur une
    occasion la retire des DEUX sommes : une mauvaise occasion ecartee monte
    `gain` et vide la queue gauche, donc baisse `baisse`. Payee deux fois.
    Miser gros au mauvais moment est puni deux fois par la meme mecanique.

    NE RIEN MISER DU TOUT rend exactement zero — pas une division par zero,
    pas un score negatif. Mieux vaut ne pas trader que trader mal ; le
    portillon du hasard, en amont, empeche qu'on retienne un modele inerte.
    """
    r = np.asarray(rendements, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0, 0.0, 0.0
    gain = float(r.mean())
    neg = np.minimum(r, 0.0)
    baisse = float(np.sqrt((neg * neg).mean()))
    return gain, baisse, gain - baisse


def places_ouvrables_compte(equity: float, marge_utilisee: float,
                            prix: float, lot_min: float, contrat: float,
                            marge_frac: float, niveau_marge: float,
                            risque_engage: float = 0.0,
                            risque_une: float = 0.0,
                            plafond_equite: float = PLAFOND_RISQUE_EQUITE
                            ) -> int:
    """Combien de positions de plus le COMPTE permet, ici et maintenant.

    SOURCE UNIQUE. L'environnement d'entrainement et `kairos_live` appellent
    tous deux cette fonction. C'est la seule facon qu'ils aient de rester
    d'accord : la version precedente vivait dans
    `BTCTradingEnvDiscrete._places_ouvrables`, et le live n'en avait aucune
    — il ouvrait UNE position par agent et passait son tour ensuite, pendant
    que l'entrainement en ouvrait soixante. Deux strategies differentes sous
    le meme nom, et rien pour le signaler.

    LA FONCTION EST PURE : elle ne lit ni environnement, ni MetaTrader, ni
    configuration. L'appelant lui donne l'etat du compte — l'env depuis ses
    propres tableaux, le live depuis `account_info()` — et elle rend un
    nombre. C'est ce qui la rend testable des deux cotes avec les memes
    entrees.

    TROIS BORNES.

      LA MARGE, contrainte du COURTIER. On s'arrete avant que le niveau de
      marge ne descende sous `niveau_marge` : la marge disponible vaut alors
      `equity / niveau_marge - marge_utilisee`, et chaque position de plus en
      consomme `prix x lot_min x contrat x marge_frac`.

      LE BUDGET, qui est NOTRE politique : `budget_part` dit QUELLE PART du
      plafond de survie on accepte d'occuper. Ce qui est deja engage se
      compte dans la meme unite — une position deux fois plus grosse que le
      minimum en consomme deux.

      LE PLAFOND DE SURVIE, en fraction d'equite, qu'aucun choix ne franchit.
      C'est desormais l'UNITE du budget : `budget_part = 1.0` vaut exactement
      le plafond, et les deux bornes coincident.

    ZERO N'EST PLUS UN PIEGE. `budget_risque = 0` signifiait ici « PAS DE
    CONTRAINTE » et non « pas de risque » : le test mesurait 129 places
    ouvrables a 0 % contre 4 a 3 %. Le palier zero etant devenu le mecanisme
    d'ABSTENTION du modele, la fonction aurait rendu l'inverse exact de
    l'intention. Deux appelants posaient un garde-fou explicite avant
    d'appeler ; c'etait deux endroits ou se tromper. Le zero est traite ici,
    une fois.

    POURQUOI UNE PART ET NON UN NOMBRE — et pourquoi `capital_reference` a
    disparu. Voir `BUDGETS_PART` pour la mesure complete ; en deux lignes :
    un nombre absolu de positions se fait ecraser par le plafond de survie
    des que l'ATR monte — quatre paliers sur six rendaient la meme capacite
    en marche agite — et `capital_reference` ajoutait au compte de positions
    une croissance en equite que la TAILLE des positions portait deja.

    LA COMPOSITION RESTE, PAR CONSTRUCTION. Le plafond vaut
    `plafond_equite x equite / risque_une` positions. Sous le lot minimum du
    courtier `risque_une` ne depend pas du capital, donc le plafond — et
    toute part de ce plafond — croit avec l'equite : le cercle vertueux est
    intact. Au-dessus, `risque_une` devient proportionnel au capital et c'est
    la taille de chaque position qui compose. Dans les deux regimes la part
    du compte reellement en risque reste bornee par `plafond_equite`.
    """
    if equity <= 0:
        return 0
    m_une = max(prix * lot_min * contrat * marge_frac, 1e-12)
    marge_max = equity / max(niveau_marge, 1e-9)
    par_marge = math.floor((marge_max - marge_utilisee) / m_une)

    r_une = max(risque_une, 1e-12)
    # CE QUI EST DEJA EN RISQUE, COMPTE EN POSITIONS MINIMALES. Une position
    # deux fois plus grosse que le minimum en occupe deux : c'est le meme
    # accounting qu'avant, change d'unite.
    deja = risque_engage / r_une

    # DEUX BORNES, ET PLUS TROIS. La troisieme etait le BUDGET — une part
    # du plafond, choisie par le modele. Elle a ete retiree le 2026-09-21 :
    # l'environnement ne tient qu'UNE position, l'appelant demandait donc
    # deja la capacite a plein, et la borne coincidait exactement avec le
    # plafond de survie. Une borne qui ne mord jamais est un chemin de code
    # qu'on ne teste pas.
    #
    # L'ABSTENTION NE PASSE PLUS PAR ICI. Elle etait le palier zero ; elle
    # est maintenant l'appartenance au sommet du classement — une occasion
    # hors des `q %` du haut n'est simplement pas prise. C'est la meme
    # decision, prise a un seul endroit au lieu de deux.
    par_survie = math.floor(
        (plafond_equite * equity - risque_engage) / r_une)

    return int(max(0, min(par_marge, par_survie)))


def compute_risk_volume(equity: float, risk_frac: float, sl_dist: float,
                        tick_value: float, tick_size: float,
                        vol_min: float = 0.01, vol_step: float = 0.01,
                        vol_max: float = 100.0,
                        contract_size: float = 0.0):
    """Volume tel qu'un stop touche coute exactement `risk_frac` de l'equity.

    C'est la MEME regle que celle de l'environnement d'entrainement
    (training.PPOEnv._compute_dynamic_size). Elle doit l'etre : le modele
    apprend sous une loi de taille donnee, et le deployer sous une autre
    revient a jouer un risque que rien n'a valide.

    L'ancienne regle etait un escalier sur l'equity — 0.10 lot sous 2000 $,
    +0.10 par tranche de 1000 $ — qui ne regardait NI le prix NI la
    volatilite. Le stop etant pose a 5xATR, le risque reel suivait l'ATR :
    sur l'or, 2.5 % du capital par jour calme, 10 % par jour agite, sans
    qu'aucun reglage ne le borne.

    La conversion passe par tick_value / tick_size plutot que par
    contract_size : c'est la seule qui reste juste quand la devise de
    cotation n'est pas celle du compte.

    Retourne (volume, risque_effectif_en_fraction, plancher_mordu).
    `plancher_mordu` signale que le volume minimum du courtier impose de
    risquer PLUS que demande — le seul cas ou le controle du risque echoue,
    et il doit etre visible.
    """
    if equity <= 0.0 or risk_frac <= 0.0 or sl_dist <= 0.0:
        return 0.0, 0.0, False

    # Valeur, en devise du compte, d'UNE unite de prix pour UN lot.
    #
    # tick_value / tick_size est la voie juste : elle porte la conversion de
    # devise. Mesure sur BTCUSD chez ce broker : 0.008621 / 0.01 = 0.8621, et
    # order_calc_profit confirme 86.23 $ pour 1 lot et +100 $ de prix — ce
    # n'est PAS 1.0, malgre un contract_size de 1.0 et un currency_profit en
    # USD.
    #
    # PIEGE MESURE : MT5 renvoie trade_tick_value = 0.0 tant que le symbole
    # n'est pas selectionne dans le Market Watch. Sans ce repli, un MT5
    # fraichement demarre annulerait CHAQUE ordre. L'appelant doit appeler
    # symbol_select() ; le repli est la pour que l'oubli coute une taille
    # approximative et un avertissement, pas un silence total.
    if tick_value > 0.0 and tick_size > 0.0:
        valeur_par_unite = tick_value / tick_size
    elif contract_size > 0.0:
        valeur_par_unite = contract_size
    else:
        return 0.0, 0.0, False

    perte_par_lot = sl_dist * valeur_par_unite
    if perte_par_lot <= 0.0:
        return 0.0, 0.0, False

    brut = (equity * risk_frac) / perte_par_lot

    # Arrondi VERS LE BAS sur le pas du courtier : on ne risque jamais plus
    # que demande par un arrondi.
    if vol_step > 0.0:
        volume = math.floor(brut / vol_step) * vol_step
    else:
        volume = brut
    volume = min(volume, vol_max)

    plancher_mordu = False
    if volume < vol_min:
        volume = vol_min
        plancher_mordu = True

    volume = round(volume, 8)
    risque_effectif = (volume * perte_par_lot) / equity
    return float(volume), float(risque_effectif), plancher_mordu


ATR_PLANCHER_FRAC = 0.0001   # 0.01 % du prix ~ un spread


def effective_atr(entry_price: float, entry_atr: float) -> float:
    """ATR plancher — garde-fou contre une bougie degeneree, RIEN DE PLUS.

    Le plancher historique de 0.15 % du prix etait cinq fois trop haut :
    mesure sur 2.74 M bougies d'or, il l'emportait sur l'ATR reel dans
    99.4 % des cas, avec un rapport median de 5.71x. Un stop annonce a
    5xATR etait donc pose a 28.5xATR, et un TP a 10xATR a 57xATR — plus
    rien ne les atteignait, et l'agent sortait presque toujours par le
    temps avec des gains et des pertes symetriques, incompatibles avec le
    R:R vise.

    Le defaut mordait aussi sur BTCUSD (plancher a 2.9x l'ATR), ce qui
    explique en partie les positions anormalement longues et le peu de
    decisions par epoch qu'on y observait.

    A 0.01 % le plancher vaut environ UN SPREAD et ne mord que sur 5.4 %
    des bougies. C'est exactement son role : empecher un stop pose a
    l'interieur du spread, donc touche instantanement. Au-dela, la
    protection vient du multiplicateur lui-meme — un SL a 5xATR represente
    deja ~14 spreads.
    """
    return max(entry_atr, ATR_PLANCHER_FRAC * entry_price, 1e-8)


def compute_sl_tp(cfg, entry_price: float, side: int, entry_atr: float):
    """SL/TP purs, sans compensation de spread.

    MT5 clôture au BID (long) / ASK (short) : l'asymétrie est modélisée côté
    déclenchement (backtest) plutôt qu'en décalant les niveaux, pour que le
    mouvement de prix requis reste celui vu à l'entraînement.
    """
    eff_atr = effective_atr(entry_price, entry_atr)

    sl_dist = cfg.atr_sl_mult * eff_atr
    tp_dist = cfg.atr_tp_mult * eff_atr * cfg.tp_shrink

    # UN tp NUL VEUT DIRE "PAS D'OBJECTIF", et MT5 l'entend exactement ainsi :
    # un champ `tp` a zero dans la requete ne pose aucun ordre limite. La regle
    # de sortie de l'entrainement n'en pose plus depuis le 2026-09-16 ; poser
    # un objectif ici ferait tourner en production une strategie que le modele
    # n'a jamais vue — il vendrait au douzieme du chemin les trades qui paient.
    #
    # ATTENTION, LA CONTREPARTIE EST DURE : sans objectif, le stop suiveur est
    # la seule sortie. L'appel a `update_sl_be_trailing_live` DOIT tourner dans
    # la boucle, sinon la position n'a plus que son stop initial.
    if not getattr(cfg, "use_tp", True):
        tp_dist = 0.0

    if side == 1:
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist if tp_dist > 0 else 0.0
    else:
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist if tp_dist > 0 else 0.0

    return max(sl, 1e-8), (max(tp, 1e-8) if tp_dist > 0 else 0.0)


def compute_entry_atr(df_closed: pd.DataFrame) -> float:
    """Dernier ATR(14) clôturé — identique backtest / live / MQL5."""
    if "atr_14" not in df_closed.columns or len(df_closed) == 0:
        return 0.0
    val = df_closed["atr_14"].iloc[-1]
    return 0.0 if pd.isna(val) else float(val)
