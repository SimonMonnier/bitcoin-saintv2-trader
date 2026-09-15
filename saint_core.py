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
N_ACTIONS = 3
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
SCALPING_MAX_HOLDING = 30


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
FEATURE_COLS_EXT = [
    "taker_ratio",       # flux agressif acheteur/vendeur, agrege 5 min (metrics)
    "taker_ma5",         # le meme, moyenne causale sur 5 barres (klines)
]

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
FEATURE_COLS_LIQ_TEMPS = ["flux_taille_trade", "flux_intensite",
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
FEATURE_COLS_SUP = [c.replace("_h1", "_h4") for c in FEATURE_COLS_H1]

# Features de la strategie "bornes d'un range" (Ichimoku / Riguet).
# Voir features_range.py pour les definitions exactes et, surtout, pour la
# raison qui fait que les figures de chandeliers y sont CONDITIONNEES a la
# proximite d'une borne : hors des bornes, les documents les disent
# "strictement inutiles", et une colonne calculee partout n'est que du bruit.
from features_range import GROUPES as _GROUPES_RANGE
FEATURE_COLS_RANGE = [c for g in _GROUPES_RANGE.values() for c in g]

FEATURE_COLS = (FEATURE_COLS_TF + FEATURE_COLS_SUP
                + FEATURE_COLS_EXT + FEATURE_COLS_LIQ_TEMPS
                + FEATURE_COLS_RANGE)

N_BASE_FEATURES = len(FEATURE_COLS)

# Embedding de position, dans cet ordre exact :
#   position (-1/0/+1), unrealized_atr, bars_held_norm, last_risk_scale
N_POS_FEATURES = 4
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


def decide_avec_barres(p_buy: float, p_sell: float, barres) -> int:
    """0 = BUY, 1 = SELL, 2 = HOLD. SOURCE UNIQUE de la règle de décision."""
    if isinstance(barres, EntryDecisionPolicy):
        return barres.decide(p_buy, p_sell)
    ok_b = p_buy >= barres[0]
    ok_s = p_sell >= barres[1]
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
    if feats_ext is not None and len(feats_ext) > 0:
        fx = feats_ext[[c for c in FEATURE_COLS_EXT if c in feats_ext.columns]]
        merged = merged.merge(fx, left_on="time", right_index=True, how="left")

        chevauche = merged["time"].between(fx.index.min(), fx.index.max())
        if chevauche.any():
            # GARDE-FOU 1 — jointure totalement ratee. Un decalage d'horodatage
            # laisserait toutes les colonnes a NaN puis le dropna viderait le
            # dataframe SANS erreur. C'est le piege qui s'est deja referme sur
            # les ticks.
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

        # 2 x d_model : la tete recoit DEUX vues concatenees (cf. forward).
        self.norm = RMSNorm(2 * d_model)

        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.GELU(),
        )

        # Intersample attention, forme deployable : voir ReferenceMemory.
        # n_ref = 0 -> aucune memoire, le modele est strictement celui d'avant.
        self.memoire = (ReferenceMemory(2 * d_model, n_ref, heads, dropout,
                                        ls_init=ls_init) if n_ref > 0 else None)

        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)
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
        self.mlp = nn.Sequential(
            nn.Linear(n_features * d_model, mlp_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim), nn.GELU(),
        )
        self.actor = nn.Linear(mlp_dim, n_actions)
        self.critic = nn.Linear(mlp_dim, 1)

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


def build_policy(device, lookback: int = 25,
                 n_features: int = OBS_N_FEATURES,
                 n_ref: int = N_REF_DEFAUT,
                 num_blocks: int = N_BLOCS_DEFAUT,
                 archi: str = ARCHI_DEFAUT,
                 d_model_patch: int = 32,
                 mlp_dim: int = 128,
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
            lb = int(state_dict["pos"].shape[1] - 1) * 8 + 16   # patchs -> lookback
            return PatchTSTPolicy(
                n_features=n_features, lookback=lb,
                d_model=int(state_dict["pos"].shape[2]),
                mlp_dim=int(state_dict["mlp.0.weight"].shape[0]),
                n_actions=N_ACTIONS).to(device)
        if est_saint:
            archi = "saint"

        cle = "memoire.bank_repr"
        n_ref = int(state_dict[cle].shape[0]) if cle in state_dict else 0
        # Profondeur deduite elle aussi : elle etait ecrite en dur ici ET dans
        # training.py, deux endroits a tenir d'accord a la main.
        vus = {int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")}
        if vus:
            num_blocks = max(vus) + 1

    if archi == "patchtst":
        return PatchTSTPolicy(n_features=n_features, lookback=lookback,
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
        mask[2] = True  # En position : seulement HOLD (sortie par SL/TP)
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
        action = decide_avec_barres(*values, thresholds)
        if self.mode == 'rolling_rank':
            for rank, value in zip(self.ranks, values):
                rank.observe(value)
        return action


def rolling_decision_spec(fraction, window, bootstrap):
    spec = {'version': 1, 'mode': 'rolling_rank', 'ties': 'conservative',
            'fraction_per_side': float(fraction), 'window': int(window),
            'bootstrap': [[float(v) for v in side[-window:]] for side in bootstrap]}
    EntryDecisionPolicy(spec)  # Fail before saving an unusable checkpoint.
    return spec


def load_decision_policy(checkpoint, fallback=None):
    """Fresh state on every load; callers retain one instance per stream."""
    import json
    from pathlib import Path
    path = Path(checkpoint).with_suffix('').with_name(Path(checkpoint).stem + '_calib.json')
    if path.exists():
        with path.open(encoding='utf-8') as f:
            metadata = json.load(f)
        if 'decision_policy' in metadata:
            return EntryDecisionPolicy(metadata['decision_policy'])
    # Older checkpoints retain their documented fixed-threshold behavior.
    return EntryDecisionPolicy({'version': 1, 'mode': 'fixed',
                                'thresholds': list(load_calib_thresholds(checkpoint, fallback))})


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

    if side == 1:
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist
    else:
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist

    return max(sl, 1e-8), max(tp, 1e-8)


def compute_entry_atr(df_closed: pd.DataFrame) -> float:
    """Dernier ATR(14) clôturé — identique backtest / live / MQL5."""
    if "atr_14" not in df_closed.columns or len(df_closed) == 0:
        return 0.0
    val = df_closed["atr_14"].iloc[-1]
    return 0.0 if pd.isna(val) else float(val)
