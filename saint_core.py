"""Noyau partagé Loup Ω — modèle, features, normalisation, masque, SL/TP.

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

# Duree de detention de reference pour normaliser bars_held.
# Doit correspondre a la detention TYPIQUE, sinon la feature sature a 3.0 en
# permanence et ne porte plus d'information.
#
# Sur XAUUSD a SL 5xATR / R:R 2.0, 92.3 % des trades se resolvent dans les
# 240 min (mesure sur barriere triple). Le deplacement etant diffusif, atteindre
# 5 ATR demande de l'ordre de 5^2 = 25 min et 10 ATR une centaine ; la detention
# mediane se situe donc vers l'heure. On prend 120 pour que la feature garde de
# la dynamique jusqu'au plafond de 240 min sans saturer.
SCALPING_MAX_HOLDING = 120


# ============================================================
# INDICATEURS (M1 & H1)
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
#  JEU DE FEATURES — XAUUSD
# ============================================================
#
# Choisi sur mesure d'apport, hors echantillon, cible = ISSUE REELLE du trade
# (le TP est-il touche avant le SL, a SL 5xATR / R:R 2.0), sur 1.32 M bougies.
#
#   PRIX seules (8)                  AUC 0.5313   PnL top1% -0.0985
#   + argent + spread (13)           AUC 0.5309   PnL       +0.3789
#   + heure (15)                     AUC 0.5327   PnL       +0.4805   <- retenu
#   ELAGUE a 9 colonnes              AUC 0.5304   PnL       +0.3170
#
# DEUX LECONS de la mise au point, qui expliquent ce jeu :
#
# 1. Une feature n'est pas bonne ou mauvaise dans l'absolu, elle l'est pour une
#    CIBLE donnee. Les colonnes horaires coutaient -0.0022 d'AUC contre une
#    cible SL 3xATR / R:R 1.4, et rapportent +0.0018 contre SL 5xATR / R:R 2.0.
#    Tout elagage doit etre refait si le SL/TP change.
#
# 2. Les apports marginaux NE SE COMPOSENT PAS. Chacune des 9 colonnes retirees
#    ci-dessous avait un apport individuel nul ou negatif ; les retirer toutes
#    coutait 0.0024. Elles se couvrent mutuellement.
#
# L'INDICE DOLLAR (USDX) a ete ECARTE malgre une correlation de -0.345 avec
# l'or : son apport marginal etait exactement 0.0000, et il limitait la fusion a
# 1.22 M bougies contre 1.32 M sans lui. Les 8 % de donnees recuperees valent
# +0.0025 a +0.0041 d'AUC selon le jeu — mesure sur les quatre.

# JEU A 30 COLONNES — choisi sur mesure.
#
# Sonde logistique, 28 653 candidats de validation, friction complete sans
# commission (ce courtier n'en facture pas), esperance par unite de risque
# apres selection du meilleur 1 % :
#
#     jeu                AUC      E[R]        t
#      8 sans Binance   0.5450   -0.2102    -4.0
#     25 M1+H1 larges   0.5783   -0.0701    -1.4
#     10 (ancien jeu)   0.6072   +0.0346    +0.6    non distinguable de zero
#     27 + Binance      0.6191   +0.1249    +2.4
#     30 (celui-ci)     0.6215   +0.1751    +3.4
#
# Les DEUX colonnes Binance pesent plus que les 25 autres reunies : 25 features
# de prix seules donnent une esperance NEGATIVE ; on ajoute taker_ratio et
# ls_ratio_top et l'AUC passe de 0.5783 a 0.6191.
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
FEATURE_COLS_EXT = [
    "taker_ratio",       # desequilibre du flux agressif acheteur/vendeur
    "ls_ratio_top",      # ratio long/short des gros comptes
]

# LIQUIDITE ET TEMPS. J'avais exclu les features d'heure sur BTCUSD en invoquant
# une mesure a -0.0022 ; la mesure comparative dit le contraire sous cette
# configuration : 27 -> 30 colonnes fait passer l'esperance de +0.1249 a +0.1751.
# `spread_rel` est rapporte a sa normale d'une journee — brut, il encoderait
# l'ANNEE (1.74 ecart-type de derive sur huit ans, mesure sur l'or).
FEATURE_COLS_LIQ_TEMPS = ["spread_rel", "heure_sin", "heure_cos"]

FEATURE_COLS = (FEATURE_COLS_M1 + FEATURE_COLS_H1
                + FEATURE_COLS_EXT + FEATURE_COLS_LIQ_TEMPS)

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

class GatedFFN(nn.Module):
    def __init__(self, d: int, mult: int = 2, dropout: float = 0.05):
        super().__init__()
        inner = d * mult
        self.lin1 = nn.Linear(d, inner * 2)
        self.lin2 = nn.Linear(inner, d)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, gate = self.lin1(h).chunk(2, dim=-1)
        h = a * torch.sigmoid(gate)
        h = self.lin2(self.dropout(h))
        return x + h


class ColumnAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F, D = x.shape
        h = x.reshape(B * T, F, D)
        h2 = self.norm(h)
        out, _ = self.attn(h2, h2, h2)
        h = h + self.drop(out)
        return h.reshape(B, T, F, D)


class RowAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F, D = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * F, T, D)
        h2 = self.norm(h)
        out, _ = self.attn(h2, h2, h2)
        h = h + self.drop(out)
        h = h.reshape(B, F, T, D).permute(0, 2, 1, 3)
        return h


class SAINTv2Block(nn.Module):
    """Un tour de melange : sur le TEMPS, puis sur les COLONNES, puis un FFN.

    L'ancienne version enchainait SIX modules — ra1, ff1, ra2, ff2, ca, ff3 —
    soit DEUX attentions sur le temps par bloc, pour une seule sur les colonnes.
    Une conception a double axe en fait normalement une de chaque : la seconde
    RowAttention remelangeait le meme axe sans rien croiser de nouveau.

    Mesure a batch 16, 34 features, GPU libre :
      2 blocs x (ra, ff, ra, ff, ca, ff)   482 836 params   7.8 ms
      2 blocs x (ra, ca, ff)               274 836 params   4.5 ms   -42 %
      1 bloc  x (ra, ff, ra, ff, ca, ff)   287 716 params   4.1 ms   -48 %

    On retient la deuxieme : presque le meme gain que la troisieme, mais elle
    conserve DEUX tours de melange croise temps/colonnes, ce qui compte avec 30
    features la ou un seul tour ne laisse chaque colonne en voir les autres
    qu'une fois.
    """

    def __init__(self, d: int, heads: int, dropout: float, mult: int):
        super().__init__()
        self.ra = RowAttention(d, heads, dropout)
        self.ca = ColumnAttention(d, heads, dropout)
        self.ff = GatedFFN(d, mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ra(x)
        x = self.ca(x)
        x = self.ff(x)
        return x


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
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_actions = n_actions

        self.input_proj = nn.Linear(1, d_model)
        self.scale = math.sqrt(d_model)
        self.row_emb = nn.Embedding(max_len, d_model)
        self.col_emb = nn.Embedding(n_features, d_model)

        self.blocks = nn.ModuleList([
            SAINTv2Block(d_model, heads, dropout, ff_mult)
            for _ in range(num_blocks)
        ])

        # 2 x d_model : la tete recoit DEUX vues concatenees (cf. forward).
        self.norm = nn.LayerNorm(2 * d_model)

        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 3, f"Input x must be (B,T,F), got {x.shape}"
        B, T, F = x.shape

        tok = self.input_proj(x.unsqueeze(-1)) * self.scale

        rows = torch.arange(T, device=x.device).view(1, T, 1).expand(B, T, F)
        cols = torch.arange(F, device=x.device).view(1, 1, F).expand(B, T, F)

        tok = tok + self.row_emb(rows) + self.col_emb(cols)

        for blk in self.blocks:
            tok = blk(tok)

        # LECTURE A DEUX VUES — corrigee.
        #
        # L'ancienne version calculait :
        #     cls_time = tok.mean(dim=1).mean(dim=1)   # moyenne T puis F
        #     cls_feat = tok.mean(dim=2).mean(dim=1)   # moyenne F puis T
        #     h = cls_time + cls_feat
        # Or moyenner sur T puis sur F donne exactement la moyenne sur (T,F),
        # dans les deux ordres : les deux vues etaient le MEME tenseur (verifie
        # numeriquement, ecart 4.5e-08), leur somme valait 2x la moyenne
        # globale, et le facteur 2 etait absorbe par le LayerNorm qui suit.
        #
        # Toute la lecture a double axe se reduisait donc a une moyenne plate.
        # Apres avoir fait travailler l'attention sur le temps ET sur les
        # colonnes, la tete jetait l'information de POSITION : quel instant,
        # quelle feature.
        #
        # On garde deux vues REELLEMENT distinctes :
        #   - le resume global de la fenetre ;
        #   - la DERNIERE bougie, celle sur laquelle la decision se prend.
        # Elles sont concatenees et non additionnees, pour que le MLP puisse
        # les ponderer au lieu de les confondre.
        h_feat = tok.mean(dim=2)              # (B, T, D) : un resume par instant
        cls_global = h_feat.mean(dim=1)       # (B, D) : moyenne de la fenetre
        cls_recent = h_feat[:, -1, :]         # (B, D) : l'instant de decision

        h = torch.cat([cls_global, cls_recent], dim=-1)
        h = self.norm(h)
        h = self.mlp(h)

        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)

        return logits, value


def build_policy(device, lookback: int = 25,
                 n_features: int = OBS_N_FEATURES) -> SAINTPolicySingleHead:
    """Instancie la policy avec les hyperparamètres d'architecture du training.

    Toute divergence ici rend les checkpoints incompatibles : passer par cette
    fabrique plutôt que de recopier les arguments.
    """
    return SAINTPolicySingleHead(
        n_features=n_features,
        d_model=80,
        num_blocks=2,
        heads=4,
        dropout=0.05,
        ff_mult=2,
        max_len=lookback,
        n_actions=N_ACTIONS,
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
        if amorce is not None:
            for v in amorce:
                self._vus.append(float(v))

    def pret(self) -> bool:
        return len(self._vus) >= self._minimum

    def seuil(self) -> float:
        """Niveau courant correspondant a la fraction visee."""
        if not self.pret():
            return float("inf")
        return float(np.quantile(self._vus, 1.0 - self.fraction))

    def accepte(self, conviction: float) -> bool:
        """Cette conviction est-elle dans le haut du rang glissant ?

        On decide AVANT d'enregistrer : une occasion ne doit pas participer au
        quantile qui la juge.
        """
        ok = self.pret() and float(conviction) >= self.seuil()
        self._vus.append(float(conviction))
        return bool(ok)

    def observe(self, conviction: float) -> None:
        """Enregistre une conviction sans decider (occasions non evaluees)."""
        self._vus.append(float(conviction))


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
