"""Construit le jeu d'entrainement H1 — neuf ans, une seule source.

POURQUOI H1. La friction est un montant FIXE (~37 $ sur BTCUSD) ; l'unite de
risque est la distance au stop, qui grandit avec l'echelle. Mesure de la
strategie "bornes d'un range", simulee telle qu'ecrite, une position a la fois,
sans chevauchement :

        UT    friction    E[R]
        M1      1.05 R   -1.69
        M5      0.40 R   -0.66
       M15      0.21 R   -0.46
        H1      0.09 R   -0.27
        H4      0.04 R   +0.12

Confirme autrement le 2026-09-15 : entrer au hasard coute -0.45 R en M1 et
-0.012 R en H1. Le changement d'echelle a fait ce qu'on lui demandait.

POURQUOI NEUF ANS ET PLUS TROIS ET DEMI. Le H1 tire de MT5 ne remontait qu'a
fevrier 2023, soit 890 occasions sans chevauchement dans la fenetre
d'entrainement, 143 par bloc de validation. Or detecter un avantage de 0.02 R
quand un trade a un ecart-type de 1.4 R demande (1.4/0.02)^2 ~ 4 900 trades.
On etait a un facteur trente de pouvoir MESURER l'avantage cherche, et c'est
ce qui a fait echouer TabM : son gain apparent de +0.030 R venait d'un seul
bloc sur quatre, celui ou acheter aveuglement rapportait deja +0.135 R.

La source est donc les archives Binance spot, qui remontent a aout 2017 :
79 453 barres au lieu de 30 379, ~2 300 occasions d'entrainement au lieu de
890. Voir telecharge_h1_long.py pour le detail du choix (spot plutot que
futures, et ce que ce choix coute).

QUATRE PIEGES TRAITES ICI, chacun deja paye une fois dans ce projet.

  1. FENETRES A L'ECHELLE. `add_indicators` code en dur rolling(1440), qui vaut
     UNE JOURNEE sur M1 et SOIXANTE sur H1. Les colonnes porteraient leur nom
     sans mesurer ce qu'il annonce, et le warmup mangerait le debut de
     l'historique. On recalcule donc tout avec la journee de chaque echelle :
     24 barres en H1, 6 en H4.

  2. CAUSALITE DU CONTEXTE SUPERIEUR. Le bloc H4 est decale de shift(1) : seule
     la derniere bougie H4 CLOSE entre. Sans ce decalage, merge_asof choisit la
     bougie en formation et injecte jusqu'a 4 heures de futur par ligne.

  3. NIVEAUX ABSOLUS SUR NEUF ANS. Le bitcoin passe de 4 000 $ a plus de
     100 000 $ sur la periode, et le nombre de trades par heure est sans
     commune mesure entre 2017 et 2026. Toute colonne exprimee en niveau
     absolu apprendrait la DATE. Les features de flux sont donc en rang
     glissant ou rapportees a leur propre normale recente.

  4. COLONNE CONSTANTE SUR UNE PARTIE DU JEU. C'est la meme fuite, en pire :
     `spread_rel` n'existait que sur les trois dernieres annees. Il est retire,
     pas constante. Voir la note dans saint_core.FEATURE_COLS_LIQ_TEMPS.

    python prepare_h1.py
"""

import numpy as np
import pandas as pd

from features_ichimoku import ajoute_features_ichimoku
from features_range import ajoute_features_range

SOURCE = "klines_h1_spot_BTCUSDT.pkl"
SORTIE = "data_cache_BTCUSD_H1.pkl"
# Jeu enrichi des 47 colonnes couvrant TOUTES les strategies Ichimoku,
# pas seulement le range : Chikou-Span (le filtre du systeme), replis sur
# plat Kijun, cassures de plus hauts, espace tradable, boussole du nuage
# et les 14 structures en chandeliers. Ecrit A PART tant que le gain
# n'est pas mesure : ajouter 47 colonnes a 56 sur 55 000 barres
# d'entrainement n'est pas neutre, et ce depot a deja paye assez cher
# les changements adoptes avant mesure.
SORTIE_COMPLET = "data_cache_BTCUSD_H1_complet.pkl"
JOUR_H1, JOUR_H4 = 24, 6
SEMAINE_H1 = 168

BASES = ["rsi_14", "returns", "vol_20", "range_norm", "open_rel", "high_rel",
         "low_rel", "close_ema_dev", "mom_5", "rsi_ok", "vol_rank",
         "high_vol_regime"]


def indicateurs(d, jour):
    """Indicateurs standards, fenetre "journee" passee en PARAMETRE."""
    h, l, c, o = d["high"], d["low"], d["close"], d["open"]
    delta = c.diff()
    ag = delta.clip(lower=0).rolling(14).mean()
    al = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi_14"] = 100 - 100 / (1 + ag / (al + 1e-8))
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    d["atr_14"] = tr.rolling(14).mean()
    d["returns"] = c.pct_change()
    d["vol_20"] = d["returns"].rolling(20).std()
    rn = (h - l) / (c + 1e-8)
    mini = max(2, jour // 4)
    d["range_norm"] = rn / (rn.rolling(jour, min_periods=mini).mean() + 1e-12)
    d["open_rel"] = (o - c) / (c + 1e-8)
    d["high_rel"] = (h - c) / (c + 1e-8)
    d["low_rel"] = (l - c) / (c + 1e-8)
    d["close_ema_dev"] = c / (c.ewm(span=60, adjust=False).mean() + 1e-8) - 1.0
    d["mom_5"] = (c > c.shift(5)).astype(float)
    d["rsi_ok"] = ((d["rsi_14"] > 28) & (d["rsi_14"] < 72)).astype(float)
    d["vol_rank"] = d["vol_20"].rolling(jour, min_periods=mini).rank(pct=True)
    d["high_vol_regime"] = (d["vol_rank"] > 0.65).astype(float)
    return d


def flux(d):
    """Flux d'ordres, ecrit pour rester comparable de 2017 a 2026.

    Aucune de ces colonnes n'est un niveau : la part acheteuse est deja un
    rapport, la taille de trade est rapportee a sa normale de la journee, et
    l'intensite est un RANG sur la semaine. Un modele ne peut pas y lire
    l'annee, ce qui est exactement le but sur un historique ou le volume
    horaire a ete multiplie par plusieurs ordres de grandeur.
    """
    vol = d["volume"].replace(0, np.nan)
    d["taker_ratio"] = (d["taker_buy_base"] / vol).clip(0, 1)
    d["taker_ma5"] = d["taker_ratio"].rolling(5, min_periods=1).mean()

    taille = d["quote_vol"] / d["nb_trades"].replace(0, np.nan)
    lt = np.log(taille.clip(lower=1e-9))
    d["flux_taille_trade"] = lt - lt.rolling(JOUR_H1, min_periods=6).mean()

    d["flux_intensite"] = (d["nb_trades"]
                           .rolling(SEMAINE_H1, min_periods=24).rank(pct=True))
    return d


def main() -> int:
    src = pd.read_pickle(SOURCE)
    src["time"] = pd.to_datetime(src["time"])
    h1 = src.sort_values("time").reset_index(drop=True)
    print(f"source : {len(h1):,} bougies H1  "
          f"{h1['time'].iloc[0]} -> {h1['time'].iloc[-1]}")

    h1 = indicateurs(h1, JOUR_H1)
    h1 = flux(h1)

    # ---------- H4, le contexte superieur ----------
    h4 = (h1.set_index("time")[["open", "high", "low", "close"]]
            .resample("4h")
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last"})
            .dropna().reset_index())
    h4 = indicateurs(h4, JOUR_H4)
    cols_h4 = [c + "_h4" for c in BASES]
    dec = h4[["time"] + BASES].copy()
    dec[BASES] = dec[BASES].shift(1)        # derniere bougie H4 CLOSE
    dec.columns = ["time"] + cols_h4
    dec["_c_h4"] = h4["close"].shift(1).values
    h1 = pd.merge_asof(h1.sort_values("time"), dec.sort_values("time"),
                       on="time", direction="backward")
    h1["close_h4_dev"] = h1["close"] / (h1["_c_h4"] + 1e-8) - 1.0
    h1 = h1.drop(columns=["_c_h4"])
    cols_h4 = cols_h4 + ["close_h4_dev"]

    # ---------- Temps ----------
    heure = h1["time"].dt.hour + h1["time"].dt.minute / 60.0
    h1["heure_sin"] = np.sin(2 * np.pi * heure / 24.0)
    h1["heure_cos"] = np.cos(2 * np.pi * heure / 24.0)

    # ---------- Features de la strategie de range ----------
    h1, cols_rng = ajoute_features_range(h1)

    # ---------- Features des AUTRES strategies Ichimoku ----------
    h1, cols_ich = ajoute_features_ichimoku(h1)

    ext = ["taker_ratio", "taker_ma5"]
    liq = ["flux_taille_trade", "flux_intensite", "heure_sin", "heure_cos"]
    colonnes = BASES + cols_h4 + ext + liq + cols_rng + cols_ich

    avant = len(h1)
    h1 = h1.replace([np.inf, -np.inf], np.nan)
    h1 = h1.dropna(subset=colonnes + ["atr_14"]).reset_index(drop=True)
    print(f"H1     : {avant:,} -> {len(h1):,} bougies apres dropna "
          f"({100*(1-len(h1)/avant):.1f} % perdues au warmup)")
    print(f"periode: {h1['time'].iloc[0]} -> {h1['time'].iloc[-1]}")
    print(f"colonnes : {len(BASES)} H1 + {len(cols_h4)} H4 + {len(ext)} flux"
          f" + {len(liq)} liquidite/temps + {len(cols_rng)} range "
          f"+ {len(cols_ich)} ichimoku "
          f"= {len(colonnes)}")

    # La liste de saint_core fait foi : si les deux divergent, l'entrainement
    # echouerait plus tard avec un KeyError peu lisible.
    from saint_core import FEATURE_COLS
    manquantes = [c for c in FEATURE_COLS if c not in h1.columns]
    en_trop = [c for c in colonnes if c not in FEATURE_COLS]
    if manquantes or en_trop:
        print("\nDESACCORD avec saint_core.FEATURE_COLS")
        print(f"  absentes du jeu   : {manquantes}")
        print(f"  produites en trop : {en_trop}")
        return 1
    print(f"accord avec saint_core.FEATURE_COLS ({len(FEATURE_COLS)} colonnes)")

    atr_med = float(h1["atr_14"].median())
    prix_med = float(h1["close"].median())
    fric = (2.61 + 1.0 + 2.0) / 1e4 * prix_med
    print(f"\nATR median {atr_med:.2f} $  |  friction {fric:.2f} $  "
          f"=  {fric/atr_med:.3f} R par ATR")

    # Le jeu COMPLET garde les deux blocs, le jeu de reference n'a
    # que les colonnes en service. Les deux partagent exactement les
    # memes lignes, donc toute comparaison entre eux est APPARIEE
    # barre a barre — c'est ce qui permet de lire un ecart sans que
    # la difference de periode vienne s'y melanger.
    complet = h1.replace([np.inf, -np.inf], np.nan)
    complet = complet.dropna(subset=cols_ich).reset_index(drop=True)
    complet.to_pickle(SORTIE_COMPLET)
    print()
    print(f'{SORTIE_COMPLET} ecrit : {complet.shape}')
    print(f'  +{len(cols_ich)} colonnes ich_*, {len(h1)-len(complet)} lignes perdues au warmup supplementaire')

    h1.to_pickle(SORTIE)
    print(f"\n{SORTIE} ecrit : {h1.shape}")

    n_tr = int(len(h1) * 0.70)
    print(f"\nDIMENSIONNEMENT : {len(h1):,} barres, {n_tr:,} en entrainement.")
    print(f"  occasions sans chevauchement (une par jour) : ~{n_tr // 24:,}"
          f"   (jeu MT5 precedent : 890)")
    print(f"  episodes de 400 barres : {n_tr // 400:,} episodes disjoints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
