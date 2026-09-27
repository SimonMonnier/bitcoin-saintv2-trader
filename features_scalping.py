# -*- coding: utf-8 -*-
"""Les features du scalping BTC M1 — ce que la recherche dit utile, et rien d'autre.

POURQUOI CE FICHIER EXISTE — 2026-09-26, demande du proprietaire : appliquer
au BTC ce que la recherche a trouve de meilleur, retirer l'Ichimoku et tout
ce qui ne sert a rien, repartir de zero. La recherche est dans
`reports/Stratégies de scalping et features PPO.md`.

CE QUE LA RECHERCHE A DIT, ET CE QUI EN DECOULE ICI :

  1. AUCUNE FAMILLE DE SCALPING FONDEE SUR LE SEUL PRIX ne depasse ~1.6 bps
     net a 1-15 minutes. L'Ichimoku et les figures de range sont des
     fonctions de l'OHLC : le depot a mesure qu'elles ne separaient rien
     (voir `prepare_m5`, detecteurs Ichimoku a +0.4 sigma). ELLES PARTENT.

  2. LE COUT ET LE REGIME RELATIFS A L'HEURE sont la famille la mieux
     etayee : un spread de 2 bps n'a pas le meme sens a 3 h du matin et a
     l'ouverture de New York. On les exprime donc en ECART A LA NORME DU
     MEME CRENEAU des jours precedents, jamais en niveau.

  3. L'HORLOGE DES EVENEMENTS : les grands mouvements de minute suivent les
     annonces macro americaines (8 h 30 et 10 h, heure de New York), la Fed
     (14 h), l'ouverture et la cloture des actions (9 h 30, 16 h). Pour le
     BTC s'y ajoute le cycle de financement des perpetuels (0 h, 8 h, 16 h
     UTC).

  4. LE FLUX SIGNE — la seule information du BTC qui ne soit pas du prix.
     `tick_desequilibre` ne portait AUCUNE direction (il compte des mises a
     jour de cotation, pas des transactions) : il part. Le desequilibre
     d'agression Binance, pondere par le volume, a 5, 15 et 60 minutes, le
     remplace.

  5. LES ANCRES DE PRIX : nombres ronds, rendement depuis l'ouverture du
     jour, position dans le range du jour, distance aux extremes de la
     veille.

  6. L'ECART ENTRE BINANCE ET LE COURTIER. Binance est le marche qui fait
     le prix ; le courtier le recopie avec retard. C'est la famille
     « arbitrage de latence » de la recherche, lue a la minute.

L'HEURE DU SERVEUR — VERIFIEE LE 2026-09-26. Le serveur MT5 de Vantage
tourne a l'heure de NEW YORK + 7 h, toute l'annee : la pause quotidienne de
l'or tombe a 00:00-01:00 serveur en janvier comme en juillet. Donc :

    heure de New York = heure serveur - 7 h        (exact, sans fuseau)
    UTC               = serveur - 2 h l'hiver, - 3 h l'ete (heure d'ete US)

CE QUE L'ANCIEN CACHE BTC FAISAIT FAUX, et c'est ce fichier qui l'a vu. Il
alignait Binance d'un decalage FIXE (+3 h, mesure sur l'echantillon recent,
donc en ete). Mesure mois par mois, correlation des rendements :

    2024-12  +2 h 0.992  +3 h 0.003        2025-06  +2 h 0.003  +3 h 0.989
    2025-11  +2 h 0.987  +3 h 0.010        2026-07  +2 h 0.004  +3 h 0.994

TOUT L'HIVER, LE FLUX AVAIT UNE HEURE DE RETARD — environ 40 % de
l'historique, sans que rien ne le signale. `aligne_binance` convertit
chaque minute avec le calendrier americain.

TOUTES LES COLONNES NE LISENT QUE LE PASSE : les normes de creneau sont
decalees d'une ligne dans leur groupe, les extremes du jour s'arretent a la
barre courante, ceux de la veille a la veille. `test_causalite_btc_m1.py`
le verifie en coupant le futur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Le serveur MT5 est a l'heure de New York + 7 h, toute l'annee.
DECALAGE_SERVEUR_NY_H = 7

# LES ANNONCES ET LES SEANCES, en minutes depuis minuit, heure de New York.
EVENEMENTS_NY = (
    ("t_annonce_0830", 8 * 60 + 30),    # emploi, inflation, PIB, ventes
    ("t_ouverture_0930", 9 * 60 + 30),  # ouverture des actions US
    ("t_annonce_1000", 10 * 60),        # ISM, confiance, JOLTS
    ("t_fed_1400", 14 * 60),            # decisions et minutes de la Fed
    ("t_cloture_1600", 16 * 60),        # cloture des actions US
)
# Au-dela d'une heure, un evenement est « loin » : la colonne sature.
FENETRE_EVENEMENT_MIN = 60

COLONNES_HORLOGE = ([n for n, _ in EVENEMENTS_NY]
                    + ["semaine_sin", "semaine_cos", "week_end",
                       "funding_sin", "funding_cos"])
COLONNES_REGIME = [
    "spread_bar",          # le cout reel de la barre, en bps
    "spread_rel_creneau",  # ce spread rapporte a la norme de son creneau
    "ticks_rel_creneau",   # activite de cotation, rapportee a son creneau
    "vol_rel_creneau",     # volatilite 15 min, rapportee a son creneau
    "vol_court_long",      # volatilite 15 min / volatilite 4 h
    "saut_ratio",          # plus gros rendement 1 min / volatilite 15 min
]
COLONNES_FLUX = [
    "taker_ratio",         # part achetee a l'agression, sur la minute
    "ofi_5",               # desequilibre d'agression signe, 5 min
    "ofi_15",              # le meme sur 15 min
    "ofi_60",              # le meme sur une heure
    "flux_taille_trade",   # taille moyenne d'un trade, contre sa normale
    "volume_rel_creneau",  # volume Binance, rapporte a son creneau
    "ecart_binance",       # courtier - Binance, contre sa normale, en bps
]
COLONNES_ANCRES = [
    "rend_jour",           # rendement depuis l'ouverture UTC, en ranges
    "pos_range_jour",      # position dans le range du jour, 0 a 1
    "dist_haut_veille",    # distance au plus haut d'hier, en ranges
    "dist_bas_veille",     # distance au plus bas d'hier, en ranges
    "dist_rond",           # distance au millier de dollars, en ranges
]
COLONNES_RANGS = ["creux_rang", "flux_rang", "creux_x_flux"]
# COINBASE — ajoutee le 2026-09-26 apres `mesure_sources_btc.py` : la seule
# des sources gratuites mesurees qui apporte quelque chose. Coinbase est en
# AVANCE d'environ une minute sur le courtier (correlation de rang avec la
# minute suivante +0.034, t = +12.6, train et validation du fold 1), et le
# signe tient a 15 et 60 minutes. Les perpetuels, la prime, les positions
# ouvertes et les ratios longs/shorts n'apportaient rien de stable.
COLONNES_COINBASE = [
    "prime_cb_dev_60",     # Coinbase - Binance, contre sa normale de l'heure
    "prime_cb_chg_15",     # la meme prime, variation sur 15 minutes
    "cb_ecart_courtier",   # Coinbase - courtier, contre sa normale de l'heure
]

# Normes de creneau : quart d'heure de la journee, semaine et week-end
# separes. 300 observations = 20 jours ouvres d'un creneau de 15 minutes.
_FEN_CRENEAU, _MIN_CRENEAU = 300, 60

# LA DUREE D'UNE BOUGIE, EN MINUTES — 2026-09-27, le jeu passe en M15.
#
# Toutes les fenetres de ce fichier etaient ecrites en LIGNES de M1 : 15,
# 60, 240, 1440, 20 000. Sur des bougies de 15 minutes, les memes nombres
# couvriraient quinze fois plus de temps, et les colonnes porteraient un nom
# qui ment. Elles sont donc exprimees en MINUTES et converties en lignes par
# `_n`. Avec `mpb = 1`, chaque nombre est inchange : le M1 est identique.


def _n(minutes: float, mpb: int, mini: int = 1) -> int:
    """Des minutes aux lignes, pour des bougies de `mpb` minutes."""
    return max(int(mini), int(round(float(minutes) / int(mpb))))


def horizons_ofi(mpb: int = 1):
    """Les trois horizons de l'OFI, en minutes. En M1 : 5, 15, 60. Plus
    court qu'une bougie n'a pas de sens : en M15, 15, 60 et 240."""
    return (5, 15, 60) if int(mpb) == 1 else (int(mpb), 4 * int(mpb), 16 * int(mpb))


def colonnes_flux(mpb: int = 1):
    """Les colonnes de flux, les noms d'OFI suivant leurs horizons REELS."""
    return (["taker_ratio"] + [f"ofi_{k}" for k in horizons_ofi(mpb)]
            + ["flux_taille_trade", "volume_rel_creneau", "ecart_binance"])
# Rangs glissants : deux semaines de M1. Voir `saint_core.FEATURE_COLS_RANGS`.
_FEN_RANG, _MIN_RANG = 20_000, 2_000
# Range journalier typique : moyenne des 20 derniers jours.
_JOURS_RANGE = 20


def serveur_depuis_utc(t_utc: pd.Series) -> pd.Series:
    """Heure serveur d'une minute UTC — le calendrier americain, pas un decalage fixe."""
    ny = (t_utc.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
          .dt.tz_localize(None))
    return ny + pd.Timedelta(hours=DECALAGE_SERVEUR_NY_H)


def utc_depuis_serveur(t: pd.Series) -> pd.Series:
    """L'inverse. L'heure repetee du passage a l'heure d'hiver est prise a
    ses voisines : elle n'existe qu'une fois par an."""
    ny = t - pd.Timedelta(hours=DECALAGE_SERVEUR_NY_H)
    loc = ny.dt.tz_localize("America/New_York", ambiguous="NaT",
                            nonexistent="NaT")
    utc = loc.dt.tz_convert("UTC").dt.tz_localize(None)
    decalage = (t - utc).ffill().bfill()
    return t - decalage


def aligne_binance(bn: pd.DataFrame) -> pd.DataFrame:
    """Les klines Binance (UTC) sur l'horloge du serveur, colonnes prefixees."""
    out = pd.DataFrame({
        "time": serveur_depuis_utc(pd.to_datetime(bn["time"])),
        "bn_close": bn["close"].astype(np.float64),
        "bn_volume": bn["volume"].astype(np.float64),
        "bn_quote_vol": bn["quote_vol"].astype(np.float64),
        "nb_trades": bn["nb_trades"].astype(np.float64),
        "taker_buy_base": bn["taker_buy_base"].astype(np.float64),
    })
    return out.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def aligne_coinbase(cb: pd.DataFrame) -> pd.DataFrame:
    """Les bougies Coinbase (UTC) sur l'horloge du serveur."""
    out = pd.DataFrame({
        "time": serveur_depuis_utc(pd.to_datetime(cb["time"])),
        "cb_close": cb["close"].astype(np.float64),
    })
    return out.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def _rel_creneau(x: pd.Series, creneau: pd.Series, mpb: int = 1) -> pd.Series:
    """x moins sa moyenne sur les ~20 derniers jours du MEME creneau.

    En M1, un creneau de 15 minutes compte 15 lignes par jour : 300 lignes
    font 20 jours. En M15 il n'en compte qu'une : 20 lignes font 20 jours.

    DECALEE D'UNE LIGNE DANS LE GROUPE : la ligne courante n'entre pas
    dans sa propre norme."""
    fen = max(20, _n(_FEN_CRENEAU, mpb))
    mini = max(4, _n(_MIN_CRENEAU, mpb))
    norme = x.groupby(creneau).transform(
        lambda s: s.shift(1).rolling(fen, min_periods=mini).mean())
    return x - norme


def horloge(d: pd.DataFrame, t: pd.Series = None) -> pd.DataFrame:
    """`t` : l'instant de la DECISION de chaque ligne. En M1 c'est `time` ;
    en M15, la derniere minute de la bougie, pas son ouverture."""
    t = d["time"] if t is None else t
    ny = t - pd.Timedelta(hours=DECALAGE_SERVEUR_NY_H)
    minute_ny = ny.dt.hour * 60 + ny.dt.minute
    ouvre = ny.dt.dayofweek < 5
    for nom, cible in EVENEMENTS_NY:
        # SIGNE : negatif avant l'evenement, positif apres, sature a une
        # heure. Le week-end n'a pas d'annonce : il est « loin apres ».
        x = (minute_ny - cible).clip(-FENETRE_EVENEMENT_MIN,
                                     FENETRE_EVENEMENT_MIN) / FENETRE_EVENEMENT_MIN
        d[nom] = x.where(ouvre, 1.0)
    utc = utc_depuis_serveur(t)
    phase = (utc.dt.dayofweek + (utc.dt.hour * 60 + utc.dt.minute) / 1440.0) / 7.0
    d["semaine_sin"] = np.sin(2 * np.pi * phase)
    d["semaine_cos"] = np.cos(2 * np.pi * phase)
    d["week_end"] = (utc.dt.dayofweek >= 5).astype(np.float64)
    # LE CYCLE DE FINANCEMENT DES PERPETUELS BINANCE : 0 h, 8 h, 16 h UTC.
    f = ((utc.dt.hour * 60 + utc.dt.minute) % 480) / 480.0
    d["funding_sin"] = np.sin(2 * np.pi * f)
    d["funding_cos"] = np.cos(2 * np.pi * f)
    return d


def regime(d: pd.DataFrame, creneau: pd.Series, mpb: int = 1) -> pd.DataFrame:
    sp = d["spread_bar"].clip(lower=1e-3)
    d["spread_rel_creneau"] = _rel_creneau(np.log(sp), creneau, mpb)
    # PAS DE « SAUT DE SPREAD » : mesure le 2026-09-26, le spread du
    # courtier ne bouge presque pas dans la minute (log max/moyen = 0.006,
    # ecart-type 0.0005). Une colonne quasi constante, une fois normalisee,
    # n'est que du bruit amplifie.
    d["ticks_rel_creneau"] = _rel_creneau(np.log1p(d["tick_n"]), creneau, mpb)
    r = np.log(d["close"]).diff()
    # FENETRE COURTE ET LONGUE : 15 et 240 minutes en M1. En M15, 15 minutes
    # ne feraient qu'une bougie : la courte passe a 4 bougies (1 h), la
    # longue a 64 (16 h).
    nc = _n(max(15, 4 * mpb), mpb)
    nl = _n(max(240, 64 * mpb), mpb)
    mc, ml = max(2, 2 * nc // 3), max(2, nl // 2)
    rvc = np.sqrt((r * r).rolling(nc, min_periods=mc).mean())
    rvl = np.sqrt((r * r).rolling(nl, min_periods=ml).mean())
    d["vol_rel_creneau"] = _rel_creneau(np.log(rvc + 1e-7), creneau, mpb)
    d["vol_court_long"] = np.log((rvc + 1e-7) / (rvl + 1e-7))
    # Entre 1/sqrt(n) — n barres egales — et 1 : une seule barre a tout fait.
    d["saut_ratio"] = (r.abs().rolling(nc, min_periods=mc).max()
                       / (np.sqrt((r * r).rolling(nc, min_periods=mc).sum()) + 1e-9))
    return d


def flux(d: pd.DataFrame, creneau: pd.Series, mpb: int = 1) -> pd.DataFrame:
    vb = d["bn_volume"]
    tb = d["taker_buy_base"]
    d["taker_ratio"] = (tb / vb.replace(0, np.nan)).clip(0, 1)
    # L'OFI SIGNE : achats agressifs moins ventes agressives, sur le volume.
    # Entre -1 (tout vendu a l'agression) et +1.
    signe = 2.0 * tb - vb
    for k in horizons_ofi(mpb):
        nk = _n(k, mpb)
        mk = max(2, nk // 2) if mpb == 1 else max(1, nk // 2)
        d[f"ofi_{k}"] = (signe.rolling(nk, min_periods=mk).sum()
                         / (vb.rolling(nk, min_periods=mk).sum() + 1e-12))
    taille = np.log((d["bn_quote_vol"]
                     / d["nb_trades"].replace(0, np.nan)).clip(lower=1e-9))
    d["flux_taille_trade"] = taille - taille.rolling(
        _n(1440, mpb), min_periods=_n(360, mpb)).mean()
    d["volume_rel_creneau"] = _rel_creneau(np.log1p(vb), creneau, mpb)
    # LE COURTIER CONTRE BINANCE. Le courtier cote un BID, Binance un
    # dernier echange : l'ecart a un niveau permanent. Seul son ecart a sa
    # normale de l'heure dit qui est en avance sur qui.
    e = (d["close"] / d["bn_close"] - 1.0) * 1e4
    d["ecart_binance"] = e - e.rolling(_n(60, mpb), min_periods=_n(30, mpb)).mean()
    return d


def coinbase(d: pd.DataFrame, mpb: int = 1) -> pd.DataFrame:
    """La demande americaine, et l'avance de Coinbase sur le courtier.

    COINBASE N'EMET PAS DE BOUGIE SANS TRANSACTION : une minute absente est
    une minute sans echange, donc au meme prix. Le dernier prix est
    prolonge, cinq minutes au plus.
    """
    cb = d["cb_close"].ffill(limit=_n(5, mpb))
    e = (cb / d["bn_close"] - 1.0) * 1e4
    d["prime_cb_dev_60"] = e - e.rolling(_n(60, mpb), min_periods=_n(30, mpb)).mean()
    d["prime_cb_chg_15"] = e - e.shift(_n(15, mpb))
    e = (cb / d["close"] - 1.0) * 1e4
    d["cb_ecart_courtier"] = e - e.rolling(_n(60, mpb), min_periods=_n(30, mpb)).mean()
    return d


def ancres(d: pd.DataFrame, t: pd.Series = None, unite_rond: float = 1000.0) -> pd.DataFrame:
    utc = utc_depuis_serveur(d["time"] if t is None else t)
    jour = utc.dt.floor("D")
    g = d.groupby(jour)
    ouverture = g["open"].transform("first")
    haut = g["high"].cummax()
    bas = g["low"].cummin()
    # LA VEILLE ET LE RANGE TYPIQUE ne lisent que des journees CLOSES.
    j = pd.DataFrame({"h": g["high"].max(), "l": g["low"].min()})
    j["h_veille"] = j["h"].shift(1)
    j["l_veille"] = j["l"].shift(1)
    j["typ"] = (j["h"] - j["l"]).shift(1).rolling(_JOURS_RANGE, min_periods=5).mean()
    typ = jour.map(j["typ"])
    c = d["close"]
    d["rend_jour"] = (c - ouverture) / typ
    largeur = (haut - bas)
    d["pos_range_jour"] = ((c - bas) / largeur.replace(0, np.nan)).fillna(0.5)
    d["dist_haut_veille"] = (c - jour.map(j["h_veille"])) / typ
    d["dist_bas_veille"] = (c - jour.map(j["l_veille"])) / typ
    # LE NOMBRE ROND DEPEND DU PRIX : le millier de dollars pour le BTC, la
    # centaine pour l'or ou le S&P 500. Voir `unite_ronde`.
    u = float(unite_rond)
    d["dist_rond"] = (((c + u / 2.0) % u) - u / 2.0) / typ
    return d


def unite_ronde(prix: float) -> float:
    """Le nombre rond d'un marche : la puissance de dix la plus proche de 1 %
    de son prix. BTC ~110 000 -> 1 000 ; or ~3 700 -> 100 ; S&P ~6 600 -> 100."""
    return float(10.0 ** np.round(np.log10(0.01 * float(prix))))


def rangs(d: pd.DataFrame, mpb: int = 1, avec_flux: bool = True) -> pd.DataFrame:
    # Voir `saint_core.FEATURE_COLS_RANGS` : le creux et le flux fort,
    # +11.95 bps a 480 minutes sur 20 mois de 23. Deux semaines, en lignes.
    fen, mini = _n(_FEN_RANG, mpb), _n(_MIN_RANG, mpb)
    d["creux_rang"] = (d["close_ema_dev"]
                       .rolling(fen, min_periods=mini).rank(pct=True))
    if not avec_flux:
        return d
    d["flux_rang"] = (d["taker_buy_base"]
                      .rolling(fen, min_periods=mini).rank(pct=True))
    # L'INTERACTION MESUREE, donnee telle quelle : fort quand le cours est
    # au fond ET que le volume agressif est fort.
    d["creux_x_flux"] = (1.0 - d["creux_rang"]) * d["flux_rang"]
    return d


def creneau_de(d: pd.DataFrame, t: pd.Series = None) -> pd.Series:
    """Le quart d'heure de la journee (heure de New York), semaine et week-end separes."""
    t = d["time"] if t is None else t
    ny = t - pd.Timedelta(hours=DECALAGE_SERVEUR_NY_H)
    q = (ny.dt.hour * 4 + ny.dt.minute // 15).astype(np.int64)
    we = (utc_depuis_serveur(t).dt.dayofweek >= 5).astype(np.int64)
    return q + 96 * we


def ajoute(d: pd.DataFrame, mpb: int = 1, sources: bool = True,
           unite_rond: float = 1000.0) -> pd.DataFrame:
    """Toutes les colonnes de ce fichier, sur un cadre deja passe par
    `prepare_m5.construit` (il faut `close_ema_dev`), en bougies de `mpb`
    minutes.

    L'HEURE D'UNE BOUGIE EST CELLE DE SON OUVERTURE ; la decision, elle, se
    prend a sa CLOTURE. L'horloge, les creneaux et les journees se lisent
    donc a la derniere minute de la bougie — en M1, c'est la meme.
    """
    d = d.sort_values("time").reset_index(drop=True)
    t = d["time"] + pd.Timedelta(minutes=int(mpb) - 1)
    cr = creneau_de(d, t)
    d = horloge(d, t)
    d = regime(d, cr, mpb)
    # SANS SOURCES : ni Binance ni Coinbase — elles n'existent que pour le
    # BTC. Le jeu multi-marches s'en passe pour tous les marches.
    if sources:
        d = flux(d, cr, mpb)
        d = coinbase(d, mpb)
    d = ancres(d, t, unite_rond)
    d = rangs(d, mpb, avec_flux=sources)
    return d


TOUTES = (COLONNES_HORLOGE + COLONNES_REGIME + COLONNES_FLUX
          + COLONNES_COINBASE + COLONNES_ANCRES + COLONNES_RANGS)
