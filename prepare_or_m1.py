# -*- coding: utf-8 -*-
"""Le cache M1 de l'or : barres, SPREAD REEL, et microstructure des ticks.

CE QUI A CHANGE LE 2026-09-21, ET POURQUOI.

Les 259 colonnes de l'observation sont TOUTES des transformations de
l'OHLC. Mesure du soir, occasions disjointes et plancher par rotation :
aucune ne porte de direction, a aucun horizon de 1 a 60 minutes. Et le
modele entraine faisait pire que pile ou face sur le choix du cote —
-1.235 bps contre +0.011 pour un tirage, soit 5.1 ecarts-types.

Ajouter un indicateur de prix de plus n'y changera rien : ce serait la
260e fonction du meme OHLC. Ce fichier va donc chercher ce que MT5 donne
D'AUTRE, et que le depot jetait :

  LE SPREAD PAR BARRE. `copy_rates` le renvoie depuis toujours ; la
  version precedente selectionnait six colonnes et le laissait tomber. Il
  dit l'incertitude du teneur de marche en temps reel.

  LES TICKS. `copy_ticks_range` rend bid et ask a chaque cotation. On en
  tire, PAR MINUTE, ce que le prix ne contient pas : combien de fois le
  marche a cote, de quel cote du carnet le mouvement est venu, et
  l'ecart bid-ask reellement pratique.

CE QU'ON NE PEUT PAS EN TIRER, et il faut l'ecrire. Le champ `last` vaut
zero sur ce CFD : ce sont des COTATIONS, pas des transactions. Il n'y a
donc aucune classification acheteur/vendeur au sens strict — pas de
volume a l'ask contre volume au bid. Ce qui reste est l'asymetrie des
MISES A JOUR du carnet, un proxy plus faible mais qui n'est pas dans le
prix.

LE COUT EST MODESTE : les ticks sont en cache local chez MT5 et sortent a
onze millions par seconde. Les 402 millions de ticks de l'historique ne
tiennent pas en memoire, donc on agrege mois par mois et on jette le brut.

    python prepare_or_m1.py [n_barres]
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pandas as pd

import prepare_m5

SYMBOLE = "XAUUSD"
SORTIE = "data_cache_XAUUSD_M1.pkl"
# LE BRUT, garde a part : barres, spread reel, agregat de ticks. Il permet de
# reconstruire les features sans MetaTrader 5 quand on change de jeu.
BRUT = "brut_XAUUSD_M1.pkl"

# LES CONTEXTES SUPERIEURS DE L'OR M1 — 2026-09-26, choix du proprietaire :
# M1, M5 et M15, et plus de H1 ni de H4. (regle de resample, suffixe,
# bougies par journee a cette echelle.)
ECHELLES_OR_M1 = [("5min", "_m5", 288), ("15min", "_m15", 96)]
JOUR_M1 = 1440
N_BARRES = 900_000

# LES COLONNES DE MICROSTRUCTURE, nommees une fois ici et lues par
# `saint_core.FEATURE_COLS`. Les ajouter ailleurs les ferait diverger.
COLONNES_TICKS = [
    "spread_bar",        # le spread de la barre, en points de base
    "tick_n",            # nombre de cotations dans la minute
    "tick_spread_moy",   # ecart bid-ask moyen, en bps
    "tick_spread_max",   # son maximum
    "tick_ask_part",     # part des cotations ou SEUL l'ask a bouge
    "tick_bid_part",     # part des cotations ou SEUL le bid a bouge
    "tick_desequilibre", # ask_part - bid_part : la pression de carnet
]


def charge_barres(symbole: str, n: int) -> pd.DataFrame:
    """Les barres M1, SPREAD COMPRIS."""
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 indisponible : {mt5.last_error()}")
    mt5.symbol_select(symbole, True)
    r = mt5.copy_rates_from_pos(symbole, mt5.TIMEFRAME_M1, 0, n)
    mt5.shutdown()
    if r is None or len(r) == 0:
        raise RuntimeError(f"aucune barre M1 pour {symbole}")
    d = pd.DataFrame(r)
    d["time"] = pd.to_datetime(d["time"], unit="s")
    # MEME SUBSTITUTION QU'EN M5 : l'or n'a pas de volume echange publie
    # par le courtier, et aucune colonne ne lit le volume en NIVEAU.
    d["volume"] = d["tick_volume"].astype(float)
    # LE SPREAD EST EN POINTS DU SYMBOLE, on le rend en POINTS DE BASE —
    # la seule unite comparable d'un instrument a l'autre. `point` vaut
    # 0.01 sur l'or ; on le deduit du prix plutot que de le supposer.
    pas = 0.01
    d["spread_bar"] = d["spread"].astype(float) * pas / d["close"] * 1e4
    return d[["time", "open", "high", "low", "close", "volume", "spread_bar"]]


def agrege_ticks(symbole: str, debut: dt.datetime, fin: dt.datetime,
                 pas_jours: int = 20) -> pd.DataFrame:
    """Les ticks, resumes PAR MINUTE. Rend un cadre indexe sur la minute.

    ON AGREGE PAR TRANCHES et on jette le brut : 402 millions de ticks
    ne tiennent pas en memoire, leur resume par minute tient largement.

    LES DRAPEAUX DISENT QUEL COTE A BOUGE. MT5 pose le bit 2 quand le bid
    change et le bit 4 quand l'ask change. Une cotation ou SEUL l'ask
    monte n'est pas la meme chose qu'une ou seul le bid descend : c'est
    l'asymetrie la plus proche d'un flux d'ordres qu'un CFD sans champ
    `last` permette d'obtenir.
    """
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 indisponible : {mt5.last_error()}")
    mt5.symbol_select(symbole, True)
    morceaux = []
    curseur = debut
    n_total = 0
    while curseur < fin:
        borne = min(curseur + dt.timedelta(days=pas_jours), fin)
        t = mt5.copy_ticks_range(symbole, curseur, borne, mt5.COPY_TICKS_ALL)
        curseur = borne
        if t is None or len(t) == 0:
            continue
        n_total += len(t)
        bid = t["bid"].astype(np.float64)
        ask = t["ask"].astype(np.float64)
        bon = (bid > 0) & (ask > 0) & (ask >= bid)
        if not bon.any():
            continue
        f = t["flags"][bon]
        sp = (ask[bon] - bid[bon]) / bid[bon] * 1e4
        minute = pd.to_datetime(t["time"][bon], unit="s").floor("min")
        _b = (f & 2) > 0
        _a = (f & 4) > 0
        g = pd.DataFrame({
            "time": minute,
            "sp": sp,
            # SEUL l'ask, ou SEUL le bid : les cotations ou les deux
            # bougent ne portent aucune asymetrie et ne comptent ni d'un
            # cote ni de l'autre.
            "ask_seul": (_a & ~_b).astype(np.float64),
            "bid_seul": (_b & ~_a).astype(np.float64),
        }).groupby("time").agg(
            tick_n=("sp", "size"),
            tick_spread_moy=("sp", "mean"),
            tick_spread_max=("sp", "max"),
            tick_ask_part=("ask_seul", "mean"),
            tick_bid_part=("bid_seul", "mean"),
        )
        morceaux.append(g)
        print(f"    {borne:%Y-%m-%d}  {n_total/1e6:7.1f} M ticks cumules",
              flush=True)
    mt5.shutdown()
    if not morceaux:
        raise RuntimeError("aucun tick recupere")
    out = pd.concat(morceaux)
    out = out.groupby(level=0).agg({
        "tick_n": "sum", "tick_spread_moy": "mean", "tick_spread_max": "max",
        "tick_ask_part": "mean", "tick_bid_part": "mean"})
    out["tick_desequilibre"] = out["tick_ask_part"] - out["tick_bid_part"]
    return out.reset_index()


def brut_depuis_cache() -> pd.DataFrame:
    """Le brut, relu sans MetaTrader 5 : `BRUT` s'il existe, sinon l'ancien cache.

    L'ANCIEN CACHE PORTE DEJA TOUT LE BRUT — barres, spread de chaque barre,
    agregat de ticks — a cote des features qu'on veut recalculer. On en
    extrait ces colonnes-la, on les ecrit dans `BRUT`, et l'ancien cache est
    renomme plutot qu'ecrase : ses features H1/H4 restent disponibles.
    """
    import os
    garder = ["time", "open", "high", "low", "close", "volume"] + COLONNES_TICKS
    if os.path.exists(BRUT):
        b = pd.read_pickle(BRUT)
        print(f"brut : {len(b):,} barres relues depuis {BRUT}", flush=True)
        return b
    ancien = pd.read_pickle(SORTIE)
    manque = [c for c in garder if c not in ancien.columns]
    if manque:
        raise RuntimeError(f"{SORTIE} ne porte pas le brut : {manque}")
    b = ancien[garder].copy()
    del ancien
    b.to_pickle(BRUT)
    os.replace(SORTIE, SORTIE.replace(".pkl", "_h1h4.pkl"))
    print(f"brut : {len(b):,} barres extraites de l'ancien cache -> {BRUT} ; "
          f"l'ancien cache est garde sous "
          f"{SORTIE.replace('.pkl', '_h1h4.pkl')}", flush=True)
    return b


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--depuis-cache":
        m1 = brut_depuis_cache().sort_values("time").reset_index(drop=True)
        return construit_et_ecrit(m1)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_BARRES
    m1 = charge_barres(SYMBOLE, n).sort_values("time").reset_index(drop=True)
    print(f"barres : {len(m1):,} M1  {m1['time'].iloc[0]} -> {m1['time'].iloc[-1]}",
          flush=True)
    print(f"  spread des barres : med {m1['spread_bar'].median():.3f} bps  "
          f"p90 {m1['spread_bar'].quantile(0.9):.3f}  "
          f"p99 {m1['spread_bar'].quantile(0.99):.3f}", flush=True)

    # L'AGREGAT DE TICKS EST MIS EN CACHE A PART. Il coute quatre minutes
    # de telechargement et d'agregation ; le refaire a chaque essai de la
    # suite du fichier est du temps jete pour rien.
    import os
    cache_tk = "cache_ticks_m1_%s.pkl" % SYMBOLE
    deb = m1["time"].iloc[0].to_pydatetime()
    fin = m1["time"].iloc[-1].to_pydatetime() + dt.timedelta(minutes=1)
    if os.path.exists(cache_tk):
        tk = pd.read_pickle(cache_tk)
        if (tk["time"].min() <= pd.Timestamp(deb)
                and tk["time"].max() >= pd.Timestamp(fin) - dt.timedelta(minutes=2)):
            print(f"ticks : {len(tk):,} minutes relues depuis {cache_tk}",
                  flush=True)
        else:
            tk = None
    else:
        tk = None
    if tk is None:
        print("ticks : agregation par minute", flush=True)
        tk = agrege_ticks(SYMBOLE, deb, fin)
        tk.to_pickle(cache_tk)
    print(f"  {len(tk):,} minutes couvertes par les ticks", flush=True)

    m1 = m1.merge(tk, on="time", how="left")
    # LES MINUTES SANS TICK SONT DES MINUTES SANS COTATION : zero cotation,
    # et un spread qu'on prend a celui de la barre plutot qu'a NaN. Un NaN
    # ferait tomber la ligne au `dropna` et trouerait l'historique.
    m1["tick_n"] = m1["tick_n"].fillna(0.0)
    for c in ("tick_spread_moy", "tick_spread_max"):
        m1[c] = m1[c].fillna(m1["spread_bar"])
    for c in ("tick_ask_part", "tick_bid_part", "tick_desequilibre"):
        m1[c] = m1[c].fillna(0.0)
    manquant = float((m1["tick_n"] == 0).mean())
    print(f"  minutes sans tick : {100*manquant:.2f} %", flush=True)
    m1[["time", "open", "high", "low", "close", "volume"]
       + COLONNES_TICKS].to_pickle(BRUT)
    return construit_et_ecrit(m1)


def construit_et_ecrit(m1: pd.DataFrame) -> int:
    """Du brut M1 au cache : features M1, contextes M5 et M15, rangs."""
    # `construit` REND UN TUPLE (df, colonnes, ichimoku), pas un cadre.
    d, _cols, _ich = prepare_m5.construit(m1, avec_flux=False, jour=JOUR_M1,
                                          echelles=ECHELLES_OR_M1)
    # `construit` MODIFIE SON ARGUMENT ET LE REND : `d` EST `m1`.
    #
    # Une premiere version faisait `d.merge(m1[...])` pour recoller les
    # colonnes de microstructure. Sur un cadre fusionne avec lui-meme,
    # pandas suffixe les colonnes communes en `_x` et `_y` — et la ligne
    # suivante tombait sur un `KeyError: 'spread_bar'`. Les colonnes sont
    # deja la ; il n'y a rien a recoller.
    #
    # ON VERIFIE PLUTOT QU'ON NE SUPPOSE : si `construit` venait un jour a
    # rendre une copie, l'absence se verrait ici et non trois heures plus
    # tard dans un run entraine sur des zeros.
    manquantes = [c for c in COLONNES_TICKS if c not in d.columns]
    if manquantes:
        raise RuntimeError(
            f"colonnes de microstructure perdues par `construit` : "
            f"{manquantes}. Elles doivent survivre au calcul des features.")
    for c in COLONNES_TICKS:
        d[c] = d[c].astype(np.float32).fillna(0.0)

    # LE FILTRE DE CHAUFFE, ET IL ETAIT TOMBE EN REECRIVANT CE FICHIER.
    #
    # `construit` calcule les colonnes mais ne filtre PAS : le `dropna` vit
    # dans le `main` de `prepare_m5`, que ce fichier n'appelle pas. La
    # premiere version rendait donc 900 000 lignes dont les ~20 000
    # premieres portent des NaN — `tend_mom_mois` seule en demande 8 640 de
    # chauffe, `ich_*_h4` davantage.
    #
    # RIEN NE L'AURAIT SIGNALE AU BON MOMENT. `MarketData` leve bien sur
    # les non-finis, mais trois heures plus tard, au lancement du run. Le
    # message nomme alors deux cents colonnes et ressemble a un cache
    # obsolete — ce qu'il etait, mais pour cette raison-la.
    #
    # ON FILTRE SUR LES COLONNES QUE L'OBSERVATION LIT VRAIMENT, plus
    # `atr_14` dont depend toute la mise a l'echelle du risque.
    # ------------------------------------------------------------------
    # LES DEUX RANGS GLISSANTS, et c'est la seule transformation du depot
    # qui ne regarde PAS une valeur mais une POSITION.
    #
    # `close_ema_dev` et le volume agressif sont deja dans le cadre. Le
    # modele ne s'en est jamais servi parce que leur NIVEAU ne veut rien
    # dire hors de son epoque : la valeur qui marquait un creux en 2024
    # n'en marque plus un en 2026. Le meme piege a ete pris en flagrant
    # delit sur le spread le 2026-09-22 — un seuil au 40e centile calcule
    # deux ans plus tot n'attrapait PLUS AUCUNE occasion.
    #
    # LA FENETRE FERME SUR LA BARRE COURANTE. `rolling` ne voit que le
    # passe ; il n'y a pas de fuite, et c'est verifiable : la valeur en t
    # ne change pas quand on ajoute des barres apres t.
    #
    # 20 000 BARRES, deux semaines de M1 : assez long pour que le rang
    # soit stable, assez court pour suivre le regime.
    _FEN_RANG = 20_000
    d["creux_rang"] = (d["close_ema_dev"]
                       .rolling(_FEN_RANG, min_periods=2_000)
                       .rank(pct=True).astype(np.float32))
    d["flux_rang"] = (d["vol_20"]
                      .rolling(_FEN_RANG, min_periods=2_000)
                      .rank(pct=True).astype(np.float32))
    print("  rangs glissants : creux_rang et flux_rang (source vol_20)",
          flush=True)

    from saint_core import FEATURE_COLS as _FC
    # LA LISTE DE SAINT_CORE FAIT FOI : une colonne qu'elle attend et que le
    # cache ne porte pas ferait lever `MarketData` trois heures plus tard.
    _absentes = [c for c in _FC if c not in d.columns]
    if _absentes:
        raise RuntimeError(f"{len(_absentes)} colonnes de FEATURE_COLS absentes "
                           f"du cache : {_absentes[:10]}")
    avant = len(d)
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=[x for x in _FC if x in d.columns] + ["atr_14"])
    d = d.reset_index(drop=True)
    print(f"  chauffe : {avant:,} -> {len(d):,} lignes "
          f"({100*(1-len(d)/avant):.1f} % perdues)", flush=True)
    d.to_pickle(SORTIE)
    print(f"\n{SORTIE} : {len(d):,} lignes, {len(d.columns)} colonnes",
          flush=True)
    for c in COLONNES_TICKS:
        print("  %-20s med %10.4f   p99 %10.4f"
              % (c, d[c].median(), d[c].quantile(0.99)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
