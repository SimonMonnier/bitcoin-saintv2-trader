# -*- coding: utf-8 -*-
"""Le cache M1 du BTC : prix MT5, microstructure MT5, ET FLUX BINANCE.

POURQUOI CE FICHIER EXISTE, ET C'EST UNE MESURE QUI L'A IMPOSE.

Le 2026-09-21, apres une journee de mesures sur l'or M1 : aucune des 266
colonnes ne porte de DIRECTION, a aucun horizon de 1 a 480 minutes,
occasions disjointes et plancher par rotation. Le modele entraine n'avait
ni avantage ni desavantage — +0.004 bps a 0.0 ecart-type contre un cote
tire au hasard. Il ne trouvait rien parce qu'il n'y a rien.

La raison est structurelle : toutes ces colonnes sont des transformations
de l'OHLC. Meme les sept colonnes de microstructure ajoutees le meme jour
— spread cote, intensite de cotation, asymetrie du carnet — ne portent que
des COTATIONS. Le champ `last` d'un CFD sur l'or vaut zero : il n'y a pas
de transactions publiques, donc pas de flux d'ordres, parce qu'un CFD n'a
pas de marche central.

CE QUE LE BTC A ET QUE L'OR N'AURA JAMAIS. Binance publie ses
TRANSACTIONS : combien d'echanges, quel volume achete a l'agression. C'est
gratuit, complet depuis 2017, et ce depot l'avait deja — six colonnes,
`nb_trades`, `taker_buy_base`, `taker_ratio`, `taker_ma5`,
`flux_taille_trade`, `flux_intensite`.

ELLES ONT ETE RETIREES LE 2026-09-16, et le commentaire de `prepare_m5`
dit pourquoi : « elles n'existent QUE sur Binance ; retirees pour que l'or
puisse partager exactement le meme jeu de colonnes ». On a donc supprime
la seule information qui ne fut pas du prix, pour qu'un instrument qui
n'en a pas puisse suivre.

TROIS SOURCES, ET CHACUNE APPORTE CE QUE LES DEUX AUTRES N'ONT PAS :

  MT5 BTCUSD      les PRIX qu'on trade, et le SPREAD reellement cote.
                  C'est le courtier qui execute ; c'est donc son prix et
                  son spread qui comptent, pas ceux de Binance.
  MT5 ticks       la microstructure de cotation, par minute.
  Binance 1m      le FLUX : transactions et agression. Gratuit, sans cle,
                  sur data.binance.vision.

L'ALIGNEMENT HORAIRE EST LE POINT DELICAT, et `build_binance_features`
l'avait deja documente : Binance horodate en UTC, le serveur du courtier
tourne en UTC+2 ou +3 selon l'heure d'ete — et ce courtier suit le
calendrier AMERICAIN. Un decalage fixe se tromperait d'une heure entiere
sur un tiers de l'historique, sans lever la moindre erreur. On le detecte
donc en correlant les RENDEMENTS, pas les prix : sur douze heures la
tendance domine et deux series decalees correlent quand meme.

LE 2026-09-26, LE JEU EST REFAIT DE ZERO pour le scalping — M1 avec les
contextes M5 et M15, sans Ichimoku ni range, et les familles que la
recherche soutient (`features_scalping`). Deux corrections en chemin :

  L'HEURE DE BINANCE. `decalage_horaire` retenait UN entier mesure sur
  l'echantillon recent. Or le serveur suit l'heure d'ete americaine :
  +2 h l'hiver, +3 h l'ete. Tout l'hiver, le flux arrivait avec une heure
  de retard. `features_scalping.aligne_binance` convertit chaque minute.

  LE BRUT EST GARDE A PART (`BRUT`), comme pour l'or : barres, spread,
  agregat de ticks. Changer de features ne demande plus MetaTrader 5.

    python prepare_btc_m1.py            depuis le brut (ou l'ancien cache)
    python prepare_btc_m1.py --mt5 [n]  barres relues dans MetaTrader 5
"""
from __future__ import annotations

import datetime as dt
import io as _io
import sys
import zipfile

import numpy as np
import pandas as pd

import features_scalping as FS
import prepare_m5

SYMBOLE = "BTCUSD"
SYMBOLE_BINANCE = "BTCUSDT"
SORTIE = "data_cache_BTCUSD_M1.pkl"
CACHE_TICKS = "cache_ticks_m1_BTCUSD.pkl"
CACHE_BINANCE = "cache_binance_1m_BTCUSDT.pkl"
# COINBASE BTC-USD, bougies 1 min — `telecharge_sources_btc.py` l'ecrit.
CACHE_COINBASE = "cache_coinbase_1m_BTCUSD.pkl"
JOUR_M1 = 1440
N_BARRES = 1_000_000
# LE BRUT : barres du courtier, spread de chaque barre, agregat de ticks.
BRUT = "brut_BTCUSD_M1.pkl"
COLONNES_BRUT = ["time", "open", "high", "low", "close", "volume",
                 "spread_bar", "tick_n", "tick_spread_moy", "tick_spread_max"]
# LES CONTEXTES DU SCALPING : M5 et M15, comme l'or M1 — plus de H1 ni de
# H4. (regle de resample, suffixe, bougies par journee a cette echelle.)
ECHELLES_BTC_M1 = [("5min", "_m5", 288), ("15min", "_m15", 96)]

COLONNES_TICKS = [
    "spread_bar", "tick_n", "tick_spread_moy", "tick_spread_max",
    "tick_ask_part", "tick_bid_part", "tick_desequilibre",
]
# LES SIX COLONNES DE FLUX, REMISES. Voir l'entete pour la mesure qui les
# avait fait retirer, et celle qui les fait revenir.
COLONNES_FLUX = [
    "nb_trades",          # nombre de TRANSACTIONS dans la minute
    "taker_buy_base",     # volume achete a l'AGRESSION
    "taker_ratio",        # sa part du volume total : la pression acheteuse
    "taker_ma5",          # la meme, lissee sur cinq minutes
    "flux_taille_trade",  # taille moyenne d'une transaction
    "flux_intensite",     # transactions par minute, rapportee a son rang
]


def charge_barres_mt5(symbole: str, n: int) -> pd.DataFrame:
    """Les barres M1 du COURTIER, spread compris. Ce sont elles qu'on trade."""
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 indisponible : {mt5.last_error()}")
    mt5.symbol_select(symbole, True)
    info = mt5.symbol_info(symbole)
    point = float(info.point) if info else 0.01
    r = mt5.copy_rates_from_pos(symbole, mt5.TIMEFRAME_M1, 0, n)
    mt5.shutdown()
    if r is None or len(r) == 0:
        raise RuntimeError(f"aucune barre M1 pour {symbole}")
    d = pd.DataFrame(r)
    d["time"] = pd.to_datetime(d["time"], unit="s")
    d["volume"] = d["tick_volume"].astype(float)
    # LE PAS DU SYMBOLE VIENT DE MT5, il n'est pas suppose : il vaut 0.01
    # sur l'or et 0.01 sur ce BTC, mais rien ne l'impose et un 1 en dur
    # ferait une friction cent fois fausse sans rien signaler.
    d["spread_bar"] = d["spread"].astype(float) * point / d["close"] * 1e4
    return d[["time", "open", "high", "low", "close", "volume", "spread_bar"]]


def agrege_ticks(symbole, debut, fin, pas_jours=10) -> pd.DataFrame:
    """Les ticks MT5, resumes PAR MINUTE. Voir `prepare_or_m1.agrege_ticks`."""
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 indisponible : {mt5.last_error()}")
    mt5.symbol_select(symbole, True)
    morceaux, curseur, n_total = [], debut, 0
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
        _b, _a = (f & 2) > 0, (f & 4) > 0
        morceaux.append(pd.DataFrame({
            "time": pd.to_datetime(t["time"][bon], unit="s").floor("min"),
            "sp": sp,
            "ask_seul": (_a & ~_b).astype(np.float64),
            "bid_seul": (_b & ~_a).astype(np.float64),
        }).groupby("time").agg(
            tick_n=("sp", "size"), tick_spread_moy=("sp", "mean"),
            tick_spread_max=("sp", "max"),
            tick_ask_part=("ask_seul", "mean"),
            tick_bid_part=("bid_seul", "mean")))
        print(f"    {borne:%Y-%m-%d}  {n_total/1e6:7.1f} M ticks", flush=True)
    mt5.shutdown()
    if not morceaux:
        raise RuntimeError("aucun tick recupere")
    out = pd.concat(morceaux).groupby(level=0).agg({
        "tick_n": "sum", "tick_spread_moy": "mean", "tick_spread_max": "max",
        "tick_ask_part": "mean", "tick_bid_part": "mean"})
    out["tick_desequilibre"] = out["tick_ask_part"] - out["tick_bid_part"]
    return out.reset_index()


def klines_api(depuis: dt.datetime, jusqu_a: dt.datetime, noms) -> pd.DataFrame:
    """Les minutes qu'aucune archive mensuelle ne couvre encore.

    POURQUOI CE REPLI EXISTE. Les archives de `data.binance.vision`
    paraissent APRES la fin du mois : le mois courant renvoie un 404, et
    la premiere version se contentait de l'annoncer « ABSENT » avant de
    continuer. Le cache s'arretait donc au dernier jour du mois clos —
    trois semaines perdues le 2026-09-21, et ce sont les plus RECENTES,
    celles ou finit la marche glissante.

    Ce n'etait pas un oubli discret : l'en-tete de `charge_binance_1m`
    ecrivait deja « un mois absent est complete par l'API ». La phrase
    decrivait une intention, pas du code.

    MILLE MINUTES PAR APPEL, ce que l'API accorde sans cle. Trois
    semaines font une trentaine d'appels.

    L'HORODATAGE DE L'API EST EN MILLISECONDES, toujours — la detection
    ligne par ligne de `charge_binance_1m` le reconnait sans rien savoir
    d'ou vient la ligne.
    """
    import json
    import urllib.request
    t0 = int(depuis.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    t1 = int(jusqu_a.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    lots, appels = [], 0
    while t0 < t1 and appels < 200:
        u = (f"https://api.binance.com/api/v3/klines?symbol={SYMBOLE_BINANCE}"
             f"&interval=1m&startTime={t0}&limit=1000")
        try:
            lot = json.loads(urllib.request.urlopen(u, timeout=60).read())
        except Exception as e:
            print(f"    API interrompue ({type(e).__name__})", flush=True)
            break
        appels += 1
        if not lot:
            break
        lots.extend(lot)
        suivant = int(lot[-1][0]) + 60_000
        # SANS CETTE GARDE LA BOUCLE NE FINIT PAS : si l'API rend le meme
        # lot deux fois, le curseur n'avance pas.
        if suivant <= t0:
            break
        t0 = suivant
    if not lots:
        return pd.DataFrame(columns=noms)
    d = pd.DataFrame(lots, columns=noms)
    for c in noms:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    print(f"    API : {len(d):,} minutes en {appels} appels", flush=True)
    return d


def charge_binance_1m(debut: dt.datetime, fin: dt.datetime) -> pd.DataFrame:
    """Les klines 1 minute de Binance, avec le FLUX. Gratuit, sans cle.

    Les archives mensuelles de `data.binance.vision` sortent a deux
    secondes le mois ; l'API REST demanderait mille appels pour la meme
    chose. Un mois absent — le mois courant, publie avec du retard — est
    complete par l'API.
    """
    import urllib.request
    noms = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "nb_trades", "taker_buy_base",
            "taker_buy_quote", "ignore"]
    morceaux = []
    mois = dt.date(debut.year, debut.month, 1)
    dernier = dt.date(fin.year, fin.month, 1)
    while mois <= dernier:
        u = (f"https://data.binance.vision/data/spot/monthly/klines/"
             f"{SYMBOLE_BINANCE}/1m/{SYMBOLE_BINANCE}-1m-{mois:%Y-%m}.zip")
        try:
            raw = urllib.request.urlopen(u, timeout=90).read()
            z = zipfile.ZipFile(_io.BytesIO(raw))
            d = pd.read_csv(z.open(z.namelist()[0]), header=None, names=noms)
            morceaux.append(d)
            print(f"    {mois:%Y-%m}  {len(d):,} minutes", flush=True)
        except Exception as e:
            print(f"    {mois:%Y-%m}  archive absente ({type(e).__name__}), "
                  f"repli sur l'API", flush=True)
            fin_mois = (mois + dt.timedelta(days=32)).replace(day=1)
            d = klines_api(dt.datetime(mois.year, mois.month, 1),
                           min(dt.datetime(fin_mois.year, fin_mois.month, 1),
                               fin + dt.timedelta(days=1)), noms)
            if len(d):
                morceaux.append(d)
        mois = (mois + dt.timedelta(days=32)).replace(day=1)
    if not morceaux:
        raise RuntimeError("aucune kline Binance")
    d = pd.concat(morceaux, ignore_index=True)
    # L'UNITE DE L'HORODATAGE CHANGE EN COURS D'HISTORIQUE, et c'est un
    # piege qui ne se voit qu'a l'execution.
    #
    # Binance est passe des millisecondes aux MICROSECONDES courant 2025,
    # sans prevenir et sans changer le format du fichier. Une detection
    # faite UNE FOIS sur la premiere ligne du cadre concatene applique
    # donc la mauvaise unite a la moitie de l'historique : la premiere
    # version est tombee sur un `OutOfBoundsDatetime` a l'an 56971.
    #
    # On tranche LIGNE PAR LIGNE, sur la grandeur. Le seuil de 1e14
    # separe sans ambiguite : en millisecondes une date de 2026 vaut
    # 1.8e12, en microsecondes 1.8e15.
    t_ = d["open_time"].astype("int64")
    micro = t_ > int(1e14)
    d["time"] = pd.NaT
    d.loc[micro, "time"] = pd.to_datetime(t_[micro], unit="us")
    d.loc[~micro, "time"] = pd.to_datetime(t_[~micro], unit="ms")
    d["time"] = pd.to_datetime(d["time"])
    print(f"    horodatage : {int(micro.sum()):,} lignes en microsecondes, "
          f"{int((~micro).sum()):,} en millisecondes", flush=True)
    return d[["time", "close", "volume", "quote_vol", "nb_trades",
              "taker_buy_base"]].sort_values("time").reset_index(drop=True)


def decalage_horaire(mt5_df: pd.DataFrame, bnc: pd.DataFrame) -> int:
    """De combien d'heures le serveur du courtier avance sur UTC ?

    ON CORRELE LES RENDEMENTS, PAS LES PRIX. Sur douze heures la tendance
    domine : deux series decalees d'une heure correlent quand meme a 0.99
    en niveau. En rendements, un decalage d'une minute suffit a faire
    tomber la correlation.

    LE DECALAGE N'EST PAS CONSTANT — ce courtier suit l'heure d'ete
    AMERICAINE, donc +2 h l'hiver et +3 h l'ete. On mesure sur un
    ECHANTILLON RECENT et on retient l'entier qui correle le mieux ; le
    reste est absorbe par le fait que les colonnes de flux sont des
    grandeurs de minute, pas des prix.
    """
    a = mt5_df.tail(60000).set_index("time")["close"].pct_change().dropna()
    b = bnc.set_index("time")["close"].pct_change().dropna()
    best, meilleur = 0, -2.0
    for h in range(-1, 6):
        bb = b.copy()
        bb.index = bb.index + pd.Timedelta(hours=h)
        j = a.align(bb, join="inner")
        if len(j[0]) < 5000:
            continue
        r = float(np.corrcoef(j[0].to_numpy(), j[1].to_numpy())[0, 1])
        print(f"    decalage +{h} h : correlation des rendements {r:+.4f}",
              flush=True)
        if r > meilleur:
            best, meilleur = h, r
    if meilleur < 0.5:
        raise RuntimeError(
            f"aucun decalage ne correle (meilleur {meilleur:+.3f}). Les deux "
            f"series ne decrivent pas le meme marche.")
    print(f"  decalage retenu : +{best} h  (correlation {meilleur:+.4f})",
          flush=True)
    return best


def brut_depuis_cache() -> pd.DataFrame:
    """Le brut sans MetaTrader 5 : `BRUT` s'il existe, sinon l'ancien cache.

    L'ancien cache porte deja les barres, le spread et l'agregat de ticks a
    cote de ses features. On en extrait le brut, et l'ancien cache est
    RENOMME plutot qu'ecrase : ses colonnes Ichimoku restent relisibles.
    """
    import os
    if os.path.exists(BRUT):
        b = pd.read_pickle(BRUT)
        print(f"brut : {len(b):,} barres relues depuis {BRUT}", flush=True)
        return b
    ancien = pd.read_pickle(SORTIE)
    manque = [c for c in COLONNES_BRUT if c not in ancien.columns]
    if manque:
        raise RuntimeError(f"{SORTIE} ne porte pas le brut : {manque}")
    b = ancien[COLONNES_BRUT].copy()
    del ancien
    b.to_pickle(BRUT)
    garde = SORTIE.replace(".pkl", "_ichimoku.pkl")
    os.replace(SORTIE, garde)
    print(f"brut : {len(b):,} barres extraites de l'ancien cache -> {BRUT} ; "
          f"l'ancien cache est garde sous {garde}", flush=True)
    return b


def brut_depuis_mt5(n: int) -> pd.DataFrame:
    """Le brut relu dans MetaTrader 5, ticks compris."""
    import os
    m1 = charge_barres_mt5(SYMBOLE, n).sort_values("time").reset_index(drop=True)
    print(f"MT5 : {len(m1):,} barres  {m1['time'].iloc[0]} -> {m1['time'].iloc[-1]}",
          flush=True)
    deb = m1["time"].iloc[0].to_pydatetime()
    fin = m1["time"].iloc[-1].to_pydatetime() + dt.timedelta(minutes=1)
    if os.path.exists(CACHE_TICKS):
        tk = pd.read_pickle(CACHE_TICKS)
        print(f"ticks : {len(tk):,} minutes relues du cache", flush=True)
    else:
        print("ticks MT5 : agregation par minute", flush=True)
        tk = agrege_ticks(SYMBOLE, deb, fin)
        tk.to_pickle(CACHE_TICKS)
    m1 = m1.merge(tk, on="time", how="left")
    m1["tick_n"] = m1["tick_n"].fillna(0.0)
    for c in ("tick_spread_moy", "tick_spread_max"):
        m1[c] = m1[c].fillna(m1["spread_bar"])
    b = m1[COLONNES_BRUT].copy()
    b.to_pickle(BRUT)
    return b


def verifie_alignement(m1: pd.DataFrame) -> None:
    """Le courtier et Binance doivent bouger ENSEMBLE chaque mois.

    C'est le test qui aurait attrape le decalage fixe : en hiver, avec une
    heure d'erreur, la correlation des rendements tombait a 0.01. Un mois
    sous 0.9 arrete la construction.
    """
    a = m1.set_index("time")["close"].pct_change()
    b = m1.set_index("time")["bn_close"].pct_change()
    ok = a.notna() & b.notna()
    j = pd.DataFrame({"a": a[ok], "b": b[ok]})
    par_mois = j.groupby(j.index.to_period("M")).apply(
        lambda x: float(np.corrcoef(x["a"], x["b"])[0, 1]) if len(x) > 1000
        else np.nan).dropna()
    pire = par_mois.idxmin()
    print(f"  alignement Binance : correlation des rendements par mois "
          f"min {par_mois.min():+.3f} ({pire})  mediane {par_mois.median():+.3f}"
          f"  sur {len(par_mois)} mois", flush=True)
    if par_mois.min() < 0.9:
        raise RuntimeError(
            f"le mois {pire} correle a {par_mois.min():+.3f} : Binance et le "
            f"courtier ne sont pas a la meme heure.")


def joint_sources(m1: pd.DataFrame) -> pd.DataFrame:
    """Le brut du courtier, plus Binance spot et Coinbase a l'heure du serveur.

    UNE ECRITURE, DEUX LECTEURS : la construction du cache et le test de
    causalite.
    """
    bn = FS.aligne_binance(pd.read_pickle(CACHE_BINANCE))
    cb = FS.aligne_coinbase(pd.read_pickle(CACHE_COINBASE))
    m1 = m1.merge(bn, on="time", how="left")
    return m1.merge(cb, on="time", how="left")


def construit_et_ecrit(m1: pd.DataFrame) -> pd.DataFrame:
    m1 = joint_sources(m1)
    print(f"  minutes couvertes par Coinbase : "
          f"{100*m1['cb_close'].notna().mean():.2f} %", flush=True)
    couvert = float(m1["taker_buy_base"].notna().mean())
    print(f"  minutes couvertes par le flux Binance : {100*couvert:.2f} %",
          flush=True)
    if couvert < 0.95:
        raise RuntimeError(f"seulement {100*couvert:.1f} % des minutes ont "
                           f"du flux Binance.")
    verifie_alignement(m1)

    d, _c, _i = prepare_m5.construit(m1, avec_flux=False, jour=JOUR_M1,
                                     echelles=ECHELLES_BTC_M1,
                                     avec_structures=False)
    d = FS.ajoute(d)

    from saint_core import FEATURE_COLS as _FC
    absentes = [c for c in _FC if c not in d.columns]
    if absentes:
        raise RuntimeError(f"colonnes de FEATURE_COLS non produites : {absentes}")
    avant = len(d)
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=list(_FC) + ["atr_14"]).reset_index(drop=True)
    print(f"  chauffe et trous : {avant:,} -> {len(d):,} lignes "
          f"({100*(1-len(d)/avant):.1f} % perdues)", flush=True)
    for c in _FC:
        d[c] = d[c].astype(np.float32)
    d.to_pickle(SORTIE)
    print(f"\n{SORTIE} : {len(d):,} lignes, {len(_FC)} features, "
          f"{d['time'].iloc[0]} -> {d['time'].iloc[-1]}", flush=True)
    print("  %-20s %12s %12s %12s" % ("feature", "p1", "mediane", "p99"))
    for c in _FC:
        q = d[c].quantile([0.01, 0.5, 0.99]).to_numpy()
        print("  %-20s %12.4f %12.4f %12.4f" % (c, q[0], q[1], q[2]), flush=True)
    return d


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--mt5":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else N_BARRES
        m1 = brut_depuis_mt5(n)
    else:
        m1 = brut_depuis_cache()
    m1 = m1.sort_values("time").reset_index(drop=True)
    print(f"  spread cote : med {m1['spread_bar'].median():.3f} bps  "
          f"p99 {m1['spread_bar'].quantile(0.99):.3f}", flush=True)
    construit_et_ecrit(m1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
