"""Recupere les donnees Binance BTCUSDT perpetuel et les aligne sur l'heure broker.

C'est la seule classe d'information absente du flux CFD : positionnement des
acteurs, flux agressif, interet ouvert et profondeur du carnet. Les features
derivees du seul prix sont epuisees -- mesure : sur 21 features, une seule
apportait plus de 0.001 d'AUC, et les 5 features de ticks exactement 0.000.

Deux sources, gratuites et sans cle :
  - REST /fapi/v1/fundingRate       taux de financement (8 h), depuis 2019
  - data.binance.vision `metrics`   OI + ratios long/short + taker (5 min), 2021-01

Le CARNET D'ORDRES a ete ecarte apres mesure, pas par choix : les archives
`bookDepth` ne descendent pas sous le palier +-1 %, alors que l'endpoint live
/fapi/v1/depth (plafonne a limit=1000) ne porte que jusqu'a +-0.17 % du mid.
Aucun recouvrement : la feature serait entrainable mais incalculable en live.
L'alternative `bookTicker` couvre les deux bouts mais renvoie 404 sur la plupart
des dates, pour 241 Mo/jour quand elle existe (~400 Go). Detail dans saint_core.

ALIGNEMENT HORAIRE -- le point critique. Binance horodate en UTC, le serveur MT5
tourne en UTC+2 ou UTC+3 selon l'heure d'ete, et ce broker suit le calendrier
DST AMERICAIN (mesure : +3 h des le 15 mars). Un offset fixe decalerait un bon
tiers de l'historique d'une heure entiere, sans la moindre erreur visible.

On le detecte donc empiriquement, en correlant les RENDEMENTS M1 du broker aux
klines Binance. Correler les prix ne marcherait pas : sur 12 h le trend domine
et tous les decalages correlent a 0.99. Sur les rendements, le bon decalage
sort a 0.99 et tous les autres a 0.00 -- sans ambiguite possible.

    python build_binance_features.py
"""

import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

SYMBOL_MT5 = "BTCUSD"
SYMBOL_BN = "BTCUSDT"

# Les archives commencent en 2021, mais Binance n'a pas publie toutes les
# colonnes d'emblee. Mesure sur le cache brut :
#   taker_ratio     absent du 2022-02-01 au 2022-05-09   (97 jours)
#   ls_ratio_top    absent ~260 jours sur 2022, dernier trou jusqu'au 2022-12-14
#   oi / retail     complets, trous de quelques minutes
# A partir du 2022-12-15 la couverture est de 100.000 % sur les quatre colonnes,
# sans un seul trou residuel. On demarre donc la : combler 97 jours par ffill
# fabriquerait une constante et apprendrait au modele un regime inexistant.
DATE_FROM = datetime(2022, 12, 15)

# On telecharge depuis 2022-01 malgre tout : le cache brut sert aussi a
# re-mesurer ces bornes si Binance complete son historique a posteriori.
DL_FROM = datetime(2022, 1, 1)
OUT_PATH = f"binance_features_{SYMBOL_MT5}.pkl"

# Duree maximale d'un trou comble par report de la derniere valeur connue.
# Ces grandeurs sont echantillonnees toutes les 5 min et bougent lentement :
# reporter sur une heure est anodin, sur une journee c'est de l'invention.
FFILL_MAX_MIN = 60

UA = {"User-Agent": "python"}
BASE_F = "https://fapi.binance.com"
VISION = "https://data.binance.vision"
N_THREADS = 16


def _get(url, timeout=60, raw=False, essais=3):
    for k in range(essais):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            return data if raw else json.loads(data.decode())
        except Exception:
            if getattr(sys.exc_info()[1], "code", None) == 404:
                raise
            if k == essais - 1:
                raise
            time.sleep(1.5 * (k + 1))


# ====================================================================
#  1. Alignement horaire
# ====================================================================
def _klines(jour, heures=12):
    t0 = int(jour.timestamp() * 1000)
    kl = _get(f"{BASE_F}/fapi/v1/klines?symbol={SYMBOL_BN}&interval=1m"
              f"&startTime={t0}&limit={heures * 60}")
    if not kl:
        return None
    return pd.Series({int(k[0]) // 1000: float(k[4]) for k in kl}).sort_index()


def _meilleur_offset(bn, rates):
    """Offset entier (h) maximisant la correlation des rendements 1 min."""
    if bn is None or rates is None or len(rates) < 500:
        return None, -2.0
    m = pd.Series(rates["close"].astype(float),
                  index=rates["time"].astype(np.int64)).sort_index()
    bn_r = bn.diff()
    m_r = m.diff()
    best, best_c = None, -2.0
    for off in range(0, 6):
        decale = pd.Series(m_r.values, index=m_r.index - off * 3600)
        com = bn_r.index.intersection(decale.index)
        if len(com) < 300:
            continue
        a = bn_r.loc[com].values
        b = decale.loc[com].values
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 300 or a[ok].std() == 0 or b[ok].std() == 0:
            continue
        c = float(np.corrcoef(a[ok], b[ok])[0, 1])
        if c > best_c:
            best_c, best = c, off
    return best, best_c


def _sonde(jours):
    """Offset detecte pour chaque date. Binance en parallele, MT5 en serie
    (le module MetaTrader5 n'est pas thread-safe)."""
    with ThreadPoolExecutor(N_THREADS) as ex:
        kl = list(ex.map(lambda j: _safe(_klines, j), jours))
    out = {}
    for j, bn in zip(jours, kl):
        r = mt5.copy_rates_range(SYMBOL_MT5, mt5.TIMEFRAME_M1,
                                 j - timedelta(hours=6), j + timedelta(hours=18))
        off, c = _meilleur_offset(bn, r)
        out[j] = (off, c) if c >= 0.90 else (None, c)
    return out


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return None


def _nieme_dimanche(annee, mois, n):
    d = datetime(annee, mois, 1)
    d += timedelta(days=(6 - d.weekday()) % 7)      # premier dimanche
    return d + timedelta(days=7 * (n - 1))


def _calendrier_dst_us(date_from, date_to, ete, hiver):
    """Bascules DST americaines : 2e dimanche de mars, 1er dimanche de novembre.

    Le changement prend effet a 02:00 heure locale ; a l'echelle d'une bougie M1
    l'erreur residuelle vaut au pire deux heures, un dimanche par semestre.
    """
    tab = {}
    for an in range(date_from.year - 1, date_to.year + 2):
        tab[_nieme_dimanche(an, 3, 2) + timedelta(hours=2)] = ete
        tab[_nieme_dimanche(an, 11, 1) + timedelta(hours=2)] = hiver
    return pd.Series(tab).sort_index()


def detecte_offsets(date_from, date_to):
    """Table date -> offset (h) en vigueur, transitions au jour pres.

    Deux etapes. D'abord une mesure empirique hebdomadaire : c'est elle qui fait
    foi, on ne suppose rien du fuseau du broker. Ensuite, si les transitions
    observees coincident avec le calendrier DST americain, on adopte ce
    calendrier -- il place les bascules au jour exact, y compris la ou aucun
    sondage n'a abouti. Sans cela, les periodes sans mesure heritent de l'offset
    precedent et peuvent rester decalees d'une heure pendant des semaines.
    """
    semaines = []
    j = date_from
    while j < date_to - timedelta(days=1):
        semaines.append(j)
        j += timedelta(days=7)

    print(f"    scan hebdomadaire : {len(semaines)} sondages...")
    res = _sonde(semaines)
    connus = [(d, o) for d, (o, _) in res.items() if o is not None]
    if not connus:
        raise SystemExit("Aucun alignement detecte -- verifier MT5 et le symbole.")
    print(f"    {len(connus)}/{len(semaines)} sondages exploitables")

    # Affinage quotidien autour de chaque changement observe.
    bornes, n_trans = [], 0
    for (d0, o0), (d1, o1) in zip(connus, connus[1:]):
        if o0 != o1:
            n_trans += 1
            if (d1 - d0).days <= 50:
                bornes.extend(d0 + timedelta(days=k)
                              for k in range(1, (d1 - d0).days))
    if bornes:
        print(f"    affinage de {len(bornes)} jours autour de "
              f"{n_trans} transitions...")
        for d, (o, _) in _sonde(bornes).items():
            if o is not None:
                connus.append((d, o))
        connus.sort()

    mesure = pd.Series({d: int(o) for d, o in connus}).sort_index()
    print(f"    offsets mesures : {mesure.value_counts().to_dict()}")

    # --- confrontation au calendrier DST americain ---
    ete = int(mesure[[d.month in (5, 6, 7, 8, 9) for d in mesure.index]].mode()[0])
    hiver = int(mesure[[d.month in (12, 1, 2) for d in mesure.index]].mode()[0])
    cal = _calendrier_dst_us(date_from, date_to, ete, hiver)
    predit = cal.values[np.clip(
        np.searchsorted([d.timestamp() for d in cal.index],
                        [d.timestamp() for d in mesure.index], side="right") - 1,
        0, len(cal) - 1)]
    accord = float((predit == mesure.values).mean())
    print(f"    ete={ete:+d}h hiver={hiver:+d}h ; accord avec le calendrier DST "
          f"US : {100 * accord:.1f} % ({int(accord * len(mesure))}/{len(mesure)})")

    if accord >= 0.98:
        tab = cal[(cal.index >= date_from - timedelta(days=400))
                  & (cal.index <= date_to + timedelta(days=1))]
        print("    -> calendrier DST retenu, bascules au jour exact :")
        for d, o in tab.items():
            print(f"       {d:%Y-%m-%d %H:%M} -> {o:+d}h")
        return tab

    print("    -> ATTENTION : le calendrier DST ne colle pas aux mesures ; "
          "on s'en tient aux mesures brutes (transitions a +/- quelques jours).")
    for (d0, o0), (d1, o1) in zip(mesure.items(), list(mesure.items())[1:]):
        if o0 != o1:
            print(f"       transition {o0:+d}h -> {o1:+d}h entre "
                  f"{d0:%Y-%m-%d} et {d1:%Y-%m-%d}")
    return mesure


def vers_heure_broker(ts_utc: pd.Series, tab: pd.Series) -> pd.Series:
    """Applique a chaque horodatage l'offset en vigueur a cette date."""
    bornes = np.array([d.timestamp() for d in tab.index])
    vals = tab.values.astype(float)
    i = np.clip(np.searchsorted(bornes, ts_utc.astype(np.int64) // 10**9,
                                side="right") - 1, 0, len(vals) - 1)
    return ts_utc + pd.to_timedelta(vals[i], unit="h")


# ====================================================================
#  2. Sources
# ====================================================================
def charge_funding(date_from, date_to):
    out, curseur = [], int(date_from.timestamp() * 1000)
    fin = int(date_to.timestamp() * 1000)
    while curseur < fin:
        d = _get(f"{BASE_F}/fapi/v1/fundingRate?symbol={SYMBOL_BN}"
                 f"&startTime={curseur}&limit=1000")
        if not d:
            break
        out.extend(d)
        suiv = int(d[-1]["fundingTime"]) + 1
        if suiv <= curseur:
            break
        curseur = suiv
        time.sleep(0.1)
    df = pd.DataFrame({
        "time_utc": pd.to_datetime([int(x["fundingTime"]) for x in out], unit="ms"),
        "funding_rate": [float(x["fundingRate"]) for x in out],
    }).drop_duplicates("time_utc").sort_values("time_utc")
    return df


def _archive(jeu, jour):
    url = (f"{VISION}/data/futures/um/daily/{jeu}/{SYMBOL_BN}/"
           f"{SYMBOL_BN}-{jeu}-{jour:%Y-%m-%d}.zip")
    z = zipfile.ZipFile(io.BytesIO(_get(url, raw=True)))
    return pd.read_csv(z.open(z.namelist()[0]))


def _jour_metrics(jour):
    try:
        df = _archive("metrics", jour)
    except Exception:
        return None
    o = pd.DataFrame({"time_utc": pd.to_datetime(df["create_time"])})
    oi = pd.to_numeric(df["sum_open_interest"], errors="coerce")
    # 463 lignes de l'historique portent un OI a zero (pannes de publication
    # cote Binance). Les prendre au mot produit des variations de -100 %, qui
    # domineraient a elles seules l'ecart-type de la feature.
    o["oi"] = oi.where(oi > 0)
    for src, dst in [("sum_taker_long_short_vol_ratio", "taker_ratio"),
                     ("sum_toptrader_long_short_ratio", "ls_ratio_top"),
                     ("count_long_short_ratio", "ls_ratio_retail")]:
        o[dst] = np.log(pd.to_numeric(df[src], errors="coerce").clip(lower=1e-3))
    return o


CACHE_DIR = ".cache_binance"


def charge_archives(jeu, fn, date_from, date_to):
    """Telecharge les archives quotidiennes manquantes, en gardant un cache brut.

    Le cache est en UTC, AVANT alignement : revoir la detection du decalage
    horaire ne doit jamais couter un nouveau telechargement de 750 Mo.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    f_cache = os.path.join(CACHE_DIR, f"{jeu}_utc.pkl")
    f_vides = os.path.join(CACHE_DIR, f"{jeu}_absents.json")

    cache = pd.read_pickle(f_cache) if os.path.exists(f_cache) else pd.DataFrame()
    vides = set(json.load(open(f_vides))) if os.path.exists(f_vides) else set()
    deja = set()
    if not cache.empty:
        deja = set(pd.to_datetime(cache["time_utc"]).dt.strftime("%Y-%m-%d"))

    jours, j = [], date_from
    while j < date_to:
        cle = j.strftime("%Y-%m-%d")
        if cle not in deja and cle not in vides:
            jours.append(j)
        j += timedelta(days=1)

    if deja:
        print(f"      cache : {len(deja)} jours deja presents, "
              f"{len(vides)} connus absents")
    if jours:
        print(f"      telechargement de {len(jours)} jours...")
        faits = [0]

        def trav(d):
            r = fn(d)
            faits[0] += 1
            if faits[0] % 200 == 0:
                print(f"      {faits[0]}/{len(jours)} "
                      f"({100 * faits[0] / len(jours):.0f} %)")
            return d, r

        with ThreadPoolExecutor(N_THREADS) as ex:
            res = list(ex.map(trav, jours))
        neufs = [r for _, r in res if r is not None and len(r)]
        for d, r in res:
            if r is None or not len(r):
                vides.add(d.strftime("%Y-%m-%d"))
        print(f"      {len(neufs)}/{len(jours)} jours recuperes, "
              f"{len(jours) - len(neufs)} absents")
        if neufs:
            cache = pd.concat([cache] + neufs, ignore_index=True)
            cache = (cache.drop_duplicates("time_utc")
                          .sort_values("time_utc").reset_index(drop=True))
            cache.to_pickle(f_cache)
        json.dump(sorted(vides), open(f_vides, "w"))

    if cache.empty:
        return pd.DataFrame()
    m = (pd.to_datetime(cache["time_utc"]) >= date_from - timedelta(days=1))
    return cache[m].sort_values("time_utc").reset_index(drop=True)


# ====================================================================
#  3. Assemblage
# ====================================================================
def sur_grille(s: pd.Series, grille: pd.DatetimeIndex,
               limite_min=FFILL_MAX_MIN) -> pd.Series:
    """Propage la derniere valeur CONNUE sur la grille M1.

    Jamais d'interpolation : elle ferait remonter de l'information future dans
    le passe, exactement le type de fuite deja corrige sur le H1.

    `limite_min=None` pour une grandeur qui tient reellement sa valeur entre
    deux publications (le funding, constant pendant 8 h par construction) ;
    une limite en minutes pour tout le reste, afin qu'un trou long ressorte en
    NaN au lieu de se deguiser en plateau.
    """
    s = s[~s.index.duplicated(keep="last")].sort_index()
    u = s.reindex(s.index.union(grille))
    if limite_min is None:
        u = u.ffill()
    else:
        # limit compte en LIGNES ; l'union melange grille M1 et points sources,
        # donc on borne par le temps ecoule depuis la derniere valeur connue.
        age = pd.Series(u.index, index=u.index).where(u.notna()).ffill()
        u = u.ffill().where(
            (pd.Series(u.index, index=u.index) - age)
            <= pd.Timedelta(minutes=limite_min))
    return u.reindex(grille)


def main() -> int:
    if not mt5.initialize():
        print(f"MT5 init KO : {mt5.last_error()}")
        return 1
    date_to = datetime.now()
    print(f"Binance {SYMBOL_BN} : sortie {DATE_FROM:%Y-%m-%d} -> {date_to:%Y-%m-%d}"
          f"  (archives telechargees depuis {DL_FROM:%Y-%m-%d})")

    print("\n[1/3] Decalage horaire broker / UTC")
    os.makedirs(CACHE_DIR, exist_ok=True)
    f_off = os.path.join(CACHE_DIR, "offsets.pkl")
    tab = None
    if os.path.exists(f_off) and "--offsets" not in sys.argv:
        tab = pd.read_pickle(f_off)
        # Un cache qui ne couvre pas la periode donnerait un offset extrapole.
        if tab.index[-1] < date_to - timedelta(days=200):
            tab = None
        else:
            print(f"    table en cache ({len(tab)} bascules, "
                  f"jusqu'au {tab.index[-1]:%Y-%m-%d}) -- "
                  f"`--offsets` pour re-mesurer")
    if tab is None:
        tab = detecte_offsets(DL_FROM, date_to)
        tab.to_pickle(f_off)
    mt5.shutdown()

    print("\n[2/3] Funding rate")
    fund = charge_funding(DL_FROM, date_to)
    print(f"      {len(fund):,} points "
          f"({fund['time_utc'].iloc[0]:%Y-%m-%d} -> {fund['time_utc'].iloc[-1]:%Y-%m-%d})")

    print("\n[3/3] Metrics (OI, ratios long/short, ratio taker)")
    met = charge_archives("metrics", _jour_metrics, DL_FROM, date_to)
    if met.empty:
        print("      AUCUNE donnee metrics -- abandon")
        return 1
    # Le cache a pu etre construit avant le garde-fou sur l'OI a zero.
    met["oi"] = pd.to_numeric(met["oi"], errors="coerce").where(lambda s: s > 0)
    print(f"      {len(met):,} points")

    # --- passage en heure broker ---
    for df in (fund, met):
        df["time"] = vers_heure_broker(df["time_utc"], tab)

    # --- grille M1 commune ---
    # On demarre une journee avant DATE_FROM pour que le pct_change(60) et le
    # report de la derniere valeur connue disposent de leur amorce.
    deb = max(met["time"].min(), pd.Timestamp(DATE_FROM) - pd.Timedelta(days=1))
    fin = met["time"].max().ceil("min")
    grille = pd.date_range(deb.floor("min"), fin, freq="1min")
    out = pd.DataFrame(index=grille)

    m = met.set_index("time")
    for c in ["taker_ratio", "ls_ratio_top", "ls_ratio_retail"]:
        out[c] = sur_grille(m[c], grille)
    oi = sur_grille(m["oi"], grille)
    out["oi_change"] = oi.pct_change(60).replace([np.inf, -np.inf], np.nan)

    # Le funding est constant par construction entre deux reglements (8 h) :
    # le reporter n'est pas combler un trou, c'est sa definition.
    f = fund.set_index("time")["funding_rate"]
    f = f[~f.index.duplicated(keep="last")].sort_index()
    out["funding_rate"] = sur_grille(f, grille, limite_min=None)
    out["funding_cum24"] = sur_grille(f.rolling(3, min_periods=1).sum(),
                                      grille, limite_min=None)

    out.index.name = "time"
    out = out.loc[pd.Timestamp(DATE_FROM):].dropna()

    out.to_pickle(OUT_PATH)
    print(f"\n-> {OUT_PATH} : {len(out):,} minutes, "
          f"{os.path.getsize(OUT_PATH) / 1e6:.1f} Mo")
    print(f"   couverture {out.index[0]} -> {out.index[-1]} (heure broker)")
    print()
    desc = out.describe().T[["mean", "std", "min", "max"]]
    desc["n_valides"] = out.notna().sum()
    desc["% couvert"] = (100 * out.notna().mean()).round(1)
    print(desc.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
