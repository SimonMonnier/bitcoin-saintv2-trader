# -*- coding: utf-8 -*-
"""Bougies 5 minutes Binance au comptant et financement, pour les cryptos du jeu.

2026-09-28, demande du proprietaire : les cryptos de Vantage presentes sur
Binance et Coinbase, avec EXACTEMENT les features du BTC. Celles du BTC
viennent de `klines_5m_spot_BTCUSDT.pkl` (archives `data.binance.vision`, avec
le volume acheteur et le nombre de trades de chaque bougie) et du
financement des perpetuels (`cache_portage.pkl`, 2020-01 -> 2026-08-31).
Ce fichier produit les memes sources pour les autres :

  klines_5m_spot_<ACTIF>USDT.pkl   memes colonnes, memes archives, arretees a
                                   la meme bougie que celles du BTC
  cache_funding_crypto.pkl         {<ACTIF>USDT : Series du financement},
                                   meme fenetre que `cache_portage.pkl`

    python telecharge_crypto_binance.py ETH ZEC
"""
import datetime as dt
import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from build_binance_features import BASE_F, N_THREADS, VISION, _get

FUNDING = "cache_funding_crypto.pkl"
PORTAGE = "cache_portage.pkl"
REF = "klines_5m_spot_BTCUSDT.pkl"
I_T, I_O, I_H, I_L, I_C, I_V, I_QV, I_NT, I_TBB = 0, 1, 2, 3, 4, 5, 7, 8, 9

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _zip(url):
    try:
        z = zipfile.ZipFile(io.BytesIO(_get(url, raw=True, essais=2)))
    except Exception:
        return None
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, low_memory=False)
    t = pd.to_numeric(df.iloc[:, I_T], errors="coerce")
    df = df[t.notna()]
    if df.empty:
        return None
    t = pd.to_numeric(df.iloc[:, I_T], errors="coerce")
    unite = "us" if t.iloc[0] > 1e14 else "ms"
    col = lambda i: pd.to_numeric(df.iloc[:, i], errors="coerce").to_numpy()
    return pd.DataFrame({
        "time": pd.to_datetime(t, unit=unite),
        "open": col(I_O), "high": col(I_H), "low": col(I_L), "close": col(I_C),
        "volume": col(I_V), "quote_vol": col(I_QV),
        "nb_trades": col(I_NT).astype("int64"), "taker_buy_base": col(I_TBB),
    })


def klines(sym: str, fin: pd.Timestamp) -> pd.DataFrame:
    mois, a, m = [], 2017, 8
    while (a, m) < (fin.year, fin.month):
        mois.append((a, m))
        a, m = (a + 1, 1) if m == 12 else (a, m + 1)
    urls = [f"{VISION}/data/spot/monthly/klines/{sym}/5m/{sym}-5m-{a:04d}-{m:02d}.zip"
            for a, m in mois]
    jours = [dt.date(fin.year, fin.month, 1) + dt.timedelta(days=i) for i in range(fin.day)]
    urls += [f"{VISION}/data/spot/daily/klines/{sym}/5m/{sym}-5m-{j:%Y-%m-%d}.zip" for j in jours]
    with ThreadPoolExecutor(N_THREADS) as ex:
        parts = [r for r in ex.map(_zip, urls) if r is not None and len(r)]
    d = pd.concat(parts, ignore_index=True).drop_duplicates("time").sort_values("time")
    return d[d["time"] <= fin].reset_index(drop=True)


def funding(sym: str, debut: pd.Timestamp, fin: pd.Timestamp) -> pd.Series:
    out, cur, stop = [], int(debut.timestamp() * 1000), int(fin.timestamp() * 1000)
    while cur < stop:
        d = _get(f"{BASE_F}/fapi/v1/fundingRate?symbol={sym}&startTime={cur}&limit=1000")
        if not d:
            break
        out.extend(d)
        suiv = int(d[-1]["fundingTime"]) + 1
        if suiv <= cur:
            break
        cur = suiv
        time.sleep(0.2)
    s = pd.Series([float(x["fundingRate"]) for x in out],
                  index=pd.to_datetime([int(x["fundingTime"]) for x in out], unit="ms"))
    s = s[~s.index.duplicated()].sort_index()
    # A LA SECONDE : l'heure du financement Binance porte quelques millisecondes.
    s.index = s.index.floor("s")
    return s[(s.index >= debut) & (s.index <= fin)]


def main() -> int:
    actifs = sys.argv[1:] or ["ETH", "ZEC"]
    fin = pd.read_pickle(REF)["time"].max()
    po = pd.read_pickle(PORTAGE)["BTCUSDT"]
    f0, f1 = pd.to_datetime(po["time"]).min(), pd.to_datetime(po["time"]).max()
    try:
        fu = pd.read_pickle(FUNDING)
    except FileNotFoundError:
        fu = {}
    for a in actifs:
        sym = f"{a}USDT"
        t0 = time.time()
        d = klines(sym, fin)
        d.to_pickle(f"klines_5m_spot_{sym}.pkl")
        trous = (d["time"].diff().dt.total_seconds() > 600).sum()
        print(f"{sym}: {len(d):,} bougies 5 min  {d['time'].iloc[0]} -> {d['time'].iloc[-1]}  "
              f"trous > 10 min : {trous}  ({time.time() - t0:.0f} s)", flush=True)
        fu[sym] = funding(sym, f0, f1)
        print(f"{sym}: financement {len(fu[sym])} valeurs  {fu[sym].index.min()} -> "
              f"{fu[sym].index.max()}", flush=True)
    pd.to_pickle(fu, FUNDING)
    print(f"ecrit {FUNDING}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
