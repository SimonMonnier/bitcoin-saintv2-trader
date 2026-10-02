# -*- coding: utf-8 -*-
"""Bougies M5 MT5 des cryptos Vantage presentes sur Binance ET Coinbase.

2026-09-28, demande du proprietaire : « toutes les crypto-monnaies presentes
sur MT5 avec Vantage FX et aussi sur Binance et Coinbase, avec exactement
les memes features que pour le BTC », dans un seul modele avec un petit
cerveau par marche.

La liste vient de la verification du 2026-09-28 (API publiques Binance et
Coinbase, symboles du chemin « Crypto Currency\\Crypto Major » de Vantage) :
paire USDT au comptant ET contrat perpetuel chez Binance (le flux acheteur,
le nombre de trades, le financement), paire USD en ligne chez Coinbase
(l'ecart Coinbase / Binance). Ecartes : IOTA, NEO, TRX, NXPC, MMT (absents de
Coinbase), OKB, CRO (absents de Binance), LRC (retire des deux).

CE FICHIER NE FAIT QUE LIRE : les bougies M5 de Vantage (spread de chaque
bougie compris), pour les couts du jeu, et les caracteristiques de chaque
symbole, dont le swap. Aucun ordre.

    python telecharge_crypto_mt5.py
"""
import sys
import time

import numpy as np
import pandas as pd

CACHE = "cache_crypto_m5_mt5.pkl"
# symbole Vantage -> actif (Binance <actif>USDT, Coinbase <actif>-USD)
CRYPTOS = {"BTCUSD": "BTC", "ETHUSD": "ETH", "LTCUSD": "LTC", "BCHUSD": "BCH",
           "XRPUSD": "XRP", "XLMUSD": "XLM", "ADAUSD": "ADA", "DOGUSD": "DOGE",
           "DOTUSD": "DOT", "LNKUSD": "LINK", "SOLUSD": "SOL", "UNIUSD": "UNI",
           "ALGUSD": "ALGO", "AVAUSD": "AVAX", "BATUSD": "BAT", "FILUSD": "FIL",
           "SHBUSD": "SHIB", "ZECUSD": "ZEC", "ATMUSD": "ATOM", "BNBUSD": "BNB",
           "CRVUSD": "CRV", "ETCUSD": "ETC", "INCUSD": "1INCH", "NERUSD": "NEAR",
           "SANUSD": "SAND", "SUSUSD": "SUSHI", "XTZUSD": "XTZ", "GRTUSD": "GRT",
           "HBARUSD": "HBAR", "WLDUSD": "WLD", "WIFUSD": "WIF", "TRUMPUSD": "TRUMP",
           "BERAUSD": "BERA", "ONDOUSD": "ONDO", "WLFIUSD": "WLFI", "SKYUSD": "SKY",
           "HYPEUSD": "HYPE"}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    import MetaTrader5 as mt5
    if not mt5.initialize():
        print("MT5 indisponible", mt5.last_error())
        return 1
    barres, infos = {}, {}
    for s in CRYPTOS:
        mt5.symbol_select(s, True)
        time.sleep(0.3)
        i = mt5.symbol_info(s)
        r = None
        for n in (99_000, 70_000, 50_000, 30_000, 10_000):
            r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M5, 0, n)
            if r is not None and len(r):
                break
        if r is None or not len(r):
            print(f"{s}: aucune donnee ({mt5.last_error()})")
            continue
        d = pd.DataFrame(r)
        d["time"] = pd.to_datetime(d["time"], unit="s")
        d["spread_bps"] = d["spread"] * i.point / d["close"] * 1e4
        barres[s] = d
        infos[s] = {"actif": CRYPTOS[s], "point": i.point, "contrat": i.trade_contract_size,
                    "lot_min": i.volume_min, "pas_lot": i.volume_step, "lot_max": i.volume_max,
                    "swap_long": i.swap_long, "swap_short": i.swap_short,
                    "swap_mode": int(i.swap_mode)}
        pc = d["close"].shift(1)
        tr = np.maximum(d["high"] - d["low"], np.maximum((d["high"] - pc).abs(), (d["low"] - pc).abs()))
        atr = (tr.rolling(14).mean() / d["close"] * 1e4).median()
        sp = d.loc[d["spread"] > 0, "spread_bps"].median()
        print(f"{s:9s} {len(d):6d} bougies M5  {d['time'].iloc[0]} -> {d['time'].iloc[-1]}  "
              f"spread median {sp:6.2f} bps  ATR M5 {atr:6.1f} bps  spread/ATR {sp / atr:5.2f}  "
              f"swap {i.swap_long:g}/{i.swap_short:g} %/an", flush=True)
    mt5.shutdown()
    pd.to_pickle({"barres": barres, "infos": infos}, CACHE)
    print(f"ecrit {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
