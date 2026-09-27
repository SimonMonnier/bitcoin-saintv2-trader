# -*- coding: utf-8 -*-
"""Bougies M15 MT5 de plusieurs marches, avec spread — pour le jeu multi-marches.

2026-09-27, demande du proprietaire : un seul jeu qui trade le BTC, l'ETH,
l'or et les indices a petit spread et bonne volatilite, pour maximiser le
nombre de trades et le profit en gardant un drawdown bas.

CE FICHIER NE FAIT QUE LIRE : bougies M15 (spread de chaque bougie compris)
et caracteristiques de chaque symbole. Aucun ordre.

    python telecharge_m15_mt5.py
"""
import sys
import time

import pandas as pd

CACHE = "cache_m15_mt5.pkl"
UNIVERS = ["BTCUSD", "ETHUSD", "XAUUSD", "NAS100", "SP500", "DJ30", "GER40"]
N_BARRES = 100_000

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
    for s in UNIVERS:
        t0 = time.time()
        mt5.symbol_select(s, True)
        i = mt5.symbol_info(s)
        r = None
        for n in (N_BARRES, 70_000, 50_000, 30_000):
            r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M15, 0, n)
            if r is not None and len(r) > 0:
                break
        if i is None or r is None or len(r) == 0:
            print(f"{s:8s} INDISPONIBLE {mt5.last_error()}", flush=True)
            continue
        d = pd.DataFrame(r)
        d["time"] = pd.to_datetime(d["time"], unit="s")
        d["spread_bps"] = d["spread"] * i.point / d["close"] * 1e4
        barres[s] = d
        infos[s] = {"point": i.point, "contrat": i.trade_contract_size,
                    "lot_min": i.volume_min, "pas_lot": i.volume_step,
                    "swap_long": i.swap_long, "swap_short": i.swap_short}
        a = ((d["high"] - d["low"]) / d["close"] * 1e4)
        nz = d["spread_bps"][d["spread_bps"] > 0]
        print(f"{s:8s} {d['time'].iloc[0]:%Y-%m-%d} -> {d['time'].iloc[-1]:%Y-%m-%d}  "
              f"{len(d):6d} bougies  spread med {nz.median() if len(nz) else float('nan'):5.2f} bps "
              f"(barres a zero {100 * (d['spread_bps'] == 0).mean():.0f} %)  "
              f"amplitude M15 med {a.median():5.1f} bps  ({time.time() - t0:.0f} s)", flush=True)
    mt5.shutdown()
    pd.to_pickle({"barres": barres, "infos": infos}, CACHE)
    print(f"\n{len(barres)} marches ecrits dans {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
