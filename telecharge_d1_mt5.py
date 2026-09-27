# -*- coding: utf-8 -*-
"""Bougies journalieres MT5 de ~20 marches, avec spread et swap — pour l'etude de tendance.

2026-09-27. Le scalping BTC M1 a montre que le signal egale le cout. L'etude
suivante porte sur ce que font les professionnels qui gagnent en
directionnel : le SUIVI DE TENDANCE diversifie, sur bougies journalieres.

CE FICHIER NE FAIT QUE LIRE : bougies D1 (spread de chaque barre compris) et
caracteristiques de chaque symbole (swap, contrat, lot minimum). Aucun
ordre. Le resultat est ecrit dans `CACHE` pour ne plus dependre de MT5.

    python telecharge_d1_mt5.py
"""
import sys
import time

import pandas as pd

CACHE = "cache_d1_mt5.pkl"
UNIVERS = ["BTCUSD", "ETHUSD", "XAUUSD", "XAGUSD", "COPPER-C", "CL-OIL",
           "SP500", "NAS100", "DJ30", "GER40", "UK100", "Nikkei225",
           "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD",
           "USDCHF", "AUDJPY", "USDX"]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    import MetaTrader5 as mt5
    if not mt5.initialize():
        print("MT5 indisponible", mt5.last_error())
        return 1
    modes = {getattr(mt5, n): n for n in dir(mt5) if n.startswith("SYMBOL_SWAP_MODE_")}
    barres, infos = {}, {}
    for s in UNIVERS:
        t0 = time.time()
        mt5.symbol_select(s, True)
        i = mt5.symbol_info(s)
        r = None
        for n in (5000, 3000, 2000):
            r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_D1, 0, n)
            if r is not None and len(r) > 0:
                break
        if i is None or r is None or len(r) == 0:
            print(f"{s:10s} INDISPONIBLE {mt5.last_error()}", flush=True)
            continue
        d = pd.DataFrame(r)
        d["time"] = pd.to_datetime(d["time"], unit="s")
        barres[s] = d
        infos[s] = {"point": i.point, "swap_long": i.swap_long,
                    "swap_short": i.swap_short,
                    "swap_mode": modes.get(i.swap_mode, str(i.swap_mode)),
                    "contrat": i.trade_contract_size, "lot_min": i.volume_min,
                    "pas_lot": i.volume_step, "devise_profit": i.currency_profit,
                    "swap_rollover3days": i.swap_rollover3days}
        sp = (d["spread"] * i.point / d["close"] * 1e4).median()
        print(f"{s:10s} {d['time'].iloc[0]:%Y-%m-%d} -> {d['time'].iloc[-1]:%Y-%m-%d} "
              f"{len(d):5d} barres  spread med {sp:6.2f} bps  swap L {i.swap_long:9.3f} "
              f"S {i.swap_short:9.3f} {infos[s]['swap_mode']}  ({time.time() - t0:.0f} s)",
              flush=True)
    mt5.shutdown()
    pd.to_pickle({"barres": barres, "infos": infos}, CACHE)
    print(f"\n{len(barres)} marches ecrits dans {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
