# -*- coding: utf-8 -*-
"""Bougies H1 MT5 des indices du jeu multi-marches H1, avec leur spread.

2026-09-28, demande du proprietaire : revenir a la configuration du run
kairos_jeu_h1_01 (+447 $ au test du fold 1) et ajouter des indices volatils
a faible spread pour augmenter le nombre de trades. Choix, mesure dans MT5 le
2026-09-28 sur les 3000 dernieres bougies H1 : ATR H1 >= 20 bps et spread
median <= ~5 % de l'ATR :
    NAS100 28 bps / 0.27   GER40 28 / 0.66   UK100 21 / 0.68
    FRA40 23 / 0.86        HK50 33 / 1.57    US2000 24 / 1.26
Ecartes : DJ30 et SP500 (ATR 16-17 bps, et tres correles au NAS100), les
indices chers ou exotiques (CHINA50, SGP20, VIX, BVSPX au swap enorme).

CE FICHIER NE FAIT QUE LIRE : bougies H1 (spread de chaque bougie compris)
et caracteristiques de chaque symbole, dont le swap. Aucun ordre.

    python telecharge_h1_mt5.py
"""
import sys
import time

import pandas as pd

CACHE = "cache_h1_mt5.pkl"
INDICES = ["NAS100", "GER40", "UK100", "FRA40", "HK50", "US2000",
           # 2026-09-28, demande du proprietaire : « beaucoup plus d'indices,
           # chacun avec son petit cerveau ». Regle : tout indice d'actions de
           # Vantage dont le spread median vaut au plus 25 % de l'ATR H1, sans
           # les doublons « ft » (contrats a terme des memes indices) ni ce qui
           # n'est pas un indice d'actions (VIX, USDX) ; SGP20 est trop cher.
           "DJ30", "SP500", "SPI200", "EU50", "ES35", "TWINDEX", "BVSPX",
           "CHINA50", "HKTECH"]
DEBUT = pd.Timestamp("2017-01-01")

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
    for s in INDICES:
        t0 = time.time()
        mt5.symbol_select(s, True)
        i = mt5.symbol_info(s)
        r = None
        for n in (99_000, 70_000, 50_000, 30_000):
            r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_H1, 0, n)
            if r is not None and len(r):
                break
        if r is None or not len(r):
            print(f"{s}: aucune donnee ({mt5.last_error()})")
            continue
        d = pd.DataFrame(r)
        d["time"] = pd.to_datetime(d["time"], unit="s")
        d = d[d["time"] >= DEBUT].reset_index(drop=True)
        barres[s] = d
        infos[s] = {"point": i.point, "contrat": i.trade_contract_size, "lot_min": i.volume_min,
                    "pas_lot": i.volume_step, "swap_long": i.swap_long, "swap_short": i.swap_short,
                    "swap_mode": str(mt5.symbol_info(s)._asdict().get("swap_mode")),
                    "swap_rollover3days": i.swap_rollover3days}
        sp = (d["spread"] * i.point / d["close"] * 1e4)
        print(f"{s:7s} {len(d):6d} bougies H1  {d['time'].iloc[0]} -> {d['time'].iloc[-1]}  "
              f"spread median {sp.median():.2f} bps (zero : {100 * (d['spread'] == 0).mean():.0f} %)  "
              f"{time.time() - t0:.1f} s", flush=True)
    mt5.shutdown()
    pd.to_pickle({"barres": barres, "infos": infos}, CACHE)
    print(f"ecrit {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
