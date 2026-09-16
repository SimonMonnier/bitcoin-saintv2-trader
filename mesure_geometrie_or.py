"""La geometrie de l'OR, derivee et non recopiee du BTC.

POURQUOI ELLE NE SE COPIE PAS. L'ATR relatif de l'or vaut 6.6 points de base
contre 15.9 pour le BTC. Un stop de 6xATR fait donc 95 bps sur le BTC et
seulement 23 sur l'or — et les sauts de reouverture de l'or ont un p95 a
24 bps. A cette largeur, un saut sur vingt franchit le stop, et la perte
depasse alors 1 R : tout le raisonnement en unites de risque s'effondre.

Le +0.1646 R mesure plus tot sur l'or portait justement sur cette geometrie
recopiee. Il est donc a la fois trop optimiste — `cibles` suppose que le stop
s'execute a son prix, alors qu'un saut le franchit — et pas representatif de
ce que l'or donnerait avec une largeur qui lui convient.

CE QUI EST MESURE ICI. La meme chose que pour le BTC : rendement par an,
friction, duree, et tenue quand la friction est plus chere que supposee. Plus
une colonne qui n'existait pas pour le BTC — la part des sauts qui franchissent
le stop — parce que c'est elle qui decide de la largeur minimale.

LA FRICTION DE L'OR N'EST PAS CELLE DU BTC. Spread releve : 0.57 point de base
contre 2.24. `cibles` porte les constantes du BTC, donc on corrige ici.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import cibles as C

SPREAD_OR_BPS = 0.57          # releve chez le courtier
SLIP_ENTREE_BPS = 1.0         # memes hypotheses que pour le BTC
SLIP_SORTIE_BPS = 2.0
STOPS = (6.0, 10.0, 16.0, 24.0, 36.0, 50.0)
PAS = 3


class Cfg:
    """Stop variable, trailing a 2 R comme sur le BTC, aucun objectif."""
    def __init__(self, sl):
        self.atr_sl_mult = sl
        self.use_tp = False
        self.atr_tp_mult = 0.0
        self.use_be_trail = True
        self.atr_be_mult = 1e9
        self.atr_trail_mult = 2.0 * sl
        self.atr_trail_dist = 2.0 * sl


def charge_or():
    import MetaTrader5 as mt5
    mt5.initialize()
    mt5.symbol_select("XAUUSD", True)
    r = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 500_000)
    mt5.shutdown()
    d = pd.DataFrame(r)
    d["time"] = pd.to_datetime(d["time"], unit="s")
    h, l, c = (d[k].to_numpy(float) for k in ("high", "low", "close"))
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    d = d.iloc[1:].reset_index(drop=True)
    d["atr_14"] = pd.Series(tr).rolling(14).mean().to_numpy()
    return d.dropna().reset_index(drop=True)


def main() -> int:
    # La friction de l'or remplace celle du BTC, le temps de cette mesure.
    C.SPREAD_BPS = SPREAD_OR_BPS
    C.SLIP_ENTREE_BPS = SLIP_ENTREE_BPS
    C.SLIP_SORTIE_BPS = SLIP_SORTIE_BPS
    fric = SPREAD_OR_BPS + SLIP_ENTREE_BPS + SLIP_SORTIE_BPS

    d = charge_or()
    t = d["time"]
    ans = (t.iloc[-1] - t.iloc[0]).days / 365.25
    atr_rel = float(np.median(d["atr_14"] / d["close"]))
    print(f"XAUUSD {len(d):,} barres, {t.iloc[0]:%Y-%m-%d} -> "
          f"{t.iloc[-1]:%Y-%m-%d} ({ans:.1f} ans)")
    print(f"ATR relatif {1e4*atr_rel:.1f} bps (BTC : 15.9)")
    print(f"friction {fric:.2f} bps par aller-retour (BTC : 4.85)\n")

    # Les sauts, une fois pour toutes : ils ne dependent pas du stop.
    dt = t.diff().dt.total_seconds().to_numpy()
    trous = np.flatnonzero(dt > 1800)
    o, c = d["open"].to_numpy(float), d["close"].to_numpy(float)
    atr = d["atr_14"].to_numpy()
    saut = np.abs(o[trous] - c[trous - 1]) / c[trous - 1]

    idx = np.arange(100, len(d) - C.BORNE_DEFAUT - 2, PAS)
    print(f"{'stop':>6} {'risque':>8} {'friction':>9} {'duree':>8} "
          f"{'trades/an':>10} {'E[R] sym':>9} {'+/-':>7} {'R/an':>7} "
          f"{'sauts > 1R':>11} {'R/an x1.5':>10}")
    print("-" * 94)
    for sl in STOPS:
        cfg = Cfg(sl)
        ra, rv, da, _ = C.rendements(d, idx, cfg, durees=True)
        ok = np.isfinite(ra) & np.isfinite(rv) & np.isfinite(da)
        if ok.sum() < 200:
            continue
        sym = (ra[ok] - rv[ok]) / 2.0
        dur = da[ok]
        pos = idx[ok]
        garde, libre = [], -1
        for k in range(len(pos)):
            if pos[k] < libre:
                continue
            garde.append(k)
            libre = pos[k] + dur[k]
        g = np.array(garde, dtype=int)
        y = sym[g]
        se = y.std(ddof=1) / np.sqrt(len(y))
        par_an = len(y) / ans
        risque_bps = sl * atr_rel * 1e4
        fric_R = fric / risque_bps
        # Part des sauts qui franchissent CE stop.
        stop_abs = sl * atr[trous - 1] / c[trous - 1]
        part = float((saut > stop_abs).mean())
        print(f"{sl:>5g}x {risque_bps:>7.0f}b {fric_R:>8.3f}R "
              f"{float(np.median(dur))*5/60:>7.1f}h {par_an:>10.0f} "
              f"{y.mean():>+9.4f} {se:>7.4f} {par_an*y.mean():>+7.1f} "
              f"{100*part:>10.1f}% {par_an*(y.mean()-0.5*fric_R):>+10.1f}")

    print("\nLECTURE. 'sauts > 1R' est la part des reouvertures qui franchissent")
    print("le stop : au-dela, la perte depasse 1 R et `cibles` ne le modelise")
    print("pas, donc la colonne R/an est optimiste d'autant. La largeur retenue")
    print("doit rendre cette part negligeable AVANT de comparer les rendements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
