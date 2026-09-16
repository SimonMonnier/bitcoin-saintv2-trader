"""L'or est-il tradable par ce modele ? Deux chiffres decident.

POURQUOI CETTE QUESTION. Le mur du depot est le nombre d'occasions
INDEPENDANTES : 4 516 sur neuf ans de BTC, quand il en faudrait de quoi porter
1 200 a 4 700 trades de test pour qu'un avantage soit lisible. On en a 787.

L'or est la seule piste qui en ajoute VRAIMENT. Les paires crypto sont
correlees a BTC — leurs trades ne s'additionnent qu'en partie — et leurs
spreads chez ce courtier vont de 1.8 a 42 fois celui du Bitcoin. L'or, lui, est
un marche sans rapport, et sa volatilite RELATIVE a cinq minutes est quasi
identique a celle de BTC (10.34 contre 11.73 bps d'ATR sur prix).

DEUX CHIFFRES MANQUENT, ET ILS TRANCHENT.

  LE SPREAD EN SEANCE. Mesure marches fermes, il lit 0.00 — un cours fige, pas
  une mesure. MT5 conserve le spread BARRE PAR BARRE dans son historique :
  c'est la vraie distribution qu'il faut lire, pas le cours de l'instant.

  LES SAUTS DE WEEK-END. L'or ferme du vendredi soir au dimanche soir et rouvre
  souvent avec un ecart. Un stop peut alors etre FRANCHI sans etre execute a son
  prix : la perte depasse 1 R, et le R:R annonce devient decoratif. Tout le
  pipeline M5 suppose des barres continues.

CE QUE CE FICHIER NE FAIT PAS. Il ne dit pas si le modele a un avantage sur
l'or — seulement si la GEOMETRIE y est jouable. C'est la meme distinction
qu'entre une friction acceptable et un avantage reel.

    python mesure_or.py [symbole]
"""

from __future__ import annotations

import sys

import numpy as np

SL_MULT = 8.0
SLIPPAGE_BPS = 3.0
N_BARRES = 200_000          # ~2 ans de M5


def _rates(sym: str, n: int = N_BARRES):
    import MetaTrader5 as mt5
    mt5.symbol_select(sym, True)
    r = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, n)
    if r is None or len(r) < 5000:
        return None
    return r


def analyse_spread(sym: str, r) -> float:
    """Distribution du spread EN SEANCE, lue barre par barre dans l'historique.

    Le champ `spread` de MT5 est en points, comme `symbol_info().spread`, mais
    il est enregistre pour CHAQUE barre : on lit donc ce qui s'est reellement
    pratique, y compris aux heures creuses, au lieu d'un instantane.
    """
    import MetaTrader5 as mt5
    i = mt5.symbol_info(sym)
    prix = r["close"].astype(np.float64)
    sp_bps = 1e4 * r["spread"].astype(np.float64) * i.point / prix
    # Les barres sans echange (volume nul) portent un spread qui n'a pas ete
    # cote : les inclure ferait passer la fermeture pour une seance.
    vif = r["tick_volume"] > 0
    s = sp_bps[vif]

    heures = ((r["time"][vif] // 3600) % 24).astype(int)
    print(f"{sym} — spread sur {len(s):,} barres cotees "
          f"({100*vif.mean():.0f} % des barres)")
    print(f"  median {np.median(s):.2f} bps   moyen {s.mean():.2f}   "
          f"p90 {np.percentile(s, 90):.2f}   p99 {np.percentile(s, 99):.2f}")
    creux = np.median(s[(heures >= 22) | (heures < 6)])
    plein = np.median(s[(heures >= 12) & (heures < 20)])
    print(f"  par heure UTC : creux (22h-6h) {creux:.2f} bps   "
          f"plein (12h-20h) {plein:.2f} bps")
    return float(np.median(s))


def analyse_sauts(sym: str, r) -> None:
    """Les ecarts de reouverture, rapportes a l'UNITE DE RISQUE.

    Un saut ne compte pas en pourcentage mais en fraction du stop : c'est lui
    qui dit si la barriere a ete franchie. Au-dela de 1.0 le stop s'execute
    plus loin qu'il n'etait pose, et la perte depasse 1 R.
    """
    t = r["time"].astype(np.int64)
    o, c = r["open"].astype(np.float64), r["close"].astype(np.float64)
    h, l = r["high"].astype(np.float64), r["low"].astype(np.float64)

    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = np.convolve(tr, np.ones(14) / 14, "same")
    risque = SL_MULT * atr                      # en prix, aligne sur r[1:]

    dt = np.diff(t) / 60.0                      # minutes entre deux barres
    saut = np.abs(o[1:] - c[:-1])
    interruption = dt > 30.0                    # au-dela, ce n'est plus la seance

    n_sem = (dt > 12 * 60).sum()
    print(f"\n{sym} — interruptions sur {len(dt):,} intervalles")
    print(f"  {interruption.sum():,} coupures de plus de 30 min "
          f"({100*interruption.mean():.2f} %), dont {n_sem:,} de plus de 12 h")
    if interruption.sum() == 0:
        print("  marche CONTINU : aucun saut de reouverture a craindre.")
        return

    s_bps = 1e4 * saut[interruption] / c[:-1][interruption]
    s_r = saut[interruption] / np.maximum(risque[interruption], 1e-9)
    print(f"  ecart de reouverture : median {np.median(s_bps):.1f} bps, "
          f"p90 {np.percentile(s_bps, 90):.1f}, max {s_bps.max():.1f}")
    print(f"  en fraction du stop ({SL_MULT:.0f}xATR) : median {np.median(s_r):.2f} R, "
          f"p90 {np.percentile(s_r, 90):.2f}, max {s_r.max():.2f}")
    depasse = float((s_r > 1.0).mean())
    print(f"  sauts qui FRANCHISSENT le stop : {100*depasse:.1f} % des coupures")

    # Part des trades exposes : un trade de duree D couvre une coupure si son
    # entree tombe dans les D minutes qui la precedent.
    duree_min = 12 * 60
    total_min = (t[-1] - t[0]) / 60.0
    expose = interruption.sum() * duree_min / total_min
    print(f"  part des trades de 12 h exposes a une coupure : {100*expose:.1f} %")
    print(f"  surcout attendu : {100*expose*depasse:.2f} % des trades "
          f"perdent plus de 1 R")


def main() -> int:
    import MetaTrader5 as mt5
    if not mt5.initialize():
        print("MT5 injoignable :", mt5.last_error())
        return 1
    syms = sys.argv[1:] or ["XAUUSD", "BTCUSD"]
    for sym in syms:
        r = _rates(sym)
        if r is None:
            print(f"{sym} : historique M5 insuffisant\n")
            continue
        d0 = np.datetime64(int(r["time"][0]), "s")
        d1 = np.datetime64(int(r["time"][-1]), "s")
        print("=" * 68)
        print(f"{sym} : {len(r):,} barres M5, {d0} -> {d1}")
        sp = analyse_spread(sym, r)
        analyse_sauts(sym, r)
        risque_bps = SL_MULT * 1e4 * np.median(
            np.convolve(np.maximum(
                r["high"][1:] - r["low"][1:],
                np.abs(r["high"][1:] - r["close"][:-1])),
                np.ones(14) / 14, "valid")) / np.median(r["close"])
        print(f"\n  friction a SL {SL_MULT:.0f}xATR : "
              f"({sp:.2f} + {SLIPPAGE_BPS:.0f}) / {risque_bps:.0f} = "
              f"{(sp + SLIPPAGE_BPS)/risque_bps:.4f} R par trade")
        print()
    mt5.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
