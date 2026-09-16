"""PAXG sur Binance est-il l'or que le courtier cote ?

POURQUOI LA QUESTION SE POSE. L'or serait le seul instrument capable d'ajouter
de vraies occasions INDEPENDANTES : marche sans rapport avec le Bitcoin, spread
trois fois plus serre, et aucun saut de reouverture n'a franchi le stop en trois
ans de mesure. Mais quatre des 260 colonnes — part acheteuse agressive, taille
moyenne de trade, intensite — n'existent pas dans un flux CFD. MT5 ne donne que
l'OHLC et un `tick_volume` qui n'est pas un volume.

PAXG les aurait toutes : c'est une paire SPOT Binance, avec les memes archives
que BTCUSDT, depuis aout 2020. L'entrainement se ferait donc sur PAXG et
l'execution sur XAUUSD chez le courtier.

CE QUE CE SAUT SUPPOSE, ET QU'IL FAUT MESURER. PAXG est un jeton adosse a l'or,
pas l'or. Il porte sa propre prime, sa propre liquidite — 22 M$ par jour contre
1 600 pour BTCUSDT — et son carnet est celui d'un marche cent fois plus mince.
Entrainer un modele sur le flux d'ordres de PAXG pour trader un CFD sur l'or
n'a de sens que si les deux series decrivent le meme marche.

TROIS CHIFFRES TRANCHENT :

  la correlation des RENDEMENTS a cinq minutes — si elle est basse, les
  colonnes de prix elles-memes ne transferent pas ;
  la volatilite relative — si PAXG bouge autrement, la geometrie des barrieres
  apprise sur lui ne decrit pas le trade execute ;
  la part de barres VIDES — un jeton mince passe des minutes sans echange, et
  une barre sans trade n'a pas de flux a offrir.

    python mesure_paxg.py
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

BASE = "https://api.binance.com/api/v3/klines"
N_BARRES = 100_000          # ~1 an de M5


def _get(url: str):
    with urllib.request.urlopen(url, timeout=25) as r:
        return json.loads(r.read().decode())


def klines(symbole: str, n: int = N_BARRES) -> pd.DataFrame:
    """M5 depuis Binance, format minimal : temps, OHLC, volume, nb de trades."""
    import time as _t
    morceaux, fin, restant = [], None, n
    while restant > 0:
        u = f"{BASE}?symbol={symbole}&interval=5m&limit={min(1000, restant)}"
        if fin is not None:
            u += f"&endTime={fin}"
        lignes = _get(u)
        if not lignes:
            break
        d = pd.DataFrame(lignes, columns=[
            "t", "o", "h", "l", "c", "v", "tc", "qv", "nt", "tb", "tq", "_"])
        morceaux.append(d)
        restant -= len(d)
        fin = int(d["t"].iloc[0]) - 1
        _t.sleep(0.10)
    d = pd.concat(morceaux, ignore_index=True).drop_duplicates(subset="t")
    d["time"] = pd.to_datetime(d["t"].astype("int64"), unit="ms")
    for c in ("o", "h", "l", "c", "v", "qv"):
        d[c] = d[c].astype(np.float64)
    d["nt"] = d["nt"].astype(np.int64)
    return d.sort_values("time").reset_index(drop=True)


def atr_bps(d: pd.DataFrame) -> float:
    h, l, c = d["h"].to_numpy(), d["l"].to_numpy(), d["c"].to_numpy()
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return float(1e4 * np.median(np.convolve(tr, np.ones(14) / 14, "valid"))
                 / np.median(c))


def mt5_or(n: int = N_BARRES) -> pd.DataFrame | None:
    """XAUUSD chez le courtier, pour la comparaison."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return None
    if not mt5.initialize():
        return None
    mt5.symbol_select("XAUUSD", True)
    r = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, n)
    mt5.shutdown()
    if r is None:
        return None
    d = pd.DataFrame(r)
    d["time"] = pd.to_datetime(d["time"], unit="s")
    return d[["time", "open", "high", "low", "close"]]


def main() -> int:
    print("PAXGUSDT depuis Binance…")
    p = klines("PAXGUSDT")
    print(f"  {len(p):,} barres M5  {p['time'].iloc[0]} -> {p['time'].iloc[-1]}")

    vide = (p["nt"] == 0).mean()
    print(f"\nLIQUIDITE")
    print(f"  barres SANS aucun echange : {100*vide:.2f} %")
    print(f"  trades par barre : median {p['nt'].median():.0f}, "
          f"p10 {p['nt'].quantile(0.10):.0f}, p01 {p['nt'].quantile(0.01):.0f}")
    print(f"  volume par barre : median {p['qv'].median():,.0f} $")
    print(f"  ATR/prix a 5 min : {atr_bps(p):.2f} bps")

    o = mt5_or()
    if o is None or len(o) < 5000:
        print("\nXAUUSD indisponible : comparaison impossible.")
        return 1

    print(f"\nXAUUSD chez le courtier : {len(o):,} barres  "
          f"{o['time'].iloc[0]} -> {o['time'].iloc[-1]}")
    print(f"  ATR/prix a 5 min : "
          f"{atr_bps(o.rename(columns={'high':'h','low':'l','close':'c'})):.2f} bps")

    # ALIGNEMENT HORAIRE, detecte et non suppose. Binance horodate en UTC, le
    # serveur du courtier tourne en UTC+2 ou +3 selon l'heure d'ete. Correler
    # les PRIX ne marcherait pas — sur douze heures le trend domine et tous les
    # decalages correlent a 0.99. Sur les RENDEMENTS, le bon decalage sort seul.
    pr = p[["time", "c"]].rename(columns={"c": "paxg"})
    orr = o[["time", "close"]].rename(columns={"close": "xau"})
    print("\nALIGNEMENT (correlation des rendements selon le decalage)")
    best, best_r = None, -9
    for dec in range(-4, 5):
        a = orr.copy()
        a["time"] = a["time"] - pd.Timedelta(hours=dec)
        m = pd.merge(pr, a, on="time", how="inner")
        if len(m) < 2000:
            continue
        rp = np.diff(np.log(m["paxg"].to_numpy()))
        rx = np.diff(np.log(m["xau"].to_numpy()))
        ok = np.isfinite(rp) & np.isfinite(rx)
        r = float(np.corrcoef(rp[ok], rx[ok])[0, 1])
        marque = ""
        if r > best_r:
            best_r, best, marque = r, dec, "  <-"
        print(f"  decalage {dec:+d} h : {r:+.4f}   ({len(m):,} barres communes){marque}")

    print(f"\nVERDICT")
    print(f"  meilleur decalage {best:+d} h, correlation des rendements "
          f"{best_r:+.4f}")
    if best_r < 0.80:
        print("  PAXG ne suit PAS l'or du courtier d'assez pres. Entrainer sur")
        print("  l'un pour executer sur l'autre reviendrait a changer de marche")
        print("  entre l'apprentissage et la production.")
    else:
        print("  Les deux series decrivent le meme marche a cinq minutes. Le")
        print("  transfert des colonnes de PRIX est defendable ; celui du FLUX")
        print("  reste a juger — un carnet a 22 M\\$/jour n'est pas celui de l'or.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
