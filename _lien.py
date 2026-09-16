"""Y a-t-il un lien entre le BTC et l'or ? Et surtout : entre leurs TRADES ?

DEUX QUESTIONS, ET LA SECONDE DECIDE.

  1. LES PRIX corrèlent-ils ? C'est ce qu'on regarde d'habitude, et ce n'est
     pas ce qui compte ici.

  2. LES RENDEMENTS DE LA STRATEGIE corrèlent-ils ? Deux actifs peuvent avoir
     des prix sans rapport et produire des trades de suivi de tendance qui
     gagnent et perdent AUX MEMES MOMENTS — si les tendances arrivent en meme
     temps, l'or n'ajoute aucune occasion independante et tout le benefice
     attendu disparait.

On mesure la meme geometrie, aux MEMES DATES, sur les deux instruments.
"""
import numpy as np, pandas as pd
import MetaTrader5 as mt5
import cibles as C, training as T

cfg = T.PPOConfig()
mt5.initialize()
res = {}
for sym in ("BTCUSD", "XAUUSD"):
    mt5.symbol_select(sym, True)
    r = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 500_000)
    if r is None or len(r) < 10000:
        print(f"{sym} : historique insuffisant ({0 if r is None else len(r)})")
        continue
    d = pd.DataFrame(r)
    d["time"] = pd.to_datetime(d["time"], unit="s")
    h, l, c = d["high"].to_numpy(float), d["low"].to_numpy(float), d["close"].to_numpy(float)
    tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    d = d.iloc[1:].reset_index(drop=True); d["atr_14"] = atr
    d = d.dropna().reset_index(drop=True)
    res[sym] = d
    print(f"{sym} : {len(d):,} barres, {d['time'].iloc[0]:%Y-%m-%d} -> "
          f"{d['time'].iloc[-1]:%Y-%m-%d}, ATR/prix "
          f"{1e4*np.median(d['atr_14']/d['close']):.1f} bps")
mt5.shutdown()
if len(res) < 2:
    raise SystemExit(0)

# Dates communes, a la minute pres.
a, b = res["BTCUSD"], res["XAUUSD"]
com = pd.merge(a[["time"]].assign(ia=np.arange(len(a))),
               b[["time"]].assign(ib=np.arange(len(b))), on="time")
print(f"\n{len(com):,} barres communes "
      f"({com['time'].iloc[0]:%Y-%m-%d} -> {com['time'].iloc[-1]:%Y-%m-%d})")

# 1. Correlation des rendements de PRIX, a l'heure.
ia, ib = com["ia"].to_numpy(), com["ib"].to_numpy()
pa = a["close"].to_numpy()[ia]; pb = b["close"].to_numpy()[ib]
ra = np.diff(np.log(pa))[::12]; rb = np.diff(np.log(pb))[::12]
n = min(len(ra), len(rb))
print(f"\n1. PRIX — correlation des rendements horaires : "
      f"{np.corrcoef(ra[:n], rb[:n])[0,1]:+.4f}")

# 2. Correlation des RENDEMENTS DE LA STRATEGIE, memes dates.
pas = 288   # une entree par jour
idx_a = np.arange(100, len(com) - C.BORNE_DEFAUT - 2, pas)
sym_r = {}
for nom, df_, col in (("BTCUSD", a, ia), ("XAUUSD", b, ib)):
    sub = df_.iloc[col].reset_index(drop=True)
    ra_, rv_ = C.rendements(sub, idx_a, cfg)
    sym_r[nom] = (ra_ - rv_) / 2.0
ok = np.isfinite(sym_r["BTCUSD"]) & np.isfinite(sym_r["XAUUSD"])
x, y = sym_r["BTCUSD"][ok], sym_r["XAUUSD"][ok]
print(f"2. STRATEGIE — {int(ok.sum()):,} dates communes resolues des deux cotes")
print(f"   correlation des rendements symetriques : {np.corrcoef(x, y)[0,1]:+.4f}")
print(f"   E[R] BTC {x.mean():+.4f} +/- {x.std(ddof=1)/np.sqrt(len(x)):.4f}")
print(f"   E[R] or  {y.mean():+.4f} +/- {y.std(ddof=1)/np.sqrt(len(y)):.4f}")
print(f"\n   Si cette correlation est proche de zero, l'or DOUBLE reellement")
print(f"   le nombre d'occasions independantes. Si elle est forte, il ne fait")
print(f"   que repeter le meme pari.")
