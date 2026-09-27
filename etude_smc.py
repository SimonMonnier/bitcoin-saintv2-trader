# -*- coding: utf-8 -*-
"""Le "Smart Money" / ICT, la "strategie des institutions", appliquee A LA LETTRE.

2026-09-27, demande du proprietaire, apres l'etude des PDF Ichimoku : coder
les regles SMC classiques avec le meme protocole (etude_pdf_ichimoku.py).
Rien n'est regle apres le premier lancement.

LES DEFINITIONS, FIXEES AVANT LE PREMIER LANCEMENT

  SWING (fractale a 2 bougies) : un plus haut plus haut que les 2 bougies de
  chaque cote. Il n'est CONNU que 2 bougies apres, a la cloture de la
  deuxieme : c'est le piege classique des backtests SMC, qui lisent les
  swings le jour meme ou ils se forment.
  STRUCTURE haussiere : les deux derniers swings hauts et bas confirmes
  montent (plus haut plus haut, plus bas plus haut). Baissiere : l'inverse.
  LIQUIDITE : un swing pas encore depasse par les prix depuis sa formation.
  FVG (fair value gap) haussier : le plus bas d'une bougie au-dessus du plus
  haut de l'avant-derniere ; la zone est entre les deux. Baissier : inverse.
  ORDER BLOCK haussier : la derniere bougie baissiere au plus tard au creux
  de la jambe qui casse la structure (5 bougies en arriere au plus).
  KILLZONES : Londres 2 h - 5 h et New York 7 h - 10 h, heure de New York,
  soit 9 h - 12 h et 14 h - 17 h heure serveur MT5 (New York + 7 h).

LES CINQ STRATEGIES

  1 CHASSE DE LIQUIDITE : la meche passe sous un swing bas encore intact et
    la bougie cloture au-dessus. Achat a l'ouverture suivante, stop sous la
    meche (-0.1 ATR), objectif le premier swing haut au-dessus.
  2 CHASSE + CHoCH + FVG (le "modele 2022" d'ICT) : chasse d'un swing bas
    intact, puis dans les 20 bougies une cloture au-dessus du dernier swing
    haut forme AVANT la chasse (changement de caractere), avec un FVG
    haussier dans la jambe. Ordre a cours limite sur le haut du FVG, valable
    20 bougies ; stop sous le creux de la chasse (-0.1 ATR) ; objectif le
    premier swing haut au-dessus (la liquidite d'en face).
  3 LA MEME, SEULEMENT DANS LES KILLZONES (M15 et H1 seulement).
  4 BOS + ORDER BLOCK : en structure haussiere, premiere cloture au-dessus du
    dernier swing haut (cassure de structure). Ordre limite a l'ouverture de
    l'order block, stop sous l'order block et le creux de la jambe (-0.1
    ATR), objectif le plus haut atteint depuis la cassure ; l'ordre n'est
    actif qu'en zone "discount" (sous la moitie de la jambe).
  5 BOS + FVG : la meme cassure, entree sur le haut du dernier FVG de la
    jambe, stop sous le creux de la jambe, objectif et discount identiques.

  Partout : ratio objectif / stop d'au moins 2 (l'usage SMC), s'il n'y a
  aucune liquidite au-dessus l'objectif est pose a 2 fois le risque, un seul
  trade a la fois par marche et par strategie. Ordre limite annule si
  l'objectif est touche sans execution ; si la meme bougie touche les deux,
  l'ordre est execute. Bougie de l'execution : seul le stop compte (on ne
  sait pas si l'objectif est venu avant). Ces deux choix sont les moins
  favorables a la strategie.

COUTS ET DONNEES : ceux de etude_pdf_ichimoku.py (spread reel de la bougie,
glissement 0.5 bps sur les ordres au marche et les stops, swap du courtier
chaque nuit, mise fixe 10 $ par trade = 1 % de 1000 $). En fin de sortie,
BTC et or sont relances SANS AUCUN FRAIS pour separer l'absence d'avantage
du poids des couts.

    python etude_smc.py
"""
import sys

import numpy as np
import pandas as pd

import etude_pdf_ichimoku as E

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

K_FRACTALE = 2
FEN_LIQ = 200
ATTENTE_CHOCH = 20
VALIDITE = 20
RECUL_OB = 5
KILLZONES = ((9, 12), (14, 17))
INTRADAY = ("M15", "H1")

STRATEGIES = ["1 chasse de liquidite", "2 chasse + CHoCH + FVG (ICT)",
              "3 idem, killzones seulement", "4 BOS + order block",
              "5 BOS + FVG"]


def base(df):
    o, h, l, c = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    tr = np.maximum(h - l, np.maximum(np.abs(h - E._dec(c)), np.abs(l - E._dec(c))))
    atr = E._roll(tr, E.ATR_N, "mean")
    n = len(c)
    k = K_FRACTALE
    # Les deux derniers swings CONFIRMES a la cloture de chaque bougie.
    sh = np.full((n, 2, 2), np.nan)          # [t, rang (0 = dernier), (indice, niveau)]
    sl = np.full((n, 2, 2), np.nan)
    hs, ls = [], []                           # (indice, niveau, bougie de confirmation)
    dh, dl = [], []
    for t in range(n):
        i = t - k
        if i >= k:
            if h[i] > h[i - k:i].max() and h[i] >= h[i + 1:t + 1].max():
                dh.append((i, h[i]))
                hs.append((i, h[i], t))
            if l[i] < l[i - k:i].min() and l[i] <= l[i + 1:t + 1].min():
                dl.append((i, l[i]))
                ls.append((i, l[i], t))
        for dst, src in ((sh, dh), (sl, dl)):
            if src:
                dst[t, 0] = src[-1]
            if len(src) > 1:
                dst[t, 1] = src[-2]
    hs = np.array(hs) if hs else np.zeros((0, 3))
    ls = np.array(ls) if ls else np.zeros((0, 3))
    temps = pd.to_datetime(df["time"]).dt.hour.to_numpy()
    return dict(o=o, h=h, l=l, c=c, atr=atr, sh=sh, sl=sl, hs=hs, ls=ls, heure=temps,
                ok=np.isfinite(atr) & (np.arange(n) >= 60))


def liquidite(x, t, sens, prix):
    """Premier swing confirme a t, dans les 200 bougies, au-dela de prix."""
    tab = x["hs"] if sens > 0 else x["ls"]
    if not len(tab):
        return np.nan
    m = (tab[:, 2] <= t) & (tab[:, 0] >= t - FEN_LIQ)
    v = tab[m, 1]
    a = x["atr"][t]
    if sens > 0:
        v = v[v > prix + E.ECART_OBST * a]
        return v.min() if len(v) else np.nan
    v = v[v < prix - E.ECART_OBST * a]
    return v.max() if len(v) else np.nan


def _intact(x, t, sens):
    """Dernier swing bas (sens +1) ou haut (-1) confirme avant t et jamais depasse."""
    s = x["sl"] if sens > 0 else x["sh"]
    i, v = s[t - 1, 0]
    if not np.isfinite(v):
        return np.nan
    i = int(i)
    if i + 1 <= t - 1:
        if sens > 0 and x["l"][i + 1:t].min() < v:
            return np.nan
        if sens < 0 and x["h"][i + 1:t].max() > v:
            return np.nan
    return v


def _objectif(x, t, sens, entree, stop):
    risque = (entree - stop) * sens
    if not risque > 0:
        return np.nan
    tp = liquidite(x, t, sens, x["c"][t])
    if not np.isfinite(tp) or (tp - entree) * sens <= 0:
        tp = entree + sens * E.RATIO_MIN * risque
    return tp if (tp - entree) * sens >= E.RATIO_MIN * risque - 1e-12 else np.nan


def chasses(x):
    """Strategie 1 : signaux au marche (t, sens, stop, objectif)."""
    l, h, c, a = x["l"], x["h"], x["c"], x["atr"]
    out = []
    for t in np.flatnonzero(x["ok"]):
        for sens in (1, -1):
            v = _intact(x, t, sens)
            if not np.isfinite(v):
                continue
            if sens > 0 and l[t] < v < c[t]:
                stop = l[t] - E.MARGE_MECHE * a[t]
            elif sens < 0 and h[t] > v > c[t]:
                stop = h[t] + E.MARGE_MECHE * a[t]
            else:
                continue
            tp = _objectif(x, t, sens, c[t], stop)
            if np.isfinite(tp):
                out.append((t, sens, stop, tp))
                break
    return out


def _fvg(x, debut, fin, sens):
    """Dernier FVG de la jambe [debut, fin] : (haut, bas) de la zone."""
    h, l = x["h"], x["l"]
    for i in range(fin, max(debut + 2, 2) - 1, -1):
        if sens > 0 and l[i] > h[i - 2]:
            return l[i], h[i - 2]
        if sens < 0 and h[i] < l[i - 2]:
            return l[i - 2], h[i]
    return None


def ordres_ict(x, killzones=False):
    """Strategies 2 et 3 : chasse, puis CHoCH avec FVG, puis ordre limite."""
    l, h, c, a = x["l"], x["h"], x["c"], x["atr"]
    n = len(c)
    out = []
    attente = {1: None, -1: None}
    for t in np.flatnonzero(x["ok"]):
        for sens in (1, -1):
            w = attente[sens]
            if w is not None:
                s, ref, extreme = w
                extreme = min(extreme, l[t]) if sens > 0 else max(extreme, h[t])
                attente[sens] = (s, ref, extreme)
                if t - s > ATTENTE_CHOCH:
                    attente[sens] = None
                elif (c[t] - ref) * sens > 0:
                    attente[sens] = None
                    if killzones and not any(a0 <= x["heure"][t] < a1 for a0, a1 in KILLZONES):
                        continue
                    z = _fvg(x, s, t, sens)
                    if z is None:
                        continue
                    p = z[0] if sens > 0 else z[1]
                    stop = extreme - sens * E.MARGE_MECHE * a[t]
                    tp = _objectif(x, t, sens, p, stop)
                    if np.isfinite(tp) and (c[t] - p) * sens >= 0:
                        out.append(dict(t=t, sens=sens, p=p, stop=stop, tp=tp, dyn=None))
                    continue
            if attente[sens] is None:
                v = _intact(x, t, sens)
                if not np.isfinite(v):
                    continue
                if (sens > 0 and l[t] < v) or (sens < 0 and h[t] > v):
                    ref = (x["sh"] if sens > 0 else x["sl"])[t - 1, 0, 1]
                    if np.isfinite(ref):
                        attente[sens] = (t, ref, l[t] if sens > 0 else h[t])
    out.sort(key=lambda d: d["t"])
    return out


def ordres_bos(x, zone):
    """Strategies 4 (order block) et 5 (FVG) : cassure de structure puis retour."""
    o, h, l, c, a = x["o"], x["h"], x["l"], x["c"], x["atr"]
    sh, sl = x["sh"], x["sl"]
    out = []
    for t in np.flatnonzero(x["ok"]):
        (ih1, h1), (_, h0) = sh[t - 1]
        (il1, l1), (_, l0) = sl[t - 1]
        if not np.all(np.isfinite([h1, h0, l1, l0])):
            continue
        for sens in (1, -1):
            if sens > 0 and h1 > h0 and l1 > l0 and c[t] > h1 >= c[t - 1]:
                d = int(ih1)
                j = d + int(np.argmin(l[d:t + 1]))
                ext = l[j]
            elif sens < 0 and h1 < h0 and l1 < l0 and c[t] < l1 <= c[t - 1]:
                d = int(il1)
                j = d + int(np.argmax(h[d:t + 1]))
                ext = h[j]
            else:
                continue
            if zone == "ob":
                ob = None
                for i in range(j, max(j - RECUL_OB, 0) - 1, -1):
                    if (c[i] - o[i]) * sens < 0:
                        ob = i
                        break
                if ob is None:
                    continue
                p = o[ob]
                stop = (min(l[ob], ext) - E.MARGE_MECHE * a[t] if sens > 0
                        else max(h[ob], ext) + E.MARGE_MECHE * a[t])
            else:
                z = _fvg(x, j, t, sens)
                if z is None:
                    continue
                p = z[0] if sens > 0 else z[1]
                stop = ext - sens * E.MARGE_MECHE * a[t]
            if (p - stop) * sens > 0 and (c[t] - p) * sens >= 0:
                out.append(dict(t=t, sens=sens, p=p, stop=stop, tp=np.nan, dyn=(t, ext)))
                break
    return out


def _suivi(x, sp, g, k0, sens, entree, stop, tp):
    """Apres l'execution en k0 : (sortie, bougie de sortie). Stop d'abord."""
    o, h, l, c = x["o"], x["h"], x["l"], x["c"]
    n = len(c)
    # Une ouverture au-dela du stop sort a l'ouverture, y compris sur la bougie
    # d'execution : un ordre limite rempli a une ouverture en gap sous le stop
    # est aussitot coupe a ce prix, pas au niveau du stop.
    for k in range(k0, n):
        if sens > 0:
            if l[k] <= stop:
                return (stop if o[k] >= stop else o[k]) - g, k
            if k > k0 and h[k] >= tp:
                return max(o[k], tp), k
        else:
            if h[k] + sp[k] >= stop:
                return (stop if o[k] + sp[k] <= stop else o[k] + sp[k]) + g, k
            if k > k0 and l[k] + sp[k] <= tp:
                return min(o[k] + sp[k], tp), k
    return (c[n - 1] if sens > 0 else c[n - 1] + sp[n - 1]), n - 1


def simule_limite(df, x, ordres, swap_l, swap_s, crypto, jour3, point, glis_bps):
    o, h, l = x["o"], x["h"], x["l"]
    sp = df["spread"].to_numpy(np.float64) * point
    temps = df["time"].to_numpy()
    n = len(o)
    trades, libre = [], 0
    for od in ordres:
        t, sens, p, stop = od["t"], od["sens"], od["p"], od["stop"]
        if t < libre or t + 1 >= n:
            continue
        rq = (p - stop) * sens
        fill, fin = None, min(t + VALIDITE, n - 1)
        for k in range(t + 1, fin + 1):
            if od["dyn"] is None:
                tp = od["tp"]
            else:
                b, ext = od["dyn"]
                tp = h[b:k].max() if sens > 0 else l[b:k].min()
                if (tp - p) * sens < E.RATIO_MIN * rq or (p - (ext + tp) / 2) * sens > 0:
                    continue
            # L'execution d'abord : si la meme bougie touche le prix d'entree ET
            # l'objectif, on suppose l'ordre execute (le cas le moins favorable,
            # puisque sur cette bougie seul le stop compte ensuite).
            if sens > 0 and l[k] + sp[k] <= p:
                fill = (min(o[k] + sp[k], p), k, tp)
                break
            if sens < 0 and h[k] >= p:
                fill = (max(o[k], p), k, tp)
                break
            if od["dyn"] is None and ((sens > 0 and h[k] >= tp) or (sens < 0 and l[k] + sp[k] <= tp)):
                fin = k
                break
        if fill is None:
            libre = fin + 1
            continue
        entree, k0, tp = fill
        g = glis_bps * 1e-4 * entree
        sortie, kf = _suivi(x, sp, g, k0, sens, entree, stop, tp)
        r = (sortie - entree) * sens / rq
        nuits = E._nuits(pd.Timestamp(temps[k0]), pd.Timestamp(temps[kf]), crypto, jour3)
        sw = (swap_l if sens > 0 else swap_s) * 1e-4 * entree * nuits / rq
        trades.append((pd.Timestamp(temps[k0]), sens, r + sw))
        libre = kf + 1
    return trades


def lance(df, s, tf, info, sans_frais=False):
    if sans_frais:
        df = df.assign(spread=0)
    x = base(df)
    prix = float(df["close"].iloc[-1])
    sw_l, sw_s = (0.0, 0.0) if sans_frais else E.swap_bps(info, prix)
    crypto, jour3, point = s in E.CRYPTO, info.get("swap_rollover3days", 3), info["point"]
    glis = 0.0 if sans_frais else E.GLISSEMENT_BPS
    ancien, E.GLISSEMENT_BPS = E.GLISSEMENT_BPS, glis
    try:
        res = {0: E.simule(df, x | {"kijun": np.full(len(df), np.nan)}, chasses(x),
                           sw_l, sw_s, crypto, jour3, point)}
    finally:
        E.GLISSEMENT_BPS = ancien
    res[1] = simule_limite(df, x, ordres_ict(x), sw_l, sw_s, crypto, jour3, point, glis)
    res[2] = (simule_limite(df, x, ordres_ict(x, True), sw_l, sw_s, crypto, jour3, point, glis)
              if tf in INTRADAY else None)
    res[3] = simule_limite(df, x, ordres_bos(x, "ob"), sw_l, sw_s, crypto, jour3, point, glis)
    res[4] = simule_limite(df, x, ordres_bos(x, "fvg"), sw_l, sw_s, crypto, jour3, point, glis)
    return res


def verif_causalite(df, essais=40, graine=1):
    """Les decisions prises a t ne changent pas si l'on coupe les donnees apres t."""
    x = base(df)
    gen = {"1": lambda y: [(t, (se, st, tp)) for t, se, st, tp in chasses(y)],
           "2": lambda y: [(d["t"], (d["sens"], d["p"], d["stop"], d["tp"])) for d in ordres_ict(y)],
           "4": lambda y: [(d["t"], (d["sens"], d["p"], d["stop"])) for d in ordres_bos(y, "ob")],
           "5": lambda y: [(d["t"], (d["sens"], d["p"], d["stop"])) for d in ordres_bos(y, "fvg")]}
    plein = {k: dict(f(x)) for k, f in gen.items()}
    rng = np.random.default_rng(graine)
    pts = set(int(v) for v in rng.integers(200, len(df) - 1, essais))
    for k, d in plein.items():
        ts = sorted(d)
        pts |= set(int(v) for v in rng.choice(ts, min(len(ts), essais // 2), replace=False)) if ts else set()
    for t in sorted(pts):
        xc = base(df.iloc[:t + 1].reset_index(drop=True))
        for k, f in gen.items():
            a, b = plein[k].get(t), dict(f(xc)).get(t)
            if (a is None) != (b is None) or (a is not None and not np.allclose(a, b, equal_nan=True)):
                return f"FUITE : strategie {k}, bougie {t} : {a} contre {b}"
    return f"causalite verifiee sur {len(pts)} bougies : les decisions ne lisent pas l'avenir"


def main() -> int:
    m15 = pd.read_pickle(E.M15)
    d1 = pd.read_pickle(E.D1)
    jeux = []
    for s, df in m15["barres"].items():
        df = df.sort_values("time").reset_index(drop=True)
        jeux += [(s, "M15", df), (s, "H1", E.agrege(df, "1h")), (s, "H4", E.agrege(df, "4h"))]
    for s, df in d1["barres"].items():
        jeux.append((s, "D1", df.sort_values("time").reset_index(drop=True)))

    btc_h1 = next(df for s, tf, df in jeux if s == "BTCUSD" and tf == "H1")
    # Le ATTENTE / VALIDITE a 20 bougies et la regle "premiere cloture" rendent la
    # coupure sure des 60 bougies : on verifie sur les 3000 dernieres.
    print(verif_causalite(btc_h1.iloc[-3000:].reset_index(drop=True)), flush=True)

    resultats = {}
    for s, tf, df in jeux:
        r = lance(df, s, tf, d1["infos"][s])
        for k, v in r.items():
            if v is not None:
                resultats[(s, tf, k)] = v
        print(f"  {s:10s} {tf:3s} " + "  ".join(
            f"{k + 1}:{len(v):5d}" for k, v in r.items() if v is not None) + " trades", flush=True)

    print("\n" + "=" * 100)
    print("VERDICT PRINCIPAL : BTC ET OR, chaque strategie, chaque unite de temps "
          "(mise 1 % = 10 $ par trade)")
    print("=" * 100)
    for s in E.PRINCIPAUX:
        for tf in ("M15", "H1", "H4", "D1"):
            print(f"\n{s} {tf}")
            for k, nom in enumerate(STRATEGIES):
                if (s, tf, k) in resultats:
                    print(E.ligne(nom, E.bilan(resultats[(s, tf, k)])))

    print("\n" + "=" * 100)
    print("CHAQUE STRATEGIE, BTC ET OR REUNIS (toutes unites de temps)")
    print("=" * 100)
    for k, nom in enumerate(STRATEGIES):
        tr = sorted(sum((v for (s, tf, kk), v in resultats.items()
                         if kk == k and s in E.PRINCIPAUX), []))
        print(E.ligne(nom, E.bilan(tr)))

    print("\n" + "=" * 100)
    print("CONTROLE : TOUS LES MARCHES (7 en M15/H1/H4, 21 en D1)")
    print("=" * 100)
    for k, nom in enumerate(STRATEGIES):
        tr = sorted(sum((v for (s, tf, kk), v in resultats.items() if kk == k), []))
        print(E.ligne(nom, E.bilan(tr)))
    combos = [(key, E.bilan(v)) for key, v in resultats.items() if len(v) >= 30]
    pos = [z for z in combos if z[1]["total"] > 0]
    sig = [z for z in combos if z[1]["t"] > 2]
    print(f"\n  combinaisons marche x unite de temps x strategie avec au moins 30 trades : "
          f"{len(combos)}")
    print(f"  rentables : {len(pos)}  |  gain significatif (t > 2) : {len(sig)} "
          f"(le hasard seul en donnerait ~{0.023 * len(combos):.0f})")
    for (s, tf, k), b in sorted(sig, key=lambda z: -z[1]["t"])[:15]:
        print(f"    {s:10s} {tf:3s} {STRATEGIES[k]:32s} {b['n']:4d} trades  PF {b['pf']:4.2f}  "
              f"{b['moy']:+.3f} R  t {b['t']:+.1f}  total {b['total']:+.2f} $")

    print("\n" + "=" * 100)
    print("SANS AUCUN FRAIS (spread, glissement et swap a zero) : BTC et or, toutes unites de temps")
    print("=" * 100)
    brut = {}
    for s, tf, df in jeux:
        if s in E.PRINCIPAUX:
            for k, v in lance(df, s, tf, d1["infos"][s], sans_frais=True).items():
                if v is not None:
                    brut.setdefault(k, []).extend(v)
    for k, nom in enumerate(STRATEGIES):
        print(E.ligne(nom, E.bilan(sorted(brut.get(k, [])))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
