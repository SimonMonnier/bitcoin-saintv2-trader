# -*- coding: utf-8 -*-
"""Les strategies des deux PDF de Patrick Riguet, appliquees A LA LETTRE.

2026-09-27, demande du proprietaire : "toutes les strategies des deux pdf".
Jusqu'ici Ichimoku et les 14 chandeliers n'avaient ete que des COLONNES
donnees au modele (features_ichimoku.py) ; aucune n'avait ete suivie comme
une regle de trading. Ce fichier le fait, sans rien regler.

SOURCES. "Ichimoku Kinko Hyo - La formation totale" (2018) et "Les chandeliers
japonais" (2017). Pages citees : celles du premier sauf mention.

LES SEPT STRATEGIES, FIXEES AVANT LE PREMIER LANCEMENT

  Communes. Reglages 9/26/52 (p.16 : "les bons reglages sont 9, 26, 52").
  Signal a la CLOTURE de la bougie (p.80 : "attendre la cloture des
  chandeliers"), entree a l'ouverture de la suivante. Ratio objectif / stop
  d'au moins 2/1, sinon on n'entre pas (p.76, 82, 89, 91). Mise fixe : 1 %
  de 1000 $ par trade, soit 10 $ par unite de risque (p.91).

  La boussole (p.25-26) : le nuage FUTUR (celui dessine a droite des prix,
  calcule avec les donnees d'aujourd'hui) est oriente a la hausse si son
  milieu a monte de plus d'un ATR en 26 bougies et que la SSA est au-dessus
  de la SSB ; symetrique a la baisse ; sinon c'est un range.

  Chikou libre (p.31-35, filtre des strategies de tendance seulement, car
  "en range, tout ne fait pas obstacle a Chikou") : a l'achat, la cloture du
  jour depasse les plus hauts des 9 bougies autour de sa position dessinee
  (26 en arriere) ET les lignes Tenkan, Kijun, SSA, SSB a cette position.

  Obstacles (p.18-28) : les plats Tenkan, Kijun et SSB (ligne immobile au
  moins 5 bougies) des 200 dernieres bougies, prolonges "en extension",
  plus les plus hauts / plus bas des 52 et 200 dernieres bougies et les
  bornes du range. L'objectif est le premier obstacle a plus d'un quart
  d'ATR ; s'il n'y en a aucun, l'objectif est pose au ratio minimal 2/1.

  1 RANGE BORNES (p.58-63, 91, 93) : en range, rectangle trace sur les
    CORPS des 52 bougies precedentes (p.59). Vente apres SOULEVEMENT (meche
    au-dessus de la borne haute, cloture dans le range), achat apres
    RESSAUT. Stop au-dela de la meche (+0.1 ATR), objectif la borne opposee.
  2 RANGE + CHANDELIER (p.62) : la meme, si la bougie du signal forme une
    structure de retournement du livre des chandeliers.
  3 RANGE + VOLUME (p.62-63) : la meme, si le volume (ticks) depasse 1.5 fois
    sa moyenne des 20 bougies precedentes.
  4 REPLI SUR PLAT KIJUN (p.75-78, 88, 92) : en tendance, Chikou libre, Kijun
    plate ; les prix touchent le plat (a 0.1 ATR) sans le casser EN CLOTURE
    (p.79 : corps entier du bon cote) et le plat a deja tenu au moins deux
    contacts (p.75). Stop sous le plat (0.5 ATR sous le plus bas du plat et
    de la meche), objectif le premier obstacle.
  5 REPLI KIJUN, SORTIE A LA CASSURE (p.79) : meme entree, on laisse courir
    jusqu'a une cloture de l'autre cote de la Kijun, stop initial conserve.
  6 CASSURE DE NOUVEAUX PLUS HAUTS / BAS (p.81-82, 88) : en tendance, Chikou
    libre, cloture au-dela du plus haut des 26 bougies precedentes, prix a
    moins d'un ATR de la Tenkan ("pas trop eloignes"). Stop sous la Kijun et
    les derniers plus bas (9 bougies), objectif le premier obstacle.
  7 CHANDELIERS SUR SUPPORT / RESISTANCE (livre des chandeliers : "le marteau
    n'a d'incidence que sur support") : structure de retournement dont la
    meche touche (a 0.25 ATR) le premier support sous la cloture, cloture
    au-dessus. Stop a l'extremite de la meche (-0.1 ATR, "l'extremite de la
    meche joue un role de support"), objectif le premier obstacle.

  Les structures (definitions du livre des chandeliers, variante "marche
  continu" pour l'avalement, le nuage noir et la penetrante, p.24-35) :
  marteau / pendu (meche basse >= 2 corps, petite meche haute, corps <= 1/3),
  etoile filante / marteau inverse (l'inverse), doji (corps <= 10 % de la
  bougie), haute vague, avalement, nuage noir, penetrante, harami, etoile du
  matin / du soir. Haussiere ou baissiere selon les 3 bougies qui precedent.

CE QUI N'EST PAS TESTE, faute de regle mecanique : les droites obliques et
canaux traces a la main, la navigation sur les unites de temps superieures,
le pyramidage, les chiffres ronds.

COUTS. Achat au prix vendeur (bid + spread REEL de la bougie d'entree),
vente au bid. Glissement 0.5 bps sur chaque ordre au marche et chaque stop,
aucun sur l'objectif. Swap du courtier a chaque nuit tenue. En cas de stop et
d'objectif dans la meme bougie, le stop d'abord.

DONNEES. Bougies MT5 Vantage : M15 (7 marches, 2022-2026) agregees en H1 et
H4, et D1 (21 marches, jusqu'a 2003). Verdict principal : BTC et or.

    python etude_pdf_ichimoku.py
"""
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

M15, D1 = "cache_m15_mt5.pkl", "cache_d1_mt5.pkl"
TENKAN, KIJUN, SSB_N, DEC = 9, 26, 52, 26
ATR_N = 14
PLAT_MIN = 5
FEN_EXT = 200
FEN_RANGE = 52
SEUIL_ORIENT = 1.0
TOL_CONTACT = 0.10
MARGE_MECHE = 0.10
MARGE_KIJUN = 0.50
PROX_TENKAN = 1.0
PROX_SUPPORT = 0.25
ECART_OBST = 0.25
RATIO_MIN = 2.0
VOL_MULT = 1.5
GLISSEMENT_BPS = 0.5
DOLLARS_PAR_R = 10.0
PRINCIPAUX = ("BTCUSD", "XAUUSD")
CRYPTO = ("BTCUSD", "ETHUSD")

STRATEGIES = ["1 range bornes", "2 range + chandelier", "3 range + volume",
              "4 repli plat Kijun", "5 repli Kijun, sortie cassure",
              "6 cassure plus haut/bas", "7 chandeliers sur S/R"]


def _dec(x, k=1):
    x = np.asarray(x, np.float64)
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def _roll(x, n, f):
    return getattr(pd.Series(x).rolling(n, min_periods=n), f)().to_numpy()


def _longueur_immobile(x):
    """Nombre de bougies consecutives, jusqu'a t, ou la ligne n'a pas bouge."""
    same = np.abs(np.diff(x, prepend=np.nan)) <= 1e-9 * np.abs(x)
    out = np.zeros(len(x))
    n = 0
    for i, s in enumerate(same):
        n = n + 1 if s else 0
        out[i] = n
    return out


def indicateurs(df):
    o, h, l, c = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    tr = np.maximum(h - l, np.maximum(np.abs(h - _dec(c)), np.abs(l - _dec(c))))
    atr = _roll(tr, ATR_N, "mean")
    tenkan = (_roll(h, TENKAN, "max") + _roll(l, TENKAN, "min")) / 2
    kijun = (_roll(h, KIJUN, "max") + _roll(l, KIJUN, "min")) / 2
    ssa_now = (tenkan + kijun) / 2
    ssb_now = (_roll(h, SSB_N, "max") + _roll(l, SSB_N, "min")) / 2
    ssa_d, ssb_d = _dec(ssa_now, DEC), _dec(ssb_now, DEC)
    mid_f = (ssa_now + ssb_now) / 2
    orient = (mid_f - _dec(mid_f, KIJUN)) / atr
    regime = np.where((orient > SEUIL_ORIENT) & (ssa_now > ssb_now), 1,
                      np.where((orient < -SEUIL_ORIENT) & (ssa_now < ssb_now), -1, 0))
    # Chikou libre : fenetre de 9 bougies centree sur sa position dessinee.
    hh_ch = _dec(_roll(h, 9, "max"), DEC - 4)
    ll_ch = _dec(_roll(l, 9, "min"), DEC - 4)
    lignes_ch = np.vstack([_dec(tenkan, DEC), _dec(kijun, DEC),
                           _dec(ssa_d, DEC), _dec(ssb_d, DEC)])
    chikou_l = (c > hh_ch) & (c > np.nanmax(lignes_ch, axis=0))
    chikou_s = (c < ll_ch) & (c < np.nanmin(lignes_ch, axis=0))
    imm_k = _longueur_immobile(kijun)
    plats = []
    for ligne in (tenkan, kijun, ssb_now):
        imm = _longueur_immobile(ligne)
        plats.append(np.where(imm >= PLAT_MIN - 1, ligne, np.nan))
    haut_corps, bas_corps = np.maximum(o, c), np.minimum(o, c)
    return dict(o=o, h=h, l=l, c=c, atr=atr, tenkan=tenkan, kijun=kijun,
                regime=regime, chikou_l=chikou_l, chikou_s=chikou_s,
                imm_k=imm_k, plats=plats,
                bh=_dec(_roll(haut_corps, FEN_RANGE, "max")),
                bb=_dec(_roll(bas_corps, FEN_RANGE, "min")),
                hh26=_dec(_roll(h, KIJUN, "max")), ll26=_dec(_roll(l, KIJUN, "min")),
                hh52=_dec(_roll(h, 52, "max")), ll52=_dec(_roll(l, 52, "min")),
                hh200=_dec(_roll(h, 200, "max")), ll200=_dec(_roll(l, 200, "min")),
                ll9=_roll(l, 9, "min"), hh9=_roll(h, 9, "max"),
                vol=df["tick_volume"].to_numpy(np.float64),
                vol_moy=_dec(_roll(df["tick_volume"].to_numpy(np.float64), 20, "mean")))


def structures(x):
    """Structures de retournement du livre des chandeliers, en booleens."""
    o, h, l, c, atr = x["o"], x["h"], x["l"], x["c"], x["atr"]
    corps = np.abs(c - o)
    et = np.maximum(h - l, 1e-12)
    mh = h - np.maximum(o, c)
    mb = np.minimum(o, c) - l
    petit = corps <= et / 3
    o1, c1, o2, c2 = _dec(o), _dec(c), _dec(o, 2), _dec(c, 2)
    corps1, corps2 = np.abs(c1 - o1), np.abs(c2 - o2)
    baisse = _dec(c) < _dec(c, 4)
    hausse = _dec(c) > _dec(c, 4)
    tol = 0.10 * atr
    f_marteau = (mb >= 2 * corps) & (mh <= 0.1 * et) & petit
    f_filante = (mh >= 2 * corps) & (mb <= 0.1 * et) & petit
    indecis = (corps <= 0.1 * et) | (petit & (mh >= corps) & (mb >= corps)
                                     & (np.minimum(mh, mb) >= 0.25 * et))
    dedans = (np.maximum(o, c) <= np.maximum(o1, c1)) & (np.minimum(o, c) >= np.minimum(o1, c1))
    englobe = (np.maximum(o, c) >= np.maximum(o1, c1)) & (np.minimum(o, c) <= np.minimum(o1, c1))
    haussier = ((f_marteau & baisse) | (f_filante & baisse) | (indecis & baisse)
                | ((c > o) & (c1 < o1) & englobe & (corps > corps1))
                | ((c1 < o1) & (c > o) & (o <= c1 + tol) & (c > (o1 + c1) / 2) & (c < o1))
                | ((c1 < o1) & dedans & (corps < corps1) & baisse)
                | ((c2 < o2) & (corps1 <= 0.5 * corps2) & (c > o) & (c > (o2 + c2) / 2)))
    baissier = ((f_marteau & hausse) | (f_filante & hausse) | (indecis & hausse)
                | ((c < o) & (c1 > o1) & englobe & (corps > corps1))
                | ((c1 > o1) & (c < o) & (o >= c1 - tol) & (c < (o1 + c1) / 2) & (c > o1))
                | ((c1 > o1) & dedans & (corps < corps1) & hausse)
                | ((c2 > o2) & (corps1 <= 0.5 * corps2) & (c < o) & (c < (o2 + c2) / 2)))
    return np.nan_to_num(haussier).astype(bool), np.nan_to_num(baissier).astype(bool)


def _candidats(x, t):
    """Tous les niveaux-obstacles connus a t."""
    d = max(0, t - FEN_EXT)
    niv = [p[d:t + 1] for p in x["plats"]]
    niv.append(np.array([x["hh52"][t], x["ll52"][t], x["hh200"][t], x["ll200"][t],
                         x["bh"][t], x["bb"][t]]))
    v = np.concatenate(niv)
    return v[np.isfinite(v)]


def obstacle(x, t, p, sens):
    v = _candidats(x, t)
    a = x["atr"][t]
    if sens > 0:
        v = v[v > p + ECART_OBST * a]
        return v.min() if len(v) else np.nan
    v = v[v < p - ECART_OBST * a]
    return v.max() if len(v) else np.nan


def support(x, t, sens):
    """Premier support sous la cloture (sens +1) ou resistance au-dessus (-1)."""
    v = _candidats(x, t)
    c = x["c"][t]
    if sens > 0:
        v = v[v < c]
        return v.max() if len(v) else np.nan
    v = v[v > c]
    return v.min() if len(v) else np.nan


def _objectif(x, t, sens, stop):
    c = x["c"][t]
    risque = (c - stop) * sens
    if not risque > 0:
        return np.nan
    tp = obstacle(x, t, c, sens)
    if not np.isfinite(tp):
        tp = c + sens * RATIO_MIN * risque
    return tp if (tp - c) * sens >= RATIO_MIN * risque else np.nan


def _contacts_kijun(x, t, sens):
    """Episodes de contact sur le plat Kijun actuel, avant t."""
    k = x["kijun"][t]
    s = int(t - x["imm_k"][t])
    n, avant = 0, False
    for j in range(max(s, 0), t):
        a = x["atr"][j]
        if sens > 0:
            ok = x["l"][j] <= k + TOL_CONTACT * a and min(x["o"][j], x["c"][j]) >= k
        else:
            ok = x["h"][j] >= k - TOL_CONTACT * a and max(x["o"][j], x["c"][j]) <= k
        n += ok and not avant
        avant = ok
    return n


def signaux(x, strat):
    """Liste (t, sens, stop, objectif) ; objectif NaN = sortie a la cassure Kijun."""
    o, h, l, c, atr = x["o"], x["h"], x["l"], x["c"], x["atr"]
    n = len(c)
    ok = np.isfinite(atr) & np.isfinite(x["hh200"]) & (np.arange(n) >= 2 * SSB_N + DEC)
    out = []
    if strat in (0, 1, 2):
        rng = ok & (x["regime"] == 0) & np.isfinite(x["bh"])
        soul = rng & (h > x["bh"]) & (c < x["bh"]) & (c > x["bb"])
        ress = rng & (l < x["bb"]) & (c > x["bb"]) & (c < x["bh"])
        if strat == 1:
            hs, bs = structures(x)
            soul, ress = soul & bs, ress & hs
        if strat == 2:
            fort = x["vol"] > VOL_MULT * x["vol_moy"]
            soul, ress = soul & fort, ress & fort
        for t in np.flatnonzero(soul | ress):
            sens = 1 if ress[t] else -1
            stop = l[t] - MARGE_MECHE * atr[t] if sens > 0 else h[t] + MARGE_MECHE * atr[t]
            tp = x["bh"][t] if sens > 0 else x["bb"][t]
            risque = (c[t] - stop) * sens
            if risque > 0 and (tp - c[t]) * sens >= RATIO_MIN * risque:
                out.append((t, sens, stop, tp))
        return out
    if strat in (3, 4):
        k = x["kijun"]
        plat = x["imm_k"] >= PLAT_MIN - 1
        long_ = (ok & plat & (x["regime"] == 1) & x["chikou_l"]
                 & (l <= k + TOL_CONTACT * atr) & (np.minimum(o, c) >= k))
        short = (ok & plat & (x["regime"] == -1) & x["chikou_s"]
                 & (h >= k - TOL_CONTACT * atr) & (np.maximum(o, c) <= k))
        for t in np.flatnonzero(long_ | short):
            sens = 1 if long_[t] else -1
            if _contacts_kijun(x, t, sens) < 2:
                continue
            stop = (min(l[t], k[t]) - MARGE_KIJUN * atr[t] if sens > 0
                    else max(h[t], k[t]) + MARGE_KIJUN * atr[t])
            tp = _objectif(x, t, sens, stop)
            if np.isfinite(tp):
                out.append((t, sens, stop, tp if strat == 3 else np.nan))
        return out
    if strat == 5:
        t_ok = np.abs(c - x["tenkan"]) <= PROX_TENKAN * atr
        long_ = ok & (x["regime"] == 1) & x["chikou_l"] & (c > x["hh26"]) & t_ok
        short = ok & (x["regime"] == -1) & x["chikou_s"] & (c < x["ll26"]) & t_ok
        for t in np.flatnonzero(long_ | short):
            sens = 1 if long_[t] else -1
            stop = (min(x["kijun"][t], x["ll9"][t]) - MARGE_MECHE * atr[t] if sens > 0
                    else max(x["kijun"][t], x["hh9"][t]) + MARGE_MECHE * atr[t])
            tp = _objectif(x, t, sens, stop)
            if np.isfinite(tp):
                out.append((t, sens, stop, tp))
        return out
    if strat == 6:
        hs, bs = structures(x)
        for t in np.flatnonzero(ok & (hs | bs)):
            for sens, m in ((1, hs), (-1, bs)):
                if not m[t]:
                    continue
                s = support(x, t, sens)
                if not np.isfinite(s):
                    continue
                touche = (l[t] <= s + PROX_SUPPORT * atr[t] if sens > 0
                          else h[t] >= s - PROX_SUPPORT * atr[t])
                if not touche:
                    continue
                stop = l[t] - MARGE_MECHE * atr[t] if sens > 0 else h[t] + MARGE_MECHE * atr[t]
                tp = _objectif(x, t, sens, stop)
                if np.isfinite(tp):
                    out.append((t, sens, stop, tp))
                    break
        return out
    raise ValueError(strat)


def _nuits(t0, t1, crypto, jour3):
    """Nuits facturees entre l'entree et la sortie (heure serveur)."""
    d0, d1 = t0.normalize(), t1.normalize()
    if d1 <= d0:
        return 0.0
    jours = pd.date_range(d0, d1 - timedelta(days=1), freq="D")
    if crypto:
        return float(len(jours))
    wd = jours.weekday
    return float(((wd < 5) * np.where(wd == (jour3 - 1) % 7, 3, 1)).sum())


def simule(df, x, sigs, swap_l, swap_s, crypto, jour3, point):
    """Un trade a la fois. Rend la liste des trades (temps, sens, R net)."""
    o, h, l, c = x["o"], x["h"], x["l"], x["c"]
    sp = df["spread"].to_numpy(np.float64) * point
    temps = df["time"].to_numpy()
    k_ = x["kijun"]
    n = len(c)
    trades, libre = [], 0
    for t, sens, stop, tp in sigs:
        if t < libre or t + 1 >= n:
            continue
        j = t + 1
        g = GLISSEMENT_BPS * 1e-4 * o[j]
        entree = o[j] + sp[j] + g if sens > 0 else o[j] - g
        risque = (entree - stop) * sens
        if not risque > 0:
            continue
        sortie, fin, sortir = None, n - 1, False
        for k in range(j, n):
            if sortir:
                sortie = (o[k] - g) if sens > 0 else (o[k] + sp[k] + g)
                fin = k
                break
            if sens > 0:
                if l[k] <= stop:
                    sortie = (min(o[k], stop) if k > j else stop) - g
                elif np.isfinite(tp) and h[k] >= tp:
                    sortie = max(o[k], tp) if k > j else tp
                elif not np.isfinite(tp) and c[k] < k_[k]:
                    sortir = True
            else:
                ha, la, oa = h[k] + sp[k], l[k] + sp[k], o[k] + sp[k]
                if ha >= stop:
                    sortie = (max(oa, stop) if k > j else stop) + g
                elif np.isfinite(tp) and la <= tp:
                    sortie = min(oa, tp) if k > j else tp
                elif not np.isfinite(tp) and c[k] > k_[k]:
                    sortir = True
            if sortie is not None:
                fin = k
                break
        if sortie is None:
            sortie = c[n - 1] if sens > 0 else c[n - 1] + sp[n - 1]
        r = (sortie - entree) * sens / risque
        nuits = _nuits(pd.Timestamp(temps[j]), pd.Timestamp(temps[fin]), crypto, jour3)
        sw = (swap_l if sens > 0 else swap_s) * 1e-4 * entree * nuits / risque
        trades.append((pd.Timestamp(temps[j]), sens, r + sw))
        libre = fin + 1
    return trades


def swap_bps(info, prix):
    m = info["swap_mode"]
    sl, ss = info["swap_long"], info["swap_short"]
    if m.endswith("DISABLED"):
        return 0.0, 0.0
    if m.endswith("POINTS"):
        return sl * info["point"] / prix * 1e4, ss * info["point"] / prix * 1e4
    if "INTEREST" in m:
        return sl / 365.0 * 100.0, ss / 365.0 * 100.0
    if "CURRENCY" in m:
        notion = info["contrat"] * prix
        return sl / notion * 1e4, ss / notion * 1e4
    return 0.0, 0.0


def agrege(m15, regle):
    d = m15.set_index("time")
    a = d.resample(regle, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "tick_volume": "sum", "spread": "first"}).dropna(subset=["open"])
    return a.reset_index()


def bilan(tr):
    if not tr:
        return dict(n=0)
    r = np.array([x[2] for x in tr])
    s = np.array([x[1] for x in tr])
    d = r * DOLLARS_PAR_R
    eq = np.cumsum(d)
    gains, pertes = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=len(r), gagnants=int((r > 0).sum()), perdants=int((r <= 0).sum()),
                longs=int((s > 0).sum()), shorts=int((s < 0).sum()),
                wr=100 * (r > 0).mean(), pf=gains / pertes if pertes > 0 else np.inf,
                moy=r.mean(), t=r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 else 0.0,
                total=d.sum(), dd=float((eq - np.maximum.accumulate(np.maximum(eq, 0))).min()),
                debut=tr[0][0], fin=tr[-1][0])


def ligne(nom, b):
    if b["n"] == 0:
        return f"  {nom:32s}   aucun trade"
    ok = "OUI" if b["total"] > 0 else "NON"
    return (f"  {nom:32s} {b['n']:5d} trades  {b['gagnants']:4d} G / {b['perdants']:4d} P  "
            f"WR {b['wr']:4.1f} %  L/S {b['longs']}/{b['shorts']}  PF {b['pf']:4.2f}  "
            f"{b['moy']:+.3f} R/trade (t {b['t']:+.1f})  total {b['total']:+8.2f} $  "
            f"DD {b['dd']:8.2f} $  RENTABLE ? {ok}")


def verif_causalite(df, strat_list=(0, 3, 5, 6), essais=25, graine=0):
    """Les signaux a t ne changent pas si l'on coupe les donnees apres t."""
    x = indicateurs(df)
    tous = {s: {t: (se, st, tp) for t, se, st, tp in signaux(x, s)} for s in strat_list}
    rng = np.random.default_rng(graine)
    pts = set()
    for s in strat_list:
        ts = list(tous[s])
        pts |= set(rng.choice(ts, min(len(ts), essais), replace=False)) if ts else set()
    pts |= set(rng.integers(400, len(df) - 1, essais))
    for t in sorted(pts):
        xc = indicateurs(df.iloc[:t + 1].reset_index(drop=True))
        for s in strat_list:
            a = tous[s].get(t)
            b = {u: (se, st, tp) for u, se, st, tp in signaux(xc, s)}.get(t)
            if (a is None) != (b is None) or (a and not np.allclose(
                    np.nan_to_num(a, nan=-1.0), np.nan_to_num(b, nan=-1.0))):
                return f"FUITE : strategie {s + 1}, barre {t} : {a} contre {b}"
    return f"causalite verifiee sur {len(pts)} barres : les signaux ne lisent pas l'avenir"


def main() -> int:
    m15 = pd.read_pickle(M15)
    d1 = pd.read_pickle(D1)
    jeux = []
    for s, df in m15["barres"].items():
        df = df.sort_values("time").reset_index(drop=True)
        jeux += [(s, "M15", df), (s, "H1", agrege(df, "1h")), (s, "H4", agrege(df, "4h"))]
    for s, df in d1["barres"].items():
        jeux.append((s, "D1", df.sort_values("time").reset_index(drop=True)))

    btc_h1 = next(df for s, tf, df in jeux if s == "BTCUSD" and tf == "H1")
    print(verif_causalite(btc_h1.iloc[-6000:].reset_index(drop=True)), flush=True)

    resultats = {}
    for s, tf, df in jeux:
        info = d1["infos"][s]
        prix = float(df["close"].iloc[-1])
        sw_l, sw_s = swap_bps(info, prix)
        x = indicateurs(df)
        for k in range(len(STRATEGIES)):
            sigs = signaux(x, k)
            resultats[(s, tf, k)] = simule(df, x, sigs, sw_l, sw_s, s in CRYPTO,
                                           info.get("swap_rollover3days", 3), info["point"])
        print(f"  {s:10s} {tf:3s} {len(df):7,d} bougies  {df['time'].iloc[0]:%Y-%m-%d} -> "
              f"{df['time'].iloc[-1]:%Y-%m-%d}  spread median "
              f"{(df['spread'] * info['point'] / df['close']).median() * 1e4:.1f} bps  "
              f"swap long/short {sw_l * 365 / 100:+.1f} / {sw_s * 365 / 100:+.1f} %/an", flush=True)

    print("\n" + "=" * 100)
    print("VERDICT PRINCIPAL : BTC ET OR, chaque strategie, chaque unite de temps "
          "(mise 1 % = 10 $ par trade)")
    print("=" * 100)
    for s in PRINCIPAUX:
        for tf in ("M15", "H1", "H4", "D1"):
            print(f"\n{s} {tf}")
            for k, nom in enumerate(STRATEGIES):
                print(ligne(nom, bilan(resultats.get((s, tf, k), []))))

    print("\n" + "=" * 100)
    print("CHAQUE STRATEGIE, BTC ET OR REUNIS (toutes unites de temps)")
    print("=" * 100)
    for k, nom in enumerate(STRATEGIES):
        tr = sorted(sum((resultats.get((s, tf, k), []) for s in PRINCIPAUX
                         for tf in ("M15", "H1", "H4", "D1")), []))
        print(ligne(nom, bilan(tr)))

    print("\n" + "=" * 100)
    print("CONTROLE : TOUS LES MARCHES (7 en M15/H1/H4, 21 en D1)")
    print("=" * 100)
    for k, nom in enumerate(STRATEGIES):
        tr = sorted(sum((v for (s, tf, kk), v in resultats.items() if kk == k), []))
        print(ligne(nom, bilan(tr)))
    combos = [(key, bilan(v)) for key, v in resultats.items() if len(v) >= 30]
    pos = [(key, b) for key, b in combos if b["total"] > 0]
    sig = [(key, b) for key, b in combos if b["t"] > 2]
    print(f"\n  combinaisons marche x unite de temps x strategie avec au moins 30 trades : "
          f"{len(combos)}")
    print(f"  rentables : {len(pos)}  |  gain significatif (t > 2) : {len(sig)} "
          f"(le hasard seul en donnerait ~{0.023 * len(combos):.0f})")
    for (s, tf, k), b in sorted(sig, key=lambda z: -z[1]["t"])[:15]:
        print(f"    {s:10s} {tf:3s} {STRATEGIES[k]:32s} {b['n']:4d} trades  PF {b['pf']:4.2f}  "
              f"{b['moy']:+.3f} R  t {b['t']:+.1f}  total {b['total']:+.2f} $")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
