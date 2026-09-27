# -*- coding: utf-8 -*-
"""Le portage du financement (cash-and-carry) sur Binance — etude honnete, frais compris.

2026-09-27. Apres le verdict du jeu BTC M15 (tests +45.95, -11.34, -70.84 $),
le proprietaire demande un jeu sur le PORTAGE DU FINANCEMENT : acheter la
crypto au comptant et vendre la meme quantite en perpetuel. La position ne
depend pas de la direction du prix ; elle encaisse le financement que les
acheteurs de perpetuels paient aux vendeurs toutes les huit heures.

AVANT DE CONSTRUIRE UN JEU, on mesure ce que la structure rapporte, avec
deux regles FIXEES D'AVANCE :

  A  TOUJOURS EN POSITION, du premier au dernier jour
  B  EN POSITION QUAND LE FINANCEMENT RECENT EST POSITIF : on entre si la
     moyenne des 9 derniers financements (3 jours) depasse 0.01 % par
     periode de 8 h (~11 %/an), on sort si elle devient negative
  C  ROTATION : une seule paire a la fois, celle dont cette moyenne est la
     plus haute ; on ne change que pour un ecart de plus de 0.01 % par
     periode (c'est la borne simple de ce qu'un jeu pourrait apprendre)

CE QUI EST COMPTE, periode de 8 h par periode de 8 h :
  financement   recu sur la jambe courte (paye si negatif)
  base          la jambe comptant contre la jambe perpetuelle, aux prix de
                cloture 8 h des deux marches : leur ecart bouge, et se paie
  frais         chaque entree et chaque sortie : 0.1 % au comptant et
                0.05 % sur le perpetuel (tarifs preneur standard), soit
                0.30 % l'aller-retour
  capital       la jambe comptant PLUS une marge egale sur le perpetuel,
                pour survivre a un doublement du prix sans liquidation :
                le rendement est rapporte a DEUX fois la taille

Donnees publiques de data.binance.vision, mises en cache.

    python etude_portage.py
"""
import io
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CACHE = "cache_portage.pkl"
PAIRES = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
DEBUT, FIN = (2020, 1), (2026, 8)
FRAIS_COMPTANT, FRAIS_PERP = 0.0010, 0.0005
SEUIL_ENTREE, FENETRE = 0.0001, 9


def _mois():
    y, m = DEBUT
    while (y, m) <= FIN:
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def _zip(url, noms=None, entete=None):
    for essai in range(3):
        try:
            raw = urllib.request.urlopen(url, timeout=60).read()
            z = zipfile.ZipFile(io.BytesIO(raw))
            d = pd.read_csv(z.open(z.namelist()[0]), header=entete, names=noms,
                            low_memory=False)
            return d
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:
            time.sleep(2 + 2 * essai)
    return None


def _temps(t):
    t = pd.to_numeric(t, errors="coerce")
    micro = t > 1e14
    out = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")
    out[micro] = pd.to_datetime(t[micro], unit="us")
    out[~micro] = pd.to_datetime(t[~micro], unit="ms")
    return out


def telecharge():
    if os.path.exists(CACHE):
        return pd.read_pickle(CACHE)
    base = "https://data.binance.vision/data"
    noms_k = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "qv", "n", "tb", "tq", "i"]
    out = {}
    for p in PAIRES:
        urls_f = [f"{base}/futures/um/monthly/fundingRate/{p}/{p}-fundingRate-{y}-{m:02d}.zip"
                  for y, m in _mois()]
        urls_s = [f"{base}/spot/monthly/klines/{p}/8h/{p}-8h-{y}-{m:02d}.zip" for y, m in _mois()]
        urls_p = [f"{base}/futures/um/monthly/klines/{p}/8h/{p}-8h-{y}-{m:02d}.zip"
                  for y, m in _mois()]
        with ThreadPoolExecutor(8) as ex:
            f = [x for x in ex.map(lambda u: _zip(u, None, 0), urls_f) if x is not None]
            s = [x for x in ex.map(lambda u: _zip(u, noms_k), urls_s) if x is not None]
            q = [x for x in ex.map(lambda u: _zip(u, noms_k), urls_p) if x is not None]
        if not f or not s or not q:
            print(f"{p}: donnees absentes", flush=True)
            continue
        # Les fichiers de financement ont une ligne d'en-tete
        # (calc_time, funding_interval_hours, last_funding_rate), les bougies non.
        fr = pd.concat(f, ignore_index=True)
        fr.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"]
        fr = fr[pd.to_numeric(fr["calc_time"], errors="coerce").notna()].copy()
        fr["time"] = _temps(fr["calc_time"]).dt.round("h")
        fr["funding"] = pd.to_numeric(fr["last_funding_rate"], errors="coerce")
        sp = pd.concat(s, ignore_index=True)
        sp = sp[pd.to_numeric(sp["open_time"], errors="coerce").notna()].copy()
        sp["time"] = _temps(sp["open_time"])
        pp = pd.concat(q, ignore_index=True)
        pp = pp[pd.to_numeric(pp["open_time"], errors="coerce").notna()].copy()
        pp["time"] = _temps(pp["open_time"])
        d = (pd.DataFrame({"time": sp["time"], "spot": pd.to_numeric(sp["close"])})
             .merge(pd.DataFrame({"time": pp["time"], "perp": pd.to_numeric(pp["close"])}),
                    on="time")
             .merge(fr[["time", "funding"]].drop_duplicates("time"), on="time", how="left"))
        d = d.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        d["funding"] = d["funding"].fillna(0.0)
        out[p] = d
        print(f"{p}: {len(d):,} periodes de 8 h  {d['time'].iloc[0]:%Y-%m-%d} -> "
              f"{d['time'].iloc[-1]:%Y-%m-%d}", flush=True)
    pd.to_pickle(out, CACHE)
    return out


def simule(d: pd.DataFrame, regle: str):
    """Rendement de chaque periode, rapporte au capital (deux fois la taille)."""
    f = d["funding"].to_numpy()
    # La jambe comptant gagne ce que le prix au comptant gagne, la jambe
    # perpetuelle courte perd ce que le perpetuel gagne : la base.
    rs = d["spot"].pct_change().fillna(0.0).to_numpy()
    rp = d["perp"].pct_change().fillna(0.0).to_numpy()
    n = len(d)
    if regle == "A":
        pos = np.ones(n)
        pos[-1] = 0.0          # on sort a la fin : la sortie se paie aussi
    else:
        moy = pd.Series(f).rolling(FENETRE, min_periods=FENETRE).mean().to_numpy()
        pos = np.zeros(n)
        for t in range(1, n - 1):
            if pos[t - 1] == 0 and moy[t] > SEUIL_ENTREE:
                pos[t] = 1
            elif pos[t - 1] == 1 and not (moy[t] < 0):
                pos[t] = 1
    return _rendement(pos, f, rs, rp), pos


def _rendement(pos, f, rs, rp):
    """LE TEMPS. La ligne t est la bougie de 8 h qui s'ouvre a T_t ; f[t] est
    le financement verse a T_t, connu a T_t. La position pos[t], decidee a T_t
    avec les financements jusqu'a f[t] compris, porte la base de la bougie t
    (rs[t] - rp[t], de T_t a T_t+1) et touche le financement verse a T_t+1.
    Les frais se paient a chaque changement de position."""
    f_suivant = np.concatenate([f[1:], [0.0]])
    rend = pos * (rs - rp + f_suivant)
    frais = np.abs(np.diff(np.concatenate([[0.0], pos]))) * (FRAIS_COMPTANT + FRAIS_PERP)
    return (rend - frais) / 2.0


def rotation(data):
    """Regle C, fixee d'avance : une seule paire a la fois, celle dont la
    moyenne des 9 derniers financements est la plus haute, si elle depasse
    le seuil d'entree ; on ne change de paire que si la nouvelle depasse
    l'actuelle d'un seuil de plus (un changement coute 0.60 %)."""
    paires = list(data)
    temps = sorted(set.intersection(*(set(data[p]["time"]) for p in paires)))
    idx = pd.DatetimeIndex(temps)
    cols = {}
    for p in paires:
        d = data[p].set_index("time").loc[idx]
        cols[p] = (d["funding"].to_numpy(),
                   d["spot"].pct_change().fillna(0.0).to_numpy(),
                   d["perp"].pct_change().fillna(0.0).to_numpy(),
                   pd.Series(d["funding"].to_numpy()).rolling(FENETRE, min_periods=FENETRE)
                   .mean().to_numpy())
    n = len(idx)
    choix = -np.ones(n, dtype=int)
    for t in range(1, n - 1):
        moy = np.array([cols[p][3][t] for p in paires])
        moy = np.where(np.isnan(moy), -np.inf, moy)
        cur = choix[t - 1]
        best = int(np.argmax(moy))
        if cur < 0:
            choix[t] = best if moy[best] > SEUIL_ENTREE else -1
        elif moy[cur] < 0:
            choix[t] = best if moy[best] > SEUIL_ENTREE else -1
        elif best != cur and moy[best] > moy[cur] + SEUIL_ENTREE:
            choix[t] = best
        else:
            choix[t] = cur
    r = np.zeros(n)
    for k, p in enumerate(paires):
        f, rs, rp, _ = cols[p]
        r += _rendement((choix == k).astype(float), f, rs, rp)
    return pd.Series(r, index=idx), choix, paires


def ligne(s, pos_moy):
    eq = (1 + s).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    an = s.groupby(s.index.year).apply(lambda x: (1 + x).prod() - 1)
    tot = float(eq.iloc[-1] - 1)
    ann = (1 + tot) ** (365.25 * 3 / max(len(s), 1)) - 1
    dern = s[s.index > s.index[-1] - pd.Timedelta(days=365)]
    return (f"{100 * ann:+6.2f} %/an sur le capital  12 derniers mois {100 * ((1 + dern).prod() - 1):+5.2f} %"
            f"  pire baisse {100 * dd:6.2f} %  en position {100 * pos_moy:3.0f} %  |  "
            + "  ".join(f"{a}:{100 * v:+.1f}%" for a, v in an.items()))


def main() -> int:
    data = telecharge()
    print(f"\nfrais : {100 * (FRAIS_COMPTANT + FRAIS_PERP):.2f} % par entree ou sortie ; "
          f"rendement rapporte a deux fois la taille (comptant + marge)\n")
    for p, d in data.items():
        f = d["funding"]
        print(f"=== {p} : financement moyen {100 * f.mean() * 3 * 365:+.1f} %/an sur la taille, "
              f"positif {100 * (f > 0).mean():.0f} % des periodes")
        for regle, nom in (("A", "toujours en position"), ("B", "en position si financement positif")):
            r, pos = simule(d, regle)
            print(f"  {regle} {nom:34s} {ligne(pd.Series(r, index=d['time']), pos.mean())}")
    s, choix, paires = rotation(data)
    print(f"\n=== C ROTATION : la paire au meilleur financement recent, une a la fois "
          f"({s.index[0]:%Y-%m-%d} -> {s.index[-1]:%Y-%m-%d})")
    print(f"  C {'':34s} {ligne(s, (choix >= 0).mean())}")
    chg = int((np.diff(choix) != 0).sum())
    print(f"  changements de position {chg}  |  temps par paire : "
          + "  ".join(f"{p} {100 * (choix == k).mean():.0f}%" for k, p in enumerate(paires)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
