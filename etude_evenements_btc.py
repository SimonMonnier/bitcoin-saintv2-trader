# -*- coding: utf-8 -*-
"""Les moments propres au bitcoin : y a-t-il une heure ou le BTC est previsible ?

2026-09-28, piste 3 de la liste des pistes non explorees avec le jeu BTC. Avant
de toucher au jeu, on mesure si des evenements STRUCTURELS du bitcoin creent un
mouvement previsible, plus grand que nos frais.

PROTOCOLE FIXE AVANT LE PREMIER LANCEMENT
  Donnees : bougies 5 minutes Binance BTCUSDT (UTC), aout 2017 - sept. 2026 ;
  financement 8 h des perpetuels depuis 2020.
  Deux periodes : DECOUVERTE (jusqu'au 31/12/2022) et CONFIRMATION (2023 et
  apres, jamais regardee pour choisir). Le sens de chaque pari est celui de
  la moyenne en decouverte.
  Un effet est VALIDE s'il est significatif en decouverte (|t| >= 3.0 pour la
  famille des 24 heures, >= 2.5 pour les autres hypotheses, qui sont une
  dizaine), puis de meme sens et t >= 2.0 en confirmation, ET si son gain
  moyen en confirmation depasse les frais : 3.5 bps l'aller-retour (spread
  Vantage BTC ~2 bps + glissement), plus le swap acheteur Vantage (-20 %/an,
  ~5.5 bps) pour chaque passage de minuit serveur (17 h New York) en achat.

LES HYPOTHESES
  H  les 24 heures UTC : rendement de chaque heure (famille de 24 tests)
  F1 avant le financement (0 h, 8 h, 16 h UTC) : rendement T-1h -> T, signe
     par le financement PRECEDENT (connu) : s'il est positif, les acheteurs
     paient et soldent avant -> baisse attendue
  F2 apres le financement : T -> T+1h, meme signature, rebond attendu
  O1 options Deribit : jeudi 20 h -> vendredi 8 h UTC (avant l'expiration)
  O2 vendredi 8 h -> 16 h UTC (apres l'expiration)
  O3 O1 les seuls vendredis d'expiration mensuelle (dernier vendredi)
  C  gap CME : entre la fermeture du vendredi (16 h Chicago) et la
     reouverture du dimanche (17 h Chicago), si l'ecart depasse 1 %, pari sur
     le comblement : sortie au retour au prix du vendredi ou apres 72 h
  U1 heure qui precede l'ouverture americaine (8h30-9h30 New York)
  U2 premiere heure de la seance americaine (9h30-10h30 New York)
  U3 derniere heure de la seance americaine (15h-16h New York)
  W1 week-end (samedi 0 h -> lundi 0 h UTC)
  W2 lundi (lundi 0 h -> mardi 0 h UTC)

    python etude_evenements_btc.py
"""
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

K5 = "klines_5m_spot_BTCUSDT.pkl"
PORTAGE = "cache_portage.pkl"
COUPURE = pd.Timestamp("2023-01-01")
COUT_BPS = 3.5
SWAP_NUIT_BPS = 20.0 / 365 * 100          # -20 %/an en achat, par nuit
T_HEURES, T_AUTRES, T_CONF = 3.0, 2.5, 2.0


def charge():
    d = pd.read_pickle(K5)[["time", "open", "close"]].copy()
    d["time"] = pd.to_datetime(d["time"])
    d = d.drop_duplicates("time").sort_values("time").set_index("time")
    return d


def prix_a(d, instants):
    """Prix d'ouverture de la bougie 5 min qui commence a chaque instant (NaN
    si la bougie manque)."""
    idx = pd.DatetimeIndex(instants)
    return d["open"].reindex(idx).to_numpy()


def minuits_serveur(debut, fin):
    """Nombre de minuits serveur MT5 (17 h New York) entre deux instants UTC."""
    ny0 = pd.Timestamp(debut).tz_localize("UTC").tz_convert("America/New_York")
    ny1 = pd.Timestamp(fin).tz_localize("UTC").tz_convert("America/New_York")
    j0 = (ny0 - pd.Timedelta(hours=17)).normalize()
    j1 = (ny1 - pd.Timedelta(hours=17)).normalize()
    return max(0, (j1 - j0).days)


def rendements(d, entrees, sorties):
    p0, p1 = prix_a(d, entrees), prix_a(d, sorties)
    r = (p1 / p0 - 1) * 1e4
    return pd.Series(r, index=pd.DatetimeIndex(entrees))


def stats(r):
    r = r.dropna()
    n = len(r)
    if n < 10:
        return dict(n=n, moy=np.nan, t=np.nan, pos=np.nan)
    return dict(n=n, moy=r.mean(), t=r.mean() / (r.std(ddof=1) / np.sqrt(n)),
                pos=100 * (r > 0).mean())


def juge(nom, r, seuil, nuits_achat=None, signe=None):
    """r : rendement brut en bps de chaque evenement (dans le sens ACHAT,
    ou deja signe). Le sens retenu est celui de la decouverte."""
    dec, conf = r[r.index < COUPURE], r[r.index >= COUPURE]
    sd = stats(dec)
    sens = signe if signe is not None else (1.0 if (sd["moy"] or 0) >= 0 else -1.0)
    sc = stats(conf * sens)
    sd_s = stats(dec * sens)
    net = np.nan
    if sc["n"] >= 10:
        rc = conf * sens - COUT_BPS
        if sens > 0 and nuits_achat is not None:
            rc = rc - nuits_achat.reindex(rc.index).fillna(0) * SWAP_NUIT_BPS
        net = rc.dropna().mean()
    ok = (abs(sd_s["t"]) >= seuil if np.isfinite(sd_s["t"]) else False) and sd_s["t"] > 0 \
        and sc["t"] >= T_CONF and net > 0
    print(f"  {nom:44s} sens {'achat' if sens > 0 else 'vente'} | decouverte n {sd_s['n']:5d} "
          f"{sd_s['moy']:+7.2f} bps t {sd_s['t']:+5.1f} | confirmation n {sc['n']:5d} "
          f"{sc['moy']:+7.2f} bps t {sc['t']:+5.1f} ({sc['pos']:.0f} % >0) | net {net:+7.2f} bps"
          f"  {'VALIDE' if ok else '-'}")
    return ok


def main() -> int:
    d = charge()
    print(f"Binance BTCUSDT 5 min : {len(d):,} bougies, {d.index[0]} -> {d.index[-1]} UTC")
    print(f"decouverte avant {COUPURE:%Y-%m-%d}, confirmation apres ; frais {COUT_BPS} bps "
          f"l'aller-retour, swap acheteur {SWAP_NUIT_BPS:.1f} bps par nuit\n")
    jours = pd.date_range(d.index[0].normalize() + pd.Timedelta(days=1), d.index[-1].normalize(), freq="D")
    valides = []

    print("H  LES 24 HEURES UTC (seuil de decouverte |t| >= 3.0)")
    for h in range(24):
        e = jours + pd.Timedelta(hours=h)
        r = rendements(d, e, e + pd.Timedelta(hours=1))
        if juge(f"H{h:02d} {h:02d}h-{(h + 1) % 24:02d}h UTC", r, T_HEURES):
            valides.append(f"heure {h} UTC")

    print("\nF  FINANCEMENT DES PERPETUELS (0 h, 8 h, 16 h UTC)")
    fu = pd.read_pickle(PORTAGE)["BTCUSDT"][["time", "funding"]].copy()
    fu["time"] = pd.to_datetime(fu["time"])
    fu = fu.set_index("time")["funding"].sort_index()
    prec = fu.shift(1)                        # financement precedent, connu a T - 8 h
    T = prec.dropna().index
    sg = np.sign(prec.reindex(T).to_numpy())
    r1 = rendements(d, T - pd.Timedelta(hours=1), T) * -sg      # vente si precedent > 0
    r2 = rendements(d, T, T + pd.Timedelta(hours=1)) * sg       # achat si precedent > 0
    r1, r2 = r1[sg != 0], r2[sg != 0]
    if juge("F1 T-1h -> T, contre le financement", r1, T_AUTRES, signe=1.0):
        valides.append("avant le financement")
    if juge("F2 T -> T+1h, avec le financement", r2, T_AUTRES, signe=1.0):
        valides.append("apres le financement")

    print("\nO  EXPIRATION DES OPTIONS DERIBIT (vendredi 8 h UTC)")
    ven = jours[jours.weekday == 4]
    o1 = rendements(d, ven - pd.Timedelta(hours=4), ven + pd.Timedelta(hours=8))
    o2 = rendements(d, ven + pd.Timedelta(hours=8), ven + pd.Timedelta(hours=16))
    nuit = pd.Series([minuits_serveur(a, a + pd.Timedelta(hours=12)) for a in o1.index], index=o1.index)
    if juge("O1 jeudi 20h -> vendredi 8h", o1, T_AUTRES, nuit):
        valides.append("avant l'expiration")
    if juge("O2 vendredi 8h -> 16h", o2, T_AUTRES):
        valides.append("apres l'expiration")
    dernier = ven[(ven + pd.Timedelta(days=7)).month != ven.month]
    o3 = o1[o1.index.isin(dernier - pd.Timedelta(hours=4))]
    if juge("O3 idem, expirations mensuelles", o3, T_AUTRES, nuit):
        valides.append("avant l'expiration mensuelle")

    print("\nC  GAP DU WEEK-END DES CONTRATS A TERME CME (ecart >= 1 %)")
    rows = []
    for v in ven:
        f_ch = (v + pd.Timedelta(hours=16)).tz_localize("America/Chicago", ambiguous="NaT",
                                                       nonexistent="NaT")
        o_ch = (v + pd.Timedelta(days=2, hours=17)).tz_localize("America/Chicago", ambiguous="NaT",
                                                                nonexistent="NaT")
        if pd.isna(f_ch) or pd.isna(o_ch):
            continue
        f_utc = f_ch.tz_convert("UTC").tz_localize(None)
        o_utc = o_ch.tz_convert("UTC").tz_localize(None)
        pf, po = prix_a(d, [f_utc])[0], prix_a(d, [o_utc])[0]
        if not (np.isfinite(pf) and np.isfinite(po)):
            continue
        gap = po / pf - 1
        if abs(gap) < 0.01:
            continue
        sens = -np.sign(gap)                                 # pari sur le comblement
        fen = d.loc[o_utc:o_utc + pd.Timedelta(hours=72)]
        if len(fen) < 100:
            continue
        cible = (fen["close"] - pf) * sens >= 0
        sortie = fen.index[int(np.argmax(cible.to_numpy()))] if cible.any() else fen.index[-1]
        px = pf if cible.any() else fen["close"].iloc[-1]
        r = (px / po - 1) * 1e4 * sens
        rows.append((o_utc, r, sens, minuits_serveur(o_utc, sortie)))
    if rows:
        g = pd.DataFrame(rows, columns=["t", "r", "sens", "nuits"]).set_index("t")
        nuits = g["nuits"].where(g["sens"] > 0, 0)
        if juge("C  comblement du gap CME (72 h max)", g["r"], T_AUTRES, nuits, signe=1.0):
            valides.append("gap CME")
        print(f"     {len(g)} gaps de plus de 1 %, comble dans les 72 h : "
              f"{100 * (g['r'] > 0).mean():.0f} % des cas (signe du rendement)")

    print("\nU  SEANCE AMERICAINE (jours de semaine)")
    sem = jours[jours.weekday < 5]

    def ny(jours_utc, h, m):
        loc = (jours_utc + pd.Timedelta(hours=h, minutes=m)).tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="NaT")
        return loc[~loc.isna()].tz_convert("UTC").tz_localize(None)
    for code, (h0, m0, h1, m1) in {"U1 8h30-9h30 New York (avant l'ouverture)": (8, 30, 9, 30),
                                   "U2 9h30-10h30 New York (1re heure)": (9, 30, 10, 30),
                                   "U3 15h-16h New York (derniere heure)": (15, 0, 16, 0)}.items():
        e, s = ny(sem, h0, m0), ny(sem, h1, m1)
        n = min(len(e), len(s))
        if juge(code, rendements(d, e[:n], s[:n]), T_AUTRES):
            valides.append(code[:2])

    print("\nW  WEEK-END ET LUNDI (UTC)")
    sam = jours[jours.weekday == 5]
    lun = jours[jours.weekday == 0]
    w1 = rendements(d, sam, sam + pd.Timedelta(days=2))
    nw = pd.Series([minuits_serveur(a, a + pd.Timedelta(days=2)) for a in w1.index], index=w1.index)
    if juge("W1 samedi 0h -> lundi 0h", w1, T_AUTRES, nw):
        valides.append("week-end")
    w2 = rendements(d, lun, lun + pd.Timedelta(days=1))
    nl = pd.Series([minuits_serveur(a, a + pd.Timedelta(days=1)) for a in w2.index], index=w2.index)
    if juge("W2 lundi 0h -> mardi 0h", w2, T_AUTRES, nl):
        valides.append("lundi")

    print("\n" + "=" * 100)
    print(f"EFFETS VALIDES (decouverte + confirmation + plus grands que les frais) : "
          f"{', '.join(valides) if valides else 'AUCUN'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
