# -*- coding: utf-8 -*-
"""Le suivi de tendance diversifie, sur bougies journalieres MT5 — etude honnete.

2026-09-27. Le scalping BTC M1 a abouti a un signal egal au cout (voir
`test_expert_seul.txt`). La strategie directionnelle la mieux documentee
chez les professionnels est le SUIVI DE TENDANCE DIVERSIFIE (fonds CTA ;
Moskowitz, Ooi et Pedersen 2012) : horizon de semaines a mois, des dizaines
de marches, taille ajustee a la volatilite.

CE QUI EST FIXE AVANT DE REGARDER, et rien n'est ajuste ensuite :

  REGLES — les trois versions classiques de la litterature, toutes
  rapportees, aucune choisie :
    A  moyennes mobiles exponentielles 50 / 200 : long si la courte est
       au-dessus, short sinon
    B  cassure de Donchian (tortues, systeme 2) : on entre sur un plus
       haut / plus bas de 55 jours, on sort sur la cassure opposee de 20
    C  momentum multi-horizon : moyenne des signes des rendements a 20,
       60, 120 et 250 jours

  TAILLE — chaque marche vise la meme volatilite : 10 % par an divises par
  la racine du nombre de marches actifs (volatilite EWMA 60 jours),
  plafonnee a 2 fois le capital par marche.

  EXECUTION — la position decidee a la cloture du jour t porte le
  rendement de t a t+1. Cout : chaque changement de position paie un
  demi-spread plus 0.5 bps. Un spread absent de l'historique (anciennes
  barres a zero) est remplace par la mediane des barres qui en ont un.

  SWAP — le financement de nuit, avec les taux ACTUELS du compte :
  l'historique des swaps n'existe pas, et les taux ont beaucoup change
  depuis 2020. C'est une approximation, et les resultats sont donnes avec
  ET sans swap pour que son poids se voie.

AUCUN PARAMETRE N'EST AJUSTE SUR CES DONNEES : les regles sortent de la
litterature. Tout l'historique est donc hors echantillon pour elles ; les
resultats sont donnes par annee et par moitie de periode.

    python etude_tendance.py
"""
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CACHE = "cache_d1_mt5.pkl"
VOL_CIBLE = 0.10
LEVIER_MAX = 2.0
GLISSEMENT_BPS = 0.5
SECTEURS = {
    "crypto": ["BTCUSD", "ETHUSD"],
    "metaux": ["XAUUSD", "XAGUSD", "COPPER-C"],
    "energie": ["CL-OIL"],
    "indices": ["SP500", "NAS100", "DJ30", "GER40", "UK100", "Nikkei225"],
    "devises": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD",
                "USDCHF", "AUDJPY", "USDX"],
}


def swap_bps_jour(info, prix, cotes_usd):
    """(long, short) en bps du notionnel par jour calendaire ; negatif = cout."""
    m = info["swap_mode"]
    sl, ss = info["swap_long"], info["swap_short"]
    if m.endswith("DISABLED"):
        return 0.0, 0.0
    if m.endswith("POINTS"):
        return (sl * info["point"] / prix * 1e4, ss * info["point"] / prix * 1e4)
    if "INTEREST" in m:
        return sl / 365.0 * 100.0, ss / 365.0 * 100.0
    if "CURRENCY" in m:
        # Argent par lot et par jour, rapporte au notionnel d'un lot.
        notionnel = info["contrat"] * prix
        return sl / notionnel * 1e4, ss / notionnel * 1e4
    return 0.0, 0.0


def signaux(c, h, l, regle):
    n = len(c)
    if regle == "A":
        e50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
        e200 = pd.Series(c).ewm(span=200, adjust=False).mean().to_numpy()
        s = np.where(e50 > e200, 1.0, -1.0)
        s[:200] = 0.0
        return s
    if regle == "B":
        hh55 = pd.Series(h).rolling(55).max().shift(1).to_numpy()
        ll55 = pd.Series(l).rolling(55).min().shift(1).to_numpy()
        hh20 = pd.Series(h).rolling(20).max().shift(1).to_numpy()
        ll20 = pd.Series(l).rolling(20).min().shift(1).to_numpy()
        s = np.zeros(n)
        pos = 0.0
        for t in range(n):
            if np.isnan(hh55[t]):
                s[t] = 0.0
                continue
            if pos > 0 and c[t] < ll20[t]:
                pos = 0.0
            elif pos < 0 and c[t] > hh20[t]:
                pos = 0.0
            if c[t] > hh55[t]:
                pos = 1.0
            elif c[t] < ll55[t]:
                pos = -1.0
            s[t] = pos
        return s
    if regle == "C":
        r = pd.Series(c)
        parts = [np.sign((r / r.shift(k) - 1.0).to_numpy()) for k in (20, 60, 120, 250)]
        s = np.nanmean(np.vstack(parts), axis=0)
        s[:250] = 0.0
        return np.nan_to_num(s)
    raise ValueError(regle)


def prepare(barres, infos):
    """Rendements, spreads et swaps alignes sur un calendrier commun."""
    dates = sorted(set().union(*[set(d["time"].dt.normalize()) for d in barres.values()]))
    cal = pd.DatetimeIndex(dates)
    out = {}
    for s, d in barres.items():
        d = d.copy()
        d["time"] = d["time"].dt.normalize()
        d = d.drop_duplicates("time").set_index("time").sort_index()
        c, h, l = (d[k].to_numpy(np.float64) for k in ("close", "high", "low"))
        sp = d["spread"].to_numpy(np.float64) * infos[s]["point"] / c * 1e4
        med = np.median(sp[sp > 0]) if (sp > 0).any() else 2.0
        sp = np.where(sp > 0, sp, med)
        sw_l, sw_s = swap_bps_jour(infos[s], float(c[-1]), None)
        out[s] = {"d": d, "c": c, "h": h, "l": l, "sp": sp,
                  "swap": (sw_l, sw_s), "spread_med": med}
    return cal, out


def backtest(cal, data, regle, avec_swap=True):
    """Rend (serie des rendements journaliers du portefeuille, contributions).

    LE CALENDRIER MELANGE DES MARCHES QUI NE COTENT PAS LES MEMES JOURS : la
    crypto cote le week-end, les devises et les indices non. La position
    d'un marche est donc PROLONGEE sur les jours ou il ne cote pas — sans
    cela, elle retomberait a zero chaque samedi et paierait un aller-retour
    fictif chaque lundi. Rendement, cout et swap ne se comptent que les
    jours ou le marche a une barre ; le swap du lundi couvre le week-end.

    Le nombre de marches qui partagent le risque compte ceux qui ont passe
    leur amorce (250 barres), qu'ils cotent ce jour-la ou non.
    """
    fr = {}
    for s, x in data.items():
        d = x["d"]
        c = x["c"]
        sig = signaux(c, x["h"], x["l"], regle)
        r = np.zeros(len(c))
        r[1:] = c[1:] / c[:-1] - 1.0
        vol = pd.Series(r).ewm(span=60, adjust=False).std().to_numpy() * np.sqrt(252)
        vol = np.where(vol > 1e-6, vol, np.nan)
        jours = np.ones(len(c))
        jours[1:] = np.diff(d.index.values).astype("timedelta64[D]").astype(float)
        pret = np.arange(len(c)) >= 250
        f = pd.DataFrame({"sig": sig, "r": r, "vol": vol, "sp": x["sp"],
                          "jours": jours, "pret": pret}, index=d.index)
        g = f.reindex(cal)
        for k in ("sig", "vol", "sp", "pret"):
            g[k] = g[k].ffill()
        fr[s] = g
    n_act = pd.DataFrame({s: fr[s]["pret"].fillna(False).astype(bool)
                          & fr[s]["vol"].notna() for s in fr}).sum(axis=1).clip(lower=1)
    port = pd.Series(0.0, index=cal)
    contrib = {}
    for s, x in fr.items():
        cote = x["r"].notna()
        w = (x["sig"] * (VOL_CIBLE / np.sqrt(n_act)) / x["vol"]).clip(-LEVIER_MAX, LEVIER_MAX)
        w = w.where(x["pret"].fillna(False).astype(bool), 0.0).fillna(0.0)
        # La position ne change que les jours ou le marche cote.
        w = w.where(cote).ffill().fillna(0.0)
        w_prec = w.shift(1).fillna(0.0)
        rend = w_prec * x["r"].fillna(0.0)
        cout = (w - w_prec).abs() * (x["sp"].fillna(0.0) / 2 + GLISSEMENT_BPS) / 1e4
        if avec_swap:
            sw_l, sw_s = data[s]["swap"]
            swap = (np.where(w_prec > 0, w_prec * sw_l, -w_prec * sw_s) / 1e4
                    * x["jours"].fillna(0.0))
        else:
            swap = 0.0
        net = (rend - cout + swap).where(cote, 0.0)
        contrib[s] = net
        port = port.add(net, fill_value=0.0)
    return port, pd.DataFrame(contrib)


def stats(r: pd.Series):
    r = r.dropna()
    ann = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return ann, vol, ann / vol if vol > 0 else float("nan"), dd


def main() -> int:
    z = pd.read_pickle(CACHE)
    barres, infos = z["barres"], z["infos"]
    cal, data = prepare(barres, infos)
    print(f"{len(data)} marches ; calendrier {cal[0]:%Y-%m-%d} -> {cal[-1]:%Y-%m-%d}\n")
    print("  marche      depuis      spread med   swap long/an   swap short/an")
    for s, x in data.items():
        print(f"  {s:10s} {x['d'].index[0]:%Y-%m-%d}  {x['spread_med']:7.2f} bps  "
              f"{x['swap'][0] * 365 / 100:+8.1f} %      {x['swap'][1] * 365 / 100:+8.1f} %")
    debut = pd.Timestamp("2010-01-01")
    for regle, nom in (("A", "moyennes mobiles 50/200"), ("B", "cassure Donchian 55/20"),
                       ("C", "momentum multi-horizon")):
        print(f"\n=== REGLE {regle} : {nom} (depuis {debut:%Y}) ===")
        for avec_swap in (False, True):
            port, contrib = backtest(cal, data, regle, avec_swap)
            port = port[port.index >= debut]
            ann, vol, sh, dd = stats(port)
            print(f"  {'avec swap' if avec_swap else 'sans swap':10s} rendement {100 * ann:+6.2f} %/an  "
                  f"volatilite {100 * vol:5.2f} %  Sharpe {sh:+.2f}  pire baisse {100 * dd:6.1f} %")
        par_an = port.groupby(port.index.year).apply(lambda x: (1 + x).prod() - 1)
        print("  par annee (avec swap) : " + "  ".join(f"{a}:{100 * v:+.0f}%" for a, v in par_an.items()))
        print(f"  annees positives : {int((par_an > 0).sum())}/{len(par_an)}")
        mi = port.index[len(port) // 2]
        for lab, part in (("1re moitie", port[port.index < mi]), ("2e moitie", port[port.index >= mi])):
            a_, v_, s_, d_ = stats(part)
            print(f"  {lab} ({part.index[0]:%Y}-{part.index[-1]:%Y}) : {100 * a_:+.2f} %/an, Sharpe {s_:+.2f}")
        c_ = contrib[contrib.index >= debut]
        print("  contribution par secteur (%/an) : " + "  ".join(
            f"{sec} {100 * c_[[m for m in ms if m in c_.columns]].sum(axis=1).mean() * 252:+.2f}"
            for sec, ms in SECTEURS.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
