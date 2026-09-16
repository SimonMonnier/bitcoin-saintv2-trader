"""Combien de trades independants, a quel prix — SANS sortie au temps.

LA QUESTION. Le mur de ce depot est le nombre d'occasions INDEPENDANTES :
2 314 en H1 sur neuf ans, quand il en faudrait ~4 900 pour qu'un avantage de
0.02 R soit significatif. Descendre en unite de temps multiplierait ce nombre.

LE PRIX. La friction est un montant FIXE en dollars ; ce qui change d'une
echelle a l'autre, c'est le denominateur — l'unite de risque, qui vaut
sl_mult x ATR. Les deux grandeurs qu'on veut optimiser, beaucoup de trades et
peu de friction, sont donc commandees par le MEME parametre : la largeur du
stop. Un stop large divise la friction par R mais met plus longtemps a etre
touche, donc rend moins de trades independants.

POURQUOI CETTE VERSION REMPLACE LA PRECEDENTE. La premiere mouture cloturait
au marche apres 30 barres. Or l'environnement d'entrainement N'A PAS de sortie
au temps — `max_holding_bars` vaut 0 et `scalping_max_holding` ne sert qu'a
normaliser une feature d'observation — et le live non plus. Toutes les mesures
a plafond decrivaient donc un trade que personne ne fait.

La difference n'est pas cosmetique : a SL 8xATR, la duree mediane atteignait
exactement le plafond de 30 barres, ce qui voulait dire que plus de la moitie
des trades sortaient au temps et que le R:R affiche etait decoratif. Sans
plafond, la duree devient celle d'une vraie course aux barrieres, et le nombre
d'occasions independantes avec elle.

CE QUI RESTE BORNE, ET POURQUOI. Une course sans aucune limite ne se termine
pas toujours dans l'historique disponible. On garde donc une borne LARGE — 30
jours — et on rapporte la part des courses non resolues plutot que de la
cacher. Une part elevee voudrait dire que la geometrie n'est pas tradable :
immobiliser un capital un mois sur un pari de deux heures n'est pas la meme
strategie.

AUCUN MODELE N'INTERVIENT. On entre a chaque occasion, un coup a l'achat un
coup a la vente, sur les 70 % initiaux seulement.

    python mesure_court_terme.py
"""

import numpy as np
import pandas as pd

from mesure_features import SPREAD_BPS, SLIP_ENTREE_BPS, SLIP_SORTIE_BPS

# LA SOURCE EST LE M5, PLUS LE CACHE M1 — corrige le 2026-09-16.
#
# Cette mesure tournait sur `data_cache_BTCUSD_20221215.pkl`, qui couvre 3.6
# ans, alors que le jeu d'entrainement en couvre 9.05. Le nombre d'occasions
# rendu etait donc systematiquement 2.6 fois trop bas — et comme il se compare
# a un seuil absolu (~4 900 occasions pour qu'un avantage de 0.02 R soit
# lisible), la comparaison n'avait de sens qu'a condition de remettre chaque
# ligne a l'echelle a la main. C'est exactement ce qui a ete oublie une fois :
# la ligne retenue mise a 9 ans, la ligne ecartee laissee a 3.6, et le stop de
# 8xATR abandonne pour 1 739 occasions quand il en vaut 4 516.
#
# On perd la ligne M1, qui n'a plus lieu d'etre : meme a 8xATR elle exige
# +0.114 R d'avantage, davantage que tout ce que ce depot a mesure.
CACHE = "klines_5m_spot_BTCUSDT.pkl"
from saint_core import ATR_PLANCHER_FRAC

ECHELLES = (("M5", None, 5), ("M15", "15min", 15),
            ("M30", "30min", 30), ("H1", "1h", 60))
SL_MULTS = (2.0, 4.0, 8.0)
RR = 2.0
# Borne large, en MINUTES, convertie en barres selon l'echelle : trente jours.
BORNE_MINUTES = 30 * 24 * 60
N_ENTREES = 4000
# Ce que le modele sait produire au-dessus du hasard, mesure sur H1.
AVANTAGE_MESURE = 0.09
# Occasions necessaires pour qu'un avantage de 0.02 R soit significatif.
OCCASIONS_REQUISES = 4900


def prepare(brut, regle):
    if regle is None:
        d = brut.copy()
    else:
        d = (brut.set_index("time")[["open", "high", "low", "close"]]
                 .resample(regle)
                 .agg({"open": "first", "high": "max", "low": "min",
                       "close": "last"})
                 .dropna().reset_index())
    h, l, c = d["high"], d["low"], d["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    d["atr_14"] = tr.rolling(14).mean()
    return d.dropna(subset=["atr_14"]).reset_index(drop=True)


def course_libre(hi, lo, cl, atr, idx, sl_mult, rr, sens, ds, se, ss, borne):
    """Course jusqu'a UNE BARRIERE, sans sortie au temps.

    Rend (rendement en R des courses resolues, durees, part non resolue).
    La recherche se fait par tranche pour chaque entree : sur des bornes de
    plusieurs milliers de barres, une boucle pas a pas sur tout le lot coute
    des milliards d'operations, alors qu'une recherche par entree n'en coute
    que la longueur de sa propre course.
    """
    n = len(cl)
    dist = sl_mult * atr[idx]
    entree = cl[idx] + sens * (ds[idx] + se[idx])
    tp = entree + sens * rr * dist
    sl = entree - sens * dist
    r = np.full(len(idx), np.nan)
    duree = np.full(len(idx), np.nan)
    for k, i in enumerate(idx):
        f = min(i + 1 + borne, n)
        if f <= i + 1:
            continue
        h, l = hi[i + 1:f], lo[i + 1:f]
        if sens > 0:
            a = np.flatnonzero(l <= sl[k])
            b = np.flatnonzero(h >= tp[k])
        else:
            a = np.flatnonzero(h >= sl[k])
            b = np.flatnonzero(l <= tp[k])
        ja = a[0] if len(a) else np.inf
        jb = b[0] if len(b) else np.inf
        if not np.isfinite(ja) and not np.isfinite(jb):
            continue
        # Egalite : les deux barrieres touchees dans la meme bougie comptent
        # comme une PERTE. On ne sait pas dans quel ordre le prix les a
        # visitees, et supposer le contraire fabriquerait du rendement.
        if ja <= jb:
            sortie, contre, j = sl[k], True, ja
        else:
            sortie, contre, j = tp[k], False, jb
        j = int(j)
        sortie = sortie + sens * (-ss[i + 1 + j] if contre else ss[i + 1 + j])
        sortie = sortie - sens * ds[i + 1 + j]
        r[k] = sens * (sortie - entree[k]) / dist[k]
        duree[k] = j + 1
    fini = np.isfinite(r)
    return r[fini], duree[fini], 1.0 - float(fini.mean())


def main() -> int:
    brut = pd.read_pickle(CACHE)[["time", "open", "high", "low", "close"]].copy()
    brut["time"] = pd.to_datetime(brut["time"])
    brut = brut.iloc[:int(len(brut) * 0.70)].reset_index(drop=True)
    minutes = (brut["time"].iloc[-1] - brut["time"].iloc[0]).total_seconds() / 60
    ans = minutes / (365.25 * 24 * 60)
    print(f"{len(brut):,} bougies M5, {ans:.1f} ans (70 % initiaux, "
          f"test intouche)")
    print(f"SORTIE SUR BARRIERE UNIQUEMENT — aucune cloture au temps, comme "
          f"l'environnement et le live")
    print(f"R:R {RR}, borne de securite {BORNE_MINUTES//1440} jours, "
          f"{N_ENTREES} entrees tirees par configuration\n")
    print(f"{'UT':>4} {'SL':>5} {'friction/R':>11} {'E[R] hasard':>12} "
          f"{'duree med':>11} {'non resolu':>11} {'occasions':>10} "
          f"{'requis':>9} {'verdict':>9}")
    print("-" * 94)

    rng = np.random.default_rng(0)
    for nom, regle, min_par_barre in ECHELLES:
        d = prepare(brut, regle)
        if len(d) < 5000:
            continue
        hi = d["high"].to_numpy(np.float64)
        lo = d["low"].to_numpy(np.float64)
        cl = d["close"].to_numpy(np.float64)
        atr = np.maximum(d["atr_14"].to_numpy(np.float64),
                         ATR_PLANCHER_FRAC * cl)
        ds = (SPREAD_BPS / 1e4) * cl / 2.0
        se = (SLIP_ENTREE_BPS / 1e4) * cl
        ss = (SLIP_SORTIE_BPS / 1e4) * cl
        fric = (SPREAD_BPS + SLIP_ENTREE_BPS + SLIP_SORTIE_BPS) / 1e4
        atr_med = float(np.median(atr))
        prix_med = float(np.median(cl))
        borne = max(50, BORNE_MINUTES // min_par_barre)

        idx = np.sort(rng.choice(np.arange(20, len(d) - 2),
                                 size=min(N_ENTREES, len(d) - 30),
                                 replace=False))
        for sl in SL_MULTS:
            rs, dus, nr = [], [], []
            for sens in (1, -1):
                r, du, part = course_libre(hi, lo, cl, atr, idx, sl, RR, sens,
                                           ds, se, ss, borne)
                rs.append(r)
                dus.append(du)
                nr.append(part)
            if min(len(x) for x in rs) < 50:
                continue
            er = float(0.5 * (rs[0].mean() + rs[1].mean()))
            dm = float(np.median(np.concatenate(dus)))
            occ = len(d) / max(dm, 1)
            fr = fric * prix_med / (sl * atr_med)
            requis = -er
            ok = "OUI" if (requis < AVANTAGE_MESURE
                           and occ >= OCCASIONS_REQUISES) else ""
            print(f"{nom:>4} {sl:5.1f} {fr:10.3f}R {er:+12.4f} "
                  f"{dm:9.0f}b {100*np.mean(nr):10.1f}% {occ:10,.0f} "
                  f"{requis:+8.4f}R {ok:>9}")

    print(f"\n'requis' = ce que le modele doit rendre AU-DESSUS du hasard pour")
    print(f"atteindre zero. Il produit {AVANTAGE_MESURE:+.2f} R sur H1.")
    print(f"'occasions' = entrees sans chevauchement ; il en faut "
          f"{OCCASIONS_REQUISES:,} pour qu'un avantage de 0.02 R soit lisible.")
    print("'verdict' marque les lignes qui satisfont les deux conditions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
