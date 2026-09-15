"""Le basis perpetuel/spot apporte-t-il quelque chose ? Protocole purge.

ALIGNEMENT HORAIRE — controle BLOQUANT. Les archives Binance sont en UTC, le
cache MT5 en heure du courtier (DST americain). On reutilise la table
d'offsets deja detectee, puis on VERIFIE : le basis doit correler avec le
taker_ratio deja en service, les deux mesurant la pression acheteuse sur le
meme marche au meme instant. Une correlation nulle signalerait un decalage,
pas une feature inutile — le script s'arrete alors plutot que de produire un
chiffre faux.

    python mesure_basis.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from saint_core import FEATURE_COLS, ATR_PLANCHER_FRAC
from build_binance_features import vers_heure_broker
from mesure_features import (barrieres, MAX_HOLD, SPREAD_BPS, SLIP_ENTREE_BPS,
                             SLIP_SORTIE_BPS, CACHE)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASIS = "binance_basis_BTCUSD.pkl"
OFFSETS = ".cache_binance/offsets.pkl"
SL, RR, N_BLOCS = 2.0, 2.0, 8
COLS = ["basis", "basis_ma5", "basis_chg", "basis_z"]


def main() -> int:
    bs = pd.read_pickle(BASIS)
    tab = pd.read_pickle(OFFSETS)
    bs.index = vers_heure_broker(pd.Series(bs.index), tab).values
    bs = bs[~bs.index.duplicated(keep="last")].sort_index()

    df = pd.read_pickle(CACHE).join(bs[COLS], on="time")
    couv = {c: df[c].notna().mean() for c in COLS}
    print("couverture : " + "  ".join(f"{c} {100*v:.1f}%" for c, v in couv.items()))

    # CONTROLE D'ALIGNEMENT : sur les RENDEMENTS, jamais sur les niveaux.
    #
    # Une premiere version comparait taker_ratio a basis_ma5 et refusait de
    # mesurer a +0.0132. C'etait la mauvaise grandeur de controle : l'un est un
    # FLUX (part du volume prise a l'achat), l'autre un ECART DE PRIX. Rien ne
    # les oblige a correler, meme parfaitement alignes — le garde-fou criait au
    # decalage la ou il n'y en avait pas.
    #
    # Les rendements M1 des deux sources, eux, DOIVENT coller : c'est le meme
    # actif. Mesure : 0.9825 avec la table DST, contre 0.0036 a +0 h, 0.4078 a
    # +2 h et 0.5783 a +3 h. Un offset constant ne suffit pas, le courtier
    # suivant le calendrier DST americain.
    pr = pd.read_pickle(".cache_binance/close1m_perp.pkl")
    pr = pr.set_index("time_utc")["close"].astype(float)
    pr.index = vers_heure_broker(pd.Series(pr.index), tab).values
    pr = pr[~pr.index.duplicated(keep="last")].sort_index().pct_change()
    j = pd.concat([df.set_index("time")["close"].pct_change().rename("a"),
                   pr.rename("b")], axis=1).dropna()
    j = j[(j.a.abs() < 0.05) & (j.b.abs() < 0.05)]
    corr = float(np.corrcoef(j.a, j.b)[0, 1])
    print(f"controle d'alignement : corr(rendements MT5, rendements perp) = "
          f"{corr:+.4f}  sur {len(j):,} minutes")
    if corr < 0.90:
        print("ABANDON : sous 0.90 les deux series ne decrivent pas le meme "
              "instant. Mesurer ici ne dirait rien.")
        return 1
    print("  les deux sources decrivent bien le meme instant.")

    df = df.dropna(subset=FEATURE_COLS + COLS + ["atr_14"]).reset_index(drop=True)
    n = len(df)
    fin = int(n * 0.70)
    hi, lo, cl = (df[c].to_numpy(np.float64) for c in ("high", "low", "close"))
    atr = np.maximum(df["atr_14"].to_numpy(np.float64), ATR_PLANCHER_FRAC * cl)
    ds = (SPREAD_BPS / 1e4) * cl / 2.0
    se = (SLIP_ENTREE_BPS / 1e4) * cl
    ss = (SLIP_SORTIE_BPS / 1e4) * cl
    toutes = FEATURE_COLS + COLS
    idx = {c: k for k, c in enumerate(toutes)}
    X = np.nan_to_num(np.column_stack(
        [df[c].to_numpy(np.float32) for c in toutes]))
    print(f"{n:,} bougies  |  barrieres SL {SL}xATR R:R {RR}\n")

    bornes = np.linspace(int(fin * 0.30), fin, N_BLOCS + 1).astype(int)
    plan = []
    for b in range(N_BLOCS):
        a, bv = bornes[b], bornes[b + 1]
        it = np.arange(0, a - 2 * MAX_HOLD, MAX_HOLD)
        iv = np.arange(a, bv - MAX_HOLD, MAX_HOLD)
        it = it[np.isfinite(atr[it]) & (atr[it] > 0)]
        iv = iv[np.isfinite(atr[iv]) & (atr[iv] > 0)]
        if len(it) < 500 or len(iv) < 40:
            continue
        for s_ in (1, -1):
            plan.append((b, it, iv,
                         barrieres(hi, lo, cl, atr, it, SL, RR, s_, ds, se, ss)[0],
                         barrieres(hi, lo, cl, atr, iv, SL, RR, s_, ds, se, ss)[0]))

    def ev(cols):
        k = [idx[x] for x in cols]
        ps, rs, ys, pb = [], [], [], {}
        for b, it, iv, rt, rv in plan:
            xt, xv = X[it][:, k], X[iv][:, k]
            m, s = xt.mean(0), xt.std(0) + 1e-8
            lr = LogisticRegression(max_iter=1000, n_jobs=1).fit(
                np.clip((xt - m) / s, -5, 5), rt > 0)
            p = lr.predict_proba(np.clip((xv - m) / s, -5, 5))[:, 1]
            ps.append(p); rs.append(rv); ys.append(rv > 0)
            pb.setdefault(b, [[], []])
            pb[b][0].append(p); pb[b][1].append(rv)
        return np.concatenate(ps), np.concatenate(rs), np.concatenate(ys), pb

    base = list(FEATURE_COLS)
    jeux = {"30 actuel": base}
    for c in COLS:
        jeux["  + " + c] = base + [c]
    jeux["  + les 4"] = base + COLS
    jeux["les 4 SEULES"] = COLS

    print(f"{'jeu':<20} {'AUC':>7} | "
          + " ".join(f"{'sel ' + str(int(100*q)) + '%':>11}" for q in (0.01, 0.02, 0.05))
          + f" {'blocs+':>8}")
    print("-" * 76)
    for nom, cols in jeux.items():
        p, r, y, pb = ev(cols)
        npos = 0
        for b in pb:
            pbb = np.concatenate(pb[b][0]); rbb = np.concatenate(pb[b][1])
            npos += rbb[pbb >= np.quantile(pbb, 0.95)].mean() > 0
        print(f"{nom:<20} {roc_auc_score(y, p):7.4f} | "
              + " ".join(f"{r[p >= np.quantile(p, 1-q)].mean():>+11.4f}"
                         for q in (0.01, 0.02, 0.05))
              + f" {npos:>6}/8")
    print("\nEcarts sous ~0.002 d'AUC : non separables du bruit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
