"""TEST UNIQUE sur fenetre vierge. A ne lancer qu'une fois, sans rien reajuster.

Tout a ete choisi en regardant la fenetre de VALIDATION : features, SL/TP,
selectivite, filtre de volatilite, methode d'entrainement. Des dizaines de
regards successifs sur la meme fenetre, en retenant a chaque fois ce qui y
marchait le mieux — c'est du surapprentissage par selection, et les marges
annoncees en cours de route valent donc moins que leur valeur faciale.

La seule mesure qui vaille est un passage UNIQUE sur des donnees qui n'ont servi
a aucune decision. Si on reajuste apres avoir vu ce resultat, il perd lui aussi
toute valeur.

CONFIGURATION GELEE :
  modele        pretrain_saintv2.pth   (AUC val 0.5407)
  SL / TP       3 x ATR / 4.2 x ATR    (R:R 1.4, 99.4 % de resolution)
  selectivite   top 1 % par cote, seuils calibres SUR LA VALIDATION
  detention max 240 min, non-resolus CLOTURES AU MARCHE
  execution     sequentielle, une position a la fois, spread paye des deux cotes

    python test_fenetre_vierge.py
"""

import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import MetaTrader5 as mt5

from saint_core import (FEATURE_COLS, N_POS_FEATURES, MASK_VALUE, merge_m1_h1,
                        charge_source_externe, SOURCE_EXT_NOM, load_norm_stats, safe_normalize,
                        build_policy, get_device, build_mask_from_pos_scalar)

CHK = "pretrain_saintv2.pth"
LOOKBACK = 25
SL_MULT, RR = 5.0, 2.0
MAX_HOLD = 240
SELECT = 0.01                  # top 1 % par cote
BATCH = 512                    # 4096 saturait la memoire GPU (~8 Go) et rendait
                               # chaque lot 10x plus lent qu'il n'aurait du
PAS_CALIB = 5                  # calibration : un quantile n'a pas besoin de
                               # chaque bougie
PAS_ENTREE = 3                 # candidats d'entree : un trade dure ~60 min,
                               # entrer a 3 min pres ne change rien et c'est
                               # conservateur (moins d'occasions)
VOLUME = 0.01                  # lot, contract_size = 1.0 -> 1 $ de move = 0.01 $
DEB = datetime(2018, 9, 15)


class _Cfg:
    force_cpu = False


def convictions(pol, dev, X, vues, bornes, mask, pas=1):
    """p_BUY / p_SELL en etat FLAT, une bougie sur `pas` de [a, b)."""
    a, b = bornes
    idx = np.arange(max(a, LOOKBACK), b, pas)
    pb, ps = [], []
    with torch.no_grad():
        for d in range(0, len(idx), BATCH):
            lot = idx[d:d + BATCH]
            f = np.transpose(vues[lot - LOOKBACK], (0, 2, 1))
            pos = np.zeros((len(lot), LOOKBACK, N_POS_FEATURES), np.float32)
            pos[:, :, 3] = 1.0
            obs = np.concatenate([f, pos], axis=2).astype(np.float32)
            lg, _ = pol(torch.as_tensor(obs, device=dev))
            p = torch.softmax(
                lg.masked_fill(~mask.unsqueeze(0), MASK_VALUE), -1).cpu().numpy()
            pb.append(p[:, 0])
            ps.append(p[:, 1])
    return idx, np.concatenate(pb), np.concatenate(ps)


def simule(idx, pb, ps, thr, hi, lo, cl, atr, spread, n):
    """Backtest SEQUENTIEL : une position a la fois, pas de chevauchement."""
    trades = []
    i = 0
    while i < len(idx):
        bar = idx[i]
        ok_b = pb[i] >= thr[0]
        ok_s = ps[i] >= thr[1]
        if not (ok_b or ok_s):
            i += 1
            continue
        if ok_b and ok_s:
            sens = 1 if (pb[i] - thr[0]) >= (ps[i] - thr[1]) else -1
        else:
            sens = 1 if ok_b else -1

        a = atr[bar]
        if not np.isfinite(a) or a <= 0:
            i += 1
            continue
        entree = cl[bar] + sens * spread / 2.0
        tp = entree + sens * RR * SL_MULT * a
        sl = entree - sens * SL_MULT * a

        sortie, motif, h_fin = None, None, MAX_HOLD
        for h in range(1, MAX_HOLD + 1):
            j = bar + h
            if j >= n:
                break
            touche_tp = (hi[j] >= tp) if sens > 0 else (lo[j] <= tp)
            touche_sl = (lo[j] <= sl) if sens > 0 else (hi[j] >= sl)
            # Une bougie touchant les deux est comptee PERDANTE : on ignore
            # l'ordre intra-minute, autant que ce soit dans le sens defavorable.
            if touche_sl:
                sortie, motif, h_fin = sl, "SL", h
                break
            if touche_tp:
                sortie, motif, h_fin = tp, "TP", h
                break
        if sortie is None:
            j = min(bar + MAX_HOLD, n - 1)
            sortie, motif, h_fin = cl[j], "temps", MAX_HOLD

        pnl_prix = sens * (sortie - entree) - spread / 2.0
        trades.append({
            "bar": bar, "sens": sens, "motif": motif, "duree": h_fin,
            "pnl_prix": pnl_prix, "pnl_atr": pnl_prix / a,
            "pnl_dollar": pnl_prix * VOLUME,
        })
        # On saute a la sortie : impossible de rouvrir avant d'avoir ferme.
        suiv = np.searchsorted(idx, bar + h_fin + 1)
        i = max(suiv, i + 1)
    return pd.DataFrame(trades)


def resume(nom, tr, requis):
    if tr.empty:
        print(f"\n--- {nom} : aucun trade ---")
        return
    w = tr[tr["pnl_prix"] > 0]
    p = tr[tr["pnl_prix"] <= 0]
    eq = tr["pnl_dollar"].cumsum()
    dd = float((eq.cummax() - eq).max())
    pf = (w["pnl_dollar"].sum() / abs(p["pnl_dollar"].sum())
          if len(p) and p["pnl_dollar"].sum() != 0 else float("inf"))
    wr = 100 * len(w) / len(tr)
    print(f"\n--- {nom} ---")
    print(f"  trades            {len(tr):,}   "
          f"({100*(tr['motif']=='TP').mean():.0f}% TP, "
          f"{100*(tr['motif']=='SL').mean():.0f}% SL, "
          f"{100*(tr['motif']=='temps').mean():.0f}% temps)")
    print(f"  winrate           {wr:.2f} %   (requis {100*requis:.2f} %)   "
          f"ecart {wr - 100*requis:+.2f} pts")
    print(f"  PnL total         {tr['pnl_dollar'].sum():+.2f} $ "
          f"(lot {VOLUME}, capital de reference 1000 $)")
    print(f"  PnL / trade       {tr['pnl_atr'].mean():+.4f} ATR   "
          f"({tr['pnl_dollar'].mean():+.4f} $)")
    print(f"  profit factor     {pf:.3f}")
    print(f"  drawdown max      {dd:.2f} $")
    print(f"  duree mediane     {tr['duree'].median():.0f} min")
    print(f"  long / short      {int((tr['sens']==1).sum())} / "
          f"{int((tr['sens']==-1).sum())}")
    # L'ecart-type dit si le resultat est distinguable de zero.
    se = tr["pnl_atr"].std() / np.sqrt(len(tr))
    t = tr["pnl_atr"].mean() / (se + 1e-12)
    print(f"  PnL/trade en sigma {t:+.2f}   "
          f"({'significatif' if abs(t) > 2 else 'DANS LE BRUIT'})")


def main() -> int:
    if not mt5.initialize():
        print(f"MT5 KO {mt5.last_error()}")
        return 1
    fin = datetime.now()
    m1 = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M1, DEB, fin)
    h1 = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_H1,
                              DEB - timedelta(days=20), fin)
    feats_ext = charge_source_externe(DEB - timedelta(days=1), fin)
    info = mt5.symbol_info("XAUUSD")
    spread = float(pd.DataFrame(m1)["spread"].mean() * info.point)
    mt5.shutdown()

    point = float(info.point)
    df = merge_m1_h1(m1, h1, feats_ext=feats_ext, point=point,
                     dropna_subset=FEATURE_COLS + ["atr_14"])
    n = len(df)
    X = safe_normalize(df[FEATURE_COLS].to_numpy(np.float32),
                       load_norm_stats(), clip_sigma=5.0).astype(np.float32)
    atr = df["atr_14"].to_numpy(np.float64)
    hi, lo, cl = (df[c].to_numpy(np.float64) for c in ("high", "low", "close"))
    vues = np.lib.stride_tricks.sliding_window_view(X, LOOKBACK, axis=0)

    win = int(n * 0.8)
    val = (int(win * 0.55), int(win * 0.70))
    fenA = (int(win * 0.70), int(win * 0.80))     # test du fold 1 : vierge
    fenB = (int(win * 0.80), n)                   # tout ce qui suit

    requis = (SL_MULT + spread / float(np.median(atr))) / (SL_MULT + RR * SL_MULT)
    print(f"{n:,} bougies | spread {spread:.2f}$ | ATR median "
          f"{np.median(atr):.2f}$ | WR d'equilibre {100*requis:.2f} %")
    for nom, (a, b) in (("validation (calibration)", val),
                        ("A  test fold 1 (VIERGE)", fenA),
                        ("B  apres fold 1", fenB)):
        print(f"  {nom:<26} [{a:>9,} : {b:>9,}]  "
              f"{df['time'].iloc[a]:%Y-%m-%d} -> {df['time'].iloc[b-1]:%Y-%m-%d}")

    dev = get_device(_Cfg())
    pol = build_policy(dev, lookback=LOOKBACK)
    pol.load_state_dict(torch.load(CHK, map_location=dev, weights_only=True))
    pol.eval()
    mask = build_mask_from_pos_scalar(0, dev, "both")
    print(f"\nmodele {CHK} sur {dev}")

    # --- seuils : calibres SUR LA VALIDATION, jamais sur le test ---
    _, pb_v, ps_v = convictions(pol, dev, X, vues, val, mask, PAS_CALIB)
    thr = (float(np.quantile(pb_v, 1.0 - SELECT)),
           float(np.quantile(ps_v, 1.0 - SELECT)))
    print(f"seuils calibres sur la validation : BUY {thr[0]:.4f}  "
          f"SELL {thr[1]:.4f}  (top {100*SELECT:g} % par cote)")

    for nom, bornes in (("A  TEST FOLD 1 (fenetre vierge)", fenA),
                        ("B  APRES FOLD 1 (contaminee pour le choix SL/TP)", fenB)):
        t0 = time.time()
        idx, pb, ps = convictions(pol, dev, X, vues, bornes, mask, PAS_ENTREE)
        tr = simule(idx, pb, ps, thr, hi, lo, cl, atr, spread, n)
        resume(nom, tr, requis)
        print(f"  (calcule en {time.time() - t0:.0f}s)")
        sys.stdout.flush()

    print()
    print("=" * 70)
    print("Ce resultat est DEFINITIF. Le reajuster apres l'avoir vu lui")
    print("retirerait toute valeur : la fenetre cesserait d'etre vierge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
