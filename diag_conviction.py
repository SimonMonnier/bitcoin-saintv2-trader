"""La conviction du modèle porte-t-elle de l'information ? Mesure DIRECTE.

Court-circuite entièrement le mécanisme de sélection (barres, quantiles,
garde-fous) : on prend le checkpoint, on calcule p_BUY / p_SELL sur chaque
bougie de la fenêtre de validation, et on confronte ces convictions à l'ISSUE
RÉELLE du trade correspondant (barrière triple, spread inclus).

    AUC(p_BUY, « un long aurait gagné ») ≈ 0.50  -> la conviction ne porte rien,
        aucune règle de sélection ne peut y changer quoi que ce soit.
    AUC nettement > 0.50                        -> le signal existe et c'est le
        mécanisme de sélection qui le gaspille.

    python diag_conviction.py [checkpoint.pth]
"""
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import MetaTrader5 as mt5
from sklearn.metrics import roc_auc_score

from saint_core import (FEATURE_COLS, OBS_N_FEATURES, MASK_VALUE, merge_m1_h1,
                        charge_source_externe, SOURCE_EXT_NOM, load_norm_stats, safe_normalize,
                        build_policy, get_device, build_mask_from_pos_scalar)

CHK = sys.argv[1] if len(sys.argv) > 1 else "best_saintv2_loup_duel_wf1_both_wf1.pth"
LOOKBACK = 25
SL_MULT, RR = 5.0, 2.0
MAX_HOLD = 240
DEB = datetime(2018, 9, 15)


class _Cfg:
    force_cpu = False


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

    # MÊME découpage que le walk-forward fold 1 : on ne mesure QUE sur la
    # fenêtre de validation, jamais sur celle qui a servi à entraîner.
    win = int(n * 0.8)
    i_deb, i_fin = int(win * 0.55), int(win * 0.70)
    print(f"{n:,} bougies | validation = [{i_deb:,} : {i_fin:,}] "
          f"({df['time'].iloc[i_deb]:%Y-%m-%d} -> {df['time'].iloc[i_fin]:%Y-%m-%d})")

    X = safe_normalize(df[FEATURE_COLS].to_numpy(np.float32),
                       load_norm_stats(), clip_sigma=5.0)
    atr = df["atr_14"].to_numpy(np.float64)
    hi, lo, cl = (df[c].to_numpy(np.float64) for c in ("high", "low", "close"))

    dev = get_device(_Cfg())
    pol = build_policy(dev, lookback=LOOKBACK)
    pol.load_state_dict(torch.load(CHK, map_location=dev))
    pol.eval()
    mask = build_mask_from_pos_scalar(0, dev, "both")
    print(f"checkpoint : {CHK}")

    # Convictions sur chaque bougie, en état FLAT (features de position à zéro)
    idx = np.arange(i_deb, i_fin - MAX_HOLD - 1, 5)   # 1 point sur 5, suffisant
    pb, ps = [], []
    with torch.no_grad():
        for d in range(0, len(idx), 4096):
            lot = idx[d:d + 4096]
            fen = np.stack([X[i - LOOKBACK:i] for i in lot])
            pos = np.zeros((len(lot), LOOKBACK, 4), np.float32)
            pos[:, :, 3] = 1.0                       # risk_scale = 1
            obs = np.concatenate([fen, pos], axis=2).astype(np.float32)
            assert obs.shape[2] == OBS_N_FEATURES, obs.shape
            lg, _ = pol(torch.as_tensor(obs, device=dev))
            p = torch.softmax(
                lg.masked_fill(~mask.unsqueeze(0), MASK_VALUE), -1).cpu().numpy()
            pb.append(p[:, 0]); ps.append(p[:, 1])
            if d % 40960 == 0:
                print(f"   {d:,}/{len(idx):,}")
    pb, ps = np.concatenate(pb), np.concatenate(ps)

    # Issue réelle du trade, pour chaque côté
    def issue(sens):
        a = atr[idx]
        px = cl[idx] + sens * spread / 2.0
        tp = px + sens * (RR * SL_MULT * a) + sens * spread / 2.0
        sl = px - sens * (SL_MULT * a) + sens * spread / 2.0
        res = np.full(len(idx), -1, np.int8)
        vivant = np.ones(len(idx), bool)
        for h in range(1, MAX_HOLD + 1):
            j = idx + h
            ok = vivant & (j < n)
            if not ok.any():
                break
            jj = j[ok]
            t_tp = (hi[jj] >= tp[ok]) if sens > 0 else (lo[jj] <= tp[ok])
            t_sl = (lo[jj] <= sl[ok]) if sens > 0 else (hi[jj] >= sl[ok])
            g, pr = t_tp & ~t_sl, t_sl
            pos_ = np.where(ok)[0]
            res[pos_[g]] = 1
            res[pos_[pr]] = 0
            v = vivant.copy(); v[pos_[g | pr]] = False; vivant = v
        return res

    print()
    print(f"{'côté':<8} {'n':>8} {'WR brut':>9} {'AUC conviction':>16} {'verdict':>12}")
    print("-" * 60)
    for sens, nom, p in ((+1, "LONG", pb), (-1, "SHORT", ps)):
        lab = issue(sens)
        ok = lab >= 0
        y, pp = lab[ok].astype(float), p[ok]
        if len(np.unique(y)) < 2:
            print(f"{nom:<8} {ok.sum():>8,}  issue constante")
            continue
        auc = roc_auc_score(y, pp)
        verdict = "SIGNAL" if auc > 0.52 else ("faible" if auc > 0.51 else "AUCUN")
        print(f"{nom:<8} {ok.sum():>8,} {100*y.mean():>8.1f}% {auc:>16.4f} {verdict:>12}")

        # Winrate par décile de conviction : le détail qui compte
        q = np.quantile(pp, [0.5, 0.8, 0.9, 0.95, 0.99])
        cells = "  ".join(
            f"top{int(100*(1-t)):>2}%={100*y[pp >= s].mean():.1f}%"
            for t, s in zip([0.5, 0.8, 0.9, 0.95, 0.99], q) if (pp >= s).sum() > 50)
        print(f"         {cells}")

    requis = (SL_MULT + spread / np.median(atr)) / (SL_MULT + RR * SL_MULT)
    print()
    print(f"WR d'équilibre à SL {SL_MULT}×ATR : {100*requis:.1f} %")
    print("AUC 0.50 = la conviction ne distingue rien. Le mécanisme de sélection")
    print("n'y peut alors rien, quelle que soit sa sophistication.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
