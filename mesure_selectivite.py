"""Elargir la selectivite donne-t-il plus de trades ? Et est-ce que ca paie ?

LA QUESTION SEMBLE ARITHMETIQUE, ELLE NE L'EST PAS. Passer de 5 % a 25 % des
barres ne donne pas cinq fois plus de trades, parce qu'UNE SEULE POSITION peut
etre tenue a la fois : tout signal emis pendant qu'un trade court est ignore.
Le nombre de trades sature donc a *duree de la fenetre / duree d'un trade*, et
cette borne ne depend pas du tout du modele.

C'est devenu le fait dominant le 16 septembre. Avec la sortie au stop suiveur
a 12xATR, la duree mediane est passee de ~17 h a ~34 h : le plafond a ete
divise par deux le jour ou la geometrie a change, sans que personne ne le
mesure. Elargir la selectivite ne peut pas le franchir.

TROIS EFFETS S'OPPOSENT, et ce fichier les separe :

  1. plus de barres eligibles           -> plus de trades
  2. occupation : un trade en cours      -> saturation, effet 1 s'annule
  3. l'avantage decroit quand on descend -> chaque trade ajoute vaut moins
     dans le classement                     que le precedent

Ce qui decide n'est aucun des trois seul, mais le produit : l'avantage TOTAL
de la fenetre vaut n x E[R], et sa lisibilite vaut E[R] x racine(n). Les deux
sont rendus, parce qu'ils ne designent pas toujours le meme optimum.

LE TEST N'EST PAS OUVERT : bornes du fold derivees comme dans
`run_walkforward`, seule la VALIDATION est lue.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

import checkpoints as CK
import cibles as C
import training as T
from saint_core import FEATURE_COLS, build_policy

T_TRAIN, T_VAL, T_TEST = 0.55, 0.15, 0.10
PAS = 3                 # une decision toutes les 15 minutes
LOT = 4096
NIVEAUX = (1.00, 0.50, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01)


def _err_bloc(y: np.ndarray, nb: int = 30) -> float:
    """Erreur-type quand les points voisins partagent leur avenir."""
    if len(y) < 3 * nb:
        nb = max(len(y) // 3, 2)
    t = max(len(y) // nb, 1)
    bl = np.array([y[i:i + t].mean() for i in range(0, len(y) - t + 1, t)])
    return float(bl.std(ddof=1) / np.sqrt(len(bl))) if len(bl) > 2 else float("nan")


def enchaine(sig, sens, rend, dur, pos, seuil):
    """Les trades REELLEMENT pris : une position a la fois, dans l'ordre.

    `pos` est l'indice de barre de chaque decision. On avance dans le temps ;
    des qu'un signal depasse le seuil et qu'on est plat, on entre, et on
    n'ecoute plus rien jusqu'a la sortie. C'est exactement la contrainte de
    l'environnement et du live — et c'est elle qui plafonne tout.
    """
    pris, libre = [], -1
    for k in range(len(sig)):
        if pos[k] < libre or sig[k] < seuil:
            continue
        if not np.isfinite(rend[k]) or not np.isfinite(dur[k]):
            continue
        pris.append(k)
        libre = pos[k] + dur[k]
    return np.array(pris, dtype=int)


def main() -> int:
    quel = sys.argv[2] if len(sys.argv) > 2 else "last"
    try:
        complet, folds = CK.resoud(sys.argv[1] if len(sys.argv) > 1 else None,
                                   famille=quel)
    except FileNotFoundError as e:
        print(f"AUCUN CHECKPOINT UTILISABLE : {e}")
        return 1

    cfg = T.PPOConfig()
    df = T.load_mt5_data(cfg)
    n = len(df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold = int(sys.argv[3]) if len(sys.argv) > 3 else folds[0]
    if fold not in folds:
        print(f"fold {fold} absent ; disponibles : {folds}")
        return 1
    depart = (fold - 1) * int(n * T_TEST)
    a_tr, b_tr = depart, depart + int(n * T_TRAIN)
    tr, va = b_tr, b_tr + int(n * T_VAL)
    print(f"checkpoint {complet}, fold {fold}")
    print(f"VALIDATION [{tr:,}, {va:,})   test [{va:,}, ...) intouche")
    print(f"sortie : stop {cfg.atr_sl_mult:g}xATR, trailing "
          f"{cfg.atr_trail_mult / cfg.atr_sl_mult:.2f} R, objectif "
          f"{'aucun' if not cfg.use_tp else cfg.atr_tp_mult}")

    pth, calib_p, _ = CK.chemins(complet, fold)
    etat = torch.load(pth, map_location=device, weights_only=True)
    calib = json.load(open(calib_p, encoding="utf-8"))
    look = int(calib.get("lookback", cfg.lookback))

    stats = T.compute_and_save_global_norm_stats(df.iloc[a_tr:b_tr],
                                                 FEATURE_COLS, path=None)
    dv = df.iloc[tr:va].reset_index(drop=True)
    mu = np.asarray(stats["mean"], np.float32)
    sd = np.maximum(np.asarray(stats["std"], np.float32), 1e-8)
    X = np.clip((dv[FEATURE_COLS].to_numpy(np.float32) - mu) / sd,
                -10.0, 10.0).astype(np.float32)
    del df

    policy = build_policy(device, lookback=look, state_dict=etat)
    policy.load_state_dict(etat, strict=True)
    policy.eval()

    idx = np.arange(look, len(dv) - C.BORNE_DEFAUT - 2, PAS)
    jours = len(dv) * 5 / 60 / 24
    print(f"{len(idx):,} decisions sur {jours:.0f} jours de validation\n")

    extra = np.zeros((look, 4), np.float32)
    extra[:, 3] = 1.0
    pb, ps = [], []
    with torch.no_grad():
        for d in range(0, len(idx), LOT):
            b = idx[d:d + LOT]
            o = np.stack([np.concatenate([X[i - look:i], extra], axis=-1)
                          for i in b])
            lg = policy(torch.from_numpy(o).to(device))
            if isinstance(lg, tuple):
                lg = lg[0]
            pr = torch.softmax(lg, dim=-1).float().cpu().numpy()
            pb.append(pr[:, 0])
            ps.append(pr[:, 1])
    pb, ps = np.concatenate(pb), np.concatenate(ps)

    ra, rv, da, dv_ = C.rendements(dv, idx, cfg, durees=True)

    # MEME REGLE QU'EN PRODUCTION : meilleur cote, puis comparaison au seuil.
    # Surtout pas d'argmax sur les trois actions — a 5 % de selectivite
    # p(HOLD) est majoritaire partout et l'argmax rendrait HOLD en permanence.
    sig = np.maximum(pb, ps)
    sens = np.where(pb >= ps, 1, -1)
    rend = np.where(sens > 0, ra, rv)
    dur = np.where(sens > 0, da, dv_)

    med = np.nanmedian(dur)
    print(f"duree mediane d'un trade : {med * 5 / 60:.1f} h")
    print(f"PLAFOND D'OCCUPATION : {jours * 24 / (med * 5 / 60):.0f} trades "
          f"au maximum sur cette fenetre, quelle que soit la selectivite\n")

    print(f"{'garde':>7} {'trades':>7} {'E[R]':>9} {'+/-':>8} {'sigma':>7} "
          f"{'R total':>9} {'achat':>7} {'duree h':>8}")
    print("-" * 70)
    for q in NIVEAUX:
        seuil = np.quantile(sig, 1.0 - q) if q < 1.0 else -np.inf
        k = enchaine(sig, sens, rend, dur, idx, seuil)
        if len(k) < 15:
            print(f"{100*q:>6.0f}% {len(k):>7}   (trop peu pour mesurer)")
            continue
        y = rend[k]
        e = _err_bloc(y)
        print(f"{100*q:>6.0f}% {len(k):>7} {y.mean():>+9.4f} {e:>8.4f} "
              f"{y.mean()/max(e,1e-9):>+7.1f} {y.sum():>+9.1f} "
              f"{100*(sens[k]>0).mean():>6.0f}% {np.mean(dur[k])*5/60:>8.1f}")

    print("\nLECTURE. 'trades' est le nombre REELLEMENT pris, occupation")
    print("comprise. Si la colonne cesse de monter quand 'garde' monte, c'est")
    print("le plafond d'occupation qui mord, et elargir n'apporte plus rien.")
    print("'R total' est l'avantage de la fenetre entiere ; 'sigma' dit si on")
    print("saurait le distinguer de zero. Les deux ne designent pas toujours")
    print("le meme reglage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
