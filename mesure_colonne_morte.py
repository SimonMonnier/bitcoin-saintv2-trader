# -*- coding: utf-8 -*-
"""LE SCORE DE RANG SOUSTRAIT-IL UNE COLONNE NON ENTRAINEE ?

LA CONTRADICTION QUI OUVRE L'ENQUETE. `cibles._regle_cible` affirme que la
cible de la tete auxiliaire transfere au rendement encaisse avec une
correlation de RANG de +0.892. Or exec31 affiche `rhoAux` NEGATIF dans les
trois folds : -0.055, -0.145, -0.037. Les deux ne peuvent pas etre vrais.

L'HYPOTHESE. Le score est forme ligne 7170 comme `a[:, 0] - a[:, 1]`, soit
ACHAT moins VENTE. C'est juste en bilateral. Mais en `side = "long"` la perte
auxiliaire est MASQUEE par le cote : la colonne VENTE ne recoit aucun
gradient et reste a son initialisation aleatoire. Le score vaudrait alors
(prediction entrainee) - (bruit), et soustraire un bruit de variance
comparable detruit le classement — voire l'inverse.

CE QUE LA MESURE COMPARE, sur les MEMES occasions de validation et le MEME
checkpoint :

    rho(a[:,0] - a[:,1], rendement reel)   le score actuel
    rho(a[:,0],          rendement reel)   la seule colonne entrainee
    rho(a[:,1],          rendement reel)   la colonne supposee morte

Si la deuxieme est positive et la premiere negative, l'hypothese est
confirmee et le correctif est d'une ligne.

LA FENETRE DE TEST N'EST PAS LUE.

    python mesure_colonne_morte.py [checkpoint.pth]
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import torch

import cibles as C
import instruments as I
import training as T
import saint_core as S
from saint_core import FEATURE_COLS, N_POS_FEATURES, BUDGETS_RISQUE

PAS = 12


def main() -> int:
    cfg = T.PPOConfig()
    chemin = sys.argv[1] if len(sys.argv) > 1 else None
    if chemin is None:
        import glob
        cands = sorted(glob.glob("best_saintv2_or_exec31*.pth"))
        if not cands:
            print("Aucun checkpoint exec31 : passer le chemin en argument.")
            return 1
        chemin = cands[-1]
    print(f"checkpoint : {chemin}")

    df = pd.read_pickle(I.INSTRUMENTS[cfg.symbol]["cache"])
    n = len(df)
    # FENETRE DE VALIDATION, jamais celle de test.
    d0, d1 = int(n * 0.55), int(n * 0.70)
    idx = np.arange(d0 + cfg.lookback + 1, d1 - C.BORNE_DEFAUT - 2, PAS)
    ra, _ = C.rendements(df, idx, cfg, indicateur=False)
    reel = np.asarray(ra, float)
    ok = np.isfinite(reel)
    idx, reel = idx[ok], reel[ok]

    stats = T.compute_and_save_global_norm_stats(df.iloc[:d0], FEATURE_COLS,
                                                 path=None)
    data = T.MarketData(df.iloc[:d1].reset_index(drop=True), FEATURE_COLS,
                        stats)

    dev = torch.device("cpu")
    etat = torch.load(chemin, map_location=dev)
    if isinstance(etat, dict) and "model" in etat:
        etat = etat["model"]
    pol = S.build_policy(dev, lookback=cfg.lookback,
                         n_features=S.OBS_N_FEATURES, archi=cfg.architecture,
                         n_ref=cfg.n_ref, num_blocks=cfg.num_blocks,
                         d_model_patch=cfg.d_model_patch, mlp_dim=cfg.mlp_dim,
                         taille_patch=cfg.taille_patch, pas=cfg.pas_patch,
                         state_dict=etat)
    pol.eval()

    xp = np.zeros((cfg.lookback, N_POS_FEATURES), np.float32)
    xp[:, 3] = 1.0
    xp[:, 5] = float(cfg.budget_risque / max(BUDGETS_RISQUE[-1], 1e-9))
    a0, a1 = [], []
    with torch.no_grad():
        for d in range(0, len(idx), 2048):
            b = idx[d:d + 2048]
            o = np.stack([np.concatenate([data.features[i - cfg.lookback:i],
                                          xp], axis=-1) for i in b])
            r = pol.rendement(torch.from_numpy(o).to(dev)).float().numpy()
            a0.append(r[:, 0]); a1.append(r[:, 1])
    a0 = np.concatenate(a0); a1 = np.concatenate(a1)

    print(f"{len(idx):,} occasions de validation  |  side = {cfg.side}\n")
    print(f"{'score':<34} {'rho avec le rendement reel':>28}")
    print("-" * 64)
    for nom, v in (("a[:,0] - a[:,1]   (score ACTUEL)", a0 - a1),
                   ("a[:,0]            (colonne ACHAT)", a0),
                   ("a[:,1]            (colonne VENTE)", a1)):
        print(f"{nom:<34} {T._correlation_rang(v, reel):>+28.4f}")
    print("-" * 64)
    print(f"\n{'colonne':<12} {'moyenne':>10} {'ecart-type':>12} "
          f"{'|correlation| avec l autre':>28}")
    print("-" * 64)
    print(f"{'ACHAT':<12} {a0.mean():>+10.4f} {a0.std():>12.4f}")
    print(f"{'VENTE':<12} {a1.mean():>+10.4f} {a1.std():>12.4f} "
          f"{abs(np.corrcoef(a0, a1)[0,1]):>28.4f}")
    print("-" * 64)
    print("LECTURE. Si la colonne ACHAT classe positivement et que le score")
    print("ACTUEL classe negativement, c'est la soustraction d'une colonne")
    print("non entrainee qui detruit le classement. L'ecart-type de la")
    print("colonne VENTE dit combien de bruit est soustrait.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
