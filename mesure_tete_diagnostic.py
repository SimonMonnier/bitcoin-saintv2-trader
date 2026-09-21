# -*- coding: utf-8 -*-
"""LA TETE PREDIT-ELLE SA PROPRE CIBLE ?

CE QUI EST ETABLI AVANT D'OUVRIR CETTE MESURE (2026-09-20, fenetre de
validation, la fenetre de test n'est pas lue) :

  LA CIBLE TRANSFERE, et mieux que le code ne l'annonce : rho(indicateur 2 R,
  rendement reel) vaut +0.954 en entrainement et +0.921 en validation, pour un
  ecart de +3.5 a +3.9 R entre indicateur a 1 et a 0. Le code annoncait +0.892.

  LA TETE CLASSE A L'ENVERS : sa colonne ACHAT — la seule entrainee en
  long-only — donne rho -0.135 avec le rendement reel.

Une cible a +0.92 et une tete a -0.14 : l'echec est donc DANS LA TETE, pas
dans ce qu'on lui demande d'apprendre. Reste a savoir ou.

DEUX HYPOTHESES, ET ELLES APPELLENT DES REMEDES OPPOSES.

  SURAPPRENTISSAGE. La tete predit bien sur l'entrainement et mal ailleurs.
  Remede : capacite, regularisation, arret precoce.

  DECALAGE DE DISTRIBUTION. La tete est entrainee sur les etats que la
  POLITIQUE visite — environ 4 100 decisions par epoch, auto-selectionnees par
  le veto de tendance et par la capacite — mais elle est evaluee sur une
  grille UNIFORME de la fenetre. Elle pourrait donc predire correctement la ou
  elle apprend et s'inverser ailleurs. Remede : l'entrainer sur une grille
  independante de la politique.

CE QUE LA MESURE FAIT : rho(tete, indicateur) ET rho(tete, rendement reel),
sur la grille d'ENTRAINEMENT et sur celle de VALIDATION. Les quatre chiffres
separent les deux hypotheses sans ambiguite.

    python mesure_tete_diagnostic.py [checkpoint.pth]
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
from saint_core import FEATURE_COLS, N_POS_FEATURES, BUDGETS_PART

PAS = 12


def main() -> int:
    cfg = T.PPOConfig()
    import glob
    chemin = (sys.argv[1] if len(sys.argv) > 1
              else sorted(glob.glob("best_saintv2_or_exec31*.pth"))[-1])
    print(f"checkpoint : {chemin}\nside : {cfg.side}\n")

    df = pd.read_pickle(I.INSTRUMENTS[cfg.symbol]["cache"])
    n = len(df)
    d0, d1 = int(n * 0.55), int(n * 0.70)
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
    xp[:, 5] = float(cfg.budget_part)

    print(f"{'fenetre':<14} {'n':>7} {'rho(tete, cible)':>19} "
          f"{'rho(tete, reel)':>18} {'rho(cible, reel)':>18}")
    print("-" * 80)
    for nom, (a, b) in (("ENTRAINEMENT", (0, d0)), ("VALIDATION", (d0, d1))):
        idx = np.arange(a + cfg.lookback + 1, b - C.BORNE_DEFAUT - 2, PAS)
        ind, _ = C.rendements(df, idx, cfg, indicateur=True)
        reel, _ = C.rendements(df, idx, cfg, indicateur=False)
        i_, r_ = np.asarray(ind, float), np.asarray(reel, float)
        ok = np.isfinite(i_) & np.isfinite(r_)
        idx, i_, r_ = idx[ok], i_[ok], r_[ok]
        sc = []
        with torch.no_grad():
            for d in range(0, len(idx), 2048):
                bb = idx[d:d + 2048]
                o = np.stack([np.concatenate(
                    [data.features[k - cfg.lookback:k], xp], axis=-1)
                    for k in bb])
                sc.append(pol.rendement(torch.from_numpy(o).to(dev))
                          .float().numpy()[:, 0])
        sc = np.concatenate(sc)
        print(f"{nom:<14} {len(idx):>7,} "
              f"{T._correlation_rang(sc, i_):>+19.4f} "
              f"{T._correlation_rang(sc, r_):>+18.4f} "
              f"{T._correlation_rang(i_, r_):>+18.4f}")
    print("-" * 80)
    print("LECTURE.")
    print("  rho(tete, cible) POSITIF en entrainement et NEGATIF en validation")
    print("     -> surapprentissage : elle a memorise, elle ne generalise pas.")
    print("  rho(tete, cible) NEGATIF PARTOUT, y compris en entrainement")
    print("     -> elle n'apprend pas sa cible du tout sur une grille uniforme,")
    print("        alors qu'elle s'entraine sur les etats de la POLITIQUE :")
    print("        c'est un decalage de distribution, pas un surapprentissage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
