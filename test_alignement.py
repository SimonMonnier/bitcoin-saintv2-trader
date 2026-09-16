"""Le live doit calculer EXACTEMENT les colonnes de l'entrainement.

CE QUE CE TEST FAIT. Il recupere des bougies par le chemin du LIVE — l'API
Binance, via `flux_live` — sur une fenetre qui se termine a l'interieur du jeu
d'entrainement, leur applique `prepare_m5.construit`, et compare les 260
colonnes obtenues a celles que le jeu porte au MEME horodatage.

POURQUOI IL EXISTE. Le live et l'entrainement construisaient leurs colonnes par
deux chemins differents : `merge_m1_h1` d'un cote, `prepare_m5.construit` de
l'autre. Rien ne verifiait qu'ils s'accordaient, et ils avaient cesse de le
faire depuis longtemps — le live etait reste en M1, 30 colonnes, stop a 2xATR,
pendant que l'entrainement passait au M5, 260 colonnes et 8xATR. Un modele
charge dans cet etat n'aurait leve aucune erreur : il aurait trade, en lisant
autre chose que ce sur quoi il a appris.

CE QU'IL COUVRE. La chaine entiere du live : recuperation, retrait de la bougie
en formation, indicateurs, contextes H1 et H4 avec leurs merge_asof, flux,
range et Ichimoku. Plus la coherence des REGLAGES entre l'entrainement et le
live : echelle, geometrie des barrieres, profondeur de fenetre.

CE QU'IL NE COUVRE PAS. L'execution chez le courtier — le prix MT5 n'est pas
le prix Binance, et cet ecart est precisement ce que la friction represente.
Ni la normalisation, qui depend du fold et vit dans le fichier `_norm.npz`.

    python test_alignement.py
"""

import sys

import numpy as np
import pandas as pd

import flux_live
import prepare_m5
import saint_core as S

# Tolerance relative. Le jeu est stocke en float32 : c'est la seule difference
# numerique attendue entre les deux chemins, qui passent par le meme code.
TOL = 3e-6
N_LIGNES = 5            # lignes comparees, en remontant depuis la fin commune


def _reglages() -> list[str]:
    """Verifie que chaque consommateur du modele decrit le MEME trade.

    Les evaluateurs (`evalue_test_exhaustif`, `evalue_ensemble`) ne sont pas
    listes : ils construisent leur environnement depuis `training.PPOConfig` et
    `BTCTradingEnvDiscrete`, donc ils suivent par construction. Seuls les
    fichiers qui RECOPIENT les reglages peuvent diverger — et c'est exactement
    pour cela qu'ils sont verifies ici.
    """
    import training as T
    import kairos_live as K
    cfg_t, cfg_l = T.PPOConfig(), K.LiveConfig()
    ecarts = []
    for nom, a, b in (
        ("stop (xATR)", cfg_t.atr_sl_mult, cfg_l.atr_sl_mult),
        ("objectif (xATR)", cfg_t.atr_tp_mult, cfg_l.atr_tp_mult),
        ("lookback", cfg_t.lookback, cfg_l.lookback),
        ("risque par trade", cfg_t.risk_per_trade, cfg_l.risk_per_trade),
    ):
        if float(a) != float(b):
            ecarts.append(f"live : {nom} — entrainement {a}, live {b}")
    if S.TIMEFRAME != "M5":
        ecarts.append(f"saint_core.TIMEFRAME vaut {S.TIMEFRAME}, pas M5")

    # Le stress-test recopie lui aussi ses reglages, dans sa propre LiveConfig.
    try:
        import backtest_saintv2_stress_test as B
        cfg_b = B.LiveConfig()
        if cfg_b.symbol != "BTCUSD":
            ecarts.append(f"stress-test : instrument {cfg_b.symbol}, "
                          f"pas BTCUSD")
        for nom, a, b in (
            ("stop (xATR)", cfg_t.atr_sl_mult, cfg_b.atr_sl_mult),
            ("objectif (xATR)", cfg_t.atr_tp_mult, cfg_b.atr_tp_mult),
            ("lookback", cfg_t.lookback, cfg_b.lookback),
        ):
            if float(a) != float(b):
                ecarts.append(f"stress-test : {nom} — entrainement {a}, "
                              f"backtest {b}")
    except Exception as e:                      # MT5 absent, par exemple
        ecarts.append(f"stress-test : non verifiable ({type(e).__name__})")
    return ecarts


def main() -> int:
    jeu = pd.read_pickle(prepare_m5.SORTIE)
    fin_jeu = jeu["time"].iloc[-1]
    print(f"jeu d'entrainement : {len(jeu):,} barres jusqu'a {fin_jeu}")

    # On demande au chemin du LIVE de quoi couvrir l'echauffement complet plus
    # les lignes comparees, en s'arretant dans la plage du jeu.
    brut = flux_live.ferme(flux_live.bougies(n=flux_live.ECHAUFFEMENT + 2000))
    brut = brut[brut["time"] <= fin_jeu].reset_index(drop=True)
    if len(brut) < flux_live.ECHAUFFEMENT:
        print(f"ECHEC : seulement {len(brut):,} bougies communes, "
              f"il en faut {flux_live.ECHAUFFEMENT:,} pour l'echauffement H4.")
        return 1
    print(f"chemin live        : {len(brut):,} bougies jusqu'a "
          f"{brut['time'].iloc[-1]}")

    vivant, colonnes, _ = prepare_m5.construit(brut.copy())
    vivant = vivant.replace([np.inf, -np.inf], np.nan)
    vivant = vivant.dropna(subset=colonnes + ["atr_14"]).reset_index(drop=True)

    cols = list(S.FEATURE_COLS)
    manquantes = [c for c in cols if c not in vivant.columns]
    if manquantes:
        print(f"ECHEC : le chemin live ne produit pas {len(manquantes)} "
              f"colonnes, dont {manquantes[:5]}")
        return 1

    communes = vivant["time"].iloc[-N_LIGNES:]
    jeu_i = jeu.set_index("time")
    absent = [t for t in communes if t not in jeu_i.index]
    if absent:
        print(f"ECHEC : horodatages absents du jeu : {absent[:3]}")
        return 1

    print(f"\n{len(cols)} colonnes comparees sur {len(communes)} horodatages "
          f"communs, tolerance relative {TOL:g}\n")

    fautives = {}
    for t in communes:
        a = jeu_i.loc[t, cols].to_numpy(np.float64)
        b = vivant.set_index("time").loc[t, cols].to_numpy(np.float64)
        na, nb = np.isnan(a), np.isnan(b)
        ecart = np.abs(a - b)
        ecart = np.where(na & nb, 0.0, ecart)
        ecart = np.where(na ^ nb, np.inf, ecart)
        echelle = np.maximum(np.abs(np.where(na, 0.0, a)), 1.0)
        for k in np.flatnonzero(ecart > TOL * echelle):
            fautives.setdefault(cols[k], []).append(
                (t, float(a[k]), float(b[k])))

    ecarts_reglages = _reglages()

    if fautives:
        print(f"{len(fautives)} COLONNES DIFFERENT entre le jeu et le live :\n")
        print(f"{'colonne':>28} {'pts':>5}  {'jeu -> live':>34}")
        print("-" * 74)
        for k, v in sorted(fautives.items(), key=lambda x: -len(x[1]))[:15]:
            t, a, b = v[0]
            print(f"{k:>28} {len(v):3d}/{len(communes):<2d}  "
                  f"{a:+15.8f} -> {b:+15.8f}")
        print("\nUn ecart ici veut dire que le modele lira en production autre")
        print("chose que ce sur quoi il a appris. Aucun n'est acceptable.")
    else:
        print(f"COLONNES ALIGNEES : les {len(cols)} colonnes du chemin live")
        print("rendent les memes valeurs que le jeu d'entrainement.")

    if ecarts_reglages:
        print(f"\n{len(ecarts_reglages)} REGLAGES DIVERGENT :")
        for e in ecarts_reglages:
            print(f"  {e}")
        print("\nLes colonnes peuvent etre identiques et le trade different :")
        print("un stop qui n'est pas celui de l'entrainement execute une autre")
        print("strategie que celle qui a ete mesuree.")
    else:
        print("\nREGLAGES ALIGNES : meme echelle, meme geometrie, meme fenetre.")

    return 1 if (fautives or ecarts_reglages) else 0


if __name__ == "__main__":
    raise SystemExit(main())
