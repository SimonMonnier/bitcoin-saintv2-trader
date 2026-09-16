"""Le modele tient-il quand l'execution se degrade ?

CE QUI REMPLACE QUOI. `backtest_saintv2_stress_test.py` reimplementait
l'environnement au-dessus de MT5 : sa propre lecture des bougies, son propre
moteur d'ordres, ses propres reglages recopies a la main. Ils avaient derive
jusqu'a l'absurde — instrument XAUUSD, stop de 5xATR annote "optimum mesure sur
l'or", lookback 25 — pendant que l'entrainement etait en BTCUSD M5 a 8xATR et
lookback 4. Le fichier ne decrivait plus aucune strategie de ce depot.

Ici il n'y a pas de second moteur. On prend `BTCTradingEnvDiscrete`, celui de
l'entrainement, et on degrade ce qu'il consomme : la friction, l'estimation de
l'ATR, le chemin de prix. Un reglage qui change dans `PPOConfig` suit donc
automatiquement, et il n'y a plus rien a maintenir en double.

CE QU'ON DEGRADE, ET POURQUOI CHACUN.

  spread         Le cout d'aller-retour est la grandeur la plus directement
                 mortelle : l'avantage mesure du modele vaut +0.09 a +0.13 R
                 pour une friction de 0.049 R. Doubler le spread mange la
                 moitie de ce qui reste.
  glissement     Un ordre au marche ne s'execute pas au prix affiche. A
                 l'entree ET a la sortie.
  meches         Les barres agregees perdent l'ORDRE des extremes : un stop
                 peut avoir ete touche intra-barre sans que le minimum de la
                 barre le montre. On etend high/low.
  ATR fausse     Les barrieres sont posees en multiples d'ATR. Si l'ATR est
                 mal estime au moment de l'entree, le trade n'a ni le risque
                 ni le rapport annonces.
  micro-gaps     Un saut de prix entre deux barres : le stop s'execute plus
                 loin que la ou il etait pose.
  pics de news   Un extreme isole qui declenche des barrieres que le chemin
                 "normal" n'aurait pas atteintes.

SUR QUELLE FENETRE. La VALIDATION par defaut. La fenetre de test ne sert
qu'une fois, et un balayage de six scenarios dessus serait exactement le genre
de reglage qui a deja coute trois a quatre points a ce depot. `--test` existe
pour le tir unique, et le dit en clair avant de partir.

    python stress_test.py [prefixe] [--test]
"""

import sys

import numpy as np
import torch

import evalue_test_exhaustif as E
import training as T
from saint_core import EntryDecisionPolicy, FEATURE_COLS, build_policy

PREFIXE = "last_saintv2_loup_duel_exec27_saint"
GRAINE = 0


def _copie(donnees):
    """Copie superficielle du jeu, avec ses series de prix DUPLIQUEES.

    On ne veut degrader que la copie : les scenarios se comparent entre eux et
    au temoin, donc ils doivent tous partir du meme etat.
    """
    import copy
    d = copy.copy(donnees)
    for nom in ("high", "low", "close", "open", "atr14"):
        if hasattr(d, nom):
            setattr(d, nom, np.array(getattr(d, nom), copy=True))
    return d


def degrade(donnees, rng, meches=0.0, atr_err=0.0, gap_prob=0.0, gap_std=0.0,
            pic_prob=0.0, pic_ampl=0.0):
    """Applique les perturbations de CHEMIN DE PRIX. Rend un nouveau jeu."""
    d = _copie(donnees)
    n = len(d.close)

    if meches > 0.0:
        # Les extremes s'ecartent, jamais ne se resserrent : une barre agregee
        # ne peut que SOUS-estimer l'amplitude reellement parcourue.
        d.high = d.high * (1.0 + np.abs(rng.normal(0, meches, n)))
        d.low = d.low * (1.0 - np.abs(rng.normal(0, meches, n)))

    if atr_err > 0.0:
        d.atr14 = d.atr14 * (1.0 + rng.uniform(-atr_err, atr_err, n))

    if gap_prob > 0.0:
        saut = np.where(rng.random(n) < gap_prob,
                        rng.normal(0, gap_std, n), 0.0)
        facteur = np.cumprod(1.0 + saut).astype(np.float32)
        for nom in ("high", "low", "close", "open"):
            setattr(d, nom, getattr(d, nom) * facteur)
        d.atr14 = d.atr14 * facteur

    if pic_prob > 0.0:
        touche = rng.random(n) < pic_prob
        ampl = np.where(touche, np.abs(rng.normal(0, pic_ampl, n)), 0.0)
        haut = rng.random(n) < 0.5
        d.high = d.high * (1.0 + np.where(haut, ampl, 0.0))
        d.low = d.low * (1.0 - np.where(~haut, ampl, 0.0))

    d.high = np.maximum.reduce([d.high, d.close, d.open])
    d.low = np.minimum.reduce([d.low, d.close, d.open])
    return d


# (nom, reglages de friction, perturbations du chemin de prix)
SCENARIOS = [
    ("temoin", {}, {}),
    ("spread x2", {"spread_bps": 5.22}, {}),
    ("glissement x3", {"entry_slippage_bps": 3.0, "slippage_bps": 6.0}, {}),
    ("meches +1.5 bps", {}, {"meches": 1.5e-4}),
    ("ATR fausse 10 %", {}, {"atr_err": 0.10}),
    ("micro-gaps", {}, {"gap_prob": 0.002, "gap_std": 6e-4}),
    ("pics de news", {}, {"pic_prob": 0.0005, "pic_ampl": 2e-3}),
    ("TOUT ENSEMBLE", {"spread_bps": 5.22, "entry_slippage_bps": 3.0,
                       "slippage_bps": 6.0},
     {"meches": 1.5e-4, "atr_err": 0.10, "gap_prob": 0.002, "gap_std": 6e-4,
      "pic_prob": 0.0005, "pic_ampl": 2e-3}),
]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sur_test = "--test" in sys.argv
    prefixe = args[0] if args else PREFIXE

    cfg_base = T.PPOConfig()
    df = T.load_mt5_data(cfg_base)
    n = len(df)
    train_len = int(n * E.TRAIN_FRAC)
    val_len = int(n * E.VAL_FRAC)
    test_len = int(n * E.TEST_FRAC)
    window = train_len + val_len + test_len
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fenetre = "TEST" if sur_test else "validation"
    print(f"{prefixe}  |  fenetre {fenetre}  |  "
          f"SL {cfg_base.atr_sl_mult}xATR  TP {cfg_base.atr_tp_mult}xATR")
    if sur_test:
        print("\n  LA FENETRE DE TEST NE SERT QU'UNE FOIS. Choisir quoi que ce")
        print("  soit d'apres ces chiffres la contamine definitivement.\n")
    else:
        print("  (la fenetre de test reste intouchee ; --test pour le tir unique)")
    print()

    resultats = {nom: [] for nom, _, _ in SCENARIOS}

    for fold in range(1, E.N_FOLDS + 1):
        start = (fold - 1) * test_len
        if start + window > n:
            break
        base = f"{prefixe}_wf{fold}_both_wf{fold}"
        try:
            etat = torch.load(base + ".pth", map_location=device,
                              weights_only=True)
            import json
            calib = json.load(open(base + "_calib.json", encoding="utf-8"))
        except FileNotFoundError:
            print(f"wf{fold} : checkpoint absent — fold saute")
            continue

        stats = T.compute_and_save_global_norm_stats(
            df.iloc[start:start + train_len], FEATURE_COLS, path=None)
        _, _, val_data, test_data = T.create_datasets_from_slices(
            df, FEATURE_COLS, start=start, train_len=train_len,
            val_len=val_len, test_len=test_len, stats=stats,
            calib_frac=cfg_base.calib_frac)
        donnees = test_data if sur_test else val_data

        policy = build_policy(device, lookback=int(calib.get("lookback",
                                                            cfg_base.lookback)),
                              state_dict=etat,
                              heads=int(calib.get("saint_heads", 0)))
        policy.load_state_dict(etat, strict=True)
        policy.eval()

        for nom, friction, chemin in SCENARIOS:
            rng = np.random.default_rng(GRAINE + fold)
            cfg = T.PPOConfig(**cfg_base.__dict__)
            for k, v in friction.items():
                setattr(cfg, k, v)
            jeu = degrade(donnees, rng, **chemin) if chemin else donnees
            env = T.BTCTradingEnvDiscrete(jeu, cfg)
            departs = T.departs_disjoints(jeu.length, cfg.lookback,
                                          cfg.episode_length)
            pnls = []
            for d in departs:
                decision = EntryDecisionPolicy(calib["decision_policy"])
                p, _ = E.joue(policy, env, decision, d, device)
                pnls += p
            resultats[nom] += pnls
        print(f"wf{fold} : {len(departs)} episodes disjoints x "
              f"{cfg_base.episode_length} barres")

    print()
    print(f"{'scenario':<18} {'trades':>7} {'WR':>7} {'PF':>6} "
          f"{'$/trade':>9} {'ecart pt mort':>14}")
    print("-" * 66)
    temoin = None
    for nom, _, _ in SCENARIOS:
        pnls = resultats[nom]
        if not pnls:
            print(f"{nom:<18} aucun trade")
            continue
        a = np.array(pnls, float)
        g, p = a[a > 0], a[a <= 0]
        wr = 100.0 * len(g) / len(a)
        pf = (g.sum() / abs(p.sum())) if len(p) and p.sum() else float("inf")
        # Point mort sur les gains et pertes REELLEMENT realises, jamais sur le
        # R:R nominal : c'est la distribution obtenue qui decide.
        rapport = (g.mean() / abs(p.mean())) if len(g) and len(p) else 0.0
        bd = 100.0 / (1.0 + rapport) if rapport > 0 else float("nan")
        ecart = wr - bd
        if temoin is None:
            temoin = ecart
        delta = "" if nom == "temoin" else f"  ({ecart - temoin:+.1f})"
        print(f"{nom:<18} {len(a):>7} {wr:>6.1f}% {pf:>6.2f} "
              f"{a.mean():>+8.2f}$ {ecart:>+13.1f}{delta}")

    print("\nLa colonne entre parentheses est la perte par rapport au temoin.")
    print("Un modele dont l'avantage ne survit pas au doublement du spread")
    print("n'a pas d'avantage : il a une mesure de friction optimiste.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
