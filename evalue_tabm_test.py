"""TabM confronte aux memes fenetres de test que PPO, dans le meme moteur.

POURQUOI TABM MERITE CE TEST. C'est la seule methode de ce depot dont le signe
ait tenu partout : sur neuf ans de H1, elle bat le tirage au hasard de
+0.085 R dans SIX periodes sur six (test des signes p = 0.016), et son
avantage grandit avec l'exigence d'abstention (+0.024 R a marge 0.10, +0.032 a
0.20). PPO, lui, montrait +4 a +6 points en validation qui ne se transferaient
pas : -0.2 +/- 2.2 points au test.

Mais TabM n'avait jamais ete mesure sur une fenetre de test, ni dans le moteur
reel. Ses chiffres etaient des esperances de rendement en unites de risque,
calculees par un simulateur de barrieres. Ceux de PPO etaient des dollars
sortis d'un environnement qui a ses propres regles de sortie. Comparer les
deux revenait a comparer deux instruments differents.

CE FICHIER SUPPRIME CETTE DIFFERENCE. TabM decide, l'environnement execute :
memes barrieres, meme friction, meme detention maximale, memes fenetres, meme
couverture exhaustive en episodes disjoints. Le tableau produit se lit ligne a
ligne contre celui d'evalue_test_exhaustif.py.

TROIS CHOSES QUI POURRAIENT FAUSSER LA COMPARAISON, ET CE QU'ON EN FAIT.

  1. LA CIBLE D'ENTRAINEMENT N'EST PAS LA SORTIE REELLE. On entraine TabM a
     predire le rendement d'une course SL 2xATR / TP 4xATR sur 30 barres, ce
     qui est le reglage de l'environnement. Mais l'environnement applique en
     plus ses propres regles de gestion en cours de trade. La cible est donc
     une APPROXIMATION de ce que l'execution rendra, et c'est assume : c'est
     exactement la situation d'un modele deploye.

  2. LA MARGE D'ABSTENTION SE CALIBRE SUR LA VALIDATION, jamais sur le test —
     la meme fenetre que PPO utilise pour ses seuils, pour que les deux aient
     vu autant de donnees avant d'etre juges.

  3. LA NORMALISATION EST CELLE DU FOLD, calculee sur son train seul, comme
     pour PPO. Reutiliser des stats calculees sur tout l'historique ferait
     entrer le futur par la porte de service.

    python evalue_tabm_test.py
"""

import sys

import numpy as np

import banc_rendement_net as B
import mesure_features as MF
import training as T
from evalue_test_exhaustif import resume
from saint_core import FEATURE_COLS

TRAIN_FRAC, VAL_FRAC, TEST_FRAC, N_FOLDS = 0.55, 0.15, 0.10, 3
# Reglages de l'environnement, repris tels quels pour que la cible apprise
# decrive le trade que l'execution fera reellement.
SL_MULT, RR, HOLD = 2.0, 2.0, 30
PAS_TRAIN = 6          # un echantillon toutes les 6 barres : le chevauchement
                       # correle les exemples d'entrainement sans les biaiser,
                       # contrairement a l'evaluation ou il est interdit.
# La selectivite n'est PAS fixee : elle se choisit fold par fold sur la
# fenetre de validation, puis s'applique au test sans etre revue.
#
# POURQUOI CE DETOUR. Un premier passage a balaye ces quatre valeurs en lisant
# directement le test : 0.10 donnait +2.8 pt, 0.05 +1.6, 0.02 -3.0, 0.01 -4.6.
# Annoncer le +2.8 aurait ete annoncer un maximum sur quatre essais, c'est-a-
# dire choisir un reglage sur les donnees censees mesurer la generalisation —
# exactement la faute que ce depot a deja payee avec la selection de
# checkpoint sur validation, qui coute trois points au test.
CANDIDATS = (0.20, 0.10, 0.05, 0.02)


def cibles_brutes(df, idx):
    """Rendement NET en unites de risque, pour un BUY puis pour un SELL."""
    MF.MAX_HOLD = HOLD
    hi = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    cl = df["close"].to_numpy(np.float64)
    atr = np.maximum(df["atr_14"].to_numpy(np.float64),
                     B.ATR_PLANCHER_FRAC * cl)
    ds = (B.SPREAD_BPS / 1e4) * cl / 2.0
    se = (B.SLIP_ENTREE_BPS / 1e4) * cl
    ss = (B.SLIP_SORTIE_BPS / 1e4) * cl
    out = []
    for sens in (1, -1):
        r, _ = MF.barrieres(hi, lo, cl, atr, idx, SL_MULT, RR, sens,
                            ds, se, ss)
        out.append(r)
    return out[0], out[1]


def joue_avec_scores(env, depart, scores_b, scores_s, marge):
    """Joue un episode ou la decision vient de deux scores par barre.

    `scores_*` sont indexes comme les barres du dataset. La decision au pas
    courant lit la barre idx-1, exactement comme l'observation de la policy :
    prendre idx ferait entrer la bougie en cours, donc du futur.
    """
    _, _ = T.reset_au_depart(env, depart)
    fini = False
    while not fini:
        i = env.idx - 1
        b, s = float(scores_b[i]), float(scores_s[i])
        if max(b, s) < marge:
            action = 2                      # abstention
        else:
            action = 0 if b >= s else 1
        env.set_risk_scale(1.0)
        _, _, fini, _, _ = env.step(action)
    return list(env.trades_pnl)


def evalue(env, departs, sb, ss, marge):
    pnls = []
    for d in departs:
        pnls += joue_avec_scores(env, d, sb, ss, marge)
    return pnls


def ecart_au_point_mort(pnls):
    """Points au-dessus du seuil de rentabilite, ou nan si trop peu de trades."""
    a = np.array(pnls, float)
    if len(a) < 20:
        return float("nan")
    g, p = a[a > 0], a[a <= 0]
    if not len(g) or not len(p):
        return float("nan")
    aw, al = float(g.mean()), abs(float(p.mean()))
    return 100.0 * len(g) / len(a) - 100.0 * al / (aw + al)


def main() -> int:
    # Un argument force une selectivite FIXEE D'AVANCE, identique sur les trois
    # folds, au lieu de la faire choisir par la validation. C'est le seul mode
    # dont le chiffre soit un vrai hors-echantillon : la validation choisit
    # systematiquement mal (mesure du 2026-09-15, -3 a -4 points sur deux
    # methodes sans rapport).
    sel_fixe = float(sys.argv[1]) if len(sys.argv) > 1 else None
    cfg = T.PPOConfig()
    df = T.load_mt5_data(cfg)
    n = len(df)
    train_len, val_len = int(n * TRAIN_FRAC), int(n * VAL_FRAC)
    test_len = int(n * TEST_FRAC)
    window = train_len + val_len + test_len

    print(f"{n:,} barres | train {train_len:,} val {val_len:,} test {test_len:,}")
    print(f"cible SL {SL_MULT}xATR  R:R {RR}  detention {HOLD} barres "
          f"(reglage de l'environnement)")
    print(f"seuil calibre sur CALIB, selectivite choisie sur VALIDATION "
          f"parmi {CANDIDATS}, test intouche")
    print()
    print(f"{'fold':>6} {'PnL':>11} {'trades':>9} {'par trade':>10} "
          f"{'WR':>7} {'BE':>7} {'ecart':>8} {'+/-':>5} {'PF':>6}")
    print("-" * 78)

    fab = B.modele_tabm()
    tous = []
    for fold in range(1, N_FOLDS + 1):
        start = (fold - 1) * test_len
        if start + window > n:
            break

        stats = T.compute_and_save_global_norm_stats(
            df.iloc[start:start + train_len], FEATURE_COLS, path=None)
        _, _, val_data, test_data = T.create_datasets_from_slices(
            df, FEATURE_COLS, start=start, train_len=train_len,
            val_len=val_len, test_len=test_len, stats=stats,
            calib_frac=cfg.calib_frac)

        # Les features normalisees avec les stats du fold, sur tout l'historique
        # utile : l'entrainement n'en lira que le train, la calibration que la
        # validation, l'execution que le test.
        X = df[FEATURE_COLS].to_numpy(np.float32)
        X = np.nan_to_num((X - stats["mean"]) / (stats["std"] + 1e-8))
        X = np.clip(X, -5.0, 5.0)

        a_tr, b_tr = start, start + train_len
        a_va, b_va = b_tr, b_tr + val_len
        a_te, b_te = b_va, b_va + test_len

        i_tr = np.arange(a_tr + 1, b_tr - HOLD - 2, PAS_TRAIN)
        rb, rs = cibles_brutes(df, i_tr)
        bon = np.isfinite(rb) & np.isfinite(rs)
        i_tr, rb, rs = i_tr[bon], rb[bon], rs[bon]

        mod_b = fab().fit(X[i_tr], rb)
        mod_s = fab().fit(X[i_tr], rs)

        # Scores sur validation et test, d'un seul coup.
        idx_all = np.arange(a_va, b_te)
        sb = np.full(len(df), -1e9, np.float64)
        ss = np.full(len(df), -1e9, np.float64)
        sb[idx_all] = mod_b.predict(X[idx_all])
        ss[idx_all] = mod_s.predict(X[idx_all])

        # TROIS FENETRES, TROIS ROLES DISTINCTS, dans l'ordre du temps :
        #   calib  -> le SEUIL de score (quantile) ;
        #   val    -> le CHOIX de la selectivite ;
        #   test   -> la mesure, qui ne decide de rien.
        n_cal = int(val_len * cfg.calib_frac)
        a_cal, b_cal = a_va, a_va + n_cal
        a_v2, b_v2 = b_cal, b_va

        env_val = T.BTCTradingEnvDiscrete(val_data, cfg)
        dep_val = T.departs_disjoints(val_data.length, cfg.lookback,
                                      cfg.episode_length)
        meilleur_cal = np.maximum(sb[a_cal:b_cal], ss[a_cal:b_cal])

        best_sel, best_ec, marges = None, -1e9, {}
        if sel_fixe is not None:
            marges[sel_fixe] = float(np.quantile(meilleur_cal, 1.0 - sel_fixe))
            best_sel, best_ec = sel_fixe, float("nan")
        for c in ([] if sel_fixe is not None else CANDIDATS):
            marges[c] = float(np.quantile(meilleur_cal, 1.0 - c))
            ec = ecart_au_point_mort(
                evalue(env_val, dep_val, sb[a_v2:b_v2], ss[a_v2:b_v2],
                       marges[c]))
            if np.isfinite(ec) and ec > best_ec:
                best_sel, best_ec = c, ec

        if best_sel is None:
            print(f"{'wf'+str(fold):>6}  aucune selectivite exploitable")
            continue

        env = T.BTCTradingEnvDiscrete(test_data, cfg)
        departs = T.departs_disjoints(test_data.length, cfg.lookback,
                                      cfg.episode_length)
        pnls = evalue(env, departs, sb[a_te:b_te], ss[a_te:b_te],
                      marges[best_sel])

        r = resume(f"wf{fold}", pnls)
        if r:
            tous += pnls
        origine = ("FIXEE d'avance" if sel_fixe is not None
                   else f"choisie sur la validation ({best_ec:+.1f} pt)")
        print(f"       selectivite {100*best_sel:.0f} % {origine}, "
              f"marge {marges[best_sel]:+.4f} R")

    print("-" * 78)
    r = resume("TOUS", tous)
    if r:
        ec, err, nn, som = r
        print(f"\n{nn} trades. Ecart au point mort {ec:+.2f} +/- {err:.2f} pt.")
        print(f"PPO sur les memes fenetres : -0.2 +/- 2.2 pt (dernier epoch), "
              f"-3.3 +/- 2.3 (checkpoint choisi sur validation).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
