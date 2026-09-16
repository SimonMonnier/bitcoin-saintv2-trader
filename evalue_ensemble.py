"""Trois modeles votent : TabM, PatchTST et SAINT, dans le meme moteur.

POURQUOI CE FICHIER EXISTE. C'est le seul levier non exploite de ce depot qui
soit adosse a une mesure. Sur les memes fenetres de test, TabM rend +3.8 points
quand LightGBM en rend +2.6 et Ridge +1.8 — et une partie de cet ecart vient de
ce que TabM est un ENSEMBLE, la ou PPO n'entraine qu'un seul reseau.

Trois modeles qui se trompent differemment :

    TabM       supervise, regression du rendement net, MLP ensemble
    PatchTST   PPO, canaux INDEPENDANTS : ne croise jamais les colonnes
    SAINT      PPO, attention sur l'axe des features : ne fait que les croiser

Les deux derniers sont opposes par construction sur la question qui separe le
mieux les modeles de ce depot. Leurs erreurs ont donc peu de raisons d'etre
correlees, ce qui est la condition pour qu'un vote reduise la variance.

CE QUI EST COMPARE, ET POURQUOI C'EST APPARIE. Un seul environnement, une seule
trajectoire, les trois modeles interroges a CHAQUE barre avec la MEME
observation. Chacun applique sa propre regle de decision calibree, puis on
combine. Les chiffres des trois modeles seuls sont produits dans la meme passe,
donc ils portent sur exactement les memes barres — aucune difference de
periode ne vient s'ajouter a la difference de methode.

DEUX FACONS DE COMBINER, et elles ne disent pas la meme chose :

    majorite   au moins deux modeles sur le meme sens, aucun sur l'autre.
    unanime    les trois. Plus selectif, donc moins de trades.

L'unanimite est ce que decrit le livre Ichimoku — un signal n'est pris que si
rien ne le contredit, et Chikou-Span sert precisement a invalider. La majorite
est le compromis habituel.

CE QUE CE FICHIER NE FAIT PAS. Il ne choisit pas entre les deux regles sur le
test : les deux sont rapportees, et le choix — s'il y en a un — devra se faire
ailleurs. Regler quoi que ce soit sur cette fenetre a deja coute trois a quatre
points cette nuit, deux fois, sur deux methodes sans rapport.

    python evalue_ensemble.py [selectivite]
"""

import json
import sys

import numpy as np
import torch

import banc_rendement_net as B
import evalue_tabm_test as E
import training as T
from evalue_test_exhaustif import resume
from saint_core import EntryDecisionPolicy, FEATURE_COLS, build_policy

SELECTIVITE = 0.05
# (nom, prefixe du checkpoint). "last" est le modele MOYENNE : le training
# ecrit ce fichier apres le tour de moyenne des poids.
RESEAUX = [("patchtst", "last_saintv2_loup_duel_exec18"),
           ("saint", "last_saintv2_loup_duel_exec20")]


def charge(prefixe, fold, lookback, device):
    """Rend (policy, spec de decision) ou None si le fold n'existe pas."""
    base = f"{prefixe}_wf{fold}_both_wf{fold}"
    try:
        etat = torch.load(base + ".pth", map_location=device, weights_only=True)
        calib = json.load(open(base + "_calib.json", encoding="utf-8"))
    except FileNotFoundError:
        return None
    # `heads` vient du fichier quand il y est ; sinon build_policy le deduit de
    # la QK-Norm, qui normalise chaque tete separement et dont la taille EST la
    # dimension par tete.
    p = build_policy(device, lookback=lookback, state_dict=etat,
                     heads=int(calib.get("saint_heads", 0)))
    p.load_state_dict(etat, strict=True)
    p.eval()
    return p, calib["decision_policy"]


def joue(env, depart, reseaux, tabm, marge_tabm, device, regle):
    """Un episode. Rend les PnL, et les votes de chacun pour comptage.

    Les trois modeles voient la MEME observation a chaque pas. Leurs regles de
    decision sont interrogees a CHAQUE barre, meme quand l'ensemble s'abstient :
    le filtre par rang glissant a besoin du flux complet des scores pour que son
    quantile ait un sens, et le priver des barres non tradees le decalerait.
    """
    etat, _ = T.reset_au_depart(env, depart)
    decisions = [EntryDecisionPolicy(spec) for _, spec in reseaux]
    fini = False
    votes = []
    while not fini:
        i = env.idx - 1
        x = torch.as_tensor(etat, dtype=torch.float32, device=device).unsqueeze(0)
        actes = []
        with torch.no_grad():
            for (pol, _), dec in zip(reseaux, decisions):
                pr = torch.softmax(pol(x)[0], dim=-1)[0].cpu().numpy()
                actes.append(dec.decide(float(pr[0]), float(pr[1])))
        # TabM : ses scores ne dependent pas de l'etat de l'environnement,
        # ils sont precalcules pour toute la fenetre.
        b, s = float(tabm[0][i]), float(tabm[1][i])
        actes.append(2 if max(b, s) < marge_tabm else (0 if b >= s else 1))

        n_buy = sum(1 for a in actes if a == 0)
        n_sell = sum(1 for a in actes if a == 1)
        if regle == "unanime":
            action = 0 if n_buy == len(actes) else (1 if n_sell == len(actes) else 2)
        else:
            # Majorite SANS opposition : deux voix pour un sens et aucune pour
            # l'autre. Un modele qui vote contre suffit a annuler, ce qui est la
            # logique du filtre de l'Ichimoku — on ne prend pas un signal
            # qu'autre chose contredit.
            action = (0 if n_buy >= 2 and n_sell == 0
                      else (1 if n_sell >= 2 and n_buy == 0 else 2))
        votes.append(actes)
        env.set_risk_scale(1.0)
        etat, _, fini, _, _ = env.step(action)
    return list(env.trades_pnl), votes


def main() -> int:
    sel = float(sys.argv[1]) if len(sys.argv) > 1 else SELECTIVITE
    cfg = T.PPOConfig()
    df = T.load_mt5_data(cfg)
    n = len(df)
    train_len, val_len = int(n * E.TRAIN_FRAC), int(n * E.VAL_FRAC)
    test_len = int(n * E.TEST_FRAC)
    window = train_len + val_len + test_len
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{n:,} barres | selectivite TabM {100*sel:.0f} % fixee d'avance")
    print(f"reseaux : {', '.join(nom for nom, _ in RESEAUX)} (modeles moyennes)")
    print()

    fab = B.modele_tabm()
    resultats = {"majorite": [], "unanime": []}
    accord = {"majorite": [], "unanime": []}

    for fold in range(1, E.N_FOLDS + 1):
        start = (fold - 1) * test_len
        if start + window > n:
            break

        reseaux = [charge(p, fold, cfg.lookback, device) for _, p in RESEAUX]
        if any(r is None for r in reseaux):
            manquants = [nom for (nom, _), r in zip(RESEAUX, reseaux) if r is None]
            print(f"wf{fold} : checkpoints absents pour {manquants} — fold saute")
            continue

        stats = T.compute_and_save_global_norm_stats(
            df.iloc[start:start + train_len], FEATURE_COLS, path=None)
        _, _, _, test_data = T.create_datasets_from_slices(
            df, FEATURE_COLS, start=start, train_len=train_len,
            val_len=val_len, test_len=test_len, stats=stats,
            calib_frac=cfg.calib_frac)

        X = df[FEATURE_COLS].to_numpy(np.float32)
        X = np.clip(np.nan_to_num((X - stats["mean"]) / (stats["std"] + 1e-8)),
                    -5.0, 5.0)
        a_tr, b_tr = start, start + train_len
        a_va = b_tr
        a_te, b_te = b_tr + val_len, b_tr + val_len + test_len
        n_cal = int(val_len * cfg.calib_frac)

        i_tr = np.arange(a_tr + 1, b_tr - E.HOLD - 2, E.PAS_TRAIN)
        rb, rs = E.cibles_brutes(df, i_tr)
        bon = np.isfinite(rb) & np.isfinite(rs)
        i_tr, rb, rs = i_tr[bon], rb[bon], rs[bon]
        mod_b = fab().fit(X[i_tr], rb)
        mod_s = fab().fit(X[i_tr], rs)
        idx = np.arange(a_va, b_te)
        sb = np.full(n, -1e9)
        ss = np.full(n, -1e9)
        sb[idx] = mod_b.predict(X[idx])
        ss[idx] = mod_s.predict(X[idx])
        marge = float(np.quantile(
            np.maximum(sb[a_va:a_va + n_cal], ss[a_va:a_va + n_cal]), 1.0 - sel))

        departs = T.departs_disjoints(test_data.length, cfg.lookback,
                                      cfg.episode_length)
        for regle in ("majorite", "unanime"):
            env = T.BTCTradingEnvDiscrete(test_data, cfg)
            pnls, votes = [], []
            for d in departs:
                p, v = joue(env, d, reseaux, (sb[a_te:b_te], ss[a_te:b_te]),
                            marge, device, regle)
                pnls += p
                votes += v
            resultats[regle] += pnls
            v = np.array(votes)
            # Part des barres ou au moins deux modeles proposent le meme sens.
            agit = ((v == 0).sum(1) >= 2) | ((v == 1).sum(1) >= 2)
            accord[regle].append(float(agit.mean()))
            r = resume(f"wf{fold} {regle[:4]}", pnls)

    print("-" * 78)
    for regle in ("majorite", "unanime"):
        r = resume(regle.upper(), resultats[regle])
        if r and accord[regle]:
            print(f"       accord d'au moins deux modeles sur "
                  f"{100*np.mean(accord[regle]):.1f} % des barres")
    print()
    print("Temoins sur les MEMES fenetres, selectivite 5 % fixee d'avance :")
    print("  TabM seul      +3.8 +/- 2.3 pt  PF 1.18  432 trades  3/3 folds")
    print("  PatchTST seul  +2.2 +/- 2.2 pt  PF 1.10  490 trades  2/3 folds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
