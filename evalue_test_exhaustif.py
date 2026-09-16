"""Reevalue les checkpoints d'un run sur TOUTE leur fenetre de test.

POURQUOI. Le test integre a l'entrainement laisse la plus grande partie de sa
fenetre inexploree, et visite le reste de travers. Deux causes, mesurees sur
exec12 :

  1. `n_test = 5` episodes de 400 barres, soit 2 000 barres sur les 7 934 de
     la fenetre. Trois quarts du test ne sont jamais joues. Resultat : 37 a 49
     trades par fold, 132 en tout, une erreur-type de 4.1 points sur le
     winrate quand l'effet cherche en vaut 3.

  2. `reset()` tire son depart via le curriculum de volatilite, qui choisit
     parmi les 30 % de barres les moins volatiles et les 30 % les plus
     volatiles. Le milieu de la distribution n'est jamais teste. Le curriculum
     est un outil d'ENTRAINEMENT ; l'appliquer au test fait mesurer autre
     chose que ce qu'on croit mesurer.

Ce script rejoue les memes checkpoints, avec la meme regle de decision et les
memes seuils calibres sur la validation, mais sur des episodes DISJOINTS qui
couvrent la fenetre entiere, dans l'ordre, sans curriculum.

EST-CE LEGITIME ? Oui, et la distinction compte. Reutiliser la fenetre de test
pour CHOISIR quelque chose — un checkpoint, un seuil, un hyperparametre —
detruirait sa valeur. Ici rien n'est choisi : les poids sont figes, les seuils
viennent de la validation, et on se contente de mesurer le meme modele sur
plus de la meme fenetre. C'est la meme question posee avec un echantillon
suffisant, pas une seconde question.

CE QUE CA NE REPARE PAS. Les tranches de test de ce depot ont deja ete
traversees par exec2 a exec12. Elles ne sont plus vierges au sens strict, et
aucune reevaluation ne leur rendra cette qualite.

    python evalue_test_exhaustif.py [prefixe] [best|bestprofit|last]
"""

import json
import sys

import numpy as np
import torch

import checkpoints as CK
import training as T
from saint_core import EntryDecisionPolicy, FEATURE_COLS, build_policy

# Memes fractions que le main de training.py.
TRAIN_FRAC, VAL_FRAC, TEST_FRAC, N_FOLDS = 0.55, 0.15, 0.10, 3
# Plus de prefixe en dur : il pointait sur exec12, quinze runs en arriere, et
# rien ne l'aurait signale — un chemin obsolete ne leve pas d'erreur tant que
# le fichier existe, il fait seulement evaluer un modele qui n'est pas celui
# qu'on croit. `checkpoints.resoud` rend le plus recent COMPATIBLE.
PREFIXE = None
QUEL = "best"


def joue(policy, env, decision, depart, device, votant=None):
    """Joue UN episode a partir d'un depart impose. Rend les PnL des trades.

    Le depart et le reset viennent de training.reset_au_depart : cette mesure
    doit rester identique a celle que fait l'entrainement, sinon les deux
    chiffres cesseraient d'etre comparables sans que rien ne le signale.

    `votant` est le VETO de TabM, le troisieme votant. Il doit etre present ici
    des lors qu'il l'etait a l'entrainement : la politique a appris a decider
    SOUS ce filtre, ses probabilites decrivent un monde ou certaines directions
    etaient interdites. L'evaluer sans lui mesurerait une autre strategie, et
    rien ne le signalerait — les deux tournent, les deux rendent des chiffres.
    """
    etat, _ = T.reset_au_depart(env, depart)

    fini = False
    while not fini:
        x = torch.as_tensor(etat, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy(x)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pb, ps = float(probs[0]), float(probs[1])
        if votant is not None:
            permis_a, permis_v = votant.veto(env.idx - 1)
            if not permis_a:
                pb = 0.0
            if not permis_v:
                ps = 0.0
        action = decision.decide(pb, ps)
        env.set_risk_scale(1.0)
        # L'observation renvoyee par step() DOIT etre reinjectee. L'oublier ne
        # provoque aucune erreur : la politique voit simplement la meme barre
        # pendant tout l'episode, produit un score constant, et le filtre par
        # rang ne declenche presque jamais. Symptome : deux trades sur trois
        # fenetres de test completes.
        etat, _, fini, _, _ = env.step(action)
    return list(env.trades_pnl), list(env.trades_side)


def resume(nom, pnls, sides=None):
    n = len(pnls)
    if n == 0:
        print(f"{nom:>6}  aucun trade")
        return None
    a = np.array(pnls, float)
    g, p = a[a > 0], a[a <= 0]
    wr = 100.0 * len(g) / n
    aw = float(g.mean()) if len(g) else 0.0
    al = abs(float(p.mean())) if len(p) else 0.0
    be = 100.0 * al / (aw + al) if (aw + al) > 0 else float("nan")
    # Erreur-type du winrate, en points : sqrt(p(1-p)/n).
    err = 100.0 * np.sqrt((wr / 100) * (1 - wr / 100) / n)
    pf = float(g.sum() / abs(p.sum())) if len(p) and p.sum() != 0 else float("inf")
    print(f"{nom:>6} {a.sum():+10.2f}$ {n:6d} tr {a.mean():+7.2f}$/tr "
          f"{wr:6.1f}% {be:6.1f}% {wr - be:+7.1f}pt {err:5.1f} {pf:6.2f}")
    return wr - be, err, n, a.sum()


def main() -> int:
    quel = sys.argv[2] if len(sys.argv) > 2 else QUEL
    # Le prefixe se RESOUT : le plus recent dont l'observation correspond au
    # pipeline courant. `checkpoints` refuse une lignee incompatible au lieu
    # de la proposer, et dit ce que "best" coute par rapport a la moyenne.
    try:
        complet, folds_dispo = CK.resoud(
            sys.argv[1] if len(sys.argv) > 1 else PREFIXE, famille=quel)
    except FileNotFoundError as e:
        print(f"AUCUN CHECKPOINT UTILISABLE : {e}")
        return 1
    prefixe = complet[len(quel) + 1:]

    cfg_base = T.PPOConfig()
    df = T.load_mt5_data(cfg_base)
    n = len(df)
    train_len = int(n * TRAIN_FRAC)
    val_len = int(n * VAL_FRAC)
    test_len = int(n * TEST_FRAC)
    window = train_len + val_len + test_len
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{n:,} barres | train {train_len:,} val {val_len:,} test {test_len:,}")
    print(f"checkpoints : {complet}_wf*  folds {folds_dispo}  |  episodes DISJOINTS")
    print()
    print(f"{'fold':>6} {'PnL':>11} {'trades':>9} {'par trade':>10} "
          f"{'WR':>7} {'BE':>7} {'ecart':>8} {'+/-':>5} {'PF':>6}")
    print("-" * 78)

    tous, ecarts = [], []
    for fold in range(1, N_FOLDS + 1):
        start = (fold - 1) * test_len
        if start + window > n:
            break
        base = f"{quel}_{prefixe}_wf{fold}_both_wf{fold}"
        try:
            etat = torch.load(base + ".pth", map_location=device,
                              weights_only=True)
            calib = json.load(open(base + "_calib.json", encoding="utf-8"))
        except FileNotFoundError as e:
            print(f"{'wf'+str(fold):>6}  fichier manquant : {e.filename}")
            continue

        # Les stats de normalisation se recalculent sur le train du fold,
        # exactement comme au moment de l'entrainement.
        stats = T.compute_and_save_global_norm_stats(
            df.iloc[start:start + train_len], FEATURE_COLS, path=None)
        _, _, _, test_data = T.create_datasets_from_slices(
            df, FEATURE_COLS, start=start, train_len=train_len,
            val_len=val_len, test_len=test_len, stats=stats,
            calib_frac=cfg_base.calib_frac)

        policy = build_policy(device, lookback=cfg_base.lookback,
                              state_dict=etat)
        policy.load_state_dict(etat, strict=True)
        policy.eval()

        cfg = T.PPOConfig(**cfg_base.__dict__)
        # LE MEME TROISIEME VOTANT QU'A L'ENTRAINEMENT. Ajuste sur la
        # fenetre d'entrainement de CE fold, applique a la fenetre evaluee :
        # aucun recouvrement. L'omettre evaluerait une politique qui a appris
        # sous un filtre, sans ce filtre.
        votant = None
        if T.PPOConfig().votant_tabm:
            import banc_rendement_net as _B
            import evalue_tabm_test as _E
            import tabm_votant as _TV
            a_ev = start + train_len + val_len if True else start + train_len
            b_ev = a_ev + (test_len if True else val_len)
            v_glob = _TV.hors_echantillon(
                df, start, start + train_len, a_ev, b_ev, _B.modele_tabm(),
                _E.cibles_brutes, FEATURE_COLS, stats, pas_train=_E.PAS_TRAIN)
            # Les indices de l'environnement partent de 0 sur SA fenetre : on
            # decale le votant pour que `veto(env.idx - 1)` tombe juste.
            votant = _TV.Votant(b_ev - a_ev, 0, b_ev - a_ev)
            votant.achat = v_glob.achat[a_ev:b_ev]
            votant.vente = v_glob.vente[a_ev:b_ev]
            votant.seuil = v_glob.seuil
        env = T.BTCTradingEnvDiscrete(test_data, cfg)
        departs = T.departs_disjoints(test_data.length, cfg.lookback,
                                      cfg.episode_length)

        pnls, sides = [], []
        for d in departs:
            decision = EntryDecisionPolicy(calib["decision_policy"])
            p, s = joue(policy, env, decision, d, device, votant)
            pnls += p
            sides += s
        r = resume(f"wf{fold}", pnls)
        if r:
            ecarts.append(r[:2])
            tous += pnls
        print(f"       {len(departs)} episodes disjoints x "
              f"{cfg.episode_length} barres = "
              f"{len(departs)*cfg.episode_length:,} barres couvertes "
              f"({100*len(departs)*cfg.episode_length/test_data.length:.0f} % "
              f"de la fenetre)")

    print("-" * 78)
    r = resume("TOUS", tous)
    if r:
        ec, err, nn, som = r
        print(f"\n{nn} trades au total. Ecart au point mort {ec:+.2f} "
              f"+/- {err:.2f} points.")
        print(f"Rappel du test integre a l'entrainement : 132 trades, "
              f"-2.67 +/- 4.09 points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
