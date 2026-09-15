"""Meta-etiquetage : un modele decide le SENS, un second decide s'il faut y aller.

L'IDEE (Lopez de Prado). Un modele primaire produit un signal directionnel. Un
second modele, entraine UNIQUEMENT sur la question "ce signal a-t-il gagne ?",
apprend quand le premier a raison. Le primaire ne sait pas s'abstenir ; le
secondaire ne sait pas choisir un sens. Separer les deux laisse chacun
apprendre une question plus simple que la conjonction.

POURQUOI CETTE STRUCTURE ICI PLUTOT QU'UNE AUTRE. Elle n'est pas importee de
force : c'est exactement ce que decrit le livre Ichimoku. Chikou-Span ne
choisit jamais un sens — "lorsque l'on detecte un potentiel signal de trading
delivre par les prix, il est necessaire de verifier si au meme moment
Chikou-Span n'est pas confrontee a un obstacle. Si elle est libre, elle VALIDE
le signal des prix. Dans le cas contraire il faut s'abstenir." Le filtre du
systeme est litteralement un meta-label.

CE QU'ON AVAIT DEJA, ET EN QUOI C'EST DIFFERENT. La marge d'abstention actuelle
est un SEUIL sur le score du primaire : on prend les 5 % ou il est le plus
confiant. C'est une version degradee du meta-etiquetage — le meme modele sert a
choisir le sens et a juger de sa propre fiabilite, avec la meme information.
Ici le second modele a ses propres features (dont la conviction du premier) et
une cible differente : gagner ou perdre, pas combien.

LE POINT QUI DECIDE DE TOUT : LES PREDICTIONS DOIVENT ETRE HORS ECHANTILLON.
Si l'on entraine le meta sur les predictions que le primaire fait sur ses
PROPRES donnees d'entrainement, il apprend a faire confiance a un primaire
artificiellement bon — celui qui a memorise. En production le primaire est
moins sur, le meta le juge trop severement, et le filtre se comporte a
l'inverse de ce qu'on a mesure. On produit donc les predictions du primaire
par chainage avant : trois blocs dans l'ordre du temps, chacun predit par un
primaire entraine uniquement sur ce qui le precede.

PROTOCOLE, identique a celui d'evalue_tabm_test pour que les deux tableaux se
lisent ligne a ligne : memes fenetres, meme moteur, meme friction, meme
couverture exhaustive, selectivite finale de 5 % fixee d'avance, seuils
calibres sur CALIB et jamais sur le test.

    python evalue_meta_etiquetage.py [selectivite]
"""

import sys

import numpy as np

import banc_rendement_net as B
import evalue_tabm_test as E
import training as T
from evalue_test_exhaustif import resume
from saint_core import FEATURE_COLS

SELECTIVITE = 0.05
# Part des occasions que le primaire propose au secondaire. Large a dessein :
# le role du primaire est de dire "il se passe quelque chose et dans quel
# sens", celui du secondaire de trancher. Un candidat trop etroit priverait le
# meta de tout ce sur quoi il doit apprendre a dire non.
PART_CANDIDATS = 0.25
N_CHAINE = 3


def primaire(fab, X, i_tr, rb, rs):
    """Entraine les deux regressions du primaire et rend les deux predicteurs."""
    return (fab().fit(X[i_tr], rb), fab().fit(X[i_tr], rs))


def predictions_chainees(fab, X, df, idx, n_blocs=N_CHAINE):
    """Predictions du primaire HORS ECHANTILLON, par chainage avant.

    Le bloc b est predit par un primaire entraine sur les blocs 0..b-1. Le
    premier bloc n'a rien avant lui : il ne produit pas de prediction et ne
    servira donc pas a entrainer le meta. C'est le prix de l'honnetete —
    un tiers des exemples en moins, mais aucun n'est contamine.
    """
    bornes = np.linspace(0, len(idx), n_blocs + 1).astype(int)
    pb = np.full(len(idx), np.nan)
    ps = np.full(len(idx), np.nan)
    for b in range(1, n_blocs):
        i_ap = idx[:bornes[b]]
        i_ce = idx[bornes[b]:bornes[b + 1]]
        if len(i_ap) < 200 or len(i_ce) < 50:
            continue
        rb, rs = E.cibles_brutes(df, i_ap)
        bon = np.isfinite(rb) & np.isfinite(rs)
        mb, ms = primaire(fab, X, i_ap[bon], rb[bon], rs[bon])
        pb[bornes[b]:bornes[b + 1]] = mb.predict(X[i_ce])
        ps[bornes[b]:bornes[b + 1]] = ms.predict(X[i_ce])
    return pb, ps


def features_meta(X, idx, pb, ps):
    """Features du secondaire : celles du primaire PLUS ce que le primaire pense.

    Les quatre colonnes ajoutees sont ce qui distingue le meta d'un simple
    second modele : la conviction du premier, son ecart entre les deux sens, et
    le sens retenu. Sans elles, le meta ne jugerait pas un signal, il en
    produirait un autre.
    """
    meilleur = np.maximum(pb, ps)
    ecart = np.abs(pb - ps)
    sens = np.where(pb >= ps, 1.0, -1.0)
    sup = np.column_stack([pb, ps, meilleur, ecart, sens]).astype(np.float32)
    return np.hstack([X[idx], sup])


def modele_meta():
    """LightGBM en classification : gagner ou perdre.

    Choix assume. TabM gagne sur la REGRESSION du rendement, mais le meta
    apprend une binaire sur quelques centaines a quelques milliers d'exemples,
    ou le gradient boosting est chez lui et n'a aucun hyperparametre a regler
    sur une fenetre qu'on n'a pas le droit de regarder.
    """
    import lightgbm as lgb
    return lambda: lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=15,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=2)


def main() -> int:
    sel = float(sys.argv[1]) if len(sys.argv) > 1 else SELECTIVITE
    cfg = T.PPOConfig()
    df = T.load_mt5_data(cfg)
    n = len(df)
    train_len, val_len = int(n * E.TRAIN_FRAC), int(n * E.VAL_FRAC)
    test_len = int(n * E.TEST_FRAC)
    window = train_len + val_len + test_len

    print(f"{n:,} barres | primaire TabM (sens et amplitude) "
          f"+ meta LightGBM (prendre ou passer)")
    print(f"candidats {100*PART_CANDIDATS:.0f} % -> selectivite finale "
          f"{100*sel:.0f} %, fixee d'avance")
    print(f"predictions du primaire hors echantillon, chainage avant "
          f"{N_CHAINE} blocs")
    print()
    print(f"{'fold':>6} {'PnL':>11} {'trades':>9} {'par trade':>10} "
          f"{'WR':>7} {'BE':>7} {'ecart':>8} {'+/-':>5} {'PF':>6}")
    print("-" * 78)

    fab = B.modele_tabm()
    fab_meta = modele_meta()
    tous = []
    for fold in range(1, E.N_FOLDS + 1):
        start = (fold - 1) * test_len
        if start + window > n:
            break

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
        a_va, b_va = b_tr, b_tr + val_len
        a_te, b_te = b_va, b_va + test_len
        n_cal = int(val_len * cfg.calib_frac)

        i_tr = np.arange(a_tr + 1, b_tr - E.HOLD - 2, E.PAS_TRAIN)
        rb_tr, rs_tr = E.cibles_brutes(df, i_tr)
        bon = np.isfinite(rb_tr) & np.isfinite(rs_tr)
        i_tr, rb_tr, rs_tr = i_tr[bon], rb_tr[bon], rs_tr[bon]

        # ---- 1. Primaire hors echantillon, pour entrainer le meta ----
        pb_h, ps_h = predictions_chainees(fab, X, df, i_tr)
        dispo = np.isfinite(pb_h) & np.isfinite(ps_h)
        if dispo.sum() < 300:
            print(f"{'wf'+str(fold):>6}  trop peu de predictions chainees")
            continue
        i_m = i_tr[dispo]
        pb_m, ps_m = pb_h[dispo], ps_h[dispo]
        meilleur_m = np.maximum(pb_m, ps_m)
        seuil_cand = float(np.quantile(meilleur_m, 1.0 - PART_CANDIDATS))
        cand = meilleur_m >= seuil_cand

        # Cible du meta : le trade que le primaire AURAIT pris a-t-il gagne ?
        gain_m = np.where(pb_m >= ps_m, rb_tr[dispo], rs_tr[dispo])
        y_meta = (gain_m[cand] > 0).astype(int)
        Xm = features_meta(X, i_m[cand], pb_m[cand], ps_m[cand])
        if len(np.unique(y_meta)) < 2 or len(y_meta) < 150:
            print(f"{'wf'+str(fold):>6}  cible meta degeneree")
            continue
        meta = fab_meta().fit(Xm, y_meta)

        # ---- 2. Primaire final, entraine sur tout le train ----
        mb, ms = primaire(fab, X, i_tr, rb_tr, rs_tr)
        idx = np.arange(a_va, b_te)
        pb = np.full(n, -1e9)
        ps = np.full(n, -1e9)
        pb[idx] = mb.predict(X[idx])
        ps[idx] = ms.predict(X[idx])

        # ---- 3. Score final = probabilite du meta, sur les candidats seuls ----
        # Hors candidats, le primaire ne propose rien : le score reste au
        # plancher pour que ces barres ne puissent jamais etre retenues.
        prob = np.full(n, -1e9)
        meilleur = np.maximum(pb[idx], ps[idx])
        est_cand = meilleur >= seuil_cand
        if est_cand.any():
            Xt = features_meta(X, idx[est_cand], pb[idx][est_cand],
                               ps[idx][est_cand])
            prob[idx[est_cand]] = meta.predict_proba(Xt)[:, 1]

        # Le SENS reste celui du primaire ; le meta ne decide que d'y aller.
        sb = np.where(pb >= ps, prob, -1e9)
        ss = np.where(ps > pb, prob, -1e9)

        # ---- 4. Seuil calibre sur CALIB pour la selectivite voulue ----
        bloc_cal = np.maximum(sb[a_va:a_va + n_cal], ss[a_va:a_va + n_cal])
        retenus = bloc_cal[bloc_cal > -1e8]
        if len(retenus) < 30:
            print(f"{'wf'+str(fold):>6}  calibration vide")
            continue
        # La selectivite se compte sur TOUTES les barres, pas sur les seuls
        # candidats : sinon le meta prendrait 5 % de 25 %, soit quatre fois
        # moins de trades que le temoin, et la comparaison ne porterait plus
        # sur la qualite du filtre mais sur son debit.
        part = min(1.0, sel * n_cal / max(len(retenus), 1))
        marge = float(np.quantile(retenus, 1.0 - part))

        env = T.BTCTradingEnvDiscrete(test_data, cfg)
        departs = T.departs_disjoints(test_data.length, cfg.lookback,
                                      cfg.episode_length)
        pnls = E.evalue(env, departs, sb[a_te:b_te], ss[a_te:b_te], marge)
        r = resume(f"wf{fold}", pnls)
        if r:
            tous += pnls
        print(f"       candidats {100*PART_CANDIDATS:.0f} %, "
              f"{len(y_meta)} exemples meta ({100*y_meta.mean():.0f} % gagnants), "
              f"seuil proba {marge:.4f}")

    print("-" * 78)
    r = resume("TOUS", tous)
    if r:
        ec, err, nn, som = r
        print(f"\n{nn} trades. Ecart au point mort {ec:+.2f} +/- {err:.2f} pt.")
        print("Temoin, meme protocole sans meta : TabM seul +3.80 +/- 2.34 pt, "
              "PF 1.18, 432 trades.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
