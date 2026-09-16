"""Les reglages des trois votants, mesures au lieu d'etre choisis.

CE QUI ETAIT ARBITRAIRE. `PART_LAISSEE = 0.50` decide a quel point le veto de
TabM mord — donc le poids du troisieme votant dans toute la decision — et il
avait ete pose sans mesure. `n_membres` et `d_cache` sont les defauts hérites
du regime H1. La geometrie de PatchTST a ete recopiee du bloc SAINT.

CE QUE CE FICHIER MESURE, ET SUR QUOI. Uniquement sur la **validation**. La
fenetre de test ne sert qu'une fois, et un balayage dessus serait exactement le
biais qui a deja coute trois a quatre points a ce depot, deux fois.

    part du veto     TabM est ajuste sur le train, note la validation, et on
                     fait varier le SEUIL sans rien reajuster. Le critere est
                     l'esperance de rendement des directions PERMISES, et sa
                     detectabilite E[R] x racine(N) — un filtre qui ameliore
                     E[R] en ne laissant que trente trades n'ameliore rien.

    TabM             n_membres et d_cache, a seuil fixe, sur le meme decoupage.

CE QU'IL NE MESURE PAS. La capacite des deux reseaux PPO : elle ne se lit pas
sur une sonde supervisee, il faut un run. Le fichier rapporte seulement le
budget, pour que le choix soit fait les yeux ouverts.

    python mesure_votants.py [selectivite|phases|part|tabm|budget]
"""

from __future__ import annotations

import sys

import numpy as np

PARTS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00)
GEOMETRIES_TABM = ((4, 128), (8, 128), (8, 256), (16, 256), (8, 512))


def _fenetres():
    import evalue_tabm_test as E
    import training as T
    from saint_core import FEATURE_COLS
    cfg = T.PPOConfig()
    df = T.load_mt5_data(cfg)
    n = len(df)
    tr = int(n * E.TRAIN_FRAC)
    va = int(n * E.VAL_FRAC)
    stats = T.compute_and_save_global_norm_stats(
        df.iloc[:tr], FEATURE_COLS, path=None)
    return df, tr, va, stats, FEATURE_COLS, E


def _scores(df, tr, va, stats, cols, E, n_membres=8, d_cache=256):
    """Ajuste TabM sur le train, rend ses scores et les cibles de validation."""
    import banc_rendement_net as B
    fab = B.modele_tabm(n_membres=n_membres, d_cache=d_cache)
    X = df[cols].to_numpy(np.float32)
    X = np.clip(np.nan_to_num((X - stats["mean"]) / (stats["std"] + 1e-8)),
                -5.0, 5.0)

    i_tr = np.arange(0, tr - E.HOLD - 2, E.PAS_TRAIN)
    ra, rv = E.cibles_brutes(df, i_tr)
    bon = np.isfinite(ra) & np.isfinite(rv)
    i_tr, ra, rv = i_tr[bon], ra[bon], rv[bon]
    m_a = fab().fit(X[i_tr], ra)
    m_v = fab().fit(X[i_tr], rv)

    # VALIDATION : entrees NON CHEVAUCHANTES, espacees du plafond de detention.
    # Deux entrees plus proches partagent leurs barres de resultat, et le meme
    # mouvement serait compte deux fois — le biais qui avait fait lire 58 % de
    # reussite la ou il y en avait 39.7.
    i_va = np.arange(tr, tr + va - E.HOLD - 2, E.HOLD)
    ya, yv = E.cibles_brutes(df, i_va)
    bon = np.isfinite(ya) & np.isfinite(yv)
    i_va, ya, yv = i_va[bon], ya[bon], yv[bon]

    sa_tr = np.concatenate([m_a.predict(X[i_tr]), m_v.predict(X[i_tr])])
    return (m_a.predict(X[i_va]), m_v.predict(X[i_va]), ya, yv, sa_tr,
            len(i_va))


SELECTIVITES = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00)


def mesure_selectivite(df, tr, va, stats, cols, E, n_phases: int = 12) -> None:
    """L'avantage est-il CONCENTRE au sommet du classement, ou ETALE ?

    CE QUE CETTE COURBE DECIDE. On ne retient que les 5 % de decisions les plus
    confiantes. Le bruit de mesure vient entierement du nombre de trades :
    a ~250 par epoch et par fold, l'erreur-type du winrate vaut 3.05 points,
    et il en faut 1.4 a 2.7 pour voir l'avantage espere. Il manque donc un
    facteur trois a six en TRADES.

    Monter la selectivite de 5 a 20 % les multiplie par quatre — gratuitement,
    SI l'avantage survit a la dilution. Deux formes possibles :

        concentre   E[R] s'effondre des qu'on descend le classement : la
                    selectivite basse est justifiee, et il faut trouver les
                    trades ailleurs.
        etale       E[R] tient jusqu'a 20 ou 30 % : quatre fois plus de trades
                    pour le meme avantage, et l'experience devient decisive.

    LE CRITERE EST LA DETECTABILITE, E[R] x racine(N), pas E[R] seul. Un
    filtre qui double l'esperance en divisant les trades par dix fait reculer
    ce qu'on peut prouver.

    Chaque direction a son propre seuil, comme dans l'environnement qui calibre
    `valB` et `valS` separement. Et tout est moyenne sur les phases : une
    grille d'entrees unique ne vaut rien, meme correctement non chevauchante.
    """
    import banc_rendement_net as B
    fab = B.modele_tabm()
    X = df[cols].to_numpy(np.float32)
    X = np.clip(np.nan_to_num((X - stats["mean"]) / (stats["std"] + 1e-8)),
                -5.0, 5.0)
    i_tr = np.arange(0, tr - E.HOLD - 2, E.PAS_TRAIN)
    ra, rv = E.cibles_brutes(df, i_tr)
    bon = np.isfinite(ra) & np.isfinite(rv)
    i_tr, ra, rv = i_tr[bon], ra[bon], rv[bon]
    m_a = fab().fit(X[i_tr], ra)
    m_v = fab().fit(X[i_tr], rv)
    # Les seuils viennent de la fenetre D'APPRENTISSAGE, jamais de celle qu'on
    # mesure : les calibrer sur la validation reviendrait a choisir le filtre
    # d'apres ce qu'il filtre.
    sa_tr, sv_tr = m_a.predict(X[i_tr]), m_v.predict(X[i_tr])

    pas_phase = max(1, E.HOLD // n_phases)
    res = {q: {"er": [], "n": []} for q in SELECTIVITES}
    for ph in range(n_phases):
        i = np.arange(tr + ph * pas_phase, tr + va - E.HOLD - 2, E.HOLD)
        if len(i) < 20:
            continue
        ya, yv = E.cibles_brutes(df, i)
        bon = np.isfinite(ya) & np.isfinite(yv)
        i, ya, yv = i[bon], ya[bon], yv[bon]
        if len(i) < 20:
            continue
        sa, sv = m_a.predict(X[i]), m_v.predict(X[i])
        for q in SELECTIVITES:
            ta = float(np.quantile(sa_tr, 1.0 - q))
            tv = float(np.quantile(sv_tr, 1.0 - q))
            pris = np.concatenate([ya[sa >= ta], yv[sv >= tv]])
            if len(pris) < 5:
                continue
            res[q]["er"].append(float(pris.mean()))
            res[q]["n"].append(len(pris))

    # Un point de winrate vaut (R:R + 1) / 100 en R : on convertit pour parler
    # la meme langue que la veille et le point mort.
    par_point = (E.RR + 1) / 100.0
    print(f"{n_phases} phases | 1 point de winrate = {par_point:.4f} R\n")
    print(f"{'select.':>8} {'trades/ph':>10} {'E[R]':>9} {'err-type':>9} "
          f"{'en points':>10} {'detectabilite':>14}")
    print("-" * 66)
    for q in SELECTIVITES:
        er = np.array(res[q]["er"])
        if len(er) < 3:
            print(f"{q:>8.0%} {'trop peu de phases':>10}")
            continue
        n = float(np.mean(res[q]["n"]))
        m = er.mean()
        err = er.std(ddof=1) / np.sqrt(len(er))
        print(f"{q:>8.0%} {n:>10.0f} {m:>+9.4f} {err:>9.4f} "
              f"{m/par_point:>+9.2f}p {m*np.sqrt(n):>+14.3f}")
    print()
    print("Une selectivite utile releve E[R] SANS effondrer le compte.")
    print("La derniere colonne tranche : c'est ce qu'on pourra prouver.")


def mesure_part_phases(df, tr, va, stats, cols, E, n_phases: int = 12) -> None:
    """Le balayage du veto, moyenne sur les PHASES et compare a phase egale.

    POURQUOI LES PHASES. Des entrees espacees du plafond de detention ne se
    recouvrent pas, mais il y a `plafond` facons de poser la grille, et elles
    ne donnent pas le meme resultat. Ce depot a mesure l'ampleur du piege le
    15 septembre : sur huit phases, E[R] allait de +0.158 a -0.071 pour le meme
    jeu, le meme protocole et les memes donnees — et la phase 0, celle qui
    avait servi a tout, etait la plus favorable des huit.

    On compare donc CHAQUE phase a elle-meme, avec et sans veto, et on lit la
    moyenne des ecarts APPARIES. C'est la seule facon d'empecher le choix de la
    grille de decider a la place de la mesure.
    """
    import banc_rendement_net as B
    fab = B.modele_tabm()
    X = df[cols].to_numpy(np.float32)
    X = np.clip(np.nan_to_num((X - stats["mean"]) / (stats["std"] + 1e-8)),
                -5.0, 5.0)
    i_tr = np.arange(0, tr - E.HOLD - 2, E.PAS_TRAIN)
    ra, rv = E.cibles_brutes(df, i_tr)
    bon = np.isfinite(ra) & np.isfinite(rv)
    i_tr, ra, rv = i_tr[bon], ra[bon], rv[bon]
    m_a = fab().fit(X[i_tr], ra)
    m_v = fab().fit(X[i_tr], rv)
    scores_tr = np.concatenate([m_a.predict(X[i_tr]), m_v.predict(X[i_tr])])

    pas_phase = max(1, E.HOLD // n_phases)
    ecarts = {p: [] for p in PARTS}
    n_permis = {p: [] for p in PARTS}
    ref_phases = []
    for ph in range(n_phases):
        i = np.arange(tr + ph * pas_phase, tr + va - E.HOLD - 2, E.HOLD)
        if len(i) < 20:
            continue
        ya, yv = E.cibles_brutes(df, i)
        bon = np.isfinite(ya) & np.isfinite(yv)
        i, ya, yv = i[bon], ya[bon], yv[bon]
        if len(i) < 20:
            continue
        sa, sv = m_a.predict(X[i]), m_v.predict(X[i])
        ref = float(np.concatenate([ya, yv]).mean())
        ref_phases.append(ref)
        for part in PARTS:
            seuil = float(np.quantile(scores_tr, 1.0 - part))
            permis = np.concatenate([ya[sa >= seuil], yv[sv >= seuil]])
            if len(permis) < 20:
                continue
            ecarts[part].append(float(permis.mean()) - ref)
            n_permis[part].append(len(permis))

    print(f"{n_phases} phases espacees de {pas_phase} barres, "
          f"chacune internement sans chevauchement")
    print(f"reference sans veto : E[R] {np.mean(ref_phases):+.4f} "
          f"+/- {np.std(ref_phases):.4f} selon la phase")
    print()
    print(f"{'part':>6} {'phases':>7} {'trades/ph':>10} "
          f"{'ecart au temoin':>16} {'err-type':>9} {'t':>7}")
    print("-" * 62)
    for part in PARTS:
        e = np.array(ecarts[part])
        if len(e) < 3:
            print(f"{part:>6.2f} {len(e):>7}  trop peu de phases exploitables")
            continue
        err = e.std(ddof=1) / np.sqrt(len(e))
        print(f"{part:>6.2f} {len(e):>7} {np.mean(n_permis[part]):>10.0f} "
              f"{e.mean():>+16.4f} {err:>9.4f} {e.mean()/max(err,1e-9):>7.2f}")
    print()
    print("L'ecart est APPARIE : chaque phase se compare a elle-meme sans")
    print("veto. Un ecart positif veut dire que le veto ajoute quelque chose.")


def mesure_part(df, tr, va, stats, cols, E) -> None:
    sa, sv, ya, yv, scores_tr, n = _scores(df, tr, va, stats, cols, E)
    print(f"{n:,} entrees de validation NON chevauchantes "
          f"(espacees de {E.HOLD} barres)\n")
    print(f"{'part':>6} {'seuil':>9} {'permises':>9} {'E[R] permis':>12} "
          f"{'E[R] refuse':>12} {'ecart':>8} {'detectabilite':>14}")
    print("-" * 78)

    # Reference : tout est permis.
    tous = np.concatenate([ya, yv])
    for part in PARTS:
        seuil = float(np.quantile(scores_tr, 1.0 - part))
        pa, pv = sa >= seuil, sv >= seuil
        permis = np.concatenate([ya[pa], yv[pv]])
        refuse = np.concatenate([ya[~pa], yv[~pv]])
        if len(permis) < 30:
            print(f"{part:>6.2f} {seuil:>+9.4f} {len(permis):>9} "
                  f"{'trop peu':>12}")
            continue
        ep, er = float(permis.mean()), (float(refuse.mean()) if len(refuse)
                                        else float("nan"))
        det = ep * np.sqrt(len(permis))
        print(f"{part:>6.2f} {seuil:>+9.4f} {len(permis):>9} {ep:>+12.4f} "
              f"{er:>+12.4f} {ep - er:>+8.4f} {det:>+14.2f}")

    print(f"\n{'reference':>6} {'(aucun veto)':>9} {len(tous):>9} "
          f"{tous.mean():>+12.4f}")
    print("\nLe veto sert s'il releve E[R] SANS effondrer le nombre de trades.")
    print("La detectabilite, E[R] x racine(N), tranche entre les deux.")


def mesure_tabm(df, tr, va, stats, cols, E) -> None:
    print(f"{'membres':>8} {'cache':>7} {'E[R] permis':>12} "
          f"{'permises':>9} {'detectabilite':>14}")
    print("-" * 56)
    for nm, dc in GEOMETRIES_TABM:
        sa, sv, ya, yv, s_tr, n = _scores(df, tr, va, stats, cols, E,
                                          n_membres=nm, d_cache=dc)
        seuil = float(np.quantile(s_tr, 0.50))
        permis = np.concatenate([ya[sa >= seuil], yv[sv >= seuil]])
        if len(permis) < 30:
            print(f"{nm:>8} {dc:>7}  trop peu de trades")
            continue
        ep = float(permis.mean())
        print(f"{nm:>8} {dc:>7} {ep:>+12.4f} {len(permis):>9} "
              f"{ep*np.sqrt(len(permis)):>+14.2f}")
    print("\nMeme seuil pour tous (mediane) : seule la GEOMETRIE varie.")


def budget() -> None:
    import torch
    import training as T
    from saint_core import (PolitiqueEnsemble, SAINTPolicySingleHead,
                            build_policy, OBS_N_FEATURES, N_ACTIONS)
    cfg = T.PPOConfig()
    occ = 4516          # occasions independantes, SL 8xATR sur neuf ans
    print(f"{occ:,} occasions independantes.  Le regime H1 qui a rendu +2.2 "
          f"portait 26 752 parametres pour 2 314, soit 11.6 par occasion.\n")
    print(f"{'d_model':>8} {'mlp SAINT':>10} {'mlp patch':>10} "
          f"{'SAINT':>9} {'PatchTST':>9} {'total':>9} {'par occ.':>9}")
    print("-" * 70)
    for d_model, mlp_s, mlp_p in ((8, 16, 32), (8, 8, 16), (8, 4, 8),
                                  (16, 8, 16)):
        try:
            sa = SAINTPolicySingleHead(
                n_features=OBS_N_FEATURES, d_model=d_model,
                num_blocks=cfg.num_blocks, heads=cfg.saint_heads,
                n_freq=cfg.saint_n_freq, mlp_dim=mlp_s,
                lecture=cfg.saint_lecture, dropout=0.05, ff_mult=2,
                max_len=cfg.lookback, n_actions=N_ACTIONS, n_ref=0)
            pa = build_policy(
                torch.device("cpu"), lookback=cfg.lookback,
                n_features=OBS_N_FEATURES, archi="patchtst", n_ref=0,
                num_blocks=cfg.num_blocks, d_model_patch=cfg.d_model_patch,
                mlp_dim=mlp_p, taille_patch=2, pas=1)
        except ValueError as e:
            print(f"{d_model:>8} {mlp_s:>10} {mlp_p:>10}  refuse : {e}")
            continue
        ns = sum(q.numel() for q in sa.parameters())
        np_ = sum(q.numel() for q in pa.parameters())
        print(f"{d_model:>8} {mlp_s:>10} {mlp_p:>10} {ns:>9,} {np_:>9,} "
              f"{ns+np_:>9,} {(ns+np_)/occ:>9.1f}")
    print("\nLa tete pese 92 a 98 % du reseau : c'est `mlp_dim` qui commande,")
    print("pas la profondeur. La capacite ne se tranche PAS sur une sonde")
    print("supervisee — ce tableau donne le budget, pas la reponse.")


def main() -> int:
    quoi = sys.argv[1] if len(sys.argv) > 1 else "part"
    if quoi == "budget":
        budget()
        return 0
    df, tr, va, stats, cols, E = _fenetres()
    print(f"train {tr:,} barres | validation {va:,} | plafond {E.HOLD} "
          f"| SL {E.SL_MULT}xATR R:R {E.RR}\n")
    if quoi == "selectivite":
        mesure_selectivite(df, tr, va, stats, cols, E)
    elif quoi == "phases":
        mesure_part_phases(df, tr, va, stats, cols, E)
    elif quoi == "tabm":
        mesure_tabm(df, tr, va, stats, cols, E)
    else:
        mesure_part(df, tr, va, stats, cols, E)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
