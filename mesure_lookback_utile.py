"""Combien de barres de passe le modele doit-il voir ? Mesure, pas intuition.

LA QUESTION, ET POURQUOI ELLE EST MAL POSEE D'HABITUDE. On choisit un lookback
et on suppose qu'il faut "assez de contexte". Or les colonnes Ichimoku SONT
deja des resumes du passe : Tenkan resume 9 barres, Kijun 26, SSB 52, et
Chikou-Span regarde 26 barres en arriere. Tout cela est present dans la ligne
de l'instant t. Le modele n'a pas a le redecouvrir.

LE FAIT QUI RETOURNE LA CHARGE DE LA PREUVE. TabM obtient +3.8 points au-dessus
du point mort sur la fenetre de test en ne voyant QU'UNE LIGNE — lookback 1.
PatchTST en voit 96. Ce n'est donc pas a la sequence d'etre "reduite au
necessaire" : c'est a elle de montrer qu'elle apporte quoi que ce soit.

CE QUE CE FICHIER MESURE. Le meme modele, le meme protocole, les memes
fenetres de test, la meme selectivite fixee d'avance — seule change la
PROFONDEUR du passe empile en entree. Les lignes k, k-1, ... sont concatenees,
ce qui donne au modele exactement l'information qu'une architecture
sequentielle pourrait en tirer, sans prejuger de la facon dont elle le ferait.

CE QUE LA REPONSE COMMANDE. Si le resultat plafonne des les premieres barres,
alors les 96 barres de PatchTST sont du poids mort, et le cout en F**2 de
SAINT — 19 700 ms par passe contre 100 pour PatchTST — serait paye pour rien.
Si au contraire il monte avec la profondeur, la sequence porte quelque chose et
l'architecture merite qu'on s'y attarde.

    python mesure_lookback_utile.py [selectivite]
"""

import sys

import numpy as np

import banc_rendement_net as B
import evalue_tabm_test as E
import training as T
from evalue_test_exhaustif import resume
from saint_core import FEATURE_COLS

# Trois profondeurs suffisent a repondre a la question posee : le resultat
# monte-t-il avec le passe empile, oui ou non. Une courbe fine coute six fois
# plus cher pour la meme reponse, et la profondeur 32 ferait 3 424 colonnes
# pour 2 300 exemples d'entrainement — on mesurerait le surajustement, pas
# l'apport du passe.
PROFONDEURS = (1, 4, 16)
SELECTIVITE = 0.05


def empile(X, profondeur):
    """Concatene les `profondeur` dernieres lignes. La ligne t reste la derniere.

    Un decalage, jamais une fenetre centree : X[t-j] pour j de 0 a profondeur-1.
    Les premieres lignes, qui n'ont pas assez de passe, recopient la plus
    ancienne disponible — elles tombent de toute facon dans le warmup.
    """
    if profondeur <= 1:
        return X
    morceaux = [X]
    for j in range(1, profondeur):
        d = np.empty_like(X)
        d[j:] = X[:-j]
        d[:j] = X[0]
        morceaux.append(d)
    return np.concatenate(morceaux, axis=1)


def main() -> int:
    sel = float(sys.argv[1]) if len(sys.argv) > 1 else SELECTIVITE
    cfg = T.PPOConfig()
    df = T.load_mt5_data(cfg)
    n = len(df)
    train_len, val_len = int(n * E.TRAIN_FRAC), int(n * E.VAL_FRAC)
    test_len = int(n * E.TEST_FRAC)
    window = train_len + val_len + test_len

    print(f"{n:,} barres | {len(FEATURE_COLS)} colonnes par ligne")
    print(f"selectivite {100*sel:.0f} % fixee d'avance, test intouche")
    print(f"modele TabM, identique a chaque profondeur\n")
    print(f"{'profondeur':>11} {'colonnes':>9} {'PnL':>11} {'trades':>8} "
          f"{'ecart':>9} {'+/-':>5} {'PF':>6} {'folds +':>8}")
    print("-" * 74)

    fab = B.modele_tabm()
    for prof in PROFONDEURS:
        tous, positifs, nfolds = [], 0, 0
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
            X = empile(X, prof)

            a_tr, b_tr = start, start + train_len
            a_va, b_va = b_tr, b_tr + val_len
            a_te, b_te = b_va, b_va + test_len
            n_cal = int(val_len * cfg.calib_frac)

            i_tr = np.arange(a_tr + 1, b_tr - E.HOLD - 2, E.PAS_TRAIN)
            rb, rs = E.cibles_brutes(df, i_tr)
            bon = np.isfinite(rb) & np.isfinite(rs)
            i_tr, rb, rs = i_tr[bon], rb[bon], rs[bon]

            mod_b = fab().fit(X[i_tr], rb)
            mod_s = fab().fit(X[i_tr], rs)
            idx = np.arange(a_va, b_te)
            sb = np.full(len(df), -1e9)
            ss = np.full(len(df), -1e9)
            sb[idx] = mod_b.predict(X[idx])
            ss[idx] = mod_s.predict(X[idx])

            marge = float(np.quantile(
                np.maximum(sb[a_va:a_va + n_cal], ss[a_va:a_va + n_cal]),
                1.0 - sel))
            env = T.BTCTradingEnvDiscrete(test_data, cfg)
            departs = T.departs_disjoints(test_data.length, cfg.lookback,
                                          cfg.episode_length)
            pnls = E.evalue(env, departs, sb[a_te:b_te], ss[a_te:b_te], marge)
            ec = E.ecart_au_point_mort(pnls)
            if np.isfinite(ec):
                nfolds += 1
                positifs += int(ec > 0)
            tous += pnls

        a = np.array(tous, float)
        g, p = a[a > 0], a[a <= 0]
        wr = 100.0 * len(g) / max(len(a), 1)
        aw = float(g.mean()) if len(g) else 0.0
        al = abs(float(p.mean())) if len(p) else 0.0
        be = 100.0 * al / (aw + al) if (aw + al) > 0 else float("nan")
        err = 100.0 * np.sqrt((wr / 100) * (1 - wr / 100) / max(len(a), 1))
        pf = float(g.sum() / abs(p.sum())) if len(p) and p.sum() else float("nan")
        print(f"{prof:11d} {prof*len(FEATURE_COLS):9d} {a.sum():+11.2f}$ "
              f"{len(a):8d} {wr-be:+8.1f}pt {err:5.1f} {pf:6.2f} "
              f"{positifs:5d}/{nfolds:<2d}")

    print("\nSi la colonne 'ecart' ne monte pas avec la profondeur, la sequence")
    print("n'apporte rien : les colonnes Ichimoku resument deja le passe, et")
    print("une architecture sequentielle paierait son cout pour rien.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
