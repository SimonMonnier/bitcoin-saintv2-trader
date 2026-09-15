"""TabM sur les geometries de barrieres les plus neutres.

CE QUI A CONDUIT ICI, en trois mesures du 2026-09-15 sur neuf ans de H1 :

  1. TabM bat le hasard de +0.085 R dans SIX periodes sur six (signes p=0.016),
     mais son E[R] absolu reste +0.034 +/- 0.031 : indistinguable de zero.
  2. La raison n'est pas le modele. Entrer au hasard avec SL 2xATR / TP 4xATR
     / 24 h coute deja -0.035 R, et le modele passe son avantage a combler ce
     trou au lieu de le mettre en resultat.
  3. Le balayage des geometries montre pourquoi : la friction est un montant
     FIXE, donc elargir le stop la divise. SL 1xATR coute -0.079 R au hasard,
     SL 3xATR seulement -0.020, et SL 3 / R:R 3 / 48 h rend +0.021.

LA QUESTION QUE CE FICHIER TRANCHE. Un taux de base plus neutre laisse-t-il
l'avantage du modele tomber dans le resultat, ou l'avantage lui-meme
disparait-il quand on elargit le stop ? Les deux sont plausibles : un stop
large veut dire moins de courses resolues, donc une cible plus bruitee et
peut-etre moins apprenable. On ne peut pas le deviner, seulement le mesurer.

PROTOCOLE. Identique pour chaque geometrie, et c'est le point : entrees
espacees de la duree de detention (aucun chevauchement), marge d'abstention
calibree sur une tranche du train qui ne juge jamais, walk-forward par blocs
ou chaque bloc n'est juge que par un modele entraine sur ce qui le precede,
fenetre de test au-dela de 70 % jamais touchee. Le REPERE au hasard est
recalcule sur exactement les memes occasions que le modele.

    python mesure_geometrie_tabm.py
"""

import warnings

import numpy as np

import banc_rendement_net as B
import mesure_features as MF

warnings.filterwarnings("ignore")

# (sl_mult, rr, hold). La premiere est le reglage actuel, gardee comme repere.
GEOMETRIES = [
    (2.0, 2.0, 24),
    (3.0, 3.0, 48),
    (3.0, 2.0, 48),
    (3.0, 1.5, 48),
]
MARGES = [0.0, 0.10, 0.20, 0.30]
PHASES = (0, 8, 16)          # trois decalages suffisent : voir l'entete
MIN_TR, MIN_VA = 400, 60


def cibles(d, idx, sl, rr):
    out = []
    for sens in (1, -1):
        r, _ = MF.barrieres(d["hi"], d["lo"], d["cl"], d["atr"], idx,
                            sl, rr, sens, d["ds"], d["se"], d["ss"])
        out.append(r)
    return out[0], out[1]


def une_geometrie(d, fab, sl, rr, hold):
    MF.MAX_HOLD = hold
    par_bloc = {b: {m: [] for m in MARGES} for b in range(B.N_BLOCS)}
    parts = {b: {m: [] for m in MARGES} for b in range(B.N_BLOCS)}
    hasard = {b: [] for b in range(B.N_BLOCS)}

    for ph in PHASES:
        for b in range(B.N_BLOCS):
            a_va, b_va = d["bornes"][b], d["bornes"][b + 1]
            i_all = np.arange(ph, a_va - 2 * hold, hold)
            i_all = i_all[np.isfinite(d["atr"][i_all]) & (d["atr"][i_all] > 0)]
            if len(i_all) < MIN_TR:
                continue
            coupe = int(len(i_all) * (1 - B.FRAC_CALIB))
            i_tr, i_ca = i_all[:coupe], i_all[coupe:]
            i_va = np.arange(a_va + ph, b_va - hold, hold)
            i_va = i_va[np.isfinite(d["atr"][i_va]) & (d["atr"][i_va] > 0)]
            if len(i_va) < MIN_VA:
                continue

            rb_tr, rs_tr = cibles(d, i_tr, sl, rr)
            p_ca, p_va = {}, {}
            for cle, y in (("b", rb_tr), ("s", rs_tr)):
                mod = fab().fit(d["X"][i_tr], y)
                p_ca[cle] = mod.predict(d["X"][i_ca])
                p_va[cle] = mod.predict(d["X"][i_va])

            rb_va, rs_va = cibles(d, i_va, sl, rr)
            gain = np.where(p_va["b"] >= p_va["s"], rb_va, rs_va)
            best = np.maximum(p_va["b"], p_va["s"])
            hasard[b].append(0.5 * (rb_va.mean() + rs_va.mean()))

            for m in MARGES:
                sel = best >= m
                parts[b][m].append(sel.mean())
                par_bloc[b][m].append(gain[sel].mean() if sel.sum() >= 10
                                      else np.nan)
    return par_bloc, parts, hasard


def main() -> int:
    B.configure("h1")
    d = B.prepare()
    fab = B.modele_tabm()
    print(f"{d['n']:,} bougies H1  |  {B.N_BLOCS} blocs  |  "
          f"{len(PHASES)} phases  |  test intouche\n")

    for sl, rr, hold in GEOMETRIES:
        pb, pt, ha = une_geometrie(d, fab, sl, rr, hold)
        base_b = [np.mean(v) for b, v in ha.items() if v]
        if not base_b:
            print(f"SL {sl} R:R {rr} hold {hold} : pas assez d'occasions\n")
            continue
        base = float(np.mean(base_b))
        print(f"=== SL {sl}xATR   R:R {rr}   detention {hold} h ===")
        print(f"repere au hasard : {base:+.4f} R "
              f"({sum(1 for x in base_b if x > 0)}/{len(base_b)} blocs positifs)")
        print(f"{'marge':>8} {'part':>8} {'E[R]':>9} {'err-type':>10} "
              f"{'blocs +':>9} {'ecart/hasard':>13} {'blocs mieux':>12}")
        print("-" * 76)
        for m in MARGES:
            moy_b, ecarts = [], []
            for b in range(B.N_BLOCS):
                v = np.array(pb[b][m], float)
                if len(v) == 0 or np.all(np.isnan(v)):
                    continue
                mb = np.nanmean(v)
                moy_b.append(mb)
                ecarts.append(mb - np.mean(ha[b]))
            if not moy_b:
                print(f"{m:+8.2f} {'abstention totale':>30}")
                continue
            part = np.mean([np.mean(pt[b][m]) for b in range(B.N_BLOCS)
                            if pt[b][m]])
            err = np.std(moy_b, ddof=1) / np.sqrt(len(moy_b))
            print(f"{m:+8.2f} {100*part:7.1f}% {np.mean(moy_b):+9.4f} "
                  f"{err:10.4f} {sum(1 for x in moy_b if x > 0):>6}/{len(moy_b)}"
                  f" {np.mean(ecarts):+13.4f} "
                  f"{sum(1 for x in ecarts if x > 0):>8}/{len(ecarts)}")
        print()

    print("'err-type' est calculee sur les BLOCS, pas sur les phases : des")
    print("phases voisines prennent presque les memes trades et ne comptent")
    print("pas comme des mesures independantes.")
    print("'blocs mieux' est le test des signes : 6/6 vaut p = 0.016.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
