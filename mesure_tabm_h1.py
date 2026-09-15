"""TabM en H1 : l'avantage tient-il dans CHAQUE periode, ou dans une seule ?

POURQUOI CE FICHIER EXISTE. banc_rendement_net.py rapporte une erreur-type
calculee sur les PHASES. En H1 les phases sont espacees de deux heures : deux
phases voisines prennent presque les memes trades decales d'un cran, donc
elles ne comptent pas comme deux mesures independantes et l'erreur affichee
est trop belle. La dimension qui separe vraiment est le TEMPS.

On lit donc les memes trades autrement : moyenne sur les phases A BLOC FIXE,
puis quatre chiffres — un par periode. Un avantage reel apparait dans les
quatre ; un artefact se concentre dans un seul.

REPERE. Le cout d'entrer au hasard est calcule ici meme, sur les memes
occasions, pour les deux sens. C'est le zero contre lequel lire le reste.

    python mesure_tabm_h1.py
"""

import warnings

import numpy as np

import banc_rendement_net as B

warnings.filterwarnings("ignore")

MARGES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]


def main() -> int:
    B.configure("h1")
    d = B.prepare()
    fab = B.modele_tabm()

    print(f"TabM  |  H1  |  {B.N_BLOCS} blocs  |  {len(B.PHASES)} phases  |  "
          f"entrees tous les {B.PAS} barres")
    print(f"barrieres SL {B.SL_MULT}xATR  R:R {B.RR}  —  couts dans la cible\n")

    # (bloc, marge) -> liste des E[R] par phase ; et le repere au hasard.
    par_bloc = {b: {m: [] for m in MARGES} for b in range(B.N_BLOCS)}
    parts = {b: {m: [] for m in MARGES} for b in range(B.N_BLOCS)}
    hasard = {b: [] for b in range(B.N_BLOCS)}
    achat = {b: [] for b in range(B.N_BLOCS)}

    for ph in B.PHASES:
        for b in range(B.N_BLOCS):
            a_va, b_va = d["bornes"][b], d["bornes"][b + 1]
            i_all = np.arange(ph, a_va - 2 * B.PAS, B.PAS)
            i_all = i_all[np.isfinite(d["atr"][i_all]) & (d["atr"][i_all] > 0)]
            if len(i_all) < B.MIN_TRAIN:
                continue
            coupe = int(len(i_all) * (1 - B.FRAC_CALIB))
            i_tr, i_ca = i_all[:coupe], i_all[coupe:]
            i_va = np.arange(a_va, b_va - B.PAS, B.PAS)
            i_va = i_va[np.isfinite(d["atr"][i_va]) & (d["atr"][i_va] > 0)]
            if len(i_va) < B.MIN_VAL:
                continue

            rb_tr, rs_tr = B.cibles(d, i_tr)
            p_ca, p_va = {}, {}
            for cle, y in (("b", rb_tr), ("s", rs_tr)):
                mod = fab().fit(d["X"][i_tr], y)
                p_ca[cle] = mod.predict(d["X"][i_ca])
                p_va[cle] = mod.predict(d["X"][i_va])

            rb_va, rs_va = B.cibles(d, i_va)
            gain = np.where(p_va["b"] >= p_va["s"], rb_va, rs_va)
            best = np.maximum(p_va["b"], p_va["s"])

            # Reperes, sur EXACTEMENT les memes occasions.
            hasard[b].append(0.5 * (rb_va.mean() + rs_va.mean()))
            achat[b].append(rb_va.mean())

            for m in MARGES:
                sel = best >= m
                parts[b][m].append(sel.mean())
                par_bloc[b][m].append(gain[sel].mean() if sel.sum() >= 10
                                      else np.nan)

    # ---------- reperes ----------
    print("REPERE — entrer a chaque occasion, sans modele :")
    print(f"{'bloc':>6} {'hasard (moy 2 sens)':>21} {'toujours acheter':>18}")
    for b in range(B.N_BLOCS):
        if not hasard[b]:
            continue
        print(f"{b:>6} {np.mean(hasard[b]):>21.4f} {np.mean(achat[b]):>18.4f}")
    tous_h = [x for b in hasard for x in hasard[b]]
    print(f"{'TOUS':>6} {np.mean(tous_h):>21.4f}\n")

    # ---------- TabM, bloc par bloc ----------
    print("TabM — E[R] net par periode (moyenne sur les phases) :")
    entete = f"{'marge':>8} " + "".join(f"{'bloc '+str(b):>14}"
                                        for b in range(B.N_BLOCS))
    print(entete + f"{'ENSEMBLE':>12} {'part':>8} {'blocs +':>9}")
    print("-" * len(entete + "            ENSEMBLE     part   blocs +"))
    for m in MARGES:
        moy_b, ligne = [], f"{m:+8.2f} "
        for b in range(B.N_BLOCS):
            v = np.array(par_bloc[b][m], float)
            if len(v) == 0 or np.all(np.isnan(v)):
                ligne += f"{'-':>14}"
                continue
            mb = np.nanmean(v)
            moy_b.append(mb)
            ligne += f"{mb:>14.4f}"
        part = np.mean([np.mean(parts[b][m]) for b in range(B.N_BLOCS)
                        if parts[b][m]])
        pos = sum(1 for x in moy_b if x > 0)
        ens = np.mean(moy_b) if moy_b else np.nan
        print(ligne + f"{ens:>12.4f} {100*part:>7.1f}% {pos:>6}/{len(moy_b)}")

    print("\nLire la colonne 'blocs +' avant l'ensemble : quatre periodes sur")
    print("quatre est un avantage ; une seule periode est un accident.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
