"""Le modele CLASSE-T-IL ? Mesure sur toutes les decisions, pas sur les 5 %.

LE PROBLEME QUE CE FICHIER RESOUT. La validation rend ~150 trades par epoch,
et l'erreur-type sur leur moyenne vaut 0.11 R. Elle ne peut donc pas separer
un modele qui ajoute +0.10 R d'un modele qui n'ajoute rien — et le "NEW BEST"
qu'elle declenche selectionne du bruit. Mesure sur exec37 : +0.169 R a
l'epoch 5, soit 1.3 ecart-type, puis +0.001 R a l'epoch 6.

Or la selectivite jette 95 % de l'information. Le modele produit une opinion
a CHAQUE barre ; n'en regarder que le sommet, c'est mesurer la moyenne d'un
echantillon de 150 quand on en a 10 000. Ici on garde tout, et on ne demande
plus "combien gagne-t-il" mais "CLASSE-T-IL" — une question a laquelle un
echantillon large repond, et qui est exactement celle que la selectivite
exploite ensuite.

CE QUI EST MESURE. Pour chaque barre echantillonnee, en etat PLAT :

  signal    p(achat) - p(vente), ce sur quoi la selectivite trie vraiment
  tete aux  la prediction directe du rendement net, si le checkpoint en a une
  realise   (R_achat - R_vente) / 2, la part SYMETRIQUE du rendement reel,
            qui deduit la derive du sous-jacent

et on rend la correlation de RANG, qui ne suppose ni linearite ni variance
finie — deux hypotheses fausses sur des rendements a queue lourde.

L'INCERTITUDE VIENT D'UN BOOTSTRAP PAR BLOCS CONTIGUS. Des barres voisines
partagent leur avenir : les traiter comme independantes diviserait l'erreur
par dix. Les blocs font plusieurs fois la duree d'un trade, donc deux blocs
differents ne partagent presque rien.

LE TEST N'EST PAS OUVERT : on mesure sur la validation du fold.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

import checkpoints as CK
import cibles as C
import training as T
from saint_core import FEATURE_COLS, build_policy

PAS = 6              # une observation toutes les 30 minutes
LOT = 4096
BLOCS = 200          # blocs du bootstrap
TIRAGES = 400


def _rang(x: np.ndarray) -> np.ndarray:
    """Rangs moyens, egalites comprises.

    Les egalites ne sont pas un detail ici : une politique saturee rend la
    meme probabilite sur des milliers de barres, et leur donner des rangs
    arbitraires fabriquerait de la correlation a partir de l'ordre du tableau.
    """
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[o] = np.arange(len(x), dtype=float)
    xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rang(a), _rang(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = ra.std() * rb.std() * len(ra)
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def _bootstrap(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float]:
    """(rho, erreur-type) par blocs contigus, tires avec remise."""
    n = len(a)
    taille = max(n // BLOCS, 1)
    debuts = np.arange(0, n - taille + 1, taille)
    ech = []
    for _ in range(TIRAGES):
        pris = rng.choice(debuts, size=len(debuts), replace=True)
        sel = np.concatenate([np.arange(d, d + taille) for d in pris])
        ech.append(_rho(a[sel], b[sel]))
    return _rho(a, b), float(np.std(ech, ddof=1))


def _erreur_bloc(y: np.ndarray, nb: int = 50) -> float:
    """Erreur-type d'une moyenne quand les points voisins partagent l'avenir."""
    taille = max(len(y) // nb, 1)
    bl = np.array([y[i:i + taille].mean()
                   for i in range(0, len(y) - taille + 1, taille)])
    return float(bl.std(ddof=1) / np.sqrt(len(bl))) if len(bl) > 2 else float("nan")


def main() -> int:
    quel = sys.argv[2] if len(sys.argv) > 2 else "last"
    try:
        complet, folds = CK.resoud(sys.argv[1] if len(sys.argv) > 1 else None,
                                   famille=quel)
    except FileNotFoundError as e:
        print(f"AUCUN CHECKPOINT UTILISABLE : {e}")
        return 1

    cfg = T.PPOConfig()
    df = T.load_mt5_data(cfg)
    n = len(df)
    tr = int(n * 0.70)
    va = C.borne_etude(n)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"checkpoint {complet}, folds {folds}")
    print(f"validation [{tr:,}, {va:,})  —  le test n'est pas ouvert\n")

    fold = folds[0]
    pth, calib_p, _ = CK.chemins(complet, fold)
    etat = torch.load(pth, map_location=device, weights_only=True)
    calib = json.load(open(calib_p, encoding="utf-8"))
    look = int(calib.get("lookback", cfg.lookback))

    # Les statistiques de normalisation se recalculent sur le TRAIN, comme a
    # l'entrainement. Les recalculer sur la validation mettrait l'observation
    # a une echelle que le reseau n'a jamais vue, et ferait passer pour une
    # incapacite du modele ce qui serait une erreur de mise a l'echelle.
    stats = T.compute_and_save_global_norm_stats(df.iloc[:tr], FEATURE_COLS,
                                                 path=None)
    dv = df.iloc[tr:va].reset_index(drop=True)
    mu = np.asarray(stats["mean"], np.float32)
    sd = np.maximum(np.asarray(stats["std"], np.float32), 1e-8)
    X = (dv[FEATURE_COLS].to_numpy(np.float32) - mu) / sd
    X = np.clip(X, -10.0, 10.0).astype(np.float32)

    policy = build_policy(device, lookback=look, state_dict=etat)
    policy.load_state_dict(etat, strict=True)
    policy.eval()

    idx = np.arange(look, len(dv) - C.BORNE_DEFAUT - 2, PAS)
    print(f"{len(idx):,} decisions echantillonnees (une toutes les "
          f"{PAS * 5} minutes)")

    # L'ETAT PLAT, exactement celui que voit la politique avant d'entrer :
    # position 0, latent 0, detention 0, echelle de risque 1.
    extra = np.zeros((look, 4), np.float32)
    extra[:, 3] = 1.0

    pb, ps, aux = [], [], []
    with torch.no_grad():
        for d in range(0, len(idx), LOT):
            bloc = idx[d:d + LOT]
            obs = np.stack([np.concatenate([X[i - look:i], extra], axis=-1)
                            for i in bloc])
            t = torch.from_numpy(obs).to(device)
            sortie = policy(t)
            logits = sortie[0] if isinstance(sortie, tuple) else sortie
            p = torch.softmax(logits, dim=-1).float().cpu().numpy()
            pb.append(p[:, 0])
            ps.append(p[:, 1])
            if aux is not None:
                try:
                    aux.append(policy.rendement(t).float().cpu().numpy())
                except Exception:
                    aux = None
    pb = np.concatenate(pb)
    ps = np.concatenate(ps)
    aux = np.concatenate(aux) if aux else None

    ra, rv = C.rendements(dv, idx, cfg)
    ok = np.isfinite(ra) & np.isfinite(rv)
    sym = (ra - rv) / 2.0            # part directionnelle, derive deduite
    print(f"{int(ok.sum()):,} resolues\n")

    rng = np.random.default_rng(0)
    print(f"{'signal':<34} {'rho de rang':>12} {'+/-':>8} {'sigma':>7}")
    print("-" * 65)
    sigs = [("p(achat) - p(vente)", pb - ps)]
    if aux is not None and aux.ndim == 2 and aux.shape[1] >= 2:
        sigs.append(("tete auxiliaire (achat - vente)", aux[:, 0] - aux[:, 1]))
    for nom, s in sigs:
        r, e = _bootstrap(s[ok], sym[ok], rng)
        print(f"{nom:<34} {r:>+12.4f} {e:>8.4f} {r / max(e, 1e-9):>+7.1f}")

    print("\nDECILES DE p(achat) - p(vente) — rendement symetrique reel")
    s, y = (pb - ps)[ok], sym[ok]
    q = np.quantile(s, np.linspace(0, 1, 11))
    print(f"{'decile':>7} {'n':>7} {'E[R] sym':>10} {'+/-':>8}")
    for k in range(10):
        m = (s >= q[k]) & ((s <= q[k + 1]) if k == 9 else (s < q[k + 1]))
        if m.sum() < 20:
            continue
        yy = y[m]
        print(f"{k + 1:>7} {int(m.sum()):>7,} {yy.mean():>+10.4f} "
              f"{_erreur_bloc(yy):>8.4f}")

    # LES DEUX COTES SEPAREMENT. La part symetrique melange les deux : un
    # modele qui saurait acheter et pas vendre y ressemblerait a un modele
    # moyen. Et le journal montre les shorts perdants a chaque epoch, donc
    # la question n'est pas theorique.
    print("\nPAR COTE — le signal de CE cote contre le rendement de CE cote")
    print(f"{'cote':<10} {'rho de rang':>12} {'+/-':>8} {'sigma':>7} "
          f"{'E[R] sommet 5%':>16}")
    print("-" * 60)
    for nom, sig, reel in (("achat", pb[ok], ra[ok]), ("vente", ps[ok], rv[ok])):
        r, e = _bootstrap(sig, reel, rng)
        haut = reel[sig >= np.quantile(sig, 0.95)]
        print(f"{nom:<10} {r:>+12.4f} {e:>8.4f} {r / max(e, 1e-9):>+7.1f} "
              f"{haut.mean():>+12.4f}+-{_erreur_bloc(haut):.3f}")

    # CE QUE LA SELECTIVITE ACHETE VRAIMENT. Le reglage en place garde 5 % des
    # barres ; la courbe dit si c'est le bon endroit, et surtout si le gain
    # par trade paie la perte d'occasions.
    print("\nCOURBE DE SELECTIVITE — meilleur cote, comme en production")
    meilleur = np.maximum(pb[ok], ps[ok])
    sens = np.where(pb[ok] >= ps[ok], 1, -1)
    rend = np.where(sens > 0, ra[ok], rv[ok])
    print(f"{'garde':>7} {'n':>8} {'E[R]':>9} {'+/-':>8} {'sigma':>7} "
          f"{'part achat':>11}")
    print("-" * 56)
    for part in (1.0, 0.50, 0.25, 0.10, 0.05, 0.02, 0.01):
        m = meilleur >= np.quantile(meilleur, 1.0 - part)
        if m.sum() < 50:
            continue
        y = rend[m]
        e = _erreur_bloc(y)
        print(f"{100 * part:>6.0f}% {int(m.sum()):>8,} {y.mean():>+9.4f} "
              f"{e:>8.4f} {y.mean() / max(e, 1e-9):>+7.1f} "
              f"{100 * (sens[m] > 0).mean():>10.0f}%")

    print("\nLECTURE. Un rho positif dit que le modele ORDONNE les occasions,")
    print("et c'est tout ce dont la selectivite a besoin. Un rho nul avec un")
    print("PnL de validation positif dit que le PnL etait un tirage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
