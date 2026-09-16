"""Banc de comparaison : REGRESSION DU RENDEMENT NET, avec abstention.

LE CHANGEMENT DE FOND. Le systeme precedent classait (le TP sera-t-il touche
avant le SL ?) puis retenait les 5 % les mieux classes. Etre bien classe n'est
PAS avoir une esperance positive : le filtre par rang prend toujours 5 % des
occasions, meme quand les meilleures sont mauvaises.

Ici on predit directement le RENDEMENT NET en unites de risque, et on n'entre
que si la meilleure prediction depasse une MARGE POSITIVE. L'abstention devient
donc possible : zero trade est une reponse valide, et c'est la bonne quand
rien ne depasse la marge.

LES COUTS SONT DEJA DANS LA CIBLE. `barrieres` place les barrieres a partir
d'un prix d'entree deja degrade (demi-spread + slippage d'entree) et facture
le slippage de sortie. Le rendement rendu est donc NET. Ne jamais les
soustraire une seconde fois.

PROTOCOLE, heritier des trois biais corriges le 15/09 :
  - entrees espacees de MAX_HOLD barres : aucun chevauchement des fenetres
    de resultat ;
  - MOYENNE SUR LES PHASES : une phase unique a un ecart-type de 0.076 sur
    l'esperance, soit l'ordre de grandeur de tous les ecarts qu'on cherche a
    mesurer. Comparaison APPARIEE a phase egale ;
  - walk-forward par blocs, chaque bloc juge par un modele entraine
    uniquement sur ce qui le precede ;
  - marge d'abstention CALIBREE sur une periode separee, jamais sur celle
    qui juge ;
  - fenetre de test (au-dela de 70 %) JAMAIS touchee.

    python banc_rendement_net.py
"""

import warnings

import numpy as np
import pandas as pd

import mesure_features as MF
from saint_core import FEATURE_COLS, ATR_PLANCHER_FRAC
from mesure_features import (barrieres, SPREAD_BPS, SLIP_ENTREE_BPS,
                             SLIP_SORTIE_BPS, CACHE)

warnings.filterwarnings("ignore")

# La geometrie suit l'echelle, elle n'est plus une constante de module : le
# M5 trade a 8xATR, le H1 a 2xATR. Les laisser confondues faisait apprendre a
# TabM une cible que l'environnement n'execute pas.
SL_MULT, RR = 8.0, 2.0
FRAC_CALIB = 0.30        # part finale du train reservee au calibrage de marge

# ------------------------------------------------------------------
#  DEUX ECHELLES, DEUX DIMENSIONNEMENTS
#
# Le protocole exige des entrees espacees de la duree maximale de detention :
# deux trades voisins ne doivent pas partager de barres de resultat, sinon le
# meme mouvement est compte plusieurs fois et l'erreur-type est sous-estimee.
# Ce meme protocole se dimensionne donc differemment selon l'echelle :
#
#          bougies   pas   entrees utilisables   phases   blocs
#   M1     1.8 M     240        ~5 200             6        6
#   H1      79 k      24         ~2 310            12        6
#
# En H1 on a soixante fois moins de barres. On compense en couvrant une plus
# grande part de l'historique par le NOMBRE de phases (12 phases de pas 2
# touchent une barre sur deux, chacune restant interne-ment sans chevauchement)
# plutot qu'en relachant l'espacement, qui est justement ce qui garantit
# l'independance des resultats.
#
# Le pas H1 vaut 24 barres — une journee — parce que la cible est une course
# SL 2xATR / TP 4xATR : au-dela d'une journee la quasi-totalite des courses est
# resolue, et allonger le pas ne ferait que reduire un echantillon deja mince.
# ------------------------------------------------------------------
CACHE_H1 = "data_cache_BTCUSD_H1.pkl"
CACHE_M5 = "data_cache_BTCUSD_M5.pkl"

# M5 : le pas vaut 288 barres — une journee — parce que la course est un
# SL 8xATR / TP 16xATR dont la duree mediane vaut 179 barres. Douze phases
# espacees de 24 barres couvrent la journee sans qu'aucune ne se chevauche
# elle-meme.
MODE = "m5"              # regle par configure() ; l'entrainement est en M5
CACHE_UTILISE = CACHE_M5
PAS = 288
N_BLOCS = 6
PHASES = list(range(0, 288, 24))
MIN_TRAIN, MIN_VAL = 600, 100


def configure(mode="m5"):
    """Fixe l'echelle. `barrieres` lit MF.MAX_HOLD : il faut le reregler AUSSI,
    sinon la course des barrieres durerait 240 heures au lieu de 24."""
    global MODE, CACHE_UTILISE, PAS, N_BLOCS, PHASES, MIN_TRAIN, MIN_VAL
    MODE = mode
    if mode == "m5":
        CACHE_UTILISE, PAS, N_BLOCS = CACHE_M5, 288, 6
        PHASES, MIN_TRAIN, MIN_VAL = list(range(0, 288, 24)), 600, 100
    elif mode == "h1":
        CACHE_UTILISE, PAS, N_BLOCS = CACHE_H1, 24, 6
        PHASES, MIN_TRAIN, MIN_VAL = list(range(0, 24, 2)), 500, 80
    else:
        CACHE_UTILISE, PAS, N_BLOCS = CACHE, 240, 6
        PHASES, MIN_TRAIN, MIN_VAL = list(range(0, 240, 40)), 600, 40
    MF.MAX_HOLD = PAS


def prepare():
    df = pd.read_pickle(CACHE_UTILISE).dropna(
        subset=FEATURE_COLS + ["atr_14"]).reset_index(drop=True)
    n = len(df)
    fin = int(n * 0.70)
    d = {
        "n": n, "fin": fin,
        "hi": df["high"].to_numpy(np.float64),
        "lo": df["low"].to_numpy(np.float64),
        "cl": df["close"].to_numpy(np.float64),
        "X": np.nan_to_num(np.column_stack(
            [df[c].to_numpy(np.float32) for c in FEATURE_COLS])),
    }
    d["atr"] = np.maximum(df["atr_14"].to_numpy(np.float64),
                          ATR_PLANCHER_FRAC * d["cl"])
    d["ds"] = (SPREAD_BPS / 1e4) * d["cl"] / 2.0
    d["se"] = (SLIP_ENTREE_BPS / 1e4) * d["cl"]
    d["ss"] = (SLIP_SORTIE_BPS / 1e4) * d["cl"]
    d["bornes"] = np.linspace(int(fin * 0.35), fin, N_BLOCS + 1).astype(int)
    return d


def cibles(d, idx):
    """Rendement NET en unites de risque, pour un BUY puis pour un SELL."""
    sortie = []
    for sens in (1, -1):
        r, _ = barrieres(d["hi"], d["lo"], d["cl"], d["atr"], idx,
                         SL_MULT, RR, sens, d["ds"], d["se"], d["ss"])
        sortie.append(r)
    return sortie[0], sortie[1]


def modele_arbres():
    import lightgbm as lgb

    def fabrique():
        return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                 num_leaves=31, min_child_samples=40,
                                 subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, reg_lambda=1.0,
                                 verbose=-1, n_jobs=2)
    return fabrique


def modele_tabm(n_membres=8, d_cache=256, epochs=30):
    """TabM — ensemble efficace a parametres partages (ICLR 2025).

    Un seul reseau dense porte l'essentiel des poids ; chaque membre de
    l'ensemble n'ajoute que deux vecteurs de rang 1 par couche, appliques en
    amont et en aval (adaptateurs multiplicatifs, facon BatchEnsemble). On
    obtient k predictions differentes pour un cout proche d'un seul reseau, et
    la moyenne d'ensemble fait le gros du travail de regularisation.

    Ce n'est PAS un transformer : ni attention, ni jeton par colonne. C'est
    precisement l'argument du papier — sur donnees tabulaires, un MLP bien
    ensemble bat les architectures a attention evaluees.
    """
    import torch
    import torch.nn as nn

    class CoucheEnsemble(nn.Module):
        def __init__(self, entree, sortie, k):
            super().__init__()
            self.lin = nn.Linear(entree, sortie)
            self.r = nn.Parameter(torch.ones(k, entree))
            self.s = nn.Parameter(torch.ones(k, sortie))
            self.b = nn.Parameter(torch.zeros(k, sortie))

        def forward(self, x):                       # x : (B, k, entree)
            return self.lin(x * self.r) * self.s + self.b

    class Reseau(nn.Module):
        def __init__(self, n_entrees, k, d):
            super().__init__()
            self.k = k
            self.c1 = CoucheEnsemble(n_entrees, d, k)
            self.c2 = CoucheEnsemble(d, d, k)
            self.tete = CoucheEnsemble(d, 1, k)
            self.drop = nn.Dropout(0.1)
            for c in (self.c1, self.c2, self.tete):
                # Adaptateurs tires dans [-1, 1] : sans cette dissymetrie les
                # k membres sont identiques et l'ensemble ne sert a rien.
                nn.init.uniform_(c.r, -1.0, 1.0)
                nn.init.uniform_(c.s, -1.0, 1.0)

        def forward(self, x):                       # x : (B, n_entrees)
            h = x.unsqueeze(1).expand(-1, self.k, -1)
            h = self.drop(torch.relu(self.c1(h)))
            h = self.drop(torch.relu(self.c2(h)))
            return self.tete(h).squeeze(-1)         # (B, k)

    class Enveloppe:
        def __init__(self):
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def fit(self, X, y):
            torch.manual_seed(0)
            # Normalisation calculee sur le TRAIN seul, conservee pour predire.
            self.m_ = X.mean(0)
            self.s_ = X.std(0) + 1e-8
            self.net = Reseau(X.shape[1], n_membres, d_cache).to(self.dev)
            opt = torch.optim.AdamW(self.net.parameters(), lr=2e-3,
                                    weight_decay=1e-4)
            xt = torch.tensor(np.clip((X - self.m_) / self.s_, -5, 5),
                              dtype=torch.float32, device=self.dev)
            yt = torch.tensor(y, dtype=torch.float32, device=self.dev)
            n = len(xt)
            self.net.train()
            for _ in range(epochs):
                perm = torch.randperm(n, device=self.dev)
                for i in range(0, n, 512):
                    b = perm[i:i + 512]
                    pred = self.net(xt[b])
                    perte = ((pred - yt[b].unsqueeze(1)) ** 2).mean()
                    opt.zero_grad()
                    perte.backward()
                    opt.step()
            return self

        def predict(self, X):
            self.net.eval()
            with torch.no_grad():
                xt = torch.tensor(np.clip((X - self.m_) / self.s_, -5, 5),
                                  dtype=torch.float32, device=self.dev)
                return self.net(xt).mean(dim=1).cpu().numpy()

    return Enveloppe


def une_phase(d, fabrique, phase, regles):
    """Rendements nets des trades pris, par regle de decision."""
    pris = {r: [] for r in regles}
    occasions = 0
    for b in range(N_BLOCS):
        a_va, b_va = d["bornes"][b], d["bornes"][b + 1]
        i_all = np.arange(phase, a_va - 2 * PAS, PAS)
        i_all = i_all[np.isfinite(d["atr"][i_all]) & (d["atr"][i_all] > 0)]
        if len(i_all) < MIN_TRAIN:
            continue
        coupe = int(len(i_all) * (1 - FRAC_CALIB))
        i_tr, i_ca = i_all[:coupe], i_all[coupe:]
        i_va = np.arange(a_va, b_va - PAS, PAS)
        i_va = i_va[np.isfinite(d["atr"][i_va]) & (d["atr"][i_va] > 0)]
        if len(i_va) < MIN_VAL:
            continue

        rb_tr, rs_tr = cibles(d, i_tr)
        p_ca, p_va = {}, {}
        for cle, y in (("b", rb_tr), ("s", rs_tr)):
            mod = fabrique().fit(d["X"][i_tr], y)
            p_ca[cle] = mod.predict(d["X"][i_ca])
            p_va[cle] = mod.predict(d["X"][i_va])

        rb_va, rs_va = cibles(d, i_va)
        best_ca = np.maximum(p_ca["b"], p_ca["s"])
        best_va = np.maximum(p_va["b"], p_va["s"])
        gain_va = np.where(p_va["b"] >= p_va["s"], rb_va, rs_va)
        occasions += len(i_va)

        for r in regles:
            # "q95" reproduit l'ANCIEN filtre par rang, comme repere.
            seuil = np.quantile(best_ca, 0.95) if r == "q95" else r
            sel = best_va >= seuil
            if sel.any():
                pris[r].append(gain_va[sel])
    return ({r: (np.concatenate(v) if v else np.array([])) for r, v in pris.items()},
            occasions)


def main(argv=None) -> int:
    import sys
    argv = sys.argv[1:] if argv is None else argv
    configure("m1" if "m1" in argv else "h1")
    d = prepare()
    print(f"echelle {MODE.upper()}  |  {CACHE_UTILISE}")
    regles = ["q95", 0.0, 0.05, 0.10, 0.20]
    print(f"{d['n']:,} bougies  |  {N_BLOCS} blocs  |  {len(PHASES)} phases  |  "
          f"entrees tous les {PAS} barres")
    print(f"barrieres SL {SL_MULT}xATR  R:R {RR}  —  cout DEJA dans la cible")
    print(f"marge calibree sur les {int(100 * FRAC_CALIB)} % finaux du train\n")

    for nom, fab in (("LightGBM", modele_arbres()), ("TabM", modele_tabm())):
        moyennes = {r: [] for r in regles}
        parts = {r: [] for r in regles}
        for ph in PHASES:
            res, occ = une_phase(d, fab, ph, regles)
            for r in regles:
                v = res[r]
                moyennes[r].append(v.mean() if len(v) >= 10 else np.nan)
                parts[r].append(len(v) / max(occ, 1))
        print(f"--- {nom} ---")
        print(f"{'regle':>18} {'part prise':>11} {'E[R] moyen':>12} "
              f"{'err-type':>10} {'phases +':>10}")
        print("-" * 66)
        for r in regles:
            v = np.array(moyennes[r], float)
            lbl = "rang 5 % (ancien)" if r == "q95" else f"marge {r:+.2f} R"
            if np.all(np.isnan(v)):
                print(f"{lbl:>18} {'abstention totale':>34}")
                continue
            ok = ~np.isnan(v)
            print(f"{lbl:>18} {100 * np.nanmean(parts[r]):10.2f}% "
                  f"{np.nanmean(v):+12.4f} "
                  f"{np.nanstd(v, ddof=1) / np.sqrt(ok.sum()):10.4f} "
                  f"{int(np.nansum(v > 0)):>7}/{len(PHASES)}")
        print()

    print("E[R] en UNITES DE RISQUE, net de tous les couts.")
    print("'part prise' = fraction des occasions ou le modele entre.")
    print("Une marge qui ne laisse passer AUCUN trade est un resultat, pas un echec.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
