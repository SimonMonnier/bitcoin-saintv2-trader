"""SAINT et PatchTST dans le cadre RENDEMENT NET, face a TabM et LightGBM.

Meme objectif, memes periodes, meme protocole que banc_rendement_net.py : on
predit le rendement NET en unites de risque et on n'entre que si la meilleure
prediction depasse une marge calibree a part.

CE QUI CHANGE ICI : ces deux modeles lisent une SEQUENCE de bougies, pas une
ligne de features. Le banc construit donc, pour chaque occasion, la fenetre
des `lookback` dernieres barres — ce qui permet aussi de repondre a la vraie
question de PatchTST : un historique plus long apporte-t-il quelque chose ?

REPERE INDISPENSABLE POUR LIRE LES CHIFFRES. Entrer au hasard coute -0.45 R ;
un oracle qui choisirait parfaitement le COTE atteindrait +0.478 R. La part
de cet ecart qu'un modele capture est la seule grandeur comparable entre
architectures. Il en faut 48 % pour atteindre zero ; TabM en capture 31 %.

    python banc_sequentiel.py

Les modeles neuronaux tournent sur moins de phases que le banc tabulaire :
leur cout est bien plus eleve et l'ecart-type inter-phases est reporte, donc
la precision reste lisible.
"""

import warnings

import numpy as np
import torch
import torch.nn as nn

import banc_rendement_net as B
from mesure_features import MAX_HOLD

warnings.filterwarnings("ignore")

PHASES = B.PHASES[:3]
LOOKBACKS_PATCH = (32, 96, 192)
LOOKBACK_SAINT = 25


# ============================================================
#  Extraction de sequences
# ============================================================

def sequences(X, idx, lookback):
    """(len(idx), lookback, n_features) — fenetre causale finissant en idx.

    Aucune barre posterieure a l'occasion n'entre : la fenetre s'arrete a idx
    inclus, comme l'observation que verrait l'agent en live.
    """
    deb = idx - lookback + 1
    pas = np.arange(lookback)
    return X[deb[:, None] + pas[None, :]]


# ============================================================
#  Modeles sequentiels
# ============================================================

class _Regresseur:
    """Boucle d'entrainement commune : normalisation sur le train, Adam, MSE."""

    def __init__(self, fabrique_net, lookback, epochs=25, lot=256, lr=1e-3):
        self.fabrique_net = fabrique_net
        self.lookback = lookback
        self.epochs = epochs
        self.lot = lot
        self.lr = lr
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, S, y):
        torch.manual_seed(0)
        plat = S.reshape(-1, S.shape[-1])
        self.m_ = plat.mean(0)
        self.s_ = plat.std(0) + 1e-8
        self.net = self.fabrique_net(S.shape[-1], self.lookback).to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr,
                                weight_decay=1e-4)
        xt = torch.tensor(np.clip((S - self.m_) / self.s_, -5, 5),
                          dtype=torch.float32, device=self.dev)
        yt = torch.tensor(y, dtype=torch.float32, device=self.dev)
        n = len(xt)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.dev)
            for i in range(0, n, self.lot):
                b = perm[i:i + self.lot]
                perte = ((self.net(xt[b]).squeeze(-1) - yt[b]) ** 2).mean()
                opt.zero_grad()
                perte.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
        return self

    def predict(self, S):
        self.net.eval()
        sorties = []
        with torch.no_grad():
            for i in range(0, len(S), 1024):
                x = torch.tensor(np.clip((S[i:i + 1024] - self.m_) / self.s_, -5, 5),
                                 dtype=torch.float32, device=self.dev)
                sorties.append(self.net(x).squeeze(-1).cpu().numpy())
        return np.concatenate(sorties)


def _net_saint(n_features, lookback):
    """Le tronc SAINT du projet, avec une tete de REGRESSION.

    On reutilise `encode` — donc exactement l'architecture entrainee par PPO,
    plongement periodique, RoPE, QK-Norm, jeton CLS compris — et on remplace
    les tetes acteur/critique par une sortie scalaire. C'est ce qui rend la
    comparaison avec TabM honnete : meme objectif, meme cible, meme decoupe.
    """
    from saint_core import SAINTPolicySingleHead, N_BLOCS_DEFAUT

    class SaintRegresseur(nn.Module):
        def __init__(self):
            super().__init__()
            self.tronc = SAINTPolicySingleHead(
                n_features=n_features, d_model=80, num_blocks=N_BLOCS_DEFAUT,
                heads=5, ff_mult=2, max_len=lookback, n_ref=0)
            self.tete = nn.Sequential(
                nn.Linear(160, 128), nn.GELU(), nn.Linear(128, 1))

        def forward(self, x):
            return self.tete(self.tronc.encode(x))

    return SaintRegresseur()


def _net_patchtst(n_features, lookback, taille_patch=16, pas=8, d=64,
                  heads=4, couches=3):
    """PatchTST — segmentation temporelle et INDEPENDANCE DES CANAUX.

    Deux idees du papier, toutes deux dirigees contre le cout de l'attention
    sur une longue sequence :

      - PATCHING. On regroupe les barres par segments de `taille_patch` avec
        un recouvrement. L'attention porte sur ~L/pas jetons au lieu de L, et
        chaque jeton voit un motif local plutot qu'une barre isolee.

      - CANAUX INDEPENDANTS. Le MEME encodeur traite chaque feature
        separement, sans melange inter-colonnes. C'est l'oppose de SAINT, qui
        fait de l'attention sur l'axe des features. Le papier defend que ce
        partage regularise : une seule serie a modeliser, vue n_features fois.

    Le melange entre colonnes n'a donc lieu qu'a la toute fin, dans la tete.
    """
    n_patchs = max(1, (lookback - taille_patch) // pas + 1)

    class PatchTST(nn.Module):
        def __init__(self):
            super().__init__()
            self.taille_patch, self.pas, self.n_patchs = taille_patch, pas, n_patchs
            self.proj = nn.Linear(taille_patch, d)
            self.pos = nn.Parameter(torch.zeros(1, n_patchs, d))
            nn.init.normal_(self.pos, std=0.02)
            couche = nn.TransformerEncoderLayer(
                d_model=d, nhead=heads, dim_feedforward=d * 4, dropout=0.1,
                batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(couche, num_layers=couches)
            self.norm = nn.LayerNorm(d)
            # TETE AGREGEE, et non "flatten" comme dans le papier.
            #
            # La tete du papier aplatit (F x N x d) : a 192 bougies cela fait
            # 11.4 M de parametres pour ~3 500 echantillons d'entrainement par
            # bloc, soit 3 000 parametres par exemple. Elle est concue pour de
            # la PREVISION, ou chaque segment doit contribuer separement a
            # chaque pas predit, et ou les jeux sont bien plus grands.
            #
            # Ici la cible est UN scalaire. On moyenne donc sur les segments
            # avant la tete : 0.5 M de parametres, invariant au nombre de
            # segments, donc comparable d'un lookback a l'autre — ce qui est
            # exactement la question posee.
            self.tete = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.1),
                nn.Linear(n_features * d, 256), nn.GELU(),
                nn.Linear(256, 1))

        def forward(self, x):                       # x : (B, L, F)
            Bt, L, F = x.shape
            h = x.permute(0, 2, 1)                  # (B, F, L)
            h = h.unfold(dimension=2, size=self.taille_patch, step=self.pas)
            h = h[:, :, :self.n_patchs, :]          # (B, F, N, taille_patch)
            h = self.proj(h) + self.pos.unsqueeze(1)
            # Canaux independants : on replie F dans le lot, donc le meme
            # encodeur voit chaque colonne separement.
            h = h.reshape(Bt * F, self.n_patchs, -1)
            h = self.norm(self.enc(h))
            h = h.reshape(Bt, F, self.n_patchs, -1).mean(dim=2)   # (B, F, d)
            return self.tete(h)

    return PatchTST()


# ============================================================
#  Evaluation
# ============================================================

def evalue(d, faire_modele, lookback, quantiles=(0.85, 0.95)):
    """Rend {quantile: (E[R] par phase, part prise par phase)}."""
    res = {q: ([], []) for q in quantiles}
    for ph in PHASES:
        pris = {q: [] for q in quantiles}
        occ = 0
        for b in range(B.N_BLOCS):
            a_va, b_va = d["bornes"][b], d["bornes"][b + 1]
            i_all = np.arange(max(ph, lookback), a_va - 2 * MAX_HOLD, MAX_HOLD)
            i_all = i_all[np.isfinite(d["atr"][i_all]) & (d["atr"][i_all] > 0)]
            if len(i_all) < 600:
                continue
            coupe = int(len(i_all) * (1 - B.FRAC_CALIB))
            i_tr, i_ca = i_all[:coupe], i_all[coupe:]
            i_va = np.arange(a_va + ph, b_va - MAX_HOLD, MAX_HOLD)
            i_va = i_va[np.isfinite(d["atr"][i_va]) & (d["atr"][i_va] > 0)]
            if len(i_va) < 40:
                continue

            S_tr = sequences(d["X"], i_tr, lookback)
            S_ca = sequences(d["X"], i_ca, lookback)
            S_va = sequences(d["X"], i_va, lookback)
            rb_tr, rs_tr = B.cibles(d, i_tr)
            p_ca, p_va = {}, {}
            for cle, y in (("b", rb_tr), ("s", rs_tr)):
                mod = faire_modele().fit(S_tr, y)
                p_ca[cle] = mod.predict(S_ca)
                p_va[cle] = mod.predict(S_va)
            rb_va, rs_va = B.cibles(d, i_va)
            best_ca = np.maximum(p_ca["b"], p_ca["s"])
            best_va = np.maximum(p_va["b"], p_va["s"])
            gain = np.where(p_va["b"] >= p_va["s"], rb_va, rs_va)
            occ += len(i_va)
            for q in quantiles:
                sel = best_va >= np.quantile(best_ca, q)
                if sel.any():
                    pris[q].append(gain[sel])
        for q in quantiles:
            v = np.concatenate(pris[q]) if pris[q] else np.array([])
            res[q][0].append(v.mean() if len(v) >= 10 else np.nan)
            res[q][1].append(len(v) / max(occ, 1))
    return res


# Mesures de reference, etablies par banc_rendement_net.py sur les memes blocs.
HASARD, ORACLE = -0.4503, 0.4780


def ligne(nom, res, quantiles=(0.85, 0.95)):
    for q in quantiles:
        v = np.array(res[q][0], float)
        p = np.array(res[q][1], float)
        if np.all(np.isnan(v)):
            print(f"{nom:<22} {'q' + str(int(100*q)):>5} {'abstention totale':>40}")
            continue
        e = np.nanmean(v)
        err = np.nanstd(v, ddof=1) / np.sqrt(np.sum(~np.isnan(v)))
        capt = 100 * (e - HASARD) / (ORACLE - HASARD)
        print(f"{nom:<22} {'q' + str(int(100*q)):>5} {100*np.nanmean(p):7.2f}% "
              f"{e:+10.4f} {err:9.4f} {capt:8.1f}% "
              f"{int(np.nansum(v > 0)):>4}/{len(v)}")
        nom = ""


def main() -> int:
    d = B.prepare()
    print(f"{d['n']:,} bougies  |  {B.N_BLOCS} blocs  |  {len(PHASES)} phases")
    print(f"repere : hasard {HASARD:+.4f}   oracle du cote {ORACLE:+.4f}   "
          f"il faut capturer 48 % pour atteindre zero\n")
    print(f"{'modele':<22} {'seuil':>5} {'part':>8} "
          f"{'E[R]':>10} {'err-type':>9} {'% oracle':>8} {'ph +':>6}")
    print("-" * 74)

    ligne("SAINT (lookback 25)",
          evalue(d, lambda: _Regresseur(_net_saint, LOOKBACK_SAINT, epochs=20),
                 LOOKBACK_SAINT))

    for lb in LOOKBACKS_PATCH:
        ligne(f"PatchTST ({lb} bougies)",
              evalue(d, lambda lb=lb: _Regresseur(_net_patchtst, lb, epochs=20),
                     lb))

    print("\n'% oracle' = part de l'ecart hasard->oracle que le modele capture.")
    print("Reperes tabulaires sur les memes blocs : TabM 31 %, LightGBM 24 %.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
