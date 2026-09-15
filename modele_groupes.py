"""Attention entre GROUPES de colonnes, pas entre colonnes.

L'IDEE, ET D'OU ELLE VIENT. Le systeme Ichimoku est explicitement conditionnel :
"si le nuage est oriente a la baisse, ne chercher QUE des ventes ; s'il n'est
pas oriente, faire du trading range et rien d'autre". Le regime commande quelles
autres informations comptent. Un MLP doit apprendre ce routage avec des
parametres ; une attention l'obtient par construction — elle decide a chaque
situation quels groupes regarder.

POURQUOI PAR GROUPES ET PAS PAR COLONNES. SAINT attend entre les 107 colonnes
prises une a une, soit 11 449 paires dont l'immense majorite n'a aucun sens, et
son cout croit en F au carre : 19 700 ms par passe contre 100 pour PatchTST,
8.9 Go sur une carte de 8. En regroupant par bloc semantique on tombe a ONZE
jetons : le terme quadratique est divise par 95, et le modele raisonne sur "le
nuage", "Chikou", "les plats" — les unites du livre, pas des scalaires.

CE QUI JOUE CONTRE, ET QU'IL FAUT LIRE AVANT LE RESULTAT. Deux mesures deja
faites vont dans l'autre sens :

  - Un arbre EST un routage conditionnel : "si le regime vaut ceci, regarde
    cette colonne". C'est exactement ce que l'attention apporterait de plus
    qu'un MLP. Sur les memes fenetres de test, LightGBM rend +2.6 points et
    TabM +3.8. Le mecanisme est deja represente, et il fait MOINS bien.
  - Le papier TabM soutient precisement qu'un MLP bien ensemble bat les
    architectures a attention sur donnees tabulaires.

Ce fichier existe donc pour trancher, pas pour confirmer. Un resultat en
dessous de TabM est une reponse, et la plus probable des deux.

PROTOCOLE. Meme boucle d'entrainement que TabM — Adam, MSE, 30 epochs,
normalisation sur le train — et meme ensemble de quatre membres moyennes, pour
que la comparaison ne mesure pas l'ensemblage au lieu de l'architecture.
"""

import numpy as np
import torch
import torch.nn as nn

import saint_core as S

# Les groupes, dans l'ordre ou le livre les lit. Un groupe = un jeton.
#
# Le decoupage suit le SENS, pas la commodite : "obstacles" reunit ce qui barre
# la route aux prix, "chikou" ce qui concerne le filtre du systeme, "nuage" la
# boussole. Regrouper autrement — par exemple tout l'Ichimoku en un bloc —
# rendrait l'attention incapable de distinguer le filtre de la boussole, qui
# jouent des roles opposes dans les regles.
GROUPES = {
    "prix_h1": S.FEATURE_COLS_TF,
    "contexte_h4": S.FEATURE_COLS_SUP,
    "flux_temps": S.FEATURE_COLS_EXT + S.FEATURE_COLS_LIQ_TEMPS,
    "range": S.FEATURE_COLS_RANGE,
    "lignes": ["ich_dev_tenkan", "ich_dev_kijun", "ich_dev_ssa", "ich_dev_ssb",
               "ich_tenkan_kijun"],
    "plats": ["ich_plat_tenkan", "ich_plat_kijun", "ich_plat_ssb",
              "ich_pente_kijun", "ich_pente_tenkan"],
    "obstacles": ["ich_obstacle_haut", "ich_obstacle_bas",
                  "ich_plat_kijun_haut", "ich_plat_kijun_bas",
                  "ich_ratio_haussier", "ich_ratio_baissier"],
    "chikou": ["ich_chikou_haut", "ich_chikou_bas", "ich_chikou_libre_h",
               "ich_chikou_libre_b", "ich_chikou_vs_prix"],
    "nuage": ["ich_kumo_epaisseur", "ich_kumo_pente", "ich_kumo_futur_pente",
              "ich_pos_kumo", "ich_regime"],
    "tendance": ["ich_repli_kijun", "ich_contacts_kijun", "ich_kijun_casse_h",
                 "ich_kijun_casse_b", "ich_cassure_haut", "ich_cassure_bas",
                 "ich_cassure_prox_tenkan"],
    "chandeliers": ["ich_marteau", "ich_pendu", "ich_filante",
                    "ich_marteau_inv", "ich_doji", "ich_haute_vague",
                    "ich_marubozu", "ich_avalement_h", "ich_avalement_b",
                    "ich_nuage_noir", "ich_penetrante", "ich_harami",
                    "ich_etoile_matin", "ich_etoile_soir"],
}


def indices_groupes(colonnes):
    """Positions de chaque groupe dans le vecteur de features.

    REFUSE de continuer si une colonne manque ou traine en trop. Un groupe
    silencieusement incomplet donnerait un modele qui tourne, qui apprend, et
    qui ignore une partie des donnees sans que rien ne le signale — le meme
    genre de defaut que la fuite de np.gradient, qui gardait la bonne forme.
    """
    pos = {c: i for i, c in enumerate(colonnes)}
    idx, vues = {}, set()
    manquantes = []
    for nom, cols in GROUPES.items():
        ids = []
        for c in cols:
            if c not in pos:
                manquantes.append(c)
            else:
                ids.append(pos[c])
                vues.add(c)
        idx[nom] = np.array(ids, np.int64)
    orphelines = [c for c in colonnes if c not in vues]
    if manquantes or orphelines:
        raise ValueError(
            f"groupes incoherents — absentes du jeu : {manquantes} ; "
            f"colonnes sans groupe : {orphelines}")
    return idx


class ReseauGroupes(nn.Module):
    """Un jeton par groupe, attention entre eux, puis tete de regression."""

    def __init__(self, idx, d=64, couches=2, tetes=4, p=0.1):
        super().__init__()
        self.noms = list(idx.keys())
        self.idx = [torch.as_tensor(idx[n]) for n in self.noms]
        # Une projection PAR groupe : les groupes n'ont ni la meme taille ni la
        # meme nature, un projecteur partage les forcerait au meme traitement.
        self.proj = nn.ModuleList([nn.Linear(len(i), d) for i in self.idx])
        # Un vecteur appris par groupe, pour que l'attention sache QUI parle :
        # sans lui, les jetons sont interchangeables et "le nuage" ne se
        # distingue pas de "Chikou".
        self.emb = nn.Parameter(torch.zeros(len(self.idx), d))
        nn.init.normal_(self.emb, std=0.02)
        couche = nn.TransformerEncoderLayer(
            d_model=d, nhead=tetes, dim_feedforward=2 * d, dropout=p,
            batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(couche, num_layers=couches)
        self.norm = nn.LayerNorm(d)
        self.tete = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(p),
                                  nn.Linear(d, 1))

    def forward(self, x):
        jetons = torch.stack(
            [p(x[:, i.to(x.device)]) for p, i in zip(self.proj, self.idx)],
            dim=1)
        h = self.enc(jetons + self.emb)
        return self.tete(self.norm(h.mean(1))).squeeze(-1)


def fabrique(colonnes, n_membres=4, epochs=30, lot=256, lr=1e-3):
    """Rend une fabrique compatible avec le banc : .fit(X, y) / .predict(X)."""
    idx = indices_groupes(colonnes)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Enveloppe:
        def fit(self, X, y):
            # Ensemble de quatre membres moyennes, comme TabM : sans lui, la
            # comparaison mesurerait l'ensemblage et non l'architecture.
            self.m_ = X.mean(0)
            self.s_ = X.std(0) + 1e-8
            xt = torch.tensor(np.clip((X - self.m_) / self.s_, -5, 5),
                              dtype=torch.float32, device=dev)
            yt = torch.tensor(y, dtype=torch.float32, device=dev)
            self.nets = []
            for graine in range(n_membres):
                torch.manual_seed(graine)
                net = ReseauGroupes(idx).to(dev)
                opt = torch.optim.AdamW(net.parameters(), lr=lr,
                                        weight_decay=1e-4)
                net.train()
                for _ in range(epochs):
                    perm = torch.randperm(len(xt), device=dev)
                    for i in range(0, len(xt), lot):
                        b = perm[i:i + lot]
                        perte = ((net(xt[b]) - yt[b]) ** 2).mean()
                        opt.zero_grad()
                        perte.backward()
                        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        opt.step()
                net.eval()
                self.nets.append(net)
            return self

        def predict(self, X):
            xt = torch.tensor(np.clip((X - self.m_) / self.s_, -5, 5),
                              dtype=torch.float32, device=dev)
            with torch.no_grad():
                p = torch.stack([n(xt) for n in self.nets]).mean(0)
            return p.cpu().numpy()

    return lambda: Enveloppe()


if __name__ == "__main__":
    idx = indices_groupes(list(S.FEATURE_COLS))
    net = ReseauGroupes(idx)
    n = sum(p.numel() for p in net.parameters())
    print(f"{len(idx)} groupes, {len(S.FEATURE_COLS)} colonnes, "
          f"{n:,} parametres par membre")
    for k, v in idx.items():
        print(f"  {k:>14} : {len(v):3d} colonnes")
    x = torch.randn(8, len(S.FEATURE_COLS))
    print(f"\nsortie : {tuple(net(x).shape)}")
