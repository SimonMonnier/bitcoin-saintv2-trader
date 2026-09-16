"""Le troisieme votant : TabM, supervise, qui oppose son veto pendant le rollout.

POURQUOI IL N'EST PAS UN MEMBRE DU MELANGE. SAINT et PatchTST sont entraines
par PPO : leurs probabilites sont differentiables, donc elles se moyennent et
le gradient les traverse. TabM n'a pas de gradient dans cette boucle — il est
ajuste une fois par fold, en supervise — et surtout ses scores dependent de la
BARRE, pas seulement de l'observation. Le mettre dans le melange demanderait de
faire porter l'indice temporel par la politique, ce qu'elle ignore par
construction.

Il entre donc par le MASQUE D'ACTIONS, la ou l'environnement dit deja ce qui
est jouable. C'est exactement la regle que decrit `evalue_ensemble` : un signal
n'est pas pris si autre chose le contredit. TabM ne propose rien, il interdit.

POURQUOI L'AJUSTEMENT EST CROISE, ET C'EST LE POINT DELICAT. Le rollout de PPO
se joue sur la fenetre d'ENTRAINEMENT — precisement celle ou TabM serait ajuste.
Un TabM qui vote sur ses propres donnees d'apprentissage y est un oracle : il
aurait raison bien plus souvent qu'en production, et les deux reseaux PPO
apprendraient a lui deferer. En live, l'oracle disparait et la politique se
retrouve a suivre un partenaire devenu ordinaire.

Chaque barre recoit donc le score d'un TabM qui NE L'A PAS VUE : la fenetre est
decoupee en K blocs contigus, et le modele qui note un bloc est ajuste sur les
autres. Des blocs contigus et non un tirage au hasard — des lignes voisines
partagent leurs barres de resultat, donc un tirage melangerait entrainement et
notation par recouvrement.

    python tabm_votant.py        # verifie l'ajustement croise sur un fold
"""

from __future__ import annotations

import numpy as np

# Le nombre de blocs d'ajustement croise. Trois suffisent : chaque modele voit
# les deux tiers de la fenetre, et le cout est trois ajustements par fold.
K_BLOCS = 3
# Part des occasions que TabM s'autorise a ne PAS contredire. A 0.50 il laisse
# passer la moitie des directions ; plus bas, il devient un filtre dur.
PART_LAISSEE = 0.50


class Votant:
    """Scores TabM par barre, et le veto qui en decoule.

    `scores[0]` est le rendement net attendu d'un ACHAT a cette barre,
    `scores[1]` celui d'une VENTE. Les deux sont en unites de risque.
    """

    def __init__(self, n: int, debut: int = 0, fin: int | None = None):
        self.achat = np.full(n, np.nan, np.float32)
        self.vente = np.full(n, np.nan, np.float32)
        self.seuil = -np.inf
        # La fenetre que ce votant avait pour mission de noter. La couverture
        # se rapporte a ELLE et non au dataframe entier : le tableau couvre
        # tout le jeu pour que l'indice de barre serve directement d'index,
        # mais un score hors fenetre n'a jamais ete promis.
        self.debut, self.fin = debut, (n if fin is None else fin)

    def veto(self, i: int) -> tuple[bool, bool]:
        """(achat permis, vente permise) a la barre i.

        Une barre que TabM n'a pas pu noter ne bloque RIEN. Refuser par
        defaut ferait taire le modele sur tout l'echauffement et sur les bords
        de fenetre, et ce silence ressemblerait a un avis.
        """
        a, v = self.achat[i], self.vente[i]
        if not np.isfinite(a) or not np.isfinite(v):
            return True, True
        return bool(a >= self.seuil), bool(v >= self.seuil)

    def couverture(self) -> float:
        """Part de la fenetre demandee qui a recu un score."""
        tranche = self.achat[self.debut:self.fin]
        return float(np.isfinite(tranche).mean()) if len(tranche) else 0.0


def ajuste(df, debut: int, fin: int, fabrique, cibles, colonnes,
           stats, k: int = K_BLOCS, part: float = PART_LAISSEE,
           pas_train: int = 144) -> Votant:
    """Ajuste TabM en croise sur [debut, fin) et rend ses scores par barre.

    `cibles(df, idx)` rend le rendement net d'un achat et d'une vente aux
    indices donnes ; `stats` porte la normalisation du fold, la MEME que celle
    de la politique.

    `fabrique` est ce que rend `banc_rendement_net.modele_tabm()` — une CLASSE,
    pas une instance ni la fonction elle-meme. `fabrique()` produit donc un
    modele neuf. Passer `modele_tabm` sans l'appeler rend la fonction, et
    `fabrique().fit(X, y)` appelle alors `fit` sur la classe : le premier
    argument part dans `self` et l'erreur parle d'un `y` manquant, ce qui
    n'aide pas a comprendre.
    """
    X = df[colonnes].to_numpy(np.float32)
    X = np.clip(np.nan_to_num((X - stats["mean"]) / (stats["std"] + 1e-8)),
                -5.0, 5.0)

    v = Votant(len(df), debut, fin)
    bornes = np.linspace(debut, fin, k + 1).astype(int)
    for b in range(k):
        a_test, b_test = bornes[b], bornes[b + 1]
        # Le modele apprend sur TOUT sauf le bloc qu'il va noter.
        idx_tr = np.concatenate([
            np.arange(bornes[j], bornes[j + 1] - pas_train - 2, pas_train)
            for j in range(k) if j != b])
        if len(idx_tr) < 200:
            continue
        ra, rv = cibles(df, idx_tr)
        bon = np.isfinite(ra) & np.isfinite(rv)
        if bon.sum() < 200:
            continue
        idx_tr, ra, rv = idx_tr[bon], ra[bon], rv[bon]
        m_a = fabrique().fit(X[idx_tr], ra)
        m_v = fabrique().fit(X[idx_tr], rv)
        cible = np.arange(a_test, b_test)
        v.achat[cible] = m_a.predict(X[cible])
        v.vente[cible] = m_v.predict(X[cible])

    fini = np.isfinite(v.achat)
    if fini.any():
        # Le seuil est un QUANTILE des scores, pas une valeur absolue : le
        # niveau des predictions derive d'un bloc a l'autre, un seuil fixe
        # laisserait passer tout un bloc et bloquerait le suivant.
        tous = np.concatenate([v.achat[fini], v.vente[fini]])
        v.seuil = float(np.quantile(tous, 1.0 - part))
    return v


def hors_echantillon(df, debut_train: int, fin_train: int,
                     debut_note: int, fin_note: int, fabrique, cibles,
                     colonnes, stats, part: float = PART_LAISSEE,
                     pas_train: int = 144) -> Votant:
    """Ajuste sur [debut_train, fin_train) et note [debut_note, fin_note).

    C'est la version pour l'EVALUATION et le LIVE, ou la fenetre notee est
    posterieure a la fenetre d'apprentissage : aucun recouvrement, donc aucun
    besoin d'ajustement croise. Celui-ci n'existe que pour l'entrainement, ou
    PPO joue sur la fenetre meme qui a servi a ajuster TabM.

    LE SEUIL VIENT DE LA FENETRE D'APPRENTISSAGE, jamais de celle qu'on note.
    Le calibrer sur la fenetre evaluee reviendrait a choisir le filtre d'apres
    les donnees qu'il filtre — le biais que ce depot a paye trois a quatre
    points, deux fois, sur deux methodes sans rapport.
    """
    X = df[colonnes].to_numpy(np.float32)
    X = np.clip(np.nan_to_num((X - stats["mean"]) / (stats["std"] + 1e-8)),
                -5.0, 5.0)

    v = Votant(len(df), debut_note, fin_note)
    idx_tr = np.arange(debut_train, fin_train - pas_train - 2, pas_train)
    ra, rv = cibles(df, idx_tr)
    bon = np.isfinite(ra) & np.isfinite(rv)
    if bon.sum() < 200:
        return v                      # trop peu d'exemples : aucun veto
    idx_tr, ra, rv = idx_tr[bon], ra[bon], rv[bon]
    m_a = fabrique().fit(X[idx_tr], ra)
    m_v = fabrique().fit(X[idx_tr], rv)

    # Le seuil se lit sur les scores de la fenetre D'APPRENTISSAGE.
    sa, sv = m_a.predict(X[idx_tr]), m_v.predict(X[idx_tr])
    v.seuil = float(np.quantile(np.concatenate([sa, sv]), 1.0 - part))

    cible = np.arange(debut_note, fin_note)
    v.achat[cible] = m_a.predict(X[cible])
    v.vente[cible] = m_v.predict(X[cible])
    return v


def main() -> int:
    import numpy as np
    import banc_rendement_net as B
    import evalue_tabm_test as E
    import training as T
    from saint_core import FEATURE_COLS

    cfg = T.PPOConfig()
    df = T.load_mt5_data(cfg)
    n = len(df)
    train_len = int(n * E.TRAIN_FRAC)
    stats = T.compute_and_save_global_norm_stats(
        df.iloc[:train_len], FEATURE_COLS, path=None)

    print(f"ajustement croise sur {train_len:,} barres, {K_BLOCS} blocs "
          f"contigus, pas {E.PAS_TRAIN}")
    v = ajuste(df, 0, train_len, B.modele_tabm(), E.cibles_brutes,
               FEATURE_COLS, stats, pas_train=E.PAS_TRAIN)
    print(f"couverture : {100*v.couverture():.1f} % des barres notees")
    print(f"seuil de veto : {v.seuil:+.4f} R  "
          f"(laisse passer {100*PART_LAISSEE:.0f} % des directions)")

    a, vte = 0, 0
    for i in range(0, train_len, 97):
        pa, pv = v.veto(i)
        a += pa
        vte += pv
    total = len(range(0, train_len, 97))
    print(f"sur un echantillon de {total:,} barres : achat permis "
          f"{100*a/total:.1f} %, vente permise {100*vte/total:.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
