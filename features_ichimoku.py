"""Les features de TOUTES les strategies Ichimoku, pas seulement le range.

SOURCE. Patrick Riguet, "Ichimoku Kinko Hyo - La formation totale" (2018) et
"Les chandeliers japonais" (2017). Les regles citees en commentaire viennent de
ces deux ouvrages ; les numeros de page sont ceux du premier sauf mention.

CE QUI MANQUAIT. features_range.py ne couvre qu'un des trois modes de trading
du systeme. Le modele ne voyait rien de :

  - CHIKOU-SPAN, que l'auteur appelle "le filtre du systeme" (p.31). C'est
    elle qui valide ou invalide TOUT signal, range compris. "Tout fait
    obstacle a Chikou-Span : les plats Tenkan-Sen, SSB, SSA, Kijun-Sen, mais
    aussi les chandeliers, les droites de tendance... tout." Si elle est libre
    elle valide, sinon il faut s'abstenir. C'etait l'absence la plus grave.
  - LE REPLI SUR PLAT KIJUN en tendance (p.75-80), avec sa distinction
    decisive : une Kijun ORIENTEE est un indicateur de momentum, seule la
    Kijun PLATE fait support. Franchir une Kijun orientee n'est pas une
    cassure.
  - LA CASSURE DE NOUVEAUX PLUS HAUTS/BAS en tendance (p.81-82), avec sa
    condition explicite : "les prix ne sont pas trop eloignes de la
    Tenkan-Sen".
  - L'ESPACE TRADABLE. Le systeme refuse toute entree en dessous d'un ratio
    2/1 (p.89) : un signal parfait est abandonne s'il n'y a pas la place. La
    distance au premier obstacle decide donc de l'entree autant que le signal.
  - LE SURACHAT / SURVENTE par eloignement de Tenkan-Sen (p.19) et l'effet
    elastique de Kijun-Sen (p.22).
  - LA BOUSSOLE : l'orientation du nuage dit tendance haussiere, baissiere ou
    range, et elle commande quelle strategie s'applique (p.25-26).
  - LES 14 STRUCTURES DE RENVERSEMENT en chandeliers, non conditionnees a la
    proximite d'une borne de range comme elles l'etaient jusqu'ici.

UNE REGLE QUE JE N'ENCODE PAS EN DUR, ET POURQUOI. L'ouvrage est formel : les
plats ne jouent leur role de support/resistance QU'EN TENDANCE, jamais en
range (p.23), et c'est vrai pour Chikou-Span aussi (p.35). J'aurais pu
multiplier chaque distance par un indicateur de regime. Je fournis plutot les
deux separement — les distances d'un cote, le regime de l'autre — parce que
coder la conjonction en dur suppose que ma detection de regime est juste, et
qu'une erreur de detection detruirait alors l'information au lieu de la
ponderer. Le modele peut apprendre la conjonction ; il ne peut pas retrouver
une information effacee.

CAUSALITE, le point ou ce fichier pourrait mentir sans rien casser :

  - Tenkan, Kijun : moyennes de plus haut/plus bas passes. Sans probleme.
  - SSA, SSB DESSINEES : decalees de +26, donc calculees il y a 26 barres.
    C'est de l'information retardee.
  - SSA, SSB FUTURES : calculees sur les donnees d'aujourd'hui et projetees en
    avant. Elles sont CONNUES maintenant — c'est le "nuage futur" de
    l'ouvrage, et sa lecture est licite.
  - CHIKOU-SPAN : sa valeur est le cours de cloture d'aujourd'hui, dessine 26
    barres en arriere. Les obstacles qu'elle rencontre sont les chandeliers et
    les lignes autour de cette position passee. Tout est donc connu a l'instant
    du calcul. C'est contre-intuitif mais parfaitement causal : on ne lit pas
    le futur de Chikou, on lit le passe qu'elle traverse.
"""

import numpy as np
import pandas as pd

from features_range import TENKAN, KIJUN, SENKOU_B, DECALAGE, ajoute_ichimoku_brut

# Fenetre de recherche des anciens plats servant de niveaux "en extension".
# L'ouvrage trace ces droites a vue, sans limite fixe ; 200 barres en H1 font
# une dizaine de jours, ce qui couvre largement ce qu'un oeil retient.
FENETRE_EXT = 200

# Tolerance de platitude, en fraction d'ATR. Une ligne bouge rarement de
# exactement zero a cause des flottants ; en dessous de ce seuil, on considere
# qu'elle n'a pas bouge.
TOL_PLAT = 0.02

# Demi-largeur de la zone ou l'on cherche les obstacles de Chikou-Span, autour
# de sa position dessinee (t - 26).
FENETRE_CHIKOU = 26


# Liste STATIQUE des colonnes produites, dans l'ordre. saint_core l'importe
# pour construire FEATURE_COLS sans avoir a executer le calcul.
#
# POURQUOI DOUBLER LA LISTE PLUTOT QUE LA DEDUIRE. saint_core doit connaitre
# les noms sans lire un fichier de donnees ni tourner sur un DataFrame ; le
# prix a payer est une liste qui peut deriver du code qui la produit. C'est
# exactement le defaut qui a deja coute un checkpoint charge avec la mauvaise
# architecture dans ce depot. Le __main__ de ce fichier compare donc les deux
# et refuse de finir si elles different : la derive devient une erreur bruyante
# au lieu d'un KeyError trois etapes plus loin.
COLONNES = [
    # A. Position relative aux lignes
    "ich_dev_tenkan", "ich_dev_kijun", "ich_dev_ssa", "ich_dev_ssb",
    "ich_tenkan_kijun",
    # B. Plats et orientation
    "ich_plat_tenkan", "ich_plat_kijun", "ich_plat_ssb",
    "ich_pente_kijun", "ich_pente_tenkan",
    # C. Obstacles pour les prix et espace tradable
    "ich_obstacle_haut", "ich_obstacle_bas",
    "ich_plat_kijun_haut", "ich_plat_kijun_bas",
    "ich_ratio_haussier", "ich_ratio_baissier",
    # D. Chikou-Span, le filtre du systeme
    "ich_chikou_haut", "ich_chikou_bas",
    "ich_chikou_libre_h", "ich_chikou_libre_b", "ich_chikou_vs_prix",
    # E. Le nuage, la boussole
    "ich_kumo_epaisseur", "ich_kumo_pente", "ich_kumo_futur_pente",
    "ich_pos_kumo", "ich_regime",
    # F. Tendance : replis et cassures
    "ich_repli_kijun", "ich_contacts_kijun",
    "ich_kijun_casse_h", "ich_kijun_casse_b",
    "ich_cassure_haut", "ich_cassure_bas", "ich_cassure_prox_tenkan",
    # G. Les 14 structures de renversement
    "ich_marteau", "ich_pendu", "ich_filante", "ich_marteau_inv",
    "ich_doji", "ich_haute_vague", "ich_marubozu",
    "ich_avalement_h", "ich_avalement_b",
    "ich_nuage_noir", "ich_penetrante", "ich_harami",
    "ich_etoile_matin", "ich_etoile_soir",
]


# ==========================================================================
#  Outils
# ==========================================================================

def _pente(valeurs, k=3):
    """Pente ARRIERE sur k barres : (x[i] - x[i-k]) / k.

    POURQUOI PAS LA FONCTION DE GRADIENT DE NUMPY. C'est une difference
    CENTREE : elle vaut (x[i+1] - x[i-1]) / 2 et lit donc la barre SUIVANTE.
    Sur une serie temporelle c'est une fuite pure et simple, et elle ne leve
    aucune erreur — la colonne s'appelle toujours "pente", elle a la bonne
    forme, le bon ordre de grandeur, et le modele s'en sert.

    Ce qu'elle a coute, le 2026-09-15 : les 47 colonnes Ichimoku affichaient
    +25.5 points au-dessus du point mort avec un profit factor de 2.85 sur la
    fenetre de TEST, contre +1.6 points sans elles. Un resultat pareil n'existe
    pas ; c'est son invraisemblance qui l'a trahi, aucune verification ne
    l'aurait attrape. `ich_kumo_futur_pente` etait le pire des cinq :
    ssb_futur[i+1] est le milieu des 52 periodes finissant en i+1, donc il
    contient le plus haut et le plus bas de DEMAIN. La boussole du systeme
    lisait l'avenir.

    Trois barres plutot qu'une : l'orientation d'une ligne Ichimoku se juge sur
    une tendance courte, pas sur le sursaut d'une barre.
    """
    v = np.asarray(valeurs, np.float64)
    out = np.full(len(v), np.nan)
    if len(v) > k:
        out[k:] = (v[k:] - v[:-k]) / float(k)
    return out


def _longueur_plat(valeurs, atr, tol=TOL_PLAT):
    """Nombre de barres consecutives ou la ligne n'a pas bouge, jusqu'a t.

    "Plus un plat est long, plus il est fort en tant que support ou
    resistance" (p.18). La longueur est donc l'information, pas seulement le
    fait d'etre plat.
    """
    v = np.asarray(valeurs, np.float64)
    a = np.asarray(atr, np.float64)
    bouge = np.abs(np.diff(v, prepend=v[0])) > tol * np.maximum(a, 1e-9)
    out = np.zeros(len(v))
    n = 0
    for i in range(len(v)):
        n = 0 if bouge[i] else n + 1
        out[i] = n
    return out


def _niveaux_extension(valeurs, plat, prix, atr, fenetre=FENETRE_EXT):
    """Distances au plat le plus proche AU-DESSUS et EN DESSOUS, en ATR.

    Ce sont les "droites en extension" que l'ouvrage fait tracer depuis les
    anciens paliers (p.19, p.24) : un plat passe reste un seuil longtemps
    apres avoir ete forme. Un niveau qu'aucun plat n'occupe rend l'infini, que
    l'on borne a 10 ATR — au-dela, l'obstacle ne contraint plus rien.
    """
    v = np.asarray(valeurs, np.float64)
    p = np.asarray(prix, np.float64)
    a = np.maximum(np.asarray(atr, np.float64), 1e-9)
    est_plat = np.asarray(plat, bool)
    n = len(v)
    haut = np.full(n, np.nan)
    bas = np.full(n, np.nan)
    for i in range(n):
        d = max(0, i - fenetre)
        niv = v[d:i][est_plat[d:i]]
        niv = niv[np.isfinite(niv)]
        if not len(niv):
            continue
        sup = niv[niv > p[i]]
        inf = niv[niv < p[i]]
        haut[i] = (sup.min() - p[i]) / a[i] if len(sup) else 10.0
        bas[i] = (p[i] - inf.max()) / a[i] if len(inf) else 10.0
    return np.clip(haut, 0, 10), np.clip(bas, 0, 10)


def _chikou_libre(close, high, low, atr, decalage=DECALAGE,
                  fenetre=FENETRE_CHIKOU):
    """Distance de Chikou-Span au premier obstacle au-dessus / en dessous.

    Chikou-Span vaut close[t], dessinee en t - 26. Ses obstacles sont les
    chandeliers qui l'entourent a cette position : leurs plus hauts au-dessus
    d'elle, leurs plus bas en dessous. L'ouvrage insiste : "tout fait obstacle
    a Chikou-Span" (p.32), et c'est elle qui valide ou invalide le signal.

    On regarde de t-26-26 a t-1 : la zone que Chikou traverse depuis sa
    position dessinee jusqu'a maintenant. Aucune barre posterieure a t n'entre.
    """
    c = np.asarray(close, np.float64)
    h = np.asarray(high, np.float64)
    l = np.asarray(low, np.float64)
    a = np.maximum(np.asarray(atr, np.float64), 1e-9)
    n = len(c)
    haut = np.full(n, np.nan)
    bas = np.full(n, np.nan)
    for i in range(n):
        j = i - decalage
        if j < 1:
            continue
        d = max(0, j - fenetre)
        f = i                      # jusqu'a la barre precedente incluse
        if f <= d:
            continue
        hh = h[d:f]
        ll = l[d:f]
        sup = hh[hh > c[i]]
        inf = ll[ll < c[i]]
        haut[i] = (sup.min() - c[i]) / a[i] if len(sup) else 10.0
        bas[i] = (c[i] - inf.max()) / a[i] if len(inf) else 10.0
    return np.clip(haut, 0, 10), np.clip(bas, 0, 10)


# ==========================================================================
#  Chandeliers — les 14 structures des deux ouvrages
# ==========================================================================

def _chandeliers(df, atr):
    """Structures de renversement, en intensite continue plutot qu'en binaire.

    CHOIX ASSUME. L'ouvrage decrit des formes ("le corps doit etre petit", "la
    meche au moins deux fois le corps"). Un booleen jette l'information de
    degre : un marteau presque parfait et un marteau limite comptent pareil, et
    le seuil devient un reglage arbitraire de plus. On rend donc des grandeurs
    continues bornees, dont le modele apprend lui-meme le seuil utile.
    """
    o, h, l, c = (df["open"].to_numpy(np.float64), df["high"].to_numpy(np.float64),
                  df["low"].to_numpy(np.float64), df["close"].to_numpy(np.float64))
    a = np.maximum(np.asarray(atr, np.float64), 1e-9)
    corps = np.abs(c - o)
    haut_corps = np.maximum(c, o)
    bas_corps = np.minimum(c, o)
    mh = h - haut_corps                       # meche haute
    mb = bas_corps - l                        # meche basse
    etendue = np.maximum(h - l, 1e-9)
    haussier = (c > o).astype(np.float64)

    def dec(x):
        return np.concatenate([[np.nan], x[:-1]])

    f = {}
    # Petit corps, longue meche basse : marteau (retournement haussier apres
    # baisse) ou pendu (retournement baissier apres hausse). La meme forme,
    # deux noms selon ce qui precede — d'ou la ponderation par la tendance
    # courte des trois barres precedentes.
    forme_marteau = np.clip(mb / etendue, 0, 1) * np.clip(1 - corps / etendue, 0, 1)
    forme_filante = np.clip(mh / etendue, 0, 1) * np.clip(1 - corps / etendue, 0, 1)
    tend3 = np.concatenate([[np.nan] * 3, np.sign(c[3:] - c[:-3])])

    f["ich_marteau"] = forme_marteau * (tend3 < 0)
    f["ich_pendu"] = forme_marteau * (tend3 > 0)
    f["ich_filante"] = forme_filante * (tend3 > 0)
    f["ich_marteau_inv"] = forme_filante * (tend3 < 0)

    # Doji : corps quasi nul. Haute vague : petit corps entre deux longues
    # meches — "l'indecision" des deux ouvrages.
    f["ich_doji"] = np.clip(1 - corps / (0.1 * etendue), 0, 1)
    f["ich_haute_vague"] = (np.clip(1 - corps / etendue, 0, 1)
                            * np.clip(np.minimum(mh, mb) / etendue * 4, 0, 1))
    # Marubozu : presque tout corps, aucune meche. "Puissant chandelier" (p.82).
    f["ich_marubozu"] = np.clip(corps / etendue, 0, 1) ** 3 * np.sign(c - o)

    # Avalement : le corps englobe entierement le precedent, en sens inverse.
    o1, c1 = dec(o), dec(c)
    corps1 = np.abs(c1 - o1)
    englobe = ((np.maximum(c, o) >= np.maximum(c1, o1))
               & (np.minimum(c, o) <= np.minimum(c1, o1))).astype(np.float64)
    ampleur = np.clip(corps / np.maximum(corps1, 1e-9) - 1.0, 0, 3) / 3.0
    f["ich_avalement_h"] = englobe * ampleur * (c > o) * (c1 < o1)
    f["ich_avalement_b"] = englobe * ampleur * (c < o) * (c1 > o1)

    # Couverture en nuage noir / penetrante, VARIANTE MARCHE CONTINU.
    #
    # Les deux ouvrages consacrent un chapitre entier a cette difference
    # ("Les variantes morphologiques des structures de renversement sur le
    # marche des devises", p.24-35 du livre sur les chandeliers). Sur le
    # marche ACTIONS, ces structures exigent une ouverture AU-DELA de la
    # cloture precedente : il faut un gap. Sur un marche ouvert en continu il
    # n'y a quasiment pas de gaps, et la regle devient : "le chandelier ouvre
    # au meme niveau de prix (ou a peine quelques pips au-dessus) que le prix
    # de cloture du chandelier precedent" et "doit cloturer au-dela de la
    # moitie du corps du chandelier precedent".
    #
    # Le bitcoin cote 24 h sur 24 et sept jours sur sept — encore moins de gaps
    # que le Forex, qui ferme le week-end. Coder la variante actions revenait
    # a produire deux colonnes muettes : ecart-type 0.004 et 0.003, soit une
    # structure detectee presque jamais sur 79 000 barres. L'auteur previent
    # exactement contre cette erreur : "beaucoup de traders ne savent pas
    # reconnaitre les configurations sur le Forex parce qu'ils tentent de
    # reproduire la morphologie de facon dogmatique".
    #
    # La tolerance d'ouverture vaut un dixieme d'ATR, l'equivalent des
    # "quelques pips" du texte.
    mid1 = (o1 + c1) / 2.0
    tol = 0.10 * a
    f["ich_nuage_noir"] = (((o >= c1 - tol) & (c < mid1) & (c1 > o1))
                           .astype(np.float64)
                           * np.clip((mid1 - c) / a, 0, 2) / 2.0)
    f["ich_penetrante"] = (((o <= c1 + tol) & (c > mid1) & (c1 < o1))
                           .astype(np.float64)
                           * np.clip((c - mid1) / a, 0, 2) / 2.0)

    # Harami : le corps courant tient ENTIEREMENT dans le precedent — l'inverse
    # de l'avalement, signe d'essoufflement.
    dedans = ((np.maximum(c, o) <= np.maximum(c1, o1))
              & (np.minimum(c, o) >= np.minimum(c1, o1))).astype(np.float64)
    f["ich_harami"] = dedans * np.clip(1 - corps / np.maximum(corps1, 1e-9), 0, 1)

    # Etoiles du matin / du soir : trois barres, la mediane ayant un petit
    # corps, la troisieme reprenant plus de la moitie de la premiere.
    o2, c2 = dec(o1), dec(c1)
    petit1 = np.clip(1 - corps1 / np.maximum(np.abs(c2 - o2), 1e-9), 0, 1)
    f["ich_etoile_matin"] = (petit1 * ((c2 < o2) & (c > o)
                                       & (c > (o2 + c2) / 2.0)).astype(np.float64))
    f["ich_etoile_soir"] = (petit1 * ((c2 > o2) & (c < o)
                                      & (c < (o2 + c2) / 2.0)).astype(np.float64))
    return f


# ==========================================================================
#  Assemblage
# ==========================================================================

def ajoute_features_ichimoku(df, atr_col="atr_14", suffixe=""):
    """Ajoute toutes les colonnes `ich_*`. Rend (df, liste des noms).

    `suffixe` permet de calculer le MEME jeu sur une autre echelle et de le
    joindre au premier sans collision de noms. C'est ce que prescrit le livre :
    "valider les signaux trouves sur une unite de temps en basculant sur les UT
    superieures". Les periodes Ichimoku sont des nombres de BOUGIES — Tenkan 9,
    Kijun 26, SSB 52 — donc les memes colonnes calculees en M5 et en H1 ne
    decrivent pas du tout les memes structures : 45 minutes contre neuf heures
    pour Tenkan.
    """
    df = ajoute_ichimoku_brut(df)
    h, l, c, o = df["high"], df["low"], df["close"], df["open"]
    atr = df[atr_col].replace(0, np.nan)
    a = atr.to_numpy(np.float64)
    cn = c.to_numpy(np.float64)
    cols = []

    def pose(nom, serie):
        nom = nom + suffixe
        df[nom] = pd.Series(serie, index=df.index).replace(
            [np.inf, -np.inf], np.nan)
        cols.append(nom)

    tenkan = df["tenkan"].to_numpy(np.float64)
    kijun = df["kijun"].to_numpy(np.float64)
    ssa = df["ssa"].to_numpy(np.float64)
    ssb = df["ssb"].to_numpy(np.float64)

    # Le nuage FUTUR : calcule aujourd'hui, projete en avant. Connu maintenant,
    # donc licite, et c'est lui que l'ouvrage lit comme boussole.
    ssa_f = (df["tenkan"] + df["kijun"]).to_numpy(np.float64) / 2.0
    ssb_f = ((h.rolling(SENKOU_B).max() + l.rolling(SENKOU_B).min())
             / 2.0).to_numpy(np.float64)

    # ---------- A. Position relative aux lignes ----------
    # "Tenkan-Sen colle aux prix ; quand les prix s'en eloignent il s'agit de
    # survente ou surachat, cette situation n'est jamais tenable" (p.19).
    pose("ich_dev_tenkan", (cn - tenkan) / a)
    # L'effet elastique : "Kijun-Sen est un aimant qui attire les prix lorsque
    # ceux-ci s'en eloignent" (p.24).
    pose("ich_dev_kijun", (cn - kijun) / a)
    pose("ich_dev_ssa", (cn - ssa) / a)
    pose("ich_dev_ssb", (cn - ssb) / a)
    pose("ich_tenkan_kijun", (tenkan - kijun) / a)

    # ---------- B. Plats et orientation ----------
    lp_tenkan = _longueur_plat(tenkan, a)
    lp_kijun = _longueur_plat(kijun, a)
    lp_ssb = _longueur_plat(ssb, a)
    pose("ich_plat_tenkan", np.clip(lp_tenkan / TENKAN, 0, 4))
    pose("ich_plat_kijun", np.clip(lp_kijun / KIJUN, 0, 4))
    pose("ich_plat_ssb", np.clip(lp_ssb / SENKOU_B, 0, 4))
    # "Une Kijun-Sen ORIENTEE est un indicateur de momentum ; en tant que
    # support, seul le PLAT Kijun-Sen compte" (p.78). Les deux informations
    # sont donc distinctes et toutes deux utiles.
    pose("ich_pente_kijun", _pente(kijun) / a)
    pose("ich_pente_tenkan", _pente(tenkan) / a)

    # ---------- C. Obstacles pour les PRIX, et espace tradable ----------
    plat_t = lp_tenkan > 0
    plat_k = lp_kijun > 0
    plat_s = lp_ssb > 0
    ht, bt = _niveaux_extension(tenkan, plat_t, cn, a)
    hk, bk = _niveaux_extension(kijun, plat_k, cn, a)
    hs, bs = _niveaux_extension(ssb, plat_s, cn, a)
    obst_h = np.minimum(np.minimum(ht, hk), hs)
    obst_b = np.minimum(np.minimum(bt, bk), bs)
    pose("ich_obstacle_haut", obst_h)
    pose("ich_obstacle_bas", obst_b)
    pose("ich_plat_kijun_haut", hk)
    pose("ich_plat_kijun_bas", bk)
    # Le ratio atteignable si l'on entrait maintenant : espace disponible d'un
    # cote rapporte a la distance au premier obstacle de l'autre, qui est ou
    # se placerait le stop. En dessous de 2, l'ouvrage dit de s'abstenir (p.82,
    # p.89) — c'est un refus d'entrer, pas un mauvais signal.
    pose("ich_ratio_haussier", np.clip(obst_h / np.maximum(obst_b, 0.05), 0, 6))
    pose("ich_ratio_baissier", np.clip(obst_b / np.maximum(obst_h, 0.05), 0, 6))

    # ---------- D. Chikou-Span, le filtre du systeme ----------
    ch_h, ch_b = _chikou_libre(cn, h.to_numpy(np.float64),
                               l.to_numpy(np.float64), a)
    pose("ich_chikou_haut", ch_h)
    pose("ich_chikou_bas", ch_b)
    # "Si Chikou-Span est libre de tout obstacle, elle valide le signal des
    # prix. Dans le cas contraire il faut s'abstenir" (p.32). Un demi-ATR de
    # marge de chaque cote.
    pose("ich_chikou_libre_h", (ch_h > 0.5).astype(float))
    pose("ich_chikou_libre_b", (ch_b > 0.5).astype(float))
    # Chikou au-dessus ou en dessous des prix qu'elle survole : la comparaison
    # la plus simple, et celle que l'oeil fait en premier.
    pose("ich_chikou_vs_prix",
         (cn - pd.Series(cn).shift(DECALAGE).to_numpy()) / a)

    # ---------- E. Le nuage : la boussole ----------
    pose("ich_kumo_epaisseur", (ssa - ssb) / a)
    pente_f = (_pente(ssa_f) + _pente(ssb_f)) / 2.0
    pose("ich_kumo_pente", (_pente(ssa) + _pente(ssb)) / 2.0 / a)
    pose("ich_kumo_futur_pente", pente_f / a)
    haut_kumo = np.maximum(ssa, ssb)
    bas_kumo = np.minimum(ssa, ssb)
    pose("ich_pos_kumo",
         np.where(cn > haut_kumo, (cn - haut_kumo) / a,
                  np.where(cn < bas_kumo, (cn - bas_kumo) / a, 0.0)))
    # "Si le nuage est oriente a la baisse, ne chercher QUE des ventes ; s'il
    # n'est pas oriente, faire du trading range et rien d'autre" (p.26). Le
    # regime commande la strategie : c'est la variable la plus structurante du
    # systeme, et le modele ne l'avait pas.
    seuil = 0.02
    pose("ich_regime", np.where(pente_f / a > seuil, 1.0,
                                np.where(pente_f / a < -seuil, -1.0, 0.0)))

    # ---------- F. Tendance : replis et cassures ----------
    # Repli sur plat Kijun : le prix revient toucher un plat Kijun en restant
    # du bon cote. "Il est prudent d'attendre au moins deux contacts qui
    # demontrent que le plat joue son role" (p.75) — d'ou le comptage.
    touche_k = (np.abs(cn - kijun) / a < 0.25) & plat_k
    pose("ich_repli_kijun", touche_k.astype(float))
    contacts = pd.Series(touche_k.astype(float)).rolling(
        KIJUN, min_periods=1).sum().to_numpy()
    pose("ich_contacts_kijun", np.clip(contacts, 0, 5))
    # "Si un bout de chandelier (niveau de CLOTURE) a casse la Kijun-Sen on ne
    # devrait pas entrer en position" (p.79). C'est la cloture qui tranche, pas
    # la meche — d'ou le corps et non les extremes.
    corps_bas = np.minimum(cn, o.to_numpy(np.float64))
    corps_haut = np.maximum(cn, o.to_numpy(np.float64))
    pose("ich_kijun_casse_h", ((corps_bas > kijun) & plat_k).astype(float))
    pose("ich_kijun_casse_b", ((corps_haut < kijun) & plat_k).astype(float))

    # Cassure de nouveaux plus hauts / plus bas, EN CLOTURE (p.81-82).
    plus_haut = h.rolling(KIJUN).max().shift(1).to_numpy(np.float64)
    plus_bas = l.rolling(KIJUN).min().shift(1).to_numpy(np.float64)
    pose("ich_cassure_haut", np.clip((cn - plus_haut) / a, -3, 3))
    pose("ich_cassure_bas", np.clip((plus_bas - cn) / a, -3, 3))
    # "À condition que les prix soient proches de la Tenkan-Sen" (p.88) : la
    # cassure ne vaut que si l'on n'est pas deja en surachat.
    pose("ich_cassure_prox_tenkan",
         np.clip(np.abs(cn - tenkan) / a, 0, 4))

    # ---------- G. Chandeliers ----------
    for nom, val in _chandeliers(df, a).items():
        pose(nom, np.asarray(val, np.float64))    # pose() ajoute le suffixe

    return df, cols


if __name__ == "__main__":
    import sys
    d = pd.read_pickle("data_cache_BTCUSD_H1.pkl")
    d, cols = ajoute_features_ichimoku(d)
    print(f"{len(cols)} colonnes ich_* produites sur {len(d):,} barres\n")
    fini = d[cols].replace([np.inf, -np.inf], np.nan)
    print(f"{'colonne':>26} {'NaN %':>7} {'moyenne':>10} {'ecart-type':>11} "
          f"{'min':>9} {'max':>9}")
    print("-" * 78)
    for k in cols:
        v = fini[k]
        print(f"{k:>26} {100*v.isna().mean():6.1f}% {v.mean():10.3f} "
              f"{v.std():11.3f} {v.min():9.2f} {v.max():9.2f}")
    mortes = [k for k in cols if fini[k].std() < 1e-9 or fini[k].isna().mean() > 0.5]
    print()
    print(f"colonnes constantes ou trop vides : {mortes if mortes else 'aucune'}")
    derive = (cols != COLONNES)
    if derive:
        print("DERIVE entre COLONNES et ce que le code produit :")
        print(f"  produites non listees : {[c for c in cols if c not in COLONNES]}")
        print(f"  listees non produites : {[c for c in COLONNES if c not in cols]}")
    else:
        print(f"COLONNES est a jour ({len(COLONNES)} noms, meme ordre)")
    sys.exit(1 if (mortes or derive) else 0)
