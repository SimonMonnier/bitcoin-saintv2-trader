"""Features de la strategie "trader les bornes d'un range" (Ichimoku, Riguet).

Tiree de la formation Ichimoku et de la formation chandeliers japonais. Les
definitions ci-dessous sont celles des documents, pas des approximations :

  TWIST        croisement de la SSA et de la SSB. Apres une tendance marquee,
               il annonce un range dans un pourcentage eleve de cas. Il ne
               signale PAS un retournement, seulement un essoufflement.

  SSB PLATE    "lors des 52 dernieres periodes il n'y a pas eu de nouveau plus
               haut ou plus bas". C'est la definition meme d'un range, et elle
               est directement calculable.

  SOULEVEMENT  chandelier qui fait une meche haute AU-DESSUS de la resistance
               et qui CLOTURE SOUS elle.

  RESSAUT      chandelier qui fait une meche basse SOUS le support et qui
               CLOTURE AU-DESSUS de lui. (aussi appele ressort)

  BORNES       on relie les CORPS des chandeliers en cloture, pas les meches —
               sinon on ne peut plus trader les ressauts et les soulevements.

  ZE           zone d'equilibre, milieu du range. Zone tres instable, on n'y
               entre pas.

LE POINT DE CONCEPTION LE PLUS IMPORTANT. Les documents insistent :

    "Le marteau n'a d'incidence, que et uniquement que, sur support."
    "Dans un range, les figures de retournement ne sont utiles que sur la
     borne haute et basse et sur la zone d'equilibre. Partout ailleurs,
     c'est strictement inutile."

Les figures de chandeliers ne sont donc PAS des features autonomes. Une colonne
"y a-t-il un marteau" calculee partout est du bruit sur 95 % des barres. Ce qui
porte l'information est le PRODUIT figure x localisation. Toutes les colonnes de
chandeliers de ce module sont donc conditionnees a la proximite d'une borne.

C'est probablement pour cela que les 30 features actuelles echouent : elles sont
toutes sans contexte.

RESERVE D'ECHELLE, a lire avant d'utiliser ce module. Les documents indiquent
"court terme : ranges en UT 15 min ; day trading : 15 min a 1H ; swing : H4 au
daily". La strategie n'est pas concue pour le M1. Mesure de cette nuit : a
SL 2xATR sur M1, la friction vaut 0.60 unite de risque — un stop structurel
serre y serait devore. Sur H1 l'ATR est environ huit fois plus grand et la meme
friction ne pese plus que ~0.08 R.
"""

import numpy as np
import pandas as pd

# Parametres Ichimoku standards.
TENKAN, KIJUN, SENKOU_B, DECALAGE = 9, 26, 52, 26

# Fenetre de detection des bornes du range, en barres.
FENETRE_RANGE = 52

# Tolerance de "touchette" et de proximite d'une borne, en fraction de la
# largeur du range. 5 % : assez large pour attraper une meche, assez etroit
# pour ne pas confondre avec le milieu.
TOL_BORNE = 0.05


def _milieu(h, l, p):
    """Milieu du canal haut/bas sur p periodes — la brique d'Ichimoku."""
    return (h.rolling(p).max() + l.rolling(p).min()) / 2.0


def ajoute_ichimoku_brut(df):
    """Tenkan, Kijun, SSA, SSB. Les Senkou sont decalees VERS L'AVANT.

    shift(+26) : la valeur affichee en t a ete CALCULEE en t-26. C'est donc de
    l'information retardee, jamais anticipee — contrairement a la Chikou, qui
    serait du futur si on l'alignait sur sa position dessinee.
    """
    h, l, c = df["high"], df["low"], df["close"]
    df["tenkan"] = _milieu(h, l, TENKAN)
    df["kijun"] = _milieu(h, l, KIJUN)
    df["ssa"] = ((df["tenkan"] + df["kijun"]) / 2.0).shift(DECALAGE)
    df["ssb"] = _milieu(h, l, SENKOU_B).shift(DECALAGE)
    return df


def ajoute_features_range(df, atr_col="atr_14", suffixe=""):
    """Ajoute les colonnes de la strategie. Rend la liste des noms crees."""
    s = suffixe
    df = ajoute_ichimoku_brut(df)
    h, l, c, o = df["high"], df["low"], df["close"], df["open"]
    atr = df[atr_col].replace(0, np.nan)
    corps = (c - o).abs()
    meche_h = h - np.maximum(c, o)
    meche_b = np.minimum(c, o) - l
    cols = []

    def pose(nom, serie):
        df[nom + s] = serie.replace([np.inf, -np.inf], np.nan)
        cols.append(nom + s)

    # ---------- A. Structure : sommes-nous en range ? ----------
    # Twist = croisement SSA/SSB.
    au_dessus = (df["ssa"] > df["ssb"]).astype(float)
    twist = au_dessus.diff().abs().fillna(0.0)
    # Recence du dernier twist, en decroissance douce : un twist d'il y a
    # 3 barres compte plus qu'un twist d'il y a 40.
    depuis = twist.copy()
    depuis[:] = np.nan
    idx_twist = np.flatnonzero(twist.to_numpy() > 0)
    age = np.full(len(df), np.nan)
    if len(idx_twist):
        pos = np.searchsorted(idx_twist, np.arange(len(df)), side="right") - 1
        valide = pos >= 0
        age[valide] = np.arange(len(df))[valide] - idx_twist[pos[valide]]
    pose("rng_twist_recence", pd.Series(np.exp(-age / 20.0), index=df.index))
    # Succession de twists = deriv latérale franche, dit le document.
    pose("rng_twists_50", twist.rolling(50).sum())

    # Nuage a plat : pente des deux bornes du nuage, rapportee a l'ATR.
    pose("rng_pente_ssb", (df["ssb"].diff(10) / (10 * atr)).abs())
    pose("rng_pente_ssa", (df["ssa"].diff(10) / (10 * atr)).abs())
    pose("rng_pente_kijun", (df["kijun"].diff(10) / (10 * atr)).abs())
    pose("rng_epaisseur_kumo", (df["ssa"] - df["ssb"]).abs() / atr)

    # SSB plate au sens du document : aucun nouveau plus haut ni plus bas sur
    # les 52 dernieres periodes. C'est la definition litterale du range.
    plus_haut = (h.rolling(SENKOU_B).max().diff() > 0)
    plus_bas = (l.rolling(SENKOU_B).min().diff() < 0)
    pose("rng_ssb_plate", (~(plus_haut | plus_bas)).astype(float))

    # ---------- B. Bornes ----------
    # On relie les CORPS en cloture, pas les meches : sinon les ressauts et
    # soulevements ne peuvent plus exister, puisque leurs meches definiraient
    # elles-memes la borne.
    haute = c.rolling(FENETRE_RANGE).max()
    basse = c.rolling(FENETRE_RANGE).min()
    largeur = (haute - basse)
    milieu = (haute + basse) / 2.0

    pose("rng_largeur_atr", largeur / atr)
    # Position dans le range, 0 = borne basse, 1 = borne haute.
    pose("rng_position", (c - basse) / largeur.replace(0, np.nan))
    pose("rng_dist_haute_atr", (haute - c) / atr)
    pose("rng_dist_basse_atr", (c - basse) / atr)
    pose("rng_dist_ze_atr", (c - milieu).abs() / atr)

    # Touchettes : le document signale que les 3e et 4e touchettes portent une
    # probabilite de cassure plus forte. On compte les contacts recents.
    pres_h = ((haute - h).abs() < TOL_BORNE * largeur).astype(float)
    pres_b = ((l - basse).abs() < TOL_BORNE * largeur).astype(float)
    pose("rng_touchettes_haute", pres_h.rolling(FENETRE_RANGE).sum())
    pose("rng_touchettes_basse", pres_b.rolling(FENETRE_RANGE).sum())

    # ---------- C. Signaux d'entree ----------
    # Soulevement : meche haute AU-DESSUS de la borne, cloture SOUS elle.
    soul = ((h > haute.shift(1)) & (c < haute.shift(1)))
    # Ressaut : meche basse SOUS la borne, cloture AU-DESSUS.
    ress = ((l < basse.shift(1)) & (c > basse.shift(1)))
    # En amplitude plutot qu'en binaire : la profondeur de la meche informe
    # sur la force du rejet.
    pose("rng_soulevement", (soul * (h - haute.shift(1)) / atr).fillna(0.0))
    pose("rng_ressaut", (ress * (basse.shift(1) - l) / atr).fillna(0.0))

    # "Si la borne a ete cassee et que le prix a reintegre le range, on ne peut
    # plus entrer en position." On marque donc cet etat plutot que de le subir.
    casse_h = (c > haute.shift(1)).astype(float)
    casse_b = (c < basse.shift(1)).astype(float)
    pose("rng_cassure_recente",
         (casse_h + casse_b).rolling(10).max().fillna(0.0))

    # ---------- D. Chandeliers, CONDITIONNES a la localisation ----------
    # Hors des bornes, ces figures sont "strictement inutiles" selon le
    # document. On les annule donc ailleurs plutot que de les laisser bruiter.
    sur_basse = ((c - basse) < TOL_BORNE * largeur).astype(float)
    sur_haute = ((haute - c) < TOL_BORNE * largeur).astype(float)

    petit_corps = corps < 0.3 * (h - l).replace(0, np.nan)
    marteau = petit_corps & (meche_b >= 2 * corps) & (meche_h <= corps)
    filante = petit_corps & (meche_h >= 2 * corps) & (meche_b <= corps)
    doji = corps < 0.1 * (h - l).replace(0, np.nan)

    haussier, baissier = (c > o), (c < o)
    prec_h, prec_b = h.shift(1), l.shift(1)
    prec_o, prec_c = o.shift(1), c.shift(1)
    aval_h = haussier & (o <= prec_c) & (c >= prec_o) & baissier.shift(1)
    aval_b = baissier & (o >= prec_c) & (c <= prec_o) & haussier.shift(1)
    # Penetrante : ouvre sous la cloture precedente, cloture au-dela du point
    # median de la bougie precedente.
    median_prec = (prec_o + prec_c) / 2.0
    penetrante = haussier & baissier.shift(1) & (o <= prec_c) & (c > median_prec)
    nuage_noir = baissier & haussier.shift(1) & (o >= prec_c) & (c < median_prec)

    pose("rng_marteau_basse", (marteau & (sur_basse > 0)).astype(float))
    pose("rng_filante_haute", (filante & (sur_haute > 0)).astype(float))
    pose("rng_doji_borne", (doji & ((sur_basse + sur_haute) > 0)).astype(float))
    pose("rng_aval_h_basse", (aval_h & (sur_basse > 0)).astype(float))
    pose("rng_aval_b_haute", (aval_b & (sur_haute > 0)).astype(float))
    pose("rng_penetrante_basse", (penetrante & (sur_basse > 0)).astype(float))
    pose("rng_nuage_noir_haute", (nuage_noir & (sur_haute > 0)).astype(float))

    # ---------- E. Confirmation ----------
    bas_n = l.rolling(14).min()
    haut_n = h.rolling(14).max()
    stoch = 100 * (c - bas_n) / (haut_n - bas_n).replace(0, np.nan)
    pose("rng_stoch", stoch / 100.0)

    return df, cols


GROUPES = {
    "structure": ["rng_twist_recence", "rng_twists_50", "rng_pente_ssb",
                  "rng_pente_ssa", "rng_pente_kijun", "rng_epaisseur_kumo",
                  "rng_ssb_plate"],
    "bornes": ["rng_largeur_atr", "rng_position", "rng_dist_haute_atr",
               "rng_dist_basse_atr", "rng_dist_ze_atr",
               "rng_touchettes_haute", "rng_touchettes_basse"],
    "signaux": ["rng_soulevement", "rng_ressaut", "rng_cassure_recente"],
    "chandeliers": ["rng_marteau_basse", "rng_filante_haute", "rng_doji_borne",
                    "rng_aval_h_basse", "rng_aval_b_haute",
                    "rng_penetrante_basse", "rng_nuage_noir_haute"],
    "confirmation": ["rng_stoch"],
}
