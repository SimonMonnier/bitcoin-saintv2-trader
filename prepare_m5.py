"""Construit le jeu d'entrainement M5 — neuf ans, 260 colonnes, trois echelles.

POURQUOI LE M5. Le mur de ce depot n'a jamais ete l'architecture ni
l'algorithme : c'est le nombre d'occasions INDEPENDANTES. Une occasion n'est
independante de la suivante que si leurs fenetres de resultat ne se recouvrent
pas, donc leur nombre vaut la duree d'historique divisee par la duree d'un
trade — jamais par le nombre de barres.

Mesure du 2026-09-16, sortie sur barriere UNIQUEMENT, comme l'environnement et
comme le live :

    UT   SL   friction/R   E[R] hasard   duree med   occasions   requis
    M1  8.0      0.132R       -0.1136        122b      10 472   +0.1136R
    M5  4.0      0.099R       -0.0892         44b       5 811   +0.0892R   <-
    M5  8.0      0.049R       -0.0425        147b       1 739   +0.0425R
   M15  2.0      0.105R       -0.1018         13b       6 559   +0.1018R
    H1  2.0      0.047R       -0.0233         13b       1 640   +0.0233R

ATTENTION : CES OCCASIONS SONT COMPTEES SUR 3.6 ANS, la duree du cache M1 qui
a servi a la mesure. Le jeu M5 en couvre 9.05, soit 2.6 fois plus. Toute
comparaison entre lignes doit se faire APRES cette mise a l'echelle — les
melanger a fait ecarter le 8xATR pour 1 739 occasions alors qu'il en vaut
4 516, et retenir le 4xATR dont la friction est deux fois plus lourde.

    config         friction   horizon   occasions 9 ans   avantage requis
    H1  SL 2xATR    0.047 R     13.0 h            2 314        +0.0233 R
    M5  SL 4xATR    0.099 R      3.7 h           15 089        +0.0892 R
    M5  SL 8xATR    0.049 R     12.2 h            4 516        +0.0425 R   <-

C'est le 8xATR qui est retenu : meme friction et meme horizon que le H1 — donc
le meme probleme de prediction, celui sur lequel l'avantage a ete mesure — pour
deux fois plus d'occasions independantes.

CE QUE LE M1 NE PEUT PAS FAIRE, et pourquoi on s'arrete au M5. Meme avec un
stop de 8xATR, le M1 exige +0.114 R d'avantage — davantage que tout ce que ce
depot a jamais mesure. La friction est un montant FIXE en dollars ; a cette
echelle elle devore une part du risque qu'aucun modele ne rattrape.

TROIS PIEGES D'ECHELLE, tous deja payes une fois ici.

  1. LES FENETRES "JOURNEE" SE COMPTENT EN BARRES. Une journee vaut 288 barres
     en M5, 24 en H1, 1440 en M1. `add_indicators` codait 1440 en dur, ce qui
     faisait porter a des colonnes un nom qui ne decrivait plus leur contenu.
     Toutes les fenetres sont donc passees en parametre.

  2. LE CONTEXTE SUPERIEUR EST DECALE. Le bloc H1 est decale de shift(1) :
     seule la derniere bougie H1 CLOSE entre. Sans ce decalage, merge_asof
     choisit la bougie en formation et injecte jusqu'a une heure de futur par
     ligne. test_causalite.py le verifie desormais, il ne se contente plus de
     le commenter.

  3. LES PERIODES ICHIMOKU NE SE CONVERTISSENT PAS. Tenkan 9, Kijun 26,
     Senkou-B 52 sont des nombres de BOUGIES, identiques a toutes les echelles
     — c'est le principe meme du systeme. On ne les multiplie pas par 12.

POURQUOI LES DEUX ECHELLES, et pas seulement le M5 (2026-09-16). Le passage au
M5 achetait 6.5 fois plus d'occasions, mais exec22 n'apprenait plus rien : huit
epochs entrainees, zero positive, gain nul sur sa propre politique gelee, alors
que la mecanique etait reparee (entropie 1.099 -> 0.996, etendue 0.53,
clipfrac 18 %). Or le point 3 ci-dessus dit exactement pourquoi : les memes
periodes ne decrivent plus les memes structures. Tenkan 9 vaut 45 minutes en M5
contre neuf heures en H1 — et c'est en H1 que l'avantage avait ete mesure.

On ne rend donc pas les occasions : on AJOUTE le range et l'Ichimoku calcules
sur les resamples H1 ET H4 (85 colonnes chacun), decales du meme shift(1)
que les indicateurs de base. La finesse du M5 pour decider, les structures H1 la ou
le signal avait ete trouve. C'est aussi ce que prescrit le livre : valider un
signal en basculant sur les unites de temps superieures.

    python prepare_m5.py
"""

import numpy as np
import pandas as pd

from features_ichimoku import ajoute_features_ichimoku
from features_range import ajoute_features_range
from prepare_h1 import BASES, indicateurs

SOURCE = "klines_5m_spot_BTCUSDT.pkl"
SORTIE = "data_cache_BTCUSD_M5.pkl"

# Une journee et une semaine, en barres de cinq minutes.
JOUR_M5 = 288
SEMAINE_M5 = 288 * 7
# CONTEXTES SUPERIEURS : (regle de resample, suffixe, barres par JOURNEE).
#
# Le troisieme nombre n'est pas un facteur de conversion : c'est le nombre de
# bougies dans une journee A CETTE ECHELLE, que `indicateurs` utilise pour ses
# fenetres "journalieres". Le coder en dur ailleurs est l'erreur deja payee une
# fois ici (piege 1 ci-dessus).
#
# Deux echelles et pas une : un trade M5 dure 3 h 40 en mediane, donc le H1
# decrit le mouvement qui CONTIENT le trade et le H4 le regime qui contient ce
# mouvement. Ajouter le H4 ne coute aucun telechargement — il se resample du
# meme M5 — et 85 colonnes, soit 13 000 parametres de plus sur ~15 100
# occasions independantes.
ECHELLES_SUP = [("1h", "_h1", 24), ("4h", "_h4", 6)]
# Compatibilite : plusieurs fichiers nomment encore le contexte immediat.
REGLE_SUP, SUFFIXE_SUP, JOUR_SUP = ECHELLES_SUP[0]


def flux(d, jour=JOUR_M5, semaine=SEMAINE_M5):
    """Flux d'ordres, ecrit pour rester comparable de 2017 a 2026.

    Aucune colonne n'est un niveau : la part acheteuse est un rapport, la
    taille de trade est rapportee a sa normale de la journee, l'intensite est
    un RANG sur la semaine. Sur un historique ou le volume par barre a change
    de plusieurs ordres de grandeur, tout niveau absolu apprendrait l'annee.
    """
    vol = d["volume"].replace(0, np.nan)
    d["taker_ratio"] = (d["taker_buy_base"] / vol).clip(0, 1)
    d["taker_ma5"] = d["taker_ratio"].rolling(5, min_periods=1).mean()
    taille = d["quote_vol"] / d["nb_trades"].replace(0, np.nan)
    lt = np.log(taille.clip(lower=1e-9))
    d["flux_taille_trade"] = lt - lt.rolling(jour, min_periods=jour // 4).mean()
    d["flux_intensite"] = (d["nb_trades"]
                           .rolling(semaine, min_periods=jour).rank(pct=True))
    return d


def _joint_echelle(m5, regle, sfx, jour_sup):
    """Calcule le jeu COMPLET a une echelle superieure et le colle aux lignes M5.

    Rend (df, noms des colonnes ajoutees). Les memes 85 colonnes qu'en M5 a
    leur suffixe pres — les periodes Ichimoku etant des nombres de bougies,
    elles ne decrivent pas du tout les memes structures.
    """
    sup = (m5.set_index("time")[["open", "high", "low", "close"]]
             .resample(regle)
             .agg({"open": "first", "high": "max", "low": "min",
                   "close": "last"})
             .dropna().reset_index())
    sup = indicateurs(sup, jour_sup)
    sup, cols_rng = ajoute_features_range(sup, suffixe=sfx)
    sup, cols_ich = ajoute_features_ichimoku(sup, suffixe=sfx)

    # Les colonnes de range et d'Ichimoku portent DEJA leur suffixe : les
    # fonctions le posent a la source. Seules les BASES doivent encore le
    # recevoir, d'ou deux listes — `bloc` nomme les colonnes telles qu'elles
    # existent dans `sup`, `noms` telles qu'elles s'appelleront une fois jointes.
    bloc = BASES + cols_rng + cols_ich
    noms = [c + sfx for c in BASES] + cols_rng + cols_ich
    dec = sup[["time"] + bloc].copy()
    # SHIFT(1) SUR TOUT LE BLOC, pas seulement sur les indicateurs de base.
    # merge_asof(backward) choisit la bougie superieure qui CONTIENT l'instant
    # M5 courant, donc une bougie encore en formation. Sans ce decalage, chaque
    # ligne recevrait jusqu'a quatre heures de futur en H4 — et aucune erreur
    # ne serait levee. test_causalite.py le verifie desormais.
    dec[bloc] = dec[bloc].shift(1)
    dec.columns = ["time"] + noms
    dec["_c_sup"] = sup["close"].shift(1).values
    m5 = pd.merge_asof(m5.sort_values("time"), dec.sort_values("time"),
                       on="time", direction="backward")
    m5["close" + sfx + "_dev"] = m5["close"] / (m5["_c_sup"] + 1e-8) - 1.0
    return m5.drop(columns=["_c_sup"]), noms + ["close" + sfx + "_dev"]


def construit(m5, avec_flux=None):
    """Du brut M5 aux colonnes de `saint_core`. Rend (df, colonnes, ichimoku).

    `avec_flux` calcule les colonnes de carnet — part acheteuse agressive,
    taille de trade, intensite. Elles n'existent QUE sur Binance : aucun CFD
    ne publie `taker_buy_base` ni `nb_trades`. Elles ont ete retirees de
    `FEATURE_COLS` le 2026-09-16 pour que l'or puisse partager exactement le
    meme jeu de colonnes, et la mesure a montre que ca ne coutait rien.

    Par defaut on les calcule si la source les porte, et on les laisse hors
    de la liste rendue : elles restent dans le cache pour qui voudrait les
    remesurer, sans entrer dans l'observation.
    """
    m5 = indicateurs(m5, JOUR_M5)
    if avec_flux is None:
        avec_flux = all(c in m5.columns
                        for c in ("taker_buy_base", "quote_vol", "nb_trades"))
    if avec_flux:
        m5 = flux(m5)

    cols_sup = []
    for regle, sfx, jour_sup in ECHELLES_SUP:
        m5, noms = _joint_echelle(m5, regle, sfx, jour_sup)
        cols_sup += noms

    # ---------- Temps ----------
    heure = m5["time"].dt.hour + m5["time"].dt.minute / 60.0
    m5["heure_sin"] = np.sin(2 * np.pi * heure / 24.0)
    m5["heure_cos"] = np.cos(2 * np.pi * heure / 24.0)

    m5, cols_rng = ajoute_features_range(m5)
    m5, cols_ich = ajoute_features_ichimoku(m5)

    # Les colonnes de carnet ne sont plus dans l'observation : la liste
    # rendue doit donc s'accorder avec `saint_core.FEATURE_COLS`, que `main`
    # verifie colonne par colonne.
    liq = ["heure_sin", "heure_cos"]
    return (m5, BASES + cols_sup + liq + cols_rng + cols_ich, cols_ich)


def main() -> int:
    src = pd.read_pickle(SOURCE)
    src["time"] = pd.to_datetime(src["time"])
    m5 = src.sort_values("time").reset_index(drop=True)
    print(f"source : {len(m5):,} bougies M5  "
          f"{m5['time'].iloc[0]} -> {m5['time'].iloc[-1]}")

    m5, colonnes, cols_ich = construit(m5)

    avant = len(m5)
    m5 = m5.replace([np.inf, -np.inf], np.nan)
    m5 = m5.dropna(subset=colonnes + ["atr_14"]).reset_index(drop=True)
    print(f"M5     : {avant:,} -> {len(m5):,} bougies apres dropna "
          f"({100*(1-len(m5)/avant):.1f} % perdues au warmup)")
    print(f"periode: {m5['time'].iloc[0]} -> {m5['time'].iloc[-1]}")

    # La liste de saint_core fait foi. Les noms du contexte superieur changent
    # d'echelle en echelle (_h4 en H1, _h1 en M5) : un desaccord ici se
    # traduirait plus tard par un KeyError peu lisible en plein entrainement.
    from saint_core import FEATURE_COLS
    manquantes = [c for c in FEATURE_COLS if c not in m5.columns]
    en_trop = [c for c in colonnes if c not in FEATURE_COLS]
    if manquantes or en_trop:
        print(f"\nDESACCORD avec saint_core.FEATURE_COLS "
              f"({len(FEATURE_COLS)} colonnes attendues)")
        print(f"  absentes du jeu   : {manquantes[:8]}")
        print(f"  produites en trop : {en_trop[:8]}")
        return 1
    print(f"accord avec saint_core.FEATURE_COLS ({len(FEATURE_COLS)} colonnes)")

    atr_med = float(m5["atr_14"].median())
    prix_med = float(m5["close"].median())
    fric = (2.61 + 1.0 + 2.0) / 1e4 * prix_med
    print(f"\nATR median {atr_med:.2f} $  |  friction {fric:.2f} $")
    print(f"  a SL 4xATR : {fric/(4*atr_med):.3f} R par trade "
          f"(0.047 R en H1 a SL 2xATR)")

    # FLOAT32 SUR LES FEATURES. Toutes sont des rapports, des ecarts normalises
    # ou des rangs : aucune ne porte de grandeur qui demande 15 chiffres
    # significatifs. Le jeu passe de 2.1 a 1.1 Go, ce qui laisse la place de
    # faire tourner l'entrainement, la veille et le test de causalite ensemble
    # sur 16 Go. Les prix et l'ATR restent en float64 : eux servent a calculer
    # des barrieres au dollar pres.
    for c in colonnes:
        m5[c] = m5[c].astype(np.float32)

    m5.to_pickle(SORTIE)
    print(f"\n{SORTIE} ecrit : {m5.shape}")

    n_tr = int(len(m5) * 0.70)
    print(f"\nDIMENSIONNEMENT : {len(m5):,} barres, {n_tr:,} en entrainement.")
    print(f"  occasions independantes (duree mediane 44 barres) : "
          f"~{n_tr // 44:,}   (H1 : 2 314)")
    print(f"  il en faut ~4 900 pour qu'un avantage de 0.02 R soit lisible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
