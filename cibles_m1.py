# -*- coding: utf-8 -*-
"""Les cibles du scalping M1 : acheter, vendre, cloturer.

CE QUI CHANGE PAR RAPPORT A `cibles.py`, ET POURQUOI IL A FALLU UN AUTRE
FICHIER. L'ancien monde avait un STOP : il definissait `R`, l'unite dans
laquelle tout se mesurait, et il bornait la perte d'une position. Le
scalping M1 n'en a pas. Deux consequences qui touchent tout le reste :

  L'UNITE DEVIENT LE POINT DE BASE DE RENDEMENT NET. Sans stop, `R` n'a
  plus de denominateur. On mesure donc en bps du prix d'entree, friction
  deduite — une unite qui a un sens sans stop et qui se convertit
  directement en argent : 1 bps sur 1 000 $ vaut 0.10 $.

  LA PERTE N'EST PLUS BORNEE. C'est la tete de cloture qui ferme, et si
  elle se trompe rien ne la rattrape. `creux_pendant` mesure donc, pour
  chaque occasion, la PIRE perte latente traversee avant la sortie. C'est
  l'information qui remplace le stop pour dimensionner les positions.

TROIS CIBLES, ET LEURS COUTS NE SONT PAS LES MEMES.

  ACHAT et VENTE : on paie l'aller ET le retour, puisque decider d'entrer
  engage les deux. La cible est donc NETTE du cout complet.

  CLOTURE : le cout de sortie sera paye de toute facon, maintenant ou plus
  tard. Il s'annule entre « fermer ici » et « tenir encore ». La cible de
  cloture est donc BRUTE — et l'oublier ferait fermer trop tot, en
  facturant deux fois un passage qu'on ne paie qu'une.

L'HORIZON EST UN PARAMETRE, ET SON CHOIX EST MESURE. `mesure_horizon_m1.py`
donne, pour chaque horizon, la fraction du plafond qu'il faut atteindre pour
seulement rentrer dans ses frais :

    horizon    hasard   plafond   pt mort   trades/j   a 16 % de competence
     1 min      -4.35     +4.16      51 %      968.3        -2895 bps/jour
     5 min      -4.32    +14.27      23 %      193.6         -260
    10 min      -4.27    +21.81      16 %       96.8          -10
    30 min      -4.10    +40.79       9 %       32.3          +99
    60 min      -3.85    +59.43       6 %       16.1         +101

En dessous de dix minutes, il faut etre exceptionnel pour ne pas perdre.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_DEFAUT = 30          # barres M1, soit trente minutes


def cout_par_barre(df, idx, symbole: str = "XAUUSD"):
    """L'aller-retour de CHAQUE occasion, au spread qui y etait cote.

    POURQUOI UNE VALEUR PAR BARRE, ET PLUS UNE CONSTANTE. Le 2026-09-21,
    le preparateur s'est mis a garder la colonne `spread_bar` que
    `copy_rates` renvoyait depuis toujours et que le depot jetait. Le
    simulateur la lit desormais pour facturer chaque trade — voir
    `training.BTCTradingEnvDiscrete._sample_trade_spread_bps`. Si
    l'etiquette restait sur une moyenne, les deux comptables
    diverger aient a nouveau, et c'est la faute que ce fichier documente le
    plus souvent.

    CE QUE LA CONSTANTE COUTAIT. Mesure sur les 900 000 barres M1 :

        spread reel        median 0.563 bps   p90 0.770   p99 0.860
        loi simulee        esperance 0.887    queue jusqu'a 1.92

    La loi venait des logs BTCUSD et etait appliquee a l'or. Elle
    facturait environ 1.6 fois le spread reel.

    LE SPREAD EST CELUI DE LA BARRE PRECEDENTE : c'est ce qu'un ordre
    passe a l'ouverture rencontre, et c'est causal.

    REPLI SUR LA CONSTANTE quand la colonne manque, pour que les caches
    anterieurs restent lisibles.
    """
    import numpy as _np
    if "spread_bar" not in getattr(df, "columns", ()):
        return cout_aller_retour(symbole)
    import instruments as I
    sp = df["spread_bar"].to_numpy(_np.float64)
    i = _np.clip(_np.asarray(idx, dtype=_np.int64) - 1, 0, len(sp) - 1)
    v = sp[i]
    moyen = float(I.spread_espere(symbole))
    v = _np.where(_np.isfinite(v) & (v > 0.0), v, moyen)
    gliss = float(I.INSTRUMENTS[symbole]["entry_slippage_bps"]) / 2.0
    # LE SPREAD COMPTE DEUX FOIS, LE GLISSEMENT UNE : meme geometrie que
    # `instruments.cout_aller_retour_espere`, et pour la meme raison — le
    # simulateur tire le spread une fois par trade et le facture a
    # l'ouverture comme a la fermeture.
    return 2.0 * v + gliss


def cout_aller_retour(symbole: str = "XAUUSD") -> float:
    """Ce que coute un trade complet. UNE SEULE SOURCE : `instruments`.

    CETTE FONCTION CALCULAIT SON PROPRE CHIFFRE, ET IL ETAIT FAUX.
    Elle valait `2 x spread_bps + GLISSEMENT_BPS` avec `GLISSEMENT_BPS`
    pose a 3.0, soit 4.36 bps la ou le simulateur en prend 2.27 — un
    facteur 1.92. Les etiquettes apprenaient aux tetes a eviter un cout
    que le compte ne paie pas, et 5.8 % des occasions d'achat etaient
    declarees perdantes alors qu'elles paient.

    LE CALCUL VIT DESORMAIS DANS `instruments.cout_aller_retour_espere`,
    avec la loi complete — bimodalite du spread, glissement a l'entree
    seulement. Voir la-bas pour la mesure. Deux comptables pour le meme
    trade, c'est exactement la faute que ce depot passe son temps a payer.
    """
    import instruments as I
    return I.cout_aller_retour_espere(symbole)


def _bornes(n: int, idx: np.ndarray, horizon: int):
    """Les occasions dont l'horizon tient dans les donnees."""
    idx = np.asarray(idx, dtype=np.int64)
    ok = (idx >= 0) & (idx + horizon < n)
    return idx[ok], ok


def rendements_entree(df: pd.DataFrame, idx, horizon: int = HORIZON_DEFAUT,
                      symbole: str = "XAUUSD"):
    """Ce que rapporte une entree en `idx`, fermee `horizon` barres plus tard.

    Rend `(achat, vente, creux_achat, creux_vente)`, tous en points de base
    et alignes sur `idx`. Les occasions dont l'horizon depasse les donnees
    rendent NaN — a masquer, jamais a remplir par zero : un zero serait une
    prediction, pas une absence.

    LES DEUX COTES SONT RENDUS ENSEMBLE parce qu'ils se deduisent du meme
    parcours de prix, et parce que la tete d'achat et celle de vente doivent
    etre entrainees sur des cibles exactement symetriques. Les calculer
    separement, c'est se donner deux occasions de diverger.
    """
    c = df["close"].to_numpy(np.float64)
    h_ = df["high"].to_numpy(np.float64)
    l_ = df["low"].to_numpy(np.float64)
    n = len(c)
    idx_ok, masque = _bornes(n, idx, horizon)

    ach = np.full(len(masque), np.nan)
    ven = np.full(len(masque), np.nan)
    cr_a = np.full(len(masque), np.nan)
    cr_v = np.full(len(masque), np.nan)
    if len(idx_ok) == 0:
        return ach, ven, cr_a, cr_v

    # LE COUT EST CELUI DE CHAQUE BARRE, plus une moyenne. Le
    # simulateur lit le spread cote ; l'etiquette doit lire le meme, sinon
    # les deux comptables divergent — voir `cout_par_barre`.
    cout = cout_par_barre(df, idx_ok, symbole)
    entree = c[idx_ok]
    sortie = c[idx_ok + horizon]
    mv = (sortie - entree) / entree * 1e4          # brut, en bps

    ach[masque] = mv - cout
    ven[masque] = -mv - cout

    # LE CREUX TRAVERSE, ET C'EST LUI QUI REMPLACE LE STOP. Sans borne de
    # perte, ce qui compte n'est pas seulement l'arrivee mais le pire point
    # du chemin : c'est lui qui declenche un appel de marge, et lui qu'un
    # humain ne supporte pas. On le mesure sur les EXTREMES des bougies, pas
    # sur les cloturess — une meche traverse le compte aussi bien qu'un corps.
    pires_bas = np.empty(len(idx_ok))
    pires_haut = np.empty(len(idx_ok))
    for j, i in enumerate(idx_ok):
        fenetre = slice(i + 1, i + horizon + 1)
        pires_bas[j] = l_[fenetre].min()
        pires_haut[j] = h_[fenetre].max()
    # Un long souffre du plus bas, un short du plus haut.
    cr_a[masque] = np.minimum((pires_bas - entree) / entree * 1e4, 0.0)
    cr_v[masque] = np.minimum((entree - pires_haut) / entree * 1e4, 0.0)
    return ach, ven, cr_a, cr_v


def cible_risque(df: pd.DataFrame, idx, sens, atr_entree,
                 horizon: int = HORIZON_DEFAUT):
    """CE QUE TENIR PEUT ENCORE COUTER : la pire excursion a venir.

    Rend un nombre POSITIF, en unites de l'ATR d'entree — la meme unite que
    `latent_atr` dans l'etat, pour que la regle de sortie puisse les
    comparer sans conversion.

    POURQUOI CETTE CIBLE A REMPLACE LE SIGNE DU MOUVEMENT, et c'est une
    mesure qui l'a impose, le 2026-09-21.

    L'ancienne cible etait `cible_cloture` : « gagne-t-on encore quelque
    chose en tenant trente minutes de plus ». La tete apprenait son SIGNE.
    Quatre epochs de suite elle est restee au hasard — juste 50.8 %, 49.2 %,
    53.1 %, 46.9 % — et sa perte collee a 0.6932, soit ln(2) a la quatrieme
    decimale.

    CE N'ETAIT PAS UN DEFAUT D'APPRENTISSAGE. La cible elle-meme est une
    piece de monnaie : sur 120 000 etats, 50.0 % de positifs. Trois tests
    independants, sur occasions DISJOINTES et contre un plancher par
    rotation :

        les 259 colonnes une par une   0 au-dessus du plancher
        une combinaison (ridge)        +0.99 bps contre un plancher +2.29
        le profit latent, par tranches aucune des 8 a 2 sigmas

    Le SENS du prix a trente minutes n'est pas dans ces donnees.

    L'AMPLITUDE, ELLE, Y EST — ET FORTEMENT. Meme protocole :

        le signe du mouvement      rho 0.0325   plancher 0.0186   1.8x
        l'amplitude parcourue      rho 0.6248   plancher 0.1357   4.6x
        le creux d'un long         rho 0.3169   plancher 0.0665   4.8x

    `vol_20` predit l'etendue des trente prochaines minutes a 0.625 de
    correlation de rang, 0.637 hors echantillon. C'est le signal le plus
    fort mesure dans ce depot.

    ET C'EST EXACTEMENT CE QU'IL FAUT POUR DECIDER. La question n'est pas
    « le prix va-t-il monter », a laquelle rien ne repond, mais « combien
    ca peut me coûter de rester », a laquelle la volatilite repond. Une
    position tres en gain devant un marche calme peut attendre ; la meme
    en perte devant un marche qui s'ouvre doit sortir. La regle de sortie
    compare `latent_atr` a ce risque — voir `training.demande_cloture`.

    MESURE DE LA REGLE, hors echantillon, 2 355 positions a entree
    aleatoire :

        horizon fixe 30 min                    moy -3.894  sd 15.77
        stop fixe -15 bps                      moy -3.839  sd 13.52
        coupe si perte > 0.50 x risque predit  moy -3.745  sd 13.05

    Elle bat le stop fixe sur la moyenne ET sur la dispersion. Sur des
    entrees aleatoires aucune regle ne peut rendre l'esperance positive —
    c'est une martingale — mais elle coupe la queue et la variance.
    """
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    n = len(c)
    idx_ok, masque = _bornes(n, idx, horizon)
    out = np.full(len(masque), np.nan)
    if len(idx_ok) == 0:
        return out
    sg = np.asarray(sens, dtype=np.float64)
    sg = sg[masque] if sg.ndim else np.full(len(idx_ok), float(sg))
    at = np.asarray(atr_entree, dtype=np.float64)
    at = at[masque] if at.ndim else np.full(len(idx_ok), float(at))
    at = np.maximum(at, 1e-9)

    # LA PIRE EXCURSION DANS LE SENS QUI FAIT MAL. Un long souffre des plus
    # BAS, un short des plus HAUTS. On balaie par fenetre glissante plutot
    # qu'en boucle : `horizon` est petit, la fenetre tient en memoire.
    ici = c[idx_ok - 1]
    pire = np.empty(len(idx_ok))
    for k in range(len(idx_ok)):
        i = idx_ok[k]
        if sg[k] > 0:
            pire[k] = ici[k] - l[i:i + horizon].min()
        else:
            pire[k] = h[i:i + horizon].max() - ici[k]
    # POSITIF PAR CONSTRUCTION : un marche qui ne va jamais contre nous
    # rend zero, pas un nombre negatif. La tete predit une AMPLITUDE.
    out[masque] = np.maximum(pire, 0.0) / at
    return out


def cible_profit(df: pd.DataFrame, idx, sens, atr_entree,
                 horizon: int = HORIZON_DEFAUT):
    """CE QU'IL RESTE A PRENDRE, en ATR d'entree. Positif ou nul.

    POURQUOI CETTE CIBLE EXISTE. Le 2026-09-22, la lecture de
    `training.demande_cloture` a montre que le systeme n'a AUCUNE prise de
    profit :

        seuil = marge + coupe x risque          (toujours >= 0.05)
        fermer si  latent < -max(seuil, 0.05)

    Le seuil est toujours positif et la condition exige un latent NEGATIF :
    une position en gain ne peut jamais declencher de cloture. La seule
    sortie d'un gagnant est le plafond de detention — mesure sur la regle
    telle qu'elle est codee, mediane ET moyenne de tenue des gagnants
    egales au plafond, sans une exception.

    CE QUE LE PLAFOND COUTE. Le sommet du rebond tombe a 49 minutes au
    dixieme centile et a 2 770 au quatre-vingt-dixieme — ecart-type 991.
    Un plafond constant ferme tous les gagnants au meme instant. Un oracle
    parfait ajouterait +246.84 bps par trade.

    LA CIBLE EST UNE AMPLITUDE, ET CE N'EST PAS NEGOCIABLE. Ce depot porte
    l'echec d'une cible SIGNEE — la tete apprenait « gagne-t-on encore en
    tenant » et rendait 50.8 %, 49.2 %, 53.1 %, 46.9 % sur quatre epochs,
    parce que la cible elle-meme est a 50.0 % de positifs. On demande donc
    COMBIEN, jamais DANS QUEL SENS :

        max(chemin futur) - valeur courante,  en ATR d'entree,  >= 0

    Meme famille que `cible_risque`, qui s'apprend a rho 0.63 hors
    echantillon la ou le signe est une piece de monnaie.

    CE QU'ELLE SE LAISSE PREDIRE, mesure du 2026-09-22 avec les seules
    quatre colonnes que l'etat porte — latent, age, `creux_rang`,
    `flux_rang` — hors echantillon, plancher par rotation :

        les 4 colonnes      IC +0.1523   plancher 0.0581   47 arbres
        latent + age        IC +0.1038   plancher 0.0554
        latent seul         IC +0.0436   plancher 0.0459   au plancher
        age seul            IC -0.0007   plancher 0.0178   au plancher

    AUCUNE COLONNE NE PORTE LE SIGNAL SEULE. C'est une interaction, et
    c'est exactement ce qui justifie une tete apprise plutot qu'un seuil.

    L'ATR EST CELUI DE L'ENTREE, comme pour `cible_risque` : les deux
    grandeurs doivent etre dans la meme unite pour que la regle de sortie
    les compare sans conversion.
    """
    c = df["close"].to_numpy(np.float64)
    n = len(c)
    idx = np.asarray(idx, dtype=np.int64)
    sens = np.asarray(sens, dtype=np.float64)
    atr_entree = np.maximum(np.asarray(atr_entree, dtype=np.float64), 1e-9)
    h = int(horizon)
    out = np.zeros(len(idx), dtype=np.float64)
    for k in range(len(idx)):
        i = int(idx[k])
        j = min(i + h + 1, n)
        if j <= i:
            continue
        fen = c[i:j]
        # LE CHEMIN INCLUT LA BARRE COURANTE, donc le reste est >= 0 par
        # construction : au pire on ne fait pas mieux que maintenant.
        extreme = fen.max() if sens[k] > 0 else fen.min()
        out[k] = max(sens[k] * (extreme - c[i]) / atr_entree[k], 0.0)
    return out


def cible_cloture(df: pd.DataFrame, idx, sens, horizon: int = HORIZON_DEFAUT):
    """Ce qu'on gagne ENCORE en tenant `horizon` barres de plus, en BRUT.

    `sens` vaut +1 pour une position longue, -1 pour une courte, et peut
    etre un scalaire ou un tableau aligne sur `idx`.

    POURQUOI EN BRUT, ET C'EST LE POINT DELICAT. Le cout de sortie sera paye
    de toute facon — maintenant ou dans trente minutes. Il est donc IDENTIQUE
    dans les deux branches de la decision et s'annule. Le facturer ici
    ferait fermer trop tot : le modele croirait payer un peage qu'il a deja
    engage en entrant.

    LE COUT D'ENTREE N'APPARAIT PAS NON PLUS : il est deja depense quand
    cette question se pose. C'est un cout irrecuperable, et le faire entrer
    dans la decision de sortie est l'erreur classique — on garderait une
    position perdante pour « amortir » un passage deja paye.

    UNE VALEUR NEGATIVE DIT DE FERMER. Positive, de tenir.
    """
    c = df["close"].to_numpy(np.float64)
    n = len(c)
    idx_ok, masque = _bornes(n, idx, horizon)
    out = np.full(len(masque), np.nan)
    if len(idx_ok) == 0:
        return out
    s = np.asarray(sens, dtype=np.float64)
    s = s[masque] if s.ndim else np.full(len(idx_ok), float(s))
    ici = c[idx_ok]
    plus_tard = c[idx_ok + horizon]
    out[masque] = s * (plus_tard - ici) / ici * 1e4
    return out


def echantillon_cloture(df: pd.DataFrame, n: int, horizon: int = HORIZON_DEFAUT,
                        tenue_max: int = None, graine: int = 0):
    """Des etats EN POSITION, avec leur cible de cloture.

    POURQUOI IL FAUT LES FABRIQUER. La tete de cloture repond a « faut-il
    fermer cette position ? ». Cette question n'a de sens que sur un etat ou
    une position EXISTE — avec son sens, son gain latent, son age. Or la
    grille dense ne contient que des etats PLATS : elle a ete batie pour les
    tetes d'entree, qui decident a plat.

    ON NE PASSE PAS PAR LE SIMULATEUR. Un etat en position se decrit
    entierement par (barre d'entree, sens, barres tenues) — le reste se
    deduit des prix. On tire donc ces trois nombres et on calcule le reste,
    au lieu de jouer des episodes pour en recolter quelques-uns. C'est plus
    rapide de plusieurs ordres de grandeur, et surtout c'est INDEPENDANT DE
    LA POLITIQUE : la tete apprend sur des positions que le modele courant
    n'aurait pas forcement prises, donc elle ne s'enferme pas dans ses
    propres habitudes.

    C'est le meme principe que la grille dense pour les tetes d'entree, et
    la meme raison : `_gr_idx` est INDEPENDANTE de la politique.

    CE QUE LA FONCTION REND, aligne sur `n` lignes :

        idx         la barre OU L'ON DECIDE (entree + tenue)
        sens        +1 long, -1 short                      -> colonne 0
        latent_atr  gain non realise, en ATR DE L'ENTREE    -> colonne 1
        tenue_norm  min(barres / 30, 3.0)                   -> colonne 2
        cible       LE RISQUE : la pire excursion a venir, en ATR
                    d'entree, positive. Voir `cible_risque`.
        latent_bps  le meme gain en points de base, pour le diagnostic

    LES TROIS PREMIERES SONT DANS LES UNITES EXACTES DE `_get_obs`. Les
    rendre autrement ferait apprendre la tete sur une distribution qu'elle
    ne reverrait jamais en decision.

    LA TENUE EST TIREE UNIFORMEMENT jusqu'a `tenue_max`, et non fixee : la
    tete doit savoir repondre a une minute comme a une heure de detention.
    Un echantillon concentre sur une seule duree lui apprendrait une regle
    qui ne vaut que la.
    """
    if tenue_max is None:
        tenue_max = horizon
    c = df["close"].to_numpy(np.float64)
    n_barres = len(c)
    rng = np.random.default_rng(graine)

    # L'entree doit laisser la place a la tenue PUIS a l'horizon.
    marge = int(tenue_max) + int(horizon) + 2
    entree = rng.integers(1, n_barres - marge, size=n)
    tenue = rng.integers(1, int(tenue_max) + 1, size=n)
    sens = rng.choice(np.array([-1.0, 1.0]), size=n)
    idx = entree + tenue

    # LE PRIX D'ENTREE EST CELUI DE L'EXECUTION, ET C'EST MESURE.
    #
    # La premiere version prenait le CLOSE de la barre d'entree. Le
    # simulateur, lui, execute a l'OUVERTURE de la barre — `step` lit
    # `data.open[idx]` — et applique le spread. Comparaison faite sur
    # quatre tenues d'une meme position :
    #
    #     tenue    col1 environnement   col1 echantillon   ecart
    #        1            -0.6137            +0.0000       0.614
    #        5            -0.0198            +0.5963       0.616
    #       12            -3.3570            -2.7546       0.602
    #       25            -2.9897            -2.3854       0.604
    #
    # UN DECALAGE CONSTANT DE 0.61 ATR — pas un arrondi, une convention
    # differente. La tete aurait appris sur des gains latents
    # systematiquement superieurs de 0.6 ATR a ceux qu'elle voit en
    # decision, et rien ne l'aurait signale.
    o = df["open"].to_numpy(np.float64) if "open" in df.columns else c
    import instruments as _I
    import training as _T
    _cfg = _T.PPOConfig()
    # LE COUT D'EXECUTION ATTENDU, pas celui d'un trade en particulier.
    #
    # Le simulateur tire le spread dans une loi bimodale et AJOUTE un
    # glissement uniforme entre 0 et `entry_slippage_bps`. Ces deux tirages
    # sont irreductibles : l'echantillon ne peut pas coller trade par
    # trade, il colle EN ESPERANCE. C'est suffisant — la tete apprend une
    # fonction du gain latent, pas la realisation d'un tirage.
    #
    # CE QU'IL RESTE, ET IL EST MESURE. Sur cent trades joues dans le
    # simulateur et compares a l'echantillon :
    #
    #     ecart moyen  -0.108 ATR    ecart-type 0.460    mediane 0.059
    #
    # Soit 2.3 ecarts-types de la moyenne : un BIAIS REEL, petit mais pas
    # nul. Sa cause est identifiee — `_sample_trade_spread_bps` tire dans
    # une loi BIMODALE dont la moyenne depasse le spread nominal, a cause
    # du mode large des seances calmes. Utiliser cette moyenne exacte
    # plutot que le nominal le supprimerait.
    #
    # ON LE LAISSE, ET ON L'ECRIT. Il vaut 0.1 ATR sur une colonne dont
    # l'ecart-type vaut plusieurs ATR — environ 5 %. Le corriger demande de
    # dupliquer ici la loi de spread du simulateur, donc de creer une
    # seconde ecriture de la meme chose : le remede serait pire que le mal.
    # Si la tete de cloture se met a fermer systematiquement trop tot, c'est
    # ICI qu'il faudra revenir.
    _sp = (float(_I.INSTRUMENTS["XAUUSD"]["spread_bps"])
           + 0.5 * float(getattr(_cfg, "entry_slippage_bps", 0.0))) / 1e4
    # On paie l'ask a l'achat, on touche le bid a la vente : le prix
    # d'entree est defavorable du spread, dans les deux sens.
    p_entree = o[entree] * (1.0 + sens * _sp)
    p_ici = c[idx]
    latent_bps = sens * (p_ici - p_entree) / p_entree * 1e4

    # LES COLONNES SONT RENDUES DANS LES UNITES DE L'ENVIRONNEMENT, ET
    # C'EST LE POINT LE PLUS FACILE A RATER DE TOUT CE FICHIER.
    #
    # `_get_obs` remplit la colonne 1 avec `position x (prix - entree) /
    # ATR_a_l_entree` et la colonne 2 avec `min(barres / 30, 3.0)`. Si
    # l'echantillon d'entrainement les remplissait autrement — en points de
    # base, par exemple, ce que la premiere version faisait — la tete
    # apprendrait sur une distribution et deciderait sur une autre, sans
    # qu'aucune erreur ne se declenche. C'est exactement la faute que ce
    # depot documente sous « deux ecritures de la meme regle ».
    #
    # L'ATR EST CELUI DE L'ENTREE, pas celui de l'instant courant :
    # l'environnement fige `entry_atr` a l'ouverture.
    atr = np.maximum(df["atr_14"].to_numpy(np.float64), 1e-9)
    latent_atr = sens * (p_ici - p_entree) / atr[entree]
    # L'ECHELLE VIENT DE `saint_core`, elle n'est plus recopiee ici. Le
    # 30.0 en dur a survecu au passage du plafond de 30 a 480 minutes : la
    # fabrique fournissait des ages satures a 90 barres pendant que
    # l'environnement en montrait jusqu'a 480. La tete apprenait sur une
    # distribution et decidait sur une autre — la faute que ce fichier
    # documente le plus souvent.
    from saint_core import SCALPING_MAX_HOLDING as _SMH
    tenue_norm = np.minimum(tenue / max(float(_SMH), 1.0), 3.0)

    # LA CIBLE EST LE RISQUE, PLUS LE SIGNE. Voir `cible_risque` pour la
    # mesure qui l'a impose : le signe du mouvement est une piece de
    # monnaie (50.0 % de positifs, tete au hasard sur quatre epochs),
    # l'amplitude se predit a rho 0.63 hors echantillon.
    #
    # L'ATR EST CELUI DE L'ENTREE, comme pour le latent : les deux
    # grandeurs doivent etre dans la MEME unite pour que la regle de
    # sortie les compare sans conversion.
    cible = cible_risque(df, idx, sens, atr[entree], horizon)

    # LA SECONDE CIBLE, POUR LA TETE DE PROFIT. Meme echantillon, memes
    # unites, meme ATR d'entree — c'est tout l'interet de la calculer ici
    # plutot que dans une seconde fabrique qui finirait par diverger.
    cible_prof = cible_profit(df, idx, sens, atr[entree], horizon)

    # LES DEUX COLONNES DE MARCHE QUE LA TETE DE PROFIT LIT, prises A
    # L'INSTANT DE LA DECISION. Absentes d'un cache ancien, on rend 0.5 —
    # le rang median, donc une information neutre plutot qu'un NaN qui
    # empoisonnerait la perte sans rien signaler.
    def _rang(nom):
        if nom in df.columns:
            v = df[nom].to_numpy(np.float64)[idx]
            return np.where(np.isfinite(v), v, 0.5)
        return np.full(len(idx), 0.5)

    return (idx, sens, latent_atr, tenue_norm, cible, latent_bps,
            cible_prof, _rang("creux_rang"), _rang("flux_rang"))


def verifie(df: pd.DataFrame, n_test: int = 20000,
            horizon: int = HORIZON_DEFAUT) -> int:
    """Les invariants que ces cibles doivent respecter. Rend un compte d'echecs."""
    rng = np.random.default_rng(0)
    idx = rng.choice(np.arange(10, len(df) - horizon - 10), n_test,
                     replace=False)
    ach, ven, cra, crv = rendements_entree(df, idx, horizon)
    clo = cible_cloture(df, idx, +1, horizon)
    # LE COUT EST CELUI DE CHAQUE BARRE depuis le 2026-09-21 : le
    # simulateur lit le spread cote, l'etiquette aussi. Comparer a une
    # constante ferait echouer des invariants qui sont pourtant tenus —
    # c'est ce qui s'est passe au premier essai, avec un ecart de 4.548
    # contre 3.353 qui n'etait que la difference entre le spread suppose
    # et le spread reel.
    _idx_ok, _masque = _bornes(len(df), idx, horizon)
    cout = np.full(len(idx), np.nan)
    cout[_masque] = cout_par_barre(df, _idx_ok)
    ok = (np.isfinite(ach) & np.isfinite(ven) & np.isfinite(clo)
          & np.isfinite(cout))
    echecs = 0

    def dit(nom, cond, detail=""):
        nonlocal echecs
        if not cond:
            echecs += 1
        print("  %s %-52s %s" % ("ok   " if cond else "ECHEC", nom, detail))

    # ACHAT + VENTE = -2 x COUT. Les deux cotes voient le meme mouvement a
    # l'envers, donc leur somme ne doit contenir que la friction.
    somme = (ach + ven)[ok]
    dit("achat + vente vaut exactement -2 x le cout",
        np.allclose(somme, -2 * cout[ok], atol=1e-9),
        "%.4f attendu %.4f  (cout median %.3f bps)"
        % (float(np.mean(somme)), float(np.mean(-2 * cout[ok])),
           float(np.median(cout[ok]))))

    # LA CLOTURE EST BRUTE : elle vaut l'achat plus le cout.
    dit("la cloture longue est l'achat, cout non deduit",
        np.allclose(clo[ok], ach[ok] + cout[ok], atol=1e-9))

    # LE CREUX EST NEGATIF OU NUL, et jamais meilleur que le resultat.
    dit("le creux d'un long est negatif ou nul",
        bool(np.all(cra[ok] <= 1e-9)))
    dit("le creux est au moins aussi mauvais que l'arrivee brute",
        bool(np.all(cra[ok] <= ach[ok] + cout[ok] + 1e-9)))

    # LES DEUX COTES PERDENT EN MOYENNE : c'est le peage, et s'il
    # disparaissait c'est que le cout ne serait pas applique.
    dit("une entree au hasard perd, des deux cotes",
        float(np.mean(ach[ok])) < 0 and float(np.mean(ven[ok])) < 0,
        "achat %+.2f  vente %+.2f bps"
        % (float(np.mean(ach[ok])), float(np.mean(ven[ok]))))
    return echecs


if __name__ == "__main__":
    import sys
    d = pd.read_pickle("data_cache_XAUUSD_M1.pkl")
    print("\n%s barres M1   horizon %d min   cout %.2f bps"
          % (format(len(d), ","), HORIZON_DEFAUT, cout_aller_retour()))
    print("\nCE QUI DOIT ETRE VRAI")
    n = verifie(d)
    rng = np.random.default_rng(1)
    idx = rng.choice(np.arange(10, len(d) - HORIZON_DEFAUT - 10), 50000,
                     replace=False)
    a, v, ca, cv = rendements_entree(d, idx)
    ok = np.isfinite(a)
    print("\nCE QUE LES CIBLES CONTIENNENT   (%s occasions)" % format(int(ok.sum()), ","))
    print("  achat  : moyenne %+7.2f   mediane %+7.2f   p95 %+8.2f bps"
          % (np.mean(a[ok]), np.median(a[ok]), np.percentile(a[ok], 95)))
    print("  creux  : mediane %+7.2f   p5  %+7.2f   p1  %+8.2f bps"
          % (np.median(ca[ok]), np.percentile(ca[ok], 5),
             np.percentile(ca[ok], 1)))
    print("\n  LE CREUX EST L'INFORMATION QUI REMPLACE LE STOP : une position")
    print("  sur cent traverse au moins %.0f bps de perte latente avant de"
          % abs(np.percentile(ca[ok], 1)))
    print("  se resoudre, et rien ne l'arrete automatiquement.")
    sys.exit(1 if n else 0)
