# -*- coding: utf-8 -*-
"""Les strategies des 44 ebooks du proprietaire, appliquees A LA LETTRE.

2026-09-27. Apres Riguet (etude_pdf_ichimoku.py) et le Smart Money
(etude_smc.py), le proprietaire a donne tout son dossier
E:\\Formation-trading\\Ebook. Chaque livre a ete lu ; ceux qui donnent des
regles precises sont codes ici, avec LEURS parametres, et rien n'est regle
apres le premier lancement. Meme protocole, memes couts, memes marches que
les deux etudes precedentes : bougies MT5 Vantage, spread reel, glissement
0.5 bps, swap du courtier, mise fixe de 10 $ par unite de risque.

UNITE DE RISQUE. Quand le livre donne un stop, une unite = la distance au
stop. Quand il n'en donne pas (croisements toujours en position, sorties a
l'ouverture du lendemain...), on n'en invente pas : la taille est fixee pour
que 2 ATR valent une unite, et la perte peut depasser une unite. Distance
plancher : 0.1 ATR, pour qu'un stop colle au prix ne divise pas par zero.

LES STRATEGIES (source ; regles ; ce qui a du etre traduit)

 Crypto-Analyse, "MACD, un systeme de trading complet"
  M1 croisement MACD (12,26,9) / signal, toujours en position (p.5-7)
  M2 MACD + MM200 : achat au croisement haussier si cours > MM200, sortie au
     croisement baissier ou sous la MM200 ; ventes symetriques (p.8-9)
  M3 retournement : croisement + divergence sur les deux derniers creux (ou
     sommets) de swing ; sortie au croisement inverse (p.14-19 ; la sortie
     "croisement ET divergence inverse" n'arrive presque jamais)
 Master Feng, "Le guide ultime des moyennes mobiles"
  F1 tendance MME200 (pente sur 10 bougies et cours du bon cote), entree au
     retour sur la MME50 (zone de valeur), stop derriere la moyenne avec une
     marge de 0.5 ATR, sortie a une cloture de l'autre cote (p.5-13)
 Eric David, Visual Forex / Visual Forex Pro
  V1 phases MME 12/20/50 ; signal : l'histogramme MACD change de sens, RSI
     14 > 50 et SAR du bon cote ; en phase de tendance sortie a la cloture
     au-dela de la MME12, en phase d'incertitude objectif 20 pips (2 ATR en
     M5 dans ses exemples) et sortie si l'histogramme se retourne (p.17-34)
 Eric David, Forex Evolution
  V2 retournement du BBand Stop (15, 1) avec MACD au-dessus de sa ligne
     signal ; stop sur le BBand Stop, objectif 10 pips (1 ATR) (p.13-15)
  V3 en tendance MME 12>20>50 : croisement MACD dans le sens du BBand Stop
     (p.45), memes stop et objectif
 Alexander Elder, "Vivre du trading"
  E1 Triple Ecran : maree = pente de l'histogramme MACD de l'unite de temps
     superieure (x4 a x7 selon nos donnees), vague = Force Index MME 2 sous
     zero (achats), ordre d'achat stop au-dessus du plus haut de la veille,
     remis chaque jour ; stop sous le plus bas du jour ou de la veille ;
     sortie quand la maree se retourne (p.176-181)
  E2 idem avec la stochastique (14,3,3) sous 30 / au-dessus de 70 (p.179)
  E3 systeme parabolique (SAR 0.02 / 0.2), stop and reverse (p.182-184)
  E4 canal MME13 contenant 95 % des cours : en canal plat, achat au bas,
     revente au haut (Appel, regle 1, p.184-186) ; sans stop dans le livre,
     sortie quand le canal cesse d'etre plat
  E5 bandes de Bollinger (MME21, 2 ecarts-types) : sortie hors de bandes
     etroites (20 % les plus etroites de l'annee), cloture quand les prix
     rentrent dans le canal (p.187)
 "Techniques et strategies de day et swing trading" (MTT)
  T1 cassure de la premiere heure (9h30-10h30 New York) a 0.1 %, sortie en
     fin de seance, sans stop (p.82-85)
  T2 le "systeme final" futures : reference 9h30-10h00, entrees 14h-16h
     seulement, a 2.25 points au-dela (0.25 % de l'ES de 2001-2003), stop a
     10 points (1.1 %), sortie a 16h15 (p.131-136)
  T3 gaps : apres une ouverture en gap, entree 0.1 % au-dela de l'extreme de
     la veille dans le sens du comblement, stop 2.5 %, sortie a l'ouverture
     suivante (variante indices, p.197-201)
  T4 volatilite hebdomadaire : achat a l'ouverture de la semaine + 40 % de
     l'amplitude de la semaine precedente, stop sous son plus bas, sortie a
     l'ouverture de la semaine suivante ; achats seulement (p.207-212)
  T5 swing points : creux de moyen terme (creux de court terme encadre par
     deux creux plus hauts), achat au-dessus du dernier sommet de court
     terme, stop sous le dernier creux de court terme, remonte a chaque
     nouveau creux (p.147-160)
 "Swing Trading" (MTT)
  S1 cours > MM20 > MM50 : achat au retour sur la MM20, stop sous le plus
     bas des 5 dernieres bougies, sortie quand la MM20 repasse sous la MM50
     (chap. 4)
  S2 swing directionnel : demain au-dessus du plus haut d'aujourd'hui si
     cloture > ouverture (vente symetrique), sortie a l'ouverture suivante
     (chap. 10 ; niveau de penetration 0, le livre ne le fixe pas ici)
  S3 idem, le lundi seulement (le "profil hebdomadaire", chap. 10)
 FX Wolf, "strategie CM" (livre en images)
  W1 range de la session asiatique (17h-2h New York), cassure pendant la
     journee, stop de l'autre cote du range, objectif 0.6 fois le range
     (reglage de l'indicateur), sortie en fin de journee
  W2 idem, dans le sens indique par le point pivot journalier
 "Fibonacci Secret" (Pips Leader, livre en images)
  P1 retracement d'une jambe de swing : ordre limite a 62 %, annule si un
     corps cloture au-dela de 79 %, stop au-dela de 100 %, objectif 5-10 pips
     (1 ATR en M15)
  P2 idem, objectif au sommet de la jambe (sa variante "prendre plus de pips")
 Ichimoku (Sam Ventura, DailyFX "l'indicateur Ichimoku", Karen Peloille)
  I1 croisement Tenkan/Kijun fort : cours au-dessus du nuage, Chikou
     au-dessus du prix et du nuage d'il y a 26 bougies ; stop sous la
     Kijun, sortie a une cloture sous la Kijun
  I2 sortie du nuage (cloture au-dessus), nuage futur haussier ; idem
  I3 croisement de la Kijun au-dessus du nuage, Chikou libre ; idem
 Guides Forex (Forex simplifie, GFC, IC Markets, XTB, BabyPips, Debutants)
  G1 croisement MM20 / MM50, toujours en position (Forex simplifie p.43)
  G2 point pivot journalier : cassure du pivot, objectif R1 / S1, stop
     S1 / R1, sortie en fin de journee (IC Markets p.16)
 Analyse technique (Clement, Le Chartisme, Maitriser l'AT, Kabbaj, Elder)
  A1 RSI 14 : achat quand il remonte au-dessus de 30, vente quand il
     redescend sous 70, toujours en position
  A2 stochastique (14,3,3) : croisement %K/%D sous 20 ou au-dessus de 80
  A3 Bollinger (20,2) : retour dans la bande, objectif la moyenne, stop sous
     le plus bas des deux dernieres bougies
  A4 double creux / double sommet : deux creux de swing a moins d'un demi
     ATR, cassure de la ligne de cou en cloture, objectif la hauteur de la
     figure reportee, stop sous les creux
  A5 rectangle de 20 bougies (au plus 4 ATR) casse en cloture avec un
     volume double de la moyenne (Clement), objectif la hauteur reportee
     (Le Chartisme), stop au milieu du rectangle
  A6 croisement MM50 / MM200 ("golden cross"), toujours en position
 Chandeliers japonais (Francois Baron et les autres livres de chandeliers)
  C1 structure de retournement apres une baisse (sous la MM20), VALIDEE par
     une cloture au-dela du plus haut de la structure (Baron, p.111) ; stop
     sous la structure, objectif 2 fois le risque

NON TESTE, faute de regle ou de donnees : les annonces economiques (Forex
Dating, Guide complet du Forex 2 : il faudrait l'historique du calendrier),
les vagues d'Elliott (comptage subjectif), les droites obliques et
figures tracees a la main, et les livres d'investissement, de psychologie
ou d'analyse fondamentale (Buffett, La Bourse pour les Nuls, gestion de
portefeuille, Reussir en Bourse, Self-Made-Man, Analyse fondamentale x2,
Conseils pour devenir un pro). Le portage (carry trade) a ete mesure a part
(etude_portage.py) et les swaps Vantage sont presque tous negatifs.

UNITES DE TEMPS. Les strategies de seance (T1, T2, W1, W2, G2) tournent en
M15 (G2 aussi en H1) ; les strategies journalieres (T3, T4, S2, S3) en D1 ;
toutes les autres en M15, H1, H4 et D1.

    python etude_livres.py
"""
import sys
from typing import NamedTuple

import numpy as np
import pandas as pd

import etude_pdf_ichimoku as E

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GLIS = E.GLISSEMENT_BPS


# ==========================================================================
#  Indicateurs
# ==========================================================================

def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def sma(x, n):
    return E._roll(np.asarray(x, np.float64), n, "mean")


def croise_haut(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a > b) & (E._dec(a) <= E._dec(b))


def croise_bas(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a < b) & (E._dec(a) >= E._dec(b))


def rsi(c, n=14):
    d = np.diff(c, prepend=np.nan)
    g = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    p = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    r = (100 - 100 / (1 + g / p.replace(0, np.nan))).to_numpy()
    r[:n] = np.nan
    return r


def macd(c, a=12, b=26, s=9):
    m = ema(c, a) - ema(c, b)
    sg = ema(m, s)
    return m, sg, m - sg


def stochastique(h, l, c, n=14, k=3, d=3):
    hh, ll = E._roll(h, n, "max"), E._roll(l, n, "min")
    brut = 100 * (c - ll) / np.where(hh - ll > 0, hh - ll, np.nan)
    kk = sma(brut, k)
    return kk, sma(kk, d)


def bollinger(c, n=20, k=2.0, exp=False):
    m = ema(c, n) if exp else sma(c, n)
    sd = pd.Series(c).rolling(n, min_periods=n).std(ddof=0).to_numpy()
    return m, m + k * sd, m - k * sd


def sar(h, l, pas=0.02, maxi=0.2):
    """Parabolique de Wilder : (niveau pour la bougie t, sens apres t)."""
    n = len(h)
    niv, sens = np.full(n, np.nan), np.zeros(n)
    if n < 3:
        return niv, sens
    tr = 1 if h[1] >= h[0] else -1
    s = l[0] if tr > 0 else h[0]
    ep = h[1] if tr > 0 else l[1]
    af = pas
    for t in range(2, n):
        s = s + af * (ep - s)
        if tr > 0:
            s = min(s, l[t - 1], l[t - 2])
            if l[t] < s:
                tr, s, ep, af = -1, ep, l[t], pas
            elif h[t] > ep:
                ep, af = h[t], min(af + pas, maxi)
        else:
            s = max(s, h[t - 1], h[t - 2])
            if h[t] > s:
                tr, s, ep, af = 1, ep, h[t], pas
            elif l[t] < ep:
                ep, af = l[t], min(af + pas, maxi)
        niv[t], sens[t] = s, tr
    return niv, sens


def bband_stop(c, n=15, dev=1.0):
    """BBands Stop (MT4) : (sens, ligne de stop)."""
    m, up, lo = bollinger(c, n, dev)
    N = len(c)
    sens, ligne = np.zeros(N), np.full(N, np.nan)
    tr = 0
    smax, smin = up.copy(), lo.copy()
    for t in range(1, N):
        if not (np.isfinite(up[t]) and np.isfinite(up[t - 1])):
            continue
        if c[t] > smax[t - 1]:
            tr = 1
        elif c[t] < smin[t - 1]:
            tr = -1
        if tr > 0 and np.isfinite(smin[t - 1]) and smin[t] < smin[t - 1]:
            smin[t] = smin[t - 1]
        if tr < 0 and np.isfinite(smax[t - 1]) and smax[t] > smax[t - 1]:
            smax[t] = smax[t - 1]
        sens[t] = tr
        ligne[t] = smin[t] if tr > 0 else smax[t] if tr < 0 else np.nan
    return sens, ligne


def swings(h, l, k=2):
    """Swings confirmes (fractale k) : tableaux (indice, niveau, confirmation)."""
    n = len(h)
    hs, ls = [], []
    for t in range(2 * k, n):
        i = t - k
        if h[i] > h[i - k:i].max() and h[i] >= h[i + 1:t + 1].max():
            hs.append((i, h[i], t))
        if l[i] < l[i - k:i].min() and l[i] <= l[i + 1:t + 1].min():
            ls.append((i, l[i], t))
    return (np.array(hs) if hs else np.zeros((0, 3))), (np.array(ls) if ls else np.zeros((0, 3)))


def derniers(tab, t, m=2):
    """Les m derniers swings confirmes a t (du plus ancien au plus recent)."""
    if not len(tab):
        return []
    j = np.searchsorted(tab[:, 2], t, side="right")
    return [tab[i] for i in range(max(0, j - m), j)]


# ==========================================================================
#  Le moteur : un ordre, une position a la fois
# ==========================================================================

class Ordre(NamedTuple):
    t: int                     # bougie de decision (a sa cloture)
    sens: int                  # +1 achat, -1 vente ; 0 pour un ordre bracket
    genre: str                 # M marche, S stop, S0 stop des l'ouverture de t,
                               # L limite, B bracket (stop haut ET stop bas)
    prix: float = np.nan       # declencheur (S, S0, L, haut de B)
    valide: int = 1            # bougies ou l'ordre reste actif
    stop: float = np.nan
    tp: float = np.nan
    risque: float = np.nan     # unite de risque (prix) quand il n'y a pas de stop
    fin: int = -1              # sortie a l'ouverture de cette bougie
    nmax: int = 0              # sortie a l'ouverture, nmax bougies apres l'entree
    sortie: str = ""           # cle des tableaux de sortie a la cloture
    suivi: str = ""            # cle des tableaux de stop suiveur
    annule: float = np.nan     # limite : annule si une cloture passe au-dela
    prix2: float = np.nan      # bas du bracket
    stop2: float = np.nan
    tp2: float = np.nan


def simule(df, x, ordres, extra, sw_l, sw_s, crypto, jour3, point, glis=GLIS):
    o, h, l, c, a = x["o"], x["h"], x["l"], x["c"], x["atr"]
    sp = df["spread"].to_numpy(np.float64) * point
    temps = df["time"].to_numpy()
    n = len(c)
    trades = []
    libre = 0                                   # premiere bougie ou l'on peut entrer
    for od in sorted(ordres, key=lambda z: z.t):
        debut = od.t if od.genre == "S0" else od.t + 1
        if debut < libre or debut >= n or (od.fin >= 0 and od.fin <= debut):
            continue
        g = glis * 1e-4 * o[debut]
        sens, k0, entree, stop, tp = od.sens, None, None, od.stop, od.tp
        dernier = min(debut + max(od.valide, 1) - 1, n - 1)
        if od.genre == "M":
            k0 = debut
            entree = o[k0] + sp[k0] + g if sens > 0 else o[k0] - g
        else:
            for k in range(debut, dernier + 1):
                if od.genre == "L":
                    if sens > 0 and l[k] + sp[k] <= od.prix:
                        k0, entree = k, min(o[k] + sp[k], od.prix)
                    elif sens < 0 and h[k] >= od.prix:
                        k0, entree = k, max(o[k], od.prix)
                    elif np.isfinite(od.annule) and (c[k] - od.annule) * sens < 0:
                        break
                    if k0 is not None:
                        break
                    continue
                if od.genre == "B":
                    haut = np.isfinite(od.prix) and h[k] >= od.prix
                    bas = np.isfinite(od.prix2) and l[k] <= od.prix2
                    if haut and bas:        # le plus proche de l'ouverture d'abord
                        haut = abs(od.prix - o[k]) <= abs(o[k] - od.prix2)
                        bas = not haut
                    if haut:
                        sens, stop, tp = 1, od.stop, od.tp
                        k0, entree = k, max(o[k], od.prix) + sp[k] + g
                    elif bas:
                        sens, stop, tp = -1, od.stop2, od.tp2
                        k0, entree = k, min(o[k], od.prix2) - g
                    if k0 is not None:
                        break
                    continue
                # ordres stop (S, S0)
                if sens > 0 and h[k] >= od.prix:
                    k0, entree = k, max(o[k], od.prix) + sp[k] + g
                    break
                if sens < 0 and l[k] <= od.prix:
                    k0, entree = k, min(o[k], od.prix) - g
                    break
        if k0 is None:
            continue
        unite = abs(entree - stop) if np.isfinite(stop) else od.risque
        if not np.isfinite(unite):
            unite = 2 * a[od.t] if np.isfinite(a[od.t]) else np.nan
        unite = max(unite, 0.1 * a[od.t]) if np.isfinite(a[od.t]) else unite
        if not (np.isfinite(unite) and unite > 0):
            continue
        s_long, s_court = (extra.get(od.sortie + "+"), extra.get(od.sortie + "-")) if od.sortie else (None, None)
        suivi = extra.get(od.suivi + ("+" if sens > 0 else "-")) if od.suivi else None
        fin_abs = od.fin if od.fin >= 0 else (k0 + od.nmax if od.nmax else n)
        sortie, kf, marche_suivant = None, n - 1, False
        stop_eff = stop
        for k in range(k0, n):
            if k >= fin_abs or marche_suivant:
                sortie = (o[k] - g) if sens > 0 else (o[k] + sp[k] + g)
                kf = k
                break
            if suivi is not None and k > k0 and np.isfinite(suivi[k - 1]):
                if not np.isfinite(stop_eff) or (suivi[k - 1] - stop_eff) * sens > 0:
                    stop_eff = suivi[k - 1]
            # Au stop : une ouverture deja au-dela sort a l'ouverture. Sur la
            # bougie d'execution d'un ordre stop, l'execution a eu lieu au
            # declencheur, donc apres l'ouverture : la sortie se fait au stop.
            # Un ordre limite rempli a l'ouverture (gap) sort a l'ouverture.
            execute_ici = k == k0 and od.genre != "M"
            ouv_passe = not (execute_ici and od.genre in ("S", "S0", "B"))
            if sens > 0:
                if np.isfinite(stop_eff) and l[k] <= stop_eff:
                    sortie = (o[k] if (ouv_passe and o[k] < stop_eff) else stop_eff) - g
                elif np.isfinite(tp) and h[k] >= tp and not execute_ici:
                    sortie = max(o[k], tp) if k > k0 else tp
            else:
                ha, la, oa = h[k] + sp[k], l[k] + sp[k], o[k] + sp[k]
                if np.isfinite(stop_eff) and ha >= stop_eff:
                    sortie = (oa if (ouv_passe and oa > stop_eff) else stop_eff) + g
                elif np.isfinite(tp) and la <= tp and not execute_ici:
                    sortie = min(oa, tp) if k > k0 else tp
            if sortie is not None:
                kf = k
                break
            arr = s_long if sens > 0 else s_court
            if arr is not None and arr[k]:
                marche_suivant = True
        if sortie is None:
            sortie = c[n - 1] if sens > 0 else c[n - 1] + sp[n - 1]
        r = (sortie - entree) * sens / unite
        nuits = E._nuits(pd.Timestamp(temps[k0]), pd.Timestamp(temps[kf]), crypto, jour3)
        sw = (sw_l if sens > 0 else sw_s) * 1e-4 * entree * nuits / unite
        trades.append((pd.Timestamp(temps[k0]), sens, r + sw))
        libre = kf if marche_suivant or kf >= fin_abs else kf + 1
    return trades


# ==========================================================================
#  Contexte commun a toutes les strategies d'un jeu de bougies
# ==========================================================================

class Ctx:
    def __init__(self, df, tf, crypto=False):
        self.df, self.tf, self.crypto = df, tf, crypto
        self.o, self.h, self.l, self.c = (df[k].to_numpy(np.float64)
                                          for k in ("open", "high", "low", "close"))
        self.v = df["tick_volume"].to_numpy(np.float64)
        tr = np.maximum(self.h - self.l, np.maximum(np.abs(self.h - E._dec(self.c)),
                                                    np.abs(self.l - E._dec(self.c))))
        self.atr = E._roll(tr, 14, "mean")
        self.n = len(self.c)
        t = pd.to_datetime(df["time"])
        self.temps = t
        self.jour = t.dt.normalize().to_numpy()
        self.minute = (t.dt.hour * 60 + t.dt.minute).to_numpy()
        self.wd = t.dt.weekday.to_numpy()
        self.pret = np.isfinite(self.atr) & (np.arange(self.n) >= 210)
        self._cache = {}

    def x(self):
        return dict(o=self.o, h=self.h, l=self.l, c=self.c, atr=self.atr)

    def get(self, cle, f):
        if cle not in self._cache:
            self._cache[cle] = f()
        return self._cache[cle]

    @property
    def macd(self):
        return self.get("macd", lambda: macd(self.c))

    @property
    def sw(self):
        return self.get("sw", lambda: swings(self.h, self.l, 2))

    def premier_du_jour(self):
        """Indice de la premiere bougie de chaque jour, et du jour suivant."""
        def f():
            nouveau = np.r_[True, self.jour[1:] != self.jour[:-1]]
            idx = np.flatnonzero(nouveau)
            suivant = np.full(self.n, self.n, dtype=int)
            for a, b in zip(idx, list(idx[1:]) + [self.n]):
                suivant[a:b] = b
            return idx, suivant
        return self.get("jours", f)

    def ut_sup(self, regle, duree):
        """Pente de l'histogramme MACD de l'unite de temps superieure, en ne
        lisant que les bougies superieures TERMINEES."""
        def f():
            d = self.df.set_index("time")
            s = d.resample(regle, label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
            _, _, hist = macd(s["close"].to_numpy(np.float64))
            fin_sup = (s.index + duree).to_numpy()
            fin_inf = (self.temps + pd.Timedelta(self._pas())).to_numpy()
            j = np.searchsorted(fin_sup, fin_inf, side="right") - 1
            pente = np.full(self.n, 0.0)
            ok = j >= 1
            pente[ok] = np.sign(hist[j[ok]] - hist[j[ok] - 1])
            return pente
        return self.get(("sup", regle), f)

    def _pas(self):
        return {"M15": "15min", "H1": "1h", "H4": "4h", "D1": "1D"}[self.tf]


def _m(t, sens, risque=np.nan, **kw):
    return Ordre(int(t), int(sens), "M", risque=risque, **kw)


# ==========================================================================
#  Les strategies
# ==========================================================================

def toujours_en_position(cx, haut, bas, cle="flip"):
    """Croisements : entree au croisement, sortie au croisement inverse."""
    ordres = [_m(t, 1, 2 * cx.atr[t], sortie=cle) for t in np.flatnonzero(haut & cx.pret)]
    ordres += [_m(t, -1, 2 * cx.atr[t], sortie=cle) for t in np.flatnonzero(bas & cx.pret)]
    return ordres, {cle + "+": bas, cle + "-": haut}


def M1(cx):
    m, s, _ = cx.macd
    return toujours_en_position(cx, croise_haut(m, s), croise_bas(m, s))


def M2(cx):
    m, s, _ = cx.macd
    mm = sma(cx.c, 200)
    up, dn = croise_haut(m, s), croise_bas(m, s)
    ordres = [_m(t, 1, 2 * cx.atr[t], sortie="x") for t in np.flatnonzero(up & (cx.c > mm) & cx.pret)]
    ordres += [_m(t, -1, 2 * cx.atr[t], sortie="x") for t in np.flatnonzero(dn & (cx.c < mm) & cx.pret)]
    return ordres, {"x+": dn | (cx.c < mm), "x-": up | (cx.c > mm)}


def M3(cx):
    m, s, _ = cx.macd
    hs, ls = cx.sw
    up, dn = croise_haut(m, s), croise_bas(m, s)
    ordres = []
    for t in np.flatnonzero((up | dn) & cx.pret):
        sens = 1 if up[t] else -1
        d = derniers(ls if sens > 0 else hs, t, 2)
        if len(d) < 2 or d[1][0] < t - 60:
            continue
        (i1, v1, _), (i2, v2, _) = d
        i1, i2 = int(i1), int(i2)
        if (sens > 0 and v2 < v1 and m[i2] > m[i1]) or (sens < 0 and v2 > v1 and m[i2] < m[i1]):
            ordres.append(_m(t, sens, 2 * cx.atr[t], sortie="x"))
    return ordres, {"x+": dn, "x-": up}


def F1(cx):
    e200, e50 = ema(cx.c, 200), ema(cx.c, 50)
    pente = e200 - E._dec(e200, 10)
    hausse = (pente > 0) & (cx.c > e200)
    baisse = (pente < 0) & (cx.c < e200)
    lo = hausse & (cx.l <= e50) & (cx.c > e50) & cx.pret
    sh = baisse & (cx.h >= e50) & (cx.c < e50) & cx.pret
    ordres = [_m(t, 1, stop=e50[t] - 0.5 * cx.atr[t], sortie="x") for t in np.flatnonzero(lo)]
    ordres += [_m(t, -1, stop=e50[t] + 0.5 * cx.atr[t], sortie="x") for t in np.flatnonzero(sh)]
    return ordres, {"x+": cx.c < e50, "x-": cx.c > e50}


def V1(cx):
    e12, e20, e50 = ema(cx.c, 12), ema(cx.c, 20), ema(cx.c, 50)
    achat = (cx.c > e12) & (e12 > e20) & (e20 > e50)
    vente = (cx.c < e12) & (e12 < e20) & (e20 < e50)
    _, _, hi = cx.macd
    vers_haut = (hi > E._dec(hi)) & (E._dec(hi) <= E._dec(hi, 2))
    vers_bas = (hi < E._dec(hi)) & (E._dec(hi) >= E._dec(hi, 2))
    r = rsi(cx.c)
    sa, _ = cx.get("sar", lambda: sar(cx.h, cx.l))
    lo = vers_haut & (r > 50) & (sa < cx.c) & ~vente & cx.pret
    sh = vers_bas & (r < 50) & (sa > cx.c) & ~achat & cx.pret
    ordres = []
    for t in np.flatnonzero(lo):
        if achat[t]:
            ordres.append(_m(t, 1, 2 * cx.atr[t], sortie="tend"))
        else:
            ordres.append(_m(t, 1, 2 * cx.atr[t], tp=cx.c[t] + 2 * cx.atr[t], sortie="inc"))
    for t in np.flatnonzero(sh):
        if vente[t]:
            ordres.append(_m(t, -1, 2 * cx.atr[t], sortie="tend"))
        else:
            ordres.append(_m(t, -1, 2 * cx.atr[t], tp=cx.c[t] - 2 * cx.atr[t], sortie="inc"))
    return ordres, {"tend+": cx.c < e12, "tend-": cx.c > e12,
                    "inc+": vers_bas, "inc-": vers_haut}


def _bband(cx):
    return cx.get("bband", lambda: bband_stop(cx.c, 15, 1.0))


def V2(cx):
    bs, ligne = _bband(cx)
    m, s, _ = cx.macd
    flip_h = (bs > 0) & (E._dec(bs) <= 0)
    flip_b = (bs < 0) & (E._dec(bs) >= 0)
    ordres = [_m(t, 1, stop=ligne[t], tp=cx.c[t] + cx.atr[t]) for t in np.flatnonzero(flip_h & (m > s) & cx.pret)]
    ordres += [_m(t, -1, stop=ligne[t], tp=cx.c[t] - cx.atr[t]) for t in np.flatnonzero(flip_b & (m < s) & cx.pret)]
    return ordres, {}


def V3(cx):
    bs, ligne = _bband(cx)
    m, s, _ = cx.macd
    e12, e20, e50 = ema(cx.c, 12), ema(cx.c, 20), ema(cx.c, 50)
    lo = croise_haut(m, s) & (bs > 0) & (e12 > e20) & (e20 > e50) & cx.pret
    sh = croise_bas(m, s) & (bs < 0) & (e12 < e20) & (e20 < e50) & cx.pret
    ordres = [_m(t, 1, stop=ligne[t], tp=cx.c[t] + cx.atr[t]) for t in np.flatnonzero(lo)]
    ordres += [_m(t, -1, stop=ligne[t], tp=cx.c[t] - cx.atr[t]) for t in np.flatnonzero(sh)]
    return ordres, {}


SUP = {"M15": ("1h", pd.Timedelta("1h")), "H1": ("4h", pd.Timedelta("4h")),
       "H4": ("1D", pd.Timedelta("1D")), "D1": ("W-MON", pd.Timedelta("7D"))}


def _triple(cx, vague_achat, vague_vente):
    regle, duree = SUP[cx.tf]
    maree = cx.ut_sup(regle, duree)
    ordres = []
    for t in np.flatnonzero(cx.pret):
        if maree[t] > 0 and vague_achat[t]:
            ordres.append(Ordre(t, 1, "S", prix=cx.h[t] + 0.01 * cx.atr[t], valide=1,
                                stop=min(cx.l[t], cx.l[t - 1]) - 0.01 * cx.atr[t], sortie="maree"))
        elif maree[t] < 0 and vague_vente[t]:
            ordres.append(Ordre(t, -1, "S", prix=cx.l[t] - 0.01 * cx.atr[t], valide=1,
                                stop=max(cx.h[t], cx.h[t - 1]) + 0.01 * cx.atr[t], sortie="maree"))
    return ordres, {"maree+": maree < 0, "maree-": maree > 0}


def E1(cx):
    fi = ema(np.diff(cx.c, prepend=np.nan) * cx.v, 2)
    return _triple(cx, fi < 0, fi > 0)


def E2(cx):
    k, _ = stochastique(cx.h, cx.l, cx.c)
    return _triple(cx, k < 30, k > 70)


def E3(cx):
    niv, sens = cx.get("sar", lambda: sar(cx.h, cx.l))
    flip = (sens != E._dec(sens)) & (sens != 0) & cx.pret
    ordres = [_m(t, int(sens[t]), stop=np.nan, risque=max(abs(cx.c[t] - niv[t]), 0.1 * cx.atr[t]),
                 sortie="flip") for t in np.flatnonzero(flip)]
    return ordres, {"flip+": sens < 0, "flip-": sens > 0}


def E4(cx):
    e13 = ema(cx.c, 13)
    ecart = pd.Series(np.abs(cx.c / e13 - 1)).rolling(250, min_periods=250).quantile(0.95).shift(1).to_numpy()
    haut, bas = e13 * (1 + ecart), e13 * (1 - ecart)
    plat = np.abs(e13 - E._dec(e13, 5)) < 0.5 * cx.atr
    lo = plat & (cx.c <= bas) & cx.pret
    sh = plat & (cx.c >= haut) & cx.pret
    # La regle ne vaut que "quand le canal est relativement plat" : sans stop
    # dans le livre, la position est soldee quand cette condition cesse
    # (sinon un canal qui part en tendance la garderait pour toujours).
    ordres = [_m(t, 1, 2 * cx.atr[t], tp=haut[t], sortie="plat") for t in np.flatnonzero(lo)]
    ordres += [_m(t, -1, 2 * cx.atr[t], tp=bas[t], sortie="plat") for t in np.flatnonzero(sh)]
    return ordres, {"plat+": ~plat, "plat-": ~plat}


def E5(cx):
    m, up, lo_b = bollinger(cx.c, 21, 2.0, exp=True)
    bw = (up - lo_b) / m
    seuil = pd.Series(bw).rolling(250, min_periods=250).quantile(0.20).shift(1).to_numpy()
    etroit = E._dec(bw) <= seuil
    lo = etroit & (cx.c > up) & cx.pret
    sh = etroit & (cx.c < lo_b) & cx.pret
    ordres = [_m(t, 1, 2 * cx.atr[t], sortie="x") for t in np.flatnonzero(lo)]
    ordres += [_m(t, -1, 2 * cx.atr[t], sortie="x") for t in np.flatnonzero(sh)]
    return ordres, {"x+": cx.c < up, "x-": cx.c > lo_b}


# ---------- strategies de seance (M15 ; heures serveur = New York + 7 h) ----------

def _seance(cx, ref_debut, ref_fin, entree_debut, entree_fin, sortie_min):
    """Pour chaque jour : (plus haut, plus bas) de la reference, bougie de
    DECISION (la derniere avant la fenetre d'entree, 15 min avant), nombre de
    bougies de la fenetre, bougie de sortie. La decision ne lit que le passe ;
    la duree de la fenetre et la sortie sont du calendrier."""
    idx, suivant = cx.premier_du_jour()
    out = []
    for a in idx:
        b = suivant[a]
        mn = cx.minute[a:b]
        dec = np.flatnonzero(mn == entree_debut - 15)
        if not len(dec):
            continue
        t = a + dec[0]
        ref = np.flatnonzero((mn >= ref_debut) & (mn < ref_fin) & (np.arange(b - a) <= dec[0]))
        if len(ref) < max(1, ((ref_fin - ref_debut) // 15) // 2):
            continue
        rh, rl = cx.h[a + ref].max(), cx.l[a + ref].min()
        fen = np.flatnonzero((mn >= entree_debut) & (mn < entree_fin))
        sor = np.flatnonzero(mn >= sortie_min) if sortie_min is not None else []
        fin = a + sor[0] if len(sor) else b
        out.append((rh, rl, t, max(len(fen), 1), fin))
    return out


def T1(cx):
    ordres = []
    for rh, rl, t, nb, fin in _seance(cx, 990, 1050, 1050, 1380, 1380):
        up, dn = rh * 1.001, rl * 0.999
        if cx.pret[t]:
            ordres.append(Ordre(t, 0, "B", prix=up, prix2=dn, valide=nb, risque=up - dn, fin=fin))
    return ordres, {}


def T2(cx):
    ordres = []
    for rh, rl, t, nb, fin in _seance(cx, 990, 1020, 1260, 1380, 1395):
        up, dn = rh * 1.0025, rl * 0.9975
        if cx.pret[t]:
            ordres.append(Ordre(t, 0, "B", prix=up, stop=up * (1 - 0.011),
                                prix2=dn, stop2=dn * (1 + 0.011), valide=nb, fin=fin))
    return ordres, {}


def _pivots(cx):
    """Pivot, R1, S1 de la veille (jour serveur), pour chaque bougie."""
    def f():
        d = pd.DataFrame({"j": cx.jour, "h": cx.h, "l": cx.l, "c": cx.c})
        g = d.groupby("j").agg(h=("h", "max"), l=("l", "min"), c=("c", "last")).shift(1)
        p = (g["h"] + g["l"] + g["c"]) / 3
        r1, s1 = 2 * p - g["l"], 2 * p - g["h"]
        m = pd.DataFrame({"p": p, "r1": r1, "s1": s1}).reindex(d["j"]).to_numpy()
        return m[:, 0], m[:, 1], m[:, 2]
    return cx.get("pivots", f)


def _asie(cx, filtre):
    p, _, _ = _pivots(cx)
    ordres = []
    for rh, rl, t, nb, fin in _seance(cx, 0, 540, 540, 1200, None):
        larg = rh - rl
        if not (larg > 0) or not cx.pret[t]:
            continue
        a = cx.atr[t]
        up, dn = rh, rl
        if filtre and np.isfinite(p[t]):
            if p[t] > rh:
                up = np.nan
            elif p[t] < rl:
                dn = np.nan
        ordres.append(Ordre(t, 0, "B", prix=up, stop=rl - 0.1 * a, tp=rh + 0.6 * larg,
                            prix2=dn, stop2=rh + 0.1 * a, tp2=rl - 0.6 * larg, valide=nb, fin=fin))
    return ordres, {}


def W1(cx):
    return _asie(cx, False)


def W2(cx):
    return _asie(cx, True)


def G2(cx):
    p, r1, s1 = _pivots(cx)
    _, suivant = cx.premier_du_jour()
    lo = croise_haut(cx.c, p) & cx.pret & (r1 > cx.c) & (s1 < cx.c)
    sh = croise_bas(cx.c, p) & cx.pret & (s1 < cx.c) & (r1 > cx.c)
    ordres = [_m(t, 1, stop=s1[t], tp=r1[t], fin=int(suivant[t])) for t in np.flatnonzero(lo)]
    ordres += [_m(t, -1, stop=r1[t], tp=s1[t], fin=int(suivant[t])) for t in np.flatnonzero(sh)]
    return ordres, {}


# ---------- strategies journalieres (D1) ----------

def T3(cx):
    ordres = []
    for t in np.flatnonzero(cx.pret):
        if t < 1:
            continue
        if cx.o[t] < cx.l[t - 1]:
            p = cx.l[t - 1] * 1.001
            ordres.append(Ordre(t, 1, "S0", prix=p, stop=p * 0.975, valide=1, fin=t + 1))
        elif cx.o[t] > cx.h[t - 1]:
            p = cx.h[t - 1] * 0.999
            ordres.append(Ordre(t, -1, "S0", prix=p, stop=p * 1.025, valide=1, fin=t + 1))
    return ordres, {}


def T4(cx):
    sem = pd.to_datetime(cx.jour).to_period("W-SUN").to_timestamp()
    idx = np.flatnonzero(np.r_[True, sem[1:] != sem[:-1]])
    bornes = list(idx) + [cx.n]
    ordres = []
    for w in range(1, len(idx)):
        a0, a, b = bornes[w - 1], bornes[w], bornes[w + 1]
        if not cx.pret[a]:
            continue
        hh, ll = cx.h[a0:a].max(), cx.l[a0:a].min()
        p = cx.o[a] + 0.4 * (hh - ll)
        ordres.append(Ordre(a, 1, "S0", prix=p, stop=ll, valide=b - a, fin=b))
    return ordres, {}


def _directionnel(cx, lundi):
    ordres = []
    # Demain est un lundi si aujourd'hui est dimanche (crypto, 7 jours sur 7)
    # ou vendredi (les autres marches).
    veille_lundi = (cx.wd == 6) if cx.crypto else (cx.wd == 4)
    for t in np.flatnonzero(cx.pret):
        if lundi and not veille_lundi[t]:
            continue
        if cx.c[t] > cx.o[t]:
            ordres.append(Ordre(t, 1, "S", prix=cx.h[t], valide=1, risque=2 * cx.atr[t], nmax=1))
        elif cx.c[t] < cx.o[t]:
            ordres.append(Ordre(t, -1, "S", prix=cx.l[t], valide=1, risque=2 * cx.atr[t], nmax=1))
    return ordres, {}


def S2(cx):
    return _directionnel(cx, False)


def S3(cx):
    return _directionnel(cx, True)


# ---------- swing points, retracements, figures ----------

def _points_court_terme(h, l):
    """Hauts et bas de court terme (J-1 et J+1 plus bas / plus hauts), en
    ignorant les bougies interieures. Rend des listes (indice, niveau,
    bougie de confirmation)."""
    n = len(h)
    utiles = [0]
    for i in range(1, n):
        if not (h[i] <= h[utiles[-1]] and l[i] >= l[utiles[-1]]):
            utiles.append(i)
    hauts, bas = [], []
    for j in range(1, len(utiles) - 1):
        a, i, b = utiles[j - 1], utiles[j], utiles[j + 1]
        if h[i] > h[a] and h[i] > h[b] and l[i] > l[a] and l[i] > l[b]:
            hauts.append((i, h[i], b))
        if l[i] < l[a] and l[i] < l[b] and h[i] < h[a] and h[i] < h[b]:
            bas.append((i, l[i], b))
    return hauts, bas


def T5(cx):
    hauts, bas = cx.get("ct", lambda: _points_court_terme(cx.h, cx.l))
    n = cx.n
    # dernier bas / haut de court terme confirme a chaque bougie
    dl, dh = np.full(n, np.nan), np.full(n, np.nan)
    for i, v, conf in bas:
        dl[conf:] = v
    for i, v, conf in hauts:
        dh[conf:] = v
    conf_opp = {1: np.array([z[2] for z in hauts]), -1: np.array([z[2] for z in bas])}
    ordres = []
    for tab, sens in ((bas, 1), (hauts, -1)):
        opp = hauts if sens > 0 else bas
        for j in range(1, len(tab) - 1):
            (_, v0, _), (i1, v1, _), (_, v2, conf2) = tab[j - 1], tab[j], tab[j + 1]
            moyen = (v1 < v0 and v1 < v2) if sens > 0 else (v1 > v0 and v1 > v2)
            if not moyen or not cx.pret[conf2]:
                continue
            # dernier sommet (creux) de court terme confirme a conf2, s'il est
            # posterieur au creux (sommet) de moyen terme
            q = np.searchsorted(conf_opp[sens], conf2, side="right") - 1
            if q < 0 or opp[q][0] <= i1:
                continue
            niveau = opp[q][1]
            a = cx.atr[conf2]
            if sens > 0:
                ordres.append(Ordre(conf2, 1, "S", prix=niveau + 0.01 * a, valide=20,
                                    stop=v2 - 0.01 * a, suivi="sp"))
            else:
                ordres.append(Ordre(conf2, -1, "S", prix=niveau - 0.01 * a, valide=20,
                                    stop=v2 + 0.01 * a, suivi="sp"))
    return ordres, {"sp+": dl - 0.01 * np.nan_to_num(cx.atr), "sp-": dh + 0.01 * np.nan_to_num(cx.atr)}


def S1(cx):
    m20, m50 = sma(cx.c, 20), sma(cx.c, 50)
    hausse = (m20 > m50) & (cx.c > m20)
    baisse = (m20 < m50) & (cx.c < m20)
    lo = hausse & (cx.l <= m20) & cx.pret
    sh = baisse & (cx.h >= m20) & cx.pret
    ll5, hh5 = E._roll(cx.l, 5, "min"), E._roll(cx.h, 5, "max")
    ordres = [_m(t, 1, stop=ll5[t] - 0.1 * cx.atr[t], sortie="x") for t in np.flatnonzero(lo)]
    ordres += [_m(t, -1, stop=hh5[t] + 0.1 * cx.atr[t], sortie="x") for t in np.flatnonzero(sh)]
    return ordres, {"x+": m20 < m50, "x-": m20 > m50}


def _fibo(cx, objectif_sommet):
    hs, ls = cx.sw
    ordres = []
    for tab, sens in ((hs, 1), (ls, -1)):
        opp = ls if sens > 0 else hs
        for i, v, conf in tab:
            t = int(conf)
            if not cx.pret[t]:
                continue
            d = derniers(opp, t, 1)
            if not d or d[0][0] >= i:
                continue
            depart = d[0][1]
            jambe = (v - depart) * sens
            a = cx.atr[t]
            if jambe < 2 * a:
                continue
            p = v - sens * 0.62 * jambe
            inval = v - sens * 0.79 * jambe
            stop = depart - sens * 0.1 * a
            tp = v if objectif_sommet else p + sens * a
            ordres.append(Ordre(t, sens, "L", prix=p, valide=20, stop=stop, tp=tp, annule=inval))
    return ordres, {}


def P1(cx):
    return _fibo(cx, False)


def P2(cx):
    return _fibo(cx, True)


def _ichi(cx):
    def f():
        h, l = cx.h, cx.l
        tk = (E._roll(h, 9, "max") + E._roll(l, 9, "min")) / 2
        kj = (E._roll(h, 26, "max") + E._roll(l, 26, "min")) / 2
        ssa = (tk + kj) / 2
        ssb = (E._roll(h, 52, "max") + E._roll(l, 52, "min")) / 2
        haut = np.fmax(E._dec(ssa, 26), E._dec(ssb, 26))
        bas = np.fmin(E._dec(ssa, 26), E._dec(ssb, 26))
        return tk, kj, ssa, ssb, haut, bas
    return cx.get("ichi", f)


def _ichi_ordres(cx, lo, sh):
    tk, kj, _, _, _, _ = _ichi(cx)
    ordres = [_m(t, 1, stop=min(kj[t], cx.l[t]) - 0.1 * cx.atr[t], sortie="kj")
              for t in np.flatnonzero(lo & cx.pret)]
    ordres += [_m(t, -1, stop=max(kj[t], cx.h[t]) + 0.1 * cx.atr[t], sortie="kj")
               for t in np.flatnonzero(sh & cx.pret)]
    return ordres, {"kj+": cx.c < kj, "kj-": cx.c > kj}


def _chikou(cx):
    _, _, _, _, haut, bas = _ichi(cx)
    c26 = E._dec(cx.c, 26)
    return ((cx.c > c26) & (cx.c > E._dec(haut, 26)), (cx.c < c26) & (cx.c < E._dec(bas, 26)))


def I1(cx):
    tk, kj, _, _, haut, bas = _ichi(cx)
    ch_l, ch_s = _chikou(cx)
    return _ichi_ordres(cx, croise_haut(tk, kj) & (cx.c > haut) & ch_l,
                        croise_bas(tk, kj) & (cx.c < bas) & ch_s)


def I2(cx):
    tk, kj, ssa, ssb, haut, bas = _ichi(cx)
    return _ichi_ordres(cx, croise_haut(cx.c, haut) & (ssa > ssb),
                        croise_bas(cx.c, bas) & (ssa < ssb))


def I3(cx):
    tk, kj, _, _, haut, bas = _ichi(cx)
    ch_l, ch_s = _chikou(cx)
    return _ichi_ordres(cx, croise_haut(cx.c, kj) & (cx.c > haut) & ch_l,
                        croise_bas(cx.c, kj) & (cx.c < bas) & ch_s)


def G1(cx):
    return toujours_en_position(cx, croise_haut(sma(cx.c, 20), sma(cx.c, 50)),
                                croise_bas(sma(cx.c, 20), sma(cx.c, 50)))


def A6(cx):
    return toujours_en_position(cx, croise_haut(sma(cx.c, 50), sma(cx.c, 200)),
                                croise_bas(sma(cx.c, 50), sma(cx.c, 200)))


def A1(cx):
    r = rsi(cx.c)
    return toujours_en_position(cx, croise_haut(r, np.full(cx.n, 30.0)),
                                croise_bas(r, np.full(cx.n, 70.0)))


def A2(cx):
    k, d = stochastique(cx.h, cx.l, cx.c)
    return toujours_en_position(cx, croise_haut(k, d) & (E._dec(k) < 20) & (E._dec(d) < 20),
                                croise_bas(k, d) & (E._dec(k) > 80) & (E._dec(d) > 80))


def A3(cx):
    m, up, lo_b = bollinger(cx.c, 20, 2.0)
    lo = croise_haut(cx.c, lo_b) & cx.pret
    sh = croise_bas(cx.c, up) & cx.pret
    ordres = [_m(t, 1, stop=min(cx.l[t], cx.l[t - 1]) - 0.1 * cx.atr[t], tp=m[t])
              for t in np.flatnonzero(lo) if m[t] > cx.c[t]]
    ordres += [_m(t, -1, stop=max(cx.h[t], cx.h[t - 1]) + 0.1 * cx.atr[t], tp=m[t])
               for t in np.flatnonzero(sh) if m[t] < cx.c[t]]
    return ordres, {}


def A4(cx):
    hs, ls = cx.sw
    ordres, vu = [], set()
    for tab, sens in ((ls, 1), (hs, -1)):
        for j in range(1, len(tab)):
            (i1, v1, _), (i2, v2, conf2) = tab[j - 1], tab[j]
            i1, i2, conf2 = int(i1), int(i2), int(conf2)
            if i2 - i1 < 5 or not cx.pret[conf2]:
                continue
            a = cx.atr[conf2]
            if abs(v2 - v1) > 0.5 * a:
                continue
            cou = cx.h[i1:i2 + 1].max() if sens > 0 else cx.l[i1:i2 + 1].min()
            extreme = min(v1, v2) if sens > 0 else max(v1, v2)
            haut_fig = (cou - extreme) * sens
            if haut_fig < 2 * a:
                continue
            for t in range(conf2, min(conf2 + 30, cx.n)):
                if (cx.c[t] - cou) * sens > 0:
                    if t not in vu:
                        vu.add(t)
                        ordres.append(_m(t, sens, stop=extreme - sens * 0.1 * a, tp=cou + sens * haut_fig))
                    break
                if (cx.c[t] - extreme) * sens < 0:
                    break
    return ordres, {}


def A5(cx):
    hh, ll = E._dec(E._roll(cx.h, 20, "max")), E._dec(E._roll(cx.l, 20, "min"))
    vm = E._dec(E._roll(cx.v, 20, "mean"))
    haut = hh - ll
    ok = cx.pret & (haut <= 4 * cx.atr) & (cx.v > 2 * vm)
    lo = ok & (cx.c > hh) & (E._dec(cx.c) <= hh)
    sh = ok & (cx.c < ll) & (E._dec(cx.c) >= ll)
    mid = (hh + ll) / 2
    ordres = [_m(t, 1, stop=mid[t], tp=hh[t] + haut[t]) for t in np.flatnonzero(lo)]
    ordres += [_m(t, -1, stop=mid[t], tp=ll[t] - haut[t]) for t in np.flatnonzero(sh)]
    return ordres, {}


def C1(cx):
    hs_, bs_ = E.structures(cx.x())
    m20 = sma(cx.c, 20)
    ordres = []
    for t in np.flatnonzero(cx.pret):
        if t < 3:
            continue
        for sens, st in ((1, hs_), (-1, bs_)):
            if not st[t - 1]:
                continue
            if sens > 0 and cx.c[t - 1] < m20[t - 1]:
                haut = max(cx.h[t - 3:t])
                if cx.c[t] > haut:
                    stop = min(cx.l[t - 3:t]) - 0.1 * cx.atr[t]
                    ordres.append(_m(t, 1, stop=stop, tp=cx.c[t] + 2 * (cx.c[t] - stop)))
            elif sens < 0 and cx.c[t - 1] > m20[t - 1]:
                bas = min(cx.l[t - 3:t])
                if cx.c[t] < bas:
                    stop = max(cx.h[t - 3:t]) + 0.1 * cx.atr[t]
                    ordres.append(_m(t, -1, stop=stop, tp=cx.c[t] - 2 * (stop - cx.c[t])))
    return ordres, {}


# ==========================================================================
#  Catalogue : (code, livre, nom, fonction, unites de temps)
# ==========================================================================

TOUTES = ("M15", "H1", "H4", "D1")
CATALOGUE = [
    ("M1", "MACD systeme complet (Crypto-Analyse)", "croisement MACD, toujours en position", M1, TOUTES),
    ("M2", "MACD systeme complet (Crypto-Analyse)", "MACD + MM200", M2, TOUTES),
    ("M3", "MACD systeme complet (Crypto-Analyse)", "MACD + divergence", M3, TOUTES),
    ("F1", "Moyennes mobiles (Master Feng)", "repli sur la MME50 en tendance MME200", F1, TOUTES),
    ("V1", "Visual Forex (Eric David)", "phases MME 12/20/50 + MACD, RSI, SAR", V1, TOUTES),
    ("V2", "Forex Evolution (Eric David)", "BBand Stop + MACD", V2, TOUTES),
    ("V3", "Forex Evolution (Eric David)", "tendance MME + croisement MACD", V3, TOUTES),
    ("E1", "Vivre du trading (Elder)", "Triple Ecran, Force Index", E1, TOUTES),
    ("E2", "Vivre du trading (Elder)", "Triple Ecran, stochastique", E2, TOUTES),
    ("E3", "Vivre du trading (Elder)", "systeme parabolique (SAR)", E3, TOUTES),
    ("E4", "Vivre du trading (Elder)", "canal MME13 plat", E4, TOUTES),
    ("E5", "Vivre du trading (Elder)", "Bollinger, sortie de bandes etroites", E5, TOUTES),
    ("T1", "Day et swing trading (MTT)", "cassure de la 1re heure", T1, ("M15",)),
    ("T2", "Day et swing trading (MTT)", "systeme final futures (14h-16h)", T2, ("M15",)),
    ("T3", "Day et swing trading (MTT)", "comblement de gap", T3, ("D1",)),
    ("T4", "Day et swing trading (MTT)", "volatilite hebdomadaire 40 %", T4, ("D1",)),
    ("T5", "Day et swing trading (MTT)", "swing points de moyen terme", T5, TOUTES),
    ("S1", "Swing Trading (MTT)", "retour sur MM20 (MM20 > MM50)", S1, TOUTES),
    ("S2", "Swing Trading (MTT)", "swing directionnel", S2, ("D1",)),
    ("S3", "Swing Trading (MTT)", "swing directionnel, lundi seulement", S3, ("D1",)),
    ("W1", "FX Wolf strategie CM", "cassure du range asiatique", W1, ("M15",)),
    ("W2", "FX Wolf strategie CM", "range asiatique + pivot", W2, ("M15",)),
    ("P1", "Fibonacci Secret (Pips Leader)", "retracement 62-79 %, objectif 1 ATR", P1, TOUTES),
    ("P2", "Fibonacci Secret (Pips Leader)", "retracement 62-79 %, objectif le sommet", P2, TOUTES),
    ("I1", "Ichimoku (Ventura, DailyFX, Peloille)", "croisement Tenkan/Kijun fort", I1, TOUTES),
    ("I2", "Ichimoku (Ventura, DailyFX, Peloille)", "sortie du nuage", I2, TOUTES),
    ("I3", "Ichimoku (Ventura, DailyFX, Peloille)", "croisement de la Kijun", I3, TOUTES),
    ("G1", "Guides Forex (Forex simplifie...)", "croisement MM20 / MM50", G1, TOUTES),
    ("G2", "Guides Forex (IC Markets)", "cassure du point pivot", G2, ("M15", "H1")),
    ("A1", "Analyse technique (Clement, Kabbaj...)", "RSI 30 / 70", A1, TOUTES),
    ("A2", "Analyse technique (Clement, Kabbaj...)", "stochastique 20 / 80", A2, TOUTES),
    ("A3", "Analyse technique (Clement, Kabbaj...)", "Bollinger, retour dans la bande", A3, TOUTES),
    ("A4", "Le Chartisme, Maitriser l'AT", "double creux / double sommet", A4, TOUTES),
    ("A5", "Le Chartisme, Clement", "rectangle casse avec volume", A5, TOUTES),
    ("A6", "Analyse technique (Clement, Kabbaj...)", "croisement MM50 / MM200", A6, TOUTES),
    ("C1", "Chandeliers japonais (Baron...)", "structure validee par une cloture", C1, TOUTES),
]


def lance(df, s, tf, info, sans_frais=False):
    if sans_frais:
        df = df.assign(spread=0)
    cx = Ctx(df, tf, s in E.CRYPTO)
    prix = float(df["close"].iloc[-1])
    sw_l, sw_s = (0.0, 0.0) if sans_frais else E.swap_bps(info, prix)
    glis = 0.0 if sans_frais else GLIS
    res = {}
    for code, _, _, f, uts in CATALOGUE:
        if tf not in uts:
            continue
        ordres, extra = f(cx)
        res[code] = simule(df, cx.x(), ordres, extra, sw_l, sw_s, s in E.CRYPTO,
                           info.get("swap_rollover3days", 3), info["point"], glis)
    return res


def verif_causalite(df, tf, essais=12, graine=3):
    """Les ordres decides a t ne changent pas si l'on coupe les donnees apres t."""
    def cles(cx, t_max=None):
        out = {}
        for code, _, _, f, uts in CATALOGUE:
            if tf not in uts:
                continue
            for od in f(cx)[0]:
                if t_max is None or od.t == t_max:
                    out[(code, od.t, od.sens)] = (od.genre, od.prix, od.stop, od.tp, od.prix2,
                                                  od.stop2, od.tp2, od.annule)
        return out
    plein = cles(Ctx(df, tf))
    rng = np.random.default_rng(graine)
    ts = sorted({k[1] for k in plein})
    pts = set(int(v) for v in rng.choice(ts, min(len(ts), essais), replace=False))
    pts |= set(int(v) for v in rng.integers(300, len(df) - 1, essais // 2))
    for t in sorted(pts):
        coupe = cles(Ctx(df.iloc[:t + 1].reset_index(drop=True), tf), t)
        attendu = {k: v for k, v in plein.items() if k[1] == t}
        if set(coupe) != set(attendu):
            return f"FUITE {tf} bougie {t} : {sorted(set(coupe) ^ set(attendu))[:3]}"
        for k in coupe:
            if not np.allclose(np.array(coupe[k][1:], float), np.array(attendu[k][1:], float),
                               equal_nan=True):
                return f"FUITE {tf} {k} : {coupe[k]} contre {attendu[k]}"
    return f"{tf} : causalite verifiee sur {len(pts)} bougies"


def main() -> int:
    m15 = pd.read_pickle(E.M15)
    d1 = pd.read_pickle(E.D1)
    jeux = []
    for s, df in m15["barres"].items():
        df = df.sort_values("time").reset_index(drop=True)
        jeux += [(s, "M15", df), (s, "H1", E.agrege(df, "1h")), (s, "H4", E.agrege(df, "4h"))]
    for s, df in d1["barres"].items():
        jeux.append((s, "D1", df.sort_values("time").reset_index(drop=True)))

    for s, tf, df in jeux:
        if s == "BTCUSD" and tf in ("M15", "H1", "D1"):
            print(verif_causalite(df.iloc[-4000:].reset_index(drop=True), tf), flush=True)

    res = {}
    for s, tf, df in jeux:
        r = lance(df, s, tf, d1["infos"][s])
        for code, v in r.items():
            res[(s, tf, code)] = v
        print(f"  {s:10s} {tf:3s} {sum(len(v) for v in r.values()):6d} trades", flush=True)

    brut = {}
    for s, tf, df in jeux:
        if s in E.PRINCIPAUX:
            for code, v in lance(df, s, tf, d1["infos"][s], sans_frais=True).items():
                brut.setdefault(code, []).extend(v)

    def reunis(filtre):
        return {code: sorted(sum((v for (s, tf, c), v in res.items() if c == code and filtre(s, tf)), []))
                for code, *_ in CATALOGUE}

    principal = reunis(lambda s, tf: s in E.PRINCIPAUX)
    tous = reunis(lambda s, tf: True)

    print("\n" + "=" * 110)
    print("CHAQUE STRATEGIE DES LIVRES : BTC ET OR REUNIS (toutes unites de temps prevues) / TOUS LES MARCHES / SANS FRAIS")
    print("=" * 110)
    livre = None
    for code, lv, nom, _, uts in CATALOGUE:
        if lv != livre:
            print(f"\n{lv}")
            livre = lv
        print(E.ligne(f"{code} {nom}"[:32], E.bilan(principal[code])))
        b = E.bilan(tous[code])
        bb = E.bilan(sorted(brut.get(code, [])))
        if b["n"]:
            print(f"      tous marches : {b['n']:6d} trades  PF {b['pf']:4.2f}  {b['moy']:+.3f} R (t {b['t']:+.1f})"
                  f"  total {b['total']:+10.2f} $   |   BTC+or SANS FRAIS : "
                  + (f"PF {bb['pf']:4.2f}  {bb['moy']:+.3f} R (t {bb['t']:+.1f})" if bb["n"] else "-"))

    print("\n" + "=" * 110)
    print("DETAIL BTC ET OR PAR UNITE DE TEMPS")
    print("=" * 110)
    for s in E.PRINCIPAUX:
        for tf in TOUTES:
            print(f"\n{s} {tf}")
            for code, lv, nom, _, uts in CATALOGUE:
                if tf in uts:
                    print(E.ligne(f"{code} {nom}"[:32], E.bilan(res.get((s, tf, code), []))))

    combos = [(k, E.bilan(v)) for k, v in res.items() if len(v) >= 30]
    pos = [z for z in combos if z[1]["total"] > 0]
    sig = [z for z in combos if z[1]["t"] > 2]
    nb_ok = sum(1 for code, *_ in CATALOGUE if principal[code] and E.bilan(principal[code])["total"] > 0)
    print("\n" + "=" * 110)
    print(f"RESUME : {len(CATALOGUE)} strategies ; rentables sur BTC + or reunis : {nb_ok}")
    print(f"  combinaisons marche x unite de temps x strategie avec au moins 30 trades : {len(combos)}")
    print(f"  rentables : {len(pos)}  |  gain significatif (t > 2) : {len(sig)} "
          f"(le hasard seul en donnerait ~{0.023 * len(combos):.0f})")
    for (s, tf, code), b in sorted(sig, key=lambda z: -z[1]["t"])[:20]:
        print(f"    {s:10s} {tf:3s} {code}  {b['n']:5d} trades  PF {b['pf']:4.2f}  "
              f"{b['moy']:+.3f} R  t {b['t']:+.1f}  total {b['total']:+.2f} $")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
