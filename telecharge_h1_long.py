"""Historique H1 long : neuf ans de bougies BTCUSDT, une seule source.

LE PROBLEME QUE CE FICHIER RESOUT, chiffre le 2026-09-15. Le passage en H1 a
fait ce qu'on lui demandait : entrer au hasard coute -0.45 R en M1 et
seulement -0.012 R en H1, la friction est effacee. Mais il a coute les
echantillons. La fenetre d'entrainement H1 contient 890 occasions sans
chevauchement, 143 par bloc de validation.

    ecart-type d'un trade          ~1.4 R
    avantage recherche            ~0.02 R
    trades necessaires pour le voir   (1.4/0.02)^2 = 4 900

On est a un facteur trente de pouvoir MESURER l'avantage, sans parler de
l'apprendre. C'est ce qui a fait echouer TabM : son gain apparent (+0.030 R)
venait d'un seul bloc sur quatre, celui ou acheter aveuglement rapportait
deja +0.135 R.

    source              debut      barres H1
    MT5 (actuel)      2023-02-21     30 379
    Binance futures   2019-10        ~61 000
    Binance spot      2017-08        ~79 500   <- retenu

POURQUOI LE SPOT ET PAS LES FUTURES. Deux ans et demi de plus, et surtout UNE
SEULE source sur toute la periode : melanger spot avant 2019 et futures apres
creerait une couture dans les features de flux exactement au milieu du jeu,
et le modele apprendrait la couture.

CE QUE CE CHOIX COUTE, ET QU'IL FAUT GARDER EN TETE. Le spot Binance n'est pas
le CFD du courtier : les prix different de quelques points de base et les
frais n'ont rien a voir. On ne reprend donc de Binance que la FORME du marche
(OHLC et flux) ; la friction reste celle du courtier, 2.61 bps de spread plus
le slippage — c'est le choix prudent, le spot reel coute moins cher.

Les annees 2017-2019 sont un marche tres different. Ce n'est pas une objection
ici : le walk-forward n'entraine que sur le passe et ne valide que sur le
futur, donc ces annees ne servent jamais que de matiere d'entrainement pour
des blocs posterieurs. Elles ne peuvent pas flatter un resultat.

    python telecharge_h1_long.py [ut]     ut = 1h, 5m, 15m...
"""

import io
import os
import json
import zipfile
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from build_binance_features import _get, VISION, SYMBOL_BN, N_THREADS, CACHE_DIR

MOIS_DEBUT = (2017, 8)
# Unite de temps, passee en argument. Les archives Binance spot existent au
# meme endroit pour toutes les echelles ; seul le nom du dossier change.
#
# POURQUOI LE M5 EXISTE ICI. Mesure du 2026-09-16, sortie sur barriere
# uniquement comme l'environnement et le live :
#
#     UT   SL   friction/R   E[R] hasard   duree med   occasions   requis
#     M5  4.0      0.099R       -0.0892         44b       5 811   +0.0892R
#     H1  2.0      0.047R       -0.0233         13b       1 640   +0.0233R
#
# Le M5 a stop 4xATR est la seule configuration qui satisfasse les deux
# conditions a la fois : un avantage requis (+0.089 R) inferieur a ce que le
# modele produit (+0.09 a +0.13 R), et assez d'occasions INDEPENDANTES pour
# qu'un ecart devienne lisible. Sur neuf ans il donnerait ~21 000 occasions
# contre 2 314 en H1 — neuf fois la contrainte qui bloque ce depot.
UT = "1h"

# Colonnes du kline spot, par position (le CSV n'a pas d'entete).
I_T, I_O, I_H, I_L, I_C, I_V, I_QV, I_NT, I_TBB = 0, 1, 2, 3, 4, 5, 7, 8, 9


def _mois(annee, mois):
    url = (f"{VISION}/data/spot/monthly/klines/{SYMBOL_BN}/{UT}/"
           f"{SYMBOL_BN}-{UT}-{annee:04d}-{mois:02d}.zip")
    try:
        z = zipfile.ZipFile(io.BytesIO(_get(url, raw=True, essais=2)))
    except Exception:
        return None
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, low_memory=False)
    t = pd.to_numeric(df.iloc[:, I_T], errors="coerce")
    df = df[t.notna()]                       # certaines archives ont une entete
    if df.empty:
        return None
    t = pd.to_numeric(df.iloc[:, I_T], errors="coerce")
    # Binance a bascule de ms a us dans les archives recentes.
    unite = "us" if t.iloc[0] > 1e14 else "ms"
    col = lambda i: pd.to_numeric(df.iloc[:, i], errors="coerce").to_numpy()
    return pd.DataFrame({
        "time": pd.to_datetime(t, unit=unite),
        "open": col(I_O), "high": col(I_H), "low": col(I_L), "close": col(I_C),
        "volume": col(I_V), "quote_vol": col(I_QV),
        "nb_trades": col(I_NT), "taker_buy_base": col(I_TBB),
    })


def _jour(jour):
    """Le mois courant n'a pas encore d'archive mensuelle."""
    url = (f"{VISION}/data/spot/daily/klines/{SYMBOL_BN}/{UT}/"
           f"{SYMBOL_BN}-{UT}-{jour:%Y-%m-%d}.zip")
    try:
        z = zipfile.ZipFile(io.BytesIO(_get(url, raw=True, essais=2)))
    except Exception:
        return None
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, low_memory=False)
    t = pd.to_numeric(df.iloc[:, I_T], errors="coerce")
    df = df[t.notna()]
    if df.empty:
        return None
    t = pd.to_numeric(df.iloc[:, I_T], errors="coerce")
    unite = "us" if t.iloc[0] > 1e14 else "ms"
    col = lambda i: pd.to_numeric(df.iloc[:, i], errors="coerce").to_numpy()
    return pd.DataFrame({
        "time": pd.to_datetime(t, unit=unite),
        "open": col(I_O), "high": col(I_H), "low": col(I_L), "close": col(I_C),
        "volume": col(I_V), "quote_vol": col(I_QV),
        "nb_trades": col(I_NT), "taker_buy_base": col(I_TBB),
    })


def main() -> int:
    global UT
    import sys
    if len(sys.argv) > 1:
        UT = sys.argv[1]
    os.makedirs(CACHE_DIR, exist_ok=True)
    f_cache = os.path.join(CACHE_DIR, f"klines_{UT}_spot.pkl")
    f_vides = os.path.join(CACHE_DIR, f"klines_{UT}_spot_absents.json")
    sortie = f"klines_{UT}_spot_{SYMBOL_BN}.pkl"

    cache = pd.read_pickle(f_cache) if os.path.exists(f_cache) else pd.DataFrame()
    vides = set(json.load(open(f_vides))) if os.path.exists(f_vides) else set()
    deja = set()
    if not cache.empty:
        deja = set(pd.to_datetime(cache["time"]).dt.strftime("%Y-%m"))

    auj = dt.date.today()
    mois = []
    a, m = MOIS_DEBUT
    while (a, m) < (auj.year, auj.month):
        cle = f"{a:04d}-{m:02d}"
        if cle not in deja and cle not in vides:
            mois.append((a, m))
        m += 1
        if m == 13:
            a, m = a + 1, 1

    print(f"cache : {len(deja)} mois presents, {len(vides)} absents connus")
    if mois:
        print(f"telechargement de {len(mois)} mois sur {N_THREADS} fils...")
        faits = [0]

        def trav(am):
            r = _mois(*am)
            faits[0] += 1
            if faits[0] % 10 == 0:
                print(f"  {faits[0]}/{len(mois)}", flush=True)
            return am, r

        with ThreadPoolExecutor(N_THREADS) as ex:
            res = list(ex.map(trav, mois))
        neufs = [r for _, r in res if r is not None and len(r)]
        for am, r in res:
            if r is None or not len(r):
                vides.add(f"{am[0]:04d}-{am[1]:02d}")
        if neufs:
            cache = pd.concat([cache] + neufs, ignore_index=True)
        json.dump(sorted(vides), open(f_vides, "w"))

    # Mois courant : archives journalieres.
    jours = [dt.date(auj.year, auj.month, 1) + dt.timedelta(days=i)
             for i in range(auj.day)]
    with ThreadPoolExecutor(N_THREADS) as ex:
        quot = [r for r in ex.map(_jour, jours) if r is not None and len(r)]
    if quot:
        cache = pd.concat([cache] + quot, ignore_index=True)

    if cache.empty:
        print("aucune donnee")
        return 1

    cache = (cache.drop_duplicates("time").sort_values("time")
                  .reset_index(drop=True))
    cache.to_pickle(f_cache)

    # Un trou de plus d'une heure signalerait une periode manquante : le
    # warmup des indicateurs le masquerait sans rien dire.
    ecarts = cache["time"].diff().dt.total_seconds() / 3600.0
    trous = ecarts[ecarts > 1.5]
    print(f"\n{len(cache):,} bougies H1  |  {cache['time'].iloc[0]} -> "
          f"{cache['time'].iloc[-1]}")
    print(f"trous > 1 h : {len(trous)}"
          + (f"  (le plus long {trous.max():.0f} h)" if len(trous) else ""))
    cache.to_pickle(sortie)
    print(f"{sortie} ecrit : {cache.shape}")

    ans = (cache["time"].iloc[-1] - cache["time"].iloc[0]).days / 365.25
    # Duree MEDIANE d'un trade a la geometrie retenue pour cette echelle,
    # mesuree sans sortie au temps par mesure_court_terme.py.
    duree = {"1h": 13, "5m": 44, "15m": 13, "1m": 33}.get(UT, 24)
    print(f"\n{ans:.1f} ans en {UT}.")
    print(f"occasions sans chevauchement (duree mediane {duree} barres, "
          f"70 % en train) : ~{int(len(cache) * 0.70 / duree):,}")
    print(f"  repere : 2 314 en H1 aujourd'hui, ~4 900 necessaires pour "
          f"qu'un avantage de 0.02 R soit lisible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
