"""Construit le jeu XAUUSD au MEME format que le BTC — 256 colonnes.

POURQUOI L'OR. Le mur de ce depot est le nombre d'occasions INDEPENDANTES, et
un second instrument n'en ajoute que s'il est DECORRELE. Mesure du
2026-09-16 : la correlation entre les rendements de la strategie sur BTC et
sur or vaut 0.088 — pratiquement zero. Les cryptos, elles, correlent a 0.62
(BCH) jusqu'a 0.81 (ETH) et n'ajouteraient que 10 % d'occasions, contre 84 %
pour l'or.

Et son avantage mesure est SUPERIEUR : +16.5 a +20.1 R par an contre +12.5
pour le meilleur reglage du BTC, avec un avantage par trade a 3.9 a 6.0
ecarts-types contre ~2.0.

CE QUI DIFFERE DU BTC, ET CE QUI N'EN DIFFERE PAS.

  LES COLONNES SONT LES MEMES. Les quatre colonnes de carnet — part acheteuse
  agressive, taille de trade, intensite — ont ete retirees le meme jour : elles
  n'existent que sur Binance et leur retrait a ete mesure gratuit. Les 256
  restantes se calculent a partir d'un OHLCV pur, donc de n'importe quel
  instrument.

  LA GEOMETRIE, NON. L'ATR relatif de l'or vaut 6.6 points de base contre 15.9
  pour le BTC : un stop de 6xATR fait 95 bps sur l'un et 40 sur l'autre. Chaque
  instrument garde sa largeur — 6xATR pour le BTC, 10x pour l'or — et sa
  friction — 0.57 point de base de spread contre 2.24. Ce fichier ne produit
  que les features ; la geometrie vit dans la configuration.

  LA SOURCE, NON PLUS. Le BTC vient des archives Binance, l'or de MetaTrader,
  qui en porte 7.1 ans en M5. Aucune donnee externe a telecharger.

LES INTERRUPTIONS. L'or ferme cinq fois par semaine — week-end et pause
quotidienne. Les indicateurs se calculent sur la suite des BARRES, pas sur le
calendrier : une moyenne mobile traverse donc le week-end, ce qui est
exactement ce que ferait un indicateur en production. En revanche le
reechantillonnage en H1 et H4 passe par le calendrier et cree des trous, que
`dropna` retire — on perd les quelques barres qui suivent chaque reouverture.
Le compte des barres perdues est affiche : c'est le prix de l'operation, et il
doit rester marginal.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import prepare_m5 as P

SYMBOLE = "XAUUSD"
SORTIE = "data_cache_XAUUSD_M5.pkl"
# 500 000 N'EST PAS UNE LIMITE TECHNIQUE, C'EST LA FIN DES DONNEES DENSES.
#
# Avec le plafond de barres du terminal porte a « illimite », MetaTrader rend
# 573 465 bougies M5 remontant a 2007 — douze annees de plus. Mesure du
# 2026-09-19 avant d'en faire quoi que ce soit :
#
#     periode              barres   ans   barres/sem   ATR relatif
#     2007-10 -> 2019-09   79 095  12.0          127      4.68 bps
#     2019-09 -> 2026-09  494 232   7.0        1 358      6.63 bps
#
# Une semaine d'or en M5 en compte ~2 016. La partie ancienne en a CENT
# VINGT-SEPT : ce ne sont pas des seances, ce sont des fragments epars. Les
# indicateurs glissants se calculent sur la suite des BARRES, donc une
# moyenne de vingt barres y enjamberait des semaines entieres et ne
# decrirait rien. Et l'ATR relatif y vaut 4.68 contre 6.63 : le meme
# multiplicateur de stop n'y designe pas le meme trade.
#
# Douze ans de calendrier pour 16 % de barres en plus, dont la geometrie
# differe de 40 % : l'historique ancien est ecarte. 500 000 barres s'arretent
# juste avant lui.
N_BARRES = 500_000


def charge_mt5(symbole: str, n: int) -> pd.DataFrame:
    """OHLCV M5 du courtier, dans le schema attendu par `prepare_m5`."""
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 indisponible : {mt5.last_error()}")
    mt5.symbol_select(symbole, True)
    r = mt5.copy_rates_from_pos(symbole, mt5.TIMEFRAME_M5, 0, n)
    mt5.shutdown()
    if r is None or len(r) == 0:
        raise RuntimeError(f"aucune barre pour {symbole}")
    d = pd.DataFrame(r)
    d["time"] = pd.to_datetime(d["time"], unit="s")
    # `tick_volume` tient lieu de volume : l'or n'a pas de volume echange
    # publie par le courtier. Aucune colonne de l'observation n'en depend en
    # niveau — seulement en rang ou en rapport — donc la substitution ne
    # change pas l'echelle des features.
    d["volume"] = d["tick_volume"].astype(float)
    return d[["time", "open", "high", "low", "close", "volume"]]


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_BARRES
    m5 = charge_mt5(SYMBOLE, n).sort_values("time").reset_index(drop=True)
    print(f"source : {len(m5):,} bougies M5  "
          f"{m5['time'].iloc[0]} -> {m5['time'].iloc[-1]}")

    dt = m5["time"].diff().dt.total_seconds()
    trous = int((dt > 1800).sum())
    semaines = (m5["time"].iloc[-1] - m5["time"].iloc[0]).days / 7
    print(f"         {trous:,} interruptions de plus de 30 min "
          f"({trous/max(semaines,1):.1f} par semaine)")

    m5, colonnes, _ = P.construit(m5, avec_flux=False)

    avant = len(m5)
    m5 = m5.replace([np.inf, -np.inf], np.nan)
    m5 = m5.dropna(subset=colonnes + ["atr_14"]).reset_index(drop=True)
    print(f"M5     : {avant:,} -> {len(m5):,} apres dropna "
          f"({100*(1-len(m5)/avant):.1f} % perdues au warmup et aux trous)")
    print(f"periode: {m5['time'].iloc[0]} -> {m5['time'].iloc[-1]}")

    # LA LISTE DE saint_core FAIT FOI, comme pour le BTC. Un desaccord ici se
    # traduirait plus tard par un KeyError illisible en plein entrainement.
    from saint_core import FEATURE_COLS
    manquantes = [c for c in FEATURE_COLS if c not in m5.columns]
    en_trop = [c for c in colonnes if c not in FEATURE_COLS]
    if manquantes or en_trop:
        print(f"\nDESACCORD avec saint_core.FEATURE_COLS "
              f"({len(FEATURE_COLS)} attendues)")
        print(f"  absentes du jeu   : {manquantes[:8]}")
        print(f"  produites en trop : {en_trop[:8]}")
        return 1
    print(f"accord avec saint_core.FEATURE_COLS ({len(FEATURE_COLS)} colonnes)")

    atr_rel = float(np.median(m5["atr_14"] / m5["close"]))
    print(f"\nATR relatif {1e4*atr_rel:.1f} points de base "
          f"(BTC : 15.9) — la geometrie de l'or n'est donc pas celle du BTC")

    m5.to_pickle(SORTIE)
    print(f"ecrit dans {SORTIE} ({len(m5):,} lignes, {len(m5.columns)} colonnes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
