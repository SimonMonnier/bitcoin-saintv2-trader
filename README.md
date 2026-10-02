# KAIROS — trader le BTCUSD en M5 par apprentissage par renforcement

KAIROS apprend à trader le bitcoin (BTCUSD) en bougies de 5 minutes, chez le
courtier Vantage via MetaTrader 5. Un modèle SAINT (transformer tabulaire)
est d'abord entraîné à imiter un expert LightGBM, puis il apprend par PPO en
jouant des journées de trading avec les vrais frais du courtier. Il choisit
le sens du trade, l'objectif, le stop et la taille de la mise.

> **Projet de recherche.** Le live ne trade que sur un **compte démo** : le
> moteur refuse tout compte réel. Les résultats passés, même sur des périodes
> jamais vues, ne garantissent rien. Rien ici n'est un conseil financier.

---

## 1. Le jeu (l'entraînement)

Tout est dans `jeu_kairos.py` (configuration `JeuConfig`, en tête du fichier).

**Les données** — bougies 5 minutes du BTC sur Binance depuis août 2017
(951 157 bougies), avec ce que le prix seul ne dit pas : le flux acheteur et le
nombre de trades de Binance, l'écart de prix Coinbase / Binance, le financement
des perpétuels et l'heure de la journée. Les frais sont ceux de Vantage :
spread mesuré par créneau (jour × heure UTC), glissements de 0,5 bps à l'entrée
et 1 bps à la sortie, swap de −20 %/an sur les achats.

**Une partie = une journée UTC** :
- 10 jetons par jour (10 trades au plus), une position à la fois ;
- un trade = un sens (achat ou vente) + un objectif et un stop de 1, 2, 4 ou
  8 ATR (ATR jamais sous 20 bps), fermé au plus tard après 96 bougies (8 h) ;
- la journée s'arrête si elle perd 3 R ;
- un trade n'est permis que si l'expert LightGBM classe la bougie dans ses
  15 % les plus prometteuses dans ce sens (`porte_rang_expert` = 0,85).

**La mise** — le modèle choisit aussi combien il risque, à chaque trade :
- le **risque** : la part du compte perdue si le stop est touché, de 1 à 100 % ;
- l'**allocation** (10 à 100 %) et la **confiance** (20 à 100 %, selon ses
  erreurs récentes), qui réduisent ce risque ;
- le **lot** : 20 tailles entre 0,01 et le maximum permis par le solde, la
  marge (levier 1:500) et le courtier.

**Un vrai compte** — en validation et en test, les journées se suivent sur un
seul solde de 1 000 $ qui grossit ou fond avec les gains et les pertes. Les
trades restent ouverts après minuit, la marge est comptée, et le solde ne
descend jamais sous zéro (`jeu_rollout.py`, `jeu_compte.py`).

## 2. Le modèle

| Partie | Détail |
|---|---|
| Entrées | 40 colonnes M5 (`prepare_btc_m5.py`) + 4 colonnes de l'expert (prédictions et rangs) + 11 colonnes d'état : jetons, score du jour, temps restant, places occupées, équité, baisse depuis le plus haut, marge libre, résultats récents, mauvaises surprises, séries de pertes |
| Mémoire | les 4 dernières bougies |
| Tronc | SAINT (`saint_core.py`) : dimension 8, 2 blocs, 1 tête d'attention |
| Têtes du signal | achat, vente, objectif, stop, valeur |
| Têtes de la mise | lot, risque, allocation, confiance |
| Expert | 2 LightGBM (achat, vente), qui prédisent le R net moyen des 16 couples objectif / stop |

**L'apprentissage, pour chaque bloc** :
1. l'expert LightGBM apprend sur les bougies d'entraînement (validation croisée
   interne purgée) ;
2. le modèle imite l'expert : 10 trades enseignés par jour, 3 passes ;
3. 40 epochs de PPO, les frais montant de 0 à 100 % sur les 10 premières.

**Deux notes séparées** (run m5_20) :
- le signal (entrée, objectif, stop, valeur) est noté sur le gain du trade,
  rapporté à 1 % du compte ;
- les quatre têtes de mise sont notées sur la **croissance du compte en
  pourcentage** (le logarithme du solde, la règle de Kelly). Miser gros paye
  quand le signal est fort ; tout miser, jamais : la ruine est la pire note ;
- un **mur** empêche les têtes de mise de modifier le tronc, donc le signal.

**Le modèle sauvegardé** — à chaque epoch, le modèle est rejoué sur la
validation. On garde celui dont le compte grossit le plus, parmi les epochs
dont la pire baisse reste au-dessus de −50 %.

## 3. La validation : 10 blocs purgés

L'historique est coupé en 10 blocs d'environ 11 mois. Chaque bloc est testé à
son tour par un modèle **neuf**, entraîné sur tous les autres. La validation
est le bloc le plus éloigné, et 388 bougies sont retirées autour du test et de
la validation, pour qu'aucun trade d'entraînement ne lise leurs prix. Pour
chaque bloc, l'expert et les normalisations sont figés dans
`pipeline_<run>_bloc<k>.json` : le test peut être rejoué à l'identique
(`rejoue_test_bloc.py`).

Un résultat négatif est définitif. Un résultat positif est un plafond : chaque
modèle a vu le passé **et** l'avenir des autres blocs.

**Le modèle final (« deploy »)** — après les 10 blocs, un nouveau modèle imite
l'expert appris sur tout l'historique, puis les 10 meilleurs modèles de blocs,
chacun sur sa propre validation, puis fait une passe de PPO sur toutes les
journées. C'est lui que trade le live :
`deploy_<run>_ensemble_validation_ddsafe.pth` et son pipeline.

## 4. Le live (compte démo)

```bash
python kairos_live.py
```

Cette commande ouvre l'interface (`kairos_gui.py`) et démarre l'agent.
- `--check` vérifie seulement que le modèle final est prêt ;
- `--console` démarre l'agent sans interface.

Le moteur (`kairos_m5_live.py`) :
- charge le modèle final et son pipeline. Il refuse de démarrer s'ils manquent
  ou s'ils ne viennent pas des 10 blocs ;
- reconstruit les mêmes 40 colonnes que le jeu, en direct : bougies et flux
  Binance (`flux_live.py`), Coinbase, financement ;
- décide à la clôture de chaque bougie M5, avec les mêmes règles que le jeu :
  10 jetons par jour UTC, vie de 3 R, porte de l'expert, mise choisie par le
  modèle et bornée par la marge et le lot du courtier ;
- passe ses ordres dans MT5 sur le symbole BTCUSD de Vantage. **Il refuse tout
  compte qui n'est pas un compte démo.**

Le run visé est fixé dans `LiveConfig.kairos_prefixe` (`kairos_live.py`).

## 5. Installation et commandes

Windows, Python 3.10, carte NVIDIA (CUDA 12.1), MetaTrader 5 connecté à Vantage.

```bash
pip install -r requirements.txt
```

**Reconstruire les données** (fichiers `.pkl`, non versionnés), dans l'ordre :

| Commande | Produit |
|---|---|
| `python telecharge_h1_long.py 5m` | `klines_5m_spot_BTCUSDT.pkl` : bougies 5 min Binance depuis 2017 |
| `python telecharge_coinbase_5m.py` | `cache_coinbase_5m_BTCUSD.pkl` : bougies 5 min Coinbase |
| `python etude_portage.py` | `cache_portage.pkl` : financement des perpétuels Binance |
| `python telecharge_m15_mt5.py` | `cache_m15_mt5.pkl` : spread Vantage (MT5 ouvert) |
| `python prepare_btc_m5.py` | `data_cache_BTCUSD_M5_BINANCE.pkl` : le cache du jeu |

**Entraîner** — double-clic sur `LANCER.bat`, ou `powershell -ExecutionPolicy
Bypass -File lancer.ps1`. Le script arrête l'entraînement en cours, range les
fichiers du run précédent dans `journaux/`, lance `jeu_kairos.py` dans sa
fenêtre et ouvre la veille. Comptez environ 2 h par bloc, donc 20 h pour les 10.

| Fichier | Rôle |
|---|---|
| `VEILLE.bat` / `VEILLE_DETAIL.bat` | suivre le journal du run, sans rien arrêter |
| `ARRETER.bat` | arrêter l'entraînement, sans rien effacer |
| `python jeu_kairos.py --rebuild-deploy-ensemble` | refaire le modèle final d'un run terminé |
| `python rejoue_test_bloc.py <run> <bloc>` | rejouer le test d'un bloc depuis son pipeline figé |

**Tests** :

```bash
python test_jeu.py
python -m unittest test_jeu_rollout test_jeu_compte test_jeu_couts test_jeu_artifacts
```

`test_jeu.py` passe en entier. Dans les tests unitaires, 9 tests de
`test_jeu_rollout.py` échouent : ils appellent le moteur sans prix, alors que
les têtes de mise en ont besoin. Les runs fournissent toujours les prix.

## 6. Les fichiers

| Fichier | Rôle |
|---|---|
| `jeu_kairos.py` | le jeu : configuration, modèle, expert, PPO, validation en blocs, modèle final |
| `jeu_rollout.py` | le déroulement des journées sur un compte (positions, marge, solde) |
| `jeu_compte.py` | la taille des positions et le compte comme en live |
| `jeu_artifacts.py` | les pipelines figés (experts, normalisations) |
| `saint_core.py` | le transformer SAINT |
| `features_ichimoku.py`, `features_range.py`, `features_scalping.py` | colonnes de l'ancien jeu M1, importées par `saint_core.py` |
| `prepare_btc_m5.py` | les 40 colonnes M5 et le cache du jeu |
| `prepare_btc_h1_binance.py` | outils communs (spread Vantage, RSI) et jeu H1 |
| `prepare_multi_h1.py`, `prepare_multi_m5.py` | jeux multi-marchés (abandonnés, gardés pour les tests) |
| `telecharge_*.py`, `build_binance_features.py`, `etude_portage.py` | téléchargement des données |
| `kairos_live.py`, `kairos_m5_live.py`, `flux_live.py` | le live |
| `kairos_gui.py`, `kairos_widgets.py`, `kairos_theme.py` | l'interface |
| `rejoue_test_bloc.py` | rejouer un test |
| `lancer.ps1`, `arreter.ps1`, `veille_fenetre.ps1`, `*.bat` | lancer, arrêter, suivre |

## 7. Où en est le projet (octobre 2026)

| Run | Mise | Résultat sur les blocs de test |
|---|---|---|
| m5_09 | 1 % fixe | +2 775 $ sur 10 blocs, 7 positifs ; les 2 derniers (fin 2024 à 2026) perdent −537 $ et −889 $ |
| m5_12 | risque libre, une seule note | +32 202 $ au total, mais 5 comptes sur 10 vidés, dont les 3 plus récents |
| m5_19 | 8 têtes, une seule note | bloc 1 : +2 719 $, pire baisse −29 % ; en validation, compte vidé dans 19 des 20 dernières epochs |
| m5_20 | deux notes séparées, sauvegarde sous −50 % | en cours |

Ce que les mesures ont établi :
- le signal du BTC M5 vient du flux Binance, de l'écart Coinbase, du
  financement et de l'heure, pas de la forme des bougies ;
- les autres cryptos de Vantage (sauf l'ETH et le ZEC) ont un spread plus
  grand que le mouvement moyen d'une bougie M5 ; l'ETH testé ne rapportait rien ;
- une mise apprise avec la même note que le signal finit par vider le compte :
  d'où les deux notes séparées.

Les anciens moteurs (M1 et H1 sur le BTC et l'or, M15, multi-marchés, l'ancien
live M1) et toutes les études sont dans l'historique git.
