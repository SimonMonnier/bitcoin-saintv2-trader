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
  10 % les plus prometteuses dans ce sens (`porte_rang_expert` = 0,90).

**La mise** — le modèle choisit aussi combien il risque, à chaque trade :
- le **risque** : la part du compte perdue si le stop est touché, de 1 à 100 % ;
- l'**allocation** (10 à 100 %) et la **confiance** (20 à 100 %, selon ses
  erreurs récentes), qui réduisent ce risque ;
- le **lot** : 20 tailles entre 0,01 et le maximum permis par le solde, la
  marge (levier 1:500) et le courtier.

**La sortie en deux temps** — le menu des objectifs est fait de paires
(objectif proche, objectif final) : 1 ou 2 ATR en une fois, ou 0,5 / 1 puis
2 ATR, 0,5 / 1 / 1,5 / 2 puis 4 ATR, 0,5 / 1 / 1,5 / 2 puis 8 ATR (12 paires ;
tout objectif au-delà de 2 ATR est coupé en deux). Le modèle choisit sa
paire. La moitié du trade sort à l'objectif proche ; le stop de l'autre moitié remonte
alors au prix d'entrée, **spread et glissement de sortie compris** (plus
1 bps), et elle court vers l'objectif choisi. Un trade qui touche l'objectif
proche ne peut plus finir perdant, sauf trou de cotation. Le modèle choisit
lui-même son mélange de styles.
Le lot minimum est de 0,02, pour pouvoir couper un trade en deux.

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

**Deux notes séparées** :
- le signal (entrée, objectif, stop, valeur) apprend sur des journées isolées,
  noté sur le gain de chaque trade rapporté à 1 % du compte ;
- les quatre têtes de mise apprennent sur des **mois entiers** (12 mois de
  30 journées d'affilée par epoch, sur un seul compte, positions et solde
  gardés d'un jour à l'autre) ; le signal y joue ses meilleures décisions, seules
  les têtes de mise explorent. Chaque décision de mise est notée sur la
  **croissance du compte depuis cette décision jusqu'à la fin du mois**
  (logarithme du solde, la règle de Kelly) : une ruine note très mal toutes
  les mises qui l'ont construite (run m5_21) ;
- un **mur** empêche les têtes de mise de modifier le tronc, donc le signal.

**La tête d'espérance** (run m5_26) — une neuvième tête prédit, pour chaque
trade possible, son résultat moyen et sa dispersion en multiples du risque.
Elle apprend sur le vrai résultat des trades joués, derrière le mur. Sa
prédiction donne la **mise de base** du trade : la moitié de la mise de Kelly,
moyenne / (moyenne² + variance). Les quatre têtes de mise ajustent autour de
cette base (lot ×0,25 à ×2, allocation et confiance ×0,5 à ×1,5) ; la tête
de risque reste un plafond. Le journal affiche à chaque epoch la corrélation
entre sa prédiction et le résultat réel en validation.

**Les têtes du déjà-vu et météo** (run m5_27) — la tête d'espérance prédisait
le résultat du trade, donc la direction : sa corrélation avec le résultat réel
est restée entre −0,11 et +0,23, elle est coupée. Deux têtes la remplacent, qui
prédisent ce qui se prévoit :
- le **déjà-vu** reconstitue la bougie regardée à partir du seul résumé du
  tronc ; son erreur mesure la nouveauté du marché (un marché jamais vu est mal
  reconstitué) ;
- la **météo** prédit l'amplitude du cours (plus haut moins plus bas) sur les
  96 bougies suivantes, quel que soit le sens : la volatilité vient par vagues,
  et le coût Vantage, fixe en bps, ne pèse presque plus rien les jours agités.

Elles sont derrière le mur et apprennent par leur seule cible : le signal et
sa note ne changent pas. Leurs prédictions deviennent deux entrées de plus des
quatre têtes de mise. Le journal affiche à chaque epoch le profit factor des
trades de validation par tranche de chaque tête (du plus familier au plus
nouveau, du plus calme au plus agité), la mise moyenne par tranche, et la
corrélation entre la météo prévue et le mouvement réel.

**Le budget de baisse** (run m5_28) — les têtes d'espérance, du déjà-vu et
météo ont cherché *quels* trades méritent une plus grosse mise : aucune n'a
trouvé de règle stable d'un bloc à l'autre. Ce qui est stable, c'est l'avantage
moyen du signal. Après chaque validation, ses trades sont rejoués 200 fois, les
journées dans le désordre, à chaque mise possible ; la **mise de base** est la
plus forte pour laquelle la pire baisse ne dépasse **30 %** qu'une fois sur
10. Si le signal perd en moyenne, elle tombe à 0,1 %. Elle sert à l'epoch
suivante, au test (celle du modèle gardé), au modèle final (médiane des dix
folds) et au live. Les quatre têtes de mise choisissent un multiplicateur de
×0,5 à ×1,5 autour d'elle ; série noire, marge et lot minimum restent des
bornes. Mesure sur les trades du m5_27, mise choisie en validation : le test
rejoué donne +7 178 $ (bloc 1, mise 1,15 %), +797 $ (bloc 2, 1,05 %) et
+7 122 $ (bloc 3, 2,25 %) contre +781, +524 et +888 $ joués, pires baisses
−17,6 / −18,9 / −20,4 % ; au bloc 4, où le signal perdait, la mise serait
restée à 0,1 %.

**La tête de conviction** (run m5_29) — un signal qui dit « mise plus » a été
trouvé dans les trades du m5_27 : quand le rang glissant de l'expert dépasse
0,99, le trade paie mieux dans les 3 blocs de validation (PF 1,06 / 1,20 /
1,14 contre 1,02 / 1,06 / 0,97) et dans les 3 tests (1,32 / 1,60 / 1,58
contre 1,28 / 1,05 / 1,37). Une cinquième tête de mise, petite, ne lit que
cette conviction et le sens, et calcule un multiplicateur continu (de ×0,1 à
×25, un garde-fou que la série noire coupe bien avant ; au run m5_29 c'était
un menu de ×0,5 à ×4). Elle apprend sur chaque trade joué la croissance
exacte qu'aurait donnée n'importe quel multiplicateur, en demi-Kelly, sans
tirage ; un prix du risque ajusté
en continu garde le multiplicateur moyen à 1 : pour quadrupler un trade, elle
doit miser moins ailleurs. Dans le modèle final, elle ne réapprend pas :
l'expert final est appris sur tout l'historique et ses rangs y « voient » les
résultats ; elle copie donc la médiane des dix têtes des folds, apprises sur
des rangs honnêtes, puis ne bouge plus (run m5_31). En live, l'expert est
rechargé avec le pipeline du modèle final et son rang recalculé à chaque
bougie. La mise de base est fixe (1 %, le R du jeu) ; le
budget de baisse (m5_28) est coupé. La veille affiche, par tranche de rang,
le multiplicateur choisi et le profit factor en validation.

**La série noire** — avant chaque trade, le risque ne dépasse jamais ce que
la pire série de pertes du modèle laisse survivre : si les N prochains trades
perdaient tous, le compte ne perdrait pas plus de 50 %. Risque maximal par
trade = (1 − 0,5^(1/N)) / k, où N est la plus longue série de pertes
d'affilée et k la perte réelle la plus forte rapportée à la perte prévue au
stop (gaps, spread, glissements). N = 8 et k = 1 donnent 8,3 %. N et k sont
remesurés à chaque validation, jamais sur le test, et la même règle s'applique
en live.

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
| m5_20 | deux notes séparées, sauvegarde sous −50 % | bloc 1 : sain jusqu'à l'epoch 11 (+1,41 $/jour, −31 %), puis comptes vidés après l'epoch 15 |
| m5_21 | note du mois et série noire | win rate tombé à 20 % (objectif à 6-8 ATR, stop à 1 ATR), sans compte vidé |
| m5_22 | sortie en deux temps, lot minimum 0,02 | bloc 1, epochs 5-9 : win rate 64-68 %, profit factor 1,02-1,07, mise réelle 0,5 % du compte |
| m5_23 | porte 0,90, objectif proche choisi par le modèle | bloc 1, frais pleins : +0,64 à +1,12 $/jour, PF 1,08-1,16, win rate 73-76 %, pire baisse −6 à −14 % |
| m5_24 | 13 paires d'objectifs, mois joués avec le signal réel | arrêté : le modèle s'est jeté sur « 4 ATR en une fois », win rate de 69 % à 27 % en 3 epochs |
| m5_25 | 12 paires (sans « 4 ATR en une fois ») | bloc 1, epochs 2-9 : +0,32 à +2,50 $/jour, win rate 63-73 %, pire baisse −6 à −20 %, mise réelle 0,3 à 1 % |
| m5_26 | tête d'espérance (mise de base demi-Kelly prédite) | bloc 1, epochs 1-17 : meilleur +2,53 $/jour (PF 1,25, epoch 16) ; corrélation prédiction / résultat entre −0,11 et +0,23 : arrêté |
| m5_27 | têtes du déjà-vu et météo, entrées des têtes de mise | tests blocs 1-3 : +2,35 / +1,59 / +2,68 $/jour (PF 1,28 / 1,15 / 1,33) ; aucune pente stable par tranche ; arrêté au bloc 4 (validation sans aucun trade) |
| m5_28 | budget de baisse : mise de base mesurée en validation (pire baisse 30 % une fois sur 10) | bloc 1, epochs 11-21 : +0,36 $/jour moyen en validation contre +0,12 au m5_27, pires baisses −8 à −37 % ; arrêté |
| m5_29 | tête de conviction : multiplicateur ×0,5 à ×4 appris sur le rang de l'expert, budget moyen constant | arrêté au bloc 1 (erreur pendant les mois), remplacé par la version continue |
| m5_30 | conviction continue : multiplicateur libre ×0,1 à ×25, demi-Kelly, budget moyen constant | arrêté aux premières epochs : le modèle final réapprenait la conviction sur l'expert final |
| m5_31 | conviction continue, celle du modèle final copiée de la médiane des 10 folds | en cours |

Ce que les mesures ont établi :
- le signal du BTC M5 vient du flux Binance, de l'écart Coinbase, du
  financement et de l'heure, pas de la forme des bougies ;
- les autres cryptos de Vantage (sauf l'ETH et le ZEC) ont un spread plus
  grand que le mouvement moyen d'une bougie M5 ; l'ETH testé ne rapportait rien ;
- une mise apprise avec la même note que le signal finit par vider le compte ;
  notée trade par trade sur des journées isolées, aussi : d'où la note du mois
  et la série noire.

Les anciens moteurs (M1 et H1 sur le BTC et l'or, M15, multi-marchés, l'ancien
live M1) et toutes les études sont dans l'historique git.
