# Audit du run BTCUSD — 14 septembre 2026

**Conclusion : plusieurs défauts vérifiés faussent la récompense ou la simulation. Ils peuvent contribuer à l'apprentissage de HOLD et à la mauvaise rentabilité, mais leur contribution exacte exige un comparatif après correction.** Le contrôle des actions PPO est désormais cohérent. Ajouter des features ou des couches avant de corriger les défauts ci-dessous rendrait les comparaisons difficiles à interpréter.

Audit effectué sur le code présent pendant le run PID 1552, lancé à 04:37. Aucun arrêt, redémarrage ou changement du code d'entraînement n'a été effectué pendant cet audit. Les reproductions utilisent le CPU et des marchés synthétiques, sans connexion de trading.

## Résultats observés

**Dernier contrôle en fin d'audit : époque 39 terminée**, PF de validation **0,5753**, PnL **−2 514,95 $** sur 16 épisodes (−157,18 $ par épisode), drawdown **40,45 %**, 731 trades et 36,66 % de gains. Toujours aucune validation bénéficiaire. Le PID 1552 reste actif. L'instantané chiffré détaillé ci-dessous a été figé à E38 pour conserver une base reproductible.

Instantané du 14/09 à 09:15:42, 38 époques complètes du fold 1 :

| Mesure | Résultat |
|---|---:|
| Validations bénéficiaires | 0 / 38 |
| Dernière validation — PF | 0,4023 |
| Dernière validation — PnL | −2 807,44 $, total de 16 épisodes |
| PnL moyen par épisode de 1 000 $ | −175,46 $ |
| Dernière validation — drawdown maximal | 40,36 % |
| Trades / taux de gain | 522 / 29,50 % |
| Trades terminés au stop | 70,50 % |
| Détention médiane / moyenne | 8 / 13,46 bougies |
| Meilleur PnL de validation | Époque 35 : −2 030,50 $, PF 0,5006 |
| PF médian des 10 dernières validations | 0,49385 |
| Entropie sur les états flat à E38 | 0,046 / maximum ln(3) ≈ 1,099 |
| Actions HOLD du rollout E38 | 98,86 %, positions ouvertes incluses |

LONG et SHORT perdent tous deux à E38 : −1 483,66 $ et −1 328,15 $. Le gain moyen est 12,29 $, la perte moyenne 12,78 $ ; le taux de gain d'équilibre empirique est proche de 51 %, contre 29,5 % observés. Ces statistiques n'établissent pas une cause unique.

La ligne E38 collecte 476 956 décisions puis n'en utilise que 16 000 pour PPO, soit 3,35 %. Le plafond économise les mises à jour, mais pas la collecte : 555 s de collecte sur environ 810 s par époque. Le journal rapporte 93 °C et 210 MHz ; c'est un problème de coût de calcul à examiner, pas une preuve de causalité des pertes.

## 1. P1 — Convention BID/ASK incohérente : stop LONG prématuré

**Fichiers :** `training.py:1046`, `training.py:1256`, `training.py:1259`, `training.py:1294`.

`_apply_micro` traite le prix comme un mid et ajoute un demi-spread, tandis que la section des déclenchements déclare que les bougies représentent le BID. Le stop LONG est testé avec `low_bar <= sl_price + spread`, au lieu de tester le niveau du stop contre le BID. La cible LONG est également décalée vers le haut. La fermeture au niveau SL/TP subit ensuite un demi-spread supplémentaire.

**Reproduction exécutée :** BID d'entrée 10 000, spread 10, sans frais, slippage ni bruit. L'entrée calculée est 10 005 (l'ASK correspondant serait 10 010). Le stop enregistré vaut 9 805. Avec un plus bas BID suivant à 9 810, donc encore au-dessus du stop, l'environnement ferme pourtant au SL.

**Effet :** stops anticipés et cibles LONG plus difficiles à atteindre ; conventions d'entrée et de sortie incompatibles avec la même série BID. L'effet total de tous les écarts peut mélanger optimisme et pessimisme : il faut corriger le moteur dans son ensemble.

**Correction proposée :** moteur d'exécution partagé avec prix BID et ASK explicites. BUY entre à l'ASK et sort au BID ; SELL entre au BID et sort à l'ASK. Les niveaux SL/TP sont testés dans la quote de fermeture. Un niveau déjà exprimé en prix exécutable ne repaie pas un spread. Gérer séparément frais, glissement et gaps, puis tester les mêmes scénarios dans training et backtest.

Référence : [principes d'exécution MetaTrader 5](https://www.metatrader5.com/en/terminal/help/trading/general_concept).

## 2. P1 — Pénalité répétée dans la récompense à prix immobile

**Fichiers :** `training.py:1159`, `training.py:1334`, `training.py:1370`.

L'equity précédente est reconstruite avec l'extrême **sans bruit** de la bougie précédente. L'equity courante utilise l'extrême **avec bruit** de la bougie courante. Le bruit payé en récompense au pas précédent disparaît de la référence du pas suivant : les variations ne décrivent plus la différence de deux évaluations cohérentes du portefeuille.

**Reproduction exécutée :** prix constant à 66 000, ATR 34, frais et spread nuls, bruit configuré à 1,2 bps et tirage déterministe au milieu de l'intervalle. Pendant 60 bougies :

- PnL réalisé : 0 ; PnL de liquidation au marché sans coûts : 0.
- Récompense de chaque pas : −0,00699068.
- Récompense cumulée : **−0,41944069**, ou **−0,19555102** actualisée à gamma 0,97.
- Avec le bruit désactivé : récompense cumulée 0.

**Effet :** pénalisation artificielle et répétée de la détention. Elle est compatible avec une préférence excessive pour HOLD, mais le test ne chiffre pas sa part dans les pertes réelles du run.

**Correction proposée :** mémoriser l'equity précédente effectivement utilisée ; calculer la récompense à partir d'une valorisation cohérente au prix de clôture ou de liquidation. Séparer le scénario de stress sur les extrêmes de la récompense économique. Tester qu'un portefeuille inchangé et sans coûts a une récompense nulle, et que le cumul des variations non clippées réconcilie les equities de début et de fin.

## 3. P1 — Fin de fenêtre : bootstrap calculé puis annulé

**Fichiers :** `training.py:1417`, `training.py:1998`, `training.py:2037`, `training.py:1520`.

Toutes les fins d'épisode sont versées au buffer avec `done=True`, y compris la fin arbitraire d'une fenêtre. Le code calcule ensuite `last_value`, mais le masque terminal du GAE annule sa contribution.

**Reproduction exécutée :** récompense 0, valeur courante 0, dernière valeur 10, gamma 0,97, durée 1. Le code avec terminal renvoie une cible 0 ; une transition tronquée d'une tâche continue donne 9,7. Ce n'est pas un défaut de la formule GAE isolée : c'est le flag envoyé par le caller qui décide du traitement.

Les positions ouvertes à la limite de fenêtre ne sont par ailleurs pas liquidées avec leurs frais ; elles sont marquées au dernier close dans le PnL de reporting et omises du tableau des trades clos.

**Correction proposée :** choisir une sémantique cohérente. Soit une tâche continue avec distinction `terminated` / `truncated` et un bootstrap valide, soit une fin financière réelle avec liquidation exécutable et résultat terminal inscrit au reward et au journal. Pour le semi-MDP, traiter explicitement la décision encore en cours plutôt que d'interroger sans validation un critique essentiellement entraîné sur les états flat.

Référence : [Farama — terminated et truncated](https://farama.org/Gymnasium-Terminated-Truncated-Step-API).

## 4. P1 — Normalisation du fold ajustée sur tout l'historique

**Fichiers :** `training.py:3036`, `training.py:3038`.

**Vérification numérique :** sur 1 911 240 lignes après filtrage, les moyennes et écarts-types sauvegardés sont exactement ceux de la totalité des données : erreur maximale 0 pour les deux. Avec seulement les 55 % de train, les écarts maximaux sont respectivement 0,3950119 et 0,1005039.

**Effet :** fuite d'information de distribution depuis validation/test. Elle invalide une prétention de test indépendant ; elle tend plutôt à rendre une mesure optimiste et n'explique pas, à elle seule, les pertes.

**Correction proposée :** calculer les stats dans chaque fold sur son train uniquement, sauvegarder leur version avec le modèle et imposer leur chargement à l'inférence. Le lanceur expérimental du second essai le faisait déjà ; ce run utilise le lanceur historique.

## Choix de conception qui peuvent limiter les résultats

Ces points sont des hypothèses à comparer, pas des bugs démontrés du run :

1. **Validation qui contourne l'abstention.** La sélection par quantiles BUY/SELL ignore que HOLD peut être l'action la plus probable. Une masse de 5 % ou 10 % de candidats n'est pas une preuve d'espérance positive. Comparer la règle actuelle à une politique d'abstention explicite, puis à deux têtes d'espérance nette LONG/SHORT. Tout filtre utilisé pendant la collecte doit être intégré à sa distribution de probabilité ; ne pas réintroduire un remplacement après tirage.
2. **Actualisation courte.** `gamma=0.97` donne 0,1608 à 60 minutes, 0,02586 à 120 et 0,000669 à 240. Une récompense tardive devient presque invisible. La détention médiane actuelle n'est toutefois que 8 minutes : le problème affecte surtout la queue des durées. Comparer gamma seulement après correction de la récompense, à horizon de trade fixé.
3. **Préentraînement sur une autre définition de la réussite.** Les paramètres BTCUSD sont maintenant alignés, mais `barrieres` supprime les exemples dont au moins un côté n'est pas résolu et n'applique pas toute l'économie du PPO. Préférer des cibles de résultat net LONG/SHORT, y compris fermeture au temps, produites par le moteur partagé. Ce script n'est pas appelé par le run courant : ce défaut ne cause pas directement son apprentissage actuel.
4. **Features économiques manquantes.** `vol_rank` est déjà sélectionné. `mom_5` reste binaire ; le code rapporte qu'une variante continue simple avait dégradé une mesure antérieure. Conserver ce baseline et tester séparément des rendements multi-horizons normalisés, les coûts rapportés au stop/ATR et les variations de flux Binance. N'ajouter aucun indicateur sans ablation contrôlée.
5. **Budget de calcul mal utilisé.** Essayer `need_weights=False` dans les attentions dont les poids sont ignorés, puis profiler. Comparer un MLP ou TCN compact au SAINT à budget égal ; éviter d'agrandir le réseau sans avantage mesuré. L'échantillonnage uniforme de 16 000 décisions n'est pas, en lui-même, une erreur de probabilité.

## Éléments déjà corrigés dans cette version

- Collecte PPO sans forçage ni remplacement, audit d'erreur autour de 3e-7.
- `vol_rank` continu dans les dix features de marché.
- Pooling SAINT concaténant résumé global et dernier instant, avec couche d'entrée adaptée.
- Préentraînement configuré pour BTCUSD, lookback 54, SL 2 ATR et R:R 1,4.

La reproduction de l'ancienne double moyenne dans le script d'audit est un contrôle historique de l'identité mathématique ; **ce n'est pas un bug du pooling actuel**.

## Ordre de travail recommandé

1. Corriger et réconcilier le moteur de prix et la récompense, dans une nouvelle expérience versionnée.
2. Régler la fin de fenêtre et la normalisation ; figer des fenêtres identiques de comparaison.
3. Refaire un court A/B à paramètres identiques, en mesurant aussi fréquence d'entrée et entropie flat. Ne pas chercher à augmenter le trading tant que chaque entrée est désavantageuse.
4. Seulement ensuite : cibles supervisées de PnL net, features de coût, variantes compactes et tête d'abstention économique.

**Je recommande de ne pas investir un long entraînement supplémentaire dans cette configuration avant ces corrections.** Le run existant a été laissé actif pendant l'analyse ; les changements ci-dessus sont des propositions et n'ont pas été appliqués au code Python de production.

## Preuves et reproductibilité

- `experiments/audit-code-20260914/reproduce.py` et `reproductions.json` : scénarios déterministes, CPU.
- `experiments/audit-code-20260914/analyze_run.py`, `run_analysis.json`, `metrics_snapshot.csv` : instantané des résultats et empreintes SHA-256 des sources.
- `roadmap-rentabilite.html` : traduction des constats en modifications concrètes, par fichier et par expérience.
