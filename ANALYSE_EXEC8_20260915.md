# Run exec8 — analyse du 15 septembre 2026

Lecture seule. Processus training.py 1892 lancé à 01:38:49. Dernier epoch complet examiné : 26, journal métrique à 11:53:33 ; collecte de l'epoch 27 enregistrée à 12:15:52. Le processus veille_epochs.py 7428 est un lecteur de journal distinct.

## Conclusion

La rentabilité ne s'améliore pas hors entraînement. Les 26 validations sont négatives. Le meilleur profit factor appartient à l'epoch 9 (0,5864), contre 0,4908 à l'epoch 26. La moyenne des cinq derniers PF vaut 0,515. La baisse de la loss du critic n'est pas une preuve d'avantage de trading.

## Epoch 26

- Validation : −18 254,92 $ au total, soit −380,31 $ par épisode de 1 000 $.
- 48 épisodes sur 48 perdants ; 28 perdent au moins 390 $.
- 4 244 trades, dont 21,32 % gagnants ; gain moyen 19,44 $, perte moyenne 10,7362 $.
- Seuil d'équilibre calculé avec ces montants : 35,58 % de gagnants, écart −14,26 points. Employer les moyennes **validation** du CSV : les champs AvgW/AvgL de la ligne META proviennent du train et donnent un autre seuil.
- 78,18 % des trades sortent au stop. Durée médiane : 5 bougies ; maximum : 607 bougies. La sortie au temps est désactivée (`max_holding_bars=0`).
- 3 149 couples entrée/côté distincts pour 4 244 trades : les épisodes se recouvrent ; le nombre de trades ne représente pas autant d'observations indépendantes.
- Train : −44 267,14 $, PF 0,6006. Il s'améliore davantage que la validation, qui demeure proche de sa limite de pertes.

## Ce qui fonctionne mieux

Le training charge une politique de décision par épisode et utilise le même composant en test. Le checkpoint contient un mode rolling_rank, une fenêtre de 500, deux fractions de 2,5 % et la gestion conservatrice des égalités. Les défauts structurels de partage du filtre d'exec5 ne sont plus présents dans le training inspecté.

Aucun avertissement de gradient ou de loss non fini relevé. Les diagnostics du tirage ne montrent ni actions forcées/remappées ni écart de log-probabilité significatif (~10⁻⁷). Cela ne démontre pas à lui seul l'exactitude de chaque aspect du PPO.

## Problèmes actuels

1. **Écart entre apprentissage et sélection.** L'entropie flat descend à 0,187/1,099. Les 13 348 ouvertures pour 238 150 décisions flat représentent environ 5,6 % d'entrées, donc 94,4 % d'attentes aux états de décision. Le réseau apprend surtout à s'abstenir ; le filtre de validation choisit encore les meilleurs scores BUY/SELL relatifs. Un haut rang ne prouve pas une espérance nette positive. Il faut comparer le rang à une règle d'entrée fondée sur un avantage net mesuré, avec calibration séparée.
2. **Collecte de plus en plus coûteuse.** 16 000 décisions sur 238 150 sont utilisées pour l'update, soit 6,72 %. Ce sous-échantillonnage uniforme n'est pas automatiquement biaisé, mais la collecte de trajectoires supplémentaires a un rendement décroissant. À l'epoch 27 : 258 223 décisions collectées. Réduire la collecte ou ajuster son budget sans tronquer incorrectement les retours.
3. **GPU fortement bridé.** À l'epoch 26 : 93 °C, 210/2 100 MHz. Phases affichées : collecte 695 s, PPO 586 s, calibration 166 s, validation 440 s ; total 31,45 minutes. Aucun changement de réglage matériel effectué.
4. **Complexité non validée.** Le manifeste indique 3 blocs et 256 références mémorisées, contre une architecture plus simple auparavant. Aucun résultat de ce run ne démontre que la mémoire améliore la généralisation. Comparaison nécessaire à protocole identique : sans mémoire, puis mémoire, sans changer simultanément SL/TP ou features.
5. **Provenance du modèle incomplète.** Les empreintes training.py, execution_quotes.py et economic_learning.py concordent avec le manifeste. Celle de saint_core.py ne concorde pas ; ce fichier a été modifié à 02:11:58, après le lancement. Une modification sur disque ne remplace pas les classes Python déjà chargées dans le processus. Les constats sur la structure du run s'appuient sur le manifeste ; les détails de l'implémentation actuelle de saint_core.py ne peuvent pas tous être attribués au run.

## Comparaisons à éviter

Ce run utilise SL=2 ATR, TP=4 ATR, commission configurée à zéro, spread/slippage actifs et bruit de mèches désactivé. Comparer directement son winrate à exec5 (TP=2,8 ATR) est trompeur : le seuil de rentabilité change. La baseline « hasard 26,7 % » du veilleur ne suffit pas non plus à une comparaison rigoureuse sans mêmes fenêtres, règles d'entrée, exécution et gestion du compte.

Le veilleur calcule son point mort à partir des moyennes TRAIN de la ligne META alors qu'il commente le winrate de validation. Pour ce diagnostic, utiliser `val_avgW` et `val_avgL` du CSV, comme ci-dessus.

Priorité : arrêter de juger les changements par la baisse de loss, figer le protocole et comparer une version simple avec/sans mémoire sur des fenêtres non chevauchantes. La poursuite inchangée jusqu'à 240 epochs n'est pas justifiée par les résultats observés. Le run et le veilleur ont été laissés actifs.

Mesures reproductibles : `experiments/audit_exec8.py`, `experiments/exec8-audit-20260915.json`.
