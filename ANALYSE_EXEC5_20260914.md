# Analyse du run Claude Code exec5 — 14 septembre 2026

Instantané : epoch 9 complet, journal horodaté 22 h 23. Processus 7336 lancé à 20 h 28, toujours actif lors de l'inspection. Les SHA-256 des quatre sources du manifeste correspondent aux fichiers inspectés : training.py, saint_core.py, execution_quotes.py et economic_learning.py. Cette analyse ne modifie ni le run ni son code.

## Résultats observés

| Mesure validation | Epoch 1 | Epoch 7 | Epoch 9 |
|---|---:|---:|---:|
| PnL total | −19 243,25 $ | −16 710,58 $ | −15 233,11 $ |
| Trades | 2 664 | 5 877 | 5 263 |
| Winrate | 16,85 % | 32,06 % | 31,77 % |
| Profit factor | 0,2398 | 0,6160 | 0,6151 |

À l'epoch 9 : 48 épisodes sur 48 sont perdants, moyenne −317,36 $ pour 1 000 $ initiaux par épisode. Douze perdent au moins 390 $. Le gain moyen d'un trade gagnant est 14,56 $, contre une perte moyenne de 11,02 $ : il faudrait environ 43,1 % de gagnants avec ces montants, contre 31,8 % observés. La médiane de détention est 6 bougies ; la plus longue détention observée en validation est 273 bougies. 67,9 % des trades sortent au stop.

L'amélioration initiale est réelle dans ces métriques, mais le profit factor plafonne autour de 0,61 sur les epochs 7–9. La diminution du PnL total négatif entre 7 et 9 accompagne aussi une diminution du nombre de trades ; le PnL moyen par trade ne s'améliore pas (environ −2,84 $ puis −2,89 $).

## Défauts prioritaires du filtre à rang

### P1 — Historique partagé entre épisodes indépendants

`training.py:2773` crée deux objets `SeuilRang`, un par côté, pour l'ensemble des 48 environnements. La boucle `training.py:2827` les consulte puis les modifie successivement pour chaque environnement.

Ces environnements correspondent à des dates différentes. Une probabilité issue d'un épisode peut donc modifier le seuil d'un autre, éventuellement situé plus tôt dans le temps. La décision dépend de l'ordre et du nombre d'environnements dans le batch. Ce n'est pas le comportement d'un bot suivant son propre historique chronologique.

Reproduction isolée avec la classe réelle : une probabilité 0,2 est acceptée après un historique de 0,1 ; la même probabilité est refusée après l'insertion dans l'historique partagé de 500 valeurs 0,9 provenant d'un autre épisode.

Correction : deux filtres indépendants par environnement, ou une validation chronologique unique. Ajouter un test d'invariance à la permutation des environnements.

### P1 — Probabilités constantes : 100 % d'acceptation

`saint_core.py:906` utilise `conviction >= quantile`. Avec un historique constant de 500 valeurs à 1/3, une fraction cible de 2,5 % et 1 000 nouvelles valeurs identiques, les **1 000 décisions sont acceptées**. Ce défaut est particulièrement pertinent pendant le warmup, où la politique est presque uniforme. La reproduction exacte ne prouve pas que toutes les probabilités du run sont identiques ; elle démontre une défaillance du filtre dans ce cas.

Correction : refuser les égalités lorsque le quantile est constitué d'ex æquo, selon une convention cohérente avec `selective_threshold`, et tester les distributions constantes et discrètes.

### P1 — Validation et test final appliquent des politiques différentes

La validation utilise maintenant les rangs glissants. Le test final, à `training.py:3201`, compare toujours les probabilités à `calib_thr_val`, fixe. Les fichiers `_calib.json` sauvegardent ces seuils fixes et ne décrivent pas complètement la nouvelle politique à rang. `kairos_live.py` et les backtests inspectés appellent encore `decide_avec_barres`, qui utilise des seuils fixes.

Correction : implémenter une politique de décision commune, avec mode explicite, paramètres de fenêtre, amorçage et état par flux ; utiliser ce même composant en validation, test et live. Le test final de ce run ne doit pas être interprété comme l'évaluation de la politique qui a sélectionné son checkpoint.

## Autres observations

- **Commission configurée à zéro.** Le manifeste confirme `fee_rate=0.0`, contrairement aux anciens essais. Spread et slippage sont conservés. Les pertes présentes ne peuvent donc plus être attribuées à l'ancienne commission de 0,0004. Ce choix doit correspondre au compte réel ; il ne prouve pas un avantage du signal.
- **Durée maximale supprimée.** `max_holding_bars=0` désactive la sortie au temps. Ce changement n'est pas une erreur en soi, mais modifie la stratégie et rend les anciens labels plafonnés à 240 bougies non interchangeables avec ce run.
- **FP32 stable jusqu'ici.** Aucun avertissement de gradient non fini dans le journal inspecté. Le contrôle des log-probabilités reste autour de 10⁻⁷, sans actions forcées ou remappées. Cela ne valide pas à lui seul toute la mise à jour PPO.
- **Bridage thermique persistant.** Derniers epochs : GPU 90–91 °C, fréquence 210/2 100 MHz. À l'epoch 9, les phases affichées totalisent environ 869 secondes, soit 14,5 minutes. L'architecture simplifiée ne permet pas d'évaluer proprement le gain de vitesse sous ce bridage.
- **Épisodes qui se recouvrent.** Les 5 263 trades de validation correspondent à 4 146 couples entrée/côté distincts. Ils ne constituent pas 5 263 observations historiques indépendantes.
- **Architecture et features.** Deux blocs allégés attention temporelle → attention colonnes → FFN, fenêtre de 25 bougies ; remplacement de `ls_ratio_top` par `taker_1m_ma5`. Ces changements exigent une ablation contrôlée pour attribuer une amélioration au modèle. Les logs seuls ne l'établissent pas.

## Conclusion

Exec5 apprend quelque chose par rapport à son départ, mais reste déficitaire et sa sélection de checkpoints est affectée par le filtre de validation. Priorité : corriger les trois défauts de rang avant d'interpréter la poursuite des epochs comme une recherche fiable de rentabilité. Ne pas confondre augmentation du nombre de trades, amélioration du signal et avantage économique.

Reproductions et mesures : `experiments/audit_exec5.py` et `experiments/exec5-audit-20260914.json`.
