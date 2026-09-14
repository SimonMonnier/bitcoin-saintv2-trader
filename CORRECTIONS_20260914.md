# Corrections du moteur et de la normalisation — 14 septembre 2026

> **Journal daté — deux points ci-dessous ont été invalidés depuis.**
> Les frais ne passent pas à `0.0004` : ce courtier ne facture **aucune
> commission** sur BTCUSD CFD, `fee_rate = 0.0`. Le lookback de l'export ONNX ne
> passe pas à 54 : il doit valoir **25**, comme le training. Voir
> `JOURNAL_MESURES.md` pour la suite.

Les quatre défauts reproduits dans `AUDIT_ENTRAINEMENT_20260914.md` ont été corrigés. L'ancien entraînement a été arrêté et ses sources, poids, statistiques et journaux copiés dans `experiments/before-fixes-20260914/`. Les deux checkpoints archivés sont lisibles par PyTorch. Aucun modèle n'a été activé en live.

## Changements

1. **Prix BID/ASK et stops.** `execution_quotes.py` définit la conversion commune : BUY à l'ASK, SELL au BID. Le training et les trois backtests utilisent cette convention. Un LONG ne déclenche plus son stop tant que le BID ne l'a pas atteint ; un SHORT teste l'ASK. Le training ne déduit plus un second spread d'un SL/TP déjà exprimé en prix exécutable. Dans les backtests, le spread échantillonné à l'entrée est conservé pour les sorties SHORT, au lieu d'en tirer un autre indépendamment.
2. **Récompense économique cohérente.** Les deux bornes du rendement utilisent le close en prix de liquidation et la même provision de frais de clôture. Les mèches bruitées restent utilisées pour les triggers, mais ne fabriquent plus une nouvelle perte latente chaque minute. Le bonus symétrique de résultat réalisé en unités de risque est conservé : il s'agit toujours d'un reward façonné, pas d'un PnL brut.
3. **Fin d'épisode complète.** Toute position restante est liquidée avec spread, frais et slippage au terme de la fenêtre ou sur un garde-fou. La clôture entre dans le capital, les trades et la dernière récompense. Le choix est celui d'épisodes finis : `terminated=True`, sans bootstrap de valeur terminale. Le calcul inutile d'une valeur finale ensuite masquée a été supprimé.
4. **Normalisation sans données futures.** Le split simple utilise seulement ses premiers 70 % pour calculer les statistiques. Chaque fold walk-forward utilise seulement sa propre tranche de train. Chaque checkpoint possède son fichier `*_norm.npz` ; live, backtests et export ONNX chargent ce fichier. En mode multi-agent, chacun reçoit une observation normalisée avec son propre scaler. Le mode à observation commune refuse des scalers différents.

Autres corrections liées : dans le training, un gap d'ouverture au-delà du stop est exécuté au prix d'ouverture défavorable ; le TP ne reçoit plus d'amélioration favorable systématique ; le break-even/trailing utilise la bougie précédente close. Les frais par défaut des backtests passent à `0.0004`, comme dans le training, sans réduire les coûts du training. Le lookback de l'export ONNX passe de 25 à 54.

## Reprise et compatibilité

- Le lancement principal utilise le nouveau préfixe `saintv2_loup_duel_exec2`, sans écraser les anciens poids.
- Les folds repartent de zéro. Le bootstrap inter-fold est refusé tant qu'aucune adaptation explicite entre scalers n'est implémentée.
- Une continuation de training exige le scaler associé au checkpoint et son égalité avec le scaler du train courant.
- Les anciens modèles restent lisibles en inférence avec un avertissement et leur ancien fichier de statistiques. Pour transférer un nouveau modèle, conserver ensemble `.pth`, `*_norm.npz` et `*_calib.json` lorsqu'il existe. Les chemins de modèles live historiques n'ont pas été remplacés.

## Vérification

**18 tests réussis** dans `test_training_economics.py` et `test_training_selection.py`. Ils couvrent notamment les stops LONG/SHORT, l'absence de pertes fictives à prix constant, les liquidations terminales, les gaps, les sorties au temps, la normalisation par train/fold, le choix du scaler et la convention d'entrée des trois backtests. Compilation des modules modifiés et `git diff --check` réussis.

Essai isolé : `experiments/corrected-smoke-20260914/`, 2 epochs, 4 épisodes de train et 2 épisodes de validation de 600 bougies, sans warmup du critique pour exercer immédiatement les mises à jour PPO. Le test final reste réservé.

| Mesure | Résultat |
|---|---:|
| Actions forcées / remappées | 0 / 0 |
| Erreur maximale de log-probabilité au contrôle du tirage | 2,01 × 10⁻⁷ |
| PnL validation epoch 1, somme de 2 épisodes | −65,18 $ |
| PnL validation epoch 2, somme de 2 épisodes | −94,41 $ |
| Profit factor validation epoch 2 | 0,6123 |
| Trades validation epoch 2 | 28 |
| Écart maximal PnL journal / somme CSV des trades | 0,0005 $, dû aux arrondis |

Les checkpoints best, bestprofit et last ont été rechargés avec leur scaler ; leur forward produit des valeurs finies. Détails dans `verification.json` et `summary.json` de l'essai.

## Limites restantes

Cet essai valide le fonctionnement et reste perdant ; il ne mesure pas un gain de rentabilité par rapport à l'ancien run, dont les fenêtres et la durée diffèrent. L'entraînement long n'a pas été relancé.

La calibration et la sélection utilisent encore les épisodes de validation du pilote. Une évaluation sur une période distincte, des ablations de features et plusieurs seeds restent nécessaires. Les frais réels du compte et la distribution du spread doivent être mesurés. Les backtests historiques conservent d'autres différences de gestion de positions et de stress : les corrections de quotes ne constituent pas une validation complète contre MT5 ticks réels. Aucun export ONNX effectif ni essai live n'a été lancé.
