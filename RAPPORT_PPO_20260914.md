Essai court PPO du 14 septembre 2026
===================================

Objectif : verifier la correction du decalage entre la distribution qui tire
les actions et les log-probabilites conservees pour PPO, puis mesurer les
resultats de validation sans toucher au test final.

Resultat termine
----------------

Les huit epoques ont termine. La correction d'echantillonnage est verifiee sur
7 631 decisions : **0 action forcee, 0 action remplacee**, ecart maximal de
log-probabilite **2,3842e-7**, compatible avec l'arrondi float32. L'acteur recoit
un gradient apres le prechauffage (norme avant clipping d'environ 0,38 a 0,68
sur les epoques 4 a 8).

| Validation | PnL total (4 episodes) | PnL moyen par episode | PF | Trades | Taux de gain | DD maximal |
|---|---:|---:|---:|---:|---:|---:|
| Meilleur checkpoint, epoque 7 | -120,26 $ | -30,07 $ | 0,5628 | 27 | 40,74 % | 12,25 % |
| Derniere epoque, 8 | -136,91 $ | -34,23 $ | 0,3323 | 16 | 31,25 % | 12,46 % |

**Aucune des huit validations n'est beneficiaire.** Corriger les probabilites
est necessaire pour un PPO coherent, mais ce court essai ne demontre aucune
rentabilite. Le checkpoint retenu reste celui de l'epoque 7; l'epoque 8 ne
franchit pas le minimum de 20 trades requis pour une nouvelle selection.
Le test reserve n'a pas ete evalue et aucun modele de production n'a ete remplace.

La suite pertinente est un comparatif avant/apres a budgets, graines et
fenetres de validation identiques, puis une evaluation sur une periode
chronologiquement ulterieure a la calibration. Allonger simplement ce run ne
suffirait pas a prouver que le correctif ameliore le rendement.

Les chiffres proviennent de `summary.json`, `training_log_both_wf1.csv` et
`sampling_audit_both_wf1.jsonl` dans le dossier de l'essai.

Correction
----------

Le curriculum remplacait certains tirages par BUY/SELL et le filtre pouvait
remplacer BUY/SELL par HOLD. Le buffer enregistrait pourtant log(pi(action))
de la politique brute. Exemple avec une politique uniforme et 87 % d'ouvertures
forcees : HOLD est reellement tire avec une probabilite de 4,33 %, mais sa
probabilite enregistree vaut 33,33 %. Le ratio PPO ne represente donc pas la
distribution ayant produit les trajectoires.

Le mode corrige tire directement dans la politique : aucune ouverture forcee,
aucun remplacement par HOLD pendant la collecte. La selection des entrees reste
active en validation. Le mode historique est conservable explicitement via
`legacy_off_policy_curriculum=True` pour les comparaisons.

Protocole
---------

- 8 epoques depuis zero, dont 3 de prechauffage du critique.
- 16 episodes d'apprentissage et 4 de validation, 1 500 bougies par episode.
- Meme jeu de features, meme seed 42, SL 2 ATR, TP 2,8 ATR, memes couts.
- Selection de 5 % des occasions de validation, budget partage entre BUY/SELL.
- Normalisation calculee uniquement sur les 1 051 490 lignes d'apprentissage.
- 286 770 lignes de validation; 191 180 lignes de test reservees dans cet essai.
- Poids, metriques, configuration et copies des sources dans
  `experiments/ppo_onpolicy_short_20260914/`.

Le premier pilote a ete arrete apres ses 8 epoques terminees pour liberer le
GPU. Ses checkpoints restent dans `experiments/profitability_20260914_v2/`.
Le second essai utilise un budget plus court : comparer directement leurs
meilleurs PnL ne permet pas d'attribuer une variation de rentabilite au correctif.

Verification
------------

Sept tests de regression couvrent la distribution effective du curriculum,
le remplacement par HOLD, la preservation de la distribution dans le mode
corrige, et les seuils de selection, notamment les probabilites constantes et
leur sauvegarde JSON. Ils passent.

Un audit par epoque enregistre le nombre de decisions, les actions forcees,
les actions remplacees et l'ecart entre log-probabilites de tirage et stockees.
Le journal est `sampling_audit_both_wf1.jsonl` dans le dossier de l'essai.

Limites de la mesure
-------------------

Les episodes de validation sont echantillonnes a chaque epoque et servent aussi
a calibrer les seuils. Ces resultats sont exploratoires, et non une preuve de
rentabilite hors echantillon. Le PnL du journal additionne quatre comptes
simules de 1 000 dollars; le PnL par episode est donc ce total divise par quatre.
Il inclut le marquage des positions ouvertes en fin d'episode, sans liquidation
terminale complete. Les frais et conventions BID/ASK du simulateur restent a
confronter a l'execution reelle avant toute conclusion de production.

La correction de normalisation s'applique au lanceur d'experiences isolees.
Le lanceur historique `training.py` utilise encore ses statistiques globales;
il ne doit pas servir a revendiquer une evaluation sans fuite de normalisation.
