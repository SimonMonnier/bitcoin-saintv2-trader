# Passation — KAIROS, état au 16 septembre 2026

Document de transfert. Il s'adresse à quelqu'un qui reprend le projet sans rien
en savoir. Tout ce qui suit est **mesuré**, sauf mention explicite du contraire.

---

## 1. Le projet en trois phrases

Agent d'apprentissage par renforcement (PPO) sur BTCUSD, entraîné en
walk-forward sur neuf ans de bougies M5 (Binance spot), exécuté sur MetaTrader 5
chez Vantage. Les positions sortent **uniquement sur barrières** — jamais au
temps, parce que le live n'a aucun chemin de fermeture au marché. Depuis le
16 septembre la sortie est un **stop 12×ATR avec stop suiveur à 1,5 R et aucun
objectif** ; l'objectif à 2 R subsiste comme *cible d'apprentissage* de la tête
auxiliaire, pas comme ordre. **Aucun modèle n'est déployable à ce jour** : rien
n'a jamais produit un avantage hors échantillon statistiquement significatif.

Ce qui a changé ce jour-là, et qui est le fait le plus utile du document : à
cette largeur de stop **la géométrie gagne sans modèle** (+0.11 R symétrique,
2.4 σ, hors test), et **le modèle classe** (ρ de rang +0.070, 2.9 σ sur 21 985
décisions de validation). Les deux moitiés du problème sont désormais mesurées
positives ; ce qui manque est de les faire tenir ensemble hors échantillon.

---

## 2. La contrainte qui commande tout

**Le nombre d'occasions INDÉPENDANTES.** Une occasion n'est indépendante de la
suivante que si leurs fenêtres de résultat ne se recouvrent pas. Leur nombre
vaut donc *durée d'historique ÷ durée d'un trade* — **jamais** le nombre de
barres. Il en faut ~4 900 pour qu'un avantage de 0.02 R soit lisible.

```
config          friction   horizon   occasions 9 ans   avantage requis
H1, SL 2xATR     0.047 R    13.0 h            2 314        +0.0233 R
M5, SL 4xATR     0.099 R     3.7 h           15 089        +0.0892 R
M5, SL 8xATR     0.049 R    12.2 h            4 516        +0.0425 R
M5, SL 12xATR    0.021 R    34.1 h            1 709        +0.0210 R   <- retenu
```

La ligne 12×ATR suppose la sortie au stop suiveur, pas des barrières fixes :
l'horizon triple et les occasions chutent, mais la friction, qui est un montant
fixe en dollars, tombe de moitié. Le produit — la détectabilité — y gagne.

Le M5 à 8×ATR domine le H1 sur tous les axes : même friction, même horizon —
donc le **même problème de prédiction**, celui sur lequel l'avantage de +0.09 à
+0.13 R avait été mesuré — pour deux fois plus d'occasions.

---

## 3. Les règles de méthode, non négociables

Elles ont toutes été payées par une erreur réelle. Les enfreindre a coûté des
journées entières.

1. **La fenêtre de test ne sert qu'une fois.** Tout choix fait dessus la
   contamine définitivement. Mesurer sur la validation, décider, puis tirer une
   seule fois.

2. **Corriger de la phase.** Des entrées non chevauchantes ne suffisent pas :
   il y a autant de grilles possibles que de barres dans le plafond de
   détention, et elles ne donnent pas le même résultat. Mesuré le 15 septembre :
   E[R] allait de **+0.158 à −0.071** selon la grille, pour le même jeu, le même
   protocole et les mêmes données. Toujours moyenner sur ≥ 12 phases et comparer
   **apparié**, chaque phase à elle-même.

3. **Un chiffre juste comparé au mauvais repère est un chiffre faux.** Quatre
   occurrences en une journée (§4). Toujours se demander *par rapport à quoi*.

4. **Vérifier l'usage, pas l'existence.** Un test qui contrôle qu'une fonction
   existe ne contrôle rien. Chercher un appel dans l'**arbre syntaxique**, pas
   dans le texte du source — la définition contient la même chaîne.

5. **Un test qu'on n'a jamais vu échouer ne prouve rien.** Réintroduire
   délibérément le défaut et vérifier que le test le voit.

6. **Un mécanisme sans valeur mesurée ne rentre pas.** Sept écartés à ce jour :
   attention entre groupes (−2.3), méta-étiquetage (−2.9), taille par
   conviction (+0.3), lookback au-delà de 4 (nul), SAINT complet (ingérable),
   colonnes Binance non branchées (0.4847 d'AUC, sous le hasard), veto TabM
   (|t| < 1 partout).

7. **Un seul chemin de calcul.** Deux implémentations qui doivent s'accorder par
   convention finissent toujours par diverger, sans lever d'erreur.

8. **Une barre d'erreur doit venir de ce qui varie indépendamment.** Douze
   phases décalées de 67 barres couvrent la même période : leurs moyennes sont
   presque le même nombre, et leur écart-type mesure la *sensibilité au
   décalage*, pas l'incertitude. Utilisé comme barre d'erreur le 16 septembre,
   il a transformé 0.8 σ en « 3.0 σ » et fait retenir une largeur de stop sur
   du bruit. Des années, ou des occasions espacées au-delà de la durée d'un
   trade, sont des échantillons ; des phases n'en sont pas.

9. **L'alternance achat/vente n'est pas un repère neutre.** Sur une fenêtre
   directionnelle, le résultat dépend de *quelles dates* reçoivent un achat :
   exactement les mêmes trades donnent **−0.0670 en alterné et +0.0283 en
   symétrique**. Mesurer (achat + vente) / 2, qui ne peut pas en dépendre et
   déduit au passage la dérive du sous-jacent — qu'aucun modèle ne peut
   promettre de revoir.

10. **Compter les décisions, pas les trades.** La sélectivité à 5 % jette 95 %
    de l'information : la validation ne rend que ~150 trades, sur lesquels
    l'erreur-type vaut 0.11 R, donc elle ne peut pas voir un avantage de
    0.10 R. La même fenêtre porte 22 000 *décisions*, et sur celles-là le
    classement se mesure à 2.9 σ. Quand un échantillon est trop petit, changer
    la question avant de changer le modèle.

---

## 4. Les erreurs, et ce qu'elles coûtaient

### 4.1 Fuites de futur

| erreur | symptôme | coût |
|---|---|---|
| `np.gradient` (différence centrée) sur 5 colonnes Ichimoku | +25.5 pt en test, PF 2.85 | toutes les mesures antérieures invalidées |
| `merge_asof(backward)` sans `shift(1)` | aucun | jusqu'à 4 h de futur par ligne |
| bougie en formation gardée dans le flux live | aucun | 5 min de futur par décision |

**Rien ne les signale.** La colonne garde son nom, sa forme, son ordre de
grandeur et sa plage. Seule l'**invraisemblance** du résultat a trahi la
première. `test_causalite.py` les cherche désormais en recalculant toute la
chaîne sur une série tronquée.

### 4.2 Le test qui ne testait rien

Sa marge d'échauffement valait 62 bougies H4 là où il en faut ~280. Les deux
calculs rendaient `NaN`, et **`NaN` contre `NaN` compte comme un accord** : les
85 colonnes H4 étaient déclarées saines sans avoir jamais été évaluées. Le test
affiche maintenant le nombre de colonnes **réellement évaluées**.

### 4.3 Quatre repères faux, en une journée

| repère | ce qu'il valait vraiment | conséquence |
|---|---|---|
| point mort calculé sur `AvgW/AvgL` du **train**, comparé au winrate de **validation** | le signe de l'écart *est* celui de `PF − 1` | une epoch affichait +0.3 pt avec PF 0.91, arithmétiquement impossible |
| plafond d'entropie fixé à ln 3 = 1.099 | sous veto : `p²·ln3 + 2p(1−p)·ln2` = **0.621** | une politique au hasard (H 0.573) aurait semblé fortement différenciée |
| table de géométrie lue à **deux échelles d'historique** | ligne retenue à 9 ans, ligne écartée à 3.6 ans | le stop de 8×ATR écarté pour 1 739 occasions quand il en vaut 4 516 |
| détection du warmup par seuil de gradient (1e-3) | l'ensemble monte à 5e-4, le seuil allait être franchi | des epochs gelées comptées comme entraînées, référence du hasard polluée |

Le warmup se lit maintenant sur le **numéro d'epoch**, exact par construction.

### 4.4 Comparaisons entre échelles

- `episode_length` ramené de 2 016 à 576 sur l'argument « des épisodes plus
  longs rendent le même nombre de décisions, 2 054 en M5 contre 2 195 en H1 ».
  **Ces deux chiffres viennent de deux échelles différentes.** À l'intérieur du
  M5 les décisions sont proportionnelles aux barres collectées. Chaque epoch ne
  voyait plus que **6 %** des occasions du jeu, tirées ailleurs à chaque fois :
  après dix epochs, 0.7 passage sur les données. Symptôme qui l'a trahi : le PF
  d'**entraînement** plafonnait à 0.95.
- `mesure_court_terme.py` lisait le cache M1 de 3.6 ans quand le jeu en couvre
  9.05, et rendait un nombre d'occasions 2.6 fois trop bas comparé à un seuil
  **absolu**.

### 4.5 Gamma réglé pour une barre qui n'existe plus

`gamma` est **par barre**, la récompense est accumulée en `gamma^dt`. Deux
changements d'échelle sans y toucher (H1 → M5, SL 4 → 8).

```
mesure : 11 971 courses, SL 8xATR
  GAINS   3993   duree med 254 b   R moyen +2.00
  PERTES  7978   duree med 146 b   R moyen -1.02
-> un gain dure 1.74 fois plus longtemps qu'une perte (mecanique : TP 2x plus loin)

gamma     poids gain   poids perte   R:R effectif
0.995       0.567        0.711          1.56      -20.2 %
0.9999      0.987        0.993          1.95       -0.5 %   <- retenu
```

À 0.995 le point mort **vu par l'agent** valait 39.1 % quand l'environnement en
applique 33.8 % : 5.3 points de winrate de trop demandés, et une incitation à
fuir les configurations lentes — c'est-à-dire les gagnantes.

### 4.6 Capacité doublée sans mesure

La tête pèse 92 à 98 % du réseau et lit `n_features × d_model`. Passer de 103 à
260 colonnes l'a multipliée par 2.5 ; ajouter un second réseau, par 1.6.

```
mlp SAINT/patch   total     par occasion (4 516)
    16 / 32      98 324         21.8
     4 /  8      46 220         10.2   <- retenu
```

Seul ancrage empirique : le régime H1 qui avait rendu +2.2 en test portait
**11.6** paramètres par occasion. Ancrage hérité, **pas une mesure**.

### 4.7 Défauts de plomberie qui ne lèvent aucune erreur

- **Masque reconstruit à l'update** au lieu d'être celui qui a agi. Tant qu'il
  ne dépendait que de la position les deux coïncidaient ; un veto qui dépend de
  la barre les fait diverger, et PPO compare alors la probabilité nouvelle d'une
  action à son ancienne probabilité **sous une autre distribution**.
- **Tri des paramètres** testant `startswith("actor.")` quand les noms valent
  `membres.0.actor.*` → trois listes vides → gradient d'acteur affiché à zéro.
- **Fonction définie et jamais appelée** (`votant_courant` dans le live) — et le
  test d'alignement la validait par `hasattr`.
- **`build_policy` unidirectionnel** : reconnaissait PatchTST, pas SAINT.
- **`lookback` non déductible des poids** — n'importe quelle profondeur se
  reconstruit en ajustant le pas. Il est écrit dans le `_calib.json`.
- **Deux entraînements simultanés** sur la même carte : 89 °C, 315 MHz, et les
  deux processus écrivant les mêmes checkpoints.

---

## 5. Les découvertes mesurées

### 5.1 Ce qui a déplacé des chiffres, et ce qui n'a rien fait

**Rien** : six expériences d'architecture (voir règle 6). **Le mur n'a jamais
été l'architecture.**

**Quelque chose** : l'échelle de temps, la longueur d'historique (3.6 → 9.1
ans), les features (+1.6 → +3.8 pt), et surtout la **réduction de taille** du
modèle — 493 796 → 15 680 paramètres a fait passer PPO de −1.4 à +2.2 en test.

### 5.2 La géométrie, seul résultat solide de la journée

Référence de la politique **gelée** (donc du pur hasard), regroupée par run :

```
SL 4xATR  n=2   -5.21 pt  +/- 0.13
SL 8xATR  n=4   -1.12 pt  +/- 0.65
ECART : +4.08 pt +/- 0.67   (6.1 ecarts-types)
```

Six écarts-types, et ça ne doit rien au modèle : c'est le coût de la structure.
Le trou à combler est divisé par près de cinq.

### 5.3 PPO dégrade la validation, de façon reproduite

```
exec24 (SL 4)   gelee -5.34   entrainee -7.29   gain -1.95 +/- 0.93  (-2.1 sigma)
exec31 (SL 8)   gelee +0.13   entrainee -3.34   gain -3.47 +/- 1.60  (-2.2 sigma)
```

Et le côté **entraînement ne bouge pas non plus** (−1.5, −0.0, +0.4, +2.5,
−0.0). **Ce n'est donc pas du sur-ajustement** : le gradient pousse la politique
vers un endroit qui n'aide ni l'un ni l'autre.

Sur exec24, la dégradation était **entièrement sur les shorts** : −5.9 pt
(−3.2 σ) contre +2.3 pt non significatif sur les longs. Et avec assurance — le
seuil de sélection des shorts montait de 0.393 à 0.520 pendant que celui des
longs restait à 0.38. Un motif trouvé à l'entraînement qui **s'inverse** en
validation, pas une distribution aplatie.

### 5.4 L'avantage existe, et il est concentré

Courbe mesurée sur la validation, 12 phases, seuils calibrés sur le train :

```
select.   trades/ph      E[R]   err-type   en points de winrate
   5 %           5    +0.5591     0.4430        +18.6
  10 %           9    +0.2023     0.1508         +6.7
  15 %          15    +0.0226     0.1243         +0.8
  30 %          50    -0.1221     0.0382         -4.1   <- t = -3.2
 100 %         195    -0.0877     0.0167         -2.9
```

**Décroissance monotone du sommet au tout-venant.** Un modèle sans capacité de
classement donnerait une courbe plate à −0.088 partout. Aucun point n'est
significatif isolément, mais la **forme** dit qu'il y a un classement réel. Et
au-delà de 30 % l'espérance est significativement négative : les occasions peu
convaincantes ne sont pas neutres, elles perdent.

**Conséquence : monter la sélectivité n'achète pas de trades.** L'avantage meurt
vers 15 %.

### 5.5 Le diagnostic central

Le README portait depuis des semaines ce fait non expliqué : *une régression
logistique atteint 0.6271 d'AUC, aucune politique entraînée n'a dépassé 0.5707*.

**Explication proposée : on entraîne la politique à AGIR, puis on s'en sert
comme CLASSEUR.** PPO maximise le rendement des actions prises ; à l'évaluation
on jette 95 % des décisions et on garde les 5 % où sa probabilité est la plus
haute. Rien dans l'objectif de PPO ne récompense un bon **ordre** de ces
probabilités. Le modèle supervisé, lui, régresse le rendement réalisé : il est
entraîné exactement à classer.

Cohérent avec tout : le signal existe (§5.4), le supervisé le voit, PPO ne
l'apprend ni en train ni en validation, et l'écart d'AUC est stable.

### 5.6 La tête auxiliaire — en cours de test (exec32)

Une couche linéaire de plus sur le tronc partagé, dans SAINT **et** PatchTST,
qui régresse le rendement net d'un achat et d'une vente à chaque barre.

```
EPOCH 001   AuxL 1.6909
EPOCH 004   AuxL 1.4373
predicteur constant (la moyenne) : 1.9354
-> 25.7 % de variance expliquee, en 4 epochs, actor encore gele
```

**Sur les données d'entraînement.** Rien ne dit encore que ça transfère.

Trois choix non neutres : elle s'entraîne **dès le warmup** (sa cible est ce que
le marché a fait, pas ce que l'agent aurait gagné) ; les occasions non résolues
sont **masquées** et non mises à zéro ; `aux_coef = 1.0` la laisse **mener le
tronc** (MSE ~1.7 contre une perte d'acteur ~0.003), ce qui est voulu.
`aux_coef = 0.0` retire la tête et redonne exactement exec31 — c'est le témoin.

---

## 6. L'arithmétique qui borne tout

Le bruit de mesure vient **entièrement du nombre de trades**. L'écart-type
observé par epoch (2.70) correspond à celui d'un winrate sur 250 trades (3.05).

```
trades   err-type   avantage detectable a 2 sigma
   250     3.05 pt                        6.11 pt
   750     1.76 pt                        3.53 pt   <- ce dont on dispose
 3 000     0.88 pt                        1.76 pt

avantage espere (brut +0.09 a +0.13 R, friction 0.049) :
  0.09 R -> +1.39 pt  ->  4 664 trades necessaires
  0.13 R -> +2.74 pt  ->  1 195 trades necessaires
```

**Si l'avantage est au bas de la fourchette, le dispositif actuel ne peut pas le
démontrer même s'il est réel.**

### Les leviers, et leur état

| levier | verdict |
|---|---|
| sélectivité plus haute | **mort** — l'avantage meurt à 15 % (§5.4) |
| stop plus large | **PRIS le 16 septembre** — 8 → 12×ATR. Le verdict « mort » reposait sur un décompte à barrières fixes ; avec le stop suiveur la friction tombe de moitié et la détectabilité gagne malgré la perte d'occasions |
| plus d'historique BTC | **épuisé** — 9 ans, tout ce que Binance a |
| ordres **maker** (futures 2×2 bp) | friction 0.0372 → 0.0265 R, **−29 %**, avec risque de non-exécution |
| **multi-symbole en exécution** | **mort sur ce courtier** — voir ci-dessous |
| **multi-symbole en entraînement** | **ouvert** — n'augmente pas les trades de test, mais régularise et teste la prémisse |
| statistique de **rang** au lieu de la moyenne du sommet | **PRIS le 16 septembre, et c'est le levier qui a payé.** ρ +0.070 ± 0.024 sur 21 985 décisions, soit 2.9 σ, quand le PnL de validation du même run plafonnait à 1.3. Journalisé à chaque epoch (`mesure_rang.py`, `PPOConfig.diag_rang`) |
| **XAUUSD** | **ouvert** — 8 ans exploitables, friction 27 % moins chère, décorrélé du BTC ; bloqué par les 4 colonnes Binance qui n'existent pas pour lui |

Spreads Vantage mesurés (marchés fermés, à revérifier en séance) :

```
BTCUSD 2.24 bps (1.0x) | ETHUSD 10.57 (4.7x) | TRXUSD 30.30 (13.5x)
BNBUSD 38.21 (17.1x)   | SOLUSD 55.52 (24.8x) | XRPUSD 64.75 (28.9x)
BCHUSD 153.75 (68.7x)  | LTCUSD 199.02 (88.9x) | ADAUSD 328.21 | DOTUSD 911.47
```

BTC est un cas isolé. ETH à 4.7× porterait la friction à ~0.090 R, le régime
écarté. **Entraîner sur dix paires, n'exécuter que BTCUSD** reste possible :
aucun spread payé sur les autres.

---

## 7. Configuration au moment du transfert

```
instrument XAUUSD SEUL — le BTC est sorti de l'entrainement
cote       LONG SEUL (PPOConfig.side = "long", declare LA et nulle part ailleurs)
jeu        data_cache_XAUUSD_M5.pkl — 256 colonnes + 4 d'etat = 260 en entree
           trois echelles (M5 / H1 / H4), float32, source MetaTrader
geometrie  SL 10xATR, AUCUN objectif, trailing 2.0 R (declenchement ET distance)
           cible d'apprentissage de la tete auxiliaire : 2 R
compte     1 000 EUR, budget de risque 3 %, marge 0.174 %, lot min 0.01
           contrat 100 ONCES — a 1 000 EUR un lot minimum risque 2.81 %
           PLUSIEURS POSITIONS SIMULTANEES, K calcule par le solde a chaque barre
modele     ensemble SAINT + PatchTST dans le MEME rollout, 45 592 parametres
           episodes de 5 760 barres, plafond 160 par epoch, cible 4 000 decisions
           90 epochs, validation une sur trois
           tete auxiliaire supervisee, et c'est ELLE qui trie (tri_par_tete_aux)
foldage    WALK-FORWARD CHAINE : wf2 herite de wf1, wf3 de wf2
           normalisation figee sur le train du fold 1, la meme pour tous
veto TabM  COUPE (votant_tabm = False) — mesure nulle, code conserve
run        or_exec05
```

**Les périodes Ichimoku sont des nombres de BOUGIES**, pas des durées : Tenkan 9
vaut 45 min en M5, 9 h en H1, 36 h en H4. On ne les convertit jamais.

---

## 8. Fichiers qui comptent

| fichier | rôle |
|---|---|
| `saint_core.py` | source unique des colonnes, des architectures, de `build_policy` |
| `training.py` | PPO, walk-forward, l'ensemble, la tête auxiliaire |
| `prepare_m5.py` | construction du jeu, trois échelles |
| `test_causalite.py` | **aucune colonne ne doit changer quand on coupe le futur** |
| `test_alignement.py` | le live calcule-t-il les mêmes colonnes ? (260/260 vérifiées) |
| `checkpoints.py` | quel modèle charger — filtre de compatibilité d'observation |
| `mesure_votants.py` | balayages corrigés de la phase (sélectivité, veto) |
| `mesure_court_terme.py` | occasions et friction par géométrie, sans sortie au temps |
| `veille_epochs.py` | lecture des epochs en direct |
| `flux_live.py` | bougies M5 en direct, format identique au jeu |
| `JOURNAL_MESURES.md` | **tout l'historique des mesures, antichronologique** |

---

## 9. Ce qui est ouvert

1. **La fenêtre de test de la lignée `or_` est déjà consommée**, et par un
   harnais défectueux : `or_exec02` a lu les trois folds avec une règle de
   décision qui prenait des ventes dans un run long-only. Tout test ultérieur
   sur ces mêmes fenêtres est une **seconde lecture**, et doit être annoncé
   comme telle.
2. **Le chaînage n'a jamais franchi le fold 2 en conditions réelles.** Le
   mécanisme est vérifié — mêmes 107 tenseurs, mêmes formes entre folds, donc
   `strict=True` passe — mais aucun run n'est encore allé jusque-là.
3. **Le stress-test n'a toujours produit aucun chiffre.** `stress_test.py` est
   écrit et vérifié syntaxiquement, à relancer GPU libre.
4. **`kairos_live` garde deux branches mortes** — `duel` et `short` — dont les
   masques sont encore écrits `"both"` en dur. Inoffensives en long-only, non
   corrigées faute de pouvoir les exécuter.
5. **La boucle multi-instruments n'existe pas.** `Portefeuille`,
   `instruments.py` et `alignement.py` sont écrits et testés, mais
   `run_training_on_split` ne joue qu'un instrument.
6. **L'écart AUC reste inexpliqué** : 0.6271 pour une régression logistique,
   0.5707 pour la meilleure politique. `tri_par_tete_aux` contourne le
   problème sans le résoudre.
7. **Le GPU est bridé thermiquement** — jusqu'à 210 MHz sur 2 100 observés.
   Les durées d'epoch ne sont pas comparables entre elles tant que la
   température dérive.

---

## 10. Le seul conseil qui compte

Sur six corrections apportées en une journée, **toutes ont déplacé le tableau
dans le même sens : moins bien qu'annoncé.** Aucune n'a révélé de performance
cachée. Cette unanimité de direction est en soi un résultat.

Les deux meilleurs chiffres du dépôt — TabM +3.8 ± 2.3 (1.65 σ) et PPO exec18
+2.2 ± 2.2 (1.0 σ) — sont tous deux compatibles avec zéro, et tous deux mesurés
à une géométrie qui n'existe plus.

**Mesurer sur la validation avant de payer un run. Toujours.**
