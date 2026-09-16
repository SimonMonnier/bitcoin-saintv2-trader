# Journal des mesures — ce qui a été vérifié, et ce qui ne l'a pas été

Ce fichier existe parce que la plupart des impasses du projet ont été des
raisonnements plausibles jamais confrontés à une mesure. Chaque entrée dit ce
qui a été mesuré, comment, et ce que la mesure **ne** permet pas de conclure.

Ordre antichronologique.

---

## 16 septembre 2026 — cinq défauts de mesure, et la géométrie choisie sur une table incohérente

Journée entière passée à corriger des **instruments**, pas des modèles. Aucune
des cinq erreurs ci-dessous n'était visible dans un résultat : chacune
produisait des chiffres plausibles, du bon ordre de grandeur, du bon signe.
C'est la raison d'être de cette entrée.

### Le jeu de données — trois échelles, 260 colonnes

`data_cache_BTCUSD_M5.pkl` : 948 532 barres, 2017-08-31 → 2026-09-14.

| bloc | colonnes | contenu |
|---|---|---|
| M5 | 103 | bases, flux, temps, range, Ichimoku |
| H1 | 85 | les mêmes structures, resamplées et décalées d'un `shift(1)` |
| H4 | 85 | idem |

Les périodes Ichimoku sont des nombres de **bougies** : Tenkan 9 couvre 45 min
en M5, 9 h en H1, 36 h en H4. Les trois blocs portent les mêmes noms et ne
décrivent pas du tout la même chose. Un trade M5 durant 3 h 40 à 12 h selon la
géométrie, le M5 décrit ce qui se passe *pendant* le trade, le H1 le mouvement
qui le contient, le H4 le régime qui contient ce mouvement.

Les features sont stockées en `float32` : 2.1 Go → 1.1 Go, ce qui laisse la
place de faire tourner l'entraînement, la veille et le test de causalité
ensemble sur 16 Go. Aucune n'est un niveau absolu, toutes sont des rapports,
des écarts normalisés ou des rangs.

**Pas de M1, et ce n'est pas une question de téléchargement.** La plus longue
ligne Ichimoku en M1, Senkou-B 52, couvre 52 minutes — moins que le Kijun M5
(2 h 10). Le bloc M1 entier tiendrait *à l'intérieur* de ce que les deux
premières lignes M5 décrivent déjà : c'est de la résolution en plus, pas un
horizon en plus. Et l'hypothèse qu'on teste est que le signal vit **au-dessus**
du M5, pas en dessous.

---

### Défaut 1 — le test de causalité passait sans rien tester

`test_causalite.py` compare chaque colonne calculée sur la série entière à la
même colonne calculée sur la série tronquée en t. Sa marge d'échauffement
valait 3 000 barres M5, soit **62 bougies H4**. Or le bloc H4 réclame
`FENETRE_EXT = 200` bougies plus `SENKOU_B = 52` et son décalage de 26 :
~280 bougies H4, soit 13 400 barres M5.

Les deux calculs rendaient donc `NaN`, et `NaN` contre `NaN` compte comme un
accord. Les **85 colonnes H4 étaient déclarées saines sans avoir jamais été
évaluées.** C'est le pire mode de défaillance possible pour un test : celui qui
rassure à tort.

Correctif : `MARGE = 16000`, et le test affiche désormais le nombre de colonnes
**réellement évaluées**. Résultat après correction : **260 / 260, aucune fuite.**

### Défaut 2 — le point mort était calculé sur la mauvaise fenêtre

La ligne `META` publie `AvgW`/`AvgL` calculés sur l'**entraînement**. La veille
les comparait au winrate de **validation**.

Preuve arithmétique, exec23 epoch 7 :

```
PF_train x pertes/gains   = 1.708
AvgW/AvgL de META         = 1.717     <- c'est l'entrainement
le meme rapport cote VAL  = 1.542
```

Le révélateur ne demande aucune donnée : **le signe de l'écart au point mort
est celui de `PF − 1`**. L'epoch 7 affichait `+0.3 pt` avec un `PF` de 0.91,
ce qui est impossible. Trois epochs de suite avaient été lues comme « revenues
au point mort » alors qu'elles perdaient. Après correction : `−2.2 pt`.

Tout se déduit maintenant de la seule ligne VAL, où les trois chiffres sont
cohérents entre eux, et une assertion vérifie l'identité à chaque epoch.

### Défaut 3 — la collecte divisée par 3.5 sur une comparaison inter-échelles

En recalculant les réglages PPO, `episode_length` est passé de 2 016 à 576
barres. Le raisonnement : « des épisodes cinq fois plus longs rendent le même
nombre de décisions, 2 054 en M5 contre 2 195 en H1 ».

**Ces deux chiffres viennent de deux échelles différentes.** À l'intérieur du
M5, les décisions sont proportionnelles aux barres collectées, et la mesure ne
disait rien de cela. C'est la même faute que la mesure à plafond de 30 barres
qui avait recommandé un stop de 8×ATR inexistant : comparer entre échelles ce
qu'il fallait comparer à échelle constante.

Coût : la collecte est tombée à 55 296 barres par epoch, soit **10 % de la
fenêtre d'entraînement**, et les décisions PPO de ~2 050 à ~850. Le jeu M5
offre ~15 100 occasions indépendantes ; chaque epoch n'en échantillonnait que
6 %, tirées ailleurs à chaque fois. Après dix epochs le modèle avait vu
**0.7 passage** sur ses données. En H1 il voyait 2 195 décisions pour 2 314
occasions — la quasi-totalité du jeu à chaque epoch.

**On avait acheté 6.5 fois plus de données et on ne les livrait jamais à
l'optimiseur.**

Le symptôme qui l'a trahi : le `PF` d'**entraînement** plafonnait à 0.87–0.95
pendant dix epochs. Un modèle qui n'arrive pas à gagner sur les données qu'il
optimise ne sur-apprend pas — il manque d'échantillons.

Correctif : `episodes_per_epoch` 96 → 336, ce qui redonne les 193 536 barres
d'origine avec 336 départs indépendants au lieu de 96. Vérifié : 3 073
décisions à l'epoch 1 contre 867.

### Défaut 4 — un plafond CUDA sur le PRODUIT features × épisodes

SAINT replie un axe dans la dimension de lot avant d'appeler l'attention : sur
l'axe temps le lot vaut `B × F`. À 336 épisodes et 261 colonnes cela fait
87 696, au-dessus de la limite de grille CUDA de 65 535 — et le message est
`CUDA error: invalid configuration argument`, qui ne nomme ni le lot, ni les
colonnes, ni l'attention.

Le plafond portait donc sur le **produit** de deux réglages choisis séparément
et pour des raisons sans rapport. Chaque enrichissement du jeu de colonnes
aurait dû se payer d'une réduction de la collecte, exactement quand on cherche
à augmenter les deux.

Correctif : le lot est tranché à 32 768 dans `AxialAttention.forward`. Le
résultat est identique au bit près — l'attention ne mélange jamais deux
éléments du lot entre eux, donc la découper n'en change aucun.

### Défaut 5 — la géométrie choisie sur une table à deux échelles d'historique

La table qui a fait retenir `SL 4×ATR` comptait les occasions de deux façons
selon la ligne. Les mesures viennent du cache M1, soit **3.6 ans** ; le jeu M5
en couvre **9.05**. La ligne retenue a été mise à l'échelle des neuf ans
(5 811 → 15 089), la ligne écartée est restée à celle des 3.6 ans (1 739), et
c'est ce 1 739 qui la faisait tomber sous le seuil de ~4 900.

À la même échelle :

```
config         friction   horizon   occasions 9 ans   avantage requis
H1  SL 2xATR    0.047 R     13.0 h            2 314        +0.0233 R
M5  SL 4xATR    0.099 R      3.7 h           15 089        +0.0892 R
M5  SL 8xATR    0.049 R     12.2 h            4 516        +0.0425 R
```

Ce qui tranche est la **détectabilité**, `(avantage − friction) × √N` :

```
avantage brut   SL 4     SL 8
     0.09 R     0.10     3.19
     0.11 R     2.56     4.54
     0.13 R     5.01     5.88
     0.15 R     7.47     7.22
```

Le croisement tombe à **0.1456 R**. L'avantage mesuré du modèle vaut +0.09 à
+0.13 R : il est tout entier du côté où 8×ATR gagne. À 0.09 R, le 4×ATR ne
laisse pas 1 % de l'avantage brut survivre à la friction.

**Et surtout, l'horizon.** Les +0.09 à +0.13 R ont été mesurés en H1, où le
trade médian dure 13 heures. Le 4×ATR en M5 dure 3 h 40 : un avantage mesuré
sur un horizon, déployé sur un autre 3.5 fois plus court, en supposant qu'il
suivrait. Rien ne le garantissait — prédire quatre heures et prédire douze
heures sont deux problèmes différents. Le 8×ATR dure 12.2 h et rend la question
identique à celle qu'on savait résoudre.

Le M5 garde alors son seul avantage réel sur le H1 : deux fois plus d'occasions
indépendantes, 4 516 contre 2 314, à friction et horizon inchangés.

---

### Ce que les runs ont mesuré

**exec23** — 260 colonnes, SL 4×ATR, 850 décisions/epoch.

```
politique GELEE   n=5  ecart moyen  -5.07 pt   ecart-type 1.97
ENTRAINEE         n=4  ecart moyen  -3.56 pt   ecart-type 2.15
gain  +1.52 +/- 1.39  (1.1 ecart-type)  ->  rien
```

**exec24** — identique, collecte réparée, 3 073 décisions/epoch.

```
politique GELEE   n=5  ecart moyen  -5.34 pt   ecart-type 0.80
ENTRAINEE         n=3  ecart moyen  -7.29 pt   ecart-type 1.49
gain  -1.95 +/- 0.93  (-2.1 ecarts-types)
```

**La collecte réparée n'a pas rendu le modèle meilleur : elle a rendu la mesure
assez fine pour montrer qu'il est plus mauvais que le hasard.** L'écart-type de
la référence passe de 1.97 à 0.80 — le mètre étalon est devenu 2.5 fois plus
fin, et ce que le bruit cachait était une dégradation.

Décomposé par sens, sur l'ensemble des trades de validation :

```
sens    etat      trades      WR      PnL      vs gele
LONG    gele        1442   31.9%   -2609$
LONG    entraine    1122   34.2%   -1383$      +2.3 pt  (+1.2 ecart-type)
SHORT   gele        1767   32.4%   -2667$
SHORT   entraine     931   26.5%   -3308$      -5.9 pt  (-3.2 ecarts-types)
```

L'amélioration des longs n'est pas significative. **Le modèle casse les
shorts**, et il les casse *avec assurance* : le seuil de sélection des shorts
monte de 0.393 à 0.440 puis 0.520 pendant que celui des longs reste à
0.37–0.385. Les 5 % de shorts retenus sont de plus en plus confiants et de plus
en plus faux — signature d'un motif trouvé dans la fenêtre d'entraînement qui
**s'inverse** en validation, pas d'une distribution aplatie où la sélection
redeviendrait aléatoire.

**exec25** — SL 8×ATR / TP 16×ATR, épisodes de 1 440 barres, 1 966 décisions.

```
politique GELEE   n=5  ecart moyen  -2.46 pt   ecart-type 4.27   positives 2/5
```

La référence du hasard passe de −5.06 à −2.46 points. La friction passait de
0.099 à 0.049 R, rapport **2.02** ; le trou dans lequel une politique aléatoire
se trouve passe de 5.06 à 2.46 points, rapport **2.06**. Les deux coïncident à
2 % près : le calcul qui a motivé le changement décrivait bien ce qui se passe.
La cible passe de « trouver 5.3 points » à « trouver 2.5 points ».

---

### Deux faits de mesure à garder

**L'écart train-val existe sans modèle.** Sur les cinq epochs **gelées**
d'exec24 il vaut `+3.28 ± 1.05 pt`. La fenêtre de validation est structurellement
plus dure que celle d'entraînement, indépendamment de ce qu'on apprend. Un gap
ne devient un sur-ajustement qu'au-delà de `+5.4 pt` ; en dessous c'est la
fenêtre qui parle. Sans ce garde-fou, les `+6.6 pt` d'exec24 auraient été lus
comme un sur-ajustement franc alors qu'ils dépassent le bruit de justesse.

**Le bruit par epoch suit la durée des trades.** À SL 8×ATR la fenêtre de
validation, 321 jours, ne porte plus que 180 à 270 trades au lieu de 600 à 800 :
un trade de 12.2 h y tient 3.3 fois moins souvent qu'un trade de 3 h 40.
L'incertitude par epoch passe de ±1.9 à ±3.4 points de winrate. Deux epochs
**gelées** consécutives ont donné +0.2 puis −7.5 sans que la politique bouge
d'un iota — quatorze points d'écart, du bruit pur. Ce bruit ne contamine rien,
puisque la moyenne des dix derniers jeux de poids a remplacé le choix du
meilleur checkpoint ; il rend seulement la veille illisible epoch par epoch.

---

### Binance — la question est close, et elle l'était déjà

Les quatre colonnes jamais branchées — `funding_rate`, `funding_cum24`,
`oi_change`, `ls_ratio_retail` — donnent **0.4847 d'AUC à elles seules, sous le
hasard**, et une corrélation plate aux rendements à tous les décalages.
`ls_ratio_top` avait été retirée pour un apport marginal de −0.0000, avec une
autocorrélation de 0.999985 : une variable de régime, constante pendant un
trade, incapable de départager deux entrées.

La raison structurelle empire au M5 : le funding est un cycle de 8 heures pour
un trade de 3 h 40 à 12 h.

La seule qui portait quelque chose, `taker_ratio` (−0.0498 si on la retire),
**est déjà là** — et en M5 elle vient des klines spot sur les neuf ans, pas du
fichier de features qui s'arrête à 2022-12-15.

Le prix serait d'ailleurs prohibitif :

```
jeu                           barres   occasions independantes
actuel, neuf ans             948 532            15 090  (SL 4)
limite aux metrics Binance   599 659             9 540
limite au fichier actuel     394 530             6 276
```

Remplir le trou par une constante serait la panne `spread_rel`, déjà payée une
fois : une colonne absente sur 60 % du jeu n'apprend pas le marché, elle
apprend la date.

### Ce que cette journée ne dit PAS

- Que le SL 8×ATR va marcher. Une seule epoch entraînée au moment d'écrire,
  `+1.6 pt` contre un bruit de `±4.3` : ça ne prouve rien. Il faudra une
  douzaine d'epochs entraînées pour que l'écart sorte à deux écarts-types.
- Que les trois échelles apportent quelque chose. exec23 et exec24 les portent
  toutes deux, et aucun des deux n'a rien montré de positif ; la seule variable
  qui a déplacé un chiffre est la géométrie.
- Que les shorts sont perdus. La dégradation à −3.2 écarts-types a été mesurée
  **à SL 4×ATR uniquement**, c'est-à-dire sur l'horizon où l'avantage n'avait
  jamais été mesuré. À SL 8×ATR la première epoch entraînée les laisse à 34.2 %,
  dans la plage gelée. Si le motif revient, l'expérience à faire est un run
  **long seul** — `cfg_long` existe déjà dans `training.py`.
- Que la fenêtre de test dise quoi que ce soit. Elle n'a pas été touchée, et
  elle ne le sera qu'une fois.

---

## 15 septembre 2026 — trois biais de mesure, et ce qu'ils ont coûté

La nuit a produit une chaîne de conclusions dont **aucune ne tenait**. Elles
sont listées ici parce que chacune semblait raisonnable isolément.

### Biais 1 — le chevauchement des fenêtres de résultat

Entrées échantillonnées toutes les 10 barres, barrières mettant jusqu'à 240
barres à se résoudre. Les résultats voisins se recouvrent presque
entièrement : on compte 1 368 trades là où il y en a **58 d'indépendants**.

    modele a arbres, 5 % de selectivite
      avec chevauchement    58.0 % de reussite   t +9.02
      sans chevauchement    39.7 %               t -0.87

C'est ce biais qui avait fait choisir le R:R 1.4, et il contaminait le tableau
inscrit dans `saint_core` (E[R] +0.1751, t +3.4).

### Biais 2 — la métrique ne correspondait pas au point de fonctionnement

L'AUC classe **toute** la distribution ; la stratégie ne touche que les 2 à
5 % du sommet. Retirer le bloc H1 améliore l'AUC de +0.0136 et dégrade
l'espérance à toutes les sélectivités utilisées. Suivre l'AUC aurait fait
supprimer treize colonnes utiles.

### Biais 3 — la phase d'échantillonnage

Une entrée toutes les 240 barres : il y a 240 phases possibles. Même jeu,
mêmes données, même protocole, seule la phase change :

    E[R] a 5 %, 8 phases : +0.1579 -0.0696 +0.0317 -0.0487
                           -0.0295 -0.0308 -0.0711 +0.0164
    moyenne -0.0054   ecart-type 0.0757   etendue 0.2290

**La phase 0, utilisée pour toutes les mesures, était la plus favorable des
huit.** Tous les écarts conclus cette nuit — R:R (0.21), H1 (0.16), H4 (0.10),
M5/M15 (0.06 à 0.29), Ichimoku (0.05) — sont du même ordre que ce bruit.

**Correctif : moyenner sur les phases et comparer APPARIÉ à phase égale.**
Le R:R 2.0 ne survit pas : écart +0.0382 (t +1.24) à 2 %, −0.0071 (t −0.31) à
5 %. Ni meilleur ni pire que 1.4.

### Ce que la mesure peut et ne peut pas dire

Seuil de détection ≈ 0.05 en E[R]. Un E[R] de **+0.02** — deux fois et demie
en dessous — donnerait +0.6 % par jour à 25 trades et 1.2 % de risque.

**La sonde ne distingue donc pas « rien » d'« excellent ».** Conclure à
l'absence d'avantage à partir d'elle est une erreur de raisonnement. Le
protocole purgé jette 99.6 % des données pour garantir l'indépendance ;
l'entraînement réel, lui, voit chaque barre.

### Candidats mesurés, tous non concluants

Ichimoku (3 jeux de périodes), bloc H4, bloc M5, bloc M15, retrait du H1,
basis perpétuel/spot. Aucun écart ne dépasse le bruit de phase. Le basis seul
donne **AUC 0.5003** — exactement le hasard.

Seul effet robuste, à ~3 écarts-types : retirer `taker_ratio` (−0.244) ou
`taker_1m_ma5` (−0.252).

### exec8 — 26 epochs, architecture complète, R:R 2.0

Apprentissage massif et mesurable : étendue ×2 400 (0.0002 → 0.489), perte du
critique ÷18 (20.75 → 1.15), entropie de 1.099 à 0.187.

**Résultat plat depuis l'epoch 7.** Meilleur écart au point mort −11.2 pt
(epoch 9) ; −13.6 pt à l'epoch 26. 48 épisodes de validation sur 48 perdants.

> **Ne pas sur-interpréter la convergence du critique.** J'avais écrit qu'un
> critique à 1.15 sur une politique perdante prouvait qu'il n'y a rien de
> mieux à trouver. C'est faux : une perte de critique basse signifie que la
> valeur est bien prédite, ce qui n'établit rien sur le plafond atteignable.
> Correction due à une analyse indépendante du même run.

---

## 14 septembre 2026 — soirée

### Le flux Binance à la minute : +0.0087 d'AUC

**Question.** Existe-t-il d'autres features Binance utiles, absentes de MT5 ?

**Mesure.** Sonde logistique, barrières SL 2.0×ATR / R:R 1.4, friction complète
de l'environnement, train 0–55 % / validation 55–70 %, fenêtre de test
intouchée. Outils : `mesure_binance_extra.py`, `mesure_flux_1m.py`.

| Opération sur le jeu de 30 | AUC | Écart |
|---|---|---|
| référence | 0.6185 | — |
| sans `taker_ratio` | 0.5680 | **−0.0498** |
| sans `ls_ratio_top` | 0.6177 | −0.0000 |
| + `taker_1m_ma5` | **0.6271** | **+0.0087** |
| les 4 colonnes Binance non branchées, seules | 0.4847 | sous le hasard |

**Conclusion.** `ls_ratio_top` remplacée par `taker_1m_ma5` (compte de features
inchangé, donc coût GPU inchangé). Les quatre colonnes inutilisées — funding,
open interest, ratio long/short retail — sont écartées.

**Le mécanisme.** Une feature doit varier à l'échelle où la décision se prend.
Détention médiane d'un trade : 7 barres. Autocorrélation à 1 minute :
`taker_ratio` 0.82, tout le reste entre 0.988 et 0.99998. Le funding change
toutes les 10 heures — sur 7 minutes, c'est une constante. Le balayage de la
fenêtre de lissage confirme, en bosse régulière à sommet unique :

```
1min +0.0016 | 3min +0.0067 | 5min +0.0087 | 10min +0.0050
15min +0.0027 | 30min +0.0017 | 60min +0.0004 | 120min -0.0000
```

À 120 minutes la série est redevenue une variable de régime et vaut exactement
zéro, comme `ls_ratio_top`.

**Ce que la mesure ne dit pas.** Elle porte sur une **sonde logistique**, pas
sur la politique. La sonde est à 0.6271 ; aucune politique entraînée n'a dépassé
0.5707. Rien ne garantit qu'un gain sur la sonde se transmette à PPO.

**Alignement horaire.** Les archives Binance sont en UTC, le cache MT5 en heure
du courtier (DST américain). `mesure_flux_1m.py` **refuse de mesurer** si la
nouvelle série ne corrèle pas avec le `taker_ratio` déjà en service : une
corrélation faible signifie un décalage, pas une feature inutile. Contrôle
obtenu : +0.5485.

### Durée de détention : la sortie par le temps ne servait plus à rien

**Mesure.** 250 000 entrées de la fenêtre d'entraînement, barrières courantes.

| | LONG | SHORT |
|---|---|---|
| résolu ≤ 60 barres | 98.21 % | 98.32 % |
| résolu ≤ 240 barres | 99.91 % | 99.92 % |
| détention médiane | **7 barres** | **7 barres** |
| encore ouverts à 240 barres | 0.09 % | 0.08 % |

**Conséquence.** `max_holding_bars` passé à 0. Il interceptait moins d'un trade
sur mille, et ceux-là finissaient 52 % TP / 48 % SL — aucun biais à préserver.
Sa justification venait de l'**or** à 5×ATR, où l'agent restait 99.9 % du temps
en position ; à 2.0×ATR ce régime n'existe pas.

Le biais de survie reste couvert par la liquidation de fin d'épisode, qui n'a
jamais été le même mécanisme.

**Corollaire.** `scalping_max_holding` ramené de 120 à 30 : à 120, la feature
`bars_held_norm` valait ~0.06 pour un trade médian et ne portait presque rien.

### Gamma : bonne valeur, mauvais argument

`gamma = 0.995` est **conservé**, mais sa justification écrite était adossée à
`max_holding_bars = 240`, désormais nul.

La durée d'un trade n'est pas le bon critère : le semi-MDP la traite déjà par
son bootstrap `γ^Δt`, et à 7–12 barres **toute** valeur raisonnable préserve le
résultat (96.6 % à 0.995, encore 80.8 % à 0.97). Ce que gamma gouverne, c'est
l'**horizon d'opportunité entre trades** — la valeur d'attendre une meilleure
configuration. Un cycle complet fait ~32 barres ; 0.995 en montre ~6.

**Toujours pas mesuré.** Aucune ablation n'a comparé les valeurs entre elles.

---

## Pièges d'exécution — à ne pas refaire

### `os.kill(pid, 0)` détruit le processus sous Windows

Un veilleur en **lecture seule** a supprimé un entraînement à l'epoch 7.

Sous POSIX, `os.kill(pid, 0)` est le test de vie standard. Sous Windows,
`os.kill` appelle `TerminateProcess(handle, sig)` : **le signal devient le code
de sortie**. `os.kill(pid, 0)` ne teste donc rien, il tue le processus avec le
code 0 — ce qui ressemble à un arrêt propre dans les journaux, sans exception,
sans trace.

Utiliser `tasklist` (voir `relais_exec5.py`).

### `sed -i` réécrit les fins de ligne du fichier entier

Sous Git Bash sur Windows, `sed -i` convertit CRLF en LF sur tout le fichier.
Un diff de trois lignes devient un diff de mille. Restaurer après coup, ou
éditer autrement.

### La configuration live avait été oubliée lors du passage à BTC

`kairos_live.py` est resté sur `symbol = "XAUUSD"`, `atr_sl_mult = 5.0`,
`atr_tp_mult = 10.0` et les checkpoints or (1.96 Mo) **des semaines après** le
pivot vers BTC. Les quatre étaient cohérents entre eux : c'était le déploiement
or complet, intact. Ne corriger que le symbole aurait fait trader BTC par un
réseau entraîné sur l'or — pire que l'état initial.

Un préfixe de checkpoint par jeu d'observation, et des chemins explicites,
valent mieux qu'un chemin générique : un mauvais fichier chargé ici trade sans
erreur visible.

---

## Comment lire les journaux d'entraînement

### Les 5 premières epochs ne mesurent rien

`critic_warmup_epochs = 5` : l'actor est **gelé**. On le voit à
`H = 1.099` (= ln 3, l'entropie maximale sur 3 actions), `clipfrac 0.0 %`,
`ActorL` et `KL` à 0.0000.

Les epochs 1 à 5 partagent donc **le même réseau**. Elles ne sont pas cinq
mesures indépendantes, mais un seul tirage observé cinq fois. Pourtant le
résultat y varie fortement — jusqu'à 2 $ par trade.

**Pourquoi.** Le seuil calibré tremble de ±0.002 d'une epoch à l'autre, alors
que l'**étendue** des probabilités du modèle vaut 0.0011 à 0.0043. Le
tremblement du seuil est du même ordre que l'écart total entre les probabilités :
déplacer la barre reshuffle presque entièrement les 5 % de situations retenues.
La validation mesure alors quel échantillon a été tiré, pas la qualité du modèle.

### Plancher de bruit mesuré

Trois runs indépendants, epoch 1, politique non entraînée :

```
exec3   -6.48$/trade   WR 19.5%
exec4   -5.02$/trade   WR 24.3%
exec5   -7.22$/trade   WR 16.9%
```

Un éventail de **2.2 $ par trade**. Toute différence inférieure à cet ordre de
grandeur, à une epoch donnée, ne signifie rien — dans un sens comme dans
l'autre.

### Le seuil d'équilibre se recalcule à chaque epoch

Sur les `AvgW` / `AvgL` **réalisés**, jamais sur le R:R nominal. Il oscille
autour de 43.7–44.3 %. La ligne du hasard, elle, est à 22.8 %.

---

## 2026-09-15 — Le mur n'était pas l'architecture, c'était la taille du jeu

### Ce qui a été mesuré, dans l'ordre

**1. exec11 a échoué, et ses trois correctifs avec lui.** Arrêt précoce
(patience 25), modèle réduit (1.19 M → 298 k), entropie 0.008 → 0.030. Résultat
sur trois folds : 0/9, 1/15 et 0/22 epochs positives après warmup, moyennes
−3.36, −4.37 et −4.97 points sous le seuil d'équilibre. L'entropie est tombée
de 1.099 à 0.51 malgré le coefficient quadruplé — ralentie, pas retenue. Le
modèle réduit a fait **pire** qu'exec10 sur le même fold (−0.57 pt au mieux
contre +5.1 pt à l'epoch 11).

Trois changements simultanés : on sait que l'ensemble n'a pas aidé, pas lequel
a coûté.

**2. Les TEST affichés par le walk-forward ne sont pas propres.** Les trois
tranches (−313 $, −82 $, −69 $) ont été traversées par exec2 à exec11. Elles
confirment la validation, elles ne la remplacent pas.

**3. Le repère qui manquait depuis le début : le coût d'entrer au hasard.**

```
entrer au hasard, M1, SL 2xATR / TP 4xATR :  -0.45  R
entrer au hasard, H1, même géométrie      :  -0.012 R   (jeu MT5, 3.6 ans)
```

Le passage en H1 avait donc **réussi** : la friction était effacée. Le problème
avait changé de nature sans qu'on s'en aperçoive.

**4. TabM, première fausse piste, attrapée à temps.** Sur le jeu MT5 de 30 379
barres, TabM montrait une progression monotone avec la sélectivité (+0.009 →
+0.030 R), 8 phases sur 12 positives. La décomposition par période l'a tuée :

```
bloc                    0        1        2        3
hasard             -0.049   +0.044   -0.023   -0.020
toujours acheter   +0.092   +0.135   -0.090   -0.004
TabM (marge 0.40)  -0.068   +0.222   +0.067       -
```

Tout venait du bloc 1, la seule période haussière, et TabM y faisait **moins
bien qu'un ordre d'achat aveugle**. Sur le bloc 0 il faisait pire que le hasard.
L'erreur-type publiée par le banc est calculée sur les PHASES, distantes de deux
heures : elles ne sont pas indépendantes et l'erreur est trop belle. **La seule
dimension qui sépare est le temps.**

**5. Le chiffre qui explique tout le projet.**

```
écart-type d'un trade                    ~1.4  R
avantage recherché                       ~0.02 R
trades indépendants pour le détecter   (1.4/0.02)^2 ≈ 4 900

fenêtre d'entraînement H1 (MT5)              890
par bloc de validation                       143
```

Facteur **trente** entre ce qu'il faut pour *mesurer* l'avantage et ce dont on
disposait. Cela réconcilie tout l'historique : en M1 on avait les échantillons
mais la friction mangeait tout (−0.45 R) ; en H1 la friction avait disparu mais
les échantillons avec.

### Ce qui a été changé

**Source de données : MT5 → archives Binance spot.** 2017-08 au lieu de
2023-02, 79 340 barres au lieu de 30 379, ~2 310 occasions d'entraînement au
lieu de 890. Spot et non futures : deux ans et demi de plus, et surtout une
seule source sur toute la période — mélanger les deux créerait une couture au
milieu du jeu, que le modèle apprendrait. La friction reste celle du courtier,
qui est le choix prudent.

**`spread_rel` retiré.** Il venait de MT5, qui ne remonte qu'à 2023 ; sur les
six années ajoutées il aurait fallu le constanter. Une colonne constante sur
70 % du jeu apprend au modèle à distinguer « avant » de « après », c'est-à-dire
la date. Remplacé par deux colonnes issues des mêmes archives que le prix
(taille moyenne d'un trade, rang d'intensité sur la semaine), toutes deux en
rang ou en écart à leur propre normale — sur neuf ans où le volume horaire a
changé d'ordre de grandeur, tout niveau absolu encoderait l'année.

### Ce que le jeu long a donné

```
TabM, 6 blocs, marge +0.10       E[R] +0.024 ± 0.006 (phases), 11/12 phases
                                 écart au hasard +0.085 R, 6 blocs sur 6
                                 test des signes : p = 0.016
```

Premier résultat du projet dont le **signe tient sur toutes les périodes**.
L'E[R] absolu (+0.034 ± 0.031 au niveau des blocs) reste indistinguable de zéro.

### La géométrie des barrières, mesurée sans aucun modèle

```
SL 1.0xATR  →  E[R] hasard  -0.079     (friction = 0.146 R aller-retour)
SL 1.5xATR  →               -0.049
SL 2.0xATR  →               -0.035     ← réglage en place
SL 3.0xATR  →               -0.020
SL 3 / R:R 3 / 48 h  →      +0.021     (57 % de courses résolues)
```

Encore la friction fixe : élargir le stop la divise. Mais tester TabM sur les
géométries neutres a donné l'inverse de l'intuition — le stop large monte le
niveau et **perd l'avantage du modèle** :

```
géométrie                repère    TabM +0.10    blocs mieux    err-type
SL 2 / R:R 2 / 24 h      -0.033      +0.028          5/6         0.027
SL 3 / R:R 3 / 48 h      +0.013      +0.063          3/6         0.065
SL 3 / R:R 2 / 48 h      -0.014      +0.009          4/6         0.055
SL 3 / R:R 1.5 / 48 h    -0.016      +0.040          4/6         0.051
```

Un stop trois fois plus large divise par deux le nombre de trades et double
l'erreur-type. **La géométrie en place était la bonne ; ce qui manquait,
c'étaient les données.**

### Un défaut de déploiement trouvé au passage

`build_policy` ne déduisait l'architecture que dans un sens : il reconnaissait
un checkpoint PatchTST, mais un checkpoint SAINT tombait dans la branche par
défaut — qui vaut « patchtst » depuis exec10. Le live aurait construit la
mauvaise architecture et échoué sur une liste de clés illisible. La déduction
couvre désormais les deux cas et refuse explicitement un fichier ambigu.

### Ce que ces mesures ne disent PAS

- Que TabM gagnerait en réel. +0.034 ± 0.031 n'est pas significativement
  positif ; seul l'**écart au hasard** l'est.
- Que PPO va y arriver. exec12 tourne avec exactement le réglage d'exec11 et
  2.6 fois plus de données (5.4 paramètres par barre au lieu de 18) — c'est la
  seule variable qui change, précisément pour que le résultat soit lisible.
- Que le spot Binance se comporte comme le CFD du courtier. Les prix diffèrent
  de quelques points de base et les frais n'ont rien à voir ; on n'en reprend
  que la forme du marché.
