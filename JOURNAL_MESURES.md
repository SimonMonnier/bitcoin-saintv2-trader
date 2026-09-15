# Journal des mesures — ce qui a été vérifié, et ce qui ne l'a pas été

Ce fichier existe parce que la plupart des impasses du projet ont été des
raisonnements plausibles jamais confrontés à une mesure. Chaque entrée dit ce
qui a été mesuré, comment, et ce que la mesure **ne** permet pas de conclure.

Ordre antichronologique.

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
