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
