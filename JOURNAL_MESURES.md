# Journal des mesures — ce qui a été vérifié, et ce qui ne l'a pas été

Ce fichier existe parce que la plupart des impasses du projet ont été des
raisonnements plausibles jamais confrontés à une mesure. Chaque entrée dit ce
qui a été mesuré, comment, et ce que la mesure **ne** permet pas de conclure.

Ordre antichronologique.

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
