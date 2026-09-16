# KAIROS — Agent RL Multi-Stratégie pour BTCUSD M5

> *Le modèle ne prédit pas le marché. Il attend le moment.*

> **PPO + SAINTv2** : agent d'apprentissage par renforcement entraîné en walk-forward sur neuf ans d'historique BTCUSD en barres de cinq minutes, lu simultanément à trois échelles (M5, H1, H4), déployable en live sur MetaTrader 5 via une interface graphique dédiée.

---

## ⚠️ État actuel — aucun modèle n'est déployable

**Le run en cours (`exec25`) n'a pas encore d'epoch entraînée concluante.** Le
meilleur résultat hors échantillon jamais obtenu reste **TabM à +3.8 ± 2.3
points** au-dessus du point mort, PF 1.18, 3 folds sur 3 — non significatif. La
meilleure politique PPO, `exec18`, donne **+2.2 ± 2.2 points**, 2 folds sur 3.

Aucun `.pth` ne doit être mis en production. `kairos_live.py` refuse d'ailleurs
de trader sans fichier `_calib.json` correspondant (`min_confidence = None`),
et il tourne **encore sur l'ancien pipeline M1** : il n'a pas été porté au M5
ni aux 260 colonnes. Le porter est un chantier ouvert, pas un détail.

### Ce que le projet a compris de lui-même

Le mur n'a jamais été l'architecture ni l'algorithme. Six expériences
d'architecture ont rendu nul ou négatif (SAINT contre PatchTST, attention entre
groupes −2.3, méta-étiquetage −2.9, lookback nul, taille par conviction +0.3).
Ce qui a déplacé des chiffres : l'**échelle de temps**, l'**historique**, les
**features**, et la **taille du modèle** — réduire le réseau de 493 796 à
15 680 paramètres a fait passer PPO de −1.4 à +2.2 en test.

La grandeur qui commande tout est le nombre d'**occasions indépendantes** :
la durée d'historique divisée par la durée d'un trade, **jamais** le nombre de
barres. Il en faut ~4 900 pour qu'un avantage de 0.02 R soit lisible.

| configuration | friction / R | horizon | occasions | avantage requis |
|---|---|---|---|---|
| H1, SL 2×ATR | 0.047 R | 13.0 h | 2 314 | +0.0233 R |
| M5, SL 4×ATR | 0.099 R | 3.7 h | 15 089 | +0.0892 R |
| **M5, SL 8×ATR** | **0.049 R** | **12.2 h** | **4 516** | **+0.0425 R** |

Le `SL 8×ATR` en M5 domine le H1 sur tous les axes : même friction, même
horizon — donc le même problème de prédiction, celui sur lequel l'avantage de
+0.09 à +0.13 R a effectivement été mesuré — pour deux fois plus d'occasions.

### Lignées de checkpoints — elles ne sont PAS interchangeables

Un préfixe par jeu d'observation. Charger le mauvais fichier ne produit aucune
erreur visible : le modèle trade, simplement il lit autre chose que ce sur quoi
il a appris. Depuis `exec23`, `build_policy` déduit l'architecture complète des
poids et le reste du fichier `_calib.json` ; un fichier ambigu est refusé.

| Préfixe | Échelle | Observation |
|---------|---------|-------------|
| `..._exec10` à `..._exec20` | H1 | 103 colonnes, contexte H4 |
| `..._exec22` | M5 | 103 colonnes, contexte H1 |
| `..._exec23`, `..._exec24` | M5 | 260 colonnes, M5 + H1 + H4, SL 4×ATR |
| `..._exec25` | M5 | idem, **SL 8×ATR / TP 16×ATR** ← courant |

### Les trois modèles votent PENDANT l'entraînement

Le vote n'existait qu'à l'évaluation, sur des réseaux entraînés chacun à agir
**seul**. Un modèle qui apprend à décider tout seul, puis qu'on met dans un
comité, n'a jamais appris à jouer sa part dans une décision commune.

| votant | nature | ce qu'il apporte |
|---|---|---|
| **SAINT** | PPO, attention sur les features | ne fait que **croiser** les colonnes |
| **PatchTST** | PPO, canaux indépendants | ne les croise **jamais** |
| **TabM** | supervisé, MLP ensemble | un veto, pas une proposition |

Les deux réseaux PPO sont opposés par construction sur la question qui sépare
le mieux les modèles de ce dépôt. Des erreurs corrélées ne s'annulent pas :
c'est la condition pour qu'un vote réduise la variance.

**Pourquoi un mélange et pas trois entraînements séparés.** Si on entraîne
séparément puis qu'on fait voter, l'action exécutée ne vient d'**aucune** des
politiques mises à jour : le rapport de PPO perd son sens et réclame un poids
d'importance non borné. En présentant le mélange comme une seule politique,
l'action est tirée de ce qui est mis à jour — PPO reste exact, et le gradient
atteint chaque membre par sa part dans la probabilité de l'action choisie.
`PolitiqueEnsemble` moyenne les **probabilités**, une voix par membre, et non
les logits qui laisseraient un membre très confiant écraser les autres.

**TabM entre par une autre porte.** Il n'a pas de gradient dans cette boucle,
et ses scores dépendent de la **barre** et non de la seule observation. Il
passe donc par le **masque d'actions** : il ne propose rien, il interdit. C'est
la fonction du Chikou-Span dans le système Ichimoku — un signal n'est pas pris
si autre chose le contredit.

Son ajustement est **croisé** : le rollout se joue sur la fenêtre même qui
servirait à l'ajuster, où il serait un **oracle**. Les deux réseaux
apprendraient à lui déférer, et en production le partenaire deviendrait
ordinaire. Chaque barre reçoit donc le score d'un TabM qui ne l'a pas vue —
trois blocs **contigus**, parce que des lignes voisines partagent leurs barres
de résultat. Coût mesuré : 12 s par fold, 100 % des barres notées.

> Le veto change le **plafond d'entropie**. Quand il interdit une direction, la
> politique choisit entre deux actions et non trois : le maximum vaut ln 2 et
> non ln 3. Le plafond attendu est `p²·ln3 + 2p(1-p)·ln2`, soit **0.621** à
> p = 0.50. Lire une entropie de 0.573 contre ln 3 ferait croire à une
> politique fortement différenciée alors qu'elle décide au hasard.

### Le checkpoint se résout, il ne s'écrit pas en dur

`checkpoints.py` rend le run le plus récent — au **numéro d'exec**, pas à la
date du fichier qu'un `touch` fausserait — dont l'observation correspond au
pipeline courant. `kairos_live` pointait sur `exec11`, dix-sept runs en
arrière, et rien ne l'aurait signalé : un chemin périmé ne lève pas d'erreur
tant que le fichier existe.

Deux garde-fous qui ont servi immédiatement : sans le filtre de compatibilité,
« le plus récent complet » rendait `exec20` et ses 107 colonnes contre 264 au
pipeline ; et un champ `n_features` absent ne vaut **pas** autorisation, sans
quoi `exec13` passait aussi.

La famille par défaut est **`last`**, la moyenne des poids. Demander « le
meilleur » rend quand même la moyenne, en disant pourquoi : choisir un
checkpoint sur son résultat de validation coûte −3.3 points mesurés ici. Le
« meilleur » checkpoint est le plus mauvais déployable.

### Le vrai sujet ouvert

Une régression logistique atteint **0.6271 d'AUC** sur ces colonnes. Aucune
politique entraînée n'a dépassé **0.5707**. Cet écart est le fait le plus
reproductible et le moins expliqué du projet : le signal est dans les features,
PPO n'arrive pas à le prendre.

Et la validation **sélectionne à l'envers**, de façon reproductible : −3.3
points pour le choix du checkpoint, −4 points pour la sélectivité TabM, et le
fold 2 d'exec18 affichait +8.14 en validation pour −0.9 en test. C'est pourquoi
le choix du meilleur checkpoint a été remplacé par la **moyenne des dix
derniers jeux de poids**, qui ne dépend d'aucun tirage particulier.

### Le live et l'entraînement calculent les mêmes colonnes — et c'est vérifié

Jusqu'au 2026-09-16 ils passaient par **deux chemins différents** : `merge_m1_h1`
d'un côté, `prepare_m5.construit` de l'autre. Deux implémentations de la même
chose, qui devaient s'accorder par convention et que rien ne vérifiait. Elles
avaient cessé de s'accorder : le live était resté en M1, lookback 96, stop
2×ATR, checkpoints `exec11`, pendant que l'entraînement passait au M5, 260
colonnes, lookback 4 et 8×ATR. Chargé tel quel, le modèle n'aurait levé aucune
erreur — il aurait tradé, en lisant autre chose que ce sur quoi il a appris.

Il n'y a plus qu'un chemin. `flux_live.py` récupère les bougies M5 au format
exact de `klines_5m_spot_BTCUSDT.pkl`, et `kairos_live` les passe au **même**
`prepare_m5.construit` que le jeu d'entraînement.

`test_alignement.py` le vérifie plutôt que de le supposer : il récupère des
bougies **par le chemin du live**, sur une fenêtre finissant dans le jeu, et
compare les 260 colonnes valeur par valeur. Il contrôle aussi les réglages —
stop, objectif, lookback, risque par trade — de tout fichier qui les **recopie**.

> Les bougies viennent de **Binance, pas de MT5**. Quatre des 260 colonnes —
> part acheteuse agressive, taille moyenne de trade, intensité — n'existent pas
> dans un flux CFD : MT5 ne publie ni `taker_buy_base`, ni `quote_vol`, ni
> `nb_trades`. MT5 ne sert plus qu'à **exécuter**, et l'écart de prix entre les
> deux est exactement ce que la friction représente.

> La **dernière bougie est toujours jetée** : l'API rend celle en formation en
> dernière ligne, et la garder injecterait cinq minutes de futur à chaque
> décision, sans qu'aucune erreur ne soit levée.

### Les backtests : un seul moteur, jamais deux

`evalue_test_exhaustif.py`, `evalue_ensemble.py` et `stress_test.py`
construisent tous leur environnement depuis `training.PPOConfig` et
`BTCTradingEnvDiscrete`. Ils suivent donc l'entraînement **par construction** :
il n'y a rien à maintenir en double.

`stress_test.py` remplace `backtest_saintv2_stress_test.py`, qui
réimplémentait l'environnement au-dessus de MT5 — sa propre lecture des
bougies, son propre moteur d'ordres, ses propres réglages recopiés à la main.
Ils avaient dérivé jusqu'à l'absurde : instrument **XAUUSD**, stop de 5×ATR
annoté « optimum mesuré sur l'or », lookback 25. Le nouveau ne réimplémente
rien : il prend le vrai moteur et dégrade ce qu'il consomme — spread doublé,
glissement triplé, mèches étendues, ATR faussé de 10 %, micro-gaps, pics de
news, puis tout ensemble. Il tourne sur la **validation** par défaut ; la
fenêtre de test ne sert qu'une fois, et `--test` le dit en clair avant de
partir.

### Divergence restante entre l'entraînement et le live

`kairos_live.py` n'a **aucun chemin de fermeture au marché** — les positions ne
sortent que par SL/TP chez le courtier. Tant que ce n'est pas écrit, toute règle
de sortie temporelle côté entraînement simulerait une stratégie inexécutable ;
c'est pourquoi `max_holding_bars` vaut 0, et pourquoi toutes les mesures de
géométrie se font **sans plafond de durée**.

### Ce qui a été retiré, et où le retrouver

Le 2026-09-16, 31 scripts et 5 rapports datés ont été supprimés, ainsi que
865 Mo de données. Tout ce qui portait sur l'**échelle de décision M1** — le
cache de 627 Mo et les seize scripts qui le lisaient — est parti avec elle :
même à un stop de 8×ATR le M1 exige +0.114 R d'avantage, davantage que tout ce
que ce dépôt a mesuré. Sont partis aussi le *basis* (aucun écart au-dessus du
bruit de phase), le flux à la minute (`taker_1m_ma5` n'est plus dans
`FEATURE_COLS`), la taille par conviction (+0.3), et le doublon exact
`data_cache_BTCUSD_H1_complet.pkl`.

Les fichiers de code et de documentation étaient suivis par git : `git log --
<fichier>` puis `git show <commit>^:<fichier>` les rend intacts. Les `.pkl` ne
l'étaient pas, mais chacun se régénère depuis son script de téléchargement.

### Où est écrit ce qu'on a appris

`JOURNAL_MESURES.md`, en ordre antichronologique. Chaque entrée dit ce qui a
été mesuré, comment, et ce que la mesure **ne** permet pas de conclure. La
plupart des impasses de ce projet ont été des raisonnements plausibles jamais
confrontés à une mesure.

---

## 📑 Table des matières

1. [Vue d'ensemble](#-vue-densemble)
2. [Architecture du modèle](#-architecture-du-modèle)
3. [Pipeline complet](#-pipeline-complet)
   - [Volume dynamique](#-volume-dynamique-position-sizing)
   - [Mode multi-agent](#-mode-multi-agent-wf1--wf2--wf3-en-parallèle)
   - [Spread et déclenchement SL/TP](#-spread-et-déclenchement-sltp)
   - [Gestion de marge](#-gestion-de-marge-anti-reject-no_money)
4. [Méthodologie d'entraînement](#-méthodologie-dentraînement)
5. [Installation](#-installation)
6. [Configuration de MetaTrader 5](#-configuration-de-metatrader-5)
7. [Utilisation](#-utilisation)
8. [Structure du projet](#-structure-du-projet)
9. [Espace d'action et masque](#-espace-daction-et-masque)
10. [Reward shaping et risk management](#-reward-shaping-et-risk-management)
11. [Walk-forward](#-walk-forward)
12. [Lecture des logs](#-lecture-des-logs)
13. [Backtest stress-test](#-backtest-stress-test)
14. [GUI live](#-gui-live)
15. [Fichiers générés](#-fichiers-générés)
16. [Troubleshooting](#-troubleshooting)
17. [Roadmap](#-roadmap)
18. [Références scientifiques](#-références-scientifiques)
19. [Disclaimer](#-disclaimer)

> 📓 **`JOURNAL_MESURES.md`** — ce qui a été mesuré, comment, et ce que chaque
> mesure ne permet pas de conclure. À lire avant de proposer une feature, un
> hyperparamètre ou une « évidence » : la plupart des impasses du projet ont été
> des raisonnements plausibles jamais confrontés à une mesure.

---

## 🎯 Vue d'ensemble

**KAIROS** est un système complet de trading algorithmique BTCUSD M1 (1 minute) basé sur l'apprentissage par renforcement profond. Il combine :

- **PPO** (Proximal Policy Optimization) — algorithme on-policy stable de référence pour le contrôle
- **SAINTv2** — transformer dual-axis (row + column attention) reconnu pour la modélisation de séries financières tabulaires
- **Walk-forward** institutionnel en 3 folds → robustesse temporelle
- **Stress-test V3** institutionnel pour valider sans illusion (slippage, news spikes, gaps, trous)
- **MetaTrader 5** pour les données historiques et l'exécution live
- **GUI PySide6** avec logs colorés et stats LONG/SHORT temps réel

Le système est conçu pour fonctionner **24/7** sur cryptos (BTCUSD), avec gestion automatique du SL initial (ATR-based), break-even, et trailing stop.

### Caractéristiques principales

| Aspect | Valeur |
|--------|--------|
| Marché | BTCUSD (crypto, 24/7) |
| Timeframe principal | M1 |
| Contexte multi-TF | M1 + H1 (concat features) |
| Algorithme RL | PPO clippé + GAE λ |
| Backbone | SAINTv2 (RowAttn + ColAttn + GatedFFN) |
| Espace d'action | **3 actions** : `BUY` / `SELL` / `HOLD` |
| Période d'entraînement | 2022-12-15 → 2026-09 (~3.7 ans, 1.83 M bougies M1 après filtrage) |
| Méthodologie | Walk-forward 3 folds (55/15/10) |
| Backbone size | d_model=80, 2 blocks, 4 heads — 407 780 paramètres |
| Features | 12 M1 + 13 H1 + 2 Binance + 3 liquidité/temps + 4 position = **34** |
| Lookback | 25 bougies M1 |

---

## 🏗️ Architecture du modèle

### SAINTv2 (Self-Attention and Intersample Attention Transformer v2)

Architecture transformer **dual-axis** spécialement adaptée aux données tabulaires temporelles, inspirée du papier _SAINT_ (Somepalli et al. 2021) avec amélioration v2 : gated FFN à la PaLM/Gemma + double row-attention par block.

```
Input (B, T=25, F=34)
   │
   ├─► Linear projection ──► (B, T, F, d_model=80)
   ├─► + Row embedding  (temporel)
   └─► + Col embedding  (feature)
              │
              ▼
   ┌──── Block × 2 ──────────────────────────────┐
   │   Attention axe T  (+ RoPE, QK-Norm)        │
   │      ↳ chaque feature regarde son histoire  │
   │   SwiGLU                                    │
   │   Attention axe F  (QK-Norm)                │
   │      ↳ chaque instant mixe ses colonnes     │
   │   SwiGLU                                    │
   │                                             │
   │   pre-RMSNorm + LayerScale sur les 4        │
   └─────────────────────────────────────────────┘
              │
              │   h = moyenne sur l'axe FEATURE ──► (B, T, d)
              │
              ├──► moyenne sur T  ──► contexte global  (B, d)
              └──► dernier pas    ──► instant courant  (B, d)
                          │
                    concat ──► (B, 2d)
                          │
                    LayerNorm(2d)
                          │
                   MLP 160 → 256 → 256
                       /        \
                  actor          critic
                  (B, 3)         (B, 1)
```

### Composants, et la recherche derrière chacun

| Composant | Référence | Pourquoi |
|---|---|---|
| pre-norm **RMSNorm** | Xiong 2020 ; Zhang & Sennrich 2019 | blocs profonds entraînables sans warmup ; le recentrage ne sert à rien après une projection linéaire |
| **QK-Norm** | Henry 2020 ; ViT-22B 2023 | borne les logits d'attention — la perte du critique oscillait entre 24 et 47 |
| **SDPA / FlashAttention** | Dao 2022 | attention exacte sans matérialiser la matrice T×T ; c'est ce qui paie la structure complète |
| **RoPE** (axe temps seul) | Su 2021 | position relative par rotation. Pas sur l'axe features : les colonnes n'ont pas d'ordre naturel |
| **SwiGLU** | Shazeer 2020 | largeur interne à 2/3 pour garder le même compte de paramètres |
| **LayerScale** 1e-4 | Touvron 2021 (CaiT) | le réseau démarre proche de l'identité et ouvre les branches utiles |
| plongement **périodique** | Gorishniy 2022 | un scalaire projeté n'occupe qu'une direction ; les fréquences apprises séparent des valeurs proches |
| jeton **CLS** | Gorishniy 2021 (FT-Transformer) | remplace la moyenne, qui pèse toutes les colonnes également |
| init acteur **gain 0.01** | Engstrom 2020 | sur PPO ce détail pèse plus que la plupart des choix algorithmiques |

### L'intersample attention, sous une forme déployable

C'est l'innovation qui donne son nom à SAINT : chaque ligne regarde **les autres
lignes du lot**. Prise au pied de la lettre, elle n'est pas exécutable ici.

En production l'agent décide sur **une** observation : le lot vaut 1. Un softmax
sur un seul élément rend 1, donc l'opération dégénère en simple projection, et
les poids appris sous un lot de 128 se comporteraient autrement en live.
Fabriquer un faux lot côté live ne réglerait rien — il ne ressemblerait pas à
celui de l'entraînement, et l'écart reviendrait dans l'autre sens.

**La correction** ([`ReferenceMemory`](saint_core.py)) : les « autres
échantillons » ne sont pas le lot courant mais une **banque de K observations
réelles**, tirée une fois de la fenêtre de *train* et rangée dans le checkpoint.
Le modèle compare l'instant présent à une bibliothèque de situations
historiques — ce que l'axe temporel ne donne pas, lui qui ne voit que les
25 dernières bougies. Parenté : Gorishniy et al. 2023, *TabR: Tabular Deep
Learning Meets Nearest Neighbors*.

Trois propriétés que cette forme conserve et que la version littérale perd :

- **Indépendance au lot.** La banque est identique pour tous les échantillons,
  donc rien ne circule *entre* les lignes du lot. Vérifié : un lot de 8 donne
  exactement 8 passes de 1, **écart 5.6e-09 avec la mémoire active**.
- **Identité entraînement / live.** Banque et encodages voyagent avec les poids ;
  `build_policy` **déduit** la taille de la banque du checkpoint au lieu de la
  supposer. Aller-retour disque vérifié à 0.0e+00.
- **Absence de fuite.** La banque vient du *train* : en validation comme en test,
  le modèle consulte des situations antérieures à la période évaluée — de la
  connaissance apprise, au même titre que les poids.

Coût : les encodages sont mis en cache et rafraîchis une fois par epoch. La
requête n'est qu'une attention croisée depuis un vecteur vers K clés —
**mesuré nul** (56.6 ms contre 60.5 ms sans mémoire, à K=256).

### Coût mesuré

Banc entrelacé, GPU bloqué à 210 MHz sur les six tours (donc même état thermique
pour toutes les variantes), forward+backward à batch 128 :

```
ancien (ra,ca,ff) x2        298 596 params   279.55 ms   1.00x
complet x2                  407 780 params   666.41 ms   2.38x
complet x3                  509 620 params  1078.59 ms   3.86x
complet x4                  611 460 params  1463.37 ms   5.23x
```

**bf16 ne rattrape rien** : 211.8 ms contre 186.0 en fp32 sur la même variante.
À cette taille le coût est dans les lancements de noyaux et la mémoire, pas dans
le calcul tensoriel, et l'autocast ajoute des conversions. Aucun gradient non
fini en bf16 — l'ancien problème d'AMP était bien spécifique à fp16.

> **Le pooling a été corrigé.** L'ancienne version moyennait le tenseur sur
> chacun des deux axes puis re-moyennait : les deux « CLS » étaient
> mathématiquement identiques (écart mesuré 4.5e-08). La moitié de la tête
> lisait donc la même chose. On concatène désormais le contexte global et le
> dernier pas de temps, qui portent une information différente.

**Pourquoi SAINTv2** : sur des données financières (M1 OHLC + indicateurs + HTF), la corrélation temporelle ET inter-features est cruciale. Les transformers classiques attaquent une seule des deux axes ; SAINTv2 mixe les deux dans chaque bloc → meilleure modélisation des patterns complexes (rejet de support, retest, divergence RSI vs prix, etc.).

### Référence
- [SAINT (Somepalli et al., NeurIPS 2021)](https://arxiv.org/abs/2106.01342) — papier original
- [Tabular Transformers benchmark](https://arxiv.org/abs/2207.08815) — SAINTv2 montre 2-5 % d'AUC en plus vs SAINT v1 sur la majorité des tâches tabulaires

### PPO (Proximal Policy Optimization)

Implémentation maison avec :
- **Clipped surrogate objective** (ratio ε = 0.18)
- **GAE semi-MDP** : bootstrap en `γ^Δt` (γ=0.995 par minute, λ=0.95). Le pas de
  décision n'est pas constant — seule une fraction des pas en position est
  conservée — donc un `γ` par transition fausserait l'actualisation.
- **Critic warmup** : 5 epochs où seul le critic apprend. **Conséquence pour
  lire les logs** : sur ces 5 epochs l'actor est gelé (`H = 1.099 = ln 3`,
  `clipfrac 0 %`). Toute variation de résultat y vient du curriculum et du
  tirage du seuil, jamais d'un apprentissage.
- **KL early stop** : interruption d'une epoch si KL > 0.03
- **Cosine LR scheduling** : 3e-4 → 1.5e-5 sur 240 epochs
- **Plafond de décisions** (16 000/epoch) : sans lui le nombre de décisions
  variait d'un facteur 17 d'une epoch à l'autre, rendant les durées et les
  quantités de gradient incomparables.

> **Reward normalization retirée.** L'estimateur de Welford initialisait sa
> variance à 1.0 puis la ramenait à 0 au premier échantillon : écart-type 1e-8,
> récompenses divisées par presque rien.

### Référence
- [PPO paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [GAE (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438)

---

## 🧬 Jeu de features — 260 colonnes, trois échelles

`saint_core.FEATURE_COLS` est la **source unique**. L'observation ajoute
4 scalaires de position → `OBS_N_FEATURES = 264`.

| bloc | colonnes | contenu |
|---|---|---|
| M5 | 103 | 12 bases, 2 Binance, 4 liquidité/temps, 25 range, 47 Ichimoku, 13 contexte |
| H1 | 85 | les mêmes structures, resamplées et décalées d'un `shift(1)` |
| H4 | 85 | idem |

Les périodes Ichimoku sont des nombres de **bougies**, pas des durées : Tenkan 9
couvre 45 minutes en M5, 9 heures en H1, 36 heures en H4. Les trois blocs
portent les mêmes noms et ne décrivent pas du tout la même chose — un trade
médian durant 12 heures, le M5 décrit ce qui se passe *pendant* le trade, le H1
le mouvement qui le contient, le H4 le régime qui contient ce mouvement.

`test_causalite.py` recalcule les 260 colonnes sur une série tronquée et exige
la valeur identique. Il affiche le nombre de colonnes **réellement évaluées** :
une marge d'échauffement trop courte rendrait `NaN` des deux côtés, ce qui
compte comme un accord et déclarerait saines des colonnes jamais calculées.

> ⚠️ L'élagage à 10 colonnes documenté dans les versions précédentes de ce
> fichier a été **annulé après mesure** : les apports marginaux NE SE COMPOSENT
> PAS. Retirer 9 features individuellement « nulles » coûtait 0.0024 d'AUC, soit
> plus que la meilleure feature du jeu n'en apporte.

### Ce que pèse chaque source — mesuré, pas supposé

Sonde logistique, barrières SL 2.0×ATR / R:R 1.4, friction complète, train
0–55 % / validation 55–70 %, fenêtre de test intouchée.

| Opération sur le jeu de 30 | AUC | Écart |
|---|---|---|
| référence | **0.6185** | — |
| sans `taker_ratio` | 0.5680 | **−0.0498** |
| sans `ls_ratio_top` | 0.6177 | −0.0000 |
| + `taker_1m_ma5` | **0.6271** | **+0.0087** |

`taker_ratio` porte à elle seule presque tout l'apport externe. `ls_ratio_top`
n'apportait **rien** et a été remplacée par `taker_1m_ma5` — même compte de
features, donc même coût GPU.

### Le critère qui sépare les features utiles des inertes

**Une feature doit varier à l'échelle où la décision se prend.** La détention
médiane d'un trade est de **7 barres** ; une série constante sur cette durée ne
peut pas départager deux entrées espacées de quelques minutes.

| Colonne | Autocorr. 1 min | Apport |
|---|---|---|
| `taker_1m_ma5` | 0.86 | **+0.0087** |
| `taker_ratio` (5 min) | 0.82 | **−0.0498 si retirée** |
| `oi_change` | 0.989 | +0.0001 |
| `funding_rate` | 0.9996 | −0.0000 |
| `ls_ratio_top` | 0.99998 | −0.0000 |
| `ls_ratio_retail` | 0.99998 | −0.0014 |

Les quatre colonnes Binance jamais branchées donnent **0.4847 d'AUC à elles
seules** — sous le hasard. Le funding change toutes les 10 heures : c'est une
variable de *régime*, pas de *timing*.

Le balayage de la fenêtre de lissage du flux 1 minute confirme le mécanisme —
bosse régulière à sommet unique, retour à zéro quand la série devient un régime :

```
1min +0.0016 | 3min +0.0067 | 5min +0.0087 | 10min +0.0050
15min +0.0027 | 30min +0.0017 | 60min +0.0004 | 120min -0.0000
```

> Ces mesures datent de l'échelle M1, et leurs scripts ont été retirés le
> 2026-09-16 avec le cache qu'ils lisaient — `git log` les rend, et leurs
> conclusions sont dans `JOURNAL_MESURES.md`. Les outils encore en service sont
> `mesure_court_terme.py`, `mesure_features_ichimoku.py`,
> `mesure_lookback_utile.py` et les `evalue_*.py`. Chacun porte dans son
> en-tête le biais de sélection qu'il subit — on lit le meilleur d'une grille
> sur la même fenêtre de validation, donc les chiffres sont des bornes
> optimistes.

**MetaTrader ne peut fournir aucune de ces grandeurs** : son `tick_volume`
compte les changements de prix, pas les montants échangés, et il n'a ni flux
taker, ni open interest, ni funding.

Le **carnet d'ordres a été écarté après mesure** : les archives `bookDepth` ne
descendent pas sous le palier ±1 %, alors que l'endpoint live `/fapi/v1/depth`
(plafonné à `limit=1000`) ne porte que jusqu'à ±0.17 % du mid. Aucun
recouvrement — la feature serait entraînable mais incalculable en live.

L'historique d'entraînement démarre au **2022-12-15**, borne où la couverture
Binance atteint 100 % sur toutes les colonnes (avant, `ls_ratio_top` a ~260
jours de trous en 2022). Coût : 2.28 M → 1.97 M bougies.

> ⏰ **Alignement horaire** — Binance horodate en UTC, le serveur MT5 tourne en
> UTC+2/+3 selon l'heure d'été, avec le calendrier DST **américain**. Le
> décalage est mesuré empiriquement (corrélation des rendements M1 : 0.99 au bon
> décalage, 0.00 partout ailleurs) puis confronté au calendrier — accord 239/239.
> Un offset fixe aurait décalé un tiers de l'historique d'une heure entière,
> sans la moindre erreur visible.

---

## 🔄 Pipeline complet

```
┌──────────────────────────────────────────────────────────────────┐
│           0. DONNÉES BINANCE (build_binance_features.py)         │
│                                                                  │
│  data.binance.vision `metrics` + REST /fundingRate  →            │
│  détection du décalage broker/UTC  →  grille M1 heure broker     │
│                                                                  │
│  Génère : binance_features_BTCUSD.pkl  (lu par training et       │
│           les 3 backtests ; cache brut dans .cache_binance/)     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    1. TRAINING (training.py)                     │
│                                                                  │
│  MT5 (BTCUSD M1+H1) + Binance  →  Indicateurs  →                 │
│  Normalisation Z-score globale  →  Walk-forward 3 folds  →       │
│  PPO + SAINTv2 240 epochs  →  Best checkpoints sauvegardés       │
│                                                                  │
│  Génère : bestprofit_saintv2_loup_duel_both_wfN.pth              │
│           best_saintv2_loup_duel_wfN_both_wfN.pth                │
│           last_saintv2_loup_duel_wfN_both_wfN.pth                │
│           training_log_both_wfN.csv                              │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│           2. VALIDATION (stress_test.py)                        │
│                                                                  │
│  Charge bestprofit_* → Simule sur OOS data avec                  │
│  stress-test V3 (slippage ±20bps, gaps, news spikes,             │
│  randomisation ATR/TP/SL, trous 1-3min)                          │
│                                                                  │
│  Génère : Verdict ROBUSTE / ACCEPTABLE / NON RENTABLE            │
│           PF, WR, DD, Sortino, Score                             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│             3. LIVE TRADING (kairos_live.py + kairos_gui.py)         │
│                                                                  │
│  GUI PySide6 → TradingAgent thread → MT5 polling 1s →            │
│  Détection nouvelle bougie M1 fermée → Construction obs →        │
│  Inférence policy → argmax → ordre BUY/SELL ou HOLD →            │
│  Tick-by-tick trailing/break-even sur SL                         │
│                                                                  │
│  Affiche : Stats LONG/SHORT temps réel, logs colorés filtrables  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🎓 Méthodologie d'entraînement

### Innovations clés vs PPO standard

1. **Reward normalization Welford online** — élimine les explosions de CriticL au début du training (de l'ordre de 1e+10 sans, ~1-10 avec)
2. **Cosine LR + KL early stop combinés** — empêche les mises à jour destructives en fin de training
3. **Curriculum learning avec biais SHORT** — sur marché bullish 2022-2026, force le modèle à explorer la direction perdante au début pour ne pas converger en long-only
4. **Critic warmup 5 epochs** — laisse le baseline V(s) se stabiliser avant d'optimiser l'actor (sinon l'actor "court après" un critic non calibré)
5. **Gradient clipping serré (norm=0.3) + unscale AMP fix** — empêche les gradient explosions ; clip appliqué AVANT scaler.unscale_ pour AMP fp16 correct
6. **Reward shaping multi-composant** — combine PnL réalisé + bonus trade gagnant + pénalité bad entry + slippage micro
7. **Garde-fous anti-NaN** — clamp logits ±30, clamp log_ratio ±10, skip-batch si NaN/Inf détecté, entropy floor ×5 si H<0.1
8. **Clip normalisation Z-score ±5σ** — aligne training / backtest / live → robustesse aux outliers de marché
9. **max_drawdown=0.4** — terminaison anticipée si DD > 40% → force le modèle à apprendre la prudence
10. **Validation 7 épisodes** (au lieu de 2) — stabilise le Sortino30 utilisé pour la sélection du best

### Hyperparamètres clés

Valeurs en vigueur sur `exec25`. Les justifications complètes, avec la mesure
qui a fait choisir chaque valeur, sont dans les commentaires de `training.py`.

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `timeframe_entrainement` | `M5` | échelle de décision ; H1 et H4 en contexte |
| `atr_sl_mult` | 8.0 | friction 0.049 R et horizon 12.2 h — ceux du H1, pour 2× ses occasions |
| `atr_tp_mult` | 16.0 | R:R 2.0 |
| `epochs` | 40 | + une itération de moyenne des poids |
| `episodes_per_epoch` | 336 | 483 840 barres/epoch = 73 % de la fenêtre, ~2 000 décisions |
| `episode_length` | 1 440 | 5 jours de M5, ~10 trades médians par épisode |
| `val_episodes` | 40 | couvre 100 % de la fenêtre de validation |
| `updates_per_epoch` | 8 | doubler les passes sans collecter une barre de plus |
| `batch_size` | 128 | deux fois plus de pas de gradient à données égales |
| `lr` | 1e-3 | la KL croît comme le carré du pas ; à 3e-4 elle valait 0.0002 pour une cible de 0.030 |
| `entropy_coef` | 0.015 | le terme de politique est 40 % plus faible en M5, celui-ci est absolu |
| `max_grad_norm` | 0.6 | la norme observée vaut 0.90 : à 0.3 chaque mise à jour était divisée par trois |
| `clip_eps` | 0.18 | standard PPO |
| `target_kl` | 0.03 | arrêt précoce à 1.5× — rend sûr d'augmenter le pas |
| `gamma` | 0.9999 | semi-MDP : l'escompte est `γ^Δt`, Δt en barres. À 0.995 il coûtait 20 % du R:R — voir ci-dessous |
| `lambda_gae` | 0.95 | standard |
| `critic_warmup_epochs` | 5 | l'actor est gelé, ces epochs servent de référence au hasard |
| `n_moyenne_poids` | 10 | remplace le choix du meilleur checkpoint, qui coûtait −3.3 points |
| `patience` | 0 | pas d'arrêt précoce : il sélectionnerait sur la validation |
| `lookback` | 4 | mesuré : le passé n'apporte rien au-delà |
| `architecture` | `ensemble` | SAINT **et** PatchTST dans le même rollout, présentés comme une seule politique |
| `membres` | saint, patchtst | opposés par construction : l'un ne fait que croiser les colonnes, l'autre ne les croise jamais |
| `votant_tabm` | True | troisième votant, supervisé, qui oppose son veto via le masque d'actions |
| `d_model` / `num_blocks` | 8 / 2 | ~45 000 paramètres pour ~4 500 occasions |
| `saint_heads` / `saint_n_freq` | 1 / 4 | `d_model // heads` doit être multiple de 8 |
| `saint_mlp_dim` | 16 | la tête pèse 92–98 % du réseau ; c'est elle qu'il faut contenir |
| `saint_lecture` | `colonnes` | lit chaque colonne, au lieu d'un CLS agrégé |
| `max_drawdown` | 0.4 | force l'apprentissage prudent |
| `tick_noise_bps` | 3.0 | extension des wicks. **Doit rester ≪ `atr_sl_mult` × ATR**, sinon le bruit déclenche le SL avant le marché |

#### Pourquoi gamma est passé de 0.995 à 0.9999

`gamma` est exprimé **par barre**, et la récompense — une plus-value latente
payée à chaque barre — est accumulée en `gamma^dt`. Deux changements ont modifié
son sens sans que personne n'y touche : H1 → M5, puis SL 4 → 8×ATR.

Mesure sur 11 971 courses résolues à SL 8×ATR / R:R 2.0 : **un gain dure
254 barres en médiane, une perte 146** — 1.74 fois plus, mécaniquement, le
take-profit étant deux fois plus loin que le stop. L'escompte frappe donc les
gains plus fort que les pertes.

| gamma | poids gain | poids perte | R:R effectif | |
|---|---|---|---|---|
| 0.995 | 0.567 | 0.711 | 1.56 | **−20.2 %** |
| 0.999 | 0.883 | 0.931 | 1.86 | −5.1 % |
| **0.9999** | 0.987 | 0.993 | **1.95** | −0.5 % |

À 0.995 le point mort **vu par l'agent** montait à 39.1 % quand l'environnement
en applique 33.8 % : on lui demandait 5.3 points de winrate de trop, et on le
poussait à fuir les configurations lentes à se résoudre — c'est-à-dire les
gagnantes. En H1 le réglage était sain (13 barres, `0.995^13 = 0.937`, moins de
2 % de distorsion) : ce n'était pas la valeur qui était fausse, c'est qu'elle
n'a pas suivi l'échelle.

---

## 💾 Installation

### Pré-requis

- **OS** : Windows 10/11 (MT5 requis)
- **Python** : 3.10 ou 3.11
- **GPU NVIDIA** (recommandé) avec CUDA 12.4 drivers
- **MetaTrader 5** terminal installé
- **Compte broker** avec accès BTCUSD M1 (Vantage, IC Markets, etc.)
- **8 Go RAM minimum**, 16 Go recommandés
- **GPU 6 Go VRAM minimum** (RTX 3060 ou +)

### Installation étape par étape

```powershell
# 1. Cloner le repo
git clone https://github.com/SimonMonnier/multi-agent-btcusd.git
cd multi-agent-btcusd

# 2. Créer un venv Python isolé
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Installer les dépendances (PyTorch CUDA 12.4 inclus)
pip install -r requirements.txt

# 4. Vérifier CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Variante CPU-only (pour tests / fallback)

Si tu n'as pas de GPU :

```powershell
pip install torch==2.5.1 numpy pandas MetaTrader5 gymnasium PySide6 colorama
```

Le training tournera **5-10× plus lentement**.

---

## ⚙️ Configuration de MetaTrader 5

### 1. Installation du terminal

Télécharge MT5 depuis le site officiel de ton broker. **Aucun MT5 générique** ne fonctionnera : il faut un broker avec BTCUSD accessible (typiquement Vantage, IC Markets, Pepperstone, FTMO).

### 2. Activer l'historique long M1

**Indispensable** pour le training :

- Ouvre MT5
- **Outils → Options → Graphiques**
  - "Max bars in chart" → **Sans limite** (ou 9 999 999)
  - "Max bars in history" → **Sans limite**
- Ouvre un graphique BTCUSD M1
- Appuie `Home` puis `Page Up` plusieurs fois → MT5 télécharge l'historique manquant depuis le serveur (peut prendre 1-2 min)
- Vérifie en bas du graphique : la date doit remonter à 2022 ou plus tôt

### 3. Activer le trading algorithmique

- Bouton **Algo Trading** en haut → doit être **vert**
- **Outils → Options → Conseillers Experts** → cocher :
  - ✅ Autoriser le trading algorithmique
  - ✅ Autoriser DLL imports
  - ✅ Désactiver les confirmations manuelles

### 4. Connexion broker

- **Fichier → Connexion à un compte de trading**
- Renseigne login, password, serveur
- Vérifie en bas de l'écran : la latence (ping) doit être < 100 ms

### 5. Vérifier l'API Python

```powershell
python -c "import MetaTrader5 as mt5; print(mt5.initialize()); print(mt5.account_info())"
```

Si `True` et un AccountInfo → tout est OK.

---

## 🚀 Utilisation

### 1. Entraîner le modèle

```powershell
python training.py
```

**Durée estimée** :
- GPU RTX 3060+ : ~12-20 heures pour 240 epochs × 3 folds
- GPU haut de gamme : ~6-12 heures
- CPU only : 5-10× plus long, à éviter

**Sortie** : logs colorés, CSV par fold, checkpoints `.pth`.

Pour relancer en partant des checkpoints existants : `training.py` détecte automatiquement les `.pth` présents et les recharge.

### 2. Backtest stress-test

```powershell
python stress_test.py
```

Par défaut backteste `bestprofit_saintv2_loup_duel_wf1_both_wf1.pth` en mode `side="both"` avec `min_confidence=0.0` (argmax pur). Modifie `LiveConfig(...)` à la fin du fichier pour tester :
- `side="both"` : modèle duel unifié
- `side="long"` / `"short"` : modèles spécialisés
- `side="duel"` : 2 modèles séparés en arbitrage
- `min_confidence=0.0` (argmax pur) ou ajuster selon le calibrage du modèle

### Variante `no_be_trail` (sans break-even / trailing)

```powershell
python stress_test.py
```

**Fichier identique** à `stress_test.py` mais avec l'appel à
`update_sl_be_trailing_backtest()` **commenté** (ligne ~980). Les positions
ne ferment qu'au SL ou TP fixe.

Génère `backtest_trades_both_no_be_trail.csv` (n'écrase pas l'original).

> ⚠️ **Résultats historiques invalidés.** Les chiffres ci-dessous ont été produits
> avant la correction de la fuite H1 : le modèle disposait alors du rendement
> complet de l'heure en cours, directement exploitable sur un TP à 1.68 × ATR(M1).
> Ils sont conservés pour mémoire, pas comme référence.

| Variante | PnL | WR | PF | DDmax | Verdict annoncé |
|----------|-----|----|----|-------|-----------------|
| Avec BE/trail | −499 $ | 46.7 % | 0.98 | — | ✗ NON RENTABLE |
| Sans BE/trail | +5 609 $ | 58.5 % | 1.85 | 4.4 % | ✓ ROBUSTE |

Le constat qualitatif sur le BE/trailing (il coupait les gagnants avant le TP)
reste plausible et le live comme le MQL5 s'en passent. En revanche l'écart chiffré
doit être remesuré après réentraînement.

Deux autres réserves méthodologiques sur ces chiffres :
- la fenêtre configurée dans le code est de **27 jours** (2026-03-04 → 03-31), pas 75 ;
- les checkpoints WF portent l'empreinte de WF3 via le bootstrap `auto_chain` du
  `__main__`, ce qui affaiblit l'indépendance walk-forward revendiquée.

Le **live applique cette leçon** : `update_sl_be_trailing_live()` est commenté
dans `kairos_live.py`.

### 3. Live trading (GUI)

```powershell
python kairos_gui.py
```

- Choisis ta taille de lot
- Clique **▶️ Démarrer l'IA**
- Le bot tournera en boucle, attendant chaque nouvelle bougie M1 fermée
- **⏹️ Arrêter l'IA** pour stopper (les positions ouvertes restent ouvertes, gérées par leur SL/TP)

### 4. Live sans GUI (CLI)

```powershell
python kairos_live.py
```

### 5. Export vers MT5 Strategy Tester (MQL5)

```powershell
python export_to_onnx.py
python export_binance_for_mql5.py
```

`AGENT` en tête du premier script pilote **à la fois** le checkpoint lu et le nom
du `.onnx` produit (`wf1` → `bestprofit_..._wf1_both_wf1.pth` → `saintv2_wf1.onnx`).
Ne jamais dissocier les deux : une version antérieure exportait les poids WF2
sous le nom `saintv2_wf3.onnx`, et l'EA « WF3 » tournait donc sur le mauvais modèle.

Copier ensuite dans `<MT5 Common>\Files\` : `saintv2_wf1.onnx`,
`norm_stats_mean.bin`, `norm_stats_std.bin`, puis régler `ModelFile` et `Magic`
dans les inputs de l'EA. Le quatrième fichier, `binance_BTCUSD.bin`, est déposé
directement au bon endroit par `export_binance_for_mql5.py`.

> 🌐 **Pourquoi Binance passe par un fichier et non par `WebRequest()`**
> `WebRequest()` est **totalement désactivé dans le Strategy Tester** : l'EA ne
> peut pas interroger Binance pendant un backtest. Les deux features sont donc
> pré-exportées dans un binaire (grille strictement minute en heure broker →
> indexation en O(1)), relu en tester **comme en live**, pour que le
> comportement soit identique dans les deux cas. L'EA refuse de démarrer si le
> fichier est absent ou si son nombre de colonnes ne correspond pas à celui
> qu'il attend — deux des dix features en dépendent.

> L'alignement H1 de l'EA est **volontairement décalé d'un bar** : `closes_h1[]`
> est rempli depuis `CopyClose(..., shift=1, ...)` tandis que `iBarShift()` compte
> depuis le bar courant. Le résultat correspond exactement au `df_h1.shift(1)`
> côté Python. Ne pas « corriger » cet écart apparent.

---

## 📁 Structure du projet

```
multi-agent-btcusd/
├── saint_core.py                    # ⭐ Noyau partagé : modèle, features, normalisation,
│                                    #    fusion M1/H1, masque, SL/TP, volume dynamique.
│                                    #    Ne JAMAIS redupliquer ces fonctions ailleurs :
│                                    #    c'est la duplication qui avait laissé diverger
│                                    #    l'alignement H1, le bruit et le levier.
├── build_binance_features.py        # ⭐ Récupère Binance (funding, metrics 5 min) et
│                                    #    résout l'alignement horaire broker/UTC (DST
│                                    #    américain). À lancer AVANT le training.
│                                    #    Seule source variant à l'échelle de la décision.
├── export_binance_for_mql5.py       # .pkl → binaire lu par l'EA (WebRequest est
│                                    #    désactivé dans le Strategy Tester)
├── training.py                      # Entraînement PPO + SAINTv2 walk-forward
├── kairos_live.py                   # Agent live MT5 (single + multi-agent + marge)
├── kairos_gui.py                    # Interface graphique PySide6 (présentation seule)
├── kairos_theme.py                  # Palette, typographie, feuille de style QSS
├── kairos_widgets.py                # Jauge winrate, sparkline, témoin d'état, tuiles
├── suivi_live.ps1                   # Fenêtre de suivi du training en direct (lecture seule)
│                                    #    ⚠️ voir le piège os.kill documenté dedans
│
├── (mesures — chacune documente son propre biais de sélection)
├── mesure_features.py               # Constantes de friction, partagées par les mesures
├── mesure_court_terme.py            # Occasions et friction par géométrie, sans sortie au temps
├── mesure_features_ichimoku.py      # Apport des 47 colonnes Ichimoku
├── mesure_lookback_utile.py         # Profondeur de passé réellement utilisée
│
├── stress_test.py                   # Degrade l'execution dans le VRAI moteur
├── flux_live.py                     # Bougies M5 en direct, format du jeu
├── test_alignement.py               # Le live calcule-t-il les memes colonnes ?
├── export_to_onnx.py                # .pth → .onnx + stats binaires pour MT5
├── SaintV2_WF3.mq5                  # EA MQL5 pour le Strategy Tester
├── requirements.txt                 # Dépendances Python
├── .gitignore                       # Exclusions git (.pth, .csv, .npz)
├── README.md                        # Ce fichier
│
├── _archive_21features/             # Ancien pipeline (21 features + ticks) :
│                                    #    checkpoints et caches.
│                                    #    Conservé pour traçabilité, plus alimenté.
│
├── (générés avant training)
├── binance_features_BTCUSD.pkl                                  # Features Binance, heure broker
├── binance_flux_1m_BTCUSD.pkl                                   # Flux 1 min, index UTC
├── data_cache_BTCUSD_20221215.pkl                               # Frame M1+H1+externe fusionnée
├── .cache_binance/                                              # Archives brutes UTC + table
│                                                                #    d'offsets DST (re-alignement
│                                                                #    sans retéléchargement)
│
├── (générés après training)
├── norm_stats_ohlc_indics.npz                                   # Stats Z-score globales
├── best_saintv2_loup_duel_exec5_wf1_both_wf1.pth                # Best Sortino30
├── bestprofit_saintv2_loup_duel_exec5_wf1_both_wf1.pth          # Best ValPNL par trade
├── *_calib.json                                                 # Seuils calibrés — SANS eux,
│                                                                #    le live refuse de trader
├── *_norm.npz                                                   # Stats de normalisation du modèle
├── run_saintv2_loup_duel_exec5_*.json                           # Manifeste : refuse d'écraser
│                                                                #    un run existant
├── runs_archives/                                               # Journaux et manifestes des runs
│                                                                #    précédents (exec3, exec4…)
├── training_log_both_wf{1,2,3}.csv                              # Logs CSV epoch-level
├── trades_both_wf{1,2,3}.csv                                    # Trade-by-trade
└── backtest_trades_both.csv                                     # Trades du backtest
```

---

## 🎮 Espace d'action et masque

L'agent dispose de **3 actions** :

| Index | Action | Description |
|-------|--------|-------------|
| `0` | `BUY` | Ouvre une position LONG avec SL+TP ATR-based |
| `1` | `SELL` | Ouvre une position SHORT |
| `2` | `HOLD` | Ne fait rien |

### Masque dynamique

Le masque dépend de la **position courante** et du **mode side** :

| Position | side="both" | side="long" | side="short" |
|----------|-------------|-------------|--------------|
| Flat (0) | BUY, SELL, HOLD | BUY, HOLD | SELL, HOLD |
| LONG (+1) | HOLD only | HOLD only | HOLD only |
| SHORT (-1) | HOLD only | HOLD only | HOLD only |

**Quand l'agent est en position, il ne peut que HOLD.** La fermeture est gérée par :
- SL touché → fermeture automatique
- TP touché → fermeture automatique

> ⚠️ **Break-even et trailing stop DÉSACTIVÉS en backtest `no_be_trail` et en live**
> (cf section [Backtest stress-test](#-backtest-stress-test)). Le backtest a montré
> que le BE/trailing actuels (triggers 1.0 / 1.5 ATR) coupent les wins trop tôt :
> PF 0.98 avec → **PF 1.70 sans**. Le training les utilise toujours dans l'env,
> mais le live et le backtest "no_be_trail" laissent les positions courir
> jusqu'au SL ou TP fixe.
> Les fonctions `update_sl_be_trailing_*` restent définies dans le code,
> simplement leur appel est commenté.

### Sélection avec seuil de confiance

| Contexte | Seuil | Note |
|----------|-------|------|
| **Training** (`CONF_THRESHOLD`) | 0.40 | Filtre exploration : HOLD si `softmax(BUY ou SELL) < 0.40` |
| **Backtest** (`min_confidence`) | 0.40 | 0.0 = argmax pur si tu veux mesurer la policy brute |
| **Live** (`min_confidence`) | 0.40 | Idem |
| **MQL5** (`MinConfidence`) | 0.40 | Input de l'EA |

**Note importante** : avec 3 actions équiprobables (1/3 ≈ 0.33), un modèle bien calibré plafonne souvent autour de 0.40-0.50 en max-prob. Un seuil > 0.50 peut bloquer toutes les entrées. Vérifier les logs `probas D BUY=X SELL=Y HOLD=Z` du backtest pour calibrer.

---

## 📐 Dimensionnement des positions — par le RISQUE

Trois modes existent, et l'ordre de priorité compte : `risk_volume` l'emporte
sur `dynamic_volume`, qui l'emporte sur `position_size`. **`risk_volume = True`
est le mode par défaut et le seul aligné sur l'entraînement.**

```python
size = capital × risk_per_trade / sl_dist
```

Le lot n'est donc **pas connu à l'avance** : il dépend de la distance au stop,
qui dépend de l'ATR à l'instant de l'entrée. Ce qui est fixe, c'est le budget de
risque — 1,2 % de l'equity par trade, `risk_per_trade` devant rester identique
entre `training.PPOConfig` et `LiveConfig` : le modèle doit trader le risque
sous lequel il a appris.

> ⚠️ **Le défaut que ça a corrigé.** Le volume était auparavant constant à 0.06
> lot et `risk_per_trade` n'était jamais lu. Comme la valeur du point diffère
> d'un symbole à l'autre, le risque réel variait d'un **facteur 108** entre
> BTCUSD et XAUUSD pour un même réglage.

### Pièges d'exécution MT5

- `symbol_info().trade_tick_value` vaut **0.0** tant que le symbole n'est pas
  dans le Market Watch. Il faut appeler `symbol_select(symbol, True)` **avant**,
  sinon le volume calculé est nul et l'ordre est annulé sans explication.
- Quand le volume minimum du courtier dépasse le volume voulu, le risque imposé
  est **supérieur** à la consigne. `compute_risk_volume` renvoie un drapeau
  `plancher` et le log l'affiche en clair : c'est le seul cas où le contrôle du
  risque échoue, il ne doit pas passer inaperçu.

### Mode par paliers d'equity (secondaire)

Utilisé seulement si `risk_volume = False`. Lot de base 0.01 sous 2 000 $, puis
+0.01 par tranche de 1 000 $, plafonné à `max_lot`.

```python
# Dans LiveConfig (kairos_live.py / backtest_*.py)
risk_volume: bool = True        # ⭐ mode par défaut
risk_per_trade: float = 0.012   # identique à training.PPOConfig
dynamic_volume: bool = True     # secondaire : paliers d'equity
max_lot: float = 100.0          # cap absolu
```

> Les projections de compounding qui figuraient ici (« +25 000 $ ») reposaient
> sur un backtest dont la formule de PnL contenait un facteur `× leverage`
> erroné, surestimant d'un facteur 6. Elles ont été retirées plutôt que
> recalculées : aucun modèle n'est actuellement rentable.

---

## 🤖 Mode multi-agent (wf1 + wf2 + wf3 en parallèle)

Les **3 checkpoints walk-forward** peuvent désormais trader simultanément sur le même compte. Chaque agent :
- voit les **mêmes données** d'entrée
- prend ses **décisions indépendamment**
- peut ouvrir **sa propre position** (max 1 par agent ⇒ jusqu'à 3 positions ouvertes en même temps)
- est identifié par un **magic MT5 dédié** :

| Agent | Magic | Checkpoint |
|-------|-------|------------|
| WF1 | 424241 | `bestprofit_saintv2_loup_duel_wf1_both_wf1.pth` |
| WF2 | 424242 | `bestprofit_saintv2_loup_duel_wf2_both_wf2.pth` |
| WF3 | 424243 | `bestprofit_saintv2_loup_duel_wf3_both_wf3.pth` |

### Configuration

```python
# LiveConfig dans kairos_live.py
multi_agent: bool = True   # active le mode 3-agents
```

Le `TradingAgent` route automatiquement vers `live_loop_multi()` quand `multi_agent=True`. Le mode classique single-agent reste disponible (`multi_agent=False`).

### Backtest multi-agent

Un fichier dédié reproduit la même logique :

```powershell
python stress_test.py
```

Affichage par-agent dans les logs (couleur cyan/jaune/magenta) + résumé final qui détaille les stats de chaque agent séparément. CSV exporté : `backtest_trades_multi_agent_no_be_trail.csv` avec colonne `agent`.

### Equity et volume partagés

Capital et equity sont **communs** aux 3 agents (un seul compte). Le volume est recalculé par le risque à chaque ouverture, sur l'equity du moment. Conséquence : si les 3 agents s'ouvrent en même temps, le risque cumulé vaut **3 × `risk_per_trade`**, soit 3,6 % de l'equity — et non 1,2 %. Le pré-check de marge borne l'exposition mais pas le risque.

---

## 🎯 Spread et déclenchement SL/TP

MT5 déclenche le SL/TP **au prix de clôture** (BID pour un LONG, ASK pour un SHORT), pas au prix d'entrée. Le spread fait donc que le TP demande un peu plus de mouvement et le SL un peu moins.

### Choix retenu : ne PAS décaler les niveaux

`saint_core.compute_sl_tp()` pose les niveaux purs (`1.2 × ATR` / `1.68 × ATR`), sans compensation de spread — identique en training, backtest, live et MQL5.

L'asymétrie BID/ASK est modélisée **au déclenchement**, là où elle se produit réellement : la boucle de backtest teste `low <= sl + s` / `high >= tp + s` (`s` = spread échantillonné à l'entrée). Décaler les niveaux eux-mêmes aurait fait diverger le mouvement de prix requis entre ce que le modèle apprend et ce qu'il rencontre à l'exécution.

Une version antérieure appliquait `sl_dist + spread` / `tp_dist - spread` dans le seul `kairos_live.py` ; elle a été retirée pour cette raison. Le paramètre `spread` a disparu de la signature.

### Poids réel du spread

Mesuré sur entrées aléatoires (SL = 1.2 × ATR ≈ 54 $ pour un ATR médian de 45 $) :

| Friction active | % SL touchés | % morts en ≤ 1 bougie |
|-----------------|-------------:|----------------------:|
| Tout (bruit 3 bps + slippage + spread) | 74.7 % | 25.3 % |
| Sans slippage d'entrée | 69.6 % | 14.1 % |
| Sans spread | 55.3 % | 2.4 % |
| Sans aucune friction | 52.8 % | 0.0 % |

Baseline théorique d'une entrée aléatoire : ~58 % de SL. Le spread bimodal (`spread_wide_prob=0.30`, `spread_bps_wide_factor=5.0`) est donc désormais la friction dominante — à recalibrer sur les spreads réellement observés chez le broker si les trades meurent encore trop vite.

---

## 🛡️ Gestion de marge (anti-reject NO_MONEY)

Pour éviter que Vantage rejette des ordres quand la marge libre est insuffisante (notamment avec lot 50-100), le live applique 3 lignes de défense :

### 1. Pré-check `adjust_volume_to_margin()`
Avant chaque `order_send`, on compare la **marge requise** (`mt5.order_calc_margin`) avec `margin_free × safety_factor` (80 % par défaut).
Si insuffisant : volume réduit linéairement, arrondi au `volume_step` du broker.

### 2. Arrondi au `volume_step` broker
Le volume ajusté est aligné au pas du broker (0.01 sur BTCUSD).

### 3. Retry automatique sur retcode 10019 (`NO_MONEY`)
Si MT5 rejette quand même : le volume est divisé par 2 et retry, jusqu'à 4 tentatives ou volume sous `min_volume`.

### Configuration

```python
# LiveConfig
margin_safety: float = 0.80              # n'utilise jamais > 80% margin_free
auto_scale_volume_to_margin: bool = True # active le pré-check
min_volume: float = 0.01                 # plancher d'ordre
```

### Logs typiques

**Cas scale-down** :
```
[VOL/WF3] equity=50000.00$ → lot=5.00
  [WF3] ↘ SCALE-DOWN volume 5.00 → 2.30 (margin_free=29500$, budget 80%=23600$, margin/lot=10260$)
```

**Cas marge insuffisante** :
```
  [WF1] ⚠ MARGE INSUFFISANTE : margin_free=300.00$ × safety=80% = 240.00$ ; required pour 5.00 lot = 51300$ → ORDRE ANNULÉ
```

**Cas retry NO_MONEY** :
```
  [WF1] ↘ NO_MONEY retry #1 : volume 10.00 → 5.00
  [WF1] ↘ NO_MONEY retry #2 : volume 5.00 → 2.50
Order exécuté [WF1] : side=1, vol=2.50, ...
```

---

## 💰 Reward shaping et risk management

### Reward composite

À chaque step (1 bougie M1) :

```
reward = pnl_step / initial_capital * scale
       + bonus_realized_trade   (× 1.8 si gain ; × -1.0 si perte)
       - penalty_micro_slippage
       - penalty_idle_in_position  (si >100 bars sans toucher SL/TP)
       + reward_for_correct_side   (si direction alignée avec micro-mouvement)
```

Tout ça est ensuite **normalisé via Welford online** avant d'être utilisé par le critic, ce qui stabilise massivement l'apprentissage.

### SL / TP dynamiques

```
ATR(14) → dernier ATR clôturé, plancher à 0.15% du prix
SL = entry_price ∓ 1.20 × ATR    (signe selon LONG/SHORT)
TP = entry_price ± 1.68 × ATR    (atr_tp_mult × tp_shrink, tp_shrink = 1.0)

→ R:R = 1:1.4  ⇒  winrate d'équilibre ≈ 42%
```

### Break-even + Trailing

```
Si mouvement favorable ≥ 1.0 × ATR :
   SL déplacé à entry_price (break-even, position garantie 0)

Si mouvement favorable ≥ 1.5 × ATR :
   SL trailé à 1.0 × ATR derrière le prix max favorable
   (s'améliore à chaque nouveau high/low favorable)
```

**État par contexte** :
| Contexte | BE/Trail actif ? |
|----------|------------------|
| Training (env) | ❌ Non — `PPOConfig.use_be_trail = False` |
| Backtest stress-test | ✅ Oui |
| Backtest **no_be_trail** | ❌ Non — variante de comparaison |
| **Live (kairos_live.py)** | ❌ **Non** |
| MQL5 `SaintV2_WF3.mq5` | ❌ Non |

---

## 🔁 Walk-forward

### Principe

Le walk-forward est la **gold standard** de validation temporelle pour les modèles financiers. Plutôt qu'un single train/val/test split, on découpe la data en plusieurs fenêtres glissantes :

```
Total bars : 2 199 868 (≈ 4 ans de M1)

Fold 1 (window 80%) :
  ├── TRAIN ─────────────────┤ ├── VAL ──┤ ├── TEST ─┤
  [══════ 55% ═══════════════] [══ 15% ══] [══ 10% ══]
  2022-01                    2024-04    2024-12    2025-05

Fold 2 (décalé de +10%) :
        ├── TRAIN ─────────────────┤ ├── VAL ──┤ ├── TEST ─┤
        [══════ 55% ═══════════════] [══ 15% ══] [══ 10% ══]
        2022-06                    2024-09    2025-05    2025-11

Fold 3 (décalé de +20%) :
              ├── TRAIN ─────────────────┤ ├── VAL ──┤ ├── TEST ─┤
              [══════ 55% ═══════════════] [══ 15% ══] [══ 10% ══]
              2022-11                    2025-02    2025-10    2026-05
```

### Garanties

- **Pas de fuite temporelle intra-barre** : les features H1 sont décalées d'un bar (`df_h1.shift(1)` dans `saint_core.merge_m1_h1`), donc seul le dernier H1 **clôturé** entre dans l'observation. Sans ce décalage, `merge_asof(backward)` renvoyait le bar H1 en formation, dont les valeurs historiques sont déjà finalisées — jusqu'à 59 minutes de futur.
- **Découpage temporel** : VAL et TEST sont toujours dans le futur du TRAIN du même fold
- **Couverture 100%** : l'union des 3 folds couvre toute la data
- **Régimes diversifiés** : chaque fold contient un mix bull/bear/sideways différent
- **Test du wf3 ≈ test out-of-sample récent** : wf3 termine en 2026-05, le plus pertinent pour le live

### Sélection du modèle live

`LiveConfig.active_agents` liste les agents à charger. **N'y mettre que des agents
dont le `.pth` existe** : le chargement lève volontairement une `FileNotFoundError`
plutôt que de tourner à vide.

```python
active_agents = ["wf1"]            # défaut actuel — seul checkpoint présent
active_agents = ["wf1", "wf3"]     # ensembling 2 agents
active_agents = None               # les 3 (exige les 3 fichiers)
```

Une fois les 3 folds réentraînés, préférer le plus récent (`wf3`), dont la fenêtre
de train est la plus proche du marché courant, ou exécuter plusieurs agents en
parallèle : chacun a son magic MT5 (424241 / 424242 / 424243) et sa propre position.

---

## 📊 Lecture des logs

### Logs training (par epoch)

```
[BOTH_wf1] EPOCH 042  TRAIN  PNL  +542.30$  trades=87  WR 41.3%  PF 1.23  DD 18.5%  L(15W/12L) +234.50$  S(21W/15L) +307.80$
[BOTH_wf1] EPOCH 042  VAL    PNL +1234.50$  trades=142 WR 43.2%  PF 1.45  DD 22.0%  L(28W/19L) +680.20$  S(33W/22L) +554.30$
[BOTH_wf1] EPOCH 042  META   Sortino +0.221  Sortino30 +0.052  AvgW +52.30$  AvgL -31.40$  ActorL +0.0042  CriticL 1.523  H 0.387  KL +0.0481  ENV [B 28.5% S 31.2% H 40.3%]
```

| Champ | Signification |
|-------|---------------|
| `TRAIN` / `VAL` | Métriques sur le set respectif |
| `PNL` | Profit total $ du fold |
| `trades` | Nombre de trades exécutés |
| `WR` | Winrate (% trades gagnants) |
| `PF` | Profit factor (gains / pertes en valeur absolue) |
| `DD` | Drawdown maximum (% equity) |
| `L(WW/LL)` | Wins / Losses **LONG** + PNL agrégé LONG |
| `S(WW/LL)` | Wins / Losses **SHORT** + PNL agrégé SHORT |
| `Sortino` | Sortino ratio de l'epoch |
| `Sortino30` | **Métrique de sélection** : moyenne Sortino sur 30 dernières epochs |
| `AvgW` / `AvgL` | Gain / perte moyenne par trade |
| `ActorL` / `CriticL` | Loss PPO actor / critic |
| `H` | Entropy de la policy (haut = exploration) |
| `KL` | KL divergence de l'update |
| `ENV [B X% S Y% H Z%]` | Ratio actions choisies (BUY / SELL / HOLD) |

### Événements

```
★ NEW BEST PROFIT  ValPNL +1234.50$  trades=142    # Nouveau best en ValPNL
★ NEW BEST SORTINO30  Sortino30=+0.052  trades=142 # Nouveau best en Sortino30
⚠ early-stop KL  KL=0.0612                        # Update interrompu (KL trop élevé)
```

### Couleurs (terminal moderne ou Windows Terminal)

- 🟣 Tag side `[BOTH_wf1]`
- 🟦 Epoch number
- 🟢 PnL positif / wins / B ratio
- 🔴 PnL négatif / losses / DD critique
- 🔵 SELL ratio / S short label
- 🟡 / 🟢 / 🔴 Sortino30 et DD selon valeur

---

## 🛡️ Backtest stress-test

Le backtest n'est **pas** un simple replay des données. Il applique 7 sources de stress pour simuler un environnement réaliste :

| Stress | Paramètre | Impact |
|--------|-----------|--------|
| Slippage random | ±20 bps | Prix d'exécution réel ≠ prix théorique |
| Tick noise | std 5 bps | Bruit OHLC gaussien |
| Spread horaire | ×0.8 à ×2 | Plus large la nuit / heures creuses |
| Micro-gaps | 0.05% bougies | Saut de prix ±5 bps |
| News spikes | 0.05% bougies | Élargissement violent 0.2-1% |
| ATR distortion | ±10% | Erreur d'estimation SL/TP |
| TP/SL random | ±10% | Variations placement |
| Trous 1-3 min | 0.05% | Saut de bougies (data outage) |

### Verdict automatique

Le backtest émet un verdict à la fin :

| Condition | Verdict |
|-----------|---------|
| PnL > 0 ET PF ≥ 1.2 ET WR ≥ 40% | ✓ ROBUSTE |
| PnL > 0 (mais critères pas tous remplis) | ~ ACCEPTABLE |
| PnL ≤ 0 | ✗ NON RENTABLE |

---

## 🖥️ GUI live

L'interface graphique offre :

La présentation vit dans `kairos_theme.py` (palette, QSS) et `kairos_widgets.py`
(widgets peints à la main). `kairos_gui.py` ne contient que l'assemblage et la
logique MT5 — une retouche esthétique ne touche jamais au code de trading.

### Barre haute
- **Témoin d'état** : un point qui pulse quand l'agent tourne. Seul élément animé
  de l'interface, donc l'œil y revient tout seul.
- **Symbole et side**, lus depuis `cfg` — le titre de la fenêtre aussi. Une
  constante écrite en dur avait fini par annoncer XAUUSD sur un agent BTC.
- **Dimensionnement** : en mode `risk_volume` le lot n'existe pas à l'avance
  (il dépend de la distance au stop à l'entrée), donc l'interface affiche le
  **budget de risque en dollars**, qui est la grandeur réellement fixe.
- **Démarrer / Arrêter**

### Bandeau de mesures
Equity, **P/L réalisé**, **P/L flottant** (séparés : les additionner donnait deux
tuiles avec le même nombre en début de session), nombre de trades, et une
**jauge de winrate marquant le point mort à 43,7 %** — l'arc passe au vert
au-dessus, ambre en dessous. Un winrate brut ne veut rien dire sans ce repère.

Sous le bandeau, une **courbe d'equity** de session : pas d'axes, elle ne sert
qu'à montrer la forme (pente, décrochages, plateau).

### Stats de session
Grille temps réel avec 3 lignes :
- 🟢 **LONG** : trades, wins, losses, WR, PF, PnL, AvgW, AvgL
- 🔵 **SHORT** : idem
- ⚪ **TOTAL** : agrégé

Source : `mt5.history_deals_get(session_start − 24h, now + 24h, group="BTCUSD")` filtré sur :
- `d.entry ∈ {DEAL_ENTRY_OUT, DEAL_ENTRY_INOUT, DEAL_ENTRY_OUT_BY}` (tous les types de sortie MT5)
- `d.magic ∈ {424241, 424242, 424243}` (les trois agents WF, ignore le manuel)
- Dédup par `d.ticket`

La fenêtre élargie ±24h absorbe le décalage timezone local ↔ serveur MT5. **Bouton Réinitialiser** pour repartir de zéro.

### Zone de logs
- **Couleurs ANSI** par catégorie (TRADE / SIGNAL / INFO / WARN / ERROR / DEBUG)
- **Classification automatique** via regex (mots-clés `buy`, `sell`, `error`, etc.)
- **Filtre texte** en temps réel
- **Pastilles colorées** cliquables pour masquer/afficher chaque catégorie
- **Auto-scroll** désactivable
- **Effacer** + **Exporter** (`.txt` horodaté)
- **Ring buffer 5000 lignes** (anti-fuite mémoire sur sessions longues)
- **Compteurs** par niveau en bas

---

## 📦 Fichiers générés

| Fichier | Contenu | Quand |
|---------|---------|-------|
| `norm_stats_ohlc_indics.npz` | Mean/std features pour Z-score (auto-recompute si shape mismatch) | Au début du training |
| `best_saintv2_loup_duel_wfN_both_wfN.pth` | Best Sortino30 (≥20 trades) | Pendant training |
| `bestprofit_saintv2_loup_duel_wfN_both_wfN.pth` | Best ValPNL (≥20 trades) | Pendant training |
| `last_saintv2_loup_duel_wfN_both_wfN.pth` | Final epoch du fold | Fin du fold |
| `training_log_both_wfN.csv` | 40+ colonnes : PNL, trades, WR, L/S split, losses, etc. | À chaque epoch |
| `trades_both_wfN.csv` | 1 ligne par trade fermé (entry/exit/pnl/SL/TP/hold_bars/phase) | À chaque trade training |
| `backtest_trades_both.csv` | Trade-by-trade du backtest stress-test (avec BE/trail) | Fin du backtest |
| `backtest_trades_both_no_be_trail.csv` | Trade-by-trade variante sans BE/trail | Fin du backtest no_be_trail |
| `loup_log_YYYYMMDD_HHMMSS.txt` | Export GUI | Bouton 💾 |

### Format des poids

```python
import torch
state = torch.load("bestprofit_saintv2_loup_duel_both_wf3.pth", map_location="cpu")
# state est un OrderedDict contenant les poids du SAINTPolicySingleHead
```

---

## 🐛 Troubleshooting

### "MT5 n'a renvoyé aucune donnée M1"
1. Vérifie que MT5 est ouvert et connecté à un compte
2. Vérifie que BTCUSD est dans le Market Watch (clic droit → Afficher)
3. Charge l'historique : ouvre BTCUSD M1, `Home` puis `Page Up` plusieurs fois
4. Augmente "Max bars in history" dans Outils → Options → Graphiques

### "CUDA out of memory"
- Baisse `batch_size` à 128 ou 64 dans `PPOConfig`
- Ferme les autres apps GPU (jeux, navigateurs avec accélération)
- Vérifie ta VRAM : `nvidia-smi` dans PowerShell

### "size mismatch for actor.weight: [3, 256] vs [5, 256]"
- Tu charges un ancien `.pth` (5 actions) avec le nouveau code (3 actions)
- Solution : retrainer le modèle (le code actuel utilise 3 actions BUY/SELL/HOLD)

### "Modèle DUEL introuvable : bestprofit_saintv2_loup_duel_exec5_*.pth"
- Le path dans `kairos_live.py` / backtest doit correspondre au pattern réel :
  `bestprofit_saintv2_loup_duel_wfN_both_wfN.pth` (avec `_wfN_` au milieu)
- Vérifie les `.pth` réellement présents : `ls bestprofit_*.pth`

### Backtest : 0 trade pendant toute la simulation
- Le seuil `min_confidence` est trop élevé pour ce modèle
- Le modèle plafonne souvent vers 0.40-0.50 en max-prob → tout seuil > 0.50 bloque tout
- Solution : `min_confidence=0.0` (argmax pur) ou inspecter les probas dans les logs

### "[NORM] Mismatch features : X vs Y actuelles → recompute"
- Normal si tu changes la liste `FEATURE_COLS` après un premier training
- Le code recompute auto les stats Z-score ; aucune action requise

### Training trop long
- Diminue `epochs` à 120 ou 160 (le cosine LR s'adapte automatiquement)
- Diminue `episode_length` à 2000
- Diminue `episodes_per_epoch` à 2

### Pas de short en backtest / live
- Vérifie que le modèle chargé a bien été entraîné avec `side="both"` (et pas `side="long"`)
- Vérifie les logs `ENV [B X% S Y% H Z%]` du training : si S < 2% sur la majorité, le modèle est long-biased
- Solution : retrainer avec un curriculum encore plus biaisé SHORT (modifier `chosen_action = random.choice([1,1,1,1,1,0,0])`)

### GUI freeze
- Le `TradingAgent.start()` lance un thread daemon — il ne devrait pas freezer la GUI
- Si ça arrive : Ctrl+C dans la console + relancer
- Bouton 🗑️ Effacer si la zone de log devient trop chargée

---

## 🗺️ Roadmap

- [ ] **Multi-symbol** : étendre à ETHUSD / SOLUSD avec partage des couches transformer (multi-task learning)
- [ ] **Position sizing dynamique** : risk-parity sur volatilité réalisée
- [ ] **Meta-learner d'ensemble** : combiner wf1+wf2+wf3 via un router neuronal
- [ ] **Régime detector** : module auxiliaire (HMM ou TCN) qui informe la policy du régime (bull / bear / range)
- [ ] **Distillation** : compresser le SAINTv2 en MLP pour inférence < 1ms
- [ ] **Fermeture au marché côté live** — prérequis à toute règle de sortie
      temporelle dans l'entraînement (voir « Divergence connue » en tête)
- [ ] **Ablation de gamma** — 0.995 est un choix par raisonnement, jamais mesuré
- [ ] **Ablation de `scalping_max_holding`** — 120 → 30 lit nul à l'epoch 6,
      la question reste ouverte

### Écartés APRÈS MESURE — ne pas y revenir sans nouvelle mesure

- ~~**Funding rate awareness**~~ — mesuré : −0.0000 d'AUC. Change toutes les
  10 heures, donc constant sur un trade de 7 barres.
- ~~**Order book features**~~ — les archives `bookDepth` ne descendent pas sous
  ±1 % alors que l'endpoint live plafonne à ±0.17 % du mid. Aucun recouvrement :
  entraînable, mais incalculable en live.
- ~~**Open interest, ratios long/short**~~ — mesuré : 0.4847 d'AUC à eux seuls,
  sous le hasard.

---

## 📚 Références scientifiques

### Reinforcement Learning
- **PPO** : Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). _Proximal Policy Optimization Algorithms_. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- **GAE** : Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). _High-Dimensional Continuous Control Using Generalized Advantage Estimation_. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
- **Reward normalization** : Engstrom, L., et al. (2020). _Implementation Matters in Deep RL: A Case Study on PPO and TRPO_. ICLR 2020.

### Tabular Transformers
- **SAINT** : Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., & Goldstein, T. (2021). _SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-training_. [arXiv:2106.01342](https://arxiv.org/abs/2106.01342)
- **FT-Transformer** : Gorishniy, Y., et al. (2021). _Revisiting Deep Learning Models for Tabular Data_. NeurIPS 2021.

### Trading RL
- **DRL Trading** : Deng, Y., et al. (2017). _Deep Direct Reinforcement Learning for Financial Signal Representation and Trading_. IEEE TNNLS.
- **Walk-forward** : Bailey, D. H., et al. (2014). _The Probability of Backtest Overfitting_. Journal of Computational Finance.

### Implémentation
- **Cosine LR** : Loshchilov, I., & Hutter, F. (2017). _SGDR: Stochastic Gradient Descent with Warm Restarts_. ICLR 2017.
- **Gated FFN** : Shazeer, N. (2020). _GLU Variants Improve Transformer_. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)

---

## ⚠️ Disclaimer

> **Ce projet est fourni à titre éducatif et de recherche uniquement.**
>
> Le trading algorithmique sur cryptomonnaies présente un **risque substantiel de perte financière**, pouvant excéder votre dépôt initial avec l'effet de levier.
>
> - Les performances passées (training, backtest) **ne garantissent en rien** les performances futures
> - Le marché crypto est volatile, manipulable, et peut subir des événements de cygne noir (FTX, Luna, halving, régulation)
> - Aucune garantie n'est donnée sur la rentabilité, la stabilité ou la qualité du code
> - L'auteur **décline toute responsabilité** pour les pertes liées à l'utilisation de ce logiciel
> - **Commencez toujours en démo / paper trading** avant le compte réel
> - **N'investissez jamais plus que ce que vous pouvez vous permettre de perdre**
> - Vérifiez la **conformité réglementaire** dans votre juridiction (AMF en France, MiFID II en EU, etc.)
>
> En utilisant ce code, vous acceptez d'agir à vos propres risques.

---

## 🤝 Contribution

PRs bienvenues sur :
- Optimisations PyTorch (compile, channels-last, AMP)
- Nouvelles features (orderbook, on-chain data, sentiment)
- Tests unitaires (l'env Gym surtout)
- Compatibilité Linux (replacement MT5 par cTrader ou Binance API)
- Docs / tutoriels

Fork → branche feature → PR avec description claire et benchmark avant/après.

---

## 📄 Licence

Code privé, tous droits réservés. Contactez l'auteur pour usage commercial.

---

**KAIROS** — _PPO + SAINTv2 pour BTCUSD M1_ — Made with ❤️ and a lot of GPUs.
