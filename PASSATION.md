# Passation — KAIROS, état au 19 septembre 2026

> **Ce document ne contient aucun chiffre de performance, et ce n'est pas un
> oubli.** Aucun résultat de backtest ni de rentabilité n'est établi dans ce
> dépôt. Tous ceux qui y ont figuré avaient été produits avec au moins un
> défaut de harnais depuis identifié, et décrivaient donc une stratégie que
> personne n'a jouée. Ce qui suit est la méthode, l'architecture, les erreurs
> et ce qui reste ouvert.

---

## 1. Le projet en trois phrases

Agent d'apprentissage par renforcement (PPO) sur **XAUUSD**, entraîné en
walk-forward **chaîné** sur sept ans de bougies M5 issues de MetaTrader,
exécuté chez Vantage. Les positions sortent **uniquement sur barrières** —
jamais au temps, parce que le live n'a aucun chemin de fermeture au marché.
Le run est **long seul** et **or seul** ; le compte ouvre **plusieurs
positions simultanées**, bornées par ce que la marge permet.

**Aucun modèle n'est déployable à ce jour.** Rien n'a jamais produit un
avantage hors échantillon qu'on puisse défendre, et la fenêtre de test de la
lignée `or_` a déjà été consultée avec un harnais défectueux : elle n'est
plus vierge.

---

## 2. La contrainte qui commande tout

**Le nombre d'occasions INDÉPENDANTES.** Une occasion n'est indépendante de
la suivante que si leurs fenêtres de résultat ne se recouvrent pas. Leur
nombre vaut donc *durée de marché ÷ durée d'un trade* — **jamais** le nombre
de barres.

Sur la fenêtre de validation du fold 1 : 381 jours de calendrier, mais
**257 jours de marché** — l'or ne cote que 68 % du temps, cinq jours sur sept
avec une pause quotidienne. À une durée médiane de trade d'environ
vingt-cinq heures, cela fait de l'ordre de **250 occasions indépendantes**
pour juger un modèle.

Le diagnostic de classement en note quelques milliers, mais elles se
**recouvrent** : un point par heure sur des trades de vingt-cinq heures.
Vingt-cinq points consécutifs décrivent en grande partie le même trade.
C'est pourquoi l'incertitude du ρ se mesure par **bootstrap par blocs** et
non par un écart-type naïf.

La détectabilité se calcule avant de payer un run : l'avantage requis pour
qu'un effet soit lisible vaut approximativement la friction divisée par la
racine du nombre d'occasions. Une géométrie qui triple l'horizon divise les
occasions mais divise aussi la friction — le produit peut y gagner. **C'est
cette arithmétique qui doit décider d'une géométrie, pas un balayage de
rendements.**

---

## 3. Les règles de méthode, non négociables

1. **La fenêtre de test ne se lit qu'une fois.** Celle de la lignée `or_` a
   déjà été lue, et avec un harnais défectueux. Toute lecture ultérieure est
   une seconde lecture et doit être annoncée comme telle.

2. **Une barre d'erreur vient de ce qui varie indépendamment.** Des phases
   décalées de quelques dizaines de barres ne sont pas des échantillons
   indépendants. Le bootstrap par blocs est le minimum sur des séries
   temporelles.

3. **La part symétrique, (achat − vente)/2, avant toute conclusion
   directionnelle.** Elle annule la dérive du sous-jacent. Un chiffre positif
   sur un marché haussier ne dit rien tant qu'on ne l'a pas retranchée.

4. **La somme des R ment sous concurrence.** Additionner des R suppose des
   trades indépendants pris à 1 R chacun. Un optimum désigné par une somme de
   R s'est déjà effondré en équité continue. Le chemin de l'équité tranche,
   pas la somme.

5. **Deux descriptions du même objet finissent toujours par diverger.**
   Chaque fois qu'une valeur est recopiée au lieu d'être lue, elle a fini par
   mentir sans lever d'erreur. Une source, et un test qui le vérifie.

6. **Mesurer sur la validation avant de payer un run.** Toujours.

---

## 4. Les erreurs, et ce qu'elles coûtaient

Toutes du même type : rien ne lève, tout rend des nombres plausibles, et le
chiffre lu ne décrit pas ce qu'on croit.

### 4.1 Un run long-only qui vendait

L'entraînement respectait le masque d'actions. La validation, la calibration
et le test ouvraient des ventes : quand `tri_par_tete_aux` est vrai, la
sortie de la tête auxiliaire **écrase** le tableau de probabilités issu des
logits masqués, et le masque de côté disparaît avec lui. Le « meilleur
modèle » a été choisi des centaines de fois sur un résultat net qui
mélangeait des achats entraînés et des ventes qui ne l'étaient pas.

Corrigé dans la **règle** de décision, qui porte désormais le côté jusque
dans le checkpoint — et le côté est **imposé au rechargement**, sans quoi un
checkpoint antérieur retombe sur « both » et se remet à vendre.

### 4.2 Validation et test jouaient une autre stratégie

Ils exigeaient d'être **entièrement à plat** pour décider, quand
l'entraînement ouvre dès qu'une place est libre. Ils restaient assis les deux
tiers du temps. On entraînait une stratégie à plusieurs positions et on en
mesurait une à une seule — et le creux le disait sans qu'on l'écoute.

### 4.3 Le cumul comptait plusieurs fois la même mesure

La validation ne tournait qu'une époque sur trois ; les autres réaffichaient
la dernière mesure, marquée `[val epN]`. Ce marqueur était dans un groupe non
capturant, invisible à la veille, qui additionnait chaque époque au cumul du
run.

### 4.4 Le plafond d'entropie était celui de trois actions

En long-only la politique n'a que deux actions à plat : son maximum vaut
ln 2, pas ln 3. Une politique **à pile ou face** s'affichait à 63 % du
plafond, et le diagnostic « apprend mais reste plat » ne pouvait
structurellement jamais se déclencher.

### 4.5 Le live ouvrait une position et passait son tour

Et la quatrième colonne d'état — celle qui porte la capacité restante, donc
le cercle vertueux — y valait une constante au lieu de varier avec le solde.

### 4.6 Le résolveur de checkpoints ne voyait pas le côté

Il ne cherchait que des fichiers `*_both_*`. Aucun checkpoint d'un run
long-only n'existait pour lui, et il proposait le dernier run **bilatéral**.
Dans le même temps les chemins multi-agents pointaient en dur sur une lignée
BTC vieille de dix-sept runs.

### 4.7 Les constantes d'instrument vivaient en double

`instruments.py` était écrit pour être la source unique et n'était appelé par
personne ; les mêmes valeurs étaient recopiées dans `PPOConfig`. Elles
avaient déjà divergé d'un chiffre sur la marge, assez pour que
l'entraînement et le live ne comptent pas les mêmes places ouvrables.

### 4.8 Défauts de plomberie qui ne lèvent aucune erreur

`Tee-Object` verrouille le journal en accès exclusif — la veille ne pouvait
même pas l'ouvrir, et se taisait. `expandable_segments` n'existe pas sous
Windows et remplissait la console de rouge. PyTorch avait été remplacé par
sa build **CPU** sans que rien ne le signale. Le terminal MetaTrader
fraîchement installé plafonnait son historique, et rafraîchir le cache aurait
remplacé sept ans de données par trois semaines.

---

## 5. Ce qui est établi, et à quel titre

### 5.1 Le classement existe, la politique ne le produit pas

Une régression linéaire ordonne les occasions sensiblement mieux que le ρ de
la politique, qui oscille dans le bruit. PPO optimise le rendement de ses
**actions**, jamais l'**ordre** de ses probabilités — et c'est pourtant tout
ce dont la sélectivité se sert. C'est le fait le plus reproductible et le
moins expliqué du projet.

`tri_par_tete_aux` contourne le problème : c'est la tête auxiliaire, entraînée
à prédire le rendement, qui trie. Elle est entraînée sur le train et jugée sur
la validation, donc son ρ est une vraie mesure hors échantillon.

### 5.2 Le trailing est le seul mécanisme qui gagne

Sans lui, la part symétrique vaut exactement zéro à toutes les largeurs de
stop et chaque trade perd sa friction. Ce n'est pas un réglage d'appoint.

### 5.3 L'or est décorrélé du BTC

La corrélation entre les rendements de la stratégie sur les deux vaut 0,088.
Les cryptos entre elles corrèlent de 0,62 à 0,81. C'est ce qui justifie l'or
comme second instrument — et ce qui disqualifie les altcoins.

### 5.4 Ce qui ne se copie jamais d'un instrument à l'autre

La largeur du stop, la friction, et surtout la **taille du contrat**. Un lot
d'or vaut cent onces : son lot minimum représente plusieurs milliers de
dollars de notionnel là où celui du BTC en vaut quelques centaines. La marge,
elle, est du même ordre des deux côtés et ne borne jamais.

---

## 6. Configuration au moment du transfert

```
instrument XAUUSD SEUL — le BTC est sorti de l'entrainement
cote       LONG SEUL (PPOConfig.side, declare LA et nulle part ailleurs)
jeu        data_cache_XAUUSD_M5.pkl — 256 colonnes + 4 d'etat = 260 en entree
           trois echelles (M5 / H1 / H4), float32, source MetaTrader
           l'historique anterieur a 2019 est ECARTE : 127 barres par semaine
           contre 1 358, et un ATR relatif qui differe de 40 %
geometrie  SL 10xATR, AUCUN objectif, trailing 2.0 R
           cible d'apprentissage de la tete auxiliaire : 2 R
compte     budget de risque a ZERO : seule la marge du courtier borne
           PLUSIEURS POSITIONS SIMULTANEES, capacite calculee a chaque barre
           les tableaux d'emplacements doublent a la demande, sans plafond
modele     ensemble SAINT + PatchTST dans le MEME rollout
           episodes de 5 760 barres, plafond 160 par epoch, cible 4 000 decisions
           90 epochs, validation et calibration a CHAQUE epoch
           tete auxiliaire supervisee, et c'est ELLE qui trie
selection  le rendement du SOMMET — les occasions que le checkpoint mettrait
           en position — sous garde de classement positif ET de survie du
           portefeuille au garde-fou de creux
foldage    WALK-FORWARD CHAINE : wf2 herite du meilleur de wf1, wf3 de wf2
           normalisation figee sur le train du fold 1, elle ne voit que du passe
veto TabM  COUPE — mesure nulle, code conserve
run        or_exec06
```

**Les périodes Ichimoku sont des nombres de BOUGIES**, pas des durées. On ne
les convertit jamais d'une échelle à l'autre.

---

## 7. Fichiers qui comptent

| fichier | rôle |
|---|---|
| `saint_core.py` | source unique des colonnes, des architectures, de la règle de décision et de la capacité du compte |
| `training.py` | PPO, walk-forward chaîné, l'ensemble, la tête auxiliaire |
| `instruments.py` | **source unique** des constantes par instrument |
| `cibles.py` / `cibles_gpu.py` | la règle de trade, et sa seconde implémentation vérifiée au 1e-9 |
| `prepare_m5.py` / `prepare_or.py` | construction du jeu, trois échelles |
| `checkpoints.py` | quel modèle charger — filtre d'observation **et de côté** |
| `kairos_live.py` | exécution, alignée sur l'environnement d'entraînement |
| `veille_epochs.py` | lecture des epochs, en direct ou depuis le journal |
| `test_causalite.py` | aucune colonne ne doit changer quand on coupe le futur |
| `test_alignement.py` | le live calcule-t-il les mêmes colonnes ? |
| `test_cote.py` | le côté interdit ne trade nulle part |
| `test_retenue.py` | un portefeuille qui a coulé n'est ni retenu ni transmis |
| `test_concurrence.py` | la récompense se répartit correctement entre emplacements |
| `JOURNAL_MESURES.md` | l'historique des mesures, antichronologique |

---

## 8. Ce qui est ouvert

1. **La fenêtre de test de la lignée `or_` est consommée**, et par un harnais
   défectueux. Tout test ultérieur est une seconde lecture.
2. **Le chaînage n'a jamais franchi le fold 2 en conditions réelles.** Le
   mécanisme est vérifié — mêmes tenseurs, mêmes formes, donc `strict=True`
   passe — mais aucun run n'est allé jusque-là.
3. **Le stress-test n'a jamais produit un seul chiffre.**
4. **`kairos_live` garde deux branches mortes**, `duel` et `short`, dont les
   masques sont encore écrits en dur. Inoffensives en long-only, non
   corrigées faute de pouvoir les exécuter.
5. **La boucle multi-instruments n'existe pas.** `Portefeuille`,
   `instruments.py` et `alignement.py` sont écrits et testés, mais
   `run_training_on_split` ne joue qu'un instrument.
6. **L'écart entre ce qu'une régression linéaire extrait et ce que PPO
   extrait reste inexpliqué.**
7. **Le budget de risque est à zéro.** Le compte peut donc porter des dizaines
   de positions dont l'exposition cumulée dépasse l'équité. C'est un choix
   assumé — le modèle doit apprendre la limite — mais il faut le savoir en
   lisant n'importe quel creux.
8. **Le GPU est bridé thermiquement.** Les durées d'epoch ne sont pas
   comparables entre elles tant que la température dérive.

---

## 9. Le seul conseil qui compte

Sur toutes les corrections apportées à ce dépôt, **presque aucune n'a révélé
de performance cachée** ; la quasi-totalité a déplacé les chiffres dans le
même sens : moins bien qu'annoncé. Cette unanimité de direction est en soi un
résultat, et c'est le seul dont ce document soit sûr.

Le corollaire pratique : **quand un chiffre est bon, le premier réflexe est de
chercher le défaut qui l'explique.** Il a presque toujours été là.

**Mesurer sur la validation avant de payer un run. Toujours.**
