
## 04:18:07 — wf1 — EPOCH 014   -1.44$/trade   273 trades   WR 32.6%   PF 0.80   1 min

PnL         -393.88$   cumul run    -3517.81$   DD 8.3%   Sortino -0.157
point mort 37.7%  ->  ecart -5.1 pt (+/- 2.8 au mieux)
vs politique gelee du meme run : -2.6 pt   (reference -2.5 pt sur 5 epochs)
vs epoch precedente : -0.08$/trade
sens    LONG    48W/89  L  35.0%    -165.55$   |   SHORT   41W/95  L  30.1%    -228.33$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 35.4%  val 5.0%
politique  H 0.925/1.099 (-0.012)   etendue 0.5898 (590x le tremblement)   KL +0.0160   clipfrac 19.6%   g_actor 9.4e-01
train   PnL   +707.02$  1301 trades  WR 39.7%  PF 1.09   ->  ecart train-val +7.1 pt
. etendue val 0.5898, soit 590x le tremblement du seuil : la selection est portee par le modele.
. GPU BRIDE : 465 MHz a 82 C. Les durees ne sont pas comparables entre epochs si la temperature derive.
temps : collecte 37s  PPO 14s  calib 5s  validation 28s

## 04:19:47 — wf1 — EPOCH 015   -1.01$/trade   273 trades   WR 34.1%   PF 0.86   1 min

PnL         -276.13$   cumul run    -3793.94$   DD 7.8%   Sortino -0.111
point mort 37.5%  ->  ecart -3.4 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -1.0 pt   (reference -2.5 pt sur 5 epochs)
vs epoch precedente : +0.43$/trade
sens    LONG    52W/93  L  35.9%     -26.50$   |   SHORT   41W/87  L  32.0%    -249.63$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 34.2%  val 5.0%
politique  H 0.910/1.099 (-0.015)   etendue 0.6102 (610x le tremblement)   KL +0.0149   clipfrac 21.5%   g_actor 8.7e-01
train   PnL   +812.20$  1274 trades  WR 40.6%  PF 1.10   ->  ecart train-val +6.5 pt
. etendue val 0.6102, soit 610x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 40s  PPO 12s  calib 4s  validation 28s

## 04:21:07 — wf1 — EPOCH 016   -1.71$/trade   287 trades   WR 31.7%   PF 0.77   1 min

PnL         -491.16$   cumul run    -4285.10$   DD 9.6%   Sortino -0.183
point mort 37.6%  ->  ecart -5.9 pt (+/- 2.7 au mieux)
vs politique gelee du meme run : -3.5 pt   (reference -2.5 pt sur 5 epochs)
vs epoch precedente : -0.70$/trade
sens    LONG    49W/99  L  33.1%    -167.87$   |   SHORT   42W/97  L  30.2%    -323.30$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 33.1%  val 5.0%
politique  H 0.869/1.099 (-0.041)   etendue 0.6348 (635x le tremblement)   KL +0.0185   clipfrac 20.4%   g_actor 8.4e-01
train   PnL  +1691.76$  1231 trades  WR 43.2%  PF 1.23   ->  ecart train-val +11.5 pt
. etendue val 0.6348, soit 635x le tremblement du seuil : la selection est portee par le modele.
. GPU BRIDE : 480 MHz a 82 C. Les durees ne sont pas comparables entre epochs si la temperature derive.
temps : collecte 37s  PPO 11s  calib 4s  validation 27s

## 04:22:28 — wf1 — EPOCH 017   -1.47$/trade   297 trades   WR 32.3%   PF 0.80   1 min

PnL         -435.38$   cumul run    -4720.48$   DD 8.2%   Sortino -0.156
point mort 37.4%  ->  ecart -5.1 pt (+/- 2.7 au mieux)
vs politique gelee du meme run : -2.6 pt   (reference -2.5 pt sur 5 epochs)
vs epoch precedente : +0.25$/trade
sens    LONG    57W/101 L  36.1%     -93.59$   |   SHORT   39W/100 L  28.1%    -341.79$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 32.0%  val 5.0%
politique  H 0.861/1.099 (-0.008)   etendue 0.6974 (697x le tremblement)   KL +0.0149   clipfrac 21.3%   g_actor 1.0e+00
train   PnL  +2496.91$  1260 trades  WR 44.1%  PF 1.34   ->  ecart train-val +11.8 pt
. etendue val 0.6974, soit 697x le tremblement du seuil : la selection est portee par le modele.
. GPU BRIDE : 450 MHz a 82 C. Les durees ne sont pas comparables entre epochs si la temperature derive.
temps : collecte 34s  PPO 12s  calib 4s  validation 34s

## 04:25:07 — wf1 — EPOCH 001   +0.91$/trade   169 trades   WR 44.4%   PF 1.15   1 min

PnL         +153.83$   cumul run     +153.83$   DD 5.9%   Sortino +0.108
point mort 41.0%  ->  ecart +3.4 pt (+/- 3.8 au mieux)
sens    LONG    52W/60  L  46.4%    +179.58$   |   SHORT   23W/34  L  40.4%     -25.75$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0002 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.4e-05
train   PnL   +202.93$  1295 trades  WR 39.7%  PF 1.02   ->  ecart train-val -4.7 pt
. ACTOR GELE : gradient 1.4e-05, aucune mise a jour. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0002 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.91$) sous l'erreur-type de cette epoch (1.02$) — non separable de zero.
temps : collecte 31s  PPO 13s  calib 4s  validation 36s

## 04:29:35 — wf1 — EPOCH 001   +0.91$/trade   169 trades   WR 44.4%   PF 1.15   2 min

PnL         +153.83$   cumul run     +153.83$   DD 5.9%   Sortino +0.108
point mort 41.0%  ->  ecart +3.4 pt (+/- 3.8 au mieux)
sens    LONG    52W/60  L  46.4%    +179.58$   |   SHORT   23W/34  L  40.4%     -25.75$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0002 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.4e-05
train   PnL   +202.93$  1295 trades  WR 39.7%  PF 1.02   ->  ecart train-val -4.7 pt
. ACTOR GELE : gradient 1.4e-05, aucune mise a jour. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0002 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.91$) sous l'erreur-type de cette epoch (1.02$) — non separable de zero.
temps : collecte 32s  PPO 13s  calib 5s  validation 42s

## 04:31:15 — wf1 — EPOCH 002   -2.20$/trade   232 trades   WR 32.3%   PF 0.70   1 min

PnL         -509.67$   cumul run     -355.84$   DD 8.0%   Sortino -0.234
point mort 40.6%  ->  ecart -8.3 pt (+/- 3.1 au mieux)
vs epoch precedente : -3.11$/trade
sens    LONG    56W/95  L  37.1%    -181.31$   |   SHORT   19W/62  L  23.5%    -328.36$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 48.9%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0003 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 3.6e-06
train   PnL   -470.78$  1295 trades  WR 37.9%  PF 0.95   ->  ecart train-val +5.6 pt
. ACTOR GELE : gradient 3.6e-06, aucune mise a jour. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0003 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GPU BRIDE : 495 MHz a 83 C. Les durees ne sont pas comparables entre epochs si la temperature derive.
temps : collecte 32s  PPO 15s  calib 5s  validation 32s

## 04:32:35 — wf1 — EPOCH 003   +1.36$/trade   264 trades   WR 42.0%   PF 1.21   1 min

PnL         +358.10$   cumul run       +2.26$   DD 7.5%   Sortino +0.156
point mort 37.5%  ->  ecart +4.5 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +6.9 pt   (reference -2.4 pt sur 2 epochs)
vs epoch precedente : +3.55$/trade
sens    LONG    53W/68  L  43.8%    +163.61$   |   SHORT   58W/85  L  40.6%    +194.49$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 47.8%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0005 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 5.8e-06
train   PnL  +1055.38$  1278 trades  WR 41.6%  PF 1.13   ->  ecart train-val -0.4 pt
. ACTOR GELE : gradient 5.8e-06, aucune mise a jour. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0005 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 32s  PPO 14s  calib 5s  validation 32s

## 04:33:55 — wf1 — EPOCH 004   -0.85$/trade   201 trades   WR 37.8%   PF 0.88   1 min

PnL         -171.81$   cumul run     -169.55$   DD 7.1%   Sortino -0.095
point mort 40.9%  ->  ecart -3.1 pt (+/- 3.4 au mieux)
vs politique gelee du meme run : -3.0 pt   (reference -0.1 pt sur 3 epochs)
vs epoch precedente : -2.21$/trade
sens    LONG    65W/90  L  41.9%     +60.92$   |   SHORT   11W/35  L  23.9%    -232.73$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 46.6%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0004 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 6.2e-06
train   PnL   -122.01$  1295 trades  WR 39.0%  PF 0.99   ->  ecart train-val +1.2 pt
. ACTOR GELE : gradient 6.2e-06, aucune mise a jour. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0004 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.85$) sous l'erreur-type de cette epoch (0.96$) — non separable de zero.
temps : collecte 30s  PPO 13s  calib 5s  validation 35s

## 04:35:35 — wf1 — EPOCH 005   +0.77$/trade   245 trades   WR 40.8%   PF 1.12   2 min

PnL         +189.65$   cumul run      +20.10$   DD 9.2%   Sortino +0.087
point mort 38.1%  ->  ecart +2.7 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +3.5 pt   (reference -0.8 pt sur 4 epochs)
vs epoch precedente : +1.63$/trade
sens    LONG    83W/108 L  43.5%    +322.77$   |   SHORT   17W/37  L  31.5%    -133.12$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0002 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 5.1e-06
train   PnL   -440.94$  1240 trades  WR 38.5%  PF 0.95   ->  ecart train-val -2.3 pt
. ACTOR GELE : gradient 5.1e-06, aucune mise a jour. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0002 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.77$) sous l'erreur-type de cette epoch (0.90$) — non separable de zero.
temps : collecte 34s  PPO 13s  calib 6s  validation 57s

## 04:41:36 — wf1 — EPOCH 006   -0.83$/trade   245 trades   WR 34.7%   PF 0.88   6 min

PnL         -203.99$   cumul run     -183.89$   DD 8.1%   Sortino -0.090
point mort 37.6%  ->  ecart -2.9 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -2.8 pt   (reference -0.1 pt sur 5 epochs)
vs epoch precedente : -1.61$/trade
sens    LONG    36W/58  L  38.3%     +51.92$   |   SHORT   49W/102 L  32.5%    -255.90$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 44.4%  val 5.0%
politique  H 1.093/1.099 (-0.006)   etendue 0.0597 (60x le tremblement)   KL +0.0036   clipfrac 9.0%   g_actor 2.1e-01
train   PnL   -558.01$  1283 trades  WR 37.5%  PF 0.93   ->  ecart train-val +2.8 pt
. APPREND MAIS RESTE PLAT : le gradient passe (2.1e-01) mais l'entropie tient a 1.093 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.0597, soit 60x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.83$) sous l'erreur-type de cette epoch (0.86$) — non separable de zero.
. GPU BRIDE : 210 MHz a 89 C. Les durees ne sont pas comparables entre epochs si la temperature derive.
temps : collecte 46s  PPO 39s  calib 46s  validation 223s

## 04:45:36 — wf1 — EPOCH 007   +1.40$/trade   220 trades   WR 42.7%   PF 1.22   4 min

PnL         +307.59$   cumul run     +123.70$   DD 7.0%   Sortino +0.160
point mort 37.9%  ->  ecart +4.8 pt (+/- 3.3 au mieux)
vs politique gelee du meme run : +4.9 pt   (reference -0.1 pt sur 5 epochs)
vs epoch precedente : +2.23$/trade
sens    LONG    56W/71  L  44.1%    +291.92$   |   SHORT   38W/55  L  40.9%     +15.67$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 43.2%  val 5.0%
politique  H 1.086/1.099 (-0.007)   etendue 0.1453 (145x le tremblement)   KL +0.0013   clipfrac 3.1%   g_actor 2.7e-01
train   PnL    +28.49$  1262 trades  WR 39.7%  PF 1.00   ->  ecart train-val -3.0 pt
. etendue val 0.1453, soit 145x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 53s  PPO 115s  calib 13s  validation 52s

## 04:46:56 — wf1 — EPOCH 008   +0.25$/trade   250 trades   WR 40.0%   PF 1.04   1 min

PnL          +63.67$   cumul run     +187.37$   DD 7.8%   Sortino +0.029
point mort 39.1%  ->  ecart +0.9 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +1.1 pt   (reference -0.1 pt sur 5 epochs)
vs epoch precedente : -1.14$/trade
sens    LONG    36W/45  L  44.4%    +159.18$   |   SHORT   64W/105 L  37.9%     -95.52$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 42.1%  val 5.0%
politique  H 1.082/1.099 (-0.004)   etendue 0.1163 (116x le tremblement)   KL +0.0039   clipfrac 5.0%   g_actor 2.6e-01
train   PnL   +769.07$  1233 trades  WR 41.8%  PF 1.10   ->  ecart train-val +1.8 pt
. etendue val 0.1163, soit 116x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.25$) sous l'erreur-type de cette epoch (0.84$) — non separable de zero.
. GPU BRIDE : 480 MHz a 84 C. Les durees ne sont pas comparables entre epochs si la temperature derive.
temps : collecte 32s  PPO 15s  calib 5s  validation 33s

## 04:48:16 — wf1 — EPOCH 009   -1.34$/trade   276 trades   WR 34.8%   PF 0.82   1 min

PnL         -370.96$   cumul run     -183.59$   DD 8.4%   Sortino -0.143
point mort 39.4%  ->  ecart -4.6 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -4.5 pt   (reference -0.1 pt sur 5 epochs)
vs epoch precedente : -1.60$/trade
sens    LONG    32W/33  L  49.2%    +175.07$   |   SHORT   64W/147 L  30.3%    -546.04$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.069/1.099 (-0.013)   etendue 0.1939 (194x le tremblement)   KL +0.0035   clipfrac 7.8%   g_actor 3.1e-01
train   PnL   +682.37$  1320 trades  WR 40.8%  PF 1.08   ->  ecart train-val +6.0 pt
. etendue val 0.1939, soit 194x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 30s  PPO 15s  calib 5s  validation 38s

## 04:57:21 — wf1 — EPOCH 001   -0.64$/trade   269 trades   WR 37.2%   PF 0.91   1 min

PnL         -173.08$   cumul run     -173.08$   DD 11.8%   Sortino -0.071
point mort 39.4%  ->  ecart -2.2 pt (+/- 2.9 au mieux)
sens    LONG    41W/59  L  41.0%     -46.86$   |   SHORT   59W/110 L  34.9%    -126.22$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0004 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.2e-04
train   PnL   +202.93$  1295 trades  WR 39.7%  PF 1.02   ->  ecart train-val +2.5 pt
. ACTOR GELE : gradient 1.2e-04, aucune mise a jour. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0004 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. gain par trade (-0.64$) sous l'erreur-type de cette epoch (0.85$) — non separable de zero.
temps : collecte 32s  PPO 15s  calib 4s  validation 28s

## 04:59:01 — wf1 — EPOCH 002   -0.77$/trade   220 trades   WR 37.3%   PF 0.89   1 min

PnL         -168.45$   cumul run     -341.53$   DD 6.4%   Sortino -0.085
point mort 40.0%  ->  ecart -2.7 pt (+/- 3.3 au mieux)
vs epoch precedente : -0.12$/trade
sens    LONG    42W/59  L  41.6%     +35.81$   |   SHORT   40W/79  L  33.6%    -204.26$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 48.9%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0011 (1x le tremblement)   KL +0.0001   clipfrac 0.0%   g_actor 5.0e-04
train   PnL   -106.61$  1272 trades  WR 38.4%  PF 0.99   ->  ecart train-val +1.1 pt
. ACTOR GELE : gradient 5.0e-04, aucune mise a jour. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0011 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.77$) sous l'erreur-type de cette epoch (0.90$) — non separable de zero.
temps : collecte 30s  PPO 15s  calib 5s  validation 33s

## 05:00:21 — wf1 — EPOCH 003   +1.22$/trade   215 trades   WR 43.3%   PF 1.19   2 min

PnL         +262.67$   cumul run      -78.86$   DD 8.3%   Sortino +0.139
point mort 39.0%  ->  ecart +4.3 pt (+/- 3.4 au mieux)
vs politique gelee du meme run : +6.7 pt   (reference -2.5 pt sur 2 epochs)
vs epoch precedente : +1.99$/trade
sens    LONG    44W/45  L  49.4%    +234.80$   |   SHORT   49W/77  L  38.9%     +27.88$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 47.8%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0004 (0x le tremblement)   KL -0.0001   clipfrac 0.0%   g_actor 2.4e-04
train   PnL   +187.97$  1254 trades  WR 39.7%  PF 1.02   ->  ecart train-val -3.6 pt
. ACTOR GELE : gradient 2.4e-04, aucune mise a jour. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0004 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 31s  PPO 16s  calib 6s  validation 39s

## 05:00:36 — wf1 — EPOCH 001   -0.64$/trade   269 trades   WR 37.2%   PF 0.91   1 min

PnL         -173.08$   cumul run     -173.08$   DD 11.8%   Sortino -0.071
point mort 39.4%  ->  ecart -2.2 pt (+/- 2.9 au mieux)
sens    LONG    41W/59  L  41.0%     -46.86$   |   SHORT   59W/110 L  34.9%    -126.22$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0004 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.2e-04
train   PnL   +202.93$  1295 trades  WR 39.7%  PF 1.02   ->  ecart train-val +2.5 pt
. ACTOR GELE : warmup du critic, gradient 1.2e-04. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0004 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. gain par trade (-0.64$) sous l'erreur-type de cette epoch (0.85$) — non separable de zero.
temps : collecte 32s  PPO 15s  calib 4s  validation 28s

## 05:00:36 — wf1 — EPOCH 002   -0.77$/trade   220 trades   WR 37.3%   PF 0.89   1 min

PnL         -168.45$   cumul run     -341.53$   DD 6.4%   Sortino -0.085
point mort 40.0%  ->  ecart -2.7 pt (+/- 3.3 au mieux)
vs epoch precedente : -0.12$/trade
sens    LONG    42W/59  L  41.6%     +35.81$   |   SHORT   40W/79  L  33.6%    -204.26$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 48.9%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0011 (1x le tremblement)   KL +0.0001   clipfrac 0.0%   g_actor 5.0e-04
train   PnL   -106.61$  1272 trades  WR 38.4%  PF 0.99   ->  ecart train-val +1.1 pt
. ACTOR GELE : warmup du critic, gradient 5.0e-04. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0011 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.77$) sous l'erreur-type de cette epoch (0.90$) — non separable de zero.
temps : collecte 30s  PPO 15s  calib 5s  validation 33s

## 05:00:36 — wf1 — EPOCH 003   +1.22$/trade   215 trades   WR 43.3%   PF 1.19   2 min

PnL         +262.67$   cumul run      -78.86$   DD 8.3%   Sortino +0.139
point mort 39.0%  ->  ecart +4.3 pt (+/- 3.4 au mieux)
vs politique gelee du meme run : +6.7 pt   (reference -2.5 pt sur 2 epochs)
vs epoch precedente : +1.99$/trade
sens    LONG    44W/45  L  49.4%    +234.80$   |   SHORT   49W/77  L  38.9%     +27.88$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 47.8%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0004 (0x le tremblement)   KL -0.0001   clipfrac 0.0%   g_actor 2.4e-04
train   PnL   +187.97$  1254 trades  WR 39.7%  PF 1.02   ->  ecart train-val -3.6 pt
. ACTOR GELE : warmup du critic, gradient 2.4e-04. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0004 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 31s  PPO 16s  calib 6s  validation 39s

## 05:01:56 — wf1 — EPOCH 004   +0.64$/trade   237 trades   WR 39.7%   PF 1.10   2 min

PnL         +150.83$   cumul run      +71.97$   DD 7.9%   Sortino +0.073
point mort 37.4%  ->  ecart +2.3 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +2.5 pt   (reference -0.2 pt sur 3 epochs)
vs epoch precedente : -0.59$/trade
sens    LONG    48W/72  L  40.0%    +246.75$   |   SHORT   46W/71  L  39.3%     -95.92$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 46.6%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0003 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 5.2e-05
train   PnL   +283.75$  1289 trades  WR 40.0%  PF 1.03   ->  ecart train-val +0.3 pt
. ACTOR GELE : warmup du critic, gradient 5.2e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0003 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.64$) sous l'erreur-type de cette epoch (0.90$) — non separable de zero.
temps : collecte 38s  PPO 18s  calib 6s  validation 33s

## 05:03:36 — wf1 — EPOCH 005   -1.87$/trade   202 trades   WR 30.7%   PF 0.74   2 min

PnL         -377.96$   cumul run     -305.99$   DD 9.8%   Sortino -0.203
point mort 37.4%  ->  ecart -6.7 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : -7.1 pt   (reference +0.4 pt sur 4 epochs)
vs epoch precedente : -2.51$/trade
sens    LONG    22W/54  L  28.9%    -178.13$   |   SHORT   40W/86  L  31.7%    -199.83$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0008 (1x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 6.0e-04
train   PnL   -137.00$  1216 trades  WR 38.1%  PF 0.98   ->  ecart train-val +7.4 pt
. ACTOR GELE : warmup du critic, gradient 6.0e-04. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0008 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 32s  PPO 16s  calib 6s  validation 37s

## 05:04:56 — wf1 — EPOCH 006   +0.59$/trade   250 trades   WR 40.4%   PF 1.09   2 min

PnL         +147.43$   cumul run     -158.56$   DD 7.5%   Sortino +0.067
point mort 38.3%  ->  ecart +2.1 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +3.1 pt   (reference -1.0 pt sur 5 epochs)
vs epoch precedente : +2.46$/trade
sens    LONG    24W/28  L  46.2%     +70.97$   |   SHORT   77W/121 L  38.9%     +76.45$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 44.4%  val 5.0%
politique  H 1.095/1.099 (-0.004)   etendue 0.0454 (45x le tremblement)   KL +0.0050   clipfrac 4.7%   g_actor 6.3e-01
train   PnL   -411.42$  1218 trades  WR 38.5%  PF 0.95   ->  ecart train-val -1.9 pt
. APPREND MAIS RESTE PLAT : le gradient passe (6.3e-01) mais l'entropie tient a 1.095 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.0454, soit 45x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (+0.59$) sous l'erreur-type de cette epoch (0.89$) — non separable de zero.
. GPU BRIDE : 450 MHz a 83 C. Les durees ne sont pas comparables entre epochs si la temperature derive.
temps : collecte 31s  PPO 17s  calib 6s  validation 37s

## 05:10:20 — wf1 — EPOCH 001   +0.23$/trade   275 trades   WR 38.5%   PF 1.03   2 min

PnL          +63.34$   cumul run      +63.34$   DD 11.2%   Sortino +0.025
point mort 37.8%  ->  ecart +0.7 pt (+/- 2.9 au mieux)
sens    LONG    54W/78  L  40.9%    +136.98$   |   SHORT   52W/91  L  36.4%     -73.64$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 50.0%  val 5.0%
politique  H 0.573/1.099   etendue 0.0004 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 9.3e-05
train   PnL   +131.24$  1239 trades  WR 40.5%  PF 1.02   ->  ecart train-val +2.0 pt
. ACTOR GELE : warmup du critic, gradient 9.3e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0004 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.23$) sous l'erreur-type de cette epoch (0.97$) — non separable de zero.
temps : collecte 40s  PPO 32s  calib 7s  validation 45s

## 05:12:20 — wf1 — EPOCH 002   -0.60$/trade   273 trades   WR 36.3%   PF 0.91   2 min

PnL         -163.28$   cumul run      -99.94$   DD 8.9%   Sortino -0.065
point mort 38.5%  ->  ecart -2.2 pt (+/- 2.9 au mieux)
vs epoch precedente : -0.83$/trade
sens    LONG    47W/80  L  37.0%     -92.97$   |   SHORT   52W/94  L  35.6%     -70.31$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 48.9%  val 5.0%
politique  H 0.566/1.099 (-0.007)   etendue 0.0002 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 7.0e-05
train   PnL   +352.01$  1237 trades  WR 41.1%  PF 1.04   ->  ecart train-val +4.8 pt
. ACTOR GELE : warmup du critic, gradient 7.0e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0002 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. gain par trade (-0.60$) sous l'erreur-type de cette epoch (0.79$) — non separable de zero.
temps : collecte 38s  PPO 34s  calib 7s  validation 34s

## 05:14:20 — wf1 — EPOCH 003   -1.29$/trade   246 trades   WR 34.6%   PF 0.82   2 min

PnL         -318.20$   cumul run     -418.14$   DD 11.4%   Sortino -0.140
point mort 39.2%  ->  ecart -4.6 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -3.8 pt   (reference -0.8 pt sur 2 epochs)
vs epoch precedente : -0.70$/trade
sens    LONG    33W/57  L  36.7%     -85.83$   |   SHORT   52W/104 L  33.3%    -232.37$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 47.8%  val 5.0%
politique  H 0.573/1.099 (+0.007)   etendue 0.0002 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 3.8e-06
train   PnL   +375.47$  1257 trades  WR 38.8%  PF 1.05   ->  ecart train-val +4.2 pt
. ACTOR GELE : warmup du critic, gradient 3.8e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0002 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 44s  PPO 35s  calib 7s  validation 32s

## 05:16:09 — wf1 — EPOCH 001   +0.23$/trade   275 trades   WR 38.5%   PF 1.03   2 min

PnL          +63.34$   cumul run      +63.34$   DD 11.2%   Sortino +0.025
point mort 37.8%  ->  ecart +0.7 pt (+/- 2.9 au mieux)
sens    LONG    54W/78  L  40.9%    +136.98$   |   SHORT   52W/91  L  36.4%     -73.64$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 50.0%  val 5.0%
politique  H 0.573/0.621   etendue 0.0004 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 9.3e-05
train   PnL   +131.24$  1239 trades  WR 40.5%  PF 1.02   ->  ecart train-val +2.0 pt
. ACTOR GELE : warmup du critic, gradient 9.3e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0004 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.23$) sous l'erreur-type de cette epoch (0.97$) — non separable de zero.
temps : collecte 40s  PPO 32s  calib 7s  validation 45s

## 05:16:09 — wf1 — EPOCH 002   -0.60$/trade   273 trades   WR 36.3%   PF 0.91   2 min

PnL         -163.28$   cumul run      -99.94$   DD 8.9%   Sortino -0.065
point mort 38.5%  ->  ecart -2.2 pt (+/- 2.9 au mieux)
vs epoch precedente : -0.83$/trade
sens    LONG    47W/80  L  37.0%     -92.97$   |   SHORT   52W/94  L  35.6%     -70.31$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 48.9%  val 5.0%
politique  H 0.566/0.621 (-0.007)   etendue 0.0002 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 7.0e-05
train   PnL   +352.01$  1237 trades  WR 41.1%  PF 1.04   ->  ecart train-val +4.8 pt
. ACTOR GELE : warmup du critic, gradient 7.0e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0002 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. gain par trade (-0.60$) sous l'erreur-type de cette epoch (0.79$) — non separable de zero.
temps : collecte 38s  PPO 34s  calib 7s  validation 34s

## 05:16:09 — wf1 — EPOCH 003   -1.29$/trade   246 trades   WR 34.6%   PF 0.82   2 min

PnL         -318.20$   cumul run     -418.14$   DD 11.4%   Sortino -0.140
point mort 39.2%  ->  ecart -4.6 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -3.8 pt   (reference -0.8 pt sur 2 epochs)
vs epoch precedente : -0.70$/trade
sens    LONG    33W/57  L  36.7%     -85.83$   |   SHORT   52W/104 L  33.3%    -232.37$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 47.8%  val 5.0%
politique  H 0.573/0.621 (+0.007)   etendue 0.0002 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 3.8e-06
train   PnL   +375.47$  1257 trades  WR 38.8%  PF 1.05   ->  ecart train-val +4.2 pt
. ACTOR GELE : warmup du critic, gradient 3.8e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0002 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 44s  PPO 35s  calib 7s  validation 32s

## 05:16:09 — wf1 — EPOCH 004   -1.26$/trade   254 trades   WR 36.2%   PF 0.82   2 min

PnL         -319.13$   cumul run     -737.27$   DD 7.1%   Sortino -0.137
point mort 40.9%  ->  ecart -4.7 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -2.7 pt   (reference -2.0 pt sur 3 epochs)
vs epoch precedente : +0.04$/trade
sens    LONG    39W/67  L  36.8%     -94.93$   |   SHORT   53W/95  L  35.8%    -224.20$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 46.6%  val 5.0%
politique  H 0.567/0.621 (-0.006)   etendue 0.0009 (1x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 9.4e-05
train   PnL   +509.95$  1278 trades  WR 39.8%  PF 1.06   ->  ecart train-val +3.6 pt
. ACTOR GELE : warmup du critic, gradient 9.4e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0009 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 40s  PPO 36s  calib 6s  validation 40s

## 05:18:09 — wf1 — EPOCH 005   -2.71$/trade   241 trades   WR 32.0%   PF 0.65   2 min

PnL         -653.91$   cumul run    -1391.18$   DD 10.8%   Sortino -0.282
point mort 41.9%  ->  ecart -9.9 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -7.2 pt   (reference -2.7 pt sur 4 epochs)
vs epoch precedente : -1.46$/trade
sens    LONG    29W/54  L  34.9%    -254.50$   |   SHORT   48W/110 L  30.4%    -399.41$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 45.5%  val 5.0%
politique  H 0.573/0.621 (+0.006)   etendue 0.0002 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 2.2e-05
train   PnL    -57.04$  1228 trades  WR 39.3%  PF 0.99   ->  ecart train-val +7.3 pt
. ACTOR GELE : warmup du critic, gradient 2.2e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0002 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 40s  PPO 34s  calib 6s  validation 38s

## 05:20:09 — wf1 — EPOCH 006   -0.36$/trade   254 trades   WR 35.8%   PF 0.95   2 min

PnL          -90.86$   cumul run    -1482.04$   DD 7.1%   Sortino -0.038
point mort 37.0%  ->  ecart -1.2 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +2.9 pt   (reference -4.1 pt sur 5 epochs)
vs epoch precedente : +2.36$/trade
sens    LONG    38W/66  L  36.5%     -84.95$   |   SHORT   53W/97  L  35.3%      -5.91$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 44.4%  val 5.0%
politique  H 0.531/0.621 (-0.042)   etendue 0.2335 (234x le tremblement)   KL +0.0044   clipfrac 9.4%   g_actor 4.2e-01
train   PnL   +251.69$  1203 trades  WR 39.6%  PF 1.03   ->  ecart train-val +3.8 pt
. etendue val 0.2335, soit 234x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (-0.36$) sous l'erreur-type de cette epoch (0.91$) — non separable de zero.
temps : collecte 36s  PPO 38s  calib 6s  validation 47s

## 05:22:29 — wf1 — EPOCH 007   -0.32$/trade   267 trades   WR 36.3%   PF 0.96   2 min

PnL          -86.06$   cumul run    -1568.10$   DD 8.9%   Sortino -0.035
point mort 37.3%  ->  ecart -1.0 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +3.2 pt   (reference -4.1 pt sur 5 epochs)
vs epoch precedente : +0.04$/trade
sens    LONG    50W/83  L  37.6%     +59.51$   |   SHORT   47W/87  L  35.1%    -145.57$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 43.2%  val 5.0%
politique  H 0.542/0.621 (+0.011)   etendue 0.3852 (385x le tremblement)   KL +0.0042   clipfrac 8.5%   g_actor 3.7e-01
train   PnL   +657.13$  1241 trades  WR 41.7%  PF 1.08   ->  ecart train-val +5.4 pt
. etendue val 0.3852, soit 385x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.32$) sous l'erreur-type de cette epoch (1.00$) — non separable de zero.
temps : collecte 43s  PPO 37s  calib 7s  validation 45s

## 05:24:49 — wf1 — EPOCH 008   -0.71$/trade   295 trades   WR 34.6%   PF 0.90   2 min

PnL         -209.69$   cumul run    -1777.79$   DD 7.4%   Sortino -0.076
point mort 37.0%  ->  ecart -2.4 pt (+/- 2.8 au mieux)
vs politique gelee du meme run : +1.8 pt   (reference -4.1 pt sur 5 epochs)
vs epoch precedente : -0.39$/trade
sens    LONG    59W/117 L  33.5%    -236.89$   |   SHORT   43W/76  L  36.1%     +27.20$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 42.1%  val 5.0%
politique  H 0.543/0.621 (+0.001)   etendue 0.4515 (452x le tremblement)   KL +0.0052   clipfrac 11.4%   g_actor 3.1e-01
train   PnL   +208.12$  1248 trades  WR 40.9%  PF 1.03   ->  ecart train-val +6.3 pt
. etendue val 0.4515, soit 452x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le SHORT rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.71$) sous l'erreur-type de cette epoch (0.81$) — non separable de zero.
temps : collecte 38s  PPO 39s  calib 7s  validation 48s

## 05:29:26 — wf1 — EPOCH 001   -0.08$/trade   227 trades   WR 38.3%   PF 0.99   2 min

PnL          -18.47$   cumul run      -18.47$   DD 6.6%   Sortino -0.009
point mort 38.6%  ->  ecart -0.3 pt (+/- 3.2 au mieux)
sens    LONG    60W/80  L  42.9%    +112.13$   |   SHORT   27W/60  L  31.0%    -130.61$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 6.7e-06
train   PnL   +202.93$  1295 trades  WR 39.7%  PF 1.02   ->  ecart train-val +1.4 pt
. ACTOR GELE : warmup du critic, gradient 6.7e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.08$) sous l'erreur-type de cette epoch (1.10$) — non separable de zero.
temps : collecte 39s  PPO 20s  calib 7s  validation 39s

## 05:31:06 — wf1 — EPOCH 002   +1.25$/trade   254 trades   WR 41.3%   PF 1.20   2 min

PnL         +317.82$   cumul run     +299.35$   DD 8.4%   Sortino +0.145
point mort 37.0%  ->  ecart +4.3 pt (+/- 3.1 au mieux)
vs epoch precedente : +1.33$/trade
sens    LONG    80W/107 L  42.8%    +245.36$   |   SHORT   25W/42  L  37.3%     +72.46$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 48.9%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 1.0e-06
train   PnL   -470.78$  1295 trades  WR 37.9%  PF 0.95   ->  ecart train-val -3.4 pt
. ACTOR GELE : warmup du critic, gradient 1.0e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 35s  PPO 20s  calib 6s  validation 37s

## 05:32:46 — wf1 — EPOCH 003   -0.95$/trade   238 trades   WR 35.3%   PF 0.87   2 min

PnL         -225.73$   cumul run      +73.62$   DD 8.9%   Sortino -0.103
point mort 38.5%  ->  ecart -3.2 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : -5.3 pt   (reference +2.0 pt sur 2 epochs)
vs epoch precedente : -2.20$/trade
sens    LONG    44W/67  L  39.6%     +38.94$   |   SHORT   40W/87  L  31.5%    -264.67$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 47.8%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 7.1e-07
train   PnL  +1055.38$  1278 trades  WR 41.6%  PF 1.13   ->  ecart train-val +6.3 pt
. ACTOR GELE : warmup du critic, gradient 7.1e-07. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 37s  PPO 20s  calib 7s  validation 48s

## 05:34:46 — wf1 — EPOCH 004   +0.07$/trade   247 trades   WR 38.1%   PF 1.01   2 min

PnL          +17.01$   cumul run      +90.63$   DD 7.9%   Sortino +0.007
point mort 37.8%  ->  ecart +0.3 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +0.0 pt   (reference +0.3 pt sur 3 epochs)
vs epoch precedente : +1.02$/trade
sens    LONG    64W/89  L  41.8%    +151.51$   |   SHORT   30W/64  L  31.9%    -134.50$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 46.6%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 2.3e-05
train   PnL   -127.68$  1296 trades  WR 39.2%  PF 0.98   ->  ecart train-val +1.1 pt
. ACTOR GELE : warmup du critic, gradient 2.3e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.07$) sous l'erreur-type de cette epoch (0.91$) — non separable de zero.
temps : collecte 36s  PPO 20s  calib 6s  validation 51s

## 05:36:06 — wf1 — EPOCH 005   -0.16$/trade   255 trades   WR 36.9%   PF 0.98   1 min

PnL          -39.56$   cumul run      +51.07$   DD 7.9%   Sortino -0.017
point mort 37.3%  ->  ecart -0.4 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -0.7 pt   (reference +0.3 pt sur 4 epochs)
vs epoch precedente : -0.22$/trade
sens    LONG    74W/116 L  38.9%    +117.28$   |   SHORT   20W/45  L  30.8%    -156.83$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 4.4e-06
train   PnL   -204.86$  1266 trades  WR 38.9%  PF 0.98   ->  ecart train-val +2.0 pt
. ACTOR GELE : warmup du critic, gradient 4.4e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.16$) sous l'erreur-type de cette epoch (0.99$) — non separable de zero.
temps : collecte 34s  PPO 19s  calib 6s  validation 29s

## 05:37:47 — wf1 — EPOCH 006   -1.99$/trade   250 trades   WR 32.0%   PF 0.74   2 min

PnL         -497.37$   cumul run     -446.30$   DD 6.7%   Sortino -0.209
point mort 38.9%  ->  ecart -6.9 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -7.0 pt   (reference +0.1 pt sur 5 epochs)
vs epoch precedente : -1.83$/trade
sens    LONG    52W/98  L  34.7%    -121.61$   |   SHORT   28W/72  L  28.0%    -375.76$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 44.4%  val 5.0%
politique  H 1.092/1.099 (-0.007)   etendue 0.0648 (65x le tremblement)   KL +0.0068   clipfrac 15.2%   g_actor 2.5e-01
train   PnL   -461.05$  1213 trades  WR 39.3%  PF 0.94   ->  ecart train-val +7.3 pt
. APPREND MAIS RESTE PLAT : le gradient passe (2.5e-01) mais l'entropie tient a 1.092 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.0648, soit 65x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 31s  PPO 18s  calib 6s  validation 39s

## 05:39:27 — wf1 — EPOCH 007   -0.40$/trade   257 trades   WR 36.2%   PF 0.94   2 min

PnL         -103.35$   cumul run     -549.65$   DD 7.6%   Sortino -0.045
point mort 37.6%  ->  ecart -1.4 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -1.6 pt   (reference +0.1 pt sur 5 epochs)
vs epoch precedente : +1.59$/trade
sens    LONG    50W/85  L  37.0%     -35.95$   |   SHORT   43W/79  L  35.2%     -67.40$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 43.2%  val 5.0%
politique  H 1.088/1.099 (-0.004)   etendue 0.0808 (81x le tremblement)   KL +0.0000   clipfrac 1.0%   g_actor 2.4e-01
train   PnL    +36.85$  1305 trades  WR 39.9%  PF 1.00   ->  ecart train-val +3.7 pt
. APPREND MAIS RESTE PLAT : le gradient passe (2.4e-01) mais l'entropie tient a 1.088 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.0808, soit 81x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (-0.40$) sous l'erreur-type de cette epoch (0.84$) — non separable de zero.
temps : collecte 33s  PPO 19s  calib 7s  validation 35s

## 05:41:07 — wf1 — EPOCH 008   -0.27$/trade   238 trades   WR 36.1%   PF 0.96   2 min

PnL          -63.24$   cumul run     -612.89$   DD 10.6%   Sortino -0.029
point mort 37.1%  ->  ecart -1.0 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : -1.1 pt   (reference +0.1 pt sur 5 epochs)
vs epoch precedente : +0.14$/trade
sens    LONG    52W/80  L  39.4%    +103.46$   |   SHORT   34W/72  L  32.1%    -166.70$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 42.1%  val 5.0%
politique  H 1.079/1.099 (-0.009)   etendue 0.1646 (165x le tremblement)   KL +0.0021   clipfrac 2.7%   g_actor 2.6e-01
train   PnL   +141.20$  1249 trades  WR 38.8%  PF 1.02   ->  ecart train-val +2.7 pt
. etendue val 0.1646, soit 165x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.27$) sous l'erreur-type de cette epoch (0.87$) — non separable de zero.
temps : collecte 31s  PPO 18s  calib 6s  validation 42s

## 05:42:47 — wf1 — EPOCH 009   -1.15$/trade   241 trades   WR 33.6%   PF 0.85   2 min

PnL         -276.05$   cumul run     -888.94$   DD 10.5%   Sortino -0.122
point mort 37.3%  ->  ecart -3.7 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -3.9 pt   (reference +0.1 pt sur 5 epochs)
vs epoch precedente : -0.88$/trade
sens    LONG    53W/85  L  38.4%     -13.40$   |   SHORT   28W/75  L  27.2%    -262.64$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.081/1.099 (+0.002)   etendue 0.2657 (266x le tremblement)   KL +0.0029   clipfrac 3.3%   g_actor 2.7e-01
train   PnL   +958.68$  1342 trades  WR 41.6%  PF 1.11   ->  ecart train-val +8.0 pt
. etendue val 0.2657, soit 266x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 36s  PPO 19s  calib 7s  validation 43s

## 05:47:18 — wf1 — EPOCH 001   -0.08$/trade   227 trades   WR 38.3%   PF 0.99   2 min

PnL          -18.47$   cumul run      -18.47$   DD 6.6%   Sortino -0.009
point mort 38.6%  ->  ecart -0.3 pt (+/- 3.2 au mieux)
sens    LONG    60W/80  L  42.9%    +112.13$   |   SHORT   27W/60  L  31.0%    -130.61$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 6.7e-06
train   PnL   +202.93$  1295 trades  WR 39.7%  PF 1.02   ->  ecart train-val +1.4 pt
. ACTOR GELE : warmup du critic, gradient 6.7e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.08$) sous l'erreur-type de cette epoch (1.10$) — non separable de zero.
temps : collecte 33s  PPO 18s  calib 5s  validation 35s

## 05:48:59 — wf1 — EPOCH 002   +1.25$/trade   254 trades   WR 41.3%   PF 1.20   2 min

PnL         +317.82$   cumul run     +299.35$   DD 8.4%   Sortino +0.145
point mort 37.0%  ->  ecart +4.3 pt (+/- 3.1 au mieux)
vs epoch precedente : +1.33$/trade
sens    LONG    80W/107 L  42.8%    +245.36$   |   SHORT   25W/42  L  37.3%     +72.46$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 1.0e-06
train   PnL   -470.78$  1295 trades  WR 37.9%  PF 0.95   ->  ecart train-val -3.4 pt
. ACTOR GELE : warmup du critic, gradient 1.0e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 33s  PPO 20s  calib 6s  validation 32s

## 05:50:19 — wf1 — EPOCH 003   -0.95$/trade   238 trades   WR 35.3%   PF 0.87   2 min

PnL         -225.73$   cumul run      +73.62$   DD 8.9%   Sortino -0.103
point mort 38.5%  ->  ecart -3.2 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : -5.3 pt   (reference +2.0 pt sur 2 epochs)
vs epoch precedente : -2.20$/trade
sens    LONG    44W/67  L  39.6%     +38.94$   |   SHORT   40W/87  L  31.5%    -264.67$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 7.1e-07
train   PnL  +1055.38$  1278 trades  WR 41.6%  PF 1.13   ->  ecart train-val +6.3 pt
. ACTOR GELE : warmup du critic, gradient 7.1e-07. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 33s  PPO 18s  calib 6s  validation 39s

## 05:52:19 — wf1 — EPOCH 004   +0.07$/trade   247 trades   WR 38.1%   PF 1.01   2 min

PnL          +17.01$   cumul run      +90.63$   DD 7.9%   Sortino +0.007
point mort 37.8%  ->  ecart +0.3 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +0.0 pt   (reference +0.3 pt sur 3 epochs)
vs epoch precedente : +1.02$/trade
sens    LONG    64W/89  L  41.8%    +151.51$   |   SHORT   30W/64  L  31.9%    -134.50$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 2.3e-05
train   PnL   -127.68$  1296 trades  WR 39.2%  PF 0.98   ->  ecart train-val +1.1 pt
. ACTOR GELE : warmup du critic, gradient 2.3e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.07$) sous l'erreur-type de cette epoch (0.91$) — non separable de zero.
temps : collecte 33s  PPO 19s  calib 6s  validation 47s

## 05:53:39 — wf1 — EPOCH 005   -0.16$/trade   255 trades   WR 36.9%   PF 0.98   2 min

PnL          -39.56$   cumul run      +51.07$   DD 7.9%   Sortino -0.017
point mort 37.3%  ->  ecart -0.4 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -0.7 pt   (reference +0.3 pt sur 4 epochs)
vs epoch precedente : -0.22$/trade
sens    LONG    74W/116 L  38.9%    +117.28$   |   SHORT   20W/45  L  30.8%    -156.83$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 32.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 4.4e-06
train   PnL   -204.86$  1266 trades  WR 38.9%  PF 0.98   ->  ecart train-val +2.0 pt
. ACTOR GELE : warmup du critic, gradient 4.4e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.16$) sous l'erreur-type de cette epoch (0.99$) — non separable de zero.
temps : collecte 31s  PPO 19s  calib 7s  validation 39s

## 05:55:39 — wf1 — EPOCH 006   -1.99$/trade   250 trades   WR 32.0%   PF 0.74   2 min

PnL         -497.37$   cumul run     -446.30$   DD 6.7%   Sortino -0.209
point mort 38.9%  ->  ecart -6.9 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -7.0 pt   (reference +0.1 pt sur 5 epochs)
vs epoch precedente : -1.83$/trade
sens    LONG    52W/98  L  34.7%    -121.61$   |   SHORT   28W/72  L  28.0%    -375.76$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 27.5%  val 5.0%
politique  H 1.092/1.099 (-0.007)   etendue 0.0648 (65x le tremblement)   KL +0.0068   clipfrac 15.2%   g_actor 2.5e-01
train   PnL   -461.05$  1213 trades  WR 39.3%  PF 0.94   ->  ecart train-val +7.3 pt
. APPREND MAIS RESTE PLAT : le gradient passe (2.5e-01) mais l'entropie tient a 1.092 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.0648, soit 65x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 36s  PPO 19s  calib 6s  validation 44s

## 05:57:19 — wf1 — EPOCH 007   -0.40$/trade   257 trades   WR 36.2%   PF 0.94   2 min

PnL         -103.35$   cumul run     -549.65$   DD 7.6%   Sortino -0.045
point mort 37.6%  ->  ecart -1.4 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -1.6 pt   (reference +0.1 pt sur 5 epochs)
vs epoch precedente : +1.59$/trade
sens    LONG    50W/85  L  37.0%     -35.95$   |   SHORT   43W/79  L  35.2%     -67.40$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 23.0%  val 5.0%
politique  H 1.088/1.099 (-0.004)   etendue 0.0808 (81x le tremblement)   KL +0.0000   clipfrac 1.0%   g_actor 2.4e-01
train   PnL    +36.85$  1305 trades  WR 39.9%  PF 1.00   ->  ecart train-val +3.7 pt
. APPREND MAIS RESTE PLAT : le gradient passe (2.4e-01) mais l'entropie tient a 1.088 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.0808, soit 81x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (-0.40$) sous l'erreur-type de cette epoch (0.84$) — non separable de zero.
temps : collecte 34s  PPO 20s  calib 8s  validation 46s

## 05:58:59 — wf1 — EPOCH 008   -0.27$/trade   238 trades   WR 36.1%   PF 0.96   2 min

PnL          -63.24$   cumul run     -612.89$   DD 10.6%   Sortino -0.029
point mort 37.1%  ->  ecart -1.0 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : -1.1 pt   (reference +0.1 pt sur 5 epochs)
vs epoch precedente : +0.14$/trade
sens    LONG    52W/80  L  39.4%    +103.46$   |   SHORT   34W/72  L  32.1%    -166.70$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 18.5%  val 5.0%
politique  H 1.079/1.099 (-0.009)   etendue 0.1646 (165x le tremblement)   KL +0.0021   clipfrac 2.7%   g_actor 2.6e-01
train   PnL   +141.20$  1249 trades  WR 38.8%  PF 1.02   ->  ecart train-val +2.7 pt
. etendue val 0.1646, soit 165x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.27$) sous l'erreur-type de cette epoch (0.87$) — non separable de zero.
temps : collecte 37s  PPO 19s  calib 7s  validation 42s

## 06:00:59 — wf1 — EPOCH 009   -1.15$/trade   241 trades   WR 33.6%   PF 0.85   2 min

PnL         -276.05$   cumul run     -888.94$   DD 10.5%   Sortino -0.122
point mort 37.3%  ->  ecart -3.7 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -3.9 pt   (reference +0.1 pt sur 5 epochs)
vs epoch precedente : -0.88$/trade
sens    LONG    53W/85  L  38.4%     -13.40$   |   SHORT   28W/75  L  27.2%    -262.64$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 14.0%  val 5.0%
politique  H 1.081/1.099 (+0.002)   etendue 0.2657 (266x le tremblement)   KL +0.0029   clipfrac 3.3%   g_actor 2.7e-01
train   PnL   +958.68$  1342 trades  WR 41.6%  PF 1.11   ->  ecart train-val +8.0 pt
. etendue val 0.2657, soit 266x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 38s  PPO 21s  calib 7s  validation 54s

## 06:02:59 — wf1 — EPOCH 010   -1.13$/trade   248 trades   WR 33.9%   PF 0.85   2 min

PnL         -280.32$   cumul run    -1169.26$   DD 10.0%   Sortino -0.121
point mort 37.6%  ->  ecart -3.7 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -3.8 pt   (reference +0.1 pt sur 5 epochs)
vs epoch precedente : +0.02$/trade
sens    LONG    53W/95  L  35.8%     -41.18$   |   SHORT   31W/69  L  31.0%    -239.14$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 9.5%  val 5.0%
politique  H 1.074/1.099 (-0.007)   etendue 0.3323 (332x le tremblement)   KL +0.0045   clipfrac 5.7%   g_actor 2.6e-01
train   PnL    -23.48$  1335 trades  WR 39.6%  PF 1.00   ->  ecart train-val +5.7 pt
. etendue val 0.3323, soit 332x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 43s  PPO 19s  calib 7s  validation 43s

## 06:04:39 — wf1 — EPOCH 011   -1.35$/trade   259 trades   WR 33.2%   PF 0.82   2 min

PnL         -349.77$   cumul run    -1519.03$   DD 11.8%   Sortino -0.143
point mort 37.7%  ->  ecart -4.5 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -4.7 pt   (reference +0.1 pt sur 5 epochs)
vs epoch precedente : -0.22$/trade
sens    LONG    56W/109 L  33.9%    -139.78$   |   SHORT   30W/64  L  31.9%    -209.98$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.058/1.099 (-0.016)   etendue 0.4007 (401x le tremblement)   KL +0.0063   clipfrac 8.2%   g_actor 2.8e-01
train   PnL   +682.96$  1243 trades  WR 40.1%  PF 1.09   ->  ecart train-val +6.9 pt
. etendue val 0.4007, soit 401x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 37s  PPO 17s  calib 7s  validation 45s

## 06:06:19 — wf1 — EPOCH 012   -1.00$/trade   272 trades   WR 34.2%   PF 0.86   2 min

PnL         -271.00$   cumul run    -1790.03$   DD 14.2%   Sortino -0.106
point mort 37.7%  ->  ecart -3.5 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -3.6 pt   (reference +0.1 pt sur 5 epochs)
vs epoch precedente : +0.35$/trade
sens    LONG    55W/113 L  32.7%    -294.14$   |   SHORT   38W/66  L  36.5%     +23.14$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.047/1.099 (-0.011)   etendue 0.4401 (440x le tremblement)   KL +0.0048   clipfrac 6.2%   g_actor 3.2e-01
train   PnL  +1368.56$  1297 trades  WR 41.1%  PF 1.17   ->  ecart train-val +6.9 pt
. etendue val 0.4401, soit 440x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le SHORT rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 37s  PPO 17s  calib 6s  validation 44s

## 06:11:03 — wf1 — EPOCH 001   -0.59$/trade   250 trades   WR 36.0%   PF 0.91   2 min

PnL         -148.26$   cumul run     -148.26$   DD 10.3%   Sortino -0.065
point mort 38.2%  ->  ecart -2.2 pt (+/- 3.0 au mieux)
sens    LONG    67W/112 L  37.4%     -12.65$   |   SHORT   23W/48  L  32.4%    -135.61$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0000 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 1.1e-06
train   PnL   +202.93$  1295 trades  WR 39.7%  PF 1.02   ->  ecart train-val +3.7 pt
. ACTOR GELE : warmup du critic, gradient 1.1e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. gain par trade (-0.59$) sous l'erreur-type de cette epoch (0.82$) — non separable de zero.
temps : collecte 39s  PPO 43s  calib 7s  validation 42s

## 06:13:23 — wf1 — EPOCH 002   -2.11$/trade   200 trades   WR 34.0%   PF 0.71   2 min

PnL         -421.23$   cumul run     -569.49$   DD 7.2%   Sortino -0.226
point mort 42.0%  ->  ecart -8.0 pt (+/- 3.3 au mieux)
vs epoch precedente : -1.51$/trade
sens    LONG    49W/84  L  36.8%    -198.41$   |   SHORT   19W/48  L  28.4%    -222.82$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0002 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 4.7e-06
train   PnL   +472.18$  1263 trades  WR 39.8%  PF 1.06   ->  ecart train-val +5.8 pt
. ACTOR GELE : warmup du critic, gradient 4.7e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0002 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 36s  PPO 45s  calib 8s  validation 45s

## 06:13:54 — wf1 — EPOCH 001   -0.59$/trade   250 trades   WR 36.0%   PF 0.91   2 min

PnL         -148.26$   cumul run     -148.26$   DD 10.3%   Sortino -0.065
point mort 38.2%  ->  ecart -2.2 pt (+/- 3.0 au mieux)
sens    LONG    67W/112 L  37.4%     -12.65$   |   SHORT   23W/48  L  32.4%    -135.61$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0000 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 1.1e-06
train   PnL   +202.93$  1295 trades  WR 39.7%  PF 1.02   ->  ecart train-val +3.7 pt
. ACTOR GELE : warmup du critic, gradient 1.1e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. gain par trade (-0.59$) sous l'erreur-type de cette epoch (0.82$) — non separable de zero.
temps : collecte 39s  PPO 43s  calib 7s  validation 42s

## 06:13:54 — wf1 — EPOCH 002   -2.11$/trade   200 trades   WR 34.0%   PF 0.71   2 min

PnL         -421.23$   cumul run     -569.49$   DD 7.2%   Sortino -0.226
point mort 42.0%  ->  ecart -8.0 pt (+/- 3.3 au mieux)
vs epoch precedente : -1.51$/trade
sens    LONG    49W/84  L  36.8%    -198.41$   |   SHORT   19W/48  L  28.4%    -222.82$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0002 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 4.7e-06
train   PnL   +472.18$  1263 trades  WR 39.8%  PF 1.06   ->  ecart train-val +5.8 pt
. ACTOR GELE : warmup du critic, gradient 4.7e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0002 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 36s  PPO 45s  calib 8s  validation 45s

## 06:15:34 — wf1 — EPOCH 003   +0.72$/trade   236 trades   WR 41.1%   PF 1.11   2 min

PnL         +169.11$   cumul run     -400.38$   DD 7.7%   Sortino +0.081
point mort 38.6%  ->  ecart +2.5 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +7.6 pt   (reference -5.1 pt sur 2 epochs)
vs epoch precedente : +2.82$/trade
sens    LONG    51W/65  L  44.0%    +218.72$   |   SHORT   46W/74  L  38.3%     -49.61$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0007 (1x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 9.4e-06
train   PnL   +189.41$  1300 trades  WR 39.2%  PF 1.02   ->  ecart train-val -1.9 pt
. ACTOR GELE : warmup du critic, gradient 9.4e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0007 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.72$) sous l'erreur-type de cette epoch (0.92$) — non separable de zero.
temps : collecte 36s  PPO 46s  calib 12s  validation 37s

## 06:17:04 — wf1 — EPOCH 001   -0.59$/trade   250 trades   WR 36.0%   PF 0.91   2 min

PnL         -148.26$   cumul run     -148.26$   DD 10.3%   Sortino -0.065
point mort 38.2%  ->  ecart -2.2 pt (+/- 3.0 au mieux)
sens    LONG    67W/112 L  37.4%     -12.65$   |   SHORT   23W/48  L  32.4%    -135.61$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0000 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 1.1e-06
train   PnL   +202.93$  1295 trades  WR 39.7%  PF 1.02   ->  ecart train-val +3.7 pt
. ACTOR GELE : warmup du critic, gradient 1.1e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. gain par trade (-0.59$) sous l'erreur-type de cette epoch (0.82$) — non separable de zero.
temps : collecte 39s  PPO 43s  calib 7s  validation 42s

## 06:17:04 — wf1 — EPOCH 002   -2.11$/trade   200 trades   WR 34.0%   PF 0.71   2 min

PnL         -421.23$   cumul run     -569.49$   DD 7.2%   Sortino -0.226
point mort 42.0%  ->  ecart -8.0 pt (+/- 3.3 au mieux)
vs epoch precedente : -1.51$/trade
sens    LONG    49W/84  L  36.8%    -198.41$   |   SHORT   19W/48  L  28.4%    -222.82$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0002 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 4.7e-06
train   PnL   +472.18$  1263 trades  WR 39.8%  PF 1.06   ->  ecart train-val +5.8 pt
. ACTOR GELE : warmup du critic, gradient 4.7e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0002 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 36s  PPO 45s  calib 8s  validation 45s

## 06:17:04 — wf1 — EPOCH 003   +0.72$/trade   236 trades   WR 41.1%   PF 1.11   2 min

PnL         +169.11$   cumul run     -400.38$   DD 7.7%   Sortino +0.081
point mort 38.6%  ->  ecart +2.5 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +7.6 pt   (reference -5.1 pt sur 2 epochs)
vs epoch precedente : +2.82$/trade
sens    LONG    51W/65  L  44.0%    +218.72$   |   SHORT   46W/74  L  38.3%     -49.61$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0007 (1x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 9.4e-06
train   PnL   +189.41$  1300 trades  WR 39.2%  PF 1.02   ->  ecart train-val -1.9 pt
. ACTOR GELE : warmup du critic, gradient 9.4e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0007 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.72$) sous l'erreur-type de cette epoch (0.92$) — non separable de zero.
temps : collecte 36s  PPO 46s  calib 12s  validation 37s

## 06:17:44 — wf1 — EPOCH 004   -0.05$/trade   220 trades   WR 38.6%   PF 0.99   2 min

PnL          -11.26$   cumul run     -411.64$   DD 6.5%   Sortino -0.006
point mort 38.9%  ->  ecart -0.3 pt (+/- 3.3 au mieux)
vs politique gelee du meme run : +2.3 pt   (reference -2.6 pt sur 3 epochs)
vs epoch precedente : -0.77$/trade
sens    LONG    37W/64  L  36.6%     -74.23$   |   SHORT   48W/71  L  40.3%     +62.97$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 6.8e-06
train   PnL   +110.60$  1251 trades  WR 38.5%  PF 1.01   ->  ecart train-val -0.1 pt
. ACTOR GELE : warmup du critic, gradient 6.8e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le SHORT rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.05$) sous l'erreur-type de cette epoch (0.70$) — non separable de zero.
temps : collecte 37s  PPO 44s  calib 8s  validation 53s

## 06:20:45 — wf1 — EPOCH 005   +0.46$/trade   114 trades   WR 38.6%   PF 1.07   3 min

PnL          +52.94$   cumul run     -358.70$   DD 5.8%   Sortino +0.051
point mort 37.0%  ->  ecart +1.6 pt (+/- 4.6 au mieux)
vs politique gelee du meme run : +3.6 pt   (reference -2.0 pt sur 4 epochs)
vs epoch precedente : +0.52$/trade
sens    LONG    31W/54  L  36.5%     +29.16$   |   SHORT   13W/16  L  44.8%     +23.78$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 32.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 2.8e-08
train   PnL   -266.86$  1293 trades  WR 38.5%  PF 0.97   ->  ecart train-val -0.1 pt
. ACTOR GELE : warmup du critic, gradient 2.8e-08. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. gain par trade (+0.46$) sous l'erreur-type de cette epoch (1.33$) — non separable de zero.
temps : collecte 40s  PPO 47s  calib 10s  validation 67s

## 06:22:45 — wf1 — EPOCH 006   -0.82$/trade   275 trades   WR 35.3%   PF 0.89   2 min

PnL         -225.07$   cumul run     -583.77$   DD 11.2%   Sortino -0.088
point mort 38.0%  ->  ecart -2.7 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -1.4 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -1.28$/trade
sens    LONG    65W/108 L  37.6%     +65.77$   |   SHORT   32W/70  L  31.4%    -290.84$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 27.5%  val 5.0%
politique  H 1.089/1.099 (-0.010)   etendue 0.2373 (237x le tremblement)   KL +0.0096   clipfrac 19.4%   g_actor 2.8e-01
train   PnL   +257.07$  1272 trades  WR 39.8%  PF 1.03   ->  ecart train-val +4.5 pt
. APPREND MAIS RESTE PLAT : le gradient passe (2.8e-01) mais l'entropie tient a 1.089 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.2373, soit 237x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.82$) sous l'erreur-type de cette epoch (0.87$) — non separable de zero.
temps : collecte 32s  PPO 54s  calib 10s  validation 35s

## 06:25:05 — wf1 — EPOCH 007   +0.06$/trade   275 trades   WR 38.9%   PF 1.01   2 min

PnL          +17.61$   cumul run     -566.16$   DD 8.3%   Sortino +0.007
point mort 38.7%  ->  ecart +0.2 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +1.5 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +0.88$/trade
sens    LONG    68W/94  L  42.0%    +237.73$   |   SHORT   39W/74  L  34.5%    -220.11$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 23.0%  val 5.0%
politique  H 1.077/1.099 (-0.012)   etendue 0.3390 (339x le tremblement)   KL +0.0101   clipfrac 18.3%   g_actor 2.5e-01
train   PnL   +596.04$  1275 trades  WR 40.9%  PF 1.07   ->  ecart train-val +2.0 pt
. etendue val 0.3390, soit 339x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.06$) sous l'erreur-type de cette epoch (0.80$) — non separable de zero.
temps : collecte 41s  PPO 47s  calib 11s  validation 44s

## 06:27:25 — wf1 — EPOCH 008   -0.06$/trade   254 trades   WR 38.6%   PF 0.99   2 min

PnL          -15.91$   cumul run     -582.07$   DD 8.8%   Sortino -0.007
point mort 38.8%  ->  ecart -0.2 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +1.1 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -0.13$/trade
sens    LONG    64W/96  L  40.0%    +180.70$   |   SHORT   34W/60  L  36.2%    -196.61$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 18.5%  val 5.0%
politique  H 1.056/1.099 (-0.021)   etendue 0.4294 (429x le tremblement)   KL +0.0257   clipfrac 26.6%   g_actor 2.0e-01
train   PnL  +1024.85$  1296 trades  WR 40.8%  PF 1.13   ->  ecart train-val +2.2 pt
. etendue val 0.4294, soit 429x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.06$) sous l'erreur-type de cette epoch (0.80$) — non separable de zero.
temps : collecte 39s  PPO 52s  calib 12s  validation 36s

## 06:29:25 — wf1 — EPOCH 009   +0.09$/trade   263 trades   WR 39.9%   PF 1.01   2 min

PnL          +24.01$   cumul run     -558.06$   DD 7.5%   Sortino +0.010
point mort 39.7%  ->  ecart +0.2 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +1.5 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +0.15$/trade
sens    LONG    68W/92  L  42.5%    +195.49$   |   SHORT   37W/66  L  35.9%    -171.48$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 14.0%  val 5.0%
politique  H 1.039/1.099 (-0.017)   etendue 0.4581 (458x le tremblement)   KL +0.0115   clipfrac 23.8%   g_actor 1.8e-01
train   PnL  +2239.82$  1193 trades  WR 45.0%  PF 1.32   ->  ecart train-val +5.1 pt
. etendue val 0.4581, soit 458x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.09$) sous l'erreur-type de cette epoch (1.16$) — non separable de zero.
temps : collecte 31s  PPO 43s  calib 9s  validation 35s

## 06:31:25 — wf1 — EPOCH 010   +0.72$/trade   249 trades   WR 41.4%   PF 1.11   2 min

PnL         +179.00$   cumul run     -379.06$   DD 7.7%   Sortino +0.081
point mort 38.9%  ->  ecart +2.5 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +3.8 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +0.63$/trade
sens    LONG    68W/90  L  43.0%    +266.51$   |   SHORT   35W/56  L  38.5%     -87.51$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 9.5%  val 5.0%
politique  H 1.022/1.099 (-0.017)   etendue 0.4807 (481x le tremblement)   KL +0.0285   clipfrac 25.3%   g_actor 1.6e-01
train   PnL  +2282.29$  1209 trades  WR 44.9%  PF 1.32   ->  ecart train-val +3.5 pt
. etendue val 0.4807, soit 481x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.72$) sous l'erreur-type de cette epoch (0.90$) — non separable de zero.
temps : collecte 33s  PPO 39s  calib 7s  validation 39s

## 06:33:25 — wf1 — EPOCH 011   -0.77$/trade   262 trades   WR 35.9%   PF 0.89   2 min

PnL         -202.66$   cumul run     -581.72$   DD 11.4%   Sortino -0.084
point mort 38.6%  ->  ecart -2.7 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -1.4 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -1.49$/trade
sens    LONG    55W/100 L  35.5%     -75.97$   |   SHORT   39W/68  L  36.4%    -126.69$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.007/1.099 (-0.015)   etendue 0.4886 (489x le tremblement)   KL +0.0357   clipfrac 25.9%   g_actor 1.6e-01
train   PnL  +3346.28$  1292 trades  WR 47.1%  PF 1.46   ->  ecart train-val +11.2 pt
. etendue val 0.4886, soit 489x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (-0.77$) sous l'erreur-type de cette epoch (0.84$) — non separable de zero.
temps : collecte 39s  PPO 40s  calib 7s  validation 43s

## 06:35:25 — wf1 — EPOCH 012   -1.02$/trade   256 trades   WR 36.3%   PF 0.86   2 min

PnL         -261.69$   cumul run     -843.41$   DD 9.8%   Sortino -0.110
point mort 39.9%  ->  ecart -3.6 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -2.3 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -0.25$/trade
sens    LONG    54W/87  L  38.3%     -36.31$   |   SHORT   39W/76  L  33.9%    -225.38$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.998/1.099 (-0.009)   etendue 0.4910 (491x le tremblement)   KL +0.0319   clipfrac 21.9%   g_actor 1.5e-01
train   PnL  +2900.50$  1212 trades  WR 47.4%  PF 1.42   ->  ecart train-val +11.1 pt
. etendue val 0.4910, soit 491x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 34s  PPO 37s  calib 6s  validation 38s

## 06:37:25 — wf1 — EPOCH 013   -0.10$/trade   286 trades   WR 39.2%   PF 0.99   2 min

PnL          -28.76$   cumul run     -872.17$   DD 10.5%   Sortino -0.011
point mort 39.4%  ->  ecart -0.2 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +1.1 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +0.92$/trade
sens    LONG    52W/83  L  38.5%     +21.03$   |   SHORT   60W/91  L  39.7%     -49.79$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.990/1.099 (-0.008)   etendue 0.4917 (492x le tremblement)   KL +0.0214   clipfrac 20.2%   g_actor 1.4e-01
train   PnL  +4367.46$  1203 trades  WR 50.7%  PF 1.69   ->  ecart train-val +11.5 pt
. etendue val 0.4917, soit 492x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.10$) sous l'erreur-type de cette epoch (1.21$) — non separable de zero.
temps : collecte 37s  PPO 38s  calib 6s  validation 34s

## 06:39:05 — wf1 — EPOCH 014   +1.66$/trade   279 trades   WR 43.4%   PF 1.26   2 min

PnL         +462.19$   cumul run     -409.98$   DD 7.9%   Sortino +0.188
point mort 37.8%  ->  ecart +5.6 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +6.9 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +1.76$/trade
sens    LONG    45W/61  L  42.5%    +167.37$   |   SHORT   76W/97  L  43.9%    +294.82$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.987/1.099 (-0.003)   etendue 0.4950 (495x le tremblement)   KL +0.0314   clipfrac 21.1%   g_actor 1.3e-01
train   PnL  +3417.77$  1244 trades  WR 48.0%  PF 1.49   ->  ecart train-val +4.6 pt
. etendue val 0.4950, soit 495x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 35s  PPO 38s  calib 7s  validation 34s

## 06:41:05 — wf1 — EPOCH 015   +0.81$/trade   262 trades   WR 42.4%   PF 1.12   2 min

PnL         +211.38$   cumul run     -198.60$   DD 10.4%   Sortino +0.091
point mort 39.6%  ->  ecart +2.8 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +4.1 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -0.85$/trade
sens    LONG    52W/61  L  46.0%    +193.91$   |   SHORT   59W/90  L  39.6%     +17.47$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.982/1.099 (-0.005)   etendue 0.4964 (496x le tremblement)   KL +0.0262   clipfrac 20.3%   g_actor 1.3e-01
train   PnL  +4419.86$  1203 trades  WR 50.3%  PF 1.71   ->  ecart train-val +7.9 pt
. etendue val 0.4964, soit 496x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (+0.81$) sous l'erreur-type de cette epoch (0.90$) — non separable de zero.
temps : collecte 36s  PPO 36s  calib 6s  validation 35s

## 06:43:05 — wf1 — EPOCH 016   -0.03$/trade   288 trades   WR 38.2%   PF 1.00   2 min

PnL           -7.63$   cumul run     -206.23$   DD 10.1%   Sortino -0.003
point mort 38.2%  ->  ecart +0.0 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +1.3 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -0.83$/trade
sens    LONG    53W/75  L  41.4%     +60.45$   |   SHORT   57W/103 L  35.6%     -68.09$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.969/1.099 (-0.013)   etendue 0.4985 (498x le tremblement)   KL +0.0328   clipfrac 16.9%   g_actor 1.1e-01
train   PnL  +4515.60$  1250 trades  WR 50.6%  PF 1.67   ->  ecart train-val +12.4 pt
. etendue val 0.4985, soit 498x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.03$) sous l'erreur-type de cette epoch (0.81$) — non separable de zero.
temps : collecte 35s  PPO 39s  calib 7s  validation 38s

## 06:45:06 — wf1 — EPOCH 017   +0.95$/trade   293 trades   WR 41.0%   PF 1.14   2 min

PnL         +278.80$   cumul run      +72.57$   DD 9.0%   Sortino +0.107
point mort 37.8%  ->  ecart +3.2 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +4.5 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +0.98$/trade
sens    LONG    53W/67  L  44.2%    +241.43$   |   SHORT   67W/106 L  38.7%     +37.38$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.969/1.099 (+0.000)   etendue 0.4991 (499x le tremblement)   KL +0.0343   clipfrac 18.0%   g_actor 1.2e-01
train   PnL  +4048.27$  1230 trades  WR 49.4%  PF 1.61   ->  ecart train-val +8.4 pt
. etendue val 0.4991, soit 499x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 34s  PPO 37s  calib 6s  validation 34s

## 06:46:46 — wf1 — EPOCH 018   -0.11$/trade   303 trades   WR 37.0%   PF 0.98   2 min

PnL          -33.69$   cumul run      +38.88$   DD 9.3%   Sortino -0.012
point mort 37.4%  ->  ecart -0.4 pt (+/- 2.8 au mieux)
vs politique gelee du meme run : +0.9 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -1.06$/trade
sens    LONG    55W/77  L  41.7%    +110.48$   |   SHORT   57W/114 L  33.3%    -144.17$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.969/1.099 (+0.000)   etendue 0.4984 (498x le tremblement)   KL +0.0232   clipfrac 17.2%   g_actor 1.2e-01
train   PnL  +4255.36$  1178 trades  WR 50.1%  PF 1.69   ->  ecart train-val +13.1 pt
. etendue val 0.4984, soit 498x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.11$) sous l'erreur-type de cette epoch (0.65$) — non separable de zero.
temps : collecte 36s  PPO 36s  calib 6s  validation 37s

## 06:48:46 — wf1 — EPOCH 019   +0.36$/trade   293 trades   WR 39.2%   PF 1.05   2 min

PnL         +105.40$   cumul run     +144.28$   DD 8.8%   Sortino +0.040
point mort 38.1%  ->  ecart +1.1 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +2.4 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +0.47$/trade
sens    LONG    51W/67  L  43.2%    +221.91$   |   SHORT   64W/111 L  36.6%    -116.51$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.967/1.099 (-0.002)   etendue 0.4988 (499x le tremblement)   KL +0.0268   clipfrac 17.4%   g_actor 1.1e-01
train   PnL  +4265.02$  1214 trades  WR 49.3%  PF 1.66   ->  ecart train-val +10.1 pt
. etendue val 0.4988, soit 499x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.36$) sous l'erreur-type de cette epoch (0.89$) — non separable de zero.
temps : collecte 38s  PPO 36s  calib 6s  validation 37s

## 06:50:46 — wf1 — EPOCH 020   +0.18$/trade   294 trades   WR 38.1%   PF 1.03   2 min

PnL          +52.78$   cumul run     +197.06$   DD 10.6%   Sortino +0.020
point mort 37.4%  ->  ecart +0.7 pt (+/- 2.8 au mieux)
vs politique gelee du meme run : +2.0 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -0.18$/trade
sens    LONG    53W/81  L  39.6%     +42.58$   |   SHORT   59W/101 L  36.9%     +10.20$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.970/1.099 (+0.003)   etendue 0.4987 (499x le tremblement)   KL +0.0291   clipfrac 18.5%   g_actor 1.2e-01
train   PnL  +4102.72$  1214 trades  WR 49.3%  PF 1.63   ->  ecart train-val +11.2 pt
. etendue val 0.4987, soit 499x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (+0.18$) sous l'erreur-type de cette epoch (0.73$) — non separable de zero.
temps : collecte 35s  PPO 38s  calib 6s  validation 37s

## 06:52:46 — wf1 — EPOCH 021   +0.07$/trade   296 trades   WR 37.5%   PF 1.01   2 min

PnL          +21.75$   cumul run     +218.81$   DD 8.1%   Sortino +0.008
point mort 37.3%  ->  ecart +0.2 pt (+/- 2.8 au mieux)
vs politique gelee du meme run : +1.5 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -0.11$/trade
sens    LONG    44W/67  L  39.6%    +109.11$   |   SHORT   67W/118 L  36.2%     -87.36$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.966/1.099 (-0.004)   etendue 0.4988 (499x le tremblement)   KL +0.0335   clipfrac 18.4%   g_actor 1.2e-01
train   PnL  +4071.05$  1163 trades  WR 50.5%  PF 1.66   ->  ecart train-val +13.0 pt
. etendue val 0.4988, soit 499x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.07$) sous l'erreur-type de cette epoch (0.89$) — non separable de zero.
temps : collecte 35s  PPO 36s  calib 6s  validation 37s

## 06:54:46 — wf1 — EPOCH 022   +0.21$/trade   268 trades   WR 39.2%   PF 1.03   2 min

PnL          +57.22$   cumul run     +276.03$   DD 6.4%   Sortino +0.024
point mort 38.5%  ->  ecart +0.7 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +2.0 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +0.14$/trade
sens    LONG    42W/61  L  40.8%     +48.16$   |   SHORT   63W/102 L  38.2%      +9.05$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.963/1.099 (-0.003)   etendue 0.4992 (499x le tremblement)   KL +0.0286   clipfrac 15.9%   g_actor 1.1e-01
train   PnL  +4513.12$  1163 trades  WR 51.1%  PF 1.75   ->  ecart train-val +11.9 pt
. etendue val 0.4992, soit 499x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (+0.21$) sous l'erreur-type de cette epoch (0.91$) — non separable de zero.
temps : collecte 36s  PPO 36s  calib 6s  validation 38s

## 06:56:46 — wf1 — EPOCH 023   -0.21$/trade   287 trades   WR 37.6%   PF 0.97   2 min

PnL          -59.12$   cumul run     +216.91$   DD 9.2%   Sortino -0.022
point mort 38.3%  ->  ecart -0.7 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +0.5 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -0.42$/trade
sens    LONG    44W/61  L  41.9%    +146.84$   |   SHORT   64W/118 L  35.2%    -205.95$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.962/1.099 (-0.001)   etendue 0.4993 (499x le tremblement)   KL +0.0243   clipfrac 13.4%   g_actor 1.1e-01
train   PnL  +4045.53$  1186 trades  WR 50.0%  PF 1.64   ->  ecart train-val +12.4 pt
. etendue val 0.4993, soit 499x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.21$) sous l'erreur-type de cette epoch (0.82$) — non separable de zero.
temps : collecte 35s  PPO 38s  calib 6s  validation 40s

## 06:58:46 — wf1 — EPOCH 024   +0.99$/trade   263 trades   WR 40.7%   PF 1.15   2 min

PnL         +260.56$   cumul run     +477.47$   DD 9.9%   Sortino +0.112
point mort 37.4%  ->  ecart +3.3 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +4.6 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +1.20$/trade
sens    LONG    48W/58  L  45.3%    +274.93$   |   SHORT   59W/98  L  37.6%     -14.37$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.961/1.099 (-0.001)   etendue 0.4994 (499x le tremblement)   KL +0.0344   clipfrac 14.5%   g_actor 1.1e-01
train   PnL  +4443.61$  1167 trades  WR 50.5%  PF 1.74   ->  ecart train-val +9.8 pt
. etendue val 0.4994, soit 499x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 38s  PPO 37s  calib 6s  validation 41s

## 07:00:46 — wf1 — EPOCH 025   +1.64$/trade   258 trades   WR 43.0%   PF 1.26   2 min

PnL         +423.66$   cumul run     +901.13$   DD 10.8%   Sortino +0.189
point mort 37.5%  ->  ecart +5.5 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +6.8 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +0.65$/trade
sens    LONG    56W/59  L  48.7%    +303.14$   |   SHORT   55W/88  L  38.5%    +120.52$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.957/1.099 (-0.004)   etendue 0.4995 (500x le tremblement)   KL +0.0171   clipfrac 12.2%   g_actor 1.2e-01
train   PnL  +4980.89$  1196 trades  WR 51.9%  PF 1.82   ->  ecart train-val +8.9 pt
. etendue val 0.4995, soit 500x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 36s  PPO 38s  calib 6s  validation 38s

## 07:02:46 — wf1 — EPOCH 026   +1.06$/trade   260 trades   WR 41.5%   PF 1.16   2 min

PnL         +275.31$   cumul run    +1176.44$   DD 9.6%   Sortino +0.119
point mort 38.0%  ->  ecart +3.5 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +4.8 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -0.58$/trade
sens    LONG    50W/60  L  45.5%    +232.28$   |   SHORT   58W/92  L  38.7%     +43.03$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.954/1.099 (-0.003)   etendue 0.4996 (500x le tremblement)   KL +0.0245   clipfrac 13.9%   g_actor 1.1e-01
train   PnL  +4932.21$  1181 trades  WR 51.3%  PF 1.83   ->  ecart train-val +9.8 pt
. etendue val 0.4996, soit 500x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 38s  PPO 38s  calib 7s  validation 44s

## 07:04:46 — wf1 — EPOCH 027   +1.39$/trade   253 trades   WR 43.5%   PF 1.21   2 min

PnL         +352.85$   cumul run    +1529.29$   DD 9.7%   Sortino +0.157
point mort 38.9%  ->  ecart +4.6 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +5.9 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +0.34$/trade
sens    LONG    59W/58  L  50.4%    +402.36$   |   SHORT   51W/85  L  37.5%     -49.51$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.949/1.099 (-0.005)   etendue 0.4997 (500x le tremblement)   KL +0.0281   clipfrac 13.6%   g_actor 1.0e-01
train   PnL  +4499.95$  1146 trades  WR 51.3%  PF 1.77   ->  ecart train-val +7.8 pt
. etendue val 0.4997, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 34s  PPO 38s  calib 6s  validation 40s

## 07:06:46 — wf1 — EPOCH 028   +2.42$/trade   260 trades   WR 44.6%   PF 1.39   2 min

PnL         +628.08$   cumul run    +2157.37$   DD 6.4%   Sortino +0.281
point mort 36.7%  ->  ecart +7.9 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +9.2 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +1.02$/trade
sens    LONG    63W/73  L  46.3%    +372.45$   |   SHORT   53W/71  L  42.7%    +255.64$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.944/1.099 (-0.005)   etendue 0.4996 (500x le tremblement)   KL +0.0175   clipfrac 12.0%   g_actor 1.6e-01
train   PnL  +3890.57$  1150 trades  WR 49.6%  PF 1.64   ->  ecart train-val +5.0 pt
. etendue val 0.4996, soit 500x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 36s  PPO 39s  calib 6s  validation 40s

## 07:08:46 — wf1 — EPOCH 029   +1.62$/trade   270 trades   WR 42.2%   PF 1.25   2 min

PnL         +437.60$   cumul run    +2594.97$   DD 8.7%   Sortino +0.184
point mort 36.9%  ->  ecart +5.3 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +6.6 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -0.79$/trade
sens    LONG    57W/70  L  44.9%    +304.19$   |   SHORT   57W/86  L  39.9%    +133.41$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.942/1.099 (-0.002)   etendue 0.4997 (500x le tremblement)   KL +0.0194   clipfrac 9.9%   g_actor 1.2e-01
train   PnL  +4904.30$  1117 trades  WR 52.9%  PF 1.89   ->  ecart train-val +10.7 pt
. etendue val 0.4997, soit 500x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 34s  PPO 38s  calib 6s  validation 40s

## 07:10:46 — wf1 — EPOCH 030   +1.57$/trade   273 trades   WR 42.1%   PF 1.24   2 min

PnL         +428.00$   cumul run    +3022.97$   DD 6.7%   Sortino +0.178
point mort 37.0%  ->  ecart +5.1 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +6.4 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -0.05$/trade
sens    LONG    59W/73  L  44.7%    +352.62$   |   SHORT   56W/85  L  39.7%     +75.38$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.940/1.099 (-0.002)   etendue 0.4996 (500x le tremblement)   KL +0.0216   clipfrac 12.0%   g_actor 1.2e-01
train   PnL  +4407.87$  1191 trades  WR 50.9%  PF 1.71   ->  ecart train-val +8.8 pt
. etendue val 0.4996, soit 500x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 36s  PPO 43s  calib 6s  validation 39s

## 07:12:47 — wf1 — EPOCH 031   +0.82$/trade   285 trades   WR 42.1%   PF 1.12   2 min

PnL         +233.81$   cumul run    +3256.78$   DD 7.6%   Sortino +0.091
point mort 39.4%  ->  ecart +2.7 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +4.0 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -0.75$/trade
sens    LONG    51W/59  L  46.4%    +276.39$   |   SHORT   69W/106 L  39.4%     -42.58$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.936/1.099 (-0.004)   etendue 0.4996 (500x le tremblement)   KL +0.0122   clipfrac 7.9%   g_actor 1.2e-01
train   PnL  +5813.44$  1148 trades  WR 54.4%  PF 2.07   ->  ecart train-val +12.3 pt
. etendue val 0.4996, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.82$) sous l'erreur-type de cette epoch (0.88$) — non separable de zero.
temps : collecte 35s  PPO 40s  calib 6s  validation 36s

## 07:14:47 — wf1 — EPOCH 032   +0.71$/trade   275 trades   WR 41.5%   PF 1.10   2 min

PnL         +193.96$   cumul run    +3450.74$   DD 8.3%   Sortino +0.078
point mort 39.2%  ->  ecart +2.3 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +3.6 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -0.12$/trade
sens    LONG    50W/67  L  42.7%    +165.37$   |   SHORT   64W/94  L  40.5%     +28.59$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.933/1.099 (-0.003)   etendue 0.4997 (500x le tremblement)   KL +0.0141   clipfrac 9.2%   g_actor 1.3e-01
train   PnL  +5370.20$  1184 trades  WR 52.6%  PF 1.92   ->  ecart train-val +11.1 pt
. etendue val 0.4997, soit 500x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (+0.71$) sous l'erreur-type de cette epoch (0.91$) — non separable de zero.
temps : collecte 35s  PPO 41s  calib 6s  validation 35s

## 07:16:47 — wf1 — EPOCH 033   +0.33$/trade   282 trades   WR 39.0%   PF 1.05   2 min

PnL          +93.84$   cumul run    +3544.58$   DD 8.0%   Sortino +0.036
point mort 37.9%  ->  ecart +1.1 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +2.4 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -0.37$/trade
sens    LONG    51W/74  L  40.8%    +143.03$   |   SHORT   59W/98  L  37.6%     -49.18$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.933/1.099 (+0.000)   etendue 0.4997 (500x le tremblement)   KL +0.0069   clipfrac 7.9%   g_actor 1.1e-01
train   PnL  +5930.41$  1201 trades  WR 53.3%  PF 2.02   ->  ecart train-val +14.3 pt
. etendue val 0.4997, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.33$) sous l'erreur-type de cette epoch (0.84$) — non separable de zero.
temps : collecte 36s  PPO 45s  calib 6s  validation 41s

## 07:19:07 — wf1 — EPOCH 034   +0.52$/trade   272 trades   WR 40.8%   PF 1.08   2 min

PnL         +141.19$   cumul run    +3685.77$   DD 8.5%   Sortino +0.057
point mort 39.0%  ->  ecart +1.8 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +3.1 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +0.19$/trade
sens    LONG    53W/75  L  41.4%    +175.81$   |   SHORT   58W/86  L  40.3%     -34.62$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.933/1.099 (+0.000)   etendue 0.4996 (500x le tremblement)   KL +0.0111   clipfrac 7.3%   g_actor 1.2e-01
train   PnL  +4037.46$  1183 trades  WR 50.5%  PF 1.64   ->  ecart train-val +9.7 pt
. etendue val 0.4996, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.52$) sous l'erreur-type de cette epoch (0.84$) — non separable de zero.
temps : collecte 39s  PPO 44s  calib 7s  validation 39s

## 07:21:07 — wf1 — EPOCH 035   +1.02$/trade   272 trades   WR 42.3%   PF 1.15   2 min

PnL         +277.50$   cumul run    +3963.27$   DD 7.4%   Sortino +0.114
point mort 38.9%  ->  ecart +3.4 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +4.7 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +0.50$/trade
sens    LONG    54W/67  L  44.6%    +225.95$   |   SHORT   61W/90  L  40.4%     +51.55$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.929/1.099 (-0.004)   etendue 0.4997 (500x le tremblement)   KL +0.0083   clipfrac 6.9%   g_actor 1.3e-01
train   PnL  +5263.45$  1136 trades  WR 52.6%  PF 1.95   ->  ecart train-val +10.3 pt
. etendue val 0.4997, soit 500x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 39s  PPO 40s  calib 6s  validation 37s

## 07:23:07 — wf1 — EPOCH 036   +1.01$/trade   283 trades   WR 42.0%   PF 1.15   2 min

PnL         +286.07$   cumul run    +4249.34$   DD 8.1%   Sortino +0.113
point mort 38.7%  ->  ecart +3.3 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +4.6 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -0.01$/trade
sens    LONG    55W/73  L  43.0%    +213.61$   |   SHORT   64W/91  L  41.3%     +72.46$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.925/1.099 (-0.004)   etendue 0.4997 (500x le tremblement)   KL +0.0066   clipfrac 5.4%   g_actor 1.4e-01
train   PnL  +6412.22$  1204 trades  WR 55.3%  PF 2.14   ->  ecart train-val +13.3 pt
. etendue val 0.4997, soit 500x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 36s  PPO 45s  calib 6s  validation 35s

## 07:25:07 — wf1 — EPOCH 037   +2.03$/trade   270 trades   WR 45.2%   PF 1.32   2 min

PnL         +547.28$   cumul run    +4796.62$   DD 7.7%   Sortino +0.234
point mort 38.4%  ->  ecart +6.8 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +8.0 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +1.02$/trade
sens    LONG    58W/72  L  44.6%    +280.17$   |   SHORT   64W/76  L  45.7%    +267.11$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.926/1.099 (+0.001)   etendue 0.4997 (500x le tremblement)   KL +0.0088   clipfrac 6.4%   g_actor 1.3e-01
train   PnL  +5568.52$  1134 trades  WR 54.8%  PF 2.02   ->  ecart train-val +9.6 pt
. etendue val 0.4997, soit 500x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 35s  PPO 41s  calib 6s  validation 36s

## 07:27:07 — wf1 — EPOCH 038   +0.91$/trade   266 trades   WR 41.4%   PF 1.14   2 min

PnL         +241.24$   cumul run    +5037.86$   DD 7.5%   Sortino +0.101
point mort 38.2%  ->  ecart +3.2 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +4.5 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -1.12$/trade
sens    LONG    57W/73  L  43.8%    +240.55$   |   SHORT   53W/83  L  39.0%      +0.68$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.925/1.099 (-0.001)   etendue 0.4998 (500x le tremblement)   KL +0.0034   clipfrac 6.8%   g_actor 1.3e-01
train   PnL  +6075.78$  1177 trades  WR 55.9%  PF 2.09   ->  ecart train-val +14.5 pt
. etendue val 0.4998, soit 500x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 37s  PPO 42s  calib 6s  validation 37s

## 07:29:07 — wf1 — EPOCH 039   -0.21$/trade   290 trades   WR 38.6%   PF 0.97   2 min

PnL          -60.31$   cumul run    +4977.55$   DD 7.7%   Sortino -0.022
point mort 39.3%  ->  ecart -0.7 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +0.5 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : -1.11$/trade
sens    LONG    57W/82  L  41.0%     +95.05$   |   SHORT   55W/96  L  36.4%    -155.37$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.922/1.099 (-0.003)   etendue 0.4997 (500x le tremblement)   KL +0.0027   clipfrac 3.8%   g_actor 1.4e-01
train   PnL  +4753.60$  1161 trades  WR 52.3%  PF 1.81   ->  ecart train-val +13.7 pt
. etendue val 0.4997, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.21$) sous l'erreur-type de cette epoch (0.82$) — non separable de zero.
temps : collecte 38s  PPO 39s  calib 6s  validation 36s

## 07:31:07 — wf1 — MODELE DEPLOYE (moyenne des poids)  EPOCH 040   +0.14$/trade   283 trades   WR 39.6%   PF 1.02   2 min

PnL          +39.85$   cumul run    +5017.40$   DD 7.2%   Sortino +0.015
point mort 39.1%  ->  ecart +0.5 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +1.8 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +0.35$/trade
sens    LONG    53W/76  L  41.1%    +168.51$   |   SHORT   59W/95  L  38.3%    -128.66$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.922/1.099 (+0.000)   etendue 0.4998 (500x le tremblement)   KL +0.0030   clipfrac 4.5%   g_actor 1.5e-01
train   PnL  +5464.89$  1145 trades  WR 55.0%  PF 1.99   ->  ecart train-val +15.4 pt
. etendue val 0.4998, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.14$) sous l'erreur-type de cette epoch (0.87$) — non separable de zero.
temps : collecte 36s  PPO 40s  calib 6s  validation 35s

## 07:32:27 — wf1 — EPOCH 041   +0.71$/trade   287 trades   WR 42.2%   PF 1.11   1 min

PnL         +204.16$   cumul run    +5221.56$   DD 7.7%   Sortino +0.079
point mort 39.6%  ->  ecart +2.6 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +3.8 pt   (reference -1.3 pt sur 5 epochs)
vs epoch precedente : +0.57$/trade
sens    LONG    59W/73  L  44.7%    +215.33$   |   SHORT   62W/93  L  40.0%     -11.17$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.922/1.099 (+0.000)   etendue 0.4997 (500x le tremblement)   KL +0.0030   clipfrac 0.0%   g_actor 0.0e+00
train   PnL  +6007.93$  1115 trades  WR 55.8%  PF 2.19   ->  ecart train-val +13.6 pt
. DESACCORD SUR L'ETAT DE L'ACTOR : l'epoch 41 est hors le warmup (5 epochs) mais son gradient vaut 0.0e+00. La configuration lue par la veille n'est peut-etre pas celle du run.
. etendue val 0.4997, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.71$) sous l'erreur-type de cette epoch (0.82$) — non separable de zero.
temps : collecte 37s  PPO 0s  calib 5s  validation 32s

## 07:34:47 — wf2 — EPOCH 001   -1.28$/trade   269 trades   WR 34.9%   PF 0.82   2 min

PnL         -343.19$   cumul run     -343.19$   DD 10.3%   Sortino -0.139
point mort 39.6%  ->  ecart -4.7 pt (+/- 2.9 au mieux)
sens    LONG    39W/54  L  41.9%     +45.38$   |   SHORT   55W/121 L  31.2%    -388.57$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.5e-06
train   PnL   -425.78$  1364 trades  WR 37.5%  PF 0.95   ->  ecart train-val +2.6 pt
. ACTOR GELE : warmup du critic, gradient 1.5e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 31s  PPO 42s  calib 6s  validation 27s

## 07:36:47 — wf2 — EPOCH 002   +1.82$/trade   253 trades   WR 45.1%   PF 1.30   2 min

PnL         +460.71$   cumul run     +117.52$   DD 7.2%   Sortino +0.217
point mort 38.7%  ->  ecart +6.4 pt (+/- 3.1 au mieux)
vs epoch precedente : +3.10$/trade
sens    LONG    70W/79  L  47.0%    +382.94$   |   SHORT   44W/60  L  42.3%     +77.77$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0003 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 7.2e-06
train   PnL    +31.79$  1279 trades  WR 39.4%  PF 1.00   ->  ecart train-val -5.7 pt
. ACTOR GELE : warmup du critic, gradient 7.2e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0003 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 32s  PPO 41s  calib 6s  validation 31s

## 07:38:28 — wf2 — EPOCH 003   -0.84$/trade   261 trades   WR 36.0%   PF 0.88   2 min

PnL         -219.81$   cumul run     -102.29$   DD 7.7%   Sortino -0.092
point mort 39.0%  ->  ecart -3.0 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -3.9 pt   (reference +0.9 pt sur 2 epochs)
vs epoch precedente : -2.66$/trade
sens    LONG    42W/52  L  44.7%     +57.10$   |   SHORT   52W/115 L  31.1%    -276.91$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 5.1e-06
train   PnL    -96.71$  1250 trades  WR 39.0%  PF 0.99   ->  ecart train-val +3.0 pt
. ACTOR GELE : warmup du critic, gradient 5.1e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 34s  PPO 42s  calib 6s  validation 29s

## 07:40:48 — wf2 — EPOCH 004   -0.06$/trade   134 trades   WR 40.3%   PF 0.99   2 min

PnL           -8.35$   cumul run     -110.64$   DD 5.5%   Sortino -0.007
point mort 40.5%  ->  ecart -0.2 pt (+/- 4.2 au mieux)
vs politique gelee du meme run : +0.2 pt   (reference -0.4 pt sur 3 epochs)
vs epoch precedente : +0.78$/trade
sens    LONG    31W/52  L  37.3%     -75.19$   |   SHORT   23W/28  L  45.1%     +66.83$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.5e-06
train   PnL   +237.87$  1280 trades  WR 38.7%  PF 1.03   ->  ecart train-val -1.6 pt
. ACTOR GELE : warmup du critic, gradient 1.5e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le SHORT rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.06$) sous l'erreur-type de cette epoch (1.09$) — non separable de zero.
temps : collecte 32s  PPO 42s  calib 6s  validation 55s

## 07:42:48 — wf2 — EPOCH 005   -0.24$/trade   240 trades   WR 39.2%   PF 0.96   2 min

PnL          -58.56$   cumul run     -169.20$   DD 5.5%   Sortino -0.027
point mort 40.1%  ->  ecart -0.9 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : -0.6 pt   (reference -0.4 pt sur 4 epochs)
vs epoch precedente : -0.18$/trade
sens    LONG    55W/75  L  42.3%     +86.39$   |   SHORT   39W/71  L  35.5%    -144.95$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 32.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.4e-06
train   PnL   -972.99$  1304 trades  WR 37.3%  PF 0.89   ->  ecart train-val -1.9 pt
. ACTOR GELE : warmup du critic, gradient 1.4e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.24$) sous l'erreur-type de cette epoch (0.79$) — non separable de zero.
temps : collecte 32s  PPO 44s  calib 11s  validation 36s

## 07:44:48 — wf2 — EPOCH 006   -2.03$/trade   262 trades   WR 31.7%   PF 0.73   2 min

PnL         -531.49$   cumul run     -700.69$   DD 7.6%   Sortino -0.215
point mort 38.8%  ->  ecart -7.1 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -6.7 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -1.78$/trade
sens    LONG    31W/46  L  40.3%     +68.59$   |   SHORT   52W/133 L  28.1%    -600.08$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 27.5%  val 5.0%
politique  H 1.090/1.099 (-0.009)   etendue 0.1869 (187x le tremblement)   KL +0.0081   clipfrac 20.5%   g_actor 2.1e-01
train   PnL   -132.92$  1307 trades  WR 39.0%  PF 0.98   ->  ecart train-val +7.3 pt
. APPREND MAIS RESTE PLAT : le gradient passe (2.1e-01) mais l'entropie tient a 1.090 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.1869, soit 187x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 35s  PPO 43s  calib 6s  validation 34s

## 07:46:48 — wf2 — EPOCH 007   -0.16$/trade   246 trades   WR 36.6%   PF 0.98   2 min

PnL          -38.15$   cumul run     -738.84$   DD 8.4%   Sortino -0.017
point mort 37.1%  ->  ecart -0.5 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +0.0 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +1.87$/trade
sens    LONG    34W/54  L  38.6%     +15.47$   |   SHORT   56W/102 L  35.4%     -53.62$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 23.0%  val 5.0%
politique  H 1.070/1.099 (-0.020)   etendue 0.3549 (355x le tremblement)   KL +0.0200   clipfrac 25.7%   g_actor 2.0e-01
train   PnL   +918.77$  1285 trades  WR 40.9%  PF 1.11   ->  ecart train-val +4.3 pt
. etendue val 0.3549, soit 355x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.16$) sous l'erreur-type de cette epoch (1.01$) — non separable de zero.
temps : collecte 31s  PPO 44s  calib 7s  validation 36s

## 07:48:48 — wf2 — EPOCH 008   -0.11$/trade   240 trades   WR 35.8%   PF 0.98   2 min

PnL          -26.30$   cumul run     -765.14$   DD 9.5%   Sortino -0.012
point mort 36.3%  ->  ecart -0.5 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : -0.0 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +0.05$/trade
sens    LONG    43W/59  L  42.2%    +151.90$   |   SHORT   43W/95  L  31.2%    -178.21$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 18.5%  val 5.0%
politique  H 1.056/1.099 (-0.014)   etendue 0.4076 (408x le tremblement)   KL +0.0193   clipfrac 25.3%   g_actor 1.9e-01
train   PnL  +1606.90$  1282 trades  WR 42.4%  PF 1.20   ->  ecart train-val +6.6 pt
. etendue val 0.4076, soit 408x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.11$) sous l'erreur-type de cette epoch (0.73$) — non separable de zero.
temps : collecte 32s  PPO 42s  calib 8s  validation 40s

## 07:50:48 — wf2 — EPOCH 009   -0.85$/trade   248 trades   WR 35.1%   PF 0.88   2 min

PnL         -210.95$   cumul run     -976.09$   DD 7.3%   Sortino -0.092
point mort 38.0%  ->  ecart -2.9 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -2.5 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -0.74$/trade
sens    LONG    39W/68  L  36.4%     -55.71$   |   SHORT   48W/93  L  34.0%    -155.24$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 14.0%  val 5.0%
politique  H 1.029/1.099 (-0.027)   etendue 0.4633 (463x le tremblement)   KL +0.0180   clipfrac 22.9%   g_actor 1.9e-01
train   PnL  +2134.82$  1271 trades  WR 45.0%  PF 1.28   ->  ecart train-val +9.9 pt
. etendue val 0.4633, soit 463x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (-0.85$) sous l'erreur-type de cette epoch (0.87$) — non separable de zero.
temps : collecte 33s  PPO 40s  calib 6s  validation 40s

## 07:52:48 — wf2 — EPOCH 010   -0.60$/trade   253 trades   WR 33.2%   PF 0.91   2 min

PnL         -152.98$   cumul run    -1129.07$   DD 8.3%   Sortino -0.066
point mort 35.3%  ->  ecart -2.1 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -1.6 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +0.25$/trade
sens    LONG    36W/76  L  32.1%     -29.93$   |   SHORT   48W/93  L  34.0%    -123.05$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 9.5%  val 5.0%
politique  H 1.016/1.099 (-0.013)   etendue 0.4739 (474x le tremblement)   KL +0.0171   clipfrac 22.4%   g_actor 1.6e-01
train   PnL  +2225.39$  1241 trades  WR 44.7%  PF 1.30   ->  ecart train-val +11.5 pt
. etendue val 0.4739, soit 474x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (-0.60$) sous l'erreur-type de cette epoch (0.84$) — non separable de zero.
temps : collecte 37s  PPO 36s  calib 6s  validation 37s

## 07:54:28 — wf2 — EPOCH 011   -1.11$/trade   272 trades   WR 33.8%   PF 0.85   2 min

PnL         -301.89$   cumul run    -1430.96$   DD 7.3%   Sortino -0.121
point mort 37.6%  ->  ecart -3.8 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -3.3 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -0.51$/trade
sens    LONG    41W/65  L  38.7%     +40.02$   |   SHORT   51W/115 L  30.7%    -341.91$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.008/1.099 (-0.008)   etendue 0.4848 (485x le tremblement)   KL +0.0351   clipfrac 26.7%   g_actor 1.4e-01
train   PnL  +1523.49$  1210 trades  WR 43.1%  PF 1.21   ->  ecart train-val +9.3 pt
. etendue val 0.4848, soit 485x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 35s  PPO 35s  calib 6s  validation 33s

## 07:56:28 — wf2 — EPOCH 012   -0.98$/trade   267 trades   WR 35.2%   PF 0.86   2 min

PnL         -262.02$   cumul run    -1692.98$   DD 9.8%   Sortino -0.107
point mort 38.7%  ->  ecart -3.5 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -3.0 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +0.13$/trade
sens    LONG    37W/51  L  42.0%    +165.51$   |   SHORT   57W/122 L  31.8%    -427.53$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.997/1.099 (-0.011)   etendue 0.4871 (487x le tremblement)   KL +0.0348   clipfrac 23.3%   g_actor 1.3e-01
train   PnL  +2335.00$  1226 trades  WR 44.9%  PF 1.33   ->  ecart train-val +9.7 pt
. etendue val 0.4871, soit 487x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 35s  PPO 36s  calib 6s  validation 33s

## 07:58:28 — wf2 — EPOCH 013   -2.85$/trade   284 trades   WR 29.6%   PF 0.63   2 min

PnL         -808.49$   cumul run    -2501.47$   DD 8.2%   Sortino -0.297
point mort 40.0%  ->  ecart -10.4 pt (+/- 2.7 au mieux)
vs politique gelee du meme run : -9.9 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -1.87$/trade
sens    LONG    29W/67  L  30.2%    -221.90$   |   SHORT   55W/133 L  29.3%    -586.58$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.989/1.099 (-0.008)   etendue 0.4926 (493x le tremblement)   KL +0.0367   clipfrac 24.3%   g_actor 1.2e-01
train   PnL  +3095.89$  1230 trades  WR 47.3%  PF 1.45   ->  ecart train-val +17.7 pt
. etendue val 0.4926, soit 493x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 37s  PPO 35s  calib 6s  validation 33s

## 08:00:08 — wf2 — EPOCH 014   -1.77$/trade   255 trades   WR 33.3%   PF 0.76   2 min

PnL         -450.33$   cumul run    -2951.80$   DD 9.6%   Sortino -0.189
point mort 39.7%  ->  ecart -6.4 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -5.9 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +1.08$/trade
sens    LONG    33W/55  L  37.5%     -22.46$   |   SHORT   52W/115 L  31.1%    -427.87$
actions B 0.2%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.986/1.099 (-0.003)   etendue 0.4906 (491x le tremblement)   KL +0.0470   clipfrac 24.8%   g_actor 1.3e-01
train   PnL  +2816.45$  1207 trades  WR 46.7%  PF 1.41   ->  ecart train-val +13.4 pt
. etendue val 0.4906, soit 491x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 36s  PPO 32s  calib 6s  validation 33s

## 08:01:48 — wf2 — EPOCH 015   -0.04$/trade   241 trades   WR 36.5%   PF 0.99   2 min

PnL          -10.21$   cumul run    -2962.01$   DD 8.6%   Sortino -0.005
point mort 36.7%  ->  ecart -0.2 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +0.2 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +1.72$/trade
sens    LONG    36W/55  L  39.6%    +142.43$   |   SHORT   52W/98  L  34.7%    -152.64$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.978/1.099 (-0.008)   etendue 0.4934 (493x le tremblement)   KL +0.0337   clipfrac 22.1%   g_actor 1.3e-01
train   PnL  +3359.01$  1210 trades  WR 49.5%  PF 1.50   ->  ecart train-val +13.0 pt
. etendue val 0.4934, soit 493x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.04$) sous l'erreur-type de cette epoch (0.56$) — non separable de zero.
temps : collecte 34s  PPO 34s  calib 6s  validation 35s

## 08:03:49 — wf2 — EPOCH 016   -1.21$/trade   244 trades   WR 33.6%   PF 0.83   2 min

PnL         -294.81$   cumul run    -3256.82$   DD 9.2%   Sortino -0.130
point mort 37.9%  ->  ecart -4.3 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -3.8 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -1.17$/trade
sens    LONG    29W/48  L  37.7%     +41.79$   |   SHORT   53W/114 L  31.7%    -336.60$
actions B 0.2%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.964/1.099 (-0.014)   etendue 0.4945 (494x le tremblement)   KL +0.0271   clipfrac 19.8%   g_actor 1.2e-01
train   PnL  +3635.63$  1201 trades  WR 48.4%  PF 1.56   ->  ecart train-val +14.8 pt
. etendue val 0.4945, soit 494x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 36s  PPO 33s  calib 6s  validation 40s

## 08:05:49 — wf2 — EPOCH 017   -0.07$/trade   250 trades   WR 36.8%   PF 0.99   2 min

PnL          -18.14$   cumul run    -3274.96$   DD 7.9%   Sortino -0.008
point mort 37.0%  ->  ecart -0.2 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +0.3 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +1.14$/trade
sens    LONG    30W/42  L  41.7%    +171.97$   |   SHORT   62W/116 L  34.8%    -190.11$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.972/1.099 (+0.008)   etendue 0.4911 (491x le tremblement)   KL +0.0205   clipfrac 20.1%   g_actor 1.2e-01
train   PnL  +3485.38$  1217 trades  WR 48.4%  PF 1.52   ->  ecart train-val +11.6 pt
. etendue val 0.4911, soit 491x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.07$) sous l'erreur-type de cette epoch (0.95$) — non separable de zero.
temps : collecte 34s  PPO 34s  calib 6s  validation 38s

## 08:07:49 — wf2 — EPOCH 018   -1.12$/trade   267 trades   WR 34.8%   PF 0.85   2 min

PnL         -298.13$   cumul run    -3573.09$   DD 9.6%   Sortino -0.119
point mort 38.6%  ->  ecart -3.8 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -3.3 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -1.04$/trade
sens    LONG    40W/57  L  41.2%     +72.33$   |   SHORT   53W/117 L  31.2%    -370.46$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.971/1.099 (-0.001)   etendue 0.4981 (498x le tremblement)   KL +0.0362   clipfrac 19.7%   g_actor 1.1e-01
train   PnL  +3652.78$  1249 trades  WR 48.4%  PF 1.53   ->  ecart train-val +13.6 pt
. etendue val 0.4981, soit 498x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 34s  PPO 37s  calib 7s  validation 37s

## 08:09:29 — wf2 — EPOCH 019   -1.15$/trade   268 trades   WR 35.4%   PF 0.84   2 min

PnL         -308.94$   cumul run    -3882.03$   DD 11.1%   Sortino -0.123
point mort 39.5%  ->  ecart -4.1 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -3.6 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -0.04$/trade
sens    LONG    39W/60  L  39.4%     +34.15$   |   SHORT   56W/113 L  33.1%    -343.09$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.962/1.099 (-0.009)   etendue 0.4984 (498x le tremblement)   KL +0.0271   clipfrac 18.3%   g_actor 1.1e-01
train   PnL  +4354.16$  1180 trades  WR 51.3%  PF 1.71   ->  ecart train-val +15.9 pt
. etendue val 0.4984, soit 498x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 32s  PPO 35s  calib 6s  validation 34s

## 08:11:29 — wf2 — EPOCH 020   -1.09$/trade   257 trades   WR 34.2%   PF 0.85   2 min

PnL         -281.23$   cumul run    -4163.26$   DD 8.9%   Sortino -0.117
point mort 38.0%  ->  ecart -3.8 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -3.3 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +0.06$/trade
sens    LONG    38W/62  L  38.0%      -5.07$   |   SHORT   50W/107 L  31.8%    -276.16$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.963/1.099 (+0.001)   etendue 0.5000 (500x le tremblement)   KL +0.0239   clipfrac 16.8%   g_actor 1.1e-01
train   PnL  +4869.38$  1211 trades  WR 51.9%  PF 1.78   ->  ecart train-val +17.7 pt
. etendue val 0.5000, soit 500x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 34s  PPO 37s  calib 6s  validation 35s

## 08:13:09 — wf2 — EPOCH 021   -0.79$/trade   261 trades   WR 34.5%   PF 0.89   2 min

PnL         -205.96$   cumul run    -4369.22$   DD 10.6%   Sortino -0.085
point mort 37.2%  ->  ecart -2.7 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -2.2 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +0.31$/trade
sens    LONG    39W/60  L  39.4%    +107.21$   |   SHORT   51W/111 L  31.5%    -313.17$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.963/1.099 (+0.000)   etendue 0.4998 (500x le tremblement)   KL +0.0337   clipfrac 18.3%   g_actor 1.0e-01
train   PnL  +2567.29$  1212 trades  WR 46.9%  PF 1.37   ->  ecart train-val +12.4 pt
. etendue val 0.4998, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.79$) sous l'erreur-type de cette epoch (0.87$) — non separable de zero.
temps : collecte 35s  PPO 36s  calib 6s  validation 40s

## 08:15:09 — wf2 — EPOCH 022   -0.49$/trade   274 trades   WR 37.2%   PF 0.93   2 min

PnL         -133.90$   cumul run    -4503.12$   DD 7.3%   Sortino -0.053
point mort 38.9%  ->  ecart -1.7 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -1.2 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +0.30$/trade
sens    LONG    36W/56  L  39.1%    +113.27$   |   SHORT   66W/116 L  36.3%    -247.18$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.964/1.099 (+0.001)   etendue 0.4991 (499x le tremblement)   KL +0.0195   clipfrac 17.9%   g_actor 1.1e-01
train   PnL  +4684.86$  1204 trades  WR 52.2%  PF 1.75   ->  ecart train-val +15.0 pt
. etendue val 0.4991, soit 499x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.49$) sous l'erreur-type de cette epoch (0.83$) — non separable de zero.
temps : collecte 34s  PPO 37s  calib 6s  validation 36s

## 08:17:09 — wf2 — EPOCH 023   -0.47$/trade   282 trades   WR 37.2%   PF 0.93   2 min

PnL         -133.33$   cumul run    -4636.45$   DD 7.9%   Sortino -0.052
point mort 38.9%  ->  ecart -1.7 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -1.3 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +0.02$/trade
sens    LONG    31W/47  L  39.7%     +92.99$   |   SHORT   74W/130 L  36.3%    -226.32$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.968/1.099 (+0.004)   etendue 0.4988 (499x le tremblement)   KL +0.0301   clipfrac 16.5%   g_actor 1.1e-01
train   PnL  +3890.43$  1246 trades  WR 48.6%  PF 1.57   ->  ecart train-val +11.4 pt
. etendue val 0.4988, soit 499x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.47$) sous l'erreur-type de cette epoch (0.80$) — non separable de zero.
temps : collecte 34s  PPO 38s  calib 6s  validation 36s

## 08:18:49 — wf2 — EPOCH 024   -0.39$/trade   264 trades   WR 36.0%   PF 0.95   2 min

PnL         -101.73$   cumul run    -4738.18$   DD 7.2%   Sortino -0.042
point mort 37.2%  ->  ecart -1.2 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -0.7 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +0.09$/trade
sens    LONG    33W/43  L  43.4%    +203.28$   |   SHORT   62W/126 L  33.0%    -305.01$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.962/1.099 (-0.006)   etendue 0.4991 (499x le tremblement)   KL +0.0250   clipfrac 15.3%   g_actor 9.8e-02
train   PnL  +4433.26$  1173 trades  WR 51.7%  PF 1.74   ->  ecart train-val +15.7 pt
. etendue val 0.4991, soit 499x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.39$) sous l'erreur-type de cette epoch (0.96$) — non separable de zero.
temps : collecte 33s  PPO 35s  calib 6s  validation 36s

## 08:20:49 — wf2 — EPOCH 025   -0.87$/trade   266 trades   WR 34.2%   PF 0.88   2 min

PnL         -231.52$   cumul run    -4969.70$   DD 7.3%   Sortino -0.095
point mort 37.1%  ->  ecart -2.9 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -2.5 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -0.49$/trade
sens    LONG    28W/55  L  33.7%     -23.69$   |   SHORT   63W/120 L  34.4%    -207.83$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.958/1.099 (-0.004)   etendue 0.4993 (499x le tremblement)   KL +0.0189   clipfrac 14.4%   g_actor 9.8e-02
train   PnL  +4821.17$  1219 trades  WR 52.1%  PF 1.77   ->  ecart train-val +17.9 pt
. etendue val 0.4993, soit 499x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 35s  PPO 37s  calib 6s  validation 36s

## 08:22:49 — wf2 — EPOCH 026   -1.83$/trade   278 trades   WR 32.4%   PF 0.76   2 min

PnL         -508.38$   cumul run    -5478.08$   DD 8.4%   Sortino -0.193
point mort 38.6%  ->  ecart -6.2 pt (+/- 2.8 au mieux)
vs politique gelee du meme run : -5.8 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -0.96$/trade
sens    LONG    29W/50  L  36.7%      +8.57$   |   SHORT   61W/138 L  30.7%    -516.95$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.956/1.099 (-0.002)   etendue 0.4995 (500x le tremblement)   KL +0.0237   clipfrac 16.2%   g_actor 9.3e-02
train   PnL  +3992.61$  1227 trades  WR 48.6%  PF 1.60   ->  ecart train-val +16.2 pt
. etendue val 0.4995, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 39s  PPO 38s  calib 6s  validation 34s

## 08:24:49 — wf2 — EPOCH 027   -1.46$/trade   268 trades   WR 33.6%   PF 0.80   2 min

PnL         -392.06$   cumul run    -5870.14$   DD 6.6%   Sortino -0.158
point mort 38.7%  ->  ecart -5.1 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -4.6 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +0.37$/trade
sens    LONG    39W/59  L  39.8%     -26.28$   |   SHORT   51W/119 L  30.0%    -365.79$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.954/1.099 (-0.002)   etendue 0.4995 (500x le tremblement)   KL +0.0215   clipfrac 13.2%   g_actor 1.1e-01
train   PnL  +4373.65$  1163 trades  WR 50.7%  PF 1.72   ->  ecart train-val +17.1 pt
. etendue val 0.4995, soit 500x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 34s  PPO 37s  calib 6s  validation 35s

## 08:26:29 — wf2 — EPOCH 028   -0.25$/trade   259 trades   WR 37.5%   PF 0.96   2 min

PnL          -65.39$   cumul run    -5935.53$   DD 8.7%   Sortino -0.028
point mort 38.4%  ->  ecart -0.9 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -0.4 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +1.21$/trade
sens    LONG    38W/45  L  45.8%    +148.91$   |   SHORT   59W/117 L  33.5%    -214.30$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.949/1.099 (-0.005)   etendue 0.4996 (500x le tremblement)   KL +0.0183   clipfrac 11.9%   g_actor 1.0e-01
train   PnL  +4501.92$  1163 trades  WR 51.2%  PF 1.76   ->  ecart train-val +13.7 pt
. etendue val 0.4996, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.25$) sous l'erreur-type de cette epoch (0.79$) — non separable de zero.
temps : collecte 35s  PPO 37s  calib 6s  validation 38s

## 08:28:29 — wf2 — EPOCH 029   -0.79$/trade   262 trades   WR 35.5%   PF 0.89   2 min

PnL         -207.79$   cumul run    -6143.32$   DD 9.7%   Sortino -0.087
point mort 38.2%  ->  ecart -2.7 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -2.2 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -0.54$/trade
sens    LONG    38W/65  L  36.9%     -23.22$   |   SHORT   55W/104 L  34.6%    -184.57$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.949/1.099 (+0.000)   etendue 0.5000 (500x le tremblement)   KL +0.0322   clipfrac 12.8%   g_actor 9.5e-02
train   PnL  +4545.63$  1162 trades  WR 52.4%  PF 1.77   ->  ecart train-val +16.9 pt
. etendue val 0.5000, soit 500x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (-0.79$) sous l'erreur-type de cette epoch (0.86$) — non separable de zero.
temps : collecte 34s  PPO 37s  calib 6s  validation 37s

## 08:30:29 — wf2 — EPOCH 030   -0.97$/trade   274 trades   WR 35.0%   PF 0.87   2 min

PnL         -264.60$   cumul run    -6407.92$   DD 8.3%   Sortino -0.105
point mort 38.3%  ->  ecart -3.3 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -2.8 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -0.17$/trade
sens    LONG    50W/73  L  40.7%     +98.61$   |   SHORT   46W/105 L  30.5%    -363.21$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.940/1.099 (-0.009)   etendue 0.5000 (500x le tremblement)   KL +0.0159   clipfrac 10.4%   g_actor 1.0e-01
train   PnL  +5700.31$  1177 trades  WR 54.2%  PF 1.99   ->  ecart train-val +19.2 pt
. etendue val 0.5000, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 34s  PPO 37s  calib 6s  validation 39s

## 08:32:30 — wf2 — EPOCH 031   -0.97$/trade   266 trades   WR 33.8%   PF 0.86   2 min

PnL         -258.79$   cumul run    -6666.71$   DD 9.7%   Sortino -0.106
point mort 37.3%  ->  ecart -3.5 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -3.0 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -0.01$/trade
sens    LONG    44W/71  L  38.3%     +16.65$   |   SHORT   46W/105 L  30.5%    -275.44$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.939/1.099 (-0.001)   etendue 0.5000 (500x le tremblement)   KL +0.0117   clipfrac 8.9%   g_actor 1.2e-01
train   PnL  +4977.87$  1150 trades  WR 52.4%  PF 1.87   ->  ecart train-val +18.6 pt
. etendue val 0.5000, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 36s  PPO 37s  calib 6s  validation 40s

## 08:34:30 — wf2 — EPOCH 032   -0.31$/trade   261 trades   WR 37.5%   PF 0.96   2 min

PnL          -80.31$   cumul run    -6747.02$   DD 6.5%   Sortino -0.033
point mort 38.5%  ->  ecart -1.0 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -0.5 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +0.67$/trade
sens    LONG    46W/62  L  42.6%     +68.57$   |   SHORT   52W/101 L  34.0%    -148.88$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.944/1.099 (+0.005)   etendue 0.4999 (500x le tremblement)   KL +0.0148   clipfrac 9.5%   g_actor 1.2e-01
train   PnL  +4267.56$  1168 trades  WR 50.4%  PF 1.70   ->  ecart train-val +12.9 pt
. etendue val 0.4999, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.31$) sous l'erreur-type de cette epoch (0.96$) — non separable de zero.
temps : collecte 32s  PPO 38s  calib 7s  validation 38s

## 08:36:30 — wf2 — EPOCH 033   -1.39$/trade   265 trades   WR 34.7%   PF 0.81   2 min

PnL         -367.44$   cumul run    -7114.46$   DD 8.8%   Sortino -0.148
point mort 39.6%  ->  ecart -4.9 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -4.4 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -1.08$/trade
sens    LONG    40W/67  L  37.4%     -95.75$   |   SHORT   52W/106 L  32.9%    -271.69$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.939/1.099 (-0.005)   etendue 0.4999 (500x le tremblement)   KL +0.0048   clipfrac 7.3%   g_actor 1.1e-01
train   PnL  +5965.89$  1175 trades  WR 54.8%  PF 2.07   ->  ecart train-val +20.1 pt
. etendue val 0.4999, soit 500x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 35s  PPO 39s  calib 7s  validation 37s

## 08:38:30 — wf2 — EPOCH 034   -0.11$/trade   254 trades   WR 38.6%   PF 0.98   2 min

PnL          -27.61$   cumul run    -7142.07$   DD 8.0%   Sortino -0.012
point mort 39.1%  ->  ecart -0.5 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +0.0 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +1.28$/trade
sens    LONG    42W/50  L  45.7%    +185.80$   |   SHORT   56W/106 L  34.6%    -213.41$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.945/1.099 (+0.006)   etendue 0.4996 (500x le tremblement)   KL +0.0069   clipfrac 7.9%   g_actor 1.1e-01
train   PnL  +4687.70$  1159 trades  WR 52.0%  PF 1.80   ->  ecart train-val +13.4 pt
. etendue val 0.4996, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.11$) sous l'erreur-type de cette epoch (0.69$) — non separable de zero.
temps : collecte 35s  PPO 38s  calib 6s  validation 38s

## 08:40:10 — wf2 — EPOCH 035   -0.64$/trade   247 trades   WR 35.2%   PF 0.91   2 min

PnL         -157.45$   cumul run    -7299.52$   DD 9.7%   Sortino -0.070
point mort 37.4%  ->  ecart -2.2 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -1.7 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -0.53$/trade
sens    LONG    39W/47  L  45.3%    +239.24$   |   SHORT   48W/113 L  29.8%    -396.69$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.947/1.099 (+0.002)   etendue 0.4996 (500x le tremblement)   KL +0.0034   clipfrac 6.0%   g_actor 1.1e-01
train   PnL  +4575.74$  1149 trades  WR 52.2%  PF 1.79   ->  ecart train-val +17.0 pt
. etendue val 0.4996, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.64$) sous l'erreur-type de cette epoch (0.89$) — non separable de zero.
temps : collecte 34s  PPO 37s  calib 6s  validation 38s

## 08:42:30 — wf2 — EPOCH 036   -0.86$/trade   269 trades   WR 35.3%   PF 0.88   2 min

PnL         -231.52$   cumul run    -7531.04$   DD 9.7%   Sortino -0.093
point mort 38.3%  ->  ecart -3.0 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -2.5 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -0.22$/trade
sens    LONG    44W/60  L  42.3%     +71.68$   |   SHORT   51W/114 L  30.9%    -303.20$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.941/1.099 (-0.006)   etendue 0.4996 (500x le tremblement)   KL +0.0063   clipfrac 6.1%   g_actor 1.0e-01
train   PnL  +5759.36$  1151 trades  WR 55.1%  PF 2.05   ->  ecart train-val +19.8 pt
. etendue val 0.4996, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 35s  PPO 38s  calib 7s  validation 43s

## 08:44:10 — wf2 — EPOCH 037   -1.28$/trade   278 trades   WR 34.2%   PF 0.83   2 min

PnL         -354.70$   cumul run    -7885.74$   DD 8.6%   Sortino -0.137
point mort 38.5%  ->  ecart -4.3 pt (+/- 2.8 au mieux)
vs politique gelee du meme run : -3.8 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -0.42$/trade
sens    LONG    39W/50  L  43.8%    +127.75$   |   SHORT   56W/133 L  29.6%    -482.45$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.949/1.099 (+0.008)   etendue 0.4996 (500x le tremblement)   KL +0.0039   clipfrac 5.8%   g_actor 1.3e-01
train   PnL  +3944.30$  1173 trades  WR 49.9%  PF 1.63   ->  ecart train-val +15.7 pt
. etendue val 0.4996, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 36s  PPO 38s  calib 6s  validation 38s

## 08:46:30 — wf2 — EPOCH 038   -0.89$/trade   273 trades   WR 34.4%   PF 0.88   2 min

PnL         -243.61$   cumul run    -8129.35$   DD 9.6%   Sortino -0.096
point mort 37.4%  ->  ecart -3.0 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -2.5 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +0.38$/trade
sens    LONG    39W/54  L  41.9%    +104.76$   |   SHORT   55W/125 L  30.6%    -348.37$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.946/1.099 (-0.003)   etendue 0.4997 (500x le tremblement)   KL +0.0024   clipfrac 5.4%   g_actor 1.1e-01
train   PnL  +5032.24$  1178 trades  WR 53.1%  PF 1.85   ->  ecart train-val +18.7 pt
. etendue val 0.4997, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 35s  PPO 39s  calib 7s  validation 39s

## 08:48:30 — wf2 — EPOCH 039   -1.25$/trade   271 trades   WR 33.2%   PF 0.83   2 min

PnL         -337.77$   cumul run    -8467.12$   DD 9.6%   Sortino -0.133
point mort 37.5%  ->  ecart -4.3 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -3.8 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -0.35$/trade
sens    LONG    32W/52  L  38.1%     +15.84$   |   SHORT   58W/129 L  31.0%    -353.61$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.937/1.099 (-0.009)   etendue 0.4997 (500x le tremblement)   KL +0.0029   clipfrac 3.4%   g_actor 1.0e-01
train   PnL  +5727.04$  1202 trades  WR 53.5%  PF 1.97   ->  ecart train-val +20.3 pt
. etendue val 0.4997, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 35s  PPO 43s  calib 10s  validation 37s

## 08:50:30 — wf2 — MODELE DEPLOYE (moyenne des poids)  EPOCH 040   -0.72$/trade   278 trades   WR 35.3%   PF 0.90   2 min

PnL         -200.05$   cumul run    -8667.17$   DD 9.6%   Sortino -0.078
point mort 37.7%  ->  ecart -2.4 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -1.9 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : +0.53$/trade
sens    LONG    35W/55  L  38.9%     +83.47$   |   SHORT   63W/125 L  33.5%    -283.52$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.947/1.099 (+0.010)   etendue 0.4996 (500x le tremblement)   KL +0.0039   clipfrac 4.7%   g_actor 1.4e-01
train   PnL  +4544.98$  1143 trades  WR 52.4%  PF 1.78   ->  ecart train-val +17.1 pt
. etendue val 0.4996, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.72$) sous l'erreur-type de cette epoch (0.85$) — non separable de zero.
temps : collecte 33s  PPO 41s  calib 10s  validation 35s

## 08:51:30 — wf2 — EPOCH 041   -0.99$/trade   258 trades   WR 35.3%   PF 0.86   1 min

PnL         -255.54$   cumul run    -8922.71$   DD 9.7%   Sortino -0.108
point mort 38.8%  ->  ecart -3.5 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -3.0 pt   (reference -0.5 pt sur 5 epochs)
vs epoch precedente : -0.27$/trade
sens    LONG    41W/50  L  45.1%    +121.00$   |   SHORT   50W/117 L  29.9%    -376.54$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.947/1.099 (+0.000)   etendue 0.4996 (500x le tremblement)   KL +0.0039   clipfrac 0.0%   g_actor 0.0e+00
train   PnL  +4015.64$  1151 trades  WR 51.1%  PF 1.66   ->  ecart train-val +15.8 pt
. DESACCORD SUR L'ETAT DE L'ACTOR : l'epoch 41 est hors le warmup (5 epochs) mais son gradient vaut 0.0e+00. La configuration lue par la veille n'est peut-etre pas celle du run.
. etendue val 0.4996, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 34s  PPO 0s  calib 5s  validation 34s

## 08:54:30 — wf3 — EPOCH 001   -1.32$/trade   218 trades   WR 35.8%   PF 0.81   2 min

PnL         -287.18$   cumul run     -287.18$   DD 6.4%   Sortino -0.145
point mort 40.8%  ->  ecart -5.0 pt (+/- 3.2 au mieux)
sens    LONG    24W/43  L  35.8%    -163.95$   |   SHORT   54W/97  L  35.8%    -123.23$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0002 (0x le tremblement)   KL -0.0001   clipfrac 0.0%   g_actor 5.2e-06
train   PnL  -1621.13$  1392 trades  WR 35.2%  PF 0.83   ->  ecart train-val -0.6 pt
. ACTOR GELE : warmup du critic, gradient 5.2e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0002 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GPU BRIDE : 480 MHz a 84 C. Les durees ne sont pas comparables entre epochs si la temperature derive.
temps : collecte 33s  PPO 43s  calib 7s  validation 35s

## 08:56:30 — wf3 — EPOCH 002   -2.06$/trade   237 trades   WR 31.6%   PF 0.73   2 min

PnL         -487.28$   cumul run     -774.46$   DD 6.9%   Sortino -0.218
point mort 38.8%  ->  ecart -7.2 pt (+/- 3.0 au mieux)
vs epoch precedente : -0.74$/trade
sens    LONG    32W/61  L  34.4%     -99.37$   |   SHORT   43W/101 L  29.9%    -387.92$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 4.6e-07
train   PnL   -838.93$  1294 trades  WR 37.6%  PF 0.90   ->  ecart train-val +6.0 pt
. ACTOR GELE : warmup du critic, gradient 4.6e-07. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 33s  PPO 42s  calib 6s  validation 40s

## 08:58:50 — wf3 — EPOCH 003   -0.39$/trade   202 trades   WR 37.1%   PF 0.94   2 min

PnL          -79.07$   cumul run     -853.53$   DD 7.1%   Sortino -0.044
point mort 38.6%  ->  ecart -1.5 pt (+/- 3.4 au mieux)
vs politique gelee du meme run : +4.6 pt   (reference -6.1 pt sur 2 epochs)
vs epoch precedente : +1.66$/trade
sens    LONG    15W/22  L  40.5%     +50.94$   |   SHORT   60W/105 L  36.4%    -130.02$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 8.3e-08
train   PnL   -873.99$  1414 trades  WR 36.4%  PF 0.91   ->  ecart train-val -0.7 pt
. ACTOR GELE : warmup du critic, gradient 8.3e-08. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.39$) sous l'erreur-type de cette epoch (0.91$) — non separable de zero.
temps : collecte 33s  PPO 57s  calib 11s  validation 38s

## 09:01:11 — wf3 — EPOCH 004   -1.23$/trade   211 trades   WR 32.2%   PF 0.83   2 min

PnL         -260.46$   cumul run    -1113.99$   DD 9.6%   Sortino -0.133
point mort 36.4%  ->  ecart -4.2 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +0.3 pt   (reference -4.5 pt sur 3 epochs)
vs epoch precedente : -0.84$/trade
sens    LONG    27W/50  L  35.1%      -8.51$   |   SHORT   41W/93  L  30.6%    -251.95$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 1.2e-08
train   PnL   -732.41$  1332 trades  WR 37.7%  PF 0.92   ->  ecart train-val +5.5 pt
. ACTOR GELE : warmup du critic, gradient 1.2e-08. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 32s  PPO 57s  calib 13s  validation 44s

## 09:03:31 — wf3 — EPOCH 005   +0.22$/trade   218 trades   WR 37.6%   PF 1.03   2 min

PnL          +47.14$   cumul run    -1066.85$   DD 7.3%   Sortino +0.024
point mort 36.9%  ->  ecart +0.7 pt (+/- 3.3 au mieux)
vs politique gelee du meme run : +5.1 pt   (reference -4.5 pt sur 4 epochs)
vs epoch precedente : +1.45$/trade
sens    LONG    48W/81  L  37.2%     -23.94$   |   SHORT   34W/55  L  38.2%     +71.08$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 32.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 2.2e-08
train   PnL   -351.95$  1354 trades  WR 37.6%  PF 0.96   ->  ecart train-val +0.0 pt
. ACTOR GELE : warmup du critic, gradient 2.2e-08. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le SHORT rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.22$) sous l'erreur-type de cette epoch (1.03$) — non separable de zero.
temps : collecte 36s  PPO 53s  calib 10s  validation 45s

## 09:06:11 — wf3 — EPOCH 006   +0.66$/trade   238 trades   WR 43.3%   PF 1.10   3 min

PnL         +157.49$   cumul run     -909.36$   DD 7.7%   Sortino +0.076
point mort 41.0%  ->  ecart +2.3 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +5.8 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +0.45$/trade
sens    LONG    47W/57  L  45.2%    +126.16$   |   SHORT   56W/78  L  41.8%     +31.33$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 27.5%  val 5.0%
politique  H 1.091/1.099 (-0.008)   etendue 0.2284 (228x le tremblement)   KL +0.0052   clipfrac 16.0%   g_actor 1.4e-01
train   PnL  -1000.69$  1356 trades  WR 37.4%  PF 0.89   ->  ecart train-val -5.9 pt
. APPREND MAIS RESTE PLAT : le gradient passe (1.4e-01) mais l'entropie tient a 1.091 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.2284, soit 228x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (+0.66$) sous l'erreur-type de cette epoch (0.91$) — non separable de zero.
temps : collecte 35s  PPO 75s  calib 17s  validation 31s

## 09:08:11 — wf3 — EPOCH 007   -0.14$/trade   233 trades   WR 38.6%   PF 0.98   2 min

PnL          -31.54$   cumul run     -940.90$   DD 9.7%   Sortino -0.015
point mort 39.1%  ->  ecart -0.5 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +2.9 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -0.80$/trade
sens    LONG    42W/59  L  41.6%     +87.57$   |   SHORT   48W/84  L  36.4%    -119.11$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 23.0%  val 5.0%
politique  H 1.078/1.099 (-0.013)   etendue 0.3061 (306x le tremblement)   KL +0.0104   clipfrac 19.3%   g_actor 1.4e-01
train   PnL   +729.37$  1319 trades  WR 40.6%  PF 1.09   ->  ecart train-val +2.0 pt
. etendue val 0.3061, soit 306x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.14$) sous l'erreur-type de cette epoch (0.90$) — non separable de zero.
temps : collecte 32s  PPO 44s  calib 6s  validation 30s

## 09:09:51 — wf3 — EPOCH 008   +0.31$/trade   246 trades   WR 39.8%   PF 1.05   2 min

PnL          +76.99$   cumul run     -863.91$   DD 12.5%   Sortino +0.036
point mort 38.7%  ->  ecart +1.1 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +4.6 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +0.45$/trade
sens    LONG    52W/78  L  40.0%      +7.96$   |   SHORT   46W/70  L  39.7%     +69.03$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 18.5%  val 5.0%
politique  H 1.061/1.099 (-0.017)   etendue 0.3614 (361x le tremblement)   KL +0.0165   clipfrac 22.2%   g_actor 1.6e-01
train   PnL  +2128.96$  1298 trades  WR 44.4%  PF 1.27   ->  ecart train-val +4.6 pt
. etendue val 0.3614, soit 361x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (+0.31$) sous l'erreur-type de cette epoch (0.84$) — non separable de zero.
temps : collecte 30s  PPO 41s  calib 6s  validation 28s

## 09:11:31 — wf3 — EPOCH 009   -0.22$/trade   233 trades   WR 39.1%   PF 0.97   2 min

PnL          -50.80$   cumul run     -914.71$   DD 7.2%   Sortino -0.024
point mort 39.8%  ->  ecart -0.7 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +2.8 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -0.53$/trade
sens    LONG    41W/61  L  40.2%     +34.15$   |   SHORT   50W/81  L  38.2%     -84.94$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 14.0%  val 5.0%
politique  H 1.046/1.099 (-0.015)   etendue 0.3926 (393x le tremblement)   KL +0.0230   clipfrac 24.3%   g_actor 1.4e-01
train   PnL  +1608.02$  1292 trades  WR 42.3%  PF 1.20   ->  ecart train-val +3.2 pt
. etendue val 0.3926, soit 393x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.22$) sous l'erreur-type de cette epoch (0.96$) — non separable de zero.
temps : collecte 29s  PPO 40s  calib 6s  validation 28s

## 09:13:31 — wf3 — EPOCH 010   +0.16$/trade   221 trades   WR 38.9%   PF 1.02   2 min

PnL          +35.80$   cumul run     -878.91$   DD 7.1%   Sortino +0.018
point mort 38.4%  ->  ecart +0.5 pt (+/- 3.3 au mieux)
vs politique gelee du meme run : +3.9 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +0.38$/trade
sens    LONG    41W/70  L  36.9%     -72.81$   |   SHORT   45W/65  L  40.9%    +108.61$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 9.5%  val 5.0%
politique  H 1.034/1.099 (-0.012)   etendue 0.4353 (435x le tremblement)   KL +0.0430   clipfrac 30.7%   g_actor 1.1e-01
train   PnL  +2021.09$  1304 trades  WR 44.2%  PF 1.26   ->  ecart train-val +5.3 pt
. etendue val 0.4353, soit 435x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le SHORT rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.16$) sous l'erreur-type de cette epoch (1.13$) — non separable de zero.
. clipfrac 30.7% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 34s  PPO 39s  calib 6s  validation 35s

## 09:15:11 — wf3 — EPOCH 011   -1.54$/trade   242 trades   WR 34.3%   PF 0.79   2 min

PnL         -371.81$   cumul run    -1250.72$   DD 8.6%   Sortino -0.166
point mort 39.8%  ->  ecart -5.5 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : -2.0 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -1.70$/trade
sens    LONG    49W/77  L  38.9%     -82.44$   |   SHORT   34W/82  L  29.3%    -289.37$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.026/1.099 (-0.008)   etendue 0.4536 (454x le tremblement)   KL +0.0367   clipfrac 29.6%   g_actor 9.9e-02
train   PnL  +1735.45$  1255 trades  WR 43.6%  PF 1.23   ->  ecart train-val +9.3 pt
. etendue val 0.4536, soit 454x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 33s  PPO 36s  calib 6s  validation 31s

## 09:17:11 — wf3 — EPOCH 012   -0.41$/trade   240 trades   WR 38.3%   PF 0.94   2 min

PnL          -98.82$   cumul run    -1349.54$   DD 8.4%   Sortino -0.046
point mort 39.8%  ->  ecart -1.5 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +1.9 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +1.12$/trade
sens    LONG    55W/83  L  39.9%     -43.86$   |   SHORT   37W/65  L  36.3%     -54.95$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.017/1.099 (-0.009)   etendue 0.4747 (475x le tremblement)   KL +0.0363   clipfrac 26.6%   g_actor 8.9e-02
train   PnL  +2514.65$  1295 trades  WR 45.4%  PF 1.33   ->  ecart train-val +7.1 pt
. etendue val 0.4747, soit 475x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (-0.41$) sous l'erreur-type de cette epoch (0.88$) — non separable de zero.
temps : collecte 34s  PPO 36s  calib 6s  validation 32s

## 09:18:51 — wf3 — EPOCH 013   -0.93$/trade   256 trades   WR 34.8%   PF 0.87   2 min

PnL         -237.69$   cumul run    -1587.23$   DD 7.5%   Sortino -0.101
point mort 38.0%  ->  ecart -3.2 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +0.3 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -0.52$/trade
sens    LONG    43W/82  L  34.4%     -95.74$   |   SHORT   46W/85  L  35.1%    -141.94$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.004/1.099 (-0.013)   etendue 0.4827 (483x le tremblement)   KL +0.0445   clipfrac 26.6%   g_actor 9.1e-02
train   PnL  +2694.84$  1300 trades  WR 45.2%  PF 1.35   ->  ecart train-val +10.4 pt
. etendue val 0.4827, soit 483x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 33s  PPO 38s  calib 6s  validation 32s

## 09:20:51 — wf3 — EPOCH 014   -1.37$/trade   248 trades   WR 33.5%   PF 0.81   2 min

PnL         -339.95$   cumul run    -1927.18$   DD 7.1%   Sortino -0.147
point mort 38.3%  ->  ecart -4.8 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -1.4 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -0.44$/trade
sens    LONG    36W/80  L  31.0%    -220.10$   |   SHORT   47W/85  L  35.6%    -119.84$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.995/1.099 (-0.009)   etendue 0.4914 (491x le tremblement)   KL +0.0433   clipfrac 24.6%   g_actor 8.3e-02
train   PnL  +3079.72$  1283 trades  WR 46.7%  PF 1.42   ->  ecart train-val +13.2 pt
. etendue val 0.4914, soit 491x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 32s  PPO 35s  calib 6s  validation 32s

## 09:22:31 — wf3 — EPOCH 015   -0.67$/trade   242 trades   WR 34.7%   PF 0.90   2 min

PnL         -161.96$   cumul run    -2089.14$   DD 7.4%   Sortino -0.074
point mort 37.1%  ->  ecart -2.4 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +1.0 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +0.70$/trade
sens    LONG    35W/67  L  34.3%     -58.82$   |   SHORT   49W/91  L  35.0%    -103.14$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.985/1.099 (-0.010)   etendue 0.4943 (494x le tremblement)   KL +0.0445   clipfrac 23.6%   g_actor 7.5e-02
train   PnL  +3150.91$  1222 trades  WR 47.3%  PF 1.46   ->  ecart train-val +12.6 pt
. etendue val 0.4943, soit 494x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (-0.67$) sous l'erreur-type de cette epoch (0.84$) — non separable de zero.
temps : collecte 32s  PPO 35s  calib 6s  validation 30s

## 09:24:11 — wf3 — EPOCH 016   +1.01$/trade   234 trades   WR 41.5%   PF 1.16   2 min

PnL         +237.34$   cumul run    -1851.80$   DD 6.2%   Sortino +0.116
point mort 37.9%  ->  ecart +3.6 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +7.0 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +1.68$/trade
sens    LONG    48W/63  L  43.2%    +164.05$   |   SHORT   49W/74  L  39.8%     +73.28$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.977/1.099 (-0.008)   etendue 0.4974 (497x le tremblement)   KL +0.0345   clipfrac 21.8%   g_actor 7.7e-02
train   PnL  +3831.47$  1224 trades  WR 48.9%  PF 1.57   ->  ecart train-val +7.4 pt
. etendue val 0.4974, soit 497x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 32s  PPO 33s  calib 6s  validation 28s

## 09:25:51 — wf3 — EPOCH 017   -0.70$/trade   261 trades   WR 35.6%   PF 0.90   2 min

PnL         -183.36$   cumul run    -2035.16$   DD 8.2%   Sortino -0.077
point mort 38.1%  ->  ecart -2.5 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +1.0 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -1.72$/trade
sens    LONG    43W/68  L  38.7%     -18.84$   |   SHORT   50W/100 L  33.3%    -164.52$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.969/1.099 (-0.008)   etendue 0.4985 (498x le tremblement)   KL +0.0387   clipfrac 22.3%   g_actor 7.1e-02
train   PnL  +4280.17$  1265 trades  WR 49.6%  PF 1.63   ->  ecart train-val +14.0 pt
. etendue val 0.4985, soit 498x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (-0.70$) sous l'erreur-type de cette epoch (0.85$) — non separable de zero.
temps : collecte 34s  PPO 33s  calib 6s  validation 30s

## 09:27:31 — wf3 — EPOCH 018   -1.03$/trade   257 trades   WR 34.2%   PF 0.86   2 min

PnL         -265.94$   cumul run    -2301.10$   DD 7.7%   Sortino -0.113
point mort 37.7%  ->  ecart -3.5 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -0.1 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -0.33$/trade
sens    LONG    37W/63  L  37.0%     -21.11$   |   SHORT   51W/106 L  32.5%    -244.83$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.964/1.099 (-0.005)   etendue 0.4991 (499x le tremblement)   KL +0.0360   clipfrac 21.4%   g_actor 7.5e-02
train   PnL  +3676.43$  1286 trades  WR 47.6%  PF 1.51   ->  ecart train-val +13.4 pt
. etendue val 0.4991, soit 499x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 36s  PPO 35s  calib 6s  validation 30s

## 09:29:12 — wf3 — EPOCH 019   -0.18$/trade   246 trades   WR 36.2%   PF 0.97   2 min

PnL          -45.04$   cumul run    -2346.14$   DD 11.4%   Sortino -0.020
point mort 36.9%  ->  ecart -0.7 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +2.8 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +0.85$/trade
sens    LONG    37W/50  L  42.5%    +140.45$   |   SHORT   52W/107 L  32.7%    -185.48$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.969/1.099 (+0.005)   etendue 0.4988 (499x le tremblement)   KL +0.0303   clipfrac 19.6%   g_actor 7.7e-02
train   PnL  +4436.91$  1270 trades  WR 50.0%  PF 1.65   ->  ecart train-val +13.8 pt
. etendue val 0.4988, soit 499x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.18$) sous l'erreur-type de cette epoch (0.79$) — non separable de zero.
temps : collecte 33s  PPO 34s  calib 6s  validation 30s

## 09:30:52 — wf3 — EPOCH 020   +0.03$/trade   239 trades   WR 37.7%   PF 1.00   2 min

PnL           +6.03$   cumul run    -2340.11$   DD 6.9%   Sortino +0.003
point mort 37.7%  ->  ecart +0.0 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +3.5 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +0.21$/trade
sens    LONG    31W/48  L  39.2%      +3.00$   |   SHORT   59W/101 L  36.9%      +3.03$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.960/1.099 (-0.009)   etendue 0.4992 (499x le tremblement)   KL +0.0452   clipfrac 17.8%   g_actor 8.0e-02
train   PnL  +3912.05$  1295 trades  WR 49.4%  PF 1.54   ->  ecart train-val +11.7 pt
. etendue val 0.4992, soit 499x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (+0.03$) sous l'erreur-type de cette epoch (0.92$) — non separable de zero.
temps : collecte 32s  PPO 32s  calib 6s  validation 28s

## 09:32:52 — wf3 — EPOCH 021   -0.01$/trade   263 trades   WR 38.8%   PF 1.00   2 min

PnL           -3.28$   cumul run    -2343.39$   DD 7.1%   Sortino -0.001
point mort 38.8%  ->  ecart +0.0 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +3.5 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -0.04$/trade
sens    LONG    37W/51  L  42.0%     +90.65$   |   SHORT   65W/110 L  37.1%     -93.93$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.957/1.099 (-0.003)   etendue 0.4993 (499x le tremblement)   KL +0.0299   clipfrac 17.1%   g_actor 6.7e-02
train   PnL  +4704.44$  1256 trades  WR 51.1%  PF 1.71   ->  ecart train-val +12.3 pt
. etendue val 0.4993, soit 499x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.01$) sous l'erreur-type de cette epoch (0.84$) — non separable de zero.
temps : collecte 33s  PPO 34s  calib 6s  validation 31s

## 09:34:32 — wf3 — EPOCH 022   -0.39$/trade   258 trades   WR 37.6%   PF 0.94   2 min

PnL         -100.34$   cumul run    -2443.73$   DD 6.9%   Sortino -0.043
point mort 39.1%  ->  ecart -1.5 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +2.0 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -0.38$/trade
sens    LONG    42W/62  L  40.4%      +3.54$   |   SHORT   55W/99  L  35.7%    -103.87$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.957/1.099 (+0.000)   etendue 0.4995 (500x le tremblement)   KL +0.0273   clipfrac 16.0%   g_actor 6.9e-02
train   PnL  +4359.78$  1231 trades  WR 50.6%  PF 1.67   ->  ecart train-val +13.0 pt
. etendue val 0.4995, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.39$) sous l'erreur-type de cette epoch (0.80$) — non separable de zero.
temps : collecte 33s  PPO 35s  calib 6s  validation 31s

## 09:36:12 — wf3 — EPOCH 023   -0.70$/trade   257 trades   WR 36.2%   PF 0.90   2 min

PnL         -180.23$   cumul run    -2623.96$   DD 8.7%   Sortino -0.077
point mort 38.7%  ->  ecart -2.5 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +1.0 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -0.31$/trade
sens    LONG    43W/78  L  35.5%     -50.09$   |   SHORT   50W/86  L  36.8%    -130.14$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.954/1.099 (-0.003)   etendue 0.4995 (500x le tremblement)   KL +0.0372   clipfrac 16.5%   g_actor 6.9e-02
train   PnL  +4025.34$  1213 trades  WR 49.0%  PF 1.62   ->  ecart train-val +12.8 pt
. etendue val 0.4995, soit 500x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (-0.70$) sous l'erreur-type de cette epoch (0.85$) — non separable de zero.
temps : collecte 32s  PPO 36s  calib 6s  validation 31s

## 09:37:52 — wf3 — EPOCH 024   +0.09$/trade   279 trades   WR 38.0%   PF 1.01   2 min

PnL          +24.75$   cumul run    -2599.21$   DD 9.0%   Sortino +0.010
point mort 37.8%  ->  ecart +0.2 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +3.7 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +0.79$/trade
sens    LONG    49W/65  L  43.0%    +186.96$   |   SHORT   57W/108 L  34.5%    -162.20$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.959/1.099 (+0.005)   etendue 0.4995 (500x le tremblement)   KL +0.0279   clipfrac 15.8%   g_actor 8.4e-02
train   PnL  +3633.86$  1238 trades  WR 48.4%  PF 1.53   ->  ecart train-val +10.4 pt
. etendue val 0.4995, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.09$) sous l'erreur-type de cette epoch (1.10$) — non separable de zero.
temps : collecte 35s  PPO 35s  calib 7s  validation 30s

## 09:39:52 — wf3 — EPOCH 025   +0.25$/trade   255 trades   WR 38.8%   PF 1.04   2 min

PnL          +62.80$   cumul run    -2536.41$   DD 8.0%   Sortino +0.028
point mort 37.9%  ->  ecart +0.9 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +4.3 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +0.16$/trade
sens    LONG    43W/62  L  41.0%    +103.77$   |   SHORT   56W/94  L  37.3%     -40.97$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.953/1.099 (-0.006)   etendue 0.4996 (500x le tremblement)   KL +0.0297   clipfrac 13.6%   g_actor 6.6e-02
train   PnL  +4694.90$  1265 trades  WR 50.8%  PF 1.70   ->  ecart train-val +12.0 pt
. etendue val 0.4996, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.25$) sous l'erreur-type de cette epoch (0.81$) — non separable de zero.
temps : collecte 32s  PPO 37s  calib 6s  validation 31s

## 09:41:32 — wf3 — EPOCH 026   +0.09$/trade   261 trades   WR 39.1%   PF 1.01   2 min

PnL          +22.45$   cumul run    -2513.96$   DD 8.3%   Sortino +0.010
point mort 38.8%  ->  ecart +0.3 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +3.7 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -0.16$/trade
sens    LONG    44W/58  L  43.1%     +90.48$   |   SHORT   58W/101 L  36.5%     -68.03$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.953/1.099 (+0.000)   etendue 0.4997 (500x le tremblement)   KL +0.0235   clipfrac 14.5%   g_actor 7.3e-02
train   PnL  +5763.95$  1294 trades  WR 53.3%  PF 1.87   ->  ecart train-val +14.2 pt
. etendue val 0.4997, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.09$) sous l'erreur-type de cette epoch (1.10$) — non separable de zero.
temps : collecte 33s  PPO 40s  calib 7s  validation 34s

## 09:43:32 — wf3 — EPOCH 027   +0.62$/trade   280 trades   WR 38.9%   PF 1.09   2 min

PnL         +172.87$   cumul run    -2341.09$   DD 9.4%   Sortino +0.069
point mort 36.9%  ->  ecart +2.0 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +5.4 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +0.53$/trade
sens    LONG    46W/65  L  41.4%    +125.24$   |   SHORT   63W/106 L  37.3%     +47.63$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.955/1.099 (+0.002)   etendue 0.4996 (500x le tremblement)   KL +0.0244   clipfrac 13.8%   g_actor 6.6e-02
train   PnL  +3902.11$  1257 trades  WR 49.9%  PF 1.56   ->  ecart train-val +11.0 pt
. etendue val 0.4996, soit 500x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (+0.62$) sous l'erreur-type de cette epoch (0.89$) — non separable de zero.
temps : collecte 35s  PPO 36s  calib 6s  validation 31s

## 09:45:12 — wf3 — EPOCH 028   +0.91$/trade   263 trades   WR 40.7%   PF 1.14   2 min

PnL         +238.65$   cumul run    -2102.44$   DD 8.1%   Sortino +0.103
point mort 37.6%  ->  ecart +3.1 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +6.6 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +0.29$/trade
sens    LONG    50W/69  L  42.0%    +229.36$   |   SHORT   57W/87  L  39.6%      +9.29$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.950/1.099 (-0.005)   etendue 0.4997 (500x le tremblement)   KL +0.0223   clipfrac 14.0%   g_actor 6.9e-02
train   PnL  +4145.65$  1231 trades  WR 49.4%  PF 1.63   ->  ecart train-val +8.7 pt
. etendue val 0.4997, soit 500x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 33s  PPO 37s  calib 6s  validation 34s

## 09:47:12 — wf3 — EPOCH 029   +0.47$/trade   278 trades   WR 38.1%   PF 1.07   2 min

PnL         +129.47$   cumul run    -1972.97$   DD 8.1%   Sortino +0.052
point mort 36.5%  ->  ecart +1.6 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +5.0 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -0.44$/trade
sens    LONG    47W/79  L  37.3%      +4.27$   |   SHORT   59W/93  L  38.8%    +125.20$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.951/1.099 (+0.001)   etendue 0.4997 (500x le tremblement)   KL +0.0215   clipfrac 11.2%   g_actor 6.3e-02
train   PnL  +4365.89$  1221 trades  WR 50.6%  PF 1.67   ->  ecart train-val +12.5 pt
. etendue val 0.4997, soit 500x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (+0.47$) sous l'erreur-type de cette epoch (0.86$) — non separable de zero.
temps : collecte 33s  PPO 38s  calib 6s  validation 36s

## 09:49:12 — wf3 — EPOCH 030   +0.74$/trade   276 trades   WR 39.5%   PF 1.11   2 min

PnL         +203.49$   cumul run    -1769.48$   DD 10.3%   Sortino +0.082
point mort 37.0%  ->  ecart +2.5 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +5.9 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +0.27$/trade
sens    LONG    51W/70  L  42.1%    +182.13$   |   SHORT   58W/97  L  37.4%     +21.36$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.948/1.099 (-0.003)   etendue 0.4997 (500x le tremblement)   KL +0.0194   clipfrac 11.6%   g_actor 6.9e-02
train   PnL  +4467.54$  1247 trades  WR 50.9%  PF 1.67   ->  ecart train-val +11.4 pt
. etendue val 0.4997, soit 500x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (+0.74$) sous l'erreur-type de cette epoch (0.88$) — non separable de zero.
temps : collecte 34s  PPO 40s  calib 6s  validation 38s

## 09:51:12 — wf3 — EPOCH 031   +0.97$/trade   271 trades   WR 39.1%   PF 1.14   2 min

PnL         +263.26$   cumul run    -1506.22$   DD 8.2%   Sortino +0.108
point mort 36.0%  ->  ecart +3.1 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +6.5 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +0.23$/trade
sens    LONG    53W/75  L  41.4%    +208.57$   |   SHORT   53W/90  L  37.1%     +54.69$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.946/1.099 (-0.002)   etendue 0.4998 (500x le tremblement)   KL +0.0123   clipfrac 12.3%   g_actor 6.8e-02
train   PnL  +5112.68$  1193 trades  WR 52.6%  PF 1.85   ->  ecart train-val +13.5 pt
. etendue val 0.4998, soit 500x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 33s  PPO 36s  calib 6s  validation 40s

## 09:53:12 — wf3 — EPOCH 032   -1.43$/trade   277 trades   WR 33.2%   PF 0.81   2 min

PnL         -396.96$   cumul run    -1903.18$   DD 9.3%   Sortino -0.151
point mort 38.0%  ->  ecart -4.8 pt (+/- 2.8 au mieux)
vs politique gelee du meme run : -1.4 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -2.40$/trade
sens    LONG    42W/74  L  36.2%     -99.55$   |   SHORT   50W/111 L  31.1%    -297.41$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.941/1.099 (-0.005)   etendue 0.4998 (500x le tremblement)   KL +0.0142   clipfrac 9.2%   g_actor 7.0e-02
train   PnL  +4674.90$  1230 trades  WR 51.6%  PF 1.73   ->  ecart train-val +18.4 pt
. etendue val 0.4998, soit 500x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 33s  PPO 40s  calib 6s  validation 40s

## 09:54:52 — wf3 — EPOCH 033   +0.39$/trade   268 trades   WR 38.1%   PF 1.06   2 min

PnL         +104.73$   cumul run    -1798.45$   DD 6.9%   Sortino +0.043
point mort 36.7%  ->  ecart +1.4 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +4.8 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +1.82$/trade
sens    LONG    46W/73  L  38.7%     +81.61$   |   SHORT   56W/93  L  37.6%     +23.12$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.943/1.099 (+0.002)   etendue 0.4998 (500x le tremblement)   KL +0.0160   clipfrac 9.6%   g_actor 6.6e-02
train   PnL  +4453.99$  1222 trades  WR 51.6%  PF 1.69   ->  ecart train-val +13.5 pt
. etendue val 0.4998, soit 500x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (+0.39$) sous l'erreur-type de cette epoch (0.85$) — non separable de zero.
temps : collecte 32s  PPO 39s  calib 6s  validation 37s

## 09:56:53 — wf3 — EPOCH 034   +0.59$/trade   255 trades   WR 38.8%   PF 1.09   2 min

PnL         +149.38$   cumul run    -1649.07$   DD 7.3%   Sortino +0.065
point mort 36.8%  ->  ecart +2.0 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +5.4 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +0.20$/trade
sens    LONG    45W/67  L  40.2%    +140.79$   |   SHORT   54W/89  L  37.8%      +8.59$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.941/1.099 (-0.002)   etendue 0.4998 (500x le tremblement)   KL +0.0056   clipfrac 7.1%   g_actor 7.2e-02
train   PnL  +4816.15$  1210 trades  WR 52.6%  PF 1.77   ->  ecart train-val +13.8 pt
. etendue val 0.4998, soit 500x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (+0.59$) sous l'erreur-type de cette epoch (0.88$) — non separable de zero.
temps : collecte 34s  PPO 39s  calib 6s  validation 38s

## 09:58:53 — wf3 — EPOCH 035   -0.03$/trade   255 trades   WR 36.9%   PF 1.00   2 min

PnL           -6.95$   cumul run    -1656.02$   DD 8.0%   Sortino -0.003
point mort 36.9%  ->  ecart +0.0 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +3.5 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -0.61$/trade
sens    LONG    44W/64  L  40.7%    +110.04$   |   SHORT   50W/97  L  34.0%    -117.00$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.933/1.099 (-0.008)   etendue 0.4998 (500x le tremblement)   KL +0.0090   clipfrac 6.6%   g_actor 7.2e-02
train   PnL  +5731.52$  1208 trades  WR 52.7%  PF 1.96   ->  ecart train-val +15.8 pt
. etendue val 0.4998, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.03$) sous l'erreur-type de cette epoch (0.86$) — non separable de zero.
temps : collecte 35s  PPO 40s  calib 6s  validation 39s

## 10:00:53 — wf3 — EPOCH 036   -0.06$/trade   283 trades   WR 36.7%   PF 0.99   2 min

PnL          -17.27$   cumul run    -1673.29$   DD 7.3%   Sortino -0.007
point mort 37.0%  ->  ecart -0.3 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +3.2 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -0.03$/trade
sens    LONG    42W/70  L  37.5%     +53.57$   |   SHORT   62W/109 L  36.3%     -70.84$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.935/1.099 (+0.002)   etendue 0.4998 (500x le tremblement)   KL +0.0068   clipfrac 7.3%   g_actor 7.8e-02
train   PnL  +5153.98$  1259 trades  WR 51.9%  PF 1.79   ->  ecart train-val +15.2 pt
. etendue val 0.4998, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.06$) sous l'erreur-type de cette epoch (0.75$) — non separable de zero.
temps : collecte 34s  PPO 41s  calib 6s  validation 37s

## 10:02:53 — wf3 — EPOCH 037   -0.03$/trade   271 trades   WR 36.9%   PF 1.00   2 min

PnL           -7.89$   cumul run    -1681.18$   DD 8.4%   Sortino -0.003
point mort 36.9%  ->  ecart -0.0 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +3.4 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +0.03$/trade
sens    LONG    41W/70  L  36.9%     +37.01$   |   SHORT   59W/101 L  36.9%     -44.90$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.935/1.099 (+0.000)   etendue 0.4998 (500x le tremblement)   KL +0.0111   clipfrac 6.1%   g_actor 6.6e-02
train   PnL  +4452.64$  1244 trades  WR 50.6%  PF 1.68   ->  ecart train-val +13.7 pt
. etendue val 0.4998, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.03$) sous l'erreur-type de cette epoch (0.85$) — non separable de zero.
temps : collecte 37s  PPO 40s  calib 6s  validation 39s

## 10:04:53 — wf3 — EPOCH 038   -0.04$/trade   271 trades   WR 36.9%   PF 0.99   2 min

PnL           -9.90$   cumul run    -1691.08$   DD 8.3%   Sortino -0.004
point mort 37.1%  ->  ecart -0.2 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +3.2 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -0.01$/trade
sens    LONG    40W/69  L  36.7%      +9.10$   |   SHORT   60W/102 L  37.0%     -19.00$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.936/1.099 (+0.001)   etendue 0.4998 (500x le tremblement)   KL +0.0013   clipfrac 5.5%   g_actor 8.6e-02
train   PnL  +5957.70$  1217 trades  WR 54.1%  PF 2.01   ->  ecart train-val +17.2 pt
. etendue val 0.4998, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.04$) sous l'erreur-type de cette epoch (0.46$) — non separable de zero.
temps : collecte 33s  PPO 40s  calib 6s  validation 36s

## 10:06:53 — wf3 — EPOCH 039   +0.70$/trade   266 trades   WR 39.1%   PF 1.10   2 min

PnL         +187.46$   cumul run    -1503.62$   DD 7.7%   Sortino +0.079
point mort 36.9%  ->  ecart +2.2 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +5.7 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +0.74$/trade
sens    LONG    46W/74  L  38.3%     +61.41$   |   SHORT   58W/88  L  39.7%    +126.05$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.937/1.099 (+0.001)   etendue 0.4998 (500x le tremblement)   KL +0.0096   clipfrac 5.4%   g_actor 8.3e-02
train   PnL  +4921.05$  1212 trades  WR 52.6%  PF 1.79   ->  ecart train-val +13.5 pt
. etendue val 0.4998, soit 500x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (+0.70$) sous l'erreur-type de cette epoch (0.94$) — non separable de zero.
temps : collecte 34s  PPO 41s  calib 6s  validation 36s

## 10:08:53 — wf3 — MODELE DEPLOYE (moyenne des poids)  EPOCH 040   -0.51$/trade   267 trades   WR 36.0%   PF 0.93   2 min

PnL         -137.19$   cumul run    -1640.81$   DD 8.1%   Sortino -0.056
point mort 37.6%  ->  ecart -1.6 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +1.8 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : -1.22$/trade
sens    LONG    37W/68  L  35.2%     -45.00$   |   SHORT   59W/103 L  36.4%     -92.19$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.936/1.099 (-0.001)   etendue 0.4998 (500x le tremblement)   KL +0.0058   clipfrac 4.9%   g_actor 7.3e-02
train   PnL  +5169.88$  1231 trades  WR 52.2%  PF 1.82   ->  ecart train-val +16.2 pt
. etendue val 0.4998, soit 500x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (-0.51$) sous l'erreur-type de cette epoch (0.89$) — non separable de zero.
temps : collecte 34s  PPO 42s  calib 6s  validation 36s

## 10:10:13 — wf3 — EPOCH 041   +0.02$/trade   271 trades   WR 37.3%   PF 1.00   1 min

PnL           +6.21$   cumul run    -1634.60$   DD 8.2%   Sortino +0.003
point mort 37.3%  ->  ecart +0.0 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +3.5 pt   (reference -3.4 pt sur 5 epochs)
vs epoch precedente : +0.54$/trade
sens    LONG    44W/66  L  40.0%    +120.01$   |   SHORT   57W/104 L  35.4%    -113.80$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.936/1.099 (+0.000)   etendue 0.4998 (500x le tremblement)   KL +0.0058   clipfrac 0.0%   g_actor 0.0e+00
train   PnL  +5318.22$  1267 trades  WR 51.9%  PF 1.82   ->  ecart train-val +14.6 pt
. DESACCORD SUR L'ETAT DE L'ACTOR : l'epoch 41 est hors le warmup (5 epochs) mais son gradient vaut 0.0e+00. La configuration lue par la veille n'est peut-etre pas celle du run.
. etendue val 0.4998, soit 500x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.02$) sous l'erreur-type de cette epoch (0.84$) — non separable de zero.
temps : collecte 35s  PPO 0s  calib 5s  validation 36s

## 12:24:28 — wf1 — EPOCH 001   -0.77$/trade   263 trades   WR 37.6%   PF 0.89   1 min

PnL         -201.50$   cumul run     -201.50$   DD 9.6%   Sortino -0.084
point mort 40.4%  ->  ecart -2.8 pt (+/- 3.0 au mieux)
sens    LONG    57W/96  L  37.3%    -132.25$   |   SHORT   42W/68  L  38.2%     -69.25$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0000 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 7.9e-07
train   PnL  +1131.53$  1152 trades  WR 43.1%  PF 1.16   ->  ecart train-val +5.5 pt
. ACTOR GELE : warmup du critic, gradient 7.9e-07. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. gain par trade (-0.77$) sous l'erreur-type de cette epoch (0.83$) — non separable de zero.
temps : collecte 25s  PPO 22s  calib 3s  validation 28s

## 12:25:48 — wf1 — EPOCH 002   -0.04$/trade   267 trades   WR 34.1%   PF 0.99   1 min

PnL          -11.12$   cumul run     -212.62$   DD 8.5%   Sortino -0.005
point mort 34.3%  ->  ecart -0.2 pt (+/- 2.9 au mieux)
vs epoch precedente : +0.72$/trade
sens    LONG    48W/97  L  33.1%    +207.19$   |   SHORT   43W/79  L  35.2%    -218.32$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.8e-06
train   PnL   +628.14$  1147 trades  WR 42.5%  PF 1.09   ->  ecart train-val +8.4 pt
. ACTOR GELE : warmup du critic, gradient 1.8e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.04$) sous l'erreur-type de cette epoch (0.53$) — non separable de zero.
temps : collecte 26s  PPO 23s  calib 4s  validation 27s

## 12:27:08 — wf1 — EPOCH 003   +1.76$/trade   176 trades   WR 44.9%   PF 1.30   1 min

PnL         +309.52$   cumul run      +96.90$   DD 6.4%   Sortino +0.212
point mort 38.5%  ->  ecart +6.4 pt (+/- 3.7 au mieux)
vs politique gelee du meme run : +7.9 pt   (reference -1.5 pt sur 2 epochs)
vs epoch precedente : +1.80$/trade
sens    LONG    19W/15  L  55.9%    +278.37$   |   SHORT   60W/82  L  42.3%     +31.15$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 4.4e-07
train   PnL   +237.20$  1234 trades  WR 40.4%  PF 1.03   ->  ecart train-val -4.5 pt
. ACTOR GELE : warmup du critic, gradient 4.4e-07. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 27s  PPO 25s  calib 4s  validation 29s

## 12:28:48 — wf1 — EPOCH 004   +0.63$/trade   257 trades   WR 37.7%   PF 1.09   1 min

PnL         +162.60$   cumul run     +259.50$   DD 7.2%   Sortino +0.070
point mort 35.7%  ->  ecart +2.0 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +0.8 pt   (reference +1.1 pt sur 3 epochs)
vs epoch precedente : -1.13$/trade
sens    LONG    37W/52  L  41.6%    +289.64$   |   SHORT   60W/108 L  35.7%    -127.04$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.0e-08
train   PnL   +119.48$  1212 trades  WR 43.1%  PF 1.02   ->  ecart train-val +5.4 pt
. ACTOR GELE : warmup du critic, gradient 1.0e-08. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.63$) sous l'erreur-type de cette epoch (0.96$) — non separable de zero.
temps : collecte 28s  PPO 26s  calib 5s  validation 28s

## 12:30:08 — wf1 — EPOCH 005   +0.17$/trade   259 trades   WR 40.9%   PF 1.03   1 min

PnL          +44.88$   cumul run     +304.38$   DD 6.6%   Sortino +0.020
point mort 40.2%  ->  ecart +0.7 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : -0.6 pt   (reference +1.3 pt sur 4 epochs)
vs epoch precedente : -0.46$/trade
sens    LONG    47W/61  L  43.5%    +268.09$   |   SHORT   59W/92  L  39.1%    -223.21$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 32.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 7.1e-08
train   PnL  +2034.84$  1158 trades  WR 44.7%  PF 1.30   ->  ecart train-val +3.8 pt
. ACTOR GELE : warmup du critic, gradient 7.1e-08. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.17$) sous l'erreur-type de cette epoch (0.74$) — non separable de zero.
temps : collecte 28s  PPO 27s  calib 5s  validation 27s

## 12:31:48 — wf1 — EPOCH 006   -0.02$/trade   274 trades   WR 40.9%   PF 1.00   2 min

PnL           -6.22$   cumul run     +298.16$   DD 9.7%   Sortino -0.003
point mort 40.9%  ->  ecart +0.0 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -1.2 pt   (reference +1.2 pt sur 5 epochs)
vs epoch precedente : -0.20$/trade
sens    LONG    61W/79  L  43.6%    +176.76$   |   SHORT   51W/83  L  38.1%    -182.98$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 27.5%  val 5.0%
politique  H 1.092/1.099 (-0.007)   etendue 0.1942 (194x le tremblement)   KL +0.0064   clipfrac 13.7%   g_actor 1.6e-01
train   PnL   +767.43$  1214 trades  WR 43.4%  PF 1.10   ->  ecart train-val +2.5 pt
. APPREND MAIS RESTE PLAT : le gradient passe (1.6e-01) mais l'entropie tient a 1.092 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.1942, soit 194x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.02$) sous l'erreur-type de cette epoch (0.79$) — non separable de zero.
temps : collecte 29s  PPO 33s  calib 5s  validation 27s

## 12:33:28 — wf1 — EPOCH 007   +1.38$/trade   255 trades   WR 42.4%   PF 1.22   2 min

PnL         +351.93$   cumul run     +650.09$   DD 9.2%   Sortino +0.158
point mort 37.6%  ->  ecart +4.8 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +3.6 pt   (reference +1.2 pt sur 5 epochs)
vs epoch precedente : +1.40$/trade
sens    LONG    64W/78  L  45.1%    +341.27$   |   SHORT   44W/69  L  38.9%     +10.67$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 23.0%  val 5.0%
politique  H 1.077/1.099 (-0.015)   etendue 0.3381 (338x le tremblement)   KL +0.0146   clipfrac 19.5%   g_actor 1.5e-01
train   PnL  +1607.48$  1173 trades  WR 45.4%  PF 1.24   ->  ecart train-val +3.0 pt
. etendue val 0.3381, soit 338x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 29s  PPO 33s  calib 6s  validation 28s

## 12:34:48 — wf1 — EPOCH 008   -0.54$/trade   269 trades   WR 38.7%   PF 0.92   2 min

PnL         -146.00$   cumul run     +504.09$   DD 7.5%   Sortino -0.061
point mort 40.7%  ->  ecart -2.0 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -3.2 pt   (reference +1.2 pt sur 5 epochs)
vs epoch precedente : -1.92$/trade
sens    LONG    50W/80  L  38.5%    +130.99$   |   SHORT   54W/85  L  38.8%    -276.99$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 18.5%  val 5.0%
politique  H 1.067/1.099 (-0.010)   etendue 0.3763 (376x le tremblement)   KL +0.0184   clipfrac 21.6%   g_actor 1.7e-01
train   PnL  +2437.65$  1156 trades  WR 46.3%  PF 1.37   ->  ecart train-val +7.6 pt
. etendue val 0.3763, soit 376x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.54$) sous l'erreur-type de cette epoch (0.81$) — non separable de zero.
temps : collecte 31s  PPO 33s  calib 6s  validation 26s

## 12:36:48 — wf1 — EPOCH 009   +0.48$/trade   250 trades   WR 38.4%   PF 1.07   2 min

PnL         +119.59$   cumul run     +623.68$   DD 8.9%   Sortino +0.054
point mort 36.8%  ->  ecart +1.6 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +0.4 pt   (reference +1.2 pt sur 5 epochs)
vs epoch precedente : +1.02$/trade
sens    LONG    44W/71  L  38.3%    +253.58$   |   SHORT   52W/83  L  38.5%    -134.00$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 14.0%  val 5.0%
politique  H 1.057/1.099 (-0.010)   etendue 0.3824 (382x le tremblement)   KL +0.0225   clipfrac 25.3%   g_actor 1.6e-01
train   PnL  +1464.35$  1183 trades  WR 44.6%  PF 1.21   ->  ecart train-val +6.2 pt
. etendue val 0.3824, soit 382x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.48$) sous l'erreur-type de cette epoch (0.93$) — non separable de zero.
temps : collecte 32s  PPO 38s  calib 6s  validation 28s

## 12:38:28 — wf1 — EPOCH 010   -0.72$/trade   275 trades   WR 36.7%   PF 0.90   2 min

PnL         -197.22$   cumul run     +426.46$   DD 9.1%   Sortino -0.078
point mort 39.2%  ->  ecart -2.5 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -3.7 pt   (reference +1.2 pt sur 5 epochs)
vs epoch precedente : -1.20$/trade
sens    LONG    58W/95  L  37.9%    +126.81$   |   SHORT   43W/79  L  35.2%    -324.04$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 9.5%  val 5.0%
politique  H 1.050/1.099 (-0.007)   etendue 0.3973 (397x le tremblement)   KL +0.0211   clipfrac 27.1%   g_actor 1.6e-01
train   PnL  +2822.86$  1143 trades  WR 47.7%  PF 1.45   ->  ecart train-val +11.0 pt
. etendue val 0.3973, soit 397x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.72$) sous l'erreur-type de cette epoch (0.84$) — non separable de zero.
temps : collecte 37s  PPO 37s  calib 7s  validation 29s

## 12:40:08 — wf1 — EPOCH 011   -1.60$/trade   278 trades   WR 35.6%   PF 0.77   2 min

PnL         -443.92$   cumul run      -17.46$   DD 8.7%   Sortino -0.175
point mort 41.8%  ->  ecart -6.2 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -7.4 pt   (reference +1.2 pt sur 5 epochs)
vs epoch precedente : -0.88$/trade
sens    LONG    70W/111 L  38.7%     -66.31$   |   SHORT   29W/68  L  29.9%    -377.61$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.038/1.099 (-0.012)   etendue 0.4217 (422x le tremblement)   KL +0.0310   clipfrac 27.7%   g_actor 1.6e-01
train   PnL  +2242.26$  1125 trades  WR 48.8%  PF 1.36   ->  ecart train-val +13.2 pt
. etendue val 0.4217, soit 422x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 34s  PPO 35s  calib 7s  validation 31s

## 12:42:08 — wf1 — EPOCH 012   -0.90$/trade   270 trades   WR 35.9%   PF 0.88   2 min

PnL         -242.03$   cumul run     -259.49$   DD 9.7%   Sortino -0.097
point mort 38.9%  ->  ecart -3.0 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -4.2 pt   (reference +1.2 pt sur 5 epochs)
vs epoch precedente : +0.70$/trade
sens    LONG    67W/110 L  37.9%     +59.63$   |   SHORT   30W/63  L  32.3%    -301.65$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.030/1.099 (-0.008)   etendue 0.4321 (432x le tremblement)   KL +0.0323   clipfrac 26.1%   g_actor 1.6e-01
train   PnL  +4058.21$  1108 trades  WR 50.3%  PF 1.69   ->  ecart train-val +14.4 pt
. etendue val 0.4321, soit 432x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 37s  PPO 35s  calib 6s  validation 31s

## 12:44:08 — wf1 — EPOCH 013   -0.68$/trade   283 trades   WR 36.7%   PF 0.90   2 min

PnL         -192.02$   cumul run     -451.51$   DD 9.7%   Sortino -0.075
point mort 39.2%  ->  ecart -2.5 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -3.7 pt   (reference +1.2 pt sur 5 epochs)
vs epoch precedente : +0.22$/trade
sens    LONG    70W/104 L  40.2%    +184.83$   |   SHORT   34W/75  L  31.2%    -376.86$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.020/1.099 (-0.010)   etendue 0.4585 (458x le tremblement)   KL +0.0285   clipfrac 28.5%   g_actor 1.4e-01
train   PnL  +3161.55$  1103 trades  WR 49.0%  PF 1.54   ->  ecart train-val +12.3 pt
. etendue val 0.4585, soit 458x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.68$) sous l'erreur-type de cette epoch (0.78$) — non separable de zero.
temps : collecte 34s  PPO 35s  calib 7s  validation 37s

## 12:45:48 — wf1 — EPOCH 014   -0.85$/trade   296 trades   WR 38.9%   PF 0.87   2 min

PnL         -253.04$   cumul run     -704.55$   DD 12.2%   Sortino -0.096
point mort 42.2%  ->  ecart -3.3 pt (+/- 2.8 au mieux)
vs politique gelee du meme run : -4.5 pt   (reference +1.2 pt sur 5 epochs)
vs epoch precedente : -0.18$/trade
sens    LONG    66W/95  L  41.0%      -1.72$   |   SHORT   49W/86  L  36.3%    -251.32$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.016/1.099 (-0.004)   etendue 0.4757 (476x le tremblement)   KL +0.0456   clipfrac 26.8%   g_actor 1.2e-01
train   PnL  +2067.81$  1094 trades  WR 47.5%  PF 1.34   ->  ecart train-val +8.6 pt
. etendue val 0.4757, soit 476x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 39s  PPO 36s  calib 7s  validation 36s

## 12:59:28 — wf1 — EPOCH 001   -0.15$/trade   218 trades   WR 43.6%   PF 0.95   2 min

PnL          -32.38$   cumul run      -32.38$   DD 5.5%   Sortino -0.039
point mort 44.8%  ->  ecart -1.2 pt (+/- 3.4 au mieux)
sens    LONG    56W/63  L  47.1%      +3.43$   |   SHORT   39W/60  L  39.4%     -35.81$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.5e-06
train   PnL   +501.45$  1152 trades  WR 43.1%  PF 1.16   ->  ecart train-val -0.5 pt
. ACTOR GELE : warmup du critic, gradient 1.5e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.15$) sous l'erreur-type de cette epoch (0.39$) — non separable de zero.
temps : collecte 34s  PPO 35s  calib 7s  validation 43s

## 12:59:28 — wf1 — EPOCH 002   +0.03$/trade   244 trades   WR 41.4%   PF 1.01   2 min

PnL           +6.30$   cumul run      -26.08$   DD 3.3%   Sortino +0.007
point mort 41.2%  ->  ecart +0.2 pt (+/- 3.2 au mieux)
vs epoch precedente : +0.17$/trade
sens    LONG    30W/42  L  41.7%     +14.79$   |   SHORT   71W/101 L  41.3%      -8.48$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 9.8e-08
train   PnL   +622.61$  1167 trades  WR 44.0%  PF 1.20   ->  ecart train-val +2.6 pt
. ACTOR GELE : warmup du critic, gradient 9.8e-08. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.03$) sous l'erreur-type de cette epoch (0.34$) — non separable de zero.
temps : collecte 32s  PPO 38s  calib 6s  validation 45s

## 12:59:28 — wf1 — EPOCH 003   -0.24$/trade   110 trades   WR 39.1%   PF 0.91   2 min

PnL          -26.59$   cumul run      -52.67$   DD 3.2%   Sortino -0.063
point mort 41.4%  ->  ecart -2.3 pt (+/- 4.7 au mieux)
vs politique gelee du meme run : -1.8 pt   (reference -0.5 pt sur 2 epochs)
vs epoch precedente : -0.27$/trade
sens    LONG    17W/27  L  38.6%     -26.99$   |   SHORT   26W/40  L  39.4%      +0.40$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0004 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 1.3e-06
train   PnL   +520.68$  1206 trades  WR 43.0%  PF 1.16   ->  ecart train-val +3.9 pt
. ACTOR GELE : warmup du critic, gradient 1.3e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0004 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le SHORT rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.24$) sous l'erreur-type de cette epoch (0.50$) — non separable de zero.
temps : collecte 35s  PPO 39s  calib 6s  validation 60s

## 12:59:28 — wf1 — EPOCH 004   -0.22$/trade   255 trades   WR 36.1%   PF 0.93   2 min

PnL          -57.10$   cumul run     -109.77$   DD 3.5%   Sortino -0.056
point mort 37.8%  ->  ecart -1.7 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -0.6 pt   (reference -1.1 pt sur 3 epochs)
vs epoch precedente : +0.02$/trade
sens    LONG    63W/101 L  38.4%     +45.03$   |   SHORT   29W/62  L  31.9%    -102.12$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 7.3e-07
train   PnL    -33.99$  1173 trades  WR 41.4%  PF 0.99   ->  ecart train-val +5.3 pt
. ACTOR GELE : warmup du critic, gradient 7.3e-07. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.22$) sous l'erreur-type de cette epoch (0.40$) — non separable de zero.
temps : collecte 31s  PPO 43s  calib 7s  validation 35s

## 13:01:09 — wf1 — EPOCH 005   -0.05$/trade   206 trades   WR 38.8%   PF 0.98   2 min

PnL          -11.32$   cumul run     -121.09$   DD 3.3%   Sortino -0.014
point mort 39.3%  ->  ecart -0.5 pt (+/- 3.4 au mieux)
vs politique gelee du meme run : +0.7 pt   (reference -1.2 pt sur 4 epochs)
vs epoch precedente : +0.17$/trade
sens    LONG    37W/59  L  38.5%     +26.34$   |   SHORT   43W/67  L  39.1%     -37.66$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 32.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 3.2e-08
train   PnL    +91.18$  1190 trades  WR 41.7%  PF 1.03   ->  ecart train-val +2.9 pt
. ACTOR GELE : warmup du critic, gradient 3.2e-08. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.05$) sous l'erreur-type de cette epoch (0.39$) — non separable de zero.
temps : collecte 33s  PPO 44s  calib 11s  validation 35s

## 13:03:29 — wf1 — EPOCH 006   -0.64$/trade   242 trades   WR 38.4%   PF 0.80   2 min

PnL         -155.07$   cumul run     -276.16$   DD 5.1%   Sortino -0.157
point mort 43.8%  ->  ecart -5.4 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : -4.3 pt   (reference -1.1 pt sur 5 epochs)
vs epoch precedente : -0.59$/trade
sens    LONG    35W/47  L  42.7%     +42.12$   |   SHORT   58W/102 L  36.2%    -197.19$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 27.5%  val 5.0%
politique  H 1.089/1.099 (-0.010)   etendue 0.2293 (229x le tremblement)   KL +0.0106   clipfrac 19.1%   g_actor 1.7e-01
train   PnL   +531.20$  1204 trades  WR 42.2%  PF 1.16   ->  ecart train-val +3.8 pt
. APPREND MAIS RESTE PLAT : le gradient passe (1.7e-01) mais l'entropie tient a 1.089 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.2293, soit 229x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 37s  PPO 46s  calib 14s  validation 41s

## 13:05:49 — wf1 — EPOCH 007   -0.82$/trade   263 trades   WR 37.6%   PF 0.74   2 min

PnL         -215.17$   cumul run     -491.33$   DD 3.6%   Sortino -0.202
point mort 44.9%  ->  ecart -7.3 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -6.2 pt   (reference -1.1 pt sur 5 epochs)
vs epoch precedente : -0.18$/trade
sens    LONG    54W/77  L  41.2%      +6.07$   |   SHORT   45W/87  L  34.1%    -221.24$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 23.0%  val 5.0%
politique  H 1.080/1.099 (-0.009)   etendue 0.3679 (368x le tremblement)   KL +0.0067   clipfrac 16.0%   g_actor 2.0e-01
train   PnL   +745.80$  1173 trades  WR 44.5%  PF 1.24   ->  ecart train-val +6.9 pt
. etendue val 0.3679, soit 368x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 37s  PPO 39s  calib 13s  validation 43s

## 13:07:49 — wf1 — EPOCH 008   -0.70$/trade   235 trades   WR 38.7%   PF 0.77   2 min

PnL         -165.37$   cumul run     -656.70$   DD 2.9%   Sortino -0.175
point mort 45.1%  ->  ecart -6.4 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : -5.3 pt   (reference -1.1 pt sur 5 epochs)
vs epoch precedente : +0.11$/trade
sens    LONG    52W/64  L  44.8%     +57.32$   |   SHORT   39W/80  L  32.8%    -222.69$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 18.5%  val 5.0%
politique  H 1.064/1.099 (-0.016)   etendue 0.4141 (414x le tremblement)   KL +0.0139   clipfrac 20.3%   g_actor 1.9e-01
train   PnL   +581.20$  1170 trades  WR 45.0%  PF 1.19   ->  ecart train-val +6.3 pt
. etendue val 0.4141, soit 414x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 32s  PPO 38s  calib 7s  validation 44s

## 13:09:49 — wf1 — EPOCH 009   -1.03$/trade   253 trades   WR 34.8%   PF 0.68   2 min

PnL         -260.26$   cumul run     -916.96$   DD 3.3%   Sortino -0.248
point mort 44.0%  ->  ecart -9.2 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -8.1 pt   (reference -1.1 pt sur 5 epochs)
vs epoch precedente : -0.32$/trade
sens    LONG    39W/57  L  40.6%      +4.22$   |   SHORT   49W/108 L  31.2%    -264.48$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 14.0%  val 5.0%
politique  H 1.052/1.099 (-0.012)   etendue 0.4442 (444x le tremblement)   KL +0.0215   clipfrac 23.0%   g_actor 1.7e-01
train   PnL   +768.42$  1180 trades  WR 44.2%  PF 1.24   ->  ecart train-val +9.4 pt
. etendue val 0.4442, soit 444x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 35s  PPO 38s  calib 10s  validation 36s

## 13:11:49 — wf1 — EPOCH 010   -1.22$/trade   210 trades   WR 37.6%   PF 0.61   2 min

PnL         -256.44$   cumul run    -1173.40$   DD 2.5%   Sortino -0.298
point mort 49.7%  ->  ecart -12.1 pt (+/- 3.3 au mieux)
vs politique gelee du meme run : -11.0 pt   (reference -1.1 pt sur 5 epochs)
vs epoch precedente : -0.19$/trade
sens    LONG    32W/45  L  41.6%      -7.27$   |   SHORT   47W/86  L  35.3%    -249.17$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 9.5%  val 5.0%
politique  H 1.043/1.099 (-0.009)   etendue 0.4669 (467x le tremblement)   KL +0.0198   clipfrac 26.2%   g_actor 1.8e-01
train   PnL  +1001.88$  1130 trades  WR 45.9%  PF 1.35   ->  ecart train-val +8.3 pt
. etendue val 0.4669, soit 467x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 41s  PPO 34s  calib 7s  validation 43s

## 13:13:49 — wf1 — EPOCH 011   -0.36$/trade   245 trades   WR 36.7%   PF 0.89   2 min

PnL          -87.71$   cumul run    -1261.11$   DD 3.4%   Sortino -0.087
point mort 39.5%  ->  ecart -2.8 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : -1.7 pt   (reference -1.1 pt sur 5 epochs)
vs epoch precedente : +0.86$/trade
sens    LONG    42W/64  L  39.6%     +39.26$   |   SHORT   48W/91  L  34.5%    -126.96$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.032/1.099 (-0.011)   etendue 0.4723 (472x le tremblement)   KL +0.0216   clipfrac 25.6%   g_actor 1.6e-01
train   PnL  +1525.73$  1146 trades  WR 48.2%  PF 1.54   ->  ecart train-val +11.5 pt
. etendue val 0.4723, soit 472x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.36$) sous l'erreur-type de cette epoch (0.40$) — non separable de zero.
temps : collecte 34s  PPO 35s  calib 7s  validation 42s

## 13:15:49 — wf1 — EPOCH 012   -0.11$/trade   242 trades   WR 36.8%   PF 0.96   2 min

PnL          -27.03$   cumul run    -1288.14$   DD 4.7%   Sortino -0.028
point mort 37.7%  ->  ecart -0.9 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +0.2 pt   (reference -1.1 pt sur 5 epochs)
vs epoch precedente : +0.25$/trade
sens    LONG    41W/48  L  46.1%    +164.32$   |   SHORT   48W/105 L  31.4%    -191.34$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.023/1.099 (-0.009)   etendue 0.4787 (479x le tremblement)   KL +0.0275   clipfrac 28.3%   g_actor 1.7e-01
train   PnL   +984.68$  1100 trades  WR 47.3%  PF 1.36   ->  ecart train-val +10.5 pt
. etendue val 0.4787, soit 479x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.11$) sous l'erreur-type de cette epoch (0.36$) — non separable de zero.
temps : collecte 39s  PPO 34s  calib 7s  validation 40s

## 13:17:49 — wf1 — EPOCH 013   +0.24$/trade   249 trades   WR 42.6%   PF 1.08   2 min

PnL          +59.11$   cumul run    -1229.03$   DD 3.6%   Sortino +0.061
point mort 40.7%  ->  ecart +1.9 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +3.0 pt   (reference -1.1 pt sur 5 epochs)
vs epoch precedente : +0.35$/trade
sens    LONG    38W/56  L  40.4%     +79.58$   |   SHORT   68W/87  L  43.9%     -20.47$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.016/1.099 (-0.007)   etendue 0.4834 (483x le tremblement)   KL +0.0360   clipfrac 26.3%   g_actor 1.6e-01
train   PnL  +1485.89$  1143 trades  WR 48.1%  PF 1.53   ->  ecart train-val +5.5 pt
. etendue val 0.4834, soit 483x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.24$) sous l'erreur-type de cette epoch (0.40$) — non separable de zero.
temps : collecte 39s  PPO 34s  calib 7s  validation 38s

## 13:19:49 — wf1 — EPOCH 014   +0.59$/trade   241 trades   WR 42.3%   PF 1.21   2 min

PnL         +141.15$   cumul run    -1087.88$   DD 2.7%   Sortino +0.154
point mort 37.8%  ->  ecart +4.5 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +5.6 pt   (reference -1.1 pt sur 5 epochs)
vs epoch precedente : +0.35$/trade
sens    LONG    39W/51  L  43.3%    +162.91$   |   SHORT   63W/88  L  41.7%     -21.76$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.020/1.099 (+0.004)   etendue 0.4792 (479x le tremblement)   KL +0.0256   clipfrac 25.0%   g_actor 1.4e-01
train   PnL  +1097.51$  1138 trades  WR 48.1%  PF 1.39   ->  ecart train-val +5.8 pt
. etendue val 0.4792, soit 479x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 43s  PPO 40s  calib 11s  validation 39s

## 13:21:49 — wf1 — EPOCH 015   +0.45$/trade   248 trades   WR 41.1%   PF 1.15   2 min

PnL         +111.09$   cumul run     -976.79$   DD 2.8%   Sortino +0.114
point mort 37.8%  ->  ecart +3.3 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +4.4 pt   (reference -1.1 pt sur 5 epochs)
vs epoch precedente : -0.14$/trade
sens    LONG    48W/61  L  44.0%    +147.43$   |   SHORT   54W/85  L  38.8%     -36.34$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.007/1.099 (-0.013)   etendue 0.4884 (488x le tremblement)   KL +0.0270   clipfrac 21.7%   g_actor 1.4e-01
train   PnL  +1316.89$  1152 trades  WR 47.3%  PF 1.45   ->  ecart train-val +6.2 pt
. etendue val 0.4884, soit 488x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 40s  PPO 36s  calib 8s  validation 35s

## 13:23:49 — wf1 — EPOCH 016   +0.02$/trade   270 trades   WR 39.6%   PF 1.00   2 min

PnL           +4.06$   cumul run     -972.73$   DD 4.9%   Sortino +0.004
point mort 39.6%  ->  ecart -0.0 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +1.1 pt   (reference -1.1 pt sur 5 epochs)
vs epoch precedente : -0.43$/trade
sens    LONG    55W/75  L  42.3%    +137.62$   |   SHORT   52W/88  L  37.1%    -133.56$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.999/1.099 (-0.008)   etendue 0.4875 (488x le tremblement)   KL +0.0256   clipfrac 21.3%   g_actor 1.4e-01
train   PnL  +1514.81$  1114 trades  WR 49.8%  PF 1.58   ->  ecart train-val +10.2 pt
. etendue val 0.4875, soit 488x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.02$) sous l'erreur-type de cette epoch (0.35$) — non separable de zero.
temps : collecte 40s  PPO 33s  calib 7s  validation 38s

## 13:26:09 — wf1 — EPOCH 017   -1.04$/trade   275 trades   WR 33.8%   PF 0.68   2 min

PnL         -287.20$   cumul run    -1259.93$   DD 5.4%   Sortino -0.253
point mort 42.9%  ->  ecart -9.1 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -8.0 pt   (reference -1.1 pt sur 5 epochs)
vs epoch precedente : -1.06$/trade
sens    LONG    50W/91  L  35.5%     -93.46$   |   SHORT   43W/91  L  32.1%    -193.73$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.003/1.099 (+0.004)   etendue 0.4912 (491x le tremblement)   KL +0.0276   clipfrac 22.3%   g_actor 1.5e-01
train   PnL  +1052.80$  1135 trades  WR 48.7%  PF 1.38   ->  ecart train-val +14.9 pt
. etendue val 0.4912, soit 491x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 36s  PPO 39s  calib 14s  validation 42s

## 13:28:10 — wf1 — EPOCH 018   -0.87$/trade   257 trades   WR 33.5%   PF 0.73   2 min

PnL         -224.62$   cumul run    -1484.55$   DD 4.0%   Sortino -0.211
point mort 40.8%  ->  ecart -7.3 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -6.2 pt   (reference -1.1 pt sur 5 epochs)
vs epoch precedente : +0.17$/trade
sens    LONG    48W/88  L  35.3%     -42.32$   |   SHORT   38W/83  L  31.4%    -182.31$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.993/1.099 (-0.010)   etendue 0.4939 (494x le tremblement)   KL +0.0240   clipfrac 19.0%   g_actor 1.4e-01
train   PnL  +1733.99$  1128 trades  WR 50.2%  PF 1.64   ->  ecart train-val +16.7 pt
. etendue val 0.4939, soit 494x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 40s  PPO 35s  calib 7s  validation 43s

## 13:30:10 — wf1 — EPOCH 019   +0.84$/trade   266 trades   WR 39.8%   PF 1.29   2 min

PnL         +223.84$   cumul run    -1260.71$   DD 4.0%   Sortino +0.217
point mort 33.9%  ->  ecart +5.9 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +7.0 pt   (reference -1.1 pt sur 5 epochs)
vs epoch precedente : +1.72$/trade
sens    LONG    58W/71  L  45.0%    +273.15$   |   SHORT   48W/89  L  35.0%     -49.31$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.991/1.099 (-0.002)   etendue 0.4936 (494x le tremblement)   KL +0.0310   clipfrac 19.6%   g_actor 1.3e-01
train   PnL  +1134.46$  1166 trades  WR 47.7%  PF 1.39   ->  ecart train-val +7.9 pt
. etendue val 0.4936, soit 494x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 38s  PPO 36s  calib 7s  validation 36s

## 13:31:50 — wf1 — EPOCH 020   -0.01$/trade   285 trades   WR 37.5%   PF 1.00   2 min

PnL           -2.85$   cumul run    -1263.56$   DD 3.9%   Sortino -0.003
point mort 37.5%  ->  ecart -0.0 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : +1.0 pt   (reference -1.1 pt sur 5 epochs)
vs epoch precedente : -0.85$/trade
sens    LONG    47W/71  L  39.8%    +118.44$   |   SHORT   60W/107 L  35.9%    -121.29$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.991/1.099 (+0.000)   etendue 0.4927 (493x le tremblement)   KL +0.0292   clipfrac 21.0%   g_actor 1.6e-01
train   PnL  +1785.85$  1097 trades  WR 52.6%  PF 1.72   ->  ecart train-val +15.1 pt
. etendue val 0.4927, soit 493x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.01$) sous l'erreur-type de cette epoch (0.36$) — non separable de zero.
temps : collecte 37s  PPO 33s  calib 7s  validation 35s

## 13:33:50 — wf1 — EPOCH 021   -0.32$/trade   296 trades   WR 37.8%   PF 0.90   2 min

PnL          -94.87$   cumul run    -1358.43$   DD 3.3%   Sortino -0.079
point mort 40.3%  ->  ecart -2.5 pt (+/- 2.8 au mieux)
vs politique gelee du meme run : -1.5 pt   (reference -1.1 pt sur 5 epochs)
vs epoch precedente : -0.31$/trade
sens    LONG    52W/72  L  41.9%     +80.48$   |   SHORT   60W/112 L  34.9%    -175.34$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.990/1.099 (-0.001)   etendue 0.4947 (495x le tremblement)   KL +0.0234   clipfrac 19.1%   g_actor 1.3e-01
train   PnL  +1461.98$  1061 trades  WR 49.9%  PF 1.58   ->  ecart train-val +12.1 pt
. etendue val 0.4947, soit 495x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.32$) sous l'erreur-type de cette epoch (0.36$) — non separable de zero.
temps : collecte 33s  PPO 32s  calib 6s  validation 37s

## 13:37:05 — wf1 — EPOCH 001   -1.12$/trade   219 trades   WR 35.6%   PF 0.65   2 min

PnL         -244.33$   cumul run     -244.33$   DD 3.5%   Sortino -0.273
point mort 46.0%  ->  ecart -10.4 pt (+/- 3.2 au mieux)
sens    LONG    24W/45  L  34.8%    -103.15$   |   SHORT   54W/96  L  36.0%    -141.17$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.6e-06
train   PnL   +751.18$  1164 trades  WR 45.1%  PF 1.25   ->  ecart train-val +9.5 pt
. ACTOR GELE : warmup du critic, gradient 1.6e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 36s  PPO 36s  calib 7s  validation 45s

## 13:39:05 — wf1 — EPOCH 002   +0.11$/trade   243 trades   WR 39.1%   PF 1.04   2 min

PnL          +27.84$   cumul run     -216.49$   DD 3.6%   Sortino +0.029
point mort 38.2%  ->  ecart +0.9 pt (+/- 3.1 au mieux)
vs epoch precedente : +1.23$/trade
sens    LONG    49W/78  L  38.6%     +61.84$   |   SHORT   46W/70  L  39.7%     -34.00$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0005 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 2.5e-06
train   PnL   -342.02$  1231 trades  WR 40.8%  PF 0.90   ->  ecart train-val +1.7 pt
. ACTOR GELE : warmup du critic, gradient 2.5e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0005 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.11$) sous l'erreur-type de cette epoch (0.39$) — non separable de zero.
temps : collecte 37s  PPO 43s  calib 11s  validation 32s

## 13:41:05 — wf1 — EPOCH 003   -0.51$/trade   223 trades   WR 34.5%   PF 0.84   2 min

PnL         -113.76$   cumul run     -330.25$   DD 3.8%   Sortino -0.127
point mort 38.6%  ->  ecart -4.1 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +0.7 pt   (reference -4.7 pt sur 2 epochs)
vs epoch precedente : -0.62$/trade
sens    LONG    27W/56  L  32.5%     -65.80$   |   SHORT   50W/90  L  35.7%     -47.95$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 1.8e-05
train   PnL   +359.40$  1223 trades  WR 41.3%  PF 1.11   ->  ecart train-val +6.8 pt
. ACTOR GELE : warmup du critic, gradient 1.8e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 35s  PPO 40s  calib 10s  validation 34s

## 13:43:05 — wf1 — EPOCH 004   -0.06$/trade   268 trades   WR 39.9%   PF 0.98   2 min

PnL          -17.21$   cumul run     -347.46$   DD 5.0%   Sortino -0.016
point mort 40.4%  ->  ecart -0.5 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +4.0 pt   (reference -4.5 pt sur 3 epochs)
vs epoch precedente : +0.45$/trade
sens    LONG    62W/94  L  39.7%     +12.62$   |   SHORT   45W/67  L  40.2%     -29.83$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.5e-05
train   PnL   -239.96$  1209 trades  WR 40.3%  PF 0.93   ->  ecart train-val +0.4 pt
. ACTOR GELE : warmup du critic, gradient 1.5e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.06$) sous l'erreur-type de cette epoch (0.40$) — non separable de zero.
temps : collecte 34s  PPO 44s  calib 11s  validation 33s

## 13:45:05 — wf1 — EPOCH 005   +0.55$/trade   224 trades   WR 43.3%   PF 1.20   2 min

PnL         +122.54$   cumul run     -224.92$   DD 4.0%   Sortino +0.144
point mort 38.9%  ->  ecart +4.4 pt (+/- 3.3 au mieux)
vs politique gelee du meme run : +7.9 pt   (reference -3.5 pt sur 4 epochs)
vs epoch precedente : +0.61$/trade
sens    LONG    63W/75  L  45.7%    +129.74$   |   SHORT   34W/52  L  39.5%      -7.20$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 32.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0004 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 1.1e-05
train   PnL   +330.05$  1193 trades  WR 44.8%  PF 1.11   ->  ecart train-val +1.5 pt
. ACTOR GELE : warmup du critic, gradient 1.1e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0004 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 32s  PPO 45s  calib 12s  validation 35s

## 13:47:05 — wf1 — EPOCH 006   -0.01$/trade   241 trades   WR 40.2%   PF 1.00   2 min

PnL           -1.21$   cumul run     -226.13$   DD 3.5%   Sortino -0.001
point mort 40.2%  ->  ecart -0.0 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +1.9 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : -0.55$/trade
sens    LONG    57W/83  L  40.7%     +43.79$   |   SHORT   40W/61  L  39.6%     -45.00$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 27.5%  val 5.0%
politique  H 1.092/1.099 (-0.007)   etendue 0.2337 (234x le tremblement)   KL +0.0112   clipfrac 15.1%   g_actor 4.3e-01
train   PnL   +440.56$  1237 trades  WR 43.3%  PF 1.13   ->  ecart train-val +3.1 pt
. APPREND MAIS RESTE PLAT : le gradient passe (4.3e-01) mais l'entropie tient a 1.092 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.2337, soit 234x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.01$) sous l'erreur-type de cette epoch (0.37$) — non separable de zero.
temps : collecte 33s  PPO 51s  calib 12s  validation 36s

## 13:49:25 — wf1 — EPOCH 007   +0.13$/trade   257 trades   WR 41.2%   PF 1.04   2 min

PnL          +32.99$   cumul run     -193.14$   DD 3.5%   Sortino +0.033
point mort 40.3%  ->  ecart +0.9 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +2.8 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : +0.13$/trade
sens    LONG    78W/101 L  43.6%     +73.29$   |   SHORT   28W/50  L  35.9%     -40.30$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 23.0%  val 5.0%
politique  H 1.084/1.099 (-0.008)   etendue 0.2583 (258x le tremblement)   KL +0.0097   clipfrac 16.6%   g_actor 4.4e-01
train   PnL   +606.18$  1175 trades  WR 44.5%  PF 1.20   ->  ecart train-val +3.3 pt
. etendue val 0.2583, soit 258x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.13$) sous l'erreur-type de cette epoch (0.42$) — non separable de zero.
temps : collecte 32s  PPO 46s  calib 13s  validation 34s

## 13:51:26 — wf1 — EPOCH 008   -0.13$/trade   244 trades   WR 39.8%   PF 0.96   2 min

PnL          -31.44$   cumul run     -224.58$   DD 3.6%   Sortino -0.033
point mort 40.7%  ->  ecart -0.9 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +1.0 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : -0.26$/trade
sens    LONG    58W/75  L  43.6%      +7.59$   |   SHORT   39W/72  L  35.1%     -39.03$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 18.5%  val 5.0%
politique  H 1.076/1.099 (-0.008)   etendue 0.3238 (324x le tremblement)   KL +0.0121   clipfrac 20.8%   g_actor 4.6e-01
train   PnL   +661.79$  1176 trades  WR 45.2%  PF 1.21   ->  ecart train-val +5.4 pt
. etendue val 0.3238, soit 324x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.13$) sous l'erreur-type de cette epoch (0.41$) — non separable de zero.
temps : collecte 32s  PPO 47s  calib 12s  validation 38s

## 13:53:26 — wf1 — EPOCH 009   -0.01$/trade   230 trades   WR 39.6%   PF 1.00   2 min

PnL           -2.24$   cumul run     -226.82$   DD 3.5%   Sortino -0.003
point mort 39.6%  ->  ecart +0.0 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +2.0 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : +0.12$/trade
sens    LONG    46W/65  L  41.4%     +50.81$   |   SHORT   45W/74  L  37.8%     -53.05$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 14.0%  val 5.0%
politique  H 1.069/1.099 (-0.007)   etendue 0.3547 (355x le tremblement)   KL +0.0166   clipfrac 21.4%   g_actor 5.2e-01
train   PnL   +516.28$  1135 trades  WR 43.3%  PF 1.17   ->  ecart train-val +3.7 pt
. etendue val 0.3547, soit 355x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.01$) sous l'erreur-type de cette epoch (0.38$) — non separable de zero.
temps : collecte 35s  PPO 41s  calib 11s  validation 36s

## 13:55:26 — wf1 — EPOCH 010   -0.64$/trade   265 trades   WR 37.0%   PF 0.79   2 min

PnL         -170.19$   cumul run     -397.01$   DD 4.9%   Sortino -0.158
point mort 42.6%  ->  ecart -5.6 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -3.7 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : -0.63$/trade
sens    LONG    41W/73  L  36.0%     -58.98$   |   SHORT   57W/94  L  37.7%    -111.21$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 9.5%  val 5.0%
politique  H 1.066/1.099 (-0.003)   etendue 0.3661 (366x le tremblement)   KL +0.0225   clipfrac 27.2%   g_actor 4.9e-01
train   PnL   +879.10$  1153 trades  WR 43.9%  PF 1.29   ->  ecart train-val +6.9 pt
. etendue val 0.3661, soit 366x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 37s  PPO 40s  calib 8s  validation 37s

## 13:57:26 — wf1 — EPOCH 011   -0.89$/trade   265 trades   WR 37.4%   PF 0.72   2 min

PnL         -234.65$   cumul run     -631.66$   DD 3.6%   Sortino -0.218
point mort 45.3%  ->  ecart -7.9 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -6.0 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : -0.24$/trade
sens    LONG    56W/79  L  41.5%     -88.00$   |   SHORT   43W/87  L  33.1%    -146.65$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.060/1.099 (-0.006)   etendue 0.3895 (390x le tremblement)   KL +0.0128   clipfrac 23.5%   g_actor 4.9e-01
train   PnL  +1105.83$  1176 trades  WR 46.5%  PF 1.38   ->  ecart train-val +9.1 pt
. etendue val 0.3895, soit 390x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 36s  PPO 41s  calib 9s  validation 35s

## 13:59:46 — wf1 — EPOCH 012   -0.77$/trade   278 trades   WR 37.1%   PF 0.76   2 min

PnL         -213.51$   cumul run     -845.17$   DD 4.6%   Sortino -0.188
point mort 43.6%  ->  ecart -6.5 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -4.6 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : +0.12$/trade
sens    LONG    60W/91  L  39.7%     -13.64$   |   SHORT   43W/84  L  33.9%    -199.87$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.049/1.099 (-0.011)   etendue 0.4289 (429x le tremblement)   KL +0.0222   clipfrac 22.8%   g_actor 4.5e-01
train   PnL  +1065.72$  1150 trades  WR 46.3%  PF 1.37   ->  ecart train-val +9.2 pt
. etendue val 0.4289, soit 429x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 36s  PPO 40s  calib 10s  validation 37s

## 14:01:26 — wf1 — EPOCH 013   -0.62$/trade   266 trades   WR 34.6%   PF 0.80   2 min

PnL         -165.53$   cumul run    -1010.70$   DD 5.4%   Sortino -0.154
point mort 39.8%  ->  ecart -5.2 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -3.3 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : +0.15$/trade
sens    LONG    64W/94  L  40.5%     -17.55$   |   SHORT   28W/80  L  25.9%    -147.98$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.042/1.099 (-0.007)   etendue 0.4425 (442x le tremblement)   KL +0.0256   clipfrac 24.0%   g_actor 4.1e-01
train   PnL  +1097.62$  1170 trades  WR 47.2%  PF 1.37   ->  ecart train-val +12.6 pt
. etendue val 0.4425, soit 442x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 38s  PPO 37s  calib 7s  validation 30s

## 14:03:26 — wf1 — EPOCH 014   -0.46$/trade   238 trades   WR 39.1%   PF 0.84   2 min

PnL         -110.43$   cumul run    -1121.13$   DD 5.0%   Sortino -0.118
point mort 43.3%  ->  ecart -4.2 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : -2.3 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : +0.16$/trade
sens    LONG    55W/76  L  42.0%     -10.36$   |   SHORT   38W/69  L  35.5%    -100.07$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.045/1.099 (+0.003)   etendue 0.4463 (446x le tremblement)   KL +0.0267   clipfrac 27.4%   g_actor 3.8e-01
train   PnL   +658.38$  1146 trades  WR 46.5%  PF 1.23   ->  ecart train-val +7.4 pt
. etendue val 0.4463, soit 446x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 37s  PPO 40s  calib 11s  validation 28s

## 14:05:26 — wf1 — EPOCH 015   -0.45$/trade   252 trades   WR 36.1%   PF 0.85   2 min

PnL         -112.81$   cumul run    -1233.94$   DD 3.7%   Sortino -0.111
point mort 39.9%  ->  ecart -3.8 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -1.9 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : +0.02$/trade
sens    LONG    48W/81  L  37.2%     -57.81$   |   SHORT   43W/80  L  35.0%     -55.00$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.036/1.099 (-0.009)   etendue 0.4676 (468x le tremblement)   KL +0.0170   clipfrac 21.8%   g_actor 4.3e-01
train   PnL  +1365.67$  1117 trades  WR 50.7%  PF 1.52   ->  ecart train-val +14.6 pt
. etendue val 0.4676, soit 468x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 36s  PPO 37s  calib 12s  validation 33s

## 14:07:26 — wf1 — EPOCH 016   -0.10$/trade   203 trades   WR 39.9%   PF 0.96   2 min

PnL          -20.81$   cumul run    -1254.75$   DD 3.0%   Sortino -0.027
point mort 40.9%  ->  ecart -1.0 pt (+/- 3.4 au mieux)
vs politique gelee du meme run : +0.9 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : +0.35$/trade
sens    LONG    52W/85  L  38.0%      -1.44$   |   SHORT   29W/37  L  43.9%     -19.37$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.029/1.099 (-0.007)   etendue 0.4763 (476x le tremblement)   KL +0.0351   clipfrac 27.1%   g_actor 5.5e-01
train   PnL   +903.51$  1146 trades  WR 45.6%  PF 1.31   ->  ecart train-val +5.7 pt
. etendue val 0.4763, soit 476x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (-0.10$) sous l'erreur-type de cette epoch (0.36$) — non separable de zero.
temps : collecte 36s  PPO 37s  calib 7s  validation 37s

## 14:09:06 — wf1 — EPOCH 017   +0.03$/trade   246 trades   WR 35.8%   PF 1.01   2 min

PnL           +6.56$   cumul run    -1248.19$   DD 3.7%   Sortino +0.007
point mort 35.5%  ->  ecart +0.3 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +2.2 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : +0.13$/trade
sens    LONG    52W/89  L  36.9%     +39.80$   |   SHORT   36W/69  L  34.3%     -33.24$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.023/1.099 (-0.006)   etendue 0.4774 (477x le tremblement)   KL +0.0275   clipfrac 23.7%   g_actor 4.2e-01
train   PnL  +1424.67$  1061 trades  WR 50.6%  PF 1.58   ->  ecart train-val +14.8 pt
. etendue val 0.4774, soit 477x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.03$) sous l'erreur-type de cette epoch (0.36$) — non separable de zero.
temps : collecte 34s  PPO 32s  calib 6s  validation 31s

## 14:10:46 — wf1 — EPOCH 018   -0.16$/trade   227 trades   WR 37.9%   PF 0.95   2 min

PnL          -36.14$   cumul run    -1284.33$   DD 3.7%   Sortino -0.040
point mort 39.1%  ->  ecart -1.2 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +0.7 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : -0.19$/trade
sens    LONG    46W/68  L  40.4%     +19.43$   |   SHORT   40W/73  L  35.4%     -55.58$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.025/1.099 (+0.002)   etendue 0.4818 (482x le tremblement)   KL +0.0228   clipfrac 21.4%   g_actor 4.1e-01
train   PnL  +1101.32$  1151 trades  WR 48.3%  PF 1.38   ->  ecart train-val +10.4 pt
. etendue val 0.4818, soit 482x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.16$) sous l'erreur-type de cette epoch (0.42$) — non separable de zero.
temps : collecte 35s  PPO 35s  calib 6s  validation 34s

## 14:12:46 — wf1 — EPOCH 019   +0.07$/trade   244 trades   WR 38.1%   PF 1.02   2 min

PnL          +16.69$   cumul run    -1267.64$   DD 3.7%   Sortino +0.017
point mort 37.6%  ->  ecart +0.5 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : +2.4 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : +0.23$/trade
sens    LONG    59W/99  L  37.3%     +17.66$   |   SHORT   34W/52  L  39.5%      -0.97$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.013/1.099 (-0.012)   etendue 0.4866 (487x le tremblement)   KL +0.0326   clipfrac 22.2%   g_actor 3.7e-01
train   PnL  +1063.54$  1105 trades  WR 47.8%  PF 1.39   ->  ecart train-val +9.7 pt
. etendue val 0.4866, soit 487x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.07$) sous l'erreur-type de cette epoch (0.46$) — non separable de zero.
temps : collecte 35s  PPO 36s  calib 7s  validation 37s

## 14:14:46 — wf1 — EPOCH 020   +0.09$/trade   256 trades   WR 39.1%   PF 1.03   2 min

PnL          +22.38$   cumul run    -1245.26$   DD 5.1%   Sortino +0.022
point mort 38.4%  ->  ecart +0.7 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : +2.7 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : +0.02$/trade
sens    LONG    53W/82  L  39.3%     +58.20$   |   SHORT   47W/74  L  38.8%     -35.82$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.014/1.099 (+0.001)   etendue 0.4876 (488x le tremblement)   KL +0.0248   clipfrac 18.7%   g_actor 3.4e-01
train   PnL  +1322.29$  1124 trades  WR 50.0%  PF 1.50   ->  ecart train-val +10.9 pt
. etendue val 0.4876, soit 488x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.09$) sous l'erreur-type de cette epoch (0.38$) — non separable de zero.
temps : collecte 38s  PPO 35s  calib 6s  validation 33s

## 14:16:26 — wf1 — EPOCH 021   -0.01$/trade   242 trades   WR 41.3%   PF 1.00   2 min

PnL           -3.30$   cumul run    -1248.56$   DD 4.6%   Sortino -0.004
point mort 41.3%  ->  ecart -0.0 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +1.9 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : -0.10$/trade
sens    LONG    64W/91  L  41.3%     +41.61$   |   SHORT   36W/51  L  41.4%     -44.91$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.005/1.099 (-0.009)   etendue 0.4869 (487x le tremblement)   KL +0.0194   clipfrac 19.9%   g_actor 3.5e-01
train   PnL  +1737.20$  1132 trades  WR 50.7%  PF 1.66   ->  ecart train-val +9.4 pt
. etendue val 0.4869, soit 487x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.01$) sous l'erreur-type de cette epoch (0.36$) — non separable de zero.
temps : collecte 37s  PPO 35s  calib 6s  validation 32s

## 14:18:26 — wf1 — EPOCH 022   -0.04$/trade   223 trades   WR 40.8%   PF 0.98   2 min

PnL          -10.01$   cumul run    -1258.57$   DD 3.5%   Sortino -0.011
point mort 41.3%  ->  ecart -0.5 pt (+/- 3.3 au mieux)
vs politique gelee du meme run : +1.4 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : -0.03$/trade
sens    LONG    57W/84  L  40.4%     +57.21$   |   SHORT   34W/48  L  41.5%     -67.22$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.014/1.099 (+0.009)   etendue 0.4922 (492x le tremblement)   KL +0.0190   clipfrac 16.4%   g_actor 3.8e-01
train   PnL  +1860.67$  1069 trades  WR 53.2%  PF 1.78   ->  ecart train-val +12.4 pt
. etendue val 0.4922, soit 492x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.04$) sous l'erreur-type de cette epoch (0.30$) — non separable de zero.
temps : collecte 35s  PPO 34s  calib 6s  validation 34s

## 14:20:27 — wf1 — EPOCH 023   -0.35$/trade   243 trades   WR 39.5%   PF 0.88   2 min

PnL          -85.46$   cumul run    -1344.03$   DD 4.8%   Sortino -0.088
point mort 42.6%  ->  ecart -3.1 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : -1.2 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : -0.31$/trade
sens    LONG    61W/91  L  40.1%     -38.95$   |   SHORT   35W/56  L  38.5%     -46.51$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.009/1.099 (-0.005)   etendue 0.4935 (494x le tremblement)   KL +0.0193   clipfrac 18.2%   g_actor 3.8e-01
train   PnL  +1620.53$  1092 trades  WR 49.6%  PF 1.64   ->  ecart train-val +10.1 pt
. etendue val 0.4935, soit 494x le tremblement du seuil : la selection est portee par le modele.
. gain par trade (-0.35$) sous l'erreur-type de cette epoch (0.36$) — non separable de zero.
temps : collecte 36s  PPO 34s  calib 7s  validation 41s

## 14:22:27 — wf1 — EPOCH 024   +0.95$/trade   238 trades   WR 43.7%   PF 1.35   2 min

PnL         +226.80$   cumul run    -1117.23$   DD 3.6%   Sortino +0.254
point mort 36.5%  ->  ecart +7.2 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +9.1 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : +1.30$/trade
sens    LONG    62W/78  L  44.3%    +156.45$   |   SHORT   42W/56  L  42.9%     +70.35$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.009/1.099 (+0.000)   etendue 0.4929 (493x le tremblement)   KL +0.0201   clipfrac 17.7%   g_actor 4.0e-01
train   PnL  +1410.21$  1133 trades  WR 50.0%  PF 1.52   ->  ecart train-val +6.3 pt
. etendue val 0.4929, soit 493x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 39s  PPO 35s  calib 7s  validation 39s

## 14:24:27 — wf1 — EPOCH 025   -0.12$/trade   240 trades   WR 40.8%   PF 0.96   2 min

PnL          -28.27$   cumul run    -1145.50$   DD 5.1%   Sortino -0.030
point mort 41.8%  ->  ecart -1.0 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +0.9 pt   (reference -1.9 pt sur 5 epochs)
vs epoch precedente : -1.07$/trade
sens    LONG    60W/76  L  44.1%     +21.37$   |   SHORT   38W/66  L  36.5%     -49.65$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.017/1.099 (+0.008)   etendue 0.4937 (494x le tremblement)   KL +0.0155   clipfrac 16.5%   g_actor 4.3e-01
train   PnL  +1527.32$  1130 trades  WR 49.6%  PF 1.57   ->  ecart train-val +8.8 pt
. etendue val 0.4937, soit 494x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.12$) sous l'erreur-type de cette epoch (0.38$) — non separable de zero.
temps : collecte 41s  PPO 35s  calib 7s  validation 40s

## 14:27:41 — wf1 — EPOCH 001   -0.23$/trade   233 trades   WR 38.2%   PF 0.93   2 min

PnL          -53.23$   cumul run      -53.23$   DD 4.1%   Sortino -0.057
point mort 39.9%  ->  ecart -1.7 pt (+/- 3.2 au mieux)
sens    LONG    33W/35  L  48.5%     +77.48$   |   SHORT   56W/109 L  33.9%    -130.71$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0000 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.3e-06
train   PnL   +501.45$  1152 trades  WR 43.1%  PF 1.16   ->  ecart train-val +4.9 pt
. ACTOR GELE : warmup du critic, gradient 1.3e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.23$) sous l'erreur-type de cette epoch (0.42$) — non separable de zero.
temps : collecte 34s  PPO 36s  calib 6s  validation 47s

## 14:29:41 — wf1 — EPOCH 002   +1.04$/trade   213 trades   WR 44.1%   PF 1.38   2 min

PnL         +220.84$   cumul run     +167.61$   DD 3.8%   Sortino +0.276
point mort 36.4%  ->  ecart +7.7 pt (+/- 3.4 au mieux)
vs epoch precedente : +1.27$/trade
sens    LONG    56W/72  L  43.8%    +243.17$   |   SHORT   38W/47  L  44.7%     -22.32$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0002 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 3.2e-07
train   PnL   +622.61$  1167 trades  WR 44.0%  PF 1.20   ->  ecart train-val -0.1 pt
. ACTOR GELE : warmup du critic, gradient 3.2e-07. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0002 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 34s  PPO 41s  calib 13s  validation 39s

## 14:32:01 — wf1 — EPOCH 003   -0.35$/trade   133 trades   WR 36.8%   PF 0.88   2 min

PnL          -47.20$   cumul run     +120.41$   DD 2.9%   Sortino -0.090
point mort 39.9%  ->  ecart -3.1 pt (+/- 4.2 au mieux)
vs politique gelee du meme run : -6.0 pt   (reference +3.0 pt sur 2 epochs)
vs epoch precedente : -1.39$/trade
sens    LONG    26W/33  L  44.1%     +51.34$   |   SHORT   23W/51  L  31.1%     -98.55$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0002 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 2.1e-06
train   PnL   +520.68$  1206 trades  WR 43.0%  PF 1.16   ->  ecart train-val +6.2 pt
. ACTOR GELE : warmup du critic, gradient 2.1e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0002 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.35$) sous l'erreur-type de cette epoch (0.49$) — non separable de zero.
temps : collecte 32s  PPO 44s  calib 12s  validation 54s

## 14:34:01 — wf1 — EPOCH 004   -0.22$/trade   278 trades   WR 40.3%   PF 0.93   2 min

PnL          -61.02$   cumul run      +59.39$   DD 3.4%   Sortino -0.055
point mort 42.0%  ->  ecart -1.7 pt (+/- 2.9 au mieux)
vs politique gelee du meme run : -2.7 pt   (reference +1.0 pt sur 3 epochs)
vs epoch precedente : +0.14$/trade
sens    LONG    77W/91  L  45.8%     +89.54$   |   SHORT   35W/75  L  31.8%    -150.57$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 5.8e-07
train   PnL    -33.99$  1173 trades  WR 41.4%  PF 0.99   ->  ecart train-val +1.1 pt
. ACTOR GELE : warmup du critic, gradient 5.8e-07. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.22$) sous l'erreur-type de cette epoch (0.37$) — non separable de zero.
temps : collecte 34s  PPO 44s  calib 9s  validation 40s

## 14:36:21 — wf1 — EPOCH 005   +0.65$/trade   238 trades   WR 44.1%   PF 1.24   2 min

PnL         +154.14$   cumul run     +213.53$   DD 4.4%   Sortino +0.171
point mort 38.9%  ->  ecart +5.2 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +4.9 pt   (reference +0.3 pt sur 4 epochs)
vs epoch precedente : +0.87$/trade
sens    LONG    58W/73  L  44.3%    +169.60$   |   SHORT   47W/60  L  43.9%     -15.46$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 32.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0004 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 3.1e-06
train   PnL    +91.18$  1190 trades  WR 41.7%  PF 1.03   ->  ecart train-val -2.4 pt
. ACTOR GELE : warmup du critic, gradient 3.1e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0004 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 37s  PPO 46s  calib 9s  validation 38s

## 14:39:01 — wf1 — EPOCH 006   +0.40$/trade   220 trades   WR 41.8%   PF 1.14   3 min

PnL          +88.43$   cumul run     +301.96$   DD 3.3%   Sortino +0.103
point mort 38.7%  ->  ecart +3.1 pt (+/- 3.3 au mieux)
vs politique gelee du meme run : +1.9 pt   (reference +1.3 pt sur 5 epochs)
vs epoch precedente : -0.25$/trade
sens    LONG    37W/49  L  43.0%     +91.11$   |   SHORT   55W/79  L  41.0%      -2.68$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 27.5%  val 5.0%
politique  H 1.091/1.099 (-0.008)   etendue 0.2049 (205x le tremblement)   KL +0.0088   clipfrac 18.5%   g_actor 1.9e-01
train   PnL    +36.20$  1251 trades  WR 39.7%  PF 1.01   ->  ecart train-val -2.1 pt
. APPREND MAIS RESTE PLAT : le gradient passe (1.9e-01) mais l'entropie tient a 1.091 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.2049, soit 205x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.40$) sous l'erreur-type de cette epoch (0.42$) — non separable de zero.
. GPU BRIDE : 420 MHz a 85 C. Les durees ne sont pas comparables entre epochs si la temperature derive.
temps : collecte 41s  PPO 62s  calib 14s  validation 46s

## 14:41:21 — wf1 — EPOCH 007   +0.44$/trade   212 trades   WR 42.9%   PF 1.16   2 min

PnL          +92.50$   cumul run     +394.46$   DD 2.8%   Sortino +0.116
point mort 39.3%  ->  ecart +3.6 pt (+/- 3.4 au mieux)
vs politique gelee du meme run : +2.3 pt   (reference +1.3 pt sur 5 epochs)
vs epoch precedente : +0.03$/trade
sens    LONG    50W/56  L  47.2%    +124.80$   |   SHORT   41W/65  L  38.7%     -32.30$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 23.0%  val 5.0%
politique  H 1.080/1.099 (-0.011)   etendue 0.3137 (314x le tremblement)   KL +0.0080   clipfrac 17.6%   g_actor 1.8e-01
train   PnL   +427.14$  1168 trades  WR 43.7%  PF 1.14   ->  ecart train-val +0.8 pt
. etendue val 0.3137, soit 314x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 37s  PPO 45s  calib 16s  validation 45s

## 14:44:01 — wf1 — EPOCH 008   +0.22$/trade   231 trades   WR 40.7%   PF 1.08   2 min

PnL          +50.82$   cumul run     +445.28$   DD 4.0%   Sortino +0.056
point mort 38.8%  ->  ecart +1.9 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : +0.6 pt   (reference +1.3 pt sur 5 epochs)
vs epoch precedente : -0.22$/trade
sens    LONG    49W/61  L  44.5%    +170.19$   |   SHORT   45W/76  L  37.2%    -119.37$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 18.5%  val 5.0%
politique  H 1.067/1.099 (-0.013)   etendue 0.3797 (380x le tremblement)   KL +0.0143   clipfrac 22.6%   g_actor 2.0e-01
train   PnL  +1078.32$  1178 trades  WR 45.5%  PF 1.36   ->  ecart train-val +4.8 pt
. etendue val 0.3797, soit 380x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.22$) sous l'erreur-type de cette epoch (0.39$) — non separable de zero.
temps : collecte 39s  PPO 51s  calib 13s  validation 46s

## 14:46:21 — wf1 — EPOCH 009   -0.37$/trade   228 trades   WR 37.7%   PF 0.87   2 min

PnL          -85.13$   cumul run     +360.15$   DD 4.3%   Sortino -0.094
point mort 41.0%  ->  ecart -3.3 pt (+/- 3.2 au mieux)
vs politique gelee du meme run : -4.6 pt   (reference +1.3 pt sur 5 epochs)
vs epoch precedente : -0.59$/trade
sens    LONG    44W/66  L  40.0%      +9.98$   |   SHORT   42W/76  L  35.6%     -95.11$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 14.0%  val 5.0%
politique  H 1.061/1.099 (-0.006)   etendue 0.3937 (394x le tremblement)   KL +0.0210   clipfrac 24.4%   g_actor 1.7e-01
train   PnL   +870.25$  1134 trades  WR 45.8%  PF 1.30   ->  ecart train-val +8.1 pt
. etendue val 0.3937, soit 394x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 37s  PPO 49s  calib 13s  validation 42s

## 14:48:41 — wf1 — EPOCH 010   -0.64$/trade   264 trades   WR 36.0%   PF 0.79   2 min

PnL         -167.75$   cumul run     +192.40$   DD 4.5%   Sortino -0.159
point mort 41.6%  ->  ecart -5.6 pt (+/- 3.0 au mieux)
vs politique gelee du meme run : -6.8 pt   (reference +1.3 pt sur 5 epochs)
vs epoch precedente : -0.26$/trade
sens    LONG    49W/80  L  38.0%      -3.66$   |   SHORT   46W/89  L  34.1%    -164.09$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 9.5%  val 5.0%
politique  H 1.046/1.099 (-0.015)   etendue 0.4186 (419x le tremblement)   KL +0.0240   clipfrac 26.8%   g_actor 1.6e-01
train   PnL  +1100.95$  1148 trades  WR 45.8%  PF 1.38   ->  ecart train-val +9.8 pt
. etendue val 0.4186, soit 419x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 37s  PPO 46s  calib 15s  validation 41s

## 14:51:01 — wf1 — EPOCH 011   +0.08$/trade   247 trades   WR 41.3%   PF 1.03   2 min

PnL          +18.65$   cumul run     +211.05$   DD 3.3%   Sortino +0.019
point mort 40.6%  ->  ecart +0.7 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : -0.6 pt   (reference +1.3 pt sur 5 epochs)
vs epoch precedente : +0.71$/trade
sens    LONG    60W/76  L  44.1%    +142.61$   |   SHORT   42W/69  L  37.8%    -123.96$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.036/1.099 (-0.010)   etendue 0.4580 (458x le tremblement)   KL +0.0186   clipfrac 26.1%   g_actor 1.7e-01
train   PnL  +1324.38$  1160 trades  WR 48.0%  PF 1.47   ->  ecart train-val +6.7 pt
. etendue val 0.4580, soit 458x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.08$) sous l'erreur-type de cette epoch (0.33$) — non separable de zero.
temps : collecte 43s  PPO 42s  calib 10s  validation 43s

## 14:53:01 — wf1 — EPOCH 012   -0.23$/trade   246 trades   WR 38.6%   PF 0.92   2 min

PnL          -56.16$   cumul run     +154.89$   DD 4.2%   Sortino -0.059
point mort 40.6%  ->  ecart -2.0 pt (+/- 3.1 au mieux)
vs politique gelee du meme run : -3.3 pt   (reference +1.3 pt sur 5 epochs)
vs epoch precedente : -0.30$/trade
sens    LONG    54W/80  L  40.3%      +0.12$   |   SHORT   41W/71  L  36.6%     -56.28$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.030/1.099 (-0.006)   etendue 0.4690 (469x le tremblement)   KL +0.0275   clipfrac 24.0%   g_actor 1.6e-01
train   PnL   +751.29$  1115 trades  WR 47.1%  PF 1.27   ->  ecart train-val +8.5 pt
. etendue val 0.4690, soit 469x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.23$) sous l'erreur-type de cette epoch (0.36$) — non separable de zero.
temps : collecte 40s  PPO 36s  calib 7s  validation 46s

## 14:56:57 — wf1 — EPOCH 001   -0.04$/trade   178 trades   WR 38.2%   PF 0.99   2 min

PnL           -7.31$   cumul run       -7.31$   DD 2.7%   Sortino -0.011
point mort 38.4%  ->  ecart -0.2 pt (+/- 3.6 au mieux)
sens    LONG    42W/63  L  40.0%     +46.95$   |   SHORT   26W/47  L  35.6%     -54.27$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.6e-06
train   PnL   +449.74$  780 trades  WR 43.6%  PF 1.24   ->  ecart train-val +5.4 pt
. ACTOR GELE : warmup du critic, gradient 1.6e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.04$) sous l'erreur-type de cette epoch (0.63$) — non separable de zero.
temps : collecte 31s  PPO 24s  calib 6s  validation 40s

## 14:58:37 — wf1 — EPOCH 002   +0.00$/trade   156 trades   WR 42.3%   PF 1.00   2 min

PnL           +0.33$   cumul run       -6.98$   DD 2.9%   Sortino +0.001
point mort 42.3%  ->  ecart -0.0 pt (+/- 4.0 au mieux)
vs epoch precedente : +0.04$/trade
sens    LONG    41W/46  L  47.1%     +83.23$   |   SHORT   25W/44  L  36.2%     -82.90$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 8.4e-07
train   PnL   +560.11$  788 trades  WR 44.5%  PF 1.30   ->  ecart train-val +2.2 pt
. ACTOR GELE : warmup du critic, gradient 8.4e-07. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.00$) sous l'erreur-type de cette epoch (0.39$) — non separable de zero.
temps : collecte 30s  PPO 25s  calib 7s  validation 38s

## 15:00:37 — wf1 — EPOCH 003   +0.28$/trade   160 trades   WR 41.2%   PF 1.11   2 min

PnL          +44.23$   cumul run      +37.25$   DD 3.4%   Sortino +0.075
point mort 38.7%  ->  ecart +2.5 pt (+/- 3.9 au mieux)
vs politique gelee du meme run : +2.6 pt   (reference -0.1 pt sur 2 epochs)
vs epoch precedente : +0.27$/trade
sens    LONG    47W/57  L  45.2%    +114.97$   |   SHORT   19W/37  L  33.9%     -70.74$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 2.0e-06
train   PnL    +62.83$  827 trades  WR 43.3%  PF 1.03   ->  ecart train-val +2.1 pt
. ACTOR GELE : warmup du critic, gradient 2.0e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.28$) sous l'erreur-type de cette epoch (0.43$) — non separable de zero.
. GPU BRIDE : 450 MHz a 84 C. Les durees ne sont pas comparables entre epochs si la temperature derive.
temps : collecte 32s  PPO 26s  calib 7s  validation 44s

## 15:02:37 — wf1 — EPOCH 004   +0.08$/trade   140 trades   WR 40.0%   PF 1.03   2 min

PnL          +10.60$   cumul run      +47.85$   DD 2.7%   Sortino +0.020
point mort 39.3%  ->  ecart +0.7 pt (+/- 4.1 au mieux)
vs politique gelee du meme run : -0.0 pt   (reference +0.7 pt sur 3 epochs)
vs epoch precedente : -0.20$/trade
sens    LONG    34W/48  L  41.5%     +34.91$   |   SHORT   22W/36  L  37.9%     -24.31$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 6.0e-06
train   PnL   +491.80$  790 trades  WR 45.4%  PF 1.26   ->  ecart train-val +5.4 pt
. ACTOR GELE : warmup du critic, gradient 6.0e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.08$) sous l'erreur-type de cette epoch (0.44$) — non separable de zero.
temps : collecte 35s  PPO 25s  calib 7s  validation 53s

## 15:04:17 — wf1 — EPOCH 005   +0.90$/trade   131 trades   WR 49.6%   PF 1.44   2 min

PnL         +117.40$   cumul run     +165.25$   DD 1.8%   Sortino +0.278
point mort 40.6%  ->  ecart +9.0 pt (+/- 4.4 au mieux)
vs politique gelee du meme run : +8.3 pt   (reference +0.7 pt sur 4 epochs)
vs epoch precedente : +0.82$/trade
sens    LONG    36W/27  L  57.1%    +128.65$   |   SHORT   29W/39  L  42.6%     -11.25$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 32.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0007 (1x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 1.3e-05
train   PnL   +106.48$  802 trades  WR 42.6%  PF 1.05   ->  ecart train-val -7.0 pt
. ACTOR GELE : warmup du critic, gradient 1.3e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0007 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 31s  PPO 25s  calib 7s  validation 43s

## 15:05:57 — wf1 — EPOCH 006   +0.01$/trade   173 trades   WR 39.9%   PF 1.00   2 min

PnL           +1.17$   cumul run     +166.42$   DD 2.6%   Sortino +0.002
point mort 39.9%  ->  ecart +0.0 pt (+/- 3.7 au mieux)
vs politique gelee du meme run : -2.4 pt   (reference +2.4 pt sur 5 epochs)
vs epoch precedente : -0.89$/trade
sens    LONG    47W/69  L  40.5%     +60.40$   |   SHORT   22W/35  L  38.6%     -59.23$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 27.5%  val 5.0%
politique  H 1.093/1.099 (-0.006)   etendue 0.2065 (206x le tremblement)   KL +0.0073   clipfrac 10.6%   g_actor 2.5e-01
train   PnL   +248.83$  825 trades  WR 42.7%  PF 1.12   ->  ecart train-val +2.8 pt
. APPREND MAIS RESTE PLAT : le gradient passe (2.5e-01) mais l'entropie tient a 1.093 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.2065, soit 206x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.01$) sous l'erreur-type de cette epoch (0.40$) — non separable de zero.
temps : collecte 30s  PPO 27s  calib 7s  validation 41s

## 15:07:37 — wf1 — EPOCH 007   +0.34$/trade   174 trades   WR 39.7%   PF 1.13   2 min

PnL          +58.34$   cumul run     +224.76$   DD 2.5%   Sortino +0.091
point mort 36.8%  ->  ecart +2.9 pt (+/- 3.7 au mieux)
vs politique gelee du meme run : +0.5 pt   (reference +2.4 pt sur 5 epochs)
vs epoch precedente : +0.33$/trade
sens    LONG    45W/60  L  42.9%    +107.83$   |   SHORT   24W/45  L  34.8%     -49.49$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 23.0%  val 5.0%
politique  H 1.081/1.099 (-0.012)   etendue 0.3120 (312x le tremblement)   KL +0.0118   clipfrac 19.2%   g_actor 2.6e-01
train   PnL    +30.34$  795 trades  WR 42.8%  PF 1.02   ->  ecart train-val +3.1 pt
. etendue val 0.3120, soit 312x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.34$) sous l'erreur-type de cette epoch (0.43$) — non separable de zero.
temps : collecte 30s  PPO 26s  calib 6s  validation 34s

## 15:16:58 — wf1 — EPOCH 008   -0.40$/trade   176 trades   WR 36.9%   PF 0.86   9 min

PnL          -70.76$   cumul run     +154.00$   DD 2.8%   Sortino -0.105
point mort 40.5%  ->  ecart -3.6 pt (+/- 3.6 au mieux)
vs politique gelee du meme run : -6.0 pt   (reference +2.4 pt sur 5 epochs)
vs epoch precedente : -0.74$/trade
sens    LONG    48W/66  L  42.1%     +33.20$   |   SHORT   17W/45  L  27.4%    -103.96$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 18.5%  val 5.0%
politique  H 1.072/1.099 (-0.009)   etendue 0.3432 (343x le tremblement)   KL +0.0182   clipfrac 23.2%   g_actor 2.3e-01
train   PnL   +490.98$  808 trades  WR 45.4%  PF 1.25   ->  ecart train-val +8.5 pt
. etendue val 0.3432, soit 343x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.40$) sous l'erreur-type de cette epoch (0.41$) — non separable de zero.
. GPU BRIDE : 210 MHz a 93 C. Les durees ne sont pas comparables entre epochs si la temperature derive.
temps : collecte 38s  PPO 122s  calib 81s  validation 324s

## 15:34:53 — wf1 — EPOCH 001   -0.04$/trade   178 trades   WR 38.2%   PF 0.99   2 min

PnL           -7.31$   cumul run       -7.31$   DD 2.7%   Sortino -0.011
point mort 38.4%  ->  ecart -0.2 pt (+/- 3.6 au mieux)
classement rho [90m-0.0017[0m (+/- 0.024 env.)  -> n'ordonne rien
sens    LONG    42W/63  L  40.0%     +46.95$   |   SHORT   26W/47  L  35.6%     -54.27$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.6e-06
train   PnL   +449.74$  780 trades  WR 43.6%  PF 1.24   ->  ecart train-val +5.4 pt
. ACTOR GELE : warmup du critic, gradient 1.6e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.04$) sous l'erreur-type de cette epoch (0.63$) — non separable de zero.
temps : collecte 37s  PPO 25s  calib 6s  validation 44s

## 15:34:53 — wf1 — EPOCH 002   +0.00$/trade   156 trades   WR 42.3%   PF 1.00   2 min

PnL           +0.33$   cumul run       -6.98$   DD 2.9%   Sortino +0.001
point mort 42.3%  ->  ecart -0.0 pt (+/- 4.0 au mieux)
classement rho [90m-0.0077[0m (+/- 0.024 env.)  -> n'ordonne rien
vs epoch precedente : +0.04$/trade
sens    LONG    41W/46  L  47.1%     +83.23$   |   SHORT   25W/44  L  36.2%     -82.90$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 8.4e-07
train   PnL   +560.11$  788 trades  WR 44.5%  PF 1.30   ->  ecart train-val +2.2 pt
. ACTOR GELE : warmup du critic, gradient 8.4e-07. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.00$) sous l'erreur-type de cette epoch (0.39$) — non separable de zero.
temps : collecte 32s  PPO 26s  calib 6s  validation 42s

## 15:34:53 — wf1 — EPOCH 003   +0.28$/trade   160 trades   WR 41.2%   PF 1.11   2 min

PnL          +44.23$   cumul run      +37.25$   DD 3.4%   Sortino +0.075
point mort 38.7%  ->  ecart +2.5 pt (+/- 3.9 au mieux)
classement rho [91m-0.0284[0m (+/- 0.024 env.)  -> classe A L'ENVERS
vs politique gelee du meme run : +2.6 pt   (reference -0.1 pt sur 2 epochs)
vs epoch precedente : +0.27$/trade
sens    LONG    47W/57  L  45.2%    +114.97$   |   SHORT   19W/37  L  33.9%     -70.74$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 2.0e-06
train   PnL    +62.83$  827 trades  WR 43.3%  PF 1.03   ->  ecart train-val +2.1 pt
. ACTOR GELE : warmup du critic, gradient 2.0e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.28$) sous l'erreur-type de cette epoch (0.43$) — non separable de zero.
temps : collecte 33s  PPO 26s  calib 7s  validation 40s

## 15:34:53 — wf1 — EPOCH 004   +0.08$/trade   140 trades   WR 40.0%   PF 1.03   2 min

PnL          +10.60$   cumul run      +47.85$   DD 2.7%   Sortino +0.020
point mort 39.3%  ->  ecart +0.7 pt (+/- 4.1 au mieux)
classement rho [91m-0.0276[0m (+/- 0.024 env.)  -> classe A L'ENVERS
vs politique gelee du meme run : -0.0 pt   (reference +0.7 pt sur 3 epochs)
vs epoch precedente : -0.20$/trade
sens    LONG    34W/48  L  41.5%     +34.91$   |   SHORT   22W/36  L  37.9%     -24.31$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 6.0e-06
train   PnL   +491.80$  790 trades  WR 45.4%  PF 1.26   ->  ecart train-val +5.4 pt
. ACTOR GELE : warmup du critic, gradient 6.0e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.08$) sous l'erreur-type de cette epoch (0.44$) — non separable de zero.
temps : collecte 32s  PPO 26s  calib 6s  validation 48s

## 15:34:53 — wf1 — EPOCH 005   +0.90$/trade   131 trades   WR 49.6%   PF 1.44   2 min

PnL         +117.40$   cumul run     +165.25$   DD 1.8%   Sortino +0.278
point mort 40.6%  ->  ecart +9.0 pt (+/- 4.4 au mieux)
classement rho [91m-0.0217[0m (+/- 0.024 env.)  -> classe A L'ENVERS
vs politique gelee du meme run : +8.3 pt   (reference +0.7 pt sur 4 epochs)
vs epoch precedente : +0.82$/trade
sens    LONG    36W/27  L  57.1%    +128.65$   |   SHORT   29W/39  L  42.6%     -11.25$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 32.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0007 (1x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 1.3e-05
train   PnL   +106.48$  802 trades  WR 42.6%  PF 1.05   ->  ecart train-val -7.0 pt
. ACTOR GELE : warmup du critic, gradient 1.3e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0007 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 32s  PPO 25s  calib 7s  validation 35s

## 15:34:53 — wf1 — EPOCH 006   +0.01$/trade   173 trades   WR 39.9%   PF 1.00   2 min

PnL           +1.17$   cumul run     +166.42$   DD 2.6%   Sortino +0.002
point mort 39.9%  ->  ecart +0.0 pt (+/- 3.7 au mieux)
classement rho [92m+0.0258[0m (+/- 0.024 env.)  -> classe
vs politique gelee du meme run : -2.4 pt   (reference +2.4 pt sur 5 epochs)
vs epoch precedente : -0.89$/trade
sens    LONG    47W/69  L  40.5%     +60.40$   |   SHORT   22W/35  L  38.6%     -59.23$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 27.5%  val 5.0%
politique  H 1.093/1.099 (-0.006)   etendue 0.2065 (206x le tremblement)   KL +0.0073   clipfrac 10.6%   g_actor 2.5e-01
train   PnL   +248.83$  825 trades  WR 42.7%  PF 1.12   ->  ecart train-val +2.8 pt
. APPREND MAIS RESTE PLAT : le gradient passe (2.5e-01) mais l'entropie tient a 1.093 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.2065, soit 206x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.01$) sous l'erreur-type de cette epoch (0.40$) — non separable de zero.
temps : collecte 29s  PPO 27s  calib 7s  validation 40s

## 15:36:33 — wf1 — EPOCH 007   +0.34$/trade   174 trades   WR 39.7%   PF 1.13   2 min

PnL          +58.34$   cumul run     +224.76$   DD 2.5%   Sortino +0.091
point mort 36.8%  ->  ecart +2.9 pt (+/- 3.7 au mieux)
classement rho [92m+0.0226[0m (+/- 0.024 env.)  -> classe
vs politique gelee du meme run : +0.5 pt   (reference +2.4 pt sur 5 epochs)
vs epoch precedente : +0.33$/trade
sens    LONG    45W/60  L  42.9%    +107.83$   |   SHORT   24W/45  L  34.8%     -49.49$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 23.0%  val 5.0%
politique  H 1.081/1.099 (-0.012)   etendue 0.3120 (312x le tremblement)   KL +0.0118   clipfrac 19.2%   g_actor 2.6e-01
train   PnL    +30.34$  795 trades  WR 42.8%  PF 1.02   ->  ecart train-val +3.1 pt
. etendue val 0.3120, soit 312x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.34$) sous l'erreur-type de cette epoch (0.43$) — non separable de zero.
temps : collecte 32s  PPO 25s  calib 6s  validation 32s

## 15:41:13 — wf1 — EPOCH 008   -0.40$/trade   176 trades   WR 36.9%   PF 0.86   5 min

PnL          -70.76$   cumul run     +154.00$   DD 2.8%   Sortino -0.105
point mort 40.5%  ->  ecart -3.6 pt (+/- 3.6 au mieux)
classement rho [92m+0.0219[0m (+/- 0.024 env.)  -> classe
vs politique gelee du meme run : -6.0 pt   (reference +2.4 pt sur 5 epochs)
vs epoch precedente : -0.74$/trade
sens    LONG    48W/66  L  42.1%     +33.20$   |   SHORT   17W/45  L  27.4%    -103.96$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 18.5%  val 5.0%
politique  H 1.072/1.099 (-0.009)   etendue 0.3432 (343x le tremblement)   KL +0.0182   clipfrac 23.2%   g_actor 2.3e-01
train   PnL   +490.98$  808 trades  WR 45.4%  PF 1.25   ->  ecart train-val +8.5 pt
. etendue val 0.3432, soit 343x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.40$) sous l'erreur-type de cette epoch (0.41$) — non separable de zero.
. GPU BRIDE : 480 MHz a 89 C. Les durees ne sont pas comparables entre epochs si la temperature derive.
temps : collecte 29s  PPO 39s  calib 73s  validation 137s

## 15:43:13 — wf1 — EPOCH 009   +0.01$/trade   186 trades   WR 39.8%   PF 1.00   2 min

PnL           +2.46$   cumul run     +156.46$   DD 3.2%   Sortino +0.004
point mort 39.8%  ->  ecart +0.0 pt (+/- 3.6 au mieux)
classement rho [90m+0.0006[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : -2.4 pt   (reference +2.4 pt sur 5 epochs)
vs epoch precedente : +0.42$/trade
sens    LONG    50W/64  L  43.9%     +52.22$   |   SHORT   24W/48  L  33.3%     -49.76$
actions B 0.1%  S 0.1%  H 99.8%   |   selectivite  train 14.0%  val 5.0%
politique  H 1.061/1.099 (-0.011)   etendue 0.3720 (372x le tremblement)   KL +0.0218   clipfrac 21.0%   g_actor 2.4e-01
train   PnL   +389.40$  792 trades  WR 47.0%  PF 1.21   ->  ecart train-val +7.2 pt
. etendue val 0.3720, soit 372x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.01$) sous l'erreur-type de cette epoch (0.39$) — non separable de zero.
temps : collecte 34s  PPO 27s  calib 8s  validation 37s

## 15:46:13 — wf1 — EPOCH 001   +0.47$/trade   114 trades   WR 43.0%   PF 1.16   2 min

PnL          +53.89$   cumul run      +53.89$   DD 4.1%   Sortino +0.121
point mort 39.4%  ->  ecart +3.6 pt (+/- 4.6 au mieux)
classement rho [92m+0.0360[0m (+/- 0.024 env.)  -> classe
sens    LONG    18W/20  L  47.4%     +75.09$   |   SHORT   31W/45  L  40.8%     -21.19$
actions B 0.0%  S 0.0%  H 99.9%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0010 (1x le tremblement)   KL +0.0001   clipfrac 0.0%   g_actor 8.9e-06
train   PnL   +273.97$  382 trades  WR 44.0%  PF 1.26   ->  ecart train-val +1.0 pt
. ACTOR GELE : warmup du critic, gradient 8.9e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0010 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.47$) sous l'erreur-type de cette epoch (0.61$) — non separable de zero.
temps : collecte 26s  PPO 10s  calib 16s  validation 82s

## 15:47:53 — wf1 — EPOCH 002   -0.48$/trade   102 trades   WR 30.4%   PF 0.86   2 min

PnL          -48.91$   cumul run       +4.98$   DD 4.4%   Sortino -0.113
point mort 33.7%  ->  ecart -3.3 pt (+/- 4.6 au mieux)
classement rho [92m+0.0228[0m (+/- 0.024 env.)  -> classe
vs epoch precedente : -0.95$/trade
sens    LONG    15W/35  L  30.0%     +27.10$   |   SHORT   16W/36  L  30.8%     -76.01$
actions B 0.0%  S 0.0%  H 99.9%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0002 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 1.9e-06
train   PnL   +563.98$  330 trades  WR 48.8%  PF 1.70   ->  ecart train-val +18.4 pt
. ACTOR GELE : warmup du critic, gradient 1.9e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0002 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.48$) sous l'erreur-type de cette epoch (0.67$) — non separable de zero.
temps : collecte 22s  PPO 7s  calib 15s  validation 57s

## 15:49:54 — wf1 — EPOCH 003   +1.54$/trade   94 trades   WR 44.7%   PF 1.57   2 min

PnL         +145.08$   cumul run     +150.06$   DD 4.0%   Sortino +0.407
point mort 34.0%  ->  ecart +10.7 pt (+/- 5.1 au mieux)
classement rho [90m+0.0152[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : +10.6 pt   (reference +0.2 pt sur 2 epochs)
vs epoch precedente : +2.02$/trade
sens    LONG    21W/21  L  50.0%    +132.24$   |   SHORT   21W/31  L  40.4%     +12.83$
actions B 0.0%  S 0.0%  H 99.9%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0003 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.2e-06
train   PnL    +46.68$  373 trades  WR 38.1%  PF 1.04   ->  ecart train-val -6.6 pt
. ACTOR GELE : warmup du critic, gradient 1.2e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0003 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 26s  PPO 9s  calib 14s  validation 69s

## 15:51:34 — wf1 — EPOCH 004   +0.20$/trade   60 trades   WR 35.0%   PF 1.06   2 min

PnL          +12.17$   cumul run     +162.23$   DD 3.1%   Sortino +0.049
point mort 33.7%  ->  ecart +1.3 pt (+/- 6.2 au mieux)
classement rho [91m-0.0217[0m (+/- 0.024 env.)  -> classe A L'ENVERS
vs politique gelee du meme run : -2.4 pt   (reference +3.7 pt sur 3 epochs)
vs epoch precedente : -1.34$/trade
sens    LONG    16W/14  L  53.3%     +70.01$   |   SHORT    5W/25  L  16.7%     -57.84$
actions B 0.0%  S 0.0%  H 99.9%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 1.3e-06
train   PnL    +99.83$  353 trades  WR 42.2%  PF 1.10   ->  ecart train-val +7.2 pt
. ACTOR GELE : warmup du critic, gradient 1.3e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.20$) sous l'erreur-type de cette epoch (0.95$) — non separable de zero.
temps : collecte 23s  PPO 9s  calib 15s  validation 54s

## 15:53:14 — wf1 — EPOCH 005   +0.96$/trade   84 trades   WR 42.9%   PF 1.33   2 min

PnL          +80.46$   cumul run     +242.69$   DD 2.9%   Sortino +0.248
point mort 36.1%  ->  ecart +6.8 pt (+/- 5.4 au mieux)
classement rho [92m+0.0323[0m (+/- 0.024 env.)  -> classe
vs politique gelee du meme run : +3.7 pt   (reference +3.1 pt sur 4 epochs)
vs epoch precedente : +0.76$/trade
sens    LONG    15W/17  L  46.9%     +79.49$   |   SHORT   21W/31  L  40.4%      +0.98$
actions B 0.0%  S 0.0%  H 99.9%   |   selectivite  train 32.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0003 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 3.7e-06
train   PnL   +255.55$  363 trades  WR 43.3%  PF 1.26   ->  ecart train-val +0.4 pt
. ACTOR GELE : warmup du critic, gradient 3.7e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0003 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 24s  PPO 9s  calib 15s  validation 60s

## 15:55:14 — wf1 — EPOCH 006   +0.35$/trade   80 trades   WR 37.5%   PF 1.11   2 min

PnL          +27.72$   cumul run     +270.41$   DD 4.9%   Sortino +0.087
point mort 35.1%  ->  ecart +2.4 pt (+/- 5.4 au mieux)
classement rho [92m+0.0336[0m (+/- 0.024 env.)  -> classe
vs politique gelee du meme run : -1.4 pt   (reference +3.8 pt sur 5 epochs)
vs epoch precedente : -0.61$/trade
sens    LONG    15W/25  L  37.5%     +32.57$   |   SHORT   15W/25  L  37.5%      -4.86$
actions B 0.0%  S 0.0%  H 99.9%   |   selectivite  train 27.5%  val 5.0%
politique  H 1.096/1.099 (-0.003)   etendue 0.1107 (111x le tremblement)   KL +0.0066   clipfrac 4.3%   g_actor 2.8e-01
train   PnL    +73.97$  370 trades  WR 39.2%  PF 1.07   ->  ecart train-val +1.7 pt
. APPREND MAIS RESTE PLAT : le gradient passe (2.8e-01) mais l'entropie tient a 1.096 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.1107, soit 111x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.35$) sous l'erreur-type de cette epoch (0.78$) — non separable de zero.
temps : collecte 27s  PPO 9s  calib 16s  validation 63s

## 15:56:54 — wf1 — EPOCH 007   +0.64$/trade   80 trades   WR 42.5%   PF 1.22   2 min

PnL          +51.30$   cumul run     +321.71$   DD 3.8%   Sortino +0.165
point mort 37.7%  ->  ecart +4.8 pt (+/- 5.5 au mieux)
classement rho [92m+0.0473[0m (+/- 0.024 env.)  -> classe
vs politique gelee du meme run : +0.9 pt   (reference +3.8 pt sur 5 epochs)
vs epoch precedente : +0.29$/trade
sens    LONG    19W/23  L  45.2%     +66.17$   |   SHORT   15W/23  L  39.5%     -14.87$
actions B 0.0%  S 0.0%  H 99.9%   |   selectivite  train 23.0%  val 5.0%
politique  H 1.089/1.099 (-0.007)   etendue 0.1704 (170x le tremblement)   KL +0.0001   clipfrac 1.9%   g_actor 3.4e-01
train   PnL   +206.77$  355 trades  WR 42.5%  PF 1.20   ->  ecart train-val +0.0 pt
. APPREND MAIS RESTE PLAT : le gradient passe (3.4e-01) mais l'entropie tient a 1.089 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.1704, soit 170x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.64$) sous l'erreur-type de cette epoch (0.74$) — non separable de zero.
temps : collecte 23s  PPO 9s  calib 14s  validation 52s

## 15:58:54 — wf1 — EPOCH 008   +0.77$/trade   76 trades   WR 44.7%   PF 1.27   2 min

PnL          +58.27$   cumul run     +379.98$   DD 4.2%   Sortino +0.200
point mort 38.9%  ->  ecart +5.8 pt (+/- 5.7 au mieux)
classement rho [92m+0.0507[0m (+/- 0.024 env.)  -> classe nettement
vs politique gelee du meme run : +1.9 pt   (reference +3.8 pt sur 5 epochs)
vs epoch precedente : +0.13$/trade
sens    LONG    19W/27  L  41.3%     +38.09$   |   SHORT   15W/15  L  50.0%     +20.18$
actions B 0.0%  S 0.0%  H 99.9%   |   selectivite  train 18.5%  val 5.0%
politique  H 1.079/1.099 (-0.010)   etendue 0.2381 (238x le tremblement)   KL +0.0080   clipfrac 9.5%   g_actor 2.5e-01
train   PnL   +110.10$  370 trades  WR 40.0%  PF 1.10   ->  ecart train-val -4.7 pt
. etendue val 0.2381, soit 238x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 24s  PPO 9s  calib 16s  validation 58s

## 16:00:54 — wf1 — EPOCH 009   +0.72$/trade   64 trades   WR 45.3%   PF 1.26   2 min

PnL          +46.14$   cumul run     +426.12$   DD 3.5%   Sortino +0.186
point mort 39.7%  ->  ecart +5.6 pt (+/- 6.2 au mieux)
classement rho [92m+0.0384[0m (+/- 0.024 env.)  -> classe
vs politique gelee du meme run : +1.8 pt   (reference +3.8 pt sur 5 epochs)
vs epoch precedente : -0.05$/trade
sens    LONG    16W/19  L  45.7%     +73.73$   |   SHORT   13W/16  L  44.8%     -27.58$
actions B 0.0%  S 0.0%  H 99.9%   |   selectivite  train 14.0%  val 5.0%
politique  H 1.072/1.099 (-0.007)   etendue 0.2937 (294x le tremblement)   KL +0.0018   clipfrac 4.7%   g_actor 2.9e-01
train   PnL    +83.76$  345 trades  WR 42.3%  PF 1.09   ->  ecart train-val -3.0 pt
. etendue val 0.2937, soit 294x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.72$) sous l'erreur-type de cette epoch (0.80$) — non separable de zero.
temps : collecte 29s  PPO 8s  calib 19s  validation 74s

## 16:03:14 — wf1 — EPOCH 010   +1.86$/trade   67 trades   WR 47.8%   PF 1.73   2 min

PnL         +124.42$   cumul run     +550.54$   DD 2.8%   Sortino +0.505
point mort 34.6%  ->  ecart +13.2 pt (+/- 6.1 au mieux)
classement rho [92m+0.0258[0m (+/- 0.024 env.)  -> classe
vs politique gelee du meme run : +9.4 pt   (reference +3.8 pt sur 5 epochs)
vs epoch precedente : +1.14$/trade
sens    LONG    18W/13  L  58.1%    +116.10$   |   SHORT   14W/22  L  38.9%      +8.33$
actions B 0.0%  S 0.0%  H 99.9%   |   selectivite  train 9.5%  val 5.0%
politique  H 1.053/1.099 (-0.019)   etendue 0.3530 (353x le tremblement)   KL +0.0153   clipfrac 17.3%   g_actor 3.3e-01
train   PnL   +423.51$  355 trades  WR 46.5%  PF 1.45   ->  ecart train-val -1.3 pt
. etendue val 0.3530, soit 353x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 32s  PPO 9s  calib 20s  validation 77s

## 16:05:54 — wf1 — EPOCH 011   -0.12$/trade   73 trades   WR 35.6%   PF 0.96   2 min

PnL           -9.12$   cumul run     +541.42$   DD 3.0%   Sortino -0.031
point mort 36.6%  ->  ecart -1.0 pt (+/- 5.6 au mieux)
classement rho [90m+0.0105[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : -4.8 pt   (reference +3.8 pt sur 5 epochs)
vs epoch precedente : -1.98$/trade
sens    LONG    15W/20  L  42.9%     +39.14$   |   SHORT   11W/27  L  28.9%     -48.26$
actions B 0.0%  S 0.0%  H 99.9%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.059/1.099 (+0.006)   etendue 0.3745 (374x le tremblement)   KL +0.0113   clipfrac 15.5%   g_actor 2.7e-01
train   PnL   +294.34$  358 trades  WR 45.0%  PF 1.32   ->  ecart train-val +9.4 pt
. etendue val 0.3745, soit 374x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.12$) sous l'erreur-type de cette epoch (0.74$) — non separable de zero.
temps : collecte 35s  PPO 9s  calib 20s  validation 82s

## 16:08:14 — wf1 — EPOCH 012   +0.25$/trade   97 trades   WR 38.1%   PF 1.08   2 min

PnL          +23.91$   cumul run     +565.33$   DD 4.5%   Sortino +0.060
point mort 36.3%  ->  ecart +1.8 pt (+/- 4.9 au mieux)
classement rho [90m+0.0019[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : -2.1 pt   (reference +3.8 pt sur 5 epochs)
vs epoch precedente : +0.37$/trade
sens    LONG    18W/21  L  46.2%     +36.00$   |   SHORT   19W/39  L  32.8%     -12.08$
actions B 0.0%  S 0.0%  H 99.9%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.049/1.099 (-0.010)   etendue 0.3616 (362x le tremblement)   KL +0.0102   clipfrac 18.0%   g_actor 2.4e-01
train   PnL   +736.65$  317 trades  WR 50.5%  PF 1.96   ->  ecart train-val +12.4 pt
. etendue val 0.3616, soit 362x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.25$) sous l'erreur-type de cette epoch (0.68$) — non separable de zero.
temps : collecte 31s  PPO 7s  calib 16s  validation 90s

## 16:10:54 — wf1 — EPOCH 013   +0.38$/trade   83 trades   WR 37.3%   PF 1.12   3 min

PnL          +31.37$   cumul run     +596.70$   DD 3.8%   Sortino +0.092
point mort 34.7%  ->  ecart +2.6 pt (+/- 5.3 au mieux)
classement rho [90m+0.0003[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : -1.3 pt   (reference +3.8 pt sur 5 epochs)
vs epoch precedente : +0.13$/trade
sens    LONG    17W/20  L  45.9%     +44.83$   |   SHORT   14W/32  L  30.4%     -13.46$
actions B 0.0%  S 0.0%  H 99.9%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.053/1.099 (+0.004)   etendue 0.3916 (392x le tremblement)   KL +0.0195   clipfrac 25.4%   g_actor 2.5e-01
train   PnL   +354.11$  353 trades  WR 48.2%  PF 1.40   ->  ecart train-val +10.9 pt
. etendue val 0.3916, soit 392x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.38$) sous l'erreur-type de cette epoch (0.77$) — non separable de zero.
temps : collecte 33s  PPO 8s  calib 20s  validation 90s

## 17:02:37 — wf1 — EPOCH 001   -0.61$/trade   108 trades   WR 36.1%   PF 0.79   5 min

PnL          -66.11$   cumul run      -66.11$   DD 4.6%   Sortino -0.138
point mort 41.7%  ->  ecart -5.6 pt (+/- 4.6 au mieux)
classement rho [90m+0.0052[0m (+/- 0.024 env.)  -> n'ordonne rien
sens    LONG    16W/24  L  40.0%     +32.48$   |   SHORT   23W/45  L  33.8%     -98.59$
actions B 1.0%  S 1.1%  H 97.9%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0002 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 1.2e-05
train   PnL     -6.05$  2443 trades  WR 39.4%  PF 1.00   ->  ecart train-val +3.3 pt
. ACTOR GELE : warmup du critic, gradient 1.2e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0002 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 92s  PPO 70s  calib 22s  validation 122s

## 17:09:17 — wf1 — EPOCH 002   +0.96$/trade   65 trades   WR 40.0%   PF 1.24   7 min

PnL          +62.37$   cumul run       -3.74$   DD 4.6%   Sortino +0.158
point mort 35.0%  ->  ecart +5.0 pt (+/- 6.1 au mieux)
classement rho [90m-0.0185[0m (+/- 0.024 env.)  -> n'ordonne rien
vs epoch precedente : +1.57$/trade
sens    LONG    22W/30  L  42.3%    +111.18$   |   SHORT    4W/9   L  30.8%     -48.80$
actions B 1.3%  S 1.3%  H 97.4%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 2.7e-06
train   PnL  +1103.06$  2963 trades  WR 42.3%  PF 1.22   ->  ecart train-val +2.3 pt
. ACTOR GELE : warmup du critic, gradient 2.7e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.96$) sous l'erreur-type de cette epoch (1.16$) — non separable de zero.
temps : collecte 112s  PPO 93s  calib 38s  validation 158s

## 17:14:37 — wf1 — EPOCH 003   +0.51$/trade   84 trades   WR 33.3%   PF 1.13   5 min

PnL          +43.22$   cumul run      +39.48$   DD 8.0%   Sortino +0.080
point mort 30.7%  ->  ecart +2.6 pt (+/- 5.1 au mieux)
classement rho [90m+0.0157[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : +2.9 pt   (reference -0.3 pt sur 2 epochs)
vs epoch precedente : -0.45$/trade
sens    LONG    19W/29  L  39.6%    +108.24$   |   SHORT    9W/27  L  25.0%     -65.02$
actions B 0.9%  S 0.9%  H 98.2%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.2e-07
train   PnL  +1290.37$  2047 trades  WR 45.2%  PF 1.25   ->  ecart train-val +11.9 pt
. ACTOR GELE : warmup du critic, gradient 1.2e-07. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.51$) sous l'erreur-type de cette epoch (1.00$) — non separable de zero.
temps : collecte 91s  PPO 55s  calib 29s  validation 128s

## 17:20:57 — wf1 — EPOCH 004   +1.22$/trade   100 trades   WR 46.0%   PF 1.55   6 min

PnL         +122.24$   cumul run     +161.72$   DD 3.5%   Sortino +0.351
point mort 35.5%  ->  ecart +10.5 pt (+/- 5.0 au mieux)
classement rho [92m+0.0204[0m (+/- 0.024 env.)  -> classe
vs politique gelee du meme run : +9.8 pt   (reference +0.7 pt sur 3 epochs)
vs epoch precedente : +0.71$/trade
sens    LONG    23W/30  L  43.4%    +110.77$   |   SHORT   23W/24  L  48.9%     +11.47$
actions B 1.2%  S 1.3%  H 97.5%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 5.8e-05
train   PnL   +801.23$  2931 trades  WR 42.2%  PF 1.20   ->  ecart train-val -3.8 pt
. ACTOR GELE : warmup du critic, gradient 5.8e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 110s  PPO 88s  calib 29s  validation 149s

## 17:55:28 — wf1 — EPOCH 001   -0.35$/trade   181 trades   WR 27.1%   PF 0.85   6 min

PnL          -62.70$   cumul run      -62.70$   DD 7.8%   Sortino -0.097
point mort 30.4%  ->  ecart -3.3 pt (+/- 3.3 au mieux)
classement rho [91m-0.0358[0m (+/- 0.024 env.)  -> classe A L'ENVERS
sens    LONG    29W/82  L  26.1%     +57.65$   |   SHORT   20W/50  L  28.6%    -120.35$
actions B 4.4%  S 4.5%  H 91.0%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 5.2e-06
train   PnL   +836.23$  3615 trades  WR 35.0%  PF 1.21   ->  ecart train-val +7.9 pt
. ACTOR GELE : warmup du critic, gradient 5.2e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 72s  PPO 141s  calib 24s  validation 121s

## 18:01:08 — wf1 — EPOCH 002   +0.45$/trade   187 trades   WR 37.4%   PF 1.21   6 min

PnL          +83.56$   cumul run      +20.86$   DD 4.7%   Sortino +0.128
point mort 33.1%  ->  ecart +4.3 pt (+/- 3.5 au mieux)
classement rho [90m-0.0096[0m (+/- 0.024 env.)  -> n'ordonne rien
vs epoch precedente : +0.79$/trade
sens    LONG    44W/63  L  41.1%    +137.46$   |   SHORT   26W/54  L  32.5%     -53.90$
actions B 4.8%  S 4.7%  H 90.5%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 5.0e-06
train   PnL    -28.48$  3277 trades  WR 33.0%  PF 0.99   ->  ecart train-val -4.4 pt
. ACTOR GELE : warmup du critic, gradient 5.0e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 72s  PPO 130s  calib 28s  validation 117s

## 18:08:48 — wf1 — EPOCH 003   +0.17$/trade   168 trades   WR 33.9%   PF 1.06   7 min

PnL          +28.70$   cumul run      +49.56$   DD 9.1%   Sortino +0.040
point mort 32.6%  ->  ecart +1.3 pt (+/- 3.7 au mieux)
classement rho [91m-0.0323[0m (+/- 0.024 env.)  -> classe A L'ENVERS
vs politique gelee du meme run : +0.8 pt   (reference +0.5 pt sur 2 epochs)
vs epoch precedente : -0.28$/trade
sens    LONG    32W/63  L  33.7%    +117.17$   |   SHORT   25W/48  L  34.2%     -88.48$
actions B 7.2%  S 6.8%  H 86.0%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 2.5e-05
train   PnL   +247.67$  4028 trades  WR 32.5%  PF 1.08   ->  ecart train-val -1.4 pt
. ACTOR GELE : warmup du critic, gradient 2.5e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.17$) sous l'erreur-type de cette epoch (0.48$) — non separable de zero.
temps : collecte 79s  PPO 187s  calib 32s  validation 149s

## 18:17:08 — wf1 — EPOCH 004   +0.48$/trade   155 trades   WR 34.8%   PF 1.20   8 min

PnL          +74.70$   cumul run     +124.26$   DD 5.6%   Sortino +0.130
point mort 30.8%  ->  ecart +4.0 pt (+/- 3.8 au mieux)
classement rho [92m+0.0207[0m (+/- 0.024 env.)  -> classe
vs politique gelee du meme run : +3.2 pt   (reference +0.8 pt sur 3 epochs)
vs epoch precedente : +0.31$/trade
sens    LONG    40W/83  L  32.5%     +48.58$   |   SHORT   14W/18  L  43.8%     +26.12$
actions B 10.2%  S 9.7%  H 80.1%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 2.4e-05
train   PnL    +95.94$  4585 trades  WR 31.9%  PF 1.04   ->  ecart train-val -2.9 pt
. ACTOR GELE : warmup du critic, gradient 2.4e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 92s  PPO 216s  calib 34s  validation 157s

## 18:21:28 — wf1 — EPOCH 005   +0.26$/trade   166 trades   WR 31.3%   PF 1.11   4 min

PnL          +43.90$   cumul run     +168.16$   DD 5.4%   Sortino +0.071
point mort 29.1%  ->  ecart +2.2 pt (+/- 3.6 au mieux)
classement rho [90m-0.0007[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : +0.6 pt   (reference +1.6 pt sur 4 epochs)
vs epoch precedente : -0.22$/trade
sens    LONG    37W/78  L  32.2%    +106.79$   |   SHORT   15W/36  L  29.4%     -62.89$
actions B 5.4%  S 5.6%  H 89.0%   |   selectivite  train 32.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.8e-08
train   PnL   -245.01$  1875 trades  WR 31.9%  PF 0.88   ->  ecart train-val +0.6 pt
. ACTOR GELE : warmup du critic, gradient 1.8e-08. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.26$) sous l'erreur-type de cette epoch (0.43$) — non separable de zero.
temps : collecte 44s  PPO 57s  calib 26s  validation 142s

## 18:28:32 — wf1 — EPOCH 001   -0.23$/trade   177 trades   WR 34.5%   PF 0.91   3 min

PnL          -39.86$   cumul run      -39.86$   DD 6.6%   Sortino -0.059
point mort 36.6%  ->  ecart -2.1 pt (+/- 3.6 au mieux)
classement rho [92m+0.0309[0m (+/- 0.024 env.)  -> classe
sens    LONG    22W/38  L  36.7%     +25.64$   |   SHORT   39W/78  L  33.3%     -65.50$
actions B 1.1%  S 1.0%  H 97.9%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0003 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 7.8e-06
train   PnL   +206.78$  859 trades  WR 35.4%  PF 1.16   ->  ecart train-val +0.9 pt
. ACTOR GELE : warmup du critic, gradient 7.8e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0003 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.23$) sous l'erreur-type de cette epoch (0.37$) — non separable de zero.
temps : collecte 34s  PPO 20s  calib 23s  validation 108s

## 18:31:52 — wf1 — EPOCH 002   -1.09$/trade   175 trades   WR 28.6%   PF 0.63   3 min

PnL         -191.50$   cumul run     -231.36$   DD 9.3%   Sortino -0.263
point mort 38.8%  ->  ecart -10.2 pt (+/- 3.4 au mieux)
classement rho [90m+0.0109[0m (+/- 0.024 env.)  -> n'ordonne rien
vs epoch precedente : -0.87$/trade
sens    LONG    21W/50  L  29.6%     -42.54$   |   SHORT   29W/75  L  27.9%    -148.95$
actions B 0.6%  S 0.8%  H 98.6%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.7e-06
train   PnL   +130.63$  1149 trades  WR 33.0%  PF 1.04   ->  ecart train-val +4.4 pt
. ACTOR GELE : warmup du critic, gradient 1.7e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 53s  PPO 26s  calib 20s  validation 101s

## 18:37:32 — wf1 — EPOCH 003   +0.29$/trade   167 trades   WR 35.3%   PF 1.10   5 min

PnL          +49.21$   cumul run     -182.15$   DD 6.0%   Sortino +0.063
point mort 33.2%  ->  ecart +2.1 pt (+/- 3.7 au mieux)
classement rho [92m+0.0231[0m (+/- 0.024 env.)  -> classe
vs politique gelee du meme run : +8.3 pt   (reference -6.2 pt sur 2 epochs)
vs epoch precedente : +1.39$/trade
sens    LONG    28W/52  L  35.0%    +111.00$   |   SHORT   31W/56  L  35.6%     -61.80$
actions B 1.3%  S 1.3%  H 97.4%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 4.1e-06
train   PnL   +527.71$  3479 trades  WR 35.6%  PF 1.12   ->  ecart train-val +0.3 pt
. ACTOR GELE : warmup du critic, gradient 4.1e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.29$) sous l'erreur-type de cette epoch (0.51$) — non separable de zero.
temps : collecte 99s  PPO 101s  calib 24s  validation 104s

## 18:42:52 — wf1 — EPOCH 004   -0.65$/trade   187 trades   WR 29.4%   PF 0.76   5 min

PnL         -121.12$   cumul run     -303.27$   DD 7.2%   Sortino -0.161
point mort 35.4%  ->  ecart -6.0 pt (+/- 3.3 au mieux)
classement rho [90m+0.0171[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : -2.6 pt   (reference -3.4 pt sur 3 epochs)
vs epoch precedente : -0.94$/trade
sens    LONG    28W/69  L  28.9%     +38.47$   |   SHORT   27W/63  L  30.0%    -159.59$
actions B 1.5%  S 1.4%  H 97.1%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0003 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 5.7e-06
train   PnL  +1078.01$  3318 trades  WR 33.8%  PF 1.34   ->  ecart train-val +4.4 pt
. ACTOR GELE : warmup du critic, gradient 5.7e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0003 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 91s  PPO 100s  calib 21s  validation 107s

## 18:47:12 — wf1 — EPOCH 005   +2.59$/trade   111 trades   WR 40.5%   PF 2.10   4 min

PnL         +287.91$   cumul run      -15.36$   DD 2.9%   Sortino +0.727
point mort 24.5%  ->  ecart +16.0 pt (+/- 4.7 au mieux)
classement rho [90m+0.0068[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : +20.1 pt   (reference -4.1 pt sur 4 epochs)
vs epoch precedente : +3.24$/trade
sens    LONG    37W/55  L  40.2%    +287.01$   |   SHORT    8W/11  L  42.1%      +0.91$
actions B 1.2%  S 1.3%  H 97.5%   |   selectivite  train 32.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0000 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 3.5e-06
train   PnL   -355.50$  2588 trades  WR 33.9%  PF 0.90   ->  ecart train-val -6.6 pt
. ACTOR GELE : warmup du critic, gradient 3.5e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0000 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 81s  PPO 70s  calib 23s  validation 93s

## 18:52:33 — wf1 — EPOCH 006   -0.03$/trade   210 trades   WR 34.3%   PF 0.99   5 min

PnL           -5.37$   cumul run      -20.73$   DD 4.6%   Sortino -0.007
point mort 34.5%  ->  ecart -0.2 pt (+/- 3.3 au mieux)
classement rho [90m-0.0010[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : -0.2 pt   (reference -0.1 pt sur 5 epochs)
vs epoch precedente : -2.62$/trade
sens    LONG    38W/72  L  34.5%     +81.94$   |   SHORT   34W/66  L  34.0%     -87.30$
actions B 1.5%  S 1.5%  H 97.0%   |   selectivite  train 27.5%  val 5.0%
politique  H 1.083/1.099 (-0.016)   etendue 0.2115 (212x le tremblement)   KL +0.0152   clipfrac 39.3%   g_actor 3.0e-01
train   PnL  +1171.95$  3141 trades  WR 33.3%  PF 1.34   ->  ecart train-val -1.0 pt
. etendue val 0.2115, soit 212x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.03$) sous l'erreur-type de cette epoch (0.37$) — non separable de zero.
. clipfrac 39.3% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 82s  PPO 93s  calib 21s  validation 108s

## 18:56:33 — wf1 — EPOCH 007   -0.21$/trade   214 trades   WR 31.3%   PF 0.90   4 min

PnL          -46.01$   cumul run      -66.74$   DD 5.2%   Sortino -0.060
point mort 33.6%  ->  ecart -2.3 pt (+/- 3.2 au mieux)
classement rho [90m+0.0074[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : -2.3 pt   (reference -0.1 pt sur 5 epochs)
vs epoch precedente : -0.19$/trade
sens    LONG    27W/65  L  29.3%      +9.78$   |   SHORT   40W/82  L  32.8%     -55.79$
actions B 1.1%  S 1.2%  H 97.7%   |   selectivite  train 23.0%  val 5.0%
politique  H 1.064/1.099 (-0.019)   etendue 0.3492 (349x le tremblement)   KL +0.0255   clipfrac 45.2%   g_actor 2.8e-01
train   PnL   -274.00$  2099 trades  WR 35.7%  PF 0.91   ->  ecart train-val +4.4 pt
. etendue val 0.3492, soit 349x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.21$) sous l'erreur-type de cette epoch (0.30$) — non separable de zero.
. clipfrac 45.2% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 68s  PPO 56s  calib 24s  validation 106s

## 19:00:53 — wf1 — EPOCH 008   -0.28$/trade   221 trades   WR 31.2%   PF 0.88   4 min

PnL          -60.86$   cumul run     -127.60$   DD 7.3%   Sortino -0.073
point mort 34.0%  ->  ecart -2.8 pt (+/- 3.1 au mieux)
classement rho [90m-0.0192[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : -2.8 pt   (reference -0.1 pt sur 5 epochs)
vs epoch precedente : -0.06$/trade
sens    LONG    36W/71  L  33.6%     +92.27$   |   SHORT   33W/81  L  28.9%    -153.13$
actions B 1.2%  S 1.0%  H 97.8%   |   selectivite  train 18.5%  val 5.0%
politique  H 1.039/1.099 (-0.025)   etendue 0.3717 (372x le tremblement)   KL +0.0321   clipfrac 38.8%   g_actor 2.8e-01
train   PnL   -285.77$  2265 trades  WR 34.7%  PF 0.91   ->  ecart train-val +3.5 pt
. etendue val 0.3717, soit 372x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.28$) sous l'erreur-type de cette epoch (0.31$) — non separable de zero.
. clipfrac 38.8% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 76s  PPO 54s  calib 20s  validation 105s

## 19:05:53 — wf1 — EPOCH 009   +0.26$/trade   186 trades   WR 34.9%   PF 1.12   5 min

PnL          +48.29$   cumul run      -79.31$   DD 4.6%   Sortino +0.074
point mort 32.4%  ->  ecart +2.5 pt (+/- 3.5 au mieux)
classement rho [91m-0.0271[0m (+/- 0.024 env.)  -> classe A L'ENVERS
vs politique gelee du meme run : +2.5 pt   (reference -0.1 pt sur 5 epochs)
vs epoch precedente : +0.54$/trade
sens    LONG    18W/33  L  35.3%    +100.88$   |   SHORT   47W/88  L  34.8%     -52.59$
actions B 1.3%  S 1.3%  H 97.4%   |   selectivite  train 14.0%  val 5.0%
politique  H 1.025/1.099 (-0.014)   etendue 0.4285 (428x le tremblement)   KL +0.0332   clipfrac 44.1%   g_actor 2.8e-01
train   PnL   +703.52$  2992 trades  WR 33.8%  PF 1.18   ->  ecart train-val -1.1 pt
. etendue val 0.4285, soit 428x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.26$) sous l'erreur-type de cette epoch (0.36$) — non separable de zero.
. clipfrac 44.1% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 88s  PPO 73s  calib 20s  validation 108s

## 19:10:33 — wf1 — EPOCH 010   +0.66$/trade   165 trades   WR 34.5%   PF 1.30   5 min

PnL         +108.70$   cumul run      +29.39$   DD 4.3%   Sortino +0.207
point mort 28.9%  ->  ecart +5.6 pt (+/- 3.7 au mieux)
classement rho [90m-0.0070[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : +5.7 pt   (reference -0.1 pt sur 5 epochs)
vs epoch precedente : +0.40$/trade
sens    LONG    23W/49  L  31.9%    +107.87$   |   SHORT   34W/59  L  36.6%      +0.83$
actions B 1.3%  S 1.1%  H 97.6%   |   selectivite  train 9.5%  val 5.0%
politique  H 0.991/1.099 (-0.034)   etendue 0.4544 (454x le tremblement)   KL +0.0302   clipfrac 41.5%   g_actor 2.9e-01
train   PnL   +442.86$  2732 trades  WR 33.3%  PF 1.13   ->  ecart train-val -1.2 pt
. etendue val 0.4544, soit 454x le tremblement du seuil : la selection est portee par le modele.
. clipfrac 41.5% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 91s  PPO 63s  calib 20s  validation 108s

## 19:15:13 — wf1 — EPOCH 011   +0.63$/trade   170 trades   WR 36.5%   PF 1.26   5 min

PnL         +106.94$   cumul run     +136.33$   DD 7.7%   Sortino +0.175
point mort 31.3%  ->  ecart +5.2 pt (+/- 3.7 au mieux)
classement rho [90m-0.0141[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : +5.3 pt   (reference -0.1 pt sur 5 epochs)
vs epoch precedente : -0.03$/trade
sens    LONG    25W/35  L  41.7%    +141.59$   |   SHORT   37W/73  L  33.6%     -34.65$
actions B 1.4%  S 1.1%  H 97.5%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.959/1.099 (-0.032)   etendue 0.4774 (477x le tremblement)   KL +0.0286   clipfrac 39.0%   g_actor 3.2e-01
train   PnL    +42.50$  3067 trades  WR 35.3%  PF 1.01   ->  ecart train-val -1.2 pt
. etendue val 0.4774, soit 477x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. clipfrac 39.0% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 98s  PPO 70s  calib 21s  validation 93s

## 19:19:53 — wf1 — EPOCH 012   +0.86$/trade   169 trades   WR 37.9%   PF 1.41   5 min

PnL         +145.68$   cumul run     +282.01$   DD 4.7%   Sortino +0.261
point mort 30.2%  ->  ecart +7.7 pt (+/- 3.7 au mieux)
classement rho [90m+0.0182[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : +7.8 pt   (reference -0.1 pt sur 5 epochs)
vs epoch precedente : +0.23$/trade
sens    LONG    25W/40  L  38.5%    +156.02$   |   SHORT   39W/65  L  37.5%     -10.35$
actions B 1.4%  S 1.2%  H 97.5%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.944/1.099 (-0.015)   etendue 0.4793 (479x le tremblement)   KL +0.0331   clipfrac 38.0%   g_actor 3.2e-01
train   PnL   +537.67$  3067 trades  WR 34.1%  PF 1.13   ->  ecart train-val -3.8 pt
. etendue val 0.4793, soit 479x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. clipfrac 38.0% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 96s  PPO 67s  calib 21s  validation 98s

## 19:24:14 — wf1 — EPOCH 013   +0.23$/trade   196 trades   WR 32.1%   PF 1.11   4 min

PnL          +44.11$   cumul run     +326.12$   DD 5.6%   Sortino +0.071
point mort 29.9%  ->  ecart +2.2 pt (+/- 3.3 au mieux)
classement rho [90m-0.0042[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : +2.2 pt   (reference -0.1 pt sur 5 epochs)
vs epoch precedente : -0.64$/trade
sens    LONG    36W/73  L  33.0%    +125.62$   |   SHORT   27W/60  L  31.0%     -81.51$
actions B 1.3%  S 0.9%  H 97.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.903/1.099 (-0.041)   etendue 0.4816 (482x le tremblement)   KL +0.0289   clipfrac 33.3%   g_actor 3.3e-01
train   PnL   +592.39$  2601 trades  WR 37.9%  PF 1.14   ->  ecart train-val +5.8 pt
. etendue val 0.4816, soit 482x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.23$) sous l'erreur-type de cette epoch (0.34$) — non separable de zero.
. clipfrac 33.3% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 87s  PPO 50s  calib 20s  validation 96s

## 19:28:34 — wf1 — EPOCH 014   +1.39$/trade   175 trades   WR 37.1%   PF 1.79   4 min

PnL         +242.73$   cumul run     +568.85$   DD 3.9%   Sortino +0.497
point mort 24.8%  ->  ecart +12.3 pt (+/- 3.7 au mieux)
classement rho [92m+0.0211[0m (+/- 0.024 env.)  -> classe
vs politique gelee du meme run : +12.3 pt   (reference -0.1 pt sur 5 epochs)
vs epoch precedente : +1.16$/trade
sens    LONG    38W/52  L  42.2%    +247.75$   |   SHORT   27W/58  L  31.8%      -5.02$
actions B 1.0%  S 0.7%  H 98.2%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.883/1.099 (-0.020)   etendue 0.4938 (494x le tremblement)   KL +0.0240   clipfrac 30.3%   g_actor 3.1e-01
train   PnL  +1086.54$  2469 trades  WR 40.8%  PF 1.25   ->  ecart train-val +3.7 pt
. etendue val 0.4938, soit 494x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. clipfrac 30.3% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 91s  PPO 45s  calib 20s  validation 92s

## 19:33:54 — wf1 — EPOCH 015   +1.40$/trade   173 trades   WR 39.9%   PF 1.75   5 min

PnL         +241.39$   cumul run     +810.24$   DD 3.8%   Sortino +0.471
point mort 27.5%  ->  ecart +12.4 pt (+/- 3.7 au mieux)
classement rho [92m+0.0215[0m (+/- 0.024 env.)  -> classe
vs politique gelee du meme run : +12.5 pt   (reference -0.1 pt sur 5 epochs)
vs epoch precedente : +0.01$/trade
sens    LONG    32W/38  L  45.7%    +230.19$   |   SHORT   37W/66  L  35.9%     +11.20$
actions B 1.5%  S 1.0%  H 97.5%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.853/1.099 (-0.030)   etendue 0.4935 (494x le tremblement)   KL +0.0271   clipfrac 32.4%   g_actor 2.9e-01
train   PnL  +1357.88$  4045 trades  WR 39.1%  PF 1.28   ->  ecart train-val -0.8 pt
. etendue val 0.4935, soit 494x le tremblement du seuil : la selection est portee par le modele.
. clipfrac 32.4% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 118s  PPO 82s  calib 23s  validation 104s

## 19:38:54 — wf1 — EPOCH 016   +0.58$/trade   182 trades   WR 30.8%   PF 1.25   5 min

PnL         +104.99$   cumul run     +915.23$   DD 4.7%   Sortino +0.169
point mort 26.2%  ->  ecart +4.6 pt (+/- 3.4 au mieux)
classement rho [90m-0.0086[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : +4.6 pt   (reference -0.1 pt sur 5 epochs)
vs epoch precedente : -0.82$/trade
sens    LONG    38W/80  L  32.2%    +150.03$   |   SHORT   18W/46  L  28.1%     -45.04$
actions B 1.5%  S 1.0%  H 97.5%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.834/1.099 (-0.019)   etendue 0.5106 (511x le tremblement)   KL +0.0216   clipfrac 32.1%   g_actor 2.9e-01
train   PnL   +804.73$  3798 trades  WR 39.6%  PF 1.15   ->  ecart train-val +8.8 pt
. etendue val 0.5106, soit 511x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. clipfrac 32.1% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 112s  PPO 74s  calib 21s  validation 95s

## 19:44:14 — wf1 — EPOCH 017   +0.51$/trade   176 trades   WR 32.4%   PF 1.22   5 min

PnL          +88.91$   cumul run    +1004.14$   DD 5.5%   Sortino +0.154
point mort 28.2%  ->  ecart +4.2 pt (+/- 3.5 au mieux)
classement rho [90m-0.0017[0m (+/- 0.024 env.)  -> n'ordonne rien
vs politique gelee du meme run : +4.3 pt   (reference -0.1 pt sur 5 epochs)
vs epoch precedente : -0.07$/trade
sens    LONG    40W/69  L  36.7%    +135.63$   |   SHORT   17W/50  L  25.4%     -46.72$
actions B 1.5%  S 1.2%  H 97.3%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.818/1.099 (-0.016)   etendue 0.5027 (503x le tremblement)   KL +0.0343   clipfrac 32.9%   g_actor 2.8e-01
train   PnL  +1776.37$  3818 trades  WR 39.9%  PF 1.38   ->  ecart train-val +7.5 pt
. etendue val 0.5027, soit 503x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. clipfrac 32.9% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 112s  PPO 72s  calib 20s  validation 101s

## 19:51:34 — wf1 — EPOCH 001   -0.23$/trade   177 trades   WR 34.5%   PF 0.91   3 min

PnL          -39.86$   cumul run      -39.86$   DD 6.6%   Sortino -0.059
point mort 36.6%  ->  ecart -2.1 pt (+/- 3.6 au mieux)
classement rho [92m+0.0309[0m (+/- 0.024 env.)  -> classe   tete auxiliaire [91m-0.0361[0m
sens    LONG    22W/38  L  36.7%     +25.64$   |   SHORT   39W/78  L  33.3%     -65.50$
actions B 1.1%  S 1.0%  H 97.9%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.0003 (0x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 7.8e-06
train   PnL   +206.78$  859 trades  WR 35.4%  PF 1.16   ->  ecart train-val +0.9 pt
. ACTOR GELE : warmup du critic, gradient 7.8e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0003 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.23$) sous l'erreur-type de cette epoch (0.37$) — non separable de zero.
temps : collecte 32s  PPO 19s  calib 21s  validation 90s

## 19:54:54 — wf1 — EPOCH 002   -1.09$/trade   175 trades   WR 28.6%   PF 0.63   3 min

PnL         -191.50$   cumul run     -231.36$   DD 9.3%   Sortino -0.263
point mort 38.8%  ->  ecart -10.2 pt (+/- 3.4 au mieux)
classement rho [90m+0.0109[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [91m-0.0263[0m
vs epoch precedente : -0.87$/trade
sens    LONG    21W/50  L  29.6%     -42.54$   |   SHORT   29W/75  L  27.9%    -148.95$
actions B 0.6%  S 0.8%  H 98.6%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.7e-06
train   PnL   +130.63$  1149 trades  WR 33.0%  PF 1.04   ->  ecart train-val +4.4 pt
. ACTOR GELE : warmup du critic, gradient 1.7e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
temps : collecte 53s  PPO 25s  calib 19s  validation 106s

## 20:01:14 — wf1 — EPOCH 003   +0.29$/trade   167 trades   WR 35.3%   PF 1.10   6 min

PnL          +49.21$   cumul run     -182.15$   DD 6.0%   Sortino +0.063
point mort 33.2%  ->  ecart +2.1 pt (+/- 3.7 au mieux)
classement rho [92m+0.0231[0m (+/- 0.024 env.)  -> classe   tete auxiliaire [90m+0.0097[0m
vs politique gelee du meme run : +8.3 pt   (reference -6.2 pt sur 2 epochs)
vs epoch precedente : +1.39$/trade
sens    LONG    28W/52  L  35.0%    +111.00$   |   SHORT   31W/56  L  35.6%     -61.80$
actions B 1.3%  S 1.3%  H 97.4%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.0001 (0x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 4.1e-06
train   PnL   +527.71$  3479 trades  WR 35.6%  PF 1.12   ->  ecart train-val +0.3 pt
. ACTOR GELE : warmup du critic, gradient 4.1e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. SELECTION TIREE AU SORT : etendue val 0.0001 contre un tremblement de seuil de ~0.0010. Les 5 % retenus changent presque entierement d'une epoch a l'autre — ce chiffre mesure le tirage.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.29$) sous l'erreur-type de cette epoch (0.51$) — non separable de zero.
temps : collecte 102s  PPO 106s  calib 33s  validation 122s

## 20:10:46 — wf1 — EPOCH 001   +0.67$/trade   215 trades   WR 30.2%   PF 1.35   5 min

PnL         +143.13$   cumul run     +143.13$   DD 6.0%   Sortino +0.239
point mort 24.3%  ->  ecart +5.9 pt (+/- 3.1 au mieux)
classement rho [92m+0.0309[0m (+/- 0.024 env.)  -> classe   tete auxiliaire [91m-0.0361[0m
sens    LONG    31W/64  L  32.6%    +177.00$   |   SHORT   34W/86  L  28.3%     -33.88$
actions B 1.1%  S 1.0%  H 97.9%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.2577 (258x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 7.8e-06
train   PnL   +206.78$  859 trades  WR 35.4%  PF 1.16   ->  ecart train-val +5.2 pt
. ACTOR GELE : warmup du critic, gradient 7.8e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.2577, soit 258x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 39s  PPO 23s  calib 33s  validation 176s

## 20:15:27 — wf1 — EPOCH 002   +0.14$/trade   211 trades   WR 28.0%   PF 1.07   5 min

PnL          +29.89$   cumul run     +173.02$   DD 4.7%   Sortino +0.046
point mort 26.6%  ->  ecart +1.4 pt (+/- 3.1 au mieux)
classement rho [90m+0.0109[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [91m-0.0263[0m
vs epoch precedente : -0.52$/trade
sens    LONG    48W/110 L  30.4%     +73.79$   |   SHORT   11W/42  L  20.8%     -43.90$
actions B 0.6%  S 0.8%  H 98.6%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.3224 (322x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.7e-06
train   PnL   +130.63$  1149 trades  WR 33.0%  PF 1.04   ->  ecart train-val +5.0 pt
. ACTOR GELE : warmup du critic, gradient 1.7e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.3224, soit 322x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.14$) sous l'erreur-type de cette epoch (0.33$) — non separable de zero.
temps : collecte 54s  PPO 29s  calib 32s  validation 163s

## 20:22:47 — wf1 — EPOCH 003   +0.07$/trade   186 trades   WR 30.1%   PF 1.03   7 min

PnL          +12.66$   cumul run     +185.68$   DD 6.6%   Sortino +0.020
point mort 29.5%  ->  ecart +0.6 pt (+/- 3.4 au mieux)
classement rho [92m+0.0231[0m (+/- 0.024 env.)  -> classe   tete auxiliaire [90m+0.0097[0m
vs politique gelee du meme run : -3.0 pt   (reference +3.6 pt sur 2 epochs)
vs epoch precedente : -0.07$/trade
sens    LONG    35W/80  L  30.4%     +75.63$   |   SHORT   21W/50  L  29.6%     -62.97$
actions B 1.3%  S 1.3%  H 97.4%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.6203 (620x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 4.1e-06
train   PnL   +527.71$  3479 trades  WR 35.6%  PF 1.12   ->  ecart train-val +5.5 pt
. ACTOR GELE : warmup du critic, gradient 4.1e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.6203, soit 620x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.07$) sous l'erreur-type de cette epoch (0.37$) — non separable de zero.
temps : collecte 105s  PPO 112s  calib 38s  validation 182s

## 20:33:07 — wf1 — EPOCH 004   -0.31$/trade   201 trades   WR 26.9%   PF 0.87   10 min

PnL          -61.42$   cumul run     +124.26$   DD 5.8%   Sortino -0.083
point mort 29.7%  ->  ecart -2.8 pt (+/- 3.1 au mieux)
classement rho [90m+0.0171[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m-0.0115[0m
vs politique gelee du meme run : -5.4 pt   (reference +2.6 pt sur 3 epochs)
vs epoch precedente : -0.37$/trade
sens    LONG    32W/74  L  30.2%     +44.69$   |   SHORT   22W/73  L  23.2%    -106.11$
actions B 1.5%  S 1.4%  H 97.1%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.6812 (681x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 5.7e-06
train   PnL  +1078.01$  3318 trades  WR 33.8%  PF 1.34   ->  ecart train-val +6.9 pt
. ACTOR GELE : warmup du critic, gradient 5.7e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.6812, soit 681x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.31$) sous l'erreur-type de cette epoch (0.34$) — non separable de zero.
temps : collecte 94s  PPO 110s  calib 39s  validation 378s

## 20:42:38 — wf1 — EPOCH 001   +0.67$/trade   215 trades   WR 30.2%   PF 1.35   5 min

PnL         +143.13$   cumul run     +143.13$   DD 6.0%   Sortino +0.239
point mort 24.3%  ->  ecart +5.9 pt (+/- 3.1 au mieux)
classement rho [92m+0.0309[0m (+/- 0.024 env.)  -> classe   tete auxiliaire [91m-0.0361[0m
sens    LONG    31W/64  L  32.6%    +177.00$   |   SHORT   34W/86  L  28.3%     -33.88$
actions B 1.1%  S 1.0%  H 97.9%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.2577 (258x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 7.8e-06
train   PnL   +206.78$  859 trades  WR 35.4%  PF 1.16   ->  ecart train-val +5.2 pt
. ACTOR GELE : warmup du critic, gradient 7.8e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.2577, soit 258x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 36s  PPO 22s  calib 34s  validation 184s

## 20:49:58 — wf1 — EPOCH 001   +0.67$/trade   215 trades   WR 30.2%   PF 1.35   5 min

PnL         +143.13$   cumul run     +143.13$   DD 6.0%   Sortino +0.239
point mort 24.3%  ->  ecart +5.9 pt (+/- 3.1 au mieux)
classement rho [92m+0.0309[0m (+/- 0.024 env.)  -> classe   tete auxiliaire [91m-0.0361[0m
sens    LONG    31W/64  L  32.6%    +177.00$   |   SHORT   34W/86  L  28.3%     -33.88$
actions B 1.1%  S 1.0%  H 97.9%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.2577 (258x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 7.8e-06
train   PnL   +206.78$  859 trades  WR 35.4%  PF 1.16   ->  ecart train-val +5.2 pt
. ACTOR GELE : warmup du critic, gradient 7.8e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.2577, soit 258x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 34s  PPO 19s  calib 36s  validation 186s

## 20:56:11 — wf1 — EPOCH 001   +0.67$/trade   215 trades   WR 30.2%   PF 1.35   5 min

PnL         +143.13$   cumul run     +143.13$   DD 6.0%   Sortino +0.239
point mort 24.3%  ->  ecart +5.9 pt (+/- 3.1 au mieux)
classement rho [92m+0.0309[0m (+/- 0.024 env.)  -> classe   tete auxiliaire [91m-0.0361[0m
sens    LONG    31W/64  L  32.6%    +177.00$   |   SHORT   34W/86  L  28.3%     -33.88$
actions B 1.1%  S 1.0%  H 97.9%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.2577 (258x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 7.8e-06
train   PnL   +206.78$  859 trades  WR 35.4%  PF 1.16   ->  ecart train-val +5.2 pt
. ACTOR GELE : warmup du critic, gradient 7.8e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.2577, soit 258x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 34s  PPO 19s  calib 36s  validation 186s

## 20:56:11 — wf1 — EPOCH 002   +0.67$/trade   215 trades   WR 30.2%   PF 1.35   1 min

PnL         +143.13$   cumul run     +286.26$   DD 6.0%   Sortino +0.239
point mort 24.3%  ->  ecart +5.9 pt (+/- 3.1 au mieux)
classement rho [90m+0.0109[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [91m-0.0263[0m
vs epoch precedente : +0.00$/trade
sens    LONG    31W/64  L  32.6%    +177.00$   |   SHORT   34W/86  L  28.3%     -33.88$
actions B 0.6%  S 0.8%  H 98.6%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.2577 (258x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.7e-06
train   PnL   +130.63$  1149 trades  WR 33.0%  PF 1.04   ->  ecart train-val +2.8 pt
. ACTOR GELE : warmup du critic, gradient 1.7e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.2577, soit 258x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 58s  PPO 28s  calib 0s  validation 0s

## 20:59:31 — wf1 — EPOCH 003   +0.07$/trade   186 trades   WR 30.1%   PF 1.03   8 min

PnL          +12.66$   cumul run     +298.92$   DD 6.6%   Sortino +0.020
point mort 29.5%  ->  ecart +0.6 pt (+/- 3.4 au mieux)
classement rho [92m+0.0231[0m (+/- 0.024 env.)  -> classe   tete auxiliaire [90m+0.0097[0m
vs politique gelee du meme run : -5.3 pt   (reference +5.9 pt sur 2 epochs)
vs epoch precedente : -0.60$/trade
sens    LONG    35W/80  L  30.4%     +75.63$   |   SHORT   21W/50  L  29.6%     -62.97$
actions B 1.3%  S 1.3%  H 97.4%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.6203 (620x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 4.1e-06
train   PnL   +527.71$  3479 trades  WR 35.6%  PF 1.12   ->  ecart train-val +5.5 pt
. ACTOR GELE : warmup du critic, gradient 4.1e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.6203, soit 620x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.07$) sous l'erreur-type de cette epoch (0.37$) — non separable de zero.
temps : collecte 115s  PPO 127s  calib 47s  validation 177s

## 21:03:11 — wf1 — EPOCH 004   +0.07$/trade   186 trades   WR 30.1%   PF 1.03   4 min

PnL          +12.66$   cumul run     +311.58$   DD 6.6%   Sortino +0.020
point mort 29.5%  ->  ecart +0.6 pt (+/- 3.4 au mieux)
classement rho [90m+0.0171[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m-0.0115[0m
vs politique gelee du meme run : -3.5 pt   (reference +4.1 pt sur 3 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    35W/80  L  30.4%     +75.63$   |   SHORT   21W/50  L  29.6%     -62.97$
actions B 1.5%  S 1.4%  H 97.1%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.6203 (620x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 5.7e-06
train   PnL  +1078.01$  3318 trades  WR 33.8%  PF 1.34   ->  ecart train-val +3.7 pt
. ACTOR GELE : warmup du critic, gradient 5.7e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.6203, soit 620x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.07$) sous l'erreur-type de cette epoch (0.37$) — non separable de zero.
. GPU BRIDE : 210 MHz a 88 C. Les durees ne sont pas comparables entre epochs si la temperature derive.
temps : collecte 100s  PPO 115s  calib 0s  validation 0s

## 21:06:31 — wf1 — EPOCH 005   +0.07$/trade   186 trades   WR 30.1%   PF 1.03   3 min

PnL          +12.66$   cumul run     +324.24$   DD 6.6%   Sortino +0.020
point mort 29.5%  ->  ecart +0.6 pt (+/- 3.4 au mieux)
classement rho [90m+0.0068[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m-0.0193[0m
vs politique gelee du meme run : -2.6 pt   (reference +3.3 pt sur 4 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    35W/80  L  30.4%     +75.63$   |   SHORT   21W/50  L  29.6%     -62.97$
actions B 1.2%  S 1.3%  H 97.5%   |   selectivite  train 32.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.6203 (620x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 3.5e-06
train   PnL   -355.50$  2588 trades  WR 33.9%  PF 0.90   ->  ecart train-val +3.8 pt
. ACTOR GELE : warmup du critic, gradient 3.5e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.6203, soit 620x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.07$) sous l'erreur-type de cette epoch (0.37$) — non separable de zero.
. GPU BRIDE : 240 MHz a 88 C. Les durees ne sont pas comparables entre epochs si la temperature derive.
temps : collecte 87s  PPO 91s  calib 0s  validation 0s

## 21:13:12 — wf1 — EPOCH 006   -0.13$/trade   206 trades   WR 32.5%   PF 0.95   7 min

PnL          -26.11$   cumul run     +298.13$   DD 10.0%   Sortino -0.033
point mort 33.7%  ->  ecart -1.2 pt (+/- 3.3 au mieux)
classement rho [90m-0.0010[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m+0.0097[0m
vs politique gelee du meme run : -3.9 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : -0.19$/trade
sens    LONG    27W/55  L  32.9%     +81.66$   |   SHORT   40W/84  L  32.3%    -107.77$
actions B 1.5%  S 1.5%  H 97.0%   |   selectivite  train 27.5%  val 5.0%
politique  H 1.083/1.099 (-0.016)   etendue 0.7046 (705x le tremblement)   KL +0.0152   clipfrac 39.3%   g_actor 3.0e-01
train   PnL  +1171.95$  3141 trades  WR 33.3%  PF 1.34   ->  ecart train-val +0.8 pt
. etendue val 0.7046, soit 705x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.13$) sous l'erreur-type de cette epoch (0.36$) — non separable de zero.
. clipfrac 39.3% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 92s  PPO 96s  calib 35s  validation 168s

## 21:15:12 — wf1 — EPOCH 007   -0.13$/trade   206 trades   WR 32.5%   PF 0.95   2 min

PnL          -26.11$   cumul run     +272.02$   DD 10.0%   Sortino -0.033
point mort 33.7%  ->  ecart -1.2 pt (+/- 3.3 au mieux)
classement rho [90m+0.0074[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m+0.0051[0m
vs politique gelee du meme run : -3.9 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    27W/55  L  32.9%     +81.66$   |   SHORT   40W/84  L  32.3%    -107.77$
actions B 1.1%  S 1.2%  H 97.7%   |   selectivite  train 23.0%  val 5.0%
politique  H 1.064/1.099 (-0.019)   etendue 0.7046 (705x le tremblement)   KL +0.0255   clipfrac 45.2%   g_actor 2.8e-01
train   PnL   -274.00$  2099 trades  WR 35.7%  PF 0.91   ->  ecart train-val +3.2 pt
. etendue val 0.7046, soit 705x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.13$) sous l'erreur-type de cette epoch (0.36$) — non separable de zero.
. clipfrac 45.2% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 65s  PPO 51s  calib 0s  validation 0s

## 21:17:32 — wf1 — EPOCH 008   -0.13$/trade   206 trades   WR 32.5%   PF 0.95   2 min

PnL          -26.11$   cumul run     +245.91$   DD 10.0%   Sortino -0.033
point mort 33.7%  ->  ecart -1.2 pt (+/- 3.3 au mieux)
classement rho [90m-0.0192[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m-0.0198[0m
vs politique gelee du meme run : -3.9 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    27W/55  L  32.9%     +81.66$   |   SHORT   40W/84  L  32.3%    -107.77$
actions B 1.2%  S 1.0%  H 97.8%   |   selectivite  train 18.5%  val 5.0%
politique  H 1.039/1.099 (-0.025)   etendue 0.7046 (705x le tremblement)   KL +0.0321   clipfrac 38.8%   g_actor 2.8e-01
train   PnL   -285.77$  2265 trades  WR 34.7%  PF 0.91   ->  ecart train-val +2.2 pt
. etendue val 0.7046, soit 705x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.13$) sous l'erreur-type de cette epoch (0.36$) — non separable de zero.
. clipfrac 38.8% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 76s  PPO 57s  calib 0s  validation 0s

## 21:23:12 — wf1 — EPOCH 009   +0.34$/trade   201 trades   WR 32.3%   PF 1.16   6 min

PnL          +68.59$   cumul run     +314.50$   DD 9.3%   Sortino +0.101
point mort 29.2%  ->  ecart +3.1 pt (+/- 3.3 au mieux)
classement rho [91m-0.0271[0m (+/- 0.024 env.)  -> classe A L'ENVERS   tete auxiliaire [91m-0.0205[0m
vs politique gelee du meme run : +0.4 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : +0.47$/trade
sens    LONG    23W/53  L  30.3%    +122.16$   |   SHORT   42W/83  L  33.6%     -53.57$
actions B 1.3%  S 1.3%  H 97.4%   |   selectivite  train 14.0%  val 5.0%
politique  H 1.025/1.099 (-0.014)   etendue 0.7230 (723x le tremblement)   KL +0.0332   clipfrac 44.1%   g_actor 2.8e-01
train   PnL   +703.52$  2992 trades  WR 33.8%  PF 1.18   ->  ecart train-val +1.5 pt
. etendue val 0.7230, soit 723x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.34$) sous l'erreur-type de cette epoch (0.36$) — non separable de zero.
. clipfrac 44.1% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 90s  PPO 80s  calib 32s  validation 149s

## 21:25:52 — wf1 — EPOCH 010   +0.34$/trade   201 trades   WR 32.3%   PF 1.16   3 min

PnL          +68.59$   cumul run     +383.09$   DD 9.3%   Sortino +0.101
point mort 29.2%  ->  ecart +3.1 pt (+/- 3.3 au mieux)
classement rho [90m-0.0070[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m+0.0028[0m
vs politique gelee du meme run : +0.4 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    23W/53  L  30.3%    +122.16$   |   SHORT   42W/83  L  33.6%     -53.57$
actions B 1.3%  S 1.1%  H 97.6%   |   selectivite  train 9.5%  val 5.0%
politique  H 0.991/1.099 (-0.034)   etendue 0.7230 (723x le tremblement)   KL +0.0302   clipfrac 41.5%   g_actor 2.9e-01
train   PnL   +442.86$  2732 trades  WR 33.3%  PF 1.13   ->  ecart train-val +1.0 pt
. etendue val 0.7230, soit 723x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.34$) sous l'erreur-type de cette epoch (0.36$) — non separable de zero.
. clipfrac 41.5% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 91s  PPO 62s  calib 0s  validation 0s

## 21:28:52 — wf1 — EPOCH 011   +0.34$/trade   201 trades   WR 32.3%   PF 1.16   3 min

PnL          +68.59$   cumul run     +451.68$   DD 9.3%   Sortino +0.101
point mort 29.2%  ->  ecart +3.1 pt (+/- 3.3 au mieux)
classement rho [90m-0.0141[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m-0.0051[0m
vs politique gelee du meme run : +0.4 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    23W/53  L  30.3%    +122.16$   |   SHORT   42W/83  L  33.6%     -53.57$
actions B 1.4%  S 1.1%  H 97.5%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.959/1.099 (-0.032)   etendue 0.7230 (723x le tremblement)   KL +0.0286   clipfrac 39.0%   g_actor 3.2e-01
train   PnL    +42.50$  3067 trades  WR 35.3%  PF 1.01   ->  ecart train-val +3.0 pt
. etendue val 0.7230, soit 723x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+0.34$) sous l'erreur-type de cette epoch (0.36$) — non separable de zero.
. clipfrac 39.0% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 99s  PPO 74s  calib 0s  validation 0s

## 21:34:52 — wf1 — EPOCH 012   +0.99$/trade   175 trades   WR 37.7%   PF 1.44   6 min

PnL         +172.76$   cumul run     +624.44$   DD 4.7%   Sortino +0.291
point mort 29.6%  ->  ecart +8.1 pt (+/- 3.7 au mieux)
classement rho [90m+0.0182[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [92m+0.0335[0m
vs politique gelee du meme run : +5.4 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : +0.65$/trade
sens    LONG    28W/47  L  37.3%    +169.85$   |   SHORT   38W/62  L  38.0%      +2.91$
actions B 1.4%  S 1.2%  H 97.5%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.944/1.099 (-0.015)   etendue 0.6802 (680x le tremblement)   KL +0.0331   clipfrac 38.0%   g_actor 3.2e-01
train   PnL   +537.67$  3067 trades  WR 34.1%  PF 1.13   ->  ecart train-val -3.6 pt
. etendue val 0.6802, soit 680x le tremblement du seuil : la selection est portee par le modele.
. clipfrac 38.0% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 100s  PPO 75s  calib 35s  validation 148s

## 21:37:32 — wf1 — EPOCH 013   +0.99$/trade   175 trades   WR 37.7%   PF 1.44   2 min

PnL         +172.76$   cumul run     +797.20$   DD 4.7%   Sortino +0.291
point mort 29.6%  ->  ecart +8.1 pt (+/- 3.7 au mieux)
classement rho [90m-0.0042[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m+0.0067[0m
vs politique gelee du meme run : +5.4 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    28W/47  L  37.3%    +169.85$   |   SHORT   38W/62  L  38.0%      +2.91$
actions B 1.3%  S 0.9%  H 97.8%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.903/1.099 (-0.041)   etendue 0.6802 (680x le tremblement)   KL +0.0289   clipfrac 33.3%   g_actor 3.3e-01
train   PnL   +592.39$  2601 trades  WR 37.9%  PF 1.14   ->  ecart train-val +0.2 pt
. etendue val 0.6802, soit 680x le tremblement du seuil : la selection est portee par le modele.
. clipfrac 33.3% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 87s  PPO 52s  calib 0s  validation 0s

## 21:39:52 — wf1 — EPOCH 014   +0.99$/trade   175 trades   WR 37.7%   PF 1.44   2 min

PnL         +172.76$   cumul run     +969.96$   DD 4.7%   Sortino +0.291
point mort 29.6%  ->  ecart +8.1 pt (+/- 3.7 au mieux)
classement rho [92m+0.0211[0m (+/- 0.024 env.)  -> classe   tete auxiliaire [92m+0.0391[0m
vs politique gelee du meme run : +5.4 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    28W/47  L  37.3%    +169.85$   |   SHORT   38W/62  L  38.0%      +2.91$
actions B 1.0%  S 0.7%  H 98.2%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.883/1.099 (-0.020)   etendue 0.6802 (680x le tremblement)   KL +0.0240   clipfrac 30.3%   g_actor 3.1e-01
train   PnL  +1086.54$  2469 trades  WR 40.8%  PF 1.25   ->  ecart train-val +3.1 pt
. etendue val 0.6802, soit 680x le tremblement du seuil : la selection est portee par le modele.
. clipfrac 30.3% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 95s  PPO 48s  calib 0s  validation 0s

## 21:46:53 — wf1 — EPOCH 015   +1.19$/trade   172 trades   WR 37.2%   PF 1.55   7 min

PnL         +204.43$   cumul run    +1174.39$   DD 4.6%   Sortino +0.355
point mort 27.7%  ->  ecart +9.5 pt (+/- 3.7 au mieux)
classement rho [92m+0.0215[0m (+/- 0.024 env.)  -> classe   tete auxiliaire [92m+0.0270[0m
vs politique gelee du meme run : +6.8 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : +0.20$/trade
sens    LONG    32W/48  L  40.0%    +222.54$   |   SHORT   32W/60  L  34.8%     -18.11$
actions B 1.5%  S 1.0%  H 97.5%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.853/1.099 (-0.030)   etendue 0.7329 (733x le tremblement)   KL +0.0271   clipfrac 32.4%   g_actor 2.9e-01
train   PnL  +1357.88$  4045 trades  WR 39.1%  PF 1.28   ->  ecart train-val +1.9 pt
. etendue val 0.7329, soit 733x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. clipfrac 32.4% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 125s  PPO 89s  calib 36s  validation 157s

## 21:49:53 — wf1 — EPOCH 016   +1.19$/trade   172 trades   WR 37.2%   PF 1.55   3 min

PnL         +204.43$   cumul run    +1378.82$   DD 4.6%   Sortino +0.355
point mort 27.7%  ->  ecart +9.5 pt (+/- 3.7 au mieux)
classement rho [90m-0.0086[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m-0.0159[0m
vs politique gelee du meme run : +6.8 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    32W/48  L  40.0%    +222.54$   |   SHORT   32W/60  L  34.8%     -18.11$
actions B 1.5%  S 1.0%  H 97.5%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.834/1.099 (-0.019)   etendue 0.7329 (733x le tremblement)   KL +0.0216   clipfrac 32.1%   g_actor 2.9e-01
train   PnL   +804.73$  3798 trades  WR 39.6%  PF 1.15   ->  ecart train-val +2.4 pt
. etendue val 0.7329, soit 733x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. clipfrac 32.1% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 116s  PPO 73s  calib 0s  validation 0s

## 21:53:13 — wf1 — EPOCH 017   +1.19$/trade   172 trades   WR 37.2%   PF 1.55   3 min

PnL         +204.43$   cumul run    +1583.25$   DD 4.6%   Sortino +0.355
point mort 27.7%  ->  ecart +9.5 pt (+/- 3.7 au mieux)
classement rho [90m-0.0017[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m-0.0121[0m
vs politique gelee du meme run : +6.8 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    32W/48  L  40.0%    +222.54$   |   SHORT   32W/60  L  34.8%     -18.11$
actions B 1.5%  S 1.2%  H 97.3%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.818/1.099 (-0.016)   etendue 0.7329 (733x le tremblement)   KL +0.0343   clipfrac 32.9%   g_actor 2.8e-01
train   PnL  +1776.37$  3818 trades  WR 39.9%  PF 1.38   ->  ecart train-val +2.7 pt
. etendue val 0.7329, soit 733x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. clipfrac 32.9% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 118s  PPO 78s  calib 0s  validation 0s

## 21:59:13 — wf1 — EPOCH 018   +0.79$/trade   199 trades   WR 32.2%   PF 1.38   6 min

PnL         +158.09$   cumul run    +1741.34$   DD 4.0%   Sortino +0.241
point mort 25.6%  ->  ecart +6.6 pt (+/- 3.3 au mieux)
classement rho [90m+0.0112[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m+0.0039[0m
vs politique gelee du meme run : +3.9 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : -0.39$/trade
sens    LONG    45W/78  L  36.6%    +196.55$   |   SHORT   19W/57  L  25.0%     -38.45$
actions B 1.4%  S 0.9%  H 97.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.800/1.099 (-0.018)   etendue 0.6829 (683x le tremblement)   KL +0.0315   clipfrac 29.5%   g_actor 2.7e-01
train   PnL   +289.03$  3167 trades  WR 37.2%  PF 1.06   ->  ecart train-val +5.0 pt
. etendue val 0.6829, soit 683x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 105s  PPO 64s  calib 32s  validation 145s

## 22:02:13 — wf1 — EPOCH 019   +0.79$/trade   199 trades   WR 32.2%   PF 1.38   3 min

PnL         +158.09$   cumul run    +1899.43$   DD 4.0%   Sortino +0.241
point mort 25.6%  ->  ecart +6.6 pt (+/- 3.3 au mieux)
classement rho [90m+0.0178[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m+0.0073[0m
vs politique gelee du meme run : +3.9 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    45W/78  L  36.6%    +196.55$   |   SHORT   19W/57  L  25.0%     -38.45$
actions B 1.3%  S 1.1%  H 97.6%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.796/1.099 (-0.004)   etendue 0.6829 (683x le tremblement)   KL +0.0407   clipfrac 37.6%   g_actor 2.3e-01
train   PnL  +1321.53$  3453 trades  WR 41.8%  PF 1.35   ->  ecart train-val +9.6 pt
. etendue val 0.6829, soit 683x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. clipfrac 37.6% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 108s  PPO 64s  calib 0s  validation 0s

## 22:05:53 — wf1 — EPOCH 020   +0.79$/trade   199 trades   WR 32.2%   PF 1.38   4 min

PnL         +158.09$   cumul run    +2057.52$   DD 4.0%   Sortino +0.241
point mort 25.6%  ->  ecart +6.6 pt (+/- 3.3 au mieux)
classement rho [90m+0.0050[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m-0.0106[0m
vs politique gelee du meme run : +3.9 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    45W/78  L  36.6%    +196.55$   |   SHORT   19W/57  L  25.0%     -38.45$
actions B 1.7%  S 1.7%  H 96.6%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.768/1.099 (-0.028)   etendue 0.6829 (683x le tremblement)   KL +0.0357   clipfrac 33.7%   g_actor 3.0e-01
train   PnL  +1650.11$  4831 trades  WR 36.9%  PF 1.35   ->  ecart train-val +4.7 pt
. etendue val 0.6829, soit 683x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. GPU BRIDE : 405 MHz a 88 C. Les durees ne sont pas comparables entre epochs si la temperature derive.
. clipfrac 33.7% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 119s  PPO 102s  calib 0s  validation 0s

## 22:13:53 — wf1 — EPOCH 021   -0.18$/trade   206 trades   WR 26.7%   PF 0.92   8 min

PnL          -37.51$   cumul run    +2020.01$   DD 7.2%   Sortino -0.052
point mort 28.4%  ->  ecart -1.7 pt (+/- 3.1 au mieux)
classement rho [90m-0.0158[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [91m-0.0314[0m
vs politique gelee du meme run : -4.4 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : -0.98$/trade
sens    LONG    33W/82  L  28.7%     +28.97$   |   SHORT   22W/69  L  24.2%     -66.48$
actions B 2.0%  S 1.6%  H 96.4%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.759/1.099 (-0.009)   etendue 0.6924 (692x le tremblement)   KL +0.0380   clipfrac 35.9%   g_actor 2.4e-01
train   PnL  +1548.64$  4545 trades  WR 38.5%  PF 1.42   ->  ecart train-val +11.8 pt
. etendue val 0.6924, soit 692x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.18$) sous l'erreur-type de cette epoch (0.34$) — non separable de zero.
. clipfrac 35.9% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 117s  PPO 100s  calib 43s  validation 203s

## 22:16:33 — wf1 — EPOCH 022   -0.18$/trade   206 trades   WR 26.7%   PF 0.92   3 min

PnL          -37.51$   cumul run    +1982.50$   DD 7.2%   Sortino -0.052
point mort 28.4%  ->  ecart -1.7 pt (+/- 3.1 au mieux)
classement rho [91m-0.0240[0m (+/- 0.024 env.)  -> classe A L'ENVERS   tete auxiliaire [91m-0.0519[0m
vs politique gelee du meme run : -4.4 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    33W/82  L  28.7%     +28.97$   |   SHORT   22W/69  L  24.2%     -66.48$
actions B 1.5%  S 1.0%  H 97.5%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.772/1.099 (+0.013)   etendue 0.6924 (692x le tremblement)   KL +0.0334   clipfrac 31.3%   g_actor 3.4e-01
train   PnL  +1701.83$  2913 trades  WR 37.7%  PF 1.50   ->  ecart train-val +11.0 pt
. etendue val 0.6924, soit 692x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.18$) sous l'erreur-type de cette epoch (0.34$) — non separable de zero.
. clipfrac 31.3% — la mise a jour tape souvent la borne de clipping, les pas sont tronques.
temps : collecte 107s  PPO 61s  calib 0s  validation 0s

## 22:19:34 — wf1 — EPOCH 023   -0.18$/trade   206 trades   WR 26.7%   PF 0.92   3 min

PnL          -37.51$   cumul run    +1944.99$   DD 7.2%   Sortino -0.052
point mort 28.4%  ->  ecart -1.7 pt (+/- 3.1 au mieux)
classement rho [91m-0.0371[0m (+/- 0.024 env.)  -> classe A L'ENVERS   tete auxiliaire [91m-0.0433[0m
vs politique gelee du meme run : -4.4 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    33W/82  L  28.7%     +28.97$   |   SHORT   22W/69  L  24.2%     -66.48$
actions B 1.1%  S 0.9%  H 98.1%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.769/1.099 (-0.003)   etendue 0.6924 (692x le tremblement)   KL +0.0264   clipfrac 27.9%   g_actor 4.1e-01
train   PnL   +338.64$  2466 trades  WR 38.0%  PF 1.07   ->  ecart train-val +11.3 pt
. etendue val 0.6924, soit 692x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.18$) sous l'erreur-type de cette epoch (0.34$) — non separable de zero.
temps : collecte 116s  PPO 51s  calib 0s  validation 0s

## 22:26:14 — wf1 — EPOCH 024   +0.62$/trade   168 trades   WR 34.5%   PF 1.28   6 min

PnL         +103.54$   cumul run    +2048.53$   DD 4.1%   Sortino +0.188
point mort 29.2%  ->  ecart +5.3 pt (+/- 3.7 au mieux)
classement rho [91m-0.0218[0m (+/- 0.024 env.)  -> classe A L'ENVERS   tete auxiliaire [91m-0.0224[0m
vs politique gelee du meme run : +2.6 pt   (reference +2.7 pt sur 5 epochs)
vs epoch precedente : +0.80$/trade
sens    LONG    38W/55  L  40.9%    +150.57$   |   SHORT   20W/55  L  26.7%     -47.03$
actions B 1.4%  S 1.0%  H 97.5%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.752/1.099 (-0.017)   etendue 0.7313 (731x le tremblement)   KL +0.0262   clipfrac 29.7%   g_actor 3.2e-01
train   PnL   +196.54$  3818 trades  WR 35.4%  PF 1.04   ->  ecart train-val +0.9 pt
. etendue val 0.7313, soit 731x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 119s  PPO 80s  calib 33s  validation 153s

## 22:33:36 — wf1 — EPOCH 001   +0.49$/trade   248 trades   WR 29.8%   PF 1.27   4 min

PnL         +122.28$   cumul run     +122.28$   DD 5.8%   Sortino +0.190
point mort 25.1%  ->  ecart +4.7 pt (+/- 2.9 au mieux)
classement rho [90m+0.0073[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [91m-0.0425[0m
sens    LONG    40W/91  L  30.5%    +153.47$   |   SHORT   34W/83  L  29.1%     -31.19$
actions B 1.1%  S 1.0%  H 97.9%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.1723 (172x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 6.6e-06
train   PnL   +206.78$  859 trades  WR 35.4%  PF 1.16   ->  ecart train-val +5.6 pt
. ACTOR GELE : warmup du critic, gradient 6.6e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.1723, soit 172x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 30s  PPO 19s  calib 30s  validation 155s

## 22:34:56 — wf1 — EPOCH 002   +0.49$/trade   248 trades   WR 29.8%   PF 1.27   1 min

PnL         +122.28$   cumul run     +244.56$   DD 5.8%   Sortino +0.190
point mort 25.1%  ->  ecart +4.7 pt (+/- 2.9 au mieux)
classement rho [92m+0.0266[0m (+/- 0.024 env.)  -> classe   tete auxiliaire [91m-0.0233[0m
vs epoch precedente : +0.00$/trade
sens    LONG    40W/91  L  30.5%    +153.47$   |   SHORT   34W/83  L  29.1%     -31.19$
actions B 0.6%  S 0.8%  H 98.6%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.1723 (172x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 5.1e-06
train   PnL   +130.63$  1149 trades  WR 33.0%  PF 1.04   ->  ecart train-val +3.2 pt
. ACTOR GELE : warmup du critic, gradient 5.1e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.1723, soit 172x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 53s  PPO 26s  calib 0s  validation 0s

## 23:12:31 — wf1 — EPOCH 001   +3.74$/trade   55 trades   WR 41.8%   PF 1.35   3 min

PnL         +205.70$   cumul run     +205.70$   DD 18.7%   Sortino +0.242
point mort 34.7%  ->  ecart +7.1 pt (+/- 6.7 au mieux)
classement rho [90m+0.0083[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m-0.0025[0m
sens    LONG    11W/15  L  42.3%    +253.94$   |   SHORT   12W/17  L  41.4%     -48.23$
actions B 1.3%  S 1.4%  H 97.3%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.3378 (338x le tremblement)   KL -0.0001   clipfrac 0.0%   g_actor 1.3e-05
train   PnL   -455.81$  960 trades  WR 34.7%  PF 0.94   ->  ecart train-val -7.1 pt
. ACTOR GELE : warmup du critic, gradient 1.3e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.3378, soit 338x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 24s  PPO 19s  calib 22s  validation 102s

## 23:39:36 — wf1 — EPOCH 001   -8.41$/trade   70 trades   WR 27.1%   PF 0.40   2 min

PnL         -588.60$   cumul run     -588.60$   DD 20.4%   Sortino -0.481
point mort 48.2%  ->  ecart -21.1 pt (+/- 5.3 au mieux)
classement rho [90m-0.0043[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [91m-0.0727[0m
sens    LONG     6W/13  L  31.6%     -80.84$   |   SHORT   13W/38  L  25.5%    -507.76$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 50.0%  val 5.0%
politique  H 1.099/1.099   etendue 0.1924 (192x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 8.3e-06
train   PnL   +325.11$  198 trades  WR 33.3%  PF 1.24   ->  ecart train-val +6.2 pt
. ACTOR GELE : warmup du critic, gradient 8.3e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.1924, soit 192x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 27s  PPO 4s  calib 21s  validation 86s

## 23:40:36 — wf1 — EPOCH 002   -8.41$/trade   70 trades   WR 27.1%   PF 0.40   1 min

PnL         -588.60$   cumul run    -1177.20$   DD 20.4%   Sortino -0.481
point mort 48.2%  ->  ecart -21.1 pt (+/- 5.3 au mieux)
classement rho [90m+0.0094[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [91m-0.0982[0m
vs epoch precedente : +0.00$/trade
sens    LONG     6W/13  L  31.6%     -80.84$   |   SHORT   13W/38  L  25.5%    -507.76$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 45.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.1924 (192x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 5.4e-06
train   PnL    -64.03$  411 trades  WR 32.6%  PF 0.98   ->  ecart train-val +5.5 pt
. ACTOR GELE : warmup du critic, gradient 5.4e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.1924, soit 192x le tremblement du seuil : la selection est portee par le modele.
temps : collecte 60s  PPO 8s  calib 0s  validation 0s

## 23:44:16 — wf1 — EPOCH 003   -0.36$/trade   75 trades   WR 34.7%   PF 0.96   4 min

PnL          -26.87$   cumul run    -1204.07$   DD 13.8%   Sortino -0.027
point mort 35.6%  ->  ecart -0.9 pt (+/- 5.5 au mieux)
classement rho [90m-0.0025[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m-0.0144[0m
vs politique gelee du meme run : +20.2 pt   (reference -21.1 pt sur 2 epochs)
vs epoch precedente : +8.05$/trade
sens    LONG     9W/14  L  39.1%    +198.92$   |   SHORT   17W/35  L  32.7%    -225.79$
actions B 0.1%  S 0.1%  H 99.7%   |   selectivite  train 41.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.1749 (175x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 3.1e-06
train   PnL   +161.07$  534 trades  WR 31.5%  PF 1.04   ->  ecart train-val -3.2 pt
. ACTOR GELE : warmup du critic, gradient 3.1e-06. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.1749, soit 175x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.36$) sous l'erreur-type de cette epoch (2.12$) — non separable de zero.
temps : collecte 75s  PPO 10s  calib 23s  validation 109s

## 23:45:56 — wf1 — EPOCH 004   -0.36$/trade   75 trades   WR 34.7%   PF 0.96   2 min

PnL          -26.87$   cumul run    -1230.94$   DD 13.8%   Sortino -0.027
point mort 35.6%  ->  ecart -0.9 pt (+/- 5.5 au mieux)
classement rho [90m-0.0167[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [90m-0.0129[0m
vs politique gelee du meme run : +13.5 pt   (reference -14.4 pt sur 3 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG     9W/14  L  39.1%    +198.92$   |   SHORT   17W/35  L  32.7%    -225.79$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 36.5%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.1749 (175x le tremblement)   KL +0.0000   clipfrac 0.0%   g_actor 4.0e-05
train   PnL   -213.68$  622 trades  WR 33.3%  PF 0.96   ->  ecart train-val -1.4 pt
. ACTOR GELE : warmup du critic, gradient 4.0e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.1749, soit 175x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.36$) sous l'erreur-type de cette epoch (2.12$) — non separable de zero.
temps : collecte 87s  PPO 12s  calib 0s  validation 0s

## 23:47:36 — wf1 — EPOCH 005   -0.36$/trade   75 trades   WR 34.7%   PF 0.96   2 min

PnL          -26.87$   cumul run    -1257.81$   DD 13.8%   Sortino -0.027
point mort 35.6%  ->  ecart -0.9 pt (+/- 5.5 au mieux)
classement rho [92m+0.0207[0m (+/- 0.024 env.)  -> classe   tete auxiliaire [90m-0.0034[0m
vs politique gelee du meme run : +10.1 pt   (reference -11.0 pt sur 4 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG     9W/14  L  39.1%    +198.92$   |   SHORT   17W/35  L  32.7%    -225.79$
actions B 0.2%  S 0.2%  H 99.7%   |   selectivite  train 32.0%  val 5.0%
politique  H 1.099/1.099 (+0.000)   etendue 0.1749 (175x le tremblement)   KL -0.0000   clipfrac 0.0%   g_actor 1.7e-05
train   PnL   +182.41$  749 trades  WR 31.8%  PF 1.03   ->  ecart train-val -2.9 pt
. ACTOR GELE : warmup du critic, gradient 1.7e-05. Cette epoch ne mesure aucun apprentissage — elle sert de reference au hasard pour les suivantes.
. etendue val 0.1749, soit 175x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-0.36$) sous l'erreur-type de cette epoch (2.12$) — non separable de zero.
temps : collecte 90s  PPO 14s  calib 0s  validation 0s

## 23:51:36 — wf1 — EPOCH 006   +2.55$/trade   62 trades   WR 35.5%   PF 1.26   4 min

PnL         +158.11$   cumul run    -1099.70$   DD 17.7%   Sortino +0.190
point mort 30.4%  ->  ecart +5.1 pt (+/- 6.1 au mieux)
classement rho [90m+0.0077[0m (+/- 0.024 env.)  -> n'ordonne rien   tete auxiliaire [92m+0.0310[0m
vs politique gelee du meme run : +14.1 pt   (reference -9.0 pt sur 5 epochs)
vs epoch precedente : +2.91$/trade
sens    LONG    11W/14  L  44.0%    +264.40$   |   SHORT   11W/26  L  29.7%    -106.29$
actions B 0.2%  S 0.2%  H 99.7%   |   selectivite  train 27.5%  val 5.0%
politique  H 1.093/1.099 (-0.006)   etendue 0.3582 (358x le tremblement)   KL +0.0061   clipfrac 11.9%   g_actor 2.3e-01
train   PnL   +180.95$  770 trades  WR 33.2%  PF 1.03   ->  ecart train-val -2.3 pt
. APPREND MAIS RESTE PLAT : le gradient passe (2.3e-01) mais l'entropie tient a 1.093 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.3582, soit 358x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+2.55$) sous l'erreur-type de cette epoch (3.04$) — non separable de zero.
temps : collecte 93s  PPO 15s  calib 26s  validation 100s

## 23:53:16 — wf1 — EPOCH 007   +2.55$/trade   62 trades   WR 35.5%   PF 1.26   2 min

PnL         +158.11$   cumul run     -941.59$   DD 17.7%   Sortino +0.190
point mort 30.4%  ->  ecart +5.1 pt (+/- 6.1 au mieux)
classement rho [92m+0.0759[0m (+/- 0.024 env.)  -> classe nettement   tete auxiliaire [92m+0.0808[0m
vs politique gelee du meme run : +14.1 pt   (reference -9.0 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    11W/14  L  44.0%    +264.40$   |   SHORT   11W/26  L  29.7%    -106.29$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 23.0%  val 5.0%
politique  H 1.090/1.099 (-0.003)   etendue 0.3582 (358x le tremblement)   KL +0.0021   clipfrac 5.2%   g_actor 3.0e-01
train   PnL   -600.08$  667 trades  WR 31.9%  PF 0.89   ->  ecart train-val -3.6 pt
. APPREND MAIS RESTE PLAT : le gradient passe (3.0e-01) mais l'entropie tient a 1.090 sur 1.099. La politique se differencie trop lentement pour que le filtre ait un sens — c'est un probleme de pas d'apprentissage ou d'echelle, pas de tirage.
. etendue val 0.3582, soit 358x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+2.55$) sous l'erreur-type de cette epoch (3.04$) — non separable de zero.
temps : collecte 92s  PPO 12s  calib 0s  validation 0s

## 23:55:16 — wf1 — EPOCH 008   +2.55$/trade   62 trades   WR 35.5%   PF 1.26   2 min

PnL         +158.11$   cumul run     -783.48$   DD 17.7%   Sortino +0.190
point mort 30.4%  ->  ecart +5.1 pt (+/- 6.1 au mieux)
classement rho [92m+0.0983[0m (+/- 0.024 env.)  -> classe nettement   tete auxiliaire [92m+0.0995[0m
vs politique gelee du meme run : +14.1 pt   (reference -9.0 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    11W/14  L  44.0%    +264.40$   |   SHORT   11W/26  L  29.7%    -106.29$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 18.5%  val 5.0%
politique  H 1.082/1.099 (-0.008)   etendue 0.3582 (358x le tremblement)   KL +0.0015   clipfrac 7.3%   g_actor 2.9e-01
train   PnL    +78.32$  666 trades  WR 33.0%  PF 1.02   ->  ecart train-val -2.5 pt
. etendue val 0.3582, soit 358x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+2.55$) sous l'erreur-type de cette epoch (3.04$) — non separable de zero.
temps : collecte 91s  PPO 13s  calib 0s  validation 0s

## 23:58:56 — wf1 — EPOCH 009   +3.11$/trade   64 trades   WR 39.1%   PF 1.30   4 min

PnL         +198.84$   cumul run     -584.64$   DD 16.6%   Sortino +0.213
point mort 33.0%  ->  ecart +6.1 pt (+/- 6.1 au mieux)
classement rho [92m+0.0684[0m (+/- 0.024 env.)  -> classe nettement   tete auxiliaire [92m+0.0612[0m
vs politique gelee du meme run : +15.1 pt   (reference -9.0 pt sur 5 epochs)
vs epoch precedente : +0.56$/trade
sens    LONG    13W/16  L  44.8%    +377.77$   |   SHORT   12W/23  L  34.3%    -178.93$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 14.0%  val 5.0%
politique  H 1.064/1.099 (-0.018)   etendue 0.5033 (503x le tremblement)   KL +0.0057   clipfrac 8.8%   g_actor 2.7e-01
train   PnL   -482.74$  680 trades  WR 34.0%  PF 0.91   ->  ecart train-val -5.1 pt
. etendue val 0.5033, soit 503x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+3.11$) sous l'erreur-type de cette epoch (3.14$) — non separable de zero.
temps : collecte 95s  PPO 12s  calib 23s  validation 94s

## 00:00:56 — wf1 — EPOCH 010   +3.11$/trade   64 trades   WR 39.1%   PF 1.30   2 min

PnL         +198.84$   cumul run     -385.80$   DD 16.6%   Sortino +0.213
point mort 33.0%  ->  ecart +6.1 pt (+/- 6.1 au mieux)
classement rho [92m+0.0695[0m (+/- 0.024 env.)  -> classe nettement   tete auxiliaire [92m+0.0530[0m
vs politique gelee du meme run : +15.1 pt   (reference -9.0 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    13W/16  L  44.8%    +377.77$   |   SHORT   12W/23  L  34.3%    -178.93$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 9.5%  val 5.0%
politique  H 1.054/1.099 (-0.010)   etendue 0.5033 (503x le tremblement)   KL +0.0058   clipfrac 19.7%   g_actor 2.3e-01
train   PnL   +789.43$  641 trades  WR 34.2%  PF 1.16   ->  ecart train-val -4.9 pt
. etendue val 0.5033, soit 503x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+3.11$) sous l'erreur-type de cette epoch (3.14$) — non separable de zero.
temps : collecte 104s  PPO 11s  calib 0s  validation 0s

## 00:02:56 — wf1 — EPOCH 011   +3.11$/trade   64 trades   WR 39.1%   PF 1.30   2 min

PnL         +198.84$   cumul run     -186.96$   DD 16.6%   Sortino +0.213
point mort 33.0%  ->  ecart +6.1 pt (+/- 6.1 au mieux)
classement rho [92m+0.0975[0m (+/- 0.024 env.)  -> classe nettement   tete auxiliaire [92m+0.0893[0m
vs politique gelee du meme run : +15.1 pt   (reference -9.0 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG    13W/16  L  44.8%    +377.77$   |   SHORT   12W/23  L  34.3%    -178.93$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.040/1.099 (-0.014)   etendue 0.5033 (503x le tremblement)   KL +0.0059   clipfrac 12.5%   g_actor 2.4e-01
train   PnL   +555.12$  639 trades  WR 33.6%  PF 1.11   ->  ecart train-val -5.5 pt
. etendue val 0.5033, soit 503x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (+3.11$) sous l'erreur-type de cette epoch (3.14$) — non separable de zero.
temps : collecte 109s  PPO 11s  calib 0s  validation 0s

## 00:07:17 — wf1 — EPOCH 012   -3.21$/trade   80 trades   WR 32.5%   PF 0.68   4 min

PnL         -257.02$   cumul run     -443.98$   DD 16.1%   Sortino -0.232
point mort 41.5%  ->  ecart -9.0 pt (+/- 5.2 au mieux)
classement rho [92m+0.0903[0m (+/- 0.024 env.)  -> classe nettement   tete auxiliaire [92m+0.0842[0m
vs politique gelee du meme run : +0.0 pt   (reference -9.0 pt sur 5 epochs)
vs epoch precedente : -6.32$/trade
sens    LONG     6W/15  L  28.6%     +34.42$   |   SHORT   20W/39  L  33.9%    -291.44$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 1.020/1.099 (-0.020)   etendue 0.5625 (562x le tremblement)   KL +0.0172   clipfrac 15.5%   g_actor 2.9e-01
train   PnL   +552.77$  652 trades  WR 33.4%  PF 1.12   ->  ecart train-val +0.9 pt
. etendue val 0.5625, soit 562x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 107s  PPO 12s  calib 29s  validation 113s

## 00:09:17 — wf1 — EPOCH 013   -3.21$/trade   80 trades   WR 32.5%   PF 0.68   2 min

PnL         -257.02$   cumul run     -701.00$   DD 16.1%   Sortino -0.232
point mort 41.5%  ->  ecart -9.0 pt (+/- 5.2 au mieux)
classement rho [92m+0.0618[0m (+/- 0.024 env.)  -> classe nettement   tete auxiliaire [92m+0.0520[0m
vs politique gelee du meme run : +0.0 pt   (reference -9.0 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG     6W/15  L  28.6%     +34.42$   |   SHORT   20W/39  L  33.9%    -291.44$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.996/1.099 (-0.024)   etendue 0.5625 (562x le tremblement)   KL +0.0153   clipfrac 19.9%   g_actor 2.3e-01
train   PnL   +862.56$  662 trades  WR 35.0%  PF 1.17   ->  ecart train-val +2.5 pt
. etendue val 0.5625, soit 562x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 105s  PPO 11s  calib 0s  validation 0s

## 00:11:17 — wf1 — EPOCH 014   -3.21$/trade   80 trades   WR 32.5%   PF 0.68   2 min

PnL         -257.02$   cumul run     -958.02$   DD 16.1%   Sortino -0.232
point mort 41.5%  ->  ecart -9.0 pt (+/- 5.2 au mieux)
classement rho [92m+0.0859[0m (+/- 0.024 env.)  -> classe nettement   tete auxiliaire [92m+0.0625[0m
vs politique gelee du meme run : +0.0 pt   (reference -9.0 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG     6W/15  L  28.6%     +34.42$   |   SHORT   20W/39  L  33.9%    -291.44$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.991/1.099 (-0.005)   etendue 0.5625 (562x le tremblement)   KL +0.0145   clipfrac 20.7%   g_actor 2.0e-01
train   PnL  +1727.56$  690 trades  WR 40.7%  PF 1.39   ->  ecart train-val +8.2 pt
. etendue val 0.5625, soit 562x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
temps : collecte 104s  PPO 12s  calib 0s  validation 0s

## 00:14:57 — wf1 — EPOCH 015   -2.21$/trade   57 trades   WR 31.6%   PF 0.82   4 min

PnL         -126.22$   cumul run    -1084.24$   DD 15.6%   Sortino -0.142
point mort 36.0%  ->  ecart -4.4 pt (+/- 6.2 au mieux)
classement rho [92m+0.1035[0m (+/- 0.024 env.)  -> classe nettement   tete auxiliaire [92m+0.0967[0m
vs politique gelee du meme run : +4.6 pt   (reference -9.0 pt sur 5 epochs)
vs epoch precedente : +1.00$/trade
sens    LONG     5W/12  L  29.4%    +142.69$   |   SHORT   13W/27  L  32.5%    -268.91$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.993/1.099 (+0.002)   etendue 0.6383 (638x le tremblement)   KL +0.0172   clipfrac 19.7%   g_actor 2.6e-01
train   PnL  +1200.26$  564 trades  WR 41.3%  PF 1.30   ->  ecart train-val +9.7 pt
. etendue val 0.6383, soit 638x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-2.21$) sous l'erreur-type de cette epoch (3.07$) — non separable de zero.
temps : collecte 96s  PPO 10s  calib 24s  validation 100s

## 00:16:57 — wf1 — EPOCH 016   -2.21$/trade   57 trades   WR 31.6%   PF 0.82   2 min

PnL         -126.22$   cumul run    -1210.46$   DD 15.6%   Sortino -0.142
point mort 36.0%  ->  ecart -4.4 pt (+/- 6.2 au mieux)
classement rho [92m+0.0843[0m (+/- 0.024 env.)  -> classe nettement   tete auxiliaire [92m+0.0562[0m
vs politique gelee du meme run : +4.6 pt   (reference -9.0 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG     5W/12  L  29.4%    +142.69$   |   SHORT   13W/27  L  32.5%    -268.91$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.986/1.099 (-0.007)   etendue 0.6383 (638x le tremblement)   KL +0.0120   clipfrac 17.6%   g_actor 2.1e-01
train   PnL   +352.74$  683 trades  WR 35.0%  PF 1.07   ->  ecart train-val +3.4 pt
. etendue val 0.6383, soit 638x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-2.21$) sous l'erreur-type de cette epoch (3.07$) — non separable de zero.
temps : collecte 100s  PPO 11s  calib 0s  validation 0s

## 00:18:57 — wf1 — EPOCH 017   -2.21$/trade   57 trades   WR 31.6%   PF 0.82   2 min

PnL         -126.22$   cumul run    -1336.68$   DD 15.6%   Sortino -0.142
point mort 36.0%  ->  ecart -4.4 pt (+/- 6.2 au mieux)
classement rho [92m+0.0724[0m (+/- 0.024 env.)  -> classe nettement   tete auxiliaire [92m+0.0597[0m
vs politique gelee du meme run : +4.6 pt   (reference -9.0 pt sur 5 epochs)
vs epoch precedente : +0.00$/trade
sens    LONG     5W/12  L  29.4%    +142.69$   |   SHORT   13W/27  L  32.5%    -268.91$
actions B 0.2%  S 0.1%  H 99.7%   |   selectivite  train 5.0%  val 5.0%
politique  H 0.968/1.099 (-0.018)   etendue 0.6383 (638x le tremblement)   KL +0.0125   clipfrac 18.0%   g_actor 2.1e-01
train   PnL  +1587.57$  725 trades  WR 38.9%  PF 1.34   ->  ecart train-val +7.3 pt
. etendue val 0.6383, soit 638x le tremblement du seuil : la selection est portee par le modele.
. GAIN UNILATERAL : seul le LONG rapporte. Un modele qui ne gagne que d'un cote a un biais directionnel, pas un avantage — verifier que l'autre sens suit avant de conclure quoi que ce soit.
. gain par trade (-2.21$) sous l'erreur-type de cette epoch (3.07$) — non separable de zero.
temps : collecte 101s  PPO 12s  calib 0s  validation 0s
