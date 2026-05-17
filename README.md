# 🐺 Loup Ω — Agent RL Multi-Stratégie pour BTCUSD M1

> **PPO + SAINTv2** : agent d'apprentissage par renforcement entraîné en walk-forward sur l'historique BTCUSD à la minute, déployable en live sur MetaTrader 5 via une interface graphique dédiée.

---

## 📑 Table des matières

1. [Vue d'ensemble](#-vue-densemble)
2. [Architecture du modèle](#-architecture-du-modèle)
3. [Pipeline complet](#-pipeline-complet)
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

---

## 🎯 Vue d'ensemble

**Loup Ω** est un système complet de trading algorithmique BTCUSD M1 (1 minute) basé sur l'apprentissage par renforcement profond. Il combine :

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
| Période d'entraînement | 2022-01-01 → maintenant (~4 ans) |
| Méthodologie | Walk-forward 3 folds (55/15/10) |
| Backbone size | d_model=80, 2 blocks, 4 heads |
| Features | 11 M1 + 5 H1 + 4 position = 20 |
| Lookback | 25 bougies M1 |

---

## 🏗️ Architecture du modèle

### SAINTv2 (Self-Attention and Intersample Attention Transformer v2)

Architecture transformer **dual-axis** spécialement adaptée aux données tabulaires temporelles, inspirée du papier _SAINT_ (Somepalli et al. 2021) avec amélioration v2 : gated FFN à la PaLM/Gemma + double row-attention par block.

```
Input (B, T=25, F=20)
   │
   ├─► Linear projection ──► (B, T, F, d_model=80)
   ├─► + Row embedding  (temporel)
   └─► + Col embedding  (feature)
              │
              ▼
   ┌──── Block × 2 ──────────────────────────────┐
   │   RowAttn (T-axis attention)                │
   │      ↳ chaque feature regarde ses voisins   │
   │        temporels                            │
   │   GatedFFN (SwiGLU-like)                    │
   │   RowAttn (2e passe : raffinement)          │
   │   GatedFFN                                  │
   │   ColAttn (F-axis attention)                │
   │      ↳ chaque pas de temps mixe ses        │
   │        features entre elles                 │
   │   GatedFFN                                  │
   └─────────────────────────────────────────────┘
              │
              ├──► time-mean ──► (B, F, d)
              └──► feat-mean ──► (B, T, d)
                          │
                  CLS pooling (mean)
                          │
                       LayerNorm
                          │
                   MLP 80 → 256 → 256
                       /        \
                  actor          critic
                  (B, 3)         (B, 1)
```

**Pourquoi SAINTv2** : sur des données financières (M1 OHLC + indicateurs + HTF), la corrélation temporelle ET inter-features est cruciale. Les transformers classiques attaquent une seule des deux axes ; SAINTv2 mixe les deux dans chaque bloc → meilleure modélisation des patterns complexes (rejet de support, retest, divergence RSI vs prix, etc.).

### Référence
- [SAINT (Somepalli et al., NeurIPS 2021)](https://arxiv.org/abs/2106.01342) — papier original
- [Tabular Transformers benchmark](https://arxiv.org/abs/2207.08815) — SAINTv2 montre 2-5 % d'AUC en plus vs SAINT v1 sur la majorité des tâches tabulaires

### PPO (Proximal Policy Optimization)

Implémentation maison avec :
- **Clipped surrogate objective** (ratio ε = 0.18)
- **GAE-λ** (γ=0.97, λ=0.95) pour le calcul des avantages
- **Critic warmup** : 5 epochs où seul le critic apprend (init du baseline V(s))
- **KL early stop** : interruption d'une epoch si KL > 0.03
- **Cosine LR scheduling** : 3e-4 → 1.5e-5 sur 240 epochs
- **Reward normalization** : algorithme de Welford online pour stabiliser CriticL

### Référence
- [PPO paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [GAE (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438)

---

## 🔄 Pipeline complet

```
┌──────────────────────────────────────────────────────────────────┐
│                    1. TRAINING (training.py)                     │
│                                                                  │
│  MT5 (BTCUSD M1+H1)  →  Indicateurs (RSI, ATR, vol, momentum) →  │
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
│           2. VALIDATION (backtest_saintv2_stress_test.py)        │
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
│             3. LIVE TRADING (loup_live.py + gui_loup.py)         │
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

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `epochs` | 240 | Plus de runway avec cosine LR |
| `episodes_per_epoch` | 4 | Diversité par epoch |
| `episode_length` | 4 000 | ~2.8 jours M1, trajectoires longues |
| `val_episodes` | 7 | Augmenté de 2 → 7 pour stabiliser le Sortino |
| `batch_size` | 256 | GPU-friendly |
| `clip_eps` | 0.18 | Standard PPO |
| `target_kl` | 0.03 | Early stop si dépassé |
| `gamma` | 0.97 | Horizon court pour scalping |
| `lambda_gae` | 0.95 | Standard |
| `lr` | 3e-4 → 1.5e-5 (cosine) | Décroissance douce |
| `entropy_coef` | 0.30 | Pousse l'exploration SHORT (×2.0 → ×0.8 sur 120 ep) |
| `value_coef` | 0.5 | Standard |
| `max_grad_norm` | 0.3 | Anti gradient-explosion (resserré + unscale AMP) |
| `max_drawdown` | 0.4 | Force l'apprentissage prudent (resserré depuis 0.8) |
| `CONF_THRESHOLD` | 0.90 | Filtre exploration training (resserré depuis 0.95) |
| `critic_warmup_epochs` | 5 | Stabilité initiale |
| `lookback` | 25 | ~25 minutes de contexte |
| `position_size` | 0.06 lot | Risk-tuned BTCUSD |
| `leverage` | 6× | Leverage modéré |
| `atr_sl_mult` | 1.2 | SL = 1.2 × ATR(14) |
| `atr_tp_mult` | 2.4 | TP = 2.4 × ATR (R:R initial = 2:1) |
| `tp_shrink` | 0.7 | TP final = 2.4 × 0.7 = 1.68 × ATR |

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
python backtest_saintv2_stress_test.py
```

Par défaut backteste `bestprofit_saintv2_loup_duel_wf1_both_wf1.pth` en mode `side="both"` avec `min_confidence=0.0` (argmax pur). Modifie `LiveConfig(...)` à la fin du fichier pour tester :
- `side="both"` : modèle duel unifié
- `side="long"` / `"short"` : modèles spécialisés
- `side="duel"` : 2 modèles séparés en arbitrage
- `min_confidence=0.0` (argmax pur) ou ajuster selon le calibrage du modèle

### Variante `no_be_trail` (sans break-even / trailing)

```powershell
python backtest_saintv2_no_be_trail.py
```

**Fichier identique** à `backtest_saintv2_stress_test.py` mais avec l'appel à
`update_sl_be_trailing_backtest()` **commenté** (ligne ~980). Les positions
ne ferment qu'au SL ou TP fixe.

Génère `backtest_trades_both_no_be_trail.csv` (n'écrase pas l'original).

**Pourquoi cette variante** : sur le run de mai 2026 (~75 jours, 9986 trades),
le backtest avec BE/trail finit à PnL **−499 $ / PF 0.98**, alors qu'au même
modèle sans BE/trail on est à **+1902 $ / PF 1.70 / WR 56.6%** après 27% du run.
Le BE/trailing actuel ferme les wins trop tôt.

### 3. Live trading (GUI)

```powershell
python gui_loup.py
```

- Choisis ta taille de lot
- Clique **▶️ Démarrer l'IA**
- Le bot tournera en boucle, attendant chaque nouvelle bougie M1 fermée
- **⏹️ Arrêter l'IA** pour stopper (les positions ouvertes restent ouvertes, gérées par leur SL/TP)

### 4. Live sans GUI (CLI)

```powershell
python loup_live.py
```

---

## 📁 Structure du projet

```
multi-agent-btcusd/
├── training.py                     # Entraînement PPO + SAINTv2 walk-forward
├── loup_live.py                    # Agent live MT5 (sans GUI) — BE/trail off
├── gui_loup.py                     # Interface graphique PySide6
├── backtest_saintv2_stress_test.py # Backtest institutionnel (BE/trail on)
├── backtest_saintv2_no_be_trail.py # Variante sans BE/trail (comparaison)
├── requirements.txt                # Dépendances Python
├── .gitignore                      # Exclusions git (.pth, .csv, .npz)
├── README.md                       # Ce fichier
│
├── (générés après training)
├── norm_stats_ohlc_indics.npz                                   # Stats Z-score globales
├── best_saintv2_loup_duel_wf{1,2,3}_both_wf{1,2,3}.pth          # Best Sortino30
├── bestprofit_saintv2_loup_duel_wf{1,2,3}_both_wf{1,2,3}.pth    # Best ValPNL
├── last_saintv2_loup_duel_wf{1,2,3}_both_wf{1,2,3}.pth          # Final epoch
├── training_log_both_wf{1,2,3}.csv                              # Logs CSV epoch-level
├── trades_both_wf{1,2,3}.csv                                    # Trade-by-trade (NEW)
└── backtest_trades_both.csv                                     # Trades du backtest (NEW)
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
| **Training** (`CONF_THRESHOLD`) | 0.90 | Filtre exploration : HOLD si `softmax(BUY ou SELL) < 0.90` |
| **Backtest** (`min_confidence`) | configurable | 0.0 = argmax pur (recommandé pour ce modèle) |
| **Live** (`min_confidence`) | configurable | 0.0 par défaut (argmax pur) |

**Note importante** : avec 3 actions équiprobables (1/3 ≈ 0.33), un modèle bien calibré plafonne souvent autour de 0.40-0.50 en max-prob. Un seuil > 0.50 peut bloquer toutes les entrées. Vérifier les logs `probas D BUY=X SELL=Y HOLD=Z` du backtest pour calibrer.

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
ATR(14) → calculé sur les 14 dernières bougies M1
SL = entry_price ∓ 1.2 × ATR     (signe selon LONG/SHORT)
TP = entry_price ± 2.4 × ATR × 0.7   (tp_shrink réaliste)

→ R:R brut = 2.0, effectif = 1.4 (après tp_shrink)
```

### Break-even + Trailing

```
Si mouvement favorable ≥ 1.0 × ATR :
   SL déplacé à entry_price (break-even, position garantie 0)

Si mouvement favorable ≥ 1.5 × ATR :
   SL trailé à 1.0 × ATR derrière le prix max favorable
   (s'améliore à chaque nouveau high/low favorable)
```

**État par contexte (mai 2026)** :
| Contexte | BE/Trail actif ? |
|----------|------------------|
| Training (env) | ✅ Oui (le modèle apprend avec) |
| Backtest stress-test original | ✅ Oui |
| Backtest **no_be_trail** | ❌ Non — variante de comparaison |
| **Live (loup_live.py)** | ❌ **Non** (désactivé après constat backtest no_be_trail : PF 0.98 → 1.70) |

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

- **Aucun data leak** : VAL et TEST sont toujours dans le futur du TRAIN du même fold
- **Couverture 100%** : l'union des 3 folds couvre toute la data
- **Régimes diversifiés** : chaque fold contient un mix bull/bear/sideways différent
- **Test du wf3 ≈ test out-of-sample récent** : wf3 termine en 2026-05, le plus pertinent pour le live

### Sélection du modèle live

Pour le live, recommandation par ordre de priorité :
1. **`bestprofit_*_wf3`** : train le plus récent, le plus proche du marché actuel
2. **`bestprofit_*_wf2`** : intermédiaire
3. **`bestprofit_*_wf1`** : train le plus ancien, le moins pertinent en live

Ou bien : exécuter les 3 en parallèle dans 3 instances MT5 et faire de l'ensembling.

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

### Panneau de contrôle
- **Lot size** (slider 0.01 → 1.00)
- **Démarrer / Arrêter l'IA**
- **Statut** : EN COURS / ARRÊTÉ avec side et lot
- **Equity bar** : Equity / Balance / P&L flottant / Margin / Free margin

### Stats de session
Grille temps réel avec 3 lignes :
- 🟢 **LONG** : trades, wins, losses, WR, PF, PnL, AvgW, AvgL
- 🔵 **SHORT** : idem
- ⚪ **TOTAL** : agrégé

Source : `mt5.history_deals_get(session_start, now, group="BTCUSD")` filtré sur les deals de sortie. **Bouton 🔄 Reset stats** pour réinitialiser.

### Zone de logs
- **Couleurs ANSI** par catégorie (TRADE / SIGNAL / INFO / WARN / ERROR / DEBUG)
- **Classification automatique** via regex (mots-clés `buy`, `sell`, `error`, etc.)
- **Filtre texte** en temps réel
- **Checkboxes** pour masquer/afficher chaque catégorie
- **Auto-scroll** désactivable
- **🗑️ Effacer** + **💾 Sauver** (export `.txt` horodaté)
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

### "Modèle DUEL introuvable : bestprofit_saintv2_loup_duel_*.pth"
- Le path dans `loup_live.py` / backtest doit correspondre au pattern réel :
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
- [ ] **Order book features** : si broker fournit DOM via MT5, intégrer top-5 bid/ask
- [ ] **Funding rate awareness** : pour les futures perpétuels
- [ ] **Distillation** : compresser le SAINTv2 en MLP pour inférence < 1ms

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

**Loup Ω** — _PPO + SAINTv2 pour BTCUSD M1_ — Made with ❤️ and a lot of GPUs.
