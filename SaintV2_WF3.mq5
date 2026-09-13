//+------------------------------------------------------------------+
//|                                                  SaintV2_WF3.mq5 |
//|                                          Loup Omega — WF3 single |
//|                                                                  |
//| Replique loup_live.py / backtest_saintv2_no_be_trail.py en MQL5 |
//| pour Strategy Tester MT5.                                        |
//|                                                                  |
//| FICHIERS REQUIS (UseCommonFolder=true) :                         |
//|   <Terminal Common>\Files\                                       |
//|     - saintv2_wf1.onnx        (généré par export_to_onnx.py)     |
//|     - norm_stats_mean.bin     (15 doubles little-endian)         |
//|     - norm_stats_std.bin      (15 doubles little-endian)         |
//|                                                                  |
//| v2.0 : bascule BTCUSD -> XAUUSD. 15 features.                    |
//|                                                                  |
//| PLUS AUCUN FICHIER DE DONNEES EXTERNE                            |
//| Les 4 features d'argent viennent de XAGUSD, lu directement par   |
//| CopyClose depuis le MEME serveur MT5 que l'or : meme horodatage, |
//| meme fuseau. Le probleme du WebRequest() desactive dans le       |
//| Strategy Tester disparait, ainsi que le binaire pre-exporte, le  |
//| cache et la detection du decalage horaire qu'imposait Binance.   |
//+------------------------------------------------------------------+

#property strict
#property copyright "Loup Omega"
#property version   "2.0"
#property description "SAINTv2 WF1 XAUUSD — 15 features (prix + argent + spread + heure)"

#include <Trade\Trade.mqh>

// =====================================================================
// PARAMETRES UTILISATEUR
// =====================================================================
input group "Modèle ONNX"
input string  ModelFile       = "saintv2_wf1.onnx";       // .onnx file (doit matcher AGENT de export_to_onnx.py)
input string  MeanFile        = "norm_stats_mean.bin";    // mean (N_BASE_FEATURES doubles)
input string  StdFile         = "norm_stats_std.bin";     // std (N_BASE_FEATURES doubles)
input string  SymbolXag      = "XAGUSD";                  // source des 4 features d'argent
input bool    UseCommonFolder = true;                     // true = Terminal\Common\Files\ (REQUIS pour Tester)

input group "Identification"
input ulong   Magic          = 424241;                    // magic (424241=WF1, 424242=WF2, 424243=WF3)
input string  Comment_       = "SAINTv2_WF1";             // commentaire ordres

input group "Risk / volume"
input bool    RiskVolume     = true;                      // lot par le RISQUE (prioritaire)
input double  RiskPerTrade   = 0.012;                     // 1.2 % de l'equity par trade
input bool    DynamicVolume  = false;                     // paliers d'equity (obsolète)
input double  FixedLot       = 0.10;                      // utilisé si ni Risk ni Dynamic
input double  MaxLot         = 100.0;                     // cap
input double  MarginSafety   = 0.80;                      // n'utilise jamais > 80% margin_free

input group "SL / TP"
// SL/TP mesures SUR L'OR (barriere triple, spread inclus, non-resolus clotures
// au marche, 1.32 M bougies). La ligne 5xATR est la seule positive sur ses
// QUATRE ratios (+0.132 / +0.027 / +0.093 / +0.317) — c'est ce qui fait preuve,
// pas la case maximale. Le 10xATR est elimine : negatif partout et seulement
// 63 % de resolution, l'or n'ayant pas assez d'amplitude pour des barrieres
// aussi lointaines. Doit rester aligne avec training.PPOConfig.
input double  AtrSlMult      = 5.0;                       // SL = 5 × ATR
input double  AtrTpMult      = 10.0;                      // TP = 10 × ATR (R:R 1:2.0)
input double  TpShrink       = 1.0;                       // pas de shrink
input int     AtrPeriod      = 14;
input int     RsiPeriod      = 14;

input group "Inférence"
input int     Lookback       = 25;                        // bougies M1 en obs
// Barres de conviction CALIBRÉES par le training, UNE PAR CÔTÉ. Elles
// réalisent la sélectivité visée (« trader les q % d'instants les plus
// favorables »). Valeurs écrites dans <checkpoint>_calib.json, champs
// calib_thr_buy / calib_thr_sell — les recopier ici à chaque nouveau modèle.
//
// Deux barres et non une : une barre unique sur max(p_BUY, p_SELL) dégénère,
// un écart de l'ordre du millième entre les côtés suffisant à verrouiller la
// sélection sur un seul (mesuré : 868 trades SHORT et zéro LONG à une epoch,
// l'inverse à une autre).
input double  ConfBuy        = 0.40;                      // ← calib_thr_buy
input double  ConfSell       = 0.40;                      // ← calib_thr_sell

// =====================================================================
// CONSTANTES (alignées avec training/loup_live)
// =====================================================================
#define ATR_PLANCHER_FRAC 0.0001   // 0.01 % du prix ~ un spread
#define N_BASE_FEATURES   15
#define N_POS_FEATURES    4
#define OBS_N_FEATURES    19
#define N_ACTIONS         3
#define CLIP_SIGMA        5.0
#define VOL_20_PERIOD     20
#define VOL_RANK_PERIOD   1440
#define MOM_5_PERIOD      5
#define EMA_DEV_SPAN      60
#define XAG_EMA_SPAN      60
#define RATIO_EMA_SPAN    240

// Indices features — DOIVENT reproduire saint_core.FEATURE_COLS dans l'ordre :
//   FEATURE_COLS_M1 + _H1 + _XAG + _LIQ + _TEMPS
//
// Jeu choisi sur mesure d'apport hors echantillon, cible = issue reelle du
// trade a SL 5xATR / R:R 2.0, sur 1.32 M bougies :
//   prix seules (8)              AUC 0.5313
//   + argent + spread (13)       AUC 0.5309
//   + heure (15)                 AUC 0.5327   <- ce jeu
//
// L'indice dollar (USDX) a ete ECARTE : apport exactement nul, et il coutait
// 8 % des bougies disponibles.
#define IDX_CLOSE_EMA_DEV  0
#define IDX_RETURNS        1
#define IDX_RANGE_NORM     2
#define IDX_MOM5           3
#define IDX_HIGH_VOL       4
#define IDX_CLOSE_H1_DEV   5
#define IDX_RSI_H1         6
#define IDX_RETURNS_H1     7
// Argent — l'or et l'argent partagent leurs moteurs. Leur correlation
// contemporaine de +0.722 ne vaut rien en soi ; c'est l'apport marginal qui
// compte (+0.0023 pour le groupe).
#define IDX_XAG_RET        8
#define IDX_XAG_RET5       9
#define IDX_XAG_DEV       10
#define IDX_RATIO_XAU_XAG 11
// Liquidite — feature la plus contributrice en apport individuel (+0.0025).
#define IDX_SPREAD_REL    12
// Temps — legitime sur l'or : sa volatilite varie de 2.43x entre la meilleure
// et la pire heure (contre ~1.5x sur BTC). Phase encodee en sin/cos pour que
// 23 h et 0 h soient voisins.
#define IDX_HEURE_SIN     13
#define IDX_HEURE_COS     14

// =====================================================================
// GLOBALS
// =====================================================================
long      g_onnx_handle      = INVALID_HANDLE;
double    g_mean[N_BASE_FEATURES];
double    g_std[N_BASE_FEATURES];
datetime  g_last_bar_time    = 0;
CTrade    g_trade;

int       g_xag_absent       = 0;   // bougies sans donnee argent, pour log

// =====================================================================
// INIT / DEINIT
// =====================================================================
int OnInit()
{
   uint onnx_flag = UseCommonFolder ? ONNX_COMMON_FOLDER : ONNX_DEFAULT;
   g_onnx_handle  = OnnxCreate(ModelFile, onnx_flag);
   if(g_onnx_handle == INVALID_HANDLE)
   {
      string where = UseCommonFolder ? "Terminal\\Common\\Files\\" : "MQL5\\Files\\";
      PrintFormat("[INIT] Erreur OnnxCreate : %d — vérifier que %s est dans %s",
                  GetLastError(), ModelFile, where);
      return INIT_FAILED;
   }

   const long input_shape[] = {1, Lookback, OBS_N_FEATURES};
   if(!OnnxSetInputShape(g_onnx_handle, 0, input_shape))
   {
      Print("[INIT] OnnxSetInputShape erreur : ", GetLastError());
      return INIT_FAILED;
   }
   const long logits_shape[] = {1, N_ACTIONS};
   if(!OnnxSetOutputShape(g_onnx_handle, 0, logits_shape))
   {
      Print("[INIT] OnnxSetOutputShape(logits) erreur : ", GetLastError());
      return INIT_FAILED;
   }
   const long value_shape[] = {1};
   OnnxSetOutputShape(g_onnx_handle, 1, value_shape);

   if(!LoadDoubleFile(MeanFile, g_mean, N_BASE_FEATURES))
   {
      Print("[INIT] Erreur lecture ", MeanFile);
      return INIT_FAILED;
   }
   if(!LoadDoubleFile(StdFile, g_std, N_BASE_FEATURES))
   {
      Print("[INIT] Erreur lecture ", StdFile);
      return INIT_FAILED;
   }

   // Quatre des quinze features viennent de l'argent. Sans lui, le modele
   // deciderait sur un vecteur qu'il n'a jamais vu a l'entrainement.
   if(!SymbolSelect(SymbolXag, true))
   {
      PrintFormat("[INIT] %s indisponible chez ce broker — demarrage refuse.",
                  SymbolXag);
      return INIT_FAILED;
   }

   g_trade.SetExpertMagicNumber(Magic);
   g_trade.SetDeviationInPoints(50);
   g_trade.SetTypeFillingBySymbol(_Symbol);

   PrintFormat("[INIT] SaintV2 v2.0 OK | %s + %s | model=%s | magic=%d | "
               "lookback=%d | features=%d | folder=%s",
               _Symbol, SymbolXag, ModelFile, (int)Magic, Lookback,
               N_BASE_FEATURES, UseCommonFolder ? "Common" : "MQL5\\Files");
   PrintFormat("[INIT] PARAMS  SL=%.2f×ATR  TP=%.2f×ATR×%.2f  "
               "barres BUY=%.3f SELL=%.3f",
               AtrSlMult, AtrTpMult, TpShrink, ConfBuy, ConfSell);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(g_onnx_handle != INVALID_HANDLE)
   {
      OnnxRelease(g_onnx_handle);
      g_onnx_handle = INVALID_HANDLE;
   }
   if(g_xag_absent > 0)
      PrintFormat("[ARGENT] %d barres sans cotation %s — verifier que "
                  "l'historique du symbole couvre la periode testee.",
                  g_xag_absent, SymbolXag);
}

// =====================================================================
// FILE LOADER
// =====================================================================
bool LoadDoubleFile(const string filename, double &arr[], int n)
{
   int flags = FILE_READ | FILE_BIN;
   if(UseCommonFolder) flags |= FILE_COMMON;
   int h = FileOpen(filename, flags);
   if(h == INVALID_HANDLE && UseCommonFolder)
      h = FileOpen(filename, FILE_READ | FILE_BIN);
   if(h == INVALID_HANDLE)
   {
      PrintFormat("[FILE] Impossible d'ouvrir %s (err=%d)", filename, GetLastError());
      return false;
   }
   for(int i=0; i<n; i++)
      arr[i] = FileReadDouble(h);
   FileClose(h);
   return true;
}

// =====================================================================
// FETCH OHLC SERIES
// =====================================================================
bool GetClosesTF(ENUM_TIMEFRAMES tf, int shift, int count, double &out[])
{
   ArrayResize(out, count);
   double tmp[];
   ArraySetAsSeries(tmp, true);
   if(CopyClose(_Symbol, tf, shift, count, tmp) != count) return false;
   for(int i=0; i<count; i++) out[i] = tmp[i];
   return true;
}
bool GetHighLowsTF(ENUM_TIMEFRAMES tf, int shift, int count, double &highs[], double &lows[])
{
   ArrayResize(highs, count); ArrayResize(lows, count);
   double th[], tl[];
   ArraySetAsSeries(th, true); ArraySetAsSeries(tl, true);
   if(CopyHigh(_Symbol, tf, shift, count, th) != count) return false;
   if(CopyLow (_Symbol, tf, shift, count, tl) != count) return false;
   for(int i=0; i<count; i++) { highs[i]=th[i]; lows[i]=tl[i]; }
   return true;
}

// =====================================================================
// INDICATEURS CUSTOM — IDENTIQUES AU PYTHON (training.py)
// =====================================================================

// RSI SMA (gains/losses → moyenne simple, comme Python)
void PrecomputeSmaRsi(const double &closes[], int n, int period, double &out[])
{
   ArrayResize(out, n);
   for(int idx = 0; idx < n - period; idx++)
   {
      double gain_sum = 0.0, loss_sum = 0.0;
      for(int k = 0; k < period; k++)
      {
         double d = closes[idx + k] - closes[idx + k + 1];
         if(d > 0) gain_sum += d;
         else      loss_sum -= d;
      }
      double avg_g = gain_sum / period;
      double avg_l = loss_sum / period;
      double rs = avg_g / (avg_l + 1e-8);
      out[idx] = 100.0 - 100.0 / (1.0 + rs);
   }
   for(int idx = n - period; idx < n; idx++) out[idx] = 50.0;
}

// ATR SMA (TR → moyenne simple, comme Python)
void PrecomputeSmaAtr(const double &highs[], const double &lows[], const double &closes[],
                     int n, int period, double &out[])
{
   ArrayResize(out, n);
   double tr[];
   ArrayResize(tr, n);
   for(int k = 0; k < n - 1; k++)
   {
      double h  = highs[k];
      double l  = lows[k];
      double cp = closes[k + 1];
      double tr1 = h - l;
      double tr2 = MathAbs(h - cp);
      double tr3 = MathAbs(l - cp);
      tr[k] = MathMax(tr1, MathMax(tr2, tr3));
   }
   tr[n-1] = 0.0;
   for(int idx = 0; idx < n - period; idx++)
   {
      double sum = 0.0;
      for(int k = 0; k < period; k++) sum += tr[idx + k];
      out[idx] = sum / period;
   }
   for(int idx = n - period; idx < n; idx++) out[idx] = 0.0;
}

// EMA récursive, équivalente à pandas .ewm(span, adjust=False).mean().
// Les tableaux sont en série (0 = plus récent), donc on remonte du plus ancien
// vers le plus récent, en amorçant sur la valeur la plus ancienne comme pandas.
void PrecomputeEma(const double &values[], int n, int span, double &out[])
{
   ArrayResize(out, n);
   if(n <= 0) return;
   double alpha = 2.0 / (span + 1.0);
   out[n-1] = values[n-1];
   for(int i = n - 2; i >= 0; i--)
      out[i] = alpha * values[i] + (1.0 - alpha) * out[i+1];
}

// Std avec ddof=1 (échantillon, comme pandas par défaut)
void PrecomputeStdDDOF1(const double &values[], int n, int period, double &out[])
{
   ArrayResize(out, n);
   for(int idx = 0; idx < n - period; idx++)
   {
      double sum = 0.0;
      for(int k = 0; k < period; k++) sum += values[idx + k];
      double mean = sum / period;
      double sumsq = 0.0;
      for(int k = 0; k < period; k++)
      {
         double d = values[idx + k] - mean;
         sumsq += d * d;
      }
      out[idx] = (period > 1) ? MathSqrt(sumsq / (period - 1)) : 0.0;
   }
   for(int idx = n - period; idx < n; idx++) out[idx] = 0.0;
}

// =====================================================================
// FEATURES D'ARGENT — lues directement depuis XAGUSD
//
// Aucun fichier, aucune API : XAGUSD vient du MEME serveur MT5 que l'or, donc
// meme horodatage et meme fuseau. Le probleme du WebRequest() desactive dans le
// Strategy Tester, qui imposait un binaire pre-exporte pour Binance, disparait.
//
// On charge une fenetre de minutes indexee par decalage depuis t_start, avec
// report de la derniere valeur connue sur les trous (l'argent et l'or peuvent
// ne pas avoir exactement les memes bougies). L'indexation est alors en O(1),
// comme cote Python ou la jointure est exacte a la minute.
// =====================================================================
bool ChargeArgent(datetime t_start, datetime t_fin,
                  datetime &o_time[], double &o_close[], double &o_ema[])
{
   MqlRates r[];
   ArraySetAsSeries(r, false);          // index 0 = plus ANCIEN
   int n = CopyRates(SymbolXag, PERIOD_M1, t_start, t_fin, r);
   if(n <= 60)
   {
      PrintFormat("[ARGENT] CopyRates %s a renvoye %d (err=%d)",
                  SymbolXag, n, GetLastError());
      return false;
   }
   ArrayResize(o_time, n);
   ArrayResize(o_close, n);
   ArrayResize(o_ema, n);

   // EMA recursive sur la SUITE DES BOUGIES, et non sur une grille de minutes.
   //
   // C'est le point critique, mesure : pandas .ewm() avance d'un pas par LIGNE
   // du dataframe, en ignorant les trous de temps. L'or ferme le week-end —
   // la ou pandas voit un seul pas entre vendredi soir et lundi matin, une
   // grille de minutes en insererait 2880, faisant decroitre l'EMA vers la
   // derniere valeur de vendredi. L'ecart mesure atteignait 38 % de
   // l'ecart-type de ratio_xau_xag, et aucune taille de fenetre ne le corrige.
   double a = 2.0 / (XAG_EMA_SPAN + 1.0);
   for(int i = 0; i < n; i++)
   {
      o_time[i]  = r[i].time;
      o_close[i] = r[i].close;
      o_ema[i]   = (i == 0) ? r[i].close
                            : a * r[i].close + (1.0 - a) * o_ema[i-1];
   }
   return true;
}

// Index de la bougie d'argent a l'horodatage EXACT `t`, ou -1.
// Recherche dichotomique : o_time est trie par construction.
int IndexArgent(const datetime &o_time[], int n, datetime t)
{
   int bas = 0, haut = n - 1;
   while(bas <= haut)
   {
      int mid = (bas + haut) / 2;
      if(o_time[mid] == t)     return mid;
      if(o_time[mid] <  t)     bas = mid + 1;
      else                     haut = mid - 1;
   }
   return -1;
}

// =====================================================================
// COMPUTE BASE FEATURES (Lookback x 15)
//   8 de prix/H1 + 4 d'argent + 1 de spread + 2 d'heure
// =====================================================================
bool ComputeBaseFeatures(double &features[])
{
   int need_m1 = Lookback + VOL_RANK_PERIOD + VOL_20_PERIOD + RsiPeriod + 5;
   double closes_m1[], highs_m1[], lows_m1[];
   if(!GetClosesTF (PERIOD_M1, 1, need_m1, closes_m1)) return false;
   if(!GetHighLowsTF(PERIOD_M1, 1, need_m1, highs_m1, lows_m1)) return false;

   double returns_m1[];
   ArrayResize(returns_m1, need_m1);
   for(int i=0; i<need_m1-1; i++)
   {
      double c  = closes_m1[i];
      double cp = closes_m1[i+1];
      returns_m1[i] = (cp > 1e-12) ? (c - cp)/cp : 0.0;
   }
   returns_m1[need_m1-1] = 0.0;

   double vol20_m1[];
   PrecomputeStdDDOF1(returns_m1, need_m1, VOL_20_PERIOD, vol20_m1);

   double ema_m1[];
   PrecomputeEma(closes_m1, need_m1, EMA_DEV_SPAN, ema_m1);

   // ATR par bougie : sert a spread_rel, qui rapporte le spread courant a la
   // volatilite du moment. C'est la feature la plus contributrice du jeu.
   double atr_m1[];
   PrecomputeSmaAtr(highs_m1, lows_m1, closes_m1, need_m1, AtrPeriod, atr_m1);

   // Seul le RSI H1 reste utilisé côté H1 : range_norm_h1 et vol_20_h1 avaient
   // une AUC de 0.501 (aucune information) et ont été retirés du jeu.
   int need_h1 = 200 + RsiPeriod + VOL_20_PERIOD;
   double closes_h1[];
   if(!GetClosesTF(PERIOD_H1, 1, need_h1, closes_h1)) return false;

   double rsi_h1[];
   PrecomputeSmaRsi(closes_h1, need_h1, RsiPeriod, rsi_h1);

   double returns_h1[];
   ArrayResize(returns_h1, need_h1);
   for(int i=0; i<need_h1-1; i++)
   {
      double c  = closes_h1[i];
      double cp = closes_h1[i+1];
      returns_h1[i] = (cp > 1e-12) ? (c - cp)/cp : 0.0;
   }
   returns_h1[need_h1-1] = 0.0;

   // ---- ARGENT ----
   // Fenetre large : l'EMA du ratio (span 240) a besoin d'environ 5 x span pour
   // converger. Mesure : a 260 min d'amorce l'ecart atteignait 1.13e-03 sur une
   // feature d'ecart-type 3.0e-03 ; a 1200 min il tombe a 1.42e-04.
   datetime t_plus_vieux = iTime(_Symbol, PERIOD_M1, Lookback);
   datetime t_start_xag  = t_plus_vieux - (5 * RATIO_EMA_SPAN + 60) * 60;
   datetime t_fin_xag    = iTime(_Symbol, PERIOD_M1, 0) + 60;

   datetime xag_time[];
   double   xag_close[], xag_ema[];
   if(!ChargeArgent(t_start_xag, t_fin_xag, xag_time, xag_close, xag_ema))
      return false;
   int n_xag = ArraySize(xag_time);

   // EMA du ratio or/argent sur la suite des bougies D'OR — c'est sur elles que
   // pandas la calcule (le ratio vit dans le dataframe fusionne, indexe par les
   // bougies d'or). On remonte du plus ancien vers le plus recent.
   int n_ratio = MathMin(need_m1, 5 * RATIO_EMA_SPAN + 60);
   double ratio_ema[], ratio_val[];
   ArrayResize(ratio_ema, n_ratio);
   ArrayResize(ratio_val, n_ratio);
   double a_r = 2.0 / (RATIO_EMA_SPAN + 1.0);
   bool ratio_ok = false;
   for(int k = n_ratio - 1; k >= 0; k--)      // k = shift, donc a l'envers
   {
      datetime tk = iTime(_Symbol, PERIOD_M1, k);
      int ix = IndexArgent(xag_time, n_xag, tk);
      double rv = (ix >= 0 && xag_close[ix] > 1e-12)
                  ? closes_m1[k] / xag_close[ix] : 0.0;
      if(rv <= 0.0)
      {
         // Pas de cotation argent : on reporte la derniere valeur connue
         // plutot que d'injecter un zero qui casserait l'EMA.
         rv = (k < n_ratio - 1) ? ratio_val[k + 1] : 0.0;
      }
      ratio_val[k] = rv;
      if(!ratio_ok) { ratio_ema[k] = rv; ratio_ok = (rv > 0.0); }
      else          ratio_ema[k] = a_r * rv + (1.0 - a_r) * ratio_ema[k + 1];
   }

   double point_or = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   ArrayResize(features, Lookback * N_BASE_FEATURES);

   for(int t=0; t<Lookback; t++)
   {
      int shift = Lookback - 1 - t;
      double hi  = highs_m1[shift];
      double lo  = lows_m1[shift];
      double c   = closes_m1[shift];
      double ret = returns_m1[shift];
      double v20 = vol20_m1[shift];
      double range_norm = (c > 1e-12) ? (hi - lo) / c : 0.0;

      double mom5 = 0.0;
      if(shift + MOM_5_PERIOD < need_m1)
         mom5 = (c > closes_m1[shift + MOM_5_PERIOD]) ? 1.0 : 0.0;

      double vol_rank = 0.0;
      if(shift + VOL_RANK_PERIOD < need_m1)
      {
         int rank_lt = 0;
         for(int k=0; k<VOL_RANK_PERIOD; k++)
            if(vol20_m1[shift + k] < v20) rank_lt++;
         vol_rank = (double)rank_lt / (double)VOL_RANK_PERIOD;
      }
      double high_vol = (vol_rank > 0.65) ? 1.0 : 0.0;

      // ALIGNEMENT H1 — ne pas "corriger" ce décalage, il est voulu.
      // closes_h1[] est rempli depuis CopyClose(..., shift=1, ...) donc
      // closes_h1[i] = bar H1 au shift (i+1). iBarShift renvoie le shift du bar
      // qui CONTIENT m1_time (bar en formation). Indexer closes_h1[h1_shift]
      // donne donc le bar H1 PRÉCÉDENT = le dernier H1 réellement clôturé,
      // exactement ce que produit le `df_h1.shift(1)` côté Python.
      datetime m1_time = iTime(_Symbol, PERIOD_M1, shift + 1);
      int h1_shift = iBarShift(_Symbol, PERIOD_H1, m1_time, false);
      if(h1_shift < 0) h1_shift = 0;
      if(h1_shift >= need_h1) h1_shift = need_h1 - 1;

      double close_h1   = closes_h1[h1_shift];
      double rsi_h1_val = rsi_h1[h1_shift];
      double ret_h1     = returns_h1[h1_shift];

      double inv_c = (c > 1e-12) ? 1.0 / c : 0.0;
      double ema_c = ema_m1[shift];

      // ---- ARGENT, SPREAD, HEURE pour CETTE bougie ----
      // Meme horodatage que le bar M1 des features de prix, donc la meme
      // minute que la jointure EXACTE faite cote Python.
      int ix = IndexArgent(xag_time, n_xag, m1_time);
      if(ix < 1)
      {
         g_xag_absent++;
         PrintFormat("[ARGENT] Pas de cotation %s pour %s — barre ignoree.",
                     SymbolXag, TimeToString(m1_time, TIME_DATE|TIME_MINUTES));
         return false;
      }
      double xc  = xag_close[ix];
      double xcp = xag_close[ix - 1];
      double xc5 = xag_close[MathMax(ix - 5, 0)];
      double xag_ret  = (xcp > 1e-12) ? (xc - xcp) / xcp : 0.0;
      double xag_ret5 = (xc5 > 1e-12) ? (xc - xc5) / xc5 : 0.0;
      double xag_dev  = (xag_ema[ix] > 1e-12) ? xc / xag_ema[ix] - 1.0 : 0.0;

      int kr = shift + 1;                    // meme bougie, en index shift
      double ratio_dev = (kr < n_ratio && ratio_ema[kr] > 1e-12)
                         ? ratio_val[kr] / ratio_ema[kr] - 1.0 : 0.0;

      // Spread de CETTE bougie, converti en prix puis rapporte a l'ATR —
      // exactement comme cote Python, ou la colonne `spread` des bougies MT5
      // est en points entiers.
      double spr_pts = (double)iSpread(_Symbol, PERIOD_M1, shift + 1);
      double atr_bar = atr_m1[shift];
      double spread_rel = (atr_bar > 1e-12)
                          ? (spr_pts * point_or) / atr_bar : 0.0;

      // Phase du jour en sin/cos : 23 h et 0 h doivent etre voisins.
      MqlDateTime st;
      TimeToStruct(m1_time, st);
      double hh = st.hour + st.min / 60.0;
      double heure_sin = MathSin(2.0 * M_PI * hh / 24.0);
      double heure_cos = MathCos(2.0 * M_PI * hh / 24.0);

      int base = t * N_BASE_FEATURES;
      features[base + IDX_CLOSE_EMA_DEV] = (ema_c > 1e-12) ? (c / ema_c - 1.0) : 0.0;
      features[base + IDX_RETURNS]       = ret;
      features[base + IDX_RANGE_NORM]    = range_norm;
      features[base + IDX_MOM5]          = mom5;
      features[base + IDX_HIGH_VOL]      = high_vol;
      features[base + IDX_CLOSE_H1_DEV]  = close_h1 * inv_c - 1.0;
      features[base + IDX_RSI_H1]        = rsi_h1_val;
      features[base + IDX_RETURNS_H1]    = ret_h1;
      features[base + IDX_XAG_RET]       = xag_ret;
      features[base + IDX_XAG_RET5]      = xag_ret5;
      features[base + IDX_XAG_DEV]       = xag_dev;
      features[base + IDX_RATIO_XAU_XAG] = ratio_dev;
      features[base + IDX_SPREAD_REL]    = spread_rel;
      features[base + IDX_HEURE_SIN]     = heure_sin;
      features[base + IDX_HEURE_COS]     = heure_cos;
   }
   return true;
}

// =====================================================================
// BUILD OBSERVATION
// =====================================================================
bool BuildObservation(float &obs[])
{
   double feats[];
   if(!ComputeBaseFeatures(feats))
   {
      Print("[OBS] ComputeBaseFeatures a échoué");
      return false;
   }
   ArrayResize(obs, Lookback * OBS_N_FEATURES);
   double pos_feat[N_POS_FEATURES] = {0.0, 0.0, 0.0, 1.0};

   for(int t=0; t<Lookback; t++)
   {
      int base_in  = t * N_BASE_FEATURES;
      int base_out = t * OBS_N_FEATURES;
      for(int k=0; k<N_BASE_FEATURES; k++)
      {
         double raw = feats[base_in + k];
         double s   = (g_std[k] > 1e-8) ? g_std[k] : 1.0;
         double z   = (raw - g_mean[k]) / s;
         if(z >  CLIP_SIGMA) z =  CLIP_SIGMA;
         if(z < -CLIP_SIGMA) z = -CLIP_SIGMA;
         obs[base_out + k] = (float)z;
      }
      for(int k=0; k<N_POS_FEATURES; k++)
         obs[base_out + N_BASE_FEATURES + k] = (float)pos_feat[k];
   }
   return true;
}

// =====================================================================
// INFÉRENCE
// =====================================================================
int RunModel(const float &obs[], double &out_probas[])
{
   float logits_out[N_ACTIONS];
   float value_out[1];
   if(!OnnxRun(g_onnx_handle, ONNX_DEFAULT, obs, logits_out, value_out))
   {
      Print("[ONNX] OnnxRun erreur : ", GetLastError());
      return 2;
   }
   double maxl = logits_out[0];
   for(int k=1; k<N_ACTIONS; k++) if(logits_out[k] > maxl) maxl = logits_out[k];
   double s = 0;
   double e[N_ACTIONS];
   for(int k=0; k<N_ACTIONS; k++) { e[k] = MathExp(logits_out[k] - maxl); s += e[k]; }
   ArrayResize(out_probas, N_ACTIONS);
   for(int k=0; k<N_ACTIONS; k++) out_probas[k] = e[k] / s;

   // UNE BARRE PAR CÔTÉ — réplique saint_core.decide_avec_barres().
   //
   // Pas d'argmax sur les trois actions : il exigerait p(BUY) > p(HOLD), ce
   // qu'une stratégie sélective ne vérifie presque jamais (mesuré : 0 trade en
   // validation, même avec un seuil nul). Pas non plus de comparaison directe
   // entre p_BUY et p_SELL : elle verrouille la sélection sur un seul côté.
   // Chaque côté est jugé sur SA barre ; à égalité, le plus net l'emporte.
   double pb = out_probas[0], ps = out_probas[1];
   bool ok_b = (pb >= ConfBuy);
   bool ok_s = (ps >= ConfSell);
   if(ok_b && ok_s) return ((pb - ConfBuy) >= (ps - ConfSell)) ? 0 : 1;
   if(ok_b) return 0;
   if(ok_s) return 1;
   return 2;   // aucun côté au-dessus de sa barre
}

// =====================================================================
// ATR pour SL/TP — DERNIER ATR(14) seul (aligné backtest no_be_trail)
// Backtest Python : df_closed["atr_14"].iloc[-1]
// =====================================================================
double ComputeEntryAtr()
{
   int need = AtrPeriod + 5;
   double closes[], highs[], lows[];
   if(!GetClosesTF (PERIOD_M1, 1, need, closes)) return 0.0;
   if(!GetHighLowsTF(PERIOD_M1, 1, need, highs, lows)) return 0.0;

   double atr_arr[];
   PrecomputeSmaAtr(highs, lows, closes, need, AtrPeriod, atr_arr);

   return atr_arr[0];  // dernière valeur (shift=1, la bougie qui vient de fermer)
}

// =====================================================================
// VOLUME / MARGE / SL-TP
// =====================================================================
//+------------------------------------------------------------------+
//| Volume tel qu'un stop touché coûte exactement RiskPerTrade.        |
//|                                                                  |
//| MEME règle que l'environnement d'entraînement (training.py,        |
//| PPOEnv._compute_dynamic_size) et que loup_live.py. Elle doit      |
//| l'être : le modèle a appris sous une loi de taille donnée, et le   |
//| déployer sous une autre revient à jouer un risque que rien n'a     |
//| validé.                                                           |
//|                                                                  |
//| L'ancienne règle était un escalier sur l'equity qui ne regardait   |
//| ni le prix ni la volatilité : le stop étant posé à 5xATR, le       |
//| risque réel suivait l'ATR, de 2.5 % à 10 % du capital sans borne.  |
//|                                                                  |
//| La conversion passe par TICK_VALUE / TICK_SIZE : c'est la seule   |
//| juste quand la devise de cotation n'est pas celle du compte.      |
//+------------------------------------------------------------------+
double ComputeRiskVolume(double equity, double sl_dist, bool &plancher_mordu)
{
   plancher_mordu = false;
   if(equity <= 0.0 || sl_dist <= 0.0) return 0.0;

   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

   // Valeur d'UNE unite de prix pour UN lot. tick_value / tick_size porte la
   // conversion de devise : mesure sur BTCUSD, 0.008621 / 0.01 = 0.8621, et
   // OrderCalcProfit confirme 86.23 pour 1 lot et +100 de prix — pas 1.0,
   // malgre un contract_size de 1.0.
   //
   // Repli sur contract_size : MT5 renvoie tick_value = 0 pour un symbole
   // absent du Market Watch. Sans lui, chaque ordre serait annule.
   double valeur_par_unite = 0.0;
   if(tick_value > 0.0 && tick_size > 0.0)
      valeur_par_unite = tick_value / tick_size;
   else
   {
      valeur_par_unite = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
      if(valeur_par_unite > 0.0)
         Print("[VOL] tick_value indisponible (symbole hors Market Watch ?) — "
               "repli sur contract_size, taille approximative");
   }
   if(valeur_par_unite <= 0.0) return 0.0;

   double perte_par_lot = sl_dist * valeur_par_unite;
   if(perte_par_lot <= 0.0) return 0.0;

   double vol_min  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vol_max  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double vol_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(vol_step <= 0.0) vol_step = 0.01;

   double brut = (equity * RiskPerTrade) / perte_par_lot;

   // Arrondi VERS LE BAS : un arrondi ne doit jamais augmenter le risque.
   double vol = MathFloor(brut / vol_step) * vol_step;
   if(vol > vol_max) vol = vol_max;
   if(vol > MaxLot)  vol = MaxLot;

   if(vol < vol_min)
   {
      // Seul cas où le contrôle du risque échoue : le minimum du courtier
      // impose de risquer PLUS que demandé. Doit être visible, pas silencieux.
      vol = vol_min;
      plancher_mordu = true;
   }

   int digits = (int)MathMax(0, MathRound(-MathLog10(vol_step)));
   return NormalizeDouble(vol, digits);
}

double ComputeDynamicVolume(double equity)
{
   if(equity <= 2000.0) return 0.10;
   int tier = (int)((equity - 1.0) / 1000.0);
   double lot = 0.10 * tier;
   if(lot > MaxLot) lot = MaxLot;
   return NormalizeDouble(lot, 2);
}

double AdjustVolumeToMargin(int side, double price, double desired_volume)
{
   double margin_free = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double required = 0;
   ENUM_ORDER_TYPE order_type = (side == 1) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   if(!OrderCalcMargin(order_type, _Symbol, desired_volume, price, required))
      return desired_volume;
   if(required <= 0) return desired_volume;

   double budget = margin_free * MarginSafety;
   if(required <= budget) return desired_volume;

   double margin_per_lot = required / desired_volume;
   if(margin_per_lot <= 0) return 0;
   double max_vol = budget / margin_per_lot;

   double vol_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(vol_step <= 0) vol_step = 0.01;
   double vol_min  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(vol_min  <= 0) vol_min  = 0.01;
   max_vol = MathFloor(max_vol / vol_step) * vol_step;
   if(max_vol < vol_min) return 0;
   return NormalizeDouble(max_vol, 2);
}

void ComputeSlTp(int side, double entry_price, double atr, double spread,
                 double &sl_out, double &tp_out)
{
   // ALIGNÉ avec backtest_saintv2_no_be_trail.compute_sl_tp() :
   // PAS de compensation spread (sl_dist=1.2×ATR, tp_dist=1.68×ATR pur).
   // Note: le paramètre `spread` est gardé pour compat mais ignoré ici.
   // Plancher d ATR : MEME valeur que saint_core.ATR_PLANCHER_FRAC.
   //
   // Il valait 0.0015, soit 0.15 % du prix. Mesure sur 2.74 M bougies d or, ce
   // plancher l emporte sur l ATR reel dans 99.4 % des cas, avec un rapport
   // median de 5.71x : un stop annonce a 5xATR aurait ete pose a 28.5xATR et un
   // TP a 10xATR a 57xATR. L EA n aurait rien trade qui ressemble a ce que le
   // modele a appris.
   //
   // A 0.01 % le plancher vaut environ UN SPREAD : il empeche seulement de
   // poser un stop a l interieur du spread, donc touche instantanement.
   double fallback = ATR_PLANCHER_FRAC * entry_price;
   double eff_atr = MathMax(atr, fallback);
   if(eff_atr < 1e-8) eff_atr = 1e-8;

   double sl_dist = AtrSlMult * eff_atr;
   double tp_dist = AtrTpMult * eff_atr * TpShrink;

   if(side == 1)
   {
      sl_out = entry_price - sl_dist;
      tp_out = entry_price + tp_dist;
   }
   else
   {
      sl_out = entry_price + sl_dist;
      tp_out = entry_price - tp_dist;
   }
   sl_out = NormalizeDouble(sl_out, _Digits);
   tp_out = NormalizeDouble(tp_out, _Digits);
}

bool HasPosition()
{
   int total = PositionsTotal();
   for(int i=0; i<total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionSelectByTicket(ticket))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == (long)Magic)
            return true;
      }
   }
   return false;
}

void SendOrder(int side, double atr)
{
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;

   double price  = (side == 1) ? tick.ask : tick.bid;
   double spread = tick.ask - tick.bid;
   if(spread < 0) spread = 0;

   // Le SL doit être connu AVANT la taille : c'est sa distance qui la fixe.
   double sl, tp;
   ComputeSlTp(side, price, atr, spread, sl, tp);

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double base_vol;
   if(RiskVolume)
   {
      bool plancher = false;
      double sl_dist = MathAbs(price - sl);
      base_vol = ComputeRiskVolume(equity, sl_dist, plancher);
      if(base_vol <= 0) { Print("[ORDER] volume par le risque = 0 — annulé"); return; }
      double risque_eff = 0.0;
      double tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      double vpu = (tv > 0.0 && ts > 0.0)
                   ? tv / ts
                   : SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
      if(vpu > 0 && equity > 0) risque_eff = base_vol * sl_dist * vpu / equity;
      if(plancher)
         PrintFormat("[VOL] equity=%.2f SL=%.2f → lot=%.2f  ⚠ VOLUME MINIMUM DU "
                     "COURTIER : risque imposé %.2f %% au lieu de %.2f %%",
                     equity, sl_dist, base_vol, 100*risque_eff, 100*RiskPerTrade);
      else
         PrintFormat("[VOL] equity=%.2f SL=%.2f → lot=%.2f (risque %.2f %%)",
                     equity, sl_dist, base_vol, 100*risque_eff);
   }
   else if(DynamicVolume)
      base_vol = ComputeDynamicVolume(equity);
   else
      base_vol = FixedLot;

   double vol = AdjustVolumeToMargin(side, price, base_vol);
   if(vol <= 0) { Print("[ORDER] vol=0 marge insuffisante"); return; }

   bool ok = (side == 1)
      ? g_trade.Buy (vol, _Symbol, price, sl, tp, Comment_)
      : g_trade.Sell(vol, _Symbol, price, sl, tp, Comment_);

   if(!ok)
   {
      PrintFormat("[ORDER] %s échoué : retcode=%d %s",
                  (side==1?"BUY":"SELL"), g_trade.ResultRetcode(),
                  g_trade.ResultRetcodeDescription());
   }
   else
   {
      PrintFormat("[ORDER] %s vol=%.2f @ %.2f  SL=%.2f  TP=%.2f  spread=%.2f  atr=%.2f",
                  (side==1?"BUY":"SELL"), vol, price, sl, tp, spread, atr);
   }
}

// =====================================================================
// ONTICK
// =====================================================================
void OnTick()
{
   datetime current_bar = iTime(_Symbol, PERIOD_M1, 0);
   if(current_bar == g_last_bar_time) return;
   g_last_bar_time = current_bar;

   if(HasPosition()) return;

   float obs[];
   if(!BuildObservation(obs)) return;

   double probas[];
   int action = RunModel(obs, probas);   // 0 = BUY, 1 = SELL, 2 = attendre
   if(action == 2) return;               // aucun côté au-dessus de sa barre

   double atr = ComputeEntryAtr();
   int side = (action == 0) ? 1 : -1;
   SendOrder(side, atr);
}
//+------------------------------------------------------------------+
