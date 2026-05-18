import sys
import io
import re
import datetime as dt
from pathlib import Path

from PySide6 import QtWidgets, QtCore, QtGui
import MetaTrader5 as mt5

from loup_live import LiveConfig, TradingAgent  # adapte le nom du fichier si besoin


# ============================================================
# Redirection des print vers la GUI
# ============================================================

class LogEmitter(QtCore.QObject):
    message = QtCore.Signal(str, str)  # (text, stream_name)


class QtLogStream(io.TextIOBase):
    def __init__(self, emitter: LogEmitter, stream_name: str):
        super().__init__()
        self.emitter = emitter
        self.stream_name = stream_name
        self._buffer = ""

    def write(self, msg: str):
        msg = str(msg)
        # On accumule pour bien découper en lignes complètes
        self._buffer += msg
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.emitter.message.emit(line.rstrip(), self.stream_name)
        return len(msg)

    def flush(self):
        if self._buffer.strip():
            self.emitter.message.emit(self._buffer.rstrip(), self.stream_name)
            self._buffer = ""


# ============================================================
# Classification des logs
# ============================================================

class LogLevel:
    ERROR  = "ERROR"
    WARN   = "WARN"
    TRADE  = "TRADE"
    SIGNAL = "SIGNAL"
    INFO   = "INFO"
    DEBUG  = "DEBUG"

LEVEL_COLORS = {
    LogLevel.ERROR:  "#ff5555",
    LogLevel.WARN:   "#f1c40f",
    LogLevel.TRADE:  "#2ecc71",
    LogLevel.SIGNAL: "#3498db",
    LogLevel.INFO:   "#dddddd",
    LogLevel.DEBUG:  "#888888",
}

# Patterns pour deviner le niveau à partir du texte
_PATTERNS = [
    (LogLevel.ERROR,  re.compile(r"\b(error|erreur|exception|traceback|fail|échec|failed)\b", re.I)),
    (LogLevel.WARN,   re.compile(r"\b(warn|warning|avertissement|attention)\b", re.I)),
    (LogLevel.TRADE,  re.compile(r"\b(buy|sell|close|open|position|order\s*sent|filled|deal|ouvert|fermé|clos[eé])\b", re.I)),
    (LogLevel.SIGNAL, re.compile(r"\b(signal|action|prob|argmax|best long|best short)\b", re.I)),
]

def classify(text: str, stream_name: str) -> str:
    if stream_name == "ERR":
        return LogLevel.ERROR
    for level, pat in _PATTERNS:
        if pat.search(text):
            return level
    return LogLevel.INFO


# ============================================================
# Fenêtre principale
# ============================================================

class MainWindow(QtWidgets.QMainWindow):

    MAX_LOG_BLOCKS = 5000  # ring buffer pour éviter de saturer la mémoire

    def __init__(self):
        super().__init__()

        # --- Config & Agent ---
        self.cfg = LiveConfig(side="both")
        self.agent = TradingAgent(self.cfg)

        self.setWindowTitle("Loup Ω – BTCUSD M1")
        self.setMinimumSize(900, 600)

        # Compteurs par niveau (affichage)
        self._counts = {lvl: 0 for lvl in LEVEL_COLORS}

        # Filtres actifs (niveau -> bool)
        self._level_filters = {lvl: True for lvl in LEVEL_COLORS}
        self._text_filter = ""

        # ---------- Redirection stdout/stderr ----------
        self.log_emitter = LogEmitter()
        self.log_emitter.message.connect(self.on_new_log)

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = QtLogStream(self.log_emitter, "OUT")
        sys.stderr = QtLogStream(self.log_emitter, "ERR")

        # ============ UI PRINCIPALE ============
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Titre
        title = QtWidgets.QLabel("🐺 Loup Ω – BTCUSD M1")
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Agent RL en live sur MT5 – Mode duel (LONG + SHORT), SL initial + break-even + trailing."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # --- Ligne info lot dynamique ---
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addWidget(QtWidgets.QLabel("Lot dynamique :"))

        # Label en lecture seule qui affiche le lot calculé selon l'equity
        # (0-2000$ → 0.01, +0.01 par tranche de 1000$, cap 100.00)
        self.lot_label = QtWidgets.QLabel("0.10 lot (auto)")
        self.lot_label.setStyleSheet(
            "color: #2ecc71; font-weight: bold; font-family: Consolas, monospace; padding: 2px 8px;"
            "background-color: #1c2833; border-radius: 4px;"
        )
        top_layout.addWidget(self.lot_label)

        # Hint de calcul
        hint = QtWidgets.QLabel(
            "(≤2000$ : 0.10  ·  +0.10 par tranche de 1000$  ·  cap 100.00)"
        )
        hint.setStyleSheet("color: #888; font-size: 11px;")
        top_layout.addWidget(hint)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        # --- Boutons Start / Stop ---
        btn_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("▶️ Démarrer l'IA")
        self.start_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 8px;")
        self.start_btn.clicked.connect(self.on_start)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("⏹️ Arrêter l'IA")
        self.stop_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 8px;")
        self.stop_btn.clicked.connect(self.on_stop)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # --- Statut du bot ---
        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.equity_label = QtWidgets.QLabel("Equity : N/A | Balance : N/A | Margin : N/A")
        self.equity_label.setWordWrap(True)
        layout.addWidget(self.equity_label)

        # ============ PANNEAU STATS LONG / SHORT ============
        # Track des trades de la session courante (depuis ouverture du GUI ou Reset)
        self.session_start = dt.datetime.now()
        self.session_balance_start: float | None = None  # capturé au 1er update_equity
        self._stats_seen_deal_tickets: set = set()
        # {"long":{"wins":[], "losses":[]}, "short":{"wins":[], "losses":[]}}
        self.session_stats = {
            "long":  {"wins": [], "losses": []},
            "short": {"wins": [], "losses": []},
        }

        stats_group = QtWidgets.QGroupBox("Stats de session (depuis ouverture)")
        stats_group.setStyleSheet("QGroupBox { color: #ccc; font-weight: bold; }")
        stats_v = QtWidgets.QVBoxLayout(stats_group)

        # Grille L / S / TOTAL
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(4)

        # Header
        def _hdr(txt):
            lbl = QtWidgets.QLabel(txt)
            lbl.setStyleSheet("color: #888; font-size: 10px;")
            return lbl

        grid.addWidget(_hdr("Côté"),    0, 0)
        grid.addWidget(_hdr("Trades"),  0, 1)
        grid.addWidget(_hdr("Wins"),    0, 2)
        grid.addWidget(_hdr("Losses"),  0, 3)
        grid.addWidget(_hdr("Winrate"), 0, 4)
        grid.addWidget(_hdr("PF"),      0, 5)
        grid.addWidget(_hdr("PnL"),     0, 6)
        grid.addWidget(_hdr("Avg W"),   0, 7)
        grid.addWidget(_hdr("Avg L"),   0, 8)

        def _stat_lbl(color="#dddddd", bold=False):
            lbl = QtWidgets.QLabel("—")
            w = "bold" if bold else "normal"
            lbl.setStyleSheet(f"color: {color}; font-weight: {w}; font-family: Consolas, 'Courier New', monospace;")
            lbl.setMinimumWidth(60)
            return lbl

        # Ligne LONG (vert)
        long_label = QtWidgets.QLabel("● LONG")
        long_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        grid.addWidget(long_label, 1, 0)
        self.long_trades_lbl = _stat_lbl()
        self.long_wins_lbl   = _stat_lbl(color="#2ecc71")
        self.long_losses_lbl = _stat_lbl(color="#e74c3c")
        self.long_wr_lbl     = _stat_lbl()
        self.long_pf_lbl     = _stat_lbl()
        self.long_pnl_lbl    = _stat_lbl(bold=True)
        self.long_avgw_lbl   = _stat_lbl(color="#2ecc71")
        self.long_avgl_lbl   = _stat_lbl(color="#e74c3c")
        grid.addWidget(self.long_trades_lbl, 1, 1)
        grid.addWidget(self.long_wins_lbl,   1, 2)
        grid.addWidget(self.long_losses_lbl, 1, 3)
        grid.addWidget(self.long_wr_lbl,     1, 4)
        grid.addWidget(self.long_pf_lbl,     1, 5)
        grid.addWidget(self.long_pnl_lbl,    1, 6)
        grid.addWidget(self.long_avgw_lbl,   1, 7)
        grid.addWidget(self.long_avgl_lbl,   1, 8)

        # Ligne SHORT (bleu)
        short_label = QtWidgets.QLabel("● SHORT")
        short_label.setStyleSheet("color: #3498db; font-weight: bold;")
        grid.addWidget(short_label, 2, 0)
        self.short_trades_lbl = _stat_lbl()
        self.short_wins_lbl   = _stat_lbl(color="#2ecc71")
        self.short_losses_lbl = _stat_lbl(color="#e74c3c")
        self.short_wr_lbl     = _stat_lbl()
        self.short_pf_lbl     = _stat_lbl()
        self.short_pnl_lbl    = _stat_lbl(bold=True)
        self.short_avgw_lbl   = _stat_lbl(color="#2ecc71")
        self.short_avgl_lbl   = _stat_lbl(color="#e74c3c")
        grid.addWidget(self.short_trades_lbl, 2, 1)
        grid.addWidget(self.short_wins_lbl,   2, 2)
        grid.addWidget(self.short_losses_lbl, 2, 3)
        grid.addWidget(self.short_wr_lbl,     2, 4)
        grid.addWidget(self.short_pf_lbl,     2, 5)
        grid.addWidget(self.short_pnl_lbl,    2, 6)
        grid.addWidget(self.short_avgw_lbl,   2, 7)
        grid.addWidget(self.short_avgl_lbl,   2, 8)

        # Séparateur
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet("color: #444;")
        grid.addWidget(sep, 3, 0, 1, 9)

        # Ligne TOTAL
        total_label = QtWidgets.QLabel("Σ TOTAL")
        total_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        grid.addWidget(total_label, 4, 0)
        self.total_trades_lbl = _stat_lbl(bold=True)
        self.total_wins_lbl   = _stat_lbl(color="#2ecc71", bold=True)
        self.total_losses_lbl = _stat_lbl(color="#e74c3c", bold=True)
        self.total_wr_lbl     = _stat_lbl(bold=True)
        self.total_pf_lbl     = _stat_lbl(bold=True)
        self.total_pnl_lbl    = _stat_lbl(bold=True)
        grid.addWidget(self.total_trades_lbl, 4, 1)
        grid.addWidget(self.total_wins_lbl,   4, 2)
        grid.addWidget(self.total_losses_lbl, 4, 3)
        grid.addWidget(self.total_wr_lbl,     4, 4)
        grid.addWidget(self.total_pf_lbl,     4, 5)
        grid.addWidget(self.total_pnl_lbl,    4, 6)
        # 2 dernières colonnes vides sur la ligne total
        grid.addWidget(_stat_lbl(), 4, 7)
        grid.addWidget(_stat_lbl(), 4, 8)

        stats_v.addLayout(grid)

        # Bouton Reset stats + info session
        bottom = QtWidgets.QHBoxLayout()
        self.session_info_lbl = QtWidgets.QLabel()
        self.session_info_lbl.setStyleSheet("color: #888; font-size: 10px;")
        bottom.addWidget(self.session_info_lbl)
        bottom.addStretch()
        self.reset_stats_btn = QtWidgets.QPushButton("🔄 Reset stats")
        self.reset_stats_btn.clicked.connect(self.on_reset_stats)
        self.reset_stats_btn.setMaximumWidth(120)
        bottom.addWidget(self.reset_stats_btn)
        stats_v.addLayout(bottom)

        layout.addWidget(stats_group)
        self._update_stats_display()

        # ============ ZONE DE LOG ============
        log_group = QtWidgets.QGroupBox("Logs en direct")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        log_layout.setSpacing(6)

        # -- Barre d'outils log --
        log_toolbar = QtWidgets.QHBoxLayout()
        log_toolbar.setSpacing(8)

        # Filtre texte
        log_toolbar.addWidget(QtWidgets.QLabel("Filtre :"))
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Rechercher dans les logs…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self.on_filter_changed)
        self.filter_edit.setFixedWidth(220)
        log_toolbar.addWidget(self.filter_edit)

        # Checkboxes par niveau
        self.level_checks = {}
        for lvl in (LogLevel.TRADE, LogLevel.SIGNAL, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR, LogLevel.DEBUG):
            cb = QtWidgets.QCheckBox(lvl)
            cb.setChecked(True)
            color = LEVEL_COLORS[lvl]
            cb.setStyleSheet(f"color: {color}; font-weight: bold;")
            cb.toggled.connect(lambda checked, l=lvl: self.on_level_toggled(l, checked))
            log_toolbar.addWidget(cb)
            self.level_checks[lvl] = cb

        log_toolbar.addStretch()

        # Auto-scroll
        self.autoscroll_cb = QtWidgets.QCheckBox("Auto-scroll")
        self.autoscroll_cb.setChecked(True)
        log_toolbar.addWidget(self.autoscroll_cb)

        # Boutons utilitaires
        self.clear_btn = QtWidgets.QPushButton("🗑️ Effacer")
        self.clear_btn.clicked.connect(self.on_clear_log)
        log_toolbar.addWidget(self.clear_btn)

        self.save_btn = QtWidgets.QPushButton("💾 Sauver")
        self.save_btn.clicked.connect(self.on_save_log)
        log_toolbar.addWidget(self.save_btn)

        log_layout.addLayout(log_toolbar)

        # Zone log (QTextEdit pour le HTML coloré)
        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        font_log = QtGui.QFont("Consolas")
        if not font_log.exactMatch():
            font_log = QtGui.QFont("Courier New")
        font_log.setPointSize(9)
        self.log_box.setFont(font_log)
        self.log_box.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #dddddd; border: 1px solid #333; }"
        )
        self.log_box.document().setMaximumBlockCount(self.MAX_LOG_BLOCKS)
        log_layout.addWidget(self.log_box, stretch=1)

        # Barre de compteurs
        self.counters_label = QtWidgets.QLabel()
        self.counters_label.setStyleSheet("color: #aaa; font-size: 11px;")
        log_layout.addWidget(self.counters_label)
        self._update_counters_label()

        layout.addWidget(log_group, stretch=1)

        # Timers
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.setInterval(1000)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start()

        self.equity_timer = QtCore.QTimer(self)
        self.equity_timer.setInterval(5000)
        self.equity_timer.timeout.connect(self.update_equity)
        self.equity_timer.start()

        # Stats LONG/SHORT depuis l'historique MT5
        self.stats_timer = QtCore.QTimer(self)
        self.stats_timer.setInterval(5000)
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start()

        self.update_status()
        self.update_equity()
        self.update_stats()

        # Petit message d'accueil
        self._append_log("Interface démarrée. En attente du lancement de l'IA…", LogLevel.INFO)

    # ====================================================
    # Boutons
    # ====================================================

    def on_start(self):
        self._append_log(
            f"[GUI] Démarrage du bot (lot dynamique selon equity)",
            LogLevel.TRADE
        )
        self.agent.start()
        self.update_status()

    def on_stop(self):
        self._append_log("[GUI] Arrêt du bot demandé.", LogLevel.WARN)
        self.agent.stop()
        self.update_status()

    def on_clear_log(self):
        self.log_box.clear()
        for k in self._counts:
            self._counts[k] = 0
        self._update_counters_label()

    def on_save_log(self):
        default_name = f"loup_log_{dt.datetime.now():%Y%m%d_%H%M%S}.txt"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Sauvegarder les logs", default_name, "Fichiers texte (*.txt);;Tous (*.*)"
        )
        if not path:
            return
        try:
            Path(path).write_text(self.log_box.toPlainText(), encoding="utf-8")
            self._append_log(f"[GUI] Logs sauvegardés dans {path}", LogLevel.INFO)
        except Exception as e:
            self._append_log(f"[GUI] Erreur sauvegarde : {e}", LogLevel.ERROR)

    def on_filter_changed(self, text: str):
        self._text_filter = text.strip().lower()

    def on_level_toggled(self, level: str, checked: bool):
        self._level_filters[level] = checked

    # ====================================================
    # Log
    # ====================================================

    def on_new_log(self, text: str, stream_name: str):
        level = classify(text, stream_name)
        self._append_log(text, level)

    def _append_log(self, text: str, level: str):
        self._counts[level] = self._counts.get(level, 0) + 1
        self._update_counters_label()

        # Filtres
        if not self._level_filters.get(level, True):
            return
        if self._text_filter and self._text_filter not in text.lower():
            return

        ts = dt.datetime.now().strftime("%H:%M:%S")
        color = LEVEL_COLORS.get(level, "#dddddd")

        # Échapper le HTML
        safe = (text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))

        html = (
            f'<span style="color:#666;">{ts}</span> '
            f'<span style="color:{color}; font-weight:bold;">[{level:<6}]</span> '
            f'<span style="color:{color};">{safe}</span>'
        )

        self.log_box.append(html)

        if self.autoscroll_cb.isChecked():
            sb = self.log_box.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _update_counters_label(self):
        parts = []
        for lvl in (LogLevel.TRADE, LogLevel.SIGNAL, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR, LogLevel.DEBUG):
            color = LEVEL_COLORS[lvl]
            parts.append(f'<span style="color:{color};">{lvl}: {self._counts.get(lvl, 0)}</span>')
        self.counters_label.setText("  |  ".join(parts))

    # ====================================================
    # Statut
    # ====================================================

    def update_status(self):
        running = self.agent._running
        # Lot affiché : si dynamique → calculé depuis equity courant
        if getattr(self.cfg, "dynamic_volume", False):
            try:
                info = mt5.account_info()
                equity = float(info.equity) if info is not None else 0.0
                from loup_live import compute_dynamic_volume
                lot = compute_dynamic_volume(equity, getattr(self.cfg, "max_lot", 100.0))
            except Exception:
                lot = 0.01
            # Met à jour le label "Lot dynamique"
            if hasattr(self, "lot_label"):
                self.lot_label.setText(f"{lot:.2f} lot (auto)")
        else:
            lot = self.cfg.position_size

        if running:
            text = f"État bot : ✅ EN COURS | side={self.cfg.side} | lot={lot:.2f}"
            self.status_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        else:
            text = f"État bot : ⏹️ ARRÊTÉ | side={self.cfg.side} | lot={lot:.2f}"
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.status_label.setText(text)

    # ====================================================
    # Stats LONG / SHORT
    # ====================================================

    def on_reset_stats(self):
        self.session_start = dt.datetime.now()
        self.session_balance_start = None   # sera recapturé au prochain update_equity
        self._stats_seen_deal_tickets.clear()
        self.session_stats = {
            "long":  {"wins": [], "losses": []},
            "short": {"wins": [], "losses": []},
        }
        self._update_stats_display()
        self._append_log("[GUI] Stats de session remises à zéro.", LogLevel.INFO)

    def update_stats(self):
        """Récupère les deals MT5 fermés depuis session_start et agrège L/S.

        Fixes :
        - Timezone : on requête depuis session_start - 24h (buffer) pour gérer
          le décalage local vs server MT5. La dédup par ticket évite les doublons.
        - Filtre magic == 424242 pour ne compter QUE les trades du bot.
        - PnL = d.profit seul (la commission est déjà incluse selon le broker ;
          mieux vaut sous-estimer légèrement que double-compter).
        """
        try:
            symbol = self.cfg.symbol if hasattr(self.cfg, "symbol") else "BTCUSD"
            # Buffer -24h pour absorber le décalage local/server, dédup par ticket
            since = self.session_start - dt.timedelta(hours=24)
            until = dt.datetime.now() + dt.timedelta(hours=24)
            deals = mt5.history_deals_get(since, until, group=symbol)
            if deals is None:
                self._update_stats_display()
                return

            _OUT_ENTRIES = (
                mt5.DEAL_ENTRY_OUT,       # 1 : sortie normale (SL/TP/manual)
                mt5.DEAL_ENTRY_INOUT,     # 2 : reverse position (netting)
                mt5.DEAL_ENTRY_OUT_BY,    # 3 : sortie par ordre opposé
            )
            # Magics du bot : 424242 (single) + 424241/424243 (multi-agent wf1/wf3)
            BOT_MAGICS = {424241, 424242, 424243}

            for d in deals:
                if d.entry not in _OUT_ENTRIES:
                    continue
                # Filtre bot : ignore manuel et autres EAs
                if int(getattr(d, "magic", 0)) not in BOT_MAGICS:
                    continue
                if d.ticket in self._stats_seen_deal_tickets:
                    continue
                self._stats_seen_deal_tickets.add(d.ticket)

                # d.profit suffit : commission souvent déjà incluse côté broker.
                pnl = float(d.profit)
                if d.type == mt5.DEAL_TYPE_SELL:
                    side = "long"   # SELL OUT ferme un LONG
                elif d.type == mt5.DEAL_TYPE_BUY:
                    side = "short"  # BUY OUT ferme un SHORT
                else:
                    continue

                if pnl > 0:
                    self.session_stats[side]["wins"].append(pnl)
                else:
                    self.session_stats[side]["losses"].append(pnl)

                # Debug : log de chaque deal accepté pour inspection
                self._append_log(
                    f"[STATS] deal #{d.ticket} {side.upper()} "
                    f"{'WIN' if pnl > 0 else 'LOSS'} {pnl:+.2f}$ "
                    f"(entry={d.entry}, magic={d.magic})",
                    LogLevel.DEBUG
                )

            self._update_stats_display()
        except Exception as e:
            self.session_info_lbl.setText(f"Erreur stats : {e}")

    def _fmt_money_html(self, x: float) -> str:
        if x > 0:
            return f'<span style="color:#2ecc71;">+{x:.2f}$</span>'
        if x < 0:
            return f'<span style="color:#e74c3c;">{x:.2f}$</span>'
        return '<span style="color:#888;">0.00$</span>'

    def _update_stats_display(self):
        def _side_stats(side):
            w = self.session_stats[side]["wins"]
            l = self.session_stats[side]["losses"]
            nw, nl = len(w), len(l)
            n = nw + nl
            wr = (nw / n) if n > 0 else 0.0
            tot_w = sum(w)
            tot_l = abs(sum(l))
            pf = (tot_w / tot_l) if tot_l > 1e-8 else 0.0
            pnl = tot_w - tot_l
            avg_w = (tot_w / nw) if nw > 0 else 0.0
            avg_l = (sum(l) / nl) if nl > 0 else 0.0
            return n, nw, nl, wr, pf, pnl, avg_w, avg_l

        # LONG
        n, nw, nl, wr, pf, pnl, aw, al = _side_stats("long")
        self.long_trades_lbl.setText(str(n))
        self.long_wins_lbl.setText(str(nw))
        self.long_losses_lbl.setText(str(nl))
        self.long_wr_lbl.setText(f"{wr*100:.1f}%" if n > 0 else "—")
        self.long_pf_lbl.setText(f"{pf:.2f}" if n > 0 else "—")
        self._set_money_label(self.long_pnl_lbl, pnl, bold=True)
        self.long_avgw_lbl.setText(f"+{aw:.2f}$" if nw > 0 else "—")
        self.long_avgl_lbl.setText(f"{al:.2f}$" if nl > 0 else "—")

        # SHORT
        n2, nw2, nl2, wr2, pf2, pnl2, aw2, al2 = _side_stats("short")
        self.short_trades_lbl.setText(str(n2))
        self.short_wins_lbl.setText(str(nw2))
        self.short_losses_lbl.setText(str(nl2))
        self.short_wr_lbl.setText(f"{wr2*100:.1f}%" if n2 > 0 else "—")
        self.short_pf_lbl.setText(f"{pf2:.2f}" if n2 > 0 else "—")
        self._set_money_label(self.short_pnl_lbl, pnl2, bold=True)
        self.short_avgw_lbl.setText(f"+{aw2:.2f}$" if nw2 > 0 else "—")
        self.short_avgl_lbl.setText(f"{al2:.2f}$" if nl2 > 0 else "—")

        # TOTAL
        tot_n = n + n2
        tot_w = nw + nw2
        tot_l = nl + nl2
        tot_wr = (tot_w / tot_n) if tot_n > 0 else 0.0
        gw = sum(self.session_stats["long"]["wins"]) + sum(self.session_stats["short"]["wins"])
        gl = abs(sum(self.session_stats["long"]["losses"]) + sum(self.session_stats["short"]["losses"]))
        tot_pf = (gw / gl) if gl > 1e-8 else 0.0
        tot_pnl = pnl + pnl2

        self.total_trades_lbl.setText(str(tot_n))
        self.total_wins_lbl.setText(str(tot_w))
        self.total_losses_lbl.setText(str(tot_l))
        self.total_wr_lbl.setText(f"{tot_wr*100:.1f}%" if tot_n > 0 else "—")
        self.total_pf_lbl.setText(f"{tot_pf:.2f}" if tot_n > 0 else "—")
        self._set_money_label(self.total_pnl_lbl, tot_pnl, bold=True)

        elapsed = dt.datetime.now() - self.session_start
        h = int(elapsed.total_seconds() // 3600)
        m = int((elapsed.total_seconds() % 3600) // 60)
        self.session_info_lbl.setText(
            f"Session ouverte depuis {self.session_start:%H:%M:%S}  ·  durée {h:d}h{m:02d}m"
        )

    def _set_money_label(self, lbl: QtWidgets.QLabel, x: float, bold: bool = False):
        if x > 0:
            color = "#2ecc71"
        elif x < 0:
            color = "#e74c3c"
        else:
            color = "#888"
        weight = "bold" if bold else "normal"
        sign = "+" if x > 0 else ""
        lbl.setText(f"{sign}{x:.2f}$")
        lbl.setStyleSheet(
            f"color: {color}; font-weight: {weight}; font-family: Consolas, 'Courier New', monospace;"
        )

    def update_equity(self):
        try:
            info = mt5.account_info()
            if info is None:
                self.equity_label.setText("Equity : N/A | Balance : N/A | Margin : N/A (MT5 non connecté)")
                self.equity_label.setStyleSheet("color: #888;")
                return

            equity  = float(info.equity)
            balance = float(info.balance)
            margin  = float(info.margin)
            free    = float(info.margin_free)
            pl_flot = equity - balance   # P/L des positions ouvertes

            # Capture du balance de référence à la 1ère lecture (= début de session)
            if self.session_balance_start is None:
                self.session_balance_start = balance

            # P/L session réalisé = balance courant - balance de départ
            pl_session = balance - self.session_balance_start
            # P/L session total = réalisé + flottant
            pl_total   = pl_session + pl_flot

            color = "#2ecc71" if pl_total >= 0 else "#e74c3c"
            self.equity_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            self.equity_label.setText(
                f"Equity : {equity:.2f}  |  Balance : {balance:.2f}  |  "
                f"P/L session : {pl_session:+.2f}  |  P/L flottant : {pl_flot:+.2f}  |  "
                f"P/L total : {pl_total:+.2f}  |  Margin : {margin:.2f}  |  Free : {free:.2f}"
            )
        except Exception as e:
            self.equity_label.setText(f"Equity : erreur ({e})")
            self.equity_label.setStyleSheet("color: #e74c3c;")

    # ====================================================
    # Fermeture
    # ====================================================

    def closeEvent(self, event: QtGui.QCloseEvent):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

        if self.agent._running:
            self._append_log("[GUI] Fermeture de la fenêtre → arrêt du bot…", LogLevel.WARN)
            self.agent.stop()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    # Style sombre par défaut pour mieux voir les couleurs de log
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
