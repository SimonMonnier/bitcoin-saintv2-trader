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
        self.cfg = LiveConfig(side="long")
        self.agent = TradingAgent(self.cfg)

        self.setWindowTitle("Loup Ω – BTCUSD M1 Long")
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
            "Agent RL en live sur MT5 – Mode Long, sans TP fixe (SL initial + break-even + trailing)."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # --- Ligne info + lot ---
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addWidget(QtWidgets.QLabel("Taille de position (lot BTC) :"))

        self.lot_spin = QtWidgets.QDoubleSpinBox()
        self.lot_spin.setDecimals(2)
        self.lot_spin.setMinimum(0.01)
        self.lot_spin.setMaximum(1.00)
        self.lot_spin.setSingleStep(0.01)
        self.lot_spin.setValue(self.cfg.position_size)
        self.lot_spin.setSuffix(" lot")
        self.lot_spin.setFixedWidth(120)
        top_layout.addWidget(self.lot_spin)
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

        self.update_status()
        self.update_equity()

        # Petit message d'accueil
        self._append_log("Interface démarrée. En attente du lancement de l'IA…", LogLevel.INFO)

    # ====================================================
    # Boutons
    # ====================================================

    def on_start(self):
        new_lot = float(self.lot_spin.value())
        self.cfg.position_size = new_lot
        self._append_log(f"[GUI] Démarrage du bot avec lot={new_lot:.2f}", LogLevel.TRADE)
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
        lot = self.cfg.position_size
        if running:
            text = f"État bot : ✅ EN COURS | side={self.cfg.side} | lot={lot:.2f}"
            self.status_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        else:
            text = f"État bot : ⏹️ ARRÊTÉ | side={self.cfg.side} | lot={lot:.2f}"
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.status_label.setText(text)

    def update_equity(self):
        try:
            mt5.initialize()
            info = mt5.account_info()
            if info is None:
                self.equity_label.setText("Equity : N/A | Balance : N/A | Margin : N/A (MT5 non connecté)")
                self.equity_label.setStyleSheet("color: #888;")
                return

            equity  = float(info.equity)
            balance = float(info.balance)
            margin  = float(info.margin)
            free    = float(info.margin_free)
            pl      = equity - balance

            color = "#2ecc71" if pl >= 0 else "#e74c3c"
            self.equity_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            self.equity_label.setText(
                f"Equity : {equity:.2f}  |  Balance : {balance:.2f}  |  "
                f"P/L flottant : {pl:+.2f}  |  Margin : {margin:.2f}  |  Free : {free:.2f}"
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
