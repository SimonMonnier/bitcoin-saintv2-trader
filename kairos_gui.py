"""KAIROS — poste de pilotage de l'agent en live sur MT5.

    python kairos_gui.py

La logique de trading n'est pas ici : elle vit dans kairos_live.TradingAgent.
Cette fenetre ne fait qu'observer, afficher et demarrer/arreter.

Organisation de l'ecran, du plus urgent au moins urgent :
  1. barre haute   — l'agent tourne-t-il, et comment l'arreter
  2. mesures       — equity, PnL de session, winrate contre le point mort
  3. courbe        — la FORME de la session, pas ses valeurs
  4. tableau       — achat / vente / total, la ou se lit l'asymetrie
  5. console       — le detail, filtrable, en bas parce qu'on y descend
"""

import sys
import io
import re
import datetime as dt
from pathlib import Path

from PySide6 import QtWidgets, QtCore, QtGui
import MetaTrader5 as mt5

from kairos_live import LiveConfig, TradingAgent
from kairos_theme import C, LEVEL_COLORS, font_ui, font_mono, stylesheet
from kairos_widgets import (StatusOrb, Sparkline, WinrateArc, MetricTile,
                            Card, filet)

# Winrate a partir duquel la strategie couvre ses frais, mesure sur BTCUSD
# avec AvgW +13.97 $ / AvgL -10.83 $. C'est le seul repere qui compte : un
# winrate brut ne dit rien sans lui.
SEUIL_EQUILIBRE = 0.437


# ============================================================
# Redirection des print vers la console de la fenetre
# ============================================================

class LogEmitter(QtCore.QObject):
    message = QtCore.Signal(str, str)   # (texte, flux)


class QtLogStream(io.TextIOBase):
    def __init__(self, emitter: LogEmitter, stream_name: str):
        super().__init__()
        self.emitter = emitter
        self.stream_name = stream_name
        self._buffer = ""

    def write(self, msg: str):
        msg = str(msg)
        self._buffer += msg
        while "\n" in self._buffer:
            ligne, self._buffer = self._buffer.split("\n", 1)
            if ligne.strip():
                self.emitter.message.emit(ligne, self.stream_name)
        return len(msg)

    def flush(self):
        if self._buffer.strip():
            self.emitter.message.emit(self._buffer, self.stream_name)
        self._buffer = ""


class LogLevel:
    ERROR  = "ERROR"
    WARN   = "WARN"
    TRADE  = "TRADE"
    SIGNAL = "SIGNAL"
    INFO   = "INFO"
    DEBUG  = "DEBUG"


_PATTERNS = [
    (LogLevel.ERROR,  re.compile(r"\b(error|erreur|exception|traceback|fail|échec|failed)\b", re.I)),
    (LogLevel.WARN,   re.compile(r"\b(warn|warning|avertissement|attention)\b", re.I)),
    (LogLevel.TRADE,  re.compile(r"\b(buy|sell|close|open|position|order\s*sent|filled|deal|ouvert|fermé|clos[eé])\b", re.I)),
    (LogLevel.SIGNAL, re.compile(r"\b(signal|action|prob|argmax|best long|best short)\b", re.I)),
]

ORDRE_NIVEAUX = (LogLevel.TRADE, LogLevel.SIGNAL, LogLevel.INFO,
                 LogLevel.WARN, LogLevel.ERROR, LogLevel.DEBUG)


def classify(text: str, stream_name: str) -> str:
    if stream_name == "ERR":
        return LogLevel.ERROR
    for level, pat in _PATTERNS:
        if pat.search(text):
            return level
    return LogLevel.INFO


# ============================================================
# Fenetre principale
# ============================================================

class MainWindow(QtWidgets.QMainWindow):

    MAX_LOG_BLOCKS = 5000     # tampon circulaire : la console ne grossit jamais

    def __init__(self):
        super().__init__()

        self.cfg = LiveConfig(side="both")
        self.agent = TradingAgent(self.cfg)

        # Le titre vient de la config, jamais d'une constante : un titre ecrit
        # en dur finit par annoncer un symbole que l'agent ne trade pas.
        self.setWindowTitle(f"KAIROS — {getattr(self.cfg, 'symbol', '?')} M1")
        self.setMinimumSize(1080, 700)
        self.resize(1320, 880)

        self._counts = {lvl: 0 for lvl in LEVEL_COLORS}
        self._level_filters = {lvl: True for lvl in LEVEL_COLORS}
        self._text_filter = ""

        # ---------- Redirection stdout/stderr ----------
        self.log_emitter = LogEmitter()
        self.log_emitter.message.connect(self.on_new_log)
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = QtLogStream(self.log_emitter, "OUT")
        sys.stderr = QtLogStream(self.log_emitter, "ERR")

        # ---------- Etat de session ----------
        self.session_start = dt.datetime.now()
        self.session_balance_start: float | None = None
        self._stats_seen_deal_tickets: set = set()
        self.session_stats = {
            "long":  {"wins": [], "losses": []},
            "short": {"wins": [], "losses": []},
        }

        # ---------- Echafaudage ----------
        racine = QtWidgets.QWidget()
        racine.setObjectName("Root")
        self.setCentralWidget(racine)
        col = QtWidgets.QVBoxLayout(racine)
        col.setContentsMargins(18, 16, 18, 16)
        col.setSpacing(13)

        col.addWidget(self._bati_barre_haute())
        col.addLayout(self._bati_mesures())
        col.addWidget(self._bati_courbe())
        col.addWidget(self._bati_tableau())
        col.addWidget(self._bati_console(), stretch=1)

        # ---------- Rythmes de rafraichissement ----------
        # L'etat toutes les secondes (c'est ce qu'on regarde), le compte et les
        # statistiques toutes les cinq : interroger MT5 plus souvent ne donne
        # rien de neuf et charge le terminal pour rien.
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.setInterval(1000)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start()

        self.equity_timer = QtCore.QTimer(self)
        self.equity_timer.setInterval(5000)
        self.equity_timer.timeout.connect(self.update_equity)
        self.equity_timer.start()

        self.stats_timer = QtCore.QTimer(self)
        self.stats_timer.setInterval(5000)
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start()

        self.update_status()
        self.update_equity()
        self.update_stats()
        self._update_stats_display()
        self._append_log("Interface prête. L'agent est à l'arrêt.", LogLevel.INFO)

    # --------------------------------------------------------
    # Construction
    # --------------------------------------------------------

    def _bati_barre_haute(self) -> QtWidgets.QWidget:
        barre = QtWidgets.QFrame()
        barre.setObjectName("TopBar")
        barre.setFixedHeight(78)
        h = QtWidgets.QHBoxLayout(barre)
        h.setContentsMargins(22, 0, 18, 0)
        h.setSpacing(16)

        # Marque
        bloc = QtWidgets.QVBoxLayout()
        bloc.setSpacing(1)
        nom = QtWidgets.QLabel("KAIROS")
        nom.setFont(font_ui(19, QtGui.QFont.Bold, spacing=4.2))
        nom.setStyleSheet(f"color: {C['ink']};")
        accroche = QtWidgets.QLabel("Le modèle ne prédit pas le marché. Il attend le moment.")
        accroche.setFont(font_ui(8.5))
        accroche.setStyleSheet(f"color: {C['dim']};")
        bloc.addWidget(nom)
        bloc.addWidget(accroche)
        h.addLayout(bloc)

        h.addSpacing(10)
        h.addWidget(filet(horizontal=False))
        h.addSpacing(10)

        # Temoin d'etat
        etat = QtWidgets.QHBoxLayout()
        etat.setSpacing(9)
        self.orb = StatusOrb()
        self.status_label = QtWidgets.QLabel()      # nom conserve : logique inchangee
        self.status_label.setFont(font_ui(11, QtGui.QFont.DemiBold))
        contexte = QtWidgets.QLabel(
            f"{getattr(self.cfg, 'symbol', 'BTCUSD')}  ·  {self.cfg.side}")
        contexte.setFont(font_mono(9))
        contexte.setStyleSheet(f"color: {C['dim']};")
        etat.addWidget(self.orb)
        etat.addWidget(self.status_label)
        etat.addWidget(contexte)
        h.addLayout(etat)

        h.addStretch()

        # Dimensionnement courant
        lot = QtWidgets.QVBoxLayout()
        lot.setSpacing(1)
        self.lot_title = QtWidgets.QLabel("VOLUME")
        self.lot_title.setFont(font_ui(8, QtGui.QFont.DemiBold, spacing=1.3))
        self.lot_title.setStyleSheet(f"color: {C['dim']};")
        self.lot_label = QtWidgets.QLabel("—")
        self.lot_label.setFont(font_mono(13, QtGui.QFont.DemiBold))
        self.lot_label.setStyleSheet(f"color: {C['kairos']};")
        lot.addWidget(self.lot_title)
        lot.addWidget(self.lot_label)
        h.addLayout(lot)

        h.addSpacing(18)

        self.start_btn = QtWidgets.QPushButton("Démarrer")
        self.start_btn.setObjectName("PrimaryBtn")
        self.start_btn.setFont(font_ui(11, QtGui.QFont.DemiBold))
        self.start_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self.on_start)

        self.stop_btn = QtWidgets.QPushButton("Arrêter")
        self.stop_btn.setObjectName("DangerBtn")
        self.stop_btn.setFont(font_ui(11, QtGui.QFont.DemiBold))
        self.stop_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.stop_btn.clicked.connect(self.on_stop)

        h.addWidget(self.start_btn)
        h.addWidget(self.stop_btn)
        return barre

    def _bati_mesures(self) -> QtWidgets.QHBoxLayout:
        rang = QtWidgets.QHBoxLayout()
        rang.setSpacing(13)

        # Realise et flottant separes : les additionner dans une seule tuile
        # donnerait deux fois le meme nombre en debut de session et masquerait
        # lequel des deux bouge.
        self.tile_equity = MetricTile("Equity")
        self.tile_pnl    = MetricTile("P/L réalisé")
        self.tile_flot   = MetricTile("P/L flottant")
        self.tile_trades = MetricTile("Trades")
        for t in (self.tile_equity, self.tile_pnl, self.tile_flot, self.tile_trades):
            rang.addWidget(t, stretch=1)

        jauge = Card("Winrate")
        self.arc = WinrateArc(SEUIL_EQUILIBRE)
        jauge.ajoute(self.arc)
        jauge.setMinimumWidth(180)
        rang.addWidget(jauge)
        return rang

    def _bati_courbe(self) -> QtWidgets.QWidget:
        carte = Card("Equity de la session")
        self.equity_label = QtWidgets.QLabel("—")      # nom conserve
        self.equity_label.setFont(font_mono(9))
        self.equity_label.setStyleSheet(f"color: {C['muted']};")
        self.equity_label.setWordWrap(False)
        self.equity_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        carte.entete.addWidget(self.equity_label)

        self.spark = Sparkline()
        carte.ajoute(self.spark)
        return carte

    def _bati_tableau(self) -> QtWidgets.QWidget:
        carte = Card("Répartition par côté")

        self.session_info_lbl = QtWidgets.QLabel("")   # nom conserve
        self.session_info_lbl.setFont(font_ui(8.5))
        self.session_info_lbl.setStyleSheet(f"color: {C['dim']};")
        carte.entete.addWidget(self.session_info_lbl)

        self.reset_stats_btn = QtWidgets.QPushButton("Réinitialiser")
        self.reset_stats_btn.setFont(font_ui(9))
        self.reset_stats_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.reset_stats_btn.clicked.connect(self.on_reset_stats)
        carte.entete.addWidget(self.reset_stats_btn)

        g = QtWidgets.QGridLayout()
        g.setHorizontalSpacing(0)
        g.setVerticalSpacing(7)

        colonnes = ["", "Trades", "Gagnants", "Perdants", "Winrate",
                    "Profit factor", "PnL", "Gain moyen", "Perte moyenne"]
        for j, titre in enumerate(colonnes):
            lbl = QtWidgets.QLabel(titre.upper())
            lbl.setFont(font_ui(7.8, QtGui.QFont.DemiBold, spacing=1.0))
            lbl.setStyleSheet(f"color: {C['dim']};")
            lbl.setAlignment(QtCore.Qt.AlignLeft if j == 0 else QtCore.Qt.AlignRight)
            g.addWidget(lbl, 0, j)
            g.setColumnStretch(j, 0 if j == 0 else 1)

        def cellule(couleur=None, gras=False):
            lbl = QtWidgets.QLabel("—")
            lbl.setFont(font_mono(10.5, QtGui.QFont.DemiBold if gras else QtGui.QFont.Normal))
            lbl.setStyleSheet(f"color: {couleur or C['ink']};")
            lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            return lbl

        def intitule(txt, couleur):
            lbl = QtWidgets.QLabel(txt)
            lbl.setFont(font_ui(10, QtGui.QFont.DemiBold))
            lbl.setStyleSheet(f"color: {couleur};")
            lbl.setMinimumWidth(78)
            return lbl

        # ACHAT
        g.addWidget(intitule("▲  ACHAT", C["long"]), 1, 0)
        self.long_trades_lbl = cellule()
        self.long_wins_lbl   = cellule(C["win"])
        self.long_losses_lbl = cellule(C["loss"])
        self.long_wr_lbl     = cellule()
        self.long_pf_lbl     = cellule()
        self.long_pnl_lbl    = cellule(gras=True)
        self.long_avgw_lbl   = cellule(C["win"])
        self.long_avgl_lbl   = cellule(C["loss"])
        for j, w in enumerate((self.long_trades_lbl, self.long_wins_lbl,
                               self.long_losses_lbl, self.long_wr_lbl,
                               self.long_pf_lbl, self.long_pnl_lbl,
                               self.long_avgw_lbl, self.long_avgl_lbl), start=1):
            g.addWidget(w, 1, j)

        # VENTE
        g.addWidget(intitule("▼  VENTE", C["short"]), 2, 0)
        self.short_trades_lbl = cellule()
        self.short_wins_lbl   = cellule(C["win"])
        self.short_losses_lbl = cellule(C["loss"])
        self.short_wr_lbl     = cellule()
        self.short_pf_lbl     = cellule()
        self.short_pnl_lbl    = cellule(gras=True)
        self.short_avgw_lbl   = cellule(C["win"])
        self.short_avgl_lbl   = cellule(C["loss"])
        for j, w in enumerate((self.short_trades_lbl, self.short_wins_lbl,
                               self.short_losses_lbl, self.short_wr_lbl,
                               self.short_pf_lbl, self.short_pnl_lbl,
                               self.short_avgw_lbl, self.short_avgl_lbl), start=1):
            g.addWidget(w, 2, j)

        g.addWidget(filet(), 3, 0, 1, 9)

        # TOTAL
        g.addWidget(intitule("Σ  TOTAL", C["ink"]), 4, 0)
        self.total_trades_lbl = cellule(gras=True)
        self.total_wins_lbl   = cellule(C["win"], gras=True)
        self.total_losses_lbl = cellule(C["loss"], gras=True)
        self.total_wr_lbl     = cellule(gras=True)
        self.total_pf_lbl     = cellule(gras=True)
        self.total_pnl_lbl    = cellule(gras=True)
        for j, w in enumerate((self.total_trades_lbl, self.total_wins_lbl,
                               self.total_losses_lbl, self.total_wr_lbl,
                               self.total_pf_lbl, self.total_pnl_lbl), start=1):
            g.addWidget(w, 4, j)

        carte.ajoute_layout(g)
        return carte

    def _bati_console(self) -> QtWidgets.QWidget:
        carte = Card("Journal")

        self.counters_label = QtWidgets.QLabel()       # nom conserve
        self.counters_label.setFont(font_mono(8.5))
        carte.entete.addWidget(self.counters_label)

        outils = QtWidgets.QHBoxLayout()
        outils.setSpacing(7)

        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Filtrer…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setFont(font_ui(9.5))
        self.filter_edit.setFixedWidth(210)
        self.filter_edit.textChanged.connect(self.on_filter_changed)
        outils.addWidget(self.filter_edit)

        outils.addSpacing(6)

        # Pastilles a la place des cases a cocher : une case a cocher grise ne
        # dit pas de quelle couleur est le niveau qu'elle filtre.
        self.level_checks = {}
        for lvl in ORDRE_NIVEAUX:
            couleur = LEVEL_COLORS[lvl]
            chip = QtWidgets.QPushButton(lvl)
            chip.setObjectName("Chip")
            chip.setCheckable(True)
            chip.setChecked(True)
            chip.setCursor(QtCore.Qt.PointingHandCursor)
            chip.setFont(font_ui(8, QtGui.QFont.DemiBold, spacing=0.6))
            chip.setStyleSheet(
                f"QPushButton#Chip:checked {{ color: {couleur}; border-color: {couleur}; }}")
            chip.toggled.connect(lambda coche, l=lvl: self.on_level_toggled(l, coche))
            outils.addWidget(chip)
            self.level_checks[lvl] = chip

        outils.addStretch()

        self.autoscroll_cb = QtWidgets.QPushButton("Suivre")
        self.autoscroll_cb.setObjectName("Chip")
        self.autoscroll_cb.setCheckable(True)
        self.autoscroll_cb.setChecked(True)
        self.autoscroll_cb.setCursor(QtCore.Qt.PointingHandCursor)
        self.autoscroll_cb.setFont(font_ui(8, QtGui.QFont.DemiBold, spacing=0.6))
        self.autoscroll_cb.setStyleSheet(
            f"QPushButton#Chip:checked {{ color: {C['kairos']}; border-color: {C['kairos']}; }}")
        outils.addWidget(self.autoscroll_cb)

        self.clear_btn = QtWidgets.QPushButton("Effacer")
        self.clear_btn.setFont(font_ui(9))
        self.clear_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.clear_btn.clicked.connect(self.on_clear_log)
        outils.addWidget(self.clear_btn)

        self.save_btn = QtWidgets.QPushButton("Exporter")
        self.save_btn.setFont(font_ui(9))
        self.save_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.save_btn.clicked.connect(self.on_save_log)
        outils.addWidget(self.save_btn)

        carte.ajoute_layout(outils)

        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setObjectName("Console")
        self.log_box.setReadOnly(True)
        self.log_box.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.log_box.setFont(font_mono(9))
        self.log_box.document().setMaximumBlockCount(self.MAX_LOG_BLOCKS)
        carte.ajoute(self.log_box, stretch=1)

        self._update_counters_label()
        return carte

    # --------------------------------------------------------
    # Boutons
    # --------------------------------------------------------

    def on_start(self):
        self._append_log("[GUI] Démarrage de l'agent (volume par le risque).",
                         LogLevel.TRADE)
        self.agent.start()
        self.update_status()

    def on_stop(self):
        self._append_log("[GUI] Arrêt de l'agent demandé.", LogLevel.WARN)
        self.agent.stop()
        self.update_status()

    def on_clear_log(self):
        self.log_box.clear()
        for k in self._counts:
            self._counts[k] = 0
        self._update_counters_label()

    def on_save_log(self):
        defaut = f"kairos_log_{dt.datetime.now():%Y%m%d_%H%M%S}.txt"
        chemin, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exporter le journal", defaut, "Fichiers texte (*.txt);;Tous (*.*)"
        )
        if not chemin:
            return
        try:
            Path(chemin).write_text(self.log_box.toPlainText(), encoding="utf-8")
            self._append_log(f"[GUI] Journal exporté vers {chemin}", LogLevel.INFO)
        except Exception as e:
            self._append_log(f"[GUI] Export impossible : {e}", LogLevel.ERROR)

    def on_filter_changed(self, text: str):
        self._text_filter = text.strip().lower()

    def on_level_toggled(self, level: str, checked: bool):
        self._level_filters[level] = checked

    # --------------------------------------------------------
    # Journal
    # --------------------------------------------------------

    def on_new_log(self, text: str, stream_name: str):
        self._append_log(text, classify(text, stream_name))

    def _append_log(self, text: str, level: str):
        self._counts[level] = self._counts.get(level, 0) + 1
        self._update_counters_label()

        if not self._level_filters.get(level, True):
            return
        if self._text_filter and self._text_filter not in text.lower():
            return

        ts = dt.datetime.now().strftime("%H:%M:%S")
        couleur = LEVEL_COLORS.get(level, C["ink"])
        safe = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

        html = (
            f'<span style="color:{C["dim"]};">{ts}</span>'
            f'<span style="color:{couleur};">  {level:<6}  </span>'
            f'<span style="color:{C["ink"] if level in (LogLevel.INFO, LogLevel.TRADE) else couleur};">{safe}</span>'
        )
        self.log_box.append(html)

        if self.autoscroll_cb.isChecked():
            sb = self.log_box.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _update_counters_label(self):
        # Initiale + compte : une suite de nombres colores seule est illisible
        # des qu'on ne se souvient plus de l'ordre des niveaux.
        parts = []
        for lvl in ORDRE_NIVEAUX:
            n = self._counts.get(lvl, 0)
            couleur = LEVEL_COLORS[lvl] if n else C["dim"]
            parts.append(f'<span style="color:{couleur};">{lvl[0]}&#8202;{n}</span>')
        self.counters_label.setText(
            f'<span style="color:{C["dim"]};"> · </span>'.join(parts))
        self.counters_label.setToolTip(
            "  ".join(f"{lvl} {self._counts.get(lvl, 0)}" for lvl in ORDRE_NIVEAUX))

    # --------------------------------------------------------
    # Etat
    # --------------------------------------------------------

    def update_status(self):
        running = self.agent._running

        # Ce qui est affiche doit etre ce que l'agent applique reellement.
        # En mode risque, le lot n'existe pas a l'avance : il depend de la
        # distance au stop, connue seulement a l'entree. La grandeur fixe,
        # elle, c'est le budget de risque — c'est donc elle qu'on montre.
        if getattr(self.cfg, "risk_volume", True):
            self.lot_title.setText("RISQUE / TRADE")
            try:
                info = mt5.account_info()
                equity = float(info.equity) if info is not None else 0.0
                frac = float(getattr(self.cfg, "risk_per_trade", 0.012))
                self.lot_label.setText(
                    f"{equity * frac:.2f}$" if equity > 0 else f"{frac*100:.1f}%")
                self.lot_label.setToolTip(
                    f"{frac*100:.2f} % de l'equity par trade — le lot est déduit "
                    f"de la distance au stop à l'entrée")
            except Exception:
                self.lot_label.setText("—")
        elif getattr(self.cfg, "dynamic_volume", False):
            self.lot_title.setText("VOLUME AUTO")
            try:
                info = mt5.account_info()
                equity = float(info.equity) if info is not None else 0.0
                from kairos_live import compute_dynamic_volume
                lot = compute_dynamic_volume(equity, getattr(self.cfg, "max_lot", 100.0))
            except Exception:
                lot = 0.01
            self.lot_label.setText(f"{lot:.2f}")
        else:
            self.lot_title.setText("VOLUME")
            self.lot_label.setText(f"{float(self.cfg.position_size):.2f}")

        self.orb.set_actif(running)
        if running:
            self.status_label.setText("EN COURS")
            self.status_label.setStyleSheet(f"color: {C['kairos']};")
        else:
            self.status_label.setText("À L'ARRÊT")
            self.status_label.setStyleSheet(f"color: {C['dim']};")

        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    # --------------------------------------------------------
    # Statistiques par cote
    # --------------------------------------------------------

    def on_reset_stats(self):
        self.session_start = dt.datetime.now()
        self.session_balance_start = None
        self._stats_seen_deal_tickets.clear()
        self.session_stats = {
            "long":  {"wins": [], "losses": []},
            "short": {"wins": [], "losses": []},
        }
        self.spark.reinit()
        self._update_stats_display()
        self._append_log("[GUI] Statistiques de session réinitialisées.", LogLevel.INFO)

    def update_stats(self):
        """Agrege les deals MT5 clotures depuis le debut de session.

        Trois precautions heritees de bugs reels :
          - fenetre elargie de +-24 h pour absorber le decalage entre l'heure
            locale et celle du serveur, avec dedoublonnage par ticket ;
          - filtre sur les magics du bot, pour ne pas compter le manuel ;
          - PnL = d.profit seul, la commission etant deja incluse selon le
            courtier — mieux vaut sous-estimer que compter deux fois.
        """
        try:
            symbol = getattr(self.cfg, "symbol", "BTCUSD")
            since = self.session_start - dt.timedelta(hours=24)
            until = dt.datetime.now() + dt.timedelta(hours=24)
            deals = mt5.history_deals_get(since, until, group=symbol)
            if deals is None:
                self._update_stats_display()
                return

            _OUT_ENTRIES = (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT,
                            mt5.DEAL_ENTRY_OUT_BY)
            BOT_MAGICS = {424241, 424242, 424243}

            for d in deals:
                if d.entry not in _OUT_ENTRIES:
                    continue
                if int(getattr(d, "magic", 0)) not in BOT_MAGICS:
                    continue
                if d.ticket in self._stats_seen_deal_tickets:
                    continue
                self._stats_seen_deal_tickets.add(d.ticket)

                pnl = float(d.profit)
                if d.type == mt5.DEAL_TYPE_SELL:
                    side = "long"      # une vente de sortie ferme un achat
                elif d.type == mt5.DEAL_TYPE_BUY:
                    side = "short"
                else:
                    continue

                if pnl > 0:
                    self.session_stats[side]["wins"].append(pnl)
                else:
                    self.session_stats[side]["losses"].append(pnl)

                self._append_log(
                    f"[STATS] deal #{d.ticket} {side.upper()} "
                    f"{'GAIN' if pnl > 0 else 'PERTE'} {pnl:+.2f}$ "
                    f"(entry={d.entry}, magic={d.magic})",
                    LogLevel.DEBUG
                )

            self._update_stats_display()
        except Exception as e:
            self.session_info_lbl.setText(f"statistiques indisponibles : {e}")

    def _set_money_label(self, lbl: QtWidgets.QLabel, x: float, bold: bool = False):
        couleur = C["win"] if x > 0 else (C["loss"] if x < 0 else C["dim"])
        lbl.setFont(font_mono(10.5, QtGui.QFont.DemiBold if bold else QtGui.QFont.Normal))
        lbl.setStyleSheet(f"color: {couleur};")
        lbl.setText(f"{'+' if x > 0 else ''}{x:.2f}$")

    def _update_stats_display(self):
        def _cote(side):
            w = self.session_stats[side]["wins"]
            l = self.session_stats[side]["losses"]
            nw, nl = len(w), len(l)
            n = nw + nl
            wr = (nw / n) if n > 0 else 0.0
            tot_w, tot_l = sum(w), abs(sum(l))
            pf = (tot_w / tot_l) if tot_l > 1e-8 else 0.0
            return (n, nw, nl, wr, pf, tot_w - tot_l,
                    (tot_w / nw) if nw else 0.0, (sum(l) / nl) if nl else 0.0)

        n, nw, nl, wr, pf, pnl, aw, al = _cote("long")
        self.long_trades_lbl.setText(str(n))
        self.long_wins_lbl.setText(str(nw))
        self.long_losses_lbl.setText(str(nl))
        self.long_wr_lbl.setText(f"{wr*100:.1f}%" if n else "—")
        self.long_pf_lbl.setText(f"{pf:.2f}" if n else "—")
        self._set_money_label(self.long_pnl_lbl, pnl, bold=True)
        self.long_avgw_lbl.setText(f"+{aw:.2f}$" if nw else "—")
        self.long_avgl_lbl.setText(f"{al:.2f}$" if nl else "—")

        n2, nw2, nl2, wr2, pf2, pnl2, aw2, al2 = _cote("short")
        self.short_trades_lbl.setText(str(n2))
        self.short_wins_lbl.setText(str(nw2))
        self.short_losses_lbl.setText(str(nl2))
        self.short_wr_lbl.setText(f"{wr2*100:.1f}%" if n2 else "—")
        self.short_pf_lbl.setText(f"{pf2:.2f}" if n2 else "—")
        self._set_money_label(self.short_pnl_lbl, pnl2, bold=True)
        self.short_avgw_lbl.setText(f"+{aw2:.2f}$" if nw2 else "—")
        self.short_avgl_lbl.setText(f"{al2:.2f}$" if nl2 else "—")

        tot_n, tot_w, tot_l = n + n2, nw + nw2, nl + nl2
        tot_wr = (tot_w / tot_n) if tot_n else 0.0
        gw = sum(self.session_stats["long"]["wins"]) + sum(self.session_stats["short"]["wins"])
        gl = abs(sum(self.session_stats["long"]["losses"]) + sum(self.session_stats["short"]["losses"]))
        tot_pf = (gw / gl) if gl > 1e-8 else 0.0

        self.total_trades_lbl.setText(str(tot_n))
        self.total_wins_lbl.setText(str(tot_w))
        self.total_losses_lbl.setText(str(tot_l))
        self.total_wr_lbl.setText(f"{tot_wr*100:.1f}%" if tot_n else "—")
        self.total_pf_lbl.setText(f"{tot_pf:.2f}" if tot_n else "—")
        self._set_money_label(self.total_pnl_lbl, pnl + pnl2, bold=True)

        # Jauge et compteur de trades
        self.arc.set_valeur(tot_wr if tot_n else None, tot_n)
        ecart = (tot_wr - SEUIL_EQUILIBRE) * 100 if tot_n else 0.0
        self.tile_trades.set(
            str(tot_n),
            f"{tot_w} gagnants · {tot_l} perdants" if tot_n else "aucun trade fermé",
        )
        if tot_n:
            self.arc.setToolTip(
                f"{tot_wr*100:.1f} % contre {SEUIL_EQUILIBRE*100:.1f} % requis "
                f"pour couvrir les frais ({ecart:+.1f} points)")

        ecoule = dt.datetime.now() - self.session_start
        h = int(ecoule.total_seconds() // 3600)
        m = int((ecoule.total_seconds() % 3600) // 60)
        self.session_info_lbl.setText(
            f"depuis {self.session_start:%H:%M:%S} · {h}h{m:02d}")

    # --------------------------------------------------------
    # Compte
    # --------------------------------------------------------

    def update_equity(self):
        try:
            info = mt5.account_info()
            if info is None:
                self.equity_label.setText("MT5 non connecté")
                self.tile_equity.set("—", "terminal injoignable", C["dim"])
                return

            equity  = float(info.equity)
            balance = float(info.balance)
            margin  = float(info.margin)
            free    = float(info.margin_free)
            pl_flot = equity - balance

            if self.session_balance_start is None:
                self.session_balance_start = balance

            pl_session = balance - self.session_balance_start
            pl_total   = pl_session + pl_flot

            self.spark.ajoute(equity)

            def teinte(x):
                return C["win"] if x > 0 else (C["loss"] if x < 0 else C["dim"])

            self.tile_equity.set(
                f"{equity:,.2f}".replace(",", " "),
                f"balance {balance:,.2f} · total {pl_total:+.2f}".replace(",", " "))
            self.tile_pnl.set(
                f"{pl_session:+,.2f}".replace(",", " "),
                f"positions fermées depuis {self.session_start:%H:%M}",
                teinte(pl_session))
            self.tile_flot.set(
                f"{pl_flot:+,.2f}".replace(",", " "),
                f"marge {margin:,.2f} · libre {free:,.0f}".replace(",", " "),
                teinte(pl_flot))

            self.equity_label.setText(
                f"marge {margin:.2f}   libre {free:.2f}   flottant {pl_flot:+.2f}")
        except Exception as e:
            self.equity_label.setText(f"lecture impossible : {e}")
            self.tile_equity.set("—", "erreur", C["loss"])

    # --------------------------------------------------------
    # Fermeture
    # --------------------------------------------------------

    def closeEvent(self, event: QtGui.QCloseEvent):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        if self.agent._running:
            self._append_log("[GUI] Fermeture de la fenêtre → arrêt de l'agent…",
                             LogLevel.WARN)
            self.agent.stop()
        event.accept()


def main():
    # Pas de AA_EnableHighDpiScaling : obsolete sous Qt6, ou la mise a l'echelle
    # est toujours active. L'appeler n'ajoutait qu'un DeprecationWarning.
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(stylesheet())
    app.setFont(font_ui(10))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
