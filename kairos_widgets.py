"""Widgets peints a la main pour KAIROS.

Qt ne fournit ni jauge en arc, ni courbe compacte, ni temoin lumineux. Les
dessiner soi-meme coute une centaine de lignes et evite d'embarquer une
bibliotheque de graphiques pour trois formes.

Tous respectent la meme regle : ils ne calculent rien. On leur donne des
valeurs deja etablies, ils les montrent.
"""

import math
from collections import deque

from PySide6 import QtCore, QtGui, QtWidgets

from kairos_theme import C, font_ui, font_mono


class StatusOrb(QtWidgets.QWidget):
    """Temoin d'etat : un point qui respire quand l'agent tourne.

    La pulsation n'est pas decorative — c'est le seul element anime de
    l'interface, donc l'oeil y revient tout seul pour savoir si le bot vit.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self._actif = False
        self._phase = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)

    def set_actif(self, actif: bool):
        if actif == self._actif:
            return
        self._actif = actif
        if actif:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _tick(self):
        self._phase = (self._phase + 0.045) % 1.0
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2
        base = QtGui.QColor(C["kairos"] if self._actif else C["dim"])

        if self._actif:
            # Halo dont le rayon suit une sinusoide : plus lisible qu'un
            # clignotement, et moins fatigant sur une session longue.
            t = 0.5 - 0.5 * math.cos(2 * math.pi * self._phase)
            rayon = 4.5 + 3.5 * t
            halo = QtGui.QColor(base)
            halo.setAlphaF(0.34 * (1.0 - t))
            p.setBrush(halo)
            p.setPen(QtCore.Qt.NoPen)
            p.drawEllipse(QtCore.QPointF(cx, cy), rayon, rayon)

        p.setBrush(base)
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(QtCore.QPointF(cx, cy), 3.6, 3.6)
        p.end()


class Sparkline(QtWidgets.QWidget):
    """Courbe d'equity de la session, remplie sous la ligne.

    Pas d'axes : sur cette hauteur ils seraient illisibles. La valeur exacte
    est affichee en toutes lettres a cote ; la courbe ne sert qu'a montrer la
    FORME — pente, decrochages, plateau.
    """

    def __init__(self, maxlen=600, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(74)
        self._pts = deque(maxlen=maxlen)
        self._ref = None            # equity de reference (debut de session)

    def ajoute(self, valeur: float):
        if self._ref is None:
            self._ref = float(valeur)
        self._pts.append(float(valeur))
        self.update()

    def reinit(self):
        self._pts.clear()
        self._ref = None
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        marge = 6

        if len(self._pts) < 2:
            p.setPen(QtGui.QColor(C["dim"]))
            p.setFont(font_ui(9))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter,
                       "en attente de la première lecture MT5")
            p.end()
            return

        vals = list(self._pts)
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            lo, hi = lo - 1.0, hi + 1.0
        etendue = hi - lo

        def xy(i, v):
            x = marge + (w - 2 * marge) * i / (len(vals) - 1)
            y = marge + (h - 2 * marge) * (1.0 - (v - lo) / etendue)
            return QtCore.QPointF(x, y)

        gagne = vals[-1] >= (self._ref if self._ref is not None else vals[0])
        teinte = QtGui.QColor(C["win"] if gagne else C["loss"])

        # Ligne de reference : le niveau de depart de la session.
        if self._ref is not None and lo <= self._ref <= hi:
            yref = marge + (h - 2 * marge) * (1.0 - (self._ref - lo) / etendue)
            stylo = QtGui.QPen(QtGui.QColor(C["line_hi"]), 1, QtCore.Qt.DashLine)
            p.setPen(stylo)
            p.drawLine(QtCore.QPointF(marge, yref), QtCore.QPointF(w - marge, yref))

        chemin = QtGui.QPainterPath()
        chemin.moveTo(xy(0, vals[0]))
        for i, v in enumerate(vals[1:], start=1):
            chemin.lineTo(xy(i, v))

        # Aplat degrade sous la courbe.
        aire = QtGui.QPainterPath(chemin)
        aire.lineTo(QtCore.QPointF(w - marge, h))
        aire.lineTo(QtCore.QPointF(marge, h))
        aire.closeSubpath()
        grad = QtGui.QLinearGradient(0, marge, 0, h)
        haut = QtGui.QColor(teinte); haut.setAlphaF(0.26)
        bas = QtGui.QColor(teinte); bas.setAlphaF(0.0)
        grad.setColorAt(0.0, haut)
        grad.setColorAt(1.0, bas)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(grad)
        p.drawPath(aire)

        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(teinte, 1.7, QtCore.Qt.SolidLine,
                            QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        p.drawPath(chemin)

        # Dernier point marque : c'est la valeur qui compte.
        dernier = xy(len(vals) - 1, vals[-1])
        halo = QtGui.QColor(teinte); halo.setAlphaF(0.30)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(halo)
        p.drawEllipse(dernier, 5.5, 5.5)
        p.setBrush(teinte)
        p.drawEllipse(dernier, 2.6, 2.6)
        p.end()


class WinrateArc(QtWidgets.QWidget):
    """Jauge en arc : winrate courant, avec le SEUIL D'EQUILIBRE marque.

    Le seuil est le vrai sujet. Un winrate de 38 % ne veut rien dire dans
    l'absolu ; il veut dire « cinq points sous le point mort ». La jauge montre
    donc les deux, et l'arc change de couleur quand il franchit le repere.
    """

    EPAISSEUR = 9.0

    def __init__(self, seuil=0.437, parent=None):
        super().__init__(parent)
        self.setMinimumSize(158, 112)
        self._wr = None
        self._seuil = seuil
        self._n = 0

    def set_valeur(self, winrate, n_trades: int):
        self._wr = winrate
        self._n = n_trades
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        marge = 8.0
        e = self.EPAISSEUR

        # Le demi-disque occupe une boite de 2r x r, plus la demi-epaisseur du
        # trait de chaque cote. On centre cette boite : sans ca l'arc deborde
        # des que la carte change de proportions.
        rayon = min((w - 2 * marge - e) / 2.0, h - 2 * marge - e)
        if rayon < 18:
            p.end()
            return
        cx = w / 2.0
        cy = (h - rayon) / 2.0 + rayon
        cadre = QtCore.QRectF(cx - rayon, cy - rayon, rayon * 2, rayon * 2)
        depart = 180 * 16                           # demi-cercle, sens horaire

        p.setPen(QtGui.QPen(QtGui.QColor(C["line"]), e, QtCore.Qt.SolidLine,
                            QtCore.Qt.FlatCap))
        p.drawArc(cadre, depart, -180 * 16)

        if self._wr is not None and self._n > 0:
            atteint = self._wr >= self._seuil
            teinte = QtGui.QColor(C["win"] if atteint else C["warn"])
            p.setPen(QtGui.QPen(teinte, e, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap))
            p.drawArc(cadre, depart, int(-180 * 16 * min(max(self._wr, 0.0), 1.0)))

        # Repere du seuil d'equilibre : il traverse le trait de part en part.
        ang = math.pi * (1.0 - self._seuil)
        r1, r2 = rayon - e / 2 - 1.5, rayon + e / 2 + 1.5
        p.setPen(QtGui.QPen(QtGui.QColor(C["ink"]), 1.6))
        p.drawLine(QtCore.QPointF(cx + r1 * math.cos(ang), cy - r1 * math.sin(ang)),
                   QtCore.QPointF(cx + r2 * math.cos(ang), cy - r2 * math.sin(ang)))

        # Textes places par rapport au RAYON, pour rester dans le disque quelle
        # que soit la taille allouee.
        taille_val = max(11.0, min(17.0, rayon * 0.30))
        if self._wr is None or self._n == 0:
            p.setPen(QtGui.QColor(C["dim"]))
            p.setFont(font_mono(taille_val))
            txt = "—"
        else:
            p.setPen(QtGui.QColor(C["ink"]))
            p.setFont(font_mono(taille_val, QtGui.QFont.DemiBold))
            txt = f"{self._wr * 100:.1f}%"
        p.drawText(QtCore.QRectF(cx - rayon * 0.9, cy - rayon * 0.68, rayon * 1.8, rayon * 0.36),
                   QtCore.Qt.AlignCenter, txt)

        p.setPen(QtGui.QColor(C["dim"]))
        p.setFont(font_ui(max(7.0, min(8.0, rayon * 0.145)), spacing=0.7))
        p.drawText(QtCore.QRectF(cx - rayon * 0.95, cy - rayon * 0.32, rayon * 1.9, rayon * 0.26),
                   QtCore.Qt.AlignCenter,
                   f"ÉQUILIBRE {self._seuil * 100:.1f}%")
        p.end()


class MetricTile(QtWidgets.QFrame):
    """Une mesure : intitule discret, valeur dominante, precision en dessous."""

    def __init__(self, titre: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(16, 13, 16, 14)
        v.setSpacing(3)

        self.lbl_titre = QtWidgets.QLabel(titre.upper())
        self.lbl_titre.setObjectName("CardTitle")
        self.lbl_titre.setFont(font_ui(8, QtGui.QFont.DemiBold, spacing=1.3))

        self.lbl_valeur = QtWidgets.QLabel("—")
        self.lbl_valeur.setFont(font_mono(19, QtGui.QFont.DemiBold))
        self.lbl_valeur.setStyleSheet(f"color: {C['ink']};")

        self.lbl_note = QtWidgets.QLabel("")
        self.lbl_note.setFont(font_ui(8.5))
        self.lbl_note.setStyleSheet(f"color: {C['dim']};")

        # Intitule en haut, chiffre et precision ancres en bas : quand la rangee
        # s'etire a la hauteur de la jauge voisine, le vide va entre les deux
        # plutot qu'autour du nombre.
        v.addWidget(self.lbl_titre)
        v.addStretch()
        v.addWidget(self.lbl_valeur)
        v.addWidget(self.lbl_note)

    def set(self, valeur: str, note: str = "", couleur: str = None):
        self.lbl_valeur.setText(valeur)
        self.lbl_valeur.setStyleSheet(f"color: {couleur or C['ink']};")
        self.lbl_note.setText(note)


class Card(QtWidgets.QFrame):
    """Panneau titre. Le titre est un intitule, pas un cadre : pas de QGroupBox,
    dont le rendu natif casse toute charte un peu tenue."""

    def __init__(self, titre: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.v = QtWidgets.QVBoxLayout(self)
        self.v.setContentsMargins(16, 13, 16, 14)
        self.v.setSpacing(10)
        if titre:
            self.entete = QtWidgets.QHBoxLayout()
            self.entete.setSpacing(8)
            lbl = QtWidgets.QLabel(titre.upper())
            lbl.setObjectName("CardTitle")
            lbl.setFont(font_ui(8, QtGui.QFont.DemiBold, spacing=1.3))
            self.entete.addWidget(lbl)
            self.entete.addStretch()
            self.v.addLayout(self.entete)

    def ajoute(self, w, stretch=0):
        self.v.addWidget(w, stretch)

    def ajoute_layout(self, l):
        self.v.addLayout(l)


def filet(horizontal=True) -> QtWidgets.QFrame:
    f = QtWidgets.QFrame()
    f.setFrameShape(QtWidgets.QFrame.HLine if horizontal else QtWidgets.QFrame.VLine)
    f.setStyleSheet(f"background: {C['line']}; border: none;")
    if horizontal:
        f.setFixedHeight(1)
    else:
        f.setFixedWidth(1)
    return f
