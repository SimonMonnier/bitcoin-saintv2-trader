"""Charte visuelle de KAIROS — palette, typographie, feuille de style Qt.

Isolee du code metier pour qu'une retouche esthetique ne touche jamais la
logique de trading. Les couleurs SEMANTIQUES (gain, perte, achat, vente) sont
distinctes de la couleur de MARQUE : melanger les deux rend un tableau illisible
des qu'on change d'accent.
"""

from PySide6 import QtGui

# ============================================================
# PALETTE
#
# Fond presque noir legerement bleute, pas un gris neutre : un instrument de
# mesure se lit mieux sur une base froide, et les verts/rouges de PnL y gagnent
# en separation. Les surfaces montent par paliers de luminosite plutot que par
# bordures, ce qui evite le quadrillage de cadres.
# ============================================================
C = {
    "void":    "#07090B",   # fond applicatif
    "panel":   "#0E1319",   # carte
    "raised":  "#151D25",   # carte survolee / champ
    "sunk":    "#0A0E12",   # console, zones creuses
    "line":    "#1B242D",   # filets
    "line_hi": "#26333E",   # filets accentues

    "ink":     "#DDE7EF",   # texte principal
    "muted":   "#7C8D9B",   # texte secondaire
    "dim":     "#4C5A67",   # texte tertiaire, unites

    # Marque — le teal d'un instrument sous tension.
    "kairos":  "#00D8B4",
    "kairos_d": "#00A88B",  # variante pressee
    "kairos_g": "#0A2C2A",  # fond teinte pour etats actifs

    # Semantique — jamais utilisee pour l'interface elle-meme.
    "long":    "#3ED598",
    "short":   "#5AA9FF",
    "win":     "#3ED598",
    "loss":    "#FF5C6E",
    "warn":    "#FFB13D",
    "neutral": "#5A6B78",
}

# Niveaux de journal : la severite se lit a la couleur, pas au texte.
LEVEL_COLORS = {
    "ERROR":  C["loss"],
    "WARN":   C["warn"],
    "TRADE":  C["long"],
    "SIGNAL": C["short"],
    "INFO":   C["muted"],
    "DEBUG":  C["dim"],
}


def font_ui(size=10, weight=QtGui.QFont.Normal, spacing=0.0):
    """Police d'interface. Segoe UI Variable si presente, sinon Segoe UI."""
    f = QtGui.QFont()
    f.setFamilies(["Segoe UI Variable Display", "Segoe UI", "Inter", "Arial"])
    f.setPointSizeF(size)
    f.setWeight(weight)
    if spacing:
        f.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, spacing)
    return f


def font_mono(size=10, weight=QtGui.QFont.Normal):
    """Police a chasse fixe pour les CHIFFRES.

    Tous les nombres de cette interface s'alignent en colonne : une chasse
    proportionnelle les ferait danser d'une mise a jour a l'autre.
    """
    f = QtGui.QFont()
    f.setFamilies(["Cascadia Mono", "Cascadia Code", "Consolas", "Courier New"])
    f.setPointSizeF(size)
    f.setWeight(weight)
    return f


# ============================================================
# FEUILLE DE STYLE
#
# QSS ne connait pas les variables CSS : on la construit par interpolation.
# ============================================================
def stylesheet() -> str:
    return f"""
QWidget {{
    background: transparent;
    color: {C['ink']};
}}
QMainWindow, #Root {{
    background: {C['void']};
}}

/* ---------- Cartes ---------- */
#Card {{
    background: {C['panel']};
    border: 1px solid {C['line']};
    border-radius: 10px;
}}
#CardTitle {{
    color: {C['dim']};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.4px;
}}
#TopBar {{
    background: {C['panel']};
    border: 1px solid {C['line']};
    border-radius: 12px;
}}

/* ---------- Boutons ---------- */
QPushButton {{
    background: {C['raised']};
    color: {C['ink']};
    border: 1px solid {C['line_hi']};
    border-radius: 7px;
    padding: 7px 14px;
    font-size: 12px;
}}
QPushButton:hover {{
    background: {C['line']};
    border-color: {C['dim']};
}}
QPushButton:pressed {{
    background: {C['sunk']};
}}
QPushButton:disabled {{
    color: {C['dim']};
    border-color: {C['line']};
    background: {C['panel']};
}}

#PrimaryBtn {{
    background: {C['kairos']};
    color: {C['void']};
    border: 1px solid {C['kairos']};
    font-weight: 700;
    padding: 9px 22px;
}}
#PrimaryBtn:hover  {{ background: #16E6C4; border-color: #16E6C4; }}
#PrimaryBtn:pressed{{ background: {C['kairos_d']}; }}
#PrimaryBtn:disabled {{
    background: {C['kairos_g']};
    color: {C['kairos_d']};
    border-color: {C['kairos_g']};
}}

#DangerBtn {{
    background: transparent;
    color: {C['loss']};
    border: 1px solid #40242A;
    font-weight: 600;
    padding: 9px 22px;
}}
#DangerBtn:hover {{ background: #1C1013; border-color: {C['loss']}; }}
#DangerBtn:disabled {{ color: #4A2F34; border-color: #241A1C; background: transparent; }}

/* ---------- Pastilles de filtre ---------- */
QPushButton#Chip {{
    background: transparent;
    border: 1px solid {C['line_hi']};
    border-radius: 11px;
    padding: 3px 11px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.6px;
    color: {C['dim']};
}}
QPushButton#Chip:hover {{ border-color: {C['muted']}; }}
QPushButton#Chip:checked {{ background: {C['raised']}; }}

/* ---------- Champs ---------- */
QLineEdit {{
    background: {C['sunk']};
    border: 1px solid {C['line']};
    border-radius: 7px;
    padding: 6px 11px;
    color: {C['ink']};
    selection-background-color: {C['kairos_d']};
    selection-color: {C['void']};
}}
QLineEdit:focus {{ border-color: {C['kairos_d']}; }}

/* ---------- Console ---------- */
QTextEdit#Console {{
    background: {C['sunk']};
    border: 1px solid {C['line']};
    border-radius: 9px;
    padding: 9px 11px;
    selection-background-color: {C['kairos_d']};
    selection-color: {C['void']};
}}

/* ---------- Ascenseurs ---------- */
QScrollBar:vertical {{
    background: transparent; width: 11px; margin: 3px 2px 3px 0;
}}
QScrollBar::handle:vertical {{
    background: {C['line_hi']}; border-radius: 4px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {C['dim']}; }}
QScrollBar:horizontal {{
    background: transparent; height: 11px; margin: 0 3px 2px 3px;
}}
QScrollBar::handle:horizontal {{
    background: {C['line_hi']}; border-radius: 4px; min-width: 28px;
}}
QScrollBar::handle:horizontal:hover {{ background: {C['dim']}; }}
QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; border: none; height: 0; width: 0; }}

QToolTip {{
    background: {C['raised']};
    color: {C['ink']};
    border: 1px solid {C['line_hi']};
    border-radius: 6px;
    padding: 5px 9px;
}}
"""
