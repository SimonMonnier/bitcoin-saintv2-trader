@echo off
REM KAIROS - ouvrir SEULEMENT la fenetre de veille.
REM Lecture seule : n arrete ni ne demarre aucun entrainement.
REM Dans la console classique (conhost) : Windows Terminal dessine avec le GPU,
REM et ses fenetres restent transparentes quand l entrainement l occupe (2026-10-04).
start "KAIROS veille" conhost.exe powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0veille_fenetre.ps1" -Journal "training_retour_m5_31.log"
