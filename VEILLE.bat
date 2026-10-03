@echo off
REM KAIROS - ouvrir SEULEMENT la fenetre de veille.
REM Lecture seule : n arrete ni ne demarre aucun entrainement.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0veille_fenetre.ps1" -Journal "training_retour_m5_31.log"
