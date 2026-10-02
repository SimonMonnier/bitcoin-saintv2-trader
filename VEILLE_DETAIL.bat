@echo off
REM KAIROS - la veille en vue DETAILLEE : 26 lignes par epoch.
REM Pour comprendre POURQUOI ca ne marche pas. VEILLE.bat suffit
REM pour savoir SI ca marche.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0veille_fenetre.ps1" -Detail -Journal "training_mois_serie_noire.log"
