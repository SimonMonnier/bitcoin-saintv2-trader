@echo off
REM KAIROS - la veille en vue DETAILLEE : 26 lignes par epoch.
REM Pour comprendre POURQUOI ca ne marche pas. VEILLE.bat suffit
REM pour savoir SI ca marche.
REM Dans la console classique (conhost), voir VEILLE.bat.
start "KAIROS veille detail" conhost.exe powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0veille_fenetre.ps1" -Detail -Journal "training_retour_m5_31.log"
