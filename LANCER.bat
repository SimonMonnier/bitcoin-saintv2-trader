@echo off
REM KAIROS - double-cliquer pour lancer entrainement + veille.
REM Le travail est dans lancer.ps1 ; ce fichier n existe que pour le
REM double-clic, que Windows ne permet pas sur un .ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0lancer.ps1"
