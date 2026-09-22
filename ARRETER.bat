@echo off
REM KAIROS - arreter l entrainement en cours, sans rien effacer.
REM Le travail est dans arreter.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0arreter.ps1"
