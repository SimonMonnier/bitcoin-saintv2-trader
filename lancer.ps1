# KAIROS — lancer l'entrainement ET la veille, sans Claude.
#
# POURQUOI CE FICHIER EXISTE. Tout ce qui suit a ete tape a la main des
# dizaines de fois dans cette session : archiver le journal, effacer les
# manifestes du run precedent, lancer Python avec le bon interpreteur et le
# bon encodage, puis ouvrir la fenetre de veille. Une seule de ces etapes
# oubliee et le run meurt — ou pire, il tourne en mentant.
#
#   clic droit -> "Executer avec PowerShell"
#   ou bien :   powershell -ExecutionPolicy Bypass -File lancer.ps1
#
# CE QU'IL FAIT, DANS L'ORDRE :
#   1. arrete l'entrainement en cours, s'il y en a un
#   2. archive le journal dans journaux/ avec la date
#   3. efface les manifestes et points de reprise du meme prefixe
#   4. lance l'entrainement dans SA fenetre
#   5. ouvre la veille dans une SECONDE fenetre
#
# RIEN N'EST PERDU : les journaux et les points de reprise partent dans
# journaux/, ils ne sont pas supprimes.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# L'INTERPRETEUR EST CELUI DE L'ENVIRONNEMENT `trading_env`, pas celui du
# PATH. Un `python` generique n'a ni torch, ni MetaTrader5, et l'erreur
# n'arrive qu'apres trente secondes de chargement.
$python = 'C:\Users\smonn\miniconda3\envs\trading_env\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host ''
    Write-Host "  INTERPRETEUR INTROUVABLE : $python" -ForegroundColor Red
    Write-Host '  Ouvrir ce fichier et corriger la ligne $python.' -ForegroundColor Yellow
    Read-Host '  Entree pour fermer'
    exit 1
}

$journal = Join-Path $PSScriptRoot 'training_btc.log'
$prefixe = 'saintv2_btc_m1_flux01'

Write-Host ''
Write-Host '  KAIROS - lancement' -ForegroundColor Cyan
Write-Host ''

# --- 1. ARRETER CE QUI TOURNE -----------------------------------------
# Deux entrainements en parallele ecrivent dans le MEME journal et se
# disputent le GPU. Le second parait lent sans raison.
$vieux = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like '*training.py*' })
if ($vieux.Count -gt 0) {
    foreach ($v in $vieux) {
        Stop-Process -Id $v.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "  entrainement precedent arrete (pid $($v.ProcessId))" -ForegroundColor DarkGray
    }
    Start-Sleep -Seconds 2
}

# --- 2. ARCHIVER LE JOURNAL -------------------------------------------
$dossier = Join-Path $PSScriptRoot 'journaux'
if (-not (Test-Path -LiteralPath $dossier)) {
    New-Item -ItemType Directory -Path $dossier | Out-Null
}
if (Test-Path -LiteralPath $journal) {
    $nom = 'training_btc_{0:yyyyMMdd_HHmm}.log' -f (Get-Date)
    Move-Item -LiteralPath $journal -Destination (Join-Path $dossier $nom) -Force
    Write-Host "  journal precedent archive : journaux\$nom" -ForegroundColor DarkGray
}

# --- 3. EFFACER LES MANIFESTES ----------------------------------------
# `run_training_on_split` REFUSE de demarrer si le manifeste du run existe
# deja — c'est une protection contre l'ecrasement silencieux d'un run.
# Elle fait donc echouer toute relance tant qu'on ne l'a pas levee.
$n = 0
foreach ($motif in @("run_$prefixe*.json", "best_$prefixe*.pth",
                     "last_$prefixe*.pth", "bestprofit_$prefixe*.pth")) {
    foreach ($f in @(Get-ChildItem -LiteralPath $PSScriptRoot -Filter $motif -ErrorAction SilentlyContinue)) {
        Move-Item -LiteralPath $f.FullName -Destination (Join-Path $dossier $f.Name) -Force
        $n++
    }
}
if ($n -gt 0) { Write-Host "  $n fichier(s) du run precedent deplaces dans journaux\" -ForegroundColor DarkGray }

# --- 4. LANCER L'ENTRAINEMENT -----------------------------------------
# `PYTHONUNBUFFERED` : sans lui, Python garde sa sortie en tampon et la
# veille ne voit rien pendant des minutes.
# `PYTHONIOENCODING` : sans lui, les accents du journal cassent sur une
# console qui n'est pas en UTF-8.
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'

$cmd = "chcp 65001 > `$null; `$env:PYTHONUNBUFFERED='1'; `$env:PYTHONIOENCODING='utf-8'; " +
       "& '$python' training.py 2>&1 | Tee-Object -FilePath '$journal'"
Start-Process powershell -ArgumentList '-NoExit','-ExecutionPolicy','Bypass','-Command',$cmd `
    -WorkingDirectory $PSScriptRoot
Write-Host '  entrainement lance dans sa fenetre' -ForegroundColor Green

# --- 5. OUVRIR LA VEILLE ----------------------------------------------
# On ferme d'abord les anciennes : plusieurs fenetres sur le meme journal
# ne servent a rien, et une fenetre restee ouverte sur un journal ARCHIVE
# donne l'illusion que l'entrainement est fige.
foreach ($v in @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
                 Where-Object { $_.CommandLine -like '*veille_fenetre*' })) {
    Stop-Process -Id $v.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 3
Start-Process powershell -ArgumentList '-NoExit','-ExecutionPolicy','Bypass','-File','veille_fenetre.ps1' `
    -WorkingDirectory $PSScriptRoot
Write-Host '  veille ouverte dans sa fenetre' -ForegroundColor Green

Write-Host ''
Write-Host '  Les deux fenetres sont independantes de celle-ci.' -ForegroundColor DarkGray
Write-Host '  Fermer la VEILLE n arrete pas l entrainement.' -ForegroundColor DarkGray
Write-Host '  Pour arreter l entrainement : fermer SA fenetre, ou relancer ce script.' -ForegroundColor DarkGray
Write-Host ''
Start-Sleep -Seconds 4
