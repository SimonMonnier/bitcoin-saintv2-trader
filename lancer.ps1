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

$journal = Join-Path $PSScriptRoot 'training_porte90_tp1.log'
$prefixe = 'kairos_jeu_m5_23_porte90_tp1'
# LE JEU DEPUIS LE 2026-09-26 : `jeu_kairos.py` remplace `training.py`.
$script = 'jeu_kairos.py'

Write-Host ''
Write-Host '  KAIROS - lancement' -ForegroundColor Cyan
Write-Host ''

# --- 1. ARRETER CE QUI TOURNE -----------------------------------------
# Deux entrainements en parallele ecrivent dans le MEME journal et se
# disputent le GPU. Le second parait lent sans raison.
$vieux = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like '*training.py*' -or
                          $_.CommandLine -like '*jeu_kairos.py*' })
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
    # Les secondes evitent qu'une relance rapide ecrase le journal archive de
    # la minute precedente. Un journal est une preuve de run, pas un cache.
    $nom = 'training_btc_{0:yyyyMMdd_HHmmss}.log' -f (Get-Date)
    Move-Item -LiteralPath $journal -Destination (Join-Path $dossier $nom)
    Write-Host "  journal precedent archive : journaux\$nom" -ForegroundColor DarkGray
}

# --- 3. EFFACER LES MANIFESTES ----------------------------------------
# `run_training_on_split` REFUSE de demarrer si le manifeste du run existe
# deja — c'est une protection contre l'ecrasement silencieux d'un run.
# Elle fait donc echouer toute relance tant qu'on ne l'a pas levee.
$n = 0
foreach ($motif in @("run_$prefixe*.json", "best_$prefixe*.pth", "bestR_$prefixe*.pth",
                     "last_$prefixe*.pth", "bestprofit_$prefixe*.pth",
                     "deploy_$prefixe*.pth", "deploy_$prefixe*.json",
                     "test_$prefixe*.json", "pipeline_$prefixe*.json",
                     "pipeline_$prefixe*_predictions.npz", "trades_$prefixe*.csv")) {
    foreach ($f in @(Get-ChildItem -LiteralPath $PSScriptRoot -Filter $motif -ErrorAction SilentlyContinue)) {
        $archive = '{0:yyyyMMdd_HHmmss}_{1}' -f (Get-Date), $f.Name
        Move-Item -LiteralPath $f.FullName -Destination (Join-Path $dossier $archive)
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

# LA SORTIE EST REDIRIGEE, PAS DEDOUBLEE PAR `Tee-Object`.
#
# Premiere version : `python ... 2>&1 | Tee-Object -FilePath $journal`,
# pour voir le texte dans la fenetre ET l'ecrire. Resultat mesure :
# l'entrainement s'arretait apres six lignes de banniere, journal de
# 655 octets, aucun processus Python vivant. `Tee-Object` met le flux en
# tampon et le pipeline meurt avec lui.
#
# ON REDIRIGE DONC, comme tout le reste de ce depot le fait. La fenetre de
# l'entrainement reste ouverte et muette ; c'est la VEILLE qui affiche, et
# c'est son metier.
# LA REDIRECTION PASSE PAR `cmd`, ET C'EST LE POINT LE PLUS SUBTIL.
#
# `*>` et `>` de PowerShell 5.1 ecrivent en UTF-16 LE, avec un octet
# nul entre chaque lettre. La veille lit en UTF-8 et n'y voit que du
# charabia — c'est exactement la panne « la veille ne marche pas » du
# 2026-09-22. Le `chcp 65001` n'y change rien : il regle la console,
# pas l'encodage de la redirection PowerShell.
#
# `cmd /c` ecrit les OCTETS BRUTS que Python produit. Avec
# `PYTHONIOENCODING=utf-8`, le journal est donc en UTF-8 et les
# accents survivent. C'est ce que faisaient les lancements manuels de
# cette session, qui n'ont jamais eu ce probleme.
$cmd = "`$env:PYTHONUNBUFFERED='1'; `$env:PYTHONIOENCODING='utf-8'; " +
       "Write-Host '  entrainement en cours - la lecture se fait dans la veille' -ForegroundColor Green; " +
       "Write-Host '  NE PAS FERMER cette fenetre' -ForegroundColor Yellow; " +
       "cmd /c '$python $script > training_porte90_tp1.log 2>&1'"
# SANS GUILLEMETS INTERIEURS, et le journal en chemin RELATIF - 2026-09-26.
# La version precedente citait l'interpreteur et le journal : PowerShell 5.1
# a retire ces guillemets en passant la commande a `cmd`, et le chemin du
# depot, qui contient des espaces, a ete coupe au premier. La sortie est
# partie dans un fichier `C:\Users\smonn\Desktop\ia\model`, et la veille
# n'a rien vu. L'interpreteur n'a pas d'espace ; le journal est ecrit dans
# le repertoire de travail, qui est $PSScriptRoot (voir -WorkingDirectory).
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
Start-Process powershell -ArgumentList '-NoExit','-ExecutionPolicy','Bypass','-File','veille_fenetre.ps1','-Journal','training_porte90_tp1.log' `
    -WorkingDirectory $PSScriptRoot
Write-Host '  veille ouverte dans sa fenetre' -ForegroundColor Green

Write-Host ''
Write-Host '  Les deux fenetres sont independantes de celle-ci.' -ForegroundColor DarkGray
Write-Host '  Fermer la VEILLE n arrete pas l entrainement.' -ForegroundColor DarkGray
Write-Host '  Pour arreter l entrainement : fermer SA fenetre, ou relancer ce script.' -ForegroundColor DarkGray
Write-Host ''
Start-Sleep -Seconds 4
