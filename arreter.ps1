# KAIROS — arreter l'entrainement, sans rien effacer.
#
# IL NE TOUCHE NI AUX JOURNAUX NI AUX POINTS DE REPRISE. Fermer la fenetre
# de l'entrainement fait la meme chose ; ce script existe pour les cas ou
# elle a ete perdue de vue, ou quand plusieurs se sont accumulees.
#
#   double-clic sur ARRETER.bat
#   ou bien :  powershell -ExecutionPolicy Bypass -File arreter.ps1

Set-Location -LiteralPath $PSScriptRoot

# ON CIBLE PAR LIGNE DE COMMANDE, PAS PAR NOM. `python.exe` peut tourner
# pour dix raisons sur cette machine — une mesure, un preparateur de cache,
# un autre projet. Seul `training.py` dans la ligne de commande identifie
# l'entrainement.
$procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like '*training.py*' })

Write-Host ''
if ($procs.Count -eq 0) {
    Write-Host '  aucun entrainement en cours' -ForegroundColor DarkGray
} else {
    foreach ($p in $procs) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "  entrainement arrete : pid $($p.ProcessId)" -ForegroundColor Yellow
    }
}
Write-Host '  journaux et points de reprise intacts' -ForegroundColor DarkGray
Write-Host ''
Start-Sleep -Seconds 3
