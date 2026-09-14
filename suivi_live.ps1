# Suivi en direct de l'entrainement.
#
# Lecture seule : cette fenetre n'agit pas sur le processus. La fermer
# n'arrete pas l'entrainement, et Ctrl+C non plus.
#
#   powershell -ExecutionPolicy Bypass -File suivi_live.ps1

$ErrorActionPreference = 'Continue'
$hote = $Host.UI.RawUI
$hote.WindowTitle = 'KAIROS - entrainement en direct'

$log = Join-Path $PSScriptRoot 'training_btc.log'

Write-Host ''
Write-Host '  KAIROS - suivi en direct' -ForegroundColor Cyan
Write-Host '  ---------------------------------------------------------------' -ForegroundColor DarkGray
Write-Host "  fichier : $log" -ForegroundColor DarkGray
Write-Host '  VAL  = validation     META = diagnostics' -ForegroundColor DarkGray
Write-Host '  ligne du hasard 22.8 %   seuil d''equilibre 43.7 %' -ForegroundColor DarkGray
Write-Host '  fermer cette fenetre n''arrete PAS l''entrainement' -ForegroundColor DarkGray
Write-Host '  ---------------------------------------------------------------' -ForegroundColor DarkGray
Write-Host ''

if (-not (Test-Path $log)) {
    Write-Host "  Journal introuvable. L'entrainement n'a peut-etre pas demarre." -ForegroundColor Yellow
    Read-Host '  Entree pour fermer'
    exit
}

# Les lignes portent des codes ANSI de couleur ; on les retire pour rester
# lisible quel que soit le terminal.
Get-Content -LiteralPath $log -Wait -Tail 60 | ForEach-Object {
    $ligne = $_ -replace "$([char]27)\[[0-9;]*m", ''

    if ($ligne -match 'EPOCH\s+(\d+)\s+VAL') {
        # On met en evidence le winrate et le profit factor.
        $couleur = 'White'
        if ($ligne -match 'WR\s+([0-9.]+)%') {
            $wr = [double]$Matches[1]
            if     ($wr -ge 43.7) { $couleur = 'Green' }
            elseif ($wr -ge 30.0) { $couleur = 'Yellow' }
            else                  { $couleur = 'Gray'  }
        }
        Write-Host $ligne -ForegroundColor $couleur
    }
    elseif ($ligne -match 'EPOCH\s+(\d+)\s+META') {
        # Les diagnostics : on garde les champs qui comptent.
        $champs = @()
        foreach ($cle in 'CriticL','H ','etendue\[','clipfrac','dec ','gpu\[','temps\[') {
            if ($ligne -match "($cle[^\s]*(\s[^\s]+)?)") { $champs += $Matches[1].Trim() }
        }
        Write-Host ('      ' + ($champs -join '   ')) -ForegroundColor DarkCyan
    }
    elseif ($ligne -match 'NEW BEST|Fold \d|MORT|Error|Traceback') {
        Write-Host $ligne -ForegroundColor Magenta
    }
}
