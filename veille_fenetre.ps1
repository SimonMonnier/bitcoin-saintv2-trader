# KAIROS — la fenetre de veille.
#
# ELLE NE FAIT QUE LIRE. `training.py` fait tourner la veille EN PROCESSUS
# et ecrit son analyse complete dans `training_btc.log` ; cette fenetre s'y
# branche. Un second processus qui relirait le meme fichier a deja coute
# quatre pannes — rotation du journal, Stop-Process, Ctrl+C recu d'un autre
# groupe de console, et un gel jamais explique. Il n'apportait rien que le
# journal ne contienne deja.
#
# LE FILTRE DESHABILLE LA LIGNE AVANT DE LA TESTER. Les lignes du journal
# commencent par des codes de couleur — `ESC[93m ESC[1m EPOCH 005` — donc
# un motif ancre sur `^EPOCH` ne correspond JAMAIS. La fenetre s'ouvrait et
# restait vide, ce qui ressemblait a « la veille ne se lance pas ».
#
#   powershell -ExecutionPolicy Bypass -File veille_fenetre.ps1

$Host.UI.RawUI.WindowTitle = 'KAIROS - veille m1_scalp01'
$journal = Join-Path $PSScriptRoot 'training_btc.log'

# LA SORTIE EST EN UTF-8, sans quoi les accents du journal ressortent en
# mojibake — le meme piege que `Set-Content -Encoding utf8` sur les .py.
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

# ET LES SEQUENCES ANSI DOIVENT ETRE INTERPRETEES. Windows Terminal le
# fait par defaut, une console classique non : sans ce drapeau les codes
# de couleur s'affichent en clair au milieu du texte.
try {
    $sig = '[DllImport("kernel32.dll")] public static extern bool GetConsoleMode(IntPtr h, out uint m);
            [DllImport("kernel32.dll")] public static extern bool SetConsoleMode(IntPtr h, uint m);
            [DllImport("kernel32.dll")] public static extern IntPtr GetStdHandle(int n);'
    $k = Add-Type -MemberDefinition $sig -Name 'Vt' -Namespace 'Kairos' -PassThru
    $h = $k::GetStdHandle(-11)
    $m = 0
    if ($k::GetConsoleMode($h, [ref]$m)) { $k::SetConsoleMode($h, $m -bor 0x0004) | Out-Null }
} catch { }

Write-Host ''
Write-Host '  KAIROS - veille : analyse par epoch' -ForegroundColor Cyan
Write-Host "  journal : $journal" -ForegroundColor DarkGray
Write-Host '  LECTURE SEULE - fermer cette fenetre n arrete pas l entrainement' -ForegroundColor DarkGray
Write-Host ''

if (-not (Test-Path -LiteralPath $journal)) {
    Write-Host "  journal absent : $journal" -ForegroundColor Yellow
    Write-Host '  (l entrainement ne l a pas encore cree — cette fenetre attend)' -ForegroundColor DarkGray
    while (-not (Test-Path -LiteralPath $journal)) { Start-Sleep -Seconds 2 }
}

# Ce qu'on garde : le bloc d'analyse par epoch, les decisions de retenue,
# les bornes de fold, et toute panne.
$garde = '^(EPOCH |  (PnL|point mort|classement|sommet du tri|vs |sens |actions |entrees |critere net|dimension|train |\. |temps :)|          SHORT|  (non )?retenu|  garde |  . NEW BEST|=== |--- Fold |Traceback|.*Error)'

Get-Content -LiteralPath $journal -Wait -Encoding utf8 | ForEach-Object {
    # ON DESHABILLE POUR TESTER, ON AFFICHE L'ORIGINALE.
    #
    # Les codes ANSI empechent le motif de mordre — `ESC[93m ESC[1m EPOCH`
    # ne commence pas par `EPOCH` — donc le test se fait sur la ligne nue.
    # Mais l'AFFICHER nue jetait toute la couleur, et la veille s'appuie
    # dessus : le vert et le rouge portent le signe de chaque chiffre. Sans
    # eux le bloc devient un mur de nombres.
    #
    # `[Console]::Out.WriteLine` et non `Write-Host` : le premier ecrit la
    # chaine telle quelle et le terminal interprete les sequences ; le
    # second passe par la mise en forme de PowerShell, qui peut les
    # echapper selon l'hote.
    $nu = $_ -replace "$([char]27)\[[0-9;]*m", ''
    if ($nu -match $garde) { [Console]::Out.WriteLine($_) }
}
