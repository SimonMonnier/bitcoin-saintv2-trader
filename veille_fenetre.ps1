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
# ELLE SUIT LE CHEMIN, PLUS LE FICHIER — 2026-09-22.
#
# `Get-Content -Wait` ouvre un HANDLE et le garde. Quand le journal est
# archive et remplace — ce que fait chaque relance — la fenetre continue de
# lire l'ancien fichier, qui ne grossit plus. Elle se fige sans rien dire,
# et l'entrainement a beau passer l'epoch 50, elle reste a la 7e.
#
# C'etait la PREMIERE des quatre pannes citees plus haut. Elle etait
# documentee et jamais corrigee : le commentaire decrivait le risque, le
# code le subissait.
#
# ET LA DATE DE CREATION NE SERT A RIEN POUR LE DETECTER. Premiere version
# de ce correctif : comparer `CreationTimeUtc`. Mesure immediate —
#
#     journal : 2 254 octets, cree 00:56:10, ecrit 14:35:07
#
# quatorze heures d'ecart entre creation et ecriture, sur un fichier cree
# a 14:34. C'est le TUNNELING de NTFS : supprimer un fichier et en recreer
# un du meme nom dans les quinze secondes fait REUTILISER l'horodatage de
# creation d'origine. La cle n'aurait jamais change, et le correctif
# n'aurait rien corrige.
#
# ON IDENTIFIE DONC LE FICHIER PAR SON CONTENU : les 256 premiers octets.
# Un nouveau journal commence par une banniere differente — au minimum une
# autre date — et le tunneling n'y peut rien. La taille qui RECULE reste
# verifiee en second : elle attrape la troncature sur place.
#
# DEUX VUES, ET C'EST UNE MESURE QUI LES IMPOSE — 2026-09-22.
#
# Le bloc d'analyse fait VINGT-SIX LIGNES par epoch. Sur cinquante epochs
# cela fait 1 400 lignes qui defilent, et le chiffre qu'on cherche — le
# PnL, le sommet, la decision de retenue — se noie dans les diagnostics.
# La fenetre etait devenue illisible, et les six lignes de phase ajoutees
# le matin meme n'ont fait que precipiter ce qui etait deja trop dense.
#
# PAR DEFAUT : SIX LIGNES. L'epoch, l'argent, le tri, la tenue, la
# decision, et toute panne. C'est ce qu'on lit pour savoir si ca marche.
#
# AVEC -Detail : TOUT. Les phases du reseau, la cadence, la collecte, les
# diagnostics d'entropie et d'etendue. C'est ce qu'on lit pour savoir
# POURQUOI ca ne marche pas.
#
#   powershell -ExecutionPolicy Bypass -File veille_fenetre.ps1
#   powershell -ExecutionPolicy Bypass -File veille_fenetre.ps1 -Detail

param([switch]$Detail)

$journal = Join-Path $PSScriptRoot 'training_btc.log'
$Host.UI.RawUI.WindowTitle = if ($Detail) { 'KAIROS - veille (detail)' } else { 'KAIROS - veille' }

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
Write-Host '  elle SUIT LE CHEMIN : une relance qui archive le journal ne la fige plus' -ForegroundColor DarkGray
if ($Detail) {
    Write-Host '  VUE DETAILLEE - 26 lignes par epoch' -ForegroundColor Yellow
} else {
    Write-Host '  vue ESSENTIELLE - 6 lignes par epoch   (VEILLE_DETAIL.bat pour tout voir)' -ForegroundColor DarkGray
}
Write-Host ''

# L'ESSENTIEL : six lignes par epoch, et rien d'autre.
#
#   SORTIE/CIBLE    la geometrie du run, une fois au demarrage
#   COLLECTE        l'epoch a commence — LE SIGNE DE VIE
#   PPO sortie      ce que la politique de sortie apprend
#   EPOCH           le resultat : $/trade, trades, WR, PF
#   PnL             l'argent, le cumul, le creux — cote VALIDATION
#   train           le meme cote ENTRAINEMENT, et l'ecart entre les deux
#   sens / SHORT    LONG et SHORT separes : trades, gagnants, perdants, PnL
#                   Deux lignes, et elles disent ce qu'aucune moyenne ne
#                   dit — un cote peut porter tout le resultat pendant que
#                   l'autre saigne, et le total le cacherait.
#                   C'est cet ecart qui montre le surapprentissage ; le
#                   couper revenait a ne montrer qu'une moitie du run.
#   tenue           gagnants contre perdants — la geometrie du trade
#   sommet du tri   le PORTILLON : il doit battre le hasard + la marge.
#                   Necessaire, pas suffisant. La veille l'annoncait comme
#                   "ce qui selectionne le checkpoint" : c'etait faux.
#   sommet 60 min   diagnostic : les entrees seules, sortie fixe a
#                   l'horizon de leur cible. Ne decide rien.
#   critere net     LE CRITERE DE SAUVEGARDE : gain - baisse.
#   sauvegarde      ce qu'il doit battre, et ce qu'il lui manque.
#   evolution       ses dernieres valeurs dans le fold.
#                   Ces trois lignes n'etaient PAS dans la vue essentielle.
#   LE SORT DU MODELE, et c'est la raison d'etre du run :
#     * NEW BEST      il est RETENU et transmis au fold suivant
#     a battu         ce qu'il a battu pour l'etre
#     x REFUSE        il avait un MEILLEUR score et il est rejete quand
#                     meme, sur le portefeuille. C'est le cas le plus
#                     important a voir — et il n'etait dans AUCUN motif.
#     non retenu      la raison du rejet
#     garde           rien de mieux trouve cette epoch
#   Fold / erreurs  les bornes et les pannes
#
# LES TROIS PREMIERS SONT DES SIGNES DE VIE, et leur absence a coute une
# alerte. Premiere version de cette vue : cinq lignes par epoch et RIEN
# entre deux. Une epoch dure cinq minutes ; la fenetre restait donc muette
# cinq minutes d'affilee, ce qui est indistinguable d'une fenetre cassee.
# Une veille qui ne dit pas « je travaille » ne sert a rien.
$essentiel = '^(EPOCH |  (PnL|train |sens |tenue |sommet du tri|sommet \d+ min|critere net|sauvegarde |evolution )|  phase  (PPO (entree|sortie)|marge entree|avantage brut|cout entrainement)|\[COLLECTE\]|SORTIE |CIBLE |          (une seule|AUCUN plafond|SHORT)|  (non )?retenu|  garde |  . NEW BEST|  . REFUSE|    a battu|=== |--- Fold |Traceback|.*Error)'

# LE DETAIL : tout le reste — diagnostics, phases du reseau, cadence.
$detaille = '^(EPOCH |  (PnL|point mort|classement|sommet du tri|sommet \d+ min|vs |sens |actions |entrees |critere net|sauvegarde |evolution |dimension|train |tenue |phase |cadence |\. |temps :)|\[COLLECTE\]|          SHORT|  (non )?retenu|  garde |  . NEW BEST|=== |--- Fold |Traceback|.*Error)'

$garde = if ($Detail) { $detaille } else { $essentiel }
$ansi = [regex]"$([char]27)\[[0-9;]*m"

$flux = $null
$lecteur = $null
$cle = ''
$pos = [long]0

while ($true) {
    if (-not (Test-Path -LiteralPath $journal)) {
        if ($null -ne $lecteur) {
            $lecteur.Dispose(); $flux.Dispose()
            $lecteur = $null; $flux = $null; $cle = ''
        }
        Start-Sleep -Milliseconds 500
        continue
    }

    $info = Get-Item -LiteralPath $journal -ErrorAction SilentlyContinue
    if ($null -eq $info) { Start-Sleep -Milliseconds 500; continue }

    # LA CLE IDENTIFIE LE FICHIER PAR SON CONTENU, pas par ses dates.
    # Voir l'entete : le tunneling NTFS rend `CreationTime` inutilisable.
    $nouvelle = ''
    try {
        $fs = New-Object System.IO.FileStream(
            $journal, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read,
            ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
        $buf = New-Object byte[] 256
        $lu = $fs.Read($buf, 0, 256)
        $fs.Dispose()
        if ($lu -gt 0) { $nouvelle = [Convert]::ToBase64String($buf, 0, $lu) }
    } catch { Start-Sleep -Milliseconds 400; continue }
    $tronque = ($null -ne $lecteur) -and ($info.Length -lt $pos)

    if ($nouvelle -ne $cle -or $tronque) {
        if ($null -ne $lecteur) {
            $lecteur.Dispose(); $flux.Dispose()
            Write-Host ''
            Write-Host '  --- journal remplace : on repart du debut ---' -ForegroundColor Yellow
            Write-Host ''
        }
        # `FileShare` autorise l'ecrivain a continuer — ET a supprimer le
        # fichier sous nos pieds, ce qu'une archive fait. Sans `Delete`,
        # `mv` echouerait et c'est l'ENTRAINEMENT qu'on bloquerait.
        $flux = New-Object System.IO.FileStream(
            $journal, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read,
            ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
        $lecteur = New-Object System.IO.StreamReader($flux, [System.Text.UTF8Encoding]::new())
        $cle = $nouvelle
        $pos = 0
    }

    $rien = $true
    while ($null -ne ($ligne = $lecteur.ReadLine())) {
        $rien = $false
        $pos = $flux.Position
        # ON DESHABILLE POUR TESTER, ON AFFICHE L'ORIGINALE.
        #
        # Les codes ANSI empechent le motif de mordre — `ESC[93m ESC[1m
        # EPOCH` ne commence pas par `EPOCH` — donc le test se fait sur la
        # ligne nue. Mais l'AFFICHER nue jetait toute la couleur, et la
        # veille s'appuie dessus : le vert et le rouge portent le signe de
        # chaque chiffre.
        #
        # `[Console]::Out.WriteLine` et non `Write-Host` : le premier ecrit
        # la chaine telle quelle et le terminal interprete les sequences ;
        # le second passe par la mise en forme de PowerShell, qui peut les
        # echapper selon l'hote.
        $nu = $ansi.Replace($ligne, '')
        if ($nu -match $garde) {
            # UNE LIGNE VIDE AVANT CHAQUE EPOCH. Sans separation les blocs
            # se collent et l'oeil ne retrouve plus ou commence l'epoch
            # suivante — c'est la moitie du probleme de lisibilite.
            if ($nu -match '^EPOCH ') { [Console]::Out.WriteLine('') }
            [Console]::Out.WriteLine($ligne)
        }
    }
    if ($rien) { Start-Sleep -Milliseconds 400 }
}
