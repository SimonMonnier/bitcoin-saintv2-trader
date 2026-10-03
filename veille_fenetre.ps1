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

#   powershell -ExecutionPolicy Bypass -File veille_fenetre.ps1 -Journal training_esperance.log
#   (un second run lance en parallele, avec son propre journal - 2026-09-28)

param([switch]$Detail, [string]$Journal = 'training_esperance.log')

$journal = Join-Path $PSScriptRoot $Journal
$titre = if ($Detail) { 'KAIROS - veille (detail)' } else { 'KAIROS - veille' }
$Host.UI.RawUI.WindowTitle = "$titre - $Journal"

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
$essentiel = '^(EPOCH |  (PnL|train |sens |tenue |sommet du tri|sommet \d+ min|critere net|sauvegarde |evolution )|  phase  (PPO (entree|sortie)|marge entree|avantage brut|cout entrainement)|\[COLLECTE\]|SORTIE |CIBLE |          (une seule|AUCUN plafond|SHORT)|  (non )?retenu|  garde |  . NEW BEST|  . REFUSE|    a battu|  expert |  reprend |  bilan |TEST fold \d+ bilan |FIN |=== |--- Fold |  bloc |  blocs gagnants|  tous les blocs|  mois |  serie noire |  sorties |  esperance |  RESUME |Traceback|.*Error)'

# LE DETAIL : tout le reste — diagnostics, phases du reseau, cadence.
$detaille = '^(EPOCH |  (PnL|point mort|classement|sommet du tri|sommet \d+ min|vs |sens |actions |entrees |critere net|sauvegarde |evolution |dimension|train |tenue |phase |cadence |\. |temps :)|\[COLLECTE\]|          SHORT|  (non )?retenu|  garde |  . NEW BEST|  jeu |  bilan |TEST fold \d+ bilan |  expert |  reprend |  (normalisation|table des coups|modele|regles|un R) |TEST |FIN |=== |--- Fold |  bloc |  blocs gagnants|  tous les blocs|  mois |  serie noire |  sorties |  esperance |  RESUME |Traceback|.*Error)'

$garde = if ($Detail) { $detaille } else { $essentiel }
$ansi = [regex]"$([char]27)\[[0-9;]*m"
$esc = [char]27
$inv = [Globalization.CultureInfo]::InvariantCulture

# LE VERDICT DU JEU, EN CLAIR - 2026-09-26, demande du proprietaire : 'je
# ne comprends rien a ce qu'il faut regarder pour savoir si le modele est
# rentable'. Les lignes du jeu sont techniques ; la question, elle, est
# simple. Chaque epoch se resume donc a UNE ligne : rentable ou pas, en
# dollars par jour, sur les journees de validation que le modele n'a jamais
# vues. Vert si l'on gagne, rouge si l'on perd. La vue detaillee garde tout.
function Verdict([string]$nu) {
    # UNE PARTIE = UN JOUR OU UNE SEMAINE - 2026-09-28, jeu H1 : le jeu
    # l'annonce dans sa banniere, et chaque montant se dit dans cette unite.
    if ($nu -match 'KAIROS EN JEU.*une semaine = une partie') { $script:unite = 'semaine' }
    elseif ($nu -match 'KAIROS EN JEU') { $script:unite = 'jour' }
    $u = $script:unite
    $vus = if ($u -eq 'semaine') { 'semaines jamais vues' } else { 'jours jamais vus' }
    $gagn = if ($u -eq 'semaine') { 'semaines gagnantes' } else { 'jours gagnants' }
    if ($nu -match '^EPOCH (\d+)\s+(.+?)\s+VAL\s+score ([+-][0-9.]+) R/partie \(([+-][0-9.]+)\$\)(?:\s+ajuste DD [+-][0-9.]+)?\s+gagnees (\d+)% perdues \d+% sur (\d+)\s+coups (\d+) \(([0-9.]+)/partie') {
        # COPIE D'ABORD : un -match reussi plus bas ecraserait $Matches.
        $m = $Matches.Clone()
        $dol = [double]::Parse($m[4], $inv)
        $c = if ($dol -gt 0) { '32' } else { '31' }
        $mot = if ($dol -gt 0) { 'OUI' } else { 'NON' }
        $niv = if ($m[2] -match 'EXPERT') { 'apres imitation de l expert' } else { "entrainement a $($m[2] -replace 'cout ', '') du cout reel" }
        return @('', "EPOCH $($m[1])   ($niv)",
                 "$esc[1;$($c)m  RENTABLE ?  $mot   $($m[4]) dollars par $u$esc[0m   (validation : $($m[6]) $vus, $($m[5])% de $gagn, $($m[8]) trades par $u)")
    }
    if ($nu -match '^EPOCH (\d+)\s+(.+?)\s+VAL\s+parties (\d+)\s+AUCUN COUP JOUE') {
        return @('', "EPOCH $($Matches[1])",
                 "$esc[1;33m  RENTABLE ?  NON   aucun trade joue sur les $($Matches[3]) parties de validation$esc[0m")
    }
    if ($nu -match '^TEST fold (\d+) \(([^)]*)\)\s+score ([+-][0-9.]+) R/partie \(([+-][0-9.]+)\$\)\s+gagnees (\d+)% perdues \d+% sur (\d+)') {
        $dol = [double]::Parse($Matches[4], $inv)
        $c = if ($dol -gt 0) { '32' } else { '31' }
        return @('', "$esc[1;$($c)m  RESULTAT FINAL du fold $($Matches[1]) sur des $($u)s JAMAIS utilise(e)s : $($Matches[4]) dollars par $u ($($Matches[5])% de $gagn sur $($Matches[6]))$esc[0m", '')
    }
    # LE BILAN DETAILLE, sous le verdict - 2026-09-27, demande du
    # proprietaire : gagnants, perdants, win rate, longs, shorts, profit
    # factor, drawdown. Le jeu l'ecrit deja lisible ; on le met en retrait.
    if ($nu -match '^  bilan  (.*)$') {
        return @("      $($Matches[1])")
    }
    if ($nu -match '^TEST fold (\d+) bilan  (.*)$') {
        return @("      $($Matches[2])", '')
    }
    if ($nu -match '^  sauvegarde  NOUVEAU MEILLEUR : ([+-][0-9.]+) R/partie') {
        return @("$esc[1;32m  >>> MODELE SAUVEGARDE : le plus rentable jusqu ici en validation$esc[0m")
    }
    return $null
}

$script:unite = 'jour'
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

    # UN JOURNAL VIDE N'A PAS ENCORE DE LECTEUR - 2026-09-26. A chaque
    # relance, `cmd` cree le journal avant que Python n'y ecrive : sa cle
    # est vide, egale a la cle de depart, donc aucun lecteur n'est ouvert.
    # Lire quand meme levait ' Impossible d'appeler une methode dans une
    # expression Null ' a chaque tour, tant que le fichier restait vide.
    if ($null -eq $lecteur) { Start-Sleep -Milliseconds 400; continue }

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
        $v = Verdict $nu
        if ($null -ne $v) {
            foreach ($x in $v) { [Console]::Out.WriteLine($x) }
            if (-not $Detail) { continue }
        }
        # LE REFUS DE SAUVEGARDER EST DEJA DIT PAR LE VERDICT.
        if (-not $Detail -and $nu -match '^  garde  rien de sauvegarde') { continue }
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
