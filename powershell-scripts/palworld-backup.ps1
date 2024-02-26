$sourceFolder = "C:\Users\blue\Desktop\steam-servers\steamapps\common\PalServer\Pal\Saved\SaveGames\0"
$backupRootFolder = "C:\Users\blue\Desktop\Palworld_backup_save"
$backupFolderName = (Get-Date -Format "Backup yyyy-MM-dd_HH-mm-ss")
$backupFolder = Join-Path $backupRootFolder $backupFolderName

# Create a new backup
Copy-Item -Path $sourceFolder -Destination $backupFolder -Recurse

# Get list of backup folders sorted by creation time
$backupFolders = Get-ChildItem -Path $backupRootFolder | Where-Object { $_.PSIsContainer } | Sort-Object CreationTime -Descending

# Keep only the last 5 backups, delete others
if ($backupFolders.Count -gt 5) {
    # Skipping the latest 5 backups
    $oldBackups = $backupFolders | Select-Object -Skip 5

    foreach ($folder in $oldBackups) {
        Remove-Item -Path $folder.FullName -Recurse -Force
    }
}
