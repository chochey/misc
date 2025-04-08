PowerShell Script for Monitoring and Moving Files/Folders
PowerShell

<#
.SYNOPSIS
Monitors a specified folder for new files and folders, ensuring they are fully written before moving them to a destination directory.

.DESCRIPTION
This script uses the FileSystemWatcher to monitor a source folder for newly created files and folders.
It implements checks to ensure that incoming items are fully written and not locked before moving them
to a destination folder. The script handles potential edge cases like incomplete files, large files,
simultaneous arrivals, network interruptions, unstable destination folders, and varying transfer speeds.
It also includes a method to verify file integrity after the move.
#>

#region Configuration Settings
# ----------------------------------------------------
# EDIT THESE PATHS TO MATCH YOUR ENVIRONMENT
# ----------------------------------------------------
$SourceFolder = "C:\Path\To\Source\Folder"      # Folder to monitor for new files/folders
$DestinationFolder = "C:\Path\To\Destination"   # Folder where files will be moved to
$LogPath = "C:\Path\To\Logs\FileMonitor.log"    # Path to the log file

# Advanced Settings (adjust as needed)
$RetryDelaySeconds = 10                         # Base delay between retries
$MaxRetries = 5                                 # Maximum number of retry attempts
$IntegrityCheckMethod = 'FileSize'              # Method for checking file integrity: 'FileSize' or 'Hash'
$DestinationCheckIntervalSeconds = 10           # How often to check if destination is available
$TimeoutWaitingForDestinationSeconds = 300      # How long to wait for destination to be available
$MinimumTransferSpeedKBps = 100                 # Used to calculate timeouts for large files
#endregion

[CmdletBinding()]
param(
    [Parameter()]
    [string]$OverrideSourceFolder,

    [Parameter()]
    [string]$OverrideDestinationFolder,

    [Parameter()]
    [string]$OverrideLogPath,

    [Parameter()]
    [int]$OverrideRetryDelaySeconds,

    [Parameter()]
    [int]$OverrideMaxRetries,

    [Parameter()]
    [ValidateSet('FileSize', 'Hash')]
    [string]$OverrideIntegrityCheckMethod,
    
    [Parameter()]
    [int]$OverrideDestinationCheckIntervalSeconds,
    
    [Parameter()]
    [int]$OverrideTimeoutWaitingForDestinationSeconds,
    
    [Parameter()]
    [int]$OverrideMinimumTransferSpeedKBps
)

# Override configured values with command line parameters if provided
if ($OverrideSourceFolder) { $SourceFolder = $OverrideSourceFolder }
if ($OverrideDestinationFolder) { $DestinationFolder = $OverrideDestinationFolder }
if ($OverrideLogPath) { $LogPath = $OverrideLogPath }
if ($OverrideRetryDelaySeconds) { $RetryDelaySeconds = $OverrideRetryDelaySeconds }
if ($OverrideMaxRetries) { $MaxRetries = $OverrideMaxRetries }
if ($OverrideIntegrityCheckMethod) { $IntegrityCheckMethod = $OverrideIntegrityCheckMethod }
if ($OverrideDestinationCheckIntervalSeconds) { $DestinationCheckIntervalSeconds = $OverrideDestinationCheckIntervalSeconds }
if ($OverrideTimeoutWaitingForDestinationSeconds) { $TimeoutWaitingForDestinationSeconds = $OverrideTimeoutWaitingForDestinationSeconds }
if ($OverrideMinimumTransferSpeedKBps) { $MinimumTransferSpeedKBps = $OverrideMinimumTransferSpeedKBps }

#region Helper Functions

function Write-Log {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Message,
        
        [ValidateSet('Info', 'Warning', 'Error')]
        [string]$Severity = 'Info'
    )
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "$Timestamp - [$Severity] - $Message"
    
    try {
        Add-Content -Path $LogPath -Value $LogEntry -ErrorAction Stop
    }
    catch {
        Write-Warning "Could not write to log file: $($_.Exception.Message)"
    }
    
    switch ($Severity) {
        'Warning' { Write-Host $LogEntry -ForegroundColor Yellow }
        'Error'   { Write-Host $LogEntry -ForegroundColor Red }
        default   { Write-Host $LogEntry }
    }
}

function Resolve-LongPath {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Path
    )
    
    if (-not $Path.StartsWith("\\?\") -and $Path.Length -gt 250) {
        if ($Path.StartsWith("\\")) {
            return "\\?\UNC\" + $Path.Substring(2)
        } else {
            return "\\?\$Path"
        }
    } else {
        return $Path
    }
}

function Test-FileLocked {
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath
    )
    
    $ResolvedPath = Resolve-LongPath -Path $FilePath
    
    try {
        $Stream = [System.IO.File]::Open($ResolvedPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
        $Stream.Close()
        return $false # File is not locked
    }
    catch [System.IO.IOException] {
        if ($_.Exception.Message -match "being used by another process") {
            return $true  # File is locked
        }
        Write-Log "Error in Test-FileLocked: $($_.Exception.Message)" -Severity 'Warning'
        return $true  # Assume locked if there's an IO exception
    }
    catch {
        Write-Log "Error in Test-FileLocked: $($_.Exception.Message)" -Severity 'Warning'
        return $true  # Assume locked if there's any other exception
    }
}

function Wait-ForFileSizeStability {
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath,
        [int]$StabilityDurationSeconds = 5
    )
    
    $ResolvedPath = Resolve-LongPath -Path $FilePath
    $InitialSize = -1
    $StabilityStartTime = [DateTime]::MinValue
    
    try {
        $InitialSize = (Get-Item -Path $ResolvedPath -ErrorAction Stop).Length
        $StabilityStartTime = Get-Date
        
        while ((Get-Date).Subtract($StabilityStartTime).TotalSeconds -lt $StabilityDurationSeconds) {
            Start-Sleep -Seconds 1
            $CurrentSize = (Get-Item -Path $ResolvedPath -ErrorAction Stop).Length
            
            if ($CurrentSize -ne $InitialSize) {
                Write-Log "File size changed from $InitialSize to $CurrentSize bytes, resetting stability timer" -Severity 'Info'
                $InitialSize = $CurrentSize
                $StabilityStartTime = Get-Date
            }
        }
        
        return $true
    }
    catch {
        Write-Log "Error checking file size stability: $($_.Exception.Message)" -Severity 'Warning'
        return $false
    }
}

function Calculate-ExpectedTransferTime {
    param(
        [Parameter(Mandatory=$true)]
        [long]$FileSizeBytes,
        [int]$TransferSpeedKBps
    )
    
    $FileSizeKB = $FileSizeBytes / 1KB
    $ExpectedSeconds = $FileSizeKB / $TransferSpeedKBps
    
    return [math]::Max(60, $ExpectedSeconds * 1.5) # At least 60 seconds, with 50% buffer
}

function WaitFor-FileComplete {
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath,
        [int]$BaseTimeoutSeconds = 60
    )
    
    $ResolvedPath = Resolve-LongPath -Path $FilePath
    $StartTime = Get-Date
    $StableDuration = 5 # Seconds to wait for size stability
    
    Write-Log "Waiting for file '$FilePath' to be fully written..." -Severity 'Info'
    
    try {
        $FileInfo = Get-Item -Path $ResolvedPath -ErrorAction Stop
        $FileSize = $FileInfo.Length
        
        # Calculate appropriate timeout based on file size and min transfer speed
        $DynamicTimeout = Calculate-ExpectedTransferTime -FileSizeBytes $FileSize -TransferSpeedKBps $MinimumTransferSpeedKBps
        $TimeoutSeconds = [math]::Max($BaseTimeoutSeconds, $DynamicTimeout)
        
        Write-Log "Using timeout of $TimeoutSeconds seconds for file of size $([math]::Round($FileSize/1MB, 2)) MB" -Severity 'Info'
        
        while ((Get-Date).Subtract($StartTime).TotalSeconds -lt $TimeoutSeconds) {
            if (Test-FileLocked -FilePath $ResolvedPath) {
                Write-Log "File is currently locked, waiting..." -Severity 'Info'
                Start-Sleep -Seconds 2
                continue
            }
            
            # Check if file size has stabilized
            if (Wait-ForFileSizeStability -FilePath $ResolvedPath -StabilityDurationSeconds $StableDuration) {
                Write-Log "File '$FilePath' appears to be fully written (size stable for $StableDuration seconds)." -Severity 'Info'
                return $true
            }
        }
        
        Write-Log "Timeout waiting for file '$FilePath' to be fully written." -Severity 'Warning'
        return $false
    }
    catch {
        Write-Log "Error checking file '$FilePath': $($_.Exception.Message)" -Severity 'Error'
        Start-Sleep -Seconds 5
        return $false
    }
}

function Test-DestinationAvailable {
    param(
        [Parameter(Mandatory=$true)]
        [string]$DestinationFolder,
        [int]$TimeoutSeconds = 300,
        [int]$CheckIntervalSeconds = 10
    )
    
    $StartTime = Get-Date
    $ResolvedPath = Resolve-LongPath -Path $DestinationFolder
    
    Write-Log "Checking if destination folder '$DestinationFolder' is available..." -Severity 'Info'
    
    while ((Get-Date).Subtract($StartTime).TotalSeconds -lt $TimeoutSeconds) {
        try {
            if (Test-Path -Path $ResolvedPath -PathType Container) {
                # Test if we can write to the destination
                $TestFile = Join-Path -Path $ResolvedPath -ChildPath "write_test_$(Get-Random).tmp"
                New-Item -Path $TestFile -ItemType File -Force | Out-Null
                Remove-Item -Path $TestFile -Force
                Write-Log "Destination folder '$DestinationFolder' is available and writable." -Severity 'Info'
                return $true
            }
            else {
                # Try to create the destination folder
                Write-Log "Destination folder does not exist. Attempting to create it..." -Severity 'Info'
                New-Item -Path $ResolvedPath -ItemType Directory -Force | Out-Null
                Write-Log "Successfully created destination folder." -Severity 'Info'
                return $true
            }
        }
        catch {
            Write-Log "Destination folder is currently unavailable: $($_.Exception.Message)" -Severity 'Warning'
            Start-Sleep -Seconds $CheckIntervalSeconds
        }
    }
    
    Write-Log "Timed out waiting for destination folder to become available." -Severity 'Error'
    return $false
}

function Get-FileHashValue {
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath
    )
    
    $ResolvedPath = Resolve-LongPath -Path $FilePath
    
    try {
        if ($PSVersionTable.PSVersion.Major -lt 5 -or ($PSVersionTable.PSVersion.Major -eq 5 -and $PSVersionTable.PSVersion.Minor -lt 1)) {
            Write-Log "Get-FileHash cmdlet requires PowerShell version 5.1 or later. Skipping hash check." -Severity 'Warning'
            return $null
        }
        return (Get-FileHash -Path $ResolvedPath -Algorithm SHA256).Hash
    }
    catch {
        Write-Log "Error calculating hash for '$FilePath': $($_.Exception.Message)" -Severity 'Error'
        return $null
    }
}

function Verify-FileIntegrity {
    param(
        [Parameter(Mandatory=$true)]
        [string]$SourcePath,
        [Parameter(Mandatory=$true)]
        [string]$DestinationPath,
        [string]$Method
    )
    
    $ResolvedSourcePath = Resolve-LongPath -Path $SourcePath
    $ResolvedDestPath = Resolve-LongPath -Path $DestinationPath
    
    Write-Log "Verifying integrity of '$SourcePath' against '$DestinationPath' using method '$Method'." -Severity 'Info'
    
    try {
        if ($Method -eq 'FileSize') {
            $SourceSize = (Get-Item -Path $ResolvedSourcePath -ErrorAction Stop).Length
            $DestinationSize = (Get-Item -Path $ResolvedDestPath -ErrorAction Stop).Length
            
            if ($SourceSize -eq $DestinationSize) {
                Write-Log "File size verification successful." -Severity 'Info'
                return $true
            } else {
                Write-Log "File size mismatch: Source ($SourceSize), Destination ($DestinationSize)." -Severity 'Warning'
                return $false
            }
        }
        elseif ($Method -eq 'Hash') {
            $SourceHash = Get-FileHashValue -FilePath $ResolvedSourcePath
            $DestinationHash = Get-FileHashValue -FilePath $ResolvedDestPath
            
            if ($SourceHash -and $DestinationHash -and ($SourceHash -ceq $DestinationHash)) {
                Write-Log "File hash verification successful." -Severity 'Info'
                return $true
            } else {
                Write-Log "File hash mismatch or error calculating hash." -Severity 'Warning'
                return $false
            }
        }
        else {
            Write-Log "Invalid integrity check method specified: '$Method'." -Severity 'Warning'
            return $false
        }
    }
    catch {
        Write-Log "Error verifying file integrity: $($_.Exception.Message)" -Severity 'Error'
        return $false
    }
}

function Preserve-FileAttributes {
    param(
        [Parameter(Mandatory=$true)]
        [string]$SourcePath,
        [Parameter(Mandatory=$true)]
        [string]$DestinationPath
    )
    
    try {
        $SourceItem = Get-Item -Path $SourcePath -ErrorAction Stop
        $DestItem = Get-Item -Path $DestinationPath -ErrorAction Stop
        
        $DestItem.Attributes = $SourceItem.Attributes
        Write-Log "Successfully preserved file attributes for '$DestinationPath'." -Severity 'Info'
        return $true
    }
    catch {
        Write-Log "Error preserving file attributes: $($_.Exception.Message)" -Severity 'Warning'
        return $false
    }
}

function Move-MonitoredItem {
    param(
        [Parameter(Mandatory=$true)]
        [string]$ItemPath,
        [Parameter(Mandatory=$true)]
        [string]$DestinationBaseFolder
    )
    
    $ResolvedItemPath = Resolve-LongPath -Path $ItemPath
    $ItemName = Split-Path -Path $ItemPath -Leaf
    $DestinationPath = Join-Path -Path $DestinationBaseFolder -ChildPath $ItemName
    $ResolvedDestPath = Resolve-LongPath -Path $DestinationPath
    
    # Check if destination exists and handle appropriately
    if (Test-Path -Path $ResolvedDestPath) {
        Write-Log "Warning: Destination path '$DestinationPath' already exists. Renaming file before move." -Severity 'Warning'
        $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $NewName = [System.IO.Path]::GetFileNameWithoutExtension($ItemName) + "_$Timestamp"
        $Extension = [System.IO.Path]::GetExtension($ItemName)
        $DestinationPath = Join-Path -Path $DestinationBaseFolder -ChildPath "$NewName$Extension"
        $ResolvedDestPath = Resolve-LongPath -Path $DestinationPath
        Write-Log "New destination path: '$DestinationPath'" -Severity 'Info'
    }
    
    # Check if destination folder is available
    if (-not (Test-DestinationAvailable -DestinationFolder $DestinationBaseFolder -TimeoutSeconds $TimeoutWaitingForDestinationSeconds -CheckIntervalSeconds $DestinationCheckIntervalSeconds)) {
        Write-Log "Destination folder '$DestinationBaseFolder' is unavailable. Cannot move '$ItemPath'." -Severity 'Error'
        return
    }
    
    $retries = 0
    while ($retries -lt $MaxRetries) {
        try {
            if (Test-Path -Path $ResolvedItemPath) {
                Write-Log "Moving '$ItemPath' to '$DestinationPath'." -Severity 'Info'
                
                # Determine if this is a file or directory
                $IsDirectory = (Get-Item -Path $ResolvedItemPath -ErrorAction Stop) -is [System.IO.DirectoryInfo]
                
                if ($IsDirectory) {
                    # For directories, use robocopy which is more robust for directories
                    $RobocopyOutput = robocopy $ResolvedItemPath $ResolvedDestPath /E /MOVE /R:3 /W:5 /MT:8 /NP /NFL /NDL /NJH /NJS
                    $RobocopyExitCode = $LASTEXITCODE
                    
                    if ($RobocopyExitCode -ge 8) {
                        throw "Robocopy encountered errors (exit code: $RobocopyExitCode)"
                    }
                    
                    Write-Log "Successfully moved directory '$ItemPath' to '$DestinationPath'." -Severity 'Info'
                    return
                }
                else {
                    # For files, use Move-Item with integrity verification
                    Copy-Item -Path $ResolvedItemPath -Destination $ResolvedDestPath -Force
                    
                    if (Test-Path -Path $ResolvedDestPath) {
                        if ($IntegrityCheckMethod -ne 'None') {
                            if (Verify-FileIntegrity -SourcePath $ResolvedItemPath -DestinationPath $ResolvedDestPath -Method $IntegrityCheckMethod) {
                                Write-Log "Successfully copied and verified '$ItemName'." -Severity 'Info'
                                
                                # Preserve file attributes
                                Preserve-FileAttributes -SourcePath $ResolvedItemPath -DestinationPath $ResolvedDestPath
                                
                                # Remove source file after successful verification
                                Remove-Item -Path $ResolvedItemPath -Force
                                Write-Log "Removed source file after successful move." -Severity 'Info'
                                return
                            } else {
                                Write-Log "Integrity check failed for '$ItemName' after copying. Will retry." -Severity 'Warning'
                                if (Test-Path -Path $ResolvedDestPath) {
                                    Remove-Item -Path $ResolvedDestPath -Force
                                }
                            }
                        } else {
                            Write-Log "Successfully copied '$ItemName' (integrity check skipped)." -Severity 'Info'
                            Remove-Item -Path $ResolvedItemPath -Force
                            return
                        }
                    } else {
                        Write-Log "Error: Destination path '$DestinationPath' not found after copy." -Severity 'Error'
                    }
                }
            } else {
                Write-Log "Warning: Source path '$ItemPath' no longer exists." -Severity 'Warning'
                return
            }
        }
        catch {
            Write-Log "Error moving '$ItemPath' to '$DestinationPath': $($_.Exception.Message)" -Severity 'Error'
        }
        
        $retries++
        if ($retries -lt $MaxRetries) {
            $CurrentDelay = $RetryDelaySeconds * [Math]::Pow(1.5, $retries - 1) # Exponential backoff
            Write-Log "Retry $retries/$MaxRetries in $([Math]::Round($CurrentDelay)) seconds..." -Severity 'Warning'
            Start-Sleep -Seconds $CurrentDelay
        }
    }
    
    Write-Log "Failed to move '$ItemPath' after $MaxRetries retries." -Severity 'Error'
}

function Handle-AlternateDataStreams {
    param(
        [Parameter(Mandatory=$true)]
        [string]$SourcePath,
        [Parameter(Mandatory=$true)]
        [string]$DestinationPath
    )
    
    try {
        $StreamInfo = Get-Item -Path $SourcePath -Stream * -ErrorAction SilentlyContinue
        
        if ($StreamInfo) {
            foreach ($Stream in $StreamInfo) {
                if ($Stream.Stream -ne ':$DATA') { # Skip the main data stream
                    $StreamContent = Get-Content -Path "$SourcePath`:$($Stream.Stream)" -Raw -ErrorAction SilentlyContinue
                    if ($StreamContent) {
                        Set-Content -Path "$DestinationPath`:$($Stream.Stream)" -Value $StreamContent -ErrorAction SilentlyContinue
                        Write-Log "Copied alternate data stream '$($Stream.Stream)' to destination file." -Severity 'Info'
                    }
                }
            }
        }
    }
    catch {
        Write-Log "Error copying alternate data streams: $($_.Exception.Message)" -Severity 'Warning'
    }
}

#endregion

#region Main Script

# Set error action preference
$ErrorActionPreference = 'Stop'

# Initialize log file directory if it doesn't exist
$LogFolder = Split-Path -Path $LogPath -Parent
if (-not (Test-Path -Path $LogFolder -PathType Container)) {
    try {
        New-Item -Path $LogFolder -ItemType Directory -Force | Out-Null
        Write-Log "Created log folder: '$LogFolder'" -Severity 'Info'
    }
    catch {
        Write-Warning "Failed to create log folder '$LogFolder': $($_.Exception.Message)"
        exit 1
    }
}

# Resolve paths to handle long paths
$SourceFolder = Resolve-LongPath -Path $SourceFolder
$DestinationFolder = Resolve-LongPath -Path $DestinationFolder

# Ensure source folder exists
if (-not (Test-Path -Path $SourceFolder -PathType Container)) {
    Write-Log "Source folder '$SourceFolder' does not exist. Please specify a valid source folder." -Severity 'Error'
    exit 1
}

# Create destination folder if it doesn't exist
if (-not (Test-DestinationAvailable -DestinationFolder $DestinationFolder -TimeoutSeconds $TimeoutWaitingForDestinationSeconds -CheckIntervalSeconds $DestinationCheckIntervalSeconds)) {
    Write-Log "Could not access or create destination folder '$DestinationFolder'. Exiting script." -Severity 'Error'
    exit 1
}

# Create the FileSystemWatcher object
try {
    $Watcher = New-Object System.IO.FileSystemWatcher
    $Watcher.Path = $SourceFolder
    $Watcher.Filter = "*.*" # Monitor all files and folders
    $Watcher.IncludeSubdirectories = $true # Monitor subdirectories as well
    $Watcher.EnableRaisingEvents = $true
    
    Write-Log "Successfully created FileSystemWatcher for '$SourceFolder'." -Severity 'Info'
}
catch {
    Write-Log "Failed to create FileSystemWatcher: $($_.Exception.Message)" -Severity 'Error'
    exit 1
}

# Define the action to take when a new file or folder is created
$Action = {
    param($EventArgs)
    $FullPath = $EventArgs.FullPath
    $ChangeType = $EventArgs.ChangeType
    $Name = $EventArgs.Name

    if ($ChangeType -eq "Created") {
        Write-Log "Detected new item: '$Name' at '$FullPath'" -Severity 'Info'
        
        # Wait for the item to be fully written (for files)
        if (Test-Path -Path $FullPath -ErrorAction SilentlyContinue) {
            # Determine if this is a file or directory
            $IsDirectory = (Get-Item -Path $FullPath -ErrorAction SilentlyContinue) -is [System.IO.DirectoryInfo]
            
            if (-not $IsDirectory) {
                # For files, wait until they're fully written
                if (WaitFor-FileComplete -FilePath $FullPath) {
                    # Check for alternate data streams and handle them
                    Handle-AlternateDataStreams -SourcePath $FullPath -DestinationPath (Join-Path -Path $using:DestinationFolder -ChildPath $Name)
                    
                    # Move the file
                    Move-MonitoredItem -ItemPath $FullPath -DestinationBaseFolder $using:DestinationFolder
                } else {
                    Write-Log "Warning: '$Name' did not become fully written within the timeout period." -Severity 'Warning'
                }
            } else {
                # For folders, wait a bit for any initial file creations to complete
                Start-Sleep -Seconds 5
                
                # Special handling for junction points and symbolic links
                $DirInfo = Get-Item -Path $FullPath -ErrorAction SilentlyContinue
                if ($DirInfo -and ($DirInfo.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
                    Write-Log "'$Name' is a junction point or symbolic link. Special handling required." -Severity 'Warning'
                    
                    # For junction points, we might need to recreate them rather than move them
                    # This is a simplistic approach - might need enhancement for production
                    $TargetPath = [System.IO.Directory]::GetDirectoryRoot($DirInfo.Target)
                    $DestJunctionPath = Join-Path -Path $using:DestinationFolder -ChildPath $Name
                    
                    try {
                        New-Item -ItemType Junction -Path $DestJunctionPath -Target $TargetPath -Force | Out-Null
                        Write-Log "Recreated junction point '$Name' at destination." -Severity 'Info'
                        Remove-Item -Path $FullPath -Force -Recurse
                    }
                    catch {
                        Write-Log "Failed to handle junction point: $($_.Exception.Message)" -Severity 'Error'
                    }
                } else {
                    # Regular directory, use standard move
                    Move-MonitoredItem -ItemPath $FullPath -DestinationBaseFolder $using:DestinationFolder
                }
            }
        } else {
            Write-Log "Warning: '$FullPath' no longer exists." -Severity 'Warning'
        }
    }
}

# Register the event handler
$EventJobName = "FileCreated_$(Get-Random)"
Register-ObjectEvent -InputObject $Watcher -EventName Created -SourceIdentifier $EventJobName -Action $Action

# Check for files already in the source folder
Write-Log "Checking for existing files in '$SourceFolder'..." -Severity 'Info'
$ExistingItems = Get-ChildItem -Path $SourceFolder -Recurse | Where-Object { -not $_.PSIsContainer }

if ($ExistingItems.Count -gt 0) {
    Write-Log "Found $($ExistingItems.Count) existing files in the source folder. Processing them..." -Severity 'Info'
    
    foreach ($Item in $ExistingItems) {
        Write-Log "Processing existing file: '$($Item.FullName)'" -Severity 'Info'
        
        if (WaitFor-FileComplete -FilePath $Item.FullName) {
            Handle-AlternateDataStreams -SourcePath $Item.FullName -DestinationPath (Join-Path -Path $DestinationFolder -ChildPath $Item.Name)
            Move-MonitoredItem -ItemPath $Item.FullName -DestinationBaseFolder $DestinationFolder
        } else {
            Write-Log "Warning: Existing file '$($Item.Name)' could not be determined as fully written." -Severity 'Warning'
        }
    }
}

Write-Log "Started monitoring folder '$SourceFolder'..." -Severity 'Info'
Write-Log "Destination folder: '$DestinationFolder'" -Severity 'Info'
Write-Log "Press Ctrl+C to stop monitoring." -Severity 'Info'

try {
    while ($true) {
        # Keep the script running
        Start-Sleep -Seconds 5
        
        # Check if the destination is still available every minute
        if ((-not (Test-Path -Path $DestinationFolder -PathType Container -ErrorAction SilentlyContinue)) -and 
            (-not (Test-DestinationAvailable -DestinationFolder $DestinationFolder -TimeoutSeconds 30 -CheckIntervalSeconds 5))) {
            Write-Log "Warning: Destination folder is currently unavailable. Will continue monitoring, but files cannot be moved until destination is available again." -Severity 'Warning'
        }
    }
}
catch [System.Management.Automation.PipelineStoppedException] {
    # This is the expected way to end the script (Ctrl+C)
    Write-Log "Received stop signal. Cleaning up..." -Severity 'Info'
}
catch {
    Write-Log "Unexpected error: $($_.Exception.Message)" -Severity 'Error'
}
finally {
    # Unregister the event and dispose of the watcher
    Unregister-Event -SourceIdentifier $EventJobName -ErrorAction SilentlyContinue
    $Watcher.EnableRaisingEvents = $false
    $Watcher.Dispose()
    Write-Log "Stopped monitoring folder '$SourceFolder'." -Severity 'Info'
}

#endregion