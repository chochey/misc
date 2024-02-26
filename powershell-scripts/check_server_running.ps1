$processName = "PalServer-Win64-Test-Cmd"
$batchFileName = "Start_Palworld_Server.bat"
$serverPath = "C:\Users\blue\Desktop\steam-servers\$batchFileName"
$workingDirectory = "C:\Users\blue\Desktop\steam-servers"

# Check if the process is already running
$runningProcess = Get-Process -Name $processName -ErrorAction SilentlyContinue

# Check if the batch file is already running
$batchFileRunning = Get-Process -Name "cmd" | Where-Object { $_.MainWindowTitle -eq $batchFileName }

if ($runningProcess -ne $null -or $batchFileRunning -ne $null) {
    # Either the process or the batch file is already running, do nothing
    Write-Host "$processName or $batchFileName is already running. No action needed."
} else {
    # Neither the process nor the batch file is running, so start the server
    Write-Host "$processName is not running. Starting the Palworld server..."
    Start-Process -FilePath $serverPath -WorkingDirectory $workingDirectory
}
