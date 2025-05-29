# VOC Analyzer Launcher v3.0
Write-Host "VOC Analyzer Launcher v3.0" -ForegroundColor Cyan

# Fix encoding issues for Windows
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    chcp 65001 | Out-Null
} catch {
    # Ignore encoding setup errors
}

# Get directories
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcDir = Join-Path -Path $scriptPath -ChildPath "src"

# Check if source directory exists
if (!(Test-Path $srcDir)) {
    Write-Host "Error: Source directory not found at $srcDir" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Change to source directory
Set-Location -Path $srcDir

# Set Python encoding environment variables
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Run the main Python script
try {
    Write-Host "Starting VOC Analyzer..." -ForegroundColor Green
    
    # Run Python code with encoding fix
    python -c "
# -*- coding: utf-8 -*-
import sys
import os

# Fix encoding for Windows
if os.name == 'nt':
    import locale
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

try:
    import main
    main.main()
except Exception as e:
    import traceback
    print('Error occurred:')
    print(traceback.format_exc())
"
}
catch {
    $errorMsg = $_.Exception.Message
    Write-Host "Error: $errorMsg" -ForegroundColor Red
}

# Return to original directory
Set-Location -Path $scriptPath
Read-Host "Press Enter to exit"