# EcoVision AI - 1-Click GitHub Push Helper
param (
    [Parameter(Mandatory=$true, HelpMessage="Enter your GitHub repository URL (e.g. https://github.com/username/EcoVision-AI.git)")]
    [string]$RepoUrl
)

$gitExe = "C:\Users\chand\AppData\Local\Programs\Git\cmd\git.exe"

if (-not (Test-Path $gitExe)) {
    Write-Host "[Error] Git executable not found at $gitExe" -ForegroundColor Red
    exit 1
}

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "      EcoVision AI - Pushing to GitHub for Cloud Deploy" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Configuring remote origin: $RepoUrl"

& $gitExe remote remove origin 2>$null
& $gitExe remote add origin $RepoUrl
& $gitExe branch -M main

Write-Host "Pushing code to GitHub main branch..." -ForegroundColor Cyan
& $gitExe push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Repository pushed to GitHub!" -ForegroundColor Green
    Write-Host "Now go to https://share.streamlit.io and deploy in 1 click!" -ForegroundColor Yellow
} else {
    Write-Host "`n[Notice] If prompted for credentials, please sign in with your GitHub account." -ForegroundColor Yellow
}
