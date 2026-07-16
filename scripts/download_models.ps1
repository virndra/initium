# PowerShell twin of download_models.sh for Windows
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Models = Join-Path $Root "models"
New-Item -ItemType Directory -Force -Path $Models | Out-Null
Set-Location $Models

function Get-IfMissing($url, $out) {
    if (Test-Path $out) {
        Write-Host "    $out already present"
        return
    }
    Write-Host "==> Downloading $out"
    Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
}

Get-IfMissing `
  "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin" `
  "stories260K.bin"

Get-IfMissing `
  "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin" `
  "tok512.bin"

if (-not (Test-Path "tokenizer.bin")) {
    Copy-Item tok512.bin tokenizer.bin -Force
}

Write-Host "Done. Models in $Models"
Get-ChildItem $Models | Format-Table Name, Length
