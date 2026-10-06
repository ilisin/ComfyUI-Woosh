param(
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [switch]$DownloadModels,
    [ValidateSet('Woosh-DFlow', 'Woosh-Flow')][string]$Model = 'Woosh-DFlow',
    [string]$ModelRoot = ''
)

$ErrorActionPreference = 'Stop'
$pluginRoot = $PSScriptRoot
$runtimeRoot = Join-Path $pluginRoot 'runtime'
$sourceRoot = Join-Path $runtimeRoot 'Woosh'
$bootstrapRoot = Join-Path $runtimeRoot 'bootstrap'
$revision = '7f238bbd5d6dbc94d27190c07827b2d0eaefa268'

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Program $Arguments" }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git is required. Install Git for Windows first.' }
New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $sourceRoot)) {
    Invoke-Checked 'git' @('clone', 'https://github.com/SonyResearch/Woosh.git', $sourceRoot)
    Invoke-Checked 'git' @('-C', $sourceRoot, 'checkout', '--detach', $revision)
}
$actualRevision = & git -C $sourceRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $actualRevision -ne $revision) {
    throw "The source checkout must be at $revision. Set up a separate runtime.json for other revisions."
}

$bootstrapPython = Join-Path $bootstrapRoot 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $bootstrapPython)) {
    Invoke-Checked 'python' @('-m', 'venv', $bootstrapRoot)
}
$uv = Join-Path $bootstrapRoot 'Scripts/uv.exe'
if (-not (Test-Path -LiteralPath $uv)) {
    Invoke-Checked $bootstrapPython @('-m', 'pip', 'install', 'uv')
}
Invoke-Checked $uv @('sync', '--project', $sourceRoot, '--python', '3.12', '--extra', $Device, '--no-default-groups')
$backendPython = Join-Path $sourceRoot '.venv/Scripts/python.exe'

# huggingface_hub uses httpx, which requires socksio for SOCKS proxy URLs.
# Install after sync because uv removes packages not declared by the upstream project.
Invoke-Checked $uv @('pip', 'install', '--python', $backendPython, 'socksio>=1.0.0,<2')

# Only tokenizer/config assets are needed; TextConditionerA supplies the text model weights.
Invoke-Checked $backendPython @('-c', "from transformers import RobertaTokenizer, AutoConfig; RobertaTokenizer.from_pretrained('roberta-large'); AutoConfig.from_pretrained('roberta-large')")
Invoke-Checked $backendPython @('-c', "from woosh.model.flowmap_from_pretrained import FlowMapFromPretrained; from woosh.model.ldm import LatentDiffusionModel; print('Woosh backend imports OK')")

if (-not $ModelRoot) { $ModelRoot = Join-Path $pluginRoot '../../models/woosh' }
$ModelRoot = [System.IO.Path]::GetFullPath($ModelRoot)
if ($DownloadModels) {
    Invoke-Checked $backendPython @((Join-Path $pluginRoot 'download_models.py'), '--model', $Model, '--destination', $ModelRoot)
}
$config = @{ python = $backendPython; source = $sourceRoot; model_root = $ModelRoot } | ConvertTo-Json
[System.IO.File]::WriteAllText((Join-Path $pluginRoot 'runtime.json'), $config, [System.Text.UTF8Encoding]::new($false))
Write-Host 'Woosh runtime configured. Restart ComfyUI, then add WooshLocalGenerate.'
if (-not $DownloadModels) { Write-Host 'Weights are still required. Re-run with -DownloadModels, or copy the three model folders manually.' }
