<#
.SYNOPSIS
    One-shot Windows installer for the Pupil object-recognition detector + plugins.

.DESCRIPTION
    Run this once on a fresh Windows machine (e.g. a laptop) that already has the
    Pupil Core app bundle installed. It:
      1. finds a Python 3.12 interpreter (3.10-3.13 accepted),
      2. creates the project .venv,
      3. detects an NVIDIA GPU and installs the CUDA build of torch (or the CPU
         build, with a clear "CPU fallback" warning if no GPU is found),
      4. installs the detector requirements,
      5. verifies torch sees the GPU and prints the device name,
      6. copies the two plugin files into ~/pupil_capture_settings/plugins and
         ~/pupil_player_settings/plugins (creating the folders if Capture/Player
         have never been launched),
      7. launches the detector server (unless -NoServer).

    Re-running is safe (idempotent): an existing venv is reused, plugins are
    re-copied only if changed (or always with -Force).

.PARAMETER Cpu
    Force the CPU build of torch even if an NVIDIA GPU is present.

.PARAMETER Cuda
    CUDA wheel tag for the torch index URL (default: cu126).

.PARAMETER NoServer
    Do everything except launch the detector at the end.

.PARAMETER Force
    Overwrite the installed plugin files even if they look identical.

.PARAMETER Recreate
    Delete and rebuild the .venv from scratch.

.EXAMPLE
    .\install.ps1
.EXAMPLE
    .\install.ps1 -Cpu          # laptop without NVIDIA GPU
#>
[CmdletBinding()]
param(
    [switch]$Cpu,
    [string]$Cuda = "cu126",
    [switch]$NoServer,
    [switch]$Force,
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv     = Join-Path $RepoRoot ".venv"
$VenvPy   = Join-Path $Venv "Scripts\python.exe"

function Info  ($m) { Write-Host "  $m" }
function Step  ($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok    ($m) { Write-Host "  [OK] $m" -ForegroundColor Green }
function Warn  ($m) { Write-Host "  [!]  $m" -ForegroundColor Yellow }
function Fail  ($m) { Write-Host "  [X]  $m" -ForegroundColor Red; exit 1 }

Write-Host "Pupil object-recognition - Windows installer" -ForegroundColor White
Info "Repo: $RepoRoot"

# ---------------------------------------------------------------------------
# 1. Find a suitable base Python (3.12 preferred, 3.10-3.13 accepted)
# ---------------------------------------------------------------------------
Step "Looking for Python 3.12"

function Find-BasePython {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $exe = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
    }
    foreach ($name in @("python3.12", "python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            $ver = & $cmd.Source -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ver -match '^3\.(1[0-3])$') { return $cmd.Source }
        }
    }
    return $null
}

$BasePython = Find-BasePython
if (-not $BasePython) {
    Fail "No Python 3.10-3.13 found. Install Python 3.12 from https://www.python.org/downloads/ (tick 'Add to PATH'), then re-run."
}
$baseVer = (& $BasePython -c "import sys; print('%d.%d.%d' % sys.version_info[:3])").Trim()
Ok "Using Python $baseVer ($BasePython)"

# ---------------------------------------------------------------------------
# 2. Create / reuse the virtual environment
# ---------------------------------------------------------------------------
Step "Virtual environment (.venv)"
if ($Recreate -and (Test-Path $Venv)) {
    Warn "Removing existing .venv (-Recreate)"
    Remove-Item -Recurse -Force $Venv
}
if (Test-Path $VenvPy) {
    Ok ".venv already present - reusing it"
} else {
    & $BasePython -m venv $Venv
    if (-not (Test-Path $VenvPy)) { Fail "venv creation failed." }
    Ok ".venv created"
}

& $VenvPy -m pip install --upgrade pip --quiet
Ok "pip up to date"

# ---------------------------------------------------------------------------
# 3. Detect the GPU and choose the torch build
# ---------------------------------------------------------------------------
Step "Detecting graphics hardware"

$gpuName = $null
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $q = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $q) { $gpuName = ($q | Select-Object -First 1).Trim() }
}

$useGpu = $false
if ($Cpu) {
    Warn "GPU disabled by -Cpu flag -> installing CPU torch."
    Warn "Detection will run on CPU (much slower; expect a low frame rate)."
} elseif ($gpuName) {
    $useGpu = $true
    Ok "NVIDIA GPU detected: $gpuName"
    Info "Installing the CUDA ($Cuda) build of torch."
} else {
    Warn "No NVIDIA GPU detected (nvidia-smi not found) -> installing CPU torch."
    Warn "CPU FALLBACK: detection will be much slower than on a CUDA GPU."
    Warn "If this machine *does* have an NVIDIA GPU, install its driver, then re-run."
}

# ---------------------------------------------------------------------------
# 4. Install torch (CUDA or CPU) then the detector requirements
# ---------------------------------------------------------------------------
Step "Installing torch"
if ($useGpu) {
    & $VenvPy -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
} else {
    & $VenvPy -m pip install torch torchvision
}
if ($LASTEXITCODE -ne 0) { Fail "torch install failed." }
Ok "torch installed"

Step "Installing detector requirements"
& $VenvPy -m pip install -r (Join-Path $RepoRoot "detector\requirements-detector.txt")
if ($LASTEXITCODE -ne 0) { Fail "requirements install failed." }
Ok "ultralytics + pyzmq + msgpack + opencv installed"

# ---------------------------------------------------------------------------
# 5. Verify the runtime device
# ---------------------------------------------------------------------------
Step "Verifying the inference device"
$probe = @"
import torch
avail = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if avail else ''
print('CUDA' if avail else 'CPU')
print(name)
"@
$out = & $VenvPy -c $probe
$mode = ($out | Select-Object -First 1).Trim()
$dev  = ($out | Select-Object -Skip 1 -First 1).Trim()

if ($mode -eq "CUDA") {
    Ok "torch will run on GPU: $dev"
} else {
    if ($useGpu) {
        Warn "torch was installed for CUDA but cannot see the GPU -> it will fall back to CPU."
        Warn "Check your NVIDIA driver, then re-run with -Recreate."
    } else {
        Warn "torch will run on CPU (slow). Re-run without -Cpu on a machine with an NVIDIA GPU for real-time speed."
    }
}

# ---------------------------------------------------------------------------
# 6. Copy the plugins into the Pupil settings folders
# ---------------------------------------------------------------------------
Step "Installing the Pupil plugins"

function Install-Plugin($srcRel, $settingsDir, $appLabel) {
    $src = Join-Path $RepoRoot $srcRel
    if (-not (Test-Path $src)) { Fail "Plugin source missing: $src" }
    $dstDir = Join-Path $HOME "$settingsDir\plugins"
    if (-not (Test-Path $dstDir)) {
        New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
        Info "Created $dstDir (Pupil $appLabel had never been launched)"
    }
    $dst = Join-Path $dstDir (Split-Path -Leaf $src)
    $needCopy = $true
    if ((Test-Path $dst) -and -not $Force) {
        $h1 = (Get-FileHash $src).Hash
        $h2 = (Get-FileHash $dst).Hash
        if ($h1 -eq $h2) { $needCopy = $false }
    }
    if ($needCopy) {
        Copy-Item $src $dst -Force
        Ok "$appLabel plugin -> $dst"
    } else {
        Ok "$appLabel plugin already up to date"
    }
}

Install-Plugin "plugins\capture_object_recognition.py" "pupil_capture_settings" "Capture"
Install-Plugin "plugins\player_object_recognition.py"  "pupil_player_settings"  "Player"

# ---------------------------------------------------------------------------
# 7. Done — launch the detector
# ---------------------------------------------------------------------------
Step "Installation complete"
Info "Next time, just run  start_detector.bat  (or .\.venv\Scripts\python detector\yolo_server.py)."
Info "In Pupil Capture/Player: Plugin Manager -> enable 'Object Recognition (YOLO)'."

if ($NoServer) {
    Info "(-NoServer) Detector not started."
    exit 0
}

Step "Launching the detector (Ctrl+C to stop)"
& $VenvPy (Join-Path $RepoRoot "detector\yolo_server.py")
