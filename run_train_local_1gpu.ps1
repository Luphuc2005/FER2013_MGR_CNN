param(
    [string]$PythonBin = "",
    [switch]$AllowCpu,
    [switch]$NoResume,
    [int]$Epochs = 50,
    [int]$BatchSizePerGpu = 1,
    [int]$MaxTrainSamples = 512,
    [int]$MaxValSamples = 128,
    [int]$MaxTestSamples = 128
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

if (-not $PythonBin) {
    if ($env:MGR_PYTHON_BIN) {
        $PythonBin = $env:MGR_PYTHON_BIN
    } else {
        $DefaultMgrPython = "C:\Users\ADMIN\anaconda3\envs\mgr_tf210\python.exe"
        $PythonBin = if (Test-Path $DefaultMgrPython) {
            $DefaultMgrPython
        } elseif ($env:CONDA_PREFIX) {
            Join-Path $env:CONDA_PREFIX "python.exe"
        } else {
            "python"
        }
    }
}

$PythonDir = Split-Path -Parent $PythonBin
$CondaEnvDir = $PythonDir
$CondaLibraryBin = Join-Path $CondaEnvDir "Library\bin"
$CondaScripts = Join-Path $CondaEnvDir "Scripts"
if (Test-Path $CondaLibraryBin) {
    $env:PATH = "$CondaLibraryBin;$CondaEnvDir;$CondaScripts;$env:PATH"
}

if ($AllowCpu) {
    $env:CUDA_VISIBLE_DEVICES = "-1"
    $env:MGR_ALLOW_CPU = "1"
} else {
    $env:CUDA_VISIBLE_DEVICES = "0"
    Remove-Item Env:\MGR_ALLOW_CPU -ErrorAction SilentlyContinue
}
$env:TF_CPP_MIN_LOG_LEVEL = "1"
$env:TF_FORCE_GPU_ALLOW_GROWTH = "true"

$env:MGR_GPU_IDS = if ($AllowCpu) { "" } else { "0" }
$env:MGR_REQUIRE_TWO_GPUS = "0"
$env:MGR_MIN_GPUS = if ($AllowCpu) { "0" } else { "1" }
$env:MGR_EPOCHS = "$Epochs"
$env:MGR_BATCH_SIZE_PER_GPU = "$BatchSizePerGpu"
if ($MaxTrainSamples -gt 0) {
    $env:MGR_MAX_TRAIN_SAMPLES = "$MaxTrainSamples"
} else {
    Remove-Item Env:\MGR_MAX_TRAIN_SAMPLES -ErrorAction SilentlyContinue
}
if ($MaxValSamples -gt 0) {
    $env:MGR_MAX_VAL_SAMPLES = "$MaxValSamples"
} else {
    Remove-Item Env:\MGR_MAX_VAL_SAMPLES -ErrorAction SilentlyContinue
}
if ($MaxTestSamples -gt 0) {
    $env:MGR_MAX_TEST_SAMPLES = "$MaxTestSamples"
} else {
    Remove-Item Env:\MGR_MAX_TEST_SAMPLES -ErrorAction SilentlyContinue
}

# Local Windows profile: keep tf.data small to avoid CPU/RAM OOM on laptop-class machines.
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:TF_NUM_INTRAOP_THREADS = "2"
$env:TF_NUM_INTEROP_THREADS = "1"
$env:MGR_TF_INTRA_OP_THREADS = "2"
$env:MGR_TF_INTER_OP_THREADS = "1"
$env:MGR_TF_DATA_NUM_PARALLEL_CALLS = "1"
$env:MGR_TF_DATA_PRIVATE_THREADPOOL_SIZE = "1"
$env:MGR_TF_DATA_DETERMINISTIC = "0"
$env:MGR_PREFETCH_BUFFER = "1"
$env:MGR_USE_TFA_ADAMW = "0"

New-Item -ItemType Directory -Force logs | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $RootDir "logs\train_local_1gpu_$Stamp.log"

Write-Host "[INFO] Local Windows 1-GPU training"
if ($AllowCpu) {
    Write-Host "[INFO] CPU fallback is enabled for smoke testing"
}
Write-Host "[INFO] epochs=$Epochs batch_size_per_gpu=$BatchSizePerGpu"
Write-Host "[INFO] max_train_samples=$MaxTrainSamples max_val_samples=$MaxValSamples max_test_samples=$MaxTestSamples"
Write-Host "[INFO] log=$LogPath"
Write-Host "[INFO] python=$PythonBin"

$PrevErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $PythonBin check_environment.py 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "check_environment.py failed with exit code $LASTEXITCODE"
    }

    Write-Host "[INFO] Starting train.py"
    $TrainArgs = @("train.py", "--config", "config.yaml")
    if (-not $NoResume) {
        $TrainArgs += "--resume"
    }
    & $PythonBin @TrainArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "train.py failed with exit code $LASTEXITCODE"
    }
}
finally {
    $ErrorActionPreference = $PrevErrorActionPreference
}
