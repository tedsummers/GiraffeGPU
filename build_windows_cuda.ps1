$ErrorActionPreference = "Stop"

if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "nvcc was not found. Install the NVIDIA CUDA Toolkit and run from a Visual Studio Developer PowerShell."
}
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    throw "cl.exe was not found. Run this from x64 Native Tools Command Prompt/Developer PowerShell for Visual Studio."
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceDir = Join-Path $root "src"
$cppSources = Get-ChildItem (Join-Path $sourceDir "*.cpp") |
    Where-Object { $_.Name -ne "neural_win.cpp" } |
    ForEach-Object { $_.FullName }
$cudaSources = Get-ChildItem (Join-Path $sourceDir "cuda\*.cu") |
    ForEach-Object { $_.FullName }
$cSources = Get-ChildItem (Join-Path $sourceDir "syzygy\*.c") |
    ForEach-Object { $_.FullName }

$output = Join-Path $root "GiraffeGPU_cuda.exe"
$arch = if ($env:GIRAFFE_CUDA_ARCH) { $env:GIRAFFE_CUDA_ARCH } else { "sm_86" }

Write-Host "Building native Windows CUDA engine ($arch)..."
& nvcc -ccbin cl -O3 -std=c++17 -arch=$arch -I$sourceDir `
    --machine 64 --allow-unsupported-compiler `
    $cppSources $cudaSources $cSources -o $output

if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE"
}

Write-Host "Build successful: $output"
