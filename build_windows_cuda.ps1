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
$cSources = @(Join-Path $sourceDir "syzygy\tbprobe.c")

$output = Join-Path $root "GiraffeGPU_cuda.exe"

if ($env:GIRAFFE_CUDA_ARCH) {
    $gencodeFlags = @("-arch=$($env:GIRAFFE_CUDA_ARCH)")
    Write-Host "Building native Windows CUDA engine ($($env:GIRAFFE_CUDA_ARCH))..."
} else {
    $gencodeFlags = @(
        "-gencode", "arch=compute_86,code=sm_86",
        "-gencode", "arch=compute_120,code=sm_120",
        "-gencode", "arch=compute_86,code=compute_86"
    )
    Write-Host "Building native Windows Dual/Multi-GPU CUDA engine (sm_86 + sm_120 + PTX)..."
}

& nvcc -ccbin cl -O3 -std=c++17 @gencodeFlags -I$sourceDir `
    -DNOMINMAX -DWIN32_LEAN_AND_MEAN `
    -Xcompiler "/O2 /EHsc /utf-8" `
    --machine 64 --allow-unsupported-compiler `
    $cppSources $cudaSources $cSources -o $output

if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE"
}

Write-Host "Build successful: $output"
