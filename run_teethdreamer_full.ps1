# ============================================
# TeethDreamer - اجرای کامل Pipeline
# ============================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "TeethDreamer - اجرای کامل Pipeline" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# تنظیم مسیرها
$TEETHDREAMER_DIR = Join-Path $PSScriptRoot "TeethDreamer"
$OUTPUT_DIR = Join-Path $PSScriptRoot "output"
$EXAMPLE_DIR = Join-Path $TEETHDREAMER_DIR "example\teeth"
$GENERATION_DIR = Join-Path $OUTPUT_DIR "generation"
$RECONSTRUCTION_DIR = Join-Path $OUTPUT_DIR "reconstruction"

# ایجاد دایرکتوری‌های خروجی
Write-Host "ایجاد دایرکتوری‌های خروجی..." -ForegroundColor Yellow
if (-not (Test-Path $GENERATION_DIR)) {
    New-Item -ItemType Directory -Path $GENERATION_DIR -Force | Out-Null
}
if (-not (Test-Path $RECONSTRUCTION_DIR)) {
    New-Item -ItemType Directory -Path $RECONSTRUCTION_DIR -Force | Out-Null
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "مرحله 1: Generation (تولید تصاویر چند منظره)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "این مرحله ممکن است 30-60 دقیقه طول بکشد..." -ForegroundColor Yellow
Write-Host ""

# تغییر به دایرکتوری TeethDreamer
Set-Location $TEETHDREAMER_DIR

# فعال‌سازی محیط مجازی
$VENV_ACTIVATE = Join-Path $PSScriptRoot "venv_teethdreamer\Scripts\Activate.ps1"
if (Test-Path $VENV_ACTIVATE) {
    & $VENV_ACTIVATE
} else {
    Write-Host "خطا: محیط مجازی پیدا نشد: $VENV_ACTIVATE" -ForegroundColor Red
    exit 1
}

# بررسی وجود مدل‌ها
$CHECKPOINT = Join-Path $TEETHDREAMER_DIR "ckpt\TeethDreamer.ckpt"
if (-not (Test-Path $CHECKPOINT)) {
    Write-Host "خطا: مدل پیدا نشد: $CHECKPOINT" -ForegroundColor Red
    exit 1
}

# بررسی وجود داده‌های نمونه
if (-not (Test-Path $EXAMPLE_DIR)) {
    Write-Host "خطا: دایرکتوری نمونه پیدا نشد: $EXAMPLE_DIR" -ForegroundColor Red
    exit 1
}

# اجرای Generation
Write-Host "اجرای Generation..." -ForegroundColor Green
$generationCmd = "python TeethDreamer.py -b configs/TeethDreamer.yaml --gpus 0 --test ckpt/TeethDreamer.ckpt --output `"$GENERATION_DIR`" data.params.test_dir=`"$EXAMPLE_DIR`""
Write-Host "دستور: $generationCmd" -ForegroundColor Gray
Write-Host ""

Invoke-Expression $generationCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "خطا در مرحله Generation!" -ForegroundColor Red
    Read-Host "برای خروج Enter را بزنید"
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "مرحله 2: پیدا کردن فایل‌های تولید شده" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# پیدا کردن اولین فایل تولید شده
$generatedFiles = Get-ChildItem -Path $GENERATION_DIR -Filter "*_upper_*.png" | Select-Object -First 1

if ($null -eq $generatedFiles) {
    Write-Host "خطا: هیچ فایل تولید شده‌ای پیدا نشد!" -ForegroundColor Red
    Write-Host "بررسی دایرکتوری: $GENERATION_DIR" -ForegroundColor Yellow
    Get-ChildItem -Path $GENERATION_DIR
    Read-Host "برای خروج Enter را بزنید"
    exit 1
}

$GENERATED_FILE = $generatedFiles.FullName
Write-Host "فایل تولید شده پیدا شد: $GENERATED_FILE" -ForegroundColor Green
Write-Host ""

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "مرحله 3: Reconstruction (بازسازی 3D)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "این مرحله ممکن است 10-20 دقیقه طول بکشد..." -ForegroundColor Yellow
Write-Host ""

# تغییر به دایرکتوری instant-nsr-pl
$INSTANT_NSR_DIR = Join-Path $TEETHDREAMER_DIR "instant-nsr-pl"
Set-Location $INSTANT_NSR_DIR

# اجرای Reconstruction
Write-Host "اجرای Reconstruction..." -ForegroundColor Green
$reconstructionCmd = "python run.py --img `"$GENERATED_FILE`" --cpu 4 --dir `"$RECONSTRUCTION_DIR`" --normal --rembg"
Write-Host "دستور: $reconstructionCmd" -ForegroundColor Gray
Write-Host ""

Invoke-Expression $reconstructionCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "خطا در مرحله Reconstruction!" -ForegroundColor Red
    Read-Host "برای خروج Enter را بزنید"
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "Pipeline کامل شد!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "فایل‌های تولید شده در: $GENERATION_DIR" -ForegroundColor Cyan
Write-Host "فایل‌های 3D در: $RECONSTRUCTION_DIR" -ForegroundColor Cyan
Write-Host ""
Write-Host "برای مشاهده نتایج، به دایرکتوری‌های بالا مراجعه کنید." -ForegroundColor Yellow
Write-Host ""

# نمایش فایل‌های نهایی
Write-Host "فایل‌های 3D تولید شده:" -ForegroundColor Yellow
Get-ChildItem -Path $RECONSTRUCTION_DIR -Recurse -Include *.ply,*.obj,*.stl | ForEach-Object {
    Write-Host "  - $($_.FullName)" -ForegroundColor Green
}

Write-Host ""
Read-Host "برای خروج Enter را بزنید"











