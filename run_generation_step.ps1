# ============================================
# TeethDreamer - اجرای مرحله Generation
# ============================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "TeethDreamer - اجرای مرحله Generation" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# تنظیم مسیرها
$TEETHDREAMER_DIR = Join-Path $PSScriptRoot "TeethDreamer"
$OUTPUT_DIR = Join-Path $PSScriptRoot "output"
$EXAMPLE_DIR = Join-Path $TEETHDREAMER_DIR "example\teeth"
$GENERATION_DIR = Join-Path $OUTPUT_DIR "generation"

# ایجاد دایرکتوری خروجی
Write-Host "ایجاد دایرکتوری خروجی..." -ForegroundColor Yellow
if (-not (Test-Path $GENERATION_DIR)) {
    New-Item -ItemType Directory -Path $GENERATION_DIR -Force | Out-Null
}

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

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "اجرای Generation..." -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "این مرحله ممکن است 30-60 دقیقه طول بکشد..." -ForegroundColor Yellow
Write-Host ""
Write-Host "دایرکتوری ورودی: $EXAMPLE_DIR" -ForegroundColor Gray
Write-Host "دایرکتوری خروجی: $GENERATION_DIR" -ForegroundColor Gray
Write-Host ""

# اجرای Generation
# توجه: test_set_npy لازم نیست چون استفاده نمی‌شود، اما باید یک مقدار به آن بدهیم
$generationCmd = "python TeethDreamer.py -b configs/TeethDreamer.yaml --gpus 0 --test ckpt/TeethDreamer.ckpt --output `"$GENERATION_DIR`" data.params.test_dir=`"$EXAMPLE_DIR`" data.params.test_set_npy=`"dummy.npy`""

Write-Host "دستور:" -ForegroundColor Yellow
Write-Host $generationCmd -ForegroundColor Gray
Write-Host ""

# اجرای دستور
Invoke-Expression $generationCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "خطا در مرحله Generation!" -ForegroundColor Red
    Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Red
    Read-Host "برای خروج Enter را بزنید"
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "Generation کامل شد!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "فایل‌های تولید شده در: $GENERATION_DIR" -ForegroundColor Cyan
Write-Host ""

# نمایش فایل‌های تولید شده
Write-Host "فایل‌های تولید شده:" -ForegroundColor Yellow
Get-ChildItem -Path $GENERATION_DIR -Filter "*.png" | ForEach-Object {
    Write-Host "  - $($_.Name) ($([math]::Round($_.Length / 1MB, 2)) MB)" -ForegroundColor Green
}

Write-Host ""
Read-Host "برای خروج Enter را بزنید"











