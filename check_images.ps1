# بررسی فایل‌های تصویر مورد نیاز
Write-Host "========================================"
Write-Host "بررسی فایل‌های تصویر مورد نیاز"
Write-Host "========================================"
Write-Host ""

$imageDir = "my_images"
$requiredFiles = @("0.png", "1.png", "2.png", "3.png", "4.png")

# بررسی وجود پوشه
if (-not (Test-Path $imageDir)) {
    Write-Host "❌ پوشه '$imageDir' موجود نیست!" -ForegroundColor Red
    Write-Host "مسیر کامل: $(Resolve-Path .)\$imageDir" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ پوشه موجود است: $imageDir" -ForegroundColor Green
Write-Host "مسیر کامل: $(Resolve-Path $imageDir)" -ForegroundColor Cyan
Write-Host ""

# بررسی فایل‌های مورد نیاز
Write-Host "بررسی فایل‌های مورد نیاز:" -ForegroundColor Yellow
Write-Host ""

$allExist = $true
foreach ($file in $requiredFiles) {
    $filePath = Join-Path $imageDir $file
    $exists = Test-Path $filePath
    
    if ($exists) {
        $size = (Get-Item $filePath).Length / 1KB
        Write-Host "  ✅ $file - موجود ($([math]::Round($size, 2)) KB)" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file - موجود نیست!" -ForegroundColor Red
        $allExist = $false
    }
}

Write-Host ""

# لیست همه فایل‌های موجود
Write-Host "همه فایل‌های موجود در پوشه:" -ForegroundColor Yellow
$allFiles = Get-ChildItem -Path $imageDir -File
if ($allFiles.Count -eq 0) {
    Write-Host "  ⚠️ پوشه خالی است!" -ForegroundColor Yellow
} else {
    foreach ($file in $allFiles) {
        $size = $file.Length / 1KB
        Write-Host "  - $($file.Name) ($([math]::Round($size, 2)) KB)" -ForegroundColor Cyan
    }
}

Write-Host ""

# نتیجه نهایی
if ($allExist) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✅ همه فایل‌های مورد نیاز موجود هستند!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "می‌توانید دستور segmentation را اجرا کنید:" -ForegroundColor Cyan
    Write-Host "  cd TeethDreamer" -ForegroundColor White
    Write-Host "  python seg_teeth.py --img ../my_images --seg ../output/segmented --suffix png" -ForegroundColor White
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "❌ برخی فایل‌ها موجود نیستند!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "لطفاً فایل‌های زیر را در پوشه '$imageDir' قرار دهید:" -ForegroundColor Yellow
    foreach ($file in $requiredFiles) {
        $exists = Test-Path (Join-Path $imageDir $file)
        if (-not $exists) {
            Write-Host "  - $file" -ForegroundColor Yellow
        }
    }
}













