#!/bin/bash

# =================================================================
# 🚀 اسکریپت تست سریع مدل‌های AI
# =================================================================
# 
# این اسکریپت به شما کمک می‌کند تا سریعاً مدل‌های مختلف را تست کنید
#
# استفاده:
#   chmod +x quick_test.sh
#   ./quick_test.sh
#
# =================================================================

echo "🦷 DentalAI - تست سریع مدل‌های AI"
echo "=================================================="

# بررسی Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 نصب نشده است"
    echo "   لطفاً ابتدا Python 3 را نصب کنید"
    exit 1
fi

echo "✅ Python3 یافت شد: $(python3 --version)"

# بررسی requests
if ! python3 -c "import requests" 2>/dev/null; then
    echo "📦 در حال نصب requests..."
    pip3 install requests
fi

# بررسی API Key
if [ -f ".env" ]; then
    echo "✅ فایل .env یافت شد"
    
    # خواندن API Key از .env
    if grep -q "VITE_OPENROUTER_API_KEY" .env; then
        API_KEY=$(grep VITE_OPENROUTER_API_KEY .env | cut -d '=' -f2)
        if [ "$API_KEY" != "sk-or-v1-your-api-key-here" ] && [ ! -z "$API_KEY" ]; then
            echo "✅ API Key یافت شد"
        else
            echo "⚠️  API Key در .env تنظیم نشده است"
            read -p "لطفاً API Key خود را وارد کنید: " API_KEY
            # ذخیره در متغیر محیطی موقت
            export OPENROUTER_API_KEY=$API_KEY
        fi
    fi
else
    echo "⚠️  فایل .env یافت نشد"
    read -p "لطفاً API Key خود را وارد کنید: " API_KEY
    export OPENROUTER_API_KEY=$API_KEY
fi

# انتخاب تصویر
echo ""
echo "📷 انتخاب تصویر:"
echo "1. استفاده از تصویر نمونه"
echo "2. وارد کردن مسیر تصویر"

read -p "انتخاب (1 یا 2): " IMAGE_CHOICE

if [ "$IMAGE_CHOICE" = "1" ]; then
    # پیدا کردن اولین تصویر در uploads
    IMAGE_PATH=$(find ../minimal-api-dev-v6/uploads/radiology -type f \( -name "*.jpg" -o -name "*.png" \) | head -n 1)
    if [ -z "$IMAGE_PATH" ]; then
        echo "❌ تصویر نمونه‌ای یافت نشد"
        read -p "لطفاً مسیر تصویر را وارد کنید: " IMAGE_PATH
    else
        echo "✅ استفاده از: $IMAGE_PATH"
    fi
else
    read -p "مسیر تصویر: " IMAGE_PATH
fi

# بررسی وجود تصویر
if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ تصویر یافت نشد: $IMAGE_PATH"
    exit 1
fi

# اجرای تست
echo ""
echo "🧪 شروع تست..."
echo "=================================================="

python3 test_openrouter_models.py "$IMAGE_PATH"

echo ""
echo "✅ تست کامل شد!"
echo "📄 نتایج در فایل test_results_*.json ذخیره شد"

