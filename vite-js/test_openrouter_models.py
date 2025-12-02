#!/usr/bin/env python3
"""
🧪 اسکریپت تست مدل‌های OpenRouter
برای تست دستی مدل‌های مختلف قبل از استفاده در پروژه

نیازمندی‌ها:
    pip install requests

استفاده:
    python test_openrouter_models.py
"""

import requests
import json
import base64
import time
import sys
from pathlib import Path
from datetime import datetime

# ========== تنظیمات ==========

# API Key خود را اینجا قرار دهید
OPENROUTER_API_KEY = "sk-or-v1-..."  # 👈 کلید خود را اینجا بگذارید

# URL سرویس OpenRouter
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# مدل‌های موجود برای تست
MODELS = [
    {
        "id": "openai/gpt-4o",
        "name": "GPT-4o",
        "provider": "OpenAI",
        "description": "بهترین مدل OpenAI با قابلیت Vision"
    },
    {
        "id": "openai/gpt-4o-mini",
        "name": "GPT-4o Mini",
        "provider": "OpenAI",
        "description": "سریع و ارزان"
    },
    {
        "id": "anthropic/claude-3.5-sonnet",
        "name": "Claude 3.5 Sonnet",
        "provider": "Anthropic",
        "description": "دقت بسیار بالا در تحلیل تصویر"
    },
    {
        "id": "anthropic/claude-3-opus",
        "name": "Claude 3 Opus",
        "provider": "Anthropic",
        "description": "قوی‌ترین مدل Claude"
    },
    {
        "id": "anthropic/claude-3-haiku",
        "name": "Claude 3 Haiku",
        "provider": "Anthropic",
        "description": "سریع‌ترین مدل Claude"
    },
    {
        "id": "google/gemini-flash-1.5",
        "name": "Gemini Flash 1.5",
        "provider": "Google",
        "description": "سریع و کارآمد"
    },
]

# Prompt برای تشخیص landmarks
PROMPT = """You are an expert in cephalometric analysis. Analyze this lateral cephalometric radiograph and identify the following anatomical landmarks with their exact pixel coordinates:

Required landmarks:
1. S (Sella) - Center of sella turcica
2. N (Nasion) - Most anterior point of frontonasal suture
3. A (Point A) - Deepest point on maxilla between ANS and prosthion
4. B (Point B) - Deepest point on mandible between infradentale and pogonion
5. Pog (Pogonion) - Most anterior point of chin
6. Go (Gonion) - Most posterior-inferior point of mandibular angle
7. Me (Menton) - Most inferior point of mandibular symphysis
8. Or (Orbitale) - Lowest point of orbital margin
9. Po (Porion) - Superior point of external auditory meatus
10. ANS (Anterior Nasal Spine) - Tip of anterior nasal spine
11. PNS (Posterior Nasal Spine) - Tip of posterior nasal spine
12. U1 (Upper Incisor) - Incisal edge of upper central incisor
13. L1 (Lower Incisor) - Incisal edge of lower central incisor

Please respond ONLY with a valid JSON object in this exact format:
{
  "landmarks": {
    "S": {"x": 0, "y": 0},
    "N": {"x": 0, "y": 0},
    "A": {"x": 0, "y": 0},
    "B": {"x": 0, "y": 0},
    "Pog": {"x": 0, "y": 0},
    "Go": {"x": 0, "y": 0},
    "Me": {"x": 0, "y": 0},
    "Or": {"x": 0, "y": 0},
    "Po": {"x": 0, "y": 0},
    "ANS": {"x": 0, "y": 0},
    "PNS": {"x": 0, "y": 0},
    "U1": {"x": 0, "y": 0},
    "L1": {"x": 0, "y": 0}
  },
  "confidence": 0.0,
  "notes": "any observations"
}

Do not include any text outside the JSON object."""

# ========== توابع کمکی ==========

def print_header(text):
    """چاپ عنوان با استایل"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_model_info(model):
    """نمایش اطلاعات مدل"""
    print(f"\n🤖 مدل: {model['name']}")
    print(f"   📦 Provider: {model['provider']}")
    print(f"   📝 {model['description']}")
    print(f"   🔗 ID: {model['id']}")

def encode_image(image_path):
    """تبدیل تصویر به base64"""
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            # تشخیص نوع فایل
            ext = Path(image_path).suffix.lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.webp': 'image/webp'
            }.get(ext, 'image/jpeg')
            
            return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        print(f"❌ خطا در خواندن تصویر: {e}")
        return None

def test_model(model, image_base64):
    """تست یک مدل"""
    print_model_info(model)
    print("⏳ در حال ارسال درخواست...")
    
    start_time = time.time()
    
    try:
        # ساخت درخواست
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "DentalAI - Cephalometric Test"
        }
        
        payload = {
            "model": model["id"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": image_base64}}
                    ]
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        # ارسال درخواست
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # بررسی پاسخ
        if response.status_code != 200:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get('error', {}).get('message', response.text)
            print(f"❌ خطا: {response.status_code}")
            print(f"   پیام: {error_msg}")
            return {
                "success": False,
                "model": model["name"],
                "error": error_msg,
                "processing_time": processing_time
            }
        
        data = response.json()
        content = data['choices'][0]['message']['content']
        
        # پاک کردن markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        # Parse JSON
        try:
            parsed = json.loads(content.strip())
            landmarks = parsed.get('landmarks', {})
            confidence = parsed.get('confidence', 0)
            
            print(f"✅ موفق!")
            print(f"   ⏱️  زمان: {processing_time:.2f}s")
            print(f"   📊 Tokens: {data.get('usage', {}).get('total_tokens', 'N/A')}")
            print(f"   🎯 اطمینان: {confidence}")
            print(f"   📍 Landmarks: {len(landmarks)}")
            
            return {
                "success": True,
                "model": model["name"],
                "landmarks": landmarks,
                "confidence": confidence,
                "processing_time": processing_time,
                "tokens": data.get('usage', {}),
                "raw_response": content
            }
            
        except json.JSONDecodeError as e:
            print(f"⚠️  خطا در Parse JSON: {e}")
            print(f"   پاسخ خام: {content[:200]}...")
            return {
                "success": False,
                "model": model["name"],
                "error": f"JSON parse error: {e}",
                "raw_response": content,
                "processing_time": processing_time
            }
            
    except requests.Timeout:
        print("❌ Timeout - مدل پاسخ نداد")
        return {
            "success": False,
            "model": model["name"],
            "error": "Request timeout"
        }
    except Exception as e:
        print(f"❌ خطا: {e}")
        return {
            "success": False,
            "model": model["name"],
            "error": str(e)
        }

def save_results(results, output_file="test_results.json"):
    """ذخیره نتایج در فایل"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 نتایج در {output_file} ذخیره شد")
    except Exception as e:
        print(f"\n❌ خطا در ذخیره فایل: {e}")

def print_summary(results):
    """چاپ خلاصه نتایج"""
    print_header("📊 خلاصه نتایج")
    
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    
    print(f"\n✅ موفق: {len(successful)}/{len(results)}")
    print(f"❌ ناموفق: {len(failed)}/{len(results)}")
    
    if successful:
        print("\n🏆 بهترین مدل‌ها (بر اساس زمان):")
        sorted_results = sorted(successful, key=lambda x: x.get('processing_time', float('inf')))
        for i, result in enumerate(sorted_results[:3], 1):
            print(f"   {i}. {result['model']}: {result.get('processing_time', 0):.2f}s")
    
    if failed:
        print("\n⚠️  مدل‌های ناموفق:")
        for result in failed:
            print(f"   - {result['model']}: {result.get('error', 'Unknown error')}")

# ========== Main ==========

def main():
    print_header("🦷 تست مدل‌های OpenRouter برای Cephalometric Analysis")
    
    # بررسی API Key
    if OPENROUTER_API_KEY == "sk-or-v1-...":
        print("\n❌ لطفاً ابتدا API Key خود را در متغیر OPENROUTER_API_KEY قرار دهید")
        print("   1. به https://openrouter.ai/keys بروید")
        print("   2. یک API key جدید ایجاد کنید")
        print("   3. کلید را در خط 19 این فایل قرار دهید")
        return
    
    # دریافت مسیر تصویر
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("\n📷 مسیر تصویر Cephalometric را وارد کنید: ").strip()
    
    if not Path(image_path).exists():
        print(f"\n❌ تصویر پیدا نشد: {image_path}")
        return
    
    print(f"✅ تصویر یافت شد: {image_path}")
    
    # تبدیل تصویر به base64
    print("📦 در حال تبدیل تصویر به base64...")
    image_base64 = encode_image(image_path)
    if not image_base64:
        return
    
    # انتخاب مدل‌ها
    print("\n🤖 مدل‌های موجود:")
    for i, model in enumerate(MODELS, 1):
        print(f"   {i}. {model['name']} ({model['provider']})")
    
    choice = input("\nتمام مدل‌ها تست شوند؟ (y/n) یا شماره مدل‌ها را وارد کنید (مثال: 1,3,5): ").strip()
    
    if choice.lower() == 'y':
        models_to_test = MODELS
    elif choice.lower() == 'n':
        print("خروج...")
        return
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            models_to_test = [MODELS[i] for i in indices if 0 <= i < len(MODELS)]
        except:
            print("❌ ورودی نامعتبر")
            return
    
    if not models_to_test:
        print("❌ هیچ مدلی انتخاب نشد")
        return
    
    # تست مدل‌ها
    results = []
    for i, model in enumerate(models_to_test, 1):
        print(f"\n{'='*70}")
        print(f"  تست {i}/{len(models_to_test)}")
        result = test_model(model, image_base64)
        results.append(result)
        
        # تاخیر بین درخواست‌ها
        if i < len(models_to_test):
            print("\n⏸️  تاخیر 2 ثانیه...")
            time.sleep(2)
    
    # نمایش خلاصه
    print_summary(results)
    
    # ذخیره نتایج
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, f"test_results_{timestamp}.json")
    
    print("\n✅ تست‌ها کامل شد!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  توسط کاربر متوقف شد")
    except Exception as e:
        print(f"\n❌ خطای غیرمنتظره: {e}")

