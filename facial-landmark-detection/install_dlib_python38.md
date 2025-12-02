# راهنمای نصب dlib برای Python 3.8 در Windows

## مشکل فعلی
CMake واقعی نصب نیست. فایل `cmake.exe` در `Python38\Scripts` یک ماژول Python است نه CMake واقعی.

## راه‌حل: استفاده از Pre-built Wheel (توصیه می‌شود)

برای Python 3.8، از pre-built wheel استفاده کنید:

```bash
pip install https://github.com/sachadee/Dlib/releases/download/v19.22/dlib-19.22.99-cp38-cp38-win_amd64.whl
```

یا اگر لینک بالا کار نکرد، این را امتحان کنید:

```bash
pip install dlib-bin
```

## راه‌حل جایگزین: نصب CMake واقعی

### مرحله 1: دانلود CMake
1. به [cmake.org/download](https://cmake.org/download/) بروید
2. "Windows x64 Installer" را دانلود کنید
3. نصب کنید و **حتماً** گزینه "Add CMake to system PATH" را انتخاب کنید

### مرحله 2: بررسی نصب CMake
```bash
cmake --version
```

اگر خطا داد، CMake به PATH اضافه نشده است. باید:
1. CMake را uninstall کنید
2. دوباره نصب کنید و این بار گزینه PATH را انتخاب کنید
3. یا دستی به PATH اضافه کنید: `C:\Program Files\CMake\bin`

### مرحله 3: نصب dlib
```bash
pip install cmake
pip install dlib
```

## راه‌حل سریع: استفاده از MediaPipe فقط

اگر نمی‌خواهید با dlib مشکل داشته باشید، فقط MediaPipe را استفاده کنید:

```bash
pip install -r requirements-basic.txt
```

MediaPipe به اندازه کافی خوب است و نیاز به CMake ندارد.

















