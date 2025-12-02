# 🔧 رفع مشکلات تشخیص لندمارک‌های صورت

## ✅ تغییرات انجام شده

مشابه بخش "آنالیز داخل دهان"، همان تغییرات را برای "تشخیص لندمارک‌های صورت" اعمال کردم:

### 1. Backend API Endpoint ✅
- فایل: `minimal-api-dev-v6/src/pages/api/patients/[id]/facial-landmark-analysis.ts`
- GET: بارگذاری آنالیز ذخیره شده
- POST: ذخیره نتایج جدید آنالیز
- فیلد جدید در Prisma Schema: `facialLandmarkAnalysis`

### 2. Frontend Changes ✅

#### فایل‌های تغییر یافته:
1. **`vite-js/src/sections/orthodontics/patient/view/patient-orthodontics-view.jsx`**
   - اضافه کردن `patientId={id}` به FacialLandmarkView

2. **`vite-js/src/sections/facial-landmark/view/facial-landmark-view.jsx`**
   - اضافه کردن Dialog imports
   - اضافه کردن axios, auth, toast imports
   - اضافه کردن `patientId` prop
   - اضافه کردن state های جدید برای history و dropdown
   - پیاده‌سازی `loadAnalysisHistory()` و `saveAnalysis()`
   - اضافه کردن dropdown انتخاب آنالیز
   - اضافه کردن dialog حذف آنالیز
   - فراخوانی خودکار `saveAnalysis()` بعد از تشخیص موفق

3. **`minimal-api-dev-v6/prisma/schema.prisma`**
   - اضافه کردن فیلد `facialLandmarkAnalysis String?`

## 📊 ساختار جدید

```
┌─────────────────────────────────────────────┐
│  😊 تشخیص لندمارک‌های صورت                 │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ [انتخاب آنالیز برای مشاهده ▼]      [🗑️]  │
│  📅 آنالیز 1 - mediapipe                   │
│     1402/08/20 - 14:30                      │
└─────────────────────────────────────────────┘

┌────────────────┬────────────────────────────┐
│   آپلود        │  🖼️ تصویر + لندمارک‌ها    │
│   مدل AI       │  [LandmarkVisualizer]      │
│   📋 فایل‌ها   │  📊 آنالیز زیبایی صورت   │
│   🔍 تشخیص     │                             │
└────────────────┴────────────────────────────┘
```

## 🔧 تغییرات تکنیکال

### State های جدید:
```javascript
const [lastSavedAnalysis, setLastSavedAnalysis] = useState(null);
const [analysisHistory, setAnalysisHistory] = useState([]);
const [selectedAnalysisIndex, setSelectedAnalysisIndex] = useState(0);
const [isLoadingHistory, setIsLoadingHistory] = useState(false);
const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
const [analysisToDelete, setAnalysisToDelete] = useState(null);
const [deleting, setDeleting] = useState(false);
```

### توابع جدید:
```javascript
// بارگذاری تاریخچه از backend
const loadAnalysisHistory = useCallback(async () => {
  const res = await axios.get(`${endpoints.patients}/${patientId}/facial-landmark-analysis`);
  // ...
}, [patientId, user?.accessToken]);

// ذخیره نتایج آنالیز در backend
const saveAnalysis = useCallback(async (resultsToSave = null) => {
  const payload = {
    analyses: [{
      serverImageId: selectedFile?.serverId || null,
      modelId: selectedModel,
      result: currentResult,
      landmarks: currentLandmarks,
      beautyAnalysis: currentBeauty,
    }]
  };
  await axios.post(`${endpoints.patients}/${patientId}/facial-landmark-analysis`, payload);
  toast.success('✅ نتایج آنالیز ذخیره شد');
}, [patientId, result, landmarks, beautyAnalysis, selectedModel, selectedFile, user?.accessToken]);
```

### Auto-Save:
بعد از تشخیص موفق، خودکار ذخیره می‌شود:
```javascript
// در handleDetect
if (patientId && parsedResult) {
  await saveAnalysis(parsedResult);
}
```

## 🎯 ویژگی‌های جدید

### 1. ✅ Dropdown انتخاب آنالیز
- نمایش تمام آنالیزهای ذخیره شده
- نمایش شماره آنالیز، مدل استفاده شده، و تاریخ
- فرمت تاریخ به فارسی
- فقط زمانی نمایش داده می‌شود که `patientId` و حداقل یک آنالیز وجود داشته باشد

### 2. ✅ دکمه حذف
- آیکون 🗑️ کنار dropdown
- باز کردن dialog تأیید
- نمایش اطلاعات آنالیز در dialog
- حالت loading در حین حذف

### 3. ✅ Dialog تأیید حذف
- عنوان: "حذف آنالیز لندمارک صورت"
- نمایش مدل استفاده شده
- دکمه‌های "انصراف" و "حذف"
- پس از حذف، اولین آنالیز باقی‌مانده انتخاب می‌شود

### 4. ✅ Toast Notifications
- `✅ نتایج آنالیز ذخیره شد` - بعد از ذخیره موفق
- `❌ خطا در ذخیره نتایج آنالیز` - در صورت خطا
- `✅ آنالیز با موفقیت حذف شد` - بعد از حذف
- `❌ خطا در حذف آنالیز` - در صورت خطا در حذف

### 5. ✅ Auto-Save
- بعد از هر تشخیص موفق، نتایج خودکار ذخیره می‌شوند
- شامل: landmarks، beauty analysis، model ID
- اگر `patientId` وجود نداشته باشد، ذخیره نمی‌شود

### 6. ✅ Load History on Mount
- با باز شدن صفحه، تاریخچه خودکار بارگذاری می‌شود
- آخرین آنالیز به طور پیش‌فرض نمایش داده می‌شود

## 🚀 نحوه استفاده

### تشخیص جدید:
```
1. به صفحه بیمار بروید
2. تب "تشخیص لندمارک صورت" را انتخاب کنید
3. تصویر صورت را آپلود کنید
4. مدل AI را انتخاب کنید (mediapipe, dlib, face_alignment, ...)
5. "تشخیص با AI" را کلیک کنید
6. نتایج نمایش داده شده و خودکار ذخیره می‌شوند
7. Toast موفقیت: "✅ نتایج آنالیز ذخیره شد"
```

### مشاهده تاریخچه:
```
1. در بالای صفحه، dropdown "انتخاب آنالیز برای مشاهده" را ببینید
2. آنالیز مورد نظر را انتخاب کنید
3. نتایج آن آنالیز (landmarks + beauty analysis) نمایش داده می‌شود
```

### حذف آنالیز:
```
1. آنالیز مورد نظر را از dropdown انتخاب کنید
2. روی آیکون 🗑️ کنار dropdown کلیک کنید
3. در dialog، "حذف" را کلیک کنید
4. آنالیز حذف می‌شود
```

## 📝 Backend API

### GET `/api/patients/[id]/facial-landmark-analysis`
```typescript
Response:
{
  analysis: {
    analyses: [{
      serverImageId: string | null,
      modelId: string,
      result: object,
      landmarks: array,
      beautyAnalysis: object
    }],
    totalAnalyses: number,
    lastUpdated: string
  },
  lastUpdated: string
}
```

### POST `/api/patients/[id]/facial-landmark-analysis`
```typescript
Request:
{
  analyses: [{
    serverImageId: string | null,
    modelId: string,
    result: object,
    landmarks: array,
    beautyAnalysis: object
  }]
}

Response:
{
  success: true,
  message: "Facial landmark analysis saved successfully",
  analysis: { analyses, totalAnalyses, lastUpdated },
  lastUpdated: string
}
```

## 🗄️ Database Schema

```prisma
model Patient {
  // ... other fields
  facialLandmarkAnalysis String?  // JSON data for facial landmark analysis results
}
```

## ⚠️ نکات مهم

1. **Migration**: برای اعمال تغییرات schema، باید migration اجرا شود:
   ```bash
   cd minimal-api-dev-v6
   npx prisma migrate dev --name add_facial_landmark_analysis
   # یا
   npx prisma db push
   ```

2. **Endpoint**: `/api/patients/[id]/facial-landmark-analysis`
   - Requires authentication (Bearer token)
   - patientId باید معتبر باشد

3. **Data Structure**: نتایج به صورت JSON ذخیره می‌شوند:
   ```json
   {
     "analyses": [
       {
         "serverImageId": null,
         "modelId": "mediapipe",
         "result": { /* detection result */ },
         "landmarks": [ /* landmark points */ ],
         "beautyAnalysis": { /* beauty scores */ }
       }
     ],
     "totalAnalyses": 1,
     "lastUpdated": "2025-11-11T..."
   }
   ```

## 🆚 مقایسه قبل و بعد

| ویژگی | قبل | بعد |
|------|-----|-----|
| **ذخیره‌سازی** | ❌ ندارد | ✅ خودکار در backend |
| **تاریخچه** | ❌ ندارد | ✅ Dropdown با تمام آنالیزها |
| **انتخاب آنالیز** | ❌ ندارد | ✅ Select از dropdown |
| **حذف آنالیز** | ❌ ندارد | ✅ دکمه trash + Dialog |
| **Toast نوتیفیکیشن** | ❌ ندارد | ✅ Success/Error messages |
| **بارگذاری خودکار** | ❌ ندارد | ✅ Load on mount |

## 📸 Console Logs

برای debugging، console logs زیر اضافه شدند:

```javascript
// Save
console.log('💾 Saving facial landmark analysis to backend:', payload);
console.log('✅ Facial landmark analysis saved successfully:', response.data);
console.error('❌ Failed to save facial landmark analysis:', err);

// Load
console.error('Failed to load analysis history:', err);

// Delete
console.error('Error deleting analysis:', error);
```

## ✨ همسان با Intra-Oral

تمام تغییرات دقیقاً مشابه بخش "آنالیز داخل دهان" پیاده‌سازی شد:
- ✅ همان ساختار dropdown
- ✅ همان dialog حذف
- ✅ همان توابع save/load
- ✅ همان toast notifications
- ✅ همان UX flow

---

**تاریخ**: 11 نوامبر 2025
**نسخه**: 1.0
**وضعیت**: ✅ کامل و آماده تست
**مشابه**: Intra-Oral Analysis System

