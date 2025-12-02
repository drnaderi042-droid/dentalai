# شروع سریع - فقط 5 دقیقه! ⚡

## می‌خواهید سریع شروع کنید؟ اینجا را بخوانید!

## قدم 1: کپی کردن کد (30 ثانیه)

```javascript
// در فایل خودتان این کد را بنویسید:
import { CephalometricRAGService } from 'src/utils/rag/cephalometric-rag-service';

const RAG = new CephalometricRAGService();
```

## قدم 2: آماده کردن اطلاعات بیمار (1 دقیقه)

```javascript
// اطلاعات بیمار را مثل این آماده کنید:
const بیمار = {
  patientId: '123',
  age: 14,                    // سن
  gender: 'male',            // 'male' یا 'female'
  cephalometricMeasurements: {
    SNA: 85,                 // زاویه فک بالا
    SNB: 78,                 // زاویه فک پایین
    ANB: 7,                  // رابطه فک‌ها
    // می‌توانید پارامترهای دیگر هم اضافه کنید
  }
};
```

## قدم 3: استفاده! (10 ثانیه)

```javascript
// فقط این یک خط را بنویسید:
const جواب = await RAG.analyzePatient(بیمار);

// تمام! جواب آماده است
console.log(جواب.تشخیص);        // "کلاس II اسکلتی"
console.log(جواب.توصیه‌ها);      // لیست توصیه‌ها
console.log(جواب.توضیحات);      // توضیحات کامل
```

## مثال کامل (کپی و استفاده کنید!)

```javascript
import { CephalometricRAGService } from 'src/utils/rag/cephalometric-rag-service';

async function مثال_ساده() {
  // 1. ساخت RAG
  const RAG = new CephalometricRAGService();
  
  // 2. اطلاعات بیمار
  const بیمار = {
    patientId: 'P001',
    age: 14,
    gender: 'male',
    cephalometricMeasurements: {
      SNA: 85,
      SNB: 78,
      ANB: 7,
    }
  };
  
  // 3. تحلیل
  const جواب = await RAG.analyzePatient(بیمار, 'چه درمانی نیاز دارد؟');
  
  // 4. نمایش نتیجه
  console.log('=== نتیجه ===');
  console.log('تشخیص:', جواب.diagnosis);
  console.log('توصیه‌ها:', جواب.recommendations);
  console.log('توضیحات:', جواب.explanation);
  
  return جواب;
}

// اجرا
مثال_ساده();
```

## استفاده در React (کامپوننت)

```javascript
import { useState, useEffect } from 'react';
import { CephalometricRAGService } from 'src/utils/rag/cephalometric-rag-service';

function PatientAnalysis({ patientId }) {
  const [نتیجه, setنتیجه] = useState(null);
  const [در_حال_بارگذاری, setدر_حال_بارگذاری] = useState(true);
  
  useEffect(() => {
    async function تحلیل() {
      // دریافت اطلاعات بیمار (از API یا state)
      const بیمار = {
        patientId: patientId,
        age: 14,
        gender: 'male',
        cephalometricMeasurements: {
          SNA: 85,
          SNB: 78,
          ANB: 7,
        }
      };
      
      // استفاده از RAG
      const RAG = new CephalometricRAGService();
      const جواب = await RAG.analyzePatient(بیمار);
      
      setنتیجه(جواب);
      setدر_حال_بارگذاری(false);
    }
    
    تحلیل();
  }, [patientId]);
  
  if (در_حال_بارگذاری) {
    return <div>در حال تحلیل...</div>;
  }
  
  return (
    <div>
      <h2>تشخیص: {نتیجه.diagnosis}</h2>
      
      <h3>توصیه‌ها:</h3>
      <ul>
        {نتیجه.recommendations.map((توصیه, i) => (
          <li key={i}>{توصیه}</li>
        ))}
      </ul>
      
      <h3>توضیحات:</h3>
      <p>{نتیجه.explanation}</p>
    </div>
  );
}
```

## سوالات سریع

### ❓ چطور اطلاعات بیمار را از API بگیرم؟

```javascript
// فرض کنید یک API دارید که اطلاعات بیمار را می‌دهد
const بیمار = await fetch(`/api/patients/${patientId}`).then(r => r.json());

// حالا از RAG استفاده کنید
const RAG = new CephalometricRAGService();
const جواب = await RAG.analyzePatient(بیمار);
```

### ❓ چطور در یک دکمه استفاده کنم؟

```javascript
function AnalyzeButton({ patientId }) {
  const [جواب, setجواب] = useState(null);
  
  async function handleClick() {
    const بیمار = await getPatientData(patientId);
    const RAG = new CephalometricRAGService();
    const نتیجه = await RAG.analyzePatient(بیمار);
    setجواب(نتیجه);
  }
  
  return (
    <div>
      <button onClick={handleClick}>تحلیل بیمار</button>
      {جواب && <div>{جواب.diagnosis}</div>}
    </div>
  );
}
```

### ❓ چطور خطاها را مدیریت کنم؟

```javascript
try {
  const RAG = new CephalometricRAGService();
  const جواب = await RAG.analyzePatient(بیمار);
  // استفاده از جواب
} catch (خطا) {
  console.error('خطا در تحلیل:', خطا);
  // نمایش پیام خطا به کاربر
}
```

## تمام! 🎉

حالا می‌توانید از RAG استفاده کنید!

**نیاز به کمک بیشتر؟**
- 📖 راهنمای کامل: `RAG_SIMPLE_GUIDE.md`
- 💡 مثال‌های بیشتر: `rag-example.ts`
- 📚 مستندات فنی: `README.md`





