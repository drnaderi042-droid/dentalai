# فرمول‌های محاسبه پارامترهای سفالومتریک

## مقدمه

این سند فرمول‌های محاسباتی پارامترهای سفالومتریک مورد استفاده در سیستم را توضیح می‌دهد. تمامی محاسبات بر اساس مختصات لندمارک‌های شناسایی شده در تصاویر رادیوگرافی لترال صورت انجام می‌شود.

## توابع پایه ریاضی

### ۱. محاسبه زاویه بین سه نقطه
```javascript
const calculateAngle = (p1, vertex, p2) => {
  const angle1 = Math.atan2(p1.y - vertex.y, p1.x - vertex.x);
  const angle2 = Math.atan2(p2.y - vertex.y, p2.x - vertex.x);
  let angle = (angle2 - angle1) * (180 / Math.PI);
  if (angle < 0) angle += 360;
  return angle > 180 ? 360 - angle : angle;
};
```

### ۲. محاسبه زاویه خط نسبت به افق
```javascript
const calculateLineAngle = (p1, p2) => Math.atan2(p2.y - p1.y, p2.x - p1.x) * (180 / Math.PI);
```

### ۳. محاسبه زاویه بین دو خط
```javascript
const calculateAngleBetweenLines = (line1Start, line1End, line2Start, line2End) => {
  const v1x = line1End.x - line1Start.x;
  const v1y = line1End.y - line1Start.y;
  const v2x = line2End.x - line2Start.x;
  const v2y = line2End.y - line2Start.y;
  const dotProduct = v1x * v2x + v1y * v2y;
  const mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
  const mag2 = Math.sqrt(v2x * v2x + v2y * v2y);
  if (mag1 === 0 || mag2 === 0) return 0;
  const cosAngle = dotProduct / (mag1 * mag2);
  const clampedCos = Math.max(-1, Math.min(1, cosAngle));
  const angleRad = Math.acos(clampedCos);
  const angleDeg = angleRad * (180 / Math.PI);
  return angleDeg > 90 ? 180 - angleDeg : angleDeg;
};
```

## پارامترهای آنالیز Steiner

### SNA (Sella-Nasion-A Point Angle)
**فرمول:** زاویه بین نقاط S، N و A
```javascript
measures.SNA = calculateAngle(S, N, A);
```
**توضیح:** نشان‌دهنده موقعیت افقی ماگزیلا نسبت به پایه جمجمه است.

### SNB (Sella-Nasion-B Point Angle)
**فرمول:** زاویه بین نقاط S، N و B
```javascript
measures.SNB = calculateAngle(S, N, B);
```
**توضیح:** نشان‌دهنده موقعیت افقی مندیبل نسبت به پایه جمجمه است.

### ANB (A Point-Nasion-B Point Angle)
**فرمول:** تفاوت زاویه SNA و SNB
```javascript
measures.ANB = SNA - SNB;
```
**توضیح:** نشان‌دهنده رابطه ساژیتال بین ماگزیلا و مندیبل است.

### FMA (Frankfort-Mandibular Plane Angle)
**فرمول:** زاویه بین خط فرانکفورت (Or-Po) و صفحه مندیبولار (Go-Me)
```javascript
measures.FMA = calculateAngleBetweenLines(Or, Po, Go, Me);
```
**توضیح:** نشان‌دهنده جهت رشد عمودی صورت است.

### FMIA (Frankfort-Mandibular Incisor Angle)
**فرمول:** زاویه بین خط فرانکفورت و محور incisor پایین
```javascript
const frankfortAngle = calculateLineAngle(Or, Po);
const incisorAngle = calculateLineAngle(L1, Me);
let angleDiff = Math.abs(frankfortAngle - incisorAngle);
if (angleDiff > 180) angleDiff = 360 - angleDiff;
if (angleDiff > 90) {
  measures.FMIA = 180 - angleDiff;
} else {
  measures.FMIA = angleDiff;
}
```
**توضیح:** نشان‌دهنده傾斜 incisor پایین است.

### IMPA (Incisor-Mandibular Plane Angle)
**فرمول:** زاویه بین محور incisor پایین و صفحه مندیبولار
```javascript
// اگر نقاط LIA و LIT موجود باشند:
measures.IMPA = calculateAngleBetweenLines(Me, Go, LIA, LIT);
// در غیر این صورت:
measures.IMPA = calculateAngleBetweenLines(Me, Go, L1, Me);
```
**توضیح:** نشان‌دهنده傾斜 incisor پایین نسبت به پایه فکی است.

### GoGn (Gonion-Gnathion Angle)
**فرمول:** زاویه خط Go-Gn نسبت به افق
```javascript
const angle = calculateLineAngle(Go, Gn);
measures.GoGn = Math.abs(angle);
if (measures.GoGn > 90) measures.GoGn = 180 - measures.GoGn;
```
**توضیح:** نشان‌دهنده شیب صفحه مندیبولار است.

### GoGn-SN (Gonion-Gnathion to Sella-Nasion Angle)
**فرمول:** تفاوت زاویه خطوط Go-Gn و S-N
```javascript
const snAngle = calculateLineAngle(S, N);
const gognAngle = calculateLineAngle(Go, Gn);
measures['GoGn-SN'] = Math.abs(snAngle - gognAngle);
```
**توضیح:** نشان‌دهنده رابطه صفحه مندیبولار با پایه جمجمه است.

### U1-SN (Upper Incisor to Sella-Nasion Angle)
**فرمول:** زاویه بین خط incisor بالا (UIA-UIT) و خط S-N
```javascript
// ابتدا خط U1 را از نقاط UIA و UIT محاسبه می‌کنیم
const uiaLandmark = getLandmark(['UIA', 'uia', 'Uia', 'upper_incisor_apex', 'Upper_incisor_apex', 'upper incisor apex']) ||
                   findLandmarkByPartial(['uia', 'upper', 'incisor', 'apex']);
const uitLandmark = getLandmark(['UIT', 'uit', 'Uit', 'upper_incisor_tip', 'Upper_incisor_tip', 'upper incisor tip']) ||
                   findLandmarkByPartial(['uit', 'upper', 'incisor', 'tip']);

if (uiaLandmark && uitLandmark && sLandmark && nLandmark) {
  const u1Angle = calculateLineAngle(uiaLandmark, uitLandmark);
  const snAngle = calculateLineAngle(sLandmark, nLandmark);
  const angleDiff = Math.abs(u1Angle - snAngle);
  measures['U1-SN'] = Math.round(Math.max(0, Math.min(180, angleDiff)) * 10) / 10;
}
```
**توضیح:** نشان‌دهنده傾斜 incisor بالا است. UIA = Upper Incisor Apex, UIT = Upper Incisor Tip.

### L1-MP (Lower Incisor to Mandibular Plane Angle)
**فرمول:** زاویه بین خط incisor پایین (LIA-LIT) و صفحه مندیبولار (Go-Me)
```javascript
// ابتدا خط L1 را از نقاط LIA و LIT محاسبه می‌کنیم
const liaLandmark = getLandmark(['LIA', 'lia', 'Lia', 'lower_incisor_apex', 'Lower_incisor_apex', 'lower incisor apex']) ||
                   findLandmarkByPartial(['lia', 'lower', 'incisor', 'apex']);
const litLandmark = getLandmark(['LIT', 'lit', 'Lit', 'lower_incisor_tip', 'Lower_incisor_tip', 'lower incisor tip']) ||
                   findLandmarkByPartial(['lit', 'lower', 'incisor', 'tip']);

if (liaLandmark && litLandmark && goLandmark && meLandmark) {
  measures['L1-MP'] = calculateAngleBetweenLines(meLandmark, goLandmark, liaLandmark, litLandmark);
  measures['L1-MP'] = Math.round(Math.max(0, Math.min(180, measures['L1-MP'])) * 10) / 10;
}
```
**توضیح:** نشان‌دهنده傾斜 incisor پایین است. LIA = Lower Incisor Apex, LIT = Lower Incisor Tip.

### Overbite
**فرمول:** فاصله عمودی بین U1 و L1 (در واحد میلی‌متر)
```javascript
const verticalDistance = Math.abs(L1.y - U1.y);
const conversionFactor = pixelToMmConversion; // یا محاسبه از نقاط P1-P2
measures.Overbite = (verticalDistance * conversionFactor);
```
**توضیح:** عمق گزش عمودی است.

### Overjet
**فرمول:** فاصله افقی بین U1 و L1 (در واحد میلی‌متر)
```javascript
const horizontalDistance = U1.x - L1.x;
const conversionFactor = pixelToMmConversion; // یا محاسبه از نقاط P1-P2
measures.Overjet = Math.abs(horizontalDistance) * conversionFactor;
```
**توضیح:** protrusion افقی است.

## پارامترهای آنالیز Ricketts

### Facial Axis
**فرمول:** زاویه بین خطوط Ba-Na و Pt-Gn
```javascript
const baNaAngle = calculateLineAngle(Ba, Na);
const ptGnAngle = calculateLineAngle(Pt, Gn);
measures.FacialAxis = Math.abs(baNaAngle - ptGnAngle);
```
**توضیح:** نشان‌دهنده جهت رشد صورت است.

### Facial Depth
**فرمول:** زاویه بین خطوط N-Pog و Or-Po (Frankfort Horizontal)
```javascript
const nPoAngle = calculateLineAngle(N, Po);
const fhAngle = calculateLineAngle(Or, Po); // FH
measures.FacialDepth = Math.abs(nPoAngle - fhAngle);
```
**توضیح:** نشان‌دهنده عمق صورت است.

### Lower Face Height
**فرمول:** نسبت فاصله عمودی صورت پایین به کل فاصله عمودی صورت
```javascript
// فقط فاصله عمودی (تفاوت y)
const ansMeVertical = Math.abs(ANS.y - Me.y);
const nMeVertical = Math.abs(N.y - Me.y);
measures.LowerFaceHeight = (ansMeVertical / nMeVertical) * 100;
```
**توضیح:** نسبت فاصله عمودی ANS-Me به فاصله عمودی N-Me است.

### Mandibular Plane
**فرمول:** زاویه صفحه مندیبولار نسبت به FH
```javascript
const goMeAngle = calculateLineAngle(Go, Me);
const fhAngle = calculateLineAngle(Or, Po);
measures.MandibularPlane = Math.abs(goMeAngle - fhAngle);
```
**توضیح:** شیب صفحه مندیبولار است.

### Convexity
**فرمول:** فاصله نقطه A از خط N-Pog
```javascript
// محاسبه فاصله نقطه A از خط N-Pog
const nPogVector = { x: Pog.x - N.x, y: Pog.y - N.y };
const aVector = { x: A.x - N.x, y: A.y - N.y };
const crossProduct = nPogVector.x * aVector.y - nPogVector.y * aVector.x;
const magnitude = Math.sqrt(nPogVector.x^2 + nPogVector.y^2);
measures.Convexity = crossProduct / magnitude;
```
**توضیح:** محدب بودن پروفایل صورت است.

## پارامترهای آنالیز McNamara

### N-A-Pog
**فرمول:** زاویه بین نقاط N، A و Pog
```javascript
measures['N-A-Pog'] = calculateAngle(N, A, Pog);
```
**توضیح:** نشان‌دهنده محدب بودن صورت است.

### Co-A (Maxillary Length)
**فرمول:** فاصله بین نقاط Co و A
```javascript
measures['Co-A'] = Math.sqrt((Co.x - A.x)^2 + (Co.y - A.y)^2) * pixelToMmConversion;
```
**توضیح:** طول ماگزیلا است.

### Co-Gn (Mandibular Length)
**فرمول:** فاصله بین نقاط Co و Gn
```javascript
measures['Co-Gn'] = Math.sqrt((Co.x - Gn.x)^2 + (Gn.y - Co.y)^2) * pixelToMmConversion;
```
**توضیح:** طول مندیبل است.

### Wits Appraisal
**فرمول:** فاصله عمودی بین نقاط AO و BO نسبت به نیروی عمودی
```javascript
// محاسبه نقاط AO و BO بر خط نیروی عمودی
// سپس اندازه‌گیری فاصله عمودی بین آنها
```
**توضیح:** ارزیابی رابطه ساژیتال فک‌ها است.

## پارامترهای آنالیز Wits

### AO-BO
**فرمول:** فاصله عمودی بین نقاط AO و BO بر خط نیروی عمودی
```javascript
// AO: نقطه A بر خط نیروی عمودی
// BO: نقطه B بر خط نیروی عمودی
// AO-BO = فاصله عمودی AO - BO
```
**توضیح:** نشان‌دهنده کلاس اسکلتال است.

### PP/Go-Gn
**فرمول:** زاویه بین صفحه پلاتین و صفحه مندیبولار
```javascript
measures['PP/Go-Gn'] = calculateAngleBetweenLines(ANS, PNS, Go, Gn);
```
**توضیح:** نشان‌دهنده جهت رشد عمودی است.

## پارامترهای آنالیز Bjork

### S-Ar/Go-Gn Ratio
**فرمول:** نسبت طول سوراسلار به گونی ذره
```javascript
const sArLength = Math.sqrt((S.x - Ar.x)^2 + (S.y - Ar.y)^2);
const goGnLength = Math.sqrt((Go.x - Gn.x)^2 + (Go.y - Gn.y)^2);
measures['S-Ar/Go-Gn Ratio'] = (sArLength / goGnLength) * 100;
```
**توضیح:** نشان‌دهنده جهت رشد است.

### MP/SN Angle
**فرمول:** زاویه صفحه مندیبولار نسبت به SN
```javascript
const mpAngle = calculateLineAngle(Go, Me);
const snAngle = calculateLineAngle(S, N);
measures['MP/SN Angle'] = Math.abs(mpAngle - snAngle);
```
**توضیح:** شیب صفحه مندیبولار است.

## پارامترهای آنالیز Jarabak

### S-Go/Ar-Go Ratio
**فرمول:** فاکتور CAG رشد
```javascript
const sGoLength = Math.sqrt((S.x - Go.x)^2 + (S.y - Go.y)^2);
const arGoLength = Math.sqrt((Ar.x - Go.x)^2 + (Ar.y - Go.y)^2);
measures['S-Go/Ar-Go Ratio'] = (sGoLength / arGoLength) * 100;
```
**توضیح:** نشان‌دهنده جهت رشد است.

### Go-Gn/SN Angle
**فرمول:** زاویه رشد
```javascript
const gognAngle = calculateLineAngle(Go, Gn);
const snAngle = calculateLineAngle(S, N);
measures['Go-Gn/SN Angle'] = Math.abs(gognAngle - snAngle);
```
**توضیح:** زاویه رشد صورت است.

## پارامترهای آنالیز Sassouni

### N-S-Ar
**فرمول:** زاویه بین نقاط N، S و Ar
```javascript
measures['N-S-Ar'] = calculateAngle(N, S, Ar);
```
**توضیح:** زاویه پایه جمجمه است.

### Go-Co//N-S
**فرمول:** میزان تمایز ساژیتال
```javascript
measures['Go-Co//N-S'] = calculateAngleBetweenLines(Go, Co, N, S);
```
**توضیح:** تمایز ساژیتال فک‌ها است.

## نکات مهم

1. **واحد اندازه‌گیری:** تمامی زاویه‌ها به درجه و فاصله‌ها به میلی‌متر تبدیل می‌شوند.

2. **ضریب تبدیل پیکسل به میلی‌متر:** معمولاً از فاصله بین نقاط P1 و P2 (معمولاً 10mm) محاسبه می‌شود.

3. **دقت محاسبات:** تمامی نتایج به یک رقم اعشار گرد می‌شوند.

4. **محدوده معتبر:** زاویه‌ها معمولاً بین 0 تا 180 درجه محدود می‌شوند.

5. **لندمارک‌های مورد استفاده:** S (Sella), N (Nasion), A (Point A), B (Point B), Go (Gonion), Me (Menton), U1 (Upper Incisor), L1 (Lower Incisor), PNS (Posterior Nasal Spine), ANS (Anterior Nasal Spine), Ar (Articulare), Co (Condylion), Pog (Pogonion), Gn (Gnathion), Or (Orbitale), Po (Porion), Pt (Pterygoid), Ba (Basion), Na (Nasion).

این فرمول‌ها بر اساس استانداردهای پذیرفته شده در ارتودنسی و جراحی دهان و فک و صورت طراحی شده‌اند.