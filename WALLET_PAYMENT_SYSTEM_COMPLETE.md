# 💳 سیستم کیف پول و پرداخت - راهنمای کامل

## 📅 تاریخ: 2025-10-30

---

## ✅ تغییرات انجام شده:

### 1. **بهبود UI صفحه Wallet** ✅

#### قبل:
```
- UI ساده
- دکمه button
- تراکنش‌ها در popup
- موجودی زیر متن
```

#### بعد:
```
- UI حرفه‌ای با gradient card
- دکمه li (کلیک‌پذیر list item)
- تراکنش‌ها حذف از popup
- موجودی کنار متن
- Quick amount buttons
- انتقال به /payment
```

**فایل‌ها:**
- ✅ `vite-js/src/layouts/components/wallet-button.jsx`
- ✅ `vite-js/src/sections/wallet/view/wallet-view.jsx`

---

### 2. **صفحه Payment فارسی و سفارشی** ✅

#### ویژگی‌ها:
```
✅ فارسی‌سازی کامل
✅ حذف billing address
✅ اضافه کردن زرینپال
✅ اضافه کردن NowPayments
✅ لوگوی درگاه‌ها
✅ نمایش نرخ دلار برای NowPayments
✅ محاسبه خودکار کارمزد
✅ Summary card با جزئیات
```

**فایل‌ها:**
- ✅ `vite-js/src/sections/payment/view/payment-view.jsx`
- ✅ `vite-js/src/sections/payment/payment-methods.jsx`
- ✅ `vite-js/src/sections/payment/payment-summary.jsx`

---

### 3. **API نرخ ارز از Bonbast** ✅

#### ویژگی‌ها:
```
✅ دریافت نرخ دلار از bonbast.com
✅ Cache 6 ساعته (هر 6 ساعت بروزرسانی)
✅ ذخیره در دیتابیس (ExchangeRate model)
✅ Fallback به last known rate
✅ تبدیل ریال به تومان
```

**فایل:**
- ✅ `minimal-api-dev-v6/src/pages/api/exchange-rate/index.ts`

**استفاده:**
```bash
GET /api/exchange-rate

Response:
{
  "success": true,
  "data": {
    "usd_to_irr": 520000,
    "eur_to_irr": 560000,
    "source": "bonbast",
    "fetched_at": "2025-10-30T...",
    "expires_at": "2025-10-30T..." // 6 hours later
  }
}
```

---

### 4. **Schema Prisma Updates** ✅

#### جداول جدید/بهبود یافته:

**Invoice Model:**
```prisma
model Invoice {
  // ... existing fields ...
  
  // NEW: Payment gateway info
  paymentGateway String? // zarinpal, nowpayments
  paymentStatus  String  @default("pending")
  transactionId  String?
  paidAt         DateTime?
  
  // NEW: User reference
  userId        String
  
  // NEW: Type
  type          String  @default("wallet_charge")
  currency      String  @default("IRR")
}
```

**ExchangeRate Model (NEW):**
```prisma
model ExchangeRate {
  id          String   @id @default(cuid())
  usdToIrr    Float    // USD to IRR (Toman)
  eurToIrr    Float?
  source      String   @default("bonbast")
  fetchedAt   DateTime @default(now())
  expiresAt   DateTime // 6 hours expiry
  createdAt   DateTime @default(now())
}
```

**فایل:**
- ✅ `minimal-api-dev-v6/prisma/schema.prisma`

**Migration:**
```bash
cd minimal-api-dev-v6
npx prisma migrate dev --name add_invoice_payment_and_exchange_rate
npx prisma generate
```

---

### 5. **Invoice System** ✅

#### APIs:

**Create Invoice:**
```bash
POST /api/invoice/create
Authorization: Bearer <token>

Body:
{
  "amount": 100000,
  "type": "wallet_charge",
  "paymentGateway": "zarinpal",
  "description": "شارژ کیف پول",
  "items": [...]
}

Response:
{
  "success": true,
  "data": {
    "id": "...",
    "invoiceNumber": "INV-...",
    "totalAmount": 101000, // با کارمزد
    "status": "pending",
    "paymentGateway": "zarinpal"
  }
}
```

**Get Invoice:**
```bash
GET /api/invoice/:id
Authorization: Bearer <token>

Response:
{
  "success": true,
  "data": {
    "id": "...",
    "invoiceNumber": "INV-...",
    "totalAmount": 101000,
    "items": [...],
    ...
  }
}
```

**List Invoices:**
```bash
GET /api/invoice/list?status=pending&limit=50
Authorization: Bearer <token>

Response:
{
  "success": true,
  "data": {
    "invoices": [...],
    "pagination": {
      "total": 10,
      "limit": 50,
      "offset": 0
    }
  }
}
```

**فایل‌ها:**
- ✅ `minimal-api-dev-v6/src/pages/api/invoice/create.ts`
- ✅ `minimal-api-dev-v6/src/pages/api/invoice/[id].ts`
- ✅ `minimal-api-dev-v6/src/pages/api/invoice/list.ts`

---

### 6. **Integration: Wallet → Payment → Invoice** ✅

#### Flow:

```
1. User: wallet page
   ↓
   انتخاب مبلغ + کلیک "شارژ کیف پول"
   ↓
2. Navigate to /payment با state: { amount, type, currency }
   ↓
3. Payment page:
   - دریافت نرخ ارز از API
   - نمایش درگاه‌های پرداخت
   - انتخاب درگاه
   ↓
4. کلیک "پرداخت":
   - Create invoice via API
   - Invoice ایجاد می‌شود با شماره یکتا
   - Navigate to /dashboard/invoice/:id
   ↓
5. Invoice page (future):
   - نمایش جزئیات invoice
   - دکمه پرداخت → redirect به درگاه واقعی
```

**فایل‌های تغییر یافته:**
- ✅ `vite-js/src/sections/wallet/view/wallet-view.jsx`
- ✅ `vite-js/src/sections/payment/view/payment-view.jsx`
- ✅ `vite-js/src/utils/axios.js` (endpoints اضافه شد)

---

## 📊 مقایسه قبل و بعد:

### Wallet Button:

**قبل:**
```jsx
<MenuItem>
  <Typography>موجودی فعلی</Typography>
  <Typography>{balance}</Typography>
</MenuItem>
<Divider />
<MenuItem>
  <Button>شارژ کیف پول</Button>
</MenuItem>
<Divider />
<Typography>تراکنش‌های اخیر</Typography>
<Scrollbar>
  {transactions.map(...)}
</Scrollbar>
```

**بعد:**
```jsx
<MenuItem component="li">
  <ListItemText
    primary="شارژ کیف پول"
    secondary={`موجودی: ${balance} تومان`}
  />
  <SvgIcon>arrow</SvgIcon>
</MenuItem>
<Divider />
<MenuItem>
  <Typography>مشاهده جزئیات و تراکنش‌ها</Typography>
</MenuItem>
```

---

### Payment Page:

**قبل:**
```jsx
<Typography>Let's finish powering you up!</Typography>
<PaymentBillingAddress />
<PaymentMethods>
  - Paypal
  - Credit Card
</PaymentMethods>
```

**بعد:**
```jsx
<Typography>پرداخت و شارژ کیف پول</Typography>
{/* No billing address */}
<PaymentMethods>
  [🟡] زرین‌پال (پیشنهادی)
  [🔵] NowPayments (crypto)
</PaymentMethods>
<PaymentSummary>
  - مبلغ
  - کارمزد
  - مبلغ نهایی
  - نرخ USD (for NowPayments)
</PaymentSummary>
```

---

## 🎨 طراحی UI:

### Wallet Page:

```
┌──────────────────────────────────────────────────┐
│  🎨 کیف پول                                     │
│  مدیریت موجودی و تراکنش‌های مالی                 │
└──────────────────────────────────────────────────┘

┌─────────────┬──────────────────────────────────┐
│ 💎 موجودی   │  💳 شارژ کیف پول                │
│ فعلی        │                                  │
│             │  مبلغ: [_______] تومان           │
│ 1,250,000   │                                  │
│ تومان       │  [50K] [100K] [200K]             │
│             │  [500K] [1M] [2M]                │
│ (Gradient)  │                                  │
│             │  [ادامه و انتخاب درگاه پرداخت]   │
└─────────────┴──────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│  📜 تراکنش‌های اخیر                             │
│                                                  │
│  [↓] شارژ کیف پول    1403/10/05    +500,000    │
│  [↑] پرداخت ویزیت    1403/10/04    -150,000    │
│  [↺] بازپرداخت        1403/10/03    +75,000     │
│                                                  │
│  [مشاهده همه تراکنش‌ها]                         │
└──────────────────────────────────────────────────┘
```

---

### Payment Page:

```
┌──────────────────────────────────────────────────┐
│  💳 پرداخت و شارژ کیف پول          [بازگشت]    │
│  انتخاب درگاه پرداخت و تکمیل تراکنش             │
└──────────────────────────────────────────────────┘

ℹ️ توجه: پس از انتخاب درگاه پرداخت، به صفحه پرداخت منتقل می‌شوید.

┌──────────────────────────┬─────────────────────┐
│  انتخاب درگاه پرداخت     │  📋 خلاصه تراکنش   │
│                          │                     │
│  ┌─────────────────────┐ │  نوع: شارژ کیف پول │
│  │ [logo] زرین‌پال     │ │  درگاه: زرین‌پال   │
│  │ [✓] پیشنهادی        │ │                     │
│  │ پرداخت با کارت ایرانی │ │  مبلغ: 100,000    │
│  │ مبلغ: 100,000 تومان │ │  کارمزد: 1,000     │
│  └─────────────────────┘ │  ─────────────────  │
│                          │  مبلغ نهایی:        │
│  ┌─────────────────────┐ │  101,000 تومان      │
│  │ [logo] NowPayments  │ │                     │
│  │ پرداخت با crypto     │ │  🛡️ پرداخت امن    │
│  │ ≈ $1.92 USD         │ │                     │
│  └─────────────────────┘ │  ☎️ پشتیبانی 24/7  │
│                          │                     │
│  [پرداخت و تکمیل تراکنش] │                     │
└──────────────────────────┴─────────────────────┘
```

---

## 🔧 نصب و راه‌اندازی:

### 1. Database Migration:

```bash
cd minimal-api-dev-v6
npx prisma migrate dev --name add_invoice_payment_and_exchange_rate
npx prisma generate
```

### 2. Install Dependencies (if needed):

```bash
# Backend (minimal-api-dev-v6)
npm install cheerio axios

# Frontend (vite-js)
# No new dependencies needed
```

### 3. Environment Variables:

در فایل `minimal-api-dev-v6/.env.local`:
```env
JWT_SECRET=your-secret-key
DATABASE_URL="file:./prisma/dev.db"
```

در فایل `vite-js/.env.local`:
```env
VITE_API_URL=http://localhost:7272
```

### 4. Start Servers:

```bash
# Backend
cd minimal-api-dev-v6
npm run dev

# Frontend
cd vite-js
npm run dev
```

---

## 🧪 تست:

### 1. Test Wallet Button:
```
1. کلیک روی آیکون wallet در header
2. باید popup با موجودی و دکمه li نمایش داده شود
3. کلیک روی "شارژ کیف پول" → انتقال به /dashboard/wallet
```

### 2. Test Wallet Page:
```
1. باز کردن /dashboard/wallet
2. انتخاب مبلغ (یا quick button)
3. کلیک "ادامه و انتخاب درگاه پرداخت"
4. انتقال به /payment با state
```

### 3. Test Payment Page:
```
1. صفحه payment باز می‌شود
2. نرخ ارز از API دریافت می‌شود
3. انتخاب یکی از درگاه‌ها (zarinpal or nowpayments)
4. کلیک "پرداخت و تکمیل تراکنش"
5. Invoice ایجاد می‌شود
6. Alert با شماره invoice نمایش داده می‌شود
7. انتقال به /dashboard/invoice/:id
```

### 4. Test Exchange Rate API:
```bash
curl http://localhost:7272/api/exchange-rate

# Should return:
{
  "success": true,
  "data": {
    "usd_to_irr": 520000,
    ...
  }
}
```

### 5. Test Invoice API:
```bash
# Create invoice
curl -X POST http://localhost:7272/api/invoice/create \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 100000,
    "type": "wallet_charge",
    "paymentGateway": "zarinpal"
  }'

# Get invoice
curl http://localhost:7272/api/invoice/:id \
  -H "Authorization: Bearer <token>"

# List invoices
curl http://localhost:7272/api/invoice/list \
  -H "Authorization: Bearer <token>"
```

---

## 📝 TODO (آینده):

```
⏳ اتصال به درگاه واقعی زرین‌پال
⏳ اتصال به درگاه واقعی NowPayments
⏳ صفحه invoice detail
⏳ Webhook برای دریافت نتیجه پرداخت
⏳ بروزرسانی موجودی wallet بعد از پرداخت موفق
⏳ Cron job برای auto-refresh exchange rate
⏳ Email/SMS notification برای تراکنش‌ها
```

---

## 🎯 خلاصه فایل‌های تغییر یافته:

### Frontend (vite-js):
```
✅ src/layouts/components/wallet-button.jsx
✅ src/sections/wallet/view/wallet-view.jsx
✅ src/sections/payment/view/payment-view.jsx
✅ src/sections/payment/payment-methods.jsx (rewritten)
✅ src/sections/payment/payment-summary.jsx (rewritten)
✅ src/utils/axios.js (endpoints added)
```

### Backend (minimal-api-dev-v6):
```
✅ prisma/schema.prisma
✅ src/pages/api/exchange-rate/index.ts (NEW)
✅ src/pages/api/invoice/create.ts (NEW)
✅ src/pages/api/invoice/[id].ts (NEW)
✅ src/pages/api/invoice/list.ts (NEW)
```

---

## 🎉 نتیجه نهایی:

```
✅ Wallet UI حرفه‌ای
✅ Payment page فارسی و سفارشی
✅ لوگوی زرین‌پال و NowPayments
✅ نرخ ارز real-time از bonbast
✅ Invoice system کامل
✅ Flow: wallet → payment → invoice
✅ Schema updated with ExchangeRate and Invoice
✅ APIs for invoice CRUD
✅ Endpoints integrated
✅ Ready for production (با اتصال به درگاه واقعی)
```

---

**تاریخ:** 2025-10-30  
**وضعیت:** ✅ Completed  
**نتیجه:** سیستم کیف پول و پرداخت کامل شد! 🎊

---

**مرحله بعدی:** اتصال به درگاه‌های واقعی زرین‌پال و NowPayments




















