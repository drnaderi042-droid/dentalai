#!/usr/bin/env node

/**
 * اسکریپت برای تحلیل یک فایل خاص و پیدا کردن کدهای استفاده نشده
 * 
 * استفاده:
 * node scripts/analyze-file.js <path-to-file>
 * 
 * مثال:
 * node scripts/analyze-file.js src/sections/orthodontics/patient/view/patient-orthodontics-view.jsx
 */

import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

// رنگ‌های ترمینال
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m',
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

// دریافت مسیر فایل از آرگومان
const filePath = process.argv[2];

if (!filePath) {
  log('❌ لطفاً مسیر فایل را وارد کنید:', 'red');
  log('   مثال: node scripts/analyze-file.js src/sections/orthodontics/patient/view/patient-orthodontics-view.jsx', 'yellow');
  process.exit(1);
}

const fullPath = path.isAbsolute(filePath) 
  ? filePath 
  : path.join(projectRoot, filePath);

if (!fs.existsSync(fullPath)) {
  log(`❌ فایل پیدا نشد: ${fullPath}`, 'red');
  process.exit(1);
}

const relativePath = path.relative(projectRoot, fullPath).replace(/\\/g, '/');

log('\n' + '='.repeat(80), 'cyan');
log(`📄 تحلیل فایل: ${relativePath}`, 'cyan');
log('='.repeat(80), 'cyan');

// 1. بررسی با ESLint
function analyzeWithESLint() {
  log('\n📦 بررسی با ESLint...', 'cyan');
  
  try {
    const result = execSync(
      `npx eslint "${relativePath}" --format compact --max-warnings 999999 2>&1`,
      {
        cwd: projectRoot,
        encoding: 'utf-8',
        stdio: 'pipe',
        maxBuffer: 10 * 1024 * 1024,
        timeout: 30000,
      }
    );

    if (!result || result.trim().length === 0) {
      log('✅ هیچ مشکلی پیدا نشد!', 'green');
      return;
    }

    const lines = result.split('\n').filter(line => line.trim());
    
    if (lines.length === 0) {
      log('✅ هیچ مشکلی پیدا نشد!', 'green');
      return;
    }

    const unusedImports = [];
    const unusedVars = [];
    const otherIssues = [];

    lines.forEach(line => {
      const match = line.match(/^(.+?):(\d+):(\d+)\s+(warning|error)\s+(.+?)\s+\((.+?)\)$/);
      if (match) {
        const [, , lineNum, col, level, message, ruleId] = match;
        const msg = {
          line: parseInt(lineNum),
          column: parseInt(col),
          level,
          message,
          ruleId,
        };

        if (ruleId === 'unused-imports/no-unused-imports') {
          unusedImports.push(msg);
        } else if (ruleId === 'no-unused-vars' || ruleId === 'unused-imports/no-unused-vars') {
          unusedVars.push(msg);
        } else {
          otherIssues.push(msg);
        }
      }
    });

    // نمایش نتایج
    if (unusedImports.length > 0) {
      log(`\n⚠️  ${unusedImports.length} Import استفاده نشده پیدا شد:`, 'yellow');
      unusedImports.slice(0, 30).forEach(msg => {
        log(`   خط ${msg.line}:${msg.column} - ${msg.message}`, 'red');
      });
      if (unusedImports.length > 30) {
        log(`   ... و ${unusedImports.length - 30} مورد دیگر`, 'yellow');
      }
    }

    if (unusedVars.length > 0) {
      log(`\n⚠️  ${unusedVars.length} متغیر/تابع استفاده نشده پیدا شد:`, 'yellow');
      unusedVars.slice(0, 30).forEach(msg => {
        log(`   خط ${msg.line}:${msg.column} - ${msg.message}`, 'red');
      });
      if (unusedVars.length > 30) {
        log(`   ... و ${unusedVars.length - 30} مورد دیگر`, 'yellow');
      }
    }

    if (otherIssues.length > 0 && otherIssues.length < 50) {
      log(`\nℹ️  ${otherIssues.length} مشکل دیگر:`, 'cyan');
      otherIssues.slice(0, 10).forEach(msg => {
        log(`   خط ${msg.line}: ${msg.message} (${msg.ruleId})`, 'yellow');
      });
    }

    if (unusedImports.length === 0 && unusedVars.length === 0) {
      log('\n✅ هیچ کد استفاده نشده‌ای پیدا نشد!', 'green');
    }

  } catch (error) {
    log('⚠️  استفاده از روش جایگزین...', 'yellow');
    
    try {
      const simpleResult = execSync(
        `npx eslint "${relativePath}" --format compact 2>&1 | findstr /C:"unused-imports"`,
        {
          cwd: projectRoot,
          encoding: 'utf-8',
          stdio: 'pipe',
          maxBuffer: 10 * 1024 * 1024,
          shell: true,
          timeout: 30000,
        }
      );
      
      if (simpleResult.trim()) {
        log('⚠️  برخی import های استفاده نشده پیدا شد:', 'yellow');
        log('   برای حذف خودکار: npm run clean:unused:imports', 'cyan');
      } else {
        log('✅ هیچ Import استفاده نشده‌ای پیدا نشد!', 'green');
      }
    } catch (e) {
      log('⚠️  برای بررسی دقیق‌تر: npm run lint', 'yellow');
    }
  }
}

// 2. تحلیل دستی فایل
function analyzeManually() {
  log('\n🔍 تحلیل دستی فایل...', 'cyan');
  
  try {
    const content = fs.readFileSync(fullPath, 'utf-8');
    const lines = content.split('\n');
    
    const imports = [];
    const importRegex = /^import\s+(.+?)\s+from\s+['"](.+?)['"]/;
    
    lines.forEach((line, index) => {
      const match = line.match(importRegex);
      if (match) {
        imports.push({
          line: index + 1,
          content: match[1],
          from: match[2],
        });
      }
    });

    log(`\n📊 آمار فایل:`, 'cyan');
    log(`   تعداد خطوط: ${lines.length}`, 'blue');
    log(`   تعداد Import ها: ${imports.length}`, 'blue');

  } catch (error) {
    log('❌ خطا در خواندن فایل:', 'red');
  }
}

// 3. پیشنهادات
function showSuggestions() {
  log('\n' + '='.repeat(80), 'cyan');
  log('💡 پیشنهادات:', 'cyan');
  log('='.repeat(80), 'cyan');
  log('\n1. برای حذف خودکار import های استفاده نشده:', 'yellow');
  log(`   npm run clean:unused:imports`, 'cyan');
  log('\n2. برای بررسی دقیق‌تر:', 'yellow');
  log(`   npm run lint -- "${relativePath}"`, 'cyan');
}

// اجرای اصلی
async function main() {
  analyzeWithESLint();
  analyzeManually();
  showSuggestions();
  
  log('\n' + '='.repeat(80), 'cyan');
  log('✅ تحلیل کامل شد!', 'green');
  log('='.repeat(80), 'cyan');
}

main().catch(console.error);
