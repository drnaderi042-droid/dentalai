#!/usr/bin/env node

/**
 * اسکریپت پیشرفته برای پیدا کردن کدهای استفاده نشده در یک فایل
 * 
 * این اسکریپت موارد زیر را بررسی می‌کند:
 * - توابع و متغیرهای تعریف شده اما استفاده نشده
 * - کامپوننت‌های JSX که render نمی‌شوند
 * - State ها و hooks استفاده نشده
 * - Handler ها و callback های استفاده نشده
 * 
 * استفاده:
 * node scripts/find-unused-code-in-file.js <path-to-file>
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

// دریافت مسیر فایل
const filePath = process.argv[2];

if (!filePath) {
  log('❌ لطفاً مسیر فایل را وارد کنید:', 'red');
  log('   مثال: node scripts/find-unused-code-in-file.js src/sections/orthodontics/patient/view/patient-orthodontics-view.jsx', 'yellow');
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
log(`📄 تحلیل پیشرفته فایل: ${relativePath}`, 'cyan');
log('='.repeat(80), 'cyan');

// خواندن فایل
const content = fs.readFileSync(fullPath, 'utf-8');
const lines = content.split('\n');

// 1. پیدا کردن تمام تعریف‌ها
function findDefinitions() {
  log('\n🔍 در حال پیدا کردن تعریف‌ها...', 'cyan');
  
  const definitions = {
    functions: [],
    variables: [],
    hooks: [],
    components: [],
    handlers: [],
    states: [],
    constants: [],
  };

  lines.forEach((line, index) => {
    const lineNum = index + 1;
    const trimmed = line.trim();

    // توابع
    const functionMatch = trimmed.match(/^(export\s+)?(const|function|let|var)\s+(\w+)\s*[=:]\s*(\([^)]*\)\s*=>|function|async\s+function)/);
    if (functionMatch) {
      const name = functionMatch[3];
      const isHandler = name.startsWith('handle') || name.startsWith('on') || name.endsWith('Handler');
      
      if (isHandler) {
        definitions.handlers.push({ name, line: lineNum, type: 'handler' });
      } else {
        definitions.functions.push({ name, line: lineNum, type: 'function' });
      }
    }

    // useState, useEffect, etc.
    const hookMatch = trimmed.match(/(const|let|var)\s+\[?(\w+)\]?\s*=\s*(use\w+)/);
    if (hookMatch) {
      const name = hookMatch[2];
      const hookType = hookMatch[3];
      
      if (hookType === 'useState') {
        definitions.states.push({ name, line: lineNum, type: 'state' });
      } else {
        definitions.hooks.push({ name, line: lineNum, type: hookType });
      }
    }

    // کامپوننت‌های React (با حرف بزرگ شروع می‌شوند)
    const componentMatch = trimmed.match(/^(export\s+)?(const|function)\s+([A-Z][a-zA-Z0-9]+)\s*[=:]/);
    if (componentMatch) {
      definitions.components.push({ name: componentMatch[3], line: lineNum, type: 'component' });
    }

    // متغیرهای const/let/var
    const varMatch = trimmed.match(/^(export\s+)?(const|let|var)\s+(\w+)\s*[=:]/);
    if (varMatch && !functionMatch && !hookMatch) {
      const name = varMatch[3];
      const isConstant = name === name.toUpperCase() || trimmed.includes('const');
      
      if (isConstant && name === name.toUpperCase()) {
        definitions.constants.push({ name, line: lineNum, type: 'constant' });
      } else {
        definitions.variables.push({ name, line: lineNum, type: 'variable' });
      }
    }
  });

  return definitions;
}

// 2. پیدا کردن استفاده‌ها
function findUsages(definitions) {
  log('🔍 در حال پیدا کردن استفاده‌ها...', 'cyan');
  
  const usages = new Set();
  
  // جمع‌آوری همه نام‌ها
  const allNames = [
    ...definitions.functions.map(d => d.name),
    ...definitions.variables.map(d => d.name),
    ...definitions.hooks.map(d => d.name),
    ...definitions.components.map(d => d.name),
    ...definitions.handlers.map(d => d.name),
    ...definitions.states.map(d => d.name),
    ...definitions.constants.map(d => d.name),
  ];

  // بررسی استفاده در فایل
  allNames.forEach(name => {
    // استفاده در JSX
    const jsxPattern = new RegExp(`<${name}[\\s>]`, 'g');
    if (jsxPattern.test(content)) {
      usages.add(name);
    }

    // استفاده در کد JavaScript
    const codePattern = new RegExp(`\\b${name}\\b`, 'g');
    const matches = content.match(codePattern) || [];
    
    // اگر بیش از یک بار استفاده شده (یک بار در تعریف)، یعنی استفاده شده
    if (matches.length > 1) {
      usages.add(name);
    }

    // استفاده در string template یا JSX attribute
    if (content.includes(`${name}=`) || content.includes(`${name}:`) || content.includes(`{${name}}`)) {
      usages.add(name);
    }
  });

  return usages;
}

// 3. پیدا کردن کدهای استفاده نشده
function findUnusedCode(definitions, usages) {
  log('\n📊 تحلیل نتایج...', 'cyan');
  
  const unused = {
    functions: [],
    variables: [],
    hooks: [],
    components: [],
    handlers: [],
    states: [],
    constants: [],
  };

  // بررسی هر دسته
  Object.keys(definitions).forEach(category => {
    definitions[category].forEach(def => {
      if (!usages.has(def.name)) {
        // استثنا: اگر export شده، ممکن است در فایل دیگری استفاده شود
        const lineContent = lines[def.line - 1];
        const isExported = lineContent.includes('export');
        
        if (!isExported) {
          unused[category].push(def);
        }
      }
    });
  });

  return unused;
}

// 4. پیدا کردن JSX blocks استفاده نشده
function findUnusedJSX() {
  log('\n🔍 بررسی JSX blocks استفاده نشده...', 'cyan');
  
  const unusedJSX = [];
  const jsxPattern = /<(\w+)[^>]*>/g;
  const components = new Set();
  
  // پیدا کردن کامپوننت‌های تعریف شده
  lines.forEach((line, index) => {
    const componentMatch = line.match(/^(export\s+)?(const|function)\s+([A-Z][a-zA-Z0-9]+)\s*[=:]/);
    if (componentMatch) {
      components.add(componentMatch[3]);
    }
  });

  // پیدا کردن استفاده از کامپوننت‌ها در JSX
  const jsxUsages = new Set();
  let match;
  while ((match = jsxPattern.exec(content)) !== null) {
    const componentName = match[1];
    if (componentName[0] === componentName[0].toUpperCase()) {
      jsxUsages.add(componentName);
    }
  }

  // کامپوننت‌های تعریف شده اما استفاده نشده
  components.forEach(comp => {
    if (!jsxUsages.has(comp) && comp !== 'PatientOrthodonticsView') {
      const lineNum = lines.findIndex((line, idx) => {
        const match = line.match(new RegExp(`(const|function)\\s+${comp}\\s*[=:]`));
        return match !== null;
      }) + 1;
      
      if (lineNum > 0) {
        unusedJSX.push({ name: comp, line: lineNum });
      }
    }
  });

  return unusedJSX;
}

// 5. پیدا کردن import های استفاده نشده (با ESLint)
function findUnusedImportsWithESLint() {
  log('\n🔍 بررسی import های استفاده نشده...', 'cyan');
  
  try {
    const result = execSync(
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

    if (result.trim()) {
      const lines = result.trim().split('\n');
      return lines.map(line => {
        const match = line.match(/:(\d+):(\d+).*?'(\w+)'/);
        if (match) {
          return {
            line: parseInt(match[1]),
            column: parseInt(match[2]),
            name: match[3],
          };
        }
        return null;
      }).filter(Boolean);
    }
  } catch (error) {
    // ignore
  }

  return [];
}

// 6. نمایش نتایج
function displayResults(unused, unusedJSX, unusedImports) {
  log('\n' + '='.repeat(80), 'cyan');
  log('📊 نتایج تحلیل', 'cyan');
  log('='.repeat(80), 'cyan');

  let totalUnused = 0;

  // Import های استفاده نشده
  if (unusedImports.length > 0) {
    log(`\n⚠️  ${unusedImports.length} Import استفاده نشده:`, 'yellow');
    unusedImports.forEach(imp => {
      log(`   خط ${imp.line}:${imp.column} - '${imp.name}'`, 'red');
    });
    totalUnused += unusedImports.length;
  }

  // Handlers استفاده نشده
  if (unused.handlers.length > 0) {
    log(`\n⚠️  ${unused.handlers.length} Handler/Callback استفاده نشده:`, 'yellow');
    unused.handlers.slice(0, 20).forEach(item => {
      log(`   خط ${item.line}: ${item.name}`, 'red');
    });
    if (unused.handlers.length > 20) {
      log(`   ... و ${unused.handlers.length - 20} مورد دیگر`, 'yellow');
    }
    totalUnused += unused.handlers.length;
  }

  // توابع استفاده نشده
  if (unused.functions.length > 0) {
    log(`\n⚠️  ${unused.functions.length} تابع استفاده نشده:`, 'yellow');
    unused.functions.slice(0, 20).forEach(item => {
      log(`   خط ${item.line}: ${item.name}`, 'red');
    });
    if (unused.functions.length > 20) {
      log(`   ... و ${unused.functions.length - 20} مورد دیگر`, 'yellow');
    }
    totalUnused += unused.functions.length;
  }

  // State های استفاده نشده
  if (unused.states.length > 0) {
    log(`\n⚠️  ${unused.states.length} State استفاده نشده:`, 'yellow');
    unused.states.slice(0, 20).forEach(item => {
      log(`   خط ${item.line}: ${item.name}`, 'red');
    });
    if (unused.states.length > 20) {
      log(`   ... و ${unused.states.length - 20} مورد دیگر`, 'yellow');
    }
    totalUnused += unused.states.length;
  }

  // متغیرهای استفاده نشده
  if (unused.variables.length > 0) {
    log(`\n⚠️  ${unused.variables.length} متغیر استفاده نشده:`, 'yellow');
    unused.variables.slice(0, 20).forEach(item => {
      log(`   خط ${item.line}: ${item.name}`, 'red');
    });
    if (unused.variables.length > 20) {
      log(`   ... و ${unused.variables.length - 20} مورد دیگر`, 'yellow');
    }
    totalUnused += unused.variables.length;
  }

  // کامپوننت‌های JSX استفاده نشده
  if (unusedJSX.length > 0) {
    log(`\n⚠️  ${unusedJSX.length} کامپوننت JSX استفاده نشده:`, 'yellow');
    unusedJSX.slice(0, 20).forEach(item => {
      log(`   خط ${item.line}: ${item.name}`, 'red');
    });
    if (unusedJSX.length > 20) {
      log(`   ... و ${unusedJSX.length - 20} مورد دیگر`, 'yellow');
    }
    totalUnused += unusedJSX.length;
  }

  // خلاصه
  log('\n' + '='.repeat(80), 'cyan');
  log(`📈 خلاصه:`, 'cyan');
  log(`   کل موارد استفاده نشده: ${totalUnused}`, totalUnused > 0 ? 'yellow' : 'green');
  log('='.repeat(80), 'cyan');

  if (totalUnused === 0) {
    log('\n✅ هیچ کد استفاده نشده‌ای پیدا نشد!', 'green');
  } else {
    log('\n💡 برای حذف خودکار import های استفاده نشده:', 'yellow');
    log('   npm run clean:unused:imports', 'cyan');
    log('\n⚠️  توجه: قبل از حذف کدها، حتماً بررسی دستی انجام دهید!', 'yellow');
  }
}

// اجرای اصلی
async function main() {
  const definitions = findDefinitions();
  const usages = findUsages(definitions);
  const unused = findUnusedCode(definitions, usages);
  const unusedJSX = findUnusedJSX();
  const unusedImports = findUnusedImportsWithESLint();

  displayResults(unused, unusedJSX, unusedImports);
}

main().catch(console.error);










