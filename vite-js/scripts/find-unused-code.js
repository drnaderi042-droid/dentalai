#!/usr/bin/env node

/**
 * اسکریپت جامع برای پیدا کردن کدهای استفاده نشده در پروژه
 * 
 * این اسکریپت موارد زیر را بررسی می‌کند:
 * 1. Import های استفاده نشده
 * 2. فایل‌های استفاده نشده
 * 3. Dependencies استفاده نشده
 * 4. Export های استفاده نشده
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const srcDir = path.join(projectRoot, 'src');

// رنگ‌های ترمینال
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

// 1. پیدا کردن Import های استفاده نشده با ESLint
function findUnusedImports() {
  log('\n📦 بررسی Import های استفاده نشده...', 'cyan');
  try {
    const result = execSync(
      'npx eslint "src/**/*.{js,jsx,ts,tsx}" --format json --max-warnings 999999',
      { 
        cwd: projectRoot, 
        encoding: 'utf-8', 
        stdio: 'pipe',
        maxBuffer: 10 * 1024 * 1024, // 10MB buffer
      }
    );
    
    const eslintResults = JSON.parse(result);
    const unusedImports = [];
    
    eslintResults.forEach(file => {
      const unusedMessages = file.messages.filter(
        msg => msg.ruleId === 'unused-imports/no-unused-imports'
      );
      
      if (unusedMessages.length > 0) {
        unusedImports.push({
          file: file.filePath.replace(projectRoot + path.sep, ''),
          messages: unusedMessages,
        });
      }
    });
    
    if (unusedImports.length > 0) {
      log(`\n⚠️  ${unusedImports.length} فایل با Import های استفاده نشده پیدا شد:`, 'yellow');
      unusedImports.forEach(({ file, messages }) => {
        log(`\n  ${file}:`, 'blue');
        messages.forEach(msg => {
          log(`    Line ${msg.line}: ${msg.message}`, 'red');
        });
      });
    } else {
      log('✅ هیچ Import استفاده نشده‌ای پیدا نشد!', 'green');
    }
    
    return unusedImports;
  } catch (error) {
    if (error.message.includes('ENOBUFS') || error.message.includes('buffer')) {
      log('⚠️  حجم فایل‌ها زیاد است. استفاده از روش جایگزین...', 'yellow');
      // استفاده از روش ساده‌تر
      try {
        const simpleResult = execSync(
          'npx eslint "src/**/*.{js,jsx,ts,tsx}" --format compact 2>&1 | findstr /C:"unused-imports"',
          { 
            cwd: projectRoot, 
            encoding: 'utf-8', 
            stdio: 'pipe',
            maxBuffer: 10 * 1024 * 1024,
            shell: true,
          }
        );
        if (simpleResult.trim()) {
          log('⚠️  برخی import های استفاده نشده پیدا شد. برای حذف خودکار از دستور زیر استفاده کنید:', 'yellow');
          log('   npm run clean:unused:imports', 'cyan');
        } else {
          log('✅ هیچ Import استفاده نشده‌ای پیدا نشد!', 'green');
        }
      } catch (e) {
        log('⚠️  برای بررسی دقیق‌تر، از دستور زیر استفاده کنید:', 'yellow');
        log('   npm run lint', 'cyan');
      }
    } else {
      log('❌ خطا در اجرای ESLint:', 'red');
      log(error.message, 'red');
    }
    return [];
  }
}

// 2. پیدا کردن فایل‌های استفاده نشده
function findUnusedFiles() {
  log('\n📁 بررسی فایل‌های استفاده نشده...', 'cyan');
  
  const allFiles = [];
  const usedFiles = new Set();
  
  // جمع‌آوری همه فایل‌های .js, .jsx, .ts, .tsx
  function collectFiles(dir) {
    const files = fs.readdirSync(dir, { withFileTypes: true });
    
    files.forEach(file => {
      const fullPath = path.join(dir, file.name);
      
      // نادیده گرفتن node_modules و سایر فولدرهای خاص
      if (file.name.startsWith('.') || 
          file.name === 'node_modules' || 
          file.name === 'dist' ||
          file.name === 'build') {
        return;
      }
      
      if (file.isDirectory()) {
        collectFiles(fullPath);
      } else if (/\.(js|jsx|ts|tsx)$/.test(file.name)) {
        const relativePath = path.relative(projectRoot, fullPath);
        allFiles.push(relativePath);
      }
    });
  }
  
  collectFiles(srcDir);
  
  // پیدا کردن فایل‌های استفاده شده از طریق import ها
  function findImports(filePath) {
    try {
      const content = fs.readFileSync(path.join(projectRoot, filePath), 'utf-8');
      const importRegex = /import\s+.*?\s+from\s+['"](.+?)['"]/g;
      const dynamicImportRegex = /import\s*\(\s*['"](.+?)['"]\s*\)/g;
      const requireRegex = /require\s*\(\s*['"](.+?)['"]\s*\)/g;
      
      const imports = [];
      let match;
      
      // Static imports
      while ((match = importRegex.exec(content)) !== null) {
        imports.push(match[1]);
      }
      
      // Dynamic imports
      while ((match = dynamicImportRegex.exec(content)) !== null) {
        imports.push(match[1]);
      }
      
      // require
      while ((match = requireRegex.exec(content)) !== null) {
        imports.push(match[1]);
      }
      
      imports.forEach(imp => {
        // تبدیل import path به فایل واقعی
        let resolvedPath = imp;
        
        // اگر با src/ شروع می‌شود
        if (imp.startsWith('src/')) {
          resolvedPath = imp;
        }
        // اگر با ./ یا ../ شروع می‌شود
        else if (imp.startsWith('./') || imp.startsWith('../')) {
          const dir = path.dirname(filePath);
          resolvedPath = path.join(dir, imp).replace(/\\/g, '/');
        }
        // اگر alias src است
        else if (!imp.startsWith('.') && !imp.startsWith('/') && !imp.includes('@')) {
          resolvedPath = `src/${imp}`;
        }
        
        // اضافه کردن extension های ممکن
        const possiblePaths = [
          resolvedPath,
          `${resolvedPath}.js`,
          `${resolvedPath}.jsx`,
          `${resolvedPath}.ts`,
          `${resolvedPath}.tsx`,
          `${resolvedPath}/index.js`,
          `${resolvedPath}/index.jsx`,
          `${resolvedPath}/index.ts`,
          `${resolvedPath}/index.tsx`,
        ];
        
        possiblePaths.forEach(p => {
          const fullPath = path.join(projectRoot, p);
          if (fs.existsSync(fullPath)) {
            usedFiles.add(path.relative(projectRoot, fullPath).replace(/\\/g, '/'));
          }
        });
      });
    } catch (error) {
      // نادیده گرفتن خطاها
    }
  }
  
  // بررسی همه فایل‌ها
  allFiles.forEach(file => {
    findImports(file);
    usedFiles.add(file); // فایل خودش هم استفاده شده محسوب می‌شود
  });
  
  // پیدا کردن فایل‌های استفاده نشده
  const unusedFiles = allFiles.filter(file => {
    // نادیده گرفتن فایل‌های خاص
    if (file.includes('_mock') || 
        file.includes('index.js') || 
        file.includes('main.jsx') ||
        file.includes('app.jsx') ||
        file.includes('.test.') ||
        file.includes('.spec.')) {
      return false;
    }
    
    return !usedFiles.has(file);
  });
  
  if (unusedFiles.length > 0) {
    log(`\n⚠️  ${unusedFiles.length} فایل احتمالاً استفاده نشده پیدا شد:`, 'yellow');
    unusedFiles.forEach(file => {
      log(`  ${file}`, 'red');
    });
    log('\n⚠️  توجه: لطفاً قبل از حذف، دستی بررسی کنید!', 'yellow');
  } else {
    log('✅ همه فایل‌ها استفاده شده‌اند!', 'green');
  }
  
  return unusedFiles;
}

// 3. پیدا کردن Dependencies استفاده نشده
function findUnusedDependencies() {
  log('\n📚 بررسی Dependencies استفاده نشده...', 'cyan');
  
  try {
    // استفاده از depcheck
    const result = execSync(
      'npx depcheck --json',
      { cwd: projectRoot, encoding: 'utf-8', stdio: 'pipe' }
    );
    
    const depcheckResult = JSON.parse(result);
    
    if (depcheckResult.dependencies && depcheckResult.dependencies.length > 0) {
      log(`\n⚠️  ${depcheckResult.dependencies.length} Dependency استفاده نشده پیدا شد:`, 'yellow');
      depcheckResult.dependencies.forEach(dep => {
        log(`  ${dep}`, 'red');
      });
    } else {
      log('✅ همه Dependencies استفاده شده‌اند!', 'green');
    }
    
    if (depcheckResult.devDependencies && depcheckResult.devDependencies.length > 0) {
      log(`\n⚠️  ${depcheckResult.devDependencies.length} DevDependency استفاده نشده پیدا شد:`, 'yellow');
      depcheckResult.devDependencies.forEach(dep => {
        log(`  ${dep}`, 'red');
      });
    }
    
    return depcheckResult;
  } catch (error) {
    if (error.message.includes('not found') || error.message.includes('ENOENT')) {
      log('⚠️  depcheck نصب نشده است. در حال نصب...', 'yellow');
      try {
        execSync('npm install --save-dev depcheck', { 
          cwd: projectRoot, 
          stdio: 'inherit',
          maxBuffer: 10 * 1024 * 1024,
        });
        // تلاش مجدد
        const result = execSync(
          'npx depcheck --json',
          { cwd: projectRoot, encoding: 'utf-8', stdio: 'pipe', maxBuffer: 10 * 1024 * 1024 }
        );
        const depcheckResult = JSON.parse(result);
        
        if (depcheckResult.dependencies && depcheckResult.dependencies.length > 0) {
          log(`\n⚠️  ${depcheckResult.dependencies.length} Dependency استفاده نشده پیدا شد:`, 'yellow');
          depcheckResult.dependencies.forEach(dep => {
            log(`  ${dep}`, 'red');
          });
        } else {
          log('✅ همه Dependencies استفاده شده‌اند!', 'green');
        }
        
        return depcheckResult;
      } catch (installError) {
        log('⚠️  برای بررسی dependencies، دستی اجرا کنید: npx depcheck', 'yellow');
      }
    } else {
      log('⚠️  خطا در اجرای depcheck:', 'yellow');
      log(error.message, 'yellow');
    }
    return null;
  }
}

// 4. گزارش خلاصه
function generateSummary(unusedImports, unusedFiles, unusedDeps) {
  log('\n' + '='.repeat(60), 'cyan');
  log('📊 خلاصه گزارش', 'cyan');
  log('='.repeat(60), 'cyan');
  
  log(`\n📦 Import های استفاده نشده: ${unusedImports.length}`, 
    unusedImports.length > 0 ? 'yellow' : 'green');
  log(`📁 فایل‌های استفاده نشده: ${unusedFiles.length}`, 
    unusedFiles.length > 0 ? 'yellow' : 'green');
  
  if (unusedDeps) {
    const totalUnusedDeps = (unusedDeps.dependencies?.length || 0) + 
                           (unusedDeps.devDependencies?.length || 0);
    log(`📚 Dependencies استفاده نشده: ${totalUnusedDeps}`, 
      totalUnusedDeps > 0 ? 'yellow' : 'green');
  }
  
  log('\n' + '='.repeat(60), 'cyan');
}

// اجرای اصلی
async function main() {
  log('🔍 شروع بررسی کدهای استفاده نشده...', 'cyan');
  log('='.repeat(60), 'cyan');
  
  const unusedImports = findUnusedImports();
  const unusedFiles = findUnusedFiles();
  const unusedDeps = findUnusedDependencies();
  
  generateSummary(unusedImports, unusedFiles, unusedDeps);
  
  log('\n✅ بررسی کامل شد!', 'green');
}

main().catch(console.error);

