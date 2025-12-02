#!/usr/bin/env node

/**
 * اسکریپت سریع برای حذف import های استفاده نشده
 * این اسکریپت فقط import های استفاده نشده را پیدا و حذف می‌کند
 */

import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import path from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

console.log('🧹 شروع حذف import های استفاده نشده...\n');

try {
  // اجرای ESLint با auto-fix
  execSync(
    'npx eslint "src/**/*.{js,jsx,ts,tsx}" --fix',
    {
      cwd: projectRoot,
      stdio: 'inherit',
    }
  );
  
  console.log('\n✅ حذف import های استفاده نشده با موفقیت انجام شد!');
} catch (error) {
  console.error('\n❌ خطا در حذف import های استفاده نشده:');
  console.error(error.message);
  process.exit(1);
}










