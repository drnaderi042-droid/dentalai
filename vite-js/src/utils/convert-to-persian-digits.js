/**
 * Convert English digits to Persian digits
 * @param {string|number} input - The input string or number
 * @returns {string} - String with Persian digits
 */
export function toPersianDigits(input) {
  if (input === null || input === undefined) return '';
  
  const persianDigits = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'];
  const englishDigits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
  
  return String(input).replace(/[0-9]/g, (digit) => persianDigits[englishDigits.indexOf(digit)]);
}

/**
 * Convert Persian digits to English digits
 * @param {string} input - The input string with Persian digits
 * @returns {string} - String with English digits
 */
export function toEnglishDigits(input) {
  if (input === null || input === undefined) return '';
  
  const persianDigits = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'];
  const englishDigits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
  
  return String(input).replace(/[۰-۹]/g, (digit) => englishDigits[persianDigits.indexOf(digit)]);
}


