/**
 * PDF Processor
 * پردازش PDF‌های واقعی برای RAG
 */

// این فایل نیاز به نصب pdf-parse دارد:
// npm install pdf-parse @types/pdf-parse

import path from 'path';

export interface PDFDocument {
  title: string;
  authors: string;
  year: number;
  pages: PDFPage[];
  metadata: {
    totalPages: number;
    filePath: string;
  };
}

export interface PDFPage {
  pageNumber: number;
  content: string;
  chapter?: string;
  section?: string;
}

/**
 * خواندن PDF و استخراج محتوا
 * 
 * توجه: این تابع نیاز به pdf-parse دارد
 * npm install pdf-parse
 */
export async function parsePDF(filePath: string): Promise<PDFDocument> {
  try {
    // Dynamic import برای pdf-parse (اگر نصب نشده باشد خطا ندهد)
    // eslint-disable-next-line import/no-unresolved
    const pdf = await import('pdf-parse');
    const fs = await import('fs');
    
    const dataBuffer = fs.readFileSync(filePath);
    const pdfData = await pdf.default(dataBuffer);
    
    // استخراج metadata
    const metadata = extractMetadata(filePath, pdfData);
    
    // تقسیم به صفحات
    const pages: PDFPage[] = [];
    const contentPerPage = pdfData.text.split(/\f/); // تقسیم بر اساس page break
    
    contentPerPage.forEach((content, index) => {
      if (content.trim()) {
        pages.push({
          pageNumber: index + 1,
          content: content.trim(),
          chapter: extractChapter(content),
          section: extractSection(content),
        });
      }
    });
    
    return {
      title: metadata.title,
      authors: metadata.authors,
      year: metadata.year,
      pages,
      metadata: {
        totalPages: pdfData.numpages,
        filePath,
      },
    };
  } catch (error) {
    console.error('Error parsing PDF:', error);
    throw new Error(`Failed to parse PDF: ${filePath}. Make sure pdf-parse is installed: npm install pdf-parse`);
  }
}

/**
 * استخراج metadata از PDF
 */
function extractMetadata(filePath: string, pdfData: any) {
  const fileName = path.basename(filePath, '.pdf');
  
  // استخراج از info (اگر موجود باشد)
  const info = pdfData.info || {};
  
  return {
    title: info.Title || fileName,
    authors: info.Author || 'Unknown',
    year: extractYear(info.CreationDate) || extractYear(fileName) || new Date().getFullYear(),
  };
}

/**
 * استخراج سال از متن
 */
function extractYear(text: string): number | null {
  if (!text) return null;
  const match = String(text).match(/\b(19|20)\d{2}\b/);
  return match ? parseInt(match[0], 10) : null;
}

/**
 * استخراج فصل از محتوا
 */
function extractChapter(content: string): string | undefined {
  // جستجوی الگوهای فصل
  const patterns = [
    /Chapter\s+(\d+)/i,
    /فصل\s+(\d+)/i,
    /Chapter\s+([IVX]+)/i,
    /CHAPTER\s+(\d+)/i,
  ];
  
  for (const pattern of patterns) {
    const match = content.match(pattern);
    if (match) {
      return `Chapter ${match[1]}`;
    }
  }
  
  return undefined;
}

/**
 * استخراج بخش از محتوا
 */
function extractSection(content: string): string | undefined {
  // جستجوی الگوهای بخش
  const patterns = [
    /Section\s+(\d+\.\d+)/i,
    /بخش\s+(\d+)/i,
    /SECTION\s+(\d+\.\d+)/i,
  ];
  
  for (const pattern of patterns) {
    const match = content.match(pattern);
    if (match) {
      return match[1];
    }
  }
  
  return undefined;
}

/**
 * پردازش همه PDF‌ها در یک پوشه
 */
export async function processAllPDFs(directory: string): Promise<PDFDocument[]> {
  try {
    const fs = await import('fs');
    const path = await import('path');
    
    const files = fs.readdirSync(directory);
    const pdfFiles = files.filter(f => f.endsWith('.pdf'));
    
    const documents: PDFDocument[] = [];
    
    for (const file of pdfFiles) {
      const filePath = path.join(directory, file);
      try {
        const doc = await parsePDF(filePath);
        documents.push(doc);
        console.log(`✅ Processed: ${file} (${doc.pages.length} pages)`);
      } catch (error) {
        console.error(`❌ Error processing ${file}:`, error);
      }
    }
    
    return documents;
  } catch (error) {
    console.error('Error reading directory:', error);
    return [];
  }
}

