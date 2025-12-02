/**
 * Image Compression Utility
 * Compresses images before sending to AI APIs
 * Handles size limits for different providers
 */

const MAX_FILE_SIZE_BYTES = {
  claude: 5 * 1024 * 1024, // 5MB for Claude/Anthropic
  'gpt-4o': 20 * 1024 * 1024, // 20MB for GPT-4o
  default: 5 * 1024 * 1024, // 5MB default
};

const TARGET_SIZE_BYTES = 4 * 1024 * 1024; // Target 4MB to be safe

/**
 * Convert data URL to blob
 */
function dataURLtoBlob(dataURL) {
  const arr = dataURL.split(',');
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n > 0) {
    n -= 1;
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new Blob([u8arr], { type: mime });
}

/**
 * Load image from URL
 */
async function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

/**
 * Compress image to target size
 * @param {string} imageUrl - Image URL to compress
 * @param {number} maxSizeBytes - Maximum size in bytes
 * @param {string} outputFormat - Output format (jpeg, webp, png)
 * @returns {Promise<{dataUrl: string, blob: Blob, size: number, compressed: boolean}>}
 */
export async function compressImage(imageUrl, maxSizeBytes = TARGET_SIZE_BYTES, outputFormat = 'jpeg') {
  try {
    console.log('📦 Starting image compression for:', imageUrl);
    
    // Load the image
    const img = await loadImage(imageUrl);
    
    console.log(`📐 Original dimensions: ${img.width}x${img.height}`);
    
    // Create canvas
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Calculate initial dimensions (keep aspect ratio)
    const { width: imgWidth, height: imgHeight } = img;
    let width = imgWidth;
    let height = imgHeight;
    
    // If image is too large, resize it first
    const MAX_DIMENSION = 2048; // Max width or height
    if (width > MAX_DIMENSION || height > MAX_DIMENSION) {
      if (width > height) {
        height = (height / width) * MAX_DIMENSION;
        width = MAX_DIMENSION;
      } else {
        width = (width / height) * MAX_DIMENSION;
        height = MAX_DIMENSION;
      }
      console.log(`📏 Resizing to: ${Math.round(width)}x${Math.round(height)}`);
    }
    
    canvas.width = width;
    canvas.height = height;
    
    // Draw image on canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);
    
    // Try different quality levels until we get under the target size
    let quality = 0.9;
    let dataUrl;
    let blob;
    let attempts = 0;
    const maxAttempts = 10;
    
    do {
      attempts += 1;
      dataUrl = canvas.toDataURL(`image/${outputFormat}`, quality);
      blob = dataURLtoBlob(dataUrl);
      
      console.log(`🎯 Attempt ${attempts}: Quality ${(quality * 100).toFixed(0)}%, Size: ${(blob.size / 1024 / 1024).toFixed(2)}MB`);
      
      if (blob.size <= maxSizeBytes) {
        break;
      }
      
      // Reduce quality for next attempt
      quality -= 0.1;
      
      // If quality is too low and still too large, resize image
      if (quality < 0.3 && blob.size > maxSizeBytes) {
        width *= 0.8;
        height *= 0.8;
        canvas.width = width;
        canvas.height = height;
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, width, height);
        ctx.drawImage(img, 0, 0, width, height);
        quality = 0.9; // Reset quality
        console.log(`📉 Reducing dimensions to: ${Math.round(width)}x${Math.round(height)}`);
      }
      
    } while (blob.size > maxSizeBytes && attempts < maxAttempts);
    
    const finalSize = blob.size;
    const compressed = finalSize < (await fetch(imageUrl).then(r => r.blob()).then(b => b.size));
    
    console.log(`✅ Compression complete: ${(finalSize / 1024 / 1024).toFixed(2)}MB (${compressed ? 'compressed' : 'same size'})`);
    
    return {
      dataUrl,
      blob,
      size: finalSize,
      compressed,
      width: Math.round(width),
      height: Math.round(height),
      quality: Math.round(quality * 100),
    };
    
  } catch (error) {
    console.error('❌ Image compression failed:', error);
    throw error;
  }
}

/**
 * Compress multiple images
 * @param {string[]} imageUrls - Array of image URLs
 * @param {number} maxSizeBytes - Maximum size per image
 * @returns {Promise<Array>}
 */
export async function compressMultipleImages(imageUrls, maxSizeBytes = TARGET_SIZE_BYTES) {
  console.log(`📦 Compressing ${imageUrls.length} images...`);
  
  const results = await Promise.all(
    imageUrls.map(url => compressImage(url, maxSizeBytes))
  );
  
  const totalSize = results.reduce((sum, r) => sum + r.size, 0);
  console.log(`✅ All images compressed. Total size: ${(totalSize / 1024 / 1024).toFixed(2)}MB`);
  
  return results;
}

/**
 * Get appropriate compression settings for AI model
 * @param {string} modelId - AI model identifier
 * @returns {Object} Compression settings
 */
export function getCompressionSettingsForModel(modelId) {
  const settings = {
    'claude-vision': {
      maxSize: MAX_FILE_SIZE_BYTES.claude,
      format: 'jpeg',
      targetSize: 4 * 1024 * 1024, // 4MB to be safe
    },
    'gpt-4o-vision': {
      maxSize: MAX_FILE_SIZE_BYTES['gpt-4o'],
      format: 'jpeg',
      targetSize: 10 * 1024 * 1024, // 10MB
    },
    'cephx-v1': {
      maxSize: 10 * 1024 * 1024,
      format: 'jpeg',
      targetSize: 8 * 1024 * 1024,
    },
    'cephx-v2': {
      maxSize: 10 * 1024 * 1024,
      format: 'jpeg',
      targetSize: 8 * 1024 * 1024,
    },
    'deepceph': {
      maxSize: 15 * 1024 * 1024,
      format: 'png',
      targetSize: 12 * 1024 * 1024,
    },
  };
  
  return settings[modelId] || {
    maxSize: MAX_FILE_SIZE_BYTES.default,
    format: 'jpeg',
    targetSize: TARGET_SIZE_BYTES,
  };
}

/**
 * Check if image needs compression
 * @param {string} imageUrl - Image URL
 * @param {number} maxSize - Maximum allowed size
 * @returns {Promise<{needsCompression: boolean, currentSize: number}>}
 */
export async function checkImageSize(imageUrl, maxSize = TARGET_SIZE_BYTES) {
  try {
    const response = await fetch(imageUrl);
    const blob = await response.blob();
    const currentSize = blob.size;
    
    return {
      needsCompression: currentSize > maxSize,
      currentSize,
      currentSizeMB: (currentSize / 1024 / 1024).toFixed(2),
      maxSizeMB: (maxSize / 1024 / 1024).toFixed(2),
    };
  } catch (error) {
    console.error('Error checking image size:', error);
    return {
      needsCompression: true, // Assume it needs compression if we can't check
      currentSize: 0,
    };
  }
}

export default {
  compressImage,
  compressMultipleImages,
  getCompressionSettingsForModel,
  checkImageSize,
};

