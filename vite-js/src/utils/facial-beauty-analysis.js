/**
 * آنالیز زیبایی صورت با استفاده از لندمارک‌های تشخیص داده شده
 * Facial Beauty Analysis using detected landmarks
 */

/**
 * محاسبه فاصله اقلیدسی بین دو نقطه
 */
export function calculateDistance(point1, point2) {
  const dx = point2.x - point1.x;
  const dy = point2.y - point1.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * dlib 68 facial landmarks indices
 * Reference: http://dlib.net/face_landmark_detection.py.html
 * 
 * 0-16: Jaw line (چانه و فک)
 * 17-21: Right eyebrow (ابرو راست)
 * 22-26: Left eyebrow (ابرو چپ)
 * 27-35: Nose (بینی)
 * 36-41: Right eye (چشم راست)
 * 42-47: Left eye (چشم چپ)
 * 48-67: Mouth (دهان)
 */
const DLIB_LANDMARK_INDICES = {
  // Jaw / Chin
  chin_center: 8,
  chin: 8,
  chin_left: 0,
  chin_right: 16,
  jaw_left: 0,
  jaw_right: 16,
  
  // Eyebrows
  right_eyebrow_outer: 17,
  right_eyebrow_inner: 21,
  left_eyebrow_inner: 22,
  left_eyebrow_outer: 26,
  glabella: 27, // Between eyebrows, top of nose
  
  // Nose
  nose_tip: 30,
  nose_bridge_top: 27, // Top of nose bridge (between eyebrows)
  nose_bridge_bottom: 30,
  nose_left: 31,
  nose_right: 35,
  subnasale: 33, // Bottom of nose, above upper lip
  
  // Eyes
  right_eye_outer: 36,
  right_eye_inner: 39,
  right_eye_top: 37,
  right_eye_bottom: 41,
  right_pupil: 68, // Calculated: average of right eye landmarks
  left_eye_inner: 42,
  left_eye_outer: 45,
  left_eye_top: 43,
  left_eye_bottom: 47,
  left_pupil: 69, // Calculated: average of left eye landmarks
  
  // Mouth
  mouth_left: 48,
  mouth_right: 54,
  mouth_top: 51, // Upper lip top
  mouth_bottom: 57, // Lower lip bottom
  mouth_center_top: 51,
  mouth_center_bottom: 57,
  upper_lip_top: 51,
  upper_lip_bottom: 62,
  lower_lip_top: 63,
  lower_lip_bottom: 57,
  
  // Additional
  pogonion: 8, // Chin tip
  menton: 8, // Chin bottom
};

/**
 * MediaPipe Face Mesh landmark indices for key facial features
 * Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
 */
const MEDIAPIPE_LANDMARK_INDICES = {
  // Left eye
  left_eye_inner: 133,
  left_eye: 33,
  left_eye_outer: 7,
  left_eye_top: 159,
  left_eye_bottom: 145,
  left_eye_corner: 33,
  left_eye_1: 7,
  left_eye_2: 133,
  
  // Right eye
  right_eye_inner: 362,
  right_eye: 263,
  right_eye_outer: 249,
  right_eye_top: 386,
  right_eye_bottom: 374,
  right_eye_corner: 263,
  right_eye_1: 249,
  right_eye_2: 362,
  
  // Nose
  nose_tip: 1,
  nose_bridge_top: 6,
  nose_bridge_1: 6,
  nose_bridge_4: 19,
  nose_left: 131,
  nose_right: 360,
  nose_bottom: 2,
  nose: 1,
  nose_tip_1: 1,
  nose_tip_2: 131,
  nose_tip_4: 360,
  
  // Mouth
  mouth_left: 61,
  mouth_right: 291,
  mouth_top: 13,
  mouth_bottom: 14,
  mouth_center_top: 13,
  mouth_center_bottom: 14,
  mouth_outer_1: 61,
  mouth_outer_3: 13,
  mouth_outer_5: 291,
  mouth_outer_9: 14,
  mouth_left_corner: 61,
  mouth_right_corner: 291,
  
  // Chin
  chin: 175,
  chin_center: 175,
  chin_8: 175,
  chin_left: 172,
  chin_right: 397,
  chin_1: 172,
  chin_17: 397,
  pogonion: 175, // تقریبی
  menton: 175, // تقریبی
  
  // Face outline
  forehead: 10,
  face_left: 172,
  face_right: 397,
  
  // Additional points
  glabella: 9,
  left_cheek: 116,
  right_cheek: 345,
  
  // Subnasale (تقریبی - نزدیک به nose_bottom)
  subnasale: 2,
  
  // Jaw (برای MediaPipe از face outline استفاده می‌کنیم)
  jaw_left: 172, // تقریبی - از face_left
  jaw_right: 397, // تقریبی - از face_right
  
  // Upper and lower lip
  upper_lip_top: 13,
  upper_lip_bottom: 14, // تقریبی
  lower_lip_top: 14, // تقریبی
  lower_lip_bottom: 14,
};

/**
 * پیدا کردن لندمارک با نام‌های مختلف (سازگار با همه مدل‌ها)
 * پشتیبانی از MediaPipe Face Mesh با استفاده از index mapping
 */
function findLandmark(landmarks, names) {
  if (!Array.isArray(names)) names = [names];
  
  // First, try to find by name (for dlib, face-alignment, RetinaFace)
  for (const name of names) {
    const found = landmarks.find(l => {
      if (!l.name) return false;
      const lName = l.name.toLowerCase();
      const searchName = name.toLowerCase();
      return lName === searchName || 
             lName.includes(searchName) || 
             searchName.includes(lName) ||
             lName.replace(/_/g, '') === searchName.replace(/_/g, '');
    });
    if (found) return found;
  }
  
  // If not found by name, try index mapping for dlib or MediaPipe
  // Check if this looks like indexed landmarks (landmark_0, landmark_1, etc.)
  const isIndexed = landmarks.some(l => l.name && l.name.startsWith('landmark_'));
  
  if (isIndexed && landmarks.length > 0) {
    // Determine if it's dlib (68 landmarks) or MediaPipe (468 landmarks)
    const landmarkCount = landmarks.length;
    const isDlib = landmarkCount === 68;
    const isMediaPipe = landmarkCount >= 400;
    
    for (const name of names) {
      const searchName = name.toLowerCase();
      let targetIndex = null;
      
      // Try dlib first if it's dlib
      if (isDlib) {
        targetIndex = DLIB_LANDMARK_INDICES[searchName];
        
        // Handle special cases for dlib (pupil calculation)
        if (isDlib && (searchName === 'left_pupil' || searchName === 'right_pupil')) {
          // Calculate pupil as center of eye landmarks
          if (searchName === 'left_pupil') {
            // Left eye: landmarks 42-47
            const leftEyeLandmarks = landmarks.filter(l => {
              const idx = l.index !== undefined ? l.index : (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
              return idx >= 42 && idx <= 47;
            });
            if (leftEyeLandmarks.length > 0) {
              const avgX = leftEyeLandmarks.reduce((sum, l) => sum + l.x, 0) / leftEyeLandmarks.length;
              const avgY = leftEyeLandmarks.reduce((sum, l) => sum + l.y, 0) / leftEyeLandmarks.length;
              return { x: avgX, y: avgY, index: 69, name: 'left_pupil' };
            }
          } else if (searchName === 'right_pupil') {
            // Right eye: landmarks 36-41
            const rightEyeLandmarks = landmarks.filter(l => {
              const idx = l.index !== undefined ? l.index : (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
              return idx >= 36 && idx <= 41;
            });
            if (rightEyeLandmarks.length > 0) {
              const avgX = rightEyeLandmarks.reduce((sum, l) => sum + l.x, 0) / rightEyeLandmarks.length;
              const avgY = rightEyeLandmarks.reduce((sum, l) => sum + l.y, 0) / rightEyeLandmarks.length;
              return { x: avgX, y: avgY, index: 68, name: 'right_pupil' };
            }
          }
        }
        
        // Handle special cases for MediaPipe (pupil calculation)
        if (isMediaPipe && (searchName === 'left_pupil' || searchName === 'right_pupil')) {
          if (searchName === 'left_pupil') {
            // MediaPipe left eye landmarks: 33, 7, 159, 145, 133
            const leftEyeLandmarks = landmarks.filter(l => {
              const idx = l.index !== undefined ? l.index : (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
              return idx === 33 || idx === 7 || idx === 133 || idx === 159 || idx === 145;
            });
            if (leftEyeLandmarks.length > 0) {
              const avgX = leftEyeLandmarks.reduce((sum, l) => sum + l.x, 0) / leftEyeLandmarks.length;
              const avgY = leftEyeLandmarks.reduce((sum, l) => sum + l.y, 0) / leftEyeLandmarks.length;
              return { x: avgX, y: avgY, index: 469, name: 'left_pupil' };
            }
          } else if (searchName === 'right_pupil') {
            // MediaPipe right eye landmarks: 263, 249, 386, 374, 362
            const rightEyeLandmarks = landmarks.filter(l => {
              const idx = l.index !== undefined ? l.index : (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
              return idx === 263 || idx === 249 || idx === 362 || idx === 386 || idx === 374;
            });
            if (rightEyeLandmarks.length > 0) {
              const avgX = rightEyeLandmarks.reduce((sum, l) => sum + l.x, 0) / rightEyeLandmarks.length;
              const avgY = rightEyeLandmarks.reduce((sum, l) => sum + l.y, 0) / rightEyeLandmarks.length;
              return { x: avgX, y: avgY, index: 468, name: 'right_pupil' };
            }
          }
        }
      }
      
      // Try MediaPipe if not dlib or if dlib didn't have it
      if (targetIndex === null || targetIndex === undefined) {
        if (isMediaPipe) {
          targetIndex = MEDIAPIPE_LANDMARK_INDICES[searchName];
        }
      }
      
      if (targetIndex !== null && targetIndex !== undefined) {
        // Find by index - check both index property and name
        const found = landmarks.find(l => {
          // First, try to use index property if available
          if (l.index !== undefined && l.index === targetIndex) {
            return true;
          }
          
          // Otherwise, try to extract index from name like "landmark_33"
          if (l.name) {
            const indexMatch = l.name.match(/landmark_(\d+)/);
            if (indexMatch) {
              const landmarkIndex = parseInt(indexMatch[1], 10);
              return landmarkIndex === targetIndex;
            }
          }
          
          return false;
        });
        
        if (found) {
          console.log(`[Facial Beauty] Found landmark "${name}" at index ${targetIndex}:`, found);
          return found;
        } 
          console.warn(`[Facial Beauty] Could not find landmark "${name}" at index ${targetIndex}`);
        
      }
    }
  }
  
  return null;
}

/**
 * محاسبه تقارن صورت (Facial Symmetry)
 */
export function calculateSymmetry(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  // پیدا کردن نقاط کلیدی - سازگار با همه مدل‌ها
  const leftEye = findLandmark(landmarks, ['left_eye', 'left_eye_inner', 'left_eye_corner', 'left_eye_1', 'left_eye_outer']);
  const rightEye = findLandmark(landmarks, ['right_eye', 'right_eye_inner', 'right_eye_corner', 'right_eye_1', 'right_eye_outer']);
  const noseTip = findLandmark(landmarks, ['nose_tip', 'nose', 'nose_tip_1']);
  const leftMouth = findLandmark(landmarks, ['mouth_left', 'mouth_left_corner']);
  const rightMouth = findLandmark(landmarks, ['mouth_right', 'mouth_right_corner']);
  const chin = findLandmark(landmarks, ['chin_center', 'chin', 'chin_8']);

  if (!leftEye || !rightEye || !noseTip) {
    return null;
  }

  // محاسبه خط مرکزی صورت (از بینی تا چانه)
  const centerLine = {
    x: noseTip.x,
    y: noseTip.y,
  };

  // فاصله از خط مرکزی
  const leftEyeDistance = Math.abs(leftEye.x - centerLine.x);
  const rightEyeDistance = Math.abs(rightEye.x - centerLine.x);
  const leftMouthDistance = leftMouth ? Math.abs(leftMouth.x - centerLine.x) : 0;
  const rightMouthDistance = rightMouth ? Math.abs(rightMouth.x - centerLine.x) : 0;

  // محاسبه تقارن (0-100)
  const eyeSymmetry = 100 - (Math.abs(leftEyeDistance - rightEyeDistance) / Math.max(leftEyeDistance, rightEyeDistance) * 100);
  const mouthSymmetry = (leftMouth && rightMouth) 
    ? 100 - (Math.abs(leftMouthDistance - rightMouthDistance) / Math.max(leftMouthDistance, rightMouthDistance) * 100)
    : 100;

  const overallSymmetry = (eyeSymmetry + mouthSymmetry) / 2;

  return {
    overall: Math.max(0, Math.min(100, overallSymmetry)),
    eyes: Math.max(0, Math.min(100, eyeSymmetry)),
    mouth: Math.max(0, Math.min(100, mouthSymmetry)),
  };
}

/**
 * محاسبه انحراف Midfacial Line (خط میانی صورت)
 * Midfacial Line: Glabella → Subnasale → Pogonion
 */
export function calculateMidfacialLineDeviation(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const glabella = findLandmark(landmarks, ['glabella', 'nose_bridge_top', 'forehead']);
  const subnasale = findLandmark(landmarks, ['subnasale', 'nose_bottom']);
  const pogonion = findLandmark(landmarks, ['pogonion', 'chin_center', 'chin']);

  if (!glabella || !subnasale || !pogonion) {
    return null;
  }

  // محاسبه خط میانی صورت (میانگین x coordinates)
  const midlineX = (glabella.x + subnasale.x + pogonion.x) / 3;
  
  // محاسبه انحراف هر نقطه از خط میانی
  const glabellaDeviation = Math.abs(glabella.x - midlineX);
  const subnasaleDeviation = Math.abs(subnasale.x - midlineX);
  const pogonionDeviation = Math.abs(pogonion.x - midlineX);
  
  // میانگین انحراف
  const avgDeviation = (glabellaDeviation + subnasaleDeviation + pogonionDeviation) / 3;
  
  // محاسبه درصد انحراف (نسبت به عرض صورت)
  const faceWidth = Math.abs(
    (findLandmark(landmarks, ['face_right', 'chin_right'])?.x || pogonion.x * 2) -
    (findLandmark(landmarks, ['face_left', 'chin_left'])?.x || 0)
  ) || 1;
  
  const deviationPercent = (avgDeviation / faceWidth) * 100;
  
  // در حالت طبیعی انحراف ≤ 2-3 میلی‌متر قابل قبول است
  // برای یک تصویر با عرض صورت حدود 200-300px، این حدود 1-1.5% است
  const idealDeviation = 1.5; // درصد
  const score = Math.max(0, 100 - ((deviationPercent - idealDeviation) / idealDeviation * 100));

  return {
    deviation: avgDeviation.toFixed(2),
    deviationPercent: deviationPercent.toFixed(2),
    score: score.toFixed(1),
    grade: deviationPercent <= 2 ? 'Ideal' : deviationPercent <= 5 ? 'Acceptable' : 'Needs Review',
  };
}

/**
 * محاسبه زاویه Interpupillary Line (خط بین مردمک‌ها)
 * باید افقی و موازی با زمین باشد
 */
export function calculateInterpupillaryLineAngle(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const leftPupil = findLandmark(landmarks, ['left_pupil', 'left_eye']);
  const rightPupil = findLandmark(landmarks, ['right_pupil', 'right_eye']);

  if (!leftPupil || !rightPupil) {
    return null;
  }

  // محاسبه زاویه خط بین مردمک‌ها نسبت به افق
  const deltaX = rightPupil.x - leftPupil.x;
  const deltaY = rightPupil.y - leftPupil.y;
  const angle = Math.atan2(Math.abs(deltaY), Math.abs(deltaX)) * 180 / Math.PI;
  
  // زاویه ایده‌آل: 0 درجه (کاملاً افقی)
  const idealAngle = 0;
  const angleDeviation = Math.abs(angle - idealAngle);
  
  // نمره: هر درجه انحراف = -5 نمره
  const score = Math.max(0, 100 - (angleDeviation * 5));

  return {
    angle: angle.toFixed(2),
    angleDeviation: angleDeviation.toFixed(2),
    score: score.toFixed(1),
    grade: angleDeviation <= 1 ? 'Ideal' : angleDeviation <= 3 ? 'Acceptable' : 'Needs Review',
  };
}

/**
 * محاسبه زاویه Commissural Line (خط بین گوشه‌های دهان)
 * باید موازی با Interpupillary Line باشد
 */
export function calculateCommissuralLineAngle(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const mouthLeft = findLandmark(landmarks, ['mouth_left', 'mouth_outer_1']);
  const mouthRight = findLandmark(landmarks, ['mouth_right', 'mouth_outer_5']);
  const leftPupil = findLandmark(landmarks, ['left_pupil', 'left_eye']);
  const rightPupil = findLandmark(landmarks, ['right_pupil', 'right_eye']);

  if (!mouthLeft || !mouthRight) {
    return null;
  }

  // محاسبه زاویه خط دهان
  const deltaX = mouthRight.x - mouthLeft.x;
  const deltaY = mouthRight.y - mouthLeft.y;
  const mouthAngle = Math.atan2(Math.abs(deltaY), Math.abs(deltaX)) * 180 / Math.PI;
  
  // اگر مردمک‌ها موجود باشند، مقایسه با زاویه خط بین مردمک‌ها
  let pupillaryAngle = 0;
  if (leftPupil && rightPupil) {
    const pupillaryDeltaX = rightPupil.x - leftPupil.x;
    const pupillaryDeltaY = rightPupil.y - leftPupil.y;
    pupillaryAngle = Math.atan2(Math.abs(pupillaryDeltaY), Math.abs(pupillaryDeltaX)) * 180 / Math.PI;
  }
  
  // تفاوت زاویه بین خط دهان و خط بین مردمک‌ها
  const angleDifference = Math.abs(mouthAngle - pupillaryAngle);
  
  // نمره: هر درجه تفاوت = -10 نمره
  const score = Math.max(0, 100 - (angleDifference * 10));

  return {
    angle: mouthAngle.toFixed(2),
    angleDifference: angleDifference.toFixed(2),
    pupillaryAngle: leftPupil && rightPupil ? pupillaryAngle.toFixed(2) : null,
    score: score.toFixed(1),
    grade: angleDifference <= 1 ? 'Ideal' : angleDifference <= 3 ? 'Acceptable' : 'Needs Review',
  };
}

/**
 * محاسبه انحراف Dental Midline از Facial Midline
 */
export function calculateDentalMidlineDeviation(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const glabella = findLandmark(landmarks, ['glabella', 'nose_bridge_top']);
  const subnasale = findLandmark(landmarks, ['subnasale', 'nose_bottom']);
  const pogonion = findLandmark(landmarks, ['pogonion', 'chin_center', 'chin']);
  const noseTip = findLandmark(landmarks, ['nose_tip']);
  const mouthTop = findLandmark(landmarks, ['mouth_top', 'mouth_center_top', 'upper_lip_top']);

  if (!glabella || !subnasale || !pogonion || !noseTip) {
    return null;
  }

  // محاسبه Facial Midline (میانگین x coordinates)
  const facialMidlineX = (glabella.x + subnasale.x + pogonion.x) / 3;
  
  // Dental Midline (تقریبی از نوک بینی)
  const dentalMidlineX = noseTip.x;
  
  // انحراف
  const deviation = Math.abs(dentalMidlineX - facialMidlineX);
  
  // محاسبه درصد انحراف
  const faceWidth = Math.abs(
    (findLandmark(landmarks, ['face_right', 'chin_right'])?.x || pogonion.x * 2) -
    (findLandmark(landmarks, ['face_left', 'chin_left'])?.x || 0)
  ) || 1;
  
  const deviationPercent = (deviation / faceWidth) * 100;
  
  // در حالت طبیعی انحراف ≤ 2 میلی‌متر قابل قبول است
  const idealDeviation = 1; // درصد
  const score = Math.max(0, 100 - ((deviationPercent - idealDeviation) / idealDeviation * 100));

  return {
    deviation: deviation.toFixed(2),
    deviationPercent: deviationPercent.toFixed(2),
    score: score.toFixed(1),
    grade: deviationPercent <= 2 ? 'Ideal' : deviationPercent <= 5 ? 'Acceptable' : 'Needs Review',
  };
}

/**
 * محاسبه نسبت‌های عمودی صورت (Vertical Proportions)
 * بررسی نسبت Glabella–Subnasale و Subnasale–Menton
 * چون لندمارکی در پیشانی (Trichion) نداریم، فقط دو بخش میانی و پایینی را بررسی می‌کنیم
 */
export function calculateVerticalProportions(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const glabella = findLandmark(landmarks, ['glabella', 'nose_bridge_top', 'forehead']);
  const subnasale = findLandmark(landmarks, ['subnasale', 'nose_bottom']);
  const menton = findLandmark(landmarks, ['menton', 'chin', 'chin_center', 'pogonion']);

  if (!glabella || !subnasale || !menton) {
    return null;
  }

  const results = {};

  // بررسی ترتیب landmarks در محور y
  // در سیستم مختصات تصویر: y از بالا به پایین افزایش می‌یابد
  // Glabella (بالا) < Subnasale (میانه) < Menton (پایین)
  
  // اعتبارسنجی: مطمئن شویم که landmarks در ترتیب صحیح هستند
  if (glabella.y >= subnasale.y || subnasale.y >= menton.y) {
    console.warn('[Vertical Proportions] Invalid landmark order:', {
      glabella: { y: glabella.y, name: glabella.name },
      subnasale: { y: subnasale.y, name: subnasale.name },
      menton: { y: menton.y, name: menton.name },
    });
    // اگر ترتیب اشتباه است، از Math.abs استفاده می‌کنیم
  }
  
  // بخش میانی صورت: Glabella → Subnasale
  // در سیستم مختصات تصویر: subnasale.y > glabella.y (subnasale پایین‌تر است)
  const middleSection = subnasale.y - glabella.y;
  results.middleThirdHeight = Math.abs(middleSection).toFixed(1);

  // بخش پایینی صورت: Subnasale → Menton
  // در سیستم مختصات تصویر: menton.y > subnasale.y (menton پایین‌تر است)
  const lowerSection = menton.y - subnasale.y;
  results.lowerThirdHeight = Math.abs(lowerSection).toFixed(1);
  
  // استفاده از مقادیر مثبت برای محاسبات بعدی
  const middleSectionAbs = Math.abs(middleSection);
  const lowerSectionAbs = Math.abs(lowerSection);

  // نسبت بین دو بخش: میانی به پایینی
  // نسبت ایده‌آل: 1:1 (هر دو بخش باید تقریباً برابر باشند)
  // در حالت ایده‌آل، Glabella–Subnasale ≈ Subnasale–Menton
  if (lowerSectionAbs > 0) {
    const ratio = middleSectionAbs / lowerSectionAbs;
    results.middleToLowerRatio = ratio.toFixed(2);
    
    // محاسبه درصد هر بخش نسبت به کل (Glabella تا Menton)
    const totalHeight = middleSectionAbs + lowerSectionAbs;
    if (totalHeight > 0) {
      results.middleThirdRatio = (middleSectionAbs / totalHeight * 100).toFixed(1);
      results.lowerThirdRatio = (lowerSectionAbs / totalHeight * 100).toFixed(1);
      
      // Debug log
      console.log('[Vertical Proportions] Calculations:', {
        glabella: { y: glabella.y, name: glabella.name },
        subnasale: { y: subnasale.y, name: subnasale.name },
        menton: { y: menton.y, name: menton.name },
        middleSection: middleSectionAbs,
        lowerSection: lowerSectionAbs,
        totalHeight,
        ratio,
        middleThirdRatio: results.middleThirdRatio,
        lowerThirdRatio: results.lowerThirdRatio,
      });
      
      // نسبت ایده‌آل: 50% برای هر بخش
      const idealRatio = 50.0;
      const middleDeviation = Math.abs(parseFloat(results.middleThirdRatio) - idealRatio);
      const lowerDeviation = Math.abs(parseFloat(results.lowerThirdRatio) - idealRatio);
      
      results.middleThirdDeviation = middleDeviation.toFixed(1);
      results.lowerThirdDeviation = lowerDeviation.toFixed(1);
      
      // نمره بر اساس انحراف از نسبت 1:1
      // اگر نسبت 1:1 باشد، نمره 100
      // هر چه اختلاف بیشتر باشد، نمره کمتر می‌شود
      const ratioDeviation = Math.abs(ratio - 1.0); // انحراف از نسبت 1:1
      const avgDeviation = (middleDeviation + lowerDeviation) / 2;
      
      // نمره: کاهش امتیاز بر اساس انحراف
      // اگر انحراف کمتر از 5% باشد، نمره خوب است
      // اگر انحراف بیشتر از 15% باشد، نمره پایین است
      let score = 100;
      if (ratioDeviation > 0.05) {
        // کاهش نمره بر اساس انحراف
        score = Math.max(0, 100 - (ratioDeviation * 100));
      }
      if (avgDeviation > 5) {
        // کاهش نمره بر اساس انحراف از 50%
        score = Math.max(0, score - (avgDeviation * 2));
      }
      
      results.score = Math.max(0, Math.min(100, score)).toFixed(1);
      
      // Grade (درجه)
      const scoreNum = parseFloat(results.score);
      if (scoreNum >= 90) {
        results.grade = 'Ideal';
      } else if (scoreNum >= 75) {
        results.grade = 'Good';
      } else if (scoreNum >= 60) {
        results.grade = 'Fair';
      } else {
        results.grade = 'Needs Improvement';
      }
    }
    
    // بررسی اینکه آیا اختلاف خیلی زیاد است
    if (ratio < 0.7 || ratio > 1.3) {
      // اختلاف بیشتر از 30% - نیاز به توجه
      results.warning = true;
      results.warningMessage = ratio < 0.7 
        ? 'بخش میانی صورت خیلی کوتاه‌تر از بخش پایینی است'
        : 'بخش پایینی صورت خیلی کوتاه‌تر از بخش میانی است';
    }
  }
  
  // بخش اول (پیشانی) - چون لندمارک نداریم، محاسبه نمی‌کنیم
  results.upperThirdRatio = null;
  results.upperThirdHeight = null;
  results.upperThirdDeviation = null;

  // تقسیم بخش پایینی صورت: Sn–Stm (Upper lip) و Stm–Me (Lower lip + Chin)
  const upperLipTop = findLandmark(landmarks, ['upper_lip_top', 'mouth_top', 'mouth_center_top']);
  const upperLipBottom = findLandmark(landmarks, ['upper_lip_bottom', 'mouth_center_bottom']);
  const lowerLipBottom = findLandmark(landmarks, ['lower_lip_bottom', 'mouth_bottom']);

  if (subnasale && upperLipBottom && menton) {
    // Upper lip: Sn–Stm (Subnasale تا Upper lip bottom)
    // در سیستم مختصات: upperLipBottom.y > subnasale.y (upperLipBottom پایین‌تر است)
    const upperLipHeight = Math.abs(upperLipBottom.y - subnasale.y);
    
    // Lower lip + Chin: Stm–Me (Upper lip bottom تا Menton)
    // در سیستم مختصات: menton.y > upperLipBottom.y (menton پایین‌تر است)
    const lowerLipChinHeight = Math.abs(menton.y - upperLipBottom.y);
    
    const lowerFaceHeight = upperLipHeight + lowerLipChinHeight;
    
    console.log('[Vertical Proportions] Lower face calculations:', {
      subnasale: { y: subnasale.y, name: subnasale.name },
      upperLipBottom: upperLipBottom ? { y: upperLipBottom.y, name: upperLipBottom.name } : null,
      menton: { y: menton.y, name: menton.name },
      upperLipHeight,
      lowerLipChinHeight,
      lowerFaceHeight,
    });
    
    if (lowerFaceHeight > 0) {
      results.upperLipRatio = (upperLipHeight / lowerFaceHeight * 100).toFixed(1);
      results.lowerLipChinRatio = (lowerLipChinHeight / lowerFaceHeight * 100).toFixed(1);
      
      // نسبت ایده‌آل: 1:2 (Upper lip ≈ 33%, Lower lip + Chin ≈ 67%)
      const idealUpperLipRatio = 33.3;
      const idealLowerLipChinRatio = 66.7;
      
      results.upperLipDeviation = Math.abs(parseFloat(results.upperLipRatio) - idealUpperLipRatio).toFixed(1);
      results.lowerLipChinDeviation = Math.abs(parseFloat(results.lowerLipChinRatio) - idealLowerLipChinRatio).toFixed(1);
      
      // نمره
      const avgLowerDeviation = (
        parseFloat(results.upperLipDeviation) +
        parseFloat(results.lowerLipChinDeviation)
      ) / 2;
      results.lowerFaceScore = Math.max(0, 100 - (avgLowerDeviation * 2)).toFixed(1);
    }
  }

  return Object.keys(results).length > 0 ? results : null;
}

/**
 * محاسبه نسبت قدامی-تحتانی (Facial Index)
 * Facial index = Face height (N–Gn) / Face width (Zy–Zy)
 */
export function calculateFacialIndex(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  // Nasion (N) - تقریبی از glabella
  const nasion = findLandmark(landmarks, ['glabella', 'nose_bridge_top', 'nasion']);
  // Gnathion (Gn) - تقریبی از menton
  const gnathion = findLandmark(landmarks, ['menton', 'chin', 'chin_center', 'pogonion']);
  // Zygomatic width (Zy–Zy) - عرض صورت در سطح zygomatic
  // در dlib، می‌توانیم از عرض صورت در سطح چشم‌ها استفاده کنیم
  const leftZygomatic = findLandmark(landmarks, ['face_left', 'chin_left']);
  const rightZygomatic = findLandmark(landmarks, ['face_right', 'chin_right']);

  if (!nasion || !gnathion) {
    return null;
  }

  // Face height: N–Gn
  const faceHeight = Math.abs(gnathion.y - nasion.y);
  
  // Face width: Zy–Zy
  let faceWidth = 0;
  if (leftZygomatic && rightZygomatic) {
    faceWidth = Math.abs(rightZygomatic.x - leftZygomatic.x);
  } else {
    // Fallback: استفاده از عرض صورت در سطح چشم‌ها
    const leftEye = findLandmark(landmarks, ['left_eye_outer', 'left_eye']);
    const rightEye = findLandmark(landmarks, ['right_eye_outer', 'right_eye']);
    if (leftEye && rightEye) {
      const eyeDistance = Math.abs(rightEye.x - leftEye.x);
      // عرض صورت معمولاً حدود 1.5 تا 2 برابر فاصله بین چشم‌ها است
      faceWidth = eyeDistance * 1.75;
    }
  }

  if (faceHeight > 0 && faceWidth > 0) {
    const facialIndex = faceHeight / faceWidth;
    
    // نرمال: حدود 1.3 ± 0.1
    const idealIndex = 1.3;
    const deviation = Math.abs(facialIndex - idealIndex);
    const score = Math.max(0, 100 - (deviation / 0.1 * 10));

    return {
      facialIndex: facialIndex.toFixed(2),
      faceHeight: faceHeight.toFixed(1),
      faceWidth: faceWidth.toFixed(1),
      deviation: deviation.toFixed(2),
      score: score.toFixed(1),
      grade: facialIndex >= 1.2 && facialIndex <= 1.4 ? 'Normal' : 
             facialIndex > 1.4 ? 'Long Face' : 'Short Face',
    };
  }

  return null;
}

/**
 * محاسبه نسبت‌های عرضی (Transverse / Horizontal Proportions)
 */
export function calculateTransverseProportions(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const results = {};

  // عرض بینی (Alar width) ≈ عرض بین داکوس گوشه داخلی چشم‌ها (intercanthal distance)
  const leftEyeInner = findLandmark(landmarks, ['left_eye_inner', 'left_eye']);
  const rightEyeInner = findLandmark(landmarks, ['right_eye_inner', 'right_eye']);
  const noseLeft = findLandmark(landmarks, ['nose_left']);
  const noseRight = findLandmark(landmarks, ['nose_right']);

  if (leftEyeInner && rightEyeInner) {
    const intercanthalDistance = Math.abs(rightEyeInner.x - leftEyeInner.x);
    results.intercanthalDistance = intercanthalDistance.toFixed(1);

    if (noseLeft && noseRight) {
      const alarWidth = Math.abs(noseRight.x - noseLeft.x);
      results.alarWidth = alarWidth.toFixed(1);
      
      // نسبت Alar width به Intercanthal distance (ایده‌آل: ≈ 1:1)
      const ratio = alarWidth / intercanthalDistance;
      results.alarToIntercanthalRatio = ratio.toFixed(2);
      const idealRatio = 1.0;
      const deviation = Math.abs(ratio - idealRatio);
      results.alarToIntercanthalScore = Math.max(0, 100 - (deviation * 50)).toFixed(1);
    }
  }

  // عرض دهان (mouth width) ≈ فاصله بین پاپیلاهای مردمک در حالت نگاه مستقیم
  const mouthLeft = findLandmark(landmarks, ['mouth_left', 'mouth_outer_1']);
  const mouthRight = findLandmark(landmarks, ['mouth_right', 'mouth_outer_5']);
  const leftPupil = findLandmark(landmarks, ['left_pupil', 'left_eye']);
  const rightPupil = findLandmark(landmarks, ['right_pupil', 'right_eye']);

  if (mouthLeft && mouthRight) {
    const mouthWidth = Math.abs(mouthRight.x - mouthLeft.x);
    results.mouthWidth = mouthWidth.toFixed(1);

    if (leftPupil && rightPupil) {
      const pupillaryDistance = Math.abs(rightPupil.x - leftPupil.x);
      results.pupillaryDistance = pupillaryDistance.toFixed(1);
      
      // نسبت Mouth width به Pupillary distance (ایده‌آل: ≈ 1:1)
      const ratio = mouthWidth / pupillaryDistance;
      results.mouthToPupillaryRatio = ratio.toFixed(2);
      const idealRatio = 1.0;
      const deviation = Math.abs(ratio - idealRatio);
      results.mouthToPupillaryScore = Math.max(0, 100 - (deviation * 50)).toFixed(1);
    }
  }

  // نسبت عرض بینی به عرض دهان ≈ 1 : 1.3
  const noseLeft2 = findLandmark(landmarks, ['nose_left']);
  const noseRight2 = findLandmark(landmarks, ['nose_right']);
  const mouthLeft2 = findLandmark(landmarks, ['mouth_left', 'mouth_outer_1']);
  const mouthRight2 = findLandmark(landmarks, ['mouth_right', 'mouth_outer_5']);

  if (noseLeft2 && noseRight2 && mouthLeft2 && mouthRight2) {
    const alarWidth = Math.abs(noseRight2.x - noseLeft2.x);
    const mouthWidth = Math.abs(mouthRight2.x - mouthLeft2.x);
    
    if (alarWidth > 0) {
      const ratio = mouthWidth / alarWidth;
      results.noseToMouthWidthRatio = ratio.toFixed(2);
      const idealRatio = 1.3;
      const deviation = Math.abs(ratio - idealRatio);
      results.noseToMouthWidthScore = Math.max(0, 100 - (deviation / idealRatio * 100)).toFixed(1);
    }
  }

  return Object.keys(results).length > 0 ? results : null;
}

/**
 * محاسبه نسبت‌های لب
 */
export function calculateLipProportions(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const results = {};

  // طول لب بالا (Sn–Stm) ≈ 20–22 میلی‌متر
  const subnasale = findLandmark(landmarks, ['subnasale', 'nose_bottom']);
  const upperLipBottom = findLandmark(landmarks, ['upper_lip_bottom', 'mouth_center_bottom', 'mouth_bottom']);

  if (subnasale && upperLipBottom) {
    const upperLipLength = Math.abs(upperLipBottom.y - subnasale.y);
    results.upperLipLength = upperLipLength.toFixed(1);
    // تبدیل به میلی‌متر (فرض: 1px ≈ 0.26mm برای یک تصویر معمولی)
    const upperLipLengthMM = upperLipLength * 0.26;
    results.upperLipLengthMM = upperLipLengthMM.toFixed(1);
    
    // نمره (ایده‌آل: 20-22mm)
    const idealLength = 21; // mm
    const deviation = Math.abs(upperLipLengthMM - idealLength);
    results.upperLipLengthScore = Math.max(0, 100 - (deviation * 5)).toFixed(1);
  }

  // طول لب پایین (Stm–Me) ≈ 40–44 میلی‌متر
  const menton = findLandmark(landmarks, ['menton', 'chin', 'chin_center', 'pogonion']);
  const lowerLipTop = findLandmark(landmarks, ['lower_lip_top', 'mouth_center_bottom']);

  if (upperLipBottom && menton) {
    const lowerLipChinLength = Math.abs(menton.y - upperLipBottom.y);
    results.lowerLipChinLength = lowerLipChinLength.toFixed(1);
    const lowerLipChinLengthMM = lowerLipChinLength * 0.26;
    results.lowerLipChinLengthMM = lowerLipChinLengthMM.toFixed(1);
    
    // نمره (ایده‌آل: 40-44mm)
    const idealLength = 42; // mm
    const deviation = Math.abs(lowerLipChinLengthMM - idealLength);
    results.lowerLipChinLengthScore = Math.max(0, 100 - (deviation * 2.5)).toFixed(1);
  }

  // نسبت لب بالا به پایین ≈ 1 : 2
  if (results.upperLipLength && results.lowerLipChinLength) {
    const ratio = parseFloat(results.lowerLipChinLength) / parseFloat(results.upperLipLength);
    results.upperToLowerLipRatio = ratio.toFixed(2);
    const idealRatio = 2.0;
    const deviation = Math.abs(ratio - idealRatio);
    results.upperToLowerLipScore = Math.max(0, 100 - (deviation * 25)).toFixed(1);
  }

  return Object.keys(results).length > 0 ? results : null;
}

/**
 * محاسبه نسبت طلایی (Golden Ratio) - نسبت 1:1.618
 */
export function calculateGoldenRatio(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;
  
  const noseTip = findLandmark(landmarks, ['nose_tip', 'nose', 'nose_tip_1']);
  const chin = findLandmark(landmarks, ['chin_center', 'chin', 'chin_8']);
  const leftEye = findLandmark(landmarks, ['left_eye', 'left_eye_inner', 'left_eye_corner']);
  const rightEye = findLandmark(landmarks, ['right_eye', 'right_eye_inner', 'right_eye_corner']);
  const mouth = findLandmark(landmarks, ['mouth_center_top', 'mouth_top']);

  if (!noseTip || !chin || !leftEye || !rightEye) {
    return null;
  }

  const results = {};

  // نسبت ارتفاع صورت (از Glabella تا Chin)
  // چون لندمارک بالای پیشانی نداریم، از Glabella به عنوان نقطه شروع استفاده می‌کنیم
  const glabella = findLandmark(landmarks, ['glabella', 'nose_bridge_top']);
  
  if (glabella && noseTip && chin) {
    // ارتفاع از Glabella تا Chin
    const glabellaToChinHeight = Math.abs(chin.y - glabella.y);
    // بخش میانی: Glabella → Nose Tip
    const glabellaToNoseHeight = Math.abs(noseTip.y - glabella.y);
    // بخش پایینی: Nose Tip → Chin
    const noseToChinHeight = Math.abs(chin.y - noseTip.y);
    
    if (glabellaToNoseHeight > 0 && noseToChinHeight > 0) {
      // نسبت Glabella→Nose به Nose→Chin (باید تقریباً 1:1.618 باشد)
      const ratio = glabellaToNoseHeight / noseToChinHeight;
      const goldenRatio = 1.618;
      const deviation = Math.abs(ratio - goldenRatio) / goldenRatio;
      results.verticalRatio = {
        ratio: ratio.toFixed(2),
        deviation: (deviation * 100).toFixed(1),
        score: Math.max(0, 100 - (deviation * 100)),
      };
    }
  }

  // نسبت عرض صورت به فاصله چشم‌ها
  const eyeCenterX = (leftEye.x + rightEye.x) / 2;
  const eyeDistance = Math.abs(rightEye.x - leftEye.x);
  
  // پیدا کردن نقاط کناری صورت (اگر وجود دارند)
  const leftFace = findLandmark(landmarks, ['chin_left', 'face_left', 'chin_1']);
  const rightFace = findLandmark(landmarks, ['chin_right', 'face_right', 'chin_17']);
  
  if (leftFace && rightFace) {
    const faceWidth = Math.abs(rightFace.x - leftFace.x);
    if (eyeDistance > 0) {
      const ratio = faceWidth / eyeDistance;
      const idealRatio = 1.618; // نسبت طلایی
      const deviation = Math.abs(ratio - idealRatio) / idealRatio;
      results.horizontalRatio = {
        ratio: ratio.toFixed(2),
        deviation: (deviation * 100).toFixed(1),
        score: Math.max(0, 100 - (deviation * 100)),
      };
    }
  }

  // نسبت فاصله چشم‌ها به عرض بینی
  const noseLeft = findLandmark(landmarks, ['nose_left', 'nose_tip_2']);
  const noseRight = findLandmark(landmarks, ['nose_right', 'nose_tip_4']);
  if (noseLeft && noseRight && eyeDistance > 0) {
    const noseWidth = Math.abs(noseRight.x - noseLeft.x);
    const ratio = eyeDistance / noseWidth;
    const idealRatio = 1.618;
    const deviation = Math.abs(ratio - idealRatio) / idealRatio;
    results.eyeToNoseRatio = {
      ratio: ratio.toFixed(2),
      deviation: (deviation * 100).toFixed(1),
      score: Math.max(0, 100 - (deviation * 100)),
    };
  }

  return results;
}

/**
 * آنالیز چشم‌ها
 */
export function analyzeEyes(landmarks) {
  const leftEye = findLandmark(landmarks, ['left_eye', 'left_eye_inner', 'left_eye_corner']);
  const rightEye = findLandmark(landmarks, ['right_eye', 'right_eye_inner', 'right_eye_corner']);
  const leftEyeInner = findLandmark(landmarks, ['left_eye_inner', 'left_eye_2', 'left_eye']);
  const rightEyeInner = findLandmark(landmarks, ['right_eye_inner', 'right_eye_2', 'right_eye']);
  const leftEyeOuter = findLandmark(landmarks, ['left_eye_outer', 'left_eye_1', 'left_eye']);
  const rightEyeOuter = findLandmark(landmarks, ['right_eye_outer', 'right_eye_1', 'right_eye']);

  if (!leftEye || !rightEye) {
    return null;
  }

  const results = {};

  // فاصله بین چشم‌ها
  if (leftEyeInner && rightEyeInner) {
    const eyeDistance = calculateDistance(leftEyeInner, rightEyeInner);
    results.eyeDistance = eyeDistance.toFixed(1);
  }

  // اندازه چشم‌ها
  if (leftEyeOuter && leftEyeInner) {
    const leftEyeWidth = calculateDistance(leftEyeOuter, leftEyeInner);
    results.leftEyeWidth = leftEyeWidth.toFixed(1);
  }

  if (rightEyeOuter && rightEyeInner) {
    const rightEyeWidth = calculateDistance(rightEyeOuter, rightEyeInner);
    results.rightEyeWidth = rightEyeWidth.toFixed(1);
  }

  // تقارن چشم‌ها
  if (results.leftEyeWidth && results.rightEyeWidth) {
    const symmetry = 100 - (Math.abs(parseFloat(results.leftEyeWidth) - parseFloat(results.rightEyeWidth)) / 
      Math.max(parseFloat(results.leftEyeWidth), parseFloat(results.rightEyeWidth)) * 100);
    results.eyeSymmetry = Math.max(0, Math.min(100, symmetry)).toFixed(1);
  }

  return results;
}

/**
 * آنالیز بینی
 */
export function analyzeNose(landmarks) {
  const noseTip = findLandmark(landmarks, ['nose_tip', 'nose', 'nose_tip_1']);
  const noseBridgeTop = findLandmark(landmarks, ['nose_bridge_top', 'nose_bridge_1']);
  const noseBridgeBottom = findLandmark(landmarks, ['nose_bridge_bottom', 'nose_bridge_4']);
  const noseLeft = findLandmark(landmarks, ['nose_left', 'nose_tip_2']);
  const noseRight = findLandmark(landmarks, ['nose_right', 'nose_tip_4']);

  if (!noseTip) {
    return null;
  }

  const results = {};

  // ارتفاع بینی
  if (noseBridgeTop && noseTip) {
    const noseHeight = calculateDistance(noseBridgeTop, noseTip);
    results.noseHeight = noseHeight.toFixed(1);
  }

  // عرض بینی
  if (noseLeft && noseRight) {
    const noseWidth = calculateDistance(noseLeft, noseRight);
    results.noseWidth = noseWidth.toFixed(1);
  }

  // نسبت ارتفاع به عرض
  if (results.noseHeight && results.noseWidth) {
    const ratio = parseFloat(results.noseHeight) / parseFloat(results.noseWidth);
    results.noseRatio = ratio.toFixed(2);
    
    // نسبت ایده‌آل برای بینی: حدود 1.5-2
    const idealRatio = 1.75;
    const deviation = Math.abs(ratio - idealRatio) / idealRatio;
    results.noseRatioScore = Math.max(0, 100 - (deviation * 100)).toFixed(1);
  }

  return results;
}

/**
 * آنالیز دهان
 */
export function analyzeMouth(landmarks) {
  const mouthLeft = findLandmark(landmarks, ['mouth_left', 'mouth_left_corner', 'mouth_outer_1']);
  const mouthRight = findLandmark(landmarks, ['mouth_right', 'mouth_right_corner', 'mouth_outer_5']);
  const mouthTop = findLandmark(landmarks, ['mouth_top', 'mouth_outer_3']);
  const mouthBottom = findLandmark(landmarks, ['mouth_bottom', 'mouth_outer_9']);
  const mouthCenterTop = findLandmark(landmarks, ['mouth_center_top', 'mouth_top']);
  const mouthCenterBottom = findLandmark(landmarks, ['mouth_center_bottom', 'mouth_bottom']);

  if (!mouthLeft || !mouthRight) {
    return null;
  }

  const results = {};

  // عرض دهان
  const mouthWidth = calculateDistance(mouthLeft, mouthRight);
  results.mouthWidth = mouthWidth.toFixed(1);

  // ارتفاع دهان
  if (mouthTop && mouthBottom) {
    const mouthHeight = calculateDistance(mouthTop, mouthBottom);
    results.mouthHeight = mouthHeight.toFixed(1);
  }

  // نسبت عرض به ارتفاع
  if (results.mouthWidth && results.mouthHeight) {
    const ratio = parseFloat(results.mouthWidth) / parseFloat(results.mouthHeight);
    results.mouthRatio = ratio.toFixed(2);
  }

  return results;
}

/**
 * آنالیز Frontal & Oblique Facial Proportions
 * چون لندمارکی در بالای پیشانی نداریم، از Glabella به عنوان نقطه شروع استفاده می‌کنیم
 */
export function analyzeFacialProportions(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const results = {};

  // پیدا کردن نقاط کلیدی
  // استفاده از Glabella به جای forehead (چون forehead در dlib وجود ندارد)
  const glabella = findLandmark(landmarks, ['glabella', 'nose_bridge_top']);
  const noseTip = findLandmark(landmarks, ['nose_tip', 'nose']);
  const chin = findLandmark(landmarks, ['chin_center', 'chin', 'chin_8']);
  const leftFace = findLandmark(landmarks, ['chin_left', 'face_left', 'chin_1']);
  const rightFace = findLandmark(landmarks, ['chin_right', 'face_right', 'chin_17']);

  // تقسیمات عمودی صورت (از Glabella تا Chin)
  // چون لندمارک بالای پیشانی نداریم، فقط از Glabella به عنوان نقطه شروع استفاده می‌کنیم
  if (glabella && noseTip && chin) {
    // ارتفاع کل از Glabella تا Chin
    const glabellaToChinHeight = Math.abs(chin.y - glabella.y);
    // بخش میانی: Glabella → Nose Tip
    const glabellaToNoseHeight = Math.abs(noseTip.y - glabella.y);
    // بخش پایینی: Nose Tip → Chin
    const noseToChinHeight = Math.abs(chin.y - noseTip.y);
    
    if (glabellaToChinHeight > 0) {
      results.glabellaToNoseRatio = (glabellaToNoseHeight / glabellaToChinHeight * 100).toFixed(1);
      results.noseToChinRatio = (noseToChinHeight / glabellaToChinHeight * 100).toFixed(1);
      
      // نسبت ایده‌آل: تقریباً 50% برای هر بخش (از Glabella تا Chin)
      const idealRatio = 50.0;
      results.glabellaToNoseDeviation = Math.abs(parseFloat(results.glabellaToNoseRatio) - idealRatio).toFixed(1);
      results.noseToChinDeviation = Math.abs(parseFloat(results.noseToChinRatio) - idealRatio).toFixed(1);
    }
  }

  // توجه: Rule of Thirds (33.3% برای هر بخش) حذف شد چون نیاز به ارتفاع کل صورت دارد

  // نسبت عرض به ارتفاع صورت (از Glabella تا Chin)
  // برای عرض صورت، از نقاط کناری صورت استفاده می‌کنیم
  // اگر نقاط کناری پیدا نشدند، از چشم‌ها استفاده می‌کنیم
  const leftEye = findLandmark(landmarks, ['left_eye', 'left_eye_outer', 'left_eye_1']);
  const rightEye = findLandmark(landmarks, ['right_eye', 'right_eye_outer', 'right_eye_1']);
  
  let faceWidth = 0;
  if (leftFace && rightFace) {
    // استفاده از نقاط کناری صورت
    faceWidth = Math.abs(rightFace.x - leftFace.x);
  } else if (leftEye && rightEye) {
    // اگر نقاط کناری پیدا نشدند، از فاصله چشم‌ها و یک تخمین استفاده می‌کنیم
    const eyeDistance = Math.abs(rightEye.x - leftEye.x);
    // معمولاً عرض صورت حدود 1.5 تا 2 برابر فاصله بین چشم‌ها است
    faceWidth = eyeDistance * 1.75;
  }
  
  // ارتفاع صورت از Glabella تا Chin (نه ارتفاع کل صورت)
  // چون لندمارک بالای پیشانی نداریم، از Glabella به عنوان نقطه شروع استفاده می‌کنیم
  let faceHeight = 0;
  if (glabella && chin) {
    faceHeight = Math.abs(chin.y - glabella.y);
  }
  
  // محاسبه نسبت
  if (faceWidth > 0 && faceHeight > 0) {
    results.widthToHeightRatio = (faceWidth / faceHeight).toFixed(2);
    // نسبت ایده‌آل: حدود 0.8-0.85 (برای Glabella تا Chin)
    // توجه: این نسبت با نسبت کل صورت متفاوت است چون از Glabella شروع می‌شود نه از بالای پیشانی
    const idealRatio = 0.825;
    const deviation = Math.abs(parseFloat(results.widthToHeightRatio) - idealRatio) / idealRatio;
    results.widthToHeightScore = Math.max(0, 100 - (deviation * 100)).toFixed(1);
  }

  return Object.keys(results).length > 0 ? results : null;
}

/**
 * آنالیز Facial Asymmetry (Nasal/Chin Deviation)
 */
export function analyzeFacialAsymmetry(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const results = {};

  // پیدا کردن نقاط صورت برای محاسبه مرکز
  const leftFace = findLandmark(landmarks, ['chin_left', 'face_left', 'chin_1']);
  const rightFace = findLandmark(landmarks, ['chin_right', 'face_right', 'chin_17']);

  if (!leftFace || !rightFace) {
    return null;
  }

  const faceCenterX = (leftFace.x + rightFace.x) / 2;
  const faceWidth = Math.abs(rightFace.x - leftFace.x);

  // انحراف بینی
  const noseTip = findLandmark(landmarks, ['nose_tip', 'nose']);
  
  if (noseTip && faceWidth > 0) {
    const noseDeviation = Math.abs(noseTip.x - faceCenterX);
    // محاسبه درصد بر اساس فاصله از مرکز نسبت به نصف عرض صورت
    const halfFaceWidth = faceWidth / 2;
    results.noseDeviation = noseDeviation.toFixed(1);
    results.noseDeviationPercent = (noseDeviation / halfFaceWidth * 100).toFixed(1);
    results.noseDeviationGrade = parseFloat(results.noseDeviationPercent) < 2 ? 'Normal' :
                                  parseFloat(results.noseDeviationPercent) < 5 ? 'Mild' :
                                  parseFloat(results.noseDeviationPercent) < 10 ? 'Moderate' : 'Severe';
  }

  // انحراف چانه
  const chin = findLandmark(landmarks, ['chin_center', 'chin', 'chin_8']);
  if (chin && faceWidth > 0) {
    const chinDeviation = Math.abs(chin.x - faceCenterX);
    // محاسبه درصد بر اساس فاصله از مرکز نسبت به نصف عرض صورت
    const halfFaceWidth = faceWidth / 2;
    results.chinDeviation = chinDeviation.toFixed(1);
    results.chinDeviationPercent = (chinDeviation / halfFaceWidth * 100).toFixed(1);
    results.chinDeviationGrade = parseFloat(results.chinDeviationPercent) < 2 ? 'Normal' :
                                 parseFloat(results.chinDeviationPercent) < 5 ? 'Mild' :
                                 parseFloat(results.chinDeviationPercent) < 10 ? 'Moderate' : 'Severe';
  }

  // عدم تقارن کلی صورت
  const leftEye = findLandmark(landmarks, ['left_eye', 'left_eye_inner', 'left_eye_corner']);
  const rightEye = findLandmark(landmarks, ['right_eye', 'right_eye_inner', 'right_eye_corner']);
  const leftMouth = findLandmark(landmarks, ['mouth_left', 'mouth_left_corner']);
  const rightMouth = findLandmark(landmarks, ['mouth_right', 'mouth_right_corner']);

  if (leftEye && rightEye && leftMouth && rightMouth) {
    // محاسبه فاصله از مرکز برای هر نقطه
    const leftEyeDeviation = Math.abs(leftEye.x - faceCenterX);
    const rightEyeDeviation = Math.abs(rightEye.x - faceCenterX);
    const leftMouthDeviation = Math.abs(leftMouth.x - faceCenterX);
    const rightMouthDeviation = Math.abs(rightMouth.x - faceCenterX);
    
    // عدم تقارن = تفاوت بین فاصله‌های چپ و راست
    const eyeAsymmetry = Math.abs(leftEyeDeviation - rightEyeDeviation);
    const mouthAsymmetry = Math.abs(leftMouthDeviation - rightMouthDeviation);
    
    // تبدیل به درصد بر اساس عرض صورت
    if (faceWidth > 0) {
      results.eyeAsymmetry = eyeAsymmetry.toFixed(1);
      results.eyeAsymmetryPercent = (eyeAsymmetry / faceWidth * 100).toFixed(1);
      results.mouthAsymmetry = mouthAsymmetry.toFixed(1);
      results.mouthAsymmetryPercent = (mouthAsymmetry / faceWidth * 100).toFixed(1);
      results.overallAsymmetry = ((eyeAsymmetry + mouthAsymmetry) / 2).toFixed(1);
      results.overallAsymmetryPercent = ((parseFloat(results.eyeAsymmetryPercent) + parseFloat(results.mouthAsymmetryPercent)) / 2).toFixed(1);
    } else {
      results.eyeAsymmetry = eyeAsymmetry.toFixed(1);
      results.mouthAsymmetry = mouthAsymmetry.toFixed(1);
      results.overallAsymmetry = ((eyeAsymmetry + mouthAsymmetry) / 2).toFixed(1);
    }
  }

  return Object.keys(results).length > 0 ? results : null;
}

/**
 * آنالیز Face Height (فقط بخش‌هایی که قابل اندازه‌گیری هستند)
 * چون لندمارکی در بالای پیشانی (Trichion) نداریم، UFH و نسبت LFH/UFH قابل محاسبه نیستند
 * فقط Middle Face Height (MFH) و Lower Face Height (LFH) را محاسبه می‌کنیم
 */
export function analyzeFaceHeight(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const results = {};

  // نقاط کلیدی
  const noseTip = findLandmark(landmarks, ['nose_tip', 'nose']);
  const chin = findLandmark(landmarks, ['chin_center', 'chin', 'chin_8']);
  const mouthTop = findLandmark(landmarks, ['mouth_top', 'mouth_center_top']);
  const mouthBottom = findLandmark(landmarks, ['mouth_bottom', 'mouth_center_bottom']);

  // Middle Face Height (MFH): Nose Tip → Mouth Top
  if (noseTip && mouthTop) {
    results.middleFaceHeight = Math.abs(mouthTop.y - noseTip.y).toFixed(1);
  }

  // Lower Face Height (LFH): Mouth Top → Chin
  if (mouthTop && chin) {
    results.lowerFaceHeight = Math.abs(chin.y - mouthTop.y).toFixed(1);
  }

  // نسبت MFH به LFH (Middle Face Height / Lower Face Height)
  // این نسبت می‌تواند به عنوان یک شاخص استفاده شود
  if (results.middleFaceHeight && results.lowerFaceHeight) {
    const mfh = parseFloat(results.middleFaceHeight);
    const lfh = parseFloat(results.lowerFaceHeight);
    if (lfh > 0) {
      results.mfhToLfhRatio = (mfh / lfh).toFixed(2);
      // نسبت ایده‌آل: حدود 0.6-0.7 (MFH معمولاً کوچکتر از LFH است)
      const idealRatio = 0.65;
      const deviation = Math.abs(parseFloat(results.mfhToLfhRatio) - idealRatio) / idealRatio;
      results.mfhToLfhScore = Math.max(0, 100 - (deviation * 100)).toFixed(1);
    }
  }

  // توجه: UFH و LFH/UFH حذف شدند چون نیاز به لندمارک بالای پیشانی دارند

  return Object.keys(results).length > 0 ? results : null;
}

/**
 * آنالیز Profile (Convex/Concave)
 */
export function analyzeProfile(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const results = {};

  // نقاط کلیدی برای پروفایل
  // استفاده از Glabella به جای forehead (چون forehead در dlib وجود ندارد)
  const glabella = findLandmark(landmarks, ['glabella', 'nose_bridge_top']);
  const noseBridge = findLandmark(landmarks, ['nose_bridge_top', 'nose_bridge_1']);
  const noseTip = findLandmark(landmarks, ['nose_tip', 'nose']);
  const upperLip = findLandmark(landmarks, ['upper_lip', 'mouth_top', 'mouth_center_top']);
  const lowerLip = findLandmark(landmarks, ['lower_lip', 'mouth_bottom', 'mouth_center_bottom']);
  const chin = findLandmark(landmarks, ['chin_center', 'chin', 'chin_8']);

  // محاسبه پروفایل بر اساس موقعیت لب نسبت به خط بینی-چانه
  // این روش دقیق‌تر از محاسبه زاویه است
  if (noseTip && chin && upperLip) {
    // خط بینی-چانه را به عنوان مرجع استفاده می‌کنیم
    const lineDistance = pointToLineDistance(upperLip, noseTip, chin);
    
    // همچنین زاویه کلی را محاسبه می‌کنیم (از Glabella به جای forehead)
    if (glabella) {
      const angle = calculateAngle(glabella, noseTip, chin);
      results.profileAngle = angle.toFixed(1);
      
      // تعیین نوع پروفایل بر اساس زاویه و موقعیت لب
      // اگر لب جلوتر از خط بینی-چانه باشد = محدب
      // اگر لب عقب‌تر از خط بینی-چانه باشد = مقعر
      if (lineDistance > 5) {
        results.profileType = 'Convex';
        results.profileDescription = 'پروفایل محدب (پیش آمدگی)';
      } else if (lineDistance < -5) {
        results.profileType = 'Concave';
        results.profileDescription = 'پروفایل مقعر (پس رفتگی)';
      } else {
        results.profileType = 'Straight';
        results.profileDescription = 'پروفایل مستقیم';
      }
    } else {
      // بدون Glabella، فقط بر اساس موقعیت لب
      if (lineDistance > 5) {
        results.profileType = 'Convex';
        results.profileDescription = 'پروفایل محدب (پیش آمدگی)';
      } else if (lineDistance < -5) {
        results.profileType = 'Concave';
        results.profileDescription = 'پروفایل مقعر (پس رفتگی)';
      } else {
        results.profileType = 'Straight';
        results.profileDescription = 'پروفایل مستقیم';
      }
    }
  }

  // زاویه لب
  if (upperLip && lowerLip && chin) {
    const lipAngle = calculateAngle(upperLip, lowerLip, chin);
    results.lipAngle = lipAngle.toFixed(1);
  }

  // برجستگی لب
  if (noseTip && upperLip && chin) {
    // محاسبه فاصله لب از خط بینی-چانه
    const lineDistance = pointToLineDistance(upperLip, noseTip, chin);
    results.lipProminence = lineDistance.toFixed(1);
    
    if (lineDistance > 5) {
      results.lipProminenceType = 'Protrusion';
      results.lipProminenceDescription = 'لب برجسته (پیش آمدگی)';
    } else if (lineDistance < -5) {
      results.lipProminenceType = 'Retrusion';
      results.lipProminenceDescription = 'لب پس رفته (پس رفتگی)';
    } else {
      results.lipProminenceType = 'Normal';
      results.lipProminenceDescription = 'لب طبیعی';
    }
  }

  return Object.keys(results).length > 0 ? results : null;
}

/**
 * آنالیز Lip (Prominence, Retro/Protrusion, Everted Incompetence)
 */
export function analyzeLipDetails(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const results = {};

  const upperLip = findLandmark(landmarks, ['upper_lip', 'mouth_top', 'mouth_center_top']);
  const lowerLip = findLandmark(landmarks, ['lower_lip', 'mouth_bottom', 'mouth_center_bottom']);
  const mouthLeft = findLandmark(landmarks, ['mouth_left', 'mouth_left_corner']);
  const mouthRight = findLandmark(landmarks, ['mouth_right', 'mouth_right_corner']);
  const noseTip = findLandmark(landmarks, ['nose_tip', 'nose']);
  const chin = findLandmark(landmarks, ['chin_center', 'chin', 'chin_8']);

  // برجستگی لب بالا
  if (upperLip && noseTip && chin) {
    const distance = pointToLineDistance(upperLip, noseTip, chin);
    results.upperLipProminence = distance.toFixed(1);
    
    if (distance > 3) {
      results.upperLipType = 'Protrusion';
    } else if (distance < -3) {
      results.upperLipType = 'Retrusion';
    } else {
      results.upperLipType = 'Normal';
    }
  }

  // برجستگی لب پایین
  if (lowerLip && noseTip && chin) {
    const distance = pointToLineDistance(lowerLip, noseTip, chin);
    results.lowerLipProminence = distance.toFixed(1);
    
    if (distance > 3) {
      results.lowerLipType = 'Protrusion';
    } else if (distance < -3) {
      results.lowerLipType = 'Retrusion';
    } else {
      results.lowerLipType = 'Normal';
    }
  }

  // عرض لب
  if (mouthLeft && mouthRight) {
    results.lipWidth = calculateDistance(mouthLeft, mouthRight).toFixed(1);
  }

  // ارتفاع لب
  if (upperLip && lowerLip) {
    results.lipHeight = Math.abs(lowerLip.y - upperLip.y).toFixed(1);
  }

  // Incompetence (عدم بسته شدن لب‌ها)
  if (upperLip && lowerLip) {
    const verticalDistance = Math.abs(lowerLip.y - upperLip.y);
    results.lipIncompetence = verticalDistance > 5 ? 'Yes' : 'No';
    results.lipIncompetenceDistance = verticalDistance.toFixed(1);
  }

  // Everted (برگشتگی لب)
  // این نیاز به آنالیز دقیق‌تر دارد که در صورت دو بعدی محدود است

  return Object.keys(results).length > 0 ? results : null;
}

/**
 * آنالیز Throat Form & Length
 */
export function analyzeThroat(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const results = {};

  const chin = findLandmark(landmarks, ['chin_center', 'chin', 'chin_8']);
  const jawLine = findLandmark(landmarks, ['jaw_line', 'chin_left', 'chin_1']);
  
  // این آنالیز نیاز به تصویر پروفایل دارد
  // در تصویر frontal محدود است
  
  if (chin) {
    results.chinPosition = { x: chin.x, y: chin.y };
  }

  return Object.keys(results).length > 0 ? results : null;
}

/**
 * آنالیز Nasolabial fold (چین نازولابیال)
 */
export function analyzeNasolabialFold(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const results = {};

  const noseLeft = findLandmark(landmarks, ['nose_left', 'nose_tip_2']);
  const noseRight = findLandmark(landmarks, ['nose_right', 'nose_tip_4']);
  const mouthLeft = findLandmark(landmarks, ['mouth_left', 'mouth_left_corner']);
  const mouthRight = findLandmark(landmarks, ['mouth_right', 'mouth_right_corner']);

  // زاویه چین نازولابیال
  if (noseLeft && mouthLeft) {
    const leftAngle = Math.atan2(mouthLeft.y - noseLeft.y, mouthLeft.x - noseLeft.x) * 180 / Math.PI;
    results.leftNasolabialAngle = leftAngle.toFixed(1);
  }

  if (noseRight && mouthRight) {
    const rightAngle = Math.atan2(mouthRight.y - noseRight.y, mouthRight.x - noseRight.x) * 180 / Math.PI;
    results.rightNasolabialAngle = rightAngle.toFixed(1);
  }

  // تقارن چین نازولابیال
  if (results.leftNasolabialAngle && results.rightNasolabialAngle) {
    const asymmetry = Math.abs(parseFloat(results.leftNasolabialAngle) - parseFloat(results.rightNasolabialAngle));
    results.nasolabialAsymmetry = asymmetry.toFixed(1);
    results.nasolabialSymmetry = asymmetry < 5 ? 'Symmetric' : 'Asymmetric';
  }

  return Object.keys(results).length > 0 ? results : null;
}

/**
 * آنالیز Incisor Show (Rest/Smile)
 */
export function analyzeIncisorShow(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const results = {};

  const upperLip = findLandmark(landmarks, ['upper_lip', 'mouth_top', 'mouth_center_top']);
  const lowerLip = findLandmark(landmarks, ['lower_lip', 'mouth_bottom', 'mouth_center_bottom']);
  const mouthTop = findLandmark(landmarks, ['mouth_top', 'mouth_center_top']);
  const mouthBottom = findLandmark(landmarks, ['mouth_bottom', 'mouth_center_bottom']);

  // در حالت استراحت (Rest)
  // این نیاز به تشخیص دقیق‌تر دارد که در تصویر frontal محدود است
  // ولی می‌توانیم فاصله لب‌ها را محاسبه کنیم
  
  if (upperLip && lowerLip) {
    const restDistance = Math.abs(lowerLip.y - upperLip.y);
    results.restLipDistance = restDistance.toFixed(1);
    
    // در حالت استراحت طبیعی، فاصله باید کم باشد
    if (restDistance < 3) {
      results.restIncisorShow = 'Normal';
    } else {
      results.restIncisorShow = 'Excessive';
    }
  }

  // در حالت لبخند (Smile)
  // این نیاز به تشخیص حالت لبخند دارد
  // می‌توانیم عرض دهان را محاسبه کنیم
  
  const mouthLeft = findLandmark(landmarks, ['mouth_left', 'mouth_left_corner']);
  const mouthRight = findLandmark(landmarks, ['mouth_right', 'mouth_right_corner']);
  
  if (mouthLeft && mouthRight) {
    const smileWidth = calculateDistance(mouthLeft, mouthRight);
    results.smileWidth = smileWidth.toFixed(1);
  }

  return Object.keys(results).length > 0 ? results : null;
}

/**
 * آنالیز Gingival Show (Ant./Post.)
 */
export function analyzeGingivalShow(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const results = {};

  const upperLip = findLandmark(landmarks, ['upper_lip', 'mouth_top', 'mouth_center_top']);
  const mouthTop = findLandmark(landmarks, ['mouth_top', 'mouth_center_top']);

  // این آنالیز نیاز به تشخیص دقیق لثه در تصویر دارد
  // در حالت فعلی محدود است
  
  if (upperLip && mouthTop) {
    const lipToMouthDistance = Math.abs(upperLip.y - mouthTop.y);
    results.lipToMouthDistance = lipToMouthDistance.toFixed(1);
    
    // اگر فاصله زیاد باشد، احتمال نمایش لثه وجود دارد
    if (lipToMouthDistance > 5) {
      results.gingivalShow = 'Possible';
    } else {
      results.gingivalShow = 'Normal';
    }
  }

  return Object.keys(results).length > 0 ? results : null;
}

/**
 * آنالیز Smile Arc (Consonant/Nonconsonant [Flat/Reverse])
 */
export function analyzeSmileArc(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const results = {};

  const mouthLeft = findLandmark(landmarks, ['mouth_left', 'mouth_left_corner']);
  const mouthRight = findLandmark(landmarks, ['mouth_right', 'mouth_right_corner']);
  const mouthTop = findLandmark(landmarks, ['mouth_top', 'mouth_center_top']);
  const mouthBottom = findLandmark(landmarks, ['mouth_bottom', 'mouth_center_bottom']);

  if (mouthLeft && mouthRight && mouthTop && mouthBottom) {
    // محاسبه قوس لبخند
    // مقایسه ارتفاع مرکز دهان با ارتفاع گوشه‌ها
    const leftCornerY = mouthLeft.y;
    const rightCornerY = mouthRight.y;
    const centerY = mouthTop.y;
    const avgCornerY = (leftCornerY + rightCornerY) / 2;
    
    // محاسبه ارتفاع قوس
    const arcHeight = Math.abs(centerY - avgCornerY);
    
    // در قوس هماهنگ (Consonant)، مرکز بالاتر از گوشه‌ها است (Y کوچکتر = بالاتر)
    // در قوس معکوس (Reverse)، مرکز پایین‌تر از گوشه‌ها است (Y بزرگتر = پایین‌تر)
    // در قوس صاف (Flat)، مرکز و گوشه‌ها در یک خط هستند
    const threshold = 3; // حد آستانه برای تشخیص (به پیکسل)
    
    if (centerY < avgCornerY - threshold) {
      results.smileArcType = 'Consonant';
      results.smileArcDescription = 'قوس هماهنگ (طبیعی)';
    } else if (centerY > avgCornerY + threshold) {
      results.smileArcType = 'Reverse';
      results.smileArcDescription = 'قوس معکوس';
    } else {
      results.smileArcType = 'Flat';
      results.smileArcDescription = 'قوس صاف';
    }
    
    results.smileArcHeight = arcHeight.toFixed(1);
  }

  return Object.keys(results).length > 0 ? results : null;
}

/**
 * آنالیز Smile Width / Buccal Corridors
 */
export function analyzeSmileWidth(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;

  const results = {};

  const mouthLeft = findLandmark(landmarks, ['mouth_left', 'mouth_left_corner']);
  const mouthRight = findLandmark(landmarks, ['mouth_right', 'mouth_right_corner']);
  const leftFace = findLandmark(landmarks, ['chin_left', 'face_left', 'chin_1']);
  const rightFace = findLandmark(landmarks, ['chin_right', 'face_right', 'chin_17']);

  if (mouthLeft && mouthRight) {
    results.smileWidth = calculateDistance(mouthLeft, mouthRight).toFixed(1);
  }

  // Buccal Corridors (راهروهای گونه‌ای)
  // فاصله بین گوشه دهان و کنار صورت در هر طرف
  if (mouthLeft && mouthRight && leftFace && rightFace) {
    const faceWidth = Math.abs(rightFace.x - leftFace.x);
    const smileWidth = Math.abs(rightFace.x - mouthRight.x) + Math.abs(mouthLeft.x - leftFace.x);
    
    // محاسبه راهروهای گونه‌ای: فاصله بین گوشه دهان و کنار صورت در هر طرف
    const leftBuccalCorridor = Math.abs(mouthLeft.x - leftFace.x);
    const rightBuccalCorridor = Math.abs(rightFace.x - mouthRight.x);
    const avgBuccalCorridor = (leftBuccalCorridor + rightBuccalCorridor) / 2;
    
    if (faceWidth > 0) {
      // نسبت راهروهای گونه‌ای به عرض صورت
      const buccalCorridorRatio = (avgBuccalCorridor / faceWidth) * 100;
      results.buccalCorridorRatio = buccalCorridorRatio.toFixed(1);
      
      // نسبت ایده‌آل: 10-15%
      if (buccalCorridorRatio < 10) {
        results.buccalCorridorType = 'Wide Smile';
        results.buccalCorridorDescription = 'لبخند عریض (راهروهای کوچک)';
      } else if (buccalCorridorRatio > 15) {
        results.buccalCorridorType = 'Narrow Smile';
        results.buccalCorridorDescription = 'لبخند باریک (راهروهای بزرگ)';
      } else {
        results.buccalCorridorType = 'Ideal';
        results.buccalCorridorDescription = 'لبخند ایده‌آل';
      }
    }
  }

  return Object.keys(results).length > 0 ? results : null;
}

/**
 * توابع کمکی
 */
function calculateAngle(point1, vertex, point2) {
  const v1 = { x: point1.x - vertex.x, y: point1.y - vertex.y };
  const v2 = { x: point2.x - vertex.x, y: point2.y - vertex.y };
  
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
  const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
  
  const angle = Math.acos(dot / (mag1 * mag2)) * 180 / Math.PI;
  return angle;
}

function pointToLineDistance(point, lineStart, lineEnd) {
  const A = point.x - lineStart.x;
  const B = point.y - lineStart.y;
  const C = lineEnd.x - lineStart.x;
  const D = lineEnd.y - lineStart.y;

  const dot = A * C + B * D;
  const lenSq = C * C + D * D;
  let param = -1;
  if (lenSq !== 0) param = dot / lenSq;

  let xx; let yy;

  if (param < 0) {
    xx = lineStart.x;
    yy = lineStart.y;
  } else if (param > 1) {
    xx = lineEnd.x;
    yy = lineEnd.y;
  } else {
    xx = lineStart.x + param * C;
    yy = lineStart.y + param * D;
  }

  const dx = point.x - xx;
  const dy = point.y - yy;
  return Math.sqrt(dx * dx + dy * dy) * (point.y > yy ? 1 : -1);
}

/**
 * آنالیز کامل زیبایی صورت
 */
export function analyzeFacialBeauty(landmarks) {
  if (!landmarks || landmarks.length === 0) {
    return {
      success: false,
      error: 'No landmarks provided',
    };
  }

  // آنالیزهای پایه
  const symmetry = calculateSymmetry(landmarks);
  const goldenRatio = calculateGoldenRatio(landmarks);
  const eyes = analyzeEyes(landmarks);
  const nose = analyzeNose(landmarks);
  const mouth = analyzeMouth(landmarks);

  // Macro-Esthetics
  const facialProportions = analyzeFacialProportions(landmarks);
  const facialAsymmetry = analyzeFacialAsymmetry(landmarks);
  const faceHeight = analyzeFaceHeight(landmarks);
  const profile = analyzeProfile(landmarks);
  const lipDetails = analyzeLipDetails(landmarks);
  const throat = analyzeThroat(landmarks);
  const nasolabialFold = analyzeNasolabialFold(landmarks);

  // Mini-Esthetics
  const incisorShow = analyzeIncisorShow(landmarks);
  const gingivalShow = analyzeGingivalShow(landmarks);
  const smileArc = analyzeSmileArc(landmarks);
  const smileWidth = analyzeSmileWidth(landmarks);

  // Frontal Analysis Lines
  const midfacialLineDeviation = calculateMidfacialLineDeviation(landmarks);
  const interpupillaryLineAngle = calculateInterpupillaryLineAngle(landmarks);
  const commissuralLineAngle = calculateCommissuralLineAngle(landmarks);
  const dentalMidlineDeviation = calculateDentalMidlineDeviation(landmarks);

  // Additional Proportions
  const verticalProportions = calculateVerticalProportions(landmarks);
  const facialIndex = calculateFacialIndex(landmarks);
  const transverseProportions = calculateTransverseProportions(landmarks);
  const lipProportions = calculateLipProportions(landmarks);

  // محاسبه امتیاز کلی
  let totalScore = 0;
  let scoreCount = 0;

  if (symmetry) {
    totalScore += symmetry.overall;
    scoreCount += 1;
  }

  if (goldenRatio) {
    if (goldenRatio.verticalRatio) {
      totalScore += goldenRatio.verticalRatio.score;
      scoreCount += 1;
    }
    if (goldenRatio.horizontalRatio) {
      totalScore += goldenRatio.horizontalRatio.score;
      scoreCount += 1;
    }
    if (goldenRatio.eyeToNoseRatio) {
      totalScore += goldenRatio.eyeToNoseRatio.score;
      scoreCount += 1;
    }
  }

  if (nose?.noseRatioScore) {
    totalScore += parseFloat(nose.noseRatioScore);
    scoreCount += 1;
  }

  if (facialProportions?.widthToHeightScore) {
    totalScore += parseFloat(facialProportions.widthToHeightScore);
    scoreCount += 1;
  }

  // LFH/UFH حذف شد - نیاز به لندمارک بالای پیشانی دارد
  // استفاده از MFH/LFH به جای آن
  if (faceHeight?.mfhToLfhScore) {
    totalScore += parseFloat(faceHeight.mfhToLfhScore);
    scoreCount += 1;
  }

  const overallScore = scoreCount > 0 ? (totalScore / scoreCount).toFixed(1) : null;

  // تعیین رتبه زیبایی
  let beautyGrade = 'N/A';
  if (overallScore !== null) {
    const score = parseFloat(overallScore);
    if (score >= 90) beautyGrade = 'عالی';
    else if (score >= 80) beautyGrade = 'خیلی خوب';
    else if (score >= 70) beautyGrade = 'خوب';
    else if (score >= 60) beautyGrade = 'متوسط';
    else if (score >= 50) beautyGrade = 'نیاز به بهبود';
    else beautyGrade = 'نیاز به اصلاح';
  }

  return {
    success: true,
    overallScore: overallScore ? parseFloat(overallScore) : null,
    beautyGrade,
    // Basic Analysis
    symmetry,
    goldenRatio,
    eyes,
    nose,
    mouth,
    // Macro-Esthetics
    facialProportions,
    facialAsymmetry,
    faceHeight,
    profile,
    lipDetails,
    throat,
    nasolabialFold,
    // Mini-Esthetics
    incisorShow,
    gingivalShow,
    smileArc,
    smileWidth,
    // Frontal Analysis Lines
    midfacialLineDeviation,
    interpupillaryLineAngle,
    commissuralLineAngle,
    dentalMidlineDeviation,
    // Additional Proportions
    verticalProportions,
    facialIndex,
    transverseProportions,
    lipProportions,
    recommendations: generateRecommendations(symmetry, goldenRatio, nose, overallScore, facialAsymmetry, profile, smileArc),
  };
}

/**
 * تولید توصیه‌ها بر اساس آنالیز
 */
function generateRecommendations(symmetry, goldenRatio, nose, overallScore, facialAsymmetry, profile, smileArc) {
  const recommendations = [];

  // تقارن
  if (symmetry && symmetry.overall < 70) {
    recommendations.push('تقارن صورت نیاز به بهبود دارد. ممکن است به درمان ارتودنسی یا جراحی نیاز باشد.');
  }

  // عدم تقارن
  if (facialAsymmetry) {
    if (facialAsymmetry.noseDeviationGrade === 'Severe' || facialAsymmetry.noseDeviationGrade === 'Moderate') {
      recommendations.push(`انحراف بینی ${facialAsymmetry.noseDeviationGrade} تشخیص داده شد. مشاوره با متخصص ارتودنسی یا جراح توصیه می‌شود.`);
    }
    if (facialAsymmetry.chinDeviationGrade === 'Severe' || facialAsymmetry.chinDeviationGrade === 'Moderate') {
      recommendations.push(`انحراف چانه ${facialAsymmetry.chinDeviationGrade} تشخیص داده شد.`);
    }
  }

  // نسبت طلایی
  if (goldenRatio && goldenRatio.verticalRatio && parseFloat(goldenRatio.verticalRatio.score) < 70) {
    recommendations.push('نسبت عمودی صورت از نسبت طلایی فاصله دارد.');
  }

  // پروفایل
  if (profile) {
    if (profile.profileType === 'Convex') {
      recommendations.push('پروفایل محدب تشخیص داده شد. ممکن است نیاز به درمان ارتودنسی یا جراحی باشد.');
    } else if (profile.profileType === 'Concave') {
      recommendations.push('پروفایل مقعر تشخیص داده شد. مشاوره با متخصص ارتودنسی توصیه می‌شود.');
    }
    if (profile.lipProminenceType === 'Protrusion') {
      recommendations.push('لب برجسته (Protrusion) تشخیص داده شد.');
    } else if (profile.lipProminenceType === 'Retrusion') {
      recommendations.push('لب پس رفته (Retrusion) تشخیص داده شد.');
    }
  }

  // نسبت بینی
  if (nose && nose.noseRatioScore && parseFloat(nose.noseRatioScore) < 70) {
    recommendations.push('نسبت بینی نیاز به بررسی دارد.');
  }

  // قوس لبخند
  if (smileArc) {
    if (smileArc.smileArcType === 'Reverse') {
      recommendations.push('قوس لبخند معکوس تشخیص داده شد. این ممکن است نیاز به درمان داشته باشد.');
    } else if (smileArc.smileArcType === 'Flat') {
      recommendations.push('قوس لبخند صاف تشخیص داده شد.');
    }
  }

  // امتیاز کلی
  if (overallScore && parseFloat(overallScore) >= 80) {
    recommendations.push('صورت شما دارای نسبت‌های زیبایی خوبی است!');
  }

  if (recommendations.length === 0) {
    recommendations.push('آنالیز اولیه انجام شد. برای نتیجه دقیق‌تر با متخصص ارتودنسی یا جراح مشورت کنید.');
  }

  return recommendations;
}

