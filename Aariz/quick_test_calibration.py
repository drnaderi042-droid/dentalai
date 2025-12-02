"""
Quick test for calibration detection - test on one image
"""

import cv2
import numpy as np
import sys

def detect_ruler_marks(image_path):
    """Simple ruler mark detection."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read: {image_path}")
        return
    
    h, w = img.shape[:2]
    print(f"📏 Image size: {w}x{h}")
    
    # Search area (upper right)
    x_start = int(w * 0.7)
    y_end = int(h * 0.25)
    
    roi = img[0:y_end, x_start:w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Find bright spots - lower threshold for better detection
    _, bright = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY)
    
    # Morphology to clean up
    kernel = np.ones((3, 3), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 500:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                # Brightness
                mask = np.zeros_like(gray_roi)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                brightness = cv2.mean(gray_roi, mask=mask)[0]
                
                points.append({
                    'x': x_start + cx,
                    'y': cy,
                    'brightness': brightness
                })
    
    # Sort by brightness
    points.sort(key=lambda p: p['brightness'], reverse=True)
    
    print(f"🔍 Found {len(points)} bright points")
    
    # Find vertical pair
    for i in range(min(10, len(points))):
        for j in range(i+1, min(10, len(points))):
            p1, p2 = points[i], points[j]
            dx = abs(p1['x'] - p2['x'])
            dy = abs(p1['y'] - p2['y'])
            
            if dx < 30 and 70 < dy < 150:
                upper = p1 if p1['y'] < p2['y'] else p2
                lower = p2 if p1['y'] < p2['y'] else p1
                
                dist = np.sqrt(dx**2 + dy**2)
                mm_px = 10 / dist
                
                print(f"\n✅ Found calibration pair:")
                print(f"   p2 (upper): ({upper['x']}, {upper['y']})")
                print(f"   p1 (lower): ({lower['x']}, {lower['y']})")
                print(f"   Distance: {dist:.1f} pixels")
                print(f"   Conversion: {mm_px:.4f} mm/pixel")
                print(f"   DPI: {25.4/mm_px:.0f}")
                
                # Visualize
                vis = img.copy()
                cv2.rectangle(vis, (x_start, 0), (w, y_end), (255,255,0), 2)
                cv2.circle(vis, (upper['x'], upper['y']), 8, (0,0,255), 2)
                cv2.circle(vis, (lower['x'], lower['y']), 8, (0,0,255), 2)
                cv2.line(vis, (upper['x'], upper['y']), (lower['x'], lower['y']), (0,0,255), 2)
                cv2.putText(vis, "p2", (upper['x']+10, upper['y']), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(vis, "p1", (lower['x']+10, lower['y']), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
                # Show all detected points
                for k, p in enumerate(points[:10]):
                    color = (0,255,0) if k in [i,j] else (255,0,0)
                    cv2.circle(vis, (p['x'], p['y']), 3, color, -1)
                
                cv2.imwrite('calibration_detection_result.png', vis)
                print(f"\n💾 Saved visualization to: calibration_detection_result.png")
                
                # Show resized
                scale = min(1.0, 1200/max(w, h))
                resized = cv2.resize(vis, None, fx=scale, fy=scale)
                cv2.imshow('Calibration Detection', resized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                return upper, lower
    
    print("❌ No calibration pair found")
    return None

if __name__ == '__main__':
    import os
    
    # Get script directory and change to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"📂 Working directory: {os.getcwd()}\n")
    
    # Use image from Aariz dataset
    test_image = 'Aariz/train/Cephalograms/cks2ip8fq29yq0yufc4scftj8.png'
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    if not os.path.exists(test_image):
        print(f"⚠️ Image not found, trying alternative paths...")
        # Try different extensions
        base_name = test_image.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        for ext in ['.png', '.jpg', '.jpeg']:
            alt_path = base_name + ext
            if os.path.exists(alt_path):
                test_image = alt_path
                print(f"✅ Found: {test_image}")
                break
        else:
            print(f"❌ Could not find image: {test_image}")
            sys.exit(1)
    
    print(f"🧪 Testing: {test_image}\n")
    detect_ruler_marks(test_image)

