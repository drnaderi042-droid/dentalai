"""
Debug script to visualize ground truth p1/p2 locations
"""

import cv2
import json
from pathlib import Path

def debug_ground_truth():
    """Show where ground truth p1/p2 are located."""
    
    # Test with first image
    image_id = 'cks2ip8fq29yq0yufc4scftj8'
    
    annotation_file = Path(f'Aariz/train/Annotations/Cephalometric Landmarks/Senior Orthodontists/{image_id}.json')
    image_path = Path(f'Aariz/train/Cephalograms/{image_id}.png')
    
    # Load annotation
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Find p1 and p2
    p1, p2 = None, None
    for landmark in data['landmarks']:
        if landmark['symbol'] == 'p1':
            p1 = landmark['value']
        elif landmark['symbol'] == 'p2':
            p2 = landmark['value']
    
    print(f"📍 Ground Truth Locations:")
    print(f"   p1: {p1}")
    print(f"   p2: {p2}")
    
    # Load image
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    print(f"\n📏 Image size: {w}x{h}")
    
    # Calculate percentage
    if p1 and p2:
        p1_pct_x = (p1['x'] / w) * 100
        p1_pct_y = (p1['y'] / h) * 100
        p2_pct_x = (p2['x'] / w) * 100
        p2_pct_y = (p2['y'] / h) * 100
        
        print(f"\n📊 As percentages:")
        print(f"   p1: ({p1_pct_x:.1f}%, {p1_pct_y:.1f}%)")
        print(f"   p2: ({p2_pct_x:.1f}%, {p2_pct_y:.1f}%)")
        
        # Distance
        dx = p1['x'] - p2['x']
        dy = p1['y'] - p2['y']
        dist = (dx**2 + dy**2)**0.5
        print(f"\n📏 Distance: {dist:.1f} pixels")
        print(f"   dx: {dx}, dy: {dy}")
        
        # Draw on image
        vis = img.copy()
        
        # Draw p1 (green - lower point)
        cv2.circle(vis, (p1['x'], p1['y']), 15, (0, 255, 0), 3)
        cv2.putText(vis, "p1 (GT)", (p1['x']+20, p1['y']), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Draw p2 (blue - upper point)
        cv2.circle(vis, (p2['x'], p2['y']), 15, (255, 0, 0), 3)
        cv2.putText(vis, "p2 (GT)", (p2['x']+20, p2['y']), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        
        # Draw line
        cv2.line(vis, (p1['x'], p1['y']), (p2['x'], p2['y']), (255, 255, 0), 2)
        
        # Draw search area (upper right 20%)
        search_x = int(w * 0.8)
        search_y_end = int(h * 0.2)
        cv2.rectangle(vis, (search_x, 0), (w, search_y_end), (0, 255, 255), 3)
        cv2.putText(vis, "Search Area", (search_x+10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Check if points are in search area
        p1_in_search = p1['x'] >= search_x and p1['y'] <= search_y_end
        p2_in_search = p2['x'] >= search_x and p2['y'] <= search_y_end
        
        print(f"\n🔍 In search area?")
        print(f"   p1: {'✅ YES' if p1_in_search else '❌ NO'}")
        print(f"   p2: {'✅ YES' if p2_in_search else '❌ NO'}")
        
        if not p1_in_search or not p2_in_search:
            print(f"\n⚠️ WARNING: Ground truth points are NOT in the search area!")
            print(f"   Search area: x >= {search_x}, y <= {search_y_end}")
            print(f"   p1 location: x={p1['x']}, y={p1['y']}")
            print(f"   p2 location: x={p2['x']}, y={p2['y']}")
        
        # Save
        cv2.imwrite('debug_ground_truth.png', vis)
        print(f"\n💾 Saved: debug_ground_truth.png")
        
        # Show resized
        scale = min(1.0, 1200/max(w, h))
        resized = cv2.resize(vis, None, fx=scale, fy=scale)
        cv2.imshow('Ground Truth Debug', resized)
        print(f"\n👀 Displaying image... Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import os
    os.chdir('aariz')
    debug_ground_truth()

