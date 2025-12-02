"""
P1/P2 Calibration Point Annotator
Simple and fast tool for annotating p1 and p2 points on cephalometric images
"""
import cv2
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import argparse


class P1P2Annotator:
    """Interactive annotator for p1/p2 calibration points"""
    
    def __init__(self, images_dir, output_file='annotations_p1_p2.json', max_images=None):
        self.images_dir = Path(images_dir)
        self.output_file = output_file
        
        # Get all images
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_files.extend(list(self.images_dir.glob(ext)))
        
        if not self.image_files:
            raise ValueError(f"No images found in {images_dir}")
        
        # Sort for consistent order
        self.image_files.sort()
        
        # Store all image files before filtering
        all_image_files = self.image_files.copy()
        
        self.current_idx = 0
        self.annotations = self.load_existing_annotations()
        
        # Filter out already annotated images
        annotated_paths = set(self.annotations.keys())
        unannotated_files = [f for f in all_image_files if str(f) not in annotated_paths]
        
        print(f"Total images in directory: {len(all_image_files)}")
        print(f"Already annotated: {len(annotated_paths)}")
        print(f"Remaining to annotate: {len(unannotated_files)}")
        
        # If max_images specified, limit unannotated images
        if max_images:
            # Calculate how many more we need
            remaining_needed = max_images - len(annotated_paths)
            if remaining_needed > 0:
                if remaining_needed < len(unannotated_files):
                    print(f"Limiting to {remaining_needed} more images (to reach {max_images} total)")
                    unannotated_files = unannotated_files[:remaining_needed]
                else:
                    print(f"Will annotate {len(unannotated_files)} more images (target: {max_images} total)")
        
        # Use unannotated files for annotation
        self.image_files = unannotated_files if unannotated_files else []
        
        if not self.image_files:
            print("\nWARNING: All images are already annotated!")
            print(f"You have {len(annotated_paths)} annotations. Increase MAX_IMAGES if you want to annotate more.")
        
        # State
        self.p1 = None
        self.p2 = None
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # Colors
        self.COLOR_P1 = (0, 165, 255)  # Orange
        self.COLOR_P2 = (0, 215, 255)  # Gold
        self.COLOR_TEMP = (0, 255, 0)   # Green
        
        print(f"\nImages ready for annotation: {len(self.image_files)}")
        print(f"Total existing annotations: {len(self.annotations)}")
    
    def load_existing_annotations(self):
        """Load existing annotations if file exists and convert from JSON format to internal format"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    
                # Convert from Aariz JSON format to internal format
                annotations = {}
                for item in json_data:
                    # Get image filename
                    image_filename = item.get('file_upload', '')
                    if not image_filename:
                        continue
                    
                    # Find corresponding image path from all available images
                    image_path = None
                    for img_file in self.image_files:
                        if img_file.name == image_filename:
                            image_path = str(img_file)
                            break
                    
                    if not image_path:
                        # Try to construct path if not found in current directory
                        potential_path = self.images_dir / image_filename
                        if potential_path.exists():
                            image_path = str(potential_path)
                        else:
                            # Use relative path as fallback
                            image_path = str(self.images_dir / image_filename)
                    
                    # Extract p1 and p2 from annotations
                    points = {}
                    if 'annotations' in item and len(item['annotations']) > 0:
                        result = item['annotations'][0].get('result', [])
                        for r in result:
                            if r.get('type') == 'keypointlabels':
                                value = r.get('value', {})
                                labels = value.get('keypointlabels', [])
                                # Convert from percentage back to normalized [0, 1]
                                x = value.get('x', 0) / 100.0
                                y = value.get('y', 0) / 100.0
                                
                                if 'p1' in labels:
                                    points['p1'] = {'x': x, 'y': y}
                                elif 'p2' in labels:
                                    points['p2'] = {'x': x, 'y': y}
                    
                    if 'p1' in points and 'p2' in points:
                        annotations[image_path] = points
                
                print(f"Loaded {len(annotations)} existing annotations from {self.output_file}")
                return annotations
            except Exception as e:
                print(f"Could not load {self.output_file}: {e}")
                import traceback
                traceback.print_exc()
                print("Starting fresh")
                return {}
        return {}
    
    def save_annotations(self):
        """Save annotations to JSON file (Aariz format), merging with existing annotations"""
        # Load existing annotations from JSON file to preserve them
        existing_data = []
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except:
                pass
        
        # Create a set of already saved image filenames
        saved_filenames = {item.get('file_upload', '') for item in existing_data}
        
        # Add new annotations
        for image_path_str, points in self.annotations.items():
            image_filename = Path(image_path_str).name
            
            # Skip if already in existing data
            if image_filename in saved_filenames:
                continue
            
            # Create Aariz-style annotation
            annotation = {
                'id': len(existing_data) + 1,
                'data': {
                    'image': f'/data/upload/{image_filename}'
                },
                'annotations': [{
                    'id': len(existing_data) + 1,
                    'completed_by': 1,
                    'created_at': datetime.now().isoformat() + 'Z',
                    'result': []
                }],
                'file_upload': image_filename
            }
            
            # Add p1
            if 'p1' in points:
                annotation['annotations'][0]['result'].append({
                    'type': 'keypointlabels',
                    'value': {
                        'x': points['p1']['x'] * 100,  # Convert to percentage
                        'y': points['p1']['y'] * 100,
                        'width': 1.0,
                        'keypointlabels': ['p1']
                    }
                })
            
            # Add p2
            if 'p2' in points:
                annotation['annotations'][0]['result'].append({
                    'type': 'keypointlabels',
                    'value': {
                        'x': points['p2']['x'] * 100,
                        'y': points['p2']['y'] * 100,
                        'width': 1.0,
                        'keypointlabels': ['p2']
                    }
                })
            
            existing_data.append(annotation)
        
        # Save to file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(existing_data)} total annotations to {self.output_file}")
        print(f"  (Existing: {len(saved_filenames)}, New: {len(existing_data) - len(saved_filenames)})")
    
    def get_current_annotation(self):
        """Get annotation for current image"""
        image_path = str(self.image_files[self.current_idx])
        return self.annotations.get(image_path, {})
    
    def set_current_annotation(self, points):
        """Set annotation for current image"""
        image_path = str(self.image_files[self.current_idx])
        self.annotations[image_path] = points
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert screen coordinates to image coordinates
            img_x = int((x - self.pan_x) / self.zoom_factor)
            img_y = int((y - self.pan_y) / self.zoom_factor)
            
            # Clamp to image bounds
            img_x = max(0, min(img_x, self.current_img.shape[1] - 1))
            img_y = max(0, min(img_y, self.current_img.shape[0] - 1))
            
            # Normalized coordinates [0, 1]
            norm_x = img_x / self.current_img.shape[1]
            norm_y = img_y / self.current_img.shape[0]
            
            # First click is p2 (top), second is p1 (bottom)
            if self.p2 is None:
                self.p2 = {'x': norm_x, 'y': norm_y}
                print(f"p2 set at ({img_x}, {img_y})")
            elif self.p1 is None:
                self.p1 = {'x': norm_x, 'y': norm_y}
                print(f"p1 set at ({img_x}, {img_y})")
                
                # Auto-save when both points are set
                self.set_current_annotation({'p1': self.p1, 'p2': self.p2})
                print("Both points annotated! Press 'n' for next image or 'r' to reset.")
    
    def draw_image(self):
        """Draw image with annotations"""
        # Copy original image
        img = self.current_img.copy()
        h, w = img.shape[:2]
        
        # Draw existing annotation (if any)
        existing = self.get_current_annotation()
        if 'p1' in existing:
            p1_x = int(existing['p1']['x'] * w)
            p1_y = int(existing['p1']['y'] * h)
            cv2.circle(img, (p1_x, p1_y), 8, self.COLOR_P1, -1)
            cv2.circle(img, (p1_x, p1_y), 10, self.COLOR_P1, 2)
            cv2.putText(img, 'p1', (p1_x + 15, p1_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, self.COLOR_P1, 2)
        
        if 'p2' in existing:
            p2_x = int(existing['p2']['x'] * w)
            p2_y = int(existing['p2']['y'] * h)
            cv2.circle(img, (p2_x, p2_y), 8, self.COLOR_P2, -1)
            cv2.circle(img, (p2_x, p2_y), 10, self.COLOR_P2, 2)
            cv2.putText(img, 'p2', (p2_x + 15, p2_y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, self.COLOR_P2, 2)
        
        # Draw current annotation (temporary)
        if self.p2 is not None:
            p2_x = int(self.p2['x'] * w)
            p2_y = int(self.p2['y'] * h)
            cv2.circle(img, (p2_x, p2_y), 8, self.COLOR_TEMP, -1)
            cv2.circle(img, (p2_x, p2_y), 10, self.COLOR_TEMP, 2)
            cv2.putText(img, 'p2 (temp)', (p2_x + 15, p2_y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, self.COLOR_TEMP, 2)
        
        if self.p1 is not None:
            p1_x = int(self.p1['x'] * w)
            p1_y = int(self.p1['y'] * h)
            cv2.circle(img, (p1_x, p1_y), 8, self.COLOR_TEMP, -1)
            cv2.circle(img, (p1_x, p1_y), 10, self.COLOR_TEMP, 2)
            cv2.putText(img, 'p1 (temp)', (p1_x + 15, p1_y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, self.COLOR_TEMP, 2)
        
        # Apply zoom and pan
        if self.zoom_factor != 1.0 or self.pan_x != 0 or self.pan_y != 0:
            # Create transform matrix
            M = np.float32([
                [self.zoom_factor, 0, self.pan_x],
                [0, self.zoom_factor, self.pan_y]
            ])
            img = cv2.warpAffine(img, M, (w, h))
        
        # Draw info text
        filename = self.image_files[self.current_idx].name
        progress = f"{self.current_idx + 1}/{len(self.image_files)}"
        status = "Complete" if 'p1' in existing and 'p2' in existing else "Incomplete"
        
        cv2.putText(img, f"{filename} [{progress}] - {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"{filename} [{progress}] - {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Draw instructions
        instructions = [
            "Click: p2 (top) then p1 (bottom)",
            "n: Next | p: Previous | r: Reset",
            "s: Save | q: Quit | +/-: Zoom"
        ]
        
        y_offset = h - 70
        for i, text in enumerate(instructions):
            cv2.putText(img, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img
    
    def load_image(self):
        """Load current image"""
        image_path = self.image_files[self.current_idx]
        self.current_img = cv2.imread(str(image_path))
        
        if self.current_img is None:
            print(f"Error loading image: {image_path}")
            return False
        
        # Reset temporary annotations
        self.p1 = None
        self.p2 = None
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        return True
    
    def next_image(self):
        """Go to next image"""
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self.load_image()
    
    def prev_image(self):
        """Go to previous image"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()
    
    def reset_annotation(self):
        """Reset current annotation"""
        self.p1 = None
        self.p2 = None
        
        # Also clear saved annotation
        image_path = str(self.image_files[self.current_idx])
        if image_path in self.annotations:
            del self.annotations[image_path]
        
        print("Annotation reset")
    
    def run(self):
        """Run the annotator"""
        if not self.image_files:
            print("\nNo images to annotate. All images are already annotated or no images found.")
            return
        
        if not self.load_image():
            return
        
        cv2.namedWindow('P1/P2 Annotator', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('P1/P2 Annotator', 1400, 900)
        cv2.setMouseCallback('P1/P2 Annotator', self.mouse_callback)
        
        print("\n" + "="*60)
        print("P1/P2 Calibration Point Annotator")
        print("="*60)
        print("\nInstructions:")
        print("  1. Click on p2 (TOP point on ruler, 1cm mark)")
        print("  2. Click on p1 (BOTTOM point on ruler, 0cm mark)")
        print("  3. Press 'n' for next image")
        print("\nControls:")
        print("  n: Next image")
        print("  p: Previous image")
        print("  r: Reset current annotation")
        print("  s: Save all annotations")
        print("  +: Zoom in")
        print("  -: Zoom out")
        print("  q: Quit and save")
        print("="*60 + "\n")
        
        while True:
            # Draw and show
            display_img = self.draw_image()
            cv2.imshow('P1/P2 Annotator', display_img)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                self.save_annotations()
                break
            
            elif key == ord('n'):
                # Next image
                self.next_image()
            
            elif key == ord('p'):
                # Previous image
                self.prev_image()
            
            elif key == ord('r'):
                # Reset
                self.reset_annotation()
            
            elif key == ord('s'):
                # Save
                self.save_annotations()
            
            elif key == ord('+') or key == ord('='):
                # Zoom in
                self.zoom_factor = min(3.0, self.zoom_factor + 0.1)
                print(f"Zoom: {self.zoom_factor:.1f}x")
            
            elif key == ord('-') or key == ord('_'):
                # Zoom out
                self.zoom_factor = max(0.5, self.zoom_factor - 0.1)
                print(f"Zoom: {self.zoom_factor:.1f}x")
        
        cv2.destroyAllWindows()
        print("\nAnnotation session ended.")
        
        # Print summary
        total = len(self.image_files)
        annotated = len(self.annotations)
        print(f"\nSummary:")
        print(f"  Total images: {total}")
        print(f"  Annotated: {annotated} ({annotated/total*100:.1f}%)")
        print(f"  Remaining: {total - annotated}")


def main():
    parser = argparse.ArgumentParser(description='P1/P2 Calibration Point Annotator')
    parser.add_argument('images_dir', nargs='?', default='Aariz/train/Cephalograms',
                       help='Directory containing images to annotate (default: Aariz/train/Cephalograms)')
    parser.add_argument('-o', '--output', default='annotations_p1_p2.json',
                       help='Output JSON file (default: annotations_p1_p2.json)')
    parser.add_argument('-n', '--max-images', type=int, default=100,
                       help='Maximum number of images to annotate (default: 100)')
    
    args = parser.parse_args()
    
    try:
        annotator = P1P2Annotator(args.images_dir, args.output, args.max_images)
        annotator.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

