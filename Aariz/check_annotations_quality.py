"""
Check quality of p1/p2 annotations
Validates annotations and reports potential issues
"""
import json
import numpy as np
from pathlib import Path
import argparse


def check_annotation_quality(annotations_file, images_dir=None):
    """Check quality of annotations"""
    
    print("="*60)
    print("P1/P2 Annotation Quality Checker")
    print("="*60)
    print()
    
    # Load annotations
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded annotations from: {annotations_file}")
    print(f"Total entries: {len(data)}")
    print()
    
    # Statistics
    stats = {
        'total': len(data),
        'complete': 0,
        'incomplete': 0,
        'distances': [],
        'dx_values': [],
        'dy_values': [],
        'suspicious': []
    }
    
    # Check each annotation
    for idx, item in enumerate(data):
        if 'annotations' not in item or not item['annotations']:
            continue
        
        annotation = item['annotations'][0]
        if 'result' not in annotation:
            continue
        
        # Extract p1 and p2
        p1 = None
        p2 = None
        
        for result_item in annotation['result']:
            if result_item.get('type') == 'keypointlabels':
                value = result_item.get('value', {})
                labels = value.get('keypointlabels', [])
                
                if 'p1' in labels:
                    p1 = {'x': value.get('x', 0), 'y': value.get('y', 0)}
                elif 'p2' in labels:
                    p2 = {'x': value.get('x', 0), 'y': value.get('y', 0)}
        
        # Check completeness
        if p1 and p2:
            stats['complete'] += 1
            
            # Calculate distance (in percentage units)
            dx = abs(p2['x'] - p1['x'])
            dy = abs(p2['y'] - p1['y'])
            distance = np.sqrt(dx**2 + dy**2)
            
            stats['distances'].append(distance)
            stats['dx_values'].append(dx)
            stats['dy_values'].append(dy)
            
            # Check for suspicious annotations
            # Vertical distance should be ~5-20% of image height
            # Horizontal distance should be < 5% (nearly vertical)
            
            issues = []
            
            if dy < 2.0:
                issues.append("too_close")
            elif dy > 25.0:
                issues.append("too_far")
            
            if dx > 5.0:
                issues.append("not_vertical")
            
            # Check if in top-right area (typical location)
            if p1['x'] < 60 or p2['x'] < 60:
                issues.append("not_in_top_right")
            
            if p1['y'] > 30 or p2['y'] > 30:
                issues.append("too_low")
            
            if issues:
                filename = item['data'].get('image', '').split('/')[-1]
                stats['suspicious'].append({
                    'index': idx + 1,
                    'filename': filename,
                    'issues': issues,
                    'p1': p1,
                    'p2': p2,
                    'dx': dx,
                    'dy': dy,
                    'distance': distance
                })
        else:
            stats['incomplete'] += 1
    
    # Print report
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print()
    print(f"Total annotations: {stats['total']}")
    print(f"Complete (both p1 and p2): {stats['complete']}")
    print(f"Incomplete: {stats['incomplete']}")
    print(f"Completion rate: {stats['complete']/stats['total']*100:.1f}%")
    print()
    
    if stats['distances']:
        print("="*60)
        print("DISTANCE STATISTICS")
        print("="*60)
        print()
        print(f"Average distance: {np.mean(stats['distances']):.2f}%")
        print(f"Median distance: {np.median(stats['distances']):.2f}%")
        print(f"Std deviation: {np.std(stats['distances']):.2f}%")
        print(f"Min distance: {np.min(stats['distances']):.2f}%")
        print(f"Max distance: {np.max(stats['distances']):.2f}%")
        print()
        print(f"Average dx (horizontal): {np.mean(stats['dx_values']):.2f}%")
        print(f"Average dy (vertical): {np.mean(stats['dy_values']):.2f}%")
        print()
        
        # Quality assessment
        avg_dy = np.mean(stats['dy_values'])
        avg_dx = np.mean(stats['dx_values'])
        
        print("="*60)
        print("QUALITY ASSESSMENT")
        print("="*60)
        print()
        
        # Vertical alignment
        if avg_dx < 2.0:
            print("Vertical alignment: EXCELLENT (dx < 2%)")
        elif avg_dx < 5.0:
            print("Vertical alignment: GOOD (dx < 5%)")
        elif avg_dx < 10.0:
            print("Vertical alignment: FAIR (dx < 10%)")
        else:
            print("Vertical alignment: POOR (dx >= 10%)")
        
        # Distance consistency
        if avg_dy >= 4.0 and avg_dy <= 15.0:
            print("Distance consistency: GOOD (4-15% vertical)")
        elif avg_dy >= 2.0 and avg_dy <= 20.0:
            print("Distance consistency: ACCEPTABLE (2-20% vertical)")
        else:
            print(f"Distance consistency: CHECK NEEDED (avg dy={avg_dy:.1f}%)")
        
        print()
    
    # Print suspicious annotations
    if stats['suspicious']:
        print("="*60)
        print(f"SUSPICIOUS ANNOTATIONS ({len(stats['suspicious'])})")
        print("="*60)
        print()
        
        for item in stats['suspicious']:
            print(f"[{item['index']}] {item['filename']}")
            print(f"    Issues: {', '.join(item['issues'])}")
            print(f"    p1: ({item['p1']['x']:.1f}, {item['p1']['y']:.1f})")
            print(f"    p2: ({item['p2']['x']:.1f}, {item['p2']['y']:.1f})")
            print(f"    dx: {item['dx']:.2f}%, dy: {item['dy']:.2f}%")
            print(f"    distance: {item['distance']:.2f}%")
            print()
    else:
        print("="*60)
        print("NO SUSPICIOUS ANNOTATIONS")
        print("="*60)
        print()
        print("All annotations look good!")
        print()
    
    # Recommendations
    print("="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print()
    
    if stats['incomplete'] > 0:
        print(f"- Complete {stats['incomplete']} remaining annotations")
    
    if len(stats['suspicious']) > 0:
        print(f"- Review {len(stats['suspicious'])} suspicious annotations")
        print("  Use: python p1_p2_annotator.py <images_dir> -o <this_file>")
        print("  Navigate to suspicious images and re-annotate (press 'r' then click)")
    
    if stats['complete'] < 50:
        print(f"- Current dataset size: {stats['complete']} samples")
        print("  Recommended: 50+ samples for decent accuracy")
        print("  Ideal: 100+ samples for best results")
    
    avg_dx = np.mean(stats['dx_values']) if stats['dx_values'] else 0
    if avg_dx > 5.0:
        print(f"- Average horizontal offset is high ({avg_dx:.1f}%)")
        print("  Ensure p1 and p2 are vertically aligned on the ruler")
    
    print()
    
    # Save report
    report_file = annotations_file.replace('.json', '_quality_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"P1/P2 Annotation Quality Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Annotations file: {annotations_file}\n")
        f.write(f"\n")
        f.write(f"Total: {stats['total']}\n")
        f.write(f"Complete: {stats['complete']}\n")
        f.write(f"Incomplete: {stats['incomplete']}\n")
        f.write(f"Suspicious: {len(stats['suspicious'])}\n")
        
        if stats['distances']:
            f.write(f"\nAverage distance: {np.mean(stats['distances']):.2f}%\n")
            f.write(f"Average dx: {np.mean(stats['dx_values']):.2f}%\n")
            f.write(f"Average dy: {np.mean(stats['dy_values']):.2f}%\n")
    
    print(f"Report saved to: {report_file}")
    print()
    
    return stats


def main():
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Check p1/p2 annotation quality')
    parser.add_argument('annotations_file', help='JSON file with annotations')
    parser.add_argument('--images_dir', help='Images directory (optional)')
    
    args = parser.parse_args()
    
    if not Path(args.annotations_file).exists():
        print(f"Error: File not found: {args.annotations_file}")
        return
    
    check_annotation_quality(args.annotations_file, args.images_dir)


if __name__ == '__main__':
    main()













