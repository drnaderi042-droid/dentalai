"""
Analyze comprehensive test results for Aariz models
"""

import json
import os

# Load results
with open('aariz_512_768_comprehensive_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']
best_per_landmark = data['best_per_landmark']

# Overall comparison
print("="*100)
print("COMPREHENSIVE COMPARISON RESULTS - AARIZ MODELS (512 & 768)")
print("="*100)
print(f"Test Date: {data['timestamp']}")
print(f"Test Images: {data['num_test_images']} (Full Validation Set)")
print(f"Pixel Size: {data['pixel_size']} mm/pixel")
print("="*100)

print("\n" + "="*100)
print("OVERALL METRICS COMPARISON")
print("="*100)
print(f"{'Model':<25} {'MRE (mm)':<12} {'Std (mm)':<12} {'SDR @ 2mm':<12} {'SDR @ 2.5mm':<12} {'SDR @ 3mm':<12} {'SDR @ 4mm':<12}")
print("-" * 100)

for config_name in sorted(results.keys()):
    result = results[config_name]
    metrics = result['metrics']
    model_label = f"{result['model_size']} {'TTA' if result['use_tta'] else 'No-TTA'}"
    print(f"{model_label:<25} {metrics['mre_mm']:<12.3f} {metrics['std_mm']:<12.3f} "
          f"{metrics['sdr']['2.0']:<12.2f}% {metrics['sdr']['2.5']:<12.2f}% "
          f"{metrics['sdr']['3.0']:<12.2f}% {metrics['sdr']['4.0']:<12.2f}%")

# Best overall model
best_overall = min(results.items(), key=lambda x: x[1]['metrics']['mre_mm'])
print("\n" + "="*100)
print("BEST OVERALL MODEL")
print("="*100)
print(f"Model: {best_overall[1]['model_size']} {'TTA' if best_overall[1]['use_tta'] else 'No-TTA'}")
print(f"MRE: {best_overall[1]['metrics']['mre_mm']:.3f} mm")
print(f"Std: {best_overall[1]['metrics']['std_mm']:.3f} mm")
print(f"SDR @ 2mm: {best_overall[1]['metrics']['sdr']['2.0']:.2f}%")
print(f"SDR @ 3mm: {best_overall[1]['metrics']['sdr']['3.0']:.2f}%")

# TTA vs No-TTA comparison
print("\n" + "="*100)
print("TTA vs NO-TTA COMPARISON")
print("="*100)

for model_size in ['512', '768', 'ensemble']:
    no_tta_key = f"{model_size}_no_tta"
    tta_key = f"{model_size}_tta"
    
    if no_tta_key in results and tta_key in results:
        no_tta_metrics = results[no_tta_key]['metrics']
        tta_metrics = results[tta_key]['metrics']
        
        mre_improvement = ((no_tta_metrics['mre_mm'] - tta_metrics['mre_mm']) / no_tta_metrics['mre_mm']) * 100
        sdr_improvement = tta_metrics['sdr']['2.0'] - no_tta_metrics['sdr']['2.0']
        
        print(f"\n{model_size.upper()}:")
        print(f"   MRE: {no_tta_metrics['mre_mm']:.3f} -> {tta_metrics['mre_mm']:.3f} mm ({mre_improvement:+.2f}%)")
        print(f"   SDR @ 2mm: {no_tta_metrics['sdr']['2.0']:.2f}% -> {tta_metrics['sdr']['2.0']:.2f}% ({sdr_improvement:+.2f}%)")

# Per-landmark best model analysis
print("\n" + "="*100)
print("BEST MODEL PER LANDMARK")
print("="*100)
print(f"{'Landmark':<10} {'Best Model':<25} {'Mean Error (mm)':<18} {'SDR @ 2mm':<12} {'Std (mm)':<12}")
print("-" * 100)

# Count which model is best for how many landmarks
model_counts = {}
for symbol, best in best_per_landmark.items():
    model = best['model']
    if model not in model_counts:
        model_counts[model] = 0
    model_counts[model] += 1
    print(f"{symbol:<10} {model:<25} {best['mean_error']:<18.3f} "
          f"{best['metrics']['sdr_2mm']:<12.2f}% {best['metrics']['std']:<12.3f}")

print("\n" + "="*100)
print("MODEL PERFORMANCE SUMMARY")
print("="*100)
for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{model:<30} Best for {count:>2} landmarks")

# Difficult landmarks analysis
difficult_landmarks = ['UMT', 'UPM', 'R', 'Ar', 'Go', 'LMT', 'LPM', 'Or', 'Co', 'PNS', 'Po', 'ANS']
print("\n" + "="*100)
print("DIFFICULT LANDMARKS ANALYSIS")
print("="*100)
print(f"{'Landmark':<10} {'Best Model':<25} {'Mean Error (mm)':<18} {'SDR @ 2mm':<12}")
print("-" * 100)

difficult_model_counts = {}
for symbol in difficult_landmarks:
    if symbol in best_per_landmark:
        best = best_per_landmark[symbol]
        model = best['model']
        if model not in difficult_model_counts:
            difficult_model_counts[model] = 0
        difficult_model_counts[model] += 1
        print(f"{symbol:<10} {model:<25} {best['mean_error']:<18.3f} {best['metrics']['sdr_2mm']:<12.2f}%")

print("\nDifficult Landmarks - Model Distribution:")
for model, count in sorted(difficult_model_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{model:<30} Best for {count:>2} difficult landmarks")

# Worst performing landmarks
print("\n" + "="*100)
print("WORST PERFORMING LANDMARKS (Across All Models)")
print("="*100)

worst_landmarks = []
for symbol, best in best_per_landmark.items():
    worst_landmarks.append((symbol, best['mean_error'], best['model']))

worst_landmarks.sort(key=lambda x: x[1], reverse=True)

print(f"{'Rank':<6} {'Landmark':<10} {'Best Model':<25} {'Mean Error (mm)':<18} {'SDR @ 2mm':<12}")
print("-" * 100)
for rank, (symbol, error, model) in enumerate(worst_landmarks[:10], 1):
    sdr = best_per_landmark[symbol]['metrics']['sdr_2mm']
    print(f"{rank:<6} {symbol:<10} {model:<25} {error:<18.3f} {sdr:<12.2f}%")

# Best performing landmarks
print("\n" + "="*100)
print("BEST PERFORMING LANDMARKS (Across All Models)")
print("="*100)

best_landmarks = sorted(worst_landmarks, key=lambda x: x[1])

print(f"{'Rank':<6} {'Landmark':<10} {'Best Model':<25} {'Mean Error (mm)':<18} {'SDR @ 2mm':<12}")
print("-" * 100)
for rank, (symbol, error, model) in enumerate(best_landmarks[:10], 1):
    sdr = best_per_landmark[symbol]['metrics']['sdr_2mm']
    print(f"{rank:<6} {symbol:<10} {model:<25} {error:<18.3f} {sdr:<12.2f}%")

print("\n" + "="*100)
print("KEY FINDINGS")
print("="*100)
print(f"1. Best Overall Model: {best_overall[1]['model_size']} {'TTA' if best_overall[1]['use_tta'] else 'No-TTA'}")
print(f"   - MRE: {best_overall[1]['metrics']['mre_mm']:.3f} mm")
print(f"   - SDR @ 2mm: {best_overall[1]['metrics']['sdr']['2.0']:.2f}%")
print(f"\n2. TTA Impact:")
tta_improvements = []
for model_size in ['512', '768', 'ensemble']:
    no_tta_key = f"{model_size}_no_tta"
    tta_key = f"{model_size}_tta"
    if no_tta_key in results and tta_key in results:
        mre_improvement = ((results[no_tta_key]['metrics']['mre_mm'] - results[tta_key]['metrics']['mre_mm']) / results[no_tta_key]['metrics']['mre_mm']) * 100
        tta_improvements.append(f"{model_size}: {mre_improvement:+.2f}%")
print(f"   - MRE Improvement: {', '.join(tta_improvements)}")
print(f"\n3. Ensemble Performance:")
if 'ensemble_tta' in results:
    ensemble = results['ensemble_tta']
    print(f"   - Ensemble TTA: MRE={ensemble['metrics']['mre_mm']:.3f}mm, SDR@2mm={ensemble['metrics']['sdr']['2.0']:.2f}%")
    print(f"   - Best for {model_counts.get('ensemble_tta', 0)} landmarks")
print(f"\n4. Most Problematic Landmarks:")
for rank, (symbol, error, model) in enumerate(worst_landmarks[:5], 1):
    print(f"   {rank}. {symbol}: {error:.3f}mm (best model: {model})")
print(f"\n5. Best Performing Landmarks:")
for rank, (symbol, error, model) in enumerate(best_landmarks[:5], 1):
    print(f"   {rank}. {symbol}: {error:.3f}mm (best model: {model})")

print("\n" + "="*100)

