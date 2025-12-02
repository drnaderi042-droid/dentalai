"""
Benchmarking script for comparing different optimization strategies
"""
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import argparse
import torch
import time
import numpy as np
from tqdm import tqdm
import json

from dataset import create_dataloaders
from model import get_model
from utils_optimized import (
    benchmark_context,
    get_gpu_memory_stats,
    print_gpu_memory,
    count_parameters
)


def benchmark_model(model, dataloader, device, num_iterations=100, mixed_precision=False):
    """Benchmark model inference speed"""
    model.eval()
    
    times = []
    with torch.no_grad():
        # Warmup
        for i, batch in enumerate(dataloader):
            if i >= 10:
                break
            images = batch['image'].to(device, non_blocking=True)
            _ = model(images)
        
        # Actual benchmark
        pbar = tqdm(enumerate(dataloader), total=min(num_iterations, len(dataloader)), desc="Benchmarking")
        for i, batch in pbar:
            if i >= num_iterations:
                break
            
            images = batch['image'].to(device, non_blocking=True)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            
            if mixed_precision:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    _ = model(images)
            else:
                _ = model(images)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()
            
            times.append(end - start)
            
            avg_time = np.mean(times) * 1000  # ms
            pbar.set_postfix({'avg_time': f'{avg_time:.2f}ms'})
    
    return times


def benchmark_training_step(model, dataloader, optimizer, criterion, device, 
                           num_iterations=100, mixed_precision=False, 
                           gradient_accumulation_steps=1):
    """Benchmark training step speed"""
    model.train()
    
    if mixed_precision:
        scaler = torch.amp.GradScaler('cuda')
    
    times = []
    forward_times = []
    backward_times = []
    
    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['target'].to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        if outputs.shape[2:] != targets.shape[2:]:
            outputs = torch.nn.functional.interpolate(
                outputs, size=targets.shape[2:], mode='bilinear', align_corners=False
            )
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Actual benchmark
    pbar = tqdm(enumerate(dataloader), total=min(num_iterations, len(dataloader)), desc="Training Benchmark")
    optimizer.zero_grad()
    
    for i, batch in pbar:
        if i >= num_iterations:
            break
        
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['target'].to(device, non_blocking=True)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        # Forward
        forward_start = time.time()
        if mixed_precision:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                if outputs.shape[2:] != targets.shape[2:]:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=targets.shape[2:], mode='bilinear', align_corners=False
                    )
                loss = criterion(outputs, targets)
                loss = loss / gradient_accumulation_steps
        else:
            outputs = model(images)
            if outputs.shape[2:] != targets.shape[2:]:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=targets.shape[2:], mode='bilinear', align_corners=False
                )
            loss = criterion(outputs, targets)
            loss = loss / gradient_accumulation_steps
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        forward_end = time.time()
        forward_times.append(forward_end - forward_start)
        
        # Backward
        backward_start = time.time()
        if mixed_precision:
            scaler.scale(loss).backward()
            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        backward_end = time.time()
        backward_times.append(backward_end - backward_start)
        
        end = time.time()
        times.append(end - start)
        
        avg_time = np.mean(times) * 1000
        pbar.set_postfix({'avg_time': f'{avg_time:.2f}ms'})
    
    return {
        'total_times': times,
        'forward_times': forward_times,
        'backward_times': backward_times
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark Cephalometric Landmark Detection Models')
    parser.add_argument('--dataset_path', type=str, default='Aariz')
    parser.add_argument('--model', type=str, default='hrnet', choices=['resnet', 'unet', 'hourglass', 'hrnet'])
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_iterations', type=int, default=100)
    parser.add_argument('--mode', type=str, default='both', choices=['inference', 'training', 'both'])
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--compile_model', action='store_true')
    parser.add_argument('--output', type=str, default='benchmark_results.json')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print("-" * 80)
    
    # Load data
    print("Loading dataset...")
    train_loader, val_loader, _ = create_dataloaders(
        dataset_folder_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
        use_heatmap=True
    )
    
    # Create model
    print(f"Creating {args.model} model...")
    model = get_model(args.model, num_landmarks=29)
    model = model.to(device)
    
    # Count parameters
    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Model size: {params['total'] * 4 / 1024**2:.2f} MB (FP32)")
    
    # Compile if requested
    if args.compile_model and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)
    
    # GPU memory before
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print_gpu_memory()
    
    results = {
        'model': args.model,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'mixed_precision': args.mixed_precision,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'parameters': params,
    }
    
    # Inference benchmark
    if args.mode in ['inference', 'both']:
        print("\n" + "="*80)
        print("INFERENCE BENCHMARK")
        print("="*80)
        
        inference_times = benchmark_model(
            model, val_loader, device, 
            num_iterations=args.num_iterations,
            mixed_precision=args.mixed_precision
        )
        
        avg_time = np.mean(inference_times) * 1000
        std_time = np.std(inference_times) * 1000
        throughput = args.batch_size / np.mean(inference_times)
        
        print(f"\nInference Results:")
        print(f"  Average time per batch: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  Time per image: {avg_time / args.batch_size:.2f} ms")
        
        results['inference'] = {
            'avg_time_ms': float(avg_time),
            'std_time_ms': float(std_time),
            'throughput_samples_per_sec': float(throughput),
            'time_per_image_ms': float(avg_time / args.batch_size)
        }
    
    # Training benchmark
    if args.mode in ['training', 'both']:
        print("\n" + "="*80)
        print("TRAINING BENCHMARK")
        print("="*80)
        
        # Create optimizer and criterion
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        criterion = torch.nn.MSELoss()
        
        training_results = benchmark_training_step(
            model, train_loader, optimizer, criterion, device,
            num_iterations=args.num_iterations,
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        
        total_times = training_results['total_times']
        forward_times = training_results['forward_times']
        backward_times = training_results['backward_times']
        
        avg_total = np.mean(total_times) * 1000
        avg_forward = np.mean(forward_times) * 1000
        avg_backward = np.mean(backward_times) * 1000
        throughput = args.batch_size / np.mean(total_times)
        
        print(f"\nTraining Results:")
        print(f"  Average time per step: {avg_total:.2f} ms")
        print(f"  Forward time: {avg_forward:.2f} ms ({avg_forward/avg_total*100:.1f}%)")
        print(f"  Backward time: {avg_backward:.2f} ms ({avg_backward/avg_total*100:.1f}%)")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
        
        # Estimate epoch time
        steps_per_epoch = len(train_loader)
        epoch_time = steps_per_epoch * np.mean(total_times) / 60  # minutes
        total_training_time = epoch_time * 250 / 60  # hours for 250 epochs
        
        print(f"\nEstimated Training Time:")
        print(f"  Time per epoch: {epoch_time:.1f} minutes")
        print(f"  Total time (250 epochs): {total_training_time:.1f} hours")
        
        results['training'] = {
            'avg_time_ms': float(avg_total),
            'forward_time_ms': float(avg_forward),
            'backward_time_ms': float(avg_backward),
            'throughput_samples_per_sec': float(throughput),
            'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
            'estimated_epoch_time_min': float(epoch_time),
            'estimated_total_time_hours': float(total_training_time)
        }
    
    # GPU memory after
    if torch.cuda.is_available():
        print("\n" + "="*80)
        print("GPU MEMORY USAGE")
        print("="*80)
        mem_stats = get_gpu_memory_stats()
        print(f"  Allocated: {mem_stats['allocated']:.2f} GB")
        print(f"  Reserved: {mem_stats['reserved']:.2f} GB")
        print(f"  Max Allocated: {mem_stats['max_allocated']:.2f} GB")
        
        results['gpu_memory'] = {
            'allocated_gb': float(mem_stats['allocated']),
            'reserved_gb': float(mem_stats['reserved']),
            'max_allocated_gb': float(mem_stats['max_allocated'])
        }
    
    # Save results
    print(f"\nSaving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()