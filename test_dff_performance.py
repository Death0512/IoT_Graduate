#!/usr/bin/env python3
"""
DFF Performance Test Script

Test Deep Feature Flow optimization với different intervals để tìm best config.
"""

import subprocess
import time
import sys
import os

# Test configurations
CONFIGS = [
    {"interval": 0, "name": "Baseline (No DFF)"},
    {"interval": 5, "name": "DFF Interval=5 (High Accuracy)"},
    {"interval": 10, "name": "DFF Interval=10 (RECOMMENDED)"},
    {"interval": 15, "name": "DFF Interval=15 (Max Speed)"},
]

# Test video (change to your video)
TEST_VIDEO = "videodemo/sample.mp4"
TEST_MODE = "display"  # or "file"
TEST_DURATION = 60  # seconds

def modify_settings(interval):
    """Temporarily modify settings.py to enable DFF with specific interval."""
    settings_path = "speedflow/settings.py"
    
    # Read current settings
    with open(settings_path, 'r') as f:
        lines = f.readlines()
    
    # Modify DFF settings
    new_lines = []
    for line in lines:
        if line.startswith("ENABLE_DFF"):
            new_lines.append(f"ENABLE_DFF = {interval > 0}\n")
        elif line.startswith("DFF_INTERVAL"):
            new_lines.append(f"DFF_INTERVAL = {interval if interval > 0 else 10}\n")
        else:
            new_lines.append(line)
    
    # Write back
    with open(settings_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"[Config] Updated settings.py: ENABLE_DFF={interval > 0}, DFF_INTERVAL={interval}")

def run_test(config):
    """Run pipeline with specific config and measure performance."""
    interval = config["interval"]
    name = config["name"]
    
    print("\n" + "="*70)
    print(f"Testing: {name}")
    print("="*70)
    
    # Modify settings
    modify_settings(interval)
    
    # Build command
    cmd = [
        "python3", "main.py",
        "--source", TEST_VIDEO,
        "--mode", TEST_MODE
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Duration: {TEST_DURATION} seconds")
    print("Starting in 3 seconds...")
    time.sleep(3)
    
    # Run pipeline
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Collect output for duration
        start_time = time.time()
        output_lines = []
        
        while time.time() - start_time < TEST_DURATION:
            line = proc.stdout.readline()
            if line:
                output_lines.append(line)
                # Print FPS info
                if "FPS" in line or "fps" in line.lower():
                    print(line.strip())
            
            # Check if process ended
            if proc.poll() is not None:
                break
        
        # Terminate if still running
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
        
        # Parse results
        fps_values = []
        for line in output_lines:
            # Extract FPS (example: "FPS: 123.45")
            if "fps" in line.lower():
                try:
                    # Simple parsing (adjust based on actual output format)
                    fps_str = line.split("fps")[0].strip().split()[-1]
                    fps = float(fps_str)
                    fps_values.append(fps)
                except:
                    pass
        
        # Calculate average FPS
        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
        
        print("\n" + "-"*70)
        print(f"Results for {name}:")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Samples: {len(fps_values)}")
        print("-"*70)
        
        return {
            "config": name,
            "interval": interval,
            "avg_fps": avg_fps,
            "samples": len(fps_values)
        }
        
    except KeyboardInterrupt:
        print("\n[Interrupted] Test stopped by user")
        proc.terminate()
        return None
    except Exception as e:
        print(f"[Error] Test failed: {e}")
        return None

def main():
    """Run all tests and compare results."""
    
    print("="*70)
    print("DFF PERFORMANCE BENCHMARK")
    print("="*70)
    print(f"Test video: {TEST_VIDEO}")
    print(f"Test mode: {TEST_MODE}")
    print(f"Duration per test: {TEST_DURATION}s")
    print(f"Total tests: {len(CONFIGS)}")
    print("="*70)
    
    # Check video exists
    if not os.path.exists(TEST_VIDEO):
        print(f"\n[Error] Test video not found: {TEST_VIDEO}")
        print("Please update TEST_VIDEO in this script.")
        sys.exit(1)
    
    input("\nPress ENTER to start tests (or Ctrl+C to cancel)...")
    
    # Run tests
    results = []
    for i, config in enumerate(CONFIGS, 1):
        print(f"\n\n{'='*70}")
        print(f"Test {i}/{len(CONFIGS)}")
        print('='*70)
        
        result = run_test(config)
        if result:
            results.append(result)
        
        # Wait between tests
        if i < len(CONFIGS):
            print("\nWaiting 5 seconds before next test...")
            time.sleep(5)
    
    # Print summary
    print("\n\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"{'Config':<30} {'Interval':<10} {'Avg FPS':<10} {'Speedup':<10}")
    print("-"*70)
    
    baseline_fps = None
    for r in results:
        fps = r['avg_fps']
        interval = r['interval']
        
        if interval == 0:
            baseline_fps = fps
            speedup = "1.00x"
        else:
            speedup = f"{fps / baseline_fps:.2f}x" if baseline_fps else "N/A"
        
        print(f"{r['config']:<30} {interval:<10} {fps:<10.2f} {speedup:<10}")
    
    print("="*70)
    
    # Recommendation
    print("\nRECOMMENDATION:")
    if results:
        # Find best interval (balance speed and not too high)
        best = max([r for r in results if r['interval'] <= 10], 
                   key=lambda x: x['avg_fps'], 
                   default=results[0])
        print(f"  → Use interval={best['interval']} ({best['config']})")
        print(f"  → Expected FPS: {best['avg_fps']:.1f}")
    
    print("\nTo use this configuration:")
    print("1. Edit speedflow/settings.py:")
    print(f"   ENABLE_DFF = True")
    print(f"   DFF_INTERVAL = {best['interval']}")
    print("2. Run normally: python3 main.py --source <video> --mode <mode>")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Benchmark cancelled by user")
        sys.exit(0)
