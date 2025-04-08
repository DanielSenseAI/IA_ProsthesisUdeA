#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Grid Search for Window Feature Extraction

This script executes wf_extractor.py with different combinations of:
- Window sizes
- Envelope types
- Filter cutoff frequencies

This allows systematic exploration of parameter space to find optimal preprocessing.
"""

import os
import subprocess
import itertools
import time
from datetime import datetime
import argparse
import sys

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run grid search for window feature extraction')
    
    parser.add_argument('--database', type=str, default='DB4',
                        help='Database to process (default: DB4)')
    parser.add_argument('--window-sizes', type=str, default='200,400,800',
                        help='Comma-separated list of window sizes to try (default: 200,400,800)')
    parser.add_argument('--envelope-types', type=str, default='0,1,6',
                        help='Comma-separated list of envelope types to try (default: 0,1,6)')
    parser.add_argument('--filter-cutoffs', type=str, default='0.6,1,5',
                        help='Comma-separated list of filter cutoffs to try (default: 0.6,1,5)')
    parser.add_argument('--overlap-pct', type=int, default=0,
                        help='Overlap percentage between windows (default: 0)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration if available')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for GPU processing (default: 64)')
    parser.add_argument('--python-executable', type=str, default=sys.executable,
                        help='Path to Python executable to use (default: current Python)')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume from last completed combination')
    return parser.parse_args()

def main():
    """Run grid search over parameter combinations."""
    args = parse_args()
    
    # Parse parameter lists
    window_sizes = [int(x.strip()) for x in args.window_sizes.split(',')]
    envelope_types = [int(x.strip()) for x in args.envelope_types.split(',')]
    filter_cutoffs = [float(x.strip()) for x in args.filter_cutoffs.split(',')]
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join('logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create preprocessed_data directory if it doesn't exist
    output_base_dir = os.path.join('preprocessed_data')
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Create database-specific output directory if it doesn't exist
    output_dir = os.path.join(output_base_dir, args.database)
    os.makedirs(output_dir, exist_ok=True)

    # In main function
    completed_file = os.path.join(log_dir, "completed_combinations.txt")
    completed_combinations = set()
    if args.resume and os.path.exists(completed_file):
        with open(completed_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    completed_combinations.add((int(parts[0]), int(parts[1]), float(parts[2])))
        print(f"Resuming from {len(completed_combinations)} completed combinations")

        
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"grid_search_{timestamp}.log")
    
    with open(log_file, 'w') as f:
        f.write(f"Grid Search Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Database: {args.database}\n")
        f.write(f"Window Sizes: {window_sizes}\n")
        f.write(f"Envelope Types: {envelope_types}\n")
        f.write(f"Filter Cutoffs: {filter_cutoffs}\n")
        f.write(f"Overlap Percentage: {args.overlap_pct}%\n")
        f.write(f"Use GPU: {args.use_gpu}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Python Executable: {args.python_executable}\n\n")
        f.write("-" * 80 + "\n\n")
    
    # Count valid combinations (excluding envelope_type=0 with cutoffs other than 0.6)
    valid_combinations = 0
    for w, e, f in itertools.product(window_sizes, envelope_types, filter_cutoffs):
        if not (e == 0 and f != 0.6):
            valid_combinations += 1

    print(f"Running grid search with {valid_combinations} parameter combinations")

    print(f"Output log: {log_file}")
    
    # Generate all combinations
    combinations = list(itertools.product(window_sizes, envelope_types, filter_cutoffs))
    
    # Check if wf_extractor.py exists
    script_path = "wf_extractor.py"
    script_abs_path = os.path.abspath(script_path)
    
    if not os.path.exists(script_path):
        print(f"Error: Script '{script_path}' not found")
        print(f"Working directory: {os.getcwd()}")
        return
    else:
        print(f"Found script at: {script_abs_path}")
    
    # Run each combination
    successful_combinations = 0
    failed_combinations = 0
    
    for i, (window_size, envelope_type, filter_cutoff) in enumerate(combinations, 1):
        print(f"\nCombination {i}/{valid_combinations}:")
        print(f"  Window Size: {window_size}")
        print(f"  Envelope Type: {envelope_type}")
        print(f"  Filter Cutoff: {filter_cutoff}")

        # Then in your combination loop
        combination = (window_size, envelope_type, filter_cutoff)
        if combination in completed_combinations:
            print(f"  Skipping already completed combination: {combination}")
            continue

        if envelope_type == 0 and filter_cutoff != 0.6:
            print("  Skipping combination due to invalid parameters")
            continue
        
        # Calculate overlap in samples based on percentage
        overlap = int(window_size * args.overlap_pct / 100)
        
        # Build command
        cmd = [
            args.python_executable,  # Use specified Python executable
            script_abs_path,  # Use absolute path to script
            "--database", args.database,
            "--window-size", str(window_size),
            "--overlap", str(overlap),
            "--envelope-type", str(envelope_type),
            "--filter-cutoff", str(filter_cutoff)
        ]
        
        # Add GPU argument if requested
        if args.use_gpu:
            cmd.extend(["--use-gpu", "--batch-size", str(args.batch_size)])
        
        # Log command
        cmd_str = " ".join(cmd)
        print(f"Running: {cmd_str}")
        with open(log_file, 'a') as f:
            f.write(f"Combination {i}/{valid_combinations} - {datetime.now().strftime('%H:%M:%S')}\n")
            f.write(f"Command: {cmd_str}\n")
        
        # Execute command and capture output
        start_time = time.time()
        try:
            # Set shell=True on Windows to avoid some path issues
            use_shell = sys.platform.startswith('win')

            # Set environment variables to help with tqdm display
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'  # Force Python to not buffer output
                    
            # Use Popen to stream output in real-time while still capturing it
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                shell=use_shell if use_shell else False,
                env=env
            )
            
            # Capture output while displaying it
            stdout_lines = []
            stderr_lines = []
            
            # Function to handle output streams
            def read_stream(stream, store_list, prefix=''):
                for line in iter(stream.readline, ''):
                    # Don't add error prefix to tqdm progress bar lines
                    if prefix and ('it/s' in line or '%, ' in line) and any(s in line for s in ['|', '/', '-', '\\']):
                        # This is likely a tqdm progress bar - print without prefix
                        print(f"{line.rstrip()}")
                    else:
                        # Normal output - add prefix if specified
                        print(f"{prefix}{line.rstrip()}")
                    store_list.append(line)
                stream.close()
            
            # Create threads to read stdout and stderr
            import threading
            stdout_thread = threading.Thread(
                target=read_stream, 
                args=(process.stdout, stdout_lines, '')
            )
            stderr_thread = threading.Thread(
                target=read_stream, 
                args=(process.stderr, stderr_lines, 'ERROR: ')
            )
            
            # Start threads
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Wait for threads to finish reading output
            stdout_thread.join()
            stderr_thread.join()
            
            # Combine captured output
            stdout_output = ''.join(stdout_lines)
            stderr_output = ''.join(stderr_lines)
            
            # Record execution time
            execution_time = time.time() - start_time
            
            if return_code == 0:
                successful_combinations += 1
                status = "Success"
            else:
                failed_combinations += 1
                status = "Failed"
            
            with open(log_file, 'a') as f:
                f.write(f"Status: {status}\n")
                f.write(f"Execution Time: {execution_time:.2f} seconds\n")
                f.write(f"Return Code: {return_code}\n")
                
                if stdout_output:
                    f.write("\nSTDOUT:\n")
                    f.write(stdout_output)
                
                if stderr_output:
                    f.write("\nSTDERR:\n")
                    f.write(stderr_output)
                
                f.write("\n" + "-" * 80 + "\n\n")
            
            print(f"\n  Combination {i} completed in {execution_time:.2f} seconds")
            # After successful completion
            with open(completed_file, 'a') as f:
                f.write(f"{window_size},{envelope_type},{filter_cutoff}\n")
            print(f"  Status: {status}")
            
        except Exception as e:
            failed_combinations += 1
            with open(log_file, 'a') as f:
                f.write(f"Error: {str(e)}\n")
                f.write("\n" + "-" * 80 + "\n\n")
            print(f"  Error: {str(e)}")
    
    # Print summary
    print("\nGrid search complete!")
    print(f"Total combinations: {valid_combinations}")
    print(f"Successful: {successful_combinations}")
    print(f"Failed: {failed_combinations}")
    print(f"All results logged to {log_file}")
    
    # Print summary of processed files
    print("\nSummary of generated files:")
    if os.path.exists(output_dir):
        generated_files = [f for f in os.listdir(output_dir) if f.endswith('.parquet') or f.endswith('.csv')]
        if generated_files:
            for file in sorted(generated_files):
                file_path = os.path.join(output_dir, file)
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  {file} ({file_size_mb:.2f} MB)")
        else:
            print("  No output files found in the directory")
    else:
        print(f"  Output directory {output_dir} not found")


if __name__ == "__main__":
    main()