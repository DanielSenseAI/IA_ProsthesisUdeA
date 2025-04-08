#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Cleaner Utility

This script cleans up duplicate plot images in preprocessed_data/DB4/plots directory.
It groups images by parameter combinations (window size, envelope type, filter cutoff)
and keeps only one representative image per category.
"""

import os
import re
import argparse
from collections import defaultdict
import shutil

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Clean duplicate plot images')
    parser.add_argument('--database', type=str, default='DB4',
                        help='Database name (default: DB4)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without actually deleting')
    parser.add_argument('--keep-per-category', type=int, default=1,
                        help='Number of images to keep per category (default: 1)')
    parser.add_argument('--move', action='store_true',
                        help='Move files to a backup directory instead of deleting')
    return parser.parse_args()

def extract_parameters(filename):
    """
    Extract window size, envelope type, filter cutoff from filename.
    
    Example: DB4_win800_env1_f5_subjS1_w0_20250406-123456.png
    Returns: (800, 1, 5.0)
    """
    win_match = re.search(r'win(\d+)', filename)
    env_match = re.search(r'env(\d+)', filename)
    filter_match = re.search(r'f(\d+(?:\.\d+)?)', filename)
    
    if win_match and env_match and filter_match:
        win_size = int(win_match.group(1))
        env_type = int(env_match.group(1))
        filter_cutoff = float(filter_match.group(1))
        return (win_size, env_type, filter_cutoff)
    
    return None

def main():
    """Main function to clean duplicate plot images."""
    args = parse_args()
    
    # Path to the plots directory
    plots_dir = os.path.abspath(os.path.join('preprocessed_data', args.database, 'plots'))
    
    if not os.path.exists(plots_dir):
        print(f"Error: Directory not found: {plots_dir}")
        return
    
    print(f"Scanning directory: {plots_dir}")
    
    # Create backup directory if moving files
    if args.move:
        backup_dir = os.path.join(plots_dir, 'backup')
        os.makedirs(backup_dir, exist_ok=True)
        print(f"Files will be moved to: {backup_dir}")
    
    # Group files by parameter combinations
    image_groups = defaultdict(list)
    all_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
    
    print(f"Found {len(all_files)} image files")
    
    for filename in all_files:
        params = extract_parameters(filename)
        if params:
            image_groups[params].append(filename)
    
    print(f"Found {len(image_groups)} unique parameter combinations")
    
    # Process each group
    total_to_remove = 0
    files_to_remove = []
    
    for params, files in sorted(image_groups.items()):
        win_size, env_type, filter_cutoff = params
        
        # If we have more files than we want to keep
        if len(files) > args.keep_per_category:
            # Sort files by name (which typically includes timestamps)
            files.sort()
            
            # Keep the first N files (oldest or whatever sorting criteria)
            to_keep = files[:args.keep_per_category]
            to_remove = files[args.keep_per_category:]
            
            print(f"\nParameter combination: win={win_size}, env={env_type}, f={filter_cutoff}")
            print(f"  Found {len(files)} images, keeping {len(to_keep)}, removing {len(to_remove)}")
            
            if args.dry_run:
                print("  Would keep:")
                for f in to_keep:
                    print(f"    - {f}")
                print("  Would remove:")
                for f in to_remove:
                    print(f"    - {f}")
            
            total_to_remove += len(to_remove)
            files_to_remove.extend(to_remove)
    
    # Delete or move the files
    if not args.dry_run and files_to_remove:
        print(f"\nRemoving {total_to_remove} duplicate images...")
        
        for filename in files_to_remove:
            file_path = os.path.join(plots_dir, filename)
            
            if args.move:
                # Move to backup directory
                dest_path = os.path.join(backup_dir, filename)
                try:
                    shutil.move(file_path, dest_path)
                    print(f"  Moved: {filename}")
                except Exception as e:
                    print(f"  Error moving {filename}: {e}")
            else:
                # Delete file
                try:
                    os.remove(file_path)
                    print(f"  Deleted: {filename}")
                except Exception as e:
                    print(f"  Error deleting {filename}: {e}")
    
    # Summary
    if args.dry_run:
        print(f"\nDry run summary: Would remove {total_to_remove} of {len(all_files)} images")
    else:
        remaining_files = len(all_files) - total_to_remove
        print(f"\nCleanup complete: Removed {total_to_remove} images, {remaining_files} remaining")
        
        if args.move:
            print(f"All removed files were backed up to: {backup_dir}")

if __name__ == "__main__":
    main()