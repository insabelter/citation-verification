#!/usr/bin/env python3
"""
Script to calculate total time from all nohup log files.
Reads all log files in the nohup_logs folder and sums up the times
from lines containing "Took X.XX seconds".
"""

import os
import re
from pathlib import Path


def extract_times_from_log(log_file_path):
    """
    Extract all times from a single log file.
    
    Args:
        log_file_path (Path): Path to the log file
        
    Returns:
        list: List of times in seconds (as floats)
    """
    times = []
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Look for pattern "Took X.XX seconds"
                match = re.search(r'Took (\d+\.?\d*) seconds', line)
                if match:
                    time_value = float(match.group(1))
                    times.append(time_value)
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
    
    return times


def get_log_files(nohup_logs_dir):
    """
    Get all log files from the nohup_logs directory, excluding 'old' folder and nohup_command.txt.
    
    Args:
        nohup_logs_dir (Path): Path to the nohup_logs directory
        
    Returns:
        list: List of log file paths
    """
    log_files = []
    
    for file_path in nohup_logs_dir.iterdir():
        # Skip directories (like 'old' folder) and nohup_command.txt
        if file_path.is_file() and file_path.name != 'nohup_command.txt':
            log_files.append(file_path)
    
    return sorted(log_files)


def format_time(seconds):
    """
    Format seconds into a readable format (hours, minutes, seconds).
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {remaining_seconds:.2f}s"
    elif minutes > 0:
        return f"{minutes}m {remaining_seconds:.2f}s"
    else:
        return f"{remaining_seconds:.2f}s"


def main():
    """Main function to process all log files and calculate total time."""
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    nohup_logs_dir = script_dir / 'nohup_logs'
    
    if not nohup_logs_dir.exists():
        print(f"Error: Directory {nohup_logs_dir} does not exist")
        return
    
    # Get all log files
    log_files = get_log_files(nohup_logs_dir)
    
    if not log_files:
        print("No log files found")
        return
    
    print(f"Found {len(log_files)} log files:")
    for log_file in log_files:
        print(f"  - {log_file.name}")
    print()
    
    # Process each log file
    total_time = 0.0
    total_entries = 0
    file_summaries = {}
    
    for log_file in log_files:
        print(f"Processing {log_file.name}...")
        times = extract_times_from_log(log_file)
        
        if times:
            file_total = sum(times)
            total_time += file_total
            total_entries += len(times)
            
            file_summaries[log_file.name] = {
                'entries': len(times),
                'total_time': file_total,
                'avg_time': file_total / len(times) if times else 0
            }
            
            print(f"  - Found {len(times)} time entries")
            print(f"  - Total time: {format_time(file_total)}")
            print(f"  - Average time per entry: {format_time(file_total / len(times))}")
        else:
            print(f"  - No time entries found")
            file_summaries[log_file.name] = {
                'entries': 0,
                'total_time': 0,
                'avg_time': 0
            }
        print()
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total log files processed: {len(log_files)}")
    print(f"Total time entries found: {total_entries}")
    print(f"Total time across all files: {format_time(total_time)}")
    
    if total_entries > 0:
        print(f"Average time per entry: {format_time(total_time / total_entries)}")
    
    print()
    print("Per-file breakdown:")
    for filename, summary in file_summaries.items():
        if summary['entries'] > 0:
            print(f"  {filename}: {summary['entries']} entries, {format_time(summary['total_time'])}")
        else:
            print(f"  {filename}: No time entries found")


if __name__ == "__main__":
    main()
