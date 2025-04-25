#!/usr/bin/env python3
import os
import glob
import yaml
import subprocess
import tempfile
from pathlib import Path
import re

def create_temp_config(template_path: str, results_path: str, lambda_value: str) -> str:
    """
    Create a temporary config file by modifying the template with results path and lambda value.
    
    Args:
        template_path: Path to the template YAML file
        results_path: Path to the results CSV file
        lambda_value: Lambda value extracted from directory name
    
    Returns:
        Path to the temporary config file
    """
    # Read the template file
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the config
    config['name'] = config['name'].replace('<lambda>', lambda_value)
    # Update the predictions path in the first item of predictions list
    config['predictions'][0]['predictions'] = config['predictions'][0]['predictions'].replace('<results_path>', results_path)
    
    # Create a temporary file
    temp_path = os.path.join('options', f"eval_config_lambda_adv_{lambda_value}.yml")
    
    # Write the updated config to the temporary file
    with open(temp_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return temp_path

def extract_lambda(dirname: str) -> str:
    """Extract lambda value from directory name."""
    match = re.search(r'lambda_opt_([^_]+)', dirname)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract lambda value from directory name: {dirname}")

def find_csv_file(directory: str) -> str:
    """Find the only CSV file in the given directory."""
    csv_files = list(Path(directory).glob('*.csv'))
    if len(csv_files) != 1:
        raise ValueError(f"Expected exactly one CSV file in {directory}, found {len(csv_files)}")
    return str(csv_files[0])

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_dir", required=True, help="Parent directory containing result subdirectories")
    parser.add_argument("--template", required=True, help="Path to template YAML file")
    args = parser.parse_args()

    # Make sure options directory exists
    os.makedirs('options', exist_ok=True)
    
    # Get all subdirectories
    subdirs = [d for d in Path(args.parent_dir).iterdir() if d.is_dir()]
    temp_files = []
    
    try:
        for subdir in subdirs:
            try:
                # Extract lambda value from directory name
                lambda_value = extract_lambda(subdir.name)
                
                # Find the CSV file
                results_path = find_csv_file(subdir)
                
                # Create temporary config
                temp_path = create_temp_config(args.template, results_path, lambda_value)
                temp_files.append(temp_path)
                
                # Run evaluation
                cmd = ["python", "evaluation.py", f"--opt={temp_path}"]
                print(f"Running evaluation for lambda {lambda_value}: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                
            except Exception as e:
                print(f"Error processing directory {subdir}: {e}")
                continue
            
    finally:
        # Clean up temporary files
        for temp_path in temp_files:
            try:
                os.remove(temp_path)
                print(f"Removed temporary config: {temp_path}")
            except Exception as e:
                print(f"Failed to remove temporary config {temp_path}: {e}")

if __name__ == "__main__":
    main() 