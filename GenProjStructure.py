import os
import sys
from datetime import datetime

def scan_project_structure(start_path, output_file):
    # Open file for writing
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Project Structure Scan - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Root Directory: {os.path.abspath(start_path)}\n\n")
        
        # Directories/files to ignore
        ignore_dirs = ['__pycache__', '.git', '.idea', 'venv', 'env', '.vscode']
        ignore_files = ['.pyc', '.pyo', '.pyd', '.git', '.DS_Store']
        
        # Walk through directory structure
        for root, dirs, files in os.walk(start_path):
            # Remove ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            # Calculate level for indentation
            level = root.replace(start_path, '').count(os.sep)
            indent = '│   ' * level
            
            # Print directory name
            dir_name = os.path.basename(root)
            if level > 0:  # Don't print the root directory as a subdirectory
                f.write(f"{indent}├── {dir_name}/\n")
            else:
                f.write(f"{dir_name}/\n")
            
            # Print files
            sub_indent = '│   ' * (level + 1)
            for file in sorted(files):
                if not any(file.endswith(ext) for ext in ignore_files):
                    f.write(f"{sub_indent}├── {file}\n")

# Usage
if __name__ == "__main__":
    # Get current directory or specified directory
    if len(sys.argv) > 1:
        project_dir = sys.argv[1]
    else:
        project_dir = "."  # Current directory
    
    output_file = "project_structure.txt"
    scan_project_structure(project_dir, output_file)
    print(f"Project structure saved to {os.path.abspath(output_file)}")