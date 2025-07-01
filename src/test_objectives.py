#!/usr/bin/env python3
"""
Test script to demonstrate different objective function types.
This script runs the optimization with each objective type and compares results.
"""

import subprocess
import os

# List of objective types to test
objective_types = ["point", "area_contrast", "uniform", "sidelobe"]

# Base configuration to modify in focus.py
config_lines = {
    "point": 'OBJECTIVE_TYPE = "point"',
    "area_contrast": 'OBJECTIVE_TYPE = "area_contrast"',
    "uniform": 'OBJECTIVE_TYPE = "uniform"',
    "sidelobe": 'OBJECTIVE_TYPE = "sidelobe"'
}

print("Testing different objective function types for acoustic lens optimization\n")

for obj_type in objective_types:
    print(f"\n{'='*60}")
    print(f"Testing objective type: {obj_type}")
    print(f"{'='*60}")
    
    # Read the current focus.py file
    with open('focus.py', 'r') as f:
        lines = f.readlines()
    
    # Find and replace the OBJECTIVE_TYPE line
    for i, line in enumerate(lines):
        if line.strip().startswith('OBJECTIVE_TYPE ='):
            lines[i] = config_lines[obj_type] + '\n'
            break
    
    # Write the modified file
    with open('focus.py', 'w') as f:
        f.writelines(lines)
    
    # Create output directory for this objective type
    output_dir = f'/workspace/hologram/outs_{obj_type}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Update output paths in focus.py to use the specific directory
    # (This is a simplified approach - in production you'd make this configurable)
    
    print(f"Running optimization with {obj_type} objective...")
    print(f"Output will be saved to: {output_dir}")
    
    # Run the focus.py script
    try:
        result = subprocess.run(['python3', 'focus.py'], 
                               capture_output=True, 
                               text=True, 
                               timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print(f"✓ {obj_type} optimization completed successfully")
            # Print last few lines of output (typically contains the results)
            output_lines = result.stdout.strip().split('\n')
            print("\nFinal results:")
            for line in output_lines[-10:]:
                if "Initial:" in line or "Final:" in line or "Change:" in line:
                    print(f"  {line}")
        else:
            print(f"✗ {obj_type} optimization failed")
            print(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print(f"✗ {obj_type} optimization timed out")
    except Exception as e:
        print(f"✗ Error running {obj_type}: {str(e)}")

print(f"\n{'='*60}")
print("All tests completed!")
print(f"{'='*60}")

# Reset to default
with open('focus.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.strip().startswith('OBJECTIVE_TYPE ='):
        lines[i] = 'OBJECTIVE_TYPE = "area_contrast"\n'
        break

with open('focus.py', 'w') as f:
    f.writelines(lines)

print("\nNote: OBJECTIVE_TYPE has been reset to 'area_contrast' in focus.py") 