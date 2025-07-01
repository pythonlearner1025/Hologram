#!/usr/bin/env python3
"""
Test script for binary material optimization.
This script runs a short optimization to verify the implementation works correctly.
"""

import subprocess
import os

print("Testing binary material optimization for 3D-printable acoustic lens")
print("="*60)

# Run the optimization with fewer iterations for testing
test_command = """
cd /workspace/hologram && source jaxenv/bin/activate && python3 -c "
import focus
# Reduce iterations for testing
focus.main.__code__ = focus.main.__code__.replace(co_consts=tuple(
    30 if c == 30 and i > 0 else c 
    for i, c in enumerate(focus.main.__code__.co_consts)
))
# Run with 5 iterations
import sys
sys.modules['__main__'].n_iterations = 5
focus.main()
"
"""

# Alternative simpler approach - modify the file temporarily
print("Creating test configuration...")

# Read the focus.py file
with open('focus.py', 'r') as f:
    content = f.read()

# Replace n_iterations
test_content = content.replace('n_iterations = 30', 'n_iterations = 5')

# Write to a temporary test file
with open('focus_test.py', 'w') as f:
    f.write(test_content)

print("Running optimization with 5 iterations for testing...")
print("-"*60)

# Run the test
result = subprocess.run(
    ['bash', '-c', 'source jaxenv/bin/activate && python3 focus_test.py'],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("\n✓ Binary material optimization test completed successfully!")
    print("\nOutput from optimization:")
    print("-"*60)
    
    # Print lines containing binary statistics
    for line in result.stdout.split('\n'):
        if 'Binary material distribution:' in line or \
           'Binary design statistics:' in line or \
           'Silicone voxels:' in line or \
           'Water/void voxels:' in line:
            print(line)
    
    print("\nCheck the output directory for generated files:")
    print("- Binary design visualization")
    print("- Material distribution CSV files")
    print("- Pressure field visualizations")
    
else:
    print("\n✗ Test failed!")
    print("Error output:")
    print(result.stderr)

# Clean up
os.remove('focus_test.py')

print("\n" + "="*60)
print("Test complete!") 