#!/bin/bash
# Test script for generating acoustic lenses for various transducer configurations
# Each configuration is run with NUM_LENSES = 1, 2, and 4

# Activate the JAX virtual environment
source /workspace/hologram/jaxenv/bin/activate

echo "Starting acoustic lens optimization for multiple transducers..."
echo "Each transducer will be optimized with 1, 2, and 4 lenses"
echo "============================================="

# Ensure the script is executable
chmod +x focus_3d_poly.py

# Array of lens counts to test
LENS_COUNTS=(4)

# OD30*2MHZ R:21
echo ""
echo "========== OD30*2MHZ R:21 =========="
for num_lenses in "${LENS_COUNTS[@]}"; do
    echo "Running with $num_lenses lenses..."
    python3 focus_3d_poly.py --freq-hz 2e6 --bowl-diam-mm 30 --bowl-roc-mm 21 --num-lenses $num_lenses
    echo "--------------------"
done
'''
# OD30*2MHZ R:24
echo ""
echo "========== OD30*2MHZ R:24 =========="
for num_lenses in "${LENS_COUNTS[@]}"; do
    echo "Running with $num_lenses lenses..."
    python3 focus_3d_poly.py --freq-hz 2e6 --bowl-diam-mm 30 --bowl-roc-mm 24 --num-lenses $num_lenses
    echo "--------------------"
done

# OD30*2MHZ R:35
echo ""
echo "========== OD30*2MHZ R:35 =========="
for num_lenses in "${LENS_COUNTS[@]}"; do
    echo "Running with $num_lenses lenses..."
    python3 focus_3d_poly.py --freq-hz 2e6 --bowl-diam-mm 30 --bowl-roc-mm 35 --num-lenses $num_lenses
    echo "--------------------"
done

# OD38*2MHZ R:38
echo ""
echo "========== OD38*2MHZ R:38 =========="
for num_lenses in "${LENS_COUNTS[@]}"; do
    echo "Running with $num_lenses lenses..."
    python3 focus_3d_poly.py --freq-hz 2e6 --bowl-diam-mm 38 --bowl-roc-mm 38 --num-lenses $num_lenses
    echo "--------------------"
done

# OD50*2.5MHZ R:65
echo ""
echo "========== OD50*2.5MHZ R:65 =========="
for num_lenses in "${LENS_COUNTS[@]}"; do
    echo "Running with $num_lenses lenses..."
    python3 focus_3d_poly.py --freq-hz 2.5e6 --bowl-diam-mm 50 --bowl-roc-mm 65 --num-lenses $num_lenses
    echo "--------------------"
done

echo ""
echo "============================================="
echo "All transducer configurations completed!"
echo "Check the test/ directory for results."
echo "Total configurations tested: ${#LENS_COUNTS[@]} lens counts × 5 transducers = $((${#LENS_COUNTS[@]} * 5)) runs" 