#!/usr/bin/env python3
"""
Lightweight profiling utilities for acoustic lens optimization.
Can be easily integrated into existing scripts.
"""

import time
from collections import defaultdict
from functools import wraps
import numpy as np
import jax

# Global timing storage
_timings = defaultdict(list)
_enabled = True

def enable_profiling(enabled=True):
    """Enable or disable profiling globally."""
    global _enabled
    _enabled = enabled

def clear_timings():
    """Clear all stored timings."""
    _timings.clear()

class profile_scope:
    """Context manager for profiling code blocks."""
    def __init__(self, name):
        self.name = name
        self.start = None
        
    def __enter__(self):
        if _enabled:
            self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        if _enabled and self.start is not None:
            elapsed = time.perf_counter() - self.start
            _timings[self.name].append(elapsed)

def profile_function(name=None):
    """Decorator to profile JAX functions."""
    def decorator(func):
        func_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _enabled:
                return func(*args, **kwargs)
                
            start = time.perf_counter()
            result = func(*args, **kwargs)
            
            # For JAX arrays, block until computation is complete
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
            elif isinstance(result, tuple) and hasattr(result[0], 'block_until_ready'):
                result[0].block_until_ready()
                
            elapsed = time.perf_counter() - start
            _timings[func_name].append(elapsed)
            
            return result
        return wrapper
    return decorator

def get_timing_summary():
    """Get a summary of all timings."""
    summary = {}
    for name, times in _timings.items():
        summary[name] = {
            'count': len(times),
            'total': sum(times),
            'mean': np.mean(times),
            'std': np.std(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times)
        }
    return summary

def print_timing_report(top_n=None):
    """Print a formatted timing report."""
    summary = get_timing_summary()
    if not summary:
        print("No timing data collected.")
        return
        
    # Sort by total time
    sorted_items = sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True)
    
    if top_n:
        sorted_items = sorted_items[:top_n]
    
    print("\n" + "="*80)
    print("TIMING REPORT")
    print("="*80)
    print(f"{'Operation':<40} {'Count':>8} {'Total(s)':>10} {'Mean(s)':>10} {'Std(s)':>10}")
    print("-"*80)
    
    for name, stats in sorted_items:
        print(f"{name:<40} {stats['count']:>8} {stats['total']:>10.3f} "
              f"{stats['mean']:>10.3f} {stats['std']:>10.3f}")
    
    # Calculate percentages
    total_time = sum(s['total'] for s in summary.values())
    if total_time > 0:
        print("\n" + "-"*80)
        print("Percentage breakdown:")
        for name, stats in sorted_items[:10]:  # Top 10 by percentage
            pct = (stats['total'] / total_time) * 100
            print(f"  {name:<35}: {pct:>5.1f}%")

def profile_iteration(iteration_func):
    """Specialized profiler for optimization iterations."""
    @wraps(iteration_func)
    def wrapper(*args, **kwargs):
        with profile_scope("iteration.total"):
            # Profile sub-components if they exist
            start = time.perf_counter()
            
            # Run the iteration
            result = iteration_func(*args, **kwargs)
            
            # Force synchronization for accurate timing
            if isinstance(result, tuple):
                for r in result:
                    if hasattr(r, 'block_until_ready'):
                        r.block_until_ready()
            
            elapsed = time.perf_counter() - start
            
        return result
    return wrapper

# Example usage functions
def add_profiling_to_optimization_step(opt_step_func, compute_field_func=None):
    """
    Wrap an optimization step function with detailed profiling.
    
    Example:
        profiled_opt_step = add_profiling_to_optimization_step(
            optimisation_step, compute_field
        )
    """
    @wraps(opt_step_func)
    def profiled_opt_step(params, opt_state):
        with profile_scope("optimization_step.total"):
            # If we have access to compute_field, profile it separately
            if compute_field_func:
                # Temporarily wrap compute_field
                original_compute = globals().get('compute_field')
                globals()['compute_field'] = profile_function("compute_field")(compute_field_func)
            
            result = opt_step_func(params, opt_state)
            
            # Restore original
            if compute_field_func and original_compute:
                globals()['compute_field'] = original_compute
                
            return result
    
    return profiled_opt_step

# Inline profiling example for the main optimization loop
def profile_optimization_loop(n_iters, opt_step_func, target_pressure_func, 
                            params, opt_state, print_every=5):
    """
    Profile an entire optimization loop.
    
    Args:
        n_iters: Number of iterations
        opt_step_func: Optimization step function
        target_pressure_func: Target pressure calculation function
        params: Initial parameters
        opt_state: Initial optimizer state
        print_every: Print timing report every N iterations
    
    Returns:
        Final params, opt_state, losses, timing_summary
    """
    losses = []
    
    for i in range(n_iters):
        iter_start = time.perf_counter()
        
        with profile_scope(f"iter_{i}.total"):
            # Optimization step
            with profile_scope(f"iter_{i}.opt_step"):
                params, opt_state, loss = opt_step_func(params, opt_state)
            losses.append(float(loss))
            
            # Target pressure (if provided)
            if target_pressure_func:
                with profile_scope(f"iter_{i}.target_pressure"):
                    p_now = target_pressure_func(params)
        
        iter_time = time.perf_counter() - iter_start
        
        # Print progress
        if i % print_every == 0:
            print(f"Iteration {i}: {iter_time:.2f}s | loss: {loss:.6f}")
            if i > 0:
                print_timing_report(top_n=10)
    
    return params, opt_state, losses, get_timing_summary() 