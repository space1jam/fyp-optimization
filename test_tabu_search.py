#!/usr/bin/env python3
"""
Test script for Tabu Search implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tabu_search import TabuSearchOptimizer
from initialize import Heuristic
import random

def create_sample_parameters():
    """Create sample parameters for testing"""
    # Sample problem with 5 clients, 2 stations, 1 depot
    parameters = {
        "clients": ["C1", "C2", "C3", "C4", "C5"],
        "stations": ["S1", "S2"],
        "all_nodes": ["D0", "C1", "C2", "C3", "C4", "C5", "S1", "S2", "D0_end"],
        "depot_start": ["D0"],
        "depot_end": ["D0_end"],
        
        # Demands (only clients have demands)
        "demand": {
            "D0": 0, "C1": 10, "C2": 15, "C3": 8, "C4": 12, "C5": 20,
            "S1": 0, "S2": 0, "D0_end": 0
        },
        
        # Time windows
        "ready_time": {
            "D0": 0, "C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0,
            "S1": 0, "S2": 0, "D0_end": 0
        },
        "due_date": {
            "D0": 1000, "C1": 100, "C2": 100, "C3": 100, "C4": 100, "C5": 100,
            "S1": 1000, "S2": 1000, "D0_end": 1000
        },
        "service_time": {
            "D0": 0, "C1": 10, "C2": 10, "C3": 10, "C4": 10, "C5": 10,
            "S1": 0, "S2": 0, "D0_end": 0
        },
        
        # Distances (symmetric)
        "arcs": {
            ("D0", "C1"): 20, ("D0", "C2"): 25, ("D0", "C3"): 30, ("D0", "C4"): 35, ("D0", "C5"): 40,
            ("D0", "S1"): 15, ("D0", "S2"): 45, ("D0", "D0_end"): 1000,
            ("C1", "C2"): 10, ("C1", "C3"): 15, ("C1", "C4"): 20, ("C1", "C5"): 25,
            ("C1", "S1"): 8, ("C1", "S2"): 35, ("C1", "D0_end"): 1000,
            ("C2", "C3"): 12, ("C2", "C4"): 18, ("C2", "C5"): 22,
            ("C2", "S1"): 12, ("C2", "S2"): 30, ("C2", "D0_end"): 1000,
            ("C3", "C4"): 8, ("C3", "C5"): 15,
            ("C3", "S1"): 18, ("C3", "S2"): 25, ("C3", "D0_end"): 1000,
            ("C4", "C5"): 10,
            ("C4", "S1"): 22, ("C4", "S2"): 20, ("C4", "D0_end"): 1000,
            ("C5", "S1"): 28, ("C5", "S2"): 15, ("C5", "D0_end"): 1000,
            ("S1", "S2"): 35, ("S1", "D0_end"): 1000,
            ("S2", "D0_end"): 1000,
        },
        
        # Travel times (same as distances for simplicity)
        "times": {
            ("D0", "C1"): 20, ("D0", "C2"): 25, ("D0", "C3"): 30, ("D0", "C4"): 35, ("D0", "C5"): 40,
            ("D0", "S1"): 15, ("D0", "S2"): 45, ("D0", "D0_end"): 1000,
            ("C1", "C2"): 10, ("C1", "C3"): 15, ("C1", "C4"): 20, ("C1", "C5"): 25,
            ("C1", "S1"): 8, ("C1", "S2"): 35, ("C1", "D0_end"): 1000,
            ("C2", "C3"): 12, ("C2", "C4"): 18, ("C2", "C5"): 22,
            ("C2", "S1"): 12, ("C2", "S2"): 30, ("C2", "D0_end"): 1000,
            ("C3", "C4"): 8, ("C3", "C5"): 15,
            ("C3", "S1"): 18, ("C3", "S2"): 25, ("C3", "D0_end"): 1000,
            ("C4", "C5"): 10,
            ("C4", "S1"): 22, ("C4", "S2"): 20, ("C4", "D0_end"): 1000,
            ("C5", "S1"): 28, ("C5", "S2"): 15, ("C5", "D0_end"): 1000,
            ("S1", "S2"): 35, ("S1", "D0_end"): 1000,
            ("S2", "D0_end"): 1000,
        },
        "normal_times": {
            ("D0", "C1"): 20, ("D0", "C2"): 25, ("D0", "C3"): 30, ("D0", "C4"): 35, ("D0", "C5"): 40,
            ("D0", "S1"): 15, ("D0", "S2"): 45, ("D0", "D0_end"): 1000,
            ("C1", "C2"): 10, ("C1", "C3"): 15, ("C1", "C4"): 20, ("C1", "C5"): 25,
            ("C1", "S1"): 8, ("C1", "S2"): 35, ("C1", "D0_end"): 1000,
            ("C2", "C3"): 12, ("C2", "C4"): 18, ("C2", "C5"): 22,
            ("C2", "S1"): 12, ("C2", "S2"): 30, ("C2", "D0_end"): 1000,
            ("C3", "C4"): 8, ("C3", "C5"): 15,
            ("C3", "S1"): 18, ("C3", "S2"): 25, ("C3", "D0_end"): 1000,
            ("C4", "C5"): 10,
            ("C4", "S1"): 22, ("C4", "S2"): 20, ("C4", "D0_end"): 1000,
            ("C5", "S1"): 28, ("C5", "S2"): 15, ("C5", "D0_end"): 1000,
            ("S1", "S2"): 35, ("S1", "D0_end"): 1000,
            ("S2", "D0_end"): 1000,
        },
        
        # Vehicle parameters
        "Q": 100,  # Fuel capacity
        "C": 50,   # Load capacity
        "g": 0.1,  # Inverse refueling rate
        "h": 1.0,  # Energy consumption rate
        "v": 1.0,  # Average velocity
        
        # Additional parameters
        "final_data": {},
        "original_stations": ["S1", "S2"],
        "std": 1.0,
        "mean": 0.0
    }
    
    # Add reverse arcs for symmetric distances
    for (i, j), dist in list(parameters["arcs"].items()):
        if (j, i) not in parameters["arcs"]:
            parameters["arcs"][(j, i)] = dist
            parameters["times"][(j, i)] = dist
            parameters["normal_times"][(j, i)] = dist
    
    return parameters

def test_tabu_search():
    """Test the tabu search implementation"""
    print("=== Testing Tabu Search Implementation ===")
    
    # Create sample parameters
    parameters = create_sample_parameters()
    
    # Create initial solution using heuristic
    print("\n1. Creating initial solution...")
    heuristic = Heuristic(parameters)
    initial_routes = heuristic.random_feasible_initial_solution()
    
    print(f"Initial solution: {len(initial_routes)} routes")
    for i, route in enumerate(initial_routes):
        print(f"  Route {i+1}: {route}")
    
    # Test tabu search
    print("\n2. Running Tabu Search...")
    try:
        ts_optimizer = TabuSearchOptimizer(
            parameters=parameters,
            initial_routes=initial_routes,
            max_iterations=50,  # Small number for testing
            tabu_tenure=10,
            max_diversification=3
        )
        
        # Run optimization
        best_routes, best_cost = ts_optimizer.optimize()
        
        print(f"\n3. Results:")
        print(f"Best cost: {best_cost:.2f}")
        print(f"Best solution: {len(best_routes)} routes")
        for i, route in enumerate(best_routes):
            print(f"  Route {i+1}: {route}")
        
        # Print statistics
        ts_optimizer.print_stats()
        
        print("\n✅ Tabu Search test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during Tabu Search: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tabu_search()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1) 