import random
import math
import copy
from typing import List, Tuple
from mip_check2 import MIPCheck
from helper_function import Helper


class SimulatedAnnealingOptimizer:
    def __init__(self, parameters, initial_routes: List[List[str]],
                 initial_temperature=1000.0, cooling_rate=0.995, max_iterations=500):
        self.parameters = parameters
        self.helper = Helper(parameters)
        self.checker = MIPCheck(parameters)

        # SA parameters
        self.initial_temp = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iter = max_iterations

        # Solution tracking
        self.current_solution = copy.deepcopy(initial_routes)
        self.current_cost = self._calculate_cost(self.current_solution)
        self.best_solution = copy.deepcopy(self.current_solution)
        self.best_cost = self.current_cost
        
        # Diagnostic counters
        self.stats = {
            'neighbors_generated': 0,
            'neighbors_identical': 0,
            'neighbors_invalid': 0,
            'neighbors_valid': 0,
            'cost_calculations_failed': 0,
            'improvements_found': 0,
            'accepted_worse': 0,
            'rejected': 0,
            'relocate_attempts': 0,
            'relocate_successes': 0,
            'swap_attempts': 0,
            'swap_successes': 0,
            'reverse_attempts': 0,
            'reverse_successes': 0
        }
        
        print(f"=== SA OPTIMIZER INITIALIZED ===")
        print(f"Initial solution: {len(initial_routes)} routes")
        for i, route in enumerate(initial_routes):
            print(f"  Route {i+1}: {route} (length: {len(route)})")
        print(f"Initial cost: {self.current_cost}")
        print(f"Parameters: temp={initial_temperature}, cooling={cooling_rate}, max_iter={max_iterations}")
        print("=" * 40)

    def _calculate_cost(self, routes: List[List[str]]) -> float:
        """Calculate total solution cost with error handling"""
        if not routes:
            return float('inf')
        
        total_cost = sum(
            self.checker.total_cost(r) for r in routes
        )
        return total_cost

    def _is_valid_solution(self, routes: List[List[str]]) -> bool:
        """Check if solution structure is valid"""
        if not routes:
            return False
        
        for route in routes:
            if len(route) < 2:
                return False
            try:
                if not self.helper.feasible_route(route):
                    return False
            except:
                return False
        
        return True

    def _generate_neighbor(self, routes: List[List[str]]) -> List[List[str]]:
        """Generate neighbor solution using random operation"""
        self.stats['neighbors_generated'] += 1
        
        if not routes:
            self.stats['neighbors_invalid'] += 1
            return routes
        
        # Try each operation type with different probabilities
        operations = [
            ('relocate', self._relocate_client_with_repair, 0.5),
            ('swap', self._swap_clients, 0.3),
            ('reverse', self._reverse_segment, 0.2)
        ]
        
        # Select operation based on weights
        rand_val = random.random()
        cumulative = 0
        selected_op = None
        
        for op_name, op_func, weight in operations:
            cumulative += weight
            if rand_val <= cumulative:
                selected_op = (op_name, op_func)
                break
        
        if selected_op is None:
            selected_op = operations[0][:2]  # Fallback to relocate
        
        try:
            neighbor = selected_op[1](copy.deepcopy(routes))
            
            if neighbor == routes:
                self.stats['neighbors_identical'] += 1
                return routes
            
            if self._is_valid_solution(neighbor):
                self.stats['neighbors_valid'] += 1
                return neighbor
            else:
                self.stats['neighbors_invalid'] += 1
                return routes
                
        except Exception as e:
            print(f"[NEIGHBOR] Exception in {selected_op[0]}: {e}")
            self.stats['neighbors_invalid'] += 1
            return routes

    def _relocate_client_with_repair(self, routes: List[List[str]]) -> List[List[str]]:
        self.stats['relocate_attempts'] += 1
        if len(routes) < 2:
            return routes

        valid_sources = [i for i, route in enumerate(routes) if len(route) > 2]
        if not valid_sources:
            return routes

        src_idx = random.choice(valid_sources)
        src_route = routes[src_idx]
        client_pos = random.randint(1, len(src_route) - 2)
        client = src_route[client_pos]

        tgt_candidates = [i for i in range(len(routes)) if i != src_idx]
        if not tgt_candidates:
            return routes
        tgt_idx = random.choice(tgt_candidates)

        new_routes = copy.deepcopy(routes)
        new_routes[src_idx].pop(client_pos)
        insert_pos = random.randint(1, len(new_routes[tgt_idx]))
        new_routes[tgt_idx].insert(insert_pos, client)

        # Try to repair: if infeasible, insert a charging station at a random position
        if not self.helper.feasible_route(new_routes[tgt_idx]):
            # Try inserting a station at a random position
            stations = self.parameters.get('stations', [])
            if stations:
                station = random.choice(stations)
                # Insert station at a random position (not depot)
                pos = random.randint(1, len(new_routes[tgt_idx]) - 1)
                new_routes[tgt_idx].insert(pos, station)
        self.stats['relocate_successes'] += 1
        return new_routes

    def _swap_clients(self, routes: List[List[str]]) -> List[List[str]]:
        """Swap two random clients between (possibly same or different) routes (always modifies if possible)"""
        self.stats['swap_attempts'] += 1
        # Find all client positions (not depot)
        all_clients = [(i, j) for i, route in enumerate(routes) for j in range(1, len(route) - 1)]
        if len(all_clients) < 2:
            return routes
        (i1, j1), (i2, j2) = random.sample(all_clients, 2)
        new_routes = copy.deepcopy(routes)
        # Swap
        new_routes[i1][j1], new_routes[i2][j2] = new_routes[i2][j2], new_routes[i1][j1]
        self.stats['swap_successes'] += 1
        return new_routes

    def _reverse_segment(self, routes: List[List[str]]) -> List[List[str]]:
        """Reverse a random segment within a random route (always modifies if possible)"""
        self.stats['reverse_attempts'] += 1
        # Find routes with at least 3 clients (so a segment can be reversed)
        valid_routes = [i for i, route in enumerate(routes) if len(route) > 3]
        if not valid_routes:
            return routes
        route_idx = random.choice(valid_routes)
        route = routes[route_idx]
        # Pick two positions to reverse between (not depot)
        positions = list(range(1, len(route) - 1))
        if len(positions) < 2:
            return routes
        a, b = sorted(random.sample(positions, 2))
        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
        new_routes = copy.deepcopy(routes)
        new_routes[route_idx] = new_route
        self.stats['reverse_successes'] += 1
        return new_routes

    def optimize(self) -> Tuple[List[List[str]], float]:
        """Main optimization method - the one called by the UI"""
        print(f"\n🔥 Starting Simulated Annealing Optimization")
        print(f"Initial cost: {self.current_cost:.2f}")
        
        temp = self.initial_temp
        iteration = 0
        last_improvement = 0
        no_improvement_limit = self.max_iter // 4  # Stop if no improvement for 1/4 of iterations
        
        while iteration < self.max_iter and temp > 0.1:
            # Generate neighbor
            neighbor = self._generate_neighbor(self.current_solution)
            
            if neighbor == self.current_solution:
                iteration += 1
                temp *= self.cooling_rate
                continue
            
            # Calculate neighbor cost
            neighbor_cost = self._calculate_cost(neighbor)
            
            if neighbor_cost == float('inf'):
                iteration += 1
                temp *= self.cooling_rate
                continue

            # Acceptance criteria
            cost_diff = neighbor_cost - self.current_cost
            accept = False
            
            if cost_diff <= 0:
                # Better solution - always accept
                accept = True
                self.stats['improvements_found'] += 1
                last_improvement = iteration
            elif temp > 0:
                # Worse solution - accept with probability
                probability = math.exp(-cost_diff / temp)
                if random.random() < probability:
                    accept = True
                    self.stats['accepted_worse'] += 1
                else:
                    self.stats['rejected'] += 1
            else:
                self.stats['rejected'] += 1

            # Update current solution
            if accept:
                self.current_solution = neighbor
                self.current_cost = neighbor_cost
                
                # Update best solution if improved
                if neighbor_cost < self.best_cost:
                    self.best_solution = copy.deepcopy(neighbor)
                    self.best_cost = neighbor_cost
                    print(f"🎯 Iteration {iteration}: New best cost = {self.best_cost:.2f}")

            # Cool down
            temp *= self.cooling_rate
            iteration += 1
            
            # Progress reporting
            if iteration % (self.max_iter // 10) == 0:
                print(f"⏳ Progress: {iteration}/{self.max_iter} iterations, temp={temp:.2f}, best={self.best_cost:.2f}")
            
            # Early stopping if no improvement for a while
            if iteration - last_improvement > no_improvement_limit:
                print(f"🛑 Early stopping: No improvement for {no_improvement_limit} iterations")
                break

        # Final report
        improvement = self.current_cost - self.best_cost
        print(f"\n✅ Optimization Complete!")
        print(f"   Initial cost: {self.current_cost:.2f}")
        print(f"   Final cost: {self.best_cost:.2f}")
        print(f"   Improvement: {improvement:.2f}")
        print(f"   Iterations: {iteration}")
        
        return self.best_solution, self.best_cost

    def optimize_with_diagnosis(self) -> Tuple[List[List[str]], float]:
        """Enhanced optimization with detailed diagnosis - for debugging"""
        print(f"\n{'='*50}")
        print("STARTING SIMULATED ANNEALING WITH DIAGNOSIS")
        print(f"{'='*50}")
        
        # First, test operations
        self.diagnose_operations()
        
        temp = self.initial_temp
        iteration = 0
        
        print(f"\nStarting main SA loop...")
        
        while iteration < min(100, self.max_iter) and temp > 1.0:  # Limited for diagnosis
            neighbor = self._generate_neighbor(self.current_solution)
            
            if neighbor == self.current_solution:
                if iteration % 10 == 0:
                    print(f"[{iteration}] No neighbor generated")
                iteration += 1
                temp *= self.cooling_rate
                continue
            
            neighbor_cost = self._calculate_cost(neighbor)
            
            if neighbor_cost == float('inf'):
                if iteration % 10 == 0:
                    print(f"[{iteration}] Neighbor has infinite cost")
                iteration += 1
                temp *= self.cooling_rate
                continue

            cost_diff = neighbor_cost - self.current_cost
            
            if cost_diff <= 0:
                # Improvement found
                self.current_solution = neighbor
                self.current_cost = neighbor_cost
                self.stats['improvements_found'] += 1
                
                if neighbor_cost < self.best_cost:
                    self.best_solution = copy.deepcopy(neighbor)
                    self.best_cost = neighbor_cost
                    print(f"[{iteration}] NEW BEST: {self.best_cost:.2f} (improvement: {-cost_diff:.2f})")
                
            elif temp > 0:
                probability = math.exp(-cost_diff / temp)
                if random.random() < probability:
                    self.current_solution = neighbor
                    self.current_cost = neighbor_cost
                    self.stats['accepted_worse'] += 1
                    if iteration % 20 == 0:
                        print(f"[{iteration}] Accepted worse solution: {neighbor_cost:.2f} (prob: {probability:.4f})")
                else:
                    self.stats['rejected'] += 1
            else:
                self.stats['rejected'] += 1

            temp *= self.cooling_rate
            iteration += 1

        # Print comprehensive statistics
        self.print_final_diagnosis()
        
        return self.best_solution, self.best_cost

    def _test_single_operation(self, routes: List[List[str]], operation_name: str, operation_func):
        """Test a single operation and return detailed results"""
        print(f"\n--- Testing {operation_name} ---")
        
        original_routes = copy.deepcopy(routes)
        attempts = 0
        successes = 0
        
        for test in range(10):  # Try 10 times
            attempts += 1
            try:
                result = operation_func(copy.deepcopy(original_routes))
                
                if result == original_routes:
                    print(f"  Attempt {test+1}: No change (identical result)")
                    continue
                
                # Check if result is valid
                if not self._is_valid_solution(result):
                    print(f"  Attempt {test+1}: Invalid solution generated")
                    continue
                
                # Calculate cost
                new_cost = self._calculate_cost(result)
                if new_cost == float('inf'):
                    print(f"  Attempt {test+1}: Infinite cost")
                    continue
                
                successes += 1
                current_cost = self._calculate_cost(original_routes)
                improvement = current_cost - new_cost
                
                print(f"  Attempt {test+1}: SUCCESS! Cost change: {improvement:.2f}")
                print(f"    Original: {original_routes}")
                print(f"    Modified: {result}")
                
                if successes >= 3:  # Stop after finding 3 successful operations
                    break
                    
            except Exception as e:
                print(f"  Attempt {test+1}: Exception - {e}")
        
        print(f"  {operation_name} Summary: {successes}/{attempts} successful")
        return successes > 0

    def diagnose_operations(self):
        """Test all neighborhood operations to see what's working"""
        print("\n" + "="*50)
        print("DIAGNOSING NEIGHBORHOOD OPERATIONS")
        print("="*50)
        
        # Test each operation
        relocate_works = self._test_single_operation(
            self.current_solution, "RELOCATE", self._relocate_client_with_repair
        )
        
        swap_works = self._test_single_operation(
            self.current_solution, "SWAP", self._swap_clients
        )
        
        reverse_works = self._test_single_operation(
            self.current_solution, "REVERSE", self._reverse_segment
        )
        
        print(f"\n=== OPERATION DIAGNOSIS SUMMARY ===")
        print(f"Relocate working: {relocate_works}")
        print(f"Swap working: {swap_works}")
        print(f"Reverse working: {reverse_works}")
        
        if not any([relocate_works, swap_works, reverse_works]):
            print("⚠️  NO OPERATIONS ARE WORKING! This is why SA is stuck.")
            self._detailed_feasibility_check()
        
        return relocate_works, swap_works, reverse_works

    def _detailed_feasibility_check(self):
        """Detailed analysis of why operations might be failing"""
        print(f"\n=== DETAILED FEASIBILITY ANALYSIS ===")
        
        # Check each route individually
        for i, route in enumerate(self.current_solution):
            print(f"\nRoute {i}: {route}")
            print(f"  Length: {len(route)}")
            print(f"  Feasible: {self.helper.feasible_route(route)}")
            
            if len(route) > 2:
                print(f"  Clients: {route[1:-1]}")
                
                # Test removing each client
                for j in range(1, len(route) - 1):
                    test_route = route[:j] + route[j+1:]
                    feasible = self.helper.feasible_route(test_route)
                    print(f"    Remove {route[j]}: feasible = {feasible}")
                
                # Test adding a client from another route
                for other_i, other_route in enumerate(self.current_solution):
                    if other_i != i and len(other_route) > 2:
                        client = other_route[1]  # First client from other route
                        test_route = route[:1] + [client] + route[1:]
                        feasible = self.helper.feasible_route(test_route)
                        print(f"    Add {client}: feasible = {feasible}")
                        break

    def print_final_diagnosis(self):
        """Print comprehensive diagnostic information"""
        print(f"\n{'='*50}")
        print("FINAL DIAGNOSTIC REPORT")
        print(f"{'='*50}")
        
        print(f"Neighbor Generation:")
        print(f"  Total attempts: {self.stats['neighbors_generated']}")
        print(f"  Identical (no change): {self.stats['neighbors_identical']}")
        print(f"  Invalid solutions: {self.stats['neighbors_invalid']}")
        print(f"  Valid neighbors: {self.stats['neighbors_valid']}")
        
        print(f"\nOperation Success Rates:")
        if self.stats['relocate_attempts'] > 0:
            relocate_rate = self.stats['relocate_successes'] / self.stats['relocate_attempts']
            print(f"  Relocate: {self.stats['relocate_successes']}/{self.stats['relocate_attempts']} ({relocate_rate:.2%})")
        
        if self.stats['swap_attempts'] > 0:
            swap_rate = self.stats['swap_successes'] / self.stats['swap_attempts']
            print(f"  Swap: {self.stats['swap_successes']}/{self.stats['swap_attempts']} ({swap_rate:.2%})")
        
        if self.stats['reverse_attempts'] > 0:
            reverse_rate = self.stats['reverse_successes'] / self.stats['reverse_attempts']
            print(f"  Reverse: {self.stats['reverse_successes']}/{self.stats['reverse_attempts']} ({reverse_rate:.2%})")
        
        print(f"\nAcceptance Statistics:")
        print(f"  Improvements found: {self.stats['improvements_found']}")
        print(f"  Worse solutions accepted: {self.stats['accepted_worse']}")
        print(f"  Solutions rejected: {self.stats['rejected']}")
        print(f"  Cost calculation failures: {self.stats['cost_calculations_failed']}")
        
        print(f"\nFinal Results:")
        print(f"  Initial cost: {self.current_cost:.2f}")
        print(f"  Best cost: {self.best_cost:.2f}")
        print(f"  Improvement: {self.current_cost - self.best_cost:.2f}")
        
        # Identify the main problem
        total_neighbors = self.stats['neighbors_generated']
        if total_neighbors > 0:
            identical_rate = self.stats['neighbors_identical'] / total_neighbors
            invalid_rate = self.stats['neighbors_invalid'] / total_neighbors
            
            print(f"\n⚠️  PROBLEM IDENTIFICATION:")
            if identical_rate > 0.8:
                print(f"   - High identical neighbor rate ({identical_rate:.1%}) - Operations not changing solution")
            if invalid_rate > 0.5:
                print(f"   - High invalid neighbor rate ({invalid_rate:.1%}) - Generated solutions violate constraints")
            if self.stats['improvements_found'] == 0:
                print(f"   - No improvements found - Neighborhood too restrictive or initial solution is local optimum")