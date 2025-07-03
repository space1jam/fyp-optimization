import random
import math
import copy
from typing import List, Tuple, Dict, Any
from collections import deque
from mip_check2 import MIPCheck  # Changed to use mip_check2 (PuLP version)
from helper_function import Helper


class TabuSearchOptimizer:
    def __init__(self, parameters, initial_routes: List[List[str]],
                 max_iterations=1000, tabu_tenure=25, max_diversification=10):
        self.parameters = parameters
        self.helper = Helper(parameters)
        self.checker = MIPCheck(parameters)

        # TS parameters
        self.max_iter = max_iterations
        self.tabu_tenure = tabu_tenure
        self.max_diversification = max_diversification
        
        # Solution tracking
        self.current_solution = copy.deepcopy(initial_routes)
        self.current_cost = self._calculate_cost(self.current_solution)
        self.best_solution = copy.deepcopy(self.current_solution)
        self.best_cost = self.current_cost
        self.best_feasible_cost = float('inf')
        self.best_feasible_solution = None
        
        # Tabu list
        self.tabu_list = deque(maxlen=tabu_tenure)
        self.frequency_memory = {}  # For frequency-based diversification
        
        # Diversification tracking
        self.diversification_count = 0
        self.no_improvement_streak = 0
        
        # Diagnostic counters
        self.stats = {
            'neighbors_generated': 0,
            'neighbors_identical': 0,
            'neighbors_invalid': 0,
            'neighbors_valid': 0,
            'cost_calculations_failed': 0,
            'improvements_found': 0,
            'accepted_worse': 0,
            'rejected_tabu': 0,
            'relocate_attempts': 0,
            'relocate_successes': 0,
            'swap_attempts': 0,
            'swap_successes': 0,
            'reverse_attempts': 0,
            'reverse_successes': 0,
            'diversifications': 0,
            'intensifications': 0,
            'feasible_solutions': 0
        }
        
        print(f"=== TABU SEARCH OPTIMIZER INITIALIZED ===")
        print(f"Initial solution: {len(initial_routes)} routes")
        for i, route in enumerate(initial_routes):
            print(f"  Route {i+1}: {route} (length: {len(route)})")
        print(f"Initial cost: {self.current_cost:.2f}")
        print(f"Parameters: max_iter={max_iterations}, tabu_tenure={tabu_tenure}")
        print("=" * 40)

    def _calculate_cost(self, routes: List[List[str]]) -> float:
        """Calculate total solution cost with comprehensive error handling"""
        if not routes:
            return float('inf')
        
        total_cost = 0.0
        for i, route in enumerate(routes):
            try:
                if len(route) < 2:
                    return float('inf')
                
                route_cost = self.checker.total_cost(
                    route,
                    time_window_weight=80.0,
                    energy_penalty_weight=150.0,
                    travel_time_weight=2.5,
                    vehicle_cost_weight=200.0
                )
                
                if math.isnan(route_cost) or math.isinf(route_cost):
                    return float('inf')
                
                total_cost += route_cost
                
            except Exception as e:
                # Log the error for debugging but don't crash
                print(f"Warning: Error calculating cost for route {i}: {e}")
                return float('inf')
        
        return total_cost

    def _is_valid_solution(self, routes: List[List[str]]) -> bool:
        """Check if solution structure is valid with detailed feasibility checks"""
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

    def _generate_move_signature(self, move_type: str, details: Dict[str, Any]) -> str:
        """Create a unique signature for a move to track in tabu list"""
        if move_type == "relocate":
            return f"relocate_{details['client']}_from_{details['src_route']}_to_{details['tgt_route']}_pos_{details['position']}"
        elif move_type == "swap":
            return f"swap_{details['client1']}_in_{details['route1']}_pos_{details['pos1']}_with_{details['client2']}_in_{details['route2']}_pos_{details['pos2']}"
        elif move_type == "reverse":
            return f"reverse_{details['route']}_between_{details['start']}_and_{details['end']}"
        return "unknown_move"

    def _is_tabu(self, move_signature: str) -> bool:
        """Check if a move is currently tabu with probabilistic expiration"""
        if move_signature in self.tabu_list:
            # 10% chance to ignore tabu status for older moves (reduced from 15%)
            if random.random() < 0.10 and self.tabu_list.index(move_signature) < len(self.tabu_list)//3:
                return False
            return True
        return False

    def _update_frequency_memory(self, solution: List[List[str]]):
        """Update frequency memory for diversification"""
        key = tuple(tuple(route) for route in solution)
        self.frequency_memory[key] = self.frequency_memory.get(key, 0) + 1

    def _generate_neighbors(self, routes: List[List[str]]) -> List[Tuple[List[List[str]], str, Dict[str, Any]]]:
        """Generate multiple neighbor solutions with move information"""
        neighbors = []
        attempts = 0
        max_attempts = 200
        
        while len(neighbors) < 30 and attempts < max_attempts:
            move_type = random.choices(
                ["relocate", "swap", "reverse"],
                weights=[0.5, 0.3, 0.2],  # Favor relocate moves
                k=1
            )[0]
            
            if move_type == "relocate":
                neighbor, move_info = self._relocate_client_with_info(copy.deepcopy(routes))
            elif move_type == "swap":
                neighbor, move_info = self._swap_clients_with_info(copy.deepcopy(routes))
            else:
                neighbor, move_info = self._reverse_segment_with_info(copy.deepcopy(routes))
            
            # Better comparison for deep nested lists
            if neighbor and move_info and self._is_different_solution(neighbor, routes):
                neighbors.append((neighbor, move_type, move_info))
            attempts += 1
        
        self.stats['neighbors_generated'] += len(neighbors)
        return neighbors

    def _is_different_solution(self, solution1: List[List[str]], solution2: List[List[str]]) -> bool:
        """Check if two solutions are different by comparing their string representations"""
        return str(solution1) != str(solution2)

    def _relocate_client_with_info(self, routes: List[List[str]]) -> Tuple[List[List[str]], Dict[str, Any]]:
        """Relocate with move information for tabu list"""
        self.stats['relocate_attempts'] += 1
        
        if len(routes) < 2:
            return routes, {}
            
        valid_sources = [i for i, route in enumerate(routes) if len(route) > 2]
        if not valid_sources:
            return routes, {}
            
        src_idx = random.choice(valid_sources)
        tgt_idx = random.choice([i for i in range(len(routes)) if i != src_idx])
        client_pos = random.randint(1, len(routes[src_idx]) - 2)
        client = routes[src_idx][client_pos]
        
        new_routes = copy.deepcopy(routes)
        new_routes[src_idx].pop(client_pos)
        
        best_pos = None
        best_cost = float('inf')
        valid_positions = []
        
        # Try all possible positions in target route
        for pos in range(1, len(new_routes[tgt_idx])):
            test_route = new_routes[tgt_idx][:pos] + [client] + new_routes[tgt_idx][pos:]
            if self.helper.feasible_route(test_route):
                valid_positions.append(pos)
        
        # If no valid positions, return original
        if not valid_positions:
            return routes, {}
        
        # Evaluate best valid position
        for pos in valid_positions:
            test_routes = copy.deepcopy(new_routes)
            test_routes[tgt_idx] = test_routes[tgt_idx][:pos] + [client] + test_routes[tgt_idx][pos:]
            cost = self._calculate_cost(test_routes)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        if best_pos is not None:
            new_routes[tgt_idx] = new_routes[tgt_idx][:best_pos] + [client] + new_routes[tgt_idx][best_pos:]
            self.stats['relocate_successes'] += 1
            move_info = {
                'client': client,
                'src_route': src_idx,
                'tgt_route': tgt_idx,
                'position': best_pos
            }
            return new_routes, move_info
            
        return routes, {}

    def _swap_clients_with_info(self, routes: List[List[str]]) -> Tuple[List[List[str]], Dict[str, Any]]:
        """Swap with move information for tabu list"""
        self.stats['swap_attempts'] += 1
        
        all_clients = [(i, j) for i, route in enumerate(routes)
                       for j in range(1, len(route) - 1)]
        
        if len(all_clients) < 2:
            return routes, {}
            
        (i1, j1), (i2, j2) = random.sample(all_clients, 2)
        new_routes = copy.deepcopy(routes)
        c1, c2 = new_routes[i1][j1], new_routes[i2][j2]
        new_routes[i1][j1], new_routes[i2][j2] = c2, c1
        
        if self.helper.feasible_route(new_routes[i1]) and self.helper.feasible_route(new_routes[i2]):
            self.stats['swap_successes'] += 1
            move_info = {
                'client1': c1,
                'route1': i1,
                'pos1': j1,
                'client2': c2,
                'route2': i2,
                'pos2': j2
            }
            return new_routes, move_info
            
        return routes, {}

    def _reverse_segment_with_info(self, routes: List[List[str]]) -> Tuple[List[List[str]], Dict[str, Any]]:
        """Reverse with move information for tabu list"""
        self.stats['reverse_attempts'] += 1
        
        valid_routes = [i for i, route in enumerate(routes) if len(route) > 3]
        if not valid_routes:
            return routes, {}
            
        route_idx = random.choice(valid_routes)
        route = routes[route_idx]
        positions = list(range(1, len(route) - 1))
        
        if len(positions) < 2:
            return routes, {}
            
        a, b = sorted(random.sample(positions, 2))
        new_route = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
        
        if self.helper.feasible_route(new_route):
            new_routes = copy.deepcopy(routes)
            new_routes[route_idx] = new_route
            self.stats['reverse_successes'] += 1
            move_info = {
                'route': route_idx,
                'start': a,
                'end': b
            }
            return new_routes, move_info
            
        return routes, {}

    def _validate_solution(self, routes: List[List[str]]) -> bool:
        """Validate that a solution is structurally sound"""
        if not routes:
            return False
        
        for route in routes:
            if not route or len(route) < 2:
                return False
            # Check that routes start and end with depots
            if route[0] != self.parameters.get("depot_start", ["D0"])[0] or route[-1] != self.parameters.get("depot_end", ["D0_end"])[0]:
                return False
        
        return True

    def _diversify(self):
        """Apply diversification when stuck in local optima"""
        self.stats['diversifications'] += 1
        self.diversification_count += 1
        self.no_improvement_streak = 0
        
        print(f"\n🚀 Applying diversification #{self.diversification_count}")
        
        candidates = []
        
        # Strategy 1: Split random routes
        for _ in range(3):
            new_solution = copy.deepcopy(self.current_solution)
            splittable_routes = [i for i, r in enumerate(new_solution) if len(r) > 4]
            if splittable_routes:
                route_idx = random.choice(splittable_routes)
                route = new_solution[route_idx]
                split_pos = random.randint(2, len(route)-2)
                new_route1 = [route[0]] + route[1:split_pos] + [route[-1]]
                new_route2 = [route[0]] + route[split_pos:-1] + [route[-1]]
                
                if (self.helper.feasible_route(new_route1) and 
                    self.helper.feasible_route(new_route2)):
                    del new_solution[route_idx]
                    new_solution.extend([new_route1, new_route2])
                    if self._validate_solution(new_solution):
                        candidates.append(new_solution)
        
        # Strategy 2: Merge random routes
        if len(self.current_solution) >= 2:
            for _ in range(3):
                new_solution = copy.deepcopy(self.current_solution)
                i, j = random.sample(range(len(new_solution)), 2)
                route1 = new_solution[i]
                route2 = new_solution[j]
                merged = route1[:-1] + route2[1:]
                
                if self.helper.feasible_route(merged):
                    del new_solution[max(i,j)]
                    del new_solution[min(i,j)]
                    new_solution.append(merged)
                    if self._validate_solution(new_solution):
                        candidates.append(new_solution)
        
        # Strategy 3: Randomly shuffle multiple routes
        for _ in range(2):
            new_solution = copy.deepcopy(self.current_solution)
            for route_idx in random.sample(range(len(new_solution)), min(3, len(new_solution))):
                route = new_solution[route_idx]
                if len(route) > 3:
                    middle = route[1:-1]
                    random.shuffle(middle)
                    new_route = [route[0]] + middle + [route[-1]]
                    if self.helper.feasible_route(new_route):
                        new_solution[route_idx] = new_route
            if self._validate_solution(new_solution):
                candidates.append(new_solution)
        
        # Select best candidate
        if candidates:
            # Filter out invalid candidates
            valid_candidates = [c for c in candidates if self._validate_solution(c)]
            if valid_candidates:
                valid_candidates.sort(key=lambda x: self._calculate_cost(x))
                best_candidate = valid_candidates[0]
                print(f"  Diversification cost: {self._calculate_cost(best_candidate):.2f}")
                return best_candidate
        
        return self.current_solution

    def _intensify(self):
        """Local search around current best solution"""
        self.stats['intensifications'] += 1
        print("\n🔍 Intensification phase")
        
        best = copy.deepcopy(self.best_solution)
        best_cost = self.best_cost
        improved = True
        iterations = 0
        
        while improved and iterations < 50:
            improved = False
            neighbors = self._generate_neighbors(best)
            
            for neighbor, move_type, move_info in neighbors:
                cost = self._calculate_cost(neighbor)
                if cost < best_cost:
                    best = neighbor
                    best_cost = cost
                    improved = True
                    move_signature = self._generate_move_signature(move_type, move_info)
                    self.tabu_list.append(move_signature)
                    break
                    
            iterations += 1
        
        if best_cost < self.best_cost:
            self.best_solution = best
            self.best_cost = best_cost
            print(f"  Intensification found better solution: {self.best_cost:.2f}")
        
        return best

    def optimize(self) -> Tuple[List[List[str]], float]:
        """Main Tabu Search optimization method"""
        print(f"\n🔥 Starting Tabu Search Optimization")
        print(f"Initial cost: {self.current_cost:.2f}")
        
        iteration = 0
        last_improvement = 0
        no_improvement_limit = self.max_iter // 4
        
        while iteration < self.max_iter:
            # Periodically apply intensification
            if self.no_improvement_streak > 20 and random.random() < 0.3:
                self.current_solution = self._intensify()
                self.current_cost = self._calculate_cost(self.current_solution)
                iteration += 1
                continue
                
            neighbors = self._generate_neighbors(self.current_solution)
            
            if not neighbors:
                iteration += 1
                continue
            
            # Evaluate all neighbors
            best_neighbor = None
            best_cost = float('inf')
            best_move_signature = None
            best_move_info = None
            
            for neighbor, move_type, move_info in neighbors:
                neighbor_cost = self._calculate_cost(neighbor)
                
                if neighbor_cost == float('inf'):
                    continue
                    
                move_signature = self._generate_move_signature(move_type, move_info)
                is_tabu = self._is_tabu(move_signature)
                
                # Aspiration criterion - accept tabu moves if they're significantly better than best cost
                if is_tabu and neighbor_cost >= self.best_cost * 0.95:
                    self.stats['rejected_tabu'] += 1
                    continue
                
                # Accept non-tabu moves or tabu moves that are significantly better
                if neighbor_cost < best_cost:
                    best_neighbor = neighbor
                    best_cost = neighbor_cost
                    best_move_signature = move_signature
                    best_move_info = move_info
            
            # Accept the best move
            if best_neighbor is not None:
                self.current_solution = best_neighbor
                self.current_cost = best_cost
                
                if best_move_signature:
                    self.tabu_list.append(best_move_signature)
                    self._update_frequency_memory(best_neighbor)
                    
                # Update best solution
                if best_cost < self.best_cost:
                    self.best_solution = copy.deepcopy(best_neighbor)
                    self.best_cost = best_cost
                    last_improvement = iteration
                    self.no_improvement_streak = 0
                    self.stats['improvements_found'] += 1
                    print(f"🎯 Iteration {iteration}: New best cost = {self.best_cost:.2f}")
                else:
                    self.no_improvement_streak += 1
                    self.stats['accepted_worse'] += 1

                # Track best feasible solution
                if all(self.helper.feasible_route(r) for r in self.current_solution):
                    self.stats['feasible_solutions'] += 1
                    feasible_cost = self._calculate_cost(self.current_solution)
                    if feasible_cost < self.best_feasible_cost:
                        self.best_feasible_solution = copy.deepcopy(self.current_solution)
                        self.best_feasible_cost = feasible_cost
            else:
                # Apply diversification if no moves found
                if self.diversification_count < self.max_diversification:
                    diversified_solution = self._diversify()
                    diversified_cost = self._calculate_cost(diversified_solution)
                    
                    if diversified_cost < float('inf'):
                        self.current_solution = diversified_solution
                        self.current_cost = diversified_cost
                        self.tabu_list.clear()  # Reset tabu list after diversification
                else:
                    print("  Max diversifications reached")
                    break
            
            iteration += 1
            
            # Progress reporting
            if iteration % (self.max_iter // 10) == 0:
                print(f"⏳ Progress: {iteration}/{self.max_iter} iterations, best={self.best_cost:.2f}")
            
            # Early stopping if no improvement for too long
            if iteration - last_improvement > no_improvement_limit:
                print(f"🛑 Early stopping: No improvement for {no_improvement_limit} iterations")
                break

        print(f"\n✅ Optimization Complete!")
        print(f"   Initial cost: {self.current_cost:.2f}")
        print(f"   Final cost: {self.best_cost:.2f}")
        print(f"   Improvement: {self.current_cost - self.best_cost:.2f}")
        print(f"   Iterations: {iteration}")
        
        # Return best feasible solution if available, otherwise best found
        if self.best_feasible_solution is not None:
            return self.best_feasible_solution, self.best_feasible_cost
        return self.best_solution, self.best_cost

    def print_stats(self):
        """Print optimization statistics"""
        print(f"\n=== TABU SEARCH STATISTICS ===")
        print(f"Neighbors generated: {self.stats['neighbors_generated']}")
        print(f"Improvements found: {self.stats['improvements_found']}")
        print(f"Worse solutions accepted: {self.stats['accepted_worse']}")
        print(f"Tabu moves rejected: {self.stats['rejected_tabu']}")
        print(f"Diversifications applied: {self.stats['diversifications']}")
        print(f"Intensifications applied: {self.stats['intensifications']}")
        print(f"Feasible solutions found: {self.stats['feasible_solutions']}")
        
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