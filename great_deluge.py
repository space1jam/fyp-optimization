import random
import copy
import logging
from typing import List, Dict
from mip_check import MIPCheck

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GreatDelugeOptimizer:
    def __init__(self, initial_routes: List[List[str]], checker: MIPCheck, params: Dict):
        """Initialize with detailed logging"""
        logger.info("Initializing GreatDelugeOptimizer")
        self.checker = checker
        self.params = params
        self.depot_start = str(params['depot_start'])
        self.depot_end = str(params['depot_end'])
        
        logger.info(f"Initial routes received: {initial_routes}")
        self.current_routes = self._clean_and_repair_routes(initial_routes)  # Combined cleaning and repair
        logger.info(f"After initial repair: {self.current_routes}")
        
        self.best_routes = copy.deepcopy(self.current_routes)
        self.current_cost = self._calculate_total_cost(self.current_routes)
        self.best_cost = self.current_cost
        self.water_level = self.current_cost * 1.1
        
        self.rain_speed = 0.995
        self.max_neighbor_attempts = 100
        self.cost_history = [self.current_cost]
        logger.info(f"Initialization complete. Starting cost: {self.current_cost}")

    def _clean_and_repair_routes(self, routes: List[List[str]]) -> List[List[str]]:
        """Hard-coded removal of specific problematic strings"""
        cleaned_routes = []
        
        for route in routes:
            # Make a copy of the route to modify
            temp_route = route.copy()
            
            # Hard-coded removal of the exact problematic strings
            if len(temp_route) > 0 and temp_route[0] == "['D0']":
                temp_route.pop(0)
            if len(temp_route) > 0 and temp_route[-1] == "['D0_end']":
                temp_route.pop(-1)
            
            # Ensure proper depot placement
            if len(temp_route) == 0:
                cleaned_routes.append([self.depot_start, self.depot_end])
            else:
                if temp_route[0] != self.depot_start:
                    temp_route.insert(0, self.depot_start)
                if temp_route[-1] != self.depot_end:
                    temp_route.append(self.depot_end)
                cleaned_routes.append(temp_route)
        
        return cleaned_routes

    def _calculate_total_cost(self, routes: List[List[str]]) -> float:
        """Calculate cost with validation logging"""
        total = 0.0
        logger.debug("Calculating total cost")
        
        for i, route in enumerate(routes):
            try:
                if len(route) >= 2 and route[0] == self.depot_start and route[-1] == self.depot_end:
                    cost = self.checker.total_cost(route)
                    total += cost
                    logger.debug(f"Route {i} cost: {cost} - {route}")
                else:
                    logger.warning(f"Invalid route {i} in cost calculation: {route}")
                    total += float('inf')
            except Exception as e:
                logger.error(f"Error calculating cost for route {i} {route}: {str(e)}")
                total += float('inf')
        
        logger.info(f"Total cost: {total}")
        return total

    def _get_neighbor(self) -> List[List[str]]:
        """Generate neighbor with detailed attempt logging"""
        logger.debug("Generating neighbor solution")
        
        for attempt in range(self.max_neighbor_attempts):
            neighbor = copy.deepcopy(self.current_routes)
            logger.debug(f"Attempt {attempt}: Starting with {neighbor}")
            
            if len(neighbor) > 1:
                if random.random() < 0.5:
                    logger.debug("Attempting node move")
                    neighbor = self._move_node(neighbor)
                else:
                    logger.debug("Attempting node swap")
                    neighbor = self._swap_nodes(neighbor)
            
            if self._is_valid_solution(neighbor):
                logger.debug(f"Found valid neighbor: {neighbor}")
                return neighbor
            else:
                logger.debug(f"Invalid neighbor attempt {attempt}: {neighbor}")
        
        logger.warning(f"Failed to find valid neighbor after {self.max_neighbor_attempts} attempts")
        return copy.deepcopy(self.current_routes)

    def _is_valid_solution(self, routes: List[List[str]]) -> bool:
        """Validate solution with detailed logging"""
        if not routes:
            logger.error("Empty solution")
            return False
            
        for i, route in enumerate(routes):
            logger.debug(f"Validating route {i}: {route}")
            
            if len(route) < 2:
                logger.error(f"Route {i} too short: {route}")
                return False
                
            if route[0] != self.depot_start:
                logger.error(f"Route {i} missing start depot: {route[0]}")
                return False
                
            if route[-1] != self.depot_end:
                logger.error(f"Route {i} missing end depot: {route[-1]}")
                return False
                
            try:
                if not self.checker.time_energy(route):
                    logger.error(f"Route {i} failed feasibility check: {route}")
                    return False
            except Exception as e:
                logger.error(f"Error checking feasibility for route {i}: {str(e)}")
                return False
        
        logger.debug("Solution is valid")
        return True

    def _swap_nodes(self, routes: List[List[str]]) -> List[List[str]]:
        """Swap nodes with detailed operation logging"""
        logger.debug(f"Attempting swap on: {routes}")
        
        if len(routes) < 2:
            logger.debug("Not enough routes to swap")
            return routes
            
        candidate_routes = [r for r in routes if len(r) > 2]
        if len(candidate_routes) < 2:
            logger.debug("Not enough routes with swappable nodes")
            return routes
            
        route1, route2 = random.sample(candidate_routes, 2)
        logger.debug(f"Selected routes for swap: {route1} and {route2}")
        
        if len(route1) <= 3 or len(route2) <= 3:
            logger.debug("Routes too short to swap without making invalid")
            return routes
            
        i = random.randint(1, len(route1) - 2)
        j = random.randint(1, len(route2) - 2)
        logger.debug(f"Swapping positions {i} and {j}")
        
        route1[i], route2[j] = route2[j], route1[i]
        logger.debug(f"After swap: {route1} and {route2}")
        
        return routes

    def _move_node(self, routes: List[List[str]]) -> List[List[str]]:
        """Move node with detailed operation logging"""
        logger.debug(f"Attempting move on: {routes}")
        
        if len(routes) < 2:
            logger.debug("Not enough routes to move between")
            return routes
            
        source_routes = [r for r in routes if len(r) > 2]
        if not source_routes:
            logger.debug("No source routes with movable nodes")
            return routes
            
        route1 = random.choice(source_routes)
        logger.debug(f"Selected source route: {route1}")
        
        if len(route1) <= 3:
            logger.debug("Source route too short to move from")
            return routes
            
        node = route1.pop(random.randint(1, len(route1) - 2))
        logger.debug(f"Moving node {node} from {route1}")
        
        target_routes = [r for r in routes if r != route1 and len(r) >= 2]
        if not target_routes:
            logger.debug("No valid target routes - putting node back")
            route1.insert(random.randint(1, len(route1) - 1), node)
            return routes
            
        route2 = random.choice(target_routes)
        pos = random.randint(1, len(route2) - 1)
        route2.insert(pos, node)
        logger.debug(f"Moved to route {route2} at position {pos}")
        
        return routes

    def optimize(self, max_iterations: int = 1000) -> List[List[str]]:
        """Run optimization with detailed iteration logging"""
        logger.info(f"Starting optimization for {max_iterations} iterations")
        
        for iteration in range(max_iterations):
            logger.info(f"\n=== Iteration {iteration} ===")
            logger.info(f"Current water level: {self.water_level}")
            logger.debug(f"Current routes: {self.current_routes}")
            
            neighbor_routes = None
            for attempt in range(10):
                neighbor = self._get_neighbor()
                if self._is_valid_solution(neighbor):
                    neighbor_routes = neighbor
                    logger.debug(f"Found valid neighbor on attempt {attempt}")
                    break
                logger.debug(f"Neighbor attempt {attempt} failed validation")
            
            if neighbor_routes is None:
                logger.warning("Could not find valid neighbor this iteration")
                continue
                
            neighbor_cost = self._calculate_total_cost(neighbor_routes)
            logger.info(f"Neighbor cost: {neighbor_cost} (Current best: {self.best_cost})")
            
            if neighbor_cost <= self.water_level:
                self.current_routes = neighbor_routes
                self.current_cost = neighbor_cost
                logger.debug(f"Accepted neighbor: {neighbor_routes}")
                
                if neighbor_cost < self.best_cost:
                    self.best_routes = copy.deepcopy(neighbor_routes)
                    self.best_cost = neighbor_cost
                    logger.info(f"New best solution found: {self.best_cost}")
            
            self.water_level *= self.rain_speed
            self.cost_history.append(self.current_cost)
            
            if iteration > 100 and abs(self.cost_history[-100] - self.current_cost) < 1:
                logger.info("Early stopping - no significant improvement")
                break
        
        final_routes = self._clean_and_repair_routes(self.best_routes)
        logger.info(f"Optimization complete. Final routes: {final_routes}")
        logger.info(f"Final cost: {self.best_cost}")
        
        return final_routes