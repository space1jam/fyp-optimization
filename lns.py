import random
import copy
from typing import List, Tuple
from initialize import Heuristic
from mip_check import MIPCheck

class LNSOptimizer:
    def __init__(self, parameters, max_iterations=200, ruin_fraction=0.3, accept_worse_prob=0.05):
        self.parameters = parameters
        self.max_iterations = max_iterations
        self.ruin_fraction = ruin_fraction  # Fraction of clients to remove in ruin phase
        self.accept_worse_prob = accept_worse_prob
        self.heuristic = Heuristic(parameters)
        self.checker = MIPCheck(parameters)
        self.clients = [str(c) for c in parameters['clients']]
        self.depot_start = str(parameters['depot_start'][0])
        self.depot_end = str(parameters['depot_end'][0])

    def _calculate_cost(self, routes: List[List[str]]) -> float:
        return sum(self.checker.total_cost(r) for r in routes)

    def _ruin(self, routes: List[List[str]], num_remove: int) -> Tuple[List[List[str]], List[str]]:
        # Flatten all clients in all routes (excluding depots)
        all_clients = [(i, j) for i, route in enumerate(routes) for j in range(1, len(route)-1)]
        if len(all_clients) <= num_remove:
            num_remove = max(1, len(all_clients) // 2)
        removed_indices = random.sample(all_clients, num_remove)
        removed_clients = []
        new_routes = copy.deepcopy(routes)
        for i, j in sorted(removed_indices, reverse=True):
            removed_clients.append(new_routes[i][j])
            del new_routes[i][j]
        # Remove empty routes (with only depots)
        new_routes = [r for r in new_routes if len(r) > 2]
        return new_routes, removed_clients

    def _recreate(self, routes: List[List[str]], removed_clients: List[str]) -> List[List[str]]:
        # Insert each removed client at the best feasible position (greedy), or start a new route if needed
        for client in removed_clients:
            best_cost = float('inf')
            best_route_idx = None
            best_pos = None
            for idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    test_route = route[:pos] + [client] + route[pos:]
                    if self.heuristic.helper.feasible_route(test_route):
                        test_routes = copy.deepcopy(routes)
                        test_routes[idx] = test_route
                        cost = self._calculate_cost(test_routes)
                        if cost < best_cost:
                            best_cost = cost
                            best_route_idx = idx
                            best_pos = pos
            if best_route_idx is not None:
                routes[best_route_idx].insert(best_pos, client)
            else:
                # Start a new route for this client
                routes.append([self.depot_start, client, self.depot_end])
        return routes

    def optimize(self, initial_routes=None) -> Tuple[List[List[str]], float]:
        if initial_routes is None:
            current_solution = self.heuristic.initial_solution()
        else:
            current_solution = copy.deepcopy(initial_routes)
        current_cost = self._calculate_cost(current_solution)
        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost
        for iteration in range(self.max_iterations):
            num_remove = max(1, int(self.ruin_fraction * sum(len(r)-2 for r in current_solution)))
            ruined_routes, removed_clients = self._ruin(current_solution, num_remove)
            new_solution = self._recreate(ruined_routes, removed_clients)
            new_cost = self._calculate_cost(new_solution)
            accept = False
            if new_cost < current_cost:
                accept = True
            elif random.random() < self.accept_worse_prob:
                accept = True
            if accept:
                current_solution = copy.deepcopy(new_solution)
                current_cost = new_cost
                if new_cost < best_cost:
                    best_solution = copy.deepcopy(new_solution)
                    best_cost = new_cost
                    print(f"Iteration {iteration+1}: New best cost = {best_cost:.2f}")
            if (iteration+1) % max(1, self.max_iterations//10) == 0:
                print(f"Iteration {iteration+1}/{self.max_iterations}, Best Cost: {best_cost:.2f}")
        return best_solution, best_cost

if __name__ == "__main__":
    from document_processor import get_parameters
    params = get_parameters("your_instance_file.txt")
    lns = LNSOptimizer(params, max_iterations=200, ruin_fraction=0.3)
    best_routes, best_cost = lns.optimize()
    print("Best Cost:", best_cost)
    for idx, route in enumerate(best_routes):
        print(f"Route {idx+1}: {route}") 