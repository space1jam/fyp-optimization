import random
import copy
from typing import List, Tuple
from initialize import Heuristic
from mip_check import MIPCheck

class Particle:
    def __init__(self, parameters, clients, depot_start, depot_end, helper, checker):
        self.parameters = parameters
        self.clients = clients[:]
        self.depot_start = depot_start
        self.depot_end = depot_end
        self.helper = helper
        self.checker = checker
        # Position: a permutation of clients
        random.shuffle(self.clients)
        self.position = self.clients[:]
        self.routes = self._decode_position(self.position)
        self.cost = self._calculate_cost(self.routes)
        self.best_position = self.position[:]
        self.best_routes = copy.deepcopy(self.routes)
        self.best_cost = self.cost
        # Velocity: a list of swap operations (as tuples)
        self.velocity = []

    def _decode_position(self, position):
        # Greedily build feasible routes from the permutation
        routes = [[self.depot_start, self.depot_end]]
        for client in position:
            inserted = False
            for route in routes:
                for pos in range(1, len(route)):
                    test_route = route[:pos] + [client] + route[pos:]
                    if self.helper.feasible_route(test_route):
                        route.insert(pos, client)
                        inserted = True
                        break
                if inserted:
                    break
            if not inserted:
                routes.append([self.depot_start, client, self.depot_end])
        return routes

    def _calculate_cost(self, routes):
        return sum(self.checker.total_cost(r) for r in routes)

    def update_personal_best(self):
        if self.cost < self.best_cost:
            self.best_cost = self.cost
            self.best_position = self.position[:]
            self.best_routes = copy.deepcopy(self.routes)

    def apply_velocity(self):
        # Apply each swap in velocity to the position
        for i, j in self.velocity:
            self.position[i], self.position[j] = self.position[j], self.position[i]
        self.routes = self._decode_position(self.position)
        self.cost = self._calculate_cost(self.routes)
        self.update_personal_best()

    def generate_velocity(self, global_best_position, w=0.5, c1=1.0, c2=1.0):
        # Generate velocity as a list of swaps to move towards pbest and gbest
        swaps = []
        # Move towards personal best
        for idx, client in enumerate(self.position):
            if client != self.best_position[idx]:
                j = self.position.index(self.best_position[idx])
                swaps.append((idx, j))
                self.position[idx], self.position[j] = self.position[j], self.position[idx]
        # Move towards global best
        for idx, client in enumerate(self.position):
            if client != global_best_position[idx]:
                j = self.position.index(global_best_position[idx])
                swaps.append((idx, j))
                self.position[idx], self.position[j] = self.position[j], self.position[idx]
        # Add random swaps for exploration
        num_random_swaps = max(1, int(w * len(self.position)))
        for _ in range(num_random_swaps):
            i, j = random.sample(range(len(self.position)), 2)
            swaps.append((i, j))
        # Undo the swaps to restore original position
        for i, j in reversed(swaps):
            self.position[i], self.position[j] = self.position[j], self.position[i]
        # With probability, keep only a subset of swaps
        random.shuffle(swaps)
        self.velocity = swaps[:max(1, int(0.5 * len(swaps)))]

class PSOOptimizer:
    def __init__(self, parameters, num_particles=20, max_iterations=200):
        self.parameters = parameters
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.helper = Heuristic(parameters).helper
        self.checker = MIPCheck(parameters)
        self.clients = [str(c) for c in parameters['clients']]
        self.depot_start = str(parameters['depot_start'][0])
        self.depot_end = str(parameters['depot_end'][0])
        self.swarm = [Particle(parameters, self.clients, self.depot_start, self.depot_end, self.helper, self.checker) for _ in range(num_particles)]
        self.global_best_cost = float('inf')
        self.global_best_position = None
        self.global_best_routes = None
        self.best_feasible_cost = float('inf')
        self.best_feasible_routes = None
        for p in self.swarm:
            if p.cost < self.global_best_cost:
                self.global_best_cost = p.cost
                self.global_best_position = p.position[:]
                self.global_best_routes = copy.deepcopy(p.routes)

    def optimize(self):
        for iteration in range(self.max_iterations):
            for particle in self.swarm:
                particle.generate_velocity(self.global_best_position)
                particle.apply_velocity()
                if particle.cost < self.global_best_cost:
                    self.global_best_cost = particle.cost
                    self.global_best_position = particle.position[:]
                    self.global_best_routes = copy.deepcopy(particle.routes)
                # Track best feasible
                if all(self.helper.feasible_route(r) for r in particle.routes):
                    feasible_cost = sum(self.checker.total_cost(r) for r in particle.routes)
                    if feasible_cost < self.best_feasible_cost:
                        self.best_feasible_cost = feasible_cost
                        self.best_feasible_routes = copy.deepcopy(particle.routes)
            if (iteration+1) % max(1, self.max_iterations//10) == 0:
                print(f"Iteration {iteration+1}/{self.max_iterations}, Best Cost: {self.global_best_cost:.2f}")
        if self.best_feasible_routes is not None:
            print("Best Feasible Cost:", self.best_feasible_cost)
            for idx, route in enumerate(self.best_feasible_routes):
                print(f"Feasible Route {idx+1}: {route}")
        else:
            print("No feasible solution found by PSO.")
        return self.global_best_routes, self.global_best_cost

if __name__ == "__main__":
    # Example usage
    from document_processor import get_parameters
    params = get_parameters("your_instance_file.txt")
    pso = PSOOptimizer(params, num_particles=20, max_iterations=200)
    best_routes, best_cost = pso.optimize()
    print("Best Cost:", best_cost)
    for idx, route in enumerate(best_routes):
        print(f"Route {idx+1}: {route}") 