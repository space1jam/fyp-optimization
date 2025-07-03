from typing import List, Dict, Any
from mip_check import MIPCheck
from station_insert import StationInsertion
from helper_function import Helper
import random

class Heuristic:
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the heuristic solver with problem parameters.
        
        Args:
            parameters: Dictionary containing all problem parameters in adapted format
                       (output from adapter.adapt_evrp_instance())
        """

        # Store parameters
        self.parameters = parameters
        self.checker = MIPCheck(self.parameters)
        self.SI = StationInsertion(self.parameters)
        self.helper = Helper(self.parameters)
        

        # Node collections
        #self.clients = self.parameters["clients"]  # Customer nodes
        #self.stations = self.parameters["stations"]  # Charging stations

        self.clients = [str(client) for client in parameters["clients"]]
        self.stations = [str(station) for station in parameters["stations"]] 

        self.all_nodes = self.parameters["all_nodes"]  # All nodes

        self.depot_start = str(parameters["depot_start"][0])
        self.depot_end = str(parameters["depot_end"][0])
        
        # Node attributes
        self.demand = self.parameters["demand"]  # Dictionary of node demands
        self.ready_time = self.parameters["ready_time"]  # Ready times
        self.due_date = self.parameters["due_date"]  # Due dates
        self.service_time = self.parameters["service_time"]  # Service times
        
        # Problem matrices
        self.arcs = self.parameters["arcs"]  # Distance between nodes (dict)
        self.times = self.parameters["times"]  # Travel times between nodes (dict)
        
        # Reference data
        self.final_data = self.parameters["final_data"]  # Original instance data
        self.original_stations = self.parameters["original_stations"]  # Station list
        
        # Vehicle parameters
        self.Q = self.parameters["Q"]  # Fuel capacity
        self.C = self.parameters["C"]  # Load capacity
        self.g = self.parameters["g"]  # Inverse refueling rate
        self.r = parameters['h']       # Refueling rate   
        self.v = self.parameters["v"]  # Average velocity

    def initial_solution(self):
        """
        Generate initial feasible solution using heuristic combined with Greedy Station Insertion.
        
        Returns:
            List of routes where each route is a list of node IDs
        """
        routes = []
        route = [self.depot_start, self.depot_end]
        routes.append(route)
        
        removal = self.clients.copy()  # Customers not yet assigned to routes
        
        while removal:
            current_route = routes[-1]
            best_insertion = None
            index_insertion = -1
            min_distance = float('inf')
            
            # Find best feasible insertion
            insertion_cost_cache = {}

            for client in removal:
                for i in range(1, len(current_route)):
                    key = (client, i)
                    if key not in insertion_cost_cache:
                        try:
                            prev_node = current_route[i-1]
                            next_node = current_route[i]

                            arc_prev = self.arcs[(prev_node, client)]
                            arc_next = self.arcs[(client, next_node)]
                            original_arc = self.arcs[(prev_node, next_node)]

                            insertion_cost_cache[key] = arc_prev + arc_next - original_arc
                        except KeyError:
                            insertion_cost_cache[key] = float('inf')  # Effectively skips invalid insertions

                    difference = insertion_cost_cache[key]
                    
                    if difference < min_distance:
                        test_route = current_route[:i] + [client] + current_route[i:]
                        if self.helper.feasible_route(test_route):
                            min_distance = difference
                            best_insertion = client
                            index_insertion = i
            
            if best_insertion:
                # Update route with best insertion
                new_route = current_route[:index_insertion] + [best_insertion] + current_route[index_insertion:]
                routes[-1] = new_route
                removal.remove(best_insertion)
            else:
                # Try inserting with charging stations
                candidates = []
                for client in removal:
                    for i in range(1, len(current_route)):
                        test_route = current_route[:i] + [client] + current_route[i:]
                        if (self.helper.cargo_check(test_route) and 
                            self.checker.time(test_route) and 
                            not self.checker.energy(test_route)):
                            candidates.append(self.SI.greedy_station_insertion_sn(test_route))
                
                if candidates:
                    # Select best candidate route
                    feasible_candidates = [c for c in candidates if self.helper.feasible_route(c)]
                    if feasible_candidates:
                        add_route = min(
                            feasible_candidates,
                            key=lambda c: self.helper.distance_one_route(c)
                        )
                        routes[-1] = add_route
                        # Remove all assigned clients from removal list
                        for node in add_route:
                            if node in removal:
                                removal.remove(node)
                    else:
                        # Start new route if no feasible insertions
                        if current_route != [self.depot_start, self.depot_end]:
                            routes.append([self.depot_start, self.depot_end])
                        else:
                            # Force insert with supplemental stations
                            forced_route = self.SI.supplement_station_insertion(
                                [self.depot_start, removal[0], self.depot_end]
                            )
                            routes[-1] = forced_route
                            removal.remove(removal[0])
                else:
                    # No candidates found - start new route
                    routes.append([self.depot_start, self.depot_end])
        
        return routes

    def random_feasible_initial_solution(self):
        """
        Generate a feasible but randomized initial solution:
        - Shuffle clients
        - Insert each client at a random feasible position in any route, or start a new route if needed
        """
        import random
        clients = self.clients[:]
        random.shuffle(clients)
        routes = [[self.depot_start, self.depot_end]]
        for client in clients:
            inserted = False
            # Try to insert into any existing route at a random position
            random.shuffle(routes)
            for route in routes:
                # Try all possible positions (except depots)
                positions = list(range(1, len(route)))
                random.shuffle(positions)
                for pos in positions:
                    test_route = route[:pos] + [client] + route[pos:]
                    if self.helper.feasible_route(test_route):
                        route.insert(pos, client)
                        inserted = True
                        break
                if inserted:
                    break
            if not inserted:
                # Start a new route for this client
                routes.append([self.depot_start, client, self.depot_end])
        return routes


