import string
from mip_check import MIPCheck


class Helper:
    def __init__(self, parameters):
        """
        Take the parameter to initiate a helper instance
        :param parameters: parameter dict of a graph instance
        """
        self.parameters = parameters
        self.checker = MIPCheck(self.parameters)
        self.clients = self.parameters["clients"]
        self.stations = self.parameters["stations"]
        self.all_nodes = self.parameters["all_nodes"]

        # Node collections
        self.depot_start = self.parameters["depot_start"]
        self.depot_end = self.parameters["depot_end"]

        # Node attributes
        self.demand = self.parameters["demand"]
        self.ready_time = self.parameters["ready_time"]
        self.due_date = self.parameters["due_date"]
        self.service_time = self.parameters["service_time"]
        self.arcs = self.parameters["arcs"]
        self.times = self.parameters["times"]
        self.final_data = self.parameters["final_data"]
        self.original_stations = self.parameters["original_stations"]

        # Vehicle parameters
        self.Q = self.parameters["Q"]
        self.C = self.parameters["C"]
        self.g = self.parameters["g"]
        self.r = self.parameters["h"]
        self.v = self.parameters["v"]

    def get_routes_dict(self, incidence_dict):
        """
        This the function to get all routes from the solution, a list contains lists, two-dimensional arrays
        :param incidence_dict: binary arcs with 0 and 1
        :return: list of routes
        """
        routes = []
        for i in self.all_nodes:
            if incidence_dict["D0", i] == 1:
                routes.append(self.get_route_dict(incidence_dict, ["D0"], i))
        return routes

    def get_route_dict(self, incidence_dict, list_of_nodes, node: str):
        """
        This is a recursive function to get just one route from the incidence dict matrix
        :param incidence_dict: binary arcs with 0 and 1
        :param list_of_nodes: the route list
        :param node: the current node pointer
        :return: list, representing one route
        """
        list_of_nodes.append(node)
        if node == "D0_end":
            return list_of_nodes

        for i in self.all_nodes:
            if incidence_dict[node, i] == 1:
                return self.get_route_dict(incidence_dict, list_of_nodes, i)

    def vehicle_number_dict(self, incidence_dict):
        """
        This is the function to get the number of vehicles from incidence dict matrix
        :param incidence_dict: binary arcs with 0 and 1
        :return: int, the number of vehicles in the solution
        """
        return sum(incidence_dict["D0", j] for j in self.all_nodes)

    def vehicle_number_list(self, routes):
        """
        This is the function to get the number of vehicles from list of all routes
        :param routes: the list of routes
        :return: the length of the list which is the number of vehicles
        """
        return len(routes)

    def total_distance_dict(self, incidence_dict):
        """
        This is the function to get the total distance from the incidence dict matrix
        :param incidence_dict: the binary arcs with 0 and 1
        :return: the total traveled distance
        """
        return sum(self.arcs[i, j] * incidence_dict[i, j] for i in self.all_nodes for j in self.all_nodes)

    def total_distance_list(self, routes):
        """
        This is the function to get total distance form all the routes
        :param routes: list of all routes
        :return: total distance of the graph
        """
        return sum(self.distance_one_route(route) for route in routes)

    def distance_one_route(self, route):
        """
        This is the function to get the total distance of a route
        :param route: list of nodes
        :return: distance of a route
        """
        total_distance = 0
        for i in range(len(route)):
            if route[i] == "D0_end":
                break
            else:
                total_distance += self.arcs[route[i], route[i + 1]]
        return total_distance

    def cargo_check(self, route):
        """
        This is the function to check the cargo feasibility for a route
        :param route: one route of one vehicle
        :return: if the cargo on this route is feasible
        """
        return sum(self.demand[i] for i in route) <= self.C

    def depot_check(self, route):
        """
        This is the function to check if for a route, the start is D0 and the end is D0_end
        :param route: list of nodes of a route
        :return: true if yes and false otherwise
        """
        return route[0] == "D0" and route[-1] == "D0_end" and route.count("D0") == 1 and route.count("D0_end") == 1

    def feasible_route(self, route):
        """
        This is the function to check if the route is completely feasible
        :param route: list of nodes
        :return: true if route is feasible and false otherwise
        """
        return self.checker.time_energy(route) and self.cargo_check(route) and self.depot_check(route)

    def feasible(self, routes):
        return all(self.feasible_route(route) for route in routes)
    
    def energy_one_route(self, route):
        """
        Calculate total energy consumption for a route based on arc distances and energy rate.
        :param route: list of nodes
        :return: total energy consumed
        """
        r = self.parameters.get("r", 1.0)  # default to 1.0 if not defined
        total_energy = 0.0
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            if from_node == "D0_end":
                break
            total_energy += self.arcs[from_node, to_node] * r
        return total_energy

    def time_one_route(self, route):
        """
        Calculate total time for a route (travel time + service time).
        :param route: list of nodes
        :return: total time taken for the route
        """
        total_time = 0.0
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            if from_node == "D0_end":
                break
            travel_time = self.times[from_node, to_node]
            service = self.service_time.get(from_node, 0)
            total_time += travel_time + service
        return total_time

    def get_route_metrics(self, route):
        """
        Extract all metrics for visualization for a single route
        Returns a dictionary with:
        - nodes: list of nodes in route
        - times: arrival times at each node
        - arrival_energies: energy levels upon arrival
        - departure_energies: energy levels upon departure
        - distances: accumulated distance at each node
        - charges: charging amounts at stations
        """
        if not self.feasible_route(route):
            raise ValueError("Route is not feasible")
        
        # Get times and energies from MIP checker
        times = self.checker.time_extractor(route)
        arrival_energies = self.checker.energy_extractor(route)
        departure_energies = self.checker.energy_extractor_departure(route)
        
        # Calculate accumulated distance
        distances = [0]
        for i in range(1, len(route)):
            distances.append(distances[-1] + self.arcs[route[i-1], route[i]])
        
        # Calculate charging amounts
        charges = [0]
        for i in range(1, len(route)):
            if route[i] in self.stations:
                charges.append(int(departure_energies[i] - arrival_energies[i]))
            else:
                charges.append(0)
        
        return {
            'nodes': route,
            'times': times,
            'arrival_energies': arrival_energies,
            'departure_energies': departure_energies,
            'distances': distances,
            'charges': charges
        }

    def get_all_metrics(self, routes):
        """
        Get metrics for all routes
        Returns a list of route metrics dictionaries
        """
        return [self.get_route_metrics(route) for route in routes]

    def get_aggregate_metrics(self, routes):
        """
        Calculate aggregate metrics across all routes
        Returns a dictionary with:
        - total_distance
        - total_energy
        - max_time (makespan)
        - num_vehicles
        - total_cost (if applicable)
        """
        all_metrics = self.get_all_metrics(routes)
        
        return {
            'total_distance': sum(metric['distances'][-1] for metric in all_metrics),
            'total_energy': sum(self.energy_one_route(route) for route in routes),
            'max_time': max(metric['times'][-1] for metric in all_metrics),
            'num_vehicles': len(routes),
            # 'total_cost': self.calculate_total_cost(routes)  # Implement this if you have cost calculations
        }

    def prepare_visualization_data(self, routes):
        """
        Prepare data in format needed for the visualization functions
        Returns a tuple of:
        - time_dict: {(vehicle_idx, node): time}
        - energy_dict: {(vehicle_idx, node): energy}
        - distance_dict: {(vehicle_idx, node): distance}
        - charge_dict: {(vehicle_idx, node): charge}
        """
        all_metrics = self.get_all_metrics(routes)
        
        time_dict = {}
        energy_dict = {}
        distance_dict = {}
        charge_dict = {}
        
        for vehicle_idx, metrics in enumerate(all_metrics):
            for node, time, energy, distance, charge in zip(
                metrics['nodes'],
                metrics['times'],
                metrics['arrival_energies'],
                metrics['distances'],
                metrics['charges']
            ):
                time_dict[(vehicle_idx, node)] = time
                energy_dict[(vehicle_idx, node)] = energy
                distance_dict[(vehicle_idx, node)] = distance
                charge_dict[(vehicle_idx, node)] = charge
                
        return time_dict, energy_dict, distance_dict, charge_dict

    # def calculate_total_cost(self, routes):
    #     """
    #     Calculate total solution cost (customize based on your cost model)
    #     """
    #     # Example implementation - adjust based on your cost factors
    #     distance_cost = sum(self.distance_one_route(route) for route in routes) * 0.5  # $0.5 per km
    #     vehicle_cost = len(routes) * 100  # $100 per vehicle
    #     time_cost = max(self.time_one_route(route) for route in routes) * 0.1  # $0.1 per minute
        
    #     # return distance_cost + vehicle_cost + time_cost