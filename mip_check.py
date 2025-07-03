import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
from typing import List, Dict
import math



class MIPCheck:
    def __init__(self, parameters: Dict):
        self.parameters = parameters
        

        # Node sets
        self.clients = self.parameters["clients"]
        self.stations = self.parameters["stations"]
        self.all_nodes = self.parameters["all_nodes"]

        # Depot handling
        self.depot_start = self.parameters["depot_start"]
        self.depot_end = self.parameters["depot_end"]

        # Node attributes
        self.demand = self.parameters["demand"]
        self.ready_time = self.parameters["ready_time"]
        self.due_date = self.parameters["due_date"]
        self.service_time = self.parameters["service_time"]

        # Matrices
        self.arcs = self.parameters["arcs"]
        self.times = self.parameters["normal_times"]

        # self.times = self.parameters["times"] -- REMOVE

        # Vehicle Parameters
        self.C = self.parameters["C"] 
        self.Q = self.parameters["Q"]
        self.g = self.parameters["g"]
        self.h = self.parameters["h"]
        self.v = self.parameters["v"]

        # Final data reference
        self.std = self.parameters["std"]
        self.mean = self.parameters["mean"]

    def update_times(self, p: float, n: float) -> None:
        """
        Update the time matrix with a stochastic process
        
        Args:
         p (float): probability of time being modified
         n (float): scaling factor for the stochastic process
         
        """
        new_times = self.parameters["times"]
        for i in self.all_nodes:
            for j in self.all_nodes:
                if i != j:
                    if random.random() < p:
                        stochastic = np.random.normal(0, n*self.std, 1)[0]
                        if new_times[i,j] + stochastic > 0:
                            new_times[i,j] = new_times[i,j] + stochastic
        self.times = new_times


    def time_energy(self, route: List[str]) -> bool:
        """
        Check if the time and energy constraints can be satisfied for a given route.

        Args:
            route (list): List of nodes representing the route.

        Returns:
            bool: True if both time and energy constraints can be satisfied, False otherwise.
        """
        # create the model first
        model = gp.Model("route_check_time_energy")

        # set the output flag as 0 to avoid outcome showing
        model.setParam('OutputFlag', 0)

        # create the decision variables, times and energy states
        t = []
        y = []
        Y = []
        for i in range(len(route)):
            t.append(model.addVar(lb=self.ready_time[route[i]], ub=self.due_date[route[i]], vtype=GRB.CONTINUOUS))
            Y.append(model.addVar(lb=0, ub=self.Q, vtype=GRB.CONTINUOUS))
            y.append(model.addVar(lb=0, ub=self.Q, vtype=GRB.CONTINUOUS))

        # create objective function, for checking feasibility, objective is set as 0
        model.setObjective(0, GRB.MINIMIZE)

        # create constraints to check the time and energy
        # time constraints
        for index in range(len(route)):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.stations:
                    model.addConstr(
                        t[index]+self.times[route[index], route[index+1]]+self.g*(Y[index]-y[index])<=t[index+1]
                    )
                else:
                    model.addConstr(
                        t[index]+self.times[route[index], route[index+1]]+self.service_time[route[index]]<=t[index+1]
                    )

        # energy constraints
        for index in range(len(route)):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.clients:
                    model.addConstr(y[index+1]<=y[index]-self.h*self.arcs[route[index], route[index+1]])
                else:
                    model.addConstr(y[index+1]<=Y[index]-self.h*self.arcs[route[index], route[index+1]])

        # departure and arrival energy constraint
        for i in range(len(route)):
            if route[i] in self.stations + self.depot_start + self.depot_end:
                model.addConstr(y[i] <= Y[i])

        # solve the model and optimize
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return True
        if model.status == GRB.INFEASIBLE or model.status == GRB.UNBOUNDED:
            return False
        return False

    def time(self, route: List[str]) -> bool:
        """
        This is the function to check if the time constraint can be satisfied (ignoring energy)

        Args:
            route (list): List of nodes representing the route.

        Returns:
            bool: True if time constraint can be satisfied, False otherwise.
        """

        # create the model first
        model = gp.Model("route_check_time_energy")

        # set the output flag as 0 to avoid outcome showing
        model.setParam('OutputFlag', 0)

        # create the decision variables, times and energy states
        t = []
        y = []
        Y = []
        for i in range(len(route)):
            t.append(model.addVar(lb=self.ready_time[route[i]], ub=self.due_date[route[i]], vtype=GRB.CONTINUOUS))
            Y.append(model.addVar(lb=0, ub=self.Q, vtype=GRB.CONTINUOUS))
            y.append(model.addVar(lb=0, ub=self.Q, vtype=GRB.CONTINUOUS))

        # create objective function, for checking feasibility, objective is set as 0
        model.setObjective(0, GRB.MINIMIZE)

        # create constraints to check the time and energy
        # time constraints
        for index in range(len(route)):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.stations:
                    model.addConstr(
                        t[index] + self.times[route[index], route[index + 1]] <= t[
                            index + 1]
                    )
                else:
                    model.addConstr(
                        t[index] + self.times[route[index], route[index + 1]] + self.service_time[route[index]] <= t[
                            index + 1]
                    )

        # solve the model and optimize
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return True
        if model.status == GRB.INFEASIBLE or model.status == GRB.UNBOUNDED:
            return False
        return False

    def energy(self, route: List[str]) -> bool:
        """
        Check if the energy constraint can be satisfied (ignoring time)
        Args:
            route (list): List of nodes representing the route.

        Returns:
            bool: True if energy constraint can be satisfied, False otherwise.
        """
        # create the model first
        model = gp.Model("route_check_time_energy")

        # set the output flag as 0 to avoid outcome showing
        model.setParam('OutputFlag', 0)

        # create the decision variables, times and energy states
        t = []
        y = []
        Y = []
        for i in range(len(route)):
            t.append(model.addVar(lb=self.ready_time[route[i]], ub=self.due_date[route[i]], vtype=GRB.CONTINUOUS))
            Y.append(model.addVar(lb=0, ub=self.Q, vtype=GRB.CONTINUOUS))
            y.append(model.addVar(lb=0, ub=self.Q, vtype=GRB.CONTINUOUS))

        # create objective function, for checking feasibility, objective is set as 0
        model.setObjective(0, GRB.MINIMIZE)

        # create constraints to check the energy

        # energy constraints
        for index in range(len(route)):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.clients:
                    model.addConstr(y[index + 1] <= y[index] - self.h * self.arcs[route[index], route[index + 1]])
                else:
                    model.addConstr(y[index + 1] <= Y[index] - self.h * self.arcs[route[index], route[index + 1]])

        # departure and arrival energy constraint
        for i in range(len(route)):
            if route[i] in self.stations + self.depot_start + self.depot_end:
                model.addConstr(y[i] <= Y[i])

        # solve the model and optimize
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return True
        if model.status == GRB.INFEASIBLE or model.status == GRB.UNBOUNDED:
            return False
        return False

    # notice: when using this function, the route must be feasible
    def time_extractor(self, route: List[str]) -> List[float]:
        # create the model first
        model = gp.Model("route_check_time_energy")

        # set the output flag as 0 to avoid outcome showing
        model.setParam('OutputFlag', 0)

        # create the decision variables, times and energy states
        t = []
        y = []
        Y = []
        for i in range(len(route)):
            t.append(model.addVar(lb=self.ready_time[route[i]], ub=self.due_date[route[i]], vtype=GRB.CONTINUOUS))
            Y.append(model.addVar(lb=0, ub=self.Q, vtype=GRB.CONTINUOUS))
            y.append(model.addVar(lb=0, ub=self.Q, vtype=GRB.CONTINUOUS))

        # create objective function, for checking feasibility, objective is set as 0
        model.setObjective(0, GRB.MINIMIZE)

        # create constraints to check the time and energy
        # time constraints
        for index in range(len(route)):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.stations:
                    model.addConstr(
                        t[index] + self.times[route[index], route[index + 1]] + self.g * (Y[index] - y[index]) <= t[
                            index + 1]
                    )
                else:
                    model.addConstr(
                        t[index] + self.times[route[index], route[index + 1]] + self.service_time[route[index]] <= t[
                            index + 1]
                    )

        # energy constraints
        for index in range(len(route)):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.clients:
                    model.addConstr(y[index + 1] <= y[index] - self.h * self.arcs[route[index], route[index + 1]])
                else:
                    model.addConstr(y[index + 1] <= Y[index] - self.h * self.arcs[route[index], route[index + 1]])

        # departure and arrival energy constraint
        for i in range(len(route)):
            if route[i] in self.stations + self.depot_start + self.depot_end:
                model.addConstr(y[i] <= Y[i])

        # solve the model and optimize
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return [time.x for time in t]
        raise Exception("This route is not feasible")

    def energy_extractor(self, route: List[str]) -> List[float]:
        # create the model first
        model = gp.Model("route_check_time_energy")

        # set the output flag as 0 to avoid outcome showing
        model.setParam('OutputFlag', 0)

        # create the decision variables, times and energy states
        t = []
        y = []
        Y = []
        for i in range(len(route)):
            t.append(model.addVar(lb=self.ready_time[route[i]], ub=self.due_date[route[i]], vtype=GRB.CONTINUOUS))
            Y.append(model.addVar(lb=0, ub=self.Q, vtype=GRB.CONTINUOUS))
            y.append(model.addVar(lb=0, ub=self.Q, vtype=GRB.CONTINUOUS))

        # create objective function, for checking feasibility, objective is set as 0
        model.setObjective(0, GRB.MINIMIZE)

        # create constraints to check the time and energy
        # time constraints
        for index in range(len(route)):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.stations:
                    model.addConstr(
                        t[index] + self.times[route[index], route[index + 1]] + self.g * (Y[index] - y[index]) <= t[
                            index + 1]
                    )
                else:
                    model.addConstr(
                        t[index] + self.times[route[index], route[index + 1]] + self.service_time[route[index]] <= t[
                            index + 1]
                    )

        # energy constraints
        for index in range(len(route)):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.clients:
                    model.addConstr(y[index + 1] <= y[index] - self.h * self.arcs[route[index], route[index + 1]])
                else:
                    model.addConstr(y[index + 1] <= Y[index] - self.h * self.arcs[route[index], route[index + 1]])

        # departure and arrival energy constraint
        for i in range(len(route)):
            if route[i] in self.stations + self.depot_start + self.depot_end:
                model.addConstr(y[i] <= Y[i])

        # solve the model and optimize
        model.optimize()

        # here for clients the departure energy should use the arrival energy
        # departure energy is the same as arrival energy
        # and for clients the departure energy decision variable is not included in the constraints, so missing value
        if model.status == GRB.OPTIMAL:
            return [arrival_energy.x for arrival_energy in y]
        raise Exception("This route is not feasible")

    def energy_extractor_departure(self, route: List[str]) -> List[float]:
        # create the model first
        model = gp.Model("route_check_time_energy")

        # set the output flag as 0 to avoid outcome showing
        model.setParam('OutputFlag', 0)

        # create the decision variables, times and energy states
        t = []
        y = []
        Y = []
        for i in range(len(route)):
            t.append(model.addVar(lb=self.ready_time[route[i]], ub=self.due_date[route[i]], vtype=GRB.CONTINUOUS))
            Y.append(model.addVar(lb=0, ub=self.Q, vtype=GRB.CONTINUOUS))
            y.append(model.addVar(lb=0, ub=self.Q, vtype=GRB.CONTINUOUS))

        # create objective function, for checking feasibility, objective is set as 0
        model.setObjective(0, GRB.MINIMIZE)

        # create constraints to check the time and energy
        # time constraints
        for index in range(len(route)):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.stations:
                    model.addConstr(
                        t[index] + self.times[route[index], route[index + 1]] + self.g * (Y[index] - y[index]) <= t[
                            index + 1]
                    )
                else:
                    model.addConstr(
                        t[index] + self.times[route[index], route[index + 1]] + self.service_time[route[index]] <= t[
                            index + 1]
                    )

        # energy constraints
        for index in range(len(route)):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.clients:
                    model.addConstr(y[index + 1] <= y[index] - self.h * self.arcs[route[index], route[index + 1]])
                else:
                    model.addConstr(y[index + 1] <= Y[index] - self.h * self.arcs[route[index], route[index + 1]])

        # departure and arrival energy constraint
        for i in range(len(route)):
            if route[i] in self.stations + self.depot_start + self.depot_end:
                model.addConstr(y[i] <= Y[i])

        # solve the model and optimize
        model.optimize()

        # here for clients the departure energy should use the arrival energy
        # departure energy is the same as arrival energy
        # and for clients the departure energy decision variable is not included in the constraints, so missing value
        if model.status == GRB.OPTIMAL:
            return [arrival_energy.x for arrival_energy in Y]
        raise Exception("This route is not feasible")

    def _time_window_penalty(self, arrival_time: float, node: str) -> float:
        """Calculate penalty for a single node's time window violation."""
        if arrival_time < self.ready_time[node]:
            return self.ready_time[node] - arrival_time  # Early arrival
        elif arrival_time > self.due_date[node]:
            return arrival_time - self.due_date[node]  # Late arrival
        return 0.0

    def _energy_violation_penalty(self, route: List[str]) -> float:
        """Calculate penalty for energy violations (if any)."""
        try:
            if self.energy(route):  # Feasible
                return 0.0
            # For infeasible routes, try to estimate violation severity
            energy_levels = self.energy_extractor(route)
            return max(0, -min(energy_levels))  # Penalize worst violation
        except Exception:
            # If energy extraction fails, return a large penalty
            return 1000.0
    
    def total_cost(
        self,
        route: List[str],
        time_window_weight: float = 80.0,
        energy_penalty_weight: float = 150.0,
        travel_time_weight: float = 2.5,
        vehicle_cost_weight: float = 200.0 
    ) -> float:
        """
        Calculate total cost for a route with comprehensive feasibility checking.
        Returns float('inf') for infeasible routes to prevent their acceptance.
        """
        # Check basic feasibility first
        if not self.time(route) or not self.energy(route):
            return float('inf')
        
        try:
            # Extract arrival times
            arrival_times = self.time_extractor(route)
            travel_time = arrival_times[-1] - arrival_times[0]
            
            # Calculate time window penalties
            time_penalty = sum(
                self._time_window_penalty(arrival_times[i], route[i])
                for i in range(len(route))
            )
            
            # Energy penalty (should be 0 for feasible routes)
            energy_penalty = self._energy_violation_penalty(route)
            
            # Calculate total cost
            total_cost = (
                travel_time * travel_time_weight
                + time_penalty * time_window_weight
                + energy_penalty * energy_penalty_weight
                + vehicle_cost_weight
            )
            
            # Ensure we don't return NaN or infinite values
            if math.isnan(total_cost) or math.isinf(total_cost):
                return float('inf')
                
            return total_cost
            
        except Exception as e:
            # If any extraction fails, the route is infeasible
            return float('inf')

    def _calculate_cost(self, routes: List[List[str]]) -> float:
        if not routes:
            print("No routes provided.")
            return float('inf')
        total_cost = 0.0
        for i, route in enumerate(routes):
            try:
                if len(route) < 2:
                    print(f"Route {i} too short: {route}")
                    return float('inf')
                route_cost = self.total_cost(
                    route,
                    time_window_weight=80.0,
                    energy_penalty_weight=150.0,
                    travel_time_weight=2.5,
                    vehicle_cost_weight=200.0
                )
                if math.isnan(route_cost) or math.isinf(route_cost):
                    print(f"Route {i} infeasible: {route}")
                    return float('inf')
                total_cost += route_cost
            except Exception as e:
                print(f"Exception in cost calculation for route {i}: {e}")
                return float('inf')
        return total_cost