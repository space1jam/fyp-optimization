# MIPCheck class using PuLP (open-source solver) instead of Gurobi
# This is a drop-in replacement for the original Gurobi-based MIPCheck
# Requires: pip install pulp

import pulp
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
        new_times = self.parameters["times"]
        for i in self.all_nodes:
            for j in self.all_nodes:
                if i != j:
                    if random.random() < p:
                        stochastic = np.random.normal(0, n*self.std, 1)[0]
                        if new_times[i,j] + stochastic > 0:
                            new_times[i,j] = new_times[i,j] + stochastic
        self.times = new_times

    def _solve_lp(self, prob) -> int:
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        return status

    def time_energy(self, route: List[str]) -> bool:
        prob = pulp.LpProblem("route_check_time_energy", pulp.LpMinimize)
        n = len(route)
        t = [pulp.LpVariable(f"t_{i}", lowBound=self.ready_time[route[i]], upBound=self.due_date[route[i]]) for i in range(n)]
        Y = [pulp.LpVariable(f"Y_{i}", lowBound=0, upBound=self.Q) for i in range(n)]
        y = [pulp.LpVariable(f"y_{i}", lowBound=0, upBound=self.Q) for i in range(n)]
        prob += 0  # Objective
        for index in range(n):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.stations:
                    prob += (
                        t[index] + self.times[route[index], route[index+1]] + self.g*(Y[index]-y[index]) <= t[index+1]
                    )
                else:
                    prob += (
                        t[index] + self.times[route[index], route[index+1]] + self.service_time[route[index]] <= t[index+1]
                    )
        for index in range(n):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.clients:
                    prob += (y[index+1] <= y[index] - self.h*self.arcs[route[index], route[index+1]])
                else:
                    prob += (y[index+1] <= Y[index] - self.h*self.arcs[route[index], route[index+1]])
        for i in range(n):
            if route[i] in self.stations + self.depot_start + self.depot_end:
                prob += (y[i] <= Y[i])
        status = self._solve_lp(prob)
        return pulp.LpStatus[status] == 'Optimal'

    def time(self, route: List[str]) -> bool:
        prob = pulp.LpProblem("route_check_time", pulp.LpMinimize)
        n = len(route)
        t = [pulp.LpVariable(f"t_{i}", lowBound=self.ready_time[route[i]], upBound=self.due_date[route[i]]) for i in range(n)]
        Y = [pulp.LpVariable(f"Y_{i}", lowBound=0, upBound=self.Q) for i in range(n)]
        y = [pulp.LpVariable(f"y_{i}", lowBound=0, upBound=self.Q) for i in range(n)]
        prob += 0
        for index in range(n):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.stations:
                    prob += (
                        t[index] + self.times[route[index], route[index+1]] <= t[index+1]
                    )
                else:
                    prob += (
                        t[index] + self.times[route[index], route[index+1]] + self.service_time[route[index]] <= t[index+1]
                    )
        status = self._solve_lp(prob)
        return pulp.LpStatus[status] == 'Optimal'

    def energy(self, route: List[str]) -> bool:
        prob = pulp.LpProblem("route_check_energy", pulp.LpMinimize)
        n = len(route)
        t = [pulp.LpVariable(f"t_{i}", lowBound=self.ready_time[route[i]], upBound=self.due_date[route[i]]) for i in range(n)]
        Y = [pulp.LpVariable(f"Y_{i}", lowBound=0, upBound=self.Q) for i in range(n)]
        y = [pulp.LpVariable(f"y_{i}", lowBound=0, upBound=self.Q) for i in range(n)]
        prob += 0
        for index in range(n):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.clients:
                    prob += (y[index+1] <= y[index] - self.h*self.arcs[route[index], route[index+1]])
                else:
                    prob += (y[index+1] <= Y[index] - self.h*self.arcs[route[index], route[index+1]])
        for i in range(n):
            if route[i] in self.stations + self.depot_start + self.depot_end:
                prob += (y[i] <= Y[i])
        status = self._solve_lp(prob)
        return pulp.LpStatus[status] == 'Optimal'

    def time_extractor(self, route: List[str]) -> List[float]:
        prob = pulp.LpProblem("route_check_time_energy", pulp.LpMinimize)
        n = len(route)
        t = [pulp.LpVariable(f"t_{i}", lowBound=self.ready_time[route[i]], upBound=self.due_date[route[i]]) for i in range(n)]
        Y = [pulp.LpVariable(f"Y_{i}", lowBound=0, upBound=self.Q) for i in range(n)]
        y = [pulp.LpVariable(f"y_{i}", lowBound=0, upBound=self.Q) for i in range(n)]
        prob += 0
        for index in range(n):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.stations:
                    prob += (
                        t[index] + self.times[route[index], route[index+1]] + self.g*(Y[index]-y[index]) <= t[index+1]
                    )
                else:
                    prob += (
                        t[index] + self.times[route[index], route[index+1]] + self.service_time[route[index]] <= t[index+1]
                    )
        for index in range(n):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.clients:
                    prob += (y[index+1] <= y[index] - self.h*self.arcs[route[index], route[index+1]])
                else:
                    prob += (y[index+1] <= Y[index] - self.h*self.arcs[route[index], route[index+1]])
        for i in range(n):
            if route[i] in self.stations + self.depot_start + self.depot_end:
                prob += (y[i] <= Y[i])
        status = self._solve_lp(prob)
        if pulp.LpStatus[status] == 'Optimal':
            return [float(tvar.varValue) for tvar in t if tvar.varValue is not None]
        raise Exception("This route is not feasible")

    def energy_extractor(self, route: List[str]) -> List[float]:
        prob = pulp.LpProblem("route_check_time_energy", pulp.LpMinimize)
        n = len(route)
        t = [pulp.LpVariable(f"t_{i}", lowBound=self.ready_time[route[i]], upBound=self.due_date[route[i]]) for i in range(n)]
        Y = [pulp.LpVariable(f"Y_{i}", lowBound=0, upBound=self.Q) for i in range(n)]
        y = [pulp.LpVariable(f"y_{i}", lowBound=0, upBound=self.Q) for i in range(n)]
        prob += 0
        for index in range(n):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.stations:
                    prob += (
                        t[index] + self.times[route[index], route[index+1]] + self.g*(Y[index]-y[index]) <= t[index+1]
                    )
                else:
                    prob += (
                        t[index] + self.times[route[index], route[index+1]] + self.service_time[route[index]] <= t[index+1]
                    )
        for index in range(n):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.clients:
                    prob += (y[index+1] <= y[index] - self.h*self.arcs[route[index], route[index+1]])
                else:
                    prob += (y[index+1] <= Y[index] - self.h*self.arcs[route[index], route[index+1]])
        for i in range(n):
            if route[i] in self.stations + self.depot_start + self.depot_end:
                prob += (y[i] <= Y[i])
        status = self._solve_lp(prob)
        if pulp.LpStatus[status] == 'Optimal':
            return [float(yvar.varValue) for yvar in y if yvar.varValue is not None]
        raise Exception("This route is not feasible")

    def energy_extractor_departure(self, route: List[str]) -> List[float]:
        prob = pulp.LpProblem("route_check_time_energy", pulp.LpMinimize)
        n = len(route)
        t = [pulp.LpVariable(f"t_{i}", lowBound=self.ready_time[route[i]], upBound=self.due_date[route[i]]) for i in range(n)]
        Y = [pulp.LpVariable(f"Y_{i}", lowBound=0, upBound=self.Q) for i in range(n)]
        y = [pulp.LpVariable(f"y_{i}", lowBound=0, upBound=self.Q) for i in range(n)]
        prob += 0
        for index in range(n):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.stations:
                    prob += (
                        t[index] + self.times[route[index], route[index+1]] + self.g*(Y[index]-y[index]) <= t[index+1]
                    )
                else:
                    prob += (
                        t[index] + self.times[route[index], route[index+1]] + self.service_time[route[index]] <= t[index+1]
                    )
        for index in range(n):
            i = route[index]
            if i == "D0_end":
                break
            else:
                if i in self.clients:
                    prob += (y[index+1] <= y[index] - self.h*self.arcs[route[index], route[index+1]])
                else:
                    prob += (y[index+1] <= Y[index] - self.h*self.arcs[route[index], route[index+1]])
        for i in range(n):
            if route[i] in self.stations + self.depot_start + self.depot_end:
                prob += (y[i] <= Y[i])
        status = self._solve_lp(prob)
        if pulp.LpStatus[status] == 'Optimal':
            return [float(Yvar.varValue) for Yvar in Y if Yvar.varValue is not None]
        raise Exception("This route is not feasible")

    def _time_window_penalty(self, arrival_time: float, node: str) -> float:
        if arrival_time < self.ready_time[node]:
            return self.ready_time[node] - arrival_time
        elif arrival_time > self.due_date[node]:
            return arrival_time - self.due_date[node]
        return 0.0

    def _energy_violation_penalty(self, route: List[str]) -> float:
        try:
            if self.energy(route):
                return 0.0
            energy_levels = self.energy_extractor(route)
            return max(0, -min(energy_levels))
        except Exception:
            return 1000.0

    def total_cost(
        self,
        route: List[str],
        time_window_weight: float = 80.0,
        energy_penalty_weight: float = 150.0,
        travel_time_weight: float = 2.5,
        vehicle_cost_weight: float = 200.0
    ) -> float:
        if not self.time(route) or not self.energy(route):
            return float('inf')
        try:
            arrival_times = self.time_extractor(route)
            travel_time = arrival_times[-1] - arrival_times[0]
            time_penalty = sum(
                self._time_window_penalty(arrival_times[i], route[i])
                for i in range(len(route))
            )
            energy_penalty = self._energy_violation_penalty(route)
            total_cost = (
                travel_time * travel_time_weight
                + time_penalty * time_window_weight
                + energy_penalty * energy_penalty_weight
                + vehicle_cost_weight
            )
            if math.isnan(total_cost) or math.isinf(total_cost):
                return float('inf')
            return total_cost
        except Exception as e:
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