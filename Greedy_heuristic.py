# -*- coding: utf-8 -*-
import numpy as np

def greedy_heuristic(distance_matrix, num_vehicles):
    num_locations = len(distance_matrix)
    routes = [[] for _ in range(num_vehicles)]
    visited = set()
    visited.add(0)  # Assuming the depot is at index 0

    for route in routes:   # Initialize routes with the depot
        route.append(0)

    # Assign locations to vehicles
    while len(visited) < num_locations:
        for route in routes:
            if len(visited) == num_locations:
                break
            last_location = route[-1]
            closest_distance = float('inf')
            closest_location = None
            for i in range(num_locations):
                if i not in visited and distance_matrix[last_location][i] < closest_distance:
                    closest_distance = distance_matrix[last_location][i]
                    closest_location = i
            route.append(closest_location)
            visited.add(closest_location)

    for route in routes:
        route.append(0)

    return routes

def calculate_total_distance(routes, distance_matrix):
    total_distance = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i+1]]
    return total_distance

