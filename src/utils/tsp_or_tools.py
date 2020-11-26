"""Utilities wrapping google OR tools for tsp."""

import math

import torch
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def create_data_model(locs, factor):
    """Stores the data for the problem, distances scaled up by factor.

    Args:
        locs: torch tensor of size (2,n), where n is the number of
            points
    """
    locs = torch.transpose(locs, 0, 1)
    dists = torch.cdist(locs, locs) * factor

    data = {}
    data["locs"] = locs
    data["factor"] = factor
    data["distance_matrix"] = dists
    data["max_travel_distance"] = int(dists.sum().item()) + 1
    data["num_vehicles"] = 1
    data["depot"] = 0
    return data


def get_route_distance(data, manager, routing, solution):
    """Returns non-scaled distance of route."""
    index = routing.Start(0)
    route_distance = 0
    while not routing.IsEnd(index):
        previous_index = index
        index = solution.Value(routing.NextVar(index))

        previous_node = manager.IndexToNode(previous_index)
        node = manager.IndexToNode(index)
        route_distance += (
            data["distance_matrix"][previous_node][node].item()
            / data["factor"]
        )
    return route_distance


def get_or_tsp(locs, timeout):
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model(locs, factor=1000)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        data["distance_matrix"].size(0), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        data["max_travel_distance"],  # vehicle maximum travel distance
        True,  # start cumul to zero (vehicles start at the same time)
        dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(
        100
    )  # not sure what this does

    # Setting guided local search heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = timeout
    search_parameters.log_search = False
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        return get_route_distance(data, manager, routing, solution)
    else:
        raise ValueError("Google OR tools failed.")


def get_batched_or_tsp(locs, timeout):
    """Compute near-optimal tsp route on batch.

    Args:
        locs: torch tensor of size (b,2,n), where b is the number of
            instances in the batch and n is the number of points
    Returns:
        torch tensor of size (b,) representing optimal routes on each instance
    """
    # run each batch
    tour_lengths = []
    for i in range(locs.size(0)):
        curr_instance = locs[i]
        tour_length = get_or_tsp(curr_instance, timeout)
        tour_lengths.append(tour_length)

    # there are bugs in get_or_tsp :(
    return torch.tensor(tour_lengths)
