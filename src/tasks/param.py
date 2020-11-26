"""Parameterization for distributions of nodes on [0,1]x[0,1] and utils."""

import math
import torch


def get_param_nodes(num_nodes, num_samples, seed, num_tiles, param):
    """Create collection of points distributed according to parameters.

    Performance notes: 1) We can usually assume that num_tiles is somewhat
    small. 2) There is probably a vectorized way to make this data generation
    faster, but it looks tricky to actually do well.

    Args:
        num_nodes (int): number of nodes per datapoint
        num_samples (int): number of datapoints
        seed (int): random seed
        num_tiles (int): number of tiles per side (for a total number of
            num_tiles^2 tiles in the parameterization)
        distrib (torch.Tensor): parameter describing distribution of data

    Returns:
        torch tensor of shape (num_samples, 2, num_nodes) with distribution of
        nodes according to param argument
    """
    # checking some preconditions
    assert isinstance(num_nodes, int)
    assert num_nodes > 0
    assert isinstance(num_samples, int)
    assert num_samples > 0
    assert isinstance(seed, int)
    assert isinstance(num_tiles, int)
    assert num_tiles > 0
    assert isinstance(param, torch.Tensor)
    assert tuple(param.size()) == (num_tiles ** 2,)
    for p in param:
        assert p >= 0 and p <= 1
    assert param.sum().item() == 1

    # setting seeds
    torch.manual_seed(seed)

    # some utility lists
    nonzero_indices = [i for i in range(len(param)) if param[i] > 0]
    nonzero_vals = [param[i].item() for i in nonzero_indices]

    # offests (where in the tile each node is located)
    offsets = torch.rand((num_samples, 2, num_nodes)) / num_tiles

    # generating random tile seeds (determines which tiles they land in)
    # can't think of a fast way to do this so we'll just iterate to get tiles
    tile_seeds = torch.rand((num_samples, num_nodes))
    tiles = torch.zeros((num_samples, 2, num_nodes))
    for i in range(num_samples):
        for j in range(num_nodes):
            tile_seed = tile_seeds[i, j]

            # generate a point
            tiles[i, 0, j], tiles[i, 1, j] = _get_tile(
                tile_seed, num_tiles, nonzero_indices, nonzero_vals
            )

    return tiles + offsets


def _get_tile(tile_seed, num_tiles, nonzero_indices, nonzero_vals):
    """Create on point from the seed distribution.

    Assumes parameters have been checked. Note that the element at the zeroth
    index of param indicates the probability that the tile selected is the
    top left tile. Example arrangement for num_tiles = 4:

    -----------------
    | 0 | 1 | 2 | 3 |
    -----------------
    | 4 | 5 | 6 | 7 |
    -----------------
    | 8 | 9 | 10| 11|
    -----------------
    | 12| 13| 14| 15|
    -----------------

    This can probably be done in torch/numpy, but I'm in a hurry and will do
    this in raw python.

    Args:
        tile_seed (int): randomly generated value of current seed (generated
            uniformly in [0,1))
        num_tiles (int): number of tiles per side (for a total number of
            num_tiles^2 tiles in the parameterization)
        nonzero_indices (List[int]): indices of param that have nonzero value
        nonzero_vals (List[float]): values at nonzero indices of param

    Returns:
        a tuple (x,y) representing the bottom left corner of the node
            specified node seed
    """
    # getting tile index
    cum_sum = 0
    tile_index = -1
    for i in range(len(nonzero_vals)):
        cum_sum += nonzero_vals[i]
        if tile_seed <= cum_sum:
            tile_index = nonzero_indices[i]
            break

    # getting coordinates of bottom left corner of tile
    x_pos = (tile_index % num_tiles) * (1 / num_tiles)
    y_pos = 1 - (math.floor(tile_index / num_tiles) + 1) * (1 / num_tiles)

    return x_pos, y_pos


# Spencer add your things here


def get_uniform_param(num_tiles):
    """Return param value that corresponds to uniform distribution.

    Args:
        num_tiles (int): number of tiles per side of unit square

    Return:
        torch tensor of shape (num_tiles ** 2, ) with nonnegeative elements
            which sum to 1
    """
    param = torch.ones((num_tiles ** 2,))
    return param / param.sum()


def get_line_param(slope, intercept, num_tiles):
    """Return param value that roughly corresponds to a line.

    Args:
        slope (float): slope of line
        intercept (float): intercept of line
        num_tiles (int): number of tiles per side of unit square

    Return:
        torch tensor of shape (num_tiles ** 2, ) with nonnegeative elements
            which sum to 1
    """
    raise NotImplementedError


def get_crossing_lines_param(slope_1, intercept_1, slope_2, intercept_2, num_tiles):
    """Return param value that roughly corresponds to two crossing lines.

    Args:
        slope_1 (float): slope of line 1
        intercept_1 (float): intercept of line 1
        slope_2 (float): slope of line 2
        intercept_2 (float): intercept of line 2
        num_tiles (int): number of tiles per side of unit square

    Return:
        torch tensor of shape (num_tiles ** 2, ) with nonnegeative elements
            which sum to 1
    """
    raise NotImplementedError


def get_circle_param(center, radius, num_tiles):
    """Return param value that roughly corresponds to a circle.

    Args:
        center (Tuple[float]): center of the circle (specified as (x,y))
        radius (float): radius of the circle
        num_tiles (int): number of tiles per side of unit square

    Return:
        torch tensor of shape (num_tiles ** 2, ) with nonnegeative elements
            which sum to 1
    """
    raise NotImplementedError


# debug functions


def _validate_param(num_tiles, param):
    """Checks param for obvious bugs."""
    assert isinstance(param, torch.Tensor)
    assert tuple(param.size()) == (num_tiles ** 2,)
    for p in param:
        assert p >= 0 and p <= 1
    assert param.sum().item() == 1


def _visualize_param(num_tiles, param):
    """Plot nodes drawn from param! For debug use only."""
    import numpy as np
    import matplotlib.pyplot as plt

    nodes = get_param_nodes(10, 10, 12345, num_tiles, param)

    x = nodes[:, 0, :].flatten().numpy()
    y = nodes[:, 1, :].flatten().numpy()

    plt.scatter(x, y)
    plt.show()


if __name__ == "__main__":
    # debug flags
    debugging_get_param_nodes = False
    debugging_get_uniform_param = False
    debugging_get_line_param = False
    debugging_get_crossing_lines_param = False
    debugging_get_circle_param = False

    if debugging_get_uniform_param:
        c_num_tiles = 8
        c_param = get_uniform_param(c_num_tiles)
        _validate_param(c_num_tiles, c_param)
        _visualize_param(c_num_tiles, c_param)

    if debugging_get_line_param:
        c_num_tiles = 8
        c_slope = 1
        c_intercept = 0.2
        c_param = get_line_param(c_slope, c_intercept, c_num_tiles)
        _validate_param(c_num_tiles, c_param)
        _visualize_param(c_num_tiles, c_param)

    if debugging_get_crossing_lines_param:
        c_num_tiles = 8
        c_slope_1 = 1
        c_intercept_1 = 0.2
        c_slope_2 = -0.5
        c_intercept_2 = 1
        c_param = get_crossing_lines_param(
            c_slope_1, c_intercept_1, c_slope_2, c_intercept_2, c_num_tiles
        )
        _validate_param(c_num_tiles, c_param)
        _visualize_param(c_num_tiles, c_param)

    if debugging_get_circle_param:
        c_num_tiles = 8
        c_center = (0.25, 0.25)
        c_radius = (0.5, 0.5)
        c_param = get_circle_param(c_center, c_radius, c_num_tiles)
        _validate_param(c_num_tiles, c_param)
        _visualize_param(c_num_tiles, c_param)

    if debugging_get_param_nodes:
        c_num_nodes = 20
        c_num_samples = 100
        c_seed = 12345
        c_num_tiles = 8
        c_param = torch.tensor([1] + [0 for i in range(63)])

        import time

        start = time.time()

        nodes = get_param_nodes(
            c_num_nodes, c_num_samples, c_seed, c_num_tiles, c_param
        )

        end = time.time()
        print(nodes)
        print()
        print(end - start)
