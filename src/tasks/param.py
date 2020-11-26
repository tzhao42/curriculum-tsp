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
    assert abs(param.sum().item() - 1.0) < 1e-5

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


def _set_tile_val(row, col, param, val, num_tiles=None):
    """Helper function."""
    if num_tiles == None:
        num_tiles = int(math.sqrt(len(param)))
    param[row * num_tiles + col] = val


def _normalize_param(param):
    """Helper function."""
    param /= param.sum()


def get_uniform_param(num_tiles):
    """Return param value that corresponds to uniform distribution.

    Args:
        num_tiles (int): number of tiles per side of unit square

    Return:
        torch tensor of shape (num_tiles ** 2, ) with nonnegeative elements
            which sum to 1
    """
    param = torch.ones((num_tiles ** 2,))
    _normalize_param(param)
    return param


def get_up_line_param(num_tiles):
    """Return param value that corresponds to up-line distribution.

    Args:
        num_tiles (int): number of tiles per side of unit square

    Return:
        torch tensor of shape (num_tiles ** 2, ) with nonnegeative elements
            which sum to 1
    """
    param = torch.zeros((num_tiles ** 2,))
    for i in range(0, num_tiles):
        _set_tile_val(num_tiles - 1 - i, i, param, 1)
    _normalize_param(param)
    return param


def get_down_line_param(num_tiles):
    """Return param value that corresponds to down-line distribution.

    Args:
        num_tiles (int): number of tiles per side of unit square

    Return:
        torch tensor of shape (num_tiles ** 2, ) with nonnegeative elements
            which sum to 1
    """
    param = torch.zeros((num_tiles ** 2,))
    for i in range(0, num_tiles):
        _set_tile_val(i, i, param, 1)
    _normalize_param(param)
    return param


def get_x_shape_param(num_tiles):
    """Return param value that corresponds to x-shape distribution.

    Args:
        num_tiles (int): number of tiles per side of unit square

    Return:
        torch tensor of shape (num_tiles ** 2, ) with nonnegeative elements
            which sum to 1
    """
    param = torch.zeros((num_tiles ** 2,))
    for i in range(0, num_tiles):
        _set_tile_val(num_tiles - 1 - i, i, param, 1)
    for i in range(0, num_tiles):
        _set_tile_val(i, i, param, 1)
    _normalize_param(param)
    return param


def get_horiz_param(num_tiles):
    """Return param value that corresponds to horiz distribution.

    Args:
        num_tiles (int): number of tiles per side of unit square

    Return:
        torch tensor of shape (num_tiles ** 2, ) with nonnegeative elements
            which sum to 1
    """
    param = torch.zeros((num_tiles ** 2,))
    row = num_tiles // 2
    for i in range(0, num_tiles):
        _set_tile_val(row, i, param, 1)
    _normalize_param(param)
    return param


def get_vert_param(num_tiles):
    """Return param value that corresponds to vert distribution.

    Args:
        num_tiles (int): number of tiles per side of unit square

    Return:
        torch tensor of shape (num_tiles ** 2, ) with nonnegeative elements
            which sum to 1
    """
    param = torch.zeros((num_tiles ** 2,))
    col = num_tiles // 2
    for i in range(0, num_tiles):
        _set_tile_val(i, col, param, 1)
    _normalize_param(param)
    return param


def get_plus_param(num_tiles):
    """Return param value that corresponds to plus distribution.

    Args:
        num_tiles (int): number of tiles per side of unit square

    Return:
        torch tensor of shape (num_tiles ** 2, ) with nonnegeative elements
            which sum to 1
    """
    param = torch.zeros((num_tiles ** 2,))
    row = num_tiles // 2
    for i in range(0, num_tiles):
        _set_tile_val(row, i, param, 1)
    col = num_tiles // 2
    for i in range(0, num_tiles):
        _set_tile_val(i, col, param, 1)

    _normalize_param(param)
    return param


def get_circle_param(num_tiles):
    """Return param value that corresponds to circle distribution.

    Args:
        num_tiles (int): number of tiles per side of unit square

    Return:
        torch tensor of shape (num_tiles ** 2, ) with nonnegeative elements
            which sum to 1
    """
    param = torch.zeros((num_tiles ** 2,))
    # so uh this prly isn't the best idk
    start_col = [0, 0, num_tiles - 1, num_tiles - 1]
    start_row = [
        (num_tiles - 1) // 2,
        num_tiles // 2,
        (num_tiles - 1) // 2,
        num_tiles // 2,
    ]
    dr = [-1, 1, -1, 1]
    dc = [1, 1, -1, -1]
    for i in range(0, 4):
        cur_row = start_row[i]
        cur_col = start_col[i]
        while (
            (cur_row >= 0)
            and (cur_row < num_tiles)
            and (cur_col >= 0)
            and (cur_col < num_tiles)
        ):
            _set_tile_val(cur_row, cur_col, param, 1)
            cur_row += dr[i]
            cur_col += dc[i]
    _normalize_param(param)
    return param


def get_border_param(num_tiles):
    """Return param value that corresponds to border distribution.

    Args:
        num_tiles (int): number of tiles per side of unit square

    Return:
        torch tensor of shape (num_tiles ** 2, ) with nonnegeative elements
            which sum to 1
    """
    param = torch.zeros((num_tiles ** 2,))
    for row in range(0, num_tiles):
        for col in range(0, num_tiles):
            good_row = (row == 0) or (row == num_tiles - 1)
            good_col = (col == 0) or (col == num_tiles - 1)
            if good_row or good_col:
                _set_tile_val(row, col, param, 1)
    _normalize_param(param)
    return param


def get_medium_pair_param(num_tiles, width=None):
    """Return param value that corresponds to medium-pair distribution.

    Args:
        num_tiles (int): number of tiles per side of unit square
        width (int): controls the width of boxes in the distribution

    Return:
        torch tensor of shape (num_tiles ** 2, ) with nonnegeative elements
            which sum to 1
    """
    if width == None:
        # idk how great this is
        width = num_tiles // 2
        if width - 2 >= 3:
            width -= 2
        elif width - 1 >= 2:
            width -= 1
        width = max(width, 1)
    width = min(width, num_tiles)
    param = torch.zeros((num_tiles ** 2,))
    for i in range(0, width):
        for j in range(0, width):
            _set_tile_val(i, num_tiles - j - 1, param, 1)
            _set_tile_val(num_tiles - 1 - i, j, param, 1)
    _normalize_param(param)
    return param


def get_tiny_pair_param(num_tiles, width=None):
    """Return param value that corresponds to tiny-pair distribution.

    Args:
        num_tiles (int): number of tiles per side of unit square
        width (int): controls the width of boxes in the distribution

    Return:
        torch tensor of shape (num_tiles ** 2, ) with nonnegeative elements
            which sum to 1
    """
    if width == None:
        width = 1
    width = min(width, num_tiles)
    param = torch.zeros((num_tiles ** 2,))
    for i in range(0, width):
        for j in range(0, width):
            _set_tile_val(i, num_tiles - j - 1, param, 1)
            _set_tile_val(num_tiles - 1 - i, j, param, 1)
    _normalize_param(param)
    return param


def get_tiny_quad_param(num_tiles, width=None):
    """Return param value that corresponds to tiny-qquad distribution.

    Args:
        num_tiles (int): number of tiles per side of unit square
        width (int): controls the width of boxes in the distribution

    Return:
        torch tensor of shape (num_tiles ** 2, ) with nonnegeative elements
            which sum to 1
    """
    if width == None:
        width = 1
    width = min(width, num_tiles)
    param = torch.zeros((num_tiles ** 2,))
    for i in range(0, width):
        for j in range(0, width):
            _set_tile_val(i, num_tiles - j - 1, param, 1)
            _set_tile_val(num_tiles - 1 - i, j, param, 1)
            _set_tile_val(i, j, param, 1)
            _set_tile_val(num_tiles - 1 - i, num_tiles - 1 - j, param, 1)
    _normalize_param(param)
    return param


# debug functions


def _validate_param(num_tiles, param):
    """Checks param for obvious bugs."""
    assert isinstance(param, torch.Tensor)
    assert tuple(param.size()) == (num_tiles ** 2,)
    for p in param:
        assert p >= 0 and p <= 1
    assert abs(param.sum().item() - 1.0) < 1e-5


def _visualize_param(num_tiles, param):
    """Plot nodes drawn from param! For debug use only."""
    import numpy as np
    import matplotlib.pyplot as plt

    nodes = get_param_nodes(10, 10, 12345, num_tiles, param)

    x = nodes[:, 0, :].flatten().numpy()
    y = nodes[:, 1, :].flatten().numpy()

    plt.scatter(x, y)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    # debug flags
    debugging_get_param_nodes = False
    debugging_get_uniform_param = False
    debugging_get_up_line_param = False
    debugging_get_down_line_param = False
    debugging_get_x_shape_param = False
    debugging_get_horiz_param = False
    debugging_get_vert_param = False
    debugging_get_line_param = False
    debugging_get_plus_param = False
    debugging_get_circle_param = False
    debugging_get_border_param = False
    debugging_get_medium_pair_param = False
    debugging_get_tiny_pair_param = False
    debugging_get_tiny_quad_param = False
    debugging_get_crossing_lines_param = False

    if debugging_get_uniform_param:
        c_num_tiles = 8
        c_param = get_uniform_param(c_num_tiles)
        _validate_param(c_num_tiles, c_param)
        _visualize_param(c_num_tiles, c_param)

    if debugging_get_up_line_param:
        c_num_tiles = 8
        c_param = get_up_line_param(c_num_tiles)
        _validate_param(c_num_tiles, c_param)
        _visualize_param(c_num_tiles, c_param)

    if debugging_get_down_line_param:
        c_num_tiles = 8
        c_param = get_down_line_param(c_num_tiles)
        _validate_param(c_num_tiles, c_param)
        _visualize_param(c_num_tiles, c_param)

    if debugging_get_x_shape_param:
        c_num_tiles = 8
        c_param = get_x_shape_param(c_num_tiles)
        _validate_param(c_num_tiles, c_param)
        _visualize_param(c_num_tiles, c_param)

    if debugging_get_horiz_param:
        c_num_tiles = 8
        c_param = get_horiz_param(c_num_tiles)
        _validate_param(c_num_tiles, c_param)
        _visualize_param(c_num_tiles, c_param)

    if debugging_get_vert_param:
        c_num_tiles = 8
        c_param = get_vert_param(c_num_tiles)
        _validate_param(c_num_tiles, c_param)
        _visualize_param(c_num_tiles, c_param)

    if debugging_get_plus_param:
        c_num_tiles = 8
        c_param = get_plus_param(c_num_tiles)
        _validate_param(c_num_tiles, c_param)
        _visualize_param(c_num_tiles, c_param)

    if debugging_get_circle_param:
        c_num_tiles = 8
        c_param = get_circle_param(c_num_tiles)
        _validate_param(c_num_tiles, c_param)
        _visualize_param(c_num_tiles, c_param)

    if debugging_get_border_param:
        c_num_tiles = 8
        c_param = get_border_param(c_num_tiles)
        _validate_param(c_num_tiles, c_param)
        _visualize_param(c_num_tiles, c_param)

    if debugging_get_medium_pair_param:
        c_num_tiles = 8
        c_param = get_medium_pair_param(c_num_tiles)
        _validate_param(c_num_tiles, c_param)
        _visualize_param(c_num_tiles, c_param)

    if debugging_get_tiny_pair_param:
        c_num_tiles = 8
        c_param = get_tiny_pair_param(c_num_tiles)
        _validate_param(c_num_tiles, c_param)
        _visualize_param(c_num_tiles, c_param)

    if debugging_get_tiny_quad_param:
        c_num_tiles = 8
        c_param = get_tiny_quad_param(c_num_tiles)
        _validate_param(c_num_tiles, c_param)
        _visualize_param(c_num_tiles, c_param)

    if debugging_get_uniform_param:
        c_num_tiles = 8
        c_param = get_uniform_param(c_num_tiles)
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
