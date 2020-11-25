"""Parameterization for collection of points on [0,1]x[0,1] square."""

import math
import torch

def gen_param_points(num_nodes, num_samples, seed, num_tiles, distrib):
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
        nodes according to distrib argument
    """
    # checking some preconditions
    assert isinstance(num_nodes, int)
    assert num_nodes > 0
    assert isinstance(num_samples, int)
    assert num_samples > 0
    assert isinstance(seed, int)
    assert isinstance(num_tiles, int)
    assert num_tiles > 0
    assert isinstance(distrib, torch.Tensor)
    assert tuple(distrib.size()) == (num_tiles ** 2, )
    for p in distrib:
        assert p >= 0 and p <= 1
    assert distrib.sum().item() == 1

    # setting seeds
    torch.manual_seed(seed)

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
            tiles[i, 0, j], tiles[i, 1, j] = _get_tile(tile_seed, num_tiles, distrib)

    return tiles + offsets


def _get_tile(tile_seed, num_tiles, distrib):
    """Create on point from the seed distribution.

    Assumes parameters have been checked. Note that the element at the zeroth
    index of distrib indicates the probability that the tile selected is the
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
        distrib (torch.Tensor): parameter describing distribution of data

    Returns:
        a tuple (x,y) representing the bottom left corner of the node
            specified node seed
    """
    nonzero_indices = [i for i in range(len(distrib)) if distrib[i] > 0]
    nonzero_vals = [distrib[i].item() for i in nonzero_indices]

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
    
