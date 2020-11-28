"""Parameterization for distributions of nodes on [0,1]x[0,1] and utils."""

import os
import math
import numpy as np
import torch
import multiprocessing as mp


def get_param_nodes(num_nodes, num_samples, num_tiles, param, num_processes):
    """Create collection of points distributed according to parameters.

    Performance notes:
        1) We can usually assume that num_tiles is somewhat small.
        2) There is probably a vectorized way to make this data generation
        faster, but it looks tricky to actually do well, so we'll use
        multiprocessing instead.
    Args:
        num_nodes (int): number of nodes per datapoint
        num_samples (int): number of datapoints
        num_tiles (int): number of tiles per side (for a total number of
            num_tiles^2 tiles in the parameterization)
        param (torch.Tensor): parameter describing distribution of data
        num_processes (int): number of processes this should spawn (should be
            at least number of cpus)
    Returns:
        torch tensor of shape (num_samples, 2, num_nodes) with distribution of
        nodes according to param argument
    """
    # print(f"Input: {num_nodes}, {num_samples}, {num_tiles}, {param}, {num_processes}")

    # checking some preconditions
    assert isinstance(num_nodes, int)
    assert num_nodes > 0
    assert isinstance(num_samples, int)
    assert num_samples > 0
    assert isinstance(num_tiles, int)
    assert num_tiles > 0
    _validate_param(num_tiles, param)
    assert isinstance(num_processes, int)
    assert num_processes > 0

    # some utility lists
    # since param is fairly short, we can do this in raw python
    nonzero_indices = [i for i in range(len(param)) if param[i] > 0]
    nonzero_vals = [param[i].item() for i in nonzero_indices]

    # precomputations for spencer's magic
    balanced = _balanced_probabilities(nonzero_indices, nonzero_vals)
    x_pos = list()
    y_pos = list()
    for i in range(0, len(param)):
        x_pos.append((i % num_tiles) * (1 / num_tiles))
        y_pos.append(1 - (math.floor(i / num_tiles) + 1) * (1 / num_tiles))

    # generating random tile seeds (determines which tiles they land in)
    # can't think of a fast way to do this so we'll just iterate to get tiles
    # generating random tensors
    offsets = torch.rand((num_samples, 2, num_nodes)) / num_tiles
    ind_seeds = torch.randint(0, len(balanced), (num_samples, num_nodes))
    val_seeds = torch.rand((num_samples, num_nodes))

    # calculating start/end values for each batch
    sample_ranges = []
    range_size = math.floor(num_samples / num_processes)
    for i in range(num_processes - 1):
        curr_range = (i * range_size, (i + 1) * range_size)
        sample_ranges.append(curr_range)
    # getting any stragglers
    curr_range = (
        (num_processes - 1) * range_size,
        max(num_processes * range_size, num_samples),
    )
    sample_ranges.append(curr_range)

    # using torch's wrapped multiprocessing
    # need to split it up pretty huge
    with mp.Pool(processes=num_processes) as pool:
        results = [
            pool.apply_async(
                _get_param_nodes_worker,
                args=(
                    num_nodes,
                    num_samples,
                    sample_ranges[i][0],
                    sample_ranges[i][1],
                    balanced,
                    x_pos,
                    y_pos,
                    ind_seeds,
                    val_seeds,
                ),
            )
            for i in range(num_processes)
        ]
        batches = [p.get() for p in results]

    tiles = torch.cat(batches, dim=0)
    ret = tiles + offsets

    return ret


def _get_param_nodes_worker(
    num_nodes,
    num_samples,
    start_samples,
    end_samples,
    balanced,
    x_pos,
    y_pos,
    ind_seeds,
    val_seeds,
):
    """Worker for generating param nodes."""
    tiles = torch.zeros((end_samples - start_samples, 2, num_nodes))

    for i in range(start_samples, end_samples):
        for j in range(num_nodes):
            tile_index = -1
            if val_seeds[i, j] < balanced[ind_seeds[i, j]][2]:
                tile_index = balanced[ind_seeds[i, j]][0]
            else:
                tile_index = balanced[ind_seeds[i, j]][1]
            tiles[i - start_samples, 0, j] = x_pos[tile_index]
            tiles[i - start_samples, 1, j] = y_pos[tile_index]

    return tiles


# param generating functions


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


# helper functions


def _balanced_probabilities(indices, vals):
    """Creates a list of length len(indices), where each entry
    has a pair of two elements, and the sum of probability mass in each
    index is 1/len(indices). This allows O(1) sampling from a distribution.
    Pretty cool!
    Args:
        indices (int): indices of each probability state
        vals (float): probability mass for corresponding state (sums to 1)

    Returns:
        list of length len(indices). In the form (state1, state2, prob1)
    select state1 with probability prob1/len(indices), state2 w.p. (1-prob1)/len(indices)

    To sample from this distribution, sample a random index "i" in the
    returned list. Then, sample a uniform value "x" in range [0,1]. Use
    ret[i][0] if x<=ret[i][2] and ret[i][1] otherwise.
    """

    ret = list()
    uniform = 1 / len(indices)
    # (state,probability) for probability <uniform
    small = list()
    # (state,probability) for probability >=uniform
    large = list()
    for i in range(0, len(indices)):
        if vals[i] < uniform:
            small.append((indices[i], vals[i]))
        else:
            large.append((indices[i], vals[i]))
    while len(ret) < len(indices):
        if len(small) > 0:
            if len(large) == 0:
                assert len(small) == (len(indices) - len(ret))
                for i in range(0, len(small)):
                    assert abs(small[i][1] - uniform) < 1e-5
                while len(small) > 0:
                    cur = small.pop()
                    ret.append((cur[0], cur[0], 1.0))
            else:
                cur_small = small.pop()
                cur_large = large.pop()
                cur_large[1] -= uniform - cur_small[1]
                ret.append((cur_small[0], cur_large[0], cur_small[1]))
                if cur_large[1] < uniform:
                    small.append(cur_large)
                else:
                    large.append(cur_large)
        else:
            assert len(large) == (len(indices) - len(ret))
            for i in range(0, len(large)):
                assert abs(large[i][1] - uniform) < 1e-5
            while len(large) > 0:
                cur = large.pop()
                ret.append((cur[0], cur[0], 1.0))

    assert len(small) == 0
    assert len(large) == 0
    assert len(ret) == len(indices)

    # testing code to make sure this works
    # is unnecessary computation so is
    # commented out

    # check = dict()
    # for ind in indices:
    #     check[ind] = 0
    # for i in range(0, len(ret)):
    #     assert ret[i][0] in check
    #     assert ret[i][1] in check
    #     check[ret[i][0]] += uniform * ret[i][2]
    #     check[ret[i][1]] += uniform * (1.0 - ret[i][2])
    # for i in range(0,len(indices)):
    #     assert abs(check[indices[i]] - vals[i]) < 1e-5

    return ret


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
    # This can probably be done in torch/numpy, but I'm in a hurry and will do
    # this in raw python.

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


def _validate_param(num_tiles, param):
    """Checks param for obvious bugs."""
    assert isinstance(param, torch.Tensor)
    assert tuple(param.size()) == (num_tiles ** 2,)
    for p in param:
        assert p >= 0 and p <= 1
    assert abs(param.sum().item() - 1.0) < 1e-5


# visualization


def _visualize_param(num_tiles, param):
    """Plot nodes drawn from param! For debug use only."""
    import matplotlib.pyplot as plt
    import numpy as np

    nodes = get_param_nodes(10, 10, 12345, num_tiles, param)

    x = nodes[:, 0, :].flatten().numpy()
    y = nodes[:, 1, :].flatten().numpy()

    plt.scatter(x, y)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


def _visualize_nodes(nodes):
    """Plot nodes drawn from param! For debug use only."""
    import matplotlib.pyplot as plt
    import numpy as np

    x = nodes[:, 0, :].flatten().numpy()
    y = nodes[:, 1, :].flatten().numpy()

    plt.scatter(x, y)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()



# stupid pipes are too small for my poop


def get_param_nodes_dev(num_nodes, num_samples, num_tiles, param, num_processes):
    """Create collection of points distributed according to parameters.

    Performance notes:
        1) We can usually assume that num_tiles is somewhat small.
        2) There is probably a vectorized way to make this data generation
        faster, but it looks tricky to actually do well, so we'll use
        multiprocessing instead.
        3) mp.Queue gets full sometimes because of the underlying pipe getting full, which causes us some problems. We get around this by writing things to disk instead
        4) Worker processes CANNOT USE TORCH. I don't know why this is, but torch causes threads to freeze for some reason. This might be a bug with torch or a bug with torch multiprocessing, which is why I'm using numpy and vanilla python multiprocessing. Plus, numpy seems to be faster than torch for this particular purpose.
    Args:
        num_nodes (int): number of nodes per datapoint
        num_samples (int): number of datapoints
        num_tiles (int): number of tiles per side (for a total number of
            num_tiles^2 tiles in the parameterization)
        param (torch.Tensor): parameter describing distribution of data
        num_processes (int): number of processes this should spawn (should be
            at least number of cpus)
    Returns:
        torch tensor of shape (num_samples, 2, num_nodes) with distribution of
        nodes according to param argument
    """
    # checking some preconditions
    assert isinstance(num_nodes, int)
    assert num_nodes > 0
    assert isinstance(num_samples, int)
    assert num_samples > 0
    assert isinstance(num_tiles, int)
    assert num_tiles > 0
    _validate_param(num_tiles, param)
    assert isinstance(num_processes, int)
    assert num_processes > 0

    # some utility lists
    # since param is fairly short, we can do this in raw python
    nonzero_indices = [i for i in range(len(param)) if param[i] > 0]
    nonzero_vals = [param[i].item() for i in nonzero_indices]

    # precomputations for spencer's magic
    balanced = _balanced_probabilities(nonzero_indices, nonzero_vals)
    x_pos = list()
    y_pos = list()
    for i in range(0, len(param)):
        x_pos.append((i % num_tiles) * (1 / num_tiles))
        y_pos.append(1 - (math.floor(i / num_tiles) + 1) * (1 / num_tiles))

    # generating random tile seeds (determines which tiles they land in)
    # can't think of a fast way to do this so we'll just iterate to get tiles
    # generating random tensors
    offsets = torch.rand((num_samples, 2, num_nodes)) / num_tiles
    ind_seeds = torch.randint(0, len(balanced), (num_samples, num_nodes))
    val_seeds = torch.rand((num_samples, num_nodes))

    # calculating start/end values for each batch
    sample_ranges = []
    range_size = math.floor(num_samples / num_processes)
    for i in range(num_processes - 1):
        curr_range = (i * range_size, (i + 1) * range_size)
        sample_ranges.append(curr_range)
    # getting any stragglers
    curr_range = (
        (num_processes - 1) * range_size,
        max(num_processes * range_size, num_samples),
    )
    sample_ranges.append(curr_range)

    # using vanilla multiprocessing and numpy
    # turns out we have problems when we don't write things to files
    processes = []
    q = mp.Queue()
    for i in range(num_processes):
        t = mp.Process(target = _get_param_nodes_dev_worker, 
                args=(
                    num_nodes,
                    num_samples,
                    sample_ranges[i][0],
                    sample_ranges[i][1],
                    balanced,
                    x_pos,
                    y_pos,
                    ind_seeds,
                    val_seeds,
                    i + 1,
                    q
                )
        )
        t.start()
        processes.append(t)

    for t in processes:
        t.join()

    batch_names = []
    for i in range(num_processes):
        batch_names.append(q.get())

    batches = [np.load(fname, allow_pickle=False) for fname in batch_names]
    for fname in batch_names:
        os.remove(fname)

    tiles = np.concatenate(batches, axis=0)
    tiles = torch.from_numpy(tiles)
    ret = tiles + offsets

    return ret


def _get_param_nodes_dev_worker(
    num_nodes,
    num_samples,
    start_samples,
    end_samples,
    balanced,
    x_pos,
    y_pos,
    ind_seeds,
    val_seeds,
    write_id,
    queue
):
    """Worker for generating param nodes."""
    fname = f"temp/temp-array-{write_id}.npy"
    
    # generating tensor
    tiles = np.zeros((end_samples - start_samples, 2, num_nodes))
    for i in range(start_samples, end_samples):
        for j in range(num_nodes):
            tile_index = -1
            if val_seeds[i, j] < balanced[ind_seeds[i, j]][2]:
                tile_index = balanced[ind_seeds[i, j]][0]
            else:
                tile_index = balanced[ind_seeds[i, j]][1]
            tiles[i - start_samples, 0, j] = x_pos[tile_index]
            tiles[i - start_samples, 1, j] = y_pos[tile_index]
    
    
    # saving tensor to disk
    np.save(fname, tiles, allow_pickle=False)
    queue.put(fname)



if __name__ == "__main__":
    profiling = True
    debugging = False

    if profiling:
        # profiling
        import time

        c_num_nodes = 100
        c_num_samples = 5000
        c_num_tiles = 8
        c_param = get_tiny_pair_param(c_num_tiles)
        c_processes = 8

        # print()
        # print("Running multiprocess version")
        # start = time.time()
        # nodes = get_param_nodes(
        #     c_num_nodes, c_num_samples, c_num_tiles, c_param, c_processes
        # )
        # end = time.time()
        # print(f"Elapsed time: {end - start}")
        # print(nodes.size())
        # print()
        # _visualize_nodes(nodes)

        print()
        print("Running new multiprocess version")
        start = time.time()
        nodes = get_param_nodes_dev(
            c_num_nodes, c_num_samples, c_num_tiles, c_param, c_processes
        )
        end = time.time()
        print(f"Elapsed time: {end - start}")
        print(nodes.size())
        print()
        # _visualize_nodes(nodes)

    if debugging:
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
            c_num_tiles = 8
            c_param = torch.tensor([1] + [0 for i in range(63)])

            import time

            start = time.time()

            nodes = get_param_nodes(
                c_num_nodes, c_num_samples, c_num_tiles, c_param
            )

            end = time.time()
            print(nodes)
            print()
            print(end - start)
