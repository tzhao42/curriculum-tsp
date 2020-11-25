"""Parameterization for collection of points on [0,1]x[0,1] square."""

def gen_param_data(num_nodes, num_samples, seed, num_tiles, distrib):
    """Create collection of points distributed according to parameters.

    Args:
        num_nodes (int): number of nodes per datapoint
        num_samples (int): number of datapoints
        seed (int): random seed
        num_tiles (int): number of tiles per side (for a total number of 
            num_tiles^2 tiles in the parameterization)
        distrib (List[float]): parameter describing distribution of data
    
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
    assert len(distrib) == num_tiles ** 2
    for p in distrib:
        assert p >= 0 and p <= 1
    assert sum(distrib) == 1

    # generating random seeds
    seeds = torch.rand((num_samples, 1, num_nodes))


