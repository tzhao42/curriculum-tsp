"""Various curriculum specifications."""

import torch
from constants import NUM_TILES
from tasks import node_distrib, tsp
from tasks.node_distrib import (
    get_border_param,
    get_circle_param,
    get_down_line_param,
    get_horiz_param,
    get_medium_pair_param,
    get_plus_param,
    get_tiny_pair_param,
    get_tiny_quad_param,
    get_uniform_param,
    get_up_line_param,
    get_vert_param,
    get_x_shape_param,
)
from tasks.tsp import TSPCurriculum


def make_curriculum(
    epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
):
    """Makes a curriculum with the specifications specified in steps."""
    # Checking that the curriculum specified matches number of epochs
    assert epochs == sum([step[1] for step in steps])

    curriculum = TSPCurriculum(num_nodes, train_size, val_size, seed, debug)

    for step in steps:
        param_fn, num_epochs = step[0], step[1]
        param = param_fn(NUM_TILES)
        curriculum.add_stage(NUM_TILES, param, num_epochs)

    curriculum.add_val(NUM_TILES, val_param)

    curriculum.start()
    return curriculum


def get_indexed_curriculum(i):
    """Return a curriculum building function indexed by i."""
    return [
        get_curriculum_zero,
        get_curriculum_one,
        get_curriculum_two,
        get_curriculum_three,
        get_curriculum_four,
        get_curriculum_five,
        get_curriculum_six,
        get_curriculum_seven,
        get_curriculum_eight,
        get_curriculum_nine,
        get_curriculum_ten,
        get_curriculum_eleven,
        get_curriculum_twelve,
        get_curriculum_thirteen,
        get_curriculum_fourteen,
        get_curriculum_fifteen,
        get_curriculum_sixteen,
        get_curriculum_seventeen,
    ][i]


def get_curriculum_zero(epochs, num_nodes, train_size, val_size, seed, debug):
    """Create the uniform curriculum."""
    steps = [(get_uniform_param, 20)]
    val_param = get_uniform_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_one(epochs, num_nodes, train_size, val_size, seed, debug):
    """Create curriculum for experiment one.

    Curriculum consists of:
        1) Tiny-pair 6
        2) Medium-pair 6
        3) Uniform 8
    """
    steps = [
        (get_tiny_pair_param, 6),
        (get_medium_pair_param, 6),
        (get_uniform_param, 8),
    ]
    val_param = get_uniform_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_two(epochs, num_nodes, train_size, val_size, seed, debug):
    """Create curriculum for experiment two.

    Curriculum consists of:
        1) Tiny-pair 4
        2) Tiny-quad 4
        3) Border 4
        4) Uniform 8
    """
    steps = [
        (get_tiny_pair_param, 4),
        (get_tiny_quad_param, 4),
        (get_border_param, 4),
        (get_uniform_param, 8),
    ]
    val_param = get_uniform_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_three(epochs, num_nodes, train_size, val_size, seed, debug):
    """Create curriculum for experiment three."""
    steps = [
        (get_tiny_pair_param, 4),
        (get_tiny_quad_param, 4),
        (get_circle_param, 4),
        (get_uniform_param, 8),
    ]
    val_param = get_uniform_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_four(epochs, num_nodes, train_size, val_size, seed, debug):
    """Create curriculum for experiment four."""
    steps = [
        (get_horiz_param, 2),
        (get_vert_param, 2),
        (get_plus_param, 2),
        (get_down_line_param, 2),
        (get_up_line_param, 2),
        (get_x_shape_param, 2),
        (get_uniform_param, 8),
    ]
    val_param = get_uniform_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_five(epochs, num_nodes, train_size, val_size, seed, debug):
    """Create curriculum for experiment five."""
    steps = [
        (get_down_line_param, 4),
        (get_up_line_param, 4),
        (get_circle_param, 4),
        (get_uniform_param, 8),
    ]
    val_param = get_uniform_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_six(epochs, num_nodes, train_size, val_size, seed, debug):
    """Create curriculum for experiment six."""
    steps = [
        (get_circle_param, 10),
        (get_uniform_param, 10),
    ]
    val_param = get_uniform_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_seven(epochs, num_nodes, train_size, val_size, seed, debug):
    """Create curriculum for experiment seven."""
    steps = [
        (get_tiny_quad_param, 10),
        (get_uniform_param, 10),
    ]
    val_param = get_uniform_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_eight(epochs, num_nodes, train_size, val_size, seed, debug):
    """Create curriculum for experiment eight."""
    steps = [
        (get_border_param, 10),
        (get_uniform_param, 10),
    ]
    val_param = get_uniform_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_nine(epochs, num_nodes, train_size, val_size, seed, debug):
    """Create curriculum for experiment nine."""
    steps = [
        (get_down_line_param, 10),
        (get_uniform_param, 10),
    ]
    val_param = get_uniform_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_ten(epochs, num_nodes, train_size, val_size, seed, debug):
    """Create curriculum for experiment ten."""
    steps = [
        (get_medium_pair_param, 20),
    ]
    val_param = get_medium_pair_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_eleven(
    epochs, num_nodes, train_size, val_size, seed, debug
):
    """Create curriculum for experiment eleven."""
    steps = [
        (get_tiny_pair_param, 10),
        (get_medium_pair_param, 10),
    ]
    val_param = get_medium_pair_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_twelve(
    epochs, num_nodes, train_size, val_size, seed, debug
):
    """Create curriculum for experiment twelve."""
    steps = [
        (get_tiny_quad_param, 20),
    ]
    val_param = get_tiny_quad_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_thirteen(
    epochs, num_nodes, train_size, val_size, seed, debug
):
    """Create curriculum for experiment thirteen."""
    steps = [
        (get_tiny_pair_param, 10),
        (get_tiny_quad_param, 10),
    ]
    val_param = get_tiny_quad_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_fourteen(
    epochs, num_nodes, train_size, val_size, seed, debug
):
    """Create curriculum for experiment fourteen."""
    steps = [
        (get_x_shape_param, 20),
    ]
    val_param = get_x_shape_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_fifteen(
    epochs, num_nodes, train_size, val_size, seed, debug
):
    """Create curriculum for experiment fifteen."""
    steps = [
        (get_down_line_param, 5),
        (get_up_line_param, 5),
        (get_x_shape_param, 10),
    ]
    val_param = get_x_shape_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_sixteen(
    epochs, num_nodes, train_size, val_size, seed, debug
):
    """Create curriculum for experiment sixteen."""
    steps = [
        (get_border_param, 20),
    ]
    val_param = get_border_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )


def get_curriculum_seventeen(
    epochs, num_nodes, train_size, val_size, seed, debug
):
    """Create curriculum for experiment seventeen."""
    steps = [
        (get_tiny_quad_param, 10),
        (get_border_param, 10),
    ]
    val_param = get_border_param(NUM_TILES)

    return make_curriculum(
        epochs, num_nodes, train_size, val_size, seed, steps, val_param, debug
    )
