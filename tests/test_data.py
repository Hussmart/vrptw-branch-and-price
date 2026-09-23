import math

import numpy as np
import pytest

from vrptw_cg.data import load_instance


def test_distance_matrix_matches_hand_computed_euclidean():
    inst = load_instance("c101", 5, reduce_time_windows=False)
    for i in range(inst.num_nodes):
        for j in range(inst.num_nodes):
            expected = round(math.hypot(inst.x[i] - inst.x[j], inst.y[i] - inst.y[j]))
            assert inst.d[i, j] == expected


def test_depot_is_duplicated_with_identical_coordinates():
    inst = load_instance("c101", 5, reduce_time_windows=False)
    assert inst.x[0] == inst.x[-1]
    assert inst.y[0] == inst.y[-1]
    assert inst.q[0] == inst.q[-1] == 0


def test_time_window_reduction_never_widens_windows():
    inst_raw = load_instance("r101", 10, reduce_time_windows=False)
    inst_reduced = load_instance("r101", 10, reduce_time_windows=True)
    assert np.all(inst_reduced.a >= inst_raw.a_raw - 1e-9)
    assert np.all(inst_reduced.b <= inst_raw.b_raw + 1e-9)


def test_time_window_reduction_keeps_depot_feasible():
    inst = load_instance("r101", 10)
    # every customer window must still be non-empty after reduction
    assert np.all(inst.a <= inst.b + 1e-9)


def test_neighbor_sets_have_expected_size_and_exclude_self():
    inst = load_instance("c101", 20, neighbor_k=5)
    for customer, neighbors in inst.neighbors.items():
        assert customer not in neighbors
        assert len(neighbors) == min(5, inst.n - 1)


def test_rejects_out_of_range_customer_count():
    with pytest.raises(ValueError):
        load_instance("c101", 0)
    with pytest.raises(ValueError):
        load_instance("c101", 101)


def test_missing_instance_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_instance("does-not-exist", 5)
