import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
from numpy.testing import assert_raises as ar
from pytest import raises
from .. import data_structures as ds 

# regular_grid_xy

def test_regular_grid_xy_bad_input():
    "must raise error if receive bad input"
    # good parameters
    area = [1, 2.2, -3.1, 4]
    shape = (12, 11)
    z0 = 12.3

    # area not list
    with raises(ValueError):
        ds.regular_grid_xy(area=(1, 2.2, -3.1, 4), shape=shape, z0=z0)
    # len area different from 4
    with raises(ValueError):
        ds.regular_grid_xy(area=[1, 2.2, -3.1], shape=shape, z0=z0)
    # shape not tuple
    with raises(ValueError):
        ds.regular_grid_xy(area=area, shape=[12,11], z0=z0)
    # len shape different from 2
    with raises(ValueError):
        ds.regular_grid_xy(area=area, shape=(12, 11, 10), z0=z0)
    # z0 complex
    with raises(ValueError):
        ds.regular_grid_xy(area=area, shape=shape, z0=12.3+5j)
    # z0 list
    with raises(ValueError):
        ds.regular_grid_xy(area=area, shape=shape, z0=[12.3])


def test_regular_grid_xy_output():
    "compare output with reference"
    area = [1, 5, 14.5, 17.5]
    shape = (5, 4)
    z0 = 10
    reference = {
        'x': np.array([1, 2, 3, 4, 5]),
        'y': np.array([14.5, 15.5, 16.5, 17.5]),
        'z': 10,
        'area': area,
        'shape': shape
    }
    computed = ds.regular_grid_xy(area=area, shape=shape, z0=z0)
    ae(reference, computed)


def test_grid_to_full_known_values():
    "compare results with reference values obtained for specific input"
    # reference grid
    grid = {
        'x' : np.arange(3),
        'y' : np.array([10.3, 12.4]),
        'z' : 10.,
        'area' : [0, 3, 10.3, 12.4],
        'shape' : (3, 2)
    }
    reference_xy = {
        'x' : np.array([0, 1, 2, 0, 1, 2]),
        'y' : np.array([10.3, 10.3, 10.3, 12.4, 12.4, 12.4]),
        'z' : np.array([10., 10., 10., 10., 10., 10.])
    }
    reference_yx = {
        'x' : np.array([0, 0, 1, 1, 2, 2]),
        'y' : np.array([10.3, 12.4, 10.3, 12.4, 10.3, 12.4]),
        'z' : np.array([10., 10., 10., 10., 10., 10.])
    }
    computed_xy = ds.grid_to_full(grid=grid, ordering='xy')
    computed_yx = ds.grid_to_full(grid=grid, ordering='yx')
    ae(computed_xy, reference_xy)
    ae(computed_yx, reference_yx)


# regular_grid_wavenumbers

def test_regular_grid_wavenumbers_bad_input():
    "must raise error if receive bad input"
    # good parameters
    shape = (12, 11)
    spacing = (1., 1.1)
    ordering='yx'

    # shape not tuple
    with raises(ValueError):
        ds.regular_grid_wavenumbers(shape=[12,11], spacing=spacing, ordering=ordering)
    # len shape different from 2
    with raises(ValueError):
        ds.regular_grid_wavenumbers(shape=(12, 11, 10), spacing=spacing, ordering=ordering)
    # shape not tuple
    with raises(ValueError):
        ds.regular_grid_wavenumbers(shape=shape, spacing=[1., 1.1], ordering=ordering)
    # len spacing different from 2
    with raises(ValueError):
        ds.regular_grid_wavenumbers(shape=shape, spacing=(1., 1.1, 3.), ordering=ordering)
    # invalid ordering
    with raises(ValueError):
        ds.regular_grid_wavenumbers(shape=shape, spacing=spacing, ordering='invalid-ordering')


def test_regular_grid_wavenumbers_output():
    "compare output with reference"
    shape = (5, 4)
    spacing = (1.3, 1.1)
    ordering='xy'
    x_reference = 2 * np.pi * np.array([-2, -1, 0, 1, 2]) / (5 * 1.3)
    y_reference = 2 * np.pi * np.array([-2, -1, 0, 1]) / (4 * 1.1)
    z_reference = np.sqrt(x_reference[:, np.newaxis]**2 + y_reference**2)
    reference = {
        'x': x_reference,
        'y': y_reference,
        'z': z_reference,
        'ordering': 'xy',
        'shape': shape,
        'spacing': spacing
    }
    computed = ds.regular_grid_wavenumbers(shape=shape, spacing=spacing, ordering=ordering)
    aae(reference['x'], computed['x'], decimal=12)
    aae(reference['y'], computed['y'], decimal=12)
    aae(reference['z'], computed['z'], decimal=12)
    ae(shape, computed['shape'])
    ae(spacing, computed['spacing'])
    ae(ordering, computed['ordering'])


# BTTB_transposed_metadata

def test_BTTB_transposed_metadata_symm_symm():
    "compare computed result with a reference for known input"
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "symm",
        "nblocks": 2,
        "columns": np.array(
            [
                [1, 2, 3],
                [10, 20, 30],
            ]
        ),
        "rows": None,
    }
    reference = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "symm",
        "nblocks": 2,
        "columns": np.array(
            [
                [1, 2, 3],
                [10, 20, 30],
            ]
        ),
        "rows": None,
    }
    computed = ds.BTTB_transposed_metadata(BTTB_metadata=BTTB)
    ae(computed, reference)


def test_BTTB_transposed_metadata_symm_skew():
    "compare computed result with a reference for known input"
    # define the data structure for the generating BTTB matrix
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "skew",
        "nblocks": 2,
        "columns": np.array(
            [
                [1, 2, 3],
                [10, 20, 30],
            ]
        ),
        "rows": None,
    }
    # define the data structure for the generating BTTB matrix transposed
    reference = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "skew",
        "nblocks": 2,
        "columns": np.array(
            [
                [1, -2, -3],
                [10, -20, -30],
            ]
        ),
        "rows": None,
    }
    computed = ds.BTTB_transposed_metadata(BTTB_metadata=BTTB)
    ae(computed, reference)


def test_BTTB_transposed_metadata_symm_gene():
    "compare computed result with a reference for known input"
    # define the data structure for the generating BTTB matrix
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "gene",
        "nblocks": 2,
        "columns": np.array(
            [
                [0, -2, 7],
                [10, 40, 50],
            ]
        ),
        "rows": np.array(
            [
                [18, 32],
                [20, 30],
            ]
        )
    }
    # define the data structure for the generating BTTB matrix transposed
    reference = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "gene",
        "nblocks": 2,
        "columns": np.array(
            [
                [0, 18, 32],
                [10, 20, 30],
            ]
        ),
        "rows": np.array(
            [
                [-2, 7],
                [40, 50],
            ]
        )
    }
    computed = ds.BTTB_transposed_metadata(BTTB_metadata=BTTB)
    ae(computed, reference)


def test_BTTB_transposed_metadata_skew_symm():
    "compare computed result with a reference for known input"
    # define the data structure for the generating BTTB matrix
    BTTB = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "symm",
        "nblocks": 2,
        "columns": np.array(
            [
                [1, 2, 3],
                [10, 20, 30],
            ]
        ),
        "rows": None,
    }
    # define the data structure for the generating BTTB matrix transposed
    reference = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "symm",
        "nblocks": 2,
        "columns": np.array(
            [
                [1, 2, 3],
                [-10, -20, -30],
            ]
        ),
        "rows": None,
    }
    computed = ds.BTTB_transposed_metadata(BTTB_metadata=BTTB)
    ae(computed, reference)


def test_BTTB_transposed_metadata_skew_skew():
    "compare computed result with a reference for known input"
    # define the data structure for the generating BTTB matrix
    BTTB = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "skew",
        "nblocks": 2,
        "columns": np.array(
            [
                [1, 2, 3],
                [-10, -20, -30],
            ]
        ),
        "rows": None,
    }
    # define the data structure for the generating BTTB matrix transposed
    reference = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "skew",
        "nblocks": 2,
        "columns": np.array(
            [
                [1, -2, -3],
                [10, -20, -30],
            ]
        ),
        "rows": None,
    }
    computed = ds.BTTB_transposed_metadata(BTTB_metadata=BTTB)
    ae(computed, reference)


def test_BTTB_transposed_metadata_skew_gene():
    "compare computed result with a reference for known input"
    # define the data structure for the generating BTTB matrix
    BTTB = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "gene",
        "nblocks": 2,
        "columns": np.array(
            [
                [0, -2, 7],
                [10, 40, 50],
            ]
        ),
        "rows": np.array(
            [
                [18, 32],
                [20, 30],
            ]
        )
    }
    # define the data structure for the generating BTTB matrix transposed
    reference = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "gene",
        "nblocks": 2,
        "columns": np.array(
            [
                [0, 18, 32],
                [-10, -20, -30],
            ]
        ),
        "rows": np.array(
            [
                [-2, 7],
                [-40, -50],
            ]
        )
    }
    computed = ds.BTTB_transposed_metadata(BTTB_metadata=BTTB)
    ae(computed, reference)


def test_BTTB_transposed_metadata_gene_symm():
    "compare computed result with a reference for known input"
    # define the data structure for the generating BTTB matrix
    BTTB = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "symm",
        "nblocks": 2,
        "columns": np.array(
            [
                [0, -2, 32],
                [60, -70, 80],
                [10, -40, -30]
            ]
        ),
        "rows": None
    }
    # define the data structure for the generating BTTB matrix transposed
    reference = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "symm",
        "nblocks": 2,
        "columns": np.array(
            [
                [0, -2, 32],
                [10, -40, -30],
                [60, -70, 80]
            ]
        ),
        "rows": None
    }
    computed = ds.BTTB_transposed_metadata(BTTB_metadata=BTTB)
    ae(computed, reference)


def test_BTTB_transposed_metadata_gene_skew():
    "compare computed result with a reference for known input"
    # define the data structure for the generating BTTB matrix
    BTTB = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "skew",
        "nblocks": 2,
        "columns": np.array(
            [
                [0, -2, 32],
                [-60, -70, 80],
                [10, -40, -30]
            ]
        ),
        "rows": None
    }
    # define the data structure for the generating BTTB matrix transposed
    reference = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "skew",
        "nblocks": 2,
        "columns": np.array(
            [
                [0, 2, -32],
                [10, 40, 30],
                [-60, 70, -80]
            ]
        ),
        "rows": None
    }
    computed = ds.BTTB_transposed_metadata(BTTB_metadata=BTTB)
    ae(computed, reference)


def test_BTTB_transposed_metadata_gene_gene():
    "compare computed result with a reference for known input"
    # define the data structure for the generating BTTB matrix
    BTTB = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "gene",
        "nblocks": 2,
        "columns": np.array(
            [
                [0, -2, 7],
                [60, -90, 100],
                [10, 40, 50]
            ]
        ),
        "rows": np.array(
            [
                [18, 32],
                [70, 80],
                [20, 30]
            ]
        )
    }
    # define the data structure for the generating BTTB matrix transposed
    reference = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "gene",
        "nblocks": 2,
        "columns": np.array(
            [
                [0, 18, 32],
                [10, 20, 30],
                [60, 70, 80]
            ]
        ),
        "rows": np.array(
            [
                [-2, 7],
                [40, 50],
                [-90, 100]
            ]
        )
    }
    computed = ds.BTTB_transposed_metadata(BTTB_metadata=BTTB)
    ae(computed, reference)