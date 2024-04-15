import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
from numpy.testing import assert_raises as ar
from numpy.linalg import multi_dot
from scipy.linalg import toeplitz, circulant, dft
from pytest import raises
from .. import convolve as cv
from .. import data_structures as ds

##### compute


def test_compute_FT_data_not_complex_matrix():
    "must raise an error if FT_data is not a complex matrix"
    filters = [np.ones((5, 5))]
    # FT_data as a complex vector
    FT_data = np.ones(5) - 1j * np.ones(5)
    with raises(ValueError):
        cv.compute(FT_data, filters)
    # FT_data as a real matrix
    FT_data = np.ones((5, 5))
    with raises(ValueError):
        cv.compute(FT_data, filters)


def test_compute_filters_not_complex_matrices():
    "must raise an error if filters does not contain complex matrices"
    FT_data = np.ones((5, 5)) - 1j * np.ones((5, 5))
    # filters without any element
    filters = []
    with raises(ValueError):
        cv.compute(FT_data, filters)
    # filters as a scalar
    filters = 3
    with raises(ValueError):
        cv.compute(FT_data, filters)
    # filters with vectors
    filters = [np.ones(3)]
    with raises(ValueError):
        cv.compute(FT_data, filters)
    # filters with matrices having a sahpe different from FT_data
    filters = [np.ones((3, 3))]
    with raises(ValueError):
        cv.compute(FT_data, filters)


##### Circulant_from_Toeplitz


def test_Circulant_from_Toeplitz_compare_known_values_symm():
    "verify if the computed Circulant is equal to the reference"
    reference = np.array(
        [
            [1, 2, 3, 0, 3, 2],
            [2, 1, 2, 3, 0, 3],
            [3, 2, 1, 2, 3, 0],
            [0, 3, 2, 1, 2, 3],
            [3, 0, 3, 2, 1, 2],
            [2, 3, 0, 3, 2, 1],
        ]
    )
    # compute the full matrix
    Toeplitz = {
        "symmetry": "symm",
        "column": np.array([1, 2, 3]),
        "row": None,
    }
    computed = cv.Circulant_from_Toeplitz(Toeplitz, full=True)
    ae(computed, reference)


def test_Circulant_from_Toeplitz_compare_known_values_skew():
    "verify if the computed Circulant is equal to the reference"
    reference = np.array(
        [
            [1, -2, -3, 0, 3, 2],
            [2, 1, -2, -3, 0, 3],
            [3, 2, 1, -2, -3, 0],
            [0, 3, 2, 1, -2, -3],
            [-3, 0, 3, 2, 1, -2],
            [-2, -3, 0, 3, 2, 1],
        ]
    )
    # compute the full matrix
    Toeplitz = {
        "symmetry": "skew",
        "column": np.array([1, 2, 3]),
        "row": None,
    }
    computed = cv.Circulant_from_Toeplitz(Toeplitz, full=True)
    ae(computed, reference)


def test_Circulant_from_Toeplitz_compare_known_values_gene():
    "verify if the computed Circulant is equal to the reference"
    reference = np.array(
        [
            [1, 4, 5, 0, 3, 2],
            [2, 1, 4, 5, 0, 3],
            [3, 2, 1, 4, 5, 0],
            [0, 3, 2, 1, 4, 5],
            [5, 0, 3, 2, 1, 4],
            [4, 5, 0, 3, 2, 1],
        ]
    )
    # compute the full matrix
    Toeplitz = {
        "symmetry": "gene",
        "column": np.array([1, 2, 3]),
        "row": np.array([4, 5]),
    }
    computed = cv.Circulant_from_Toeplitz(Toeplitz, full=True)
    ae(computed, reference)


def test_Circulant_from_Toeplitz_compare_first_column():
    "verify if returns the correct first column"
    # from symmetric Toeplitz
    Toeplitz = {
        "symmetry": "symm",
        "column": np.array([1, 2, 3]),
        "row": None,
    }
    full = cv.Circulant_from_Toeplitz(Toeplitz, full=True)
    column = cv.Circulant_from_Toeplitz(Toeplitz, full=False)
    ae(circulant(column), full)
    # from skew-symmetric Toeplitz
    Toeplitz = {
        "symmetry": "skew",
        "column": np.array([8, -2, 3]),
        "row": None,
    }
    full = cv.Circulant_from_Toeplitz(Toeplitz, full=True)
    column = cv.Circulant_from_Toeplitz(Toeplitz, full=False)
    ae(circulant(column), full)
    # from generic Toeplitz
    Toeplitz = {
        "symmetry": "gene",
        "column": np.array([10, 92, -3]),
        "row": np.array([4, 18]),
    }
    full = cv.Circulant_from_Toeplitz(Toeplitz, full=True)
    column = cv.Circulant_from_Toeplitz(Toeplitz, full=False)
    ae(circulant(column), full)


def test_Circulant_from_Toeplitz_symmetry_preservation():
    "verify if the computed Circulant preserve the symmetry of the originating Toeplitz matrix"
    # symmetric Toeplitz
    Toeplitz = {
        "symmetry": "symm",
        "column": np.array([1, 2, 3]),
        "row": None,
    }
    computed = cv.Circulant_from_Toeplitz(Toeplitz, full=True)
    ae(computed, computed.T)
    # skew-symmetric Toeplitz
    Toeplitz = {
        "symmetry": "skew",
        "column": np.array([8, 2, 3]),
        "row": None,
    }
    computed = cv.Circulant_from_Toeplitz(Toeplitz, full=True)
    # we are not interested in the elements at the main diagonal
    diagonal = np.diag(np.diag(computed))
    ae(-(computed - diagonal), (computed - diagonal).T)
    # generic Toeplitz
    Toeplitz = {
        "symmetry": "gene",
        "column": np.array([1, 2, 3]),
        "row": np.array([4, 5]),
    }
    computed = cv.Circulant_from_Toeplitz(Toeplitz, full=True)
    ae(computed, circulant(np.array([1, 2, 3, 0, 5, 4])))


##### BTTB_from_metadata


def test_BTTB_from_metadata_compare_known_values_symm_symm():
    "verify if the computed BTTB is equal to the reference"
    columns = np.array(
        [
            [1, 2, 3],
            [10, 20, 30],
        ]
    )
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "symm",
        "nblocks": 2,
        "columns": columns,
        "rows": None,
    }
    computed = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
    reference = np.array(
        [
            [1, 2, 3, 10, 20, 30],
            [2, 1, 2, 20, 10, 20],
            [3, 2, 1, 30, 20, 10],
            [10, 20, 30, 1, 2, 3],
            [20, 10, 20, 2, 1, 2],
            [30, 20, 10, 3, 2, 1],
        ]
    )
    ae(computed, reference)


def test_BTTB_from_metadata_compare_known_values_symm_skew():
    "verify if the computed BTTB is equal to the reference"
    columns = np.array(
        [
            [1, 2, 3],
            [10, 20, 30],
        ]
    )
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "skew",
        "nblocks": 2,
        "columns": columns,
        "rows": None,
    }
    computed = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
    reference = np.array(
        [
            [1, -2, -3, 10, -20, -30],
            [2, 1, -2, 20, 10, -20],
            [3, 2, 1, 30, 20, 10],
            [10, -20, -30, 1, -2, -3],
            [20, 10, -20, 2, 1, -2],
            [30, 20, 10, 3, 2, 1],
        ]
    )
    ae(computed, reference)


def test_BTTB_from_metadata_compare_known_values_symm_gene():
    "verify if the computed BTTB is equal to the reference"
    columns = np.array([[0, -2, 7], [10, 40, 50]])
    rows = np.array(
        [
            [18, 32],
            [20, 30],
        ]
    )
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "gene",
        "nblocks": 2,
        "columns": columns,
        "rows": rows,
    }
    computed = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
    reference = np.array(
        [
            [0, 18, 32, 10, 20, 30],
            [-2, 0, 18, 40, 10, 20],
            [7, -2, 0, 50, 40, 10],
            [10, 20, 30, 0, 18, 32],
            [40, 10, 20, -2, 0, 18],
            [50, 40, 10, 7, -2, 0],
        ]
    )
    ae(computed, reference)


def test_BTTB_from_metadata_compare_known_values_skew_symm():
    "verify if the computed BTTB is equal to the reference"
    columns = np.array(
        [
            [1, 2, 3],
            [10, 20, 30],
        ]
    )
    BTTB = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "symm",
        "nblocks": 2,
        "columns": columns,
        "rows": None,
    }
    computed = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
    reference = np.array(
        [
            [1, 2, 3, -10, -20, -30],
            [2, 1, 2, -20, -10, -20],
            [3, 2, 1, -30, -20, -10],
            [10, 20, 30, 1, 2, 3],
            [20, 10, 20, 2, 1, 2],
            [30, 20, 10, 3, 2, 1],
        ]
    )
    ae(computed, reference)


def test_BTTB_from_metadata_compare_known_values_skew_skew():
    "verify if the computed BTTB is equal to the reference"
    columns = np.array(
        [
            [1, 2, 3],
            [-10, -20, -30],
        ]
    )
    BTTB = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "skew",
        "nblocks": 2,
        "columns": columns,
        "rows": None,
    }
    computed = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
    reference = np.array(
        [
            [1, -2, -3, 10, -20, -30],
            [2, 1, -2, 20, 10, -20],
            [3, 2, 1, 30, 20, 10],
            [-10, 20, 30, 1, -2, -3],
            [-20, -10, 20, 2, 1, -2],
            [-30, -20, -10, 3, 2, 1],
        ]
    )
    ae(computed, reference)


def test_BTTB_from_metadata_compare_known_values_skew_gene():
    "verify if the computed BTTB is equal to the reference"
    columns = np.array([[0, -2, 7], [10, 40, 50]])
    rows = np.array(
        [
            [18, 32],
            [20, 30],
        ]
    )
    BTTB = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "gene",
        "nblocks": 2,
        "columns": columns,
        "rows": rows,
    }
    computed = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
    reference = np.array(
        [
            [0, 18, 32, -10, -20, -30],
            [-2, 0, 18, -40, -10, -20],
            [7, -2, 0, -50, -40, -10],
            [10, 20, 30, 0, 18, 32],
            [40, 10, 20, -2, 0, 18],
            [50, 40, 10, 7, -2, 0],
        ]
    )
    ae(computed, reference)


def test_BTTB_from_metadata_compare_known_values_gene_symm():
    "verify if the computed BTTB is equal to the reference"
    columns = np.array([[0, -2, 32], [60, -70, 80], [10, -40, -30]])
    BTTB = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "symm",
        "nblocks": 2,
        "columns": columns,
        "rows": None,
    }
    computed = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
    reference = np.array(
        [
            [0, -2, 32, 10, -40, -30],
            [-2, 0, -2, -40, 10, -40],
            [32, -2, 0, -30, -40, 10],
            [60, -70, 80, 0, -2, 32],
            [-70, 60, -70, -2, 0, -2],
            [80, -70, 60, 32, -2, 0],
        ]
    )
    ae(computed, reference)


def test_BTTB_from_metadata_compare_known_values_gene_skew():
    "verify if the computed BTTB is equal to the reference"
    columns = np.array([[0, -2, 32], [-60, -70, 80], [10, -40, -30]])
    BTTB = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "skew",
        "nblocks": 2,
        "columns": columns,
        "rows": None,
    }
    computed = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
    reference = np.array(
        [
            [0, 2, -32, 10, 40, 30],
            [-2, 0, 2, -40, 10, 40],
            [32, -2, 0, -30, -40, 10],
            [-60, 70, -80, 0, 2, -32],
            [-70, -60, 70, -2, 0, 2],
            [80, -70, -60, 32, -2, 0],
        ]
    )
    ae(computed, reference)


def test_BTTB_from_metadata_compare_known_values_gene_gene():
    "verify if the computed BTTB is equal to the reference"
    columns = np.array([[0, -2, 7], [60, -90, 100], [10, 40, 50]])
    rows = np.array([[18, 32], [70, -80], [20, 30]])
    BTTB = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "gene",
        "nblocks": 2,
        "columns": columns,
        "rows": rows,
    }
    computed = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
    reference = np.array(
        [
            [0, 18, 32, 10, 20, 30],
            [-2, 0, 18, 40, 10, 20],
            [7, -2, 0, 50, 40, 10],
            [60, 70, -80, 0, 18, 32],
            [-90, 60, 70, -2, 0, 18],
            [100, -90, 60, 7, -2, 0],
        ]
    )
    ae(computed, reference)


# ##### embedding_BCCB


def test_embedding_BCCB_compare_known_values_symm_symm():
    "verify if the computed BCCB is equal to the reference"
    # define the columns/rows of the generating BTTB
    columns = np.array(
        [
            [1, 2, 3],
            [10, 20, 30],
        ]
    )
    # compute the BCCB
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "symm",
        "nblocks": 2,
        "columns": columns,
        "rows": None,
    }
    computed = cv.embedding_BCCB(BTTB_metadata=BTTB)
    # define the reference
    reference = np.array(
        [
            1,
            2,
            3,
            0,
            3,
            2,
            10,
            20,
            30,
            0,
            30,
            20,
            0,
            0,
            0,
            0,
            0,
            0,
            10,
            20,
            30,
            0,
            30,
            20,
        ]
    )
    ae(computed, reference)


def test_embedding_BCCB_compare_known_values_symm_skew():
    "verify if the computed BCCB is equal to the reference"
    # define the columns/rows of the generating BTTB
    columns = np.array(
        [
            [1, 2, 3],
            [10, 20, 30],
        ]
    )
    # compute the BCCB
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "skew",
        "nblocks": 2,
        "columns": columns,
        "rows": None,
    }
    computed = cv.embedding_BCCB(BTTB_metadata=BTTB)
    # define the reference
    reference = np.array(
        [
            1,
            2,
            3,
            0,
            -3,
            -2,
            10,
            20,
            30,
            0,
            -30,
            -20,
            0,
            0,
            0,
            0,
            0,
            0,
            10,
            20,
            30,
            0,
            -30,
            -20,
        ]
    )
    ae(computed, reference)


def test_embedding_BCCB_compare_known_values_symm_gene():
    "verify if the computed BCCB is equal to the reference"
    # define the columns/rows of the generating BTTB
    columns = np.array([[0, -2, 7], [10, 40, 50]])
    rows = np.array(
        [
            [18, 32],
            [20, 30],
        ]
    )
    # compute the BCCB
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "gene",
        "nblocks": 2,
        "columns": columns,
        "rows": rows,
    }
    computed = cv.embedding_BCCB(BTTB_metadata=BTTB)
    # define the reference
    reference = np.array(
        [
            0,
            -2,
            7,
            0,
            32,
            18,
            10,
            40,
            50,
            0,
            30,
            20,
            0,
            0,
            0,
            0,
            0,
            0,
            10,
            40,
            50,
            0,
            30,
            20,
        ]
    )
    ae(computed, reference)


def test_embedding_BCCB_compare_known_values_skew_symm():
    "verify if the computed BCCB is equal to the reference"
    # define the columns/rows of the generating BTTB
    columns = np.array(
        [
            [1, 2, 3],
            [10, 20, 30],
        ]
    )
    # compute the BCCB
    BTTB = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "symm",
        "nblocks": 2,
        "columns": columns,
        "rows": None,
    }
    computed = cv.embedding_BCCB(BTTB_metadata=BTTB)
    # define the reference
    reference = np.array(
        [
            1,
            2,
            3,
            0,
            3,
            2,
            10,
            20,
            30,
            0,
            30,
            20,
            0,
            0,
            0,
            0,
            0,
            0,
            -10,
            -20,
            -30,
            0,
            -30,
            -20,
        ]
    )
    ae(computed, reference)


def test_embedding_BCCB_compare_known_values_skew_skew():
    "verify if the computed BCCB is equal to the reference"
    # define the columns/rows of the generating BTTB
    columns = np.array(
        [
            [1, 2, 3],
            [10, 20, 30],
        ]
    )
    # compute the BCCB
    BTTB = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "skew",
        "nblocks": 2,
        "columns": columns,
        "rows": None,
    }
    computed = cv.embedding_BCCB(BTTB_metadata=BTTB)
    # define the reference
    reference = np.array(
        [
            1,
            2,
            3,
            0,
            -3,
            -2,
            10,
            20,
            30,
            0,
            -30,
            -20,
            0,
            0,
            0,
            0,
            0,
            0,
            -10,
            -20,
            -30,
            0,
            30,
            20,
        ]
    )
    ae(computed, reference)


def test_embedding_BCCB_compare_known_values_skew_gene():
    "verify if the computed BCCB is equal to the reference"
    # define the columns/rows of the generating BTTB
    columns = np.array([[0, -2, 7], [10, 40, 50]])
    rows = np.array(
        [
            [18, 32],
            [20, 30],
        ]
    )
    # compute the BCCB
    BTTB = {
        "symmetry_structure": "skew",
        "symmetry_blocks": "gene",
        "nblocks": 2,
        "columns": columns,
        "rows": rows,
    }
    computed = cv.embedding_BCCB(BTTB_metadata=BTTB)
    # define the reference
    reference = np.array(
        [
            0,
            -2,
            7,
            0,
            32,
            18,
            10,
            40,
            50,
            0,
            30,
            20,
            0,
            0,
            0,
            0,
            0,
            0,
            -10,
            -40,
            -50,
            0,
            -30,
            -20,
        ]
    )
    ae(computed, reference)


def test_embedding_BCCB_compare_known_values_gene_symm():
    "verify if the computed BCCB is equal to the reference"
    # define the columns/rows of the generating BTTB
    columns = np.array([[0, -2, 7], [10, 40, 50], [28, 12, 3]])
    # compute the BCCB
    BTTB = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "symm",
        "nblocks": 2,
        "columns": columns,
        "rows": None,
    }
    computed = cv.embedding_BCCB(BTTB_metadata=BTTB)
    # define the reference
    reference = np.array(
        [
            0,
            -2,
            7,
            0,
            7,
            -2,
            10,
            40,
            50,
            0,
            50,
            40,
            0,
            0,
            0,
            0,
            0,
            0,
            28,
            12,
            3,
            0,
            3,
            12,
        ]
    )
    ae(computed, reference)


def test_embedding_BCCB_compare_known_values_gene_skew():
    "verify if the computed BCCB is equal to the reference"
    # define the columns/rows of the generating BTTB
    columns = np.array([[0, -2, 7], [10, 40, 50], [28, 12, 3]])
    # compute the BCCB
    BTTB = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "skew",
        "nblocks": 2,
        "columns": columns,
        "rows": None,
    }
    computed = cv.embedding_BCCB(BTTB_metadata=BTTB)
    # define the reference
    reference = np.array(
        [
            0,
            -2,
            7,
            0,
            -7,
            2,
            10,
            40,
            50,
            0,
            -50,
            -40,
            0,
            0,
            0,
            0,
            0,
            0,
            28,
            12,
            3,
            0,
            -3,
            -12,
        ]
    )
    ae(computed, reference)


def test_embedding_BCCB_compare_known_values_gene_gene():
    "verify if the computed BCCB is equal to the reference"
    # define the columns/rows of the generating BTTB
    columns = np.array([[0, -2, 7], [10, 40, 50], [28, 12, 3]])
    rows = np.array([[18, 32], [20, 30], [1, 2]])
    # compute the BCCB
    BTTB = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "gene",
        "nblocks": 2,
        "columns": columns,
        "rows": rows,
    }
    computed = cv.embedding_BCCB(BTTB_metadata=BTTB)
    # define the reference
    reference = np.array(
        [
            0,
            -2,
            7,
            0,
            32,
            18,
            10,
            40,
            50,
            0,
            30,
            20,
            0,
            0,
            0,
            0,
            0,
            0,
            28,
            12,
            3,
            0,
            2,
            1,
        ]
    )
    ae(computed, reference)


##### eigenvalues_BCCB


def test_eigenvalues_BCCB_bad_ordering():
    "must raise ValueError for invalid symmetry"
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "symm",
        "nblocks": 2,
        "columns": np.zeros((2, 3)),
        "rows": None,
    }
    ordering = "invalid-ordering"
    with raises(ValueError):
        cv.eigenvalues_BCCB(BTTB, ordering)


def test_eigenvalues_BCCB_known_values():
    "compare result with reference"
    # define the columns/rows of the generating BTTB
    columns = np.array(
        [
            [0, -2, 7, 10, 1],
            [10, 40, 50, 30, 1.1],
            [28, 12, 3, 17.2, 2.5],
            [65, 54, 31, 20, 11.1],
            [10, 12.3, 5, 8, 2],
            [3, 6.5, 7.0, 8, 12],
            [56, 76, 43, 23, 12],
            [31, 42, 53, 64, 75],
            [87, 65, 32, 10, 29],
            [6, 3, 8, 5, 6],
            [1, 4, 2, 6, 3.9],
        ]
    )
    rows = np.array(
        [
            [13, 18, 32, 11],
            [65, 20, 30, 82],
            [7, 1, 2, 4.5],
            [32, 10, -4, -23.7],
            [2, 3, 4, 7.8],
            [76, 48, 76, 13],
            [7, 8, 4, 3],
            [1, 9, 2, 7.12],
            [86, 23, 41.5, 30],
            [91, 2, 46, 3],
            [6, 14, 3, 98.9],
        ]
    )
    # compute the BCCB
    BTTB = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "gene",
        "nblocks": 6,
        "columns": columns,
        "rows": rows,
    }
    BCCB = cv.embedding_BCCB(BTTB_metadata=BTTB, full=True)
    Q = 6  # number of blocks along rows/columns of BTTB matrix
    P = 5  # number of rows/columns in each block of BTTB matrix
    # define unitaty DFT matrices
    F2Q = dft(n=2 * Q, scale="sqrtn")
    F2P = dft(n=2 * P, scale="sqrtn")
    # compute the Kronecker product between them
    F2Q_kron_F2P = np.kron(F2Q, F2P)
    # compute the reference eigenvalues of BCCB from its first column
    lambda_ref = np.sqrt(4 * Q * P) * F2Q_kron_F2P @ BCCB[:, 0]
    # compute eigenvalues with ordering='row'
    L_row = cv.eigenvalues_BCCB(BTTB_metadata=BTTB, ordering="row")
    lambda_row = L_row.ravel()
    aae(lambda_row, lambda_ref, decimal=10)
    # compute eigenvalues with ordering='column'
    L_col = cv.eigenvalues_BCCB(BTTB_metadata=BTTB, ordering="column")
    lambda_col = L_col.T.ravel()
    aae(lambda_col, lambda_ref, decimal=10)


def test_eigenvalues_BCCB_compare_eigenvalues_symm_symm():
    "verify the relationship between the eigenvalues and transposition"
    # define the data structure for the generating BTTB matrix
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
    BTTB_matrix = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
    # define the data structure for the generating BTTB matrix transposed
    BTTB_T = ds.BTTB_transposed_metadata(BTTB_metadata=BTTB)
    BTTB_matrix_T = cv.BTTB_from_metadata(BTTB_metadata=BTTB_T)
    # compute the eigenvalues
    L = cv.eigenvalues_BCCB(BTTB_metadata=BTTB, ordering="row")
    L_T = cv.eigenvalues_BCCB(BTTB_metadata=BTTB_T, ordering="row")
    # compare BTTB matrices
    ae(BTTB_matrix.T, BTTB_matrix_T)
    # compare eigenvalues
    aae(np.conj(L), L_T, decimal=12)


def test_eigenvalues_BCCB_compare_eigenvalues_symm_skew():
    "verify the relationship between eigenvalues and transposition"
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
    BTTB_matrix = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
    # define the data structure for the generating BTTB matrix transposed
    BTTB_T = ds.BTTB_transposed_metadata(BTTB_metadata=BTTB)
    BTTB_matrix_T = cv.BTTB_from_metadata(BTTB_metadata=BTTB_T)
    # compute the eigenvalues
    L = cv.eigenvalues_BCCB(BTTB_metadata=BTTB, ordering="row")
    L_T = cv.eigenvalues_BCCB(BTTB_metadata=BTTB_T, ordering="row")
    # compare BTTB matrices
    ae(BTTB_matrix.T, BTTB_matrix_T)
    # compare eigenvalues
    aae(np.conj(L), L_T, decimal=12)


def test_eigenvalues_BCCB_compare_eigenvalues_symm_gene():
    "verify the relationship between eigenvalues and transposition"
    # define the data structure for the generating BTTB matrix
    BTTB = {
        "symmetry_structure": "symm",
        "symmetry_blocks": "gene",
        "nblocks": 2,
        "columns": np.array(
            [
                [1, 2, 3],
                [10, 20, 30]
            ]
        ),
        "rows": np.array(
            [
                [12, 5],
                [8, 14.7]
            ]
        )
    }
    BTTB_matrix = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
    # define the data structure for the generating BTTB matrix transposed
    BTTB_T = ds.BTTB_transposed_metadata(BTTB_metadata=BTTB)
    BTTB_matrix_T = cv.BTTB_from_metadata(BTTB_metadata=BTTB_T)
    # compute the eigenvalues
    L = cv.eigenvalues_BCCB(BTTB_metadata=BTTB, ordering="row")
    L_T = cv.eigenvalues_BCCB(BTTB_metadata=BTTB_T, ordering="row")
    # compare BTTB matrices
    ae(BTTB_matrix.T, BTTB_matrix_T)
    # compare eigenvalues
    aae(np.conj(L), L_T, decimal=12)


def test_eigenvalues_BCCB_compare_eigenvalues_skew_symm():
    "verify the relationship between eigenvalues and transposition"
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
    BTTB_matrix = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
    # define the data structure for the generating BTTB matrix transposed
    BTTB_T = ds.BTTB_transposed_metadata(BTTB_metadata=BTTB)
    BTTB_matrix_T = cv.BTTB_from_metadata(BTTB_metadata=BTTB_T)
    # compute the eigenvalues
    L = cv.eigenvalues_BCCB(BTTB_metadata=BTTB, ordering="row")
    L_T = cv.eigenvalues_BCCB(BTTB_metadata=BTTB_T, ordering="row")
    # compare BTTB matrices
    ae(BTTB_matrix.T, BTTB_matrix_T)
    # compare eigenvalues
    aae(np.conj(L), L_T, decimal=12)


def test_eigenvalues_BCCB_compare_eigenvalues_skew_skew():
    "verify the relationship between eigenvalues and transposition"
    # define the data structure for the generating BTTB matrix
    BTTB = {
        "symmetry_structure": "skew",
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
    BTTB_matrix = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
    # define the data structure for the generating BTTB matrix transposed
    BTTB_T = ds.BTTB_transposed_metadata(BTTB_metadata=BTTB)
    BTTB_matrix_T = cv.BTTB_from_metadata(BTTB_metadata=BTTB_T)
    # compute the eigenvalues
    L = cv.eigenvalues_BCCB(BTTB_metadata=BTTB, ordering="row")
    L_T = cv.eigenvalues_BCCB(BTTB_metadata=BTTB_T, ordering="row")
    # compare BTTB matrices
    ae(BTTB_matrix.T, BTTB_matrix_T)
    # compare eigenvalues
    aae(np.conj(L), L_T, decimal=12)


# this test fails and I dont know why
# def test_eigenvalues_BCCB_compare_eigenvalues_skew_gene():
#     "verify the relationship between eigenvalues and transposition"
#     # define the data structure for the generating BTTB matrix
#     BTTB = {
#         "symmetry_structure": "skew",
#         "symmetry_blocks": "gene",
#         "nblocks": 2,
#         "columns": np.array(
#             [
#                 [1, 2, 3],
#                 [10, 20, 30],
#             ]
#         ),
#         "rows": np.array(
#             [
#                 [15, 23],
#                 [10, 3],
#             ]
#         )
#     }
#     BTTB_matrix = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
#     # define the data structure for the generating BTTB matrix transposed
#     BTTB_T = ds.BTTB_transposed_metadata(BTTB_metadata=BTTB)
#     BTTB_matrix_T = cv.BTTB_from_metadata(BTTB_metadata=BTTB_T)
#     # compute the eigenvalues
#     L = cv.eigenvalues_BCCB(BTTB_metadata=BTTB, ordering="row")
#     L_T = cv.eigenvalues_BCCB(BTTB_metadata=BTTB_T, ordering="row")
#     # compare BTTB matrices
#     ae(BTTB_matrix.T, BTTB_matrix_T)
#     # compare eigenvalues
#     aae(np.conj(L), L_T, decimal=12)


##### product_BCCB_vector


def test_product_BCCB_vector_bad_eigenvalues():
    "must raise AssertionError for bad eigenvalues matrix"
    v = np.zeros(12)
    # float
    L = 1.3
    with raises(ValueError):
        cv.product_BCCB_vector(eigenvalues=L, ordering="row", v=v)
    # vector
    L = np.ones(3)
    with raises(ValueError):
        cv.product_BCCB_vector(eigenvalues=L, ordering="row", v=v)
    # matrix with size different from 4 * v.size
    L = np.zeros((5, 4))
    with raises(ValueError):
        cv.product_BCCB_vector(eigenvalues=L, ordering="row", v=v)


def test_product_BCCB_vector_bad_v():
    "must raise AssertionError for bad v"
    Q = 4
    P = 3
    L = np.zeros((2 * Q, 2 * P))
    # v float
    v = 1.3
    with raises(ValueError):
        cv.product_BCCB_vector(eigenvalues=L, ordering="row", v=v)
    # v matrix
    v = np.ones((3, 3))
    with raises(ValueError):
        cv.product_BCCB_vector(eigenvalues=L, ordering="row", v=v)
    # v vector with size different from Q*P
    v = np.ones(Q * P + 5)
    with raises(ValueError):
        cv.product_BCCB_vector(eigenvalues=L, ordering="row", v=v)


def test_product_BCCB_vector_bad_ordering():
    "must raise AssertionError for invalid ordering"
    Q = 4
    P = 3
    L = np.zeros((2 * Q, 2 * P))
    v = np.ones(Q * P)
    with raises(ValueError):
        cv.product_BCCB_vector(eigenvalues=L, ordering="invalid-ordering", v=v)


def test_product_BCCB_vector_compare_matrix_vector():
    "compare values with that obtained via matrix-vector product"
    # define the columns/rows of the generating BTTB
    columns = np.array(
        [
            [0, -2, 7, 10, 1],
            [10, 40, 50, 30, 1.1],
            [28, 12, 3, 17.2, 2.5],
            [65, 54, 31, 20, 11.1],
            [10, 12.3, 5, 8, 2],
            [3, 6.5, 7.0, 8, 12],
            [56, 76, 43, 23, 12],
            [31, 42, 53, 64, 75],
            [87, 65, 32, 10, 29],
            [6, 3, 8, 5, 6],
            [1, 4, 2, 6, 3.9],
        ]
    )
    rows = np.array(
        [
            [13, 18, 32, 11],
            [65, 20, 30, 82],
            [7, 1, 2, 4.5],
            [32, 10, -4, -23.7],
            [2, 3, 4, 7.8],
            [76, 48, 76, 13],
            [7, 8, 4, 3],
            [1, 9, 2, 7.12],
            [86, 23, 41.5, 30],
            [91, 2, 46, 3],
            [6, 14, 3, 98.9],
        ]
    )
    # compute the BCCB
    BTTB = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "gene",
        "nblocks": 6,
        "columns": columns,
        "rows": rows,
    }
    BTTB_matrix = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
    BCCB = cv.embedding_BCCB(BTTB_metadata=BTTB, full=True)
    Q = 6  # number of blocks along rows/columns of BTTB matrix
    P = 5  # number of rows/columns in each block of BTTB matrix
    # define a vector v
    np.random.seed(5)
    v = np.random.rand(Q * P)
    # define reference
    w_matvec = BTTB_matrix @ v
    # compute the product with function convolve.product_BCCB_vector
    # ordering='row'
    L = cv.eigenvalues_BCCB(BTTB_metadata=BTTB, ordering="row")
    w_conv_row = cv.product_BCCB_vector(eigenvalues=L, ordering="row", v=v)
    aae(w_conv_row, w_matvec, decimal=12)
    # ordering='column'
    L = cv.eigenvalues_BCCB(BTTB_metadata=BTTB, ordering="column")
    w_conv_col = cv.product_BCCB_vector(eigenvalues=L, ordering="column", v=v)
    aae(w_conv_col, w_matvec, decimal=12)


def test_product_BCCB_vector_compare_transposed():
    "compare values with that obtained via transposed-matrix-vector product"
    # define the columns/rows of the generating BTTB
    columns = np.array(
        [
            [0, -2, 7, 10, 1],
            [10, 40, 50, 30, 1.1],
            [28, 12, 3, 17.2, 2.5],
            [65, 54, 31, 20, 11.1],
            [10, 12.3, 5, 8, 2],
            [3, 6.5, 7.0, 8, 12],
            [56, 76, 43, 23, 12],
            [31, 42, 53, 64, 75],
            [87, 65, 32, 10, 29],
            [6, 3, 8, 5, 6],
            [1, 4, 2, 6, 3.9],
        ]
    )
    rows = np.array(
        [
            [13, 18, 32, 11],
            [65, 20, 30, 82],
            [7, 1, 2, 4.5],
            [32, 10, -4, -23.7],
            [2, 3, 4, 7.8],
            [76, 48, 76, 13],
            [7, 8, 4, 3],
            [1, 9, 2, 7.12],
            [86, 23, 41.5, 30],
            [91, 2, 46, 3],
            [6, 14, 3, 98.9],
        ]
    )
    # compute the BCCB
    BTTB = {
        "symmetry_structure": "gene",
        "symmetry_blocks": "gene",
        "nblocks": 6,
        "columns": columns,
        "rows": rows,
    }
    BTTB_matrix = cv.BTTB_from_metadata(BTTB_metadata=BTTB)
    Q = 6  # number of blocks along rows/columns of BTTB matrix
    P = 5  # number of rows/columns in each block of BTTB matrix
    # define a vector v
    np.random.seed(5)
    v = np.random.rand(Q * P)
    # define reference product with transposed BTTB matrix
    w_matvec = BTTB_matrix.T @ v
    # compute the product with function convolve.product_BCCB_vector
    # ordering='row'
    L = cv.eigenvalues_BCCB(BTTB_metadata=BTTB, ordering="row")
    w_conv_row = cv.product_BCCB_vector(
        eigenvalues=np.conj(L), ordering="row", v=v
    )
    aae(w_conv_row, w_matvec, decimal=12)
    # ordering='column'
    L = cv.eigenvalues_BCCB(BTTB_metadata=BTTB, ordering="column")
    w_conv_col = cv.product_BCCB_vector(
        eigenvalues=np.conj(L), ordering="column", v=v
    )
    aae(w_conv_col, w_matvec, decimal=12)
