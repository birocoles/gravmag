import numpy as np
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_equal as ae
from pytest import raises
from .. import eqlayer


#### kernel_matrix_monopoles

def test_kernel_matrix_monopoles_invalid_field():
    "Verify if passing an invalid field raises an error"
    # single source
    S = {
        'x' : np.array([0.]),
        'y' : np.array([0.]),
        'z' : np.array([0.])
    }
    # singe data point
    P = {
        'x' : np.array([  0.]),
        'y' : np.array([  0.]),
        'z' : np.array([-10.])
    }
    # string
    with raises(ValueError):
        eqlayer.kernel_matrix_monopoles(data_points=P, source_points=S, field='invalid-field')
    # float
    with raises(ValueError):
        eqlayer.kernel_matrix_monopoles(data_points=P, source_points=S, field=3.4)
    # tuple
    with raises(ValueError):
        eqlayer.kernel_matrix_monopoles(data_points=P, source_points=S, field=(9,0))
    # list
    with raises(ValueError):
        eqlayer.kernel_matrix_monopoles(data_points=P, source_points=S, field=[4., 5.6, 3])
    # uppercase
    with raises(ValueError):
        eqlayer.kernel_matrix_monopoles(data_points=P, source_points=S, field='X')


def test_kernel_matrix_monopoles_shape():
    "Verify if returns the correct shape"
    # single source
    S = {
        'x' : np.ones(5), 
        'y' : np.ones(5), 
        'z' : np.ones(5)
    }
    # singe data point
    P = {
        'x' : np.zeros(5),
        'y' : np.zeros(5),
        'z' : np.zeros(5)
    }
    G = eqlayer.kernel_matrix_monopoles(data_points=P, source_points=S, field='y')
    ae(G.shape, (5,5))


##### kernel_matrix_dipoles


def test_kernel_matrix_dipoles_invalid_field():
    "Verify if passing an invalid field raises an error"
    # single source
    S = {
        'x' : np.array([0.]),
        'y' : np.array([0.]),
        'z' : np.array([0.])
    }
    # singe data point
    P = {
        'x' : np.array([  0.]),
        'y' : np.array([  0.]),
        'z' : np.array([-10.])
    }
    # inc, dec, inct, dect
    inc, dec = -34.5, 19.
    inct, dect = 10, 28.1

    # string
    with raises(ValueError):
        eqlayer.kernel_matrix_dipoles(data_points=P, source_points=S, inc=inc, dec=dec, field='invalid-field', inct=inct, dect=dect)
    # float
    with raises(ValueError):
        eqlayer.kernel_matrix_dipoles(data_points=P, source_points=S, inc=inc, dec=dec, field=3.4, inct=inct, dect=dect)
    # tuple
    with raises(ValueError):
        eqlayer.kernel_matrix_dipoles(data_points=P, source_points=S, inc=inc, dec=dec, field=(9,0), inct=inct, dect=dect)
    # list
    with raises(ValueError):
        eqlayer.kernel_matrix_dipoles(data_points=P, source_points=S, inc=inc, dec=dec, field=[4., 5.6, 3], inct=inct, dect=dect)
    # uppercase
    with raises(ValueError):
        eqlayer.kernel_matrix_dipoles(data_points=P, source_points=S, inc=inc, dec=dec, field='X', inct=inct, dect=dect)


def test_kernel_matrix_dipoles_invalid_inc_dec():
    "Verify if passing invalid inc and/or dec raises an error for field not 't'"
    # single source
    S = {
        'x' : np.array([0.]),
        'y' : np.array([0.]),
        'z' : np.array([0.])
    }
    # singe data point
    P = {
        'x' : np.array([  0.]),
        'y' : np.array([  0.]),
        'z' : np.array([-10.])
    }
    # inct, dect
    inct, dect = 'invalid-inct', 'invalid-dect'

    # inc string, dec ok
    inc, dec = 'invalid-inc', 23.
    with raises(ValueError):
        eqlayer.kernel_matrix_dipoles(data_points=P, source_points=S, inc=inc, dec=dec, field='x', inct=inct, dect=dect)
    # inc ok, dec string
    inc, dec = 34., 'invalid-dec'
    with raises(ValueError):
        eqlayer.kernel_matrix_dipoles(data_points=P, source_points=S, inc=inc, dec=dec, field='y', inct=inct, dect=dect)
    # inc tuple, dec ok
    inc, dec = (1.6,), 32
    with raises(ValueError):
        eqlayer.kernel_matrix_dipoles(data_points=P, source_points=S, inc=inc, dec=dec, field='z', inct=inct, dect=dect)
    # inc ok, dec list
    inc, dec = 12, [23.]
    with raises(ValueError):
        eqlayer.kernel_matrix_dipoles(data_points=P, source_points=S, inc=inc, dec=dec, field='x', inct=inct, dect=dect)
    # inc complex, dec ok
    inc, dec = 1-5j, 23.
    with raises(ValueError):
        eqlayer.kernel_matrix_dipoles(data_points=P, source_points=S, inc=inc, dec=dec, field='y', inct=inct, dect=dect)


def test_kernel_matrix_dipoles_ignore_inct_dect():
    "Verify if passing invalid inct and/or dect raises an error for the case in which field is 't'"
    # single source
    S = {
        'x' : np.array([0.]),
        'y' : np.array([0.]),
        'z' : np.array([0.])
    }
    # singe data point
    P = {
        'x' : np.array([  0.]),
        'y' : np.array([  0.]),
        'z' : np.array([-10.])
    }
    # inc, dec
    inc, dec = 10, 28.1

    # inc, dec string
    inct, dect = 'invalid-inc', 'invalid-dec'
    with raises(ValueError):
        eqlayer.kernel_matrix_dipoles(data_points=P, source_points=S, inc=inc, dec=dec, field='t', inct=inct, dect=dect)
    # inc, dec complex
    inct, dect = 28+3j, 14.5
    with raises(ValueError):
        eqlayer.kernel_matrix_dipoles(data_points=P, source_points=S, inc=inc, dec=dec, field='t', inct=inct, dect=dect)
    # inc, dec list
    inct, dect = [13.,], 24.
    with raises(ValueError):
        eqlayer.kernel_matrix_dipoles(data_points=P, source_points=S, inc=inc, dec=dec, field='t', inct=inct, dect=dect)
    # inc, dec tuple
    inct, dect = 13., (24.,)
    with raises(ValueError):
        eqlayer.kernel_matrix_dipoles(data_points=P, source_points=S, inc=inc, dec=dec, field='t', inct=inct, dect=dect)


def test_kernel_matrix_dipoles_shape():
    "Verify if returns the correct shape"
    # single source
    S = {
        'x' : np.ones(5), 
        'y' : np.ones(5), 
        'z' : np.ones(5)
    }
    # singe data point
    P = {
        'x' : np.zeros(5),
        'y' : np.zeros(5),
        'z' : np.zeros(5)
    }
    G = eqlayer.kernel_matrix_dipoles(data_points=P, source_points=S, inc=4., dec=5, field='y')
    ae(G.shape, (5,5))