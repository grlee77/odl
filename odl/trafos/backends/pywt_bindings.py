# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Bindings to the PyWavelets backend for wavelet transforms.

`PyWavelets <https://pywavelets.readthedocs.io/>`_ is a Python library
for wavelet transforms in arbitrary dimensions, featuring a large number
of built-in wavelet filters.
"""

from __future__ import print_function, division, absolute_import
from itertools import product
import numpy as np
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
from pywt._utils import _wavelets_per_axis, _modes_per_axis
from pywt._multilevel import _check_level


__all__ = ('PAD_MODES_ODL2PYWT', 'PYWT_SUPPORTED_MODES', 'PYWT_AVAILABLE',
           'pywt_wavelet', 'pywt_pad_mode', 'pywt_max_nlevels',
           'ncoeffs_from_shapes', 'wavedecn_shapes', 'ravel_coeffs',
           'unravel_coeffs')


PAD_MODES_ODL2PYWT = {'constant': 'zero',
                      'periodic': 'periodic',
                      'symmetric': 'symmetric',
                      'order0': 'constant',
                      'order1': 'smooth',
                      'pywt_periodic': 'periodization',
                      # Upcoming version of Pywavelets adds this
                      # 'reflect': 'reflect'
                      }
PYWT_SUPPORTED_MODES = PAD_MODES_ODL2PYWT.values()


def pywt_wavelet(wavelet):
    """Convert ``wavelet`` to a `pywt.Wavelet` instance."""
    if isinstance(wavelet, pywt.Wavelet):
        return wavelet
    else:
        return pywt.Wavelet(wavelet)


def pywt_pad_mode(pad_mode, pad_const=0):
    """Convert ODL-style padding mode to pywt-style padding mode."""
    pad_mode = str(pad_mode).lower()
    if pad_mode == 'constant' and pad_const != 0.0:
        raise ValueError('constant padding with constant != 0 not supported '
                         'for `pywt` back-end')
    try:
        return PAD_MODES_ODL2PYWT[pad_mode]
    except KeyError:
        raise ValueError("`pad_mode` '{}' not understood".format(pad_mode))


def _prep_axes_wavedecn(shape, axes):
    if len(shape) < 1:
        raise ValueError("Expected at least 1D input data.")
    ndim = len(shape)
    if np.isscalar(axes):
        axes = (axes, )
    if axes is None:
        axes = range(ndim)
    else:
        axes = tuple(axes)
    if len(axes) != len(set(axes)):
        raise ValueError("The axes passed to wavedecn must be unique.")
    try:
        axes_shapes = [shape[ax] for ax in axes]
    except IndexError:
        raise ValueError("Axis greater than data dimensions")
    ndim_transform = len(axes)
    return axes, axes_shapes, ndim_transform


def _prepare_coeffs_axes(coeffs, axes):
    """Helper function to check type of coeffs and axes.

    This code is used by both coeffs_to_array and ravel_coeffs.
    """
    from pywt._multilevel import (_coeffs_wavedec_to_wavedecn,
                                  _coeffs_wavedec2_to_wavedecn)
    if not isinstance(coeffs, list) or len(coeffs) == 0:
        raise ValueError("input must be a list of coefficients from wavedecn")
    if coeffs[0] is None:
        raise ValueError("coeffs_to_array does not support missing "
                         "coefficients.")
    if not isinstance(coeffs[0], np.ndarray):
        raise ValueError("first list element must be a numpy array")
    ndim = coeffs[0].ndim

    if len(coeffs) > 1:
        # convert wavedec or wavedec2 format coefficients to waverecn format
        if isinstance(coeffs[1], dict):
            pass
        elif isinstance(coeffs[1], np.ndarray):
            coeffs = _coeffs_wavedec_to_wavedecn(coeffs)
        elif isinstance(coeffs[1], (tuple, list)):
            coeffs = _coeffs_wavedec2_to_wavedecn(coeffs)
        else:
            raise ValueError("invalid coefficient list")

    if len(coeffs) == 1:
        # no detail coefficients were found
        return coeffs, axes, ndim, None

    # Determine the number of dimensions that were transformed via key length
    ndim_transform = len(list(coeffs[1].keys())[0])
    if axes is None:
        if ndim_transform < ndim:
            raise ValueError(
                "coeffs corresponds to a DWT performed over only a subset of "
                "the axes.  In this case, axes must be specified.")
        axes = np.arange(ndim)

    if len(axes) != ndim_transform:
        raise ValueError(
            "The length of axes doesn't match the number of dimensions "
            "transformed.")

    return coeffs, axes, ndim, ndim_transform


def _determine_coeff_array_size(coeffs):
    arr_size = np.asarray(coeffs[0].size)
    for d in coeffs[1:]:
        for k, v in d.items():
            arr_size += v.size
    return arr_size


def ravel_coeffs(coeffs, axes=None):
    """
    Ravel a wavelet coefficient list from `wavedecn` into a single 1D array.

    Parameters
    ----------
    coeffs: array-like
        dictionary of wavelet coefficients as returned by pywt.wavedecn
    axes : sequence of ints, optional
        Axes over which the DWT that created ``coeffs`` was performed.  The
        default value of ``None`` corresponds to all axes.

    Returns
    -------
    coeff_arr : array-like
        Wavelet transform coefficient array.
    coeff_slices : list
        List of slices corresponding to each coefficient.  As a 2D example,
        `coeff_arr[coeff_slices[1]['dd']]` would extract the first level detail
        coefficients from `coeff_arr`.
    coeff_shapes : list
        List of shapes corresponding to each coefficient.  For example, in 2D,
        coeff_shapes[1]['dd'] would contain the original shape of the first
        level detail coefficients array.

    See Also
    --------
    unravel_coeffs : the inverse of ravel_coeffs

    Examples
    --------
    >>> import pywt
    >>> cam = pywt.data.camera()
    >>> coeffs = pywt.wavedecn(cam, wavelet='db2', level=3)
    >>> arr, coeff_slices, coeff_shapes = pywt.ravel_coeffs(coeffs)

    """
    coeffs, axes, ndim, ndim_transform = _prepare_coeffs_axes(coeffs, axes)

    # initialize with the approximation coefficients.
    a_coeffs = coeffs[0]
    a_size = a_coeffs.size

    if len(coeffs) == 1:
        # only a single approximation coefficient array was found
        return a_coeffs, [[slice(None)] * ndim]

    # preallocate output array
    arr_size = _determine_coeff_array_size(coeffs)
    coeff_arr = np.empty((arr_size, ), dtype=a_coeffs.dtype)

    a_slice = slice(a_size)
    coeff_arr[a_slice] = a_coeffs.ravel()

    # initialize list of coefficient slices
    coeff_slices = []
    coeff_shapes = []
    coeff_slices.append(a_slice)
    coeff_shapes.append(coeffs[0].shape)

    # loop over the detail cofficients, embedding them in coeff_arr
    ds = coeffs[1:]
    offset = a_size
    for coeff_dict in ds:
        # new dictionaries for detail coefficient slices and shapes
        coeff_slices.append({})
        coeff_shapes.append({})
        if np.any([d is None for d in coeff_dict.values()]):
            raise ValueError("coeffs_to_array does not support missing "
                             "coefficients.")
        for key, d in coeff_dict.items():
            sl = slice(offset, offset + d.size)
            offset += d.size
            coeff_arr[sl] = d.ravel()
            coeff_slices[-1][key] = sl
            coeff_shapes[-1][key] = d.shape
    return coeff_arr, coeff_slices, coeff_shapes


def unravel_coeffs(arr, coeff_slices, coeff_shapes, output_format='wavedecn'):
    """
    Convert a raveled array of coefficients back to a list compatible with
    `waverecn`.

    Parameters
    ----------

    arr: array-like
        An array containing all wavelet coefficients.  This should have been
        generated via `coeffs_to_array`.
    coeff_slices : list of tuples
        List of slices corresponding to each coefficient as obtained from
        `array_to_coeffs`.
    output_format : {'wavedec', 'wavedec2', 'wavedecn'}
        Make the form of the coefficients compatible with this type of
        multilevel transform.

    Returns
    -------
    coeffs: array-like
        Wavelet transform coefficient array.

    See Also
    --------
    coeffs_to_array : the inverse of array_to_coeffs

    Examples
    --------
    >>> import pywt
    >>> from numpy.testing import assert_array_almost_equal
    >>> cam = pywt.data.camera()
    >>> coeffs = pywt.wavedecn(cam, wavelet='db2', level=3)
    >>> arr, coeff_slices, coeff_shapes = pywt.ravel_coeffs(coeffs)
    >>> coeffs_from_arr = pywt.unravel_coeffs(arr, coeff_slices, coeff_shapes)
    >>> cam_recon = pywt.waverecn(coeffs_from_arr, wavelet='db2')
    >>> assert_array_almost_equal(cam, cam_recon)

    """
    arr = np.asarray(arr)
    coeffs = []
    if len(coeff_slices) == 0:
        raise ValueError("empty list of coefficient slices")
    elif len(coeff_shapes) == 0:
        raise ValueError("empty list of coefficient shapes")
    elif len(coeff_shapes) != len(coeff_slices):
        raise ValueError("coeff_shapes and coeff_slices have unequal length")
    else:
        coeffs.append(arr[coeff_slices[0]].reshape(coeff_shapes[0]))

    # difference coefficients at each level
    for n in range(1, len(coeff_slices)):
        slice_dict = coeff_slices[n]
        shape_dict = coeff_shapes[n]
        if output_format == 'wavedec':
            d = arr[slice_dict['d']].reshape(shape_dict['d'])
        elif output_format == 'wavedec2':
            d = (arr[slice_dict['da']].reshape(shape_dict['da']),
                 arr[slice_dict['ad']].reshape(shape_dict['ad']),
                 arr[slice_dict['dd']].reshape(shape_dict['dd']))
        elif output_format == 'wavedecn':
            d = {}
            for k, v in coeff_slices[n].items():
                d[k] = arr[v].reshape(shape_dict[k])
        else:
            raise ValueError(
                "Unrecognized output format: {}".format(output_format))
        coeffs.append(d)
    return coeffs


def wavedecn_shapes(shape, wavelet, mode='symmetric', level=None, axes=None):
    """Precompute the shapes of all wavedecn coefficients."""
    axes, axes_shapes, ndim_transform = _prep_axes_wavedecn(shape, axes)
    wavelets = _wavelets_per_axis(wavelet, axes)
    modes = _modes_per_axis(mode, axes)
    dec_lengths = [w.dec_len for w in wavelets]

    level = _check_level(min(axes_shapes), max(dec_lengths), level)

    shapes = []
    for i in range(level):
        detail_keys = [''.join(c) for c in product('ad', repeat=len(axes))]
        new_shapes = {k: list(shape) for k in detail_keys}
        for axis, wav, mode in zip(axes, wavelets, modes):
            s = pywt.dwt_coeff_len(shape[axis], filter_len=wav.dec_len,
                                   mode=mode)
            for k in detail_keys:
                new_shapes[k][axis] = s
        for k, v in new_shapes.items():
            new_shapes[k] = tuple(v)
        shapes.append(new_shapes)
        shape = new_shapes.pop('a' * ndim_transform)
    shapes.append(shape)
    shapes.reverse()
    return shapes


def ncoeffs_from_shapes(shapes):
    """Total number of wavedecn coefficients."""
    ncoeffs = np.prod(shapes[0])
    for d in shapes[1:]:
        for k, v in d.items():
            ncoeffs += np.prod(v)
    return ncoeffs


def pywt_max_nlevels(shape, wavelet, axes=None):
    """Return the maximum number of wavelet levels.

    Parameters
    ----------
    shape : sequence of ints
        Shape of an input to the transform.
    wavelet : string or `pywt.Wavelet`
        Specification of the wavelet to be used in the transform.
        If a string is given, it is converted to a `pywt.Wavelet`.
        Use `pywt.wavelist` to get a list of available wavelets.  This can also
        be a list of wavelets equal in length to the number of axes to be
        transformed.
    axes : tuple of int or None
        The set of axes to transform.  The default (None) is to transform all
        axes.

    Returns
    -------
    max_nlevels : int
        Maximum value for the nlevels option.

    Examples
    --------
    Find maximum nlevels for Haar wavelet:

    >>> pywt_max_nlevels([10], 'haar')
    3
    >>> pywt_max_nlevels([1024], 'haar')
    10

    For multiple axes, the maximum nlevels is determined by the smallest shape:

    >>> pywt_max_nlevels([10, 1024], 'haar')
    3
    """
    axes, axes_shapes, ndim_transform = _prep_axes_wavedecn(shape, axes)
    wavelets = _wavelets_per_axis(wavelet, axes)
    max_levels = [pywt.dwt_max_level(n, wav.dec_len)
                  for n, wav in zip(axes_shapes, wavelets)]
    return min(max_levels)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests(skip_if=not PYWT_AVAILABLE)
