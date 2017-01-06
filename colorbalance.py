"""
Color balance using colorchecker/color card.

Color balance includes two gamma correction algorithms described in [1].
The new gamma correction algorithm yields much better accuracy than the classic
one. Ground-truth RGB values of CameraTrax [2] and X-Rite colorchecker [3]
are also included.


The API consists of functions to:
* Extract colors from cropped image of a known color pallette.
* Estimate coefficient matrices of correction function
* Apply correction function to a image captured in the same condition as the
  input color card.


Supported color spaces
----------------------
* RGB : Red Green Blue.
* Other 3-channels color spaces


References
----------
.. [1] Constantinou (2013) A comparison of color correction algorithms for
       endoscopic cameras
.. [2] CameraTrax color card - https://www.cameratrax.com/color_balance_2x3.php
...[3] http://xritephoto.com/documents/literature/en/ColorData-1p_EN.pdf
"""

import numpy as np
from scipy import optimize
from scipy import stats


ColorCheckerRGB_CameraTrax = np.asarray(
    [[115., 196., 91., 94., 129., 98., 223., 58., 194., 93., 162., 229.,
      49., 77., 173., 241., 190., 0., 242., 203., 162., 120., 84., 50.],
     [83., 147., 122., 108., 128., 190., 124., 92., 82., 60., 190., 158.,
      66., 153., 57., 201., 85., 135., 243., 203., 163., 120., 84., 50.],
     [68., 127., 155., 66., 176., 168., 47., 174., 96., 103., 62., 41.,
      147., 71., 60., 25., 150., 166., 245., 204., 162., 120., 84., 52.]])

ColorCheckerRGB_XRite = np.asarray(
    [[115., 194., 98., 87., 133., 103., 214., 80., 193., 94., 157., 224.,
      56., 70., 175., 231., 187., 8., 243., 200., 160., 122., 85., 52.],
     [82., 150., 122., 108., 128., 189., 126., 91., 90., 60., 188., 163.,
      61., 148., 54., 199., 86., 133., 243., 200., 160., 122., 85., 52.],
     [68., 130., 157., 67., 177., 170., 44., 166., 99., 108., 64., 46.,
      150., 73., 60., 31., 149., 161., 242., 200., 160., 121., 85., 52.]])


def _classic_gamma_correction_model(colors, color_alpha, color_constant,
                                    color_gamma):
    """Apply color correction to a list of colors.
    This uses classic gamma correction algorithm:
       |R_out|   |alpha_R    0       0   |   |R_in|^|gamma_R|   |beta_R|
       |G_out| = |   0    alpha_G    0   | * |G_in| |gamma_G| + |beta_G|
       |B_out|   |   0       0    alpha_B|   |B_in| |gamma_B|   |beta_B|

    """
    assert(colors.shape[0] == 3)
    assert(color_alpha.size == 3)
    assert(color_constant.size == 3)
    assert(color_gamma.size == 3)

    corrected_colors = np.zeros_like(colors)
    for j in range(3):
        corrected_colors[j, :] = \
            color_alpha[j] * np.power(colors[j, :], color_gamma[j]) + \
            color_constant[j]
    return corrected_colors


def _gamma_correction_model(colors, color_alpha, color_constant,
                            color_gamma):
    """Apply color correction to a list of colors.
    This uses a modified gamma correction algorithm:
       |R_out'|   |alpha_11 alpha_12 alpha_13|   |R_in|   |beta_R|
       |G_out'| = |alpha_21 alpha_22 alpha_23| * |G_in| + |beta_G|
       |B_out'|   |alpha_31 alpha_32 alpha_33|   |B_in|   |beta_B|

       |R_out|         |R_out'/255|^|gamma_R|
       |G_out| = 255 * |G_out'/255| |gamma_G|
       |B_out|         |B_out'/255| |gamma_B|

    """
    assert(colors.shape[0] == 3)
    assert(color_alpha.shape == (3, 3))
    assert(color_constant.size == 3)
    assert(color_gamma.size == 3)

    scaled_colors = np.dot(color_alpha, colors) + color_constant
    np.clip(scaled_colors, 0, None, scaled_colors)  # set min values to zeros # I (MEF) commented this. This is now like the pipeline!!
    corrected_colors = np.zeros_like(scaled_colors)
    for j in range(3):
        corrected_colors[j, :] = 255.0*np.power(scaled_colors[j, :]/255.0,
                                                color_gamma[j])
    return corrected_colors


def _get_color_error(args2, true_colors, actual_colors, algorithm):
    """Calculated color error after applying color correction.
    This function is used in `get_color_correction_parameters` function.

    """
    if algorithm == "classic_gamma_correction":
        color_alpha = args2[:3].reshape([3, 1])
        color_constant = args2[3:6].reshape([3, 1])
        # forced non-negative exponential component
        color_gamma = np.abs(args2[6:9].reshape([3, 1]))
        corrected_colors = \
            _classic_gamma_correction_model(actual_colors, color_alpha,
                                            color_constant, color_gamma)
    elif algorithm == "gamma_correction":
        color_alpha = args2[:9].reshape([3, 3])
        color_constant = args2[9:12].reshape([3, 1])
        # forced non-negative exponential component
        color_gamma = np.abs(args2[12:15].reshape([3, 1]))
        corrected_colors = \
            _gamma_correction_model(actual_colors, color_alpha,
                                    color_constant, color_gamma)
    else:
        raise ValueError("Unsupported algorithm {}.".format(algorithm))

    diff_colors = true_colors - corrected_colors
    errors = np.sqrt(np.sum(diff_colors * diff_colors, axis=0)).tolist()
    return errors


def get_color_correction_parameters(true_colors, actual_colors,
                                    algorithm="gamma_correction"):
    """Estimate parameters of color correction function.

    Parameters
    ----------
    true_colors : 3xN ndarray
        The input ground-truth colors.
    actual_colors : 3xN ndarray
        The input actual color as captured in image.
    algorithm : string
        The correction algorithm, either `classic_gamma_correction` or
        `gamma_correction` (default)

    Returns
    -------
    color_alpha : ndarray
        The scaling coefficient.
    color_constant : ndarray
        The color constant component.
    color_gamma : ndarray
        The gamma coefficient or the exponential component of
        correction function.

    Raises
    ------
    ValueError
        If the input algorithm is not supported.
    """
    if algorithm == "classic_gamma_correction":
        color_alpha = np.ones([3, 1])
    elif algorithm == "gamma_correction":
        color_alpha = np.eye(3)
    else:
        raise ValueError("Unsupported algorithm {}.".format(algorithm))

    color_constant = np.zeros([3, 1])
    color_gamma = np.ones([3, 1])

    args_init = np.concatenate((color_alpha.reshape([color_alpha.size]),
                                color_constant.reshape([color_constant.size]),
                                color_gamma.reshape([color_gamma.size])))
    args_refined, _ = \
        optimize.leastsq(_get_color_error, args_init,
                         args=(true_colors, actual_colors, algorithm),
                         maxfev = 20000)

    if algorithm == "classic_gamma_correction":
        color_alpha = args_refined[:3].reshape([3, 1])
        color_constant = args_refined[3:6].reshape([3, 1])
        # forced non-negative exponential compnent
        color_gamma = np.abs(args_refined[6:9].reshape([3, 1]))
    elif algorithm == "gamma_correction":
        color_alpha = args_refined[:9].reshape([3, 3])
        color_constant = args_refined[9:12].reshape([3, 1])
        # forced non-negative exponential compnent
        color_gamma = np.abs(args_refined[12:15].reshape([3, 1]))
    else:
        raise ValueError("Unsupported algorithm {}.".format(algorithm))

    return color_alpha, color_constant, color_gamma


def get_colorcard_colors(color_card, grid_size):
    """Extract color information from a cropped image of a color card.
    containing squares of different colors.

    Parameters
    ----------
    color_card : ndarray
        The input cropped image containing only color card.
    grid_size : list, [horizontal_grid_size, vertical_grid_size]
        The number of columns and rows in color card.

    Returns
    -------
    colors : 3xN ndarray
        List of colors with color channels go along the first array axis.
    """
    grid_cols, grid_rows = grid_size
    colors = np.zeros([3, grid_rows * grid_cols])
    colors_std = np.zeros(grid_rows * grid_cols)

    sample_size_row = int(0.2 * color_card.shape[0] / grid_rows)
    sample_size_col = int(0.2 * color_card.shape[1] / grid_cols)
    for row in range(grid_rows):
        for col in range(grid_cols):
            r = int((row + 0.5) * color_card.shape[0] / grid_rows)
            c = int((col + 0.5) * color_card.shape[1] / grid_cols)
            i = row * grid_cols + col
            for j in range(colors.shape[0]):
                channel = color_card[r - sample_size_row:r + sample_size_row,
                                     c - sample_size_col:c + sample_size_col,
                                     j]
                colors[j, i] = np.median(channel.astype(np.float))
                colors_std[i] = colors_std[i] + np.std(channel.astype(np.float))

    return colors, colors_std


def correct_color(image, color_alpha, color_constant, color_gamma,
                  algorithm="gamma_correction"):
    """Apply color correction to an image.

    Parameters
    ----------
    image : ndarray
        The input image to correct.
    color_alpha : ndarray
        The scaling coefficient.
    color_constant : ndarray
        The color constant component.
    color_gamma : ndarray
        The gamma coefficient or the exponential component of
        correction function.
    algorithm : string
        The correction algorithm, either `classic_gamma_correction` or
        `gamma_correction` (default)

    Returns
    -------
    corrected_image : ndarray
        The color-corrected image of the same size as input image.

    Raises
    ------
    ValueError
        If the input algorithm is not supported.
    
    """
    # first turn it to [M*N, 3] matrix, then [3,M*N] matrix
    assert(len(image.shape) == 3)
    colors = image.reshape([image.shape[0] * image.shape[1], 3])
    colors = colors.transpose()

    if algorithm == "classic_gamma_correction":
        corrected_colors = \
            _classic_gamma_correction_model(colors, color_alpha,
                                            color_constant, color_gamma)
    elif algorithm == "gamma_correction":
        corrected_colors = \
            _gamma_correction_model(colors, color_alpha,
                                    color_constant, color_gamma)
    else:
        raise ValueError("Unsupported algorithm {}.".format(algorithm))

    # now turn it back to [M*N, 3] matrix, then [M,N,3] matrix
    corrected_colors = corrected_colors.transpose()
    corrected_image = corrected_colors.reshape([image.shape[0],
                                                image.shape[1], 3])
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
    return corrected_image
