import numpy as np
from scipy.ndimage import gaussian_filter1d


def lfo_dither(path, sigma=9, lfo_period=100, dither_magnitude=2.0):
    """
        Processes a path of y, x coordinates using vectorized operations.

        Args:
            path (list): A list of (y,x) coordinates representing the path.
            sigma (int, optional): Standard deviation for Gaussian smoothing. Defaults to 9.
            lfo_period (int, optional): Period of the sine wave (LFO). Defaults to 100.
            dither_magnitude (float, optional): Magnitude of dither to add. Defaults to 2.0.

        Returns:
            list: A list of tuples representing the processed path.
        """

    # Calculate displacements
    path_nd = np.array(path)
    displacements = np.diff(path_nd, axis=0)
    displacements = np.vstack((displacements, displacements[-1]))

    # Rotate coordinates (y = x, x = -y) for ortho
    gradient = np.array([displacements[:, 1], -displacements[:, 0]]).T.astype(np.float32)

    # Smooth rotated coordinates with 1D Gaussian kernel
    smoothed_y = gaussian_filter1d(gradient[:, 0], sigma=sigma)
    smoothed_x = gaussian_filter1d(gradient[:, 1], sigma=sigma)
    smoothed_grad = np.column_stack((smoothed_y, smoothed_x))

    # Generate sine wave
    wave_length = lfo_period
    lfo = np.sin(np.linspace(0, 2 * np.pi, len(path)) * (len(path) / wave_length))

    # Pointwise multiplication
    modulated_grad = dither_magnitude * smoothed_grad * lfo[:, np.newaxis]

    # Add original path
    result = path + modulated_grad

    # Output as list of tuples
    result_tuples = [tuple(point) for point in result]

    return result_tuples
