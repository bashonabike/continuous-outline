import math
import numpy as np
from scipy.signal import convolve

import helpers.mazify.temp_options as options
import helpers.mazify.EdgeNode as EdgeNode
import helpers.mazify.MazeSections as sections

class EdgePath:
    def __init__(self, path_num, path_raw, maze_sections: sections.MazeSections):
        self.path, self.num = [], path_num
        self.parse_path(path_raw, maze_sections)


    def parse_path(self, path, maze_sections: sections.MazeSections):
        #Determine explicit directions
        dys,dxs = [],[]
        for i in range(len(path)):
            i_next = (i + 1) % len(path)
            dy, dx = path[i_next][0][0] - path[i][0][0], path[i_next][0][1] - path[i][0][1]
            dys.append(dy)
            dxs.append(dx)
        dy_nd, dx_nd = np.array(dys), np.array(dxs)
        path_fwd_dirs = np.arctan2(dy_nd, dx_nd).tolist()
        path_displs = np.sqrt(dy_nd**2 + dx_nd**2).tolist()

        #Set smoothed directions
        kernel = self.gaussian_kernel_1d(options.dir_smoothing_size, options.dir_smoothing_sigma)
        smoothed_dy_nd, smoothed_dx_nd = convolve(dy_nd, kernel, mode='wrap'),  convolve(dx_nd, kernel, mode='wrap')
        path_fwd_dirs_smoothed = np.arctan2(smoothed_dy_nd, smoothed_dx_nd).tolist()

        #Set up nodes
        for i in range(len(path)):
            node = EdgeNode.EdgeNode(path[i][0][0], path[i][0][1],
                                     path_fwd_dirs[i], path_fwd_dirs_smoothed[i], path_displs[i])
            self.path.append(node)
            cur_section = maze_sections.get_section_from_coords(node.y, node.x)
            cur_section.add_node(node)

    def gaussian_kernel_1d(self, kernel_size, sigma):
        """
        Generates a 1D normalized Gaussian kernel using NumPy.

        Args:
            kernel_size: The size of the kernel (must be odd).
            sigma: The standard deviation of the Gaussian distribution.

        Returns:
            A 1D NumPy array representing the normalized Gaussian kernel.
        """

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        center = kernel_size // 2
        x = np.arange(kernel_size) - center

        kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))

        # Normalize the kernel
        kernel /= np.sum(kernel)

        return kernel
