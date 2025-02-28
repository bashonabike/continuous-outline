import numpy as np

class MazeSections:
    def __init__(self, outer_edge, m, n):
        self.outer_edge = outer_edge
        self.sections, self.y_grade, self.x_grade = self.count_true_pixels_in_sections(outer_edge, m, n)


    def count_true_pixels_in_sections(self, boolean_image, m, n):
        """
        Breaks a boolean image into m x n rectangular sections and counts the number of
        True pixels in each section.

        Args:
            boolean_image (numpy.ndarray): The boolean image (True/False or 1/0).
            m (int): The number of rows of sections.
            n (int): The number of columns of sections.

        Returns:
            numpy.ndarray: A 2D array where each element represents the count of True
                           pixels in the corresponding section.
        """

        height, width = boolean_image.shape
        section_height = height // m
        section_width = width // n

        # Handle cases where the image cannot be divided evenly
        remainder_height = height % m
        remainder_width = width % n

        sections = np.zeros((m, n), dtype=MazeSection)

        for i in range(m):
            for j in range(n):
                # Calculate section boundaries, handling remainders
                y_start = i * section_height
                y_end = (i + 1) * section_height + (1 if i == m - 1 and remainder_height > 0 else 0)
                x_start = j * section_width
                x_end = (j + 1) * section_width + (1 if j == n - 1 and remainder_width > 0 else 0)

                # Extract the section
                section = boolean_image[y_start:y_end, x_start:x_end]

                # Count True pixels
                count = np.count_nonzero(section)
                sections[i, j] = MazeSection((y_start, y_end, x_start, x_end), count)

        return sections, section_height, section_width

class MazeSection:
    def __init__(self, bounds, edge_pixels):
        (self.ymin, self.ymax, self.xmin, self.xmax) = bounds
        self.edge_pixels = edge_pixels
        self.filled_pixels = 0
        self.saturation = 0.0

    def update_saturation(self, fill_count):
        #TODO: improve this so it doesn't double-count saturation
        self.filled_pixels += fill_count
        self.saturation = float(self.filled_pixels) / self.edge_pixels