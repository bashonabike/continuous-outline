import cv2
import numpy as np
from collections import deque


# TODO: rewrite so node based should be o(n) examine each link only 1x
# each pixel is a node
# cursor thru each examing all neighbours from 5pi/4 to 0 in pi/4 increments
# if that pixel is part of glob object, add self to that glob object
# else new glob object with both pixels
# if wind up bridging globs, merge globs
# keep track of size of each glob
# at end, sort globs by size, delete globs within bounds for removal

class StrayPixelRemover:

    def __init__(self, min_threshold, max_threshold):
        """
        Initialize the StrayPixelCleaner with threshold values.
        :param min_threshold: Minimum number of connected pixels considered stray.
        :param max_threshold: Maximum number of connected pixels to not be removed.
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def process(self, image):
        """
        Remove stray pixels from the given image.
        :param image: NumPy ndarray to process (from cv2.imread).
        :return: Processed NumPy ndarray.
        """
        height, width = image.shape
        background = image[0, 0]  # Assume the top-left pixel as background
        processed_pixels = set()

        for y in range(height):
            for x in range(width):
                current_pixel = (x, y)
                if image[y, x] == background or current_pixel in processed_pixels:
                    continue

                connected_pixels = self.find_connected_pixels(image, background, x, y)
                processed_pixels.update(connected_pixels)

                if self.min_threshold <= len(connected_pixels) < self.max_threshold:
                    print(f"Removing {len(connected_pixels)} stray pixels")
                    for px, py in connected_pixels:
                        image[py, px] = background

        return image

    def find_connected_pixels(self, image, background, x, y):
        """
        Find all connected non-background pixels starting from (x, y).
        :param image: NumPy ndarray.
        :param background: Background pixel value (tuple for RGB).
        :param x: Starting x-coordinate.
        :param y: Starting y-coordinate.
        :return: Set of connected pixels.
        """
        height, width = image.shape
        to_process = deque([(x, y)])
        connected_pixels = set()

        while to_process:
            cx, cy = to_process.popleft()
            if (cx, cy) in connected_pixels:
                continue

            connected_pixels.add((cx, cy))

            for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in connected_pixels:
                    if image[ny, nx] != background:
                        to_process.append((nx, ny))

        return connected_pixels

    @staticmethod
    def crop_image(image, background):
        """
        Crop the image to remove the background area.
        :param image: NumPy ndarray.
        :param background: Background pixel value (tuple for RGB).
        :return: Cropped NumPy ndarray.
        """
        mask = np.any(image != np.array(background), axis=-1)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return image  # No non-background pixels, return the original image
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return image[y_min:y_max + 1, x_min:x_max + 1]
