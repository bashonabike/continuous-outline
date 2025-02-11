import helpers.edge_detection as edge
import os
import cv2
import helpers.StrayPixelRemover as spr
import helpers.SLIC_Segmentation as slic
from skimage.segmentation import mark_boundaries
from PIL import Image, ImageDraw
from skimage.util import img_as_float
from skimage import io
import numpy as np
import helpers.draw_outline as draw


def draw_open_paths(image, paths, color=(0, 0, 255), thickness=2):
  """
  Draws a series of open paths on an image.

  Args:
      image: The image (NumPy array) to draw on.
      paths: A list of NumPy arrays, where each array represents a path
             and contains the (x, y) coordinates of the nodes.
      color: The color of the paths (BGR tuple, default: red).
      thickness: The thickness of the path lines.

  Returns:
      The image with the paths drawn.  (A copy is created if the input image is modified)
  """

  image_with_paths = image.copy()  # Create a copy to avoid modifying the original

  for path in paths:
    for i in range(len(path) - 1):  # Iterate through nodes, drawing lines between them
      start_point = tuple(path[i][0].astype(int))  # Cast to int for pixel indexing
      end_point = tuple(path[i + 1][0].astype(int))
      cv2.line(image_with_paths, start_point, end_point, color, thickness)

  return image_with_paths

# edge.k_means_clustering('helpers.png')

remover = spr.StrayPixelRemover(1, 10)



for file in os.listdir("Trial-AI-Base-Images"):
    if file.endswith(".jpg"):
      image_path = os.path.join("Trial-AI-Base-Images", file)
      postedge, filtered_edges, pixels_removed, split_contours = edge.detect_edges(image_path)

      #Segment
      im_float = img_as_float(io.imread(image_path))
      near_boudaries_contours, segments = slic.slic_image_test_boundaries(im_float, split_contours)
      image_with_contours = (mark_boundaries(cv2.imread(image_path), segments) * 255).astype(np.uint8)  # Create a copy to avoid modifying the original
      # cv2.drawContours(image_with_contours, tuple(split_contours), -1, (255,0,0), 1)  # -1 draws all contours
      # cv2.drawContours(image_with_contours, tuple(near_boudaries_contours), -1, (0, 255, 255), 3)  # -1 draws all contours
      image_splits = draw_open_paths(image_with_contours, split_contours, color=(255, 0, 0), thickness=1) # -1 draws all contours
      image_splits_2 = draw_open_paths(image_splits, near_boudaries_contours, color=(0, 255, 0), thickness=3)


      output_path = os.path.join("Trial-AI-Base-Images", f"border_edges_{file}")
      cv2.imwrite(output_path, image_splits_2)


      # Save the edge map (optional)
      output_path = os.path.join("Trial-AI-Base-Images", f"edges_{file}")
      cv2.imwrite(output_path, postedge)


      output_path = os.path.join("Trial-AI-Base-Images", f"edges_pxs_rmvd_{file}")
      cv2.imwrite(output_path, pixels_removed)


      output_path = os.path.join("Trial-AI-Base-Images", f"filtered_edges_{file}")
      cv2.imwrite(output_path, filtered_edges)

      connected_path, obliterated_mask = draw.connect_paths(image_splits_2, near_boudaries_contours)

      image_with_path = draw.draw_path(image_splits_2, connected_path, color=(0, 0, 255), thickness=3)

      image_with_obliterated = draw.draw_obliterated(image_with_path, obliterated_mask, color=(0, 255, 0))


      output_path = os.path.join("Trial-AI-Base-Images", f"drawn_{file}")
      cv2.imwrite(output_path, image_with_path)


      output_path = os.path.join("Trial-AI-Base-Images", f"oblit_{file}")
      cv2.imwrite(output_path, image_with_obliterated)





      #
      # output_path = os.path.join("Trial-AI-Base-Images", f"thinned_{file}")
      # cv2.imwrite(output_path, thinned)
      # edge.vectorize_edgified_image(output_path)




        # testtt = str(os.path.join(root, f"edges_{file}_.svg"))
        # edge.split_svg_file("test.svg", str(os.path.join(root, f"edges_{file}_.svg")))