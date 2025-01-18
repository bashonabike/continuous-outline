import helpers.edge_detection as edge
import os
import cv2
import helpers.StrayPixelRemover as spr
from PIL import Image, ImageDraw

# edge.k_means_clustering('helpers.png')

remover = spr.StrayPixelRemover(1, 10)

for file in os.listdir("Trial-AI-Base-Images"):
    if file.endswith(".jpg"):
      image_path = os.path.join("Trial-AI-Base-Images", file)
      postedge, filtered_edges, pixels_removed = edge.detect_edges(image_path)
      # Save the edge map (optional)
      output_path = os.path.join("Trial-AI-Base-Images", f"edges_{file}")
      cv2.imwrite(output_path, postedge)


      output_path = os.path.join("Trial-AI-Base-Images", f"edges_pxs_rmvd_{file}")
      cv2.imwrite(output_path, pixels_removed)


      output_path = os.path.join("Trial-AI-Base-Images", f"filtered_edges_{file}")
      cv2.imwrite(output_path, filtered_edges)

      #
      # output_path = os.path.join("Trial-AI-Base-Images", f"thinned_{file}")
      # cv2.imwrite(output_path, thinned)
      # edge.vectorize_edgified_image(output_path)




        # testtt = str(os.path.join(root, f"edges_{file}_.svg"))
        # edge.split_svg_file("test.svg", str(os.path.join(root, f"edges_{file}_.svg")))