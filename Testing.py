import os
import cv2
# from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import numpy as np
import time
# import svgwrite as svg
import matplotlib.pyplot as plt
from datetime import date

from helpers.mazify.MazeAgent import MazeAgent
import helpers.edge_detect.SLIC_Segmentation as slic
# import helpers.edge_detect.edge_detection as edge
# import helpers.old_method.DrawingAgent as draw
import helpers.Enums as Enums
# import helpers.old_method.ParsePathsIntoObjects as parse
# from helpers.old_method import TourConstraints as constr
# import helpers.old_method.NodeSet as NodeSet
from helpers.mazify.MazeSections import MazeSections, MazeSection
import helpers.mazify.temp_options as options

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

def draw_object_node_path(image, object_path, color=(0, 0, 255), thickness=2):
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

  for i in range(len(object_path) - 1):  # Iterate through nodes, drawing lines between them
    start_point = (object_path[i].x, object_path[i].y)  # Cast to int for pixel indexing
    end_point = (object_path[i+1].x, object_path[i+1].y)
    cv2.line(image_with_paths, start_point, end_point, color, thickness)

  return image_with_paths

# edge.k_means_clustering('helpers.png')

# remover = spr.StrayPixelRemover(1, 10)



for file in os.listdir("Trial-AI-Base-Images"):
    if file.endswith(".jpg") or file.endswith(".png"):
      start_pre = time.time_ns()
      image_path = os.path.join("Trial-AI-Base-Images", file)
      # postedge, split_contours = edge.detect_edges(image_path)

      #TODO: maybe just run edge detect on details regions spec by user
      #use SLIC  for tracing, maybe build agent so it follows line lke maze then jumps to next as needed
      #try to do this intelligently? prioritize maximizing coverage (length)
      #Think where place node, maybe track deflection once gets past certain amt then nodify
      #maybe break into sub-ranges?  try to get coverage across many?
      #review notes from actual physical one line tracing!

      # #Segment
      # im_unch = cv2.imread(image_path, cv2.IMREAD_COLOR)
      # lab = cv2.cvtColor(im_unch, cv2.COLOR_BGR2LAB)
      # clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(16,16))  # Adjust clipLimit and tileGridSize
      # lab[:,:,0] = clahe.apply(lab[:,:,0])
      # clahe_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
      # cv2.imshow("hi contrat", clahe_image)
      # cv2.waitKey(0)
      # if clahe_image.shape[2] == 4:  # Check if it has an alpha channel
      #   im_float = img_as_float(cv2.cvtColor(clahe_image, cv2.COLOR_BGRA2BGR)) # Convert from BGRA to BGR
      # else:
      #   im_float = img_as_float(clahe_image)  # If it does not have an alpha channel, just use the image directly.


      im_unch = cv2.imread(image_path, cv2.IMREAD_COLOR)
      if im_unch.shape[2] == 4:  # Check if it has an alpha channel
        im_float = img_as_float(cv2.cvtColor(im_unch, cv2.COLOR_BGRA2BGR)) # Convert from BGRA to BGR
      else:
        im_float = img_as_float(im_unch)  # If it does not have an alpha channel, just use the image directly.




      # near_boudaries_contours, segments = slic.slic_image_test_boundaries(im_float, split_contours)
      # near_boudaries_contours, segments = slic.mask_test_boundaries(image_path, split_contours)

      outer_edges, outer_contours_yx, mask, bounds_outer = slic.mask_boundary_edges(image_path)
      inner_edges, inner_contours_yx, segments, num_segs, bounds_inner = slic.slic_image_boundary_edges(im_float,
                                                                                          num_segments=options.slic_regions,
                                                                                          enforce_connectivity=False,
                                                                                          contour_offset = len(outer_contours_yx))
      start = time.time_ns()
      crop = (tuple([min(o, c) for o, c in zip(bounds_outer[0], bounds_inner[0])]),
              tuple([max(o, c) for o, c in zip(bounds_outer[1], bounds_inner[1])]))

      outer_edges_cropped = outer_edges[crop[0][0]:crop[1][0] + 1, crop[0][1]:crop[1][1] + 1]
      inner_edges_cropped = inner_edges[crop[0][0]:crop[1][0] + 1, crop[0][1]:crop[1][1] + 1]
      outer_contours_yx_cropped = slic.shift_contours(outer_contours_yx, (-1)*crop[0][0], (-1)*crop[0][1])
      inner_contours_yx_cropped = slic.shift_contours(inner_contours_yx, (-1)*crop[0][0], (-1)*crop[0][1])

      edges = outer_edges + inner_edges

      # transition_nodes = slic.find_transition_nodes(segments)
      #
      edges_show = edges.astype(np.uint8) * 255
      end = time.time_ns()
      print(str((end - start)/1e6) + " ms to do crop stuff")

      end_pre = time.time_ns()
      print(str((end_pre - start_pre)/1e6) + " ms to do all pre-processing")
      # cv2.imshow("outer", outer_edges.astype(np.uint8) * 255)
      cv2.imshow("inner", inner_edges.astype(np.uint8) * 255)
      # cv2.imshow("outer", outer_edges.astype(np.uint8) * 255)
      # cv2.imshow("nodes", transition_nodes.astype(np.uint8) * 255)
      cv2.waitKey(0)

      y_lower, y_upper, x_lower, x_upper = 400, 600, 400, 700
      detail_req_mask = np.zeros_like(outer_edges_cropped, dtype=np.bool)
      # Create boolean masks for y and x bounds
      y_mask = (np.arange(detail_req_mask.shape[0]) >= y_lower) & (np.arange(detail_req_mask.shape[0]) <= y_upper)
      x_mask = (np.arange(detail_req_mask.shape[1]) >= x_lower) & (np.arange(detail_req_mask.shape[1]) <= x_upper)

      # Use meshgrid to create 2D masks
      y_mask_2d, x_mask_2d = np.meshgrid(x_mask, y_mask)

      # Apply the masks to set values to True
      detail_req_mask[y_mask_2d & x_mask_2d] = True

      maze_sections = MazeSections(outer_edges_cropped, options.maze_sections_across, options.maze_sections_across,
                                   [detail_req_mask])

      maze_agent = MazeAgent(outer_edges_cropped, outer_contours_yx_cropped, inner_edges_cropped,
                             inner_contours_yx_cropped, maze_sections)

      # raw_path_coords = maze_agent.run_round_dumb(image_path)
      raw_path_coords =  maze_agent.run_round_trace(Enums.TraceTechnique.snake)

      raw_path_coords_centered = slic.shift_contours([raw_path_coords], crop[0][0], crop[0][1])[0]

      # flipped_coords_nd = np.array(raw_path_coords_centered)
      # flipped_coords_nd[:, 0] = outer_edges.shape[0] - 1 - flipped_coords_nd[:, 0]
      # flipped_coords = flipped_coords_nd.tolist()  # Flip y-coordinates

      y_coords, x_coords = zip(*raw_path_coords_centered)  # Unzip the coordinates

      plt.plot(x_coords, y_coords, marker='o', markersize=1)  # Plot the line with markers
      plt.gca().invert_yaxis()
      plt.xlabel("X-coordinate")
      plt.ylabel("Y-coordinate")
      plt.title("Line Plot of Coordinates")
      plt.grid(True)
      plt.savefig(str(date.today()) + ".png" , dpi=600)
      plt.show()




      #TODO: Pass in vector segs, use compass pick dir, lock onto seg, criteria for losing seg, use dir just for between segs
      #TODO: pre-walk each edge forward, set smoothed direction for each node maybe do as kernal so backward and forward predictyion
      #Have it walk-dir invariant, but relative to forward walk, have smoothing as func of angle and displacement with next node
      #While walking, parse each node into a quadrant for easy retrieval, maybe make quadrant object, so can do refs to nodes

      testtt = 0


      # details_boundaries_contours, detail_segments = slic.slic_image_test_boundaries(im_float, split_contours,
      #                                                                  num_segments=constr.mindetailsegments,
      #                                                                  enforce_connectivity=False)
      # # image_with_contours = (mark_boundaries(cv2.imread(image_path, cv2.IMREAD_COLOR), segments, color=(255, 0, 0)) * 255).astype(np.uint8)
      # # cv2.imshow("image_with_contours", image_with_contours)
      # # cv2.waitKey(0)
      #
      # # image_with_contours = (mark_boundaries(image_with_contours, detail_segments, color=(255, 255, 0)) * 255).astype(
      # #   np.uint8)  # Create a copy to avoid modifying the original
      # image_with_contours = (mark_boundaries(cv2.imread(image_path, cv2.IMREAD_COLOR), detail_segments, color=(255, 255, 0)) * 255).astype(
      #   np.uint8)  # Create a copy to avoid modifying the original
      #
      # maze = np.zeros(image_with_contours.shape, dtype=np.uint8)
      # maze = mark_boundaries(maze, detail_segments, color=(255, 255, 255))[:,:, 0].astype(bool)
      #
      # cv2.imshow("image_with_contours", image_with_contours)
      # cv2.waitKey(0)
      # # cv2.drawContours(image_with_contours, tuple(split_contours), -1, (255,0,0), 1)  # -1 draws all contours
      # # cv2.drawContours(image_with_contours, tuple(near_boudaries_contours), -1, (0, 255, 255), 3)  # -1 draws all contours
      #
      #
      # # image_splits = draw_open_paths(image_with_contours, near_boudaries_contours, color=(255, 0, 0), thickness=1) # -1 draws all contours
      # # image_splits_2 = draw_open_paths(image_splits, details_boundaries_contours, color=(0, 255, 0), thickness=1)
      #
      # image_splits_2 = draw_open_paths(image_with_contours, details_boundaries_contours, color=(0, 255, 0), thickness=1)
      #
      # output_path = os.path.join("Trial-AI-Base-Images", f"border_edges_{file}")
      # # cv2.imwrite(output_path, image_splits_2)
      #
      #
      # # Save the edge map (optional)
      # output_path = os.path.join("Trial-AI-Base-Images", f"edges_{file}")
      # # cv2.imwrite(output_path, postedge)
      #
      #
      # # outer_nodes = parse.create_tour_nodes_from_paths(near_boudaries_contours, Enums.NodeSet.OUTER)
      # detail_nodes = parse.create_tour_nodes_from_paths(details_boundaries_contours, Enums.NodeSet.DETAIL)
      # # parsed_nodes = outer_nodes + detail_nodes
      # parsed_nodes =  detail_nodes
      #
      # n, m, _ = im_float.shape
      # dims = (n,m)
      #
      # node_set = NodeSet.NodeSet(parsed_nodes, dims)
      # agents = []
      # iters = 10
      # for agent in range(iters):
      #   print(f"\rAgent # {agent+1}/{iters} ({int((agent+1)/iters*100)}%)", end="", flush=True)
      #   start = time.perf_counter_ns() // 1000
      #   cur_agent = draw.DrawingAgent(dims, node_set)
      #   cur_agent.tour()
      #   agents.append(cur_agent)
      #   end = time.perf_counter_ns() // 1000
      #   # print("agenttot: "+ str(float(end-start)/1000.0))
      #   node_set.reset_oblit()
      # print()  # Add a newline at the end to move the cursor down
      #
      # agents.sort(key=lambda a: a.stats.final_score, reverse=True)
      #
      # dwg = svg.Drawing("TestAgent.svg", profile='full')
      # nodes_final = agents[0].tour_path
      # path_data = "M" + str(nodes_final[0].x) + "," + str(nodes_final[0].y)  # Move to the first point
      # for node in nodes_final[1:]:
      #   path_data += " L" + str(node.x) + "," + str(node.y)  # Line to each subsequent node
      #
      # path = dwg.path(d=path_data, stroke='black', fill="none", stroke_width=2)
      # dwg.add(path)
      # dwg.save()
      #
      # agent_num = 0
      # for agent in agents[:10]:
      #   agent_num += 1
      #   image_agent = draw_object_node_path(image_splits_2, agent.tour_path,
      #                                 color=(0, 0, 255), thickness=2)
      #   output_path = os.path.join("Trial-AI-Base-Images", f"agent_{str(agent_num)}_output_{file}")
      #   cv2.imwrite(output_path, image_agent)
      #   print(f"Oblit mask:{agent.stats.oblit_mask_size} - accum defl:{agent.stats.accum_defl_rad} - crowding:{agent.stats.crowding} - length conn:{agent.stats.length_of_connectors}")
      #

        # cv2.imshow(str(agent_num), image_agent)







      #
      # connected_path, obliterated_mask = draw.connect_paths(image_splits_2, near_boudaries_contours)
      #
      # image_with_path = draw.draw_path(image_splits_2, connected_path, color=(0, 0, 255), thickness=3)
      #
      # image_with_obliterated = draw.draw_obliterated(image_with_path, obliterated_mask, color=(0, 255, 0))
      #
      #
      # output_path = os.path.join("Trial-AI-Base-Images", f"drawn_{file}")
      # cv2.imwrite(output_path, image_with_path)
      #
      #
      # output_path = os.path.join("Trial-AI-Base-Images", f"oblit_{file}")
      # cv2.imwrite(output_path, image_with_obliterated)
      #




      #
      # output_path = os.path.join("Trial-AI-Base-Images", f"thinned_{file}")
      # cv2.imwrite(output_path, thinned)
      # edge.vectorize_edgified_image(output_path)




        # testtt = str(os.path.join(root, f"edges_{file}_.svg"))
        # edge.split_svg_file("test.svg", str(os.path.join(root, f"edges_{file}_.svg")))