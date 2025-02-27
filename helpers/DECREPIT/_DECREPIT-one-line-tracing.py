
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from svgpathtools import svg2paths, wsvg
from svgwrite import Drawing
import cairosvg
from scipy import interpolate
import matplotlib.pyplot as plt
import copy as cp
from scipy.ndimage import gaussian_filter1d
from PIL import Image


def path_length(path):
    """Calculate the length of a given path."""
    length = 0
    for segment in path:
        length += segment.length()
    return length


def filter_svg_by_length(input_svg, output_svg, min_length, pctlengthcutoff=0.0):
    # Load paths and attributes from the SVG
    paths, attributes = svg2paths(input_svg)
    lensum, lennum, lenavg = 0.0,0.0,0.0

    # Filter out paths shorter than the min_length
    filtered_paths = []
    filtered_attributes = []
    for path, attr in zip(paths, attributes):
        curlen = path_length(path)
        lensum += curlen
        lennum += 1
        lenavg = lensum/lennum
        if curlen >= min_length and curlen > pctlengthcutoff*2*lenavg:
            filtered_paths.append(path)
            filtered_attributes.append(attr)

    # Write the filtered paths to a new SVG
    wsvg(filtered_paths, attributes=filtered_attributes, filename=output_svg)
def extract_paths_from_svg(svg_file):
    # Load paths from the SVG file
    paths, attributes = svg2paths(svg_file)

    # Print path information
    for i, path in enumerate(paths):
        print(f"Path {i}: {path}")

    return paths


def svg_to_png(svg_file, output_png):
    cairosvg.svg2png(url=svg_file, write_to=output_png, background_color="white")




def trace_image_contours(image_path):
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if np.mean(img) > 170:
        img = cv2.bitwise_not(img)


    # Thresholding to get binary image
    _, binary = cv2.threshold(img, np.max(img)//2, np.max(img), cv2.THRESH_BINARY)

    # Detect contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on a blank image
    # traced_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    iterImg = binary
    contoursall = cp.deepcopy(contours)
    iters = 0
    while np.mean(iterImg) > 0.01*np.max(iterImg) and iters < 1000:
        iters += 1
        mask = np.zeros_like(img)
        cv2.drawContours(mask, contoursall, -1, (255), thickness=20)
        cv2.imshow('Traced Image', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        img_without_contours = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
        iterImg = cp.deepcopy(img_without_contours)
        # Detect contours
        _, binary2 = cv2.threshold(img_without_contours, 1, np.max(img_without_contours), cv2.THRESH_BINARY)
        contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoursall = contoursall + contours2

    # contoursall = contours2



    # Display traced image
    # cv2.imshow('Traced Image', traced_image)
    # cv2.imwrite('Traced Image.png', traced_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return contoursall



def simplify_contours(contours, epsilon=0.01):
    simplified_contours = []
    for contour in contours:
        epsilon = epsilon * cv2.arcLength(contour, True)  # Epsilon is proportional to contour length
        approx = cv2.approxPolyDP(contour, epsilon, True)
        simplified_contours.append(approx)
    return simplified_contours



def save_svg_from_contours(output_file, contours):
    dwg = Drawing(output_file)

    for contour in contours:
        points = contour[:, 0, :]  # Extract points
        path_data = 'M ' + ' L '.join([f'{x},{y}' for x, y in points]) + ' Z'
        dwg.add(dwg.path(d=path_data, fill='none', stroke='black', stroke_width=1))

    dwg.save()

def conjoin_contours(contours):
    # Concatenate all contours along axis 0 (flattening)
    flattened_contours = np.vstack(contours)
    return flattened_contours

# Function to compute the Euclidean distance between two points
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# Function to reorder and flatten contours based on minimizing distance
def flatten_contours_minimize_distance(contours):
    # Choose the first contour as the starting point
    current_contour = contours[0]
    combined_contour = [current_contour.squeeze()]  # Remove extra dimension

    # Remove the first contour from the list
    remaining_contours = list(contours[1:])
    output_sets = []

    while remaining_contours:
        # Get the end point of the current contour
        current_end = current_contour[-1].squeeze()

        # Find the contour that starts closest to the current contour's end
        min_dist = float('inf')
        next_contour_idx = None
        reverse = False

        for i, contour in enumerate(remaining_contours):
            contour_start = contour[0].squeeze()
            contour_end = contour[-1].squeeze()

            # Calculate distances from current_end to both start and end of the next contour
            dist_start = distance(current_end, contour_start)
            dist_end = distance(current_end, contour_end)

            # Pick the minimum distance, and track if we need to reverse the contour
            if dist_start < min_dist:
                min_dist = dist_start
                next_contour_idx = i
                reverse = False
            if dist_end < min_dist:
                min_dist = dist_end
                next_contour_idx = i
                reverse = True

        # Retrieve the next contour and reverse it if necessary
        next_contour = remaining_contours.pop(next_contour_idx)
        if reverse:
            next_contour = np.flip(next_contour, axis=0)
        if min_dist > 0:
            output_sets.append(np.vstack(combined_contour))
            combined_contour = []
        # Add the next contour to the combined contour
        combined_contour.append(next_contour.squeeze())

        # Update current_contour for the next iteration
        current_contour = next_contour

    output_sets.append(np.vstack(combined_contour))

    return output_sets


# Step 2: Draw a spline through all the points
def draw_spline_through_points(points, smoothness=3):
    # Extract x and y coordinates from points
    x = points[:, 0]
    y = points[:, 1]

    # Create parameter t for spline, based on arc length
    t = np.arange(len(x))

    # Perform spline interpolation using B-spline
    tck, u = interpolate.splprep([x, y], s=smoothness)
    u_fine = np.linspace(0, 1, 1*len(x))  # Interpolation points
    x_fine, y_fine = interpolate.splev(u_fine, tck)

    # Apply Gaussian filter for smoothing
    smoothed_x = gaussian_filter1d(x_fine, sigma=3)
    smoothed_y = gaussian_filter1d(y_fine, sigma=3)

    return smoothed_x, smoothed_y


#
# # Example usage
# svg_file = 'jacktrace1.svg'
# paths = extract_paths_from_svg(svg_file)
#
# svg_to_png('jacktrace4.svg', 'tempout.png')
# charcoal_effect('tempout.png', 'charcoal.png')

# Example usage
filter_svg_by_length('handswater.svg', 'cleaned.svg', 5, pctlengthcutoff=0.0)

svg_to_png('cleaned.svg', 'output_file.png')

# Example usage
contours = trace_image_contours('output_file.png')
# contours = trace_image_contours('output_file.jpg')


# contours = trace_image_contours('jack for tracing.png')


flattened_contours_sets = flatten_contours_minimize_distance(contours)
# Open the image file
img = Image.open('output_file.png')

# Get the dimensions
width, height = img.size
width, height = 12*width/max(width, height), 12*height/max(width, height)

plt.figure(figsize=(width, height))

for flattened_contours in flattened_contours_sets:
    if len(flattened_contours) > 0:
        # Calculate x_max and y_max from the array
        x_max = np.max(flattened_contours[:, 0])
        y_max = np.max(flattened_contours[:, 1])
        try:
            x_spline, y_spline = draw_spline_through_points(flattened_contours, 20)

            # Plot original points
            # plt.scatter(flattened_contours[:, 0], flattened_contours[:, 1], color='red', label='Original Points')

            # Plot spline curve
            plt.plot(x_spline, y_spline, color='black', linewidth=1.3)
        except:
            print("Failed to plot sub-spline, likely too few points")

plt.legend()
plt.gca().invert_yaxis()
plt.grid(False)
plt.axis('off')
plt.savefig('spline_curve.jpg', format='jpeg', bbox_inches='tight', pad_inches=0, dpi=600)
plt.show()

#
#
# # Example usage after tracing contours
# simplified = simplify_contours(contours)
#
#
# # Save to SVG
# save_svg_from_contours('traced_output.svg', simplified)
#
