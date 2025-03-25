import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_edges_canny_hull(image_path):
    """
    Opens an image file using OpenCV, finds contours, and returns them.

    Args:
        image_path: The path to the image file.

    Returns:
        A list of contours found in the image, or None if an error occurs.
    """
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[img[:,:,3] == 0] = 0

    gray = 5*(gray//5)

    # # Apply thresholding (you might need to adjust the threshold value)
    # _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 100, 200)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_contours = []

    contours = [c for c in contours if len(c) > 100]
    for c in contours:
        contour = c[:,0,:]
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)
        if area/perimeter > 10:
            final_contours.append(c)



    return final_contours

# Example Usage:
image_path = "C:\\Users\\liamc\\PycharmProjects\\continuous-outline\\Trial-AI-Base-Images\\bg_removed\\TEMPP\\bgrem_British_Columbia_Parliament_Buildings_-_Pano_-_HDR.png"
contours = find_image_contours(image_path)

if contours is not None:
    print(f"Found {len(contours)} contours.")

    # You can then process the contours, e.g., draw them on the image:
    img = cv2.imread(image_path) #reloads the image
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)  # Draw contours in green

    plt.imshow(img)
    plt.show()
else:
    print("No contours found or an error occurred.")