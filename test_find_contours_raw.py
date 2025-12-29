import cv2
import numpy as np
import matplotlib.pyplot as plt


def split_contour_on_inflection_apex(contour):
    if len(contour) <= 2:
        return [contour]  # Need at least 2 points for differences

    #Check if contour is closed
    closed = np.linalg.norm(contour[0,:,:] - contour[-1,:,:]) < 20

    #Determine normalized gradients
    next_points = np.roll(contour, -1, axis=0)  # Shift points to get next points
    gradients = next_points - contour
    mags = np.linalg.norm(gradients, axis=2, keepdims=True)
    grad_norm = np.where(mags == 0, gradients, gradients / mags)

    #Find instances of turnaround
    roll_fwd, roll_back = np.roll(grad_norm, 1, axis=0), np.roll(grad_norm, -1, axis=0)
    apex_negation = np.linalg.norm(roll_back + grad_norm, axis=2)
    bridge_negation = np.linalg.norm(roll_fwd + roll_back, axis=2)
    turnarounds = np.where(np.logical_or(apex_negation < 0.5, bridge_negation < 0.5))[0]
    turnarounds.sort()
    apex_indices = turnarounds

    #
    #
    # #Convolve with 1 wide kernel to find bridge loop points
    # kernel_11 = np.ones(11)
    # if closed:
    #     #Pad for wrapping around
    #     padded_grads = np.vstack((gradients[-5:,:], gradients, gradients[:5,:]))
    #     summed_bridge_x = np.convolve(padded_grads[:,0,0], kernel_11, mode='valid')
    #     summed_bridge_y = np.convolve(padded_grads[:,0,1], kernel_11, mode='valid')
    # else:
    #     summed_bridge_x = np.convolve(gradients[:,0,0], kernel_11, mode='same')
    #     summed_bridge_y = np.convolve(gradients[:,0,1], kernel_11, mode='same')
    #
    # #Add 12th point for apex loop points
    # grads_12th = np.roll(gradients, -6, axis=0)
    # summed_apex_x = summed_bridge_x + grads_12th[:,0,0]
    # summed_apex_y = summed_bridge_y + grads_12th[:,0,1]
    #
    # #Find points where path converges on itself
    # bridge_points = np.logical_and(summed_bridge_x == 0, summed_bridge_y == 0)
    # apex_points = np.logical_and(summed_apex_x == 0, summed_apex_y == 0)
    # apex_indices = np.where(np.logical_or(bridge_points, apex_points))[0]
    # if apex_indices.size == 0:
    #     return [contour]

    # mags = np.linalg.norm(gradients, axis=2, keepdims=True)
    # grad_norm = np.where(mags == 0, gradients, gradients / mags)
    #
    # #Determine laplacians and inflection points (sign changes)
    # laplacians = np.roll(grad_norm, -1, axis=0) - grad_norm
    # inflection_points = np.sum(np.roll(laplacians, -1, axis=0)*laplacians, axis=2)[:,0] < 0
    # sharp_turns = np.linalg.norm(laplacians, axis=2)[:,0] >= 2
    #
    # #Find likely apexes and split on them
    # apex_indices = np.where(np.logical_and(inflection_points, sharp_turns))[0]
    final_contours = []
    for i in range(len(apex_indices)):
        if closed:
            i_nx = (i + 1) % len(apex_indices)
            if apex_indices[i_nx] > apex_indices[i]:
                final_contours.append(contour[apex_indices[i]:apex_indices[i_nx] + 1])
            else:
                #Loop around
                split_contour = np.vstack((contour[apex_indices[i]:], contour[:apex_indices[i_nx] + 1]))
                final_contours.append(split_contour)
        else:
            if i >= len(apex_indices) - 1 or apex_indices[i] < 1 or apex_indices[i] >= len(contour) - 1:
                continue #Ignore last point since not wrapping around
            i_nx = i + 1
            final_contours.append(contour[apex_indices[i]:apex_indices[i_nx] + 1])

    return final_contours

def find_edges_canny_hull(image_path):
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
    print(f"Found {len(contours)} initial contours.")
    # for c in contours:
    #     contour = c[:,0,:]
    #     hull = cv2.convexHull(contour)
    #     area = cv2.contourArea(hull)
    #     perimeter = cv2.arcLength(hull, True)
    #     if area/perimeter > 10:
    #         final_contours.append(c)

    for c in contours:
        final_contours.extend(split_contour_on_inflection_apex(c))


    return final_contours

# Example Usage:
image_path = "C:\\Users\\liamc\\PycharmProjects\\continuous-outline\\Trial-AI-Base-Images\\bg_removed\\TEMPP\\bgrem_British_Columbia_Parliament_Buildings_-_Pano_-_HDR.png"
contours = find_edges_canny_hull(image_path)

if contours is not None:
    print(f"Found {len(contours)} contours.")

    # You can then process the contours, e.g., draw them on the image:
    img = cv2.imread(image_path) #reloads the image
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)  # Draw contours in green

    plt.imshow(img)
    plt.show()
else:
    print("No contours found or an error occurred.")