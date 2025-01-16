import numpy as np
import cv2
import copy as cp

def detect_image_contours(image_path):
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if np.mean(img) > 170:
        img = cv2.bitwise_not(img)

    # Thresholding to get binary image
    _, binary = cv2.threshold(img, np.max(img) // 2, np.max(img), cv2.THRESH_BINARY)

    # Detect contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on a blank image
    # traced_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    iterImg = binary
    contoursall = cp.deepcopy(contours)
    iters = 0
    # while np.mean(iterImg) > 0.01 * np.max(iterImg) and iters < 1000:
    #     iters += 1
    #     mask = np.zeros_like(img)
    #     cv2.drawContours(mask, contoursall, -1, (255), thickness=2000)
    #     # cv2.imshow('Traced Image', mask)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #
    #     img_without_contours = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    #     iterImg = cp.deepcopy(img_without_contours)
    #     # Detect contours
    #     _, binary2 = cv2.threshold(img_without_contours, 1, np.max(img_without_contours), cv2.THRESH_BINARY)
    #     contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contoursall = contoursall + contours2

    return contoursall