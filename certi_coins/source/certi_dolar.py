# Desenvolvido Lucas Cavalcante, lidcc87@gmail.com 
# Desafio Visão Computacional - CERTI

import numpy as np
import cv2
import os

im = cv2.imread('../resources/dolar_original.png') # read image
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # convert to gray
blur = cv2.blur(gray, (7,7)) # blur image with size (7,7)

_ , binary = cv2.threshold(blur, 63, 255, cv2.THRESH_BINARY_INV) # bitwise binary conversion

def create_kernel(num=20):
    """
    Input: number of matrix square dimension 
    Output: matrix square with ones - array numpy
    """
    return np.ones((num,num),np.uint8)

kernel = create_kernel(num=20)

morphological_im = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

def param_builder():
# Create the params
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = False
    params.filterByColor = False
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByCircularity = True
    return params

# Detect blobs
detector = cv2.SimpleBlobDetector_create(param_builder())
keypoints = detector.detect(morphological_im)

# Draw detected blobs as red circles
im_with_keypoints = cv2.drawKeypoints(morphological_im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

centers_coordinates = tuple(np.array([keypoints[idx].pt for idx in range(0, len(keypoints))]).astype("int"))
radius = (np.array([keypoints[idx].size for idx in range(0, len(keypoints))])/2).astype("int")

image = None

def circles_mounts():
    # Draw circles
    global image
    global im
    for (center_coordinates, r) in zip(centers_coordinates, radius):
        image = cv2.circle(im, tuple(center_coordinates), r, (0, 255, 0), 2) # draw circumference
        image = cv2.circle(im, tuple(center_coordinates), 1, (255, 0, 0), 5) # draw circumference center 
        cv2.imshow("circle", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

# Quantity of coins
quantity_of_coins = len(keypoints)
print('Quantity of Coins (USD):', quantity_of_coins)

# Save the image 'output'
cv2.imwrite(os.path.join('image_result/dolar_result.png'), circles_mounts())