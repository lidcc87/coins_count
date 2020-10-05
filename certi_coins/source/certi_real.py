# Desenvolvido Lucas Cavalcante, lidcc87@gmail.com
# Desafio Visão Computacional - CERTI

import numpy as np
import cv2
import os

im = cv2.imread('../resources/real_original.jpg') # read image
output = im.copy() # copy im
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # convert to gray
blur = cv2.blur(gray, (10,10)) # blur image with size (10,10)

# detect circles in the image
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 2, 100)

# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.circle(output, (x, y), 1, (0, 0, 255), 5)
        
    # show the output image
    cv2.imshow("output", output)
    
    cv2.waitKey(0)
cv2.destroyAllWindows()

# Quantity of coins (brl)
quantity_of_coins = circles.shape[0]
print('Quantity of Coins (BRL):', quantity_of_coins)

# Save the image 'output'
cv2.imwrite(os.path.join('image_result/real_result.jpg'), output)