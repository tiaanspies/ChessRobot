import cv2
import cv2.aruco as aruco
import numpy as np
import pi_debugging

# Define the size and number of bits of the marker
marker_size = 200
marker_id = 10

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

for i in range(4):
    # Generate the marker image
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_image = aruco.generateImageMarker(dictionary, i, marker_size, 20)

    # Display the generated marker
    cv2.imshow("Marker Image", marker_image)
    cv2.waitKey(0)

    # Save the marker image to a file
    cv2.imwrite("marker_4x4_"+str(i)+".png", marker_image)