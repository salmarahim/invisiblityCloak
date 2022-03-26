# This technique is opposite to Green screening.
# In green screening, we remove the background but here we will remove the foreground frame.

# So our algorithm or the steps will be as follows:-
# 1. Capture and store the background frame. [ This will bedone for some seconds ]
# 2. Detect the red colored cloth using color detection and segmentation algorithm.
# 3. Segment out the red colored cloth by generating a mask. [ used in code ]
# 4. Generate the final augmented output to create a magical effect. [video.mp4 ]

import cv2
import time
import numpy as np

# To save the output in a file output.avi, for that we will use the VideoWriter_fourcc
# FourCC is a 4-byte code used to specify the video codec. The list of available codes 
# can be found in fourcc.org. It is platform dependent.FourCC code is passed as 
# `cv.VideoWriter_fourcc('M','J','P','G') or cv.VideoWriter_fourcc(*'MJPG')` for MJPEG.

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Now we'll start reading the video from the webcam of our system. 
# We'll use VideoCapture() function to capture the video.
cap = cv2.VideoCapture(0)


# We'll allow the camera to warm up.
# Allowing the webcam to start by making the code sleep for 2 seconds
time.sleep(2)
bg = 0


# We need to have a video that has some seconds dedicated to the background
# frame so that it could easily save the background image.
# So we'll be capturing the background in the range of 60.
# Capturing background for 60 frames

for i in range(60):
    ret, bg = cap.read()
#Flipping the background
bg = np.flip(bg, axis=1)
# Here we are flipping the background because the camera captures the image inverted.

# Now that we have our background ready, we need to read every frame from the camera.
# Reading the captured frame until the camera is open
while (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    #Flipping the image for consistency
    img = np.flip(img, axis=1)

    # Here we are using cap.isOpened to check if the camera is opened or not.
    # ret returns a boolean value of true or false.And using np.flip to flip the image.

    # As we capture frames we are also capturing the colors in those frames.
    # And we need to convert the images from BGR (Blue Green Red) to HSV(Hue, Saturation, Value).
    # We need to do this so that we can detect the red color more efficiently.

    #Converting the color from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # https://medium.com/programming-fever/invisibility-cloak-using-opencv-cf2d7d8894
    # https://www.peachpit.com/content/images/art_krause2_colortips/elementLinks/krause1_fig01.jpg

    # Now that we have converted the color from BGR To HSV it will be easy for us to identify 
    # the colors.We have to create masks which will check for the colors in the specified
    # range and then mask it with the background image.
    # We'll be creating 2 different masks which will help us detect the colors in
    # that given range.

    # We are creating masks to detect the red color. You can change the color
    # value depending on the color you want.

    # Generating mask to detect red colour
    # These values can also be changed as per the color
    # lower mask (0-10)
    lower_red = np.array([0, 120, 50])
    upper_red = np.array([10, 255,255])
    mask_1 = cv2.inRange(hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask_2 = cv2.inRange(hsv, lower_red, upper_red)

    mask_1 = mask_1 + mask_2

# Now we need to add effects on the colors that we have detected. What kind of effect do you
# think we'll be adding? We'll be adding the diluting effect to the image in the video.
# For that we'll be using the morphologyEx() method.

# Syntax of this method: morphologyEx(src, dst, op, kernel)
# This method accepts the following parameters:
# ● src − An object representing the source (input) image.
# ● dst − object representing the destination (output) image.
# ● op − An integer representing the type of the Morphological operation.
# ● kernel − A kernel matrix. 

# morphologyEx() is the method of the class Img Processing which is used to perform operations
# on a given image.

    #Open and expand the image where there is mask 1 (color)
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # MORPH_OPEN and MORPH_DILATE are the two types of effects.
    # Now we need to create a mask to segment out the red color from the frame.
    # To do so we'll be using the bitwise_not() method.

    # Selecting only the part that does not have mask one and saving in mask 2
    mask_2 = cv2.bitwise_not(mask_1)

    # Now, we need to create 2 resolutions. First one would be an image without 
    # color red (or any other color that you choose) and the second one would be
    # the background from the background image we captured earlier just for the 
    # parts where red color was (mask 1).

    # Keeping only the part of the images without the red color 
    # (or any other color you may choose)
    res_1 = cv2.bitwise_and(img, img, mask=mask_2)

    # Keeping only the part of the images with the red color
    # (or any other color you may choose)
    res_2 = cv2.bitwise_and(bg, bg, mask=mask_1)

    # All that is left is to generate the final video with the invisible effect and 
    # also, to display the result to the user.

    # Generating the final output by merging res_1 and res_2
    final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)
    output_file.write(final_output)
    # Displaying the output to the user
    cv2.imshow("magic", final_output)
    cv2.waitKey(1)


cap.release()
output_file.release()
cv2.destroyAllWindows()