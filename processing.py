"""This program pre-processes self-captured images containing handwritten
digits so that the trained NN model can recognize efficiently. The idea is to
convert digits in the original image to digits that have the same format with
MNIST images that were used to train the NN model.

References:
http://www.hackevolve.com/recognize-handwritten-digits-1/
"""
import cv2
import imutils
import numpy as np
import matplotlib.image as mpimg


def processing(path):
    """Process image: resize, grey, remove noise, normalize
    """
    # load image
    i_orig = mpimg.imread(path)
    # resize but keep the aspect ratio
    i_resize = imutils.resize(i_orig, width=320)
    # convert to grays cale image
    i_gray = cv2.cvtColor(i_resize, cv2.COLOR_BGR2GRAY)
    # Blackhat to reveal dark regions on a light background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    i_black_hat = cv2.morphologyEx(i_gray, cv2.MORPH_BLACKHAT, kernel)
    # threshold the image to further reduce noise
    _, i_thresh = cv2.threshold(i_black_hat, 0, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # dilate the image to grow foreground pixels
    i_dilate = cv2.dilate(i_thresh, None)
    # check processing results
    # plt.imshow(i_dilate)  # i_resize, i_gray, i_black_hat, i_thresh, i_dilate
    # plt.show()
    return i_resize, i_dilate


def segmentation(i_resize, i_dilate, flat=False):
    """Segment the processed image and extract the digits in the image
    """
    _, contours, _ = cv2.findContours(i_dilate.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    img = i_resize.copy()
    digits = []
    boxes = []
    i = 0
    for cnt in contours:
        # Check the area of contour, if it is very small ignore it
        if cv2.contourArea(cnt) < 10:
            continue
        # filtered contours are detected
        (x, y, w, h) = cv2.boundingRect(cnt)
        # take ROI (i.e. digits) of the contour
        roi = i_dilate[y:y+h, x:x+w]
        # resize to a size of MNIST data without border
        roi = cv2.resize(roi, (20, 20))
        # add border with black (background) pixels
        black = [0, 0, 0]
        roi = cv2.copyMakeBorder(roi, 4, 4, 4, 4,
                                 cv2.BORDER_CONSTANT, value=black)
        # resize to a size compatible with MNIST data
        roi = cv2.resize(roi, (28, 28))
        # save (before normalization) the digits as image files for checking
        cv2.imwrite('roi' + str(i) + '.png', roi)
        # normalize the image since the MNIST data were normalized
        roi = roi / 255.0
        if flat:
            # reshape to a flat vector for fully connected network model
            roi = np.reshape(roi, (784,))
        else:
            # reshape to a array for CNN model
            roi = np.reshape(roi, (28, 28, 1))
        # mark the digits on the image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        digits.append(roi)  # or roi_array
        boxes.append((x, y, w, h))
        i += 1
    # check if segmentation correct
    # cv2.imshow('Image', img)
    # cv2.waitKey(0)
    digits = np.array(digits)
    return digits, boxes
