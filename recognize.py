"""This program recognize handwritten digits using trained NN model.
"""
import cv2
import numpy as np
from keras.models import load_model
from processing import processing, segmentation


# load data to predict #
path = 'IMG_1702.jpg'  # IMG_1702.jpg, IMG_1703.jpg
i_resize, i_dilate = processing(path)
# load model #
vanilla_model = load_model('vanilla_model_classic.h5')
CNN_model = load_model('cnn_model_classic.h5')


def show_prediction(image, prediction, box, model_name):
    """Show the prediction digits above original digits on the image
    """
    image = image.copy()
    # cv2.imshow('Original', image)
    for (x, y, w, h), p in sorted(zip(box, prediction)):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(image, str(p), (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
    cv2.imshow('Using ' + model_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# recognize #
# using vanilla model
digits, boxes = segmentation(i_resize, i_dilate, True)
result_v = vanilla_model.predict(digits)
result_v = [np.argmax(r) for r in result_v]
show_prediction(i_resize, result_v, boxes, 'vanilla model')
# using cnn model
digits, boxes = segmentation(i_resize, i_dilate)
result_c = CNN_model.predict(digits)
result_c = [np.argmax(r) for r in result_c]
show_prediction(i_resize, result_c, boxes, 'CNN model')
