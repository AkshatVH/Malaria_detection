import streamlit as st
import numpy as np
import cv2
from keras import utils
import tensorflow as tf
from tensorflow.keras import preprocessing
from PIL import Image
import keras



st.title('Malaria??')

def main():
    file_uploaded = st.file_uploader("Choose the cell image", type = ['png', 'jpg', 'jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)

        st.image(file_uploaded, caption='Uploaded cell Image', use_column_width=None)

        test_image = image.resize((64, 64))
        test_image = preprocessing.image.img_to_array(test_image, dtype='uint8')
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        class_names = ['Parasitized', 'Uninfected']

        result = predict_class(test_image)
        st.write(result)
        if result == class_names[0]:
            pred_img = get_infected_area(image)
            st.image(pred_img, caption='Malaria Detected', use_column_width=None)






def predict_class(image):
    model = keras.models.load_model('Malaria_neural_network.h5')

    class_names = ['Parasitized', 'Uninfected']
    predictions = model.predict(image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]

    return image_class

def get_infected_area(img):
    img = preprocessing.image.img_to_array(img, dtype='uint8')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coordinates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        coordinates.append((x, y, width, height))

    for points in coordinates:
        add_Bbox(img, points)
    return img

def add_Bbox(img, bbox):
    x, y, width, height = bbox
    cv2.rectangle(img, (x, y), (x + width, y+height), (0, 255, 0), 2)



if __name__ == "__main__":
    main()

