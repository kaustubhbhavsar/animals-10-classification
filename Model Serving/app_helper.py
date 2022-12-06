# import the model and other libraries
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# class labels
class_labels = ['CANE', 'CAVALLO', 'ELEFANTE', 'FARFALLA', 'GALLINA', 'GATTO', 'MUCCA',
               'PECORA', 'RAGNO', 'SCOIATTOLO']
# model path
model_path = 'Models/mobilenetv3l_basic_10_0.113.h5'

def get_classes(file_path):
    '''
    Function to preprocess the given image and get its predicted class label.
    
    Parameters:
      file_path (string): Path to image to be predicted.
    
    Returns:
      class_label (string): Predicted class label.
    '''
    # create an instance of saved model
    model = load_model(model_path)

    # load image and preprocess it
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.array([x])
    x = preprocess_input(x)

    # inferencing
    prediction = model.predict(x)

    return class_labels[np.argmax(prediction)]
