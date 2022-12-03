# AIM: To create LeNet class.


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Dense
from tensorflow.keras.layers import Flatten, MaxPooling2D, Rescaling
from tensorflow.keras import backend as K

class LeNet:
    '''
    Class to construct LeNet model.
    
    LeNet Architecture:
        INPUT => CONV => RELU => POOL => CONV => RELU => POOL => 
            FC => RELU => FC => SOFTMAX
    '''
    @staticmethod
    def build(width, height, depth, classes):
        '''
        Static method to build the LeNet model architecture.
    
        Parameters:
            width (int): Width of the input image.
            height (int): Height of the input image.
            depth (int): Depth of the input image.
            classes (int): Number of output classes to learn to predict.
           
        Returns:
            model: Constructed lenet network architecture.
        '''
        model = Sequential()
        inputShape = (height, width, depth) # initialize the model along with the input shape to be "channels last"
        if K.image_data_format() == "channels_first": # if we are using "channels first", update the input shape
            inputShape = (depth, height, width)

        # normalizing the images
        model.add(Rescaling(1./255, input_shape=inputShape))

  	    # define the model layers
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # model name		
        model._name = 'LeNet'

        return model
