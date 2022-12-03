# AIM: To create ShallowNet class.


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Dense
from tensorflow.keras.layers import Flatten, Rescaling
from tensorflow.keras import backend as K

class ShallowNet:
    '''
    Class to construct ShallowNet model.
    
    ShallowNet Architecture:
        INPUT => CONV => RELU => FC => SOFTMAX
    '''
    @staticmethod
    def build(width, height, depth, classes):
        '''
        Static method to build the ShallowNet model architecture.
    
        Parameters:
            width (int): Width of the input image.
            height (int): Height of the input image.
            depth (int): Depth of the input image.
            classes (int): Number of output classes to learn to predict.
           
        Returns:
            model: Constructed shallownet network architecture.
        '''
        model = Sequential()  # initialize the model along with the input shape to be "channels last"
        inputShape = (height, width, depth)
        if K.image_data_format() == "channels_first": # if we are using "channels first", update the input shape
            inputShape = (depth, height, width)

        # normalizing the images
        model.add(Rescaling(1./255, input_shape=inputShape))

	# define the model layers
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # model name		
        model._name = 'ShallowNet'

        return model
