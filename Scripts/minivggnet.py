# AIM: To create MiniVGGNet class.


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense
from tensorflow.keras.layers import Dropout, Flatten, MaxPooling2D, Rescaling
from tensorflow.keras import backend as K


class MiniVGGNet:
    '''
    Class to construct MiniVGGNet model.
    
    MiniVGGNet Architecture:
        INPUT => (CONV => RELU => CONV => RELU => POOL)*2 => 
            FC => RELU => FC => SOFTMAX
    '''
    @staticmethod
    def build(width, height, depth, classes):
        '''
        Static method to build the MiniVGGNet model architecture.
    
        Parameters:
            width (int): Width of the input image.
            height (int): Height of the input image.
            depth (int): Depth of the input image.
            classes (int): Number of output classes to learn to predict.
          
        Returns:
            model: Constructed minivggnet network architecture.
        '''
        model = Sequential() # initialize the model along with the input shape to be "channels last" and the channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

	# if we are using "channels first", update the input shape and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        # normalizing the images
        model.add(Rescaling(1./255, input_shape=inputShape))

  	# first CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

  	# second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # remaining, FC => RELU => FC => SOFTMAX
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # model name       
        model._name = 'MiniVGGNet'

        return model
