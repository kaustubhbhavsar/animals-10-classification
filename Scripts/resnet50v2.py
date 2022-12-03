# AIM: To create ResNet50(transfer learning) class.


from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras import Model, Input
from tensorflow.keras import backend as K


class ResNet50_V2:
    '''
    Class to construct ResNet50V2 model.
    '''
    @staticmethod
    def build(width, height, depth, classes):
        '''
        Static method to build the ResNet50V2 model architecture.
    
        Parameters:
            width (int): Width of the input image.
            height (int): Height of the input image.
            depth (int): Depth of the input image.
            classes (int): Number of output classes to learn to predict.
         
        Returns:
            model: Constructed ResNet50V2 network architecture.
        '''
        # initialize the model along with the input shape to be "channels last" and the channels dimension itself
        inputShape = (height, width, depth)
        seed = 42 # set seed
        
		# if we are using "channels first", update the input shape and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        
        # defining resnet50v2 network
        base_model = ResNet50V2(include_top=False,
                   input_shape=inputShape,
                   weights='imagenet')
        base_model.trainable=False
        
        # input layer
        inputs = Input(shape=inputShape)
        # preprocessing layer
        x = preprocess_input(inputs) 
        # base model
        x = base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        # output layer
        outputs = Dense(classes, activation="softmax")(x)
        model = Model(inputs, outputs)

        # model name
        model._name = 'ResNet50V2'

        return model