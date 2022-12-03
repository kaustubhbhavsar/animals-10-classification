# AIM: To create MobileNetV3Large(transfer learning) class.
# NOTE: For MobileNetV3Large, input preprocessing is part of the model by default. As we use
#       the preprocessing layer separately, it just acts as pass through function and has no 
#       effect whatsoever. 


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras import layers, Model, Input
from tensorflow.keras import backend as K


class MobileNetV3L:
    '''
    Class to construct MobileNetV3Large model.
    '''
    @staticmethod
    def build(width, height, depth, classes, data_aug, dense_layer):
        '''
        Static method to build the MobileNetV3Large model architecture.
    
        Parameters:
            width (int): Width of the input image.
            height (int): Height of the input image.
            depth (int): Depth of the input image.
            classes (int): Number of output classes to learn to predict.
            data_aug (boolean): If value set to True, then add data augmentation layer; else do not add.
            dense_layer (boolean): If value set to True, then add dense layer; else do not add.
        
        Returns:
            model: Constructed mobilenetv3large network architecture.
        '''
        # initialize the model along with the input shape to be "channels last" and the channels dimension itself
        inputShape = (height, width, depth)
        seed = 42 # set seed
        
		# if we are using "channels first", update the input shape and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        
        # add data augmentation layer if boolean value of dataAugmentation variable is set to 1
        data_augmentation = Sequential([
                layers.RandomFlip("horizontal", input_shape=inputShape, seed=seed),
                layers.RandomZoom(height_factor=-0.4, seed=seed),
                layers.RandomTranslation(0.3, 0.2, seed=seed)
            ])
        
        # defining mobilenetv3large network
        base_model = MobileNetV3Large(include_top=False,
                   input_shape=inputShape,
                   weights='imagenet')
        base_model.trainable=False
        
        # input layer
        inputs = Input(shape=inputShape)
        # data augmentation layer to be added if data_aug is set to True
        if data_aug==True:
            x = data_augmentation(inputs) # data augmentation layer
            x = preprocess_input(x) # preprocessing layer
        else:
            x = preprocess_input(inputs) # preprocessing layer
        # base model
        x = base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        # dense layer to be added if dense_layer is set to True
        if dense_layer==True:
            x = Dense(128, activation="relu")(x)
        x = Dropout(0.2)(x)
        # output layer
        outputs = Dense(classes, activation="softmax")(x)
        model = Model(inputs, outputs)

        # model name
        model._name = 'MobileNetV3Large'

        return model


