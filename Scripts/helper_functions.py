# AIM: To create a python file that maintains helper functions.


from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import os
import shutil


def move_images(imagePaths):
    '''
    Function to move images into seperate train and validation directories.
    
    Parameters:
        imagePaths (list): List containing paths to all the images.
        
    Returns:
        None.
    '''
    labels = [] # list to collect labels
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
        # create training directory with class folders
        if not os.path.exists('train_dir/' + label + '/'):
            os.makedirs('train_dir/' + label + '/')
        # create validation directory with class folders
        if not os.path.exists('valid_dir/' + label + '/'):
            os.makedirs('valid_dir/' + label + '/')

    # do stratified sampling
    (trainX, validX, _, _) = train_test_split(imagePaths, labels,
	    test_size=0.3, random_state=42, stratify=labels)

    # move images into separate train directory
    for trainx in trainX: 
        to_path = os.path.join("train_dir/", os.sep.join(trainx.rsplit(r"/")[-2:]))
        from_path = os.path.join("raw-img/", os.sep.join(trainx.rsplit(r"/")[-2:]))
        shutil.move(from_path, to_path)

    # move images into separate validation directory
    for validx in validX: 
        to_path = os.path.join("valid_dir/", os.sep.join(validx.rsplit(r"/")[-2:]))
        from_path = os.path.join("raw-img/", os.sep.join(validx.rsplit(r"/")[-2:]))
        shutil.move(from_path, to_path)

    return 0


def plot_graphs(history):
    '''
    Function to plot accuracy and loss graph.
    
    Parameters:
        history (object): Trained model.
        
    Returns:
        fig (object): Accuracy and loss plots.
    '''
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    N = np.arange(0, len(history.history['loss']))

    fig = plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(N, accuracy, label='Training Accuracy')
    plt.plot(N, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(N, loss, label='Training Loss')
    plt.plot(N, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Cross Entropy Loss')

    return fig
