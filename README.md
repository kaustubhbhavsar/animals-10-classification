<!-- PROJECT NAME -->

<br />
<div align="center">
  <h3 align="center">Identifying Animals from Images</h3>
  <p align="center">
    Multiclass Classification | Transfer Learning | Class Imbalance
    
  </p>
</div>

<!-- ABOUT PROJECT -->
## What Is It?
<a href="https://www.kaggle.com/datasets/alessiocorrado99/animals10">Animals-10</a> is an image dataset with 10 output classes, i.e., 10 classes of 10 different animals. According to the information given by the dataset provider, the data is collected from Google Images and verified by a human. The dataset contains around 26,000 images.

<b>Challenges:</b>

    1.  Classifying 26,000 images into 10 classes results in fewer images per category.
    2.  The dataset is imbalanced, i.e., few categories have a large number of images compared to others.
    3.  Images are varying in size and medium in quality.


> <b>Aim of the project is to handle class imbalance and perform multiclass image classification.</b>

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- PROJECT SUMMARY -->
## Summary


<p align="right">(<a href="#top">back to top</a>)</p>


<!-- Project Directory Structure -->
## Directory Structure
```
├── Data Analysis                         # Data analysis files
    ├── data_analysis.ipynb               # Data analysis (notebook) 
    └── image_size_analysis.ipynb         # Image size analysis (notebook)
├── Models                                # Saved trained model files
    ├── mobilenetv3l_basic_10_0.113.h5    # MobileNetV3Large model 1 
    ├── mobilenetv3l_dense_04_0.110.h5    # MobileNetV3Large model 2
    ├── mobilenetv3s_dense_07_0.178.h5    # MobileNetV3Small model 2
    └── resnet50v2_basic_06_0.152.h5      # ResNet50V2
├── Scripts                               # Scripts 
    ├── helper_functions.py               # Functions to make life easy
    ├── lenet.py                          # LeNet model architecture (from scratch)
    ├── minivggnet.py                     # MiniVGGNet model architecture (from scratch)
    ├── mobilenetv3l.py                   # MobileNetV3Large model architecture (transfer learning)
    ├── mobilenetv3s.py                   # MobileNetV3Small model architecture (transfer learning)
    ├── resnet50v2.py                     # ResNet50V2 model architecture
    └── shallownet.py                     # ShallowNet model architecture (from scratch, baseline)
├── 1_training_shallownet.ipynb           # Shallownet training (notebook) 
├── 2_training_lenet.ipynb                # Lenet training (notebook)
├── 3_training_miniVGGnet.ipynb           # MiniVGGnet training (notebook)
├── 4_training_mobileNetV3Small.ipynb     # MobileNetV3Small training (notebook)
├── 5_training_mobileNetV3Large.ipynb     # MobileNetV3Large training (notebook)
├── 6_training_resnet50v2.ipynb           # Resnet50V2 training (notebook)
```
> NOTE: Download the dataset from <a href="https://www.kaggle.com/datasets/alessiocorrado99/animals10">here</a>.
<p align="right">(<a href="#top">back to top</a>)</p>


<!-- Tools and Libraries used -->
## Languge and Libraries

*   Language: Python
*   Libraries: Tensorflow, Keras, Matplotlib, Seaborn, Numpy, Pandas, Shutil, PIL, OS, Imutils.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Final Notes -->
## Final Notes
To run the entire project use JupyterLab or similar IDE.

Notebooks can also be run directly on google colab (make sure to upload required .py files in working directory if required).

> NOTE: Notebooks may use python scripts to run.

> NOTE: High RAM is required to run the project. GPU can make life easier.

<p align="right">(<a href="#top">back to top</a>)</p>
