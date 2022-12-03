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

    1.  Classifying more than 26,000 images into 10 classes results in fewer images per category.
    2.  The dataset is imbalanced, i.e., few categories have a large number of images compared to others.
    3.  Images are varying in size and medium in quality.


> <b>Aim of the project is to handle class imbalance and perform multiclass image classification.</b>

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- PROJECT SUMMARY -->
## Summary
At first, the image data is analyzed to understand the structure of the dataset, the number of classes and images per class, as well as the quality of the images (<a href="data_analysis.ipynb">data_analysis.ipynb</a>). Image sizes are analyzed seprately (<a href="image_size_analysis.ipynb">image_size_analysis.ipynb</a>). Image sizes are found to be of varying sizes, thus suggesting that progressive resizing could lead to better results. 

Shallownet (<a href="shallownet.py">shallownet.py</a>), Lenet (<a href="lenet.py">lenet.py</a>), and miniVGGnet (<a href="minivggnet.py">minivggnet.py</a>) model architectures are implemented from scratch. Whereas, MobileNetV3Large (<a href="mobilenetv3l.py">mobilenetv3l.py</a>) and (<a href="mobilenetv3s.py">mobilenetv3s.py</a>) model architecture is specifically chosen for transfer learning. Why? What could be the application of identifying animals through images?
*   In forests, to send alerts when specific animal is identified through surveillance camera or drones.
*   Common users, using a mobile application to identify animals.

In both the mentioned scenarios, mobile and embedded vision system is used and MobileNet model architecture is specifically desined for it.
Resnet50v2 (<a href="resnet50v2.py">resnet50v2.py</a>) model architecture is used to comparative analysis with MobileNetV3 model architectures.

Training results, and observations from various experiments carried out are explained in depth in training files (notebooks).

Summarizing top results:

<div align="center">

Rank | Model Architecture | Model Number | Trainable Parameters | Training Loss | Training Acc | Validation Loss | Validation Acc
:---: | :------------------------------------------------------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------:
1 | MobileNetV3Large | 1 | 9,610 | 0.1080 | 0.9818 | 0.1134 | 0.9716 
2 | MobileNetV3Large | 2 | 124,298 | 0.0939 | 0.9842 | 0.1100 | 0.9731
3 | ResNet50V2       | - | 20,490 | 0.1240 | 0.9780 | 0.1517 | 0.9593 
4 | MobileNetV3Small | 2 | 75,146 | 0.1769 | 0.9687 | 0.1780 | 0.9459 

</div>

> NOTE: <b>'Model Number'</b> is the number given to a particular configuration of the given model architecture. Refer training notebooks for in depth experiments, their configurations, and results.

> NOTE: Trainable parameters are result of the model configuration. Therefore, high number of trainable parameters is seen in MobileNetV3Small Model 3.

Although, <b>MobileNetV3Large Model 2</b> performs slightly better than <b>MobileNetV3Large Model 1</b>, however, the later one uses approximately 12X fewer trainable parameters compared to the former. Therefore, <b>MobileNetV3Large Model 2</b> is chosen as the best model.

All the four models are saved in h5 format in <a href="Models">Models</a> directory.


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
