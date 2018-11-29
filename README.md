## Deep Learning Project -Follow me 
This project's goal is control a drone track a specific person. In this project deep learning method is used in recognizing person. Pixel-wise segmentation problem.
In order to train a deep learning network, images and labeled mask can be generated with simulator which udacity provides.(But I just use the data in workspace)

### Contents
1. make environment
2. write encoder, decoder block and fully convolutional neural network.
3. Tuning the hyperparameters based on loss and final score.
4. Results
5. Problems

#### 1. make environment
The whole environment was setted up in anaconda virtual environment. With the udacity provided resource, environment was built easily.

#### 2. write encoder, decoder block and fully convolutional neural network.
Add encoder and decoder block with TODO hint.

My neural network structure show as below. It is pretty deep and with 1x1 convolutional layer.

![layer structure](https://github.com/Fred159/RoboticsND-Follow-me-deeplearning/blob/master/Project%20Image/layer.png)

Encoder step is just general convolution process.The convolustion process was implemented to image for extract features of image. The more deeper, the more abstract and important features can be extracted. But why it called encoder? Because, the neural network encode the image in to a feature map with many channels. However, we will lose some spatial information from the origin image since the channel increase. When we compress image with a person, then with convolusion step, each slide window's feature can be extracted. However, spatial information of each pixel is not be extracted.
So we use 1x1 connvolution layer to extract spatial information from whole channels. After this step, feature map with spatial information is made.We encode image from [160,160,3]  to [10,10.256]<=this is feature. 
For segmentation problem, we need to upsample the each feature map. This is why we call the process as decoder. Because decoder's feature is using the feature map to recover the original image's size [160,160,3]. The pixel-wise segmentation mission is done.
When decoder recover, it can't recover the feature map to original image. Because when pooling step and relu step, the feature map actually change a lot, so when recover step, decoder can't upsample the pixel as origin image's accuracy. It means we always lose spatial information of origin image step by step.

Stride is always [2,2] except in 1x1 convolutional layer. The layer's depth is finally change to 256 with 1x1 convolutional network.
The 1x1 convolutional network can extract spatial information in 256 layer.1x1 convolution layer also can decrease the layers' depth, it make layer can be trained more easily. It give each pixel a specific label, so this 1x1 convolution is the key of pixel wise segmentation problems!

![convolution 1x1 feature](https://github.com/Fred159/RoboticsND-Follow-me-deeplearning/blob/master/Project%20Image/1x1%20convolution.jpg)

After that, in order to get enough information in original image, decoder layers were concatenated with encoder layer by using the 'decoder_block' function.

##### 3. Tuning the hyperparameters based on loss and final score.
Hyperparamter finally confirmed as below.

![hyperparamte](https://github.com/Fred159/RoboticsND-Follow-me-deeplearning/blob/master/Project%20Image/hyperparameter.png)

I also tried other parameter set like below. Overfittig or underfitting happened during training. Those Hyperparameter gave a final score very near to 0.4(like 0.3999), but no one set can gave a score >  0.4. 
batch size was selected with the memory of system.
epoch was selected by training loss and trail
steps_per_epoch was selected by 'image number/batch size'

![other hyperparamter](https://github.com/Fred159/RoboticsND-Follow-me-deeplearning/blob/master/Project%20Image/hyperpara.png)

#### 4. Results
After training, training results shown as below. This traning step took 6hours......

![traning loss](https://github.com/Fred159/RoboticsND-Follow-me-deeplearning/blob/master/Project%20Image/trainingloss.png)

And the prediction results shown as below. Final score is 0.42. 

* Actually, I think model can be enhanced. 1st, change the train data. 2nd, increase train epoch. (Based on train loss figure, I think this model training step can stop at 50epoch.)3rd, make a deeper network.(It needs more time) 4. Add preprocess step in preprocess_ims.py. As far as I know preprocess step is pretty inportant to CNN.

![test1](https://github.com/Fred159/RoboticsND-Follow-me-deeplearning/blob/master/Project%20Image/test1.png)

![test2](https://github.com/Fred159/RoboticsND-Follow-me-deeplearning/blob/master/Project%20Image/test2.png)

![test3](https://github.com/Fred159/RoboticsND-Follow-me-deeplearning/blob/master/Project%20Image/test3.png)

Finally , the drone was tested in simulator. simulation results shown as below.

![result1](https://github.com/Fred159/RoboticsND-Follow-me-deeplearning/blob/master/Project%20Image/20181126_151004.jpg)

![result2](https://github.com/Fred159/RoboticsND-Follow-me-deeplearning/blob/master/Project%20Image/20181126_151018.jpg)

![result3](https://github.com/Fred159/RoboticsND-Follow-me-deeplearning/blob/master/Project%20Image/20181126_151037.jpg)

![result4](https://github.com/Fred159/RoboticsND-Follow-me-deeplearning/blob/master/Project%20Image/20181126_151125.jpg)

![result5](https://github.com/Fred159/RoboticsND-Follow-me-deeplearning/blob/master/Project%20Image/20181126_151134.jpg)

* One of the question from udacity is"whether this model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required?"
ans: Of course this model can't be used to recognize the cat,dog. This model is only trained to recognize human. If we wan't to recognize the cat or dog, we need to train the network with specific image and labeled image of cat or dog.

#### 5. Problems
* Deep learning processing is so slow. So it needs many time to tune paramter, however as long as change one parameter, it needs several hours to confirm the result.
* Preprocess_ims.py file can't run correctly with anaconda virtual environment. I don't know how to fix that problem, so I couldn't generate extrac dataset.
* ROS is not used in this project. Maybe drone control is implyed, but I don't figure it out yet.
* Actually, I think with more train and more data , final score can be increase a lot with more preprocess.
* Generate image label is pretty hard...



------------------------------------------------------------------------------------------------------------------------------



[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##

In this project, you will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 

## Setup Instructions
**Clone the repository**
```
$ git clone https://github.com/udacity/RoboND-DeepLearning.git
```

**Download the data**

Save the following three files into the data folder of the cloned repository. 

[Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip) 

[Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip)

[Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

**Download the QuadSim binary**

To interface your neural net with the QuadSim simulator, you must use a version QuadSim that has been custom tailored for this project. The previous version that you might have used for the Controls lab will not work.

The simulator binary can be downloaded [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)

**Install Dependencies**

You'll need Python 3 and Jupyter Notebooks installed to do this project.  The best way to get setup with these if you are not already is to use Anaconda following along with the [RoboND-Python-Starterkit](https://github.com/udacity/RoboND-Python-StarterKit).

If for some reason you choose not to use Anaconda, you must install the following frameworks and packages on your system:
* Python 3.x
* Tensorflow 1.2.1
* NumPy 1.11
* SciPy 0.17.0
* eventlet 
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* transforms3d
* PyQt4/Pyqt5

## Implement the Segmentation Network
1. Download the training dataset from above and extract to the project `data` directory.
2. Implement your solution in model_training.ipynb
3. Train the network locally, or on [AWS](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us).
4. Continue to experiment with the training data and network until you attain the score you desire.
5. Once you are comfortable with performance on the training dataset, see how it performs in live simulation!

## Collecting Training Data ##
A simple training dataset has been provided in this project's repository. This dataset will allow you to verify that your segmentation network is semi-functional. However, if your interested in improving your score,you may want to collect additional training data. To do it, please see the following steps.

The data directory is organized as follows:
```
data/runs - contains the results of prediction runs
data/train/images - contains images for the training set
data/train/masks - contains masked (labeled) images for the training set
data/validation/images - contains images for the validation set
data/validation/masks - contains masked (labeled) images for the validation set
data/weights - contains trained TensorFlow models

data/raw_sim_data/train/run1
data/raw_sim_data/validation/run1
```

### Training Set ###
1. Run QuadSim
2. Click the `DL Training` button
3. Set patrol points, path points, and spawn points. **TODO** add link to data collection doc
3. With the simulator running, press "r" to begin recording.
4. In the file selection menu navigate to the `data/raw_sim_data/train/run1` directory
5. **optional** to speed up data collection, press "9" (1-9 will slow down collection speed)
6. When you have finished collecting data, hit "r" to stop recording.
7. To reset the simulator, hit "`<esc>`"
8. To collect multiple runs create directories `data/raw_sim_data/train/run2`, `data/raw_sim_data/train/run3` and repeat the above steps.


### Validation Set ###
To collect the validation set, repeat both sets of steps above, except using the directory `data/raw_sim_data/validation` instead rather than `data/raw_sim_data/train`.

### Image Preprocessing ###
Before the network is trained, the images first need to be undergo a preprocessing step. The preprocessing step transforms the depth masks from the sim, into binary masks suitable for training a neural network. It also converts the images from .png to .jpeg to create a reduced sized dataset, suitable for uploading to AWS. 
To run preprocessing:
```
$ python preprocess_ims.py
```
**Note**: If your data is stored as suggested in the steps above, this script should run without error.

**Important Note 1:** 

Running `preprocess_ims.py` does *not* delete files in the processed_data folder. This means if you leave images in processed data and collect a new dataset, some of the data in processed_data will be overwritten some will be left as is. It is recommended to **delete** the train and validation folders inside processed_data(or the entire folder) before running `preprocess_ims.py` with a new set of collected data.

**Important Note 2:**

The notebook, and supporting code assume your data for training/validation is in data/train, and data/validation. After you run `preprocess_ims.py` you will have new `train`, and possibly `validation` folders in the `processed_ims`.
Rename or move `data/train`, and `data/validation`, then move `data/processed_ims/train`, into `data/`, and  `data/processed_ims/validation`also into `data/`

**Important Note 3:**

Merging multiple `train` or `validation` may be difficult, it is recommended that data choices be determined by what you include in `raw_sim_data/train/run1` with possibly many different runs in the directory. You can create a temporary folder in `data/` and store raw run data you don't currently want to use, but that may be useful for later. Choose which `run_x` folders to include in `raw_sim_data/train`, and `raw_sim_data/validation`, then run  `preprocess_ims.py` from within the 'code/' directory to generate your new training and validation sets. 


## Training, Predicting and Scoring ##
With your training and validation data having been generated or downloaded from the above section of this repository, you are free to begin working with the neural net.

**Note**: Training CNNs is a very compute-intensive process. If your system does not have a recent Nvidia graphics card, with [cuDNN](https://developer.nvidia.com/cudnn) and [CUDA](https://developer.nvidia.com/cuda) installed , you may need to perform the training step in the cloud. Instructions for using AWS to train your network in the cloud may be found [here](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us)

### Training your Model ###
**Prerequisites**
- Training data is in `data` directory
- Validation data is in the `data` directory
- The folders `data/train/images/`, `data/train/masks/`, `data/validation/images/`, and `data/validation/masks/` should exist and contain the appropriate data

To train complete the network definition in the `model_training.ipynb` notebook and then run the training cell with appropriate hyperparameters selected.

After the training run has completed, your model will be stored in the `data/weights` directory as an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file, and a configuration_weights file. As long as they are both in the same location, things should work. 

**Important Note** the *validation* directory is used to store data that will be used during training to produce the plots of the loss, and help determine when the network is overfitting your data. 

The **sample_evalution_data** directory contains data specifically designed to test the networks performance on the FollowME task. In sample_evaluation data are three directories each generated using a different sampling method. The structure of these directories is exactly the same as `validation`, and `train` datasets provided to you. For instance `patrol_with_targ` contains an `images` and `masks` subdirectory. If you would like to the evaluation code on your `validation` data a copy of the it should be moved into `sample_evaluation_data`, and then the appropriate arguments changed to the function calls in the `model_training.ipynb` notebook.

The notebook has examples of how to evaulate your model once you finish training. Think about the sourcing methods, and how the information provided in the evaluation sections relates to the final score. Then try things out that seem like they may work. 

## Scoring ##

To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. 

In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. 

We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the label mask. 

Using the above the number of detection true_positives, false positives, false negatives are counted. 

**How the Final score is Calculated**

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data

**Ideas for Improving your Score**

Collect more data from the sim. Look at the predictions think about what the network is getting wrong, then collect data to counteract this. Or improve your network architecture and hyperparameters. 

**Obtaining a Leaderboard Score**

Share your scores in slack, and keep a tally in a pinned message. Scores should be computed on the sample_evaluation_data. This is for fun, your grade will be determined on unreleased data. If you use the sample_evaluation_data to train the network, it will result in inflated scores, and you will not be able to determine how your network will actually perform when evaluated to determine your grade.

## Experimentation: Testing in Simulation
1. Copy your saved model to the weights directory `data/weights`.
2. Launch the simulator, select "Spawn People", and then click the "Follow Me" button.
3. Run the realtime follower script
```
$ python follower.py my_amazing_model.h5
```

**Note:** If you'd like to see an overlay of the detected region on each camera frame from the drone, simply pass the `--pred_viz` parameter to `follower.py`
