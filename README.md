# Visual-Odometry

## Introduction
 Localization is the main task for autonomous vehicles to be able to track their paths and properly detect and avoid obstacles. Vision-based odometry is one of the robust techniques used for vehicle localization. This section comprehensively discusses VO and its types, approaches, applications, and challenges.
 VO is the pose estimation process of an agent (e.g., vehicle, human, and robot) that involves the use of only a stream of images acquired from a single or from multiple cameras attached to it. The core of VO is camera pose estimation. It is an ego-motion online estimation process from a video input. This approach is a non-contact method for the effective positioning of mobile robots. VO provides an incremental online estimation of a vehicle’s position by analyzing the image sequences captured by a camera.
The idea of estimating a vehicle’s pose from visual input alone was introduced and described by Moravec in the early 1980s. From 1980 to 2000, VO research was dominated by NASA in preparation for the 2004 Mars Mission. The term "visual odometry" was selected because vision-based localization is similar to wheel odometry in that it incrementally estimates the motion of a vehicle by integrating the number of turns of its wheels over time. In the same manner, VO integrates pixel displacements between image frames over time.

### Requirements:
* opencv-3.0 
* numpy

In this project, there is more than 10 features extraction methods that are be tested with their combination to achieve to a best result in different external conditions.

For testing the project, only change the dataset path and the paths for saving results and put into your terminal:
* python Code.py

The main dataset used in this project is the KITTI- dataset, it is a open-source and useful dataset.
Also, for testing and comparing the different Features extraction, detection and matching methods, other datasets are used where they contain several images which differ by point of view, brightness, noise ratio, dark light,... These datasets can be found in the datasets file.

##### For details: alirida.sahili@gmail.com
