# Deep Learning-assisted Driving Attention Aid

![](https://img.shields.io/badge/python-3.7-blue.svg)
![](https://img.shields.io/badge/tensorflow-1.14-orange.svg)
![](https://img.shields.io/badge/opencv-4.2-brightgreen.svg)
![](https://img.shields.io/badge/nvidia%20driver-430.50-green)

![](example.gif)

(This work was done as a part of an engineering capstone project with a team of 4 people. Final "polished" version included additional features such as traffic light colour detection using k-NN and traffic sign classification with VGG19 models. This is not included in this public repository)

This application highlights and warns driver of objects-of-interest on the road using only video feed and provides feedback using a number of computer vision and deep learning approaches. Only objects that are of *immediate attention and in close proximity* are highlighted (within ~20m in front of the vehicle), rest are ignored. This includes lane departure, object recognition, and traffic sign/light recognition (see Features for more info). In order to run, Anaconda enviroment is **highly** recommended, as matching different packages versions can be very tedious with a lot of errors.

Using an NVIDIA GeForce GTX 1080 (2560 CUDA cores) resulted in about 15 FPS. In order to speed up the processing, Mask R-CNN detection and processing is done only on every *third frame*, since it was deemed that there are not many changes on frame-by-frame basis for road applications. This lowers overhead, without sacrificing any real-world performance.

## Features
* Lane detection using Probabalistic Hough Transforms (bottom of screen); used for lane drift warning
* Mask R-CNN model pre-trained on Microsoft COCO dataset for road object detection/segmentation to identify vehicles, cyclists, parking spots, traffic lights, etc.
* Visual feedback to user after any of the above were detected
* Distributed client/worker: has provisions to have a networked camera connect to a CUDA enabled worker for portable deployment with Flask
* *Traffic sign and traffic light colour classification with VGG19 and kNN (not included in this public repo)*

# Acknowledgements

This work is based on [Matterport's Mask R-CNN](https://github.com/matterport/Mask_RCNN) repository, who have done amazing job of implementing Mask R-CNN and providing additional materials. Additionally, [noxouille's rt-mrcnn repo](https://github.com/noxouille/rt-mrcnn) offered good ideas on how to improve real-time performance. And of course, rest of the team (D, D, A)!

# How To Run

Run the primary script is `webTest.py` which will create a Flask server at 127.0.0.1:5000. Alternatively, you can run `demo.sh` from shell which will run it.

 From there, a demo video called *drive15_trim.mp4* will be used to run inference on (can be changed in lines 86,89 in `webTest.py`; *any video with 640x480 preferably can be used*), and processed frames will be seen in the web browser at the server link. 
 
 There are provisions in the code to use live video feed from a web camera using OpenCV (local or network-based), however, they are commented out in `webTest.py`; re-enable by uncommenting accordingly.

# Installation (taken from Matterport's repo)

1. Clone this repository and go into your conda environment
2. Install dependencies in order
   ```bash
   cat requirements.txt | xargs -n 1 -L 1 pip install
   ```
3. Run setup from the repository root directory
    ```bash
    python setup.py install
    ``` 
4. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).

## Requirements
Python 3.7, TensorFlow 1.14 and other common packages listed in `requirements.txt`. Tested to be working on Ubuntu 18.04.1, with CUDA 10.1, cuDNN v7.3.1.20 and, NVIDIA Driver 430.50 for GPU support.