# SIGN LANGUAGE TRANSLATOR
**Skills learnt and applied** - Computer Vision, OpenCV, TensorFlow, Mediapipe,Matplotlib, Image processing and Video processing.

## The Idea
The basic idea as of now is to create a sign language translator that could understand the gestures done by anyone in real-time where the input is provided through the webcam and the output should be the meaning of the gestures displayed on the screen in real-time as well. For this a model was trained on 3 basic and common gestures from the German Sign Language but it could also be leveraged using more signs or gestures to the complete **German Sign Language or Deutsche Gebärdensprache(DGS)**.

## Installing and importing the libraries
First we need to install and import the necessary libraries. But before that we can also check the libraries that we already have. This can be done by running the command

```
!pip list
```

Now moving on to installing the libraries we need. 

```
!pip install tensorflow mediapipe
```

The rest of the libraries were already installed on my laptop so will directly import it. There are some more modules we need but will import them when they are needed.

```
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import mediapipe as mp
import cv2
import tensorflow as tf
```






