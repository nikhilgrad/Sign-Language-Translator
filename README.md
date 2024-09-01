# SIGN LANGUAGE TRANSLATOR
**Skills learnt and applied** - Computer Vision, Time-Series data processing(Image processing & Video processing), LSTM, OpenCV, TensorFlow, Mediapipe, Matplotlib.

## The Idea
The basic idea as of now is to create a sign language translator that could understand the gestures done by anyone in real-time where the input is provided through the webcam and the output should be the meaning of these gestures that would be displayed on the screen in real-time as well. For this a model was trained on 3 basic and common gestures from the German Sign Language but it could also be leveraged using more signs or gestures to complete **German Sign Language or Deutsche Gebärdensprache(DGS)**.

![Sign Language Translator](https://github.com/user-attachments/assets/291c56f0-d883-4b56-a941-908349bb7603)


## Installing and importing the libraries
First we need to install and import the necessary libraries. But before that we can also check the libraries that we already have. This can be done by running the command

```
!pip list
```

Now moving on to installing the libraries we need. 

```
!pip install tensorflow mediapipe
```

MediaPipe library is a framework for building machine learning pipelines for processing time-series data like video and audio.

The rest of the libraries were already installed on my laptop so will directly import it. There are some more modules that we need but will import them at a later stage.

```
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import mediapipe as mp
import cv2
import tensorflow as tf
```

## Creating keypoints using Mediapipe

For tracking and reading the gestures we are using Mediapipe library. With mediapipe we will create keypoints on face, left hand and right hand. It will also detect the pose and create a **Holistic model** with **MediaPipe Holistic** (which is a pipeline that combines pose, hands and face detection to create a computer vision model). We have to make 2 variables one for mediapipe holistic which does the detection and another for drawing of keypoints on ourselves.

```
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils
```

**Now we need to define 2 functions one that detects the images and passes it to the model for processing and another that takes the images and results from first function and renders it by drawing the markings**

As OpenCV reads frames in BGR format and Mediapipe processes the frames(images) in RGB format so we have to convert the BGR format frames taken from webcam by OpenCV to RGB format for Mediapipe to process. For this we use `cv2.COLOR_BGR2RGB` and then after passing image to model for processing we will again change the format to BGR by using `cv2.COLOR_RGB2BGR` so that it can again be given back by OpenCV as output.

```
def mp_detect(image,model):
    
    # Converting the colour using cv2.cvtColor from BGR format to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Image is not writeable that is no editing can be done to image now
    image.flags.writeable = False
    
    # Detecting the image or the feed using the mediapipe
    results = model.process(image)
    
    # We again make the image writeable
    image.flags.writeable = True
    
    # Reconverting the color back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, results 
```

The below function uses `results.face_landmarks` to place the markings on the face whereas `mp_holistic.FACEMESH_CONTOURS` is used for detecting where exactly the face is situated to place the markings on it. The first `mp_drawing.DrawingSpec` will colour the dots(markings) whereas the second is for colouring the line(connections) that joins the markings(landmarks) to each other. This is same for other markers for pose and both hands.

```
def draw_landmarks(image, results):

    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )     

    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(22,44,221), thickness=2, circle_radius=2)
                             )

    # Draw  left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,235,255), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(45,66,30), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2)
                             ) 
```

## Extract keypoint values

We define a function to extract the keypoints when they are already drawn using the above functions so that we can further process it. `pose` extracts the pose landmarks (33 points) if they are available. Each landmark includes the x, y, z coordinates and visibility score. If pose landmarks are not available, it returns an array of zeros of size 33*4. Similar is the case with `face`, `lh` and `rh`. Finally, it concatenates all these arrays into a single array and returns it. This function is useful for collecting all the keypoints from different parts of the body detected by MediaPipe into a single, flattened array for further processing or analysis. Also the values in parantheses of `np.zeros` represents the number of keypoints in that type of landmark. So the total number of landmarks becomes (33 *4) + (468 *3) + (21 *3) + (21 *3) = 1662 keypoints.

```
def extract_keypoints(results):

    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)

    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, face, lh, rh])
```

One thing to observe in the above code is that there is a visibility score for pose landmarks given by `res.visibility` which is absent in other types of landmarks. The other landmarks in MediaPipe do not include a visibility score is because the face and hands mesh model used by MediaPipe is designed to provide highly detailed and dense landmark points (468 for face and 21 for each hand) for facial features. These landmarks are primarily used for applications like facial recognition, hand gestures recognition etc. So the primary focus is on the precise location of each point rather than its visibility. 

In contrast, the pose landmarks include a visibility score because they are used in scenarios where the visibility of body parts can vary significantly due to occlusion, movement, or camera angle. The visibility score helps to determine how reliable each landmark is, which is crucial for applications like pose estimation and activity recognition.

![Markings3](https://github.com/user-attachments/assets/3d771980-0964-46e0-b985-719c3ec085b9)

*Keypoints marked on face*


## Creating folders for collecting keypoints

We have to save the extracted keypoints in order to use them to decode our sign language at a later stage.

```
#Path for exported data
DATA_PATH = os.path.join('MP_Data') 

#Actions or gestures that we try to detect
actions = np.array(['hallo', 'danke', 'liebe'])

#Thirty videos 
video_sequence = 30

#Each video will be 30 frames in length
sequence_length = 30

#loop through our actions 
for action in actions: 
    
    #loop through our videos
    for sequence in range(video_sequence):
        
        #create a folder if it is not present and if present pass it on
        try:
            #make directory at location DATA_PATH with the name of one of the actions and then make a subdirectory with name as that of sequence which are just 0,1,2 etc
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            
            #if we have the folder then just skip or pass
        except:
            pass
```

## Data collection

We need to collect keypoint values through videos so that we can use that for Training and Testing of our translator. For this we will directly take the input from our webcam feed by capturing videos through it.

**1. Initialize the webcam for video capture**
```
cap = cv2.VideoCapture(0)
```

**2. Set Up MediaPipe Holistic Model**

```
# Set up the MediaPipe holistic model with specified confidence thresholds for detection and tracking
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    # Loop through actions, videos and frames
    for action in actions:
        for sequence in range(video_sequence):
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mp_detect(frame, holistic)

                # Draw landmarks
                draw_landmarks(image, results)
                
                # If we are at frame 0 then we will wait for 2 seconds and print 'STARTING COLLECTION' and then print 'Collecting frames for {} Video Number {}' 
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)

                    # This 2000 is in milliseconds thus it corresponds to 2 seconds of waiting time.
                    cv2.waitKey(2000)
                
                # If we are not at frame 0 then we will directly print the below lines and won't print 'STARTING COLLECTION'
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    
                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                
                # This line saves our arrays of keypoints as a numpy array at location npy_path 
                np.save(npy_path, keypoints)
                
                
                # Break allows the user to break the loop and stop the video capture by pressing the 'q' key
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
```

**Release the webcam and close all OpenCV windows**                  

```
cap.release()
cv2.destroyAllWindows()
```

## Preprocess data and then create labels and features

To do this we need to import some libraries 
```
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
```

The `train_test_split` will split the data into training data and test data which will be used for validation of our model. It is just a good practice to test our model before deploying.

The `to_categorical` function from TensorFlow’s Keras API is used to convert integer labels into one-hot encoded vectors. **One-Hot Encoding** is a technique used to convert categorical data into a binary matrix representation. Each category is represented as a binary vector, where only one element is “hot” (1) and all others are “cold” (0). This is useful for machine learning models that require numerical input. It is also good for categories that do not have a natural order i.e the data is unique and unrelated. 

For example we have an array of actions ['hallo', 'danke', 'liebe']. One-hot encoding these categories would result in:

[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
 
where each row corresponds to a one-hot encoded vector for the respective label. 

Now we convert our actions from categorical labels to numerical labels which are often required for machine learning models. We do this by creating a dictionary using dictionary comprehension where our label(action name) is the key and num(index of that action) is value.

```
label_map = {label:num for num, label in enumerate(actions)}
```

We need to create features and labels so that the model can use them to learn. Here sequences are feature data which is a list and is intended to store multiple sequences, where each sequence is a list of keypoints for all frames in that sequence and labels are the label data which is again a list to store corresponding label for each sequence.

```
sequences, labels = [], []

# Loop through actions and then video_sequence for each action
for action in actions:
    for sequence in range(video_sequence):

        # Collect keypoints for all frames in a sequence
        window = []
        for frame_num in range(sequence_length):

            # Load the keypoints for the current frame from a .npy file and append them to window list.
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)

        # Append window list which contains keypoints for entire sequence to sequences list
        sequences.append(window)

        # The label_map dictionary maps each action to a unique integer and then the corresponding label for the current action is appended to the labels list. 
        labels.append(label_map[action])
```

Create training and test data from total data of features and labels. `test_size=0.05` specifies that 5% of total data will be used for testing. 

```
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
```

## Build and train LSTM Neural Network

We need to import some dependencies from tensorflow, which we have already installed. 

```
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense, Dropout

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
```

Now we need to setup a Tensorboard callback for monitoring and visualizing the training process. The `TensorBoard` class is part of TensorFlow’s Keras API, and it allows us to log events for visualization in TensorBoard. Thus it is a visualization tool provided by TensorFlow that helps us understand and debug our machine learning models. The `log_dir` parameter specifies the directory where the logs will be saved.

**Setting the model architecture**
The architecture consists of 3 `LSTM` layers(a type of RNN that are good at learning from sequences of data) with 64, 128 and 64 neurons each and with `relu` activation function which introduces non-linearity. Also each one of them is followed by a `Dropout` layer. Then we have 3 dense layers with `relu` activation function

```
model = Sequential()

model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
```

**Start the training**

```
model.fit(X_train, y_train, epochs=500, batch_size=32, shuffle=False, callbacks=[tb_callback])
```

To see the summary of the model as in the layer type, output shape and number of parameters in each layer and it also shows that out of all the parameters how many are trainable etc. 

```
model.summary()
```

## Save the model and make predictions in real-time
To save the model

```
model.save('sign.h5')
```

This function `prob_viz` visualizes the probabilities of different actions on an input frame using colored rectangles and text. The `cv2.rectangle` draws the rectangle on output frame for each action probability. The rectangle’s height is fixed, but its width is proportional to the probability `(int(prob*100))`. Position of the rectangle is determined by `num`, ensuring each rectangle is drawn below the previous one. The `cv2.putText` writes the action's name in the rectangles that were drawn  

```
colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):

    # Copy the input frame to avoid modifying the original image
    output_frame = input_frame.copy()

    # Loop through probabilities and their corresponding indices
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    return output_frame
```

Now since we have already trained the model and created the rectangles that contain our actions(our gestures) namely hallo, danke and liebe we can move on to writing the final part of the code and that is real-time detection.

```
# 1. Detection variables

# Sequence will collect our 30 frames in order to generate prediction and as we loop through openCV we will append to this and then pass it our prediction algorithm to start predictions 
sequence = []

# Sentence will allow us to concatenate our history of detections together 
sentence = []
predictions = []

# Threshold will render the result when it is above certain threshold which is chosen arbitrarily
threshold = 0.7

# Start the webcam
cap = cv2.VideoCapture(0)

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mp_detect(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_landmarks(image, results)

        # 2. Prediction logic

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        # We will only predict if we have 30 frames 
        if len(sequence) == 30:
            # Expand_dims adds an extra dimension allowing us to pass one sequence as well
            # It makes our one sequence from 2D to 3D and as we have given axis=0 then it will be along rows
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))    
            
        # 3. Vizualization logic
            
            # Grabs the last 10 predictions and prints only the unique actions from it, this gives more stability while predicting our actions
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 

                    # Check if we have words in the sentences array. If not, append to it.
                    # If we do,check the current predicted word is same or not. If it is same then skip the append to prevent duplication.
                    # If it is not same then append
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            # If appended sentence is >5 then take the last 5 value to not end up with the giant array we want to render.
            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Visualization probabilities
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
```

![Hallo2](https://github.com/user-attachments/assets/8dfa513c-27ae-4cdf-98f5-2bb117e744f3)

![Danke2](https://github.com/user-attachments/assets/3943584f-fcf9-48ec-aa38-3daea406ff41)

![Liebe2](https://github.com/user-attachments/assets/d79ec9ee-a0b0-4276-a903-70d3c4c90222)




Then press **"interrupt the kernel"** button  and run the below code.

**To release the camera and close all windows**

```
cap.release()
cv2.destroyAllWindows()
```
 














