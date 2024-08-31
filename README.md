# Facial Pose Estimation System

This project was developed as part of the National Telecommunication Institute (NTI) Machine Learning for Data Analysis Internship. The Facial Pose Estimation System is designed to detect and predict the orientation of faces in images and videos by estimating the pitch, yaw, and roll angles using machine learning techniques. This system combines the capabilities of dlib for facial landmark detection, SVM models for pose prediction, and MediaPipe for real-time visualization.

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
  - [Pose Estimation on Images](#pose-estimation-on-images)
  - [Pose Estimation on Videos](#pose-estimation-on-videos)
- [Results](#results)

## Overview

The Facial Pose Estimation System accurately predicts the **pitch**, **yaw**, and **roll** of a face using a combination of machine learning models and real-time facial landmark detection. This system can be applied in various fields such as driver monitoring systems, virtual reality, human-computer interaction, and more.

## Technologies Used

- **Python 3.x**: The programming language used for development.
- **OpenCV**: For processing images and videos.
- **dlib**: Utilized for facial landmark detection.
- **scikit-learn**: Employed for training and using SVM models.
- **MediaPipe**: Used for visualizing facial landmarks and pose estimation.
- **tkinter**: Provides a graphical interface for file selection.

## Installation

To get started with the Facial Pose Estimation System, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AbdelrahmanAboegela/NTI-face-pose-estimation.git
   cd NTI-face-pose-estimation
   
2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt

3. **Download the dlib pre-trained model:**
   Download ```shape_predictor_68_face_landmarks.dat```.

## Usage

## 1.Pose Estimation on Images
The SVM models for predicting pitch, yaw, and roll angles are already trained and ready to use. To perform facial           pose estimation on individual images:
   
**Run the script:**
   ```bash
   python predict_face_pose_image.py
   ```
This script will prompt you to select an image file, process it, and display the image with the predicted pose angles.

## 2.Pose Estimation on Videos
To perform facial pose estimation on video files or live camera feeds:
   
**Run the script:**
```bash
python predict_face_pose_video.py
```
This script will allow you to select a video file. It processes each frame, predicts the pose angles, and saves the output video with visualizations.

## Results
The results from the system include visualizations of the facial landmarks and the estimated pitch, yaw, and roll angles superimposed on the images or video frames.





