# Facial Pose Estimation System

This project was developed as part of the National Telecommunication Institute (NTI) Machine Learning for Data Analysis Internship. The Facial Pose Estimation System detects and predicts the pose of faces in images and videos using machine learning models. The project leverages tools like dlib for facial landmark detection, SVM models for pose prediction, and MediaPipe for visualization.

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

## Overview

The Facial Pose Estimation System predicts the **pitch**, **yaw**, and **roll** of a face using a combination of machine learning models and real-time facial landmark detection. This system can be used to analyze head orientation in various applications such as driver monitoring systems, virtual reality, and human-computer interaction.

## Technologies Used

- **Python 3.x**
- **OpenCV**: For image and video processing.
- **dlib**: For facial landmark detection.
- **scikit-learn**: For training and using SVM models.
- **MediaPipe**: For visualizing facial landmarks and pose estimation.
- **tkinter**: For file selection dialogs.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AbdelrahmanAboegela/NTI-face-pose-estimation.git
   cd NTI-face-pose-estimation
   
2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt

3. **Download the dlib pre-trained model:**
   Download shape_predictor_68_face_landmarks.dat from dlib's model page and place it in the project directory.

   ## Usage
   ### Running the System
   To run the facial pose estimation system on an image, video, or camera feed, follow these steps:

   1. **Run the script:**
   ```bash
   python facial_pose_estimation.py
