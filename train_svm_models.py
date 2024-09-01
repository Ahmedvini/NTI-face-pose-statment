import os
import cv2
import numpy as np
import dlib
import scipy.io
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to detect facial landmarks from an image
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    if len(rects) > 0:
        # Assuming one face per image
        shape = predictor(gray, rects[0])
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        return landmarks.flatten()  # Flatten the landmarks into a 1D array
    else:
        return None

# Function to load data and extract features and labels
def load_data_and_extract_features(directory):
    X = []  # Feature vectors (landmarks)
    y_pitch = []  # Pitch angles
    y_yaw = []  # Yaw angles
    y_roll = []  # Roll angles
    
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            image_path = os.path.join(directory, filename)
            mat_path = os.path.join(directory, filename.replace('.jpg', '.mat'))
            
            # Check if the corresponding .mat file exists
            if os.path.exists(mat_path):
                image = cv2.imread(image_path)
                landmarks = get_landmarks(image)
                
                if landmarks is not None:
                    X.append(landmarks)
                    
                    # Load the ground truth values from the .mat file
                    mat_data = scipy.io.loadmat(mat_path)
                    pose_params = mat_data['Pose_Para'][0][:3]  # Pitch, Yaw, Roll
                    
                    y_pitch.append(pose_params[0])
                    y_yaw.append(pose_params[1])
                    y_roll.append(pose_params[2])
                    
    return np.array(X), np.array(y_pitch), np.array(y_yaw), np.array(y_roll)

# Directory containing the images and mat files
directory = '/home/ahmed/Documents/GitHub/NTI-face-pose-statment-1/DataSet'

# Load data and extract features
X, y_pitch, y_yaw, y_roll = load_data_and_extract_features(directory)

# Split data into training and testing sets
X_train, X_test, y_pitch_train, y_pitch_test = train_test_split(X, y_pitch, test_size=0.2, random_state=42)
_, _, y_yaw_train, y_yaw_test = train_test_split(X, y_yaw, test_size=0.2, random_state=42)
_, _, y_roll_train, y_roll_test = train_test_split(X, y_roll, test_size=0.2, random_state=42)

# Initialize SVM regressors for each angle
svr_pitch = SVR(kernel='rbf')
svr_yaw = SVR(kernel='rbf')
svr_roll = SVR(kernel='rbf')

# Train the regressors
svr_pitch.fit(X_train, y_pitch_train)
svr_yaw.fit(X_train, y_yaw_train)
svr_roll.fit(X_train, y_roll_train)

# Predict on the test set
y_pitch_pred = svr_pitch.predict(X_test)
y_yaw_pred = svr_yaw.predict(X_test)
y_roll_pred = svr_roll.predict(X_test)

# Evaluate the model
mse_pitch = mean_squared_error(y_pitch_test, y_pitch_pred)
mse_yaw = mean_squared_error(y_yaw_test, y_yaw_pred)
mse_roll = mean_squared_error(y_roll_test, y_roll_pred)

print(f'Pitch MSE: {mse_pitch}')
print(f'Yaw MSE: {mse_yaw}')
print(f'Roll MSE: {mse_roll}')

# Plot predicted vs actual values for each angle
plt.figure(figsize=(15, 5))

# Pitch
plt.subplot(1, 3, 1)
plt.scatter(y_pitch_test, y_pitch_pred, c='blue')
plt.plot([min(y_pitch_test), max(y_pitch_test)], [min(y_pitch_test), max(y_pitch_test)], 'k--', lw=2)
plt.xlabel('True Pitch')
plt.ylabel('Predicted Pitch')
plt.title('Pitch: True vs Predicted')

# Yaw
plt.subplot(1, 3, 2)
plt.scatter(y_yaw_test, y_yaw_pred, c='green')
plt.plot([min(y_yaw_test), max(y_yaw_test)], [min(y_yaw_test), max(y_yaw_test)], 'k--', lw=2)
plt.xlabel('True Yaw')
plt.ylabel('Predicted Yaw')
plt.title('Yaw: True vs Predicted')

# Roll
plt.subplot(1, 3, 3)
plt.scatter(y_roll_test, y_roll_pred, c='red')
plt.plot([min(y_roll_test), max(y_roll_test)], [min(y_roll_test), max(y_roll_test)], 'k--', lw=2)
plt.xlabel('True Roll')
plt.ylabel('Predicted Roll')
plt.title('Roll: True vs Predicted')

plt.tight_layout()
plt.show()

# Visualizing overfitting with train vs test loss
y_pitch_train_pred = svr_pitch.predict(X_train)
y_yaw_train_pred = svr_yaw.predict(X_train)
y_roll_train_pred = svr_roll.predict(X_train)

mse_pitch_train = mean_squared_error(y_pitch_train, y_pitch_train_pred)
mse_yaw_train = mean_squared_error(y_yaw_train, y_yaw_train_pred)
mse_roll_train = mean_squared_error(y_roll_train, y_roll_train_pred)

print(f'Train Pitch MSE: {mse_pitch_train}')
print(f'Train Yaw MSE: {mse_yaw_train}')
print(f'Train Roll MSE: {mse_roll_train}')

# Plot train vs test loss for each angle
plt.figure(figsize=(15, 5))

# Pitch
plt.subplot(1, 3, 1)
plt.bar(['Train', 'Test'], [mse_pitch_train, mse_pitch], color=['blue', 'orange'])
plt.title('Pitch: Train vs Test MSE')

# Yaw
plt.subplot(1, 3, 2)
plt.bar(['Train', 'Test'], [mse_yaw_train, mse_yaw], color=['blue', 'orange'])
plt.title('Yaw: Train vs Test MSE')

# Roll
plt.subplot(1, 3, 3)
plt.bar(['Train', 'Test'], [mse_roll_train, mse_roll], color=['blue', 'orange'])
plt.title('Roll: Train vs Test MSE')

plt.tight_layout()
plt.show()




joblib.dump(svr_pitch, 'svr_pitch_model.pkl')
joblib.dump(svr_yaw, 'svr_yaw_model.pkl')
joblib.dump(svr_roll, 'svr_roll_model.pkl')
