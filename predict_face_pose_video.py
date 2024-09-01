import os
import cv2
import numpy as np
import dlib
import joblib
import mediapipe as mp
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to detect facial landmarks from an image
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    landmarks_list = []
    if len(rects) > 0:
        for rect in rects[:5]:  # Limit to the first 5 faces
            shape = predictor(gray, rect)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            landmarks_list.append(landmarks.flatten())  # Flatten the landmarks into a 1D array
        return landmarks_list
    else:
        return []

# Load your pre-trained SVM models
svr_pitch = joblib.load('svr_pitch_model.pkl')
svr_yaw = joblib.load('svr_yaw_model.pkl')
svr_roll = joblib.load('svr_roll_model.pkl')

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, min_detection_confidence=0.2)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Set up tkinter root (hidden)
Tk().withdraw()

# Open file dialog to select a video
video_path = askopenfilename(title="Select a Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
if not video_path:
    print("No file selected.")
    exit()

# Read and process the selected video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
output_video_path = os.path.splitext(video_path)[0] + "_output.avi"
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get landmarks using dlib and predict angles using SVM models
    landmarks_list = get_landmarks(frame)
    if not landmarks_list:
        continue

    for landmarks in landmarks_list:
        pitch_pred = svr_pitch.predict([landmarks])[0]
        yaw_pred = svr_yaw.predict([landmarks])[0]
        roll_pred = svr_roll.predict([landmarks])[0]

        # Use MediaPipe to get face landmarks and calculate pose direction
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True
        img_h, img_w, img_c = frame.shape

        if results.multi_face_landmarks:
            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                face_3d = []
                face_2d = []
                
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (landmark.x * img_w, landmark.y * img_h)
                            
                        x, y = int(landmark.x * img_w), int(landmark.y * img_h)
                        
                        # 2D coordinates
                        face_2d.append([x, y])
                        
                        # 3D coordinates
                        face_3d.append([x, y, landmark.z])
                
                # Convert lists to numpy arrays
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                
                # Rotation degree (predicted by SVM)
                x = pitch_pred * 180 / np.pi  # Convert pitch to degrees
                y = -yaw_pred * 180 / np.pi    # Convert yaw to degrees
                z = roll_pred * 180 / np.pi   # Convert roll to degrees
                
                # Nose direction
                length = 100
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + length * np.sin(y * np.pi / 180)), 
                      int(nose_2d[1] - length * np.sin(x * np.pi / 180))) 
                
                # Draw axis lines
                cv2.line(frame, p1, p2, (255, 0, 0), 3)  # Blue line for nose direction
                cv2.line(frame, p1, (p1[0], p1[1] + 100), (0, 255, 0), 3)  # Green line
                cv2.line(frame, p1, (p1[0] + 100, p1[1]), (0, 0, 255), 3)  # Red line
                
                # Text on image with smaller font size
                font_scale = 0.4
                thickness = 1
                #cv2.putText(frame, f"SVM Pitch: {pitch_pred:.2f}, Yaw: {yaw_pred:.2f}, Roll: {roll_pred:.2f}", 
                            #(50, 50 + face_idx * 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

    # Write the processed frame to the output video
    out.write(frame)

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved as {output_video_path}")
