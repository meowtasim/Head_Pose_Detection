import cv2
import mediapipe as mp
import math

#Getting Face mesh from mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

#Webcam video
cap = cv2.VideoCapture(0)

while True:#Video capture is going on
    ret, image = cap.read()
    if not ret:#Unable to read frame
        break
    results = face_mesh.process(image)
    if results.multi_face_landmarks:#Face is detected
        face_landmarks = results.multi_face_landmarks[0].landmark
        left_ear = [face_landmarks[454].x, face_landmarks[454].y]
        right_ear = [face_landmarks[234].x, face_landmarks[234].y]
        nose_tip = [face_landmarks[4].x, face_landmarks[4].y]
        landmark_10 = [face_landmarks[10].x, face_landmarks[10].y]
        landmark_152 = [face_landmarks[152].x, face_landmarks[152].y]
        dx = landmark_152[0] - landmark_10[0]
        dy = landmark_152[1] - landmark_10[1]
        angle = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle)
        cv2.putText(image, f"Vertical face rotation: {int(angle_degrees)} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        mp_drawing.draw_landmarks(image, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_CONTOURS)
    else:
        cv2.putText(image, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
