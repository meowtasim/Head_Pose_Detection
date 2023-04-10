import cv2
import mediapipe as mp
import math

# Getting Face mesh from mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Webcam video
cap = cv2.VideoCapture(0)

while True:  # Video capture is going on
    ret, image = cap.read()
    if not ret:  # Unable to read frame
        break
    results = face_mesh.process(image)
    if results.multi_face_landmarks:  # Face is detected
        face_landmarks = results.multi_face_landmarks[0].landmark
        forehead = [face_landmarks[10].x, face_landmarks[10].y]
        chin = [face_landmarks[152].x, face_landmarks[152].y]
        nose_tip = [face_landmarks[4].x, face_landmarks[4].y]
        dy = nose_tip[1] - (forehead[1] + chin[1])/2
        angle = math.atan2(dy, 1.0)  # Calculate the angle using atan2 function
        angle_degrees = math.degrees(angle)
        mapped_value = ((angle_degrees + 3) / 6) * 180 - 90  # Map angle_degrees from -3 to +3 to -90 to +90
        cv2.putText(image, f"Head rotation amount: {mapped_value:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        mp_drawing.draw_landmarks(image, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_CONTOURS)
    else:
        cv2.putText(image, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
