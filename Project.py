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

        # First getting the various feature points for calculation  
        face_landmarks = results.multi_face_landmarks[0].landmark
        left_ear = [face_landmarks[454].x, face_landmarks[454].y]
        right_ear = [face_landmarks[234].x, face_landmarks[234].y]
        nose_tip = [face_landmarks[4].x, face_landmarks[4].y]
        forehead = [face_landmarks[10].x, face_landmarks[10].y]
        chin = [face_landmarks[152].x, face_landmarks[152].y]

        # Head Tilt
        vx = chin[0] - forehead[0]
        vy = chin[1] - forehead[1]
        angle = math.atan2(vy, vx)#gets the angle of vector x,y with respect to x axis
        tilt_angle = math.degrees(angle)
        cv2.putText(image, f"Inclined face rotation: {int(tilt_angle)} degrees", (
            10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Horizontal Rotation
        hx = nose_tip[0] - (left_ear[0] + right_ear[0])/2
        angle = math.atan2(hx,1.0)# Calculate the angle using atan2 function
        angle_degrees = math.degrees(angle)
        mapped_value=round(angle_degrees*10)
        cv2.putText(image, f"Horizontal Head rotation: {mapped_value} degrees", (
            10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Vertical Rotation
        dy = nose_tip[1] - (forehead[1] + chin[1])/2
        angle = math.atan2(dy, 1.0)  # Calculate the angle using atan2 function
        angle_degrees = math.degrees(angle)
        # Map angle_degrees from -3 to +3 to -90 to +90 and round off to whole numbers
        mapped_value = round(((angle_degrees + 3) / 6) * 180 - 90)
        cv2.putText(image, f"Vertical Head rotation: {mapped_value} degrees", (
            10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw the mesh on face
        mp_drawing.draw_landmarks(
            image, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_CONTOURS)
    else:  # No face in current frame
        cv2.putText(image, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Mediapipe Face Mesh', image)
    if cv2.waitKey(1) == 27:  # When esc key is pressed, break out of while loop
        break
cap.release()  # releasing webcam resource
cv2.destroyAllWindows()  # closing the display window
