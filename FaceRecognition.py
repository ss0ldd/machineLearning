import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

YOUR_NAME = "Julia"
YOUR_SURNAME = "Yaldynova"

# Emotion labels
EMOTIONS = ["Angry", "Happy", "Sad", "Neutral"]

def count_raised_fingers(hand_landmarks):
    if not hand_landmarks:
        return 0

    finger_tips = [8, 12, 16, 20]
    finger_joints = [6, 10, 14, 18]
    
    raised_fingers = 0

    for tip, joint in zip(finger_tips, finger_joints):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[joint].y:
            raised_fingers += 1

    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        raised_fingers += 1
    
    return raised_fingers

def detect_emotion(face_mesh_results):
    if not face_mesh_results.multi_face_landmarks:
        return "Neutral"
    
    face_landmarks = face_mesh_results.multi_face_landmarks[0].landmark
    

    left_mouth = face_landmarks[61]
    right_mouth = face_landmarks[291]

    upper_lip = face_landmarks[13]
    lower_lip = face_landmarks[14]

    left_eyebrow = face_landmarks[70]
    right_eyebrow = face_landmarks[300]

    mouth_opening = abs(upper_lip.y - lower_lip.y)

    mouth_width = abs(left_mouth.x - right_mouth.x)

    eyebrow_height = (left_eyebrow.y + right_eyebrow.y) / 2

    if mouth_opening > 0.03 and mouth_width > 0.3:
        return "Happy"
    elif eyebrow_height < 0.1:
        return "Angry"
    elif mouth_opening < 0.01 and mouth_width < 0.2:
        return "Sad"
    else:
        return "Neutral"

def main():
    cap = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
         mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to read from webcam")
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face_results = face_detection.process(image_rgb)

            face_mesh_results = face_mesh.process(image_rgb)

            hand_results = hands.process(image_rgb)

            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            raised_fingers = 0
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    raised_fingers = count_raised_fingers(hand_landmarks)
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if face_results.detections:
                for detection in face_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, c = image.shape
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    width, height = int(bbox.width * w), int(bbox.height * h)

                    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

                    if raised_fingers == 1:
                        text = YOUR_NAME
                    elif raised_fingers == 2:
                        text = YOUR_SURNAME
                    elif raised_fingers == 3:
                        text = detect_emotion(face_mesh_results)
                    else:
                        text = "Unknown"

                    cv2.putText(image, text, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('Face and Hand Detection', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 