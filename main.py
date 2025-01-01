import cv2
import dlib
import pyttsx3
from scipy.spatial import distance
import threading
import queue
import numpy as np
import serial

# INITIALIZING THE pyttsx3 SO THAT 
# ALERT AUDIO MESSAGE CAN BE DELIVERED
engine = pyttsx3.init()

# CREATING A QUEUE FOR TEXT-TO-SPEECH MESSAGES
speech_queue = queue.Queue()

# FUNCTION TO HANDLE TEXT-TO-SPEECH REQUESTS
def tts_worker():
    while True:
        message = speech_queue.get()
        if message is None:
            break
        engine.say(message)
        engine.runAndWait()
        speech_queue.task_done()

# STARTING THE TTS WORKER THREAD
tts_thread = threading.Thread(target=tts_worker)
tts_thread.daemon = True
tts_thread.start()

# Setup serial communication
ser = serial.Serial('COM13', 9600, timeout=1)  # Replace 'COM6' with your actual port

# Load the pre-trained MobileNet SSD model and the prototxt file
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# Define the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# SETTING UP OF CAMERA TO 1 YOU CAN
# EVEN CHOOSE 0 IN PLACE OF 1
cap = cv2.VideoCapture(0)  # Use 0 instead of 1 for most built-in webcams
countDrowsiness = 0
alert_frames = 0  # Counter to keep track of frames to display the alert
count = 0

# FACE DETECTION OR MAPPING THE FACE TO
# GET THE Eye AND EYES DETECTED
face_detector = dlib.get_frontal_face_detector()

# PUT THE LOCATION OF .DAT FILE (FILE FOR
# PREDICTING THE LANDMARKS ON FACE )
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# FUNCTION CALCULATING THE ASPECT RATIO FOR
# THE Eye BY USING EUCLIDEAN DISTANCE FUNCTION
def Detect_Eye(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_Eye = (poi_A + poi_B) / (2 * poi_C)
    return aspect_ratio_Eye

# FUNCTION CALCULATING THE ASPECT RATIO FOR
# THE Mouth BY USING EUCLIDEAN DISTANCE FUNCTION
def Detect_Mouth(mouth):
    poi_A = distance.euclidean(mouth[2], mouth[10])  # 51 to 59
    poi_B = distance.euclidean(mouth[4], mouth[8])   # 53 to 57
    poi_C = distance.euclidean(mouth[0], mouth[6])   # 49 to 55
    aspect_ratio_Mouth = (poi_A + poi_B) / (2 * poi_C)
    return aspect_ratio_Mouth

# MAIN LOOP IT WILL RUN ALL THE UNLESS AND 
# UNTIL THE PROGRAM IS BEING KILLED BY THE USER
while True:
    null, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # print(frame.shape)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Variable to hold the serial message
    serial_message = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = f"{CLASSES[idx]}: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector(gray_scale)
                if len(faces) > 0:
                    for face in faces:
                        x, y, w, h = face.left(), face.top(), face.width(), face.height()
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        face_landmarks = dlib_facelandmark(gray_scale, face)
                        leftEye = []
                        rightEye = []
                        mouth = []

                        for n in range(42, 48):
                            x = face_landmarks.part(n).x
                            y = face_landmarks.part(n).y
                            rightEye.append((x, y))
                            next_point = n + 1
                            if n == 47:
                                next_point = 42
                            x2 = face_landmarks.part(next_point).x
                            y2 = face_landmarks.part(next_point).y
                            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

                        for n in range(36, 42):
                            x = face_landmarks.part(n).x
                            y = face_landmarks.part(n).y
                            leftEye.append((x, y))
                            next_point = n + 1
                            if n == 41:
                                next_point = 36
                            x2 = face_landmarks.part(next_point).x
                            y2 = face_landmarks.part(next_point).y
                            cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

                        for n in range(48, 60):
                            x = face_landmarks.part(n).x
                            y = face_landmarks.part(n).y
                            mouth.append((x, y))
                            next_point = n + 1
                            if n == 59:
                                next_point = 48
                            x2 = face_landmarks.part(next_point).x
                            y2 = face_landmarks.part(next_point).y
                            cv2.line(frame, (x, y), (x2, y2), (0, 0, 255), 1)

                        right_Eye = Detect_Eye(rightEye)
                        left_Eye = Detect_Eye(leftEye)
                        Eye_Rat = (left_Eye + right_Eye) / 2
                        Eye_Rat = round(Eye_Rat, 2)

                        mouth_Rat = Detect_Mouth(mouth)
                        mouth_Rat = round(mouth_Rat, 2)

                        print("Mouth Ratio is ", mouth_Rat)

                        if Eye_Rat < 0.25:
                            countDrowsiness += 1
                            if countDrowsiness > 20:
                                alert_frames = 100
                                cv2.putText(frame, "Alert!!!! DROWSINESS DETECTED", (50, 450),
                                            cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
                                if speech_queue.empty():
                                    speech_queue.put("Alert!!!! DROWSINESS DETECTED")
                                countDrowsiness = 0
                                serial_message = 2

                            if alert_frames > 0:
                                alert_frames -= 1
                                cv2.putText(frame, "Alert!!!! DROWSINESS DETECTED", (50, 450),
                                            cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
                                serial_message = 2


                        elif (mouth_Rat > 0.55):
                            alert_frames = 100
                            cv2.putText(frame, "Alert!!!! YAWNING DETECTED", (50, 450),
                            cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
                            if speech_queue.empty():
                                speech_queue.put("Alert!!!! YAWNING DETECTED")
                            countDrowsiness = 0
                            serial_message = 3

                            if alert_frames > 0:
                                alert_frames -= 1
                                cv2.putText(frame, "Alert!!!! YAWNING DETECTED", (50, 450),
                                            cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
                                serial_message = 3
                        else:
                            countDrowsiness = 0

                    # if alert_frames > 0:
                    #     alert_frames -= 1
                    #     cv2.putText(frame, "Alert!!!! Drowsiness Detected", (50, 450),
                    #                 cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
                    #     serial_message = 1
                else:
                    countDrowsiness += 1
                    if countDrowsiness > 20:
                        alert_frames = 100
                        cv2.putText(frame, "Alert!!!! PAY ATTENTION", (50, 450),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
                        if speech_queue.empty():
                            speech_queue.put("Alert!!!! PAY ATTENTION")
                        countDrowsiness = 0
                        serial_message = 2

                    if alert_frames > 0:
                        alert_frames -= 1
                        cv2.putText(frame, "Alert!!!! PAY ATTENTION", (50, 450),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
                        serial_message = 2

    count += 1
    if(count > 60):
        if(serial_message == 0):
            ser.write(b'0')
            count = 0
            print("Sent ", serial_message)
        elif(serial_message == 1):
            ser.write(b'1')
            count = 0
            print("Sent ", serial_message)
        elif(serial_message == 2):
            ser.write(b'2')
            count = 0
            print("Sent ", serial_message)
        elif(serial_message == 3):
            ser.write(b'3')
            count = 0
            print("Sent ", serial_message)

    cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# SIGNALING THE TTS WORKER THREAD TO EXIT
speech_queue.put(None)
tts_thread.join()
ser.close()
