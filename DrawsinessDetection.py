import cv2
import dlib
import pyttsx3
from scipy.spatial import distance
import threading
import queue
import numpy as np

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


# Load the pre-trained MobileNet SSD model and the prototxt file
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# Define the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]


# SETTING UP OF CAMERA TO 1 YOU CAN
# EVEN CHOOSE 0 IN PLACE OF 1
cap = cv2.VideoCapture(1)  # Use 0 instead of 1 for most built-in webcams
countDrowsiness = 0
alert_frames = 0  # Counter to keep track of frames to display the alert

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

# MAIN LOOP IT WILL RUN ALL THE UNLESS AND 
# UNTIL THE PROGRAM IS BEING KILLED BY THE USER
while True:
    null, frame = cap.read()
    frame = cv2.flip(frame,1)
    # Get the frame dimensions
    (h, w) = frame.shape[:2]
    # Preprocess the frame: resize to 300x300 pixels and normalize it
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    # Pass the blob through the network to get the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        # Extract the confidence (probability) of the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than a threshold
        if confidence > 0.2:
            # Extract the index of the class label from the detection
            idx = int(detections[0, 0, i, 1])

            # If the detected object is a person, proceed with drawing the bounding box
            if CLASSES[idx] == "person":
                # Compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box around the detected person
                label = f"{CLASSES[idx]}: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_detector(gray_scale)
                print("length of faces : ", len(faces))
                if(len(faces) > 0):
                    for face in faces:
                        # Draw rectangle around the face
                        x, y, w, h = face.left(), face.top(), face.width(), face.height()
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        face_landmarks = dlib_facelandmark(gray_scale, face)
                        leftEye = []
                        rightEye = []

                        # THESE ARE THE POINTS ALLOCATION FOR THE 
                        # LEFT EYES IN .DAT FILE THAT ARE FROM 42 TO 47
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

                        # THESE ARE THE POINTS ALLOCATION FOR THE 
                        # RIGHT EYES IN .DAT FILE THAT ARE FROM 36 TO 41
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

                        # CALCULATING THE ASPECT RATIO FOR LEFT 
                        # AND RIGHT EYE
                        right_Eye = Detect_Eye(rightEye)
                        left_Eye = Detect_Eye(leftEye)
                        Eye_Rat = (left_Eye + right_Eye) / 2

                        # NOW ROUND OF THE VALUE OF AVERAGE MEAN 
                        # OF RIGHT AND LEFT EYES
                        Eye_Rat = round(Eye_Rat, 2)

                        # THIS VALUE OF 0.25 (YOU CAN EVEN CHANGE IT) 
                        # WILL DECIDE WHETHER THE PERSON'S EYES ARE CLOSE OR NOT
                        if Eye_Rat < 0.25:
                            countDrowsiness = countDrowsiness + 1
                            print("Count : ", countDrowsiness)
                            if countDrowsiness > 20:
                                alert_frames = 100  # Display the alert message for the next 100 frames
                                cv2.putText(frame, "DROWSINESS DETECTED", (50, 450),
                                            cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)

                                # ENQUEUE THE ALERT MESSAGE
                                if speech_queue.empty():
                                    speech_queue.put("Alert!!!! Drowsiness Detected")

                                countDrowsiness = 0
                        else:
                            countDrowsiness = 0

                    if alert_frames > 0:
                        alert_frames -= 1
                        cv2.putText(frame, "Alert!!!! Drowsiness Detected", (50, 450),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
                else :
                    countDrowsiness = countDrowsiness + 1
                    if(countDrowsiness > 20):
                        alert_frames = 100  # Display the alert message for the next 100 frames
                        cv2.putText(frame, "Alert!!!! PAY ATTENTION", (50, 450),cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)

                        # ENQUEUE THE ALERT MESSAGE
                        if speech_queue.empty():
                            speech_queue.put("Alert!!!! PAY ATTENTION")

                        countDrowsiness = 0
                
                    if alert_frames > 0:
                            alert_frames -= 1
                            cv2.putText(frame, "Alert!!!! PAY ATTENTION", (50, 450),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)


    cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# SIGNALING THE TTS WORKER THREAD TO EXIT
speech_queue.put(None)
tts_thread.join()
