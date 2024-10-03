import openai
openai.api_key='|||||key'
import cv2

def monitor_workshop_feed():
    cap = cv2.VideoCapture('camera_ip')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Display the video frame
        cv2.imshow('Workshop Feed', frame)
        # Add motion detection logic here
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        # More logic to detect changes or specific objects (like safety gear)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
monitor_workshop_feed()

def analyze_behavior(behavior_description):
    prompt = f"Determine if the following behavior is hazardous or violates workshop rules: {behavior_description}"
    
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()
description_1 = "Worker detected in workshop without wearing a mask."
description_2 = "Unknown person entered the restricted secret area."

# Analyze behavior using OpenAI
result_1 = analyze_behavior(description_1)
result_2 = analyze_behavior(description_2)

print(result_1)  
print(result_2)  

mask_model = load_model('mask_detector.model')
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Add face recognition encoding for authorized personnel here...

def detect_behavior(frame):
    # Perform mask detection and face recognition
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))
        face_array = img_to_array(face_resized)
        face_array = np.expand_dims(face_array, axis=0) / 255.0

        (mask, no_mask) = mask_model.predict(face_array)[0]
        if mask < no_mask:
            # Convert to behavior description for OpenAI API
            behavior_description = "A worker is detected in the workshop without wearing a mask."
            response = analyze_behavior(behavior_description)
            return response

    return None

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			
		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


def analyze_behavior(behavior_description):
    prompt = f"Determine if the following behavior is hazardous or violates workshop rules: {behavior_description}"
    
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=100
    )
    
    return response.choices[0].text.strip()

# Example behavior descriptions generated from computer vision
description_1 = "Worker detected in workshop without wearing a mask."
description_2 = "Unknown person entered the restricted secret area."

# Analyze behavior using OpenAI
result_1 = analyze_behavior(description_1)
result_2 = analyze_behavior(description_2)

print(result_1)  # Output: Provides an evaluation of whether this is a safety violation
print(result_2)  # Output: Provides evaluation on whether action needs to be tak






import cv2
import openai

openai.api_key = 'your-openai-api-key'

# Load pre-trained models (face detection, mask detection, face recognition, etc.)
mask_model = load_model('mask_detector.model')
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Add face recognition encoding for authorized personnel here...

def detect_behavior(frame):
    # Perform mask detection and face recognition
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))
        face_array = img_to_array(face_resized)
        face_array = np.expand_dims(face_array, axis=0) / 255.0

        (mask, no_mask) = mask_model.predict(face_array)[0]
        if mask < no_mask:
            # Convert to behavior description for OpenAI API
            behavior_description = "A worker is detected in the workshop without wearing a mask."
            response = analyze_behavior(behavior_description)
            return response

    return None

def analyze_behavior(behavior_description):
    # Send behavior description to OpenAI API
    prompt = f"Determine if the following behavior is hazardous or violates workshop rules: {behavior_description}"
    
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=100
    )
    
    return response.choices[0].text.strip()

# Monitoring the workshop
def monitor_workshop():
    cap = cv2.VideoCapture(0)  # Capture video feed from camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect behaviors
        result = detect_behavior(frame)

        if result:
            print("Analysis result from OpenAI:", result)  # Alert based on OpenAI's analysis
        
        cv2.imshow('Workshop Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

monitor_workshop()

from twilio.rest import Client

def send_alert(alert_message, phone_number):
    client = Client("twilio-sid", "twilio-auth-token")
    message = client.messages.create(
        body=alert_message,
        from_="+1234567890",  # Twilio number
        to=phone_number
    )
    return message.sid

alert_message = "High gas levels detected in Workshop Zone A. Immediate action required."
send_alert(alert_message, "+1987654321")

from googleapiclient.discovery import build

def generate_report(report_content):
    service = build('docs', 'v1', credentials=credentials)
    
    # Create a new document for the incident report
    document = service.documents().create(body={'title': 'Incident Report'}).execute()
    doc_id = document.get('documentId')

    # Add the report content to the document
    service.documents().batchUpdate(documentId=doc_id, body={
        'requests': [
            {'insertText': {'location': {'index': 1}, 'text': report_content}}
        ]
    }).execute()

report_content = "Hazard: High gas levels detected in Zone A at 2:00 PM. Immediate action taken."
generate_report(report_content)