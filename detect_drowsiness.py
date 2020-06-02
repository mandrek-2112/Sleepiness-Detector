# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import datetime
import dlib
import sys
import cv2

def sound_alarm(path):
	# play an alarm sound
	# print(path)
	playsound.playsound(path)

class myThread (Thread):
   def __init__(self, threadID, name, counter):
      Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
   def run(self):
      sound_alarm(args["alarm"])

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-s", "--EAR-sensi", type=float, required=True,
	help="value of EAR to determine sensitivity (recommended =0.25)")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = args["EAR_sensi"]
EYE_AR_CONSEC_FRAMES = 40

# initialize the frame counter, time list as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False
timeDrowsiness = []

# initialize dlib's face detector (HOG-based) and then create
# the eye landmark predictor
print("[RESULT] Getting eye predictor model.")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# print(face_utils.FACIAL_LANDMARKS_IDXS['left_eye'])
# print(face_utils.FACIAL_LANDMARKS_IDXS['right_eye'])
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
# (lStart, lEnd) = (0, 6)
# face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (rStart, rEnd) = (6, 12)
# face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# define indices for eye model
lStart = 0
lEnd = 6
rStart = 6
rEnd = 12

# start the video stream thread
print("[RESULT] Starting webcam stream.")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the eye region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# print(shape)

		# extract the left and right eye coordinates, then use the
		# resultant coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		# print(ear)

		# compute the convex hull for the left and right eye, then
		# display it for each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)

		thread1 = myThread(1, "Thread-1", 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on and note down
				# the time

				if not ALARM_ON:
					ALARM_ON = True
					tD = str(datetime.datetime.now())
					timeDrowsiness.append(tD)

					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
				if args["alarm"] != "":
					# t = Thread(target=sound_alarm,
						# args=(args["alarm"],))
					# t.deamon = True
					# t.start()
					# thread1 = myThread(1, "Thread-1", 1)
					thread1.start()
					# thread1.join()

				# draw an alarm on the frame
				cv2.putText(frame, "SLEEPINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (141,36, 80), 2)
				if thread1.is_alive():
					thread1.join()

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			ALARM_ON = False

		# draw the computed eye aspect ratio on the frame
		cv2.putText(frame, "EAR Value: {:.2f}".format(ear), (250, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		print("[RESULT] You were detected drowsy at: ")
		for val, items in enumerate(timeDrowsiness):
			print("{}. {}".format(val+1, items))
		print("\nEXITING.")
		break
'''
print("You were detected drowsy at: ")
for val, items in enumerate(timeDrowsiness):
	print("{}. {}".format(val, items))
'''

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()