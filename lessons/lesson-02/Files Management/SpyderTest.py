# import the necessary packages
from imutils.video import VideoStream
from collections import deque
from threading import Thread
import argparse
import datetime
import imutils
import time
import cv2
import sys
import numpy as np
import yagmail
is_py2 = sys.version[0] == '2'
if is_py2:
    import Queue
else:
    from queue import Queue

minArea = 5000
firstFrame = None
text = "Unoccupied"
recordingNumber = 0
yag = yagmail.SMTP('abbas.chokor','Colorado2017')
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-f", "--fps", type=int, default=20,
	help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
	help="codec of output video")
ap.add_argument("-b", "--buffer-size", type=int, default=32,
	help="buffer size of video clip writer")
args = vars(ap.parse_args())

# Write class of functions
class KeyClipWriter:
	def __init__(self, bufSize=64, timeout=0.5):
		# store the maximum buffer size of frames to be kept
		# in memory along with the sleep timeout during threading
		self.bufSize = bufSize
		self.timeout = timeout

		# initialize the buffer of frames, queue of frames that
		# and boolean indicating whether recording has started or not
		self.frames = deque(maxlen=bufSize)
		self.Q = None
		self.writer = None
		self.thread = None
		self.recording = False

	def update(self, frame):
		# update the frames buffer
		self.frames.appendleft(frame)

		# if we are recording, update the queue as well
		if self.recording:
			self.Q.put(frame)

	def start(self, outputPath, fourcc, fps):
         self.recording = True
         self.writer = cv2.VideoWriter(outputPath, fourcc, fps,(self.frames[0].shape[1], self.frames[0].shape[0]), True)
         self.Q = Queue()
         for i in range(len(self.frames), 0, -1):
             self.Q.put(self.frames[i - 1])
         self.thread = Thread(target=self.write, args=())
         self.thread.daemon = True
         self.thread.start()

	def write(self):
		# keep looping
		while True:
			# if we are done recording, exit the thread
			if not self.recording:
				return

			# check to see if there are entries in the queue
			if not self.Q.empty():
				# grab the next frame in the queue and write it
				# to the video file
				frame = self.Q.get()
				self.writer.write(frame)

			# otherwise, the queue is empty, so sleep for a bit
			# so we don't waste CPU cycles
			else:
				time.sleep(self.timeout)

	def flush(self):
		# empty the queue by flushing all remaining frames to file
		while not self.Q.empty():
			frame = self.Q.get()
			self.writer.write(frame)

	def finish(self):
		# indicate that we are done recording, join the thread,
		# flush all remaining frames in the queue to file, and
		# release the writer pointer
		self.recording = False
		self.thread.join()
		self.flush()
		self.writer.release()

class BasicMotionDetector:
	def __init__(self, accumWeight=0.5, deltaThresh=5, minArea=5000):
		# determine the OpenCV version, followed by storing the
		# the frame accumulation weight, the fixed threshold for
		# the delta image, and finally the minimum area required
		# for "motion" to be reported
		self.isv2 = imutils.is_cv2()
		self.accumWeight = accumWeight
		self.deltaThresh = deltaThresh
		self.minArea = minArea

		# initialize the average image for motion detection
		self.avg = None

	def update(self, image):
		# initialize the list of locations containing motion
		locs = []

		# if the average image is None, initialize it
		if self.avg is None:
			self.avg = image.astype("float")
			return locs

		# otherwise, accumulate the weighted average between
		# the current frame and the previous frames, then compute
		# the pixel-wise differences between the current frame
		# and running average
		cv2.accumulateWeighted(image, self.avg, self.accumWeight)
		frameDelta = cv2.absdiff(image, cv2.convertScaleAbs(self.avg))

		# threshold the delta image and apply a series of dilations
		# to help fill in holes
		thresh = cv2.threshold(frameDelta, self.deltaThresh, 255,
			cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, None, iterations=2)

		# find contours in the thresholded image, taking care to
		# use the appropriate version of OpenCV
		cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if self.isv2 else cnts[1]

		# loop over the contours
		for c in cnts:
			# only add the contour to the locations list if it
			# exceeds the minimum area
			if cv2.contourArea(c) > self.minArea:
				locs.append(c)

		# return the set of locations
		return locs

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# initialize key clip writer and the consecutive number of frames that have *not* contained any action
kcw = KeyClipWriter(bufSize=args["buffer_size"])
consecFrames = 0

motion = BasicMotionDetector(minArea=minArea)
# function to read data from video stream

# keep looping
while True:
	# grab the current frame, resize it, and initialize a
	# boolean used to indicate if the consecutive frames
	# counter should be updated
    frame = vs.read()
    frame = imutils.resize(frame, width=680)
    updateConsecFrames = True

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    locs = motion.update(gray)
    # only process the panorama for motion if a nice average has
	# been built up
    if len(locs) > 0:
		# initialize the minimum and maximum (x, y)-coordinates,
		# respectively
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)

		# loop over the locations of motion and accumulate the
		# minimum and maximum locations of the bounding boxes
        for l in locs:
            (x, y, w, h) = cv2.boundingRect(l)
            (minX, maxX) = (min(minX, x), max(maxX, x + w))
            (minY, maxY) = (min(minY, y), max(maxY, y + h))

		# draw the bounding box
        cv2.rectangle(frame, (minX, minY), (maxX, maxY),(0, 0, 255), 2)

        text = "Occupied"
        consecFrames = 0
        if not kcw.recording:
            timestamp = datetime.datetime.now()
            p = "{}/{}.avi".format(args["output"],timestamp.strftime("%Y%m%d-%H%M%S"))
            kcw.start(p, cv2.VideoWriter_fourcc(*args["codec"]),args["fps"])

    else:
        text = "Unoccupied"
        consecFrames += 1

    kcw.update(frame)
    print ("text", text)
    print ("consecframe", consecFrames)
    print ("shapes", frame.shape)
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
	# if we are recording and reached a threshold on consecutive
	# number of frames with no action, stop recording the clip
    if kcw.recording and consecFrames == args["buffer_size"]:
        kcw.finish()
        yag.send("abbas.chokor@seagate.com", 'Motion Detection', 'A motion was detected in your cleanroom.')
        recordingNumber += 1
        print ('recordingNumber = ',recordingNumber)

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# if we are in the middle of recording a clip, wrap it up
if kcw.recording:
	kcw.finish()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
