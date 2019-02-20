import cv2
import time
import numpy
from multiprocessing import Process
from multiprocessing import Queue
from picamera.array import PiRGBArray
from picamera import PiCamera

#hacked from:
#https://software.intel.com/articles/OpenVINO-Install-RaspberryPI
#https://opencv2-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
#https://github.com/PINTO0309/MobileNet-SSD-RealSense/blob/master/SingleStickSSDwithUSBCamera_OpenVINO_NCS2.py
#https://raspberrypi.stackexchange.com/questions/87062/overhead-counter

#Les Wright Dec 24 2018 (modified to support picam 30 Dec 2018)
#refined to warp speed (30 fps video, 28 fps inferencing 20 Feb 2019)

#Note cv2.dnn.blobFromImage, the size is present in the XML files, we could write a preamble to go get that data,
#Then we dont have to explicitly set it!


# Load the model
net = cv2.dnn.readNet('models/MobileNetSSD_deploy.xml', 'models/MobileNetSSD_deploy.bin')


# Specify target device
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

#Misc vars
font = cv2.FONT_HERSHEY_SIMPLEX
frameWidth = 304
frameHeight = 304
secPerFrame = 0.0
detections = 0.0

confThreshold = 0.6



#initialize the camera and grab a reference to the raw camera capture
#well this is interesting, we can closely match the input of the network!
#this 'seems' to have improved accuracy!
camera = PiCamera()
camera.resolution = (304,304)
camera.framerate = 35
rawCapture = PiRGBArray(camera, size=(304, 304)) 

# allow the camera to warmup
time.sleep(0.1)


labels_file = 'models/labels.txt'
with open(labels_file, 'r') as f:
	labels = [x.strip() for x in f]
print(labels)


#define the function that handles our processing thread
def classify_frame(net, inputQueue, outputQueue):
# keep looping
	while True:
		# check to see if there is a frame in our input queue
		if not inputQueue.empty():
			# grab the frame from the input queue, resize it, and
			# construct a blob from it
			frame = inputQueue.get()
			frame = cv2.resize(frame, (300, 300))
			blob = cv2.dnn.blobFromImage(frame, 0.007843, size=(300, 300),\
			mean=(127.5,127.5,127.5), swapRB=False, crop=False)
			net.setInput(blob)
			out = net.forward()

			data_out = []

			for detection in out.reshape(-1, 7):
				inference = []
				obj_type = int(detection[1]-1)
				confidence = float(detection[2])
				xmin = detection[3]
				ymin = detection[4]
				xmax = detection[5]
				ymax = detection[6]

				if confidence > 0: #ignore garbage
					inference.extend((obj_type,confidence,xmin,ymin,xmax,ymax))
					data_out.append(inference)

			outputQueue.put(data_out)
			

# initialize the input queue (frames), output queue (out),
# and the list of actual detections returned by the child process
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
out = None

# construct a child process *indepedent* from our main process of
# execution
print("[INFO] starting process...")
p = Process(target=classify_frame, args=(net,inputQueue,outputQueue,))
p.daemon = True
p.start()

print("[INFO] starting capture...")

#time the frame rate....
start = time.time()
frames = 0

for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
	# Capture frame-by-frame
	frame = frame.array

	# if the input queue *is* empty, give the current frame to
	# classify
	if inputQueue.empty():
		inputQueue.put(frame)

	# if the output queue *is not* empty, grab the detections
	if not outputQueue.empty():
		out = outputQueue.get()


	# check to see if 'out' is not empty
	if out is not None:
		# loop over the detections
		for detection in out:
			#print(detection)
			#print("\n")
			
			objID = detection[0]
			confidence = detection[1]

			xmin = int(detection[2] * frame.shape[1])
			ymin = int(detection[3] * frame.shape[0])
			xmax = int(detection[4] * frame.shape[1])
			ymax = int(detection[5] * frame.shape[0])

			if confidence > confThreshold:
				#bounding box
				cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 255))

				#label
				cv2.rectangle(frame, (xmin-1, ymin-1),\
				(xmin+70, ymin-10), (0,255,255), -1)
				#labeltext
				cv2.putText(frame,' '+labels[objID]+' '+str(round(confidence,2)),\
				(xmin,ymin-2), font, 0.3,(0,0,0),1,cv2.LINE_AA)
	
		detections += 1

	# Display the resulting frame
	cv2.putText(frame,'Threshold: '+str(round(confThreshold,1)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 0, 0), 1, cv2.LINE_AA)
	cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('frame',frameWidth,frameHeight)
	cv2.imshow('frame',frame)
	
	

	frames+=1

	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
	
	keyPress = cv2.waitKey(1)

	if keyPress == 113:
        	break

	if keyPress == 82:
		confThreshold += 0.1

	if keyPress == 84:
		confThreshold -= 0.1

	if confThreshold >1:
		confThreshold = 1
	if confThreshold <0:
		confThreshold = 0


end = time.time()
seconds = end-start
fps = frames/seconds
print("Avg Frames Per Sec: "+str(fps))
dts = detections/seconds
print("Avg detections Per Sec: "+str(dts))


cv2.destroyAllWindows()


