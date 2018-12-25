import cv2
import time
import numpy
from multiprocessing import Process
from multiprocessing import Queue

#hacked from:
#https://software.intel.com/articles/OpenVINO-Install-RaspberryPI
#https://opencv2-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
#https://github.com/PINTO0309/MobileNet-SSD-RealSense/blob/master/SingleStickSSDwithUSBCamera_OpenVINO_NCS2.py
#https://raspberrypi.stackexchange.com/questions/87062/overhead-counter

#Les Wright Dec 24 2018

#Note cv2.dnn.blobFromImage, the size is present in the XML files, write a preamble to go get that data,
#Then we dont have to explicitly set it!


# Load the model
net = cv2.dnn.readNet('models/face-detection-retail-0004.xml', 'models/face-detection-retail-0004.bin')

# Specify target device
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

#Misc vars
font = cv2.FONT_HERSHEY_SIMPLEX
frameWidth = 320
frameHeight = 240
secPerFrame = 0.0



#Specify a camera
cap = cv2.VideoCapture(0)

#Get the camera data:
def capProperties():
	print("[info] W, H, FPS")
	print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	print(cap.get(cv2.CAP_PROP_FPS))

capProperties()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)


#time the frame rate....
start = time.time()
frames = 0

def classify_frame(net, inputQueue, outputQueue):
# keep looping
	while True:
		# check to see if there is a frame in our input queue
		if not inputQueue.empty():
			# grab the frame from the input queue, resize it, and
			# construct a blob from it
			frame = inputQueue.get()
			frame = cv2.resize(frame, (300, 300))

			blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U) 

			net.setInput(blob)
			out = net.forward()
			# write the detections to the output queue
			outputQueue.put(out)

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

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

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
		# Draw detections on the frame
		for detection in out.reshape(-1, 7):

			confidence = float(detection[2])

			xmin = int(detection[3] * frame.shape[1])
			ymin = int(detection[4] * frame.shape[0])
			xmax = int(detection[5] * frame.shape[1])
			ymax = int(detection[6] * frame.shape[0])


			if confidence > 0.5:
				#bounding box
				cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 255))

				#label
				cv2.rectangle(frame, (xmin-1, ymin-1),\
				(xmin+60, ymin-10), (0,255,255), -1)
				#labeltext
				cv2.putText(frame,'Face: '+str(round(confidence,2)),\
				(xmin,ymin-2), font, 0.3,(0,0,0),1,cv2.LINE_AA)


	# Display the resulting frame
	cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('frame',frameWidth,frameHeight)
	cv2.imshow('frame',frame)
	

	frames+=1


	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break


end = time.time()
seconds = end-start
fps = frames/seconds
print("Avg Frames Per Sec: "+str(fps))



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


