import cv2 as cv
import time

#hacked from:
#https://software.intel.com/articles/OpenVINO-Install-RaspberryPI
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
#Les Wright Dec 20 2018
#Note cv.dnn.blobFromImage, the size is present in the XML files, write a preamble to go get that data,
#Then we dont have to explicitly set it!


# Load the model
#net = cv.dnn.readNet('models/face-detection-adas-0001.xml', 'models/face-detection-adas-0001.bin')
net = cv.dnn.readNet('models/face-detection-retail-0004.xml', 'models/face-detection-retail-0004.bin')


# Specify target device
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

#Specify a camera
cap = cv.VideoCapture(0)

#Get the camera data:
def capProperties():
	print("[info] W, H, FPS")
	print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
	print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
	print(cap.get(cv.CAP_PROP_FPS))

capProperties()
cap.set(3,320)
cap.set(4,240)



#time the frame rate....
start = time.time()
frames = 0

network_width = 160
network_height = 120
def preprocess(source_cap):
	resized_cap = cv.resize(source_cap,(network_width,network_height))
	return resized_cap


while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	#preprocess
	resized = preprocess(frame)

	# Prepare input blob and perform an inference
	blob = cv.dnn.blobFromImage(resized, size=(300, 300), ddepth=cv.CV_8U) 


	net.setInput(blob)
	out = net.forward()


	# Draw detected faces on the frame
	for detection in out.reshape(-1, 7):
		confidence = float(detection[2])
		xmin = int(detection[3] * resized.shape[1])
		ymin = int(detection[4] * resized.shape[0])
		xmax = int(detection[5] * resized.shape[1])
		ymax = int(detection[6] * resized.shape[0])


		# Our operations on the frame come here
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

		# Display the resulting frame
		#cv.namedWindow('frame',cv.WINDOW_NORMAL)
		#cv.resizeWindow('frame',320,240)
		cv.imshow('frame',gray)

		if confidence > 0.5:
			cv.rectangle(frame, (xmin*2, ymin*2), (xmax*2, ymax*2), color=(0, 255, 255))
			#font = cv.FONT_HERSHEY_PLAIN
			#cv.putText(frame,'Face: '+str(round(confidence,2)),(xmin,ymin), font, 1,(0,255,255),2,cv.LINE_AA)


	frames+=1


	if cv.waitKey(1) & 0xFF == ord('q'):
        	break


end = time.time()
seconds = end-start
fps = frames/seconds
print("Frames Per Sec: "+str(fps))


# When everything done, release the capture
cap.release()
cv.destroyAllWindows()


