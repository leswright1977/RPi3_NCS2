# RPi3_NCS2
Intel Neural Compute Stick 2 Running on RPI 3 with Threading

These Scripts allow you to do Object detection on the Raspberry Pi 3B+ using the Intel Neural Compute stick.
Frame rate is around 20 FPS as the blocking operations are removed to a thread.
Detection is a little lower than the framerate, but not by much!

Follow the instructions here to set up and configure your Pi:
https://software.intel.com/en-us/articles/OpenVINO-Install-RaspberryPI

The following scripts are provided:

openvino_fd_myriad.py 
Original single image detection script from: https://software.intel.com/en-us/articles/OpenVINO-Install-RaspberryPI

pi_NCS_USB_cam_test1.py
pi_NCS_USB_cam_test2.py
Naive initial scripts (experiments)

pi_NCS2_USB_cam_threaded_mobilenet.py
Threaded example using the mobilenet-ssd model, converted from the caffe model using the OpenVino toolkit
The models are present in src/models

pi_NCS2_USB_cam_threaded_faces.py
Threaded example using the models from: https://download.01.org/openvinotoolkit/2018_R4/open_model_zoo/
face-detection-retail-0004





