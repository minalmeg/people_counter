# people_counter
People counter identifies and counts unique people in a video stream
Prerequisite to run the solution -

NumPy - pip install numpy

OpenCV - pip install opencv-contrib-python

dlib - pip install dlib

imutils - pip install imutils

On cmd run the following command- 
A) To get output in mp4 -

python main2.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4  --output output/output_01.mp4v

B)To get output in avi -
python main2.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4  --output output/output_01.avi

To terminate processing, press Q.

Made with help of https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/
