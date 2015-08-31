cmake_minimum_required(VERSION 2.8)
project( FD_opencv )
find_package( OpenCV REQUIRED )

add_executable( faceDetection_OpenCVExample face_detection_opencv.cpp )
target_link_libraries( faceDetection_OpenCVExample ${OpenCV_LIBS} )

