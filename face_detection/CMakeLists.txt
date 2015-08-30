cmake_minimum_required(VERSION 2.8)
project( FT )
find_package( OpenCV REQUIRED )

add_executable( faceTracking face_tracking.cpp )
target_link_libraries( faceTracking ${OpenCV_LIBS} )

