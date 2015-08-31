cmake_minimum_required(VERSION 2.8)
project( cap )
find_package( OpenCV REQUIRED )

add_executable( capture capture.cpp )
target_link_libraries( capture ${OpenCV_LIBS} )

