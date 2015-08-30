cmake_minimum_required(VERSION 2.8)
project( BS )
find_package( OpenCV REQUIRED )

add_executable( backgroundSub background_sub.cpp )
add_executable( realtimeBS realtimeBS.cpp)
target_link_libraries( backgroundSub ${OpenCV_LIBS} )
target_link_libraries( realtimeBS ${OpenCV_LIBS} )
