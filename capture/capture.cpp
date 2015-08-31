#include <opencv2/core/core.hpp>   
#include "opencv2/highgui.hpp"

using namespace cv;

int main() {

  VideoCapture cap(0);      // Opens default camera
  Mat frame;                // Open a Mat
  
  for(;;) {

    cap >> frame;           // Grab frame

    if(waitKey(30) >= 0)    // Pause key
      break;

    imshow("video", frame); // Show video
  }

  return 0;
}
