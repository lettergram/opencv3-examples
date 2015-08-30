#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <cmath>

using namespace std;
using namespace cv;

/* Function Headers */
Point detectAndDisplay( Mat frame, Point priorCenter );

/* Global variables */

String face_cascade_name = "lbpcascade_frontalface.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

String display_window = "Display";
String face_window = "Face View";


int main() {
  VideoCapture cap;
  Mat frame;
  Point priorCenter(0,0);
  
  // Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){
    printf("Error loading face cascade\n");
    return -1;
  };
  
  if( !eyes_cascade.load( eyes_cascade_name ) ){
    printf("Error loading eyes cascade\n");
    return -1;
  };
  
  // Read the video stream
  cap.open( -1 );
  if ( !cap.isOpened() ){
    printf("Error opening video capture\n");
    return -1;
  }

  // Loop to capture frames
  while (  cap.read(frame) ) {
    if( frame.empty() ) {
      printf("No captured frame -- Break!");
      break;
    }

    // Apply the classifier to the frame, i.e. find face
    priorCenter = detectAndDisplay( frame, priorCenter );
    cout << priorCenter << endl;
    
    if( waitKey(30) >= 0 ) { break; } // space
  }
  return 0;
}

Mat outputFrame( Mat frame, Point center, int w, int h) {

  int x = center.x - w/2;
  int y = center.y - 3*h/5;

  if(x > frame.size().width || x < 0 ||
     y > frame.size().height || y < 0)
    return frame(Rect(0, 0, w, h));
  
  // output frame of only face
  return frame(Rect(x, y, w, h));
}

// Detect face and display it
Point detectAndDisplay( Mat frame, Point priorCenter) {
  
  std::vector<Rect> faces;
  Mat frame_gray, frame_lab, output;

  output = frame;
  
  cvtColor( frame, frame_gray, COLOR_BGR2GRAY );

  equalizeHist( frame_gray, frame_gray );

  int minNeighbors = 2;
  
  // Detect face with open source cascade
  face_cascade.detectMultiScale( frame_gray, faces,
				 1.1, minNeighbors,
				 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

  for( size_t i = 0; i < faces.size(); i++ ) {

    Point center( faces[i].x + faces[i].width/2,
		  faces[i].y + faces[i].height/2 );

    
    if(abs(center.x - priorCenter.x) < frame.size().width / 5 &&
       abs(center.y - priorCenter.y) < frame.size().height / 5) {

      center.x = (center.x + 2*priorCenter.x) / 3;
      center.y = (center.y + 2*priorCenter.y) / 3;

      priorCenter = center;

      // output frame of only face
      output = outputFrame(frame, center,
			   frame.size().width / 3,
			   frame.size().height / 2);
                 
      break;
      
    } else {

      Mat faceROI = frame_gray( faces[i] );
      std::vector<Rect> eyes;
      
      // Try to detect eyes, if no face is found
      eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2,
				     0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
      
      for( size_t j = 0; j < eyes.size(); j++ ) {
	
	Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2,
			  faces[i].y + eyes[j].y + eyes[j].height/2 );

	priorCenter.x += eye_center.x;
	priorCenter.y += eye_center.y;
	
      }

      // Use average location of eyes
      if(eyes.size() > 0) {
	priorCenter.x = priorCenter.x / eyes.size();
	priorCenter.y = priorCenter.y / eyes.size();

	// output frame of only face
	output = outputFrame(frame, priorCenter,
			     frame.size().width / 3,
			     frame.size().height / 2);
      }      
    }
  }
  
  // Display output
  imshow( display_window, frame );

  // Display only face
  imshow( face_window, output );
  
  return priorCenter;
}
