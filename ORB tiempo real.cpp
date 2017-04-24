#include <iostream>
#include <string>
#include <sstream>
#include <ctime>
using namespace std;
// OpenCV includes
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"

using namespace cv;

int main( int argc, const char** argv )
{
  // Read images	
 
// 0 is the ID of the built-in laptop camera, change if you want to use other camera
VideoCapture cap(1);

//check if the file was opened properly
if(!cap.isOpened())
{
cout << "Capture could not be opened successfully" << endl;
return -1;
}
 
namedWindow("Video");
  Mat edges;
// Play the video in a loop till it ends
  Ptr<ORB> orb_detector = ORB::create();  
while(char(waitKey(1)) != 'c' && cap.isOpened())
{
  
Mat frame;
cap >> frame;
  
  
  
  std::vector<KeyPoint> kp;
  clock_t cl = clock();
  orb_detector->detect(frame, kp);
  cout << (clock()-cl)*1000/CLOCKS_PER_SEC << "ms" << endl;
  
  Mat out;
  drawKeypoints(frame, kp, out, Scalar::all(255));
  imshow("Lena Keypoints", out);
}
  
  // wait for any key press
  waitKey(0);
  return 0;
}
