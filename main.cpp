#include <opencv2/opencv.hpp>
#include <ctime>
#include <iostream>
 
using namespace cv;
using namespace std;

Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;

const char* source_window = "Source image";
const char* corners_window = "Corners detected";

/// Function header
void cornerHarris_demo( int, void* );
 
int main()
{
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
  while(char(waitKey(1)) != 'c' && cap.isOpened())
  {
    
  Mat frame;
  cap >> frame;

  /**
  * @function main
  */
    /// Load source image and convert it to gray
    src = frame;
    cvtColor( src, src_gray, COLOR_BGR2GRAY );

    /// Create a window and a trackbar
    namedWindow( source_window, WINDOW_AUTOSIZE );
    createTrackbar( "Threshold: ", source_window, &thresh, max_thresh);
    imshow( source_window, src );

    
    Mat dst, dst_norm, dst_norm_scaled;
  dst = Mat::zeros( src.size(), CV_32FC1 );

  /// Detector parameters
  int blockSize = 2;
  int apertureSize = 3;
  double k = 0.04;

  clock_t cl = clock();
  /// Detecting corners
  cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
  cout << (clock()-cl)*1000/CLOCKS_PER_SEC << "ms" << endl;
  /// Normalizing
  normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  convertScaleAbs( dst_norm, dst_norm_scaled );

  /// Drawing a circle around corners
  for( int j = 0; j < dst_norm.rows ; j++ )
     { for( int i = 0; i < dst_norm.cols; i++ )
          {
            if( (int) dst_norm.at<float>(j,i) > thresh )
              {
               circle( src, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
              }
          }
     }
  /// Showing the result
  namedWindow( corners_window, WINDOW_AUTOSIZE );
  imshow( corners_window,src );
  
  }
    
      waitKey();
    return(0);
}
