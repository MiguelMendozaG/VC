#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int, char** argv)
{
//! [load_image]
    // Load the image
    Mat src = imread("arbol.png");

    // Check if everything was fine
    if (!src.data)
        return -1;

    // Show source image
    imshow("Source Image", src);
//! [load_image]

//! [black_bg]
    // Change the background from white to black, since that will help later to extract
    // better results during the use of Distance Transform
    for( int x = 0; x < src.rows; x++ ) {
      for( int y = 0; y < src.cols; y++ ) {
          if ( src.at<Vec3b>(x, y) == Vec3b(255,255,255) ) {
            src.at<Vec3b>(x, y)[0] = 0;
            src.at<Vec3b>(x, y)[1] = 0;
            src.at<Vec3b>(x, y)[2] = 0;
          }
        }
    }

    // Show output image
    //imshow("Black Background Image", src);
//! [black_bg]

//! [sharp]
    // Create a kernel that we will use for accuting/sharpening our image
    Mat kernel = (Mat_<float>(3,3) <<
            1,  1, 1,
            1, -8, 1,
            1,  1, 1); // an approximation of second derivative, a quite strong kernel

    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    Mat imgLaplacian;
    Mat sharp = src; // copy source image to another temporary one
    filter2D(sharp, imgLaplacian, CV_32F, kernel);
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;

    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

    // imshow( "Laplace Filtered Image", imgLaplacian );
  //  imshow( "New Sharped Image", imgResult );
//! [sharp]

    src = imgResult; // copy back

//! [bin]
    // Create binary image from source image
    Mat bw;
    cvtColor(src, bw, CV_BGR2GRAY);
    threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
   // imshow("Binary Image", bw);
//! [bin]

//! [dist]
    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, CV_DIST_L2, 3);

    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1., NORM_MINMAX);
   // imshow("Distance Transform Image", dist);
//! [dist]

//! [peaks]
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, .4, 1., CV_THRESH_BINARY);

    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    dilate(dist, dist, kernel1);
   // imshow("Peaks", dist);
//! [peaks]

//! [seeds]
    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32SC1);

    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);

    // Draw the background marker
    circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
    //imshow("Markers", markers*10000);
//! [seeds]

//! [watershed]
    // Perform the watershed algorithm
    watershed(src, markers);

    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    bitwise_not(mark, mark);
//    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
                                  // image looks like at that point

    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);

        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);

    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
                dst.at<Vec3b>(i,j) = colors[index-1];
            else
                dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
        }
    }

    // Visualize the final image
  //  imshow("Final Result", dst);
//! [watershed]
    
    cvtColor(dst,dst,CV_RGB2GRAY);
    imshow("Gray",dst);
    
    vector<Mat> bgr_planes;
    split( dst, bgr_planes );
    
    int histSize = 256;
    float range[] = {0,256};
    const float* histRange = {range};
    Mat b_hist;
    
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, true, false );
    cout << b_hist << endl;
    
    
    int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
  }

  /// Display
  namedWindow("calcHist Demo", WINDOW_AUTOSIZE );
imshow("calcHist Demo", histImage );
    
    
    
    
    
    //////////////////////////////////////////
    //////////////////////////////////////
    ///inicia deteccion de regiones por su umbral
    
    
    int region[3];
    int j=0, i, m;
    for (i = 1; i < 256 ; i++){
      if (b_hist.at<float>(0,i) !=0) {
      region[j] = i;
      //cout << region[j] << endl;
	j++;
      }
    }
    
    
    Mat dst_aux = Mat::zeros(dst.size(),CV_8UC1);
    int area = 0, perimetro = 0, area_aux;
    int i_per = 0, j_per=0, per_aux =0;
    for (m = 0 ; m < 3; m++){
      for (i = 0; i < dst.rows ; i++){
	for (j = 0; j < dst.cols ; j++){
	  if (dst.at<uchar>(i,j) == region[m])
	    dst_aux.at<uchar>(i,j) = region[m];
	  else
	    dst_aux.at<uchar>(i,j) = 0;
	}
      }
    area =0; perimetro = 0;
    for (i=1 ; i<dst.rows ; i++){
	for (j=1 ; j<dst.cols; j++){
	  area_aux = (dst_aux.at<uchar>(i-1,j-1) && 1 ) + ( dst_aux.at<uchar>(i-1,j) && 1 ) + ( dst_aux.at<uchar>(i-1,j+1) && 1 ) + 		     (dst_aux.at<uchar>(i,j-1) && 1 ) +   (dst_aux.at<uchar>(i,j) && 1) +     (dst_aux.at<uchar>(i,j+1) && 1) + 		     (dst_aux.at<uchar>(i+1,j-1) && 1) +  (dst_aux.at<uchar>(i+1,j) && 1) +   (dst_aux.at<uchar>(i+1,j+1) && 1);
	  if ( dst_aux.at<uchar>(i,j)==region[m])
	    area++; 
	  if (area_aux > 2 && area_aux < 8 && dst_aux.at<uchar>(i,j)==region[m])
	    perimetro++;
	}
      }
    
    cout << "area - perimetro - compactitud de Region:" << m << endl;
    cout << area << " - " << perimetro << " - "<< pow(perimetro,2)/area << endl;
    imshow ("region",dst_aux);
    cvWaitKey(0);
    }

    waitKey(0);
    return 0;
}