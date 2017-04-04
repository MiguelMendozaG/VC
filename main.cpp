#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void ConnectedComponentsStats(Mat img)
{
// Use connected components with stats
Mat labels, stats, centroids;
int num_objects= connectedComponentsWithStats(img, labels, stats,
centroids);
// Check the number of objects detected
if(num_objects < 2 ){
cout << "No objects detected" << endl;
return;
}else{
cout << "Number of objects detected: " << num_objects - 1 << endl;
}
// Create output image coloring the objects and show area
Mat output= Mat::zeros(img.rows,img.cols, CV_8UC3);
RNG rng( 0xFFFFFFFF );
for(int i=1; i<num_objects; i++){
cout << "Object "<< i << " with pos: " << centroids.at<Point2d>(i)
<< " with area " << stats.at<int>(i, CC_STAT_AREA) << endl;
Mat mask= labels==i;
output.setTo(255/i, mask);
// draw text with area
stringstream ss;
ss << "area: " << stats.at<int>(i, CC_STAT_AREA);
putText(output,
ss.str(),
centroids.at<Point2d>(i),
FONT_HERSHEY_SIMPLEX,
0.4,
Scalar(255,255,255));
}
imshow("Result", output);
}

void FindContoursBasic(Mat img)
{
vector<vector<Point> > contours;
findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
Mat output= Mat::zeros(img.rows,img.cols, CV_8UC3);
// Check the number of objects detected
if(contours.size() == 0 ){
cout << "No objects detected" << endl;
return;
}else{
cout << "Number of objects detected: " << contours.size() << endl;
}
RNG rng( 0xFFFFFFFF );
for(int i=0; i<contours.size(); i++)
drawContours(output, contours, i, (float)255/(i+1));
cvtColor(output,output,COLOR_RGB2GRAY);
imshow("Result2", output);

  int histSize = 256;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat b_hist, g_hist, r_hist;
  calcHist( &output, 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
  //cout << b_hist.at<float>(0,38) <<endl;
  int tam = contours.size();
  int per[tam];
  int i=0;

    for (int j =1 ; j < 256 ; j++){
	if(b_hist.at<float>(0,j) != 0){
	  cout << b_hist.at<float>(0,j)<< endl;
	  per[i] =(int) b_hist.at<float>(0,j);
	  i++;
	}
 
  }


}
  

int main(int, char** argv){
  
  Mat imagen = imread("foto1.jpg",0);
  Mat no_objs = imread("foto3.jpg",0);
  int method=1, fondo=1;
  
  resize(imagen,imagen,cv::Size(),0.15,0.15);
  resize(no_objs,no_objs,cv::Size(),0.15,0.15);
  imshow("imagen", imagen);
  
  medianBlur(imagen,imagen,3);
  //imshow("imagen_blur", imagen);
  Mat res;
 // imagen=no_objs- imagen;
  
  if (fondo == 1){
  
  blur(imagen,no_objs,Size(imagen.cols/3,imagen.rows/3));
  imshow("new pat",no_objs);}
  
  if (method == 0) ///hace division
  {
    imagen.convertTo(imagen,CV_32F);
    no_objs.convertTo(no_objs,CV_32F);
    Mat res = imagen;
    //cv::divide(imagen,no_objs,res);
    res=imagen/no_objs;
    res.convertTo(res,CV_8U);
    res = 255*(1-res);
    imagen=res;
    
    imshow("division",imagen);}
  
  else{
    imagen=no_objs- imagen;
    imshow("resta",imagen);
  }
  
  //////////////////////
  ////calcula histograma
  ////
  /////////////////////

  int histSize = 256;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat b_hist, g_hist, r_hist;
  calcHist( &imagen, 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
  
  ////fin calculo histograma
  /////////////////////
  ////////////////////
  //calculo de umbral ///
  int T = 0, u1=0, u2=0, u1_aux=0, u2_aux=0, flag=0, cuenta=0, times=0;
  for (int i=0 ; i < 256 ; i++){ //calcula intensidad promedio del histograma
    T = b_hist.at<uchar>(0,i) + T;
  }
  cout << T/255 << endl;
  while (flag == 0){
    cuenta =0;
    u1_aux=u1;
    u2_aux=u2;
  for (int i=0; i< 256 ; i++){
    if (b_hist.at<uchar>(0,i) <= T){
      u1 = b_hist.at<uchar>(0,i)+u1;
      cuenta++;
    }
    else {
      u2 = b_hist.at<uchar>(0,i)+u2;
    }
  }
  u1=u1/cuenta;
  u2=u2/(255-cuenta);
    if (u1==u1_aux && u2==u2_aux)
      flag=1;
  T = 0.5*(u1+u2); times++;}
  //cout << T << endl;
  //cout << times << endl;

  ////////////////////////////////
  //fin calculo histograma
  ///////////////////////
  
  threshold(imagen, imagen, T, 255, CV_THRESH_BINARY);
  imshow("umbral",imagen);
  
  ///////////////////////////////
  ///////////
  //dilatacion
  ////////////
  int erosion_size = 5;
  int dilation_size = 5;
  Mat ima_dil;
  Mat element = getStructuringElement( MORPH_ELLIPSE,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );

  dilate( imagen, ima_dil, element );
  //imshow("dilatada",ima_dil);
  
  element = getStructuringElement( MORPH_ELLIPSE,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
  //////////////////////////
  ////////////////////
  //erosion
  ////////////
  erode(ima_dil,ima_dil,element);
  //imshow("erosion",ima_dil);

  ConnectedComponentsStats(ima_dil);	
  FindContoursBasic(ima_dil);


  waitKey();
  return 0;}
  