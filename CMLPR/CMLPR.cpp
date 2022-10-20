#include <iostream>
#include <string>
#include <windows.graphics.h>
#include <baseapi.h>
#include <allheaders.h>
#include "core/core.hpp"
#include "highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

Mat RGB2Grey(Mat RGB)
{
  Mat greyscale = Mat::zeros(RGB.size(), CV_8UC1);

  for (int i = 0; i < RGB.rows; i++)
  {
    for (int j = 0; j < RGB.cols * 3 ; j += 3)
    {
      greyscale.at<uchar>(i, j/3) = (RGB.at<uchar>(i, j) + RGB.at<uchar>(i, j + 1)+ RGB.at<uchar>(i, j+2))/3; 
    }
  }

  return greyscale; 
}

Mat Grey2BW(Mat Grey, int threshold)
{
  Mat bw = Mat::zeros(Grey.size(), CV_8UC1);

  for (int i = 0; i < Grey.rows; i++)
  {
    for (int j = 0; j < Grey.cols; j++)
    {
      if (Grey.at<uchar>(i,j) >= threshold)
      {
        bw.at<uchar>(i,j) = 255; 
      }
    }
  }

  return bw; 
}

Mat InverseGrey(Mat Grey)
{
  Mat inv = Mat::zeros(Grey.size(), CV_8UC1);

  for (int i = 0; i < Grey.rows; i++)
  {
    for (int j = 0; j < Grey.cols; j++)
    {
      
        inv.at<uchar>(i,j) = 255 - Grey.at<uchar>(i,j); 
      
    }
  }

  return inv; 
}

Mat Negative(Mat RGB)
{
  Mat neg = Mat::zeros(RGB.size(), CV_8UC3);

  for (int i = 0; i < RGB.rows; i++)
  {
    for (int j = 0; j < RGB.cols * 3; j++)
    {
      
      neg.at<uchar>(i,j) = 255 - RGB.at<uchar>(i,j); 
      
    }
  }

  return neg; 
}

Mat Step(Mat Grey)
{
  Mat stepping = Mat::zeros(Grey.size(), CV_8UC1);
  int stepMin = 80;
  int stepMax = 160; 

  for (int i = 0; i < Grey.rows; i++)
  {
    for (int j = 0; j < Grey.cols; j++)
    {
      if (Grey.at<uchar>(i,j) >= stepMin && Grey.at<uchar>(i,j) < stepMax) stepping.at<uchar>(i,j) = 255; 
    }
  }

  return stepping;
}

Mat Blur(Mat Source, int BlurRadius)
{
  Mat blurred = Mat::zeros(Source.size(), CV_8UC1);
  int avg = pow((2 * BlurRadius) + 1, 2) ; 
  
  
  for (int i = BlurRadius ; i < Source.rows - BlurRadius ; i++)
  {
    for (int j = BlurRadius ; j < Source.cols - BlurRadius ; j++)
    {
      int sum = 0;
      
      for (int n = -BlurRadius ; n <= BlurRadius; n++)
      {
        for (int p = -BlurRadius; p <= BlurRadius; p++)
        {
          sum += Source.at<uchar>((i+n), (j+p)); 
        }
        
      }

      blurred.at<uchar>(i,j) = sum / avg; 
    }
  }
  
  return blurred; 
}

Mat MaxVal(Mat Source, int Radius)
{
  Mat maxed = Mat::zeros(Source.size(), CV_8UC1);
  int avg = pow((2 * Radius) + 1, 2) ; 
  
  
  for (int i = Radius ; i < Source.rows - Radius ; i++)
  {
    for (int j = Radius ; j < Source.cols - Radius ; j++)
    {
      int max = -1;
      
      for (int n = -Radius ; n <= Radius; n++)
      {
        for (int p = -Radius; p <= Radius; p++)
        {
          if (Source.at<uchar>((i+n), (j+p)) > max) max = Source.at<uchar>((i+n), (j+p));   
        }
        
      }

      maxed.at<uchar>(i,j) = max; 
    }
  }
  
  return maxed; 
}

Mat MinVal(Mat Source, int Radius)
{
  Mat minmat = Mat::zeros(Source.size(), CV_8UC1);
  int avg = pow((2 * Radius) + 1, 2) ; 
  
  
  for (int i = Radius ; i < Source.rows - Radius ; i++)
  {
    for (int j = Radius ; j < Source.cols - Radius ; j++)
    {
      int min = 255;
      
      for (int n = -Radius ; n <= Radius; n++)
      {
        for (int p = -Radius; p <= Radius; p++)
        {
          if (Source.at<uchar>((i+n), (j+p)) < min) min = Source.at<uchar>((i+n), (j+p));   
        }
        
      }

      minmat.at<uchar>(i,j) = min; 
    }
  }
  
  return minmat; 
}

Mat AllEdgeDetect(Mat Source, int Radius, int Threshold)
{

  Mat allEdged = Mat::zeros(Source.size(), CV_8UC1);
  int avg = pow((2 * Radius) + 1, 2) ; 
  
  for (int i = Radius ; i < Source.rows - Radius ; i++)
  {
    for (int j = Radius ; j < Source.cols - Radius ; j++)
    {
      int avgL = 0;
      int avgR = 0;
      int countL = 0; 
      int countR = 0;
      
      for (int n = -Radius ; n < 0; n++)
      {
        for (int p = -Radius; p < 0; p++)
        {
          countL++; 
          avgL += (Source.at<uchar>((i+n), (j+p)))/countL;
          
        }
        
      }

      for (int n = 1 ; n <= Radius; n++)
      {
        for (int p = 1; p <= Radius; p++)
        {
          countR++; 
          avgR += (Source.at<uchar>((i+n), (j+p)))/countR;
          
        }
        
      }

      if (abs(avgL - avgR) > Threshold) allEdged.at<uchar>(i, j) = 255; 
      
    }
  }

  return allEdged; 
  
}

Mat VertEdgeDetect(Mat Source, int Radius, int Threshold)
{

  Mat vertEdged = Mat::zeros(Source.size(), CV_8UC1);
  int avg = pow((2 * Radius) + 1, 2) ; 
  
  for (int i = Radius ; i < Source.rows - Radius ; i++)
  {
    for (int j = Radius ; j < Source.cols - Radius ; j++)
    {
      int avgL = 0;
      int avgR = 0;
      int countL = 0; 
      int countR = 0;

      for (int n = -Radius; n <= Radius; n++)
      {
        for (int p = -Radius; p < 0; p++)
        {
      
          countL++; 
          avgL += (Source.at<uchar>((i+n), (j+p)))/countL;
          
        }
        
      }

      for (int n = -Radius; n <= Radius; n++)
      {
        for (int p = 1; p <= Radius; p++)
        {
      
          countR++; 
          avgR += (Source.at<uchar>((i+n), (j+p)))/countR;
          
        }
        
      }

      if (abs(avgL - avgR) > Threshold) vertEdged.at<uchar>(i, j) = 255; 
      
    }
  }

  return vertEdged; 
  
}

Mat HorEdgeDetect(Mat Source, int Radius, int Threshold)
{

  Mat horEdged = Mat::zeros(Source.size(), CV_8UC1);
  int avg = pow((2 * Radius) + 1, 2) ; 
  
  for (int i = Radius ; i < Source.rows - Radius ; i++)
  {
    for (int j = Radius ; j < Source.cols - Radius ; j++)
    {
      int avgU = 0;
      int avgD = 0;
      int countU = 0; 
      int countD = 0;
      
      for (int n = -Radius ; n < 0; n++)
      {
        for (int p = -Radius; p <= Radius; p++)
        {
          countU++; 
          avgU += (Source.at<uchar>((i+n), (j+p)))/countU;
          
        }
        
      }

      for (int n = 1 ; n <= Radius; n++)
      {
        for (int p = -Radius; p <= Radius; p++)
        {
          countD++; 
          avgD += (Source.at<uchar>((i+n), (j+p)))/countD;
          
        }
        
      }

      if (abs(avgU - avgD) > Threshold) horEdged.at<uchar>(i, j) = 255; 
      
    }
  }

  return horEdged; 
  
}

Mat Dilation (Mat Source , int Radius)
{
  Mat dilated = Mat::zeros(Source.size(), CV_8UC1);

  for (int i = Radius ; i < Source.rows - Radius ; i++)
  {
    for (int j = Radius ; j < Source.cols - Radius ; j++)
    {
      if (Source.at<uchar>(i,j) == 255) dilated.at<uchar>(i,j) = 255;
      else
      {
        for (int n = -Radius ; n <= Radius; n++)
        {
          for (int p = -Radius; p <= Radius; p++)
          {
            if (Source.at<uchar>((i+n), (j+p)) == 255 && Source.at<uchar>(i,j) == 0)
            {
              dilated.at<uchar>(i,j) = 255;
              break;
            }
          }
        }
      }
      
    }
  }

  return dilated; 
  
}

Mat Erosion (Mat Source , int Radius)
{
  Mat eroded = Mat::zeros(Source.size(), CV_8UC1);

  for (int i = Radius ; i < Source.rows - Radius ; i++)
  {
    for (int j = Radius ; j < Source.cols - Radius ; j++)
    {
      eroded.at<uchar>(i,j) = Source.at<uchar>(i,j);

      for (int n = -Radius ; n <= Radius; n++)
      {
        for (int p = -Radius; p <= Radius; p++)
        {
          if (Source.at<uchar>((i+n), (j+p)) == 0 && Source.at<uchar>(i,j) == 255)
          {
            eroded.at<uchar>(i,j) = 0;
            break;
          }
        }
      }
    }
  }

  return eroded; 
  
}

Mat EQHisto (Mat Source)
{
  Mat EQImg = Mat::zeros(Source.size(), CV_8UC1);
  // count
  int count[256] = {0};
  for (int i = 0; i < Source.rows; i++)
    for (int j = 0; j < Source.cols; j++)
      count[Source.at<uchar>(i, j)]++;
		

  // probability
  float prob[256] = { 0.0 };
  for (int i = 0; i < 256; i++)
    prob[i] = (float)count[i] / (float)(Source.rows * Source.cols);

  // accumulated probability
  float accprob[256] = { 0.0 };
  accprob[0] = prob[0];
  for (int i = 1; i < 256; i++)
    accprob[i] = prob[i] + accprob[i - 1];
  // new = 255 * accprob 
  int newvalue[256] = { 0 };
  for (int i = 0; i < 256; i++)
    newvalue[i] = 255 * accprob[i];

  for (int i = 0; i < Source.rows; i++)
    for (int j = 0; j < Source.cols; j++)
      EQImg.at<uchar>(i, j) = newvalue[Source.at<uchar>(i, j)];

  return EQImg;
}

Mat Masking(Mat Source, float ROI_W_P , float ROI_H_P, int Threshold)
{
  Mat masked = Mat::zeros(Source.size(), CV_8UC1);
  
  float roi_height = ROI_H_P * Source.rows;
  float roi_width = ROI_W_P * Source.cols;
 
  int y = (int)(Source.rows - roi_height)/2;
  int x = (int)(Source.cols - roi_width)/2;

  for (int i = y; i < Source.rows - y; i++ )
  {
    for (int j = x; j < Source.cols - x; j++)
    {
      if (Source.at<uchar>(i,j) > Threshold) masked.at<uchar>(i,j) = Source.at<uchar>(i,j);
    }
  }

  return masked; 
}



Mat Colourize (Mat BW)
{
  Mat colored = Mat::zeros(BW.size() , CV_8UC3);

  for (int i = 0; i < BW.rows; i++)
  {
    for (int j = 0; j < BW.cols; j++)
    {
      if (BW.at<uchar>(i,j) == 255)
      {
        colored.at<Vec3b>(i,j)[0] = 255;
        colored.at<Vec3b>(i,j)[1] = 0;
        colored.at<Vec3b>(i,j)[2] = 0;
        
      }

      else
      {
        colored.at<Vec3b>(i,j)[0] = 0;
        colored.at<Vec3b>(i,j)[1] = 0;
        colored.at<Vec3b>(i,j)[2] = 255;
      }
    }
  }

  return colored; 
}

int main()
{
  int threshold = 120;
  int blurradius = 2;
  int maxradius = 2;
  int minradius = 2;

  int edgeradius = 2; 
  int edgethreshold = 50; 

  float roi_width_percent = 0.7;
  float roi_height_percent = 0.7; 
  
  Mat RGBImg = imread("..\\Img\\Data_10.jpg");
  imshow("RGB Image", RGBImg);

  cout << "Image Size: " << RGBImg.cols << "(w) x " << RGBImg.rows << " (h)" << std::endl;
  cout << "BW Threshold Set At: " << threshold << std::endl;

  Mat GreyImg = RGB2Grey(RGBImg);
  imshow("Greyscale", GreyImg);

  Mat EQImg = EQHisto(GreyImg);
  imshow("EQ Grey image", EQImg);

  Mat MaskedImg = Masking(EQImg, roi_width_percent, roi_height_percent, threshold);
  imshow("Masked to ROI", MaskedImg); ;

  Mat BlurredImg = Blur(GreyImg, blurradius);
  string BlurTitle = "Blurred Greyscale @ " + std::to_string(blurradius) + " Blur Radius"; 
  imshow(BlurTitle, BlurredImg);
  
  Mat BWImg = Grey2BW(EQImg, threshold);
  string BWTitle = "Black & White Image, Threshold set at : " + std::to_string(threshold);  
  imshow(BWTitle, BWImg);

  Mat VertBlurredEdgedImg = VertEdgeDetect(BlurredImg, edgeradius, edgethreshold);
  imshow("Blurred & Vertically Edged Image", VertBlurredEdgedImg);
  
  Mat ErodedImg = Erosion(VertBlurredEdgedImg, 1);
  imshow("Eroded Image", ErodedImg);

  Mat DilatedImg = Dilation(ErodedImg, 6);
  imshow("Dilated Image After Erosion", DilatedImg);


  Mat DilatedImgCpy;
  DilatedImgCpy = DilatedImg.clone();
  vector<vector<Point>> contours1;
  vector<Vec4i> hierachy1;
  findContours(DilatedImg, contours1, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
  Mat dst = Mat::zeros(GreyImg.size(), CV_8UC3);



  if (!contours1.empty())
  {
    for (int i = 0; i < contours1.size(); i++)
    {
      Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
      drawContours(dst, contours1, i, colour, -1, 8, hierachy1);
    }
  }



  imshow("Segmented Image", dst);

  Mat plate;
  Rect rect;
  Scalar black = CV_RGB(0, 0, 0);
  for (int i = 0; i < contours1.size(); i++)
  {
    rect = boundingRect(contours1[i]);
    
    if (rect.height < 0.05 * GreyImg.rows || rect.width < 0.05 * GreyImg.cols || rect.width <= rect.height || rect.width < 1.5 * rect.height || rect.width > 4 * rect.height || rect.x < 0.2 * GreyImg.cols || rect.x > 0.8 * GreyImg.cols || rect.y < 0.2 * GreyImg.rows || rect.y > 0.8 * GreyImg.rows)
    {
      drawContours(DilatedImgCpy, contours1, i, black, -1, 8, hierachy1);
    }

    else plate = EQImg(rect); 
        
  }



  imshow("Filtered Image", DilatedImgCpy);
  if(plate.cols !=0 && plate.rows !=0)
  {
    imshow("Detected Plate", plate);
  }

  Mat ColorizedImg = Colourize(BWImg);
  imshow("Colourized Red Blue", ColorizedImg); 
  
  waitKey();

  tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
  
    
}

