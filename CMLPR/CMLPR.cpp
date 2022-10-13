#include <iostream>
#include <string>
#include <windows.graphics.h>
#include "core/core.hpp"
#include "highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace cv; 

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

Mat EdgeDetect(Mat Source, int Radius, int Threshold)
{

  Mat edged = Mat::zeros(Source.size(), CV_8UC1);
  int avg = pow((2 * Radius) + 1, 2) ; 
  
  for (int i = Radius ; i < Source.rows - Radius ; i++)
  {
    for (int j = Radius ; j < Source.cols - Radius ; j++)
    {
      int sum = 0;
      
      for (int n = -Radius ; n <= Radius; n++)
      {
        for (int p = -Radius; p <= Radius; p++)
        {
          sum += Source.at<uchar>((i+n), (j+p));
          
        }
        
      }
      edged.at<uchar>(i,j) = sum / avg;

      if (edged.at<uchar>(i,j) >= Threshold) edged.at<uchar>(i,j) = 0;
      else edged.at<uchar>(i,j) = 255; 
    }
  }

  return edged; 
  
}



// Mat Colourize (Mat BW)
// {
//   static COLORREF blue = RGB(0,0,255);
//   static COLORREF red = RGB(255,0,0); 
//   Mat colored = Mat::zeros(BW.rows(), BW.cols()*3 , CV_8UC3); 
//
//   for (int i = 0; i < BW.rows(); i++)
//   {
//     for (int j = 0; j < BW.cols() * 3 ; j+=3)
//     {
//       if (BW.at<uchar>(i,j /3) == 255)
//       {
//         colored.at<uchar>(i,j) = 255;
//         colored.at<uchar>(i,j+1)= 0;
//         colored.at<uchar>(i,j+2) = 0; 
//       }
//
//       else
//       {
//         colored.at<uchar>(i,j) = 0;
//         colored.at<uchar>(i,j+1)= 0;
//         colored.at<uchar>(i,j+2) = 255;
//       }
//     }
//   }
//
//   return colored; 
// }

int main()
{
  int threshold = 128;
  int blurradius = 2;
  int maxradius = 5; 
  
  Mat RGBImg = imread("..\\Img\\Data_1.jpg");
  imshow("RGB Image", RGBImg);

  Mat GreyImg = RGB2Grey(RGBImg);
  imshow("Greyscale", GreyImg);

  Mat InvGreyImg = InverseGrey(GreyImg);
  imshow("Inverted Greyscale", InvGreyImg);

  Mat BlurredImg = Blur(GreyImg, blurradius);
  std::string BlurTitle = "Blurred Greyscale @ " + std::to_string(blurradius) + " Blur Radius"; 
  imshow(BlurTitle, BlurredImg);

  Mat MaxImg = MaxVal(GreyImg, maxradius);
  std::string MaxTitle = "Blown Out @ " + std::to_string(maxradius) + " Radius"; 
  imshow(MaxTitle, MaxImg);
  
  Mat ColNeg = Negative(RGBImg);
  imshow("Colour Negative", ColNeg);
  
  std::cout << "Image Size: " << RGBImg.rows << "(w) x " << RGBImg.cols << " (h)" << std::endl;
  std::cout << "BW Threshold Set At: " << threshold << std::endl;

  Mat BWImg = Grey2BW(GreyImg, threshold);
  std::string BWTitle = "Black & White Image, Threshold set at : " + std::to_string(threshold);  
  imshow(BWTitle, BWImg);

  Mat SteppedImg = Step(GreyImg);
  imshow("Stepped Image", SteppedImg);

  Mat EdgedImg = EdgeDetect(BlurredImg, 1, 55);
  imshow("Edged Image", EdgedImg);

  // Mat Colorized = Colourize(BWImg);
  // imshow("Colourized Image", Colorized); 
  waitKey();

  
  
    
}

