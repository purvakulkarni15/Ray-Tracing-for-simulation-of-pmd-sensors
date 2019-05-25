#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"	
#include "stdio.h"

using namespace cv;

/*
Function: Visualize the results in an image file
Parameters: mat_image_data_output_host->color of every pixels, x->image(x_Pix), y->image(y_pix)
*/

void VisualizeResults(Vec3* mat_image_data_output_host, int x, int y)
{
	Mat img(y, x, CV_8UC3, Scalar(250,250,250));

	if(img.empty())
	{
		printf("Image Not Created");
	}
	else
	{
		for(int i = 0; i < y; i++)
		{
			for(int j = 0; j < x; j++)
			{
				img.at<Vec3b>(i,j)[0] = mat_image_data_output_host[i*x+j].z;  
				img.at<Vec3b>(i,j)[1] = mat_image_data_output_host[i*x+j].y;  
				img.at<Vec3b>(i,j)[2] = mat_image_data_output_host[i*x+j].x;  

			}
		}
	}

	char imageName[50] = "IMAGE";

	namedWindow(imageName, WINDOW_AUTOSIZE );
	imshow(imageName, img );
	waitKey(0);
	imwrite( "Image.jpg", img );
}

void VisualizeMatrix(Vec3* mat_image_data_output_host, int x, int y)
{
	Mat img(y, x, CV_8UC3, Scalar(250,250,250));

	if(img.empty())
	{
		printf("Image Not Created");
	}
	else
	{
		for(int i = 0; i < y; i++)
		{
			for(int j = 0; j < x; j++)
			{
				img.at<Vec3b>(i,j)[0] = mat_image_data_output_host[i*x+j].x*10;  
				img.at<Vec3b>(i,j)[1] = mat_image_data_output_host[i*x+j].y*10;  
				img.at<Vec3b>(i,j)[2] = mat_image_data_output_host[i*x+j].z*10;  

			}
		}
	}

	char imageName[50] = "IMAGE";

	namedWindow(imageName, WINDOW_AUTOSIZE );
	imshow(imageName, img );
	waitKey(0);
	imwrite( "ImageVisib.jpg", img );
}
