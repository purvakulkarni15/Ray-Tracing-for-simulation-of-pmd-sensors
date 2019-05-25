#include "vectorFuncs.h"

/*
Function: Populate the input matrix with data for the calculation of ray's direction
Parameters:x_pixels->matrix cols, y_pixels->matrix Rows, cCenter->center of the camera in world coordinate system
Output: matrix with appropriate coordinates in accordance with the cCenter
*/

Vec3* GenerateMatrix(int x_pixels, int y_pixels)
{
	FILE* fpx = fopen("sensor_grid_x.dat", "r");
	FILE* fpy = fopen("sensor_grid_y.dat", "r");

	Vec3* matrix = (Vec3*)malloc(sizeof(Vec3)*y_pixels*x_pixels);

	for(int i = 0; i < y_pixels; i++)
	{
		for(int j = 0; j < x_pixels; j++)
		{
			fscanf(fpx,"%lf,", &matrix[i*x_pixels+j].x);
			fscanf(fpy,"%lf,", &matrix[i*x_pixels+j].y);

			matrix[i*x_pixels+j].z = 0;
		}
	}

	return matrix;
}

void SensorParameterInitization()
{
	//Camera Parameter Initialization
	cameraParameterList.camera_cols = 224;
	cameraParameterList.camera_rows = 172;
	cameraParameterList.frame_rate = 5;
	cameraParameterList.super_sampling_factor = 1;

	//Light Parameter Initialization
	lightParameterList.light_position = Vector3(15.475, 0, 0);
	lightParameterList.wavelength = 850;
	//lightParameterList.profile_type = "msmt_picoflexx";
	//lightParameterList.profile_filename = "VCSEL_picoflexx.txt";

	//Optical Parameter Initialization
	opticalParameterList.fNumber = 1.58;
	opticalParameterList.focal_length = 3.67;

	//Pixel Parameter Initialization
	pixelParameterList.pixel_length = 17.5;
	pixelParameterList.pixel_width = 17.5;
	pixelParameterList.fill_factor = 0.76;
}

void IntensityProfileMatrixBuilder(int x_pixels, int y_pixels)
{
	intensityProfilePlane = (double*)malloc(x_pixels*y_pixels*sizeof(double));
	intensityProfileTriangle = (double*)malloc(x_pixels*y_pixels*sizeof(double));
	intensityProfileSphere = (double*)malloc(x_pixels*y_pixels*sizeof(double));
	intensityProfileBox = (double*)malloc(x_pixels*y_pixels*sizeof(double));

	FILE* fp_ip_plane, *fp_ip_triangle, *fp_ip_sphere, *fp_ip_box;

	for(int i = 0; i < y_pixels; i++)
	{
		for(int j = 0; j < x_pixels; j++)
		{
			fscanf(fp_ip_plane,"%lf,", &intensityProfilePlane[i*x_pixels+j]);
			fscanf(fp_ip_triangle,"%lf,", &intensityProfileTriangle[i*x_pixels+j]);
			fscanf(fp_ip_sphere,"%lf,", &intensityProfileSphere[i*x_pixels+j]);
			fscanf(fp_ip_box,"%lf,", &intensityProfileBox[i*x_pixels+j]);
		}
	}
}
