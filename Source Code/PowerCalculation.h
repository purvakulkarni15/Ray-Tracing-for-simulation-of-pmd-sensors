#include "vectorFuncs.h"

// PI 3.14159

__global__ void PowerCalculation(	CameraParameterList cameraparameterList, 
									OpticalParameterList opticalparameterList, 
									LightParameterList lightparameterList, 
									PixelParameterList pixelparameterList, 
									Vec3* mat_in_intersectionPoint, 
									Vec3* mat_in_normal, 
									double* mat_in_reflectivity, 
									double* mat_in_distance, 
									double* mat_in_intensityProfile, 
									int ipX, 
									int ipY, 
									int offsetX, 
									int offsetY, 
									int matX, 
									int matY, 
									double cosAlpha, 
									double PI,
									double* mat_out_powerCalculation)
{
	int blockOffset = threadIdx.x + threadIdx.y*blockDim.x;
	int gridOffset = blockIdx.x + blockIdx.y*gridDim.x;
	int index = blockOffset+ (gridOffset*blockDim.x*blockDim.y);

	if(index > matX*matY)
		return;

	//Calculate and normalize Light Direction
	Vec3 lightVec = Normalize(Sub(mat_in_intersectionPoint[index], lightParameterList.light_position));
	
	//Calculate Inclination = arctan(z/sqrt(x2+y2)).
	double sqrtX2plusY2 = sqrt(lightVec.x*lightVec.x + lightVec.y*lightVec.y);
	double inclination = atan2(lightVec.z, sqrtX2plusY2);

	//Calculate azimuth angle = arctan(y/x)
	double azimuth = atan2(lightVec.y, lightVec.x);

	//Calculate Intensity Profile of the light source for per pixel (172 X 168)
	int yScale = (int)(ipY/(2*offsetY));
	int xScale = (int)(ipX/(2*offsetX));
	int rowIndex = floor(inclination*sin(azimuth)*180/PI)*yScale;
	int colIndex = floor(inclination*cos(azimuth)*180/PI)*xScale;

	if(rowIndex < 0)
		rowIndex = 0;
	if(rowIndex > ipY)
		rowIndex = ipY;

	if(colIndex < 0)
		colIndex = 0;
	if(colIndex > ipX)
		colIndex = ipX;

	//Calculate radiance at the light source
	double L = (mat_in_reflectivity[index] * mat_in_intensityProfile[rowIndex * ipX + colIndex] * DotProduct(lightVec, mat_in_normal[index]))/(mat_in_distance[index] * mat_in_distance[index]*PI);

	//Calculate irradiance at the ray-object Instersection
	double E = (L*PI*cosAlpha)/(opticalparameterList.fNumber*opticalparameterList.fNumber);

	//Calculate power received at the sensor

	mat_out_powerCalculation[index] = E * pixelParameterList.pixel_length * pixelParameterList.pixel_width * pixelParameterList.fill_factor;
}
