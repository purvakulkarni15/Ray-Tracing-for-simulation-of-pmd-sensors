#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include<cuda.h>
#include<stdio.h>
#include<conio.h>
#include<stdlib.h>
#include<vector>
#include<math.h>
#include<time.h>
using namespace std;

#define XMAX 224
#define YMAX 172
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 4

typedef struct Vector3f
{
	double x, y, z;
}Vec3;

typedef struct Ray
{
	Vec3 dir;
	Vec3 orig;
	Vec3 hitNormal;
	Vec3 nearIntersectionPoint;
	Vec3 color;
	double len;
	double distance;
	double reflectivity;
	double visibility;
}Ray;

//Camera Parameters
typedef struct CameraParameterList
{
	int camera_cols;
	int camera_rows;
	int super_sampling_factor;
	double frame_rate;

}CameraParameterList;

//Optical Parameters
typedef struct OpticalParameterList
{
	double focal_length;
	double fNumber;

}OpticalParameterList;

//Pixel Parameters
typedef struct PixelParameterList
{
	double pixel_length;
	double pixel_width;
	double fill_factor;

}PixelParameterList;

//Light Parameters
typedef struct LightParameterList
{
	Vec3 light_position;
	double wavelength;
	char profile_type[50];
	char profile_filename[50];

}LightParameterList;


typedef struct Plane
{
	double A, B, C, D;
	double reflectivity;
	Vec3 color;
}Plane;

typedef struct Triangle
{
	Vec3 P1, P2, P3;
	double reflectivity;
	Vec3 color;
}Triangle;

typedef struct Box
{
	Vec3 minPt, maxPt;
	double reflectivity;
	Vec3 color;
}Box;

typedef struct Sphere
{
	double radius;
	Vec3 center;
	double reflectivity;
	Vec3 color;
}Sphere;

vector<Plane> planeListHolder;
vector<Sphere> sphereListHolder;
vector<Triangle> triangleListHolder;
vector<Box> boxListHolder;


CameraParameterList cameraParameterList;
LightParameterList lightParameterList;
PixelParameterList pixelParameterList;
OpticalParameterList opticalParameterList;

double* intensityProfilePlane, *intensityProfileTriangle, *intensityProfileSphere, *intensityProfileBox;

