#include "RayTracing.h"
#include "visualizeResults.h"
#include <time.h>

void DisplayMenu();
void StoreInFile(Vec3* mat_out_intersectionPoint, Vec3* mat_out_normal, double* mat_out_distance, double* mat_out_reflectivity, int* mat_out_visibility, int x, int y);

int main()
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	Vec3* mat_in_sensorGridData = GenerateMatrix(XMAX, YMAX);

	Vec3* mat_out_intersectionPoint = (Vec3*)malloc(XMAX*YMAX * sizeof(Vec3));
	Vec3* mat_out_normal = (Vec3*)malloc(XMAX*YMAX * sizeof(Vec3));
	Vec3* mat_out_imageData = (Vec3*)malloc(XMAX*YMAX * sizeof(Vec3));
	double* mat_out_distance = (double*)malloc(XMAX*YMAX * sizeof(double));
	double* mat_out_reflectivity = (double*)malloc(XMAX*YMAX * sizeof(double));
	int* mat_out_visibility = (int*)malloc(XMAX*YMAX * sizeof(int));

	SensorParameterInitization();
	DisplayMenu();

	float milliseconds = 0;
	cudaEventRecord(start, 0);

	RayTracer(mat_in_sensorGridData,
		XMAX,
		YMAX,
		mat_out_intersectionPoint,
		mat_out_normal,
		mat_out_distance,
		mat_out_reflectivity,
		mat_out_visibility,
		mat_out_imageData);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("%f\n", milliseconds*0.001);

	StoreInFile(mat_out_intersectionPoint, mat_out_normal, mat_out_distance, mat_out_reflectivity, mat_out_visibility, XMAX, YMAX);
	VisualizeResults(mat_out_imageData, XMAX, YMAX);

	free(mat_in_sensorGridData);
	free(mat_out_intersectionPoint);
	free(mat_out_normal);
	free(mat_out_distance);
	free(mat_out_reflectivity);
	free(mat_out_imageData);

	getch();
}

/*
Function: Display Menu to the user
Parameters:
Output:
*/

void DisplayMenu()
{
	int *objects;
	int numObjects;

	printf("Enter the number of objects to launch: ");
	scanf(" %d", &numObjects);

	objects = (int*)malloc(numObjects * sizeof(int));

	for (int i = 0; i < numObjects; i++)
	{
		printf("___________________\n");
		printf("| Plane     |  1  |\n");
		printf("| Triangle  |  2  |\n");
		printf("| Box       |  3  |\n");
		printf("| Sphere    |  4  |\n");
		printf("|___________|_____|\n");

		printf("Enter your choice: %d\n", i + 1);
		scanf("%d", &objects[i]);
		printf("\n");

		if (objects[i] == 1)
		{
			Plane plane;

			printf("A: ");
			scanf("%lf", &plane.A);
			printf("B: ");
			scanf("%lf", &plane.B);
			printf("C: ");
			scanf("%lf", &plane.C);
			printf("D: ");
			scanf("%lf", &plane.D);

			printf("Reflectivity:");
			scanf("%lf", &plane.reflectivity);

			printf("Color:\n");
			printf("Red Component: ");
			scanf("%lf", &plane.color.x);
			printf("Green Component: ");
			scanf("%lf", &plane.color.y);
			printf("Blue Component: ");
			scanf("%lf", &plane.color.z);

			planeListHolder.push_back(plane);
		}
		else if (objects[i] == 2)
		{
			Triangle triangle;

			printf("Vertex 1: X: ");
			scanf("%lf", &triangle.P1.x);
			printf("Vertex 1: Y: ");
			scanf("%lf", &triangle.P1.y);
			printf("Vertex 1: Z: ");
			scanf("%lf", &triangle.P1.z);

			printf("Vertex 2: X: ");
			scanf("%lf", &triangle.P2.x);
			printf("Vertex 2: Y: ");
			scanf("%lf", &triangle.P2.y);
			printf("Vertex 2: Z: ");
			scanf("%lf", &triangle.P2.z);

			printf("Vertex 3: X: ");
			scanf("%lf", &triangle.P3.x);
			printf("Vertex 3: Y: ");
			scanf("%lf", &triangle.P3.y);
			printf("Vertex 3: Z: ");
			scanf("%lf", &triangle.P3.z);

			printf("Reflectivity:");
			scanf("%lf", &triangle.reflectivity);

			printf("Color:\n");
			printf("Red Component: ");
			scanf("%lf", &triangle.color.x);
			printf("Green Component: ");
			scanf("%lf", &triangle.color.y);
			printf("Blue Component: ");
			scanf("%lf", &triangle.color.z);

			triangleListHolder.push_back(triangle);
		}
		else if (objects[i] == 3)
		{
			Box box;

			printf("MinPoint: X: ");
			scanf("%lf", &box.minPt.x);
			printf("MinPoint: Y: ");
			scanf("%lf", &box.minPt.y);
			printf("MinPoint: Z: ");
			scanf("%lf", &box.minPt.z);

			printf("MaxPoint: X: ");
			scanf("%lf", &box.maxPt.x);
			printf("MaxPoint: Y: ");
			scanf("%lf", &box.maxPt.y);
			printf("MaxPoint: Z: ");
			scanf("%lf", &box.maxPt.z);

			printf("Reflectivity:");
			scanf("%lf", &box.reflectivity);

			printf("Color:\n");
			printf("Red Component: ");
			scanf("%lf", &box.color.x);
			printf("Green Component: ");
			scanf("%lf", &box.color.y);
			printf("Blue Component: ");
			scanf("%lf", &box.color.z);

			boxListHolder.push_back(box);
		}
		else if (objects[i] == 4)
		{
			Sphere sphere;

			printf("Radius: ");
			scanf("%lf", &sphere.radius);
			printf("CenterX: ");
			scanf("%lf", &sphere.center.x);
			printf("CenterY: ");
			scanf("%lf", &sphere.center.y);
			printf("CenterZ: ");
			scanf("%lf", &sphere.center.z);

			printf("Reflectivity:");
			scanf("%lf", &sphere.reflectivity);

			printf("Color:\n");
			printf("Red Component: ");
			scanf("%lf", &sphere.color.x);
			printf("Green Component: ");
			scanf("%lf", &sphere.color.y);
			printf("Blue Component: ");
			scanf("%lf", &sphere.color.z);

			sphereListHolder.push_back(sphere);
		}
	}
}

void StoreInFile(Vec3* mat_out_intersectionPoint, Vec3* mat_out_normal, double* mat_out_distance, double* mat_out_reflectivity, int* mat_out_visibility, int x, int y)
{
	FILE* fp1 = fopen("IntersectionPoint.csv", "w");
	FILE* fp2 = fopen("Normal.csv", "w");
	FILE* fp3 = fopen("Distance.csv", "w");
	FILE* fp4 = fopen("Reflectivity.csv", "w");
	FILE* fp5 = fopen("Visibility.csv", "w");

	for (int row = 0; row < y; row++)
	{
		for (int col = 0; col < x; col++)
		{
			if (mat_out_intersectionPoint[row*x + col].x != 9999.0)
			{
				//Intersection Point
				fprintf(fp1, "x: %lf y: %lf z: %lf,", mat_out_intersectionPoint[row*x + col].x, mat_out_intersectionPoint[row*x + col].y, mat_out_intersectionPoint[row*x + col].z);
				fprintf(fp2, "x: %lf y: %lf z: %lf,", mat_out_normal[row*x + col].x, mat_out_normal[row*x + col].y, mat_out_normal[row*x + col].z);
				fprintf(fp3, "%lf,", mat_out_distance[row*x + col]);
				fprintf(fp4, "%lf,", mat_out_reflectivity[row*x + col]);
				fprintf(fp5, "%d,", mat_out_visibility[row*x + col]);
			}
			else
			{
				//No IntersectionPoint
				fprintf(fp1, "x: %lf y: %lf z: %lf,", -1.0, -1.0, -1.0);
				fprintf(fp2, "x: %lf y: %lf z: %lf,", -1.0, -1.0, -1.0);
				fprintf(fp3, "%lf,", 0);
				fprintf(fp4, "%lf,", 0);
				fprintf(fp5, "%d,", 0);

			}
		}
		fprintf(fp1, "\n");
		fprintf(fp2, "\n");
		fprintf(fp3, "\n");
		fprintf(fp4, "\n");
		fprintf(fp5, "\n");
	}

	fclose(fp1);
	fclose(fp2);
	fclose(fp3);
	fclose(fp4);
	fclose(fp5);
}
