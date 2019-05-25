#include "RayObjectIntersection.h"

/*
Function: 
1. It is device function i.e. called and executed both on GPU.
2. Compute Ray-Scene Intersection
3. The scene contains geometric primitives like viz - Plane, Triangle, Box and Sphere.

Parameters:
1. planeList        -> Contains Plane Data for all the planes in the list  
2. sphereList       -> Contains Sphere Data for all the spheres in the list  
3. triangleList     -> Contains Triangle Data for all the triangles in the list  
4. boxList          -> Contains Box Data for all the boxes in the list  
5. planeListSize    -> Total number of planes in the planeList 
6. sphereListSize   -> Total number of spheres in the sphereList
7. triangleListSize -> Total number of triangles in the triangleList
8. boxListSize      -> Total number of boxes in the boxList
9. ray              -> Ray object (origin, direction, nearPoint, nearPointNormal, nearPointdistance, nearPointReflectivity, nearPointVisibity, objectColor, distBetPixelAndFocalPt)
					   (OUT PARAMETER)
10.objectColor      -> RGB color value of the object defined by user (OUT PARAMETER)
11.isBackCullingOn  -> Culling the backside of the object in Ray Tracing. (No RayObject Intersection will be tested for Object's Back side.)

Output: 
1. Whether or not the ray intersects with the scene.
2. Parameter 10 is an Out Parameter.
3. Some members of Ray structure also act as out parameters.
*/

__device__ bool TraceRay( Plane* planeList, 
						  Sphere* sphereList, 
						  Triangle* triangleList, 
					      Box* boxList,
					      int planeListSize, 
					      int sphereListSize, 
					      int triangleListSize, 
					      int boxListSize,
					      Ray* ray,
						  Vec3* objectColor,
						  bool isBackCullingOn)
{

	Vec3 intersectionPoint, hitNormal;
	double distance;
	bool isIntersect = false;
	ray->distance = 99999;


	for(int i = 0; i < planeListSize; i++)
	{
		if(IsRayPlaneIntersect(	planeList[i].A, 
								planeList[i].B, 
								planeList[i].C, 
								planeList[i].D, 
								*ray, 
								&intersectionPoint, 
								&hitNormal,
								&distance,
								isBackCullingOn))
		{
			isIntersect = true;
			if(distance < ray->distance)
			{
				ray->nearIntersectionPoint = intersectionPoint;
				ray->hitNormal = hitNormal;
				ray->distance = distance;
				ray->reflectivity = planeList[i].reflectivity;

				if(objectColor != NULL)
				*objectColor = planeList[i].color;
			}
		}
	}

	for(int i = 0; i < triangleListSize; i++)
	{	
		if(IsRayTriangleIntersect(	triangleList[i].P1, 
									triangleList[i].P2, 
									triangleList[i].P3, 
									*ray, 
									&intersectionPoint, 
									&hitNormal,
									&distance,
								    isBackCullingOn))
		{
			isIntersect = true;
			if(distance < ray->distance)
			{
				ray->nearIntersectionPoint = intersectionPoint;
				ray->hitNormal = hitNormal;
				ray->distance = distance;
				ray->reflectivity = triangleList[i].reflectivity;

				if(objectColor != NULL)
				*objectColor = triangleList[i].color;
			}
		}
	}

	for(int i = 0; i < boxListSize; i++)
	{
		if(IsRayBoxIntersect(	boxList[i].minPt, 
								boxList[i].maxPt, 
								*ray, 
								&intersectionPoint, 
								&hitNormal,
								&distance))
		{
			isIntersect = true;
			if(distance < ray->distance)
			{
				ray->nearIntersectionPoint = intersectionPoint;
				ray->hitNormal = hitNormal;
				ray->distance = distance;
				ray->reflectivity = boxList[i].reflectivity;

				if(objectColor != NULL)
				*objectColor = boxList[i].color;
			}
		}
	}

	for(int i = 0; i < sphereListSize; i++)
	{
		if(IsRaySphereIntersect(sphereList[i].center, 
								sphereList[i].radius, 
								*ray, 
								&intersectionPoint, 
								&hitNormal,
								&distance))
		{
			isIntersect = true;
			if(distance < ray->distance)
			{
				ray->nearIntersectionPoint = intersectionPoint;
				ray->hitNormal = hitNormal;
				ray->distance = distance;
				ray->reflectivity = sphereList[i].reflectivity;

				if(objectColor != NULL)
				*objectColor = sphereList[i].color;
			}
		}
	}
	if (ray->reflectivity > 0)
	{

	}
	return isIntersect;
}

/*
Function:
1. It is a global function i.e called from CPU and executed on GPU.
2. Calculates Ray Index by using threadID, blockID, blockDim and GridDim
3. Calculates the Ray Direction as SensorGrid[RayIndex] - RayOrigin.
4. Computes Ray Object Intersection
5. Compute Shadow Ray Tracing
6. Compute reflected Ray Tracing
7. Computes pixel color by using Lambert's Shading.

Parameters:
1. mat_in_sensorGridData,
2. planeList				 -> Contains Plane Data for all the planes in the list  
3. sphereList				 -> Contains Sphere Data for all the spheres in the list  
4. triangleList				 -> Contains Triangle Data for all the triangles in the list 
5. x						 -> X-Resolution of the image [224]
6. y                         -> Y-Resolution of the image [172]
7. boxList					 -> Contains Box Data for all the boxes in the list  
8. planeListSize			 -> Total number of planes in the planeList 
9. sphereListSize            -> Total number of spheres in the sphereList
10. triangleListSize         -> Total number of triangles in the triangleList
11.boxListSize               -> Total number of boxes in the boxList
12.lightSourceVec            -> Position Vector of Light Source (15.47, 0, 0)
13.Vec3 FocalPoint           -> Focal Point of the Camera (0, 0, -3.67)
14.mat_out_intersectionPoint -> Matrix to store nearest Ray-Object Intersection Point.
15.mat_out_normal            -> Matrix to store Object Normal.
16.mat_out_imageData         -> Matrix to store image color Data for Visualization
17.mat_out_distance          -> Matrix to store object-sensor distance
18.mat_out_reflectivity      -> Matrix to store Object relflectivity
19.mat_out_visibility        -> Matrix to store Object Visibility

Output:
1. Parameters 12 to 17 are out parameters
*/

__global__ void RayTracer_kernel(	Vec3* mat_in_sensorGridData,
									Plane* planeList, 
									Sphere* sphereList, 
									Triangle* triangleList, 
									Box* boxList,
									int x,
									int y, 
									int planeListSize, 
									int sphereListSize, 
									int triangleListSize, 
									int boxListSize,
									Vec3 lightSourceVec,
									Vec3 focalPoint,
									Vec3* mat_out_intersectionPoint, 
									Vec3* mat_out_normal, 
									Vec3* mat_out_imageData, 
									double* mat_out_distance, 
									double* mat_out_reflectivity, 
									int* mat_out_visibility)
{
	int blockOffset = threadIdx.x + threadIdx.y*blockDim.x;
	int gridOffset = blockIdx.x + blockIdx.y*gridDim.x;
	int index = blockOffset+ (gridOffset*blockDim.x*blockDim.y);

	if(index > x*y)
		return;
	
	double Ka = 0.2, Kd = 0.8, Ks = 0.38, diffuseColor = 100, specularColor = 256, shininessVal = 90;
	double bias = 1.5;
	Vec3 objectColor;

	Ray ray;
	ray.nearIntersectionPoint = Vector3(9999.0, 9999.0, 9999.0);
	ray.color = Vector3(0, 0, 0);
	ray.orig = Vector3(0, 0, -3.67);
			
	ray.len = sqrt( (ray.orig.x - mat_in_sensorGridData[index].x)*(ray.orig.x - mat_in_sensorGridData[index].x) +
					(ray.orig.y - mat_in_sensorGridData[index].y)*(ray.orig.y - mat_in_sensorGridData[index].y) +
					(ray.orig.z - mat_in_sensorGridData[index].z)*(ray.orig.z - mat_in_sensorGridData[index].z));

	ray.dir = Normalize(Sub(ray.orig, mat_in_sensorGridData[index]));

	//Compute Ray-Scene Intersection
	if(TraceRay(planeList, sphereList, triangleList, boxList, planeListSize, sphereListSize, triangleListSize, boxListSize, &ray, &objectColor, false))
	{
		//Lambert's Shading Calculations
		Vec3 lightVec, reflLightVec;
		double specular;

		lightVec = Sub(ray.nearIntersectionPoint, lightSourceVec);
		lightVec = Normalize(lightVec);
					
		double lambertian = DotProduct(lightVec, ray.hitNormal);
				
		if(lambertian < 0.0)
		{
			lambertian = 1.0;
		}
		else
		{
			reflLightVec = Sub(lightVec, ScalarMul(ray.hitNormal, DotProduct(lightVec, ray.hitNormal)*2));

			float specAngle = DotProduct(reflLightVec, ScalarMul(ray.dir, -1.0));
			if(specAngle < 0.0)
			{
				specAngle = 0.0;
			}
			specular = pow((double)specAngle, shininessVal);
		}

		ray.color.x = (Ka * objectColor.x + Kd * lambertian * diffuseColor + Ks * specular * specularColor);
		ray.color.y = (Ka * objectColor.y + Kd * lambertian * diffuseColor + Ks * specular * specularColor);
		ray.color.z = (Ka * objectColor.z + Kd * lambertian * diffuseColor + Ks * specular * specularColor);
		
		//Reflected Ray Tracing
		if(ray.reflectivity > 0)
		{
			Ray reflectedRay;
			reflectedRay.orig = Add(ray.nearIntersectionPoint, ScalarMul(ray.hitNormal,bias));
			reflectedRay.dir = reflect(ray.dir, ray.hitNormal);
			if(!TraceRay(planeList, sphereList, triangleList, boxList, planeListSize, sphereListSize, triangleListSize, boxListSize, &reflectedRay, &objectColor, false))
			{
				objectColor.x = 10;
				objectColor.y = 10;
				objectColor.z = 10;
			}
			else
			{
				lightVec = Sub(reflectedRay.nearIntersectionPoint, lightSourceVec);
				lightVec = Normalize(lightVec);
					
				double lambertian = DotProduct(lightVec, reflectedRay.hitNormal);
				
				if(lambertian < 0.0)
				{
					lambertian = 1.0;
				}
				else
				{
					reflLightVec = Sub(lightVec, ScalarMul(reflectedRay.hitNormal, DotProduct(lightVec, reflectedRay.hitNormal)*2));

					float specAngle = DotProduct(reflLightVec, ScalarMul(reflectedRay.dir, -1.0));
					if(specAngle < 0.0)
					{
						specAngle = 0.0;
					}
					specular = pow((double)specAngle, shininessVal);
				}

				ray.color.x = ray.color.x + ray.reflectivity*0.4*(Ka * objectColor.x + Kd * lambertian * diffuseColor + Ks * specular * specularColor);
				ray.color.y = ray.color.y + ray.reflectivity*0.4*(Ka * objectColor.y + Kd * lambertian * diffuseColor + Ks * specular * specularColor);
				ray.color.z = ray.color.z + ray.reflectivity*0.4*(Ka * objectColor.z + Kd * lambertian * diffuseColor + Ks * specular * specularColor);
			}
		}


		//Shadow Ray Tracing
		Ray shadowRay;
		shadowRay.orig = Add(ray.nearIntersectionPoint, ScalarMul(ray.hitNormal,bias));
		shadowRay.dir = Normalize(Sub(shadowRay.orig, lightSourceVec));

		double dist_lightSource = sqrt((shadowRay.orig.x - lightSourceVec.x) * (shadowRay.orig.x - lightSourceVec.x) +
								       (shadowRay.orig.y - lightSourceVec.y) * (shadowRay.orig.y - lightSourceVec.y) +
					                   (shadowRay.orig.z - lightSourceVec.z) * (shadowRay.orig.z - lightSourceVec.z));


		if(TraceRay(planeList, sphereList, triangleList, boxList, planeListSize, sphereListSize, triangleListSize, boxListSize, &shadowRay, NULL, true))
		{
			mat_out_visibility[index] = 0;
			if(shadowRay.distance > dist_lightSource)
			{
				mat_out_visibility[index] = 1;
			}
		}
		else
		{
			mat_out_visibility[index] = 1;
		}

		
		if(!mat_out_visibility[index])
		{
			ray.color.x = (Ka * objectColor.x + Kd * lambertian * diffuseColor + Ks * specular * specularColor)/10;///distanceSquare;
			ray.color.y = (Ka * objectColor.y + Kd * lambertian * diffuseColor + Ks * specular * specularColor)/10;///distanceSquare;
			ray.color.z = (Ka * objectColor.z + Kd * lambertian * diffuseColor + Ks * specular * specularColor)/10;///distanceSquare;
		}

		mat_out_intersectionPoint[index] = ScalarDiv(ray.nearIntersectionPoint,1000);
		mat_out_normal[index] = ray.hitNormal;
		mat_out_distance[index] = (ray.distance - ray.len)/1000;
		mat_out_reflectivity[index] = ray.reflectivity;
		mat_out_imageData[index] = ray.color;
	}
	else
	{
		mat_out_imageData[index] = Vector3(0, 0, 0); 
		mat_out_intersectionPoint[index] = Vector3(-1, -1, -1);
		mat_out_normal[index] = Vector3(-1, -1, -1);
		mat_out_visibility[index] = -1;
		mat_out_distance[index] = -1;
		mat_out_reflectivity[index] = -1;
	}

}

void RayTracer(	Vec3* mat_in_sensorGridData, 
				int x, 
				int y, 
				Vec3* mat_out_intersectionPoint, 
				Vec3* mat_out_normal, 
				double* mat_out_distance, 
				double* mat_out_reflectivity, 
				int* mat_out_visibility, 
				Vec3* mat_out_imageData)
{
	Plane* planeList_host = (Plane*)malloc(sizeof(Plane)*planeListHolder.size());
	Sphere* sphereList_host = (Sphere*)malloc(sizeof(Sphere)*sphereListHolder.size());
	Triangle* triangleList_host  = (Triangle*)malloc(sizeof(Triangle)*triangleListHolder.size());
	Box* boxList_host  = (Box*)malloc(sizeof(Box)*boxListHolder.size());
	
	Vec3 lightSourceVec = lightParameterList.light_position;
	Vec3 focalPoint = Vector3(0, 0, -1*opticalParameterList.focal_length);

	
	Plane* planeList;
	Triangle* triangleList;
	Sphere* sphereList;
	Box* boxList;


	for(int i = 0; i < planeListHolder.size(); i++)
	{
		planeList_host [i] = planeListHolder[i];
	}

	for(int i = 0; i < sphereListHolder.size(); i++)
	{
		sphereList_host[i] = sphereListHolder[i];
	}

	for(int i = 0; i < triangleListHolder.size(); i++)
	{
		triangleList_host[i] = triangleListHolder[i];
	}

	for(int i = 0; i < boxListHolder.size(); i++)
	{
		boxList_host[i] = boxListHolder[i];
	}

	Vec3* mat_in_sensorGridData_dev;
	Vec3* mat_out_intersectionPoint_dev;
	Vec3* mat_out_normal_dev;
	Vec3* mat_out_imageData_dev;
	double* mat_out_distance_dev;
	double* mat_out_reflectivity_dev;
	int* mat_out_visibility_dev;

	cudaMalloc((void**)&mat_in_sensorGridData_dev, x*y*sizeof(Vec3));
	cudaMalloc((void**)&mat_out_intersectionPoint_dev, x*y*sizeof(Vec3));
	cudaMalloc((void**)&mat_out_normal_dev, x*y*sizeof(Vec3));
	cudaMalloc((void**)&mat_out_imageData_dev, x*y*sizeof(Vec3));
	cudaMalloc((void**)&mat_out_distance_dev, x*y*sizeof(double));
	cudaMalloc((void**)&mat_out_reflectivity_dev, x*y*sizeof(double));
	cudaMalloc((void**)&mat_out_visibility_dev, x*y*sizeof(int));

	cudaMalloc((void**)&planeList, planeListHolder.size()*sizeof(Plane));
	cudaMalloc((void**)&triangleList, triangleListHolder.size()*sizeof(Triangle));
	cudaMalloc((void**)&sphereList, sphereListHolder.size()*sizeof(Sphere));
	cudaMalloc((void**)&boxList, boxListHolder.size()*sizeof(Box));

	cudaMemcpy(mat_in_sensorGridData_dev, mat_in_sensorGridData, x*y*sizeof(Vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(planeList, planeList_host, planeListHolder.size()*sizeof(Plane), cudaMemcpyHostToDevice);
	cudaMemcpy(triangleList, triangleList_host, triangleListHolder.size()*sizeof(Triangle), cudaMemcpyHostToDevice);
	cudaMemcpy(sphereList, sphereList_host, sphereListHolder.size()*sizeof(Sphere), cudaMemcpyHostToDevice);
	cudaMemcpy(boxList, boxList_host, boxListHolder.size()*sizeof(Box), cudaMemcpyHostToDevice);

	dim3 block(32, 2, 1);
	dim3 grid((int)ceil((float)x/block.x), (int)ceil((float)y/block.y), 1);

	RayTracer_kernel<<<grid, block>>>(	mat_in_sensorGridData_dev,
										planeList,
										sphereList,
										triangleList,
										boxList,
										x,
										y, 
										planeListHolder.size(), 
										sphereListHolder.size(), 
										triangleListHolder.size(), 
										boxListHolder.size(),
										lightSourceVec,
										focalPoint,
										mat_out_intersectionPoint_dev, 
										mat_out_normal_dev, 
										mat_out_imageData_dev, 
										mat_out_distance_dev, 
										mat_out_reflectivity_dev, 
										mat_out_visibility_dev);
	cudaDeviceSynchronize();

	cudaMemcpy(mat_out_intersectionPoint, mat_out_intersectionPoint_dev, x*y*sizeof(Vec3), cudaMemcpyDeviceToHost);
	cudaMemcpy(mat_out_normal, mat_out_normal_dev, x*y*sizeof(Vec3), cudaMemcpyDeviceToHost);
	cudaMemcpy(mat_out_distance, mat_out_distance_dev, x*y*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(mat_out_reflectivity, mat_out_reflectivity_dev, x*y*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(mat_out_visibility, mat_out_visibility_dev, x*y*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(mat_out_imageData, mat_out_imageData_dev, x*y*sizeof(Vec3), cudaMemcpyDeviceToHost);

	cudaFree(mat_in_sensorGridData_dev);
	cudaFree(mat_out_intersectionPoint_dev);
	cudaFree(mat_out_normal_dev);
	cudaFree(mat_out_distance_dev);
	cudaFree(mat_out_reflectivity_dev);
	cudaFree( mat_out_imageData_dev);

	planeListHolder.clear();
	sphereListHolder.clear();
	triangleListHolder.clear();
	boxListHolder.clear();

	free(planeList_host);
	free(sphereList_host);
	free(boxList_host);
	free(triangleList_host);
	
}

