#include "SensorParameterInitialization.h"

/*
Function: Checks for the intersection point of the ray and the plane by a method given in ScratchAPixel.  
Parameters: Plane Eqn->[ Ax + By + Cz + D = 0], instance of Ray struct
Output: Returns the whether the ray intersects with plane.
Out Param: intersection Point, Distance from the plane.
*/

__device__ bool IsRayPlaneIntersect(double A, 
									double B, 
									double C, 
									double D, 
									Ray ray, 
									Vec3* intersectionPoint, 
									Vec3* hitNormal,
									double* distance,
									bool isBackCullingOn)
{
	Vec3 pNormal = Vector3(A, B, C); //Compute plane normal
	pNormal = ScalarDiv(pNormal, Magnitude(pNormal));
	double t;

	double denominator = DotProduct(pNormal, ray.dir);

	if(isBackCullingOn && denominator > 0)
	{
		pNormal = ScalarMul(pNormal, -1);
		denominator = DotProduct(pNormal, ray.dir);
	}
	double numerator = D + DotProduct(pNormal, ray.orig);

	if(denominator != 0)
	{
		t = -1*(numerator/denominator); //Parametric variable

		if(t > 0.0)
		{
			*intersectionPoint = ScalarMul(ray.dir, t);
			*intersectionPoint  = Add(ray.orig, *intersectionPoint);
			*hitNormal = pNormal;
			*distance = sqrt((ray.orig.x - intersectionPoint->x) * (ray.orig.x - intersectionPoint->x) +
						     (ray.orig.y - intersectionPoint->y) * (ray.orig.y - intersectionPoint->y) +
					         (ray.orig.z - intersectionPoint->z) * (ray.orig.z - intersectionPoint->z)); 
			return true;
		}
		else
		{
			*intersectionPoint = Vector3(0, 0, 0);//No intersection point found
		}
	}

	return false;//No intersection point found
}


/*
Function: Checks for the intersection point of the ray and the sphere by a method given in ScratchAPixel.  
Parameters: Center(Sphere), Radius (Sphere), instance of Ray struct.
Output: Returns the whether the ray intersects with Sphere.
Out Param: intersection Point, Distance from the sphere.
*/


__device__ bool IsRaySphereIntersect(	Vec3 centerS, 
										double radius, 
										Ray ray, 
										Vec3* intersectionPoint, 
										Vec3* hitNormal,
										double* distance)
{
    double t0, t1;//parametric constants
    bool isIntersect = true;
    double radius2 = radius*radius;
    Vec3 L = Sub(ray.orig, centerS); 
    double tca = DotProduct(L, ray.dir); 

    if (tca < 0) isIntersect = false; 

    double d2 = DotProduct(L, L) - tca * tca; 

    if (d2 > radius2) isIntersect = false; 
	double thc = sqrt(radius2 - d2); 

    t0 = tca - thc; 
    t1 = tca + thc; 

    if (t0 > t1) 
    {
		double temp = t0;
		t0 = t1;
		t1 = temp;
    }
 
    if (t0 < 0)
    { 
        t0 = t1; // if t0 is negative, let's use t1 instead 
        if (t0 < 0) 
		isIntersect = false; // both t0 and t1 are negative 
    } 

    if(isIntersect)
    {
		*intersectionPoint  = ScalarMul(ray.dir, t0);
		*intersectionPoint  = Add(ray.orig, *intersectionPoint);
		*hitNormal = Normalize(Sub(centerS, *intersectionPoint));
		*distance = sqrt((ray.orig.x - intersectionPoint->x) * (ray.orig.x - intersectionPoint->x) +
						 (ray.orig.y - intersectionPoint->y) * (ray.orig.y - intersectionPoint->y) +
					     (ray.orig.z - intersectionPoint->z) * (ray.orig.z - intersectionPoint->z)); 

		return true;
    }
    else
    {
		*intersectionPoint = Vector3(0, 0, 0);//No intersection point found
		return false;
    }
}

/*
Function: Checks for the intersection point of the ray and the triangle by a method given in ScratchAPixel. 
Parameters: Vertices of the triangle (P1, P2, P3), instance of Ray struct.
Output: Returns the whether the ray intersects with triangle.
Out Param: intersection Point, Distance from the triangle.
*/

__device__ bool IsRayTriangleIntersect(	Vec3 P1, 
										Vec3 P2, 
										Vec3 P3, 
										Ray ray, 
										Vec3* intersectionPoint, 
										Vec3* hitNormal,
										double* distance,
										bool isBackCullingOn)
{
	Vec3 tNormal = CrossProduct(Sub(P1, P2), Sub(P1, P3));//compute the normal of the triangle
	tNormal = ScalarDiv(tNormal, Magnitude(tNormal));
	double t;
	double denominator = DotProduct(tNormal, ray.dir);

	if(isBackCullingOn && denominator > 0)
	{
		tNormal = ScalarMul(tNormal, -1);
		denominator = DotProduct(tNormal, ray.dir);
	}

	double D = DotProduct(tNormal, P1);
	double numerator = (D + DotProduct(tNormal, ray.orig));

	if(denominator != 0)
	{
		t = numerator/denominator;

		if(t > 0.0)
		{
			*intersectionPoint  = ScalarMul(ray.dir, t);
			*intersectionPoint  = Add(ray.orig, *intersectionPoint);

			Vec3 edge0 = Sub(P1,  P2); 
			Vec3 edge1 = Sub(P2,  P3); 
			Vec3 edge2 = Sub(P3,  P1); 

			Vec3 C0 = Sub(P1, *intersectionPoint);
			Vec3 C1 = Sub(P2, *intersectionPoint);
			Vec3 C2 = Sub(P3, *intersectionPoint);

			if(DotProduct(tNormal, CrossProduct(edge0, C0)) >= 0 && DotProduct(tNormal, CrossProduct(edge1, C1)) >= 0 && DotProduct(tNormal, CrossProduct(edge2, C2)) >= 0)
			{
				*hitNormal = tNormal;
				*distance = sqrt((ray.orig.x - intersectionPoint->x) * (ray.orig.x - intersectionPoint->x) +
						         (ray.orig.y - intersectionPoint->y) * (ray.orig.y - intersectionPoint->y) +
					             (ray.orig.z - intersectionPoint->z) * (ray.orig.z - intersectionPoint->z)); 
				return true;
			}
		}
	}
	 
	*intersectionPoint = Vector3(0, 0, 0) ;
	return false;//No intersection point found
}

/*
Function: Checks for the intersection point of the ray and the box by a method given in ScratchAPixel. 
Parameters: opposite points of a axis aligned rectangular box, instance of Ray struct.
Output: Returns the whether the ray intersects with box.
Out Param: intersection Point, Distance from the box.
*/


__device__ bool IsRayBoxIntersect(	Vec3 minPt, 
									Vec3 maxPt, 
									Ray ray, 
									Vec3* intersectionPoint, 
									Vec3* hitNormal,
									double* distance)
{
	bool isIntersect = true;

	Ray invRay;
	invRay.dir.x = 1/ray.dir.x;
	invRay.dir.y = 1/ray.dir.y;
	invRay.dir.z = 1/ray.dir.z;

	double tmin, tmax, tymin, tymax, tzmin, tzmax;

	if(invRay.dir.x >= 0)
	{
		tmin = (minPt.x - ray.orig.x) * invRay.dir.x; 
		tmax = (maxPt.x - ray.orig.x) * invRay.dir.x; 
	}
	else
	{
		tmin = (maxPt.x - ray.orig.x) * invRay.dir.x; 
		tmax = (minPt.x - ray.orig.x) * invRay.dir.x; 
	}
 
	if(invRay.dir.y >= 0)
	{ 
		tymin = (minPt.y - ray.orig.y) * invRay.dir.y; 
		tymax = (maxPt.y - ray.orig.y) * invRay.dir.y; 
	}
	else
	{
		tymin = (maxPt.y - ray.orig.y) * invRay.dir.y; 
		tymax = (minPt.y - ray.orig.y) * invRay.dir.y; 
	}
    
    if ((tmin > tymax) || (tymin > tmax)) 
	{
		return false;
	}

    if (tymin > tmin) 
        tmin = tymin; 
 
    if (tymax < tmax) 
        tmax = tymax; 
 
   if(invRay.dir.z >= 0)
	{ 
		tzmin = (minPt.z - ray.orig.z) * invRay.dir.z; 
		tzmax = (maxPt.z - ray.orig.z) * invRay.dir.z; 
	}
	else
	{
		tzmin = (maxPt.z - ray.orig.z) * invRay.dir.z; 
		tzmax = (minPt.z - ray.orig.z) * invRay.dir.z; 
	}
 
    if ((tmin > tzmax) || (tzmin > tmax)) 
	{
		return false;
	}
 
    if (tzmin > tmin) 
        tmin = tzmin; 
 
    if (tzmax < tmax) 
        tmax = tzmax; 
 
	
	if(floor(tmin) <= 0)
		return false;

	*intersectionPoint  = ScalarMul(ray.dir, tmin);
	*intersectionPoint  = Add(ray.orig, *intersectionPoint);

	//Find normal of the cube
	float bias = 1.00001;
	Vec3 cubeCenter = Vector3((minPt.x + maxPt.x)/2, (minPt.y + maxPt.y)/2, (minPt.z + maxPt.z)/2);
	Vec3 vectorBetCenterIntersectionPt = Sub(cubeCenter, *intersectionPoint);
	double dx = abs(maxPt.x - minPt.x)/2;
	double dy = abs(maxPt.y - minPt.y)/2;
	double dz = abs(maxPt.z - minPt.z)/2;

	*hitNormal = Normalize(Vector3((int)(bias*vectorBetCenterIntersectionPt.x/dx), (int)(bias*vectorBetCenterIntersectionPt.y/dy), (int)(bias*vectorBetCenterIntersectionPt.z/dz)));
	
	if(DotProduct(ray.dir, *hitNormal) > 0)
		*hitNormal = ScalarMul(*hitNormal, -1);
	
	*distance = sqrt((ray.orig.x - intersectionPoint->x) * (ray.orig.x - intersectionPoint->x) +
					(ray.orig.y - intersectionPoint->y) * (ray.orig.y - intersectionPoint->y) +
					(ray.orig.z - intersectionPoint->z) * (ray.orig.z - intersectionPoint->z)); 

	return true;
}
