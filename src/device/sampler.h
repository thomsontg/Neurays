#ifndef SAMPLER_H
#define SAMPLER_H

#include <optix.h>
#include <src/device/cuda/random.h>

// Given a direction vector v sampled around the z-axis of a
// local coordinate system, this function applies the same
// rotation to v as is needed to rotate the z-axis to the
// actual direction n that v should have been sampled around
// [Frisvad, Journal of Graphics Tools 16, 2012;
//  Duff et al., Journal of Computer Graphics Techniques 6, 2017].
__inline__ __host__ __device__ void rotate_to_normal(const float3& normal, float3& v)
{
  float sign = copysignf(1.0f, normal.z);
  const float a = -1.0f/(1.0f + fabsf(normal.z));
  const float b = normal.x*normal.y*a;
  v = make_float3(1.0f + normal.x*normal.x*a, b, -sign*normal.x)*v.x
    + make_float3(sign*b, sign*(1.0f + normal.y*normal.y*a), -normal.y)*v.y
    + normal*v.z;
}

// Given spherical coordinates, where theta is the 
// polar angle and phi is the azimuthal angle, this
// function returns the corresponding direction vector
__inline__ __host__ __device__ float3 spherical_direction(float sin_theta, float cos_theta, float phi)
{
  float sin_phi = sinf(phi), cos_phi = cosf(phi);
  return make_float3(sin_theta*cos_phi, sin_theta*sin_phi, cos_theta);
}

SUTILFN float3 sample_hemisphere(const float3& normal, unsigned int& t)
{
  // Get random numbers
  float cos_theta = rnd(t);
  float phi = 2.0f*M_PIf*rnd(t);

  // Calculate new direction as if the z-axis were the normal
  float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
  float3 v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual normal and return
  rotate_to_normal(normal, v);
  return v;
}

SUTILFN float3 sample_direction(unsigned int t)
{
  float3 dir = make_float3(0.0f);

  dir.x = rnd(t);
  dir.y = rnd(t);
  dir.z = rnd(t);

  dir = normalize(dir);
  
  return dir;
}

SUTILFN float3 sample_cosine_weighted(const float3& normal, unsigned int& t)
{
  // Get random numbers
  float cos_theta = sqrtf(rnd(t));
  float phi = 2.0f*M_PIf*rnd(t);

  // Calculate new direction as if the z-axis were the normal
  float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
  float3 v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual normal and return
  rotate_to_normal(normal, v);
  return v;
}

SUTILFN float3 sample_isotropic(float3& forward, unsigned int& t)
{
  float xi = rnd(t);
  float cos_theta = 1.0f - 2.0f*xi;
  float phi = 2.0f*M_PIf*rnd(t);
  float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
  float3 v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual normal and return
  rotate_to_normal(forward, v);
  return v;
}

SUTILFN float3 sample_HG(const float3& forward, float g, unsigned int& t)
{
  float xi = rnd(t);
  float cos_theta;
  if(fabs(g) < 1.0e-3f)
    cos_theta = 1.0f - 2.0f*xi;
  else
  {
    float two_g = 2.0f*g;
    float g_sqr = g*g;
    float tmp = (1.0f - g_sqr)/(1.0f - g + two_g*xi);
    cos_theta = 1.0f/two_g*(1.0f + g_sqr - tmp*tmp);
  }
  float phi = 2.0f*M_PIf*rnd(t);

  // Calculate new direction as if the z-axis were the forward direction
  float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
  float3 v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual forward direction and return
  rotate_to_normal(forward, v);
  return v;
}

// EVAL and SAMPLE for the Draine (and therefore Cornette-Shanks) phase function
//   g = HG shape parameter
//   a = "alpha" shape parameter

// Warning: these functions don't special case isotropic scattering and can numerically fail for certain inputs

// eval:
//   u = dot(prev_dir, next_dir)
SUTILFN float evalDraine(float u, float g, float a)
{
    return ((1 - g*g)*(1 + a*u*u))/(4.*(1 + (a*(1 + 2*g*g))/3.) * M_PIf * powf(1 + g*g - 2*g*u,1.5));
}

// sample: (sample an exact deflection cosine)
//   xi = a uniform random real in [0,1]
SUTILFN float sampleDraineCos(float t, float g, float a)
{
    const float g2 = g * g;
	const float g3 = g * g2;
	const float g4 = g2 * g2;
	const float g6 = g2 * g4;
	const float pgp1_2 = (1 + g2) * (1 + g2);
	const float T1 = (-1 + g2) * (4 * g2 + a * pgp1_2);
	const float T1a = -a + a * g4;
	const float T1a3 = T1a * T1a * T1a;
	const float T2 = -1296 * (-1 + g2) * (a - a * g2) * (T1a) * (4 * g2 + a * pgp1_2);
	const float T3 = 3 * g2 * (1 + g * (-1 + 2 * t)) + a * (2 + g2 + g3 * (1 + 2 * g2) * (-1 + 2 * t));
	const float T4a = 432 * T1a3 + T2 + 432 * (a - a * g2) * T3 * T3;
	const float T4b = -144 * a * g2 + 288 * a * g4 - 144 * a * g6;
	const float T4b3 = T4b * T4b * T4b;
	const float T4 = T4a + sqrtf(-4 * T4b3 + T4a * T4a);
	const float T4p3 = powf(T4, 1.0 / 3.0);
	const float T6 = (2 * T1a + (48 * powf(2, 1.0 / 3.0) *
		(-(a * g2) + 2 * a * g4 - a * g6)) / T4p3 + T4p3 / (3. * powf(2, 1.0 / 3.0))) / (a - a * g2);
	const float T5 = 6 * (1 + g2) + T6;
	return (1 + g2 - powf(-0.5 * sqrtf(T5) + sqrtf(6 * (1 + g2) - (8 * T3) / (a * (-1 + g2) * sqrtf(T5)) - T6) / 2., 2)) / (2. * g);
}

SUTILFN float3 sample_Draine(const float3& forward, float g, float a, unsigned int& t)
{
  float cos_theta = sampleDraineCos(t, g, a);
  float phi = 2.0f*M_PIf*rnd(t);

  // Calculate new direction as if the z-axis were the forward direction
  float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
  float3 v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual forward direction and return
  rotate_to_normal(forward, v);
}

SUTILFN float3 sample_barycentric(unsigned int& t)
{
  // Get random numbers
  float sqrt_xi1 = sqrtf(rnd(t));
  float xi2 = rnd(t);

  // Calculate Barycentric coordinates
  float u = 1.0f - sqrt_xi1;
  float v = (1.0f - xi2)*sqrt_xi1;
  float w = xi2*sqrt_xi1;

  // Return barycentric coordinates
  return make_float3(u, v, w);
}

SUTILFN float3 sample_Phong_distribution(const float3& normal, const float3& dir, float shininess, unsigned int& t)
{
  // Get random numbers
  float cos_theta = powf(rnd(t), 1.0f/(shininess + 2.0f));
  float phi = 2.0f*M_PIf*rnd(t);

  // Calculate sampled direction as if the z-axis were the reflected direction
  float sin_theta = sqrtf(fmaxf(1.0f - cos_theta*cos_theta, 0.0f));
  float3 v = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual reflected direction
  rotate_to_normal(2.0f*dot(normal, dir)*normal - dir, v);
  return v;
}

SUTILFN float3 sample_Blinn_normal(const float3& normal, float shininess, unsigned int& t)
{
  // Get random numbers
  float cos_theta = powf(rnd(t), 1.0f/(shininess + 2.0f));
  float phi = 2.0f*M_PIf*rnd(t);

  // Calculate sampled half-angle vector as if the z-axis were the normal
  float sin_theta = sqrtf(fmaxf(1.0f - cos_theta*cos_theta, 0.0f));
  float3 hv = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual normal
  rotate_to_normal(normal, hv);
  return hv;
}

SUTILFN float3 sample_Beckmann_normal(const float3& normal, float width, unsigned int& t)
{
  // Get random numbers
  float tan_theta_sqr = -width*width*logf(1.0f - rnd(t));
  float phi = 2.0f*M_PIf*rnd(t);

  // Calculate sampled half-angle vector as if the z-axis were the normal
  float cos_theta_sqr = 1.0f/(1.0f + tan_theta_sqr);
  float cos_theta = sqrtf(cos_theta_sqr);
  float sin_theta = sqrtf(1.0f - cos_theta_sqr);
  float3 hv = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual normal
  rotate_to_normal(normal, hv);
  return hv;
}

SUTILFN float3 sample_GGX_normal(const float3& normal, float roughness, unsigned int& t)
{
  // Get random numbers
  float xi1 = rnd(t);
  float tan_theta_sqr = roughness*roughness*xi1/(1.0f - xi1);
  float phi = 2.0f*M_PIf*rnd(t);

  // Calculate sampled half-angle vector as if the z-axis were the normal
  float cos_theta_sqr = 1.0f/(1.0f + tan_theta_sqr);
  float cos_theta = sqrtf(cos_theta_sqr);
  float sin_theta = sqrtf(1.0f - cos_theta_sqr);
  float3 hv = spherical_direction(sin_theta, cos_theta, phi);

  // Rotate from z-axis to actual normal
  rotate_to_normal(normal, hv);
  return hv;
}

#endif // SAMPLER_H