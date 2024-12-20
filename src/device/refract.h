
#include <optix.h>
#include "src/sutil/math/complex.h"

// Note: n2 is the ior of the medium into which we refract to.
__device__ __host__ __inline__ bool _refract(const float3 & i, const float3 & n, const float n1_over_n2, float3 & refracted, float & cos_theta_i, float & cos_theta_t)
{
	cos_theta_i = fmaxf(0.0f,dot(i, n));
	float cos_theta_i_sqr = cos_theta_i*cos_theta_i;
	float sin_theta_t_sqr = n1_over_n2*n1_over_n2*(1.0f - cos_theta_i_sqr);
	if (sin_theta_t_sqr >= 1.0f)
	{
		// Total internal refraction
		cos_theta_t = 1.0f;
		return false;
	}
	cos_theta_t = sqrt(1.0f - sin_theta_t_sqr);
	refracted = n1_over_n2*(cos_theta_i*n - i) - n*cos_theta_t;
	return true;
}

// Refraction convention.
// i always points away from the medium.
// refracted always points inside the medium
// IOR has to be refracted medium index over source medium index
__device__ __host__ __inline__ bool refract(const float3 & i, const float3 & n, const float n1_over_n2, float3 & refracted, float & F_r, float & cos_theta_i, float & cos_theta_t)
{
	bool pass = _refract(i, n, n1_over_n2, refracted, cos_theta_i, cos_theta_t);
	F_r = !pass ? 1.0f : fresnel_R(cos_theta_i, cos_theta_t, 1.0f/n1_over_n2);
	return pass;
}

__device__ __host__ __inline__ bool refract(const float3 & i, const float3 & n, const float n1_over_n2, float3 & refracted, float & F_r)
{
	float cos_theta_i, cos_theta_t;
	return refract(i, n, n1_over_n2, refracted, F_r, cos_theta_i, cos_theta_t);
}


__device__ __host__ __inline__ bool refract(const float3 & i, const float3 & n, const float n1_over_n2, float3 & refracted)
{
	float cos_theta_i, cos_theta_t;
	return _refract(i, n, n1_over_n2, refracted, cos_theta_i, cos_theta_t);
}

__host__ __device__ __inline__ bool
refract(const float3 &w, float3 n, float n1_over_n2, float &R, float3 &wt) {

  float cos_theta_i = dot(w, n);
  if (cos_theta_i < 0.0f) {
    cos_theta_i = -cos_theta_i;
    n = -n;
  }

  const float sin_theta_out_sqr =
      n1_over_n2 * n1_over_n2 * (1.0f - cos_theta_i * cos_theta_i);
  if (sin_theta_out_sqr > 1.0f) {
    R = 1.0f;
    return false;
  }

  const float cos_theta_out = sqrtf(1.0f - sin_theta_out_sqr);
  R = fresnel_R(cos_theta_i, cos_theta_out, n1_over_n2);
  wt = n1_over_n2 * (n * cos_theta_i - w) - n * cos_theta_out;
  return true;
}
