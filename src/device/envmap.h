#ifndef ENVMAP_H
#define ENVMAP_H

#include <optix.h>
#include <src/sutil/math/vec_math.h>

__device__ __forceinline__ float3 env_lookup(const float3& dir)
{
  const LaunchParams& lp = launch_params;
  float theta = acosf(dir.y);
  float phi = atan2f(dir.x, -dir.z);
  float u = phi < 0.0 ? (phi + 2.0f * M_PIf) : phi;
  u /= (2.0f * M_PIf);
  float v = theta*M_1_PIf;
  return make_float3(tex2D<float4>(lp.envmap, u, v));
}

#endif // ENVMAP_H
