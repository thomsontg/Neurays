#pragma once

#include <sampler.h>

#include "../Render/structs.h"
#include "fresnel.h"
#include "microfacet.h"
#include "refract.h"

__device__ __inline__ float refract_ray(const struct Layer &layer,
                                        const float3 &w, float3 n,
                                        float n1_over_n2, float3 &wt,
                                        RNDTYPE &t, bool flip = false)
{

  float T;
  switch (layer.bsdf.type)
  {
  case SMOOTH:
  {
    float R = 1.0f;
    refract(w, n, n1_over_n2, R, wt);
    T = 1.0f - R;
    break;
  }
  case MICROFACET:
  {
    T = ggx_refract(wt, w, n, w.z, n1_over_n2, layer.bsdf.ggx.roughness, t);
    if (dot(wt, -n) < 0.0f || isnan(T))
    {
      return 0.0f;
    }
    break;
  }
  }
  if (flip)
  {
    wt = -wt;
  }
  return T;
}
