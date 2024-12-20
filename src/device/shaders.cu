#include <optix.h>

#include <cuda/LocalGeometry.h>
#include <cuda/helpers.h>
#include <cuda/random.h>
#include <cuda_runtime.h>
#include <src/sutil/math/vec_math.h>

// #include <torch/torch.h>
// #include <torch/script.h>

#include "src/sutil/shared/structs.h"
// #include "Scene.h"
#include "trace.h"
#include "cdf_bsearch.h"
#include "sampler.h"
#include "fresnel.h"
#include "refract.h"
#include "microfacet.h"
#include "src/Render/Lights/SunSky.h"
#include "phasefunc.h"
#include "../network/KernelNetwork.h"
// #include "tiny-cuda-nn/common.h"

#include "../gaussian/Gaussian.h"

extern "C" {
  __constant__ LaunchParams launch_params;
}

#include "envmap.h"
#include "AreaLight.h"
#include "env_cameras.cu"

#define DIRECT
#define INDIRECT

SUTILFN float3 smoothstep(const float3& edge0, const float3& edge1, const float3& x)
{
  const float3 t = clamp((x - edge0)/(edge1 - edge0), 0.0, 1.0);
  return t*t*(3.0f - 2.0f*t);
}

SUTILFN uchar4 make_rgba(const float3& color)
{
  float3 c = clamp(color, 0.0f, 1.0f);
  return make_uchar4(quantizeUnsigned8Bits(c.x), quantizeUnsigned8Bits(c.y), quantizeUnsigned8Bits(c.z), 255u);
}

#include "shaders/shared.cu"
#include "shaders/raygen.cu"
#include "shaders/miss.cu"
#include "shaders/helpers.cu"

#include "shaders/absorbing.cu"
#include "shaders/normals.cu"
#include "shaders/basecolor.cu"
#include "shaders/lambertian.cu"
#include "shaders/directional.cu"
#include "shaders/arealight.cu"
#include "shaders/mirror.cu"
#include "shaders/transparent.cu"
// #include "shaders/rough_transparent.cu"
#include "shaders/glossy.cu"
#include "shaders/metal.cu"
#include "shaders/volume.cu"
#include "shaders/translucent.cu"
#include "shaders/sub_surf.cu"
#include "shaders/fresnel.cu"
#include "shaders/heterogeneous.cu"
#include "shaders/network.cu"
#include "shaders/volume_rough.cu"
#include "shaders/network_inf.cu"
#include "shaders/distance.cu"
#include "shaders/distance_firsthit.cu"
#include "shaders/check_normals.cu"
#include "shaders/holdout.cu"
#include "shaders/radiance_holdout.cu"
#include "shaders/spheres.cu"
#include "shaders/blood.cu"
#include "shaders/bone.cu"
#include "shaders/bidirectional_volume.cu"

// #include "../Render/Scene/Photons.h"