#include <optix.h>

#include <cuda/LocalGeometry.h>
#include <cuda/helpers.h>
#include <cuda/random.h>
#include <sampler.h>
#include <src/sutil/vec_math.h>

// #include "bsdf/bsdf.h"
#include <src/device/shaders/mc_color.cu>
#include "phasefunc.h"
#include "fresnel.h"
#include "bsdf.h"
#include <src/Render/structs.h>

// #include <src/Render/Lights/Lights.h>

extern "C"
{
  __constant__ LaunchParams launch_params;
}

#include "shaders/mc_helpers.cu"

__device__ __inline__ float sample_emitter(const float3 &x, float3 &wi,
                                           float3 &Le, float &dist,
                                           const SimpleLight &light, RNDTYPE &t) {
  const SimpleLight &dir_light = light;
  wi = -dir_light.direction;
  Le = dir_light.emission;
  dist = 1e32f;
  return 1.0f;
}

__device__ __inline__ void cross_boundary_up(struct Photon &p, const float3 &n,
                                             RNDTYPE &t)
{
  const LaunchParams &lp = launch_params;
  // Compute distance to layer upper boundary
  const float d =
      dist_to_surface(launch_params.layers[p.layer_idx].start, p.x.z, p.w.z);
  p.x += p.w * d;
  // Probability
  const float x = rnd_closed(t);
  float n1_over_n2 = compute_ior_up(p.layer_idx);
  float R = fresnel_R(p.w.z, n1_over_n2);
  if (x < R)
  {
    // Reflect
    p.w = reflect(p.w, n);
    p.weight *= R;
  }
  else
  {
    // Else penetrate through upper surface
    if (p.layer_idx == 0)
    {
      p.layer_idx = lp.mode == REFLECT ? -1 : -2;
    }
    else
    {
      p.layer_idx--;
    }
  }
}

__device__ __inline__ void cross_boundary_down(struct Photon &p,
                                               const float3 &n, RNDTYPE &t)
{

  const LaunchParams &lp = launch_params;
  // Compute distance to layer upper boundary
  const float d =
      dist_to_surface(launch_params.layers[p.layer_idx].end, p.x.z, p.w.z);
  p.x += p.w * d;
  // Probability
  const float x = rnd_closed(t);
  float n1_over_n2 = compute_ior_down(p.layer_idx);
  float R = fresnel_R(-p.w.z, n1_over_n2);
  if (x < R)
  {
    // Reflect
    p.w = reflect(p.w, -n);
    p.weight *= R;
  }
  else
  {
    if (p.layer_idx == (launch_params.n_layers - 1))
    {
      p.layer_idx = lp.mode == TRANSMIT ? -1 : -2;
    }
    else
    {

      p.layer_idx++;
    }
  }
}

// Returns true touched a boundary
__device__ __inline__ bool cross_boundary(struct Photon &p, const float3 &n,
                                          RNDTYPE &t)
{
  const struct Layer &layer = launch_params.layers[p.layer_idx];
  if (p.x.z >= layer.start && p.w.z > 0)
  {
    cross_boundary_up(p, n, t);
    return true;
  }
  else if (p.x.z < layer.end && p.w.z < 0)
  {
    cross_boundary_down(p, n, t);
    return true;
  }

  return false;
}

extern "C" __global__ void __raygen__volume_sim_diffuse()
{
  const LaunchParams &lp = launch_params;
  const uint3 launch_idx = optixGetLaunchIndex();
  const uint3 launch_dims = optixGetLaunchDimensions();
  const unsigned int frame = lp.subframe_index;
  const unsigned int pixel_idx = launch_idx.y * launch_dims.x + launch_idx.x;
  const float2 idx = make_float2(launch_idx.x, launch_idx.y);
  const float2 res = make_float2(launch_dims.x, launch_dims.y);
  RNDTYPE t = tea<16>(pixel_idx, frame);

  // Accumulate result from previous run
  const uint2 p_offset = make_uint2(frame % 2u, (frame + 1u) % 2u);
  const float new_sum = accum_result(p_offset, pixel_idx);

  // Current layer
  unsigned int layer_idx = 0;
  Layer const *cur_layer = &lp.layers[layer_idx];
  const float3 n = lp.normal;

  // Light ray
  const float3 xi = make_float3(0.0f, 0.0f, cur_layer->start);
  float3 wi;
  float3 Le;
  float l_dist = 0.0f;
  float pdf = sample_emitter(xi, wi, Le, l_dist, lp.lights[0], t);
  float cos_theta_i = wi.z;
  if (cos_theta_i < 1.0e-4f)
  {
    return;
  }

  // Refract ray
  const float n1_over_n2 = 1.0f / cur_layer->ior;
  float3 w12;
  float T12 = refract_ray(*cur_layer, wi, n, n1_over_n2, w12, t);

  // Optical properties
  const float sigma_t = cur_layer->extinction;
  const float albedo = cur_layer->albedo;
  const float g = cur_layer->asymmetry;

  for (unsigned int i = 0u; i < lp.subframe_index; ++i)
  {
    struct Photon p = init_photon(w12, xi, T12, layer_idx);
    unsigned int counter = 0u;
    while (counter++ < 1000000u && p.weight > 0.0f)
    {
      // Propagation
      const float d1 = -log(1.0f - rnd(t)) / sigma_t;
      p.x += p.w * d1;
      // Handle surface interactions
      if (cross_boundary(p, n, t))
      {
        if ((int)p.layer_idx == -1)
        {
          record_contrib(p.x, p.weight);
        }
        if (p.layer_idx < 0)
        {
          break;
        }
        else
        {
          cur_layer = &lp.layers[p.layer_idx];
        }
      }
      else
      {
        // Scattering or absorption
        const float x = rnd(t);
        if (x < albedo)
        {
          p.w = sample_HG(p.w, g, t);
        }
        else
        {
          break;
        }
      }
    }
  }

  // Progressive update and display
  write_frame_buffer(frame, new_sum, pixel_idx);
}

extern "C" __global__ void __raygen__volume_sim()
{
  const LaunchParams &lp = launch_params;
  const uint3 launch_idx = optixGetLaunchIndex();
  const uint3 launch_dims = optixGetLaunchDimensions();
  const unsigned int frame = lp.subframe_index;
  const unsigned int pixel_idx = launch_idx.y * launch_dims.x + launch_idx.x;
  const float2 idx = make_float2(launch_idx.x, launch_idx.y);
  const float2 res = make_float2(launch_dims.x, launch_dims.y);
  RNDTYPE t = tea<16>(pixel_idx, frame);

  // Accumulate result from previous run
  const uint2 p_offset = make_uint2(frame % 2u, (frame + 1u) % 2u);
  const float new_sum = accum_result(p_offset, pixel_idx);

  // Current layer
  unsigned int layer_idx = 0;
  Layer const *cur_layer = &lp.layers[layer_idx];

  // Observation ray
  const float3 n = make_float3(0.0f, 0.0f, 1.0f);
  float3 wo = lp.wo;

  if ((lp.mode == REFLECT && wo.z < 0.0f))
  {
    return;
  }
  if ((lp.mode == TRANSMIT && wo.z > 0.0f))
  {
    return;
  }
  assert((lp.mode == REFLECT && wo.z >= 0.0f) ||
         (lp.mode == TRANSMIT && wo.z <= 0.0f));

  // // Sample location on gaussian beam by sampling a disk
  // const float beam_radius = lp.beam_radius;
  // const float2 sd = lp.beam_std;
  // const float cos_angle = 0.0f; // sqrtf(rnd(t));
  // const float sin_angle = 1.0f; // sqrtf(1.0f - cos_angle * cos_angle);
  // const float phi = 2.0f * M_PIf * rnd(t);
  // float3 xi = spherical_direction(sin_angle, cos_angle, phi);
  // xi.z = 0.0f; // cur_layer->start;
  // xi *= beam_radius * rnd(t);
  // float area_weight = M_PIf * beam_radius * beam_radius;
  // float weight = area_weight / (sqrtf(2 * M_PIf) * sd.x) *
  //                expf(-dot(xi, xi) / (2.0f * sd.x * sd.x)) * 1e5 / lp.total_flux;
  // // printf("%f\n", lp.total_flux);

  // float beam_radius = lp.beam_radius;
  // const float2 &mean = lp.beam_mean;   // Mean (average)
  // const float2 &std_dev = lp.beam_std; // Standard deviation
  // float3 xi;
  // xi.x = sample_normal_distribution(mean.x, std_dev.x, rnd(t)) /100.f;
  // xi.y = sample_normal_distribution(mean.y, std_dev.y, rnd(t)) /100.f;
  // // printf("%f %f\n", xi.x, xi.y);
  // xi.z = 0.0f;
  // const float weight = 1838213.875 / lp.total_flux;


  // find index in
  int index = cdf_bsearch(rnd(t));
  int ii = index / lp.n_light_flux;
  int jj = index % lp.n_light_flux;
  float dx = lp.dx;
  float flux = lp.light_flux[index];
  float flux_normalized = flux / lp.total_flux;
  float pdf_flux = lp.light_flux_pdf[index];
  float3 xi = make_float3((ii - lp.mid_i) * dx, (jj - lp.mid_j) * dx, 0.0);
  float weight = flux_normalized / pdf_flux;
  if (pdf_flux <= 0.0f) {
    return;
  }

  // Light ray
  float3 wi;
  float3 Le;
  float l_dist = 0.0f;
  float pdf = sample_emitter(xi, wi, Le, l_dist, lp.lights[0], t);
  float cos_theta_i = wi.z;
  if (cos_theta_i < 1.0e-4f)
  {
    return;
  }

  // Reflect/Refract ray
  const float n1_over_n2 = 1.0f / cur_layer->ior;

  // // Compute reflection
  // if (lp.include_reflection)
  // {
  //   float R =
  //       microfacet_reflect(wi, wo, n, n1_over_n2, cur_layer->bsdf.ggx,
  //                          lp.microfacet_normals, 0, t);
  //   {
  //     record_contrib(xi, R * weight);
  //   }
  // }
  // return;

  // Refract ray and scattering
  float3 w12, w21;
  float T12, T21;
  // if (lp.number_of_normals > 0)
  // {
  //   T12 = refract_ray_measured(*cur_layer, wi, n, n1_over_n2, w12,
  //                              lp.microfacet_normals, 0, t,
  //                              T12);
  //   T21 = refract_ray_measured(*cur_layer, wo, n, n1_over_n2, w21,
  //                              lp.microfacet_normals, 0, t,
  //                              T21);
  //   if (T12 == 0 || T21 == 0)
  //   {
  //     return;
  //   }
  //   w21 = -w21;
  // }
  // else
  // {
    T12 = refract_ray(*cur_layer, wi, n, n1_over_n2, w12, t);
    T21 = refract_ray(*cur_layer, wo, n, n1_over_n2, w21, t, true);
  // }

  // Optical properties
  const float sigma_t = cur_layer->extinction;
  const float albedo = cur_layer->albedo;
  const float g = cur_layer->asymmetry;
  float pixel_area =
      launch_params.pixel_size * launch_params.pixel_size * lp.subframe_index;

  for (unsigned int i = 0u; i < lp.subframe_index; ++i)
  {

    struct Photon p = init_photon(w12, xi, T12, layer_idx);
    unsigned int counter = 0u;
    while (counter++ < 1000000u)
    {
      // Propagation
      const float d1 = -log(1.0f - rnd(t)) / sigma_t;
      p.x += p.w * d1;
      // Handle surface interactions
      if (cross_boundary(p, n, t))
      {
        if ((int)p.layer_idx < 0)
        {
          // we ended penetrating through an outer layer
          break;
        }
      }
      else
      {
        if (counter >= 1u)
        {
          const float z = cur_layer->start;
          const float d2 = dist_to_surface(z, p.x.z, w21.z);
          const float3 xo = p.x + d2 * w21;
          const float factor =
              albedo * p_hg(dot(p.w, w21), g) * expf(-d2 * sigma_t);
          const float result = p.weight * T21 * factor;
          record_contrib(xo, weight * result / pixel_area);
        }

        // Scattering or absorption
        const float x = rnd(t);
        if (x < albedo)
        {
          p.w = sample_HG(p.w, g, t);
        }
        else
        {
          break;
        }
      }
    }
  }
  write_frame_buffer(frame, new_sum, pixel_idx);
}

extern "C" __global__ void __miss__radiance() {}
