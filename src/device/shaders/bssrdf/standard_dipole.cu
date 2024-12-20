#pragma once

#include <optix.h>

SUTILFN float3 dipole(float dist, const OptProps &properties) {
  float3 real_source = properties.three_D * properties.three_D;
  float3 extrapolation = 4.0f * properties.A * properties.D;
  float3 virtual_source = extrapolation * extrapolation;
  float3 corrected_mean_free = properties.three_D + extrapolation;
  float3 r_sqr = make_float3(dist * dist);

  // Compute distances to dipole sources
  float3 d_r_sqr = r_sqr + real_source;
  float3 d_r = sqrtf(d_r_sqr);
  float3 d_v_sqr = r_sqr + virtual_source;
  float3 d_v = sqrtf(d_v_sqr);

  // Compute intensities for dipole sources
  float3 tr_r = properties.transport * d_r;
  float3 S_r = properties.three_D * (1.0f + tr_r) / (d_r_sqr * d_r);
  S_r *= expf(-tr_r);
  float3 tr_v = properties.transport * d_v;
  float3 S_v = corrected_mean_free * (1.0f + tr_v) / (d_v_sqr * d_v);
  S_v *= expf(-tr_v);
  return S_r + S_v;
}

// SUTILFN void test_dipole(const OptProps &p) {

//   // Optical properties
//   OptProps props1 = p;
//   props1.use_deltaEdd = false;
//   props1.extinction = make_float3(10.0f);
//   props1.albedo = make_float3(0.9f);
//   props1.meancosine = make_float3(0.0f);
//   props1.f0 = make_float3(0.3f);
//   float ior_rgb = 1.3f;

//   // Derived
//   props1.scattering = props1.extinction * props1.albedo;
//   props1.absorption = props1.extinction - props1.scattering;
//   props1.reducedScattering = (1.0 - props1.meancosine) * props1.scattering;
//   props1.reducedExtinction = props1.reducedScattering + props1.absorption;
//   props1.D = 1.0 / (3.0 * (props1.reducedScattering + props1.absorption));
//   props1.transport = sqrtf(props1.absorption / props1.D);
//   props1.C_phi = make_float3(C_phi(ior_rgb));
//   props1.C_phi_inv = make_float3(C_phi(1.f / ior_rgb));
//   props1.C_E = make_float3(C_E(ior_rgb));
//   props1.reducedAlbedo = props1.reducedScattering / props1.reducedExtinction;
//   props1.de = 2.131f * props1.D / sqrtf(props1.reducedAlbedo);
//   props1.A = (1.f - props1.C_E) / (2.0 * props1.C_phi);
//   props1.three_D = 3.0f * props1.D;
//   props1.two_A_de = 2.f * props1.A * props1.de;
//   props1.global_coeff =
//       (1.f) / (4.0f * props1.C_phi_inv) * 1.0f / (4.0f * M_PIf * M_PIf);
//   props1.one_over_three_ext = (1.0) / (3.0f * props1.extinction);

//   // Model param
//   float3 w12 = normalize(make_float3(0.0f, -0.5, -0.5f));
//   float3 n = make_float3(0.0f, 0.0f, 1.0f);
//   float3 xi = make_float3(0.0f);
//   float3 xo = make_float3(0.0f, 1.0f, 0.0f);

//   float3 subsurf = dipole(length(xo - xi), props1);
//   printf("%f\n", subsurf.x);
//   return;

//   float result = 0.f;
//   int width = 10;
//   for (int i = 0; i < width; i++) {
//     for (int j = 0; j < width; j++) {
//       float3 xo = make_float3(static_cast<float>(i) / width,
//                               static_cast<float>(j) / width, 0.0f);
//       float3 subsurf = dipole(length(xo - xi), props1);
//       result += subsurf.x;
//     }
//   }
//   printf("%f\n", result);
// }

// SUTILFN float3 dipole_bssrdf(const float3 &xi, const float3 &ni,
//                              const float3 &wi, const float3 &xo,
//                              const float3 &no, const float3 &w21,
//                              const HitGroupData *data, RNDTYPE &t) {
//   /* test_dipole(properties); */
//   /* return make_float3(0.0f); */
//   const OptProps &properties = data->optprops;
//   float3 w12;
//   float3 T =
//       btdf_sample(data->mtl_inside, wi, ni, properties.n1_over_n2, w12, t);
//   if (!non_zero(T)) {
//     return make_float3(0.0f);
//   }

//   float3 Rd =
//       dipole(length(xo - xi), properties) * properties.one_over_4PI * M_1_PIf;
//   return Rd * T;
// }

