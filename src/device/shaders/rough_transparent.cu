// 02941 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2017
// Copyright (c) DTU Informatics 2017

// Closest hit program for drawing shading normals
closesthit_fn __closesthit__rough_transparent() {
  SHADER_HEADER
  
  const float3 ray_dir = optixGetWorldRayDirection();
  const float3 &xo = geom.P;
  float3 n = normalize(geom.N);
  const float3 wo = -ray_dir;
  const float thit = optixGetRayTmax();
  float n1_over_n2 = hit_group_data->mtl_outside.ior/hit_group_data->mtl_inside.ior;

  const float s = hit_group_data->mtl_inside.shininess;

  float cos_theta_o = dot(wo, n);
  bool inside = cos_theta_o < 0.0f;
  if (inside) {
    cos_theta_o = -cos_theta_o;
    n1_over_n2 = 1.0f / n1_over_n2;
  }

  float3 no = faceforward(n, wo, n);
  // Sample direction towards env map
  float3 wi;
  float3 Le;
  float pdf;
  float dist = TMAX;
  sampleLight(xo, wi, Le, dist, pdf, t);
  const bool V = !traceOcclusion(lp.handle, xo, wi, TMIN, dist);
  if (V) {
    float3 bsdf = ggx_bsdf_cos(xo, wi, wo, n, cos_theta_o, n1_over_n2, s);
    float bsdf_sum = bsdf.x + bsdf.y + bsdf.z;
    if (bsdf_sum > 0.0f && !isnan(bsdf_sum)) {
      result += bsdf * Le / pdf;
    }
  }

  float3 wt;
  float3 G = ggx_refract(wt, wo, no, dot(wo, no), n1_over_n2, s, t);
  float G_sum = G.x + G.y + G.z;
  if (G_sum > 0.0f && !isnan(G_sum)) {
    setPayloadSeed(t);
    setPayloadDirection(wt);
    setPayloadOrigin(xo);
    setPayloadDead(0);
    setPayloadAttenuation(getPayloadAttenuation() * G); 
    setPayloadEmit(0);
  }

  setPayloadResult(result);
}
