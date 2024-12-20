
closesthit_fn __closesthit__glossy()
{
  SHADER_HEADER
  
  if(depth > lp.max_depth)
    return;

  // Retrieve material data
  complex3 n1_over_n2 = hit_group_data->mtl_inside.c_recip_ior;
  //const float ior = hit_group_data->mtl_inside.ior;
  const float s = 1.0f/hit_group_data->mtl_inside.shininess;

  // Retrieve ray and hit info
  const float dist = optixGetRayTmax();
  const float3 ray_dir = optixGetWorldRayDirection();
  const float3& x = geom.P;
  const float3 wo = -ray_dir;
  float3 n = normalize(geom.N);
  const float tmin = 1.0e-4f;
  const float tmax = 1.0e16f;

  // Compute cosine of the angle of observation
  float cos_theta_o = dot(wo, n);
  bool inside = cos_theta_o < 0.0f;
  const float3& extinction = inside ? hit_group_data->mtl_inside.ext : hit_group_data->mtl_outside.ext;
  float3 Tr = expf(-extinction*dist);
  float Tr_prob = (Tr.x + Tr.y + Tr.z)/3.0f;
  if(rnd(t) > Tr_prob)
    return;

  // Compute relative index of refraction
  //float n1_over_n2 = 1.0f/ior;

  float3 result = make_float3(0.0f);
#ifdef DIRECT
  for(unsigned int i = 0; i < lp.lights.count; ++i)
  {
    const SimpleLight& light = lp.lights[i];
    const float3 wi = -light.direction;
    const float cos_theta_i = dot(wi, n);
    const bool V = !traceOcclusion(lp.handle, x, wi, tmin, tmax);
    if(V)
    {
      float3 bsdf = ggx_bsdf_cos(x, wi, wo, n, cos_theta_o, n1_over_n2, s);
      float bsdf_sum = bsdf.x + bsdf.y + bsdf.z;
      if(bsdf_sum > 0.0f) // && !isnan(bsdf_sum))
        result += bsdf*light.emission;
    }
  }
#endif
#ifdef INDIRECT
  if(inside)
  {
    n = -n;
    cos_theta_o = -cos_theta_o;
    n1_over_n2 = 1.0f/n1_over_n2;
  }

  // Rough reflection/refraction based on a normal distribution
  float3 wi;
  float3 G = ggx_refract(wi, wo, n, cos_theta_o, n1_over_n2, s, t);
  float G_sum = G.x + G.y + G.z;
  if(G_sum > 0.0f)
  {
    // Trace new ray
    PayloadRadiance payload;
    payload.depth = depth + 1;
    payload.seed = t;
    payload.emit = 1;
    traceRadiance(lp.handle, x, wi, tmin, tmax, &payload);
    result += G*payload.result;
  }
#endif
  if(inside)
    result *= Tr/Tr_prob;

  setPayloadResult(result);
}
