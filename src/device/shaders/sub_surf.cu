
closesthit_fn __closesthit__subsurf_brdf()
{
  SHADER_HEADER
  
  // Retrieve material data
  const float3& emission = hit_group_data->mtl_inside.emission;
  float3 rho_d = hit_group_data->mtl_inside.base_color_tex
    ? make_float3(tex2D<float4>(hit_group_data->mtl_inside.base_color_tex, geom.UV.x, geom.UV.y))
    : hit_group_data->mtl_inside.rho_d;
  const float ior = hit_group_data->mtl_inside.ior;
  const float n1_over_n2 = 1.0f/ior;
  const float3& albedo = hit_group_data->mtl_inside.alb;
  const float3& g = hit_group_data->mtl_inside.asym;

  // Retrieve hit info
  float3 ray_dir = optixGetWorldRayDirection();
  const float3 wo = -ray_dir;
  const float3& x = geom.P;
  const float3 n = normalize(geom.N);
  float3 result = emission;
  const float tmin = 1.0e-4f;
  const float tmax = 1.0e16f;
  const float cos_theta_o = fabsf(dot(wo, n));

  // Prepare Fresnel calculations
  const float recip_n_sqr = n1_over_n2*n1_over_n2;
  const float cos_theta_ot = sqrtf(1.0f - recip_n_sqr*(1.0f - cos_theta_o*cos_theta_o));
  const float T21 = 1.0f - fresnel_R(cos_theta_o, cos_theta_ot, n1_over_n2);
  const float3 wot = n1_over_n2*(cos_theta_o*n - wo) - n*cos_theta_ot;

#ifdef DIRECT
  // Lambertian reflection
  for(unsigned int i = 0; i < lp.lights.count; ++i)
  {
    const SimpleLight& light = lp.lights[i];
    const float3 wi = -light.direction;
    const float cos_theta_i = dot(wi, n);
    if(cos_theta_i > 0.0f)
    {
      const bool V = !traceOcclusion(lp.handle, x, wi, tmin, tmax);
      if(V)
      {
        const float cos_theta_it = sqrtf(1.0f - recip_n_sqr*(1.0f - cos_theta_i*cos_theta_i));
        const float T12 = 1.0f - fresnel_R(cos_theta_i, cos_theta_it, n1_over_n2);
        const float3 wit = n1_over_n2*(cos_theta_i*n - wi) - n*cos_theta_it;
        const float F = T12*T21;
        result += (albedo*F*p_hg(dot(wit, wot), g)/(cos_theta_it + cos_theta_ot) + rho_d*M_1_PIf)*light.emission*cos_theta_i;
      }
    }
  }
#endif
#ifdef INDIRECT
  // Indirect illumination
  float prob = (rho_d.x + rho_d.y + rho_d.z)/3.0f;
  if(rnd(t) < prob)
  {
    const float3 wi = sample_cosine_weighted(n, t);
    PayloadRadiance payload;
    payload.depth = depth + 1;
    payload.seed = t;
    payload.emit = 0;
    traceRadiance(lp.handle, x, wi, tmin, tmax, &payload);
    const float cos_theta_i = dot(wi, n);
    const float cos_theta_it = sqrtf(1.0f - recip_n_sqr*(1.0f - cos_theta_i*cos_theta_i));
    const float T12 = 1.0f - fresnel_R(cos_theta_i, cos_theta_it, n1_over_n2);
    const float3 wit = n1_over_n2*(cos_theta_i*n - wi) - n*cos_theta_it;
    const float F = T12*T21;
    result += (M_PIf*albedo*F*p_hg(dot(wit, wot), g)/(cos_theta_it + cos_theta_ot) + rho_d)*payload.result/prob;
  }
#endif
  setPayloadResult(result);
}