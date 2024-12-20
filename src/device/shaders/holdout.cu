
closesthit_fn __closesthit__holdout()
{
  SHADER_HEADER
  
  // Retrieve material data
  const float3& emission = hit_group_data->mtl_inside.emission;
  const float3 ray_dir = optixGetWorldRayDirection();
  float3 rho_d = env_lookup(ray_dir);

  // Retrieve hit info
  const float3& x = geom.P;
  const float3 n = normalize(geom.N);
  float3 result = emission;
  const float tmin = 1.0e-4f;
  const float tmax = 1.0e16f;

// DIRECT
  // Lambertian reflection
  for(unsigned int i = 0; i < lp.lights.count; ++i)
  {
    const SimpleLight& light = lp.lights[i];
    const float3 wi = -light.direction;
    const float cos_theta_i = dot(wi, n);
    if(cos_theta_i > 0.0f)
    {
      const bool V = !traceOcclusion(lp.handle, x, wi, tmin, tmax);
      result += V*rho_d;
    }

  }
// INDIRECT
  // Indirect illumination
  const bool V = !traceOcclusion(lp.handle, x, sample_cosine_weighted(n, t), tmin, tmax);
  result += V*rho_d;

  // ifdef direct and indirect
  result *= 0.5f;
  setPayloadResult(result);
}
