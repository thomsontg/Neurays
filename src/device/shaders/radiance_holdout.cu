
closesthit_fn __closesthit__radiance_holdout()
{
  SHADER_HEADER
  
  if(depth > lp.max_depth)
    return;
  // Retrieve material data
  const float3& emission = hit_group_data->mtl_inside.emission;
  const float3 ray_dir = optixGetWorldRayDirection();
  float3 rho_d;

  if(lp.envmap)
  {
    rho_d = env_lookup(ray_dir);
  }
  else
  {
    rho_d = lp.miss_color;
  }

  // Retrieve hit info
  const float3& x = geom.P;
  const float3 n = normalize(geom.N);
  float3 result = make_float3(0.0f);
  float3 env_result = make_float3(0.0f);
  const float tmin = 1.0e-4f;
  const float tmax = 1.0e16f;

  float3 w_i, L_i = make_float3(0.0f);
  float cos_theta_i = -1.0f;
  while(cos_theta_i <= 0.0f && lp.envmap)
  {
    sample_environment(x, w_i, L_i, t);
    cos_theta_i = dot(w_i, n);
  }
  // Trace ray
  
  // printf("env : %f\n",w_i.x);
  PayloadRadiance payload;
  payload.depth = 0;
  payload.seed = t;
  payload.emit = 1;
  traceRadiance(lp.handle, x, w_i, tmin, tmax, &payload);
  float3 V = payload.result/env_lookup(w_i);
  result += V*rho_d;
  // printf("result : %f\n",result.x);
  
  setPayloadResult(result);
}
