
// #### FIX ME ####


closesthit_fn __closesthit__network_inf()
{
  SHADER_HEADER
  
  if(depth > lp.max_depth)
    return;

  // Retrieve material data
  const float ior1 = hit_group_data->mtl_outside.ior;
  const float3& extinction1 = hit_group_data->mtl_outside.scat +  hit_group_data->mtl_outside.absp;
  const float3& albedo1 = hit_group_data->mtl_outside.scat / extinction1;
  const float3& asymmetry1 = hit_group_data->mtl_outside.asym;
  // const float ior2 = hit_group_data->mtl_inside.ior;
  // const float3& extinction2 = hit_group_data->mtl_inside.scat +  hit_group_data->mtl_inside.absp;
  // const float3& albedo2 = hit_group_data->mtl_inside.scat / extinction2;
  // const float3& asymmetry2 = hit_group_data->mtl_inside.asym;
  float ior2 = 0.0f;
  float3 albedo2 = make_float3(0.0f);
  float3 extinction2 = make_float3(0.0f);
  float3 asymmetry2 = make_float3(0.0f);

  // Retrieve ray and hit info
  float3 ray_dir = optixGetWorldRayDirection();
  float3 x = geom.P;
  float3 n = normalize(geom.N);
  float dist = optixGetRayTmax();
  
  const float noise_scale = hit_group_data->mtl_inside.noise_scale;
  const float density = hit_group_data->mtl_inside.density;
  
  float weight = 1.0f;
  int colorband = -1;
  bool inside_main = true;

    // Compute cosine of the angle of observation
  float cos_theta_o = dot(-ray_dir, n);
  bool inside = cos_theta_o < 0.0f;
  float cos_theta_t;
  
  float input[9] = {x.x, x.y, x.z, -ray_dir.x, -ray_dir.y, -ray_dir.z, lp.lights[0].direction.x, lp.lights[0].direction.y, lp.lights[0].direction.z};
  float output[3] = {0.0, 0.0, 0.0};

  // Network inference
  evalNetwork(lp.inf_network, input, output);

  // Compute relative index of refraction
  float n1_over_n2 = ior1/ior2;
  float n2_over_n3 = hit_group_data->mtl_inside.ior/hit_group_data->mtl_inside_2.ior;
  if(!inside_main) n2_over_n3 = 1/n2_over_n3;


  // Russian roulette to select reflection or refraction
  float xi = rnd(t);
  float3 wi;

  if(xi < 1.0f){
    wi = reflect(ray_dir, n);
      // Trace the new ray
    PayloadRadiance payload;
    payload.depth = depth + 1;
    payload.seed = t;
    payload.emit = 1;
#ifdef PASS_PAYLOAD_POINTER
    payload.hit_normal = n;
#endif
    traceRadiance(lp.handle, x, wi, 1.0e-4f, 1.0e16f, &payload);
    setPayloadResult(payload.result);
    return;
  }
  else{
    wi = n1_over_n2*(cos_theta_o*n + ray_dir) - n*cos_theta_t;
  }

  // Trace the new ray
  PayloadRadiance payload;
  payload.depth = depth + 1;
  payload.seed = t;
  payload.emit = 1;
#ifdef PASS_PAYLOAD_POINTER
  payload.hit_normal = n;
#endif
  traceRadiance(lp.handle, x, wi, 1.0e-4f, 1.0e16f, &payload);

  // payload.result *= rho_d;

  if(colorband >= 0)
  {
    // *(&payload.result.x + colorband) *= weight;
    // *(&payload.result.x + (colorband + 1)%3) = 0.0f;
    // *(&payload.result.x + (colorband + 2)%3) = 0.0f;
  }
#ifdef PASS_PAYLOAD_POINTER
  prd->result = payload.result;
  prd->hit_normal = payload.hit_normal;
  // prd->mi_dot_n = n.z;
  prd->dist = optixGetRayTmax();
#else
  setPayloadResult(payload.result);
#endif
}