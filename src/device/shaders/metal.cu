
closesthit_fn __closesthit__metal()
{
  SHADER_HEADER
  
  if(depth > lp.max_depth)
    return;

  // Retrieve material data
  const complex3& recip_ior = hit_group_data->mtl_inside.c_recip_ior;

  // Retrieve ray and hit info
  const float3 ray_dir = optixGetWorldRayDirection();
  const float3& x = geom.P;
  float3 n = normalize(geom.N);

  // Do reflection or abosrption
  float3 result = make_float3(0.0f);
  float cos_theta_i = dot(-ray_dir, n);
  float3 R = make_float3(fresnel_R(cos_theta_i, recip_ior.x),
                         fresnel_R(cos_theta_i, recip_ior.y),
                         fresnel_R(cos_theta_i, recip_ior.z));
  float prob = (R.x + R.y + R.z)/3.0f;
  float xi = rnd(t);
  if(xi < prob)
  {
    // Mirror reflection
    PayloadRadiance payload;
    payload.depth = depth + 1;
    payload.seed = t;
    payload.emit = 1;
    traceRadiance(lp.handle, x, 2.0f*n*cos_theta_i + ray_dir, 1.0e-4f, 1.0e16f, &payload);
    result = R*payload.result/prob;
    setPayloadSeed(t);
    setPayloadDirection(2.0f*n*cos_theta_i + ray_dir);
    setPayloadOrigin(x);
    setPayloadAttenuation(getPayloadAttenuation() * R / prob);
    setPayloadDead(0);
    setPayloadEmit(1);
  }
}

