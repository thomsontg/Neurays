closesthit_fn __closesthit__mirror()
{
  SHADER_HEADER
  
  if(depth > lp.max_depth)
    return;
  
  // Retrieve ray and hit info
  const float3 ray_dir = optixGetWorldRayDirection();
  const float3& x = geom.P;
  const float3 n = normalize(geom.N);

  // Mirror reflection
  setPayloadSeed(t);
  setPayloadDirection(reflect(ray_dir, n));
  setPayloadOrigin(x);
  setPayloadDead(0);
  setPayloadEmit(1);
  setPayloadResult(make_float3(0.0f));
}