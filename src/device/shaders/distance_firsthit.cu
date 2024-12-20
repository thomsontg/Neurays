
closesthit_fn __closesthit__distance_firsthit()
{
  SHADER_HEADER
  if(depth > lp.max_depth)
    return;

  // Retrieve ray and hit info
  const float3 ray_dir = optixGetWorldRayDirection();
  const float3& x = geom.P;

  PayloadFeeler feeler;
  bool hit = traceFeeler(lp.handle, x, ray_dir, TMIN, TMAX, &feeler);
  // printf("%f : ", feeler.dist);
  if(hit)
  {
    setPayloadResult(make_float3(feeler.dist));
  }
  else
  {
    setPayloadResult(make_float3(0.0f));
  }
  setPayloadIntersect(1);
  return;

}