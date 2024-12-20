
closesthit_fn __closesthit__distance()
{
  SHADER_HEADER
  if(depth > lp.max_depth)
    return;

  setPayloadDistance(optixGetRayTmax());

  // Retrieve ray and hit info
  const float3 ray_dir = optixGetWorldRayDirection();
  const float3& x = geom.P;
  float3 n = normalize(geom.N);

  // Compute cosine of the angle of observation
  float cos_theta_o = dot(-ray_dir, n);

  bool from_inside = cos_theta_o < 0.0f;
  if(from_inside)
  {
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
    return;
  }

  // Trace the new ray
  setPayloadSeed(t);
  setPayloadDirection(ray_dir);
  setPayloadOrigin(x);
  setPayloadDead(0);
  setPayloadEmit(1);
  setPayloadResult(make_float3(0.0f));
  setPayloadIntersect(1);
}