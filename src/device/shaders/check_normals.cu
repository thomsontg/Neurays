
closesthit_fn __closesthit__check_normals()
{
  SHADER_HEADER
  // Retrieve ray and hit info
  const float3 ray_dir = optixGetWorldRayDirection();
  const float3& x = geom.P;
  float3 n = normalize(geom.N);

  // Compute cosine of the angle of observation
  float cos_theta_o = dot(-ray_dir, n);

  bool from_inside = cos_theta_o < 0.0f;
  if(from_inside)
  {
    setPayloadResult(make_float3(1.0f, 0.0f, 0.0f));
  }
  else{
    setPayloadResult(make_float3(0.0f, 1.0f, 0.0f));
  }
}