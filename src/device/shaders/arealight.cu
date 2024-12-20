closesthit_fn __closesthit__arealight()
{
  SHADER_HEADER

  // Retrieve material data
  const float3& emission = hit_group_data->mtl_inside.emission;
  float3 rho_d = hit_group_data->mtl_inside.base_color_tex
    ? make_float3(tex2D<float4>(hit_group_data->mtl_inside.base_color_tex, geom.UV.x, geom.UV.y))
    : hit_group_data->mtl_inside.rho_d;
  float3 alpha_d = hit_group_data->mtl_inside.alpha_map_tex
    ? make_float3(tex2D<float4>(hit_group_data->mtl_inside.alpha_map_tex, geom.UV.x, geom.UV.y))
    : hit_group_data->mtl_inside.rho_d;

  // float dist = optixGetRayTmax();

  // Retrieve ray and hit info
  const float3 ray_dir = optixGetWorldRayDirection();
  const float3& x = geom.P;
  const float3 n = normalize(geom.N)*copysignf(1.0f, -dot(geom.N, ray_dir));
  float3 result = emit ? emission : make_float3(0.0f);
  const float tmin = 1.0e-4f;
  const float tmax = 1.0e16f;
  const float xi = rnd(t);
  if(xi < alpha_d.x)
  {
    setPayloadDead(0);
  }
  setPayloadResult(rho_d);
  // setPayloadResult(rho_d);
}