
closesthit_fn __closesthit__basecolor()
{
  SHADER_HEADER
  
  // Retrieve material data
  // const float3& emission = hit_group_data->mtl_inside.emission;
  float3 rho_d = hit_group_data->mtl_inside.base_color_tex
    ? make_float3(tex2D<float4>(hit_group_data->mtl_inside.base_color_tex, geom.UV.x, geom.UV.y))
    : hit_group_data->mtl_inside.rho_d;

  float3 result = rho_d;
  // float3 result = make_float3(1.0f);
  setPayloadResult(result);
}
