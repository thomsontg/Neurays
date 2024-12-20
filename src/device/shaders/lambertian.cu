
closesthit_fn __closesthit__lambertian()
{
  SHADER_HEADER
  
  // Retrieve material data
  // const float3& emission = hit_group_data->mtl_inside.emission;
  float3 rho_d = hit_group_data->mtl_inside.base_color_tex
    ? make_float3(tex2D<float4>(hit_group_data->mtl_inside.base_color_tex, geom.UV.x, geom.UV.y))
    : hit_group_data->mtl_inside.rho_d;

  float3 result;

  if(getPayloadIntersect())
  {
    result = getPayloadResult() * rho_d * M_1_PIf;
  }
  else{
    result = rho_d * M_1_PIf;
    setPayloadIntersect(1);
  }

  float3 wi = sample_hemisphere(geom.N, t);

  setPayloadSeed(t);
  setPayloadDirection(wi);
  setPayloadOrigin(geom.P);
  setPayloadAttenuation(getPayloadAttenuation() * rho_d * M_1_PIf);
  setPayloadDead(0);
  setPayloadEmit(0);

  setPayloadResult(make_float3(0.0f));
}
