closesthit_fn __closesthit__occlusion()
{
  setPayloadOcclusion(true);
}


closesthit_fn __closesthit__feeler() {
  SHADER_HEADER
  
  const float ior1 = hit_group_data->mtl_outside.ior;
  const float ior2 = hit_group_data->mtl_inside.ior;
  setPayloadOcclusion(true);
  setPayloadDistance(optixGetRayTmax());
  setPayloadNormal(normalize(geom.N));
  setPayloadRelIOR(ior1 / ior2);
}
