closesthit_fn __closesthit__normals() {
  SHADER_HEADER
  
  float3 result = normalize(geom.N) * 0.5f + 0.5f;
  setPayloadResult(result);
}
