
closesthit_fn __closesthit__directional()
{
  SHADER_HEADER

  if(depth > lp.max_depth)
    return;

  // Retrieve material data
  const float3& emission = hit_group_data->mtl_inside.emission;
  float3 rho_d = hit_group_data->mtl_inside.base_color_tex
    ? make_float3(tex2D<float4>(hit_group_data->mtl_inside.base_color_tex, geom.UV.x, geom.UV.y))
    : hit_group_data->mtl_inside.rho_d;

  // Retrieve hit info
  const float3& x = geom.P;
  const float3 n = normalize(geom.N);
  float3 result = emission;
  const float tmin = 1.0e-4f;
  const float tmax = 1.0e16f;

#ifdef DIRECT
  // Lambertian reflection
  for(unsigned int i = 0; i < lp.lights.count; ++i)
  {
    const SimpleLight& light = lp.lights[i];
    const float3 wi = -light.direction;
    const float cos_theta_i = dot(wi, n);
    if(cos_theta_i > 0.0f)
    {
      const bool V = !traceOcclusion(lp.handle, x, wi, tmin, tmax);
      if(V)
        result += rho_d*M_1_PIf*light.emission*cos_theta_i;
    }
  }
#endif
#ifdef INDIRECT
  // Indirect illumination
  float prob = (rho_d.x + rho_d.y + rho_d.z)/3.0f;
  if(rnd(t) < prob)
  {
    PayloadRadiance payload;
    payload.depth = depth + 1;
    payload.seed = t;
    payload.emit = 0;
    // payload.tmax = __float_as_int(optixGetRayTmax());
    traceRadiance(lp.handle, x, sample_cosine_weighted(n, t), tmin, tmax, &payload);
    result += rho_d*payload.result/prob;
  }
  
  float3 d = x - 	optixGetWorldRayOrigin();
  float dist = sqrtf(dot(d , d));
  // lp.hit_data[pixel_idx] = dist;
#endif
  setPayloadResult(result);
}