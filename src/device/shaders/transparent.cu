
closesthit_fn __closesthit__transparent()
{
  SHADER_HEADER
  if(depth > lp.max_depth)
    return;

  // Retrieve material data
  const float ior1 = hit_group_data->mtl_outside.ior;
  const float ior2 = hit_group_data->mtl_inside.ior;

  // Retrieve ray and hit info
  const float3 ray_dir = optixGetWorldRayDirection();
  const float3& x = geom.P;
  float3 n = normalize(geom.N);

  // Compute cosine of the angle of observation
  float cos_theta_o = dot(-ray_dir, n);

  // Compute relative index of refraction
  float n1_over_n2 = ior1/ior2;
  bool inside = cos_theta_o < 0.0f;
  if(inside)
  {
    n = -n;
    cos_theta_o = -cos_theta_o;
    n1_over_n2 = 1.0f/n1_over_n2;
  }

  // Compute Fresnel reflectance (R) and trace refracted ray if necessary
  float cos_theta_t;
  float R = 1.0f;
  float sin_theta_t_sqr = n1_over_n2*n1_over_n2*(1.0f - cos_theta_o*cos_theta_o);
  if(sin_theta_t_sqr < 1.0f)
  {
    cos_theta_t = sqrtf(1.0f - sin_theta_t_sqr);
    R = fresnel_R(cos_theta_o, cos_theta_t, n1_over_n2);
  }

  // Russian roulette to select reflection or refraction
  float xi = rnd(t);
  float3 wi = xi < R ? reflect(ray_dir, n) : n1_over_n2*(cos_theta_o*n + ray_dir) - n*cos_theta_t;

  // Trace the new ray
  setPayloadSeed(t);
  setPayloadDirection(wi);
  setPayloadOrigin(x);
  setPayloadDead(0);
  setPayloadEmit(1);
  setPayloadResult(make_float3(0.0f));
}