closesthit_fn __closesthit__fresnel()
{
  SHADER_HEADER

  if(depth > lp.max_depth)
    return;
  
  // Retrieve material data
  const float ior1 = hit_group_data->mtl_outside.ior;
  const float3& extinction1 = hit_group_data->mtl_outside.ext;
  const float ior2 = hit_group_data->mtl_inside.ior;
  const float3& extinction2 = hit_group_data->mtl_inside.ext;

  // Retrieve ray and hit info
  float3 ray_dir = optixGetWorldRayDirection();
  float3 wo = -ray_dir;
  float3 xo = geom.P;
  float3 no = normalize(geom.N);
  float dist = optixGetRayTmax();

  // Compute cosine of the angle of observation
  float cos_theta_o = dot(wo, no);
  bool inside = cos_theta_o < 0.0f;
  float3 result = make_float3(0.0f);

  // Compute relative index of refraction
  float n1_over_n2 = ior1/ior2;
  float3 ext = extinction1;
  bool scatter = hit_group_data->mtl_inside.illum == 14;
  if(inside)
  {
    n1_over_n2 = 1.0f/n1_over_n2;
    no = -no;
    cos_theta_o = -cos_theta_o;
    scatter = hit_group_data->mtl_outside.illum == 14;
    ext = extinction2;
  }

  // Evaluate surface reflection and direct transmission as in earlier shaders
  float3 Tr = expf(-ext*dist);
  float prob = (Tr.x + Tr.y + Tr.z)/3.0f;
  if(rnd(t) > prob)
    return;

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
  float3 wi;
  if(xi < R)
  {
    wi = reflect(ray_dir, no);
    scatter = false;

    // Trace the new ray
    setPayloadSeed(t);
    setPayloadDirection(wi);
    setPayloadOrigin(xo);
    setPayloadAttenuation(getPayloadAttenuation());
    setPayloadDead(0);
    setPayloadEmit(1);
    setPayloadResult(make_float3(0.0f));
  }

  setPayloadResult(result);
}