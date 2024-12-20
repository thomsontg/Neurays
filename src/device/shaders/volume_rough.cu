closesthit_fn __closesthit__volume_rough()
{
  SHADER_HEADER
  if(depth > lp.max_depth)
    return;

  // Retrieve material data
  const float ior1 = hit_group_data->mtl_outside.ior;
  const float3& extinction1 = hit_group_data->mtl_outside.scat +  hit_group_data->mtl_outside.absp;
  const float3& albedo1 = hit_group_data->mtl_outside.scat / extinction1;
  const float3& asymmetry1 = hit_group_data->mtl_outside.asym;
  const float ior2 = hit_group_data->mtl_inside.ior;
  const float3& extinction2 = hit_group_data->mtl_inside.scat +  hit_group_data->mtl_inside.absp;
  const float3& albedo2 = hit_group_data->mtl_inside.scat / extinction2;
  const float3& asymmetry2 = hit_group_data->mtl_inside.asym;

  const float s = hit_group_data->mtl_inside.shininess;

  // Retrieve ray and hit info
  float3 ray_dir = optixGetWorldRayDirection();
  float3 x = geom.P;
  float3 n = normalize(geom.N);
  float dist = optixGetRayTmax();

  // Compute cosine of the angle of observation
  float cos_theta_o = dot(-ray_dir, n);
  bool inside = cos_theta_o < 0.0f;

  // Compute relative index of refraction
  float n1_over_n2 = ior1/ior2;
  bool scatter = hit_group_data->mtl_outside.illum == 21;
  if(inside)
  {
    n1_over_n2 = 1.0f/n1_over_n2;
    n = -n;
    cos_theta_o = -cos_theta_o;
    scatter = hit_group_data->mtl_inside.illum == 21;
  }
  int colorband = -1;
  float weight = 1.0f;
  if(scatter)
  {
    weight = 3.0f;
    colorband = int(rnd(t)*weight);
    float sigma_t = inside ? *(&extinction2.x + colorband) : *(&extinction1.x + colorband);
    float tmin = 1.0e-4f;
    float tmax = -log(1.0f - rnd(t))/sigma_t;
    if(tmax < dist)
    {
      x = optixGetWorldRayOrigin();
      float alb = inside ? *(&albedo2.x + colorband) : *(&albedo1.x + colorband);
      float g = inside ? *(&asymmetry2.x + colorband) : *(&asymmetry1.x + colorband);
      unsigned int scatter_depth = depth;
      do
      {
        if(rnd(t) > alb)
          return;
        x += tmax*ray_dir;
        tmax = -log(1.0f - rnd(t))/sigma_t;
        ray_dir = sample_HG(ray_dir, g, t);
        inside = !traceOcclusion(lp.handle, x, ray_dir, tmin, tmax);
        tmin = 0.0f;
      }
      while(inside && ++scatter_depth < 500u);
      if(inside) return;

      PayloadFeeler feeler;
      traceFeeler(lp.handle, x, ray_dir, tmin, tmax, &feeler);
      x += feeler.dist*ray_dir;
      n = feeler.normal;
      n1_over_n2 = feeler.n1_over_n2;
      cos_theta_o = dot(-ray_dir, n);
      if(cos_theta_o < 0.0f)
      {
        n1_over_n2 = 1.0f/n1_over_n2;
        n = -n;
        cos_theta_o = -cos_theta_o;
      }
    }
  }
  const float tmin = 1.0e-4f;
  const float tmax = 1.0e16f;
  const float3 wo = ray_dir;

  float3 result = make_float3(0.0f);
  float3 wt;
  float G = ggx_refract(wt, wo, n, dot(wo, n), n1_over_n2, s, t);
  if (G) {
    // Trace the new ray
    setPayloadSeed(t);
    setPayloadDirection(wt);
    setPayloadOrigin(x);
    setPayloadAttenuation(result + G * getPayloadAttenuation());
    setPayloadDead(0);
    setPayloadEmit(1);
    setPayloadResult(make_float3(0.0f));
  }

  float3 attenuation = make_float3(1.0f);
  if (colorband >= 0) {
    *(&attenuation.x + colorband) *= weight;
    *(&attenuation.x + (colorband + 1) % 3) = 0.0f;
    *(&attenuation.x + (colorband + 2) % 3) = 0.0f;
  }
}
