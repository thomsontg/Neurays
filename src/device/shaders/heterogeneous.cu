
// Signed distance function for randomly placed spheres inspired by sparse convolution noise
float spheres(const float3& p)
{
  const unsigned int spheres = 1;
  const float3 pi0 = floor(p - 0.5f);
  float result = 1.0e10f;
  for(unsigned int i = 0; i < 8; ++i)
  {
    const float3 corner = make_float3(i&1, (i>>1)&1, (i>>2)&1);
    const float3 pi = pi0 + corner;
    unsigned int t = 4*spheres*static_cast<unsigned int>(fabsf(pi.x + pi.y*1000.0f + pi.z*576.0f));
    for(int j = 0; j < spheres; ++j)
    {
      float r = rnd(t)*0.1f + 0.5f;
      float3 xi = make_float3(rnd(t), rnd(t), rnd(t));
      float3 x = pi + xi;
      result = fmin(result, length(p - x) - r);
    }
  } 
  return -result;
}

closesthit_fn __closesthit__heterogeneous()
{
  SHADER_HEADER
  
  if(depth > lp.max_depth)
    return;

  // Retrieve material data
  const float ior1 = hit_group_data->mtl_outside.ior;
  const float3& extinction1 = hit_group_data->mtl_outside.scat +  hit_group_data->mtl_outside.absp;
  const float3& albedo1 = hit_group_data->mtl_outside.scat / extinction1;
  const float3& asymmetry1 = hit_group_data->mtl_outside.asym;
  // const float ior2 = hit_group_data->mtl_inside.ior;
  // const float3& extinction2 = hit_group_data->mtl_inside.scat +  hit_group_data->mtl_inside.absp;
  // const float3& albedo2 = hit_group_data->mtl_inside.scat / extinction2;
  // const float3& asymmetry2 = hit_group_data->mtl_inside.asym;
  float ior2 = 0.0f;
  float3 albedo2 = make_float3(0.0f);
  float3 extinction2 = make_float3(0.0f);
  float3 asymmetry2 = make_float3(0.0f);
  
  if(spheres(optixGetWorldRayOrigin()) < 0)
  {
    ior2 = hit_group_data->mtl_inside.ior;
    extinction2 = hit_group_data->mtl_inside.scat +  hit_group_data->mtl_inside.absp;
    albedo2 = hit_group_data->mtl_inside.scat / extinction2;
    asymmetry2 = hit_group_data->mtl_inside.asym;
  }
  else
  {
    ior2 = hit_group_data->mtl_inside_2.ior;
    extinction2 = hit_group_data->mtl_inside_2.scat +  hit_group_data->mtl_inside_2.absp;
    albedo2 = hit_group_data->mtl_inside_2.scat / extinction2;
    asymmetry2 = hit_group_data->mtl_inside_2.asym;
  }

  float3 rho_d = hit_group_data->mtl_inside.base_color_tex
  ? make_float3(tex2D<float4>(hit_group_data->mtl_inside.base_color_tex, geom.UV.x, geom.UV.y))
  : hit_group_data->mtl_inside.rho_d;

  // Retrieve ray and hit info
  float3 ray_dir = optixGetWorldRayDirection();
  float3 x = geom.P;
  float3 n = normalize(geom.N);
  float dist = optixGetRayTmax();

  float3 p_l = make_float3(0.0f);

  // Compute cosine of the angle of observation
  float cos_theta_o = dot(-ray_dir, n);
  bool inside = cos_theta_o < 0.0f;

  // Compute relative index of refraction
  float n1_over_n2 = ior1/ior2;
  bool scatter = hit_group_data->mtl_outside.illum == 18;
  if(inside)
  {
    n1_over_n2 = 1.0f/n1_over_n2;
    n = -n;
    cos_theta_o = -cos_theta_o;
    scatter = hit_group_data->mtl_inside.illum == 18;
  }
  int colorband = -1;
  float weight = 1.0f;
  if(scatter)
  {
    weight = 3.0f;
    colorband = int(rnd(t)*weight);
    float sigma_t = inside ? *(&extinction2.x + colorband) : *(&extinction1.x + colorband);
    float alb = inside ? *(&albedo2.x + colorband) : *(&albedo1.x + colorband);
    float g = inside ? *(&asymmetry2.x + colorband) : *(&asymmetry1.x + colorband);
    float tmin = 1.0e-4f;
    float tmax = -log(1.0f - rnd(t))/sigma_t;
    if(tmax < dist)
    {
      x = optixGetWorldRayOrigin();
      unsigned int scatter_depth = depth;
      do
      {
        // if(spheres(x) < 0)
        // {
        //   ior2 = hit_group_data->mtl_inside.ior;
        //   extinction2 = hit_group_data->mtl_inside.scat +  hit_group_data->mtl_inside.absp;
        //   sigma_t = *(&extinction2.x + colorband);
        //   alb = *(&(hit_group_data->mtl_inside.scat.x) + colorband) / *(&(extinction2.x) + colorband);
        //   g = *(&(hit_group_data->mtl_inside.asym).x + colorband);
        //   tmax = -log(1.0f - rnd(t))/sigma_t;
        // }
        // else
        // {
        //   ior2 = hit_group_data->mtl_inside_2.ior;
        //   extinction2 = hit_group_data->mtl_inside_2.scat +  hit_group_data->mtl_inside_2.absp;
        //   sigma_t = *(&extinction2.x + colorband);
        //   alb = *(&(hit_group_data->mtl_inside_2.scat.x) + colorband) / *(&(extinction2.x) + colorband);
        //   g = *(&(hit_group_data->mtl_inside_2.asym).x + colorband);
        //   tmax = -log(1.0f - rnd(t))/sigma_t;
        // }
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
    // for(int i = 0; i < lp.lights.count; i++)
    // {
    //   const float3& li_p = lp.lights[i].position;
    //   const float3& li_I = lp.lights[i].emission;
    //   const float3& li_D = lp.lights[i].direction;
    //   if(lp.lights[i].type == POINTLIGHT)
    //   {
    //     float li_dist = sqrtf(powf(abs(x.x- li_p.x), 2) + powf(abs(x.y- li_p.y), 2) + powf(abs(x.z- li_p.z), 2));
    //     const float3& li_dir = -normalize(x- li_p);
    //     if (!traceOcclusion(lp.handle, x, li_dir, 1.0e-4f, 1.0e16f))
    //     {
    //       p_l += li_I / (powf(li_dist, 2.0f));
    //     }
    //   }
    //   else if(lp.lights[i].type == DIRECTIONALLIGHT)
    //   {
    //     if (!traceOcclusion(lp.handle, x, normalize(li_D), 1.0e-4f, 1.0e16f))
    //     {
    //       p_l += li_I;
    //     }
    //   }
    // }
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
  // float3 wi = xi < R ? reflect(ray_dir, n) : n1_over_n2*(cos_theta_o*n + ray_dir) - n*cos_theta_t;
  float3 wi = n1_over_n2*(cos_theta_o*n + ray_dir) - n*cos_theta_t;
// if(xi < R) return;
// float3 wi = n1_over_n2*(cos_theta_o*n + ray_dir) - n*cos_theta_t;
  float3 attenuation = make_float3(1.0f);
  if (colorband >= 0) {
    *(&attenuation.x + colorband) *= weight;
    *(&attenuation.x + (colorband + 1) % 3) = 0.0f;
    *(&attenuation.x + (colorband + 2) % 3) = 0.0f;
  }

  attenuation += p_l;

  // Trace the new ray
  setPayloadSeed(t);
  setPayloadDirection(wi);
  setPayloadOrigin(x);
  setPayloadAttenuation(getPayloadAttenuation() * attenuation);
  setPayloadDead(0);
  setPayloadEmit(1);
  setPayloadResult(make_float3(0.0f));

}
