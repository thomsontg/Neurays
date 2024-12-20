const float precis = 1.0e-6f;

float sdspheres(const float3& p, const float &ns, const float& density, const float3& center, const bool& invert)
{
  // const unsigned int spheres = 167;
  const float3 pi0 = floor(p - 0.5f);
  float result = 1.0e10f;
  for(unsigned int i = 0; i < 8; ++i)
  {
    const float3 corner = make_float3(i&1, (i>>1)&1, (i>>2)&1);
    const float3 pi = pi0 + corner;
    const int3 pii = make_int3 (pi.x, pi.y, pi.z);
    // unsigned int t = 4*static_cast<unsigned int>(fabsf(pi.x + pi.y*1000.0f + pi.z*576.0f));
    unsigned int t = 4u*static_cast<unsigned int>(pii.x*29 + pii.y*23 + pii.z*19 +pii.x*pii.x*17+pii.y*pii.y*13+pii.z*pii.z*37+pii.x*pii.y*pii.z*19);
    unsigned int spheres = rnd(t) < density ? 1 : 0;
    // unsigned int spheres = 167;
    for(int j = 0; j < spheres; ++j)
    {
      float r = (rnd(t)*0.0019f + 0.0001f)/* / ns*/;
      float3 xi = make_float3(rnd(t), rnd(t), rnd(t));
      float3 x = pi + xi;
      r *= 1000/length(x - center);
      result = fmin(result, length(p - x) - r);
    }
  }
  if(invert)
    return -result;
  else
    return result;
}

float3 calcNormal(const float3& pos, const float& ns, const float& density, const float3& center, const bool& invert)
{
  float2 e = make_float2(1.0f, -1.0f)*0.5773f*precis;
  return normalize(
    make_float3(e.x, e.y, e.y)*sdspheres(pos + make_float3(e.x, e.y, e.y), ns, density, center, invert) +
    make_float3(e.y, e.y, e.x)*sdspheres(pos + make_float3(e.y, e.y, e.x), ns, density, center, invert) +
    make_float3(e.y, e.x, e.y)*sdspheres(pos + make_float3(e.y, e.x, e.y), ns, density, center, invert) +
    make_float3(e.x)*sdspheres(pos + make_float3(e.x), ns, density, center, invert));
}

float raycast_ns(const float3& ro, const float3& rd, float tmin, float tmax, float ns, const float& density, const float3& center, const bool& invert)
{
  // ray marching
  float t = tmin;
  float d = sdspheres(ro + t*rd, ns, density, center, invert);
  const float sgn = copysignf(1.0f, d);
  // if(sgn >= 0.0f)
  //   return 100.0f;
  for(unsigned int i = 0; i < 10000u; ++i)
  {
    if(fabsf(d) < precis*t || t > tmax) break;
    t += sgn*d; // *1.2f;
    d = sdspheres(ro + t*rd, ns, density, center, invert);
  }
  return t; //t < tmax ? t : 1.0e16f;
}

float fresnel(const float3& ray_dir, const float3& n, const float& n1_over_n2, float& cos_theta_t)
{
    // Compute Fresnel reflectance (R) and trace refracted ray if necessary
  float cos_theta_o = dot(-ray_dir, n);
  float R = 1.0f;
  float sin_theta_t_sqr = n1_over_n2*n1_over_n2*(1.0f - cos_theta_o*cos_theta_o);
  if(sin_theta_t_sqr < 1.0f)
  {
    cos_theta_t = sqrtf(1.0f - sin_theta_t_sqr);
    R = fresnel_R(cos_theta_o, cos_theta_t, n1_over_n2);
  }
  return R;
}

float get_sigma_t(const HitGroupData* HGD, int& color, bool& in)
{
  float3 t;
  if(in) t = HGD->mtl_inside.scat +  HGD->mtl_inside.absp;
  else t = HGD->mtl_inside_2.scat +  HGD->mtl_inside_2.absp;
  return *(&t.x + color);
}

float get_alb(const HitGroupData* HGD, int& color, bool& in)
{
  float3 t;
  if(in) t = HGD->mtl_inside.scat / (HGD->mtl_inside.scat +  HGD->mtl_inside.absp);
  else t = HGD->mtl_inside_2.scat / (HGD->mtl_inside_2.scat +  HGD->mtl_inside_2.absp);
  return *(&t.x + color);
}

float get_g(const HitGroupData* HGD, int& color, bool& in)
{
  float3 t;
  if(in) t = HGD->mtl_inside.asym;
  else t = HGD->mtl_inside_2.asym;
  return *(&t.x + color);
}

closesthit_fn __closesthit__network()
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

  // Retrieve ray and hit info
  float3 ray_dir = optixGetWorldRayDirection();
  float3 x = geom.P;
  float3 n = normalize(geom.N);
  float dist = optixGetRayTmax();
  
  const float noise_scale = hit_group_data->mtl_inside.noise_scale;
  const float density = hit_group_data->mtl_inside.density;
  
  float weight = 1.0f;
  int colorband = -1;
  bool inside_main = true;

    // Compute cosine of the angle of observation
  float cos_theta_o = dot(-ray_dir, n);
  bool inside = cos_theta_o < 0.0f;
  float cos_theta_t;
  
  if(spheres(x) < 0)
  {
    ior2 = hit_group_data->mtl_inside.ior;
    extinction2 = hit_group_data->mtl_inside.scat +  hit_group_data->mtl_inside.absp;
    albedo2 = hit_group_data->mtl_inside.scat / extinction2;
    asymmetry2 = hit_group_data->mtl_inside.asym;
    inside_main = true;
  }
  else
  {
    ior2 = hit_group_data->mtl_inside_2.ior;
    extinction2 = hit_group_data->mtl_inside_2.scat +  hit_group_data->mtl_inside_2.absp;
    albedo2 = hit_group_data->mtl_inside_2.scat / extinction2;
    asymmetry2 = hit_group_data->mtl_inside_2.asym;
    inside_main = false;
  }
  
  // Compute relative index of refraction
  float n1_over_n2 = ior1/ior2;
  float n2_over_n3 = hit_group_data->mtl_inside.ior/hit_group_data->mtl_inside_2.ior;
  if(!inside_main) n2_over_n3 = 1/n2_over_n3;

  bool scatter = hit_group_data->mtl_outside.illum == 19;
  if(inside)
  {
    n1_over_n2 = 1.0f/n1_over_n2;
    n = -n;
    cos_theta_o = -cos_theta_o;
    scatter = hit_group_data->mtl_inside.illum == 19;
  }

  float R = fresnel(ray_dir, n, n1_over_n2, cos_theta_t);

  // Russian roulette to select reflection or refraction
  float xi = rnd(t);
  float3 wi;

  if(xi < R){
    wi = reflect(ray_dir, n);
      // Trace the new ray
    PayloadRadiance payload;
    payload.depth = depth + 1;
    payload.seed = t;
    payload.emit = 1;
    traceRadiance(lp.handle, x, wi, 1.0e-4f, 1.0e16f, &payload);
    setPayloadResult(payload.result);
    return;
  }
  else{
    wi = n1_over_n2*(cos_theta_o*n + ray_dir) - n*cos_theta_t;
  }

  // Ray Loop
  if(scatter)
  {
    // Calculate Transmission
    weight = 3.0f;
    colorband = int(rnd(t)*weight);
    float sigma_t = get_sigma_t(hit_group_data, colorband, inside_main);
    float alb = get_alb(hit_group_data, colorband, inside_main);
    float g = get_g(hit_group_data, colorband, inside_main);
    float trans_min = 1.0e-4f;
    float trans_max = -log(1.0f - rnd(t))/sigma_t;
    
    if(trans_max < dist)
    {
      unsigned int scatter_depth = depth;
      do{
        if(rnd(t) > alb)
          return;
        // Cast ray
        float s = raycast_ns(x*noise_scale, wi, 0.01f, dist*noise_scale, noise_scale, density, lp.geom_center, lp.invert_distances)/noise_scale;

        // Change Direction if hit
        if(s < trans_max)
        {
          x += s * wi;
          n = calcNormal(x*noise_scale, noise_scale, density, lp.geom_center, lp.invert_distances);

          R = fresnel(ray_dir, n, n2_over_n3, cos_theta_t);
          xi = rnd(t);
          if(xi < R){
            wi = reflect(ray_dir, n);
          }
          else{
            wi = n1_over_n2*(cos_theta_o*n + ray_dir) - n*cos_theta_t;
            n2_over_n3 = 1/n2_over_n3;
            inside_main = !inside_main;

            sigma_t = get_sigma_t(hit_group_data, colorband, inside_main);
            alb = get_alb(hit_group_data, colorband, inside_main);
            g = get_g(hit_group_data, colorband, inside_main);
          }
          if(rnd(t) > alb) return;
        }
        // else return
        else
        {
          x += trans_max*ray_dir;
          wi = sample_HG(ray_dir, g, t);
        }
        
        inside = !traceOcclusion(lp.handle, x, ray_dir, trans_min, trans_max);
        trans_max = -log(1.0f - rnd(t))/sigma_t;
        trans_min = 0.0f;
      }
      while(inside && ++scatter_depth < 500u);
      if(inside) return;
      
      PayloadFeeler feeler;      
      traceFeeler(lp.handle, x, ray_dir, trans_min, trans_max, &feeler);
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

  // Trace the new ray
  PayloadRadiance payload;
  payload.depth = depth + 1;
  payload.seed = t;
  payload.emit = 1;
#ifdef PASS_PAYLOAD_POINTER
  payload.hit_normal = n;
#endif
  traceRadiance(lp.handle, x, wi, 1.0e-4f, 1.0e16f, &payload);

  // payload.result *= rho_d;

  if(colorband >= 0)
  {
    // *(&payload.result.x + colorband) *= weight;
    // *(&payload.result.x + (colorband + 1)%3) = 0.0f;
    // *(&payload.result.x + (colorband + 2)%3) = 0.0f;
  }
  setPayloadResult(payload.result);
}