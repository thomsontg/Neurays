closesthit_fn __closesthit__spheres()
{
  SHADER_HEADER
  
  const unsigned int max_depth = 200;
  if(depth > lp.max_depth)
    return;
  const float tmin = 1.0e-3f;
  const float tmax = 1.0e16f;
  float3 x = geom.P;
  float3 w = optixGetWorldRayDirection();
  float3 n = normalize(geom.N);
  float dist = optixGetRayTmax();
  float3 result = make_float3(1.0f);

  const float noise_scale = hit_group_data->mtl_inside.noise_scale;
  const float density = hit_group_data->mtl_inside.density;
  
  // Retrieve material data
  const float ior1 = hit_group_data->mtl_outside.ior;
  const float ior2 = hit_group_data->mtl_inside.ior;

  // Ray casting
  bool inside_mesh = dot(w, n) > 0.0f;
  float sgn = 1.0f;
  if(inside_mesh)
  {
    // if(!traceOcclusion(lp.handle, x, w, tmin, tmax)) result += make_float3(1000.0f, 0.0f, 0.0f);
    PayloadFeeler feeler;
    float3 p = optixGetWorldRayOrigin();
    unsigned int i = 0;
    for(; i < max_depth; ++i)
    {
      float s = raycast_ns(p*noise_scale, w, 0.01f, dist*noise_scale, noise_scale, density, lp.geom_center, lp.invert_distances)/noise_scale;
      if(s >0.0f && s < dist)
      {
        p += s*w;
        n = calcNormal(p*noise_scale, noise_scale, density, lp.geom_center, lp.invert_distances);

        // Compute cosine of the angle of observation
        float cos_theta_o = dot(-w, n);

        // Check for absorption
        const bool inside = cos_theta_o < 0.0f;
        const float3& extinction = inside ? hit_group_data->mtl_inside.ext : hit_group_data->mtl_outside.ext;
        const float3 Tr = expf(-extinction*s);
        const float prob = (Tr.x + Tr.y + Tr.z)/3.0f;
        if(rnd(t) > prob)
          return;
        result *= Tr/prob;

        // Compute relative index of refraction
        float n1_over_n2 = ior1/ior2;
        if(inside)
        {
          n = -n;
          cos_theta_o = -cos_theta_o;
          n1_over_n2 = 1.0f/n1_over_n2;
          // result = make_float3(1000.0f, 0.0f, 0.0f);
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
        w = xi < R ? reflect(w, n) : n1_over_n2*(cos_theta_o*n + w) - n*cos_theta_t;

        // inside_mesh = !traceOcclusion(lp.handle, p, w, tmin, tmax);
        // if(inside_mesh)
        // {
        //   result += make_float3(1000.0f, 0.0f, 0.0f);
        //   if(inside) break;
        //   else
        //   {
        //     n = -normalize(geom.N);
        //     cos_theta_o = - cos_theta_o;
        //     n1_over_n2 = 1.0f/n1_over_n2;

        //     R = 1.0f;
        //     sin_theta_t_sqr = n1_over_n2*n1_over_n2*(1.0f - cos_theta_o*cos_theta_o);
            
        //     if(sin_theta_t_sqr < 1.0f)
        //     {
        //       cos_theta_t = sqrtf(1.0f - sin_theta_t_sqr);
        //       R = fresnel_R(cos_theta_o, cos_theta_t, n1_over_n2);
        //     }
            
        //     // Russian roulette to select reflection or refraction
        //     xi = rnd(t);
        //     w = xi < R ? reflect(w, n) : n1_over_n2*(cos_theta_o*n + w) - n*cos_theta_t;
        //     if(xi >= R) break;
        //   }
        // }

        traceFeeler(lp.handle, p, w, tmin, tmax, &feeler);
        dist = feeler.dist;
      }
      else
        break;
    }
    if(i == max_depth)
      return;
    x = p + w*dist;
    sgn = copysignf(1.0f, sdspheres(x*noise_scale, noise_scale, density, lp.geom_center, lp.invert_distances));

    // Check for absorption
    const float3& extinction = sgn < 0.0f ? hit_group_data->mtl_inside.ext : hit_group_data->mtl_outside.ext;
    const float3 Tr = expf(-extinction*dist);
    const float prob = (Tr.x + Tr.y + Tr.z)/3.0f;
    if(rnd(t) > prob)
      return;
    result *= Tr/prob;

    n = feeler.normal;
      // result += make_float3(1000.0f, 0.0f, 0.0f);
  }
  else
  {
    // if(traceOcclusion(lp.handle, x, w, tmin, tmax)) result += make_float3(1000.0f, 0.0f, 0.0f);
    sgn = copysignf(1.0f, sdspheres(x*noise_scale, noise_scale, density, lp.geom_center, lp.invert_distances));
    if(sgn<0.0f)
    {
  
     // Compute cosine of the angle of observation
     float cos_theta_o = dot(-w, n);

     // Compute relative index of refraction
     float n1_over_n2 = sgn < 0.0f ? 1.0f/ior2 : 1.0f/ior1;
     if(inside_mesh)
     {
      n = -n;
      cos_theta_o = -cos_theta_o;
      n1_over_n2 = 1.0f/n1_over_n2;
      // result = make_float3(1000.0f, 0.0f, 0.0f);
     }

     // Compute Fresnel reflectance (R) and trace refracted ray if necessary
     float cos_theta_t;
     float R = 1.0f;
     const float sin_theta_t_sqr = n1_over_n2*n1_over_n2*(1.0f - cos_theta_o*cos_theta_o);
     if(sin_theta_t_sqr < 1.0f)
     {
      cos_theta_t = sqrtf(1.0f - sin_theta_t_sqr);
      R = fresnel_R(cos_theta_o, cos_theta_t, n1_over_n2);
     }

     // Russian roulette to select reflection or refraction
     float xi = rnd(t);
     w = xi < R ? reflect(w, n) : n1_over_n2*(cos_theta_o*n + w) - n*cos_theta_t;
    }
    
      // result += make_float3(1000.0f, 0.0f, 0.0f);
  }
  // Trace the new ray
  setPayloadSeed(t);
  setPayloadDirection(w);
  setPayloadOrigin(x);
  setPayloadDead(0);
  setPayloadEmit(1);
  setPayloadResult(result * getPayloadResult());
}
