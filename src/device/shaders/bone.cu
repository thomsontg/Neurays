float sdspheres_bone(const float3& p, const float &ns)
{
  const unsigned int spheres = 1;
  const float3 pi0 = floor(p - 0.5f);
  float result = 1.0e10f;
  for(unsigned int i = 0; i < 8; ++i)
  {
    const float3 corner = make_float3(i&1, (i>>1)&1, (i>>2)&1);
    const float3 pi = pi0 + corner;
    const int3 pii = make_int3 (pi.x, pi.y, pi.z);
    // unsigned int t = 4*spheres*static_cast<unsigned int>(fabsf(pi.x + pi.y*1000.0f + pi.z*576.0f));
    unsigned int t = 4u*spheres*static_cast<unsigned int>(pii.x*29 + pii.y*23 + pii.z*19 +pii.x*pii.x*17+pii.y*pii.y*13+pii.z*pii.z*37+pii.x*pii.y*pii.z*19);
    for(int j = 0; j < spheres; ++j)
    {
      // r *= 10;
      float3 xi = make_float3(rnd(t), rnd(t), rnd(t));
      float3 x = pi + xi;
      float3 c = make_float3(0.0f);
      float dist = sqrtf(powf(pi.x - c.x, 2)+ powf(pi.y - c.y, 2));
      float d = dist == 0 ? 0.0001f : dist;
      float r = (0.8/d) / ns;
      result = fmin(result, length(p - x) - r);
    }
  } 
  return -result;
}

float3 calcNormal_bone(const float3& pos, const float& ns)
{
  float2 e = make_float2(1.0f, -1.0f)*0.5773f*precis;
  return normalize(
    make_float3(e.x, e.y, e.y)*sdspheres_bone(pos + make_float3(e.x, e.y, e.y), ns) +
    make_float3(e.y, e.y, e.x)*sdspheres_bone(pos + make_float3(e.y, e.y, e.x), ns) +
    make_float3(e.y, e.x, e.y)*sdspheres_bone(pos + make_float3(e.y, e.x, e.y), ns) +
    make_float3(e.x)*sdspheres_bone(pos + make_float3(e.x), ns));
}

float raycast_ns_bone(const float3& ro, const float3& rd, float tmin, float tmax, float ns)
{
  // ray marching
  float t = tmin;
  float d = sdspheres_bone(ro + t*rd, ns);
  const float sgn = copysignf(1.0f, d);
  for(unsigned int i = 0; i < 10000u; ++i)
  {
    if(fabsf(d) < precis*t || t > tmax) break;
    t += sgn*d*1.0f; // *1.2f;
    d = sdspheres_bone(ro + t*rd, ns);
  }
  return t; //t < tmax ? t : 1.0e16f;
}

closesthit_fn __closesthit__bone()
{
  SHADER_HEADER

  if(depth > lp.max_depth)
    return;

  const unsigned int max_depth = 200;
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
  const bool inside_mesh = dot(w, n) > 0.0f;
  float sgn = 1.0f;
  if(inside_mesh)
  {
    PayloadFeeler feeler;
    float3 p = optixGetWorldRayOrigin();
    unsigned int i = 0;
    for(; i < max_depth; ++i)
    {
      float s = raycast_ns_bone(p*noise_scale, w, 0.01f, dist*noise_scale, noise_scale)/noise_scale;
      if(s >0.0f && s < dist)
      {
        p += s*w;
        n = calcNormal_bone(p*noise_scale, noise_scale);

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

        traceFeeler(lp.handle, p, w, tmin, tmax, &feeler);
        dist = feeler.dist;
      }
      else
        break;
    }
    if(i == max_depth)
      return;
    x = p + w*dist;
    sgn = copysignf(1.0f, sdspheres_bone(x*noise_scale, noise_scale));

    // Check for absorption
    const float3& extinction = sgn < 0.0f ? hit_group_data->mtl_inside.ext : hit_group_data->mtl_outside.ext;
    const float3 Tr = expf(-extinction*dist);
    const float prob = (Tr.x + Tr.y + Tr.z)/3.0f;
    if(rnd(t) > prob)
      return;
    result *= Tr/prob;

    n = feeler.normal;
  }
  else
  {
    // return;
    sgn = copysignf(1.0f, sdspheres_bone(x*noise_scale, noise_scale));
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
  }
  
  // Trace the new ray
  setPayloadSeed(t);
  setPayloadDirection(w);
  setPayloadOrigin(x);
  setPayloadAttenuation(getPayloadAttenuation());
  setPayloadDead(0);
  setPayloadEmit(1);
  setPayloadResult(make_float3(0.0f));
}
