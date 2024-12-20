
closesthit_fn __closesthit__bidirectional_volume()
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
 
  float3 rho_d = hit_group_data->mtl_inside.base_color_tex
  ? make_float3(tex2D<float4>(hit_group_data->mtl_inside.base_color_tex, geom.UV.x, geom.UV.y))
  : hit_group_data->mtl_inside.rho_d;
   
  float3 bump_d = hit_group_data->mtl_inside.bump_map_tex
  ? make_float3(tex2D<float4>(hit_group_data->mtl_inside.bump_map_tex, geom.UV.x, geom.UV.y))
  : hit_group_data->mtl_inside.rho_d;

  // Retrieve ray and hit info
  float3 ray_dir = optixGetWorldRayDirection();
  float3 x = geom.P;
  float3 n = normalize(geom.N);
  float dist = optixGetRayTmax();

  float3 p_l = make_float3(0.0f);

  if(hit_group_data->mtl_inside.bump_map_tex)
  {
    bump_d = (2.0 * bump_d) - 1.0;

    float3 normal = normalize(geom.N);
    float3 tang = normalize(geom.T);
    
    tang = normalize(tang - dot(tang, normal) * normal);
    float3 btang = cross(tang, normal);
    
    n = normalize(make_float3(tang.x * bump_d.x + btang.x * bump_d.y + normal.x * bump_d.z,
                              tang.y * bump_d.x + btang.y * bump_d.y + normal.y * bump_d.z,
                              tang.z * bump_d.x + btang.z * bump_d.y + normal.z * bump_d.z));
  }

  // Compute cosine of the angle of observation
  float cos_theta_o = dot(-ray_dir, n);
  bool inside = cos_theta_o < 0.0f;

  // Compute relative index of refraction
  float n1_over_n2 = ior1/ior2;
  bool scatter = hit_group_data->mtl_outside.illum == 23;
  if(inside)
  {
    n1_over_n2 = 1.0f/n1_over_n2;
    n = -n;
    cos_theta_o = -cos_theta_o;
    scatter = hit_group_data->mtl_inside.illum == 23;
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

  float3 attenuation = make_float3(1.0f);
  if (colorband >= 0) {
    *(&attenuation.x + colorband) *= weight;
    *(&attenuation.x + (colorband + 1) % 3) = 0.0f;
    *(&attenuation.x + (colorband + 2) % 3) = 0.0f;
  }

  // Trace the new ray
  setPayloadSeed(t);
  setPayloadDirection(wi);
  setPayloadOrigin(x);
  setPayloadAttenuation(getPayloadAttenuation() * attenuation);
  setPayloadDead(0);
  setPayloadEmit(1);
  setPayloadResult(make_float3(0.0f));
}