closesthit_fn __closesthit__volume() {
  SHADER_HEADER
  
  // Retrieve material data
  const float ior1 = hit_group_data->mtl_outside.ior;
  const float3 &extinction1 = hit_group_data->mtl_outside.ext;
  const float3 &albedo1 = hit_group_data->mtl_outside.alb;
  const float3 &asymmetry1 = hit_group_data->mtl_outside.asym;
  const float ior2 = hit_group_data->mtl_inside.ior;
  const float3 &extinction2 = hit_group_data->mtl_inside.ext;
  const float3 &albedo2 = hit_group_data->mtl_inside.alb;
  const float3 &asymmetry2 = hit_group_data->mtl_inside.asym;
  // Compute relative index of refraction (from the outside)
  float n1_over_n2 = ior1/ior2;

  // Retrieve ray and hit info
  float3 ray_dir = optixGetWorldRayDirection();
  float3 x = geom.P;
  float3 n = normalize(geom.N);
  const float dist = optixGetRayTmax();

  bool scatter = true;

  // Compute cosine of the angle of observation
  float cos_theta_o = dot(-ray_dir, n);
  bool inside = cos_theta_o < 0.0f;
  if (inside) {
    scatter = true;
    n1_over_n2 = 1/n1_over_n2;
    n = -n;
  }

  int colorband = -1;
  float weight = 1.0f;
  if (scatter) {
    weight = 3.0f;
    colorband = irnd_mod(t, 3);
    const float mfp = inside ? 1.0f / (*(&extinction2.x + colorband))
                             : 1.0f / (*(&extinction1.x + colorband));
    const float tmin = 0.f;
    float tmax = -log(1.0f - rnd(t)) * mfp;
    if (tmax < dist) {
      x = optixGetWorldRayOrigin();
      const float alb =
          inside ? *(&albedo2.x + colorband) : *(&albedo1.x + colorband);
      const float g =
          inside ? *(&asymmetry2.x + colorband) : *(&asymmetry1.x + colorband);
      unsigned int scatter_depth = 0;
      do {
        if (rnd(t) > alb) {
          return;
        }

        x += tmax * ray_dir;
        tmax = -log(rnd(t)) * mfp;
        ray_dir = sample_HG(ray_dir, g, t);
        scatter_depth++;
        inside = !traceOcclusion(lp.handle, x, ray_dir, tmin, tmax);
      } while (inside && (scatter_depth < lp.max_volume_bounces));
      if (inside) {
        return;
      }
      if (scatter_depth < 1) {
        return;
      }

      PayloadFeeler feeler;
      // float new_tmax = tmax + 0.5* tmax;       // Jeppe's suggestion for the t_max problem
      traceFeeler(lp.handle, x, ray_dir, tmin, tmax, &feeler);
      x += feeler.dist * ray_dir;
      n = feeler.normal;

      cos_theta_o = dot(-ray_dir, n);
      bool from_inside = cos_theta_o < 0.0f;
      if (from_inside) {
        // n1_over_n2 = 1.0f / n1_over_n2;
        n = -n;
        cos_theta_o = -cos_theta_o;
      }
    }
  }

  // Compute Fresnel reflectance (R) and trace refracted ray if necessary
  float cos_theta_t;
  float R = 1.0f;
  const float sin_theta_t_sqr =
      n1_over_n2 * n1_over_n2 * (1.0f - cos_theta_o * cos_theta_o);
  if (sin_theta_t_sqr < 1.0f) {
    cos_theta_t = sqrtf(1.0f - sin_theta_t_sqr);
    R = fresnel_R(cos_theta_o, cos_theta_t, n1_over_n2);
  }

  // Russian roulette to select reflection or refraction through the surface
  const float xi = rnd(t);
  const float3 wi =
      xi < R ? reflect(ray_dir, n)
             : n1_over_n2 * (cos_theta_o * n + ray_dir) - n * cos_theta_t;

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
