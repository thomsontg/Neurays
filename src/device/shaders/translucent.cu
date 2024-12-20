
closesthit_fn __closesthit__translucent()
{
  SHADER_HEADER
  
  if(depth > lp.max_depth)
    return;

  // Retrieve material data
  const float ior1 = hit_group_data->mtl_outside.ior;
  const float3& albedo1 = hit_group_data->mtl_outside.alb;
  const float3& extinction1 = hit_group_data->mtl_outside.ext;
  const float3& asymmetry1 = hit_group_data->mtl_outside.asym;
  const float ior2 = hit_group_data->mtl_inside.ior;
  const float3& albedo2 = hit_group_data->mtl_inside.alb;
  const float3& extinction2 = hit_group_data->mtl_inside.ext;
  const float3& asymmetry2 = hit_group_data->mtl_inside.asym;

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
  }
  else
    wi = n1_over_n2*(cos_theta_o*no + ray_dir) - no*cos_theta_t;

  // Trace new ray
  PayloadRadiance payload;
  payload.depth = depth + 1;
  payload.seed = t;
  payload.emit = 1;
  traceRadiance(lp.handle, xo, wi, 1.0e-4f, 1.0e16f, &payload);
  result += payload.result;

  // Add subsurface scattering by evaluating the surface samples
  if(scatter)
  {
    const float3& sigma_t = inside ? extinction1 : extinction2;
    const float3& alpha = inside ? albedo1 : albedo2;
    const float3& g = inside ? asymmetry1 : asymmetry2;
    float3 sigma_s = alpha*sigma_t;
    float3 sigma_a = sigma_t - sigma_s;
    float3 sigma_s_prime = (1.0f - g)*sigma_s;
    float3 sigma_t_prime = sigma_s_prime + sigma_a;
    float3 alpha_prime = sigma_s_prime/sigma_t_prime;
    float3 D = 1.0f/(3.0f*sigma_t_prime);
    float3 sigma_tr = sqrtf(sigma_a/D);
    float F_dr = fresnel_diffuse(n1_over_n2);
    float A = (1.0f + F_dr)/(1.0f - F_dr);
    //float A = (1.0f - C_E(1.0f/n1_over_n2))/(2.0f*C_phi(1.0f/n1_over_n2));
    float3 zr = 1.0f/sigma_t_prime;
    float3 zv = zr + 4.0f*A*D;
    float3 flux_factor = alpha_prime/(4.0f*M_PIf*M_PIf);

    float3 subsurf = make_float3(0.0f);
    for(unsigned int i = 0; i < lp.translucent_no_of_samples; ++i)
    {
      const PositionSample& sample = lp.translucent_samples[i];
      const float3& xi = sample.pos;
      float r = length(xi - xo);
      float3 transport = expf(-sigma_tr*r);
      float prob = (transport.x + transport.y + transport.z)/3.0f;
      if(rnd(t) > prob)
        continue;

      const float3& wi = lp.lights[0].direction;
      const float3& ni = sample.normal;
      float cos_theta_i = fabsf(dot(wi, ni));
      float cos_theta_i_sqr = cos_theta_i*cos_theta_i;
      float sin_theta_t_sqr = n1_over_n2*n1_over_n2*(1.0f - cos_theta_i_sqr);
      if(sin_theta_t_sqr > 1.0f)
        continue;

      float cos_theta_t = sqrtf(1.0f - sin_theta_t_sqr);
      //float3 wt = n1_over_n2*(cos_theta_i*ni - wi) - ni*cos_theta_t;
      float T12 = 1.0f - fresnel_R(cos_theta_i, cos_theta_t, n1_over_n2);

      float3 dr = sqrtf(r*r + zr*zr);
      float3 dv = sqrtf(r*r + zv*zv);
      float3 S_real = zr*(1.0f + sigma_tr*dr)/(dr*dr*dr)*expf(-sigma_tr*dr);
      float3 S_virt = zv*(1.0f + sigma_tr*dv)/(dv*dv*dv)*expf(-sigma_tr*dv);
      float3 bssrdf = flux_factor*(S_real + S_virt);
      subsurf += T12*bssrdf*sample.L/prob;
    }
    result += subsurf/static_cast<float>(lp.translucent_no_of_samples);
  }
  result *= Tr/prob;

  
  const uint3 launch_idx = optixGetLaunchIndex();
  const uint3 launch_dims = optixGetLaunchDimensions();
  const unsigned int pixel_idx = launch_idx.y*launch_dims.x + launch_idx.x;
  lp.hit_data[pixel_idx] = dist;
  lp.pos[pixel_idx] = geom.P;

  setPayloadResult(result);
}