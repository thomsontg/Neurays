extern "C" __global__ void __closesthit__volume()
{
#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  unsigned int depth = prd->depth;
  unsigned int& t = prd->seed;
#else
  unsigned int depth = getPayloadDepth();
  unsigned int t = getPayloadSeed();
#endif
  const LaunchParams& lp = launch_params;
  if(depth > lp.max_depth)
    return;

  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry);

  // Retrieve material data
  const float s = 1.0f/hit_group_data->mtl_inside.shininess;
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
  float3 x = geom.P;
  float3 n = normalize(geom.N);
  float dist = optixGetRayTmax();

  // Compute cosine of the angle of observation
  float cos_theta_o = dot(-ray_dir, n);
  bool inside = cos_theta_o < 0.0f;

  // Compute relative index of refraction
  float n1_over_n2 = ior1/ior2;
  bool scatter = hit_group_data->mtl_outside.illum == 13;
  if(inside)
  {
    n1_over_n2 = 1.0f/n1_over_n2;
    n = -n;
    cos_theta_o = -cos_theta_o;
    scatter = hit_group_data->mtl_inside.illum == 13;
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
      PayloadFeeler feeler;      
      do
      {
        if(rnd(t) > alb)
          return;
        x += tmax*ray_dir;
        tmax = -log(1.0f - rnd(t))/sigma_t;
        ray_dir = sample_HG(ray_dir, g, t);
        inside = !traceFeeler(lp.handle, x, ray_dir, tmin, tmax, &feeler);
        tmin = 0.0f;
      }
      while(inside && ++scatter_depth < 500u);
      if(inside) return;

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

  // Do rough reflection/refraction and early exit using G
  float3 m, wi;
  float G = ggx_refract(m, wi, -ray_dir, n, cos_theta_o, n1_over_n2, s, t);
  if(rnd(t) > G)
    return;

  // Trace the new ray
  PayloadRadiance payload;
  payload.depth = depth + 1;
  payload.seed = t;
  payload.emit = 1;
#ifdef PASS_PAYLOAD_POINTER
  payload.hit_normal = n;
#endif
  traceRadiance(lp.handle, x, wi, 1.0e-4f, 1.0e16f, &payload);

  if(colorband >= 0)
  {
    *(&payload.result.x + colorband) *= weight;
    *(&payload.result.x + (colorband + 1)%3) = 0.0f;
    *(&payload.result.x + (colorband + 2)%3) = 0.0f;
  }
#ifdef PASS_PAYLOAD_POINTER
  prd->result = payload.result;
  prd->hit_normal = payload.hit_normal;
  prd->mi_dot_n = n.z;
#else
  setPayloadResult(payload.result);
#endif
}
