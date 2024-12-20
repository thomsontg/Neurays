SUTILFN float pow2(float val)
{
  return val * val;
}


SUTILFN void compute_squared_error(unsigned int pxl_idx)
{
  const LaunchParams &lp = launch_params;
  lp.error[0] +=  pow2(lp.accum_buffer[pxl_idx].x - lp.ref_img[pxl_idx].x) + 
                  pow2(lp.accum_buffer[pxl_idx].y - lp.ref_img[pxl_idx].y) + 
                  pow2(lp.accum_buffer[pxl_idx].z - lp.ref_img[pxl_idx].z);

}

SUTILFN float3 generate_camera_ray(RNDTYPE &t, const uint3 &launch_idx,
                                   const uint3 &launch_dims) {

  const LaunchParams &lp = launch_params;
  float3 direction;
  const float2 jitter = make_float2(rnd(t), rnd(t));
  const float2 idx = make_float2(launch_idx.x, launch_idx.y);
  const float2 res = make_float2(launch_dims.x, launch_dims.y);
  const float2 ip_coords = (idx + jitter) / res * 2.0f - 1.0f;
  direction = normalize(ip_coords.x * lp.U + ip_coords.y * launch_params.V + launch_params.W);
  
  return direction;
}

SUTILFN float3 trace_primary_ray(const float3 &direction, const float3 &origin,
                                 bool &Isect, RNDTYPE &t) {
  const LaunchParams &lp = launch_params;

  float3 result = make_float3(0.0f);
  float3 o = origin;
  float3 attenuation = make_float3(1.0f);

  // Setup payload
  PayloadRadiance payload;
  payload.direction = direction;
  payload.origin = make_float3(0.0f);
  payload.attenuation = attenuation;
  payload.depth = 0;
  payload.seed = t;
  payload.emit = 1;
  payload.dead = 1;
  payload.intersect = 0;

  for (;;) {
    // Trace camera ray
    traceRadiance(lp.handle, o, payload.direction, TMIN, TMAX, &payload);
    result += payload.result * attenuation;
    payload.depth++;
    if (payload.dead || payload.depth >= lp.max_depth) {
      break;
    }
    payload.dead = 1;
    o = payload.origin;
    attenuation = payload.attenuation;
  }
  Isect = payload.intersect;
  return result;
}

raygen_fn __raygen__pinhole()
{
  const LaunchParams& lp = launch_params;
  const uint3 launch_idx = optixGetLaunchIndex();
  const uint3 launch_dims = optixGetLaunchDimensions();
  const unsigned int frame = lp.subframe_index;
  const unsigned int pixel_idx = launch_idx.y*launch_dims.x + launch_idx.x;
  unsigned int t = tea<16>(pixel_idx, frame);
  bool Isect = 0;

  float3 result = make_float3(0.0f);
  for (int i = 0; i < lp.no_of_samples; i++) {
    // Generate camera ray (the center of each pixel is at (0.5, 0.5))
    float3 direction = generate_camera_ray(t, launch_idx, launch_dims);
    result += trace_primary_ray(direction, lp.eye, Isect, t);
  }
  result /= lp.no_of_samples;

  // // Handle fireflies
  // float max_res = fmaxf(fmaxf(result.x, result.y), result.z);
  // if (max_res > lp.firefly_threshold) {
  //   result *= lp.firefly_threshold / max_res;
  // }
  
  // Progressive update of image
  float3 accum_color = color::accumColor(result, frame,
                                         lp.accum_buffer, pixel_idx);

  lp.accum_buffer[pixel_idx] = make_float4(accum_color, Isect);

  if (lp.frame_buffer) 
  {
    lp.frame_buffer[pixel_idx] = color::toColor(accum_color, lp.tone_mapping);
  }

  if (lp.width == lp.ref_width && lp.height == lp.ref_height)
  {
    compute_squared_error(pixel_idx);
  }
}

#ifdef PASS_PAYLOAD_POINTER
SUTILFN float3 rainbow(float f)
{
  const float dx = 0.8f;
  float g = (6.0f - 2.0f*dx)*f + dx;
  float R = fmaxf(0.0f, (3.0f - abs(g - 4.0f) - abs(g - 5.0f))*0.5f);
  float G = fmaxf(0.0f, (4.0f - abs(g - 2.0f) - abs(g - 4.0f))*0.5f);
  float B = fmaxf(0.0f, (3.0f - abs(g - 1.0f) - abs(g - 2.0f))*0.5f);
  return make_float3(R, G, B);
}

SUTILFN float3 hsv2rgb(float h, float s, float v)
{
  float h6 = h*6.0;
  float frac = h6 - floor(h6);
  float4 ell = v*make_float4(1.0 - s, 1.0 - s*frac, 1.0 - s*(1.0 - frac), 1.0);
  return h6 < 1.0 ? make_float3(ell.w, ell.z, ell.x)
    : (h6 < 2.0 ? make_float3(ell.y, ell.w, ell.x)
      : (h6 < 3.0 ? make_float3(ell.x, ell.w, ell.z)
        : (h6 < 4.0 ? make_float3(ell.x, ell.y, ell.w)
          : (h6 < 5.0 ? make_float3(ell.z, ell.x, ell.w)
            : make_float3(ell.w, ell.x, ell.y)))));
}

SUTILFN float3 val2rainbow(float f)
{
  float t = clamp((log10(f) + 5.5f)/5.5f, 0.0f, 1.0f);
  float h = clamp((1.0f - t)*2.0f, 0.0f, 0.65f);
  return hsv2rgb(h, 1.0f, 1.0f);
}

raygen_fn __raygen__bsdf()
{
  const LaunchParams& lp = launch_params;
  const uint3 launch_idx = optixGetLaunchIndex();
  const uint3 launch_dims = optixGetLaunchDimensions();
  const unsigned int frame = lp.subframe_index;
  unsigned int pixel_idx = launch_idx.y*launch_dims.x + launch_idx.x;
  unsigned int t = tea<16>(pixel_idx, frame);

  // Generate camera ray (the center of each pixel is at (0.5, 0.5))
  const float2 jitter = make_float2(rnd(t), rnd(t));
  const float2 idx = make_float2(launch_idx.x, launch_idx.y);
  const float2 res = make_float2(launch_dims.x, launch_dims.y);
  float2 x = 2.0f*make_float2(launch_idx.x, launch_idx.y)/float(launch_dims.y) - 1.0f;

  // Trace camera ray
  PayloadRadiance payload;
  payload.result = make_float3(0.0f);
  payload.depth = 0;
  payload.seed = t;
  payload.emit = 1;
  float tmin = 1.0e-4f;
  float tmax = 1.0e16f;

#define REFLECTION

#if defined(PERSPECTIVE)
  const float2 ip_coords = (idx + jitter)/res*2.0f - 1.0f;
  const float3 direction = normalize(ip_coords.x*lp.U + ip_coords.y*lp.V + lp.W);
  traceRadiance(lp.handle, lp.eye, direction, tmin, tmax, &payload);
  float3 result = 0.5f*payload.result + 0.5f;
#elif defined(NORMALS)
  const float2 ip_coords = (idx + jitter)/res*2.0f - 1.0f;
  const float3 direction = lp.lights[0].direction;
  const float3 origin = make_float3(ip_coords.x, ip_coords.y, 0.0f)*lp.beam_factor - 10.0f*direction;
  traceRadiance(lp.handle, origin, direction, tmin, tmax, &payload);
  float3 result = 0.5f*payload.hit_normal + 0.5f;
#else

  float3 result = make_float3(0.0f);
  const float2 ip_coords = (idx + jitter)/res*2.0f - 1.0f;
  const float3 direction = lp.lights[0].direction;
  const float3 origin = make_float3(ip_coords.x, ip_coords.y, 0.0f)*lp.beam_factor - 10.0f*direction;
  traceRadiance(lp.handle, origin, direction, tmin, tmax, &payload);
  //if(payload.depth > 1)
  {
#ifdef REFLECTION
    if(payload.result.z > 0.0f)
    {
#else
    if(prd.result.z < 0.0f)
    {
#endif
      // normal distribution
      //uint2 new_idx = make_uint2(res*(0.5f + make_float2(payload.hit_normal.x, payload.hit_normal.y)*0.5f));
      //result = make_float3(0.2f);

      // geometric optics
      float mo_dot_wo = fabsf(dot(payload.result, payload.hit_normal));
      uint2 new_idx = make_uint2(res*(0.5f + make_float2(payload.result.x, payload.result.y)*0.5f));
      float denom = lp.surface_area*fmaxf(fabsf(direction.z*payload.mi_dot_n), 1.0e-8f);
      result = make_float3(mo_dot_wo/denom);

      // scalar diffraction theory
      //payload.result.z = 0.0f;
      //uint2 new_idx = make_uint2(res*(0.5f + make_float2(payload.hit_normal.x, payload.hit_normal.y)*0.5f));
      //float denom = total_area*fmaxf(abs(direction.z*payload.hit_normal.z), 1.0e-8f);
      //result = make_float4(payload.result*2.0f/denom, 0.0f);

      pixel_idx = new_idx.y*launch_dims.x + new_idx.x;
      x = 2.0f*make_float2(new_idx.x, new_idx.y)/float(launch_dims.y) - 1.0f;
    }
  }
#endif
  float3 accum_result = make_float3(lp.accum_buffer[pixel_idx]); // frame != 0 ? make_float3(lp.accum_buffer[pixel_idx]) : make_float3(0.0f);
  accum_result += result;
  lp.accum_buffer[pixel_idx] = make_float4(accum_result, 1.0f);
#if defined(NORMALS) || defined(PERSPECTIVE)
  lp.frame_buffer[pixel_idx] = make_rgba(accum_result/static_cast<float>(frame + 1));
#else
  //atomicAdd(&lp.accum_buffer[pixel_idx].x, fmaxf(result.x, 0.0f));
  float3 output = dot(x, x) < 1.0f ? val2rainbow(accum_result.x/static_cast<float>(frame + 1)) : make_float3(0.0f);
  lp.frame_buffer[pixel_idx] = make_rgba(output);
#endif
}
#endif


raygen_fn __raygen__sample_translucent()
{
  const LaunchParams& lp = launch_params;
  const uint3 launch_idx = optixGetLaunchIndex();
  const unsigned int frame = lp.subframe_index;
  const unsigned int idx = launch_idx.x;
  unsigned int t = tea<16>(idx, frame);
  PositionSample& sample = lp.translucent_samples[idx];

  // printf("Translucent\n");

  // sample a triangle
  unsigned int triangle_id = cdf_bsearch(rnd(t), lp.translucent_face_area_cdf);
  const uint3& idx_vxt = lp.translucent_idxs[triangle_id];
  const float3& v0 = lp.translucent_verts[idx_vxt.x];
  const float3& v1 = lp.translucent_verts[idx_vxt.y];
  const float3& v2 = lp.translucent_verts[idx_vxt.z];

  // sample a point in the triangle
  float3 bary = sample_barycentric(t);
  sample.pos = bary.x*v0 + bary.y*v1 + bary.z*v2;

  // compute the sample normal
  if(lp.translucent_norms.count > 0)
  {
    const float3& n0 = lp.translucent_norms[idx_vxt.x];
    const float3& n1 = lp.translucent_norms[idx_vxt.y];
    const float3& n2 = lp.translucent_norms[idx_vxt.z];
    sample.normal = normalize(bary.x*n0 + bary.y*n1 + bary.z*n2);
  }
  else
    sample.normal = normalize(cross(v1 - v0, v2 - v0));

  // evaluate incoming light
  float3 Le, wi;
  float tmin = 1.0e-4f;
  float tmax = 1.0e16f;
#ifdef DIRECT
#ifdef INDIRECT
  float xi = rnd(t);
  if(xi < 0.5f)
  {
#endif
//*
    // float dist;
    // ::sample(sample.pos, wi, Le, dist, t);

    // // Lambertian reflection
    // const bool V = !traceOcclusion(launch_params.handle, sample.pos, wi, tmin, dist - tmin);
    // if(V)
    //   Le *= fmaxf(dot(wi, sample.normal), 0.0f);
//*/

    unsigned int light_idx = lp.lights.count*rnd(t);
    SimpleLight light = lp.lights[light_idx];
    wi = light.direction;
    // Point Light test
    // wi = normalize(make_float3(100.0f) - sample.pos);
    // wi = normalize(wi) * 200.0f ;
    // wi = normalize(wi - sample.pos);
    // Directional light test
    // wi = normalize(make_float3(1.0f));
    Le = light.emission;
    float cos_theta_i = fmaxf(dot(wi, sample.normal), 0.0f);

    // trace a shadow ray to compute the visibility term
    const bool V = !traceOcclusion(lp.handle, sample.pos, wi, tmin, tmax);
    Le *= V*cos_theta_i*lp.lights.count;

#ifdef INDIRECT
  }
  else
  {
#endif
#endif
#ifdef INDIRECT
    wi = sample_cosine_weighted(sample.normal, t);

    PayloadRadiance payload;
    payload.depth = 0;
    payload.seed = t;
    payload.emit = 0;
    traceRadiance(lp.handle, sample.pos, wi, tmin, tmax, &payload);
    Le = payload.result*M_PIf;
#ifdef DIRECT
  }
  Le *= 2.0f;
#endif
#endif
  sample.dir = wi;
  sample.L = Le*lp.surface_area;
}

raygen_fn __raygen__bdsamples()
{
  const LaunchParams& lp = launch_params;

  const uint3 launch_idx = optixGetLaunchIndex();
  const uint3 launch_dims = optixGetLaunchDimensions();
  const unsigned int frame = lp.subframe_index;
  const unsigned int pixel_idx = launch_idx.y*launch_dims.x + launch_idx.x;
  unsigned int t = tea<16>(pixel_idx, frame);

  // lp.bd_samples[pixel_idx] = make_float3(pixel_idx);
  PayloadRadiance payload;

  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  // const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry);
  const float ior1 = hit_group_data->mtl_outside.ior;
  const float ior2 = hit_group_data->mtl_inside.ior;

  // unsigned int t = getPayloadSeed();

  // // Generate light samples
  const float3& extinction2 = hit_group_data->mtl_inside.scat +  hit_group_data->mtl_inside.absp;
  const float3& asymmetry2 = hit_group_data->mtl_inside.asym;

  const float3& light_pos = lp.lights[0].position;
  const float3& light_emi = lp.lights[0].emission;
  const float3& g_center = lp.geom_center;

  float3 light_dir = normalize(light_pos - g_center);

  float3 x_l = light_pos;
  // float n1_over_n2 = ior2/ior1;
  bool object_hit = false;

  do{
    float3 ray_dir = sample_hemisphere(light_dir, t);
          traceRadiance(lp.handle, light_pos, ray_dir, TMIN, TMAX, &payload);
    // if(traceOcclusion(lp.handle, light_pos, ray_dir, 1.0e-4f, 1.0e16f))
    // {
    //   PayloadFeeler feeler;
    //   traceFeeler(lp.handle, light_pos, ray_dir, 1.0e-4f, 1.0e16f, &feeler);
    //   x_l += feeler.dist * ray_dir;
    //   int colorband = int(rnd(t)*3.0);
    //   lp.bd_samples[pixel_idx] = x_l;
    //   // float n1_over_n2 = feeler.n1_over_n2;
    //   // float3 n = feeler.normal;
    //   // // Compute cosine of the angle of observation
    //   // float cos_theta_o = dot(-ray_dir, n);
    //   // float sigma_t = *(&extinction2.x + colorband);
    //   // float g = *(&asymmetry2.x + colorband);
    //   // float tmax = -log(1.0f - rnd(t))/sigma_t;
  
    //   // // Compute Fresnel reflectance (R) and trace refracted ray if necessary
    //   // float cos_theta_t;
    //   // float R = 1.0f;
    //   // const float sin_theta_t_sqr =
    //   //     n1_over_n2 * n1_over_n2 * (1.0f - cos_theta_o * cos_theta_o);
    //   // if (sin_theta_t_sqr < 1.0f) {
    //   //   cos_theta_t = sqrtf(1.0f - sin_theta_t_sqr);
    //   //   R = fresnel_R(cos_theta_o, cos_theta_t, n1_over_n2);
    //   // }

    //   // // Russian roulette to select reflection or refraction through the surface
    //   // const float xi = rnd(t);

    //   // if(xi > R){
    //   //   const float3 wi = n1_over_n2 * (cos_theta_o * n + ray_dir) - n * cos_theta_t;
    //   // }

    //   // ray_dir = sample_HG(ray_dir, g, t);
      
    //   // Compute cosine of the angle of observation
    //   // bool inside = dot(-ray_dir, normalize(feeler.normal)) < 0.0f;
    //   bool inside = true;
    //   if(inside)
    //     {
    //       // x_l += tmax*ray_dir;
    //       printf("Before Trace");
    //       printf("After Trace");
    //       // lp.bd_samples[pixel_idx] = x_l;
    //       object_hit = true;
    //     }
    // }
    object_hit = true;
  }while(!object_hit);
  // lp.bd_samples[pixel_idx] = light_pos;
    // printf("%d %d %d \n", x_l.x, x_l.y, x_l.z);
}


closesthit_fn __closesthit__bdsample()
{
  printf("Closesnt hit called \n");
}