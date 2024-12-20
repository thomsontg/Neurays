SUTILFN float dir_error(float3 ref, float3 val)
{
  float3 x =  ref-val;
  float res = x.x * x.x + x.y * x.y + x.z * x.z;
  return res;
}


miss_fn __miss__radiance() {

  unsigned int emit = getPayloadEmit();
  const float3 ray_dir = optixGetWorldRayDirection();
  switch (launch_params.missType) {
  case MISS_CONSTANT: {
    setPayloadResult(launch_params.miss_color);
    break;
  }
  case MISS_ENVMAP: {
    if (!emit) {
      setPayloadResult(make_float3(0.f));
      return;
    }
    float3 env_radiance = env_lookup(ray_dir);
    setPayloadResult(env_radiance);
    break;
  }
  case MISS_SUNSKY: {
    const LaunchParams &lp = launch_params;
    const float overcast = lp.sunsky.overcast;
    const float3 ray_dir = optixGetWorldRayDirection();
    float3 result = overcast * overcast_sky_color(ray_dir, lp.sunsky);
    // TODO fix this
    // if (overcast < 1.0f)  {
    //   result += (1.0f - overcast) * clear_sky_color(ray_dir, lp.sunsky);
    // }
    setPayloadResult(result);
    break;
  }
  case MISS_DIRECTION: {
    if (dir_error(normalize(ray_dir), normalize(launch_params.light_dir)) < 0.001f)
    {
      setPayloadResult(make_float3(50.0f));
    }
    else setPayloadResult(make_float3(0.0f));
    break;
  }
  }
}

miss_fn __miss__ray_direction() {
  const float3 ray_dir = optixGetWorldRayDirection();
  setPayloadResult(ray_dir);
}
