#pragma once

#define record_contrib(pos, result)                                            \
  const float2 ip_coords =                                                     \
      make_float2(pos.x / lp.film_width, pos.y / lp.film_height);              \
  float2 idx = ip_coords + 0.5f;                                               \
  if (idx.x >= 0.0f && idx.x < 1.0f && idx.y >= 0.0f && idx.y < 1.0f) {        \
    const uint2 new_idx = make_uint2(idx * res);                               \
    const unsigned int scatter_idx = new_idx.y * launch_dims.x + new_idx.x;    \
    atomicAdd(&lp.accum_buffer[scatter_idx].x + p_offset.x, result);           \
  }

SUTILFN float accum_result(const uint2 &p_offset,
                                         const unsigned int pixel_idx) {
  const LaunchParams &lp = launch_params;
  // Accumulate result from previous run
  float4 curr_sum = lp.accum_buffer[pixel_idx];
  float new_sum = (*(&curr_sum.x + p_offset.y)) + curr_sum.z;
  *(&lp.accum_buffer[pixel_idx].x + p_offset.y) = 0.0f;
  lp.accum_buffer[pixel_idx].z = new_sum;
  return new_sum;
}

// Progressive update and display
SUTILFN void write_frame_buffer(const unsigned int frame,
                                              const float val,
                                              const unsigned int pixel_idx) {
  const LaunchParams &lp = launch_params;
  const float accum_result = frame > 0 ? val / frame : 0.0f;
  const float3 accum_color = launch_params.use_rainbow
                                 ? val2rainbow(accum_result)
                                 : make_float3(0.01f * accum_result);
  lp.frame_buffer[pixel_idx] = make_rgba(accum_color);
}

SUTILFN bool is_at_top_or_bottom_layer(unsigned int layer_idx) {
  const LaunchParams &lp = launch_params;
  if (lp.mode == REFLECT && layer_idx == 0) {
    return true;
  } else if (lp.mode == TRANSMIT && layer_idx == lp.n_layers - 1) {
    return true;
  }
  return false;
}

// Compute the distance from x_z to surf_z in direction w_z
SUTILFN float dist_to_surface(const float surf_z, const float x_z,
                                            const float w_z) {
  assert(w_z != 0);
  return (surf_z - x_z) / w_z;
}

SUTILFN float
compute_ior_up(const unsigned int current_layer_idx) {
  const LaunchParams &lp = launch_params;
  if (current_layer_idx == 0) {
    // Top boundary towards air
    return lp.layers[current_layer_idx].ior;
  } else {
    return lp.layers[current_layer_idx].ior /
           lp.layers[current_layer_idx - 1].ior;
  }
}

SUTILFN float
compute_ior_down(const unsigned int current_layer_idx) {
  const LaunchParams &lp = launch_params;
  if (current_layer_idx == lp.n_layers - 1) {
    // Bottom boundary towards air
    return lp.layers[current_layer_idx].ior;
  } else { // from li-1 to li
    return lp.layers[current_layer_idx].ior /
           lp.layers[current_layer_idx + 1].ior;
  }
}

SUTILFN struct Photon init_photon(const float3 &w,
                                                const float3 &x,
                                                const float weight,
                                                int layer_idx) {
  struct Photon photon;
  photon.w = w;
  photon.x = x;
  photon.weight = weight;
  photon.layer_idx = layer_idx;
  return photon;
}

// Searching in width direction
SUTILFN unsigned int cdf_bsearch(float xi) {

  const LaunchParams &lp = launch_params;
  unsigned int table_size = lp.n_light_flux * lp.n_light_flux;
  unsigned int middle = table_size = table_size >> 1;
  unsigned int odd = 0;
  while (table_size > 0) {
    odd = table_size & 1;
    table_size = table_size >> 1;
    unsigned int tmp = table_size + odd;
    float cdf = lp.light_flux_cdf[middle];
    float cdf1 = lp.light_flux_cdf[middle-1];
    middle = xi > cdf ? middle + tmp : (xi < cdf1 ? middle - tmp : middle);
  }

  return middle;
}
