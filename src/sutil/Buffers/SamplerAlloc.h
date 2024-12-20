#pragma once

#include <cuda_runtime.h>
#include <sutil/Preprocessor.h>
#include <sutil/sutilapi.h>

#include <optix.h>

#include <memory>
#include <string>
#include <vector>

namespace sutil
{

  class SamplerAlloc
  {

  public:
    SUTILAPI void cleanup();
    SUTILAPI void addSampler(cudaTextureAddressMode address_s,
                             cudaTextureAddressMode address_t,
                             cudaTextureFilterMode filter_mode,
                             const int32_t image_idx, const bool is_hdr = false);
    SUTILAPI void replaceSampler(cudaTextureAddressMode address_s,
                                 cudaTextureAddressMode address_t,
                                 cudaTextureFilterMode filter_mode,
                                 const int32_t image_idx, const bool is_hdr = false, const int idx = 0);
    SUTILAPI void addImage(const int32_t width, const int32_t height,
                           const int32_t bits_per_component,
                           const int32_t num_components, const void *data);
    SUTILAPI void replaceImage(const int32_t width, const int32_t height,
                               const int32_t bits_per_component,
                               const int32_t num_components, const void *data, const int idx = 0);

    SUTILAPI cudaTextureObject_t getSampler(int32_t sampler_index) const;
    SUTILAPI cudaArray_t getImage(int32_t image_index) const;

  private:
    std::vector<cudaTextureObject_t> m_samplers;
    std::vector<cudaArray_t> m_images;
  };

} // namespace sutil
