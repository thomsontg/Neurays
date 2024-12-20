#include "SamplerAlloc.h"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sutil/Exception.h>
#include <sutil/math/Matrix.h>
#include <sutil/sutil.h>
#include <Render/Scene/Scene.h>

namespace sutil
{

  void SamplerAlloc::cleanup()
  {
    // Destroy textures (base_color, metallic_roughness, normal)
    for (cudaTextureObject_t &texture : m_samplers)
      CUDA_CHECK(cudaDestroyTextureObject(texture));
    m_samplers.clear();

    for (cudaArray_t &image : m_images)
      CUDA_CHECK(cudaFreeArray(image));
    m_images.clear();
  }
  void SamplerAlloc::addSampler(cudaTextureAddressMode address_s,
                                cudaTextureAddressMode address_t,
                                cudaTextureFilterMode filter,
                                const int32_t image_idx, const bool is_hdr)
  {
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = getImage(image_idx);

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = address_s;
    tex_desc.addressMode[1] = address_t;
    tex_desc.filterMode = filter;
    tex_desc.readMode =
        is_hdr ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 1.0f;
    tex_desc.sRGB = 0; // Is the renderer using conversion to sRGB?

    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
    m_samplers.push_back(cuda_tex);
  }

  void SamplerAlloc::replaceSampler(cudaTextureAddressMode address_s,
                                    cudaTextureAddressMode address_t,
                                    cudaTextureFilterMode filter,
                                    const int32_t image_idx, const bool is_hdr, const int idx)
  {
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = getImage(image_idx);

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = address_s;
    tex_desc.addressMode[1] = address_t;
    tex_desc.filterMode = filter;
    tex_desc.readMode =
        is_hdr ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 1.0f;
    tex_desc.sRGB = 0; // Is the renderer using conversion to sRGB?

    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
    m_samplers[idx] = cuda_tex;
  }

  cudaTextureObject_t SamplerAlloc::getSampler(int32_t sampler_index) const
  {
    return m_samplers[sampler_index];
  }

  void SamplerAlloc::addImage(const int32_t width, const int32_t height,
                              const int32_t bits_per_component,
                              const int32_t num_components, const void *data)
  {
    // Allocate CUDA array in device memory
    int32_t pitch;
    cudaChannelFormatDesc channel_desc;
    if (bits_per_component == 8)
    {
      pitch = width * num_components * sizeof(uint8_t);
      channel_desc = cudaCreateChannelDesc<uchar4>();
    }
    else if (bits_per_component == 16)
    {
      pitch = width * num_components * sizeof(uint16_t);
      channel_desc = cudaCreateChannelDesc<ushort4>();
    }
    else if (bits_per_component == 32)
    {
      pitch = width * num_components * sizeof(float);
      channel_desc = cudaCreateChannelDesc<float4>();
    }
    else
    {
      throw Exception("Unsupported bits/component in glTF image");
    }

    cudaArray_t cuda_array = nullptr;
    CUDA_CHECK(cudaMallocArray(&cuda_array, &channel_desc, width, height));
    CUDA_CHECK(cudaMemcpy2DToArray(cuda_array,
                                   0, // X offset
                                   0, // Y offset
                                   data, pitch, pitch, height,
                                   cudaMemcpyHostToDevice));
    m_images.push_back(cuda_array);
  }

  void SamplerAlloc::replaceImage(const int32_t width, const int32_t height,
                                  const int32_t bits_per_component,
                                  const int32_t num_components, const void *data, const int idx)
  {
    // Allocate CUDA array in device memory
    int32_t pitch;
    cudaChannelFormatDesc channel_desc;
    if (bits_per_component == 8)
    {
      pitch = width * num_components * sizeof(uint8_t);
      channel_desc = cudaCreateChannelDesc<uchar4>();
    }
    else if (bits_per_component == 16)
    {
      pitch = width * num_components * sizeof(uint16_t);
      channel_desc = cudaCreateChannelDesc<ushort4>();
    }
    else if (bits_per_component == 32)
    {
      pitch = width * num_components * sizeof(float);
      channel_desc = cudaCreateChannelDesc<float4>();
    }
    else
    {
      throw Exception("Unsupported bits/component in glTF image");
    }

    cudaArray_t cuda_array = nullptr;
    CUDA_CHECK(cudaMallocArray(&cuda_array, &channel_desc, width, height));
    CUDA_CHECK(cudaMemcpy2DToArray(cuda_array,
                                   0, // X offset
                                   0, // Y offset
                                   data, pitch, pitch, height,
                                   cudaMemcpyHostToDevice));
    m_images[idx] = cuda_array;
  }

  cudaArray_t SamplerAlloc::getImage(int32_t image_index) const
  {
    return m_images[image_index];
  }
} // namespace sutil
