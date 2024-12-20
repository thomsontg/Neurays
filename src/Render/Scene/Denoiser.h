#pragma once

#include <map>

#include <lib/oidn/include/OpenImageDenoise/oidn.hpp>
#include <src/sutil/math/vec_math.h>

class Denoiser {



public:
  struct BufferInfo {
    oidn::BufferRef buffer;
    oidn::Format format;
    unsigned int width;
    unsigned int height;
    bool active = false;
  };
  Denoiser() = default;
  ~Denoiser();

  void init(unsigned int width, unsigned int height,
            oidn::DeviceType type = oidn::DeviceType::Default);
  void denoise();

  void create_buffers(unsigned int width, unsigned int height);
  bool set_buffer(float3* buffer, const char*);
  bool set_buffer(float4* buffer, const char*);
  bool set_buffer_active(const char *);
  bool get_buffer(const char *, float3 *dst_buffer);
  float3* get_output_buffer();

  const std::map<const char *, Denoiser::BufferInfo> &get_buffers() {
    return m_buffers;
  }

private:
  // Denoiser
  oidn::DeviceRef m_device;
  oidn::FilterRef m_filter;
  std::map<const char *, Denoiser::BufferInfo> m_buffers;

  float3 *m_host_output_buffer;
};
