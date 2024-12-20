#include "Denoiser.h"

#include <iostream>
#include <src/sutil/math/vec_math.h>

/********************************************/
Denoiser::~Denoiser() 
/********************************************/
{
  for (auto it = m_buffers.begin(); it != m_buffers.end(); it++) {
    it->second.buffer.release();
  }
  //delete[] m_host_output_buffer;
}

/********************************************/
void Denoiser::init(unsigned int width, unsigned int height,
                    oidn::DeviceType type)
/********************************************/
{
  m_device = oidn::newDevice(type);
  m_device.commit();

  create_buffers(width, height);

  m_host_output_buffer = new float3[width * height];
}

/********************************************/
void Denoiser::create_buffers(unsigned int width, unsigned int height)
/********************************************/
{
  Denoiser::BufferInfo color_buf = {
      m_device.newBuffer(width * height * sizeof(float3)), oidn::Format::Float3,
      width, height, false};
  Denoiser::BufferInfo albedo_buf = {
      m_device.newBuffer(width * height * sizeof(float3)), oidn::Format::Float3,
      width, height, false};
  Denoiser::BufferInfo normal_buf = {
      m_device.newBuffer(width * height * sizeof(float3)), oidn::Format::Float3,
      width, height, false};
  m_buffers["color"] = color_buf;
  m_buffers["albedo"] = albedo_buf;
  m_buffers["normal"] = normal_buf;
  m_buffers["output"] = color_buf;
}

/********************************************/
bool Denoiser::set_buffer(float3* host_buffer, const char* name)
/********************************************/
{

    auto it = m_buffers.find(name);
    if (it == m_buffers.end()) {
        printf("Couldn't find %s in buffer map (Denoiser)\n", name);
        return false;
    }

    BufferInfo& buffer_info = it->second;
    buffer_info.buffer.write(
        0, sizeof(float3) * buffer_info.width * buffer_info.height, host_buffer);
    buffer_info.active = true;
    return true;
}

/********************************************/
bool Denoiser::set_buffer(float4* host_buffer, const char* name)
/********************************************/
{

    auto it = m_buffers.find(name);
    if (it == m_buffers.end()) {
        printf("Couldn't find %s in buffer map (Denoiser)\n", name);
        return false;
    }

    BufferInfo& buffer_info = it->second;
    buffer_info.buffer.write(
        0, sizeof(float3) * buffer_info.width * buffer_info.height, host_buffer);
    buffer_info.active = true;
    return true;
}

/********************************************/
bool Denoiser::set_buffer_active(const char *name)
/********************************************/
{
  auto it = m_buffers.find(name);
  if (it == m_buffers.end()) {
    printf("Couldn't find %s in buffer map (Denoiser)\n", name);
    return false;
  }

  BufferInfo &buffer_info = it->second;
  buffer_info.active = true;
  return true;
}

/********************************************/
bool Denoiser::get_buffer(const char *name, float3 *dst_buffer)
/********************************************/
{
  auto it = m_buffers.find(name);
  if (it == m_buffers.end()) {
    printf("Couldn't find %s in buffer map (Denoiser)\n", name);
    return false;
  }

  BufferInfo &buffer_info = it->second;
  buffer_info.buffer.read(
      0, sizeof(float3) * buffer_info.width * buffer_info.height, dst_buffer);
  return true;
}

/********************************************/
float3 *Denoiser::get_output_buffer()
/********************************************/
{
  auto it = m_buffers.find("output");
  BufferInfo &buffer_info = it->second;
  buffer_info.buffer.read(
      0, sizeof(float3) * buffer_info.width * buffer_info.height,
      m_host_output_buffer);
  return m_host_output_buffer;
}

/********************************************/
void Denoiser::denoise()
/********************************************/
{
  m_filter = m_device.newFilter("RT");

  // Create buffers for denoising
  for (auto it = m_buffers.begin(); it != m_buffers.end(); it++) {
    BufferInfo &buffer_info = it->second;
    if (buffer_info.active) {
      m_filter.setImage(it->first, buffer_info.buffer, buffer_info.format,
                        buffer_info.width, buffer_info.height);
    }
    buffer_info.active = false;
  }
  m_filter.set("hdr", true);
  m_filter.commit();
  m_filter.execute();

  // Check for errors
  const char *error_message;
  if (m_device.getError(error_message) != oidn::Error::None) {
    std::cout << "Error: " << error_message << std::endl;
  }
}
