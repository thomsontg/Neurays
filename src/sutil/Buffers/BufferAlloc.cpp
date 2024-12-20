#include "BufferAlloc.h"

#include <optix.h>
#include <optix_stubs.h>

#include <sutil/Exception.h>

namespace sutil
{

  void BufferAlloc::cleanup()
  {
    // Free buffers for mesh (indices, positions, normals, texcoords)
    for (CUdeviceptr &buffer : m_buffers)
      CUDA_CHECK(cudaFree(reinterpret_cast<void *>(buffer)));
    m_buffers.clear();
  }

  unsigned int BufferAlloc::addBuffer(const uint64_t buf_size, const void *data)
  {
    CUdeviceptr buffer = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer), buf_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(buffer), data, buf_size,
                          cudaMemcpyHostToDevice));
    m_buffers.push_back(buffer);
    return static_cast<unsigned int>(m_buffers.size() - 1);
  }

  CUdeviceptr BufferAlloc::getBuffer(int32_t buffer_index) const
  {
    return m_buffers[buffer_index];
  }

} // namespace sutil
