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

  class BufferAlloc
  {
  public:
    SUTILAPI void cleanup();
    SUTILAPI unsigned int addBuffer(const uint64_t buf_size, const void *data);
    SUTILAPI CUdeviceptr getBuffer(int32_t buffer_index) const;

  private:
    std::vector<CUdeviceptr> m_buffers;
  };
} // namespace sutil
