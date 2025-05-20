#pragma once

#include <string.h>

#include <optix.h>

#include "sutil/shared/structs.h"
#include <sutil/Buffers/BufferAlloc.h>

class Network
{
public:
  Network(const std::string &network_path) : networkPath(network_path) {}

  void createNetworkBuffers(LaunchParams *lp, sutil::BufferAlloc &bufferalloc);

private:
  std::string networkPath;

  BufferView<float> createNetworkLayerBuffer(sutil::BufferAlloc &bufferalloc,
                                             unsigned int input_n,
                                             unsigned int output_n,
                                             const std::string &file);
  float *loadArrayFromFile(std::string &filename, unsigned int size);
};
