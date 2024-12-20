#pragma once

#include <string.h>

#include <optix.h>
#include <sutil/Buffers/BufferAlloc.h>
#include <sutil/Buffers/SamplerAlloc.h>
#include <sutil/IO/HDRLoader.h>

#include "sutil/shared/structs.h"

class HDRLoader;

class Envmap
{
public:
  Envmap() {}

  void set_envmap_file(const std::string &envfile_)
  {
    envfile = envfile_;
    use_envmap_ = !envfile_.empty();
  }
  bool use_envmap() const { return use_envmap_; }
  bool load(sutil::BufferAlloc &b_alloc, sutil::SamplerAlloc &s_alloc, LaunchParams *lp);
  bool replace(sutil::BufferAlloc &b_alloc, sutil::SamplerAlloc &s_alloc, LaunchParams *lp);

private:
  std::string envfile;
  bool use_envmap_ = false;

  void createEnvMapBuffers(HDRLoader &hdr, sutil::BufferAlloc &b_alloc, LaunchParams *lp);
};
