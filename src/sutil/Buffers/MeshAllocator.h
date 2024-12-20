//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <vector_types.h>

#include <src/device/cuda/BufferView.h>
#include <src/device/cuda/GeometryData.h>

#include <sutil/math/Aabb.h>
#include <sutil/Exception.h>
#include <sutil/math/Matrix.h>
#include <sutil/Preprocessor.h>
#include <sutil/sutilapi.h>

#include <cuda_runtime.h>

#include <optix.h>

#include <memory>
#include <string>
#include <vector>

namespace sutil
{

  struct Instance
  {
    sutil::Matrix4x4 transform;
    sutil::Aabb world_aabb;

    unsigned int mesh_idx;
    struct sutil::Transform transforms;

    std::string material_name;
    int material_illum;
    unsigned int material_id;
    unsigned int instance_id;
  };

  struct MeshGroup
  {
    std::string name;

    std::vector<GenericBufferView> indices;
    std::vector<BufferView<float3>> positions;
    std::vector<BufferView<float3>> normals;
    std::vector<BufferView<float2>> texcoords;

    std::vector<int32_t> material_idx;
    std::string material_name;
    int material_illum;

    OptixTraversableHandle gas_handle = 0;
    CUdeviceptr d_gas_output = 0;

    sutil::Aabb object_aabb;
    sutil::Transform transform;
    sutil::Aabb world_aabb;

    unsigned int instance_count = 0;
  };

  class MeshAlloc
  {

  public:
    void cleanup()
    {
      for (auto mesh : m_meshes)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(mesh->d_gas_output)));
      m_meshes.clear();
      m_instances.clear();
    }

    void addInstance(std::shared_ptr<sutil::Instance> instance)
    {
      m_instances.push_back(instance);
      m_meshes[instance->mesh_idx].get()->instance_count++;
    }
    void addMesh(std::shared_ptr<sutil::MeshGroup> mesh)
    {
      m_meshes.push_back(mesh);
    }
    void delete_instance(int idx)
    {
      m_instances.erase(m_instances.begin() + idx);
    }
    void delete_all_instances(int idx)
    {
      bool deleted = false;
      for (int i = 0; i < m_instances.size(); i++)
      {
        do
        {
          if (m_instances.size() == 0)
            return;
          if (m_instances[i].get()->mesh_idx == idx)
          {
            m_instances.erase(m_instances.begin() + i);
            deleted = true;
          }
          else
            deleted = false;
        } while (deleted);
      }
    }

    const std::vector<std::shared_ptr<MeshGroup>> &get_meshes() const { return m_meshes; };
    const std::vector<std::shared_ptr<Instance>> &get_instances() const { return m_instances; };

  private:
    std::vector<std::shared_ptr<Instance>> m_instances;
    std::vector<std::shared_ptr<MeshGroup>> m_meshes;
  };

} // end namespace sutil
