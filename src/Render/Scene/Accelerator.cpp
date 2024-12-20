#pragma once

#include "Accelerator.h"

#include <iostream>
#include <vector>
#include <map>

#include <sutil/sutil.h>

/********************************************/
void Accelerator::build(const sutil::MeshAlloc &allocator,
                        const OptixDeviceContext &context)
/********************************************/
{
  build_mesh_accels(allocator.get_meshes(), context);
  build_instance_accel(allocator, context);
}

/********************************************/
void Accelerator::rebuild(const sutil::MeshAlloc &allocator,
                          const OptixDeviceContext &context)
/********************************************/
{
  build_instance_accel(allocator, context, false);
}

/********************************************/
void Accelerator::cleanup()
/********************************************/
{
  if (m_d_ias_output_buffer)
  {
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_ias_output_buffer)));
    m_d_ias_output_buffer = 0;
  }
}

/*****************************************/
void Accelerator::build_mesh_accels(
    const std::vector<std::shared_ptr<sutil::MeshGroup>> &meshes,
    const OptixDeviceContext &context)
/*****************************************/
{
  // Problem:
  // The memory requirements of a compacted GAS are unknown prior to building
  // the GAS. Hence, compaction of a GAS requires to build the GAS first and
  // allocating memory for the compacted GAS afterwards. This causes a
  // device-host synchronization point, potentially harming performance. This is
  // most likely the case for small GASes where the actual building and
  // compaction of the GAS is very fast. A naive algorithm processes one GAS at
  // a time with the following steps:
  // 1. compute memory sizes for the build process (temporary buffer size and
  // build buffer size)
  // 2. allocate temporary and build buffer
  // 3. build the GAS (with temporary and build buffer) and compute the
  // compacted size If compacted size is smaller than build buffer size (i.e.,
  // compaction is worth it):
  // 4. allocate compacted buffer (final output buffer)
  // 5. compact GAS from build buffer into compact buffer
  //
  // Idea of the algorithm:
  // Batch process the building and compaction of multiple GASes to avoid
  // host-device synchronization. Ideally, the number of synchronization points
  // would be linear with the number of batches rather than the number of GASes.
  // The main constraints for selecting batches of GASes are:
  // a) the peak memory consumption when batch processing GASes, and
  // b) the amount of memory for the output buffer(s), containing the compacted
  // GASes. This is also part of a), but is also important after the build
  // process. For the latter we try to keep it as minimal as possible, i.e., the
  // total memory requirements for the output should equal the sum of the
  // compacted sizes of the GASes. Hence, it should be avoided to waste memory
  // by allocating buffers that are bigger than what is required for a compacted
  // GAS.
  //
  // The peak memory consumption effectively defines the efficiency of the
  // algorithm. If memory was unlimited, compaction isn't needed at all. A lower
  // bound for the peak memory consumption during the build is the output of the
  // process, the size of the compacted GASes. Peak memory consumption
  // effectively defines the memory pool available during the batch building and
  // compaction of GASes.
  //
  // The algorithm estimates the size of the compacted GASes by a give
  // compaction ratio as well as the computed build size of each GAS. The
  // compaction ratio is defined as: size of compacted GAS / size of build
  // output of GAS. The validity of this estimate therefore depends on the
  // assumed compaction ratio. The current algorithm assumes a fixed compaction
  // ratio. Other strategies could be:
  // - update the compaction ration on the fly by do statistics on the already
  // processed GASes to have a better guess for the remaining batches
  // - multiple compaction rations by type of GAS (e.g., motion vs static),
  // since the type of GAS impacts the compaction ratio Further, compaction may
  // be skipped for GASes that do not benefit from compaction (compaction ratio
  // of 1.0).
  //
  // Before selecting GASes for a batch, all GASes are sorted by size (their
  // build size). Big GASes are handled before smaller GASes as this will
  // increase the likelihood of the peak memory consumption staying close to the
  // minimal memory consumption. This also increase the benefit of batching
  // since small GASes that benefit most from avoiding synchronizations are
  // built "together". The minimum batch size is one GAS to ensure forward
  // process.
  //
  // Goal:
  // Estimate the required output size (the minimal peak memory consumption) and
  // work within these bounds. Batch process GASes as long as they are expected
  // to fit into the memory bounds (non strict).
  //
  // Assumptions:
  // The inputs to each GAS are already in device memory and are needed
  // afterwards. Otherwise this could be factored into the peak memory
  // consumption. E.g., by uploading the input data to the device only just
  // before building the GAS and releasing it right afterwards.
  //
  // Further, the peak memory consumption of the application / system is
  // influenced by many factors unknown to this algorithm. E.g., if it is known
  // that a big pool of memory is needed after GAS building anyways (e.g.,
  // textures that need to be present on the device), peak memory consumption
  // will be higher eventually and the GAS build process could already make use
  // of a bigger memory pool.
  //
  // TODOs:
  // - compaction ratio estimation / updating
  // - handling of non-compactable GASes
  // - integration of GAS input data upload / freeing
  // - add optional hard limits / check for hard memory limits (shrink batch
  // size / abort, ...)
  //////////////////////////////////////////////////////////////////////////

  // Magic constants:

  // see explanation above
  constexpr double initialCompactionRatio = 0.5;

  // It is assumed that trace is called later when the GASes are still in
  // memory. We know that the memory consumption at that time will at least be
  // the compacted GASes + some CUDA stack space. Add a "random" 250MB that we
  // can use here, roughly matching CUDA stack space requirements.
  constexpr size_t additionalAvailableMemory = 250 * 1024 * 1024;

  //////////////////////////////////////////////////////////////////////////

  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  struct GASInfo
  {
    std::vector<OptixBuildInput> buildInputs;
    OptixAccelBufferSizes gas_buffer_sizes;
    std::shared_ptr<sutil::MeshGroup> mesh;
  };
  std::multimap<size_t, GASInfo> gases;
  size_t totalTempOutputSize = 0;

  unsigned opaque_triangle_input_flags[2] = {
      OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
      OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT |
          OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING};
  //   unsigned mask_triangle_input_flags[2] = {
  //       OPTIX_GEOMETRY_FLAG_NONE,
  //       OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING};
  //   unsigned blend_triangle_input_flags[2] = {
  //       OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL,
  //       OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL |
  //           OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING};

  for (size_t i = 0; i < meshes.size(); ++i)
  {
    auto &mesh = meshes[i];

    const size_t num_subMeshes = mesh->indices.size();
    std::vector<OptixBuildInput> buildInputs(num_subMeshes);

    assert(mesh->positions.size() == num_subMeshes &&
           mesh->normals.size() ==
               num_subMeshes); //&&
                               // mesh->colors.size() == num_subMeshes );

    // for( size_t j = 0; j < GeometryData::num_textcoords; ++j )
    //     assert( mesh->texcoords[j].size() == num_subMeshes );

    for (size_t j = 0; j < num_subMeshes; ++j)
    {
      OptixBuildInput &triangle_input = buildInputs[j];
      memset(&triangle_input, 0, sizeof(OptixBuildInput));
      triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
      triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
      triangle_input.triangleArray.vertexStrideInBytes =
          mesh->positions[j].byte_stride ? mesh->positions[j].byte_stride
                                         : sizeof(float3),
      triangle_input.triangleArray.numVertices = mesh->positions[j].count;
      triangle_input.triangleArray.vertexBuffers = &(mesh->positions[j].data);
      triangle_input.triangleArray.indexFormat =
          mesh->indices[j].elmt_byte_size == 0 ? OPTIX_INDICES_FORMAT_NONE
          : mesh->indices[j].elmt_byte_size == 2
              ? OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3
              : OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      triangle_input.triangleArray.indexStrideInBytes =
          mesh->indices[j].byte_stride ? mesh->indices[j].byte_stride * 3
                                       : mesh->indices[j].elmt_byte_size * 3;
      triangle_input.triangleArray.numIndexTriplets =
          mesh->indices[j].count / 3;
      triangle_input.triangleArray.indexBuffer = mesh->indices[j].data;
      triangle_input.triangleArray.numSbtRecords = 1;
      /*
                  const int32_t mat_idx = mesh->material_idx[j];
                  if( mat_idx >= 0 )
                  {
                      auto alpha_mode = m_materials[mat_idx].alpha_mode;
                      switch( alpha_mode )
                      {
                      case MaterialData::ALPHA_MODE_MASK:
                          triangle_input.triangleArray.flags =
         &mask_triangle_input_flags[m_materials[mat_idx].doubleSided]; break;
                      case MaterialData::ALPHA_MODE_BLEND:
                          triangle_input.triangleArray.flags =
         &blend_triangle_input_flags[m_materials[mat_idx].doubleSided]; break;
                      default:
                          triangle_input.triangleArray.flags =
         &opaque_triangle_input_flags[m_materials[mat_idx].doubleSided]; break;
                      };
                  }
                  else */
      {
        triangle_input.triangleArray.flags =
            &opaque_triangle_input_flags[0]; // default is single sided
      }
    }

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context, &accel_options, buildInputs.data(),
        static_cast<unsigned int>(num_subMeshes), &gas_buffer_sizes));

    totalTempOutputSize += gas_buffer_sizes.outputSizeInBytes;
    GASInfo g = {std::move(buildInputs), gas_buffer_sizes, mesh};
    gases.emplace(gas_buffer_sizes.outputSizeInBytes, g);
  }

  size_t totalTempOutputProcessedSize = 0;
  size_t usedCompactedOutputSize = 0;
  double compactionRatio = initialCompactionRatio;

  CuBuffer<char> d_temp;
  CuBuffer<char> d_temp_output;
  CuBuffer<size_t> d_temp_compactedSizes;

  OptixAccelEmitDesc emitProperty = {};
  emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

  while (!gases.empty())
  {
    // The estimated total output size that we end up with when using
    // compaction. It defines the minimum peak memory consumption, but is
    // unknown before actually building all GASes. Working only within these
    // memory constraints results in an actual peak memory consumption that is
    // very close to the minimal peak memory consumption.
    size_t remainingEstimatedTotalOutputSize =
        (size_t)((totalTempOutputSize - totalTempOutputProcessedSize) *
                 compactionRatio);
    size_t availableMemPoolSize =
        remainingEstimatedTotalOutputSize + additionalAvailableMemory;
    // We need to fit the following things into availableMemPoolSize:
    // - temporary buffer for building a GAS (only during build, can be cleared
    // before compaction)
    // - build output buffer of a GAS
    // - size (actual number) of a compacted GAS as output of a build
    // - compacted GAS

    size_t batchNGASes = 0;
    size_t batchBuildOutputRequirement = 0;
    size_t batchBuildMaxTempRequirement = 0;
    size_t batchBuildCompactedRequirement = 0;
    for (auto it = gases.rbegin(); it != gases.rend(); it++)
    {
      batchBuildOutputRequirement +=
          it->second.gas_buffer_sizes.outputSizeInBytes;
      batchBuildCompactedRequirement +=
          (size_t)(it->second.gas_buffer_sizes.outputSizeInBytes *
                   compactionRatio);
      // roughly account for the storage of the compacted size, although that
      // goes into a separate buffer
      batchBuildOutputRequirement += 8ull;
      // make sure that all further output pointers are 256 byte aligned
      batchBuildOutputRequirement =
          roundUp<size_t>(batchBuildOutputRequirement, 256ull);
      // temp buffer is shared for all builds in the batch
      batchBuildMaxTempRequirement =
          std::max(batchBuildMaxTempRequirement,
                   it->second.gas_buffer_sizes.tempSizeInBytes);
      batchNGASes++;
      if ((batchBuildOutputRequirement + batchBuildMaxTempRequirement +
           batchBuildCompactedRequirement) > availableMemPoolSize)
        break;
    }

    // d_temp may still be available from a previous batch, but is freed later
    // if it is "too big"
    d_temp.allocIfRequired(batchBuildMaxTempRequirement);

    // trash existing buffer if it is more than 10% bigger than what we need
    // if it is roughly the same, we keep it
    if (d_temp_output.byteSize() > batchBuildOutputRequirement * 1.1)
      d_temp_output.free();
    d_temp_output.allocIfRequired(batchBuildOutputRequirement);

    // this buffer is assumed to be very small
    // trash d_temp_compactedSizes if it is at least 20MB in size and at least
    // double the size than required for the next run
    if (d_temp_compactedSizes.reservedCount() > batchNGASes * 2 &&
        d_temp_compactedSizes.byteSize() > 20 * 1024 * 1024)
      d_temp_compactedSizes.free();
    d_temp_compactedSizes.allocIfRequired(batchNGASes);

    auto it = gases.rbegin();
    for (size_t i = 0, tempOutputAlignmentOffset = 0; i < batchNGASes; ++i)
    {
      emitProperty.result = d_temp_compactedSizes.get(i);
      GASInfo &info = it->second;

      OPTIX_CHECK(optixAccelBuild(
          context, 0, // CUDA stream
          &accel_options, info.buildInputs.data(),
          static_cast<unsigned int>(info.buildInputs.size()), d_temp.get(),
          d_temp.byteSize(), d_temp_output.get(tempOutputAlignmentOffset),
          info.gas_buffer_sizes.outputSizeInBytes, &info.mesh->gas_handle,
          &emitProperty, // emitted property list
          1              // num emitted properties
          ));

      tempOutputAlignmentOffset +=
          roundUp<size_t>(info.gas_buffer_sizes.outputSizeInBytes, 256ull);
      it++;
    }

    // trash d_temp if it is at least 20MB in size
    if (d_temp.byteSize() > 20 * 1024 * 1024)
      d_temp.free();

    // download all compacted sizes to allocate final output buffers for these
    // GASes
    std::vector<size_t> h_compactedSizes(batchNGASes);
    d_temp_compactedSizes.download(h_compactedSizes.data());

    //////////////////////////////////////////////////////////////////////////
    // TODO:
    // Now we know the actual memory requirement of the compacted GASes.
    // Based on that we could shrink the batch if the compaction ratio is bad
    // and we need to strictly fit into the/any available memory pool.
    bool canCompact = false;
    it = gases.rbegin();
    for (size_t i = 0; i < batchNGASes; ++i)
    {
      GASInfo &info = it->second;
      if (info.gas_buffer_sizes.outputSizeInBytes > h_compactedSizes[i])
      {
        canCompact = true;
        break;
      }
      it++;
    }

    // sum of size of compacted GASes
    size_t batchCompactedSize = 0;

    if (canCompact)
    {
      //////////////////////////////////////////////////////////////////////////
      // "batch allocate" the compacted buffers
      it = gases.rbegin();
      for (size_t i = 0; i < batchNGASes; ++i)
      {
        GASInfo &info = it->second;
        batchCompactedSize += h_compactedSizes[i];
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&info.mesh->d_gas_output),
                       h_compactedSizes[i]));
        totalTempOutputProcessedSize += info.gas_buffer_sizes.outputSizeInBytes;
        it++;
      }

      it = gases.rbegin();
      for (size_t i = 0; i < batchNGASes; ++i)
      {
        GASInfo &info = it->second;
        OPTIX_CHECK(optixAccelCompact(
            context, 0, info.mesh->gas_handle, info.mesh->d_gas_output,
            h_compactedSizes[i], &info.mesh->gas_handle));
        it++;
      }
    }
    else
    {
      it = gases.rbegin();
      for (size_t i = 0, tempOutputAlignmentOffset = 0; i < batchNGASes; ++i)
      {
        GASInfo &info = it->second;
        info.mesh->d_gas_output = d_temp_output.get(tempOutputAlignmentOffset);
        batchCompactedSize += h_compactedSizes[i];
        totalTempOutputProcessedSize += info.gas_buffer_sizes.outputSizeInBytes;

        tempOutputAlignmentOffset +=
            roundUp<size_t>(info.gas_buffer_sizes.outputSizeInBytes, 256ull);
        it++;
      }
      d_temp_output.release();
    }

    usedCompactedOutputSize += batchCompactedSize;

    gases.erase(it.base(), gases.end());
  }
}

/*****************************************/
void Accelerator::build_instance_accel(const sutil::MeshAlloc &allocator,
                                       const OptixDeviceContext &context,
                                       bool first, int rayTypeCount)
/*****************************************/
{

  const auto &instances = allocator.get_instances();
  const auto &meshes = allocator.get_meshes();
  const size_t num_instances = instances.size();

  std::vector<OptixInstance> optix_instances(num_instances);

  unsigned int sbt_offset = 0;
  for (size_t i = 0; i < num_instances; ++i)
  {
    auto instance = instances[i];
    auto &optix_instance = optix_instances[i];
    memset(&optix_instance, 0, sizeof(OptixInstance));

    optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
    optix_instance.instanceId = static_cast<unsigned int>(i);
    optix_instance.sbtOffset = sbt_offset;
    optix_instance.visibilityMask = 1;
    optix_instance.traversableHandle =
        meshes[instance->mesh_idx]->gas_handle;
    memcpy(optix_instance.transform, instance->transform.getData(),
           sizeof(float) * 12);

    sbt_offset +=
        static_cast<unsigned int>(
            meshes[instance->mesh_idx]
                ->indices.size()) *
        rayTypeCount; // one sbt record per GAS build input per RAY_TYPE
  }

  const size_t instances_size_in_bytes = sizeof(OptixInstance) * num_instances;
  CUdeviceptr d_instances;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_instances),
                        instances_size_in_bytes));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_instances),
                        optix_instances.data(), instances_size_in_bytes,
                        cudaMemcpyHostToDevice));

  OptixBuildInput instance_input = {};
  instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  instance_input.instanceArray.instances = d_instances;
  instance_input.instanceArray.numInstances =
      static_cast<unsigned int>(num_instances);

  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  accel_options.operation =
      first ? OPTIX_BUILD_OPERATION_BUILD : OPTIX_BUILD_OPERATION_UPDATE;

  OptixAccelBufferSizes ias_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options,
                                           &instance_input,
                                           1, // num build inputs
                                           &ias_buffer_sizes));

  CUdeviceptr d_temp_buffer;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer),
                        ias_buffer_sizes.tempSizeInBytes));
  if (first)
  {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_d_ias_output_buffer),
                          ias_buffer_sizes.outputSizeInBytes));
  }

  OPTIX_CHECK(optixAccelBuild(context,
                              nullptr, // CUDA stream
                              &accel_options, &instance_input,
                              1, // num build inputs
                              d_temp_buffer, ias_buffer_sizes.tempSizeInBytes,
                              m_d_ias_output_buffer,
                              ias_buffer_sizes.outputSizeInBytes, &m_ias_handle,
                              nullptr, // emitted property list
                              0        // num emitted properties
                              ));

  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_instances)));
}
