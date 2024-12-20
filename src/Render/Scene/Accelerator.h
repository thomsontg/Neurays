#pragma once

#include <optix.h>
#include <optix_stubs.h>

#include <sutil/shared/structs.h>

#include <sutil/Buffers/MeshAllocator.h>
#include <sutil/Buffers/SamplerAlloc.h>

class Accelerator
{
public:
    OptixTraversableHandle traversable_handle() const { return m_ias_handle; }

    void build(const sutil::MeshAlloc &, const OptixDeviceContext &m_context);
    void rebuild(const sutil::MeshAlloc &, const OptixDeviceContext &m_context);

    void cleanup();

private:
    OptixTraversableHandle m_ias_handle = 0;

    CUdeviceptr m_d_ias_output_buffer = 0;

    void build_mesh_accels(const std::vector<std::shared_ptr<sutil::MeshGroup>> &meshes,
                           const OptixDeviceContext &context);
    void build_instance_accel(const sutil::MeshAlloc &allocator,
                              const OptixDeviceContext &context,
                              bool first = true, int rayTypeCount = RAY_TYPE_COUNT);

    template <typename T = char>
    class CuBuffer
    {
    public:
        CuBuffer(size_t count = 0) { alloc(count); }
        ~CuBuffer() { free(); }
        void alloc(size_t count)
        {
            free();
            m_allocCount = m_count = count;
            if (m_count)
            {
                CUDA_CHECK(cudaMalloc(&m_ptr, m_allocCount * sizeof(T)));
            }
        }
        void allocIfRequired(size_t count)
        {
            if (count <= m_allocCount)
            {
                m_count = count;
                return;
            }
            alloc(count);
        }
        CUdeviceptr get() const { return reinterpret_cast<CUdeviceptr>(m_ptr); }
        CUdeviceptr get(size_t index) const
        {
            return reinterpret_cast<CUdeviceptr>(m_ptr + index);
        }
        void free()
        {
            m_count = 0;
            m_allocCount = 0;
            CUDA_CHECK(cudaFree(m_ptr));
            m_ptr = nullptr;
        }
        CUdeviceptr release()
        {
            m_count = 0;
            m_allocCount = 0;
            CUdeviceptr current = reinterpret_cast<CUdeviceptr>(m_ptr);
            m_ptr = nullptr;
            return current;
        }
        void upload(const T *data)
        {
            CUDA_CHECK(
                cudaMemcpy(m_ptr, data, m_count * sizeof(T), cudaMemcpyHostToDevice));
        }

        void download(T *data) const
        {
            CUDA_CHECK(
                cudaMemcpy(data, m_ptr, m_count * sizeof(T), cudaMemcpyDeviceToHost));
        }
        void downloadSub(size_t count, size_t offset, T *data) const
        {
            assert(count + offset <= m_allocCount);
            CUDA_CHECK(cudaMemcpy(data, m_ptr + offset, count * sizeof(T),
                                  cudaMemcpyDeviceToHost));
        }
        size_t count() const { return m_count; }
        size_t reservedCount() const { return m_allocCount; }
        size_t byteSize() const { return m_allocCount * sizeof(T); }

    private:
        size_t m_count = 0;
        size_t m_allocCount = 0;
        T *m_ptr = nullptr;
    };
};