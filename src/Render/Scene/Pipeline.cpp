
#include "Pipeline.h"

#include <optix_stubs.h>

/********************************************/
void Pipeline::init(unsigned int width, unsigned int height)
/********************************************/
{
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&m_d_params), sizeof(LaunchParams)));
    create_ptx_module();
    create_program_groups();
    create_pipeline();
    // if(use_simulator){
    //   create_sbt_sim();
    //   initDefaultLayer(true);
    // }
    // else{
    create_sbt();
    // }

    create_denoiser(width, height);
}

/********************************************/
void Pipeline::create_denoiser(unsigned int width, unsigned int height)
/********************************************/
{
  m_denoise.init(width, height);
}

/********************************************/
void Pipeline::on_change(unsigned int width, unsigned int height)
/********************************************/
{
  m_denoise.create_buffers(width, height);
}

/********************************************/
float3 *Pipeline::denoise(struct LaunchParams &lp, unsigned int width,
                          unsigned int height)
/********************************************/
{
  // Set buffers
  m_denoise.set_buffer(lp.accum_buffer, "color");
  m_denoise.set_buffer(lp.denoise_albedo, "albedo");
  m_denoise.set_buffer(lp.denoise_normals, "normal");
  m_denoise.set_buffer_active("output");
  m_denoise.denoise();

  return m_denoise.get_output_buffer();
}

/********************************************/
void Pipeline::cleanup()
/********************************************/
{
    // OptiX cleanup
    if (m_pipeline)
    {
        OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
        m_pipeline = 0;
    }
    if (m_raygen_prog_group)
    {
        OPTIX_CHECK(optixProgramGroupDestroy(m_raygen_prog_group));
        m_raygen_prog_group = 0;
    }
    if (m_radiance_miss_group)
    {
        OPTIX_CHECK(optixProgramGroupDestroy(m_radiance_miss_group));
        m_radiance_miss_group = 0;
    }
    if (m_occlusion_miss_group)
    {
        OPTIX_CHECK(optixProgramGroupDestroy(m_occlusion_miss_group));
        m_occlusion_miss_group = 0;
    }
    if (m_occlusion_hit_group)
    {
        OPTIX_CHECK(optixProgramGroupDestroy(m_occlusion_hit_group));
        m_occlusion_hit_group = 0;
    }
    if (m_ptx_module)
    {
        OPTIX_CHECK(optixModuleDestroy(m_ptx_module));
        m_ptx_module = 0;
    }

    if (m_sbt.raygenRecord)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_sbt.raygenRecord)));
        m_sbt.raygenRecord = 0;
    }
    if (m_sbt.missRecordBase)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_sbt.missRecordBase)));
        m_sbt.missRecordBase = 0;
    }
    if (m_sbt.hitgroupRecordBase)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_sbt.hitgroupRecordBase)));
        m_sbt.hitgroupRecordBase = 0;
    }

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_params)));
}

/********************************************/
void Pipeline::launch_pre_pass(LaunchParams &lp)
/********************************************/
{
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(m_d_params), &lp,
                               sizeof(LaunchParams), cudaMemcpyHostToDevice,
                               0 // stream
                               ));
    OPTIX_CHECK(
        optixLaunch(m_pipeline,
                    0, // stream
                    reinterpret_cast<CUdeviceptr>(m_d_params),
                    sizeof(LaunchParams), &m_sample_sbt,
                    lp.translucent_no_of_samples, // launch width
                    1,                            // launch height
                    1                             // launch depth
                    ));

    CUDA_SYNC_CHECK();
}

/********************************************/
void Pipeline::launch_bdpt_pass(LaunchParams &lp)
/********************************************/
{

    if (m_bdsamples_sbt.hitgroupRecordCount != 0)
    {
        OPTIX_CHECK(optixLaunch(
            m_bd_pipeline,
            0, // stream
            reinterpret_cast<CUdeviceptr>(m_d_params),
            sizeof(LaunchParams),
            &m_bdsamples_sbt,
            lp.bdsample_count, // launch width
            1,                 // launch height
            1                  // launch depth
            ));
    }

    //   float3 *mapped = new float3[launch_params.bdsample_count];
    //   CUDA_CHECK(cudaMemcpyAsync(mapped, launch_params.bd_samples, launch_params.bdsample_count * sizeof(float3),
    //                               cudaMemcpyDeviceToHost, 0));

    //   for(int i = 0; i < launch_params.bdsample_count; i++)
    //   {
    //     cout<< i << " : \t" << mapped[i].x << " \t" << mapped[i].y << " \t" << mapped[i].z << endl;
    //   }
}

/********************************************/
void Pipeline::launch_main_pass(LaunchParams &lp, unsigned int width, unsigned int height)
/********************************************/
{
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(m_d_params),
                               &lp,
                               sizeof(LaunchParams),
                               cudaMemcpyHostToDevice,
                               0 // stream
                               ));

    OPTIX_CHECK(optixLaunch(
        m_pipeline,
        0, // stream
        reinterpret_cast<CUdeviceptr>(m_d_params),
        sizeof(LaunchParams),
        &m_sbt,
        width,  // launch width
        height, // launch height
        1       // launch depth
        ));

    CUDA_SYNC_CHECK();
}

/********************************************/
void Pipeline::launch_env_pass(LaunchParams &lp)
/********************************************/
{
    OPTIX_CHECK(optixLaunch(
        m_pipeline,
        0, // stream
        reinterpret_cast<CUdeviceptr>(m_d_params),
        sizeof(LaunchParams),
        &m_env_luminance_sbt,
        lp.env_width,  // launch width
        lp.env_height, // launch height
        1              // launch depth
        ));
    OPTIX_CHECK(optixLaunch(
        m_pipeline,
        0, // stream
        reinterpret_cast<CUdeviceptr>(m_d_params),
        sizeof(LaunchParams),
        &m_env_marginal_sbt,
        1,             // launch width
        lp.env_height, // launch height
        1              // launch depth
        ));
    OPTIX_CHECK(optixLaunch(
        m_pipeline,
        0, // stream
        reinterpret_cast<CUdeviceptr>(m_d_params),
        sizeof(LaunchParams),
        &m_env_pdf_sbt,
        lp.env_width,  // launch width
        lp.env_height, // launch height
        1              // launch depth
        ));
}

/********************************************/
void Pipeline::create_ptx_module()
/********************************************/
{
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    m_pipeline_compile_options = {};
    m_pipeline_compile_options.usesMotionBlur = false;
    m_pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_pipeline_compile_options.numPayloadValues = NUM_PAYLOAD_VALUES;
    m_pipeline_compile_options.numAttributeValues = 2;                     // TODO
    m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE; // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = "launch_params";

    size_t inputSize = 0;
    std::string ptx;
    if (!use_simulator)
    {
        const std::string ptx_tmp(sutil::getInputData(nullptr, "src/device", "shaders.cu", inputSize));
        ptx = ptx_tmp;
    }
    else
    {
        const std::string ptx_tmp(sutil::getInputData(nullptr, "src/device", "mc_simulator.cu", inputSize));
        ptx = ptx_tmp;
    }

    m_ptx_module = {};
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreate(
        m_context,
        &module_compile_options,
        &m_pipeline_compile_options,
        ptx.c_str(),
        ptx.size(),
        log,
        &sizeof_log,
        &m_ptx_module));
}

/********************************************/
void Pipeline::create_program_groups()
/********************************************/
{
    OptixProgramGroupOptions program_group_options = {};
    char log[2048];
    size_t sizeof_log = sizeof(log);

    //
    // Ray generation
    //
    {
        std::string raygen = use_simulator ? "__raygen__volume_sim_diffuse" : "__raygen__pinhole";

        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = m_ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = raygen.c_str();

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context,
            &raygen_prog_group_desc,
            1, // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &m_raygen_prog_group));
    }
    if (!use_simulator)
    {
        if (has_translucent)
        {

            OptixProgramGroupDesc raygen_prog_group_desc = {};
            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module = m_ptx_module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__sample_translucent";

            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                m_context,
                &raygen_prog_group_desc,
                1, // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &m_sample_prog_group));
        }
        if (use_bdpt)
        {

            OptixProgramGroupDesc raygen_prog_group_desc = {};
            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module = m_ptx_module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__bdsamples";

            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                m_context,
                &raygen_prog_group_desc,
                1, // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &m_bdsample_prog_group));
        }

        if (use_envmap)
        {
            create_env_prog_groups();
        }
    }

    //
    // Miss
    //
    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = m_ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context, &miss_prog_group_desc,
            1, // num program groups
            &program_group_options, log, &sizeof_log, &m_radiance_miss_group));

        memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module =
            nullptr; // NULL miss program for occlusion rays
        miss_prog_group_desc.miss.entryFunctionName = nullptr;
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context, &miss_prog_group_desc,
            1, // num program groups
            &program_group_options, log, &sizeof_log, &m_occlusion_miss_group));
    }

    //
    // Hit group
    //
    if (!use_simulator)
    {
      int shader_count = 0;
        // Associate the shader selected in the command line with illum 0, 1, and 2
        OptixProgramGroup m_radiance_hit_group = create_shader(shader_count++, m_shadername);
        set_shader(shader_count++, m_radiance_hit_group);
        set_shader(shader_count++, m_radiance_hit_group);
        for(std::vector<std::string> &shaders: *m_shaderlist)
        {
          for(std::string shader : shaders)
          {
            create_shader(shader_count++, shader);
          }
        }
        
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = m_ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            m_context,
            &hit_prog_group_desc,
            1, // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &m_occlusion_hit_group));

        memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = m_ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__feeler";
        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            m_context,
            &hit_prog_group_desc,
            1, // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &m_feeler_hit_group));

        if (use_bdpt)
        {
            char log[2048];
            size_t sizeof_log = sizeof(log);
            OptixProgramGroupDesc hit_prog_group_desc = {};
            hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hit_prog_group_desc.hitgroup.moduleCH = m_ptx_module;
            hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__bdsample";
            sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                m_context,
                &hit_prog_group_desc,
                1, // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &m_bdclosest_hit_group));
        }
    }
}

/********************************************/
void Pipeline::create_env_prog_groups()
/********************************************/
{

    OptixProgramGroupOptions program_group_options = {};
    char log[2048];
    size_t sizeof_log = sizeof(log);

    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = m_ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__env_luminance";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context,
            &raygen_prog_group_desc,
            1, // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &m_env_luminance_prog_group));
    }
    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = m_ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__env_marginal";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context,
            &raygen_prog_group_desc,
            1, // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &m_env_marginal_prog_group));
    }
    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = m_ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__env_pdf";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            m_context,
            &raygen_prog_group_desc,
            1, // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &m_env_pdf_prog_group));
    }
}

/********************************************/
void Pipeline::create_pipeline()
/********************************************/
{
    OptixProgramGroup program_groups_ren[] =
        {
            m_raygen_prog_group,
            m_radiance_miss_group,
            m_occlusion_miss_group,
            get_shader(1),
            m_occlusion_hit_group,
        };

    OptixProgramGroup program_groups_sim[] =
        {
            m_raygen_prog_group,
        };

    OptixProgramGroup program_groups_bdsample[] =
        {
            m_bdsample_prog_group,
            m_occlusion_miss_group,
            m_bdclosest_hit_group,
            m_occlusion_hit_group,
        };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 25u;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    if (!use_simulator)
    {
        OPTIX_CHECK_LOG(optixPipelineCreate(
            m_context,
            &m_pipeline_compile_options,
            &pipeline_link_options,
            program_groups_ren,
            sizeof(program_groups_ren) / sizeof(program_groups_ren[0]),
            log,
            &sizeof_log,
            &m_pipeline));
    }
    else
    {
        OPTIX_CHECK_LOG(optixPipelineCreate(
            m_context,
            &m_pipeline_compile_options,
            &pipeline_link_options,
            program_groups_sim,
            sizeof(program_groups_sim) / sizeof(program_groups_sim[0]),
            log,
            &sizeof_log,
            &m_pipeline));
    }

    if (use_bdpt)
    {
        OPTIX_CHECK_LOG(optixPipelineCreate(
            m_context,
            &m_pipeline_compile_options,
            &pipeline_link_options,
            program_groups_bdsample,
            sizeof(program_groups_bdsample) / sizeof(program_groups_bdsample[0]),
            log,
            &sizeof_log,
            &m_bd_pipeline));
    }
}

/********************************************/
void Pipeline::create_sbt()
/********************************************/
{
    {
        const size_t raygen_record_size = sizeof(EmptyRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_sbt.raygenRecord), raygen_record_size));

        EmptyRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(m_sbt.raygenRecord),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice));
    }

    {
        const unsigned int ray_type_count = RAY_TYPE_COUNT;
        const size_t miss_record_size = sizeof(EmptyRecord);
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&m_sbt.missRecordBase),
            miss_record_size * ray_type_count));

        std::vector<EmptyRecord> ms_sbt(ray_type_count);
        OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_miss_group, &ms_sbt[RAY_TYPE_RADIANCE]));
        OPTIX_CHECK(optixSbtRecordPackHeader(m_occlusion_miss_group, &ms_sbt[RAY_TYPE_OCCLUSION]));

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(m_sbt.missRecordBase),
            &ms_sbt[0],
            miss_record_size * ray_type_count,
            cudaMemcpyHostToDevice));
        m_sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
        m_sbt.missRecordCount = ray_type_count;
    }

    if (use_bdpt)
    {
        {
            const size_t raygen_record_size = sizeof(EmptyRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_bdsamples_sbt.raygenRecord), raygen_record_size));

            EmptyRecord rg_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_prog_group, &rg_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(m_bdsamples_sbt.raygenRecord),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice));
        }

        {
            const unsigned int ray_type_count = RAY_TYPE_COUNT;
            const size_t miss_record_size = sizeof(EmptyRecord);
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>(&m_bdsamples_sbt.missRecordBase),
                miss_record_size * ray_type_count));

            std::vector<EmptyRecord> ms_sbt(ray_type_count);
            OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_miss_group, &ms_sbt[RAY_TYPE_RADIANCE]));
            OPTIX_CHECK(optixSbtRecordPackHeader(m_occlusion_miss_group, &ms_sbt[RAY_TYPE_OCCLUSION]));

            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(m_bdsamples_sbt.missRecordBase),
                &ms_sbt[0],
                miss_record_size * ray_type_count,
                cudaMemcpyHostToDevice));
            m_bdsamples_sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
            m_bdsamples_sbt.missRecordCount = ray_type_count;
        }
    }
}

/********************************************/
void Pipeline::create_sbt_sim()
/********************************************/
{
    {
        const size_t raygen_record_size = sizeof(EmptyRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_sbt.raygenRecord),
                              raygen_record_size));

        EmptyRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(m_sbt.raygenRecord), &rg_sbt,
                              raygen_record_size, cudaMemcpyHostToDevice));
    }

    {
        const size_t miss_record_size = sizeof(EmptyRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_sbt.missRecordBase),
                              miss_record_size));
        m_sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
        m_sbt.missRecordCount = 1;
    }

    const size_t hitgroup_record_size = sizeof(EmptyRecord);
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&m_sbt.hitgroupRecordBase), 0));

    m_sbt.hitgroupRecordStrideInBytes =
        static_cast<unsigned int>(hitgroup_record_size);
    m_sbt.hitgroupRecordCount = 0;
}

/********************************************/
void Pipeline::create_hit_group(std::vector<std::shared_ptr<sutil::MeshGroup>> meshes,
                                std::vector<std::shared_ptr<sutil::Instance>> instances,
                                std::map<std::string, std::pair<MtlData, MtlData>> materialdata,
                                std::vector<MtlData> materials,
                                bool first)
/********************************************/
{
    std::vector<HitGroupRecord> hitgroup_records;
    HitGroupRecord bd_hitgroup_records = {};
    for (const auto instance : instances)
    {
        const auto mesh = meshes[instance.get()->mesh_idx];
        for (size_t i = 0; i < mesh->material_idx.size(); ++i)
        {
            // cout << i << " : " << mesh->material_idx[i] << endl;
            HitGroupRecord rec = {};
            HitGroupRecord bd_rec = {};
            const int32_t mat_idx = mesh->material_idx[i] < 0 ? 0 : mesh->material_idx[i];
            const std::string selectedItem = instance->material_name;

            if (materialdata.find(selectedItem) != materialdata.end())
            {
                rec.data.mtl_inside = materialdata.find(selectedItem)->second.first;
                rec.data.mtl_inside_2 = materialdata.find(selectedItem)->second.second;
                rec.data.mtl_outside = MtlData();

                if (first)
                {
                    rec.data.mtl_inside.illum = materials[mat_idx].illum;
                    mesh->material_illum = materials[mat_idx].illum;
                }
                else
                {
                    rec.data.mtl_inside.illum = instance->material_illum;
                    rec.data.mtl_inside_2.illum = instance->material_illum;
                    rec.data.mtl_outside.illum = instance->material_illum;
                }
            }
            else
            {
                rec.data.mtl_inside = MtlData();
                rec.data.mtl_inside_2 = MtlData();
                rec.data.mtl_outside = MtlData();

                rec.data.mtl_inside.illum = instance->material_illum;
                rec.data.mtl_inside_2.illum = instance->material_illum;
                rec.data.mtl_outside.illum = instance->material_illum;
            }
            OptixProgramGroup m_radiance_hit_group = get_shader(rec.data.mtl_inside.illum);
            if (use_bdpt)
                bd_rec = rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_hit_group, &rec));
            rec.data.geometry.type = GeometryData::TRIANGLE_MESH;
            rec.data.geometry.triangle_mesh.positions = mesh->positions[i];
            rec.data.geometry.triangle_mesh.normals = mesh->normals[i];
            rec.data.geometry.triangle_mesh.texcoords = mesh->texcoords[i];
            rec.data.geometry.triangle_mesh.indices = mesh->indices[i];
            hitgroup_records.push_back(rec);

            if (rec.data.mtl_inside.illum >= (2 + m_shaderlist[0].size()))
                OPTIX_CHECK(optixSbtRecordPackHeader(m_feeler_hit_group, &rec));
            else
                OPTIX_CHECK(optixSbtRecordPackHeader(m_occlusion_hit_group, &rec));
            hitgroup_records.push_back(rec);

            if (use_bdpt)
            {
                OPTIX_CHECK(optixSbtRecordPackHeader(m_bdclosest_hit_group, &bd_rec));
                bd_hitgroup_records.data.geometry.type = GeometryData::TRIANGLE_MESH;
                bd_hitgroup_records.data.geometry.triangle_mesh.positions = mesh->positions[i];
                bd_hitgroup_records.data.geometry.triangle_mesh.normals = mesh->normals[i];
                bd_hitgroup_records.data.geometry.triangle_mesh.texcoords = mesh->texcoords[i];
                bd_hitgroup_records.data.geometry.triangle_mesh.indices = mesh->indices[i];
            }
        }
    }

    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&m_sbt.hitgroupRecordBase),
        hitgroup_record_size * hitgroup_records.size()));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(m_sbt.hitgroupRecordBase),
        hitgroup_records.data(),
        hitgroup_record_size * hitgroup_records.size(),
        cudaMemcpyHostToDevice));

    m_sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(hitgroup_record_size);
    m_sbt.hitgroupRecordCount = static_cast<unsigned int>(hitgroup_records.size());

    if (has_translucent)
    {
        m_sample_sbt = m_sbt;
        const size_t raygen_record_size = sizeof(EmptyRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_sample_sbt.raygenRecord), raygen_record_size));

        EmptyRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_sample_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(m_sample_sbt.raygenRecord),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice));
    }

    if (use_bdpt)
    {
        const size_t hitgroup_record_size = sizeof(HitGroupRecord);
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&m_bdsamples_sbt.hitgroupRecordBase),
            hitgroup_record_size));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(m_bdsamples_sbt.hitgroupRecordBase),
            &bd_hitgroup_records,
            hitgroup_record_size,
            cudaMemcpyHostToDevice));

        m_bdsamples_sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(hitgroup_record_size);
        m_bdsamples_sbt.hitgroupRecordCount = static_cast<unsigned int>(hitgroup_records.size());

        // m_bdsamples_sbt = m_sbt;
        {
            const size_t raygen_record_size = sizeof(EmptyRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_bdsamples_sbt.raygenRecord), raygen_record_size));

            EmptyRecord rg_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(m_bdsample_prog_group, &rg_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(m_bdsamples_sbt.raygenRecord),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice));
        }
    }

    if (use_envmap)
    {
        create_env_hitgroup();
    }
}

/********************************************/
void Pipeline::create_env_hitgroup()
/********************************************/
{

    m_env_luminance_sbt = m_env_marginal_sbt = m_env_pdf_sbt = m_sbt;
    {
        const size_t raygen_record_size = sizeof(EmptyRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_env_luminance_sbt.raygenRecord), raygen_record_size));

        EmptyRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_env_luminance_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(m_env_luminance_sbt.raygenRecord),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice));
    }
    {
        const size_t raygen_record_size = sizeof(EmptyRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_env_marginal_sbt.raygenRecord), raygen_record_size));

        EmptyRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_env_marginal_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(m_env_marginal_sbt.raygenRecord),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice));
    }
    {
        const size_t raygen_record_size = sizeof(EmptyRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_env_pdf_sbt.raygenRecord), raygen_record_size));

        EmptyRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_env_pdf_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(m_env_pdf_sbt.raygenRecord),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice));
    }
}

/********************************************/
void Pipeline::update_env()
/********************************************/
{
    create_env_prog_groups();
    create_env_hitgroup();
}

/********************************************/
OptixProgramGroup Pipeline::create_shader(int illum, std::string name)
/********************************************/
{
    OptixProgramGroupOptions program_group_options = {};
    char log[2048];
    size_t sizeof_log = sizeof(log);
    std::string shader = "__closesthit__" + name;
    OptixProgramGroup m_radiance_hit_group = 0;
    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH = m_ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = shader.c_str();
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context,
        &hit_prog_group_desc,
        1, // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &m_radiance_hit_group));
    set_shader(illum, m_radiance_hit_group);
    return m_radiance_hit_group;
}

/********************************************/
void Pipeline::set_shader(int illum, OptixProgramGroup closest_hit_program)
/********************************************/
{
    if (illum < 0)
    {
        std::cerr << "Error: Negative identification numbers are not supported for illumination models." << std::endl;
        return;
    }
    while (illum >= static_cast<int>(shaders.size()))
        shaders.push_back(0);
    shaders[illum] = closest_hit_program;
}

/********************************************/
OptixProgramGroup Pipeline::get_shader(int illum)
/********************************************/
{
    OptixProgramGroup shader = 0;
    if (use_simulator)
        return shader;
    if (illum >= 0 && illum < static_cast<int>(shaders.size()))
        shader = shaders[illum];

    if (!shader)
    {
        std::cerr << "Warning: An object uses a material with an unsupported illum identifier(" << illum << "). Using the default shader instead." << std::endl;
        shader = shaders[0];
    }
    return shader;
}
