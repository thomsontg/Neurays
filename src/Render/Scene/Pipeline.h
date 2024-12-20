#pragma once

#include <iostream>
#include <vector>
#include <map>

#include <optix.h>

#include <sutil/sutil.h>
#include <sutil/Buffers/MeshAllocator.h>
#include <sutil/shared/structs.h>
#include <sutil/shared/CUDAOutputBuffer.h>

#include "Denoiser.h"

template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptyData
{
};

typedef Record<EmptyData> EmptyRecord;

class Pipeline
{
public:
  bool has_translucent;
  bool use_bdpt;
  bool use_envmap;
  bool use_simulator;

  void init(unsigned int width, unsigned int height);
  OptixDeviceContext &get_context() { return m_context; }
  void cleanup();

  typedef Record<HitGroupData> HitGroupRecord;

  void launch_main_pass(LaunchParams &lp, unsigned int width, unsigned int height);
  void launch_pre_pass(struct LaunchParams &lp);
  void launch_env_pass(LaunchParams &lp);
  void launch_bdpt_pass(LaunchParams &lp);

  float3 *denoise(struct LaunchParams &lp, unsigned int width, unsigned int height);  

  void set_default_shader(std::string shader_name)
  {
    m_shadername = shader_name;
  }

  void set_shaderlist(std::vector<std::vector<std::string>> *list)
  {
    m_shaderlist = list;
  }

  void create_sbt();
  void create_hit_group(std::vector<std::shared_ptr<sutil::MeshGroup>> meshes,
                        std::vector<std::shared_ptr<sutil::Instance>> instances,
                        std::map<std::string, std::pair<MtlData, MtlData>> materialdata,
                        std::vector<MtlData> materials,
                        bool first = false);
  void create_env_hitgroup();
  void create_denoiser(unsigned int width, unsigned int height);


  void on_change(unsigned int width, unsigned int height);

  void update_env();

  void set_device_context(OptixDeviceContext context) { m_context = context; }

private:
  OptixDeviceContext m_context;

  OptixPipelineCompileOptions m_pipeline_compile_options = {};
  OptixPipeline m_pipeline = 0;
  OptixPipeline m_bd_pipeline = 0;
  OptixModule m_ptx_module = 0;
  OptixShaderBindingTable m_sbt = {};
  OptixShaderBindingTable m_sample_sbt = {};
  OptixShaderBindingTable m_env_luminance_sbt = {};
  OptixShaderBindingTable m_env_marginal_sbt = {};
  OptixShaderBindingTable m_env_pdf_sbt = {};
  OptixShaderBindingTable m_bdsamples_sbt = {};
  OptixProgramGroup m_raygen_prog_group = 0;
  OptixProgramGroup m_sample_prog_group = 0;
  OptixProgramGroup m_bdsample_prog_group = 0;
  OptixProgramGroup m_env_luminance_prog_group = 0;
  OptixProgramGroup m_env_marginal_prog_group = 0;
  OptixProgramGroup m_env_pdf_prog_group = 0;
  OptixProgramGroup m_radiance_miss_group = 0;
  OptixProgramGroup m_occlusion_miss_group = 0;
  OptixProgramGroup m_feeler_miss_group = 0;
  OptixProgramGroup m_bdsample_miss_group = 0;
  OptixProgramGroup m_occlusion_hit_group = 0;
  OptixProgramGroup m_feeler_hit_group = 0;
  OptixProgramGroup m_bdsample_hit_group = 0;
  OptixProgramGroup m_bdclosest_hit_group = 0;
  std::vector<OptixProgramGroup> shaders;

  std::string m_shadername;
  std::vector<std::vector<std::string>> *m_shaderlist;

  void create_ptx_module();
  void create_program_groups();
  void create_pipeline();
  void create_sbt_sim();

  void create_env_prog_groups();

  OptixProgramGroup create_shader(int illum, std::string name);
  void set_shader(int illum, OptixProgramGroup closest_hit_program);
  OptixProgramGroup get_shader(int illum);

  LaunchParams *m_d_params = nullptr;
      
  Denoiser m_denoise;
};