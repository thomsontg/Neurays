#pragma once

#define USE_FP32_OUTPUT 1

#include <string>
#include <vector>
#include <optix.h>
#include <sutil/shared/Camera.h>
#include <sutil/shared/CUDAOutputBuffer.h>

#include <Render/optics/Medium.h>
#include <Render/optics/Interface.h>
#include <Render/optics/load_mpml.h>

#include <OIDN/include/OpenImageDenoise/oidn.hpp>

//#include "src/network/EvalNetwork.h"

#include <misc/Globals.h>
#include <Render/Lights/Lights.h>
#include "Accelerator.h"
#include "Pipeline.h"
#include "Denoiser.h"

class Scene
{
public:
  Scene(struct Globals *_gbx);

  void cleanup();

  ~Scene();

  void render(sutil::CUDAOutputBuffer<uchar4> &output_buffer);
  float3* denoise();

  void initScene(Globals *_gbx);

  void initLight(const float3 &light_direction, const float3 &light_emission,
                 const std::string &env_file = "")
  {
    lights.init(env_file, light_direction, light_emission);
  }
  void initLaunchParams();
  void initCameraState();
  void initDefaultLayer(bool first);

  void initCameraFromCalibration(const float3 &eye_pos,
                                 const mat3 &calib_matrix,
                                 const mat3 &camera_rot_matrix,
                                 const mat3 &stereo_rot_matrix,
                                 const float2 &res, bool cam_res);

  void init_gl_buffer(int32_t w, int32_t h) { output_buffer.initialise_gl_buffer(w, h); }

  void handleCameraUpdate();
  void handleResize(sutil::CUDAOutputBuffer<uchar4> &output_buffer, int32_t w, int32_t h);
  void handleSunSkyUpdate(float &solar_time, float overcast);
  void handleLightUpdate() { update_light(); }
  void handleGeometryUpdate();


  // Window resize state
  bool resize_dirty;
  bool minimized;
  bool use_denoiser;

  // Camera state
  sutil::Camera camera;
  int32_t width;
  int32_t height;
  float scene_scale;
  bool cam_mat;
  float3 cam_mat_eye;

  // Default light configuration
  float3 light_dir;
  float3 light_rad;

  void append_files(std::vector<std::string> append_names);

  void update_mesh_structures(bool &update_bbox);

  void update_env(std::string env);

  void add_Instance(int mesh_id);

  void delete_Instance(int idx);

  void delete_all_Instances(int idx);

  void update_accel_structure();

  const bool use_simulator() { return m_pipeline.use_simulator; }

  float3 get_bbox_center() { return bbox.center(); }

  oidn::BufferRef get_colorBuf() { return colorBuf; }

  const std::vector<std::shared_ptr<sutil::MeshGroup>> &get_meshes() { return m_mesh_allocator.get_meshes(); }

  const std::vector<std::shared_ptr<sutil::Instance>> &get_instances() { return m_mesh_allocator.get_instances(); }

  std::vector<MtlData> &get_materials() { return m_materials; }

  const std::vector<std::string> &get_material_names() const { return mtl_names; }

  const std::vector<std::string> &get_all_mtl() const { return m_all_mtl; }

  const std::map<std::string, Medium> &get_media() const { return m_media; }

  const std::map<std::string, Interface> &get_interfaces() const { return m_interfaces; }

  MtlData get_materialdata(std::string &selecteditem) { return m_materialdata.find(selecteditem) != m_materialdata.end() ? m_materialdata.find(selecteditem)->second.first : MtlData(); }

  Lights &get_light() { return this->lights; }

  void set_materialdata(std::string &selecteditem, MtlData &data);

  void replace_material(std::string &name, int const &idx, int &illum, float &noise, float &density);

  void update_mtl_name(std::string &name, int &idx) { mtl_names[idx] = name; }

  void list_mtl();

  sutil::CUDAOutputBuffer<uchar4> &get_output_buffer() { return output_buffer; } //  Fix Output_buffer destructor

  void dealloc_output_buffer();

  void destroy_output_buffer() { output_buffer.~CUDAOutputBuffer(); }

  void cleanup_allocator();

  MtlData fill_rgb(Medium *med);

  float3 scene_center() { return bbox.center(); }

  std::vector<std::string> &get_shaderlist() { return m_single_shaderlist; }

private:
  void createContext();
  void initDenoiser();
  void setDenoiserImages();
  void run_denoiser();

  void list_shaders();
  void loadObjs(std::vector<std::string> files);
  void scanMeshObj(std::string m_filename);
  void add_default_light();
  void update_light();
  void compute_diffuse_reflectance(MtlData &mtl);
  void analyze_materials();
  unsigned int extract_area_lights();
  sutil::Transform get_object_transform(std::string filename) const;

  void alloc_env();

  int m_mesh_count = 0;
  int m_mtl_count = 0;
  int m_tex_count = 0;
  std::map<std::string, Medium> m_media;
  std::map<std::string, Interface> m_interfaces;
  std::vector<std::string> m_all_mtl;
  std::vector<MtlData> m_materials;
  std::vector<std::string> mtl_names;
  std::map<std::string, std::pair<MtlData, MtlData>> m_materialdata;
  std::vector<std::vector<std::string>> m_shaderlist;
  std::vector<std::string> m_single_shaderlist;

  // Allocators
  sutil::MeshAlloc m_mesh_allocator;
  sutil::SamplerAlloc m_sample_allocator;
  sutil::BufferAlloc m_buffer_allocator;

  // Output Buffer
  OptixDeviceContext m_context = 0;
  sutil::CUDAOutputBuffer<uchar4> output_buffer;

  oidn::DeviceRef device;
  oidn::BufferRef colorBuf;
  oidn::BufferRef albedoBuf;
  oidn::BufferRef normalBuf;
  oidn::FilterRef filter;

  // MC simulate variables
  std::vector<struct Layer> layers;
  BufferView<struct Layer> m_layers;

  struct Surface
  {
    unsigned int no_of_faces = 0;
    std::vector<uint3> indices;
    std::vector<float3> positions;
    std::vector<float3> normals;
    int mesh_idx;
  };

  std::vector<std::string> filenames;
  std::vector<Surface> surfaces;
  std::string envfile;
  sutil::Aabb bbox;
  bool use_sunsky;
  bool use_interop;

  PreethamSunSky sun_sky;
  float day_of_year;
  float time_of_day;
  float latitude;
  float angle_with_south;

  // Misc
  // Network network_creator;

  // Lights
  Lights lights;

  // Accelerator
  Accelerator m_accel;

  // Pipeline
  Pipeline m_pipeline;

  // Global Variables
  Globals *m_gbx = nullptr;
};
