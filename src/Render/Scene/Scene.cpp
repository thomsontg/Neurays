#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>

#include <sutil/shared/CUDAOutputBuffer.h>

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <sutil/shared/structs.h>

#include <sutil/Exception.h>
#include <sutil/math/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/IO/HDRLoader.h>

#include "lib/GLFW/include/GLFW/glfw3.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <lib/tinyobjloader/tiny_obj_loader.h>

#include "device/fresnel.h"
#include "Lights/SunSky.h"
#include "Scene.h"

#include "sh/sh.h"

using namespace std;
using namespace sutil;

namespace
{
  vector<tinyobj::shape_t> obj_shapes;
  vector<tinyobj::material_t> obj_materials;
  int32_t bufidx = 0;

  void context_log_cb(unsigned int level, const char *tag, const char *message,
                      void * /*cbdata */)
  {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag
              << "]: " << message << "\n";
  }
}

Scene::Scene(struct Globals *_gbx)
  : 
  network_creator(_gbx->networkdir),
  output_buffer(_gbx->output_buffer_type, _gbx->rd_width, _gbx->rd_height)
{
  m_gbx = _gbx;
  filenames = m_gbx->filenames;
  envfile = m_gbx->env_name;
  if (!m_gbx->camera)
  {
    m_gbx->camera = &camera;
  }
  else
  {
    camera = *m_gbx->camera;
  }
  width = m_gbx->rd_width;
  height = m_gbx->rd_height;
  light_dir = m_gbx->light_dir;
  light_rad = make_float3(0.0f);
  use_interop = m_gbx->output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP;
  resize_dirty = false;
  minimized = false;
  use_denoiser = false;
  day_of_year = 255.0f;
  time_of_day = 12.0f;
  latitude = 55.78f;
  angle_with_south = 0.0f;

  list_shaders();
  m_pipeline.set_shaderlist(&m_shaderlist);

  if (!m_gbx->shadername.empty())
    m_pipeline.set_default_shader(m_gbx->shadername);
  else
    m_pipeline.set_default_shader("lambertian");
  m_pipeline.use_envmap = !envfile.empty();
  m_pipeline.use_bdpt = false;
  m_pipeline.use_simulator = false;
  m_pipeline.has_translucent = false;
  if (m_pipeline.use_envmap)
    m_tex_count++;
  scene_scale = m_gbx->scene_scale;
  lights.set_envfile(envfile);
  initScene(_gbx);
  // handleSunSkyUpdate(time_of_day, sun_sky.getOvercast());
  
  initLight(m_gbx->get_light_direction(), m_gbx->emission, m_gbx->env_name);
  
  if (m_gbx->use_sunsky)
  {
    get_light().initSunSky(m_gbx->ordinal_day, m_gbx->time_of_day, m_gbx->latitude,
                                  m_gbx->turbidity, m_gbx->clouds, m_gbx->sky_angle, m_gbx->sky_up);
  }

  if(!m_gbx->xml_loading)
  {
    m_gbx->trackball = new QuatTrackBall(camera.lookat(), length(camera.lookat() - camera.eye()), m_gbx->rd_width, m_gbx->rd_height);
    m_gbx->camera_changed = true;
    m_gbx->xml_loading = false;
  }

}

void Scene::initScene(Globals *_gbx)
{
  cleanup();

  // Load env
  if (!m_pipeline.use_simulator)
    lights.loadEnvMap(m_buffer_allocator, m_sample_allocator, &m_gbx->launch_params);

  if (!filenames.empty())
  {
    loadObjs(filenames);
    analyze_materials();
    createContext();

    // Denoiser variables and setup
    initDenoiser();

    m_accel.build(m_mesh_allocator, m_pipeline.get_context());

    // Load network
    network_creator.createNetworkBuffers(&m_gbx->launch_params, m_buffer_allocator);

    OPTIX_CHECK(optixInit()); // Need to initialize function table
    m_pipeline.set_device_context(m_context);
    m_pipeline.init(width, height);
    m_pipeline.create_hit_group(get_meshes(), get_instances(), m_materialdata, m_materials, true);

    bbox.invalidate();
    for (const auto &instance : m_mesh_allocator.get_instances())
      if (instance->world_aabb.area() < 1.0e4f) // Objects with a very large bounding box are considered background
        bbox.include(instance->world_aabb);
    // cout << "Scene bounding box maximum extent: " << bbox.maxExtent() << endl;

    if (&camera == nullptr || !m_gbx->xml_loading)
    {
      initCameraState();
    }
    initLaunchParams();
  }
}

Scene::~Scene()
{
  try
  {
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_gbx->launch_params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_gbx->launch_params.buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_gbx->launch_params.lights.data)));
  }
  catch (exception &e)
  {
    cerr << "Caught exception: " << e.what() << "\n";
  }
}

void Scene::list_shaders()
{
  std::vector<std::string> occlusion_shaders, feeler_shaders;
  occlusion_shaders.push_back("mirror");
  occlusion_shaders.push_back("transparent");
  occlusion_shaders.push_back("glossy");
  occlusion_shaders.push_back("basecolor");
  occlusion_shaders.push_back("lambertian");
  occlusion_shaders.push_back("directional");
  occlusion_shaders.push_back("arealight");
  occlusion_shaders.push_back("metal");
  occlusion_shaders.push_back("absorbing");
  // occlusion_shaders.push_back("translucent");
  occlusion_shaders.push_back("subsurf_brdf");
  occlusion_shaders.push_back("fresnel");
  occlusion_shaders.push_back("network");
  // occlusion_shaders.push_back("network_inf");
  // occlusion_shaders.push_back("rough_transparent");
  occlusion_shaders.push_back("holdout");
  occlusion_shaders.push_back("radiance_holdout");
  occlusion_shaders.push_back("check_normals");
  // feeler_shaders.push_back("distance");
  // feeler_shaders.push_back("distance_firsthit");
  feeler_shaders.push_back("heterogeneous");
  feeler_shaders.push_back("bidirectional_volume");
  feeler_shaders.push_back("volume");
  feeler_shaders.push_back("volume_rough");
  // Microgeometry
  feeler_shaders.push_back("spheres");
  feeler_shaders.push_back("blood");
  feeler_shaders.push_back("bone");

  m_shaderlist.push_back(occlusion_shaders);
  m_shaderlist.push_back(feeler_shaders);

  m_single_shaderlist = occlusion_shaders;
  m_single_shaderlist.insert(m_single_shaderlist.end(), feeler_shaders.begin(), feeler_shaders.end());
}

void Scene::handleGeometryUpdate()
{
  for (auto &instance : m_mesh_allocator.get_instances())
  {
    if (instance->transforms.dirty)
    {
      instance->transform =
          sutil::Matrix4x4::compute_transform(instance->transforms);
      instance->world_aabb.transform(instance->transform);
    }
  }
  m_accel.rebuild(m_mesh_allocator, m_pipeline.get_context());
}

void Scene::cleanup()
{
  m_buffer_allocator.cleanup();
  m_sample_allocator.cleanup();

  m_accel.cleanup();
  m_pipeline.cleanup();
  m_mesh_allocator.cleanup();

  if (m_context)
  {
    OPTIX_CHECK(optixDeviceContextDestroy(m_context));
    m_context = 0;
  }
}

void Scene::createContext()
{
  // Initialize CUDA
  CUDA_CHECK(cudaFree(nullptr));

  CUcontext cuCtx = nullptr; // zero means take the current context
  OPTIX_CHECK(optixInit());
  OptixDeviceContextOptions options = {};
  // options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
  options.logCallbackFunction = &context_log_cb;
  options.logCallbackLevel = 4;
  OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &m_context));
  m_pipeline.set_device_context(m_context);
}

void Scene::initDenoiser()
{
  device = oidn::newDevice();
  device.commit();

  // Create buffers for input/output images accessible by both host (CPU) and device (CPU/GPU)
  colorBuf = device.newBuffer(width * height * 4 * sizeof(float));
  normalBuf = device.newBuffer(width * height * 3 * sizeof(float));
  albedoBuf = device.newBuffer(width * height * 3 * sizeof(float));

  setDenoiserImages();
}

// This is also called by reshape.
void Scene::setDenoiserImages()
{
  // filter.setImage()
  float4 *res = new float4[width * height];

  colorBuf.write(0, sizeof(float), res);
  filter.setImage("color", colorBuf, oidn::Format::Float4, width, height);
  filter.setImage("albedo", colorBuf, oidn::Format::Float4, width, height);
  filter.setImage("normal", colorBuf, oidn::Format::Float4, width, height);
  filter.setImage("output", colorBuf, oidn::Format::Float4, width, height);
  filter.set("hdr", true);
  filter.commit();
}

void Scene::run_denoiser()
{
  filter.execute();
}

void Scene::initLaunchParams()
{
  LaunchParams &lp = m_gbx->launch_params;
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void **>(&lp.accum_buffer),
      width * height * sizeof(float4)));
  // CUDA_CHECK(cudaMalloc(
  //     reinterpret_cast<void **>(&lp.buffer),
  //     width * height * sizeof(float3)));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void **>(&lp.error),
      sizeof(float)));
  // CUDA_CHECK(cudaMalloc(
  //     reinterpret_cast<void **>(&lp.buffer),
  //     width * height * sizeof(float3)));
  // CUDA_CHECK(cudaMalloc(
  //     reinterpret_cast<void **>(&lp.hit_data),
  //     width * height * sizeof(unsigned int)));
  // CUDA_CHECK(cudaMalloc(
  //     reinterpret_cast<void **>(&lp.pos),
  //     width * height * sizeof(float3)));
  // CUDA_CHECK(cudaMalloc(
  //     reinterpret_cast<void **>(&lp.ray_dir),
  //     width * height * sizeof(float3)));
  // lp.error[0] = 0.0f;
  lp.frame_buffer = nullptr; // Will be set when output buffer is mapped
  lp.translucent_no_of_samples = 1000u;
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void **>(&lp.translucent_samples),
      lp.translucent_no_of_samples * sizeof(PositionSample)));
  lp.width = width;
  lp.height = height;
  lp.subframe_index = 0u;
  lp.max_depth = 32u;
  lp.lightType = LIGHTING_SIMPLE;
  lp.missType = MISS_CONSTANT;

  lp.miss_color = make_float3(1.0f, 1.0f, 1.0f);
  lp.no_of_samples = 1u;
  lp.tone_mapping = color::COLORMODE_NONE;
  lp.firefly_threshold = 5000u;
  lp.max_volume_bounces = 30000u;
  lp.light_dir = make_float3(1.0f);

  // BDPT params
  lp.bdsample_count = 20u;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&lp.bd_samples), lp.bdsample_count * sizeof(float3)));

  if (m_pipeline.use_simulator)
  {
    lp.n_light_flux = 100;
    lp.film_height = lp.pixel_size * height;
    lp.film_width = lp.pixel_size * width;
  }
  // const float loffset = bbox.maxExtent();
  m_gbx->launch_params.max_bbox = bbox.maxExtent();
  // Add light sources depending on chosen shader
  // if(shadername == "arealight")
  // {
  //   if(!extract_area_lights())
  //   {
  //     cerr << "Error: no area lights in scene. "
  //          << "You cannot use the area light shader if there are no emissive objects in the scene. "
  //          << "Objects are emissive if their ambient color is not zero."
  //          << endl;
  //     exit(0);
  //   }
  // }
  // else
  add_default_light();

  if (m_pipeline.use_envmap && !m_pipeline.use_simulator)
  {
    lp.lightType = LIGHTING_ENVMAP;
    lp.missType = MISS_ENVMAP;
    lp.env_scale = 0.8;
    alloc_env();
  }

  // Initialise simulation params
  lp.use_rainbow = true;
  lp.mode = TRANSMIT;
  lp.n_layers = 1;

  // Microgeometry params
  lp.invert_distances = false;
  lp.geom_center = bbox.center();
  // End Microgeometry params

  // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_d_params), sizeof(LaunchParams)));

  lp.handle = m_accel.traversable_handle();
  if (use_denoiser)
  {
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&denoiser.noisy_buffer), sizeof(float4) * width * height));
  }
}

void Scene::initCameraState()
{
  camera.setFovY(45.0f);
  camera.setLookat(bbox.center());
  camera.setEye(bbox.center() + make_float3(0.0f, 0.0f, 1.8f * bbox.maxExtent()));
}

void Scene::handleCameraUpdate()
{
  camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
  m_gbx->launch_params.eye = camera.eye();
  camera.UVWFrame(m_gbx->launch_params.U, m_gbx->launch_params.V, m_gbx->launch_params.W);
  m_gbx->camera = &camera;
}

void Scene::initCameraFromCalibration(const float3 &eye_pos,
                                      const mat3 &calib_matrix,
                                      const mat3 &camera_rot_matrix,
                                      const mat3 &stereo_rot_matrix,
                                      const float2 &res, bool cam_res)
{
  mat3 cam_rot = mult(camera_rot_matrix, stereo_rot_matrix);
  cam_mat_eye = mult(cam_rot, -eye_pos); // if tvec is provided instead of eye
  m_gbx->launch_params.eye = cam_mat_eye;
  mat3 inv_calib_matrix = identity_mat3();
  invert(calib_matrix, inv_calib_matrix);
  m_gbx->launch_params.inv_calibration_matrix = mult(cam_rot, inv_calib_matrix);
  if (cam_res)
    m_gbx->launch_params.inv_calibration_matrix =
        mult(m_gbx->launch_params.inv_calibration_matrix,
             make_mat3(make_float3(res.x / width, res.y / height, 1.0f)));
  camera.setEye(cam_mat_eye);
  handleCameraUpdate();
  cam_mat = true;

  // The following is for debugging. Should be a noop for a proper rotation
  // matrix.
  // mat3 inv_cam_rot;
  // invert(launch_params.inv_calibration_matrix, inv_cam_rot);
  // launch_params.inv_calibration_matrix = transpose(inv_cam_rot); // inverse
  // transpose to get normal matrix
}

void Scene::handleResize(CUDAOutputBuffer<uchar4> &output_buffer, int32_t w, int32_t h)
{
  LaunchParams &lp = m_gbx->launch_params;
  width = w;
  height = h;
  lp.width = width;
  lp.height = height;
  output_buffer.resize(width, height);

  // Realloc accumulation buffer
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_gbx->launch_params.accum_buffer)));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void **>(&m_gbx->launch_params.accum_buffer),
      width * height * sizeof(float4)));
  
  m_pipeline.on_change(w, h);
}

void Scene::render(sutil::CUDAOutputBuffer<uchar4> &output_buffer)
{
  // Launch Environment samples
  if (m_gbx->first_loop && m_pipeline.use_envmap && !m_pipeline.use_simulator)
    m_pipeline.launch_env_pass(m_gbx->launch_params);
  // Luunch BDPT pass
  if (m_pipeline.use_bdpt)
    m_pipeline.launch_bdpt_pass(m_gbx->launch_params);
  // Translucent pass
  if (m_pipeline.has_translucent && !m_pipeline.use_simulator)
    m_pipeline.launch_pre_pass(m_gbx->launch_params);
  // Launch main frame
  uchar4 *result_buffer_data = output_buffer.map();
  m_gbx->launch_params.frame_buffer = result_buffer_data;
  m_pipeline.launch_main_pass(m_gbx->launch_params, width, height);
  output_buffer.unmap();

  if (use_denoiser)
    run_denoiser();
}

/********************************************/
float3 *Scene::denoise()
/********************************************/
{
  return m_pipeline.denoise(m_gbx->launch_params, width, height);
}

void Scene::append_files(std::vector<std::string> append_names)
{
  for (const std::string name : append_names)
  {
    if (std::find(append_names.begin(), append_names.end(), name) != append_names.end())
    {
      filenames.insert(filenames.end(), append_names.begin(), append_names.end());
    }
    else
    {
      std::cout << name << " : Already exists!" << std::endl;
    }
  }
  // Update global variable
  m_gbx->filenames = filenames;

  for (int i = 0; i < append_names.size(); i++)
    std::cout << "Adding : " << append_names[i] << " to the Buffer" << std::endl;

  loadObjs(append_names);
}

void Scene::update_mesh_structures(bool &update_bbox)
{
  m_accel.build(m_mesh_allocator, m_context);
  m_pipeline.create_sbt();
  m_gbx->launch_params.handle = m_accel.traversable_handle();
  // bbox.invalidate();
  // for (const auto &instance : m_mesh_allocator.get_instances())
  //   if (instance->world_aabb.area() < 1.0e4f) // Objects with a very large bounding box are considered background
  //     bbox.include(instance->world_aabb);
}

void Scene::update_env(std::string env)
{
  lights.set_envfile(env);
  if (!m_pipeline.use_envmap)
  {
    lights.loadEnvMap(m_buffer_allocator, m_sample_allocator, &m_gbx->launch_params);
    m_pipeline.update_env();
    m_pipeline.use_envmap = true;
    m_gbx->launch_params.lightType = LIGHTING_ENVMAP;
    m_gbx->launch_params.missType = MISS_ENVMAP;
  }
  else
  {
    lights.replaceEnvMap(m_buffer_allocator, m_sample_allocator, &m_gbx->launch_params);
  }
  alloc_env();
}

void Scene::initDefaultLayer(bool first)
{
  unsigned int n_layers = 1;
  layers.resize(n_layers);
  auto &layer = layers[0];

  // Optical params
  layer.ior = 1.5867f;
  layer.albedo = 0.99919;
  layer.extinction = 14.836f;
  layer.asymmetry = 0.9745;
  // Layer thickness
  layer.start = 0.0f;
  layer.thickness = 6.4f; // 1.0f / layer.extinction;
  layer.end = layer.start - layer.thickness;
  // Surface
  layer.bsdf.type = MICROFACET;
  layer.bsdf.ggx.roughness = 0.0084f;
  layer.bsdf.ggx.mean = 0.0065f;
  layer.bsdf.ggx.scale = 0.0018f;

  // auto &layer2 = layers[1];
  // layer2.ior = launch_params.ior;
  // layer2.albedo = 0.6f;
  // layer2.extinction = launch_params.extinction;
  // layer2.asymmetry = launch_params.asymmetry;
  // // Layer thickness
  // layer2.start = layer.end;
  // layer2.thickness = 10000.0f; // 1.0f / layer.extinction;
  // layer2.end = layer2.start - layer2.thickness;
  // // Surface
  // layer2.bsdf.type = SMOOTH;
  // layer2.bsdf.ggx.roughness = 0.3;

  m_gbx->launch_params.n_layers = n_layers;
  if (first)
  {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_layers.data),
                          n_layers * sizeof(Layer)));
  }
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(m_layers.data), layers.data(),
                        sizeof(Layer) * n_layers, cudaMemcpyHostToDevice));

  m_layers.byte_stride = 0;
  m_layers.count = static_cast<uint32_t>(n_layers);
  m_layers.elmt_byte_size = static_cast<uint16_t>(sizeof(struct Layer));
  m_gbx->launch_params.layers = m_layers;
}

void Scene::alloc_env()
{
  LaunchParams &lp = m_gbx->launch_params;
  lp.envmap = m_sample_allocator.getSampler(0);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&lp.env_luminance), lp.env_width * lp.env_height * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&lp.marginal_f), lp.env_height * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&lp.marginal_pdf), lp.env_height * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&lp.marginal_cdf), lp.env_height * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&lp.conditional_pdf), lp.env_width * lp.env_height * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&lp.conditional_cdf), lp.env_width * lp.env_height * sizeof(float)));
}

void Scene::cleanup_allocator()
{
  m_mesh_allocator.cleanup();
  m_mtl_count = 0;
  m_mesh_count = 0;
  m_tex_count = m_pipeline.use_envmap ? 1 : 0;
}

void Scene::add_Instance(int mesh_id)
{
  const auto mesh = m_mesh_allocator.get_meshes()[mesh_id];
  auto instance = std::make_shared<sutil::Instance>();
  instance->world_aabb = mesh->object_aabb;
  instance->world_aabb.transform(instance->transform);
  instance->material_name = mesh->material_name;
  instance->material_illum = mesh->material_illum;
  instance->instance_id = mesh->instance_count;
  m_mesh_allocator.addInstance(instance);
}

void Scene::delete_Instance(int idx)
{
  m_mesh_allocator.delete_instance(idx);
  m_accel.build(m_mesh_allocator, m_pipeline.get_context());
  m_pipeline.create_hit_group(get_meshes(), get_instances(), m_materialdata, m_materials, true);
  m_gbx->launch_params.handle = m_accel.traversable_handle();
}

void Scene::delete_all_Instances(int idx)
{
  m_mesh_allocator.delete_all_instances(idx);
  m_accel.build(m_mesh_allocator, m_pipeline.get_context());
  m_pipeline.create_hit_group(get_meshes(), get_instances(), m_materialdata, m_materials, true);
  m_gbx->launch_params.handle = m_accel.traversable_handle();
}

void Scene::update_accel_structure()
{
  m_accel.rebuild(m_mesh_allocator, m_pipeline.get_context());
  m_pipeline.create_sbt();
  m_gbx->launch_params.handle = m_accel.traversable_handle();
}

void Scene::loadObjs(std::vector<std::string> files)
{
  for (string filename : files)
  {
    scanMeshObj(filename);

    for (tinyobj::material_t &mtl : obj_materials)
    {
      MtlData m_mtl;
      m_mtl.rho_d = make_float3(mtl.diffuse[0], mtl.diffuse[1], mtl.diffuse[2]);
      m_mtl.rho_s = make_float3(mtl.specular[0], mtl.specular[1], mtl.specular[2]);
      m_mtl.emission = make_float3(mtl.ambient[0], mtl.ambient[1], mtl.ambient[2]);
      m_mtl.shininess = mtl.shininess;
      m_mtl.ior = mtl.ior;
      m_mtl.illum = mtl.illum;
      if (!mtl.diffuse_texname.empty())
      {
        string path;
        size_t idx = filename.find_last_of("/\\");
        
        if (idx < filename.length())
        {
          path = filename.substr(0, idx + 1);
        }
        ImageBuffer img;
        
        try
        {
          img = load_Image((path + mtl.diffuse_texname).c_str());
        }
        catch(std::exception &e)
        {
          std::cout << e.what() << std::endl;
        }
        
        if (img.pixel_format != UNSIGNED_BYTE4)
          cerr << "Texture image with unknown pixel format: " << mtl.diffuse_texname << endl;
        else
        {
          cout << "Loaded texture image " << mtl.diffuse_texname << endl;
          m_sample_allocator.addImage(img.width, img.height, 8, 4, img.data);
          m_sample_allocator.addSampler(cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, m_tex_count);
          m_mtl.base_color_tex = m_sample_allocator.getSampler(m_tex_count++);
        }
      }

      if (!mtl.bump_texname.empty())
      {
        string path;
        size_t idx = filename.find_last_of("/\\");
      
        if (idx < filename.length())
          path = filename.substr(0, idx + 1);

        ImageBuffer img;
      
        try{
          img = load_Image((path + mtl.bump_texname).c_str());
        }
        catch(std::exception &e)
        {
          std::cout << e.what() << std::endl;
        }
      
        if (img.pixel_format != UNSIGNED_BYTE4)
          cerr << "Texture image with unknown pixel format: " << mtl.bump_texname << endl;
        else
        {
          cout << "Loaded texture image " << mtl.bump_texname << endl;
          m_sample_allocator.addImage(img.width, img.height, 8, 4, img.data);
          m_sample_allocator.addSampler(cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, m_tex_count);
          m_mtl.bump_map_tex = m_sample_allocator.getSampler(m_tex_count++);
        }
      }      
      
      if (!mtl.alpha_texname.empty())
      {
        string path;
        size_t idx = filename.find_last_of("/\\");
      
        if (idx < filename.length())
          path = filename.substr(0, idx + 1);

        ImageBuffer img;
      
        try{
          img = load_Image((path + mtl.alpha_texname).c_str());
        }
        catch(std::exception &e)
        {
          std::cout << e.what() << std::endl;
        }
      
        if (img.pixel_format != UNSIGNED_BYTE4)
          cerr << "Texture image with unknown pixel format: " << mtl.alpha_texname << endl;
        else
        {
          cout << "Loaded alpha texture image " << mtl.alpha_texname << endl;
          m_sample_allocator.addImage(img.width, img.height, 8, 4, img.data);
          m_sample_allocator.addSampler(cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, m_tex_count);
          m_mtl.alpha_map_tex = m_sample_allocator.getSampler(m_tex_count++);
        }
      }

      m_materials.push_back(m_mtl);
      mtl_names.push_back(mtl.name);
    }
    std::vector<std::string> obj_names;
    int count = 0;
    for (vector<tinyobj::shape_t>::const_iterator it = obj_shapes.begin(); it < obj_shapes.end(); ++it)
    {
      const tinyobj::shape_t &shape = *it;
      CUdeviceptr buffer;
      auto mesh = std::make_shared<sutil::MeshGroup>();
      m_mesh_allocator.addMesh(mesh);
      
      if (std::find(obj_names.begin(), obj_names.end(), shape.name) == obj_names.end())
      {
        obj_names.push_back(shape.name);
        mesh->name = shape.name;
      }
      else
      {
        mesh->name = shape.name + std::to_string(count++);
      }
      
      {
        BufferView<unsigned int> buffer_view;
        m_buffer_allocator.addBuffer(shape.mesh.indices.size() * sizeof(unsigned int), reinterpret_cast<const void *>(&shape.mesh.indices[0]));
        buffer = m_buffer_allocator.getBuffer(bufidx++);
        buffer_view.data = buffer;
        buffer_view.byte_stride = 0;
        buffer_view.count = static_cast<uint32_t>(shape.mesh.indices.size());
        buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(unsigned int));
        mesh->indices.push_back(buffer_view);
      }
      {
        BufferView<float3> buffer_view;
        m_buffer_allocator.addBuffer(shape.mesh.positions.size() * sizeof(float), reinterpret_cast<const void *>(&shape.mesh.positions[0]));
        buffer = m_buffer_allocator.getBuffer(bufidx++);
        buffer_view.data = buffer;
        buffer_view.byte_stride = 0;
        buffer_view.count = static_cast<uint32_t>(shape.mesh.positions.size() / 3);
        buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
        mesh->positions.push_back(buffer_view);
      }
      {
        BufferView<float3> buffer_view;
        if (shape.mesh.normals.size() > 0)
        {
          m_buffer_allocator.addBuffer(shape.mesh.normals.size() * sizeof(float), reinterpret_cast<const void *>(&shape.mesh.normals[0]));
          buffer = m_buffer_allocator.getBuffer(bufidx++);
          buffer_view.data = buffer;
          buffer_view.byte_stride = 0;
          buffer_view.count = static_cast<uint32_t>(shape.mesh.normals.size() / 3);
          buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
        }
        mesh->normals.push_back(buffer_view);
      }
      {
        BufferView<float2> buffer_view;
        if (shape.mesh.texcoords.size() > 0)
        {
          m_buffer_allocator.addBuffer(shape.mesh.texcoords.size() * sizeof(float), reinterpret_cast<const void *>(&shape.mesh.texcoords[0]));
          buffer = m_buffer_allocator.getBuffer(bufidx++);
          buffer_view.data = buffer;
          buffer_view.byte_stride = 0;
          buffer_view.count = static_cast<uint32_t>(shape.mesh.texcoords.size() / 2);
          buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float2));
        }
        mesh->texcoords.push_back(buffer_view);
      }
      mesh->material_idx.push_back(shape.mesh.material_ids[0] + m_mtl_count);
      mesh->material_name = mtl_names[mesh->material_idx[0]];
      cerr << "\t\tNum triangles: " << mesh->indices.back().count / 3 << endl;
      cerr << "\t\tMesh name    : " << mesh->material_name << endl;
      auto instance = std::make_shared<sutil::Instance>();
      
      // Custom transforms
      instance->transforms = get_object_transform(filename);
      
      instance->transform = sutil::Matrix4x4::compute_transform(instance->transforms);

      Surface surface;
      surface.indices.resize(shape.mesh.indices.size() / 3);
      copy(shape.mesh.indices.begin(), shape.mesh.indices.end(), &surface.indices.front().x);
      surface.positions.resize(shape.mesh.positions.size() / 3);
      for (unsigned int i = 0; i < surface.positions.size(); ++i)
      {
        float4 pos = make_float4(shape.mesh.positions[i * 3], shape.mesh.positions[i * 3 + 1], shape.mesh.positions[i * 3 + 2], 1.0f);
        surface.positions[i] = make_float3(instance->transform * pos);
      }

      if (shape.mesh.normals.size() > 0)
      {
        surface.normals.resize(shape.mesh.normals.size() / 3);
        for (unsigned int i = 0; i < surface.normals.size(); ++i)
        {
          float4 normal = make_float4(shape.mesh.positions[i * 3], shape.mesh.positions[i * 3 + 1], shape.mesh.positions[i * 3 + 2], 0.0f);
          surface.normals[i] = make_float3(instance->transform * normal);
        }
      }
      surface.no_of_faces = static_cast<unsigned int>(surface.indices.size());
      surface.mesh_idx = m_mesh_count;
      surfaces.push_back(surface);

      mesh->object_aabb.invalidate();
      for (unsigned int i = 0; i < shape.mesh.positions.size() / 3; ++i)
        mesh->object_aabb.include(make_float3(shape.mesh.positions[i * 3], shape.mesh.positions[i * 3 + 1], shape.mesh.positions[i * 3 + 2]));

      instance->mesh_idx = m_mesh_count++;
      instance->world_aabb = mesh->object_aabb;
      instance->world_aabb.transform(instance->transform);
      instance->material_name = mesh->material_name;
      instance->material_illum = mesh->material_illum;
      instance->material_id = mesh->material_idx[0];
      instance->instance_id = mesh->instance_count;
      m_mesh_allocator.addInstance(instance);
    }
    m_mtl_count += static_cast<int>(obj_materials.size());
    obj_materials.clear();
    obj_shapes.clear();
  }
}

void Scene::scanMeshObj(string m_filename)
{
  int32_t num_triangles = 0;
  int32_t num_vertices = 0;
  int32_t num_materials = 0;
  bool has_normals = false;
  bool has_texcoords = false;

  if (obj_shapes.empty())
  {
    std::string err;
    bool ret = tinyobj::LoadObj(
        obj_shapes,
        obj_materials,
        err,
        m_filename.c_str(),
        m_filename.substr(0, m_filename.find_last_of("\\/") + 1).c_str());

    if (!err.empty())
      cerr << err << endl;

    if (!ret)
      throw runtime_error("MeshLoader: " + err);
  }

  //
  // Iterate over all shapes and sum up number of vertices and triangles
  //
  uint64_t num_groups_with_normals = 0;
  uint64_t num_groups_with_texcoords = 0;
  for (vector<tinyobj::shape_t>::const_iterator it = obj_shapes.begin(); it < obj_shapes.end(); ++it)
  {
    const tinyobj::shape_t &shape = *it;

    num_triangles += static_cast<int32_t>(shape.mesh.indices.size()) / 3;
    num_vertices += static_cast<int32_t>(shape.mesh.positions.size()) / 3;

    if (!shape.mesh.normals.empty())
      ++num_groups_with_normals;

    if (!shape.mesh.texcoords.empty())
      ++num_groups_with_texcoords;
  }

  //
  // We ignore normals and texcoords unless they are present for all shapes
  //
  if (num_groups_with_normals != 0)
  {
    if (num_groups_with_normals != obj_shapes.size())
      cerr << "MeshLoader - WARNING: mesh '" << m_filename
           << "' has normals for some groups but not all.  "
           << "Ignoring all normals." << endl;
    else
      has_normals = true;
  }
  if (num_groups_with_texcoords != 0)
  {
    if (num_groups_with_texcoords != obj_shapes.size())
      cerr << "MeshLoader - WARNING: mesh '" << m_filename
           << "' has texcoords for some groups but not all.  "
           << "Ignoring all texcoords." << endl;
    else
      has_texcoords = true;
  }
  num_materials = (int32_t)m_materials.size();
}

// #define GEN_NERF
void Scene::add_default_light()
{
  // The radiance of a directional source modeling the Sun should be equal
  // to the irradiance at the surface of the Earth.
  // We convert radiance to irradiance at the surface of the Earth using the
  // solid angle 6.74e-5 subtended by the solar disk as seen from Earth.

  // Default directional light
  lights.get_lights().resize(1);
  lights.get_lights()[0].emission = use_sunsky ? sun_sky.sunColor() * 6.74e-5f * (1.0f - sun_sky.getOvercast()) : light_rad;
  lights.get_lights()[0].direction = use_sunsky ? -sun_sky.getSunDir() : normalize(light_dir);
  lights.get_lights()[0].position = make_float3(30.0f, 30.0f, 30.0f);
  lights.get_lights()[0].type = POINTLIGHT;
  update_light();
#ifdef GEN_NERF
  Directional tmp;
  tmp.emission = make_float3(1.0f);
  tmp.direction = make_float3(1.0f, -1.0f, 1.0f);
  dir_lights.push_back(tmp);
  tmp.direction = make_float3(-1.0f, 1.0f, 1.0f);
  dir_lights.push_back(tmp);
  tmp.direction = make_float3(-1.0f, -1.0f, 1.0f);
  dir_lights.push_back(tmp);
  tmp.direction = make_float3(1.0f, 0.0f, 0.0f);
  dir_lights.push_back(tmp);
#endif
}

void Scene::update_light()
{
  if (m_gbx->launch_params.lights.count == 0)
  {
    m_gbx->launch_params.lights.count = static_cast<uint32_t>(lights.get_lights().size());
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_gbx->launch_params.lights.data),
                          lights.get_lights().size() * sizeof(SimpleLight)));
  }
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(m_gbx->launch_params.lights.data),
                        lights.get_lights().data(), lights.get_lights().size() * sizeof(SimpleLight),
                        cudaMemcpyHostToDevice));
}

void Scene::compute_diffuse_reflectance(MtlData &mtl)
{
  float n1_over_n2 = 1.0f / mtl.ior;
  float3 sca = mtl.alb * mtl.ext;
  float3 abs = mtl.ext - sca;
  float3 sca_p = sca * (1.0f - mtl.asym);
  float3 alb_p = sca_p / (sca_p + abs);
  float F_dr = fresnel_diffuse(n1_over_n2);
  float A = (1.0f + F_dr) / (1.0f - F_dr);
  float3 transport = sqrtf(3.0f * (1.0f - alb_p));
  mtl.rho_d = alb_p * 0.5f * (1.0f + expf(-4.0f / 3.0f * A * transport)) * expf(-transport);
}

void Scene::analyze_materials()
{
  vector<int> adv_mtls;
  auto &meshes = m_mesh_allocator.get_meshes();
  unsigned int mesh_idx = 0;
  bool calculate_area = false;
  for (auto mesh : meshes)
  {
    bool translucent = false;
    for (unsigned int j = 0; j < mesh->material_idx.size(); ++j)
    {
      int mtl_idx = mesh->material_idx[j];
      if (mtl_idx >= 0)
      {
        const MtlData &mtl = m_materials[mtl_idx];
        if (mtl.illum == 5 || mtl.illum > 10)
          adv_mtls.push_back(mtl_idx);
        if (mtl.illum == 23)
          m_pipeline.use_bdpt = true;
        translucent = true;
        calculate_area = true;
      }
    }
    if (calculate_area)
    {
      const Surface &surface = surfaces[mesh_idx];
      float surface_area = 0.0f;
      vector<float> face_areas(surface.indices.size());
      vector<float> face_area_cdf(surface.indices.size());
      for (unsigned int i = 0; i < surface.indices.size(); ++i)
      {
        uint3 face = surface.indices[i];
        float3 p0 = surface.positions[face.x];
        float3 a = surface.positions[face.y] - p0;
        float3 b = surface.positions[face.z] - p0;
        face_areas[i] = 0.5f * length(cross(a, b));
        face_area_cdf[i] = surface_area + face_areas[i];
        surface_area += face_areas[i];
      }
      m_gbx->launch_params.surface_area = surface_area;

      if (translucent && !m_pipeline.has_translucent)
      {
        m_pipeline.has_translucent = true;
        {
          BufferView<uint3> buffer_view;
          m_buffer_allocator.addBuffer(surface.indices.size() * sizeof(uint3), reinterpret_cast<const void *>(&surface.indices[0]));
          buffer_view.data = m_buffer_allocator.getBuffer(bufidx++);
          buffer_view.byte_stride = 0;
          buffer_view.count = static_cast<uint32_t>(surface.indices.size());
          buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(uint3));
          m_gbx->launch_params.translucent_idxs = buffer_view;
        }
        {
          BufferView<float3> buffer_view;
          m_buffer_allocator.addBuffer(surface.positions.size() * sizeof(float3), reinterpret_cast<const void *>(&surface.positions[0]));
          buffer_view.data = m_buffer_allocator.getBuffer(bufidx++);
          buffer_view.byte_stride = 0;
          buffer_view.count = static_cast<uint32_t>(surface.positions.size());
          buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
          m_gbx->launch_params.translucent_verts = buffer_view;
        }
        {
          BufferView<float3> buffer_view;
          m_buffer_allocator.addBuffer(surface.normals.size() * sizeof(float3), reinterpret_cast<const void *>(&surface.normals[0]));
          buffer_view.data = m_buffer_allocator.getBuffer(bufidx++);
          buffer_view.byte_stride = 0;
          buffer_view.count = static_cast<uint32_t>(surface.normals.size());
          buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
          m_gbx->launch_params.translucent_norms = buffer_view;
        }
        if (surface.normals.size() == 0)
          cerr << "Warning: Translucent object was loaded for surface sampling but has no vertex normals." << endl;
        if (surface_area > 0.0f)
          for (unsigned int i = 0; i < surface.indices.size(); ++i)
            face_area_cdf[i] /= surface_area;
        {
          BufferView<float> buffer_view;
          m_buffer_allocator.addBuffer(face_area_cdf.size() * sizeof(float), reinterpret_cast<const void *>(&face_area_cdf[0]));
          buffer_view.data = m_buffer_allocator.getBuffer(bufidx++);
          buffer_view.byte_stride = 0;
          buffer_view.count = static_cast<uint32_t>(face_area_cdf.size());
          buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float));
          m_gbx->launch_params.translucent_face_area_cdf = buffer_view;
        }
      }
    }
    ++mesh_idx;
  }

  load_mpml(std::string(SOURCE_DIR) + "/models/media.mpml", m_media, m_interfaces);
  list_mtl();
  if (m_pipeline.use_bdpt)
    std::cout << "\n\t\tEnabling light samples for Bi-directional tracing. \n\n"
              << std::endl;
}

unsigned int Scene::extract_area_lights()
{
  vector<uint2> lights;
  auto &meshes = m_mesh_allocator.get_meshes();
  int mesh_idx = 0;
  for (auto mesh : meshes)
  {
    for (unsigned int j = 0; j < mesh->material_idx.size(); ++j)
    {
      int mtl_idx = mesh->material_idx[j];
      const MtlData &mtl = m_materials[mtl_idx];
      bool emissive = false;
      for (unsigned int k = 0; k < 3; ++k)
        emissive = emissive || *(&mtl.emission.x + k) > 0.0f;
      if (emissive)
        lights.push_back(make_uint2(mesh_idx, mtl_idx));
    }
    ++mesh_idx;
  }
  Surface lightsurf;
  vector<float3> emission;
  for (unsigned int j = 0; j < lights.size(); ++j)
  {
    uint2 light = lights[j];
    auto mesh = meshes[light.x];
    const Surface &surface = surfaces[light.x];
    unsigned int no_of_verts = static_cast<unsigned int>(lightsurf.positions.size());
    lightsurf.positions.insert(lightsurf.positions.end(), surface.positions.begin(), surface.positions.end());
    lightsurf.normals.insert(lightsurf.normals.end(), surface.positions.begin(), surface.positions.end());
    lightsurf.indices.insert(lightsurf.indices.end(), surface.indices.begin(), surface.indices.end());
    if (surface.normals.size() > 0)
      for (unsigned int k = no_of_verts; k < lightsurf.normals.size(); ++k)
        lightsurf.normals[k] = surface.normals[k - no_of_verts];
    for (unsigned int k = lightsurf.no_of_faces; k < lightsurf.indices.size(); ++k)
    {
      lightsurf.indices[k] += make_uint3(no_of_verts);
      if (surface.normals.size() == 0)
      {
        uint3 face = lightsurf.indices[k];
        float3 p0 = lightsurf.positions[face.x];
        float3 a = lightsurf.positions[face.y] - p0;
        float3 b = lightsurf.positions[face.z] - p0;
        lightsurf.normals[face.x] = lightsurf.normals[face.y] = lightsurf.normals[face.z] = normalize(cross(a, b));
      }
    }
    emission.insert(emission.end(), surface.no_of_faces, m_materials[light.y].emission);
    lightsurf.no_of_faces += surface.no_of_faces;
  }
  {
    BufferView<uint3> buffer_view;
    m_buffer_allocator.addBuffer(lightsurf.indices.size() * sizeof(uint3), reinterpret_cast<const void *>(&lightsurf.indices[0]));
    buffer_view.data = m_buffer_allocator.getBuffer(bufidx++);
    buffer_view.byte_stride = 0;
    buffer_view.count = static_cast<uint32_t>(lightsurf.indices.size());
    buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(uint3));
    m_gbx->launch_params.light_idxs = buffer_view;
  }
  {
    BufferView<float3> buffer_view;
    m_buffer_allocator.addBuffer(lightsurf.positions.size() * sizeof(float3), reinterpret_cast<const void *>(&lightsurf.positions[0]));
    buffer_view.data = m_buffer_allocator.getBuffer(bufidx++);
    buffer_view.byte_stride = 0;
    buffer_view.count = static_cast<uint32_t>(lightsurf.positions.size());
    buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
    m_gbx->launch_params.light_verts = buffer_view;
  }
  {
    BufferView<float3> buffer_view;
    m_buffer_allocator.addBuffer(lightsurf.normals.size() * sizeof(float3), reinterpret_cast<const void *>(&lightsurf.normals[0]));
    buffer_view.data = m_buffer_allocator.getBuffer(bufidx++);
    buffer_view.byte_stride = 0;
    buffer_view.count = static_cast<uint32_t>(lightsurf.normals.size());
    buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
    m_gbx->launch_params.light_norms = buffer_view;
  }
  {
    BufferView<float3> buffer_view;
    m_buffer_allocator.addBuffer(emission.size() * sizeof(float3), reinterpret_cast<const void *>(&emission[0]));
    buffer_view.data = m_buffer_allocator.getBuffer(bufidx++);
    buffer_view.byte_stride = 0;
    buffer_view.count = static_cast<uint32_t>(emission.size());
    buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
    m_gbx->launch_params.light_emission = buffer_view;
  }
  float surface_area = 0.0f;
  vector<float> face_areas(lightsurf.no_of_faces);
  vector<float> face_area_cdf(lightsurf.no_of_faces);
  for (unsigned int i = 0; i < lightsurf.no_of_faces; ++i)
  {
    uint3 face = lightsurf.indices[i];
    float3 p0 = lightsurf.positions[face.x];
    float3 a = lightsurf.positions[face.y] - p0;
    float3 b = lightsurf.positions[face.z] - p0;
    face_areas[i] = 0.5f * length(cross(a, b));
    face_area_cdf[i] = surface_area + face_areas[i];
    surface_area += face_areas[i];
  }
  if (surface_area > 0.0f)
    for (unsigned int i = 0; i < lightsurf.no_of_faces; ++i)
      face_area_cdf[i] /= surface_area;
  m_gbx->launch_params.light_area = surface_area;
  {
    BufferView<float> buffer_view;
    m_buffer_allocator.addBuffer(face_area_cdf.size() * sizeof(float), reinterpret_cast<const void *>(&face_area_cdf[0]));
    buffer_view.data = m_buffer_allocator.getBuffer(bufidx++);
    buffer_view.byte_stride = 0;
    buffer_view.count = static_cast<uint32_t>(face_area_cdf.size());
    buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float));
    m_gbx->launch_params.light_face_area_cdf = buffer_view;
  }
  return static_cast<unsigned int>(lights.size());
}

sutil::Transform Scene::get_object_transform(string filename) const
{
  size_t idx = filename.find_last_of("\\/") + 1;

  struct sutil::Transform t;
  if (idx >= filename.length())
    return t;

  if (filename.compare(idx, 7, "cornell") == 0)
  {
    t.scale = make_float3(0.025f);
    t.rotate = make_float3(0.0f, M_PIf, 0.0f);
  }
  else if (filename.compare(idx, 6, "dragon") == 0)
  {
    t.rotate = make_float3(-M_PI_2f, 0.0, 0.0);
  }
  else if (filename.compare(idx, 5, "bunny") == 0)
  {
    t.translate = make_float3(0.0f);
    t.scale = make_float3(25.0f);
    t.rotate = make_float3(0.0f);
  }
  else if (filename.compare(idx, 12, "justelephant") == 0)
  {
    t.translate = make_float3(-10.0f, 3.0f, -2.0f);
    t.rotate = make_float3(0.0f, 0.5f, 0.0f);
  }
  else if (filename.compare(idx, 10, "glass_wine") == 0)
  {
    t.scale = make_float3(5.0f);
  }
  return t;
}

void Scene::set_materialdata(std::string &selecteditem, MtlData &data)
{
  if (m_materialdata.find(selecteditem) != m_materialdata.end())
  {
    m_materialdata.find(selecteditem)->second.first = data;
  }
  else
  {
    std::cout << "Material not found " << std::endl;
  }
  m_pipeline.create_hit_group(get_meshes(), get_instances(), m_materialdata, m_materials);
}

void Scene::replace_material(std::string &name, int const &idx, int &illum, float &noise, float &density)
{
  MtlData tmp;
  auto m_i = m_media.find(name);
  auto i_i = m_interfaces.find(name);

  Medium *med = nullptr;
  if (m_i != m_media.end())
    med = &m_i->second;
  else if (i_i != m_interfaces.end())
    med = i_i->second.med_in;
  if (med)
  {
    med->fill_rgb_data(true);
    Color<complex<double>> &ior = med->get_ior(rgb);
    Color<double> &alb = med->get_albedo(rgb);
    Color<double> &ext = med->get_extinction(rgb);
    Color<double> &asym = med->get_asymmetry(rgb);
    Color<double> &scat = med->get_scattering(rgb);
    Color<double> &absp = med->get_absorption(rgb);
    m_complex ior_x = {static_cast<float>(ior[0].real()), static_cast<float>(ior[0].imag())};
    m_complex ior_y = {static_cast<float>(ior[1].real()), static_cast<float>(ior[1].imag())};
    m_complex ior_z = {static_cast<float>(ior[2].real()), static_cast<float>(ior[2].imag())};
    tmp.ior = static_cast<float>(ior[0].real() + ior[1].real() + ior[2].real()) / 3.0f;
    tmp.c_recip_ior = {1.0f / ior_x, 1.0f / ior_y, 1.0f / ior_z};
    tmp.alb = make_float3(static_cast<float>(alb[0]), static_cast<float>(alb[1]), static_cast<float>(alb[2]));
    tmp.ext = make_float3(static_cast<float>(ext[0]), static_cast<float>(ext[1]), static_cast<float>(ext[2])) * scene_scale;
    tmp.asym = make_float3(static_cast<float>(asym[0]), static_cast<float>(asym[1]), static_cast<float>(asym[2]));
    tmp.scat = make_float3(static_cast<float>(scat[0]), static_cast<float>(scat[1]), static_cast<float>(scat[2])) * scene_scale;
    tmp.absp = make_float3(static_cast<float>(absp[0]), static_cast<float>(absp[1]), static_cast<float>(absp[2])) * scene_scale;
    tmp.material_scale = scene_scale;
    tmp.illum = illum;
    tmp.noise_scale = noise;
    tmp.density = density;
  }
  m_materials[idx] = tmp;
}

void Scene::list_mtl()
{
  std::pair<MtlData, MtlData> t_p_mtl;
  for (unsigned int j = 0; j < m_materials.size(); j++)
  {
    m_all_mtl.push_back(mtl_names[j]);
    t_p_mtl.first = m_materials[j], t_p_mtl.second = MtlData();
    m_materialdata[mtl_names[j]] = t_p_mtl;
  }

  for (auto it = m_media.begin(); it != m_media.end(); ++it)
  {
    m_all_mtl.push_back(it->first);
    MtlData t_mtl = fill_rgb(&it->second);
    t_p_mtl.first = t_mtl, t_p_mtl.second = MtlData();
    m_materialdata[it->first] = t_p_mtl;
  }

  for (auto it = m_interfaces.begin(); it != m_interfaces.end(); ++it)
  {
    m_all_mtl.push_back(it->first);
    MtlData t1_mtl = fill_rgb(it->second.med_in);
    MtlData t2_mtl = fill_rgb(it->second.med_out);
    t_p_mtl.first = t1_mtl, t_p_mtl.second = t2_mtl;
    m_materialdata[it->first] = t_p_mtl;
  }
}

void Scene::dealloc_output_buffer()
{
  int size_buffer = width * height * 4;
  uchar4 *result_buffer_data = output_buffer.map();
  CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(result_buffer_data), 0, size_buffer * sizeof(unsigned char)));
  CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(m_gbx->launch_params.accum_buffer), 0, size_buffer * sizeof(float)));
  output_buffer.unmap();
  CUDA_SYNC_CHECK();
}

MtlData Scene::fill_rgb(Medium *med)
{
  if (med)
  {
    MtlData mtl;
    med->fill_rgb_data();
    Color<complex<double>> &ior = med->get_ior(rgb);
    Color<double> &alb = med->get_albedo(rgb);
    Color<double> &ext = med->get_extinction(rgb);
    Color<double> &asym = med->get_asymmetry(rgb);
    Color<double> &scat = med->get_scattering(rgb);
    Color<double> &absp = med->get_absorption(rgb);
    m_complex ior_x = {static_cast<float>(ior[0].real()), static_cast<float>(ior[0].imag())};
    m_complex ior_y = {static_cast<float>(ior[1].real()), static_cast<float>(ior[1].imag())};
    m_complex ior_z = {static_cast<float>(ior[2].real()), static_cast<float>(ior[2].imag())};
    mtl.ior = static_cast<float>(ior[0].real() + ior[1].real() + ior[2].real()) / 3.0f;
    mtl.c_recip_ior = {1.0f / ior_x, 1.0f / ior_y, 1.0f / ior_z};
    mtl.alb = make_float3(static_cast<float>(alb[0]), static_cast<float>(alb[1]), static_cast<float>(alb[2]));
    mtl.ext = make_float3(static_cast<float>(ext[0]), static_cast<float>(ext[1]), static_cast<float>(ext[2])) * scene_scale;
    mtl.asym = make_float3(static_cast<float>(asym[0]), static_cast<float>(asym[1]), static_cast<float>(asym[2]));
    mtl.scat = make_float3(static_cast<float>(scat[0]), static_cast<float>(scat[1]), static_cast<float>(scat[2])) * scene_scale;
    mtl.absp = make_float3(static_cast<float>(absp[0]), static_cast<float>(absp[1]), static_cast<float>(absp[2])) * scene_scale;
    mtl.material_scale = scene_scale;
    compute_diffuse_reflectance(mtl);
    return mtl;
  }
  else
    return MtlData();
}
