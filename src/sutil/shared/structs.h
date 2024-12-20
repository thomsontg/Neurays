#pragma once

#include <optix.h>
#include <src/sutil/math/mat3.h>
#include "src/device/cuda/util.h"
// #include <sutil/CUDAOutputBuffer.h>
#include <src/device/cuda/GeometryData.h>
#include <src/device/cuda/random.h>
#include <texture_types.h>
#include <src/sutil/math/complex.h>
#include "src/render/Lights/SunSky.h"
// #include <src/Render/Lights/sim_Light.h>
// #include <build/src/Config.h>
// #include <lib/tiny-cuda-nn/include/tiny-cuda-nn/common_host.h>

// #define PASS_PAYLOAD_POINTER

enum FileType
{
  PNG,
  PPM,
  EXR,
  RAW,
};

enum DataGen 
{
    GEN_NONE,
    GEN_NERF,
    GEN_NETWORK,
    GEN_INVERSE,
    GEN_DISTANCE,
    GEN_ERROR
};

// struct Photon
// {
//   float weight;
//   float3 w;      // Direction
//   float3 x;      // Position
//   int layer_idx; // Layer photn currently in
// };

enum Mode
{
  REFLECT,
  TRANSMIT,
};

struct Microfacet
{
  float mean = 0.0f;
  float roughness;
  float scale = 0.0f;
};

enum BSDF_type
{
  SMOOTH = 0,
  MICROFACET = 1,
  MEASURED = 2,
};

struct BSDF
{
  enum BSDF_type type;
  Microfacet ggx;
};

struct Layer
{
  struct BSDF bsdf;

  // layer start and end along z-axis
  float thickness;
  float start;
  float end;

  // Optical properties
  float ior;
  float albedo;
  float extinction;
  float asymmetry;
};

enum LightType
{
  POINTLIGHT,
  DIRECTIONALLIGHT,
};

// light vertex holding
struct LightVertex
{
  float3 vertex;
  float dist;
};

// Forward declarations
struct SimpleLight
{
  float3 direction;
  float3 emission;
  float3 position;
  LightType type;
};

enum Lighting
{
  LIGHTING_SIMPLE,
  LIGHTING_ENVMAP,
  LIGTHING_AREALIGHT,
  LIGHTING_SUNSKY
};

// TODO handle this name conflict
enum MissType
{
  MISS_CONSTANT,
  MISS_ENVMAP,
  MISS_SUNSKY,
  MISS_DIRECTION
};

struct PositionSample
{
  float3 pos;
  float3 dir;
  float3 normal;
  float3 L;
};

namespace infnetwork
{

  const unsigned int input_size = 9;
  const unsigned int output_size = 3;
  const unsigned int layer_size = 256;
  const unsigned int embedding_size = 4;

  struct data
  {
    // Layer 0 (intermediate_n x 10)
    BufferView<float> w0;
    BufferView<float> b0;

    // Layer 1 (intermediate_n x intermediate_n)
    BufferView<float> w1;
    BufferView<float> b1;

    // Layer 2 (intermediate_n x intermediate_n)
    BufferView<float> w2;
    BufferView<float> b2;

    // Layer 3 (intermediate_n x intermediate_n)
    BufferView<float> w3;
    BufferView<float> b3;

    // Layer 4 (intermediate_n x intermediate_n)
    BufferView<float> w4;
    BufferView<float> b4;

    // Layer 5 (intermediate_n x intermediate_n)
    BufferView<float> w5;
    BufferView<float> b5;

    // Layer 6 (intermediate_n x intermediate_n)
    BufferView<float> w6;
    BufferView<float> b6;

    // Layer 7 (intermediate_n x intermediate_n)
    BufferView<float> w7;
    BufferView<float> b7;

    // Layer 8 (3 x intermediate_n)
    BufferView<float> w8;
    BufferView<float> b8;

    // BufferView<float> flux;

    // Beckmann layer
    // BufferView<float> beckmann_shininess;

    float3 normal;
  };
} // network

struct LaunchParams
{
  unsigned int subframe_index;
  float4 *accum_buffer;
  uchar4 *frame_buffer;
  unsigned int width, height;
  unsigned int max_depth;
  unsigned int no_of_samples;
  color::ColorMode tone_mapping;

  float *hit_data;
  float3 *ray_dir;
  float3 *buffer;

  float4 *temp;
  float max_bbox;

  float firefly_threshold;
  unsigned int max_volume_bounces;

  // std::vector<std::vector<float>> data_input;
  // std::vector<std::vector<float>> data_output;
  int batchsize;

  float3 *result_buffer;
  float3 light_dir;
  float *result_hit_data;
  float3 *result_ray_dir;
  float2 *result_ray_vec;
  unsigned int buffer_width;
  unsigned int buffer_height;
  float3 *pos;
  float3 m_eye;
  float3 m_lookat;
  float3 m_up;
  float3 *result_pos;

  // Camera parameters
  float3 eye;
  float3 U;
  float3 V;
  float3 W;
  mat3 inv_calibration_matrix;

  // Lights
  enum Lighting lightType;
  enum MissType missType;
  BufferView<SimpleLight> lights;
  BufferView<LightVertex> light_vertex;

  // Arealight
  BufferView<float3> light_verts;
  BufferView<float3> light_norms;
  BufferView<uint3> light_idxs;
  BufferView<float3> light_emission;
  BufferView<float> light_face_area_cdf;
  float light_area;
  Preetham sunsky;
  float beam_factor;

  BufferView<float3> translucent_verts;
  BufferView<float3> translucent_norms;
  BufferView<uint3> translucent_idxs;
  BufferView<float> translucent_face_area_cdf;
  PositionSample *translucent_samples;
  unsigned int translucent_no_of_samples;
  float surface_area;

  cudaTextureObject_t envmap;
  float env_scale;
  float envmap_phi;
  unsigned int env_width;
  unsigned int env_height;
  float *env_luminance;
  float *marginal_f;
  float *marginal_pdf;
  float *marginal_cdf;
  float *conditional_pdf;
  float *conditional_cdf;

  // Error comparison
  float4* ref_img;
  unsigned int ref_width, ref_height;
  float* error;

  // Bi-directional Samples
  unsigned int bdsample_count;
  float3 *bd_samples;

  // Micro Geometry
  bool invert_distances;
  float3 geom_center;

  // Network
  infnetwork::data inf_network;

  float3 miss_color;
  OptixTraversableHandle handle;

  // Denoiser
  float3 *denoise_normals;
  float3 *denoise_albedo;

  // Simulator Params
  bool use_rainbow;
  Mode mode;

  // Number of layers
  unsigned int n_layers;
  // Layers
  BufferView<Layer> layers;

  BufferView<float> light_flux;     //
  BufferView<float> light_flux_cdf; //
  BufferView<float> light_flux_pdf; //
  unsigned int n_light_flux;
  float total_flux;
  float mid_i, mid_j; //
  float dx;
  // Light light;
  float3 normal;           //
  float3 wo;               //
  bool include_reflection; //

  float pixel_size; //
  float film_width;
  float film_height;

  // End Simulator Params

  void update_networkdata(float3 &light, const float3 &eye, const float3 &lookat, const float3 &up)
  {

    // delete[] (result_hit_data);
    // delete[] (result_ray_dir);
    // delete[] (result_ray_vec);
    delete[] (result_buffer);
    // delete[] (result_pos);

    light_dir = light;
    m_eye = eye;
    m_lookat = lookat;
    m_up = up;
    int size_buffer = buffer_width * buffer_height;
    // float* result_buffer = new float[size_buffer];
    // result_hit_data = new float[size_buffer];
    // result_ray_dir = new float3[size_buffer];
    // result_pos = new float3[size_buffer];
    // result_ray_vec = new float2[size_buffer];
    result_buffer = new float3[size_buffer];
    // temp = new float4[size_buffer];
    cudaMemcpyAsync(result_buffer, buffer, size_buffer * sizeof(float3), cudaMemcpyDeviceToHost, 0);
    // cudaMemcpyAsync(result_ray_dir, ray_dir, size_buffer * sizeof(float3), cudaMemcpyDeviceToHost, 0);
    // cudaMemcpyAsync(temp, accum_buffer, size_buffer * sizeof(float4), cudaMemcpyDeviceToHost, 0);
    // cudaMemcpyAsync(result_pos, pos, size_buffer * sizeof(float3), cudaMemcpyDeviceToHost, 0);
    // for (int i = 0; i < size_buffer; i++)
    // {

    //   float theta_o = atan2(result_ray_dir[i].y, result_ray_dir[i].x);
    //   float phi_o = acos(result_ray_dir[i].z);

    //   result_ray_vec[i] = make_float2(theta_o, phi_o);
    // }
  }
};

enum RayType
{
  RAY_TYPE_RADIANCE = 0,
  RAY_TYPE_OCCLUSION = 1,
  RAY_TYPE_COUNT = 2
};

struct PayloadRadiance
{
  float3 origin;
  float3 direction;
  float3 result;
  float3 attenuation;
  unsigned int depth;
  unsigned int seed;
  unsigned int emit;
  unsigned int dead;
  unsigned int intersect;
};

const unsigned int NUM_PAYLOAD_VALUES = 17u; // no. of 32 bit fields in payload

struct PayloadOcclusion
{
};

struct PayloadFeeler
{
  float dist;
  float3 normal;
  float n1_over_n2;
};

struct MtlData
{
  float3 rho_d = {1.0f, 1.0f, 1.0f};
  float3 rho_s = {0.0f, 0.0f, 0.0f};
  float3 emission = {0.0f, 0.0f, 0.0f};
  float shininess = 0.0f;
  float ior = 1.0f;
  int illum = 1;
  complex3 c_recip_ior = {{1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f}};
  float3 alb = {1.0f, 1.0f, 1.0f};
  float3 ext = {0.0f, 0.0f, 0.0f};
  float3 absp = {0.0f, 0.0f, 0.0f};
  float3 asym = {0.0f, 0.0f, 0.0f};
  float3 scat = {0.0f, 0.0f, 0.0f};
  cudaTextureObject_t base_color_tex = 0;
  cudaTextureObject_t bump_map_tex = 0;
  cudaTextureObject_t alpha_map_tex = 0;
  float material_scale = 1.0f;
  int opposite = -1;
  int translucent_index = -1;
  float noise_scale = 1.0f;
  float density = 1.0f;

  void update_scale(float new_scale)
  {
    float ratio = new_scale / material_scale;
    material_scale = new_scale;

    ext *= ratio;
    scat *= ratio;
    absp *= ratio;
  }
};

struct HitGroupData
{
  GeometryData geometry;
  MtlData mtl_inside;
  MtlData mtl_inside_2;
  MtlData mtl_outside;
};
