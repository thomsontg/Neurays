#pragma once

#include <iostream>
#include <fstream>
#include <vector>

#include "scene.h"
#include <misc/Spline.h>
#include "sutil/sutil.h"
#include "sutil/shared/structs.h"

struct Globals
{
  sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

  int32_t rd_width = 500;
  int32_t rd_height = 500;
  int32_t panel_width = 300;
  int32_t win_width = rd_width + panel_width;
  int32_t win_height = rd_height;

  float aspect_ratio = 1.0f;
  int frame_dim[2] = {rd_width, rd_height};
  bool custom_title_bar = true;

  bool first_loop = false;
  bool resize_dirty = false;
  bool minimized = false;
  bool save_image = false;
  bool progressive = true;
  bool pause_render = false;

  FileType file_type = EXR;

  DataGen gen_type = GEN_NONE;
  std::vector<std::string> env_list;
  unsigned int env_counter = 0;
  unsigned int datapoint_num = 0;
  unsigned int total_datapoints = 200;

  // Data Pointers
  float3* env = new float3[rd_width*rd_height];

  std::ofstream train;
  std::ofstream test;
  std::ofstream val;
  std::ofstream data_file;
  std::ofstream error_file;

  //
  // Parse command line options
  //
  std::vector<std::string> filenames;
  std::string filename;
  std::string shadername;
  std::string outfile;
  std::string networkdir;
  bool outfile_selected = false;
  bool use_sunsky = false;
  bool bsdf = false;
  bool run_network = false;
  // bool network_run_stat =  false;
  float latitude = 55.78f;
  float turbidity = 2.4f;
  float ordinal_day = 255.0f;
  float sky_angle = 0.0f;
  float3 sky_up = make_float3(0.0f, 1.0f, 0.0f);
  unsigned int samples = 16;
  float scene_scale = 1.0e-2f;

  // Drop handle
  bool obj_dropped = false;
  bool env_dropped = false;
  bool scene_dropped = false;
  bool ref_dropped = false;
  std::vector<std::string> dropped_files;

  // Sky state
  bool sky_changed = false;
  float time_of_day = 12.0f;
  float clouds = 0.0f;

  // Light state
  float3 light_dir = make_float3(1.0f, 1.0f, 1.0f);
  float3 emission = make_float3(M_PIf) * 0.0f;
  bool light_changed = false;
  float theta_i = 54.7356f;
  float phi_i = 45.0f;
  std::string scene_name = "";
  std::string env_name = "";
  std::string reference_name = "";

  // Mouse state
  sutil::Camera *camera = nullptr;
  bool camera_changed = true;
  QuatTrackBall *trackball = 0;
  int32_t mouse_button = -1;
  float cam_const = 0.41421356f;

  GLFWwindow *window;

  // Render state
  bool reset = false;
  bool save_scene = false;
  bool xml_loading = false;
  bool stats_on_render = false;
  bool show_panels = false;
  bool save_as_open = false;
  bool compute_error = false;
  bool update_bbox = true;
  bool show_panel_changed = false;
  bool material_changed = false;
  bool settings_changed = false;
  bool videoset_changed = false;
  bool instance_transformed = false;
  bool add_instance = false;
  bool delete_instance = false;
  bool delete_all_instance = false;
  int mesh_idx;

  std::string materialname;
  MtlData mtldata;

  // Video render
  bool video = false;
  bool video_save = false;
  int frame = 0;
  float vid_len = 6.0f; // in seconds for 30 fps
  int keypoints = 5;
  int video_samples = 500;
  float step = 0.0, x;

  // video recording
  int view = 0;
  SplineData splinedata;

  void save_view(const std::string &filename);
  void load_view(const std::string &filename);
  float3 get_light_direction();

  float* error = new float;

  LaunchParams launch_params;
};
