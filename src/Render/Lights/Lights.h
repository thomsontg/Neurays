#pragma once

#include <vector>
#include <vector_types.h>
// #include <Render/Materials/Materials.h>
#include <sutil/shared/structs.h>

#include "Envmap.h"

class Lights
{
public:
  Lights(){};

  void init(const std::string &env_file, const float3 &light_direction,
            const float3 &light_emission)
  {
    light_dir = light_direction;
    light_rad = light_emission;
    env.set_envmap_file(env_file);
  }

  // Basic lights
  void add_default_light(LaunchParams &lp);
  void update_light(LaunchParams &lp);
  std::vector<SimpleLight> &get_lights() { return lights; }

  // Arealight
  // unsigned int extract_area_lights(
  //     LaunchParams &lp, std::vector<std::shared_ptr<sutil::MeshGroup>> &meshes,
  //     Materials &materials, const std::vector<Surface> &surfaces,
  //     sutil::BufferAlloc &alloc, bool first);

  // envmap
  const bool use_envmap() const { return env.use_envmap(); }
  void loadEnvMap(sutil::BufferAlloc &b_alloc, sutil::SamplerAlloc &s_alloc,
                  LaunchParams *lp)
  {
    env.set_envmap_file(env_file);
    env.load(b_alloc, s_alloc, lp);
  }

  void replaceEnvMap(sutil::BufferAlloc &b_alloc, sutil::SamplerAlloc &s_alloc,
                     LaunchParams *lp)
  {
    env.set_envmap_file(env_file);
    env.replace(b_alloc, s_alloc, lp);
  }

  void set_envfile(std::string &envfile) { this->env_file = envfile; }

  // Sunsky
  void initSunSky(float ordinal_day, float solar_time, float globe_latitude,
                  float turbidity, float overcast, float sky_angle,
                  const float3 &sky_up);
  void handleSunSkyUpdate(LaunchParams &lp, float &solar_time, float overcast);
  bool use_sunsky() { return sunsky; }
  PreethamSunSky &get_sunsky() { return sun_sky; }

private:
  // Default light configuration
  std::vector<SimpleLight> lights;
  float3 light_dir;
  float3 light_rad;

  std::string env_file;
  Envmap env;
  bool sunsky = false;
  PreethamSunSky sun_sky;
  bool has_arealight = false;
};
