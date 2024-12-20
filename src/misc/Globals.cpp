#include "Globals.h"

void Globals::save_view(const std::string &filename)
{
  if (trackball)
  {
    std::ofstream ofs(filename.c_str(), std::ofstream::binary);
    if (ofs)
    {
    ofs.write(reinterpret_cast<const char *>(trackball), sizeof(QuatTrackBall));
    ofs.write(reinterpret_cast<const char *>(&cam_const), sizeof(float));
    }
    ofs.close();
    std::cout << "Camera settings stored in a file called " << filename << std::endl;
  }
}

void Globals::load_view(const std::string &filename)
{
  if (trackball)
  {
    std::ifstream ifs_view(filename.c_str(), std::ifstream::binary);
    if (ifs_view)
    {
    ifs_view.read(reinterpret_cast<char *>(trackball), sizeof(QuatTrackBall));
    ifs_view.read(reinterpret_cast<char *>(&cam_const), sizeof(float));
    }
    ifs_view.close();
    float3 eye, lookat, up;
    float vfov = atanf(cam_const) * 360.0f * M_1_PIf;
    trackball->get_view_param(eye, lookat, up);
    std::cout << "Loaded view: eye [" << eye.x << ", " << eye.y << ", " << eye.z
            << "], lookat [" << lookat.x << ", " << lookat.y << ", " << lookat.z
            << "], up [" << up.x << ", " << up.y << ", " << up.z
            << "], vfov " << vfov << std::endl;
    camera_changed = true;
  }
}

float3 Globals::get_light_direction()
{
  float theta = theta_i * M_PIf / 180.0f;
  float phi = phi_i * M_PIf / 180.0f;
  float sin_theta = sinf(theta);
  return -make_float3(sin_theta * cosf(phi), sin_theta * sinf(phi), cosf(theta));
}
