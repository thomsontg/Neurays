// Based on NVIDIA's OptiX implementation of 2009.
// Code modified by Jeppe Revall Frisvad, 2013 and 2021.

#pragma once

#include <optix.h>
#include <src/sutil/math/vec_math.h>
#include <src/device/cuda/color.h>

//------------------------------------------------------------------------------
//
// Implements the Preetham analytic sun/sky model ( Preetham, SIGGRAPH 99 )
//
//------------------------------------------------------------------------------

struct Preetham
{
  float3 sky_up;
  float3 sun_color;
  float3 sun_dir;
  float overcast;
  float3 c0;
  float3 c1;
  float3 c2;
  float3 c3;
  float3 c4;
  float3 inv_divisor_Yxy;
  float tonemap;
};

class PreethamSunSky
{
public:
  PreethamSunSky();

  void setSunTheta(float sun_theta)
  {
    _sun_theta = sun_theta;
    _dirty = true;
  }
  void setSunPhi(float sun_phi)
  {
    _sun_phi = sun_phi;
    _dirty = true;
  }
  void setTurbidity(float turbidity)
  {
    _turbidity = turbidity;
    _dirty = true;
  }

  void setUpDir(const float3 &up)
  {
    _p.sky_up = up;
    _dirty = true;
  }
  void setOvercast(float overcast)
  {
    _p.overcast = overcast;
    _p.tonemap = 0.1f / (1.0f + expf(-2.0f * (overcast - 0.5f)));
  }

  float getSunTheta() { return _sun_theta; }
  float getSunPhi() { return _sun_phi; }
  float getTurbidity() { return _turbidity; }

  float getOvercast() { return _p.overcast; }
  float3 getUpDir() { return _p.sky_up; }
  float3 getSunDir()
  {
    update();
    return _p.sun_dir;
  }

  const Preetham &getParams() { return _p; }

  // Query the sun color at current sun position and air turbidity
  float3 sunColor();

  // Query the sky color in a given direction
  float3 skyColor(const float3 &direction, bool CEL = false);

  void init(float ordinal_day, float solar_time, float globe_latitude,
            float turbidity, float overcast, float sky_angle,
            const float3 &sky_up);
  void update();
  void updateParams(float solar_time, float overcast);

private:
  // Represents one entry from table 2 in the paper
  struct Datum
  {
    float wavelength;
    float sun_spectral_radiance;
    float k_o;
    float k_wa;
  };

  static const float cie_table[38][4]; // CIE spectral sensitivy curves
  static const Datum data[38];         // Table2

  // Calculate absorption for a given wavelength of direct sunlight
  static float calculateAbsorption(float sun_theta, // Sun angle from zenith
                                   float m,         // Optical mass of atmosphere
                                   float lambda,    // light wavelength
                                   float turbidity, // atmospheric turbidity
                                   float k_o,       // atten coeff for ozone
                                   float k_wa);     // atten coeff for h2o vapor

  // Model parameters
  float _sun_theta;
  float _sun_phi;
  float _turbidity;
  bool _dirty;
  Preetham _p;
  float day_of_year;
  float time_of_day;
  float latitude;
  float angle_with_south;
};

// Preetham skylight model
SUTILFN float3 clear_sky_color(float3 ray_dir, const Preetham &p)
{
  float inv_dir_dot_up = 1.f / dot(ray_dir, p.sky_up);
  if (inv_dir_dot_up < 0.f)
  {
    ray_dir = reflect(ray_dir, p.sky_up);
    inv_dir_dot_up = -inv_dir_dot_up;
  }

  float gamma = dot(p.sun_dir, ray_dir);
  float acos_gamma = acosf(gamma);
  float3 A = p.c1 * inv_dir_dot_up;
  float3 B = p.c3 * acos_gamma;
  float3 color_Yxy =
      (1.0f + p.c0 * expf(A)) * (1.0f + p.c2 * expf(B) + p.c4 * gamma * gamma);
  color_Yxy *= p.inv_divisor_Yxy;

  float3 color_XYZ = color::Yxy2XYZ(color_Yxy);
  return color::XYZ2rgb(color_XYZ) * p.tonemap /
         683.0f; // return result in W/m^2 (using a tone map scale to handle
                 // dynamic range)
}

// CIE standard overcast sky model
SUTILFN float3 overcast_sky_color(const float3 &ray_dir,
                                  const Preetham &p)
{
  float Y = 15.0f;
  return make_float3(Y * (1.0f + 2.0f * fabsf(dot(ray_dir, p.sky_up))) / 3.0f) *
         p.tonemap; // (a scale to handle dynamic range)
}
