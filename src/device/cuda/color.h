#ifndef _COLOR_H_
#define _COLOR_H_

#include <src/Render/optics/spectrum2rgb.h>

namespace color {

enum ColorMode { COLORMODE_NONE = 0, COLORMODE_SRGB = 1 };

const float cie_table[38][4] = {
    {380.f, 0.0002f, 0.0000f, 0.0007f}, {390.f, 0.0024f, 0.0003f, 0.0105f},
    {400.f, 0.0191f, 0.0020f, 0.0860f}, {410.f, 0.0847f, 0.0088f, 0.3894f},
    {420.f, 0.2045f, 0.0214f, 0.9725f},

    {430.f, 0.3147f, 0.0387f, 1.5535f}, {440.f, 0.3837f, 0.0621f, 1.9673f},
    {450.f, 0.3707f, 0.0895f, 1.9948f}, {460.f, 0.3023f, 0.1282f, 1.7454f},
    {470.f, 0.1956f, 0.1852f, 1.3176f},

    {480.f, 0.0805f, 0.2536f, 0.7721f}, {490.f, 0.0162f, 0.3391f, 0.4153f},
    {500.f, 0.0038f, 0.4608f, 0.2185f}, {510.f, 0.0375f, 0.6067f, 0.1120f},
    {520.f, 0.1177f, 0.7618f, 0.0607f},

    {530.f, 0.2365f, 0.8752f, 0.0305f}, {540.f, 0.3768f, 0.9620f, 0.0137f},
    {550.f, 0.5298f, 0.9918f, 0.0040f}, {560.f, 0.7052f, 0.9973f, 0.0000f},
    {570.f, 0.8787f, 0.9556f, 0.0000f},

    {580.f, 1.0142f, 0.8689f, 0.0000f}, {590.f, 1.1185f, 0.7774f, 0.0000f},
    {600.f, 1.1240f, 0.6583f, 0.0000f}, {610.f, 1.0305f, 0.5280f, 0.0000f},
    {620.f, 0.8563f, 0.3981f, 0.0000f},

    {630.f, 0.6475f, 0.2835f, 0.0000f}, {640.f, 0.4316f, 0.1798f, 0.0000f},
    {650.f, 0.2683f, 0.1076f, 0.0000f}, {660.f, 0.1526f, 0.0603f, 0.0000f},
    {670.f, 0.0813f, 0.0318f, 0.0000f},

    {680.f, 0.0409f, 0.0159f, 0.0000f}, {690.f, 0.0199f, 0.0077f, 0.0000f},
    {700.f, 0.0096f, 0.0037f, 0.0000f}, {710.f, 0.0046f, 0.0018f, 0.0000f},
    {720.f, 0.0022f, 0.0008f, 0.0000f},

    {730.f, 0.0010f, 0.0004f, 0.0000f}, {740.f, 0.0005f, 0.0002f, 0.0000f},
    {750.f, 0.0003f, 0.0001f, 0.0000f}};

/********************************************/
SUTILFN float3 Yxy2XYZ(const float3 &Yxy)
/********************************************/
{
  return make_float3(Yxy.y * (Yxy.x / Yxy.z), Yxy.x,
                     (1.0f - Yxy.y - Yxy.z) * (Yxy.x / Yxy.z));
}

/*
// sRGB color space
__host__ SUTILFN float3 XYZ2rgb( const float3& xyz)
{
  const float R = dot( xyz, make_float3(  3.2410f, -1.5374f, -0.4986f ) );
  const float G = dot( xyz, make_float3( -0.9692f,  1.8760f,  0.0416f ) );
  const float B = dot( xyz, make_float3(  0.0556f, -0.2040f,  1.0570f ) );
  return make_float3( R, G, B );
}

// NTSC color space
__host__ SUTILFN float3 XYZ2rgb( const float3& xyz)
{
  const float R = dot( xyz, make_float3(  1.9100f, -0.5325f, -0.2882f ) );
  const float G = dot( xyz, make_float3( -0.9847f,  1.9992f, -0.0283f ) );
  const float B = dot( xyz, make_float3(  0.0583f, -0.1184f,  0.8976f ) );
  return make_float3( R, G, B );
}
*/
// Wide gamut color space
/********************************************/
SUTILFN float3 XYZ2rgb(const float3 &xyz)
/********************************************/
{
  const float R = dot(xyz, make_float3(1.4625f, -0.1845f, -0.2734f));
  const float G = dot(xyz, make_float3(-0.5228f, 1.4479f, 0.0681f));
  const float B = dot(xyz, make_float3(0.0346f, -0.0958f, 1.2875f));
  return make_float3(R, G, B);
}

/********************************************/
template <unsigned int spec_n = 4>
SUTILFN float3 spectrum_to_rgb(const float spec[spec_n])
/********************************************/
{
  float3 v_rgb = make_float3(0.0f);
  double rgb_sums[3] = {0.0, 0.0, 0.0};
  double lambda = 450.0;
  double step_size = 50.0;
  for (unsigned int i = 0; i < spec_n; ++i) {
    unsigned int idx = get_nearest_rgb_index(lambda);

    if (idx == spectrum_rgb_samples - 1)
      break;
    double lambda_clamped = fminf(fmaxf(lambda, spectrum_rgb[0][0]),
        spectrum_rgb[spectrum_rgb_samples - 1][0]);
    double c = (lambda_clamped - spectrum_rgb[idx][0]) /
               (spectrum_rgb[idx + 1][0] - spectrum_rgb[idx][0]);
    for (unsigned int j = 1; j < 4; ++j) {
      double cie =
          (1.0 - c) * spectrum_rgb[idx][j] + c * spectrum_rgb[idx + 1][j];
      ((float *)&v_rgb)[j - 1] += cie * spec[i];
      rgb_sums[j - 1] += cie;
    }
    lambda += step_size;
  }
  for (int i = 0; i < 3; ++i) {
    ((float *)&v_rgb)[i] /= rgb_sums[i];
  }
  return v_rgb;
}

/********************************************/
SUTILFN float3 spectrum_to_rgb_fixed(const float spec[4])
/********************************************/
{
  static const float spec2rgb[3][4] = {{-0.04780f, -0.435f, 1.0059f, 3.16730f},
                                       {0.02830f, 0.626f, 1.0029f, 0.2591f},
                                       {0.99960f, 0.091f, -0.0097f, -0.00277f}};

  static const float sums[3] = {3.690400f, 1.9163f, 1.07813f};

  float rgb[3] = {0, 0, 0};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      rgb[i] += spec[j] * spec2rgb[i][j];
    }
  }

  const float3 res =
      make_float3(rgb[0] / sums[0], rgb[1] / sums[1], rgb[2] / sums[2]);
  return res;
}

/********************************************/
SUTILFN float3 to_SRGB(const float3 &ic)
/********************************************/
{ //*
  // const float3 cie_to_sr = make_float3(1.2904f, -0.2087f, -0.0492f);
  // const float3 cie_to_sg = make_float3(-0.2334f, 1.2787f, -0.0522f);
  // const float3 cie_to_sb = make_float3(0.0245f, -0.1314f, 0.8571f);
  const float3 c = ic; // make_float3(dot(ic, cie_to_sr), dot(ic, cie_to_sg),
                       // dot(ic, cie_to_sb)); //*/
  float invGamma = 1.0f / 2.4f;
  float3 powed = make_float3(powf(c.x, invGamma), powf(c.y, invGamma),
                             powf(c.z, invGamma));
  return make_float3(
      c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
      c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
      c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f);
}

/********************************************/
SUTILFN float dequantizeUnsigned8Bits(const unsigned char i)
/********************************************/
{
  enum { N = (1 << 8) - 1 };
  return fminf((float)i / (float)N, 1.f);
}

/********************************************/
SUTILFN unsigned char quantizeUnsigned8Bits(float x)
/********************************************/
{
  x = clamp(x, 0.0f, 1.0f);
  enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
  return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

/********************************************/
SUTILFN uchar4 make_srgb(const float3 &c)
/********************************************/
{

  // first apply gamma, then convert to unsigned char
  float3 srgb = to_SRGB(clamp(c, 0.0f, 1.0f));
  return make_uchar4(quantizeUnsigned8Bits(srgb.x),
                     quantizeUnsigned8Bits(srgb.y),
                     quantizeUnsigned8Bits(srgb.z), 255u);
}
SUTILFN uchar4 make_srgb(const float4 &c) {
  return make_srgb(make_float3(c.x, c.y, c.z));
}

/********************************************/
SUTILFN float3 hsv2rgb(float h, float s, float v)
/********************************************/
{
  const float h6 = h * 6.0f;
  const float frac = h6 - floor(h6);
  const float4 ell = v * make_float4(1.0f - s, 1.0f - s * frac,
                                     1.0f - s * (1.0f - frac), 1.0f);
  return h6 < 1.0f
             ? make_float3(ell.w, ell.z, ell.x)
             : (h6 < 2.0f
                    ? make_float3(ell.y, ell.w, ell.x)
                    : (h6 < 3.0f
                           ? make_float3(ell.x, ell.w, ell.z)
                           : (h6 < 4.0f
                                  ? make_float3(ell.x, ell.y, ell.w)
                                  : (h6 < 5.0f
                                         ? make_float3(ell.z, ell.x, ell.w)
                                         : make_float3(ell.w, ell.x, ell.y)))));
}

/********************************************/
SUTILFN float3 val2rainbow(float f)
/********************************************/
{
  const float t = clamp((log10f(f) + 14.0f) / 10.0f, 0.0f, 1.0f);
  const float h = clamp((1.0f - t) * 2.0f, 0.0f, 0.65f);
  return hsv2rgb(h, 1.0f, 1.0f);
}

/********************************************/
SUTILFN uchar4 make_rgba(const float3 &color)
/********************************************/
{
  float3 c = clamp(color, 0.0f, 1.0f);
  return make_uchar4(quantizeUnsigned8Bits(c.x), quantizeUnsigned8Bits(c.y),
                     quantizeUnsigned8Bits(c.z), 255u);
}

/********************************************/
SUTILFN uchar4 toColor(const float3 &color, enum ColorMode mode)
/********************************************/
{
  switch (mode) {
  case COLORMODE_SRGB:
    return make_srgb(color);
  case COLORMODE_NONE:
  default:
    return make_rgba(color);
  }
}

/********************************************/
SUTILFN float3 accumColor(const float3 &color, 
                          unsigned int frame, float4 *accum_buffer,
                          unsigned int image_idx)
/********************************************/
{
  float3 curr_sum =
      make_float3(accum_buffer[image_idx]) * static_cast<float>(frame);
  return (color + curr_sum) / static_cast<float>(frame + 1);
}

/********************************************/
SUTILFN float3 accumColor(const float3 &color, 
                          unsigned int frame, float3 *accum_buffer,
                          unsigned int image_idx)
/********************************************/
{
  float3 curr_sum = accum_buffer[image_idx] * static_cast<float>(frame);
  return (color + curr_sum) / static_cast<float>(frame + 1);
}

/********************************************/
SUTILFN float luminance(const float3 &val)
/********************************************/
{
  return 0.299f * val.x + 0.587f * val.y + 0.114f * val.z;
}

} // namespace color

#endif
