#pragma once

SUTILFN uchar4 make_rgba(const float3 &color) {
  float3 c = clamp(color, 0.0f, 1.0f);
  return make_uchar4(quantizeUnsigned8Bits(c.x), quantizeUnsigned8Bits(c.y),
                     quantizeUnsigned8Bits(c.z), 255u);
}

// __device__ __inline__ float3 rainbow(float f)
// {
//   const float dx = 0.8f;
//   const float g = (6.0f - 2.0f*dx)*f + dx;
//   const float R = fmaxf(0.0f, (3.0f - abs(g - 4.0f) - abs(g - 5.0f))*0.5f);
//   const float G = fmaxf(0.0f, (4.0f - abs(g - 2.0f) - abs(g - 4.0f))*0.5f);
//   const float B = fmaxf(0.0f, (3.0f - abs(g - 1.0f) - abs(g - 2.0f))*0.5f);
//   return make_float3(R, G, B);
// }

// __device__ __inline__ float3 hsv2rgb(float h, float s, float v)
// {
//   const float h6 = h*6.0f;
//   const float frac = h6 - floor(h6);
//   const float4 ell = v*make_float4(1.0f - s, 1.0f - s*frac, 1.0f - s*(1.0f - frac), 1.0f);
//   return h6 < 1.0f ? make_float3(ell.w, ell.z, ell.x)
//     : (h6 < 2.0f ? make_float3(ell.y, ell.w, ell.x)
//       : (h6 < 3.0f ? make_float3(ell.x, ell.w, ell.z)
//         : (h6 < 4.0f ? make_float3(ell.x, ell.y, ell.w)
//           : (h6 < 5.0f ? make_float3(ell.z, ell.x, ell.w)
//             : make_float3(ell.w, ell.x, ell.y)))));
// }

// __device__ __inline__ float3 val2rainbow(float f)
// {
//   const float t = clamp((log10(f) + 14.0f)/10.0f, 0.0f, 1.0f);
//   const float h = clamp((1.0f - t)*2.0f, 0.0f, 0.65f);
//   return hsv2rgb(h, 1.0f, 1.0f);
// }

SUTILFN float3 rainbow(float f)
{
  const float dx = 0.8f;
  const float g = (6.0f - 2.0f*dx)*f + dx;
  const float R = fmaxf(0.0f, (3.0f - abs(g - 4.0f) - abs(g - 5.0f))*0.5f);
  const float G = fmaxf(0.0f, (4.0f - abs(g - 2.0f) - abs(g - 4.0f))*0.5f);
  const float B = fmaxf(0.0f, (3.0f - abs(g - 1.0f) - abs(g - 2.0f))*0.5f);
  return make_float3(R, G, B);
}

SUTILFN float3 hsv2rgb(float h, float s, float v)
{
  const float h6 = h*6.0f;
  const float frac = h6 - floor(h6);
  const float4 ell = v*make_float4(1.0f - s, 1.0f - s*frac, 1.0f - s*(1.0f - frac), 1.0f);
  return h6 < 1.0f ? make_float3(ell.w, ell.z, ell.x)
    : (h6 < 2.0f ? make_float3(ell.y, ell.w, ell.x)
      : (h6 < 3.0f ? make_float3(ell.x, ell.w, ell.z)
        : (h6 < 4.0f ? make_float3(ell.x, ell.y, ell.w)
          : (h6 < 5.0f ? make_float3(ell.z, ell.x, ell.w)
            : make_float3(ell.w, ell.x, ell.y)))));
}

SUTILFN float3 val2rainbow(float f)
{
  const float t = clamp((log10(f) + 7.5f)/8.8f, 0.0f, 1.0f);
  const float h = clamp((1.0f - t)*2.0f, 0.0f, 0.65f);
  return hsv2rgb(h, 1.0f, 1.0f);
}