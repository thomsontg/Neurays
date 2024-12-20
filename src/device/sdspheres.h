#ifndef SDSPHERES_H
#define SDSPHERES_H

#include <cuda/random.h>

#define SDF scnoise
//#define SDF sdspheres

inline float cubic(const float3& v)
{
  const float x = 1.0f - dot(v, v)*4.0f;
  return fmaxf(0.0f, x)*x*x;
}

// Sparse convolution noise [Frisvad and Wyvill 2007, Luongo et al. 2020]
inline float scnoise(const float3& p)
{
  const unsigned int sources = 30;
  const float a = 0.8*powf(sources, -1.0f/3.0f);
  const float3 pi0 = floor(p - 0.5f);
  float result = 0.0f;
  for(unsigned int i = 0; i < 8; ++i)
  {
    const float3 corner = make_float3(i&1, (i>>1)&1, (i>>2)&1);
    const float3 pi = pi0 + corner;
    const int3 pii = make_int3(pi.x, pi.y, pi.z);
    unsigned int t = 4u*sources*static_cast<unsigned int>(pii.x + pii.y*1000 + pii.z*576 + pii.x*pii.y*pii.z*3);
    for(unsigned int j = 0; j < sources; ++j)
    {
      float c = a*rnd(t)*(j < sources/2 ? 1.0 : -1.0);
      float3 xi = make_float3(rnd(t), rnd(t), rnd(t));
      float3 x = pi + xi;
      result += c*cubic(x - p);
    }
  }
  return result;
}

// Signed distance function for randomly placed spheres inspired by sparse convolution noise
inline float sdspheres(const float3& p)
{
  const unsigned int spheres = 1;
  const float3 pi0 = floor(p - 0.5f);
  float result = 1.0e10f;
  for(unsigned int i = 0; i < 8; ++i)
  {
    const float3 corner = make_float3(i&1, (i>>1)&1, (i>>2)&1);
    const float3 pi = pi0 + corner;
    const int3 pii = make_int3(pi.x, pi.y, pi.z);
    unsigned int t = 4u*spheres*static_cast<unsigned int>(pii.x + pii.y*1000 + pii.z*576 + pii.x*pii.y*pii.z*3);
    for(unsigned int j = 0; j < spheres; ++j)
    {
      float r = rnd(t)*0.1f + 0.2f;
      float3 xi = make_float3(rnd(t), rnd(t), rnd(t));
      float3 x = pi + xi;
      result = fmin(result, length(p - x) - r);
    }
  }
  return result;
}

const float precis = 0.001f;

inline float raycast(const float3& ro, const float3& rd, float tmin, float tmax)
{
  // ray marching
  float t = tmin;
  float d = SDF(ro + t*rd);
  const float sgn = copysignf(1.0f, d);
  for(unsigned int i = 0; i < 100u; ++i)
  {
    if(fabsf(d) < precis*t || t > tmax) break;
    t += sgn*d; // *1.2f;
    d = SDF(ro + t*rd);
  }
  return t; // t < tmax ? t : 1.0e16f;
}

// https://iquilezles.org/articles/normalsSDF
float3 calcNormal(const float3& pos)
{
  float2 e = make_float2(1.0f, -1.0f)*0.5773f*precis;
  return normalize(
    make_float3(e.x, e.y, e.y)*SDF(pos + make_float3(e.x, e.y, e.y)) +
    make_float3(e.y, e.y, e.x)*SDF(pos + make_float3(e.y, e.y, e.x)) +
    make_float3(e.y, e.x, e.y)*SDF(pos + make_float3(e.y, e.x, e.y)) +
    make_float3(e.x)*SDF(pos + make_float3(e.x)));
}

#endif // SDSPHERES_H